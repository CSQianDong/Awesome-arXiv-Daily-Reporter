# Stop Summation: Min-Form Credit Assignment Is All Process Reward Model Needs for Reasoning 

**Authors**: Jie Cheng, Ruixi Qiao, Lijun Li, Chao Guo, Junle Wang, Gang Xiong, Yisheng Lv, Fei-Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15275)  

**Abstract**: Process reward models (PRMs) have proven effective for test-time scaling of Large Language Models (LLMs) on challenging reasoning tasks. However, reward hacking issues with PRMs limit their successful application in reinforcement fine-tuning. In this paper, we identify the main cause of PRM-induced reward hacking: the canonical summation-form credit assignment in reinforcement learning (RL), which defines the value as cumulative gamma-decayed future rewards, easily induces LLMs to hack steps with high rewards. To address this, we propose PURE: Process sUpervised Reinforcement lEarning. The key innovation of PURE is a min-form credit assignment that formulates the value function as the minimum of future rewards. This method significantly alleviates reward hacking by limiting the value function range and distributing advantages more reasonably. Through extensive experiments on 3 base models, we show that PRM-based approaches enabling min-form credit assignment achieve comparable reasoning performance to verifiable reward-based methods within only 30% steps. In contrast, the canonical sum-form credit assignment collapses training even at the beginning! Additionally, when we supplement PRM-based fine-tuning with just 10% verifiable rewards, we further alleviate reward hacking and produce the best fine-tuned model based on Qwen2.5-Math-7B in our experiments, achieving 82.5% accuracy on AMC23 and 53.3% average accuracy across 5 benchmarks. Moreover, we summarize the observed reward hacking cases and analyze the causes of training collapse. Code and models are available at this https URL. 

---
# Leveraging Language Models for Automated Patient Record Linkage 

**Authors**: Mohammad Beheshti, Lovedeep Gondara, Iris Zachary  

**Link**: [PDF](https://arxiv.org/pdf/2504.15261)  

**Abstract**: Objective: Healthcare data fragmentation presents a major challenge for linking patient data, necessitating robust record linkage to integrate patient records from diverse sources. This study investigates the feasibility of leveraging language models for automated patient record linkage, focusing on two key tasks: blocking and matching. Materials and Methods: We utilized real-world healthcare data from the Missouri Cancer Registry and Research Center, linking patient records from two independent sources using probabilistic linkage as a baseline. A transformer-based model, RoBERTa, was fine-tuned for blocking using sentence embeddings. For matching, several language models were experimented under fine-tuned and zero-shot settings, assessing their performance against ground truth labels. Results: The fine-tuned blocking model achieved a 92% reduction in the number of candidate pairs while maintaining near-perfect recall. In the matching task, fine-tuned Mistral-7B achieved the best performance with only 6 incorrect predictions. Among zero-shot models, Mistral-Small-24B performed best, with a total of 55 incorrect predictions. Discussion: Fine-tuned language models achieved strong performance in patient record blocking and matching with minimal errors. However, they remain less accurate and efficient than a hybrid rule-based and probabilistic approach for blocking. Additionally, reasoning models like DeepSeek-R1 are impractical for large-scale record linkage due to high computational costs. Conclusion: This study highlights the potential of language models for automating patient record linkage, offering improved efficiency by eliminating the manual efforts required to perform patient record linkage. Overall, language models offer a scalable solution that can enhance data integration, reduce manual effort, and support disease surveillance and research. 

---
# FlowReasoner: Reinforcing Query-Level Meta-Agents 

**Authors**: Hongcheng Gao, Yue Liu, Yufei He, Longxu Dou, Chao Du, Zhijie Deng, Bryan Hooi, Min Lin, Tianyu Pang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15257)  

**Abstract**: This paper proposes a query-level meta-agent named FlowReasoner to automate the design of query-level multi-agent systems, i.e., one system per user query. Our core idea is to incentivize a reasoning-based meta-agent via external execution feedback. Concretely, by distilling DeepSeek R1, we first endow the basic reasoning ability regarding the generation of multi-agent systems to FlowReasoner. Then, we further enhance it via reinforcement learning (RL) with external execution feedback. A multi-purpose reward is designed to guide the RL training from aspects of performance, complexity, and efficiency. In this manner, FlowReasoner is enabled to generate a personalized multi-agent system for each user query via deliberative reasoning. Experiments on both engineering and competition code benchmarks demonstrate the superiority of FlowReasoner. Remarkably, it surpasses o1-mini by 10.52% accuracy across three benchmarks. The code is available at this https URL. 

---
# SuoiAI: Building a Dataset for Aquatic Invertebrates in Vietnam 

**Authors**: Tue Vo, Lakshay Sharma, Tuan Dinh, Khuong Dinh, Trang Nguyen, Trung Phan, Minh Do, Duong Vu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15252)  

**Abstract**: Understanding and monitoring aquatic biodiversity is critical for ecological health and conservation efforts. This paper proposes SuoiAI, an end-to-end pipeline for building a dataset of aquatic invertebrates in Vietnam and employing machine learning (ML) techniques for species classification. We outline the methods for data collection, annotation, and model training, focusing on reducing annotation effort through semi-supervised learning and leveraging state-of-the-art object detection and classification models. Our approach aims to overcome challenges such as data scarcity, fine-grained classification, and deployment in diverse environmental conditions. 

---
# A Self-Improving Coding Agent 

**Authors**: Maxime Robeyns, Martin Szummer, Laurence Aitchison  

**Link**: [PDF](https://arxiv.org/pdf/2504.15228)  

**Abstract**: We demonstrate that an LLM coding agent, equipped with basic coding tools, can autonomously edit itself, and thereby improve its performance on benchmark tasks. We find performance gains from 17% to 53% on a random subset of SWE Bench Verified, with additional performance gains on LiveCodeBench, as well as synthetically generated agent benchmarks. Our work represents an advancement in the automated and open-ended design of agentic systems, and provides a reference agent framework for those seeking to post-train LLMs on tool use and other agentic tasks. 

---
# Position: Bayesian Statistics Facilitates Stakeholder Participation in Evaluation of Generative AI 

**Authors**: Yanan Long  

**Link**: [PDF](https://arxiv.org/pdf/2504.15211)  

**Abstract**: The evaluation of Generative AI (GenAI) systems plays a critical role in public policy and decision-making, yet existing methods are often limited by reliance on benchmark-driven, point-estimate comparisons that fail to capture uncertainty and broader societal impacts. This paper argues for the use of Bayesian statistics as a principled framework to address these challenges. Bayesian methods enable the integration of domain expertise through prior elicitation, allow for continuous learning from new data, and provide robust uncertainty quantification via posterior inference. We demonstrate how Bayesian inference can be applied to GenAI evaluation, particularly in incorporating stakeholder perspectives to enhance fairness, transparency, and reliability. Furthermore, we discuss Bayesian workflows as an iterative process for model validation and refinement, ensuring robust assessments of GenAI systems in dynamic, real-world contexts. 

---
# Synergistic Weak-Strong Collaboration by Aligning Preferences 

**Authors**: Yizhu Jiao, Xuchao Zhang, Zhaoyang Wang, Yubo Ma, Zhun Deng, Rujia Wang, Chetan Bansal, Saravan Rajmohan, Jiawei Han, Huaxiu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2504.15188)  

**Abstract**: Current Large Language Models (LLMs) excel in general reasoning yet struggle with specialized tasks requiring proprietary or domain-specific knowledge. Fine-tuning large models for every niche application is often infeasible due to black-box constraints and high computational overhead. To address this, we propose a collaborative framework that pairs a specialized weak model with a general strong model. The weak model, tailored to specific domains, produces initial drafts and background information, while the strong model leverages its advanced reasoning to refine these drafts, extending LLMs' capabilities to critical yet specialized tasks. To optimize this collaboration, we introduce a collaborative feedback to fine-tunes the weak model, which quantifies the influence of the weak model's contributions in the collaboration procedure and establishes preference pairs to guide preference tuning of the weak model. We validate our framework through experiments on three domains. We find that the collaboration significantly outperforms each model alone by leveraging complementary strengths. Moreover, aligning the weak model with the collaborative preference further enhances overall performance. 

---
# Behavioral Universe Network (BUN): A Behavioral Information-Based Framework for Complex Systems 

**Authors**: Wei Zhou, Ailiya Borjigin, Cong He  

**Link**: [PDF](https://arxiv.org/pdf/2504.15146)  

**Abstract**: Modern digital ecosystems feature complex, dynamic interactions among autonomous entities across diverse domains. Traditional models often separate agents and objects, lacking a unified foundation to capture their interactive behaviors. This paper introduces the Behavioral Universe Network (BUN), a theoretical framework grounded in the Agent-Interaction-Behavior (AIB) formalism. BUN treats subjects (active agents), objects (resources), and behaviors (operations) as first-class entities, all governed by a shared Behavioral Information Base (BIB). We detail the AIB core concepts and demonstrate how BUN leverages information-driven triggers, semantic enrichment, and adaptive rules to coordinate multi-agent systems. We highlight key benefits: enhanced behavior analysis, strong adaptability, and cross-domain interoperability. We conclude by positioning BUN as a promising foundation for next-generation digital governance and intelligent applications. 

---
# Contemplative Wisdom for Superalignment 

**Authors**: Ruben Laukkonen, Fionn Inglis, Shamil Chandaria, Lars Sandved-Smith, Jakob Hohwy, Jonathan Gold, Adam Elwood  

**Link**: [PDF](https://arxiv.org/pdf/2504.15125)  

**Abstract**: As artificial intelligence (AI) improves, traditional alignment strategies may falter in the face of unpredictable self-improvement, hidden subgoals, and the sheer complexity of intelligent systems. Rather than externally constraining behavior, we advocate designing AI with intrinsic morality built into its cognitive architecture and world model. Inspired by contemplative wisdom traditions, we show how four axiomatic principles can instil a resilient Wise World Model in AI systems. First, mindfulness enables self-monitoring and recalibration of emergent subgoals. Second, emptiness forestalls dogmatic goal fixation and relaxes rigid priors. Third, non-duality dissolves adversarial self-other boundaries. Fourth, boundless care motivates the universal reduction of suffering. We find that prompting AI to reflect on these principles improves performance on the AILuminate Benchmark using GPT-4o, particularly when combined. We offer detailed implementation strategies for state-of-the-art models, including contemplative architectures, constitutions, and reinforcement of chain-of-thought. For future systems, the active inference framework may offer the self-organizing and dynamic coupling capabilities needed to enact these insights in embodied agents. This interdisciplinary approach offers a self-correcting and resilient alternative to prevailing brittle control schemes. 

---
# Mitigating Degree Bias in Graph Representation Learning with Learnable Structural Augmentation and Structural Self-Attention 

**Authors**: Van Thuy Hoang, Hyeon-Ju Jeon, O-Joun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.15075)  

**Abstract**: Graph Neural Networks (GNNs) update node representations through message passing, which is primarily based on the homophily principle, assuming that adjacent nodes share similar features. However, in real-world graphs with long-tailed degree distributions, high-degree nodes dominate message passing, causing a degree bias where low-degree nodes remain under-represented due to inadequate messages. The main challenge in addressing degree bias is how to discover non-adjacent nodes to provide additional messages to low-degree nodes while reducing excessive messages for high-degree nodes. Nevertheless, exploiting non-adjacent nodes to provide valuable messages is challenging, as it could generate noisy information and disrupt the original graph structures. To solve it, we propose a novel Degree Fairness Graph Transformer, named DegFairGT, to mitigate degree bias by discovering structural similarities between non-adjacent nodes through learnable structural augmentation and structural self-attention. Our key idea is to exploit non-adjacent nodes with similar roles in the same community to generate informative edges under our augmentation, which could provide informative messages between nodes with similar roles while ensuring that the homophily principle is maintained within the community. To enable DegFairGT to learn such structural similarities, we then propose a structural self-attention to capture the similarities between node pairs. To preserve global graph structures and prevent graph augmentation from hindering graph structure, we propose a Self-Supervised Learning task to preserve p-step transition probability and regularize graph augmentation. Extensive experiments on six datasets showed that DegFairGT outperformed state-of-the-art baselines in degree fairness analysis, node classification, and node clustering tasks. 

---
# Text-to-Decision Agent: Learning Generalist Policies from Natural Language Supervision 

**Authors**: Shilin Zhang, Zican Hu, Wenhao Wu, Xinyi Xie, Jianxiang Tang, Chunlin Chen, Daoyi Dong, Yu Cheng, Zhenhong Sun, Zhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15046)  

**Abstract**: RL systems usually tackle generalization by inferring task beliefs from high-quality samples or warmup explorations. The restricted form limits their generality and usability since these supervision signals are expensive and even infeasible to acquire in advance for unseen tasks. Learning directly from the raw text about decision tasks is a promising alternative to leverage a much broader source of supervision. In the paper, we propose Text-to-Decision Agent (T2DA), a simple and scalable framework that supervises generalist policy learning with natural language. We first introduce a generalized world model to encode multi-task decision data into a dynamics-aware embedding space. Then, inspired by CLIP, we predict which textual description goes with which decision embedding, effectively bridging their semantic gap via contrastive language-decision pre-training and aligning the text embeddings to comprehend the environment dynamics. After training the text-conditioned generalist policy, the agent can directly realize zero-shot text-to-decision generation in response to language instructions. Comprehensive experiments on MuJoCo and Meta-World benchmarks show that T2DA facilitates high-capacity zero-shot generalization and outperforms various types of baselines. 

---
# Evaluating Code Generation of LLMs in Advanced Computer Science Problems 

**Authors**: Emir Catir, Robin Claesson, Rodothea Myrsini Tsoupidi  

**Link**: [PDF](https://arxiv.org/pdf/2504.14964)  

**Abstract**: Large Language Models (LLMs), such as GitHub Copilot and ChatGPT have become popular among programming students. Students use LLMs to assist them in programming courses, including generating source code. Previous work has evaluated the ability of LLMs in solving introductory-course programming assignments. The results have shown that LLMs are highly effective in generating code for introductory Computer Science (CS) courses. However, there is a gap in research on evaluating LLMs' ability to generate code that solves advanced programming assignments. In this work, we evaluate the ability of four LLM tools to solve programming assignments from advanced CS courses in three popular programming languages, Java, Python, and C. We manually select 12 problems, three problems from introductory courses as the baseline and nine programming assignments from second- and third-year CS courses. To evaluate the LLM-generated code, we generate a test suite of 1000 test cases per problem and analyze the program output. Our evaluation shows that although LLMs are highly effective in generating source code for introductory programming courses, solving advanced programming assignments is more challenging. Nonetheless, in many cases, LLMs identify the base problem and provide partial solutions that may be useful to CS students. Furthermore, our results may provide useful guidance for teachers of advanced programming courses on how to design programming assignments. 

---
# Generative Semantic Communications: Principles and Practices 

**Authors**: Xiaojun Yuan, Haoming Ma, Yinuo Huang, Zhoufan Hua, Yong Zuo, Zhi Ding  

**Link**: [PDF](https://arxiv.org/pdf/2504.14947)  

**Abstract**: Semantic communication leverages artificial intelligence (AI) technologies to extract semantic information from data for efficient transmission, theraby significantly reducing communication cost. With the evolution towards artificial general intelligence (AGI), the increasing demands for AGI services pose new challenges to semantic communication. In response, we propose a new paradigm for AGI-driven communications, called generative semantic communication (GSC), which utilizes advanced AI technologies such as foundation models and generative models. We first describe the basic concept of GSC and its difference from existing semantic communications, and then introduce a general framework of GSC, followed by two case studies to verify the advantages of GSC in AGI-driven applications. Finally, open challenges and new research directions are discussed to stimulate this line of research and pave the way for practical applications. 

---
# EducationQ: Evaluating LLMs' Teaching Capabilities Through Multi-Agent Dialogue Framework 

**Authors**: Yao Shi, Rongkeng Liang, Yong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14928)  

**Abstract**: Large language models (LLMs) increasingly serve as educational tools, yet evaluating their teaching capabilities remains challenging due to the resource-intensive, context-dependent, and methodologically complex nature of teacher-student interactions. We introduce EducationQ, a multi-agent dialogue framework that efficiently assesses teaching capabilities through simulated dynamic educational scenarios, featuring specialized agents for teaching, learning, and evaluation. Testing 14 LLMs across major AI Organizations (OpenAI, Meta, Google, Anthropic, and others) on 1,498 questions spanning 13 disciplines and 10 difficulty levels reveals that teaching effectiveness does not correlate linearly with model scale or general reasoning capabilities - with some smaller open-source models outperforming larger commercial counterparts in teaching contexts. This finding highlights a critical gap in current evaluations that prioritize knowledge recall over interactive pedagogy. Our mixed-methods evaluation, combining quantitative metrics with qualitative analysis and expert case studies, identifies distinct pedagogical strengths employed by top-performing models (e.g., sophisticated questioning strategies, adaptive feedback mechanisms). Human expert evaluations show 78% agreement with our automated qualitative analysis of effective teaching behaviors, validating our methodology. EducationQ demonstrates that LLMs-as-teachers require specialized optimization beyond simple scaling, suggesting next-generation educational AI prioritize targeted enhancement of specific pedagogical effectiveness. 

---
# OTC: Optimal Tool Calls via Reinforcement Learning 

**Authors**: Hongru Wang, Cheng Qian, Wanjun Zhong, Xiusi Chen, Jiahao Qiu, Shijue Huang, Bowen Jin, Mengdi Wang, Kam-Fai Wong, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2504.14870)  

**Abstract**: Tool-integrated reasoning (TIR) augments large language models (LLMs) with the ability to invoke external tools, such as search engines and code interpreters, to solve tasks beyond the capabilities of language-only reasoning. While reinforcement learning (RL) has shown promise in improving TIR by optimizing final answer correctness, existing approaches often overlook the efficiency and cost associated with tool usage. This can lead to suboptimal behavior, including excessive tool calls that increase computational and financial overhead, or insufficient tool use that compromises answer quality. In this work, we propose Optimal Tool Call-controlled Policy Optimization (OTC-PO), a simple yet effective RL-based framework that encourages models to produce accurate answers with minimal tool calls. Our method introduces a tool-integrated reward that jointly considers correctness and tool efficiency, promoting high tool productivity. We instantiate this framework within both Proximal Policy Optimization (PPO) and Group Relative Preference Optimization (GRPO), resulting in OTC-PPO and OTC-GRPO. Experiments with Qwen-2.5 and Qwen-Math across multiple QA benchmarks show that our approach reduces tool calls by up to 73.1\% and improves tool productivity by up to 229.4\%, while maintaining comparable answer accuracy. To the best of our knowledge, this is the first RL-based framework that explicitly optimizes tool-use efficiency in TIR. 

---
# AlignRAG: An Adaptable Framework for Resolving Misalignments in Retrieval-Aware Reasoning of RAG 

**Authors**: Jiaqi Wei, Hao Zhou, Xiang Zhang, Di Zhang, Zijie Qiu, Wei Wei, Jinzhe Li, Wanli Ouyang, Siqi Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.14858)  

**Abstract**: Retrieval-augmented generation (RAG) has emerged as a foundational paradigm for knowledge-grounded text generation. However, existing RAG pipelines often fail to ensure that the reasoning trajectories align with the evidential constraints imposed by retrieved content. In this paper, we reframe RAG as a problem of retrieval-aware reasoning and identify a core challenge: reasoning misalignment-the mismatch between a model's reasoning trajectory and the retrieved evidence. To address this challenge, we propose AlignRAG, a novel test-time framework that mitigates reasoning misalignment through iterative Critique-Driven Alignment (CDA) steps. In contrast to prior approaches that rely on static training or post-hoc selection, AlignRAG actively refines reasoning trajectories during inference by enforcing fine-grained alignment with evidence. Our framework introduces a new paradigm for retrieval-aware reasoning by: (1) constructing context-rich training corpora; (2) generating contrastive critiques from preference-aware reasoning trajectories; (3) training a dedicated \textit{Critic Language Model (CLM)} to identify reasoning misalignments; and (4) applying CDA steps to optimize reasoning trajectories iteratively. Empirical results demonstrate that AlignRAG consistently outperforms all baselines and could integrate as a plug-and-play module into existing RAG pipelines without further changes. By reconceptualizing RAG as a structured reasoning trajectory and establishing the test-time framework for correcting reasoning misalignments in RAG, AlignRAG provides practical advancements for retrieval-aware generation. 

---
# Establishing Reliability Metrics for Reward Models in Large Language Models 

**Authors**: Yizhou Chen, Yawen Liu, Xuesi Wang, Qingtao Yu, Guangda Huzhang, Anxiang Zeng, Han Yu, Zhiming Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.14838)  

**Abstract**: The reward model (RM) that represents human preferences plays a crucial role in optimizing the outputs of large language models (LLMs), e.g., through reinforcement learning from human feedback (RLHF) or rejection sampling. However, a long challenge for RM is its uncertain reliability, i.e., LLM outputs with higher rewards may not align with actual human preferences. Currently, there is a lack of a convincing metric to quantify the reliability of RMs. To bridge this gap, we propose the \textit{\underline{R}eliable at \underline{$\eta$}} (RETA) metric, which directly measures the reliability of an RM by evaluating the average quality (scored by an oracle) of the top $\eta$ quantile responses assessed by an RM. On top of RETA, we present an integrated benchmarking pipeline that allows anyone to evaluate their own RM without incurring additional Oracle labeling costs. Extensive experimental studies demonstrate the superior stability of RETA metric, providing solid evaluations of the reliability of various publicly available and proprietary RMs. When dealing with an unreliable RM, we can use the RETA metric to identify the optimal quantile from which to select the responses. 

---
# DONOD: Robust and Generalizable Instruction Fine-Tuning for LLMs via Model-Intrinsic Dataset Pruning 

**Authors**: Jucheng Hu, Surong Yang, Dongzhan Zhou, Lijun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14810)  

**Abstract**: Ad-hoc instruction fine-tuning of large language models (LLMs) is widely adopted for domain-specific adaptation. While domain-specific supervised fine-tuning (SFT) is effective and efficient, it often weakens cross-domain generalization and struggles with noisy training data. To address these challenges, we propose DONOD, a lightweight model-intrinsic data pruning method. Our approach evaluates data using two model-parameter-based metrics: Delta of Norm (DON), which captures the cumulative influence on model weights, and Norm of Delta (NOD), which quantifies weight instability. Moreover, by employing the Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS) algorithm, we effectively filter noisy, unlearnable, and generalization-harming samples without relying on auxiliary models during the SFT process. Experiments on mathematical tasks demonstrate that data selected by DONOD achieve superior fine-tuning efficiency and improved robustness against noisy data. By filtering out 70% of the full dataset, we improve target-domain accuracy by 14.90% and cross-domain accuracy by 5.67%. Meanwhile, our selected data present superior cross-architecture generalization. Data pruned by smaller models (e.g., Llama 3.1-8B) generalize effectively on larger models (e.g., Llama 2-13B). Compared to existing related methodologies, DONOD demonstrates comparable or superior performance while remaining dataset-agnostic, enabling broader applicability. 

---
# PLANET: A Collection of Benchmarks for Evaluating LLMs' Planning Capabilities 

**Authors**: Haoming Li, Zhaoliang Chen, Jonathan Zhang, Fei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14773)  

**Abstract**: Planning is central to agents and agentic AI. The ability to plan, e.g., creating travel itineraries within a budget, holds immense potential in both scientific and commercial contexts. Moreover, optimal plans tend to require fewer resources compared to ad-hoc methods. To date, a comprehensive understanding of existing planning benchmarks appears to be lacking. Without it, comparing planning algorithms' performance across domains or selecting suitable algorithms for new scenarios remains challenging. In this paper, we examine a range of planning benchmarks to identify commonly used testbeds for algorithm development and highlight potential gaps. These benchmarks are categorized into embodied environments, web navigation, scheduling, games and puzzles, and everyday task automation. Our study recommends the most appropriate benchmarks for various algorithms and offers insights to guide future benchmark development. 

---
# AI with Emotions: Exploring Emotional Expressions in Large Language Models 

**Authors**: Shin-nosuke Ishikawa, Atsushi Yoshino  

**Link**: [PDF](https://arxiv.org/pdf/2504.14706)  

**Abstract**: The human-level performance of Large Language Models (LLMs) across various tasks has raised expectations for the potential of Artificial Intelligence (AI) to possess emotions someday. To explore the capability of current LLMs to express emotions in their outputs, we conducted an experiment using several LLMs (OpenAI GPT, Google Gemini, Meta Llama3, and Cohere Command R+) to role-play as agents answering questions with specified emotional this http URL defined the emotional states using Russell's Circumplex model, a well-established framework that characterizes emotions along the sleepy-activated (arousal) and pleasure-displeasure (valence) axes. We chose this model for its simplicity, utilizing two continuous parameters, which allows for better controllability in applications involving continuous changes in emotional states. The responses generated were evaluated using a sentiment analysis model, independent of the LLMs, trained on the GoEmotions dataset. The evaluation showed that the emotional states of the generated answers were consistent with the specifications, demonstrating the LLMs' capability for emotional expression. This indicates the potential for LLM-based AI agents to simulate emotions, opening up a wide range of applications for emotion-based interactions, such as advisors or consultants who can provide advice or opinions with a personal touch. 

---
# A Framework for Benchmarking and Aligning Task-Planning Safety in LLM-Based Embodied Agents 

**Authors**: Yuting Huang, Leilei Ding, Zhipeng Tang, Tianfu Wang, Xinrui Lin, Wuyang Zhang, Mingxiao Ma, Yanyong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14650)  

**Abstract**: Large Language Models (LLMs) exhibit substantial promise in enhancing task-planning capabilities within embodied agents due to their advanced reasoning and comprehension. However, the systemic safety of these agents remains an underexplored frontier. In this study, we present Safe-BeAl, an integrated framework for the measurement (SafePlan-Bench) and alignment (Safe-Align) of LLM-based embodied agents' behaviors. SafePlan-Bench establishes a comprehensive benchmark for evaluating task-planning safety, encompassing 2,027 daily tasks and corresponding environments distributed across 8 distinct hazard categories (e.g., Fire Hazard). Our empirical analysis reveals that even in the absence of adversarial inputs or malicious intent, LLM-based agents can exhibit unsafe behaviors. To mitigate these hazards, we propose Safe-Align, a method designed to integrate physical-world safety knowledge into LLM-based embodied agents while maintaining task-specific performance. Experiments across a variety of settings demonstrate that Safe-BeAl provides comprehensive safety validation, improving safety by 8.55 - 15.22%, compared to embodied agents based on GPT-4, while ensuring successful task completion. 

---
# Consensus in Motion: A Case of Dynamic Rationality of Sequential Learning in Probability Aggregation 

**Authors**: Polina Gordienko, Christoph Jansen, Thomas Augustin, Martin Rechenauer  

**Link**: [PDF](https://arxiv.org/pdf/2504.14624)  

**Abstract**: We propose a framework for probability aggregation based on propositional probability logic. Unlike conventional judgment aggregation, which focuses on static rationality, our model addresses dynamic rationality by ensuring that collective beliefs update consistently with new information. We show that any consensus-compatible and independent aggregation rule on a non-nested agenda is necessarily linear. Furthermore, we provide sufficient conditions for a fair learning process, where individuals initially agree on a specified subset of propositions known as the common ground, and new information is restricted to this shared foundation. This guarantees that updating individual judgments via Bayesian conditioning-whether performed before or after aggregation-yields the same collective belief. A distinctive feature of our framework is its treatment of sequential decision-making, which allows new information to be incorporated progressively through multiple stages while maintaining the established common ground. We illustrate our findings with a running example in a political scenario concerning healthcare and immigration policies. 

---
# UFO2: The Desktop AgentOS 

**Authors**: Chaoyun Zhang, He Huang, Chiming Ni, Jian Mu, Si Qin, Shilin He, Lu Wang, Fangkai Yang, Pu Zhao, Chao Du, Liqun Li, Yu Kang, Zhao Jiang, Suzhen Zheng, Rujia Wang, Jiaxu Qian, Minghua Ma, Jian-Guang Lou, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14603)  

**Abstract**: Recent Computer-Using Agents (CUAs), powered by multimodal large language models (LLMs), offer a promising direction for automating complex desktop workflows through natural language. However, most existing CUAs remain conceptual prototypes, hindered by shallow OS integration, fragile screenshot-based interaction, and disruptive execution.
We present UFO2, a multiagent AgentOS for Windows desktops that elevates CUAs into practical, system-level automation. UFO2 features a centralized HostAgent for task decomposition and coordination, alongside a collection of application-specialized AppAgent equipped with native APIs, domain-specific knowledge, and a unified GUI--API action layer. This architecture enables robust task execution while preserving modularity and extensibility. A hybrid control detection pipeline fuses Windows UI Automation (UIA) with vision-based parsing to support diverse interface styles. Runtime efficiency is further enhanced through speculative multi-action planning, reducing per-step LLM overhead. Finally, a Picture-in-Picture (PiP) interface enables automation within an isolated virtual desktop, allowing agents and users to operate concurrently without interference.
We evaluate UFO2 across over 20 real-world Windows applications, demonstrating substantial improvements in robustness and execution accuracy over prior CUAs. Our results show that deep OS integration unlocks a scalable path toward reliable, user-aligned desktop automation. 

---
# Toward the Axiomatization of Intelligence: Structure, Time, and Existence 

**Authors**: Kei Itoh  

**Link**: [PDF](https://arxiv.org/pdf/2504.14596)  

**Abstract**: This study aims to construct an axiomatic definition of intelligence within a meta-framework that defines the method of definition, addressing intelligence as an inherently naive and polysemous concept. Initially, we formalize a set-theoretic representation of the universe as the domain wherein intelligence exists and characterize intelligence as a structure that involves temporal evolution and interaction with other sets. Starting from a naive definition of intelligence as "an entity possessing structures for externally inputting, internally processing, and externally outputting information or matter," we axiomatically reformulate it within this set-theoretical depiction of the universe. Applying this axiomatic definition, we compare and interpret three examples -- Hebbian non-optimized neural networks (NNs), backpropagation-optimized NNs, and biological reflexive systems -- in terms of their intelligence, structural properties, and biological plausibility. Furthermore, by extending our definition into a categorical framework, we introduce two categories, "Time Category" and "Intelligence Category," along with the functorial relationships between them, demonstrating the potential to represent changes and mimicry relationships among intelligent systems abstractly. Additionally, since intelligence, as defined herein, functions effectively only when accompanied by temporal interactions, we introduce the concept of "activity" and explore how activity-based conditions influence classifications and interpretations of intelligence. Finally, we suggest that our definitional methodology is not limited to intelligence alone, but can be similarly applied to other concepts, such as consciousness and emotion, advocating for their formal reinterpretation through the same procedural steps: defining a universal representation, selecting naive definitions, and axiomatic formalization. 

---
# LLM-Enabled In-Context Learning for Data Collection Scheduling in UAV-assisted Sensor Networks 

**Authors**: Yousef Emami, Hao Gao, SeyedSina Nabavirazani, Luis Almeida  

**Link**: [PDF](https://arxiv.org/pdf/2504.14556)  

**Abstract**: Unmanned Aerial Vehicles (UAVs) are increasingly being used in various private and commercial applications, e.g. traffic control, package delivery, and Search and Rescue (SAR) operations. Machine Learning (ML) methods used in UAV-assisted Sensor Networks (UASNETs) and especially in Deep Reinforcement Learning (DRL) face challenges such as complex and lengthy model training, gaps between simulation and reality, and low sample efficiency, which conflict with the urgency of emergencies such as SAR operations. This paper proposes In-Context Learning (ICL)-based Data Collection Scheduling (ICLDC) scheme, as an alternative to DRL in emergencies. The UAV collects and transmits logged sensory data, to an LLM, to generate a task description in natural language, from which it obtains a data collection schedule to be executed by the UAV. The system continuously adapts by adding feedback to task descriptions and utilizing feedback for future decisions. This method is tested against jailbreaking attacks, where task description is manipulated to undermine network performance, highlighting the vulnerability of LLMs to such attacks. The proposed ICLDC outperforms the Maximum Channel Gain by reducing cumulative packet loss by approximately 56\%. ICLDC presents a promising direction for intelligent scheduling and control in UAV-assisted data collection. 

---
# Learning from Reasoning Failures via Synthetic Data Generation 

**Authors**: Gabriela Ben Melech Stan, Estelle Aflalo, Avinash Madasu, Vasudev Lal, Phillip Howard  

**Link**: [PDF](https://arxiv.org/pdf/2504.14523)  

**Abstract**: Training models on synthetic data has emerged as an increasingly important strategy for improving the performance of generative AI. This approach is particularly helpful for large multimodal models (LMMs) due to the relative scarcity of high-quality paired image-text data compared to language-only data. While a variety of methods have been proposed for generating large multimodal datasets, they do not tailor the synthetic data to address specific deficiencies in the reasoning abilities of LMMs which will be trained with the generated dataset. In contrast, humans often learn in a more efficient manner by seeking out examples related to the types of reasoning where they have failed previously. Inspired by this observation, we propose a new approach for synthetic data generation which is grounded in the analysis of an existing LMM's reasoning failures. Our methodology leverages frontier models to automatically analyze errors produced by a weaker LMM and propose new examples which can be used to correct the reasoning failure via additional training, which are then further filtered to ensure high quality. We generate a large multimodal instruction tuning dataset containing over 553k examples using our approach and conduct extensive experiments demonstrating its utility for improving the performance of LMMs on multiple downstream tasks. Our results show that models trained on our synthetic data can even exceed the performance of LMMs trained on an equivalent amount of additional real data, demonstrating the high value of generating synthetic data targeted to specific reasoning failure modes in LMMs. We will make our dataset and code publicly available. 

---
# Meta-Thinking in LLMs via Multi-Agent Reinforcement Learning: A Survey 

**Authors**: Ahsan Bilal, Muhammad Ahmed Mohsin, Muhammad Umer, Muhammad Awais Khan Bangash, Muhammad Ali Jamshed  

**Link**: [PDF](https://arxiv.org/pdf/2504.14520)  

**Abstract**: This survey explores the development of meta-thinking capabilities in Large Language Models (LLMs) from a Multi-Agent Reinforcement Learning (MARL) perspective. Meta-thinking self-reflection, assessment, and control of thinking processes is an important next step in enhancing LLM reliability, flexibility, and performance, particularly for complex or high-stakes tasks. The survey begins by analyzing current LLM limitations, such as hallucinations and the lack of internal self-assessment mechanisms. It then talks about newer methods, including RL from human feedback (RLHF), self-distillation, and chain-of-thought prompting, and each of their limitations. The crux of the survey is to talk about how multi-agent architectures, namely supervisor-agent hierarchies, agent debates, and theory of mind frameworks, can emulate human-like introspective behavior and enhance LLM robustness. By exploring reward mechanisms, self-play, and continuous learning methods in MARL, this survey gives a comprehensive roadmap to building introspective, adaptive, and trustworthy LLMs. Evaluation metrics, datasets, and future research avenues, including neuroscience-inspired architectures and hybrid symbolic reasoning, are also discussed. 

---
# Seeing Through Risk: A Symbolic Approximation of Prospect Theory 

**Authors**: Ali Arslan Yousaf, Umair Rehman, Muhammad Umair Danish  

**Link**: [PDF](https://arxiv.org/pdf/2504.14448)  

**Abstract**: We propose a novel symbolic modeling framework for decision-making under risk that merges interpretability with the core insights of Prospect Theory. Our approach replaces opaque utility curves and probability weighting functions with transparent, effect-size-guided features. We mathematically formalize the method, demonstrate its ability to replicate well-known framing and loss-aversion phenomena, and provide an end-to-end empirical validation on synthetic datasets. The resulting model achieves competitive predictive performance while yielding clear coefficients mapped onto psychological constructs, making it suitable for applications ranging from AI safety to economic policy analysis. 

---
# The Geometry of Self-Verification in a Task-Specific Reasoning Model 

**Authors**: Andrew Lee, Lihao Sun, Chris Wendler, Fernanda Viégas, Martin Wattenberg  

**Link**: [PDF](https://arxiv.org/pdf/2504.14379)  

**Abstract**: How do reasoning models verify their own answers? We study this question by training a model using DeepSeek R1's recipe on the CountDown task. We leverage the fact that preference tuning leads to mode collapse, resulting in a model that always produces highly structured and easily parse-able chain-of-thought sequences. With this setup, we do a top-down and bottom-up analysis to reverse-engineer how the model verifies its outputs. Our top-down analysis reveals Gated Linear Unit (GLU) weights encoding verification-related tokens, such as ``success'' or ``incorrect'', which activate according to the correctness of the model's reasoning steps. Our bottom-up analysis reveals that ``previous-token heads'' are mainly responsible for model verification. Our analyses meet in the middle: drawing inspiration from inter-layer communication channels, we use the identified GLU vectors to localize as few as three attention heads that can disable model verification, pointing to a necessary component of a potentially larger verification circuit. 

---
# Mathematical Programming Models for Exact and Interpretable Formulation of Neural Networks 

**Authors**: Masoud Ataei, Edrin Hasaj, Jacob Gipp, Sepideh Forouzi  

**Link**: [PDF](https://arxiv.org/pdf/2504.14356)  

**Abstract**: This paper presents a unified mixed-integer programming framework for training sparse and interpretable neural networks. We develop exact formulations for both fully connected and convolutional architectures by modeling nonlinearities such as ReLU activations through binary variables and encoding structural sparsity via filter- and layer-level pruning constraints. The resulting models integrate parameter learning, architecture selection, and structural regularization within a single optimization problem, yielding globally optimal solutions with respect to a composite objective that balances prediction accuracy, weight sparsity, and architectural compactness. The mixed-integer programming formulation accommodates piecewise-linear operations, including max pooling and activation gating, and permits precise enforcement of logic-based or domain-specific constraints. By incorporating considerations of interpretability, sparsity, and verifiability directly into the training process, the proposed framework bridges a range of research areas including explainable artificial intelligence, symbolic reasoning, and formal verification. 

---
# Time Up! An Empirical Study of LLM Reasoning Ability Under Output Length Constraint 

**Authors**: Yi Sun, Han Wang, Jiaqiang Li, Jiacheng Liu, Xiangyu Li, Hao Wen, Huiwen Zheng, Yan Liang, Yuanchun Li, Yunxin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14350)  

**Abstract**: Recent work has demonstrated the remarkable potential of Large Language Models (LLMs) in test-time scaling. By making the models think before answering, they are able to achieve much higher accuracy with extra inference computation. However, in many real-world scenarios, models are used under time constraints, where an answer should be given to the user within a certain output length. It is unclear whether and how the reasoning abilities of LLMs remain effective under such constraints. We take a first look at this problem by conducting an in-depth empirical study. Specifically, we test more than 25 LLMs on common reasoning datasets under a wide range of output length budgets, and we analyze the correlation between the inference accuracy and various properties including model type, model size, prompt style, etc. We also consider the mappings between the token budgets and the actual on-device latency budgets. The results have demonstrated several interesting findings regarding the budget-aware LLM reasoning that differ from the unconstrained situation, e.g. the optimal choices of model sizes and prompts change under different budgets. These findings offer practical guidance for users to deploy LLMs under real-world latency constraints. 

---
# FAIRGAME: a Framework for AI Agents Bias Recognition using Game Theory 

**Authors**: Alessio Buscemi, Daniele Proverbio, Alessandro Di Stefano, Anh Han, German Castignani, Pietro Di Liò  

**Link**: [PDF](https://arxiv.org/pdf/2504.14325)  

**Abstract**: Letting AI agents interact in multi-agent applications adds a layer of complexity to the interpretability and prediction of AI outcomes, with profound implications for their trustworthy adoption in research and society. Game theory offers powerful models to capture and interpret strategic interaction among agents, but requires the support of reproducible, standardized and user-friendly IT frameworks to enable comparison and interpretation of results. To this end, we present FAIRGAME, a Framework for AI Agents Bias Recognition using Game Theory. We describe its implementation and usage, and we employ it to uncover biased outcomes in popular games among AI agents, depending on the employed Large Language Model (LLM) and used language, as well as on the personality trait or strategic knowledge of the agents. Overall, FAIRGAME allows users to reliably and easily simulate their desired games and scenarios and compare the results across simulation campaigns and with game-theoretic predictions, enabling the systematic discovery of biases, the anticipation of emerging behavior out of strategic interplays, and empowering further research into strategic decision-making using LLM agents. 

---
# RadioDiff-Inverse: Diffusion Enhanced Bayesian Inverse Estimation for ISAC Radio Map Construction 

**Authors**: Xiucheng Wang, Zhongsheng Fang, Nan Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.14298)  

**Abstract**: Radio maps (RMs) are essential for environment-aware communication and sensing, providing location-specific wireless channel information. Existing RM construction methods often rely on precise environmental data and base station (BS) locations, which are not always available in dynamic or privacy-sensitive environments. While sparse measurement techniques reduce data collection, the impact of noise in sparse data on RM accuracy is not well understood. This paper addresses these challenges by formulating RM construction as a Bayesian inverse problem under coarse environmental knowledge and noisy sparse measurements. Although maximum a posteriori (MAP) filtering offers an optimal solution, it requires a precise prior distribution of the RM, which is typically unavailable. To solve this, we propose RadioDiff-Inverse, a diffusion-enhanced Bayesian inverse estimation framework that uses an unconditional generative diffusion model to learn the RM prior. This approach not only reconstructs the spatial distribution of wireless channel features but also enables environmental structure perception, such as building outlines, and location of BS just relay on pathloss, through integrated sensing and communication (ISAC). Remarkably, RadioDiff-Inverse is training-free, leveraging a pre-trained model from Imagenet without task-specific fine-tuning, which significantly reduces the training cost of using generative large model in wireless networks. Experimental results demonstrate that RadioDiff-Inverse achieves state-of-the-art performance in accuracy of RM construction and environmental reconstruction, and robustness against noisy sparse sampling. 

---
# CHAINSFORMER: Numerical Reasoning on Knowledge Graphs from a Chain Perspective 

**Authors**: Ze Zhao, Bin Lu, Xiaoying Gan, Gu Tang, Luoyi Fu, Xinbing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14282)  

**Abstract**: Reasoning over Knowledge Graphs (KGs) plays a pivotal role in knowledge graph completion or question answering systems, providing richer and more accurate triples and attributes. As numerical attributes become increasingly essential in characterizing entities and relations in KGs, the ability to reason over these attributes has gained significant importance. Existing graph-based methods such as Graph Neural Networks (GNNs) and Knowledge Graph Embeddings (KGEs), primarily focus on aggregating homogeneous local neighbors and implicitly embedding diverse triples. However, these approaches often fail to fully leverage the potential of logical paths within the graph, limiting their effectiveness in exploiting the reasoning process. To address these limitations, we propose ChainsFormer, a novel chain-based framework designed to support numerical reasoning. Chainsformer not only explicitly constructs logical chains but also expands the reasoning depth to multiple hops. Specially, we introduces Relation-Attribute Chains (RA-Chains), a specialized logic chain, to model sequential reasoning patterns. ChainsFormer captures the step-by-step nature of multi-hop reasoning along RA-Chains by employing sequential in-context learning. To mitigate the impact of noisy chains, we propose a hyperbolic affinity scoring mechanism that selects relevant logic chains in a variable-resolution space. Furthermore, ChainsFormer incorporates an attention-based numerical reasoner to identify critical reasoning paths, enhancing both reasoning accuracy and transparency. Experimental results demonstrate that ChainsFormer significantly outperforms state-of-the-art methods, achieving up to a 20.0% improvement in performance. The implementations are available at this https URL. 

---
# ProtPainter: Draw or Drag Protein via Topology-guided Diffusion 

**Authors**: Zhengxi Lu, Shizhuo Cheng, Yuru Jiang, Yan Zhang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14274)  

**Abstract**: Recent advances in protein backbone generation have achieved promising results under structural, functional, or physical constraints. However, existing methods lack the flexibility for precise topology control, limiting navigation of the backbone space. We present ProtPainter, a diffusion-based approach for generating protein backbones conditioned on 3D curves. ProtPainter follows a two-stage process: curve-based sketching and sketch-guided backbone generation. For the first stage, we propose CurveEncoder, which predicts secondary structure annotations from a curve to parametrize sketch generation. For the second stage, the sketch guides the generative process in Denoising Diffusion Probabilistic Modeling (DDPM) to generate backbones. During this process, we further introduce a fusion scheduling scheme, Helix-Gating, to control the scaling factors. To evaluate, we propose the first benchmark for topology-conditioned protein generation, introducing Protein Restoration Task and a new metric, self-consistency Topology Fitness (scTF). Experiments demonstrate ProtPainter's ability to generate topology-fit (scTF > 0.8) and designable (scTM > 0.5) backbones, with drawing and dragging tasks showcasing its flexibility and versatility. 

---
# Rethinking Traffic Flow Forecasting: From Transition to Generatation 

**Authors**: Li Shijiao, Ma Zhipeng, He Huajun, Chen Haiyue  

**Link**: [PDF](https://arxiv.org/pdf/2504.14248)  

**Abstract**: Traffic flow prediction plays an important role in Intelligent Transportation Systems in traffic management and urban planning. There have been extensive successful works in this area. However, these approaches focus only on modelling the flow transition and ignore the flow generation process, which manifests itself in two ways: (i) The models are based on Markovian assumptions, ignoring the multi-periodicity of the flow generation in nodes. (ii) The same structure is designed to encode both the transition and generation processes, ignoring the differences between them. To address these problems, we propose an Effective Multi-Branch Similarity Transformer for Traffic Flow Prediction, namely EMBSFormer. Through data analysis, we find that the factors affecting traffic flow include node-level traffic generation and graph-level traffic transition, which describe the multi-periodicity and interaction pattern of nodes, respectively. Specifically, to capture traffic generation patterns, we propose a similarity analysis module that supports multi-branch encoding to dynamically expand significant cycles. For traffic transition, we employ a temporal and spatial self-attention mechanism to maintain global node interactions, and use GNN and time conv to model local node interactions, respectively. Model performance is evaluated on three real-world datasets on both long-term and short-term prediction tasks. Experimental results show that EMBSFormer outperforms baselines on both tasks. Moreover, compared to models based on flow transition modelling (e.g. GMAN, 513k), the variant of EMBSFormer(93K) only uses 18\% of the parameters, achieving the same performance. 

---
# A Knowledge-Informed Deep Learning Paradigm for Generalizable and Stability-Optimized Car-Following Models 

**Authors**: Chengming Wang, Dongyao Jia, Wei Wang, Dong Ngoduy, Bei Peng, Jianping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14241)  

**Abstract**: Car-following models (CFMs) are fundamental to traffic flow analysis and autonomous driving. Although calibrated physics-based and trained data-driven CFMs can replicate human driving behavior, their reliance on specific datasets limits generalization across diverse scenarios and reduces reliability in real-world deployment. Moreover, these models typically focus on behavioral fidelity and do not support the explicit optimization of local and string stability, which are increasingly important for the safe and efficient operation of autonomous vehicles (AVs). To address these limitations, we propose a Knowledge-Informed Deep Learning (KIDL) paradigm that distills the generalization capabilities of pre-trained Large Language Models (LLMs) into a lightweight and stability-aware neural architecture. LLMs are used to extract fundamental car-following knowledge beyond dataset-specific patterns, and this knowledge is transferred to a reliable, tractable, and computationally efficient model through knowledge distillation. KIDL also incorporates stability constraints directly into its training objective, ensuring that the resulting model not only emulates human-like behavior but also satisfies the local and string stability requirements essential for real-world AV deployment. We evaluate KIDL on the real-world NGSIM and HighD datasets, comparing its performance with representative physics-based, data-driven, and hybrid CFMs. Both empirical and theoretical results consistently demonstrate KIDL's superior behavioral generalization and traffic flow stability, offering a robust and scalable solution for next-generation traffic systems. 

---
# InfiGUI-R1: Advancing Multimodal GUI Agents from Reactive Actors to Deliberative Reasoners 

**Authors**: Yuhang Liu, Pengxiang Li, Congkai Xie, Xavier Hu, Xiaotian Han, Shengyu Zhang, Hongxia Yang, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14239)  

**Abstract**: Multimodal Large Language Models (MLLMs) have powered Graphical User Interface (GUI) Agents, showing promise in automating tasks on computing devices. Recent works have begun exploring reasoning in GUI tasks with encouraging results. However, many current approaches rely on manually designed reasoning templates, which may result in reasoning that is not sufficiently robust and adaptive for complex GUI environments. Meanwhile, some existing agents continue to operate as Reactive Actors, relying primarily on implicit reasoning that may lack sufficient depth for GUI tasks demanding planning and error recovery. We argue that advancing these agents requires a shift from reactive acting towards acting based on deliberate reasoning. To facilitate this transformation, we introduce InfiGUI-R1, an MLLM-based GUI agent developed through our Actor2Reasoner framework, a reasoning-centric, two-stage training approach designed to progressively evolve agents from Reactive Actors to Deliberative Reasoners. The first stage, Reasoning Injection, focuses on establishing a basic reasoner. We employ Spatial Reasoning Distillation to transfer cross-modal spatial reasoning capabilities from teacher models to MLLMs through trajectories with explicit reasoning steps, enabling models to integrate GUI visual-spatial information with logical reasoning before action generation. The second stage, Deliberation Enhancement, refines the basic reasoner into a deliberative one using Reinforcement Learning. This stage introduces two approaches: Sub-goal Guidance, which rewards models for generating accurate intermediate sub-goals, and Error Recovery Scenario Construction, which creates failure-and-recovery training scenarios from identified prone-to-error steps. Experimental results show InfiGUI-R1 achieves strong performance in GUI grounding and trajectory tasks. Resources at this https URL. 

---
# Assessing AI-Generated Questions' Alignment with Cognitive Frameworks in Educational Assessment 

**Authors**: Antoun Yaacoub, Jérôme Da-Rugna, Zainab Assaghir  

**Link**: [PDF](https://arxiv.org/pdf/2504.14232)  

**Abstract**: This study evaluates the integration of Bloom's Taxonomy into OneClickQuiz, an Artificial Intelligence (AI) driven plugin for automating Multiple-Choice Question (MCQ) generation in Moodle. Bloom's Taxonomy provides a structured framework for categorizing educational objectives into hierarchical cognitive levels. Our research investigates whether incorporating this taxonomy can improve the alignment of AI-generated questions with specific cognitive objectives. We developed a dataset of 3691 questions categorized according to Bloom's levels and employed various classification models-Multinomial Logistic Regression, Naive Bayes, Linear Support Vector Classification (SVC), and a Transformer-based model (DistilBERT)-to evaluate their effectiveness in categorizing questions. Our results indicate that higher Bloom's levels generally correlate with increased question length, Flesch-Kincaid Grade Level (FKGL), and Lexical Density (LD), reflecting the increased complexity of higher cognitive demands. Multinomial Logistic Regression showed varying accuracy across Bloom's levels, performing best for "Knowledge" and less accurately for higher-order levels. Merging higher-level categories improved accuracy for complex cognitive tasks. Naive Bayes and Linear SVC also demonstrated effective classification for lower levels but struggled with higher-order tasks. DistilBERT achieved the highest performance, significantly improving classification of both lower and higher-order cognitive levels, achieving an overall validation accuracy of 91%. This study highlights the potential of integrating Bloom's Taxonomy into AI-driven assessment tools and underscores the advantages of advanced models like DistilBERT for enhancing educational content generation. 

---
# Pets: General Pattern Assisted Architecture For Time Series Analysis 

**Authors**: Xiangkai Ma, Xiaobin Hong, Wenzhong Li, Sanglu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14209)  

**Abstract**: Time series analysis has found widespread applications in areas such as weather forecasting, anomaly detection, and healthcare. However, real-world sequential data often exhibit a superimposed state of various fluctuation patterns, including hourly, daily, and monthly frequencies. Traditional decomposition techniques struggle to effectively disentangle these multiple fluctuation patterns from the seasonal components, making time series analysis challenging. Surpassing the existing multi-period decoupling paradigms, this paper introduces a novel perspective based on energy distribution within the temporal-spectrum space. By adaptively quantifying observed sequences into continuous frequency band intervals, the proposed approach reconstructs fluctuation patterns across diverse periods without relying on domain-specific prior knowledge. Building upon this innovative strategy, we propose Pets, an enhanced architecture that is adaptable to arbitrary model structures. Pets integrates a Fluctuation Pattern Assisted (FPA) module and a Context-Guided Mixture of Predictors (MoP). The FPA module facilitates information fusion among diverse fluctuation patterns by capturing their dependencies and progressively modeling these patterns as latent representations at each layer. Meanwhile, the MoP module leverages these compound pattern representations to guide and regulate the reconstruction of distinct fluctuations hierarchically. Pets achieves state-of-the-art performance across various tasks, including forecasting, imputation, anomaly detection, and classification, while demonstrating strong generalization and robustness. 

---
# AI Idea Bench 2025: AI Research Idea Generation Benchmark 

**Authors**: Yansheng Qiu, Haoquan Zhang, Zhaopan Xu, Ming Li, Diping Song, Zheng Wang, Kaipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14191)  

**Abstract**: Large-scale Language Models (LLMs) have revolutionized human-AI interaction and achieved significant success in the generation of novel ideas. However, current assessments of idea generation overlook crucial factors such as knowledge leakage in LLMs, the absence of open-ended benchmarks with grounded truth, and the limited scope of feasibility analysis constrained by prompt design. These limitations hinder the potential of uncovering groundbreaking research ideas. In this paper, we present AI Idea Bench 2025, a framework designed to quantitatively evaluate and compare the ideas generated by LLMs within the domain of AI research from diverse perspectives. The framework comprises a comprehensive dataset of 3,495 AI papers and their associated inspired works, along with a robust evaluation methodology. This evaluation system gauges idea quality in two dimensions: alignment with the ground-truth content of the original papers and judgment based on general reference material. AI Idea Bench 2025's benchmarking system stands to be an invaluable resource for assessing and comparing idea-generation techniques, thereby facilitating the automation of scientific discovery. 

---
# Direct Advantage Regression: Aligning LLMs with Online AI Reward 

**Authors**: Li He, He Zhao, Stephen Wan, Dadong Wang, Lina Yao, Tongliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14177)  

**Abstract**: Online AI Feedback (OAIF) presents a promising alternative to Reinforcement Learning from Human Feedback (RLHF) by utilizing online AI preference in aligning language models (LLMs). However, the straightforward replacement of humans with AI deprives LLMs from learning more fine-grained AI supervision beyond binary signals. In this paper, we propose Direct Advantage Regression (DAR), a simple alignment algorithm using online AI reward to optimize policy improvement through weighted supervised fine-tuning. As an RL-free approach, DAR maintains theoretical consistency with online RLHF pipelines while significantly reducing implementation complexity and improving learning efficiency. Our empirical results underscore that AI reward is a better form of AI supervision consistently achieving higher human-AI agreement as opposed to AI preference. Additionally, evaluations using GPT-4-Turbo and MT-bench show that DAR outperforms both OAIF and online RLHF baselines. 

---
# Adaptation Method for Misinformation Identification 

**Authors**: Yangping Chen, Weijie Shi, Mengze Li, Yue Cui, Hao Chen, Jia Zhu, Jiajie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14171)  

**Abstract**: Multimodal fake news detection plays a crucial role in combating online misinformation. Unfortunately, effective detection methods rely on annotated labels and encounter significant performance degradation when domain shifts exist between training (source) and test (target) data. To address the problems, we propose ADOSE, an Active Domain Adaptation (ADA) framework for multimodal fake news detection which actively annotates a small subset of target samples to improve detection performance. To identify various deceptive patterns in cross-domain settings, we design multiple expert classifiers to learn dependencies across different modalities. These classifiers specifically target the distinct deception patterns exhibited in fake news, where two unimodal classifiers capture knowledge errors within individual modalities while one cross-modal classifier identifies semantic inconsistencies between text and images. To reduce annotation costs from the target domain, we propose a least-disagree uncertainty selector with a diversity calculator for selecting the most informative samples. The selector leverages prediction disagreement before and after perturbations by multiple classifiers as an indicator of uncertain samples, whose deceptive patterns deviate most from source domains. It further incorporates diversity scores derived from multi-view features to ensure the chosen samples achieve maximal coverage of target domain features. The extensive experiments on multiple datasets show that ADOSE outperforms existing ADA methods by 2.72\% $\sim$ 14.02\%, indicating the superiority of our model. 

---
# TALES: Text Adventure Learning Environment Suite 

**Authors**: Christopher Zhang Cui, Xingdi Yuan, Zhang Xiao, Prithviraj Ammanabrolu, Marc-Alexandre Côté  

**Link**: [PDF](https://arxiv.org/pdf/2504.14128)  

**Abstract**: Reasoning is an essential skill to enable Large Language Models (LLMs) to interact with the world. As tasks become more complex, they demand increasingly sophisticated and diverse reasoning capabilities for sequential decision-making, requiring structured reasoning over the context history to determine the next best action. We introduce TALES, a diverse collection of synthetic and human-written text-adventure games designed to challenge and evaluate diverse reasoning capabilities. We present results over a range of LLMs, open- and closed-weights, performing a qualitative analysis on the top performing models. Despite an impressive showing on synthetic games, even the top LLM-driven agents fail to achieve 15% on games designed for human enjoyment. Code and visualization of the experiments can be found at this https URL. 

---
# Large Language Model Enhanced Particle Swarm Optimization for Hyperparameter Tuning for Deep Learning Models 

**Authors**: Saad Hameed, Basheer Qolomany, Samir Brahim Belhaouari, Mohamed Abdallah, Junaid Qadir, Ala Al-Fuqaha  

**Link**: [PDF](https://arxiv.org/pdf/2504.14126)  

**Abstract**: Determining the ideal architecture for deep learning models, such as the number of layers and neurons, is a difficult and resource-intensive process that frequently relies on human tuning or computationally costly optimization approaches. While Particle Swarm Optimization (PSO) and Large Language Models (LLMs) have been individually applied in optimization and deep learning, their combined use for enhancing convergence in numerical optimization tasks remains underexplored. Our work addresses this gap by integrating LLMs into PSO to reduce model evaluations and improve convergence for deep learning hyperparameter tuning. The proposed LLM-enhanced PSO method addresses the difficulties of efficiency and convergence by using LLMs (particularly ChatGPT-3.5 and Llama3) to improve PSO performance, allowing for faster achievement of target objectives. Our method speeds up search space exploration by substituting underperforming particle placements with best suggestions offered by LLMs. Comprehensive experiments across three scenarios -- (1) optimizing the Rastrigin function, (2) using Long Short-Term Memory (LSTM) networks for time series regression, and (3) using Convolutional Neural Networks (CNNs) for material classification -- show that the method significantly improves convergence rates and lowers computational costs. Depending on the application, computational complexity is lowered by 20% to 60% compared to traditional PSO methods. Llama3 achieved a 20% to 40% reduction in model calls for regression tasks, whereas ChatGPT-3.5 reduced model calls by 60% for both regression and classification tasks, all while preserving accuracy and error rates. This groundbreaking methodology offers a very efficient and effective solution for optimizing deep learning models, leading to substantial computational performance improvements across a wide range of applications. 

---
# Bayesian Principles Improve Prompt Learning In Vision-Language Models 

**Authors**: Mingyu Kim, Jongwoo Ko, Mijung Park  

**Link**: [PDF](https://arxiv.org/pdf/2504.14123)  

**Abstract**: Prompt learning is a popular fine-tuning method for vision-language models due to its efficiency. It requires a small number of additional learnable parameters while significantly enhancing performance on target tasks. However, most existing methods suffer from overfitting to fine-tuning data, yielding poor generalizability. To address this, we propose a new training objective function based on a Bayesian learning principle to balance adaptability and generalizability. We derive a prior over the logits, where the mean function is parameterized by the pre-trained model, while the posterior corresponds to the fine-tuned model. This objective establishes a balance by allowing the fine-tuned model to adapt to downstream tasks while remaining close to the pre-trained model. 

---
# CODECRASH: Stress Testing LLM Reasoning under Structural and Semantic Perturbations 

**Authors**: Man Ho Lam, Chaozheng Wang, Jen-tse Huang, Michael R. Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14119)  

**Abstract**: Large Language Models (LLMs) have recently showcased strong capabilities in code-related tasks, yet their robustness in code comprehension and reasoning remains underexplored. In this paper, we present CodeCrash, a unified benchmark that evaluates LLM robustness under code structural and textual distraction perturbations, applied to two established benchmarks -- CRUXEval and LiveCodeBench -- across both input and output prediction tasks. We evaluate seventeen LLMs using direct and Chain-of-Thought inference to systematically analyze their robustness, identify primary reasons for performance degradation, and highlight failure modes. Our findings reveal the fragility of LLMs under structural noise and the inherent reliance on natural language cues, highlighting critical robustness issues of LLMs in code execution and understanding. Additionally, we examine three Large Reasoning Models (LRMs) and discover the severe vulnerability of self-reflective reasoning mechanisms that lead to reasoning collapse. CodeCrash provides a principled framework for stress-testing LLMs in code understanding, offering actionable directions for future evaluation and benchmarking. The code of CodeCrash and the robustness leaderboard are publicly available at this https URL . 

---
# Linking forward-pass dynamics in Transformers and real-time human processing 

**Authors**: Jennifer Hu, Michael A. Lepori, Michael Franke  

**Link**: [PDF](https://arxiv.org/pdf/2504.14107)  

**Abstract**: Modern AI models are increasingly being used as theoretical tools to study human cognition. One dominant approach is to evaluate whether human-derived measures (such as offline judgments or real-time processing) are predicted by a model's output: that is, the end-product of forward pass(es) through the network. At the same time, recent advances in mechanistic interpretability have begun to reveal the internal processes that give rise to model outputs, raising the question of whether models and humans might arrive at outputs using similar "processing strategies". Here, we investigate the link between real-time processing in humans and "layer-time" dynamics in Transformer models. Across five studies spanning domains and modalities, we test whether the dynamics of computation in a single forward pass of pre-trained Transformers predict signatures of processing in humans, above and beyond properties of the model's output probability distribution. We consistently find that layer-time dynamics provide additional predictive power on top of output measures. Our results suggest that Transformer processing and human processing may be facilitated or impeded by similar properties of an input stimulus, and this similarity has emerged through general-purpose objectives such as next-token prediction or image recognition. Our work suggests a new way of using AI models to study human cognition: not just as a black box mapping stimuli to responses, but potentially also as explicit processing models. 

---
# Think Deep, Think Fast: Investigating Efficiency of Verifier-free Inference-time-scaling Methods 

**Authors**: Junlin Wang, Shang Zhu, Jon Saad-Falcon, Ben Athiwaratkun, Qingyang Wu, Jue Wang, Shuaiwen Leon Song, Ce Zhang, Bhuwan Dhingra, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2504.14047)  

**Abstract**: There is intense interest in investigating how inference time compute (ITC) (e.g. repeated sampling, refinements, etc) can improve large language model (LLM) capabilities. At the same time, recent breakthroughs in reasoning models, such as Deepseek-R1, unlock the opportunity for reinforcement learning to improve LLM reasoning skills. An in-depth understanding of how ITC interacts with reasoning across different models could provide important guidance on how to further advance the LLM frontier. This work conducts a comprehensive analysis of inference-time scaling methods for both reasoning and non-reasoning models on challenging reasoning tasks. Specifically, we focus our research on verifier-free inference time-scaling methods due to its generalizability without needing a reward model. We construct the Pareto frontier of quality and efficiency. We find that non-reasoning models, even with an extremely high inference budget, still fall substantially behind reasoning models. For reasoning models, majority voting proves to be a robust inference strategy, generally competitive or outperforming other more sophisticated ITC methods like best-of-N and sequential revisions, while the additional inference compute offers minimal improvements. We further perform in-depth analyses of the association of key response features (length and linguistic markers) with response quality, with which we can improve the existing ITC methods. We find that correct responses from reasoning models are typically shorter and have fewer hedging and thinking markers (but more discourse markers) than the incorrect responses. 

---
# Metacognition and Uncertainty Communication in Humans and Large Language Models 

**Authors**: Mark Steyvers, Megan A.K. Peters  

**Link**: [PDF](https://arxiv.org/pdf/2504.14045)  

**Abstract**: Metacognition, the capacity to monitor and evaluate one's own knowledge and performance, is foundational to human decision-making, learning, and communication. As large language models (LLMs) become increasingly embedded in high-stakes decision contexts, it is critical to assess whether, how, and to what extent they exhibit metacognitive abilities. Here, we provide an overview of current knowledge of LLMs' metacognitive capacities, how they might be studied, and how they relate to our knowledge of metacognition in humans. We show that while humans and LLMs can sometimes appear quite aligned in their metacognitive capacities and behaviors, it is clear many differences remain. Attending to these differences is crucial not only for enhancing human-AI collaboration, but also for promoting the development of more capable and trustworthy artificial systems. Finally, we discuss how endowing future LLMs with more sensitive and more calibrated metacognition may also help them develop new capacities such as more efficient learning, self-direction, and curiosity. 

---
# Multi-Stage Retrieval for Operational Technology Cybersecurity Compliance Using Large Language Models: A Railway Casestudy 

**Authors**: Regan Bolton, Mohammadreza Sheikhfathollahi, Simon Parkinson, Dan Basher, Howard Parkinson  

**Link**: [PDF](https://arxiv.org/pdf/2504.14044)  

**Abstract**: Operational Technology Cybersecurity (OTCS) continues to be a dominant challenge for critical infrastructure such as railways. As these systems become increasingly vulnerable to malicious attacks due to digitalization, effective documentation and compliance processes are essential to protect these safety-critical systems. This paper proposes a novel system that leverages Large Language Models (LLMs) and multi-stage retrieval to enhance the compliance verification process against standards like IEC 62443 and the rail-specific IEC 63452. We first evaluate a Baseline Compliance Architecture (BCA) for answering OTCS compliance queries, then develop an extended approach called Parallel Compliance Architecture (PCA) that incorporates additional context from regulatory standards. Through empirical evaluation comparing OpenAI-gpt-4o and Claude-3.5-haiku models in these architectures, we demonstrate that the PCA significantly improves both correctness and reasoning quality in compliance verification. Our research establishes metrics for response correctness, logical reasoning, and hallucination detection, highlighting the strengths and limitations of using LLMs for compliance verification in railway cybersecurity. The results suggest that retrieval-augmented approaches can significantly improve the efficiency and accuracy of compliance assessments, particularly valuable in an industry facing a shortage of cybersecurity expertise. 

---
# Going Whole Hog: A Philosophical Defense of AI Cognition 

**Authors**: Herman Cappelen, Josh Dever  

**Link**: [PDF](https://arxiv.org/pdf/2504.13988)  

**Abstract**: This work defends the 'Whole Hog Thesis': sophisticated Large Language Models (LLMs) like ChatGPT are full-blown linguistic and cognitive agents, possessing understanding, beliefs, desires, knowledge, and intentions. We argue against prevailing methodologies in AI philosophy, rejecting starting points based on low-level computational details ('Just an X' fallacy) or pre-existing theories of mind. Instead, we advocate starting with simple, high-level observations of LLM behavior (e.g., answering questions, making suggestions) -- defending this data against charges of metaphor, loose talk, or pretense. From these observations, we employ 'Holistic Network Assumptions' -- plausible connections between mental capacities (e.g., answering implies knowledge, knowledge implies belief, action implies intention) -- to argue for the full suite of cognitive states. We systematically rebut objections based on LLM failures (hallucinations, planning/reasoning errors), arguing these don't preclude agency, often mirroring human fallibility. We address numerous 'Games of Lacks', arguing that LLMs do not lack purported necessary conditions for cognition (e.g., semantic grounding, embodiment, justification, intrinsic intentionality) or that these conditions are not truly necessary, often relying on anti-discriminatory arguments comparing LLMs to diverse human capacities. Our approach is evidential, not functionalist, and deliberately excludes consciousness. We conclude by speculating on the possibility of LLMs possessing 'alien' contents beyond human conceptual schemes. 

---
# Birds of a Different Feather Flock Together: Exploring Opportunities and Challenges in Animal-Human-Machine Teaming 

**Authors**: Myke C. Cohen, David A. Grimm, Reuth Mirsky, Xiaoyun Yin  

**Link**: [PDF](https://arxiv.org/pdf/2504.13973)  

**Abstract**: Animal-Human-Machine (AHM) teams are a type of hybrid intelligence system wherein interactions between a human, AI-enabled machine, and animal members can result in unique capabilities greater than the sum of their parts. This paper calls for a systematic approach to studying the design of AHM team structures to optimize performance and overcome limitations in various applied settings. We consider the challenges and opportunities in investigating the synergistic potential of AHM team members by introducing a set of dimensions of AHM team functioning to effectively utilize each member's strengths while compensating for individual weaknesses. Using three representative examples of such teams -- security screening, search-and-rescue, and guide dogs -- the paper illustrates how AHM teams can tackle complex tasks. We conclude with open research directions that this multidimensional approach presents for studying hybrid human-AI systems beyond AHM teams. 

---
# Evaluation and Incident Prevention in an Enterprise AI Assistant 

**Authors**: Akash V. Maharaj, David Arbour, Daniel Lee, Uttaran Bhattacharya, Anup Rao, Austin Zane, Avi Feller, Kun Qian, Yunyao Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.13924)  

**Abstract**: Enterprise AI Assistants are increasingly deployed in domains where accuracy is paramount, making each erroneous output a potentially significant incident. This paper presents a comprehensive framework for monitoring, benchmarking, and continuously improving such complex, multi-component systems under active development by multiple teams. Our approach encompasses three key elements: (1) a hierarchical ``severity'' framework for incident detection that identifies and categorizes errors while attributing component-specific error rates, facilitating targeted improvements; (2) a scalable and principled methodology for benchmark construction, evaluation, and deployment, designed to accommodate multiple development teams, mitigate overfitting risks, and assess the downstream impact of system modifications; and (3) a continual improvement strategy leveraging multidimensional evaluation, enabling the identification and implementation of diverse enhancement opportunities. By adopting this holistic framework, organizations can systematically enhance the reliability and performance of their AI Assistants, ensuring their efficacy in critical enterprise environments. We conclude by discussing how this multifaceted evaluation approach opens avenues for various classes of enhancements, paving the way for more robust and trustworthy AI systems. 

---
# The Model Counting Competitions 2021-2023 

**Authors**: Johannes K. Fichte, Markus Hecher  

**Link**: [PDF](https://arxiv.org/pdf/2504.13842)  

**Abstract**: Modern society is full of computational challenges that rely on probabilistic reasoning, statistics, and combinatorics. Interestingly, many of these questions can be formulated by encoding them into propositional formulas and then asking for its number of models. With a growing interest in practical problem-solving for tasks that involve model counting, the community established the Model Counting (MC) Competition in fall of 2019 with its first iteration in 2020. The competition aims at advancing applications, identifying challenging benchmarks, fostering new solver development, and enhancing existing solvers for model counting problems and their variants. The first iteration, brought together various researchers, identified challenges, and inspired numerous new applications. In this paper, we present a comprehensive overview of the 2021-2023 iterations of the Model Counting Competition. We detail its execution and outcomes. The competition comprised four tracks, each focusing on a different variant of the model counting problem. The first track centered on the model counting problem (MC), which seeks the count of models for a given propositional formula. The second track challenged developers to submit programs capable of solving the weighted model counting problem (WMC). The third track was dedicated to projected model counting (PMC). Finally, we initiated a track that combined projected and weighted model counting (PWMC). The competition continued with a high level of participation, with seven to nine solvers submitted in various different version and based on quite diverging techniques. 

---
# Roll the dice & look before you leap: Going beyond the creative limits of next-token prediction 

**Authors**: Vaishnavh Nagarajan, Chen Henry Wu, Charles Ding, Aditi Raghunathan  

**Link**: [PDF](https://arxiv.org/pdf/2504.15266)  

**Abstract**: We design a suite of minimal algorithmic tasks that are a loose abstraction of open-ended real-world tasks. This allows us to cleanly and controllably quantify the creative limits of the present-day language model. Much like real-world tasks that require a creative, far-sighted leap of thought, our tasks require an implicit, open-ended stochastic planning step that either (a) discovers new connections in an abstract knowledge graph (like in wordplay, drawing analogies, or research) or (b) constructs new patterns (like in designing math problems or new proteins). In these tasks, we empirically and conceptually argue how next-token learning is myopic and memorizes excessively; comparatively, multi-token approaches, namely teacherless training and diffusion models, excel in producing diverse and original output. Secondly, in our tasks, we find that to elicit randomness from the Transformer without hurting coherence, it is better to inject noise right at the input layer (via a method we dub hash-conditioning) rather than defer to temperature sampling from the output layer. Thus, our work offers a principled, minimal test-bed for analyzing open-ended creative skills, and offers new arguments for going beyond next-token learning and softmax-based sampling. We make part of the code available under this https URL 

---
# Bringing Diversity from Diffusion Models to Semantic-Guided Face Asset Generation 

**Authors**: Yunxuan Cai, Sitao Xiang, Zongjian Li, Haiwei Chen, Yajie Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.15259)  

**Abstract**: Digital modeling and reconstruction of human faces serve various applications. However, its availability is often hindered by the requirements of data capturing devices, manual labor, and suitable actors. This situation restricts the diversity, expressiveness, and control over the resulting models. This work aims to demonstrate that a semantically controllable generative network can provide enhanced control over the digital face modeling process. To enhance diversity beyond the limited human faces scanned in a controlled setting, we introduce a novel data generation pipeline that creates a high-quality 3D face database using a pre-trained diffusion model. Our proposed normalization module converts synthesized data from the diffusion model into high-quality scanned data. Using the 44,000 face models we obtained, we further developed an efficient GAN-based generator. This generator accepts semantic attributes as input, and generates geometry and albedo. It also allows continuous post-editing of attributes in the latent space. Our asset refinement component subsequently creates physically-based facial assets. We introduce a comprehensive system designed for creating and editing high-quality face assets. Our proposed model has undergone extensive experiment, comparison and evaluation. We also integrate everything into a web-based interactive tool. We aim to make this tool publicly available with the release of the paper. 

---
# Values in the Wild: Discovering and Analyzing Values in Real-World Language Model Interactions 

**Authors**: Saffron Huang, Esin Durmus, Miles McCain, Kunal Handa, Alex Tamkin, Jerry Hong, Michael Stern, Arushi Somani, Xiuruo Zhang, Deep Ganguli  

**Link**: [PDF](https://arxiv.org/pdf/2504.15236)  

**Abstract**: AI assistants can impart value judgments that shape people's decisions and worldviews, yet little is known empirically about what values these systems rely on in practice. To address this, we develop a bottom-up, privacy-preserving method to extract the values (normative considerations stated or demonstrated in model responses) that Claude 3 and 3.5 models exhibit in hundreds of thousands of real-world interactions. We empirically discover and taxonomize 3,307 AI values and study how they vary by context. We find that Claude expresses many practical and epistemic values, and typically supports prosocial human values while resisting values like "moral nihilism". While some values appear consistently across contexts (e.g. "transparency"), many are more specialized and context-dependent, reflecting the diversity of human interlocutors and their varied contexts. For example, "harm prevention" emerges when Claude resists users, "historical accuracy" when responding to queries about controversial events, "healthy boundaries" when asked for relationship advice, and "human agency" in technology ethics discussions. By providing the first large-scale empirical mapping of AI values in deployment, our work creates a foundation for more grounded evaluation and design of values in AI systems. 

---
# A Genetic Fuzzy-Enabled Framework on Robotic Manipulation for In-Space Servicing 

**Authors**: Nathan Steffen, Wilhelm Louw, Nicholas Ernest, Timothy Arnett, Kelly Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2504.15226)  

**Abstract**: Automation of robotic systems for servicing in cislunar space is becoming extremely important as the number of satellites in orbit increases. Safety is critical in performing satellite maintenance, so the control techniques utilized must be trusted in addition to being highly efficient. In this work, Genetic Fuzzy Trees are combined with the widely used LQR control scheme via Thales' TrUE AI Toolkit to create a trusted and efficient controller for a two-degree-of-freedom planar robotic manipulator that would theoretically be used to perform satellite maintenance. It was found that Genetic Fuzzy-LQR is 18.5% more performant than optimal LQR on average, and that it is incredibly robust to uncertainty. 

---
# M$^2$AD: Multi-Sensor Multi-System Anomaly Detection through Global Scoring and Calibrated Thresholding 

**Authors**: Sarah Alnegheimish, Zelin He, Matthew Reimherr, Akash Chandrayan, Abhinav Pradhan, Luca D'Angelo  

**Link**: [PDF](https://arxiv.org/pdf/2504.15225)  

**Abstract**: With the widespread availability of sensor data across industrial and operational systems, we frequently encounter heterogeneous time series from multiple systems. Anomaly detection is crucial for such systems to facilitate predictive maintenance. However, most existing anomaly detection methods are designed for either univariate or single-system multivariate data, making them insufficient for these complex scenarios. To address this, we introduce M$^2$AD, a framework for unsupervised anomaly detection in multivariate time series data from multiple systems. M$^2$AD employs deep models to capture expected behavior under normal conditions, using the residuals as indicators of potential anomalies. These residuals are then aggregated into a global anomaly score through a Gaussian Mixture Model and Gamma calibration. We theoretically demonstrate that this framework can effectively address heterogeneity and dependencies across sensors and systems. Empirically, M$^2$AD outperforms existing methods in extensive evaluations by 21% on average, and its effectiveness is demonstrated on a large-scale real-world case study on 130 assets in Amazon Fulfillment Centers. Our code and results are available at this https URL. 

---
# Integrating Symbolic Execution into the Fine-Tuning of Code-Generating LLMs 

**Authors**: Marina Sakharova, Abhinav Anand, Mira Mezini  

**Link**: [PDF](https://arxiv.org/pdf/2504.15210)  

**Abstract**: Code-generating Large Language Models (LLMs) have become essential tools in modern software development, enhancing productivity and accelerating development. This paper aims to investigate the fine-tuning of code-generating LLMs using Reinforcement Learning and Direct Preference Optimization, further improving their performance. To achieve this, we enhance the training data for the reward model with the help of symbolic execution techniques, ensuring more comprehensive and objective data. With symbolic execution, we create a custom dataset that better captures the nuances in code evaluation. Our reward models, fine-tuned on this dataset, demonstrate significant improvements over the baseline, CodeRL, in estimating the quality of generated code. Our code-generating LLMs, trained with the help of reward model feedback, achieve similar results compared to the CodeRL benchmark. 

---
# A Causal Convolutional Low-rank Representation Model for Imputation of Water Quality Data 

**Authors**: Xin Liao, Bing Yang, Tan Dongli, Cai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15209)  

**Abstract**: The monitoring of water quality is a crucial part of environmental protection, and a large number of monitors are widely deployed to monitor water quality. Due to unavoidable factors such as data acquisition breakdowns, sensors and communication failures, water quality monitoring data suffers from missing values over time, resulting in High-Dimensional and Sparse (HDS) Water Quality Data (WQD). The simple and rough filling of the missing values leads to inaccurate results and affects the implementation of relevant measures. Therefore, this paper proposes a Causal convolutional Low-rank Representation (CLR) model for imputing missing WQD to improve the completeness of the WQD, which employs a two-fold idea: a) applying causal convolutional operation to consider the temporal dependence of the low-rank representation, thus incorporating temporal information to improve the imputation accuracy; and b) implementing a hyperparameters adaptation scheme to automatically adjust the best hyperparameters during model training, thereby reducing the tedious manual adjustment of hyper-parameters. Experimental studies on three real-world water quality datasets demonstrate that the proposed CLR model is superior to some of the existing state-of-the-art imputation models in terms of imputation accuracy and time cost, as well as indicating that the proposed model provides more reliable decision support for environmental monitoring. 

---
# Compute-Optimal LLMs Provably Generalize Better With Scale 

**Authors**: Marc Finzi, Sanyam Kapoor, Diego Granziol, Anming Gu, Christopher De Sa, J. Zico Kolter, Andrew Gordon Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2504.15208)  

**Abstract**: Why do larger language models generalize better? To investigate this question, we develop generalization bounds on the pretraining objective of large language models (LLMs) in the compute-optimal regime, as described by the Chinchilla scaling laws. We introduce a novel, fully empirical Freedman-type martingale concentration inequality that tightens existing bounds by accounting for the variance of the loss function. This generalization bound can be decomposed into three interpretable components: the number of parameters per token, the loss variance, and the quantization error at a fixed bitrate. As compute-optimal language models are scaled up, the number of parameters per data point remains constant; however, both the loss variance and the quantization error decrease, implying that larger models should have smaller generalization gaps. We examine why larger models tend to be more quantizable from an information theoretic perspective, showing that the rate at which they can integrate new information grows more slowly than their capacity on the compute-optimal frontier. From these findings we produce a scaling law for the generalization gap, with bounds that become predictably stronger with scale. 

---
# Support Evaluation for the TREC 2024 RAG Track: Comparing Human versus LLM Judges 

**Authors**: Nandan Thakur, Ronak Pradeep, Shivani Upadhyay, Daniel Campos, Nick Craswell, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15205)  

**Abstract**: Retrieval-augmented generation (RAG) enables large language models (LLMs) to generate answers with citations from source documents containing "ground truth", thereby reducing system hallucinations. A crucial factor in RAG evaluation is "support", whether the information in the cited documents supports the answer. To this end, we conducted a large-scale comparative study of 45 participant submissions on 36 topics to the TREC 2024 RAG Track, comparing an automatic LLM judge (GPT-4o) against human judges for support assessment. We considered two conditions: (1) fully manual assessments from scratch and (2) manual assessments with post-editing of LLM predictions. Our results indicate that for 56% of the manual from-scratch assessments, human and GPT-4o predictions match perfectly (on a three-level scale), increasing to 72% in the manual with post-editing condition. Furthermore, by carefully analyzing the disagreements in an unbiased study, we found that an independent human judge correlates better with GPT-4o than a human judge, suggesting that LLM judges can be a reliable alternative for support assessment. To conclude, we provide a qualitative analysis of human and GPT-4o errors to help guide future iterations of support assessment. 

---
# Zero-Shot, But at What Cost? Unveiling the Hidden Overhead of MILS's LLM-CLIP Framework for Image Captioning 

**Authors**: Yassir Benhammou, Alessandro Tiberio, Gabriel Trautmann, Suman Kalyan  

**Link**: [PDF](https://arxiv.org/pdf/2504.15199)  

**Abstract**: MILS (Multimodal Iterative LLM Solver) is a recently published framework that claims "LLMs can see and hear without any training" by leveraging an iterative, LLM-CLIP based approach for zero-shot image captioning. While this MILS approach demonstrates good performance, our investigation reveals that this success comes at a hidden, substantial computational cost due to its expensive multi-step refinement process. In contrast, alternative models such as BLIP-2 and GPT-4V achieve competitive results through a streamlined, single-pass approach. We hypothesize that the significant overhead inherent in MILS's iterative process may undermine its practical benefits, thereby challenging the narrative that zero-shot performance can be attained without incurring heavy resource demands. This work is the first to expose and quantify the trade-offs between output quality and computational cost in MILS, providing critical insights for the design of more efficient multimodal models. 

---
# Breast density in MRI: an AI-based quantification and relationship to assessment in mammography 

**Authors**: Yaqian Chen, Lin Li, Hanxue Gu, Haoyu Dong, Derek L. Nguyen, Allan D. Kirk, Maciej A. Mazurowski, E. Shelley Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15192)  

**Abstract**: Mammographic breast density is a well-established risk factor for breast cancer. Recently there has been interest in breast MRI as an adjunct to mammography, as this modality provides an orthogonal and highly quantitative assessment of breast tissue. However, its 3D nature poses analytic challenges related to delineating and aggregating complex structures across slices. Here, we applied an in-house machine-learning algorithm to assess breast density on normal breasts in three MRI datasets. Breast density was consistent across different datasets (0.104 - 0.114). Analysis across different age groups also demonstrated strong consistency across datasets and confirmed a trend of decreasing density with age as reported in previous studies. MR breast density was correlated with mammographic breast density, although some notable differences suggest that certain breast density components are captured only on MRI. Future work will determine how to integrate MR breast density with current tools to improve future breast cancer risk prediction. 

---
# Existing Industry Practice for the EU AI Act's General-Purpose AI Code of Practice Safety and Security Measures 

**Authors**: Lily Stelling, Mick Yang, Rokas Gipiškis, Leon Staufer, Ze Shen Chin, Siméon Campos, Michael Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.15181)  

**Abstract**: This report provides a detailed comparison between the measures proposed in the EU AI Act's General-Purpose AI (GPAI) Code of Practice (Third Draft) and current practices adopted by leading AI companies. As the EU moves toward enforcing binding obligations for GPAI model providers, the Code of Practice will be key to bridging legal requirements with concrete technical commitments. Our analysis focuses on the draft's Safety and Security section which is only relevant for the providers of the most advanced models (Commitments II.1-II.16) and excerpts from current public-facing documents quotes that are relevant to each individual measure.
We systematically reviewed different document types - including companies' frontier safety frameworks and model cards - from over a dozen companies, including OpenAI, Anthropic, Google DeepMind, Microsoft, Meta, Amazon, and others. This report is not meant to be an indication of legal compliance nor does it take any prescriptive viewpoint about the Code of Practice or companies' policies. Instead, it aims to inform the ongoing dialogue between regulators and GPAI model providers by surfacing evidence of precedent. 

---
# An Efficient Aerial Image Detection with Variable Receptive Fields 

**Authors**: Liu Wenbin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15165)  

**Abstract**: Aerial object detection using unmanned aerial vehicles (UAVs) faces critical challenges including sub-10px targets, dense occlusions, and stringent computational constraints. Existing detectors struggle to balance accuracy and efficiency due to rigid receptive fields and redundant architectures. To address these limitations, we propose Variable Receptive Field DETR (VRF-DETR), a transformer-based detector incorporating three key components: 1) Multi-Scale Context Fusion (MSCF) module that dynamically recalibrates features through adaptive spatial attention and gated multi-scale fusion, 2) Gated Convolution (GConv) layer enabling parameter-efficient local-context modeling via depthwise separable operations and dynamic gating, and 3) Gated Multi-scale Fusion (GMCF) Bottleneck that hierarchically disentangles occluded objects through cascaded global-local interactions. Experiments on VisDrone2019 demonstrate VRF-DETR achieves 51.4\% mAP\textsubscript{50} and 31.8\% mAP\textsubscript{50:95} with only 13.5M parameters. This work establishes a new efficiency-accuracy Pareto frontier for UAV-based detection tasks. 

---
# Landmark-Free Preoperative-to-Intraoperative Registration in Laparoscopic Liver Resection 

**Authors**: Jun Zhou, Bingchen Gao, Kai Wang, Jialun Pei, Pheng-Ann Heng, Jing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15152)  

**Abstract**: Liver registration by overlaying preoperative 3D models onto intraoperative 2D frames can assist surgeons in perceiving the spatial anatomy of the liver clearly for a higher surgical success rate. Existing registration methods rely heavily on anatomical landmark-based workflows, which encounter two major limitations: 1) ambiguous landmark definitions fail to provide efficient markers for registration; 2) insufficient integration of intraoperative liver visual information in shape deformation modeling. To address these challenges, in this paper, we propose a landmark-free preoperative-to-intraoperative registration framework utilizing effective self-supervised learning, termed \ourmodel. This framework transforms the conventional 3D-2D workflow into a 3D-3D registration pipeline, which is then decoupled into rigid and non-rigid registration subtasks. \ourmodel~first introduces a feature-disentangled transformer to learn robust correspondences for recovering rigid transformations. Further, a structure-regularized deformation network is designed to adjust the preoperative model to align with the intraoperative liver surface. This network captures structural correlations through geometry similarity modeling in a low-rank transformer network. To facilitate the validation of the registration performance, we also construct an in-vivo registration dataset containing liver resection videos of 21 patients, called \emph{P2I-LReg}, which contains 346 keyframes that provide a global view of the liver together with liver mask annotations and calibrated camera intrinsic parameters. Extensive experiments and user studies on both synthetic and in-vivo datasets demonstrate the superiority and potential clinical applicability of our method. 

---
# C2RUST-BENCH: A Minimized, Representative Dataset for C-to-Rust Transpilation Evaluation 

**Authors**: Melih Sirlanci, Carter Yagemann, Zhiqiang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15144)  

**Abstract**: Despite the effort in vulnerability detection over the last two decades, memory safety vulnerabilities continue to be a critical problem. Recent reports suggest that the key solution is to migrate to memory-safe languages. To this end, C-to-Rust transpilation becomes popular to resolve memory-safety issues in C programs. Recent works propose C-to-Rust transpilation frameworks; however, a comprehensive evaluation dataset is missing. Although one solution is to put together a large enough dataset, this increases the analysis time in automated frameworks as well as in manual efforts for some cases. In this work, we build a method to select functions from a large set to construct a minimized yet representative dataset to evaluate the C-to-Rust transpilation. We propose C2RUST-BENCH that contains 2,905 functions, which are representative of C-to-Rust transpilation, selected from 15,503 functions of real-world programs. 

---
# KGMEL: Knowledge Graph-Enhanced Multimodal Entity Linking 

**Authors**: Juyeon Kim, Geon Lee, Taeuk Kim, Kijung Shin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15135)  

**Abstract**: Entity linking (EL) aligns textual mentions with their corresponding entities in a knowledge base, facilitating various applications such as semantic search and question answering. Recent advances in multimodal entity linking (MEL) have shown that combining text and images can reduce ambiguity and improve alignment accuracy. However, most existing MEL methods overlook the rich structural information available in the form of knowledge-graph (KG) triples. In this paper, we propose KGMEL, a novel framework that leverages KG triples to enhance MEL. Specifically, it operates in three stages: (1) Generation: Produces high-quality triples for each mention by employing vision-language models based on its text and images. (2) Retrieval: Learns joint mention-entity representations, via contrastive learning, that integrate text, images, and (generated or KG) triples to retrieve candidate entities for each mention. (3) Reranking: Refines the KG triples of the candidate entities and employs large language models to identify the best-matching entity for the mention. Extensive experiments on benchmark datasets demonstrate that KGMEL outperforms existing methods. Our code and datasets are available at: this https URL. 

---
# EasyEdit2: An Easy-to-use Steering Framework for Editing Large Language Models 

**Authors**: Ziwen Xu, Shuxun Wang, Kewei Xu, Haoming Xu, Mengru Wang, Xinle Deng, Yunzhi Yao, Guozhou Zheng, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15133)  

**Abstract**: In this paper, we introduce EasyEdit2, a framework designed to enable plug-and-play adjustability for controlling Large Language Model (LLM) behaviors. EasyEdit2 supports a wide range of test-time interventions, including safety, sentiment, personality, reasoning patterns, factuality, and language features. Unlike its predecessor, EasyEdit2 features a new architecture specifically designed for seamless model steering. It comprises key modules such as the steering vector generator and the steering vector applier, which enable automatic generation and application of steering vectors to influence the model's behavior without modifying its parameters. One of the main advantages of EasyEdit2 is its ease of use-users do not need extensive technical knowledge. With just a single example, they can effectively guide and adjust the model's responses, making precise control both accessible and efficient. Empirically, we report model steering performance across different LLMs, demonstrating the effectiveness of these techniques. We have released the source code on GitHub at this https URL along with a demonstration notebook. In addition, we provide a demo video at this https URL for a quick introduction. 

---
# Neural ATTF: A Scalable Solution to Lifelong Multi-Agent Path Planning 

**Authors**: Kushal Shah, Jihyun Park, Seung-Kyum Choi  

**Link**: [PDF](https://arxiv.org/pdf/2504.15130)  

**Abstract**: Multi-Agent Pickup and Delivery (MAPD) is a fundamental problem in robotics, particularly in applications such as warehouse automation and logistics. Existing solutions often face challenges in scalability, adaptability, and efficiency, limiting their applicability in dynamic environments with real-time planning requirements. This paper presents Neural ATTF (Adaptive Task Token Framework), a new algorithm that combines a Priority Guided Task Matching (PGTM) Module with Neural STA* (Space-Time A*), a data-driven path planning method. Neural STA* enhances path planning by enabling rapid exploration of the search space through guided learned heuristics and ensures collision avoidance under dynamic constraints. PGTM prioritizes delayed agents and dynamically assigns tasks by prioritizing agents nearest to these tasks, optimizing both continuity and system throughput. Experimental evaluations against state-of-the-art MAPD algorithms, including TPTS, CENTRAL, RMCA, LNS-PBS, and LNS-wPBS, demonstrate the superior scalability, solution quality, and computational efficiency of Neural ATTF. These results highlight the framework's potential for addressing the critical demands of complex, real-world multi-agent systems operating in high-demand, unpredictable settings. 

---
# A General Infrastructure and Workflow for Quadrotor Deep Reinforcement Learning and Reality Deployment 

**Authors**: Kangyao Huang, Hao Wang, Yu Luo, Jingyu Chen, Jintao Chen, Xiangkui Zhang, Xiangyang Ji, Huaping Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15129)  

**Abstract**: Deploying robot learning methods to a quadrotor in unstructured outdoor environments is an exciting task. Quadrotors operating in real-world environments by learning-based methods encounter several challenges: a large amount of simulator generated data required for training, strict demands for real-time processing onboard, and the sim-to-real gap caused by dynamic and noisy conditions. Current works have made a great breakthrough in applying learning-based methods to end-to-end control of quadrotors, but rarely mention the infrastructure system training from scratch and deploying to reality, which makes it difficult to reproduce methods and applications. To bridge this gap, we propose a platform that enables the seamless transfer of end-to-end deep reinforcement learning (DRL) policies. We integrate the training environment, flight dynamics control, DRL algorithms, the MAVROS middleware stack, and hardware into a comprehensive workflow and architecture that enables quadrotors' policies to be trained from scratch to real-world deployment in several minutes. Our platform provides rich types of environments including hovering, dynamic obstacle avoidance, trajectory tracking, balloon hitting, and planning in unknown environments, as a physical experiment benchmark. Through extensive empirical validation, we demonstrate the efficiency of proposed sim-to-real platform, and robust outdoor flight performance under real-world perturbations. Details can be found from our website this https URL. 

---
# Kuwain 1.5B: An Arabic SLM via Language Injection 

**Authors**: Khalil Hennara, Sara Chrouf, Mohamed Motaism Hamed, Zeina Aldallal, Omar Hadid, Safwan AlModhayan  

**Link**: [PDF](https://arxiv.org/pdf/2504.15120)  

**Abstract**: Enhancing existing models with new knowledge is a crucial aspect of AI development. This paper introduces a novel method for integrating a new language into a large language model (LLM). Our approach successfully incorporates a previously unseen target language into an existing LLM without compromising its prior knowledge. We trained a tiny model with 1.5 billion parameters named Kuwain by injecting the Arabic language into a small open-source model mainly trained in English. Our method demonstrates significant improvements in Arabic language performance, with an average 8% improvement across various benchmarks, while retaining the model's existing knowledge with a minimum amount of the original model's data. This offers a cost-effective alternative to training a comprehensive model in both English and Arabic. The results highlight the potential for efficient, targeted language model expansion without extensive retraining or resource-intensive processes. 

---
# A triple-branch network for latent fingerprint enhancement guided by orientation fields and minutiae 

**Authors**: Yurun Wang, Zerong Qi, Shujun Fu, Mingzheng Hu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15105)  

**Abstract**: Latent fingerprint enhancement is a critical step in the process of latent fingerprint identification. Existing deep learning-based enhancement methods still fall short of practical application requirements, particularly in restoring low-quality fingerprint regions. Recognizing that different regions of latent fingerprints require distinct enhancement strategies, we propose a Triple Branch Spatial Fusion Network (TBSFNet), which simultaneously enhances different regions of the image using tailored strategies. Furthermore, to improve the generalization capability of the network, we integrate orientation field and minutiae-related modules into TBSFNet and introduce a Multi-Level Feature Guidance Network (MLFGNet). Experimental results on the MOLF and MUST datasets demonstrate that MLFGNet outperforms existing enhancement algorithms. 

---
# NeuGaze: Reshaping the future BCI 

**Authors**: Yiqian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15101)  

**Abstract**: Traditional brain-computer interfaces (BCIs), reliant on costly electroencephalography or invasive implants, struggle with complex human-computer interactions due to setup complexity and limited precision. We present NeuGaze, a novel webcam-based system that leverages eye gaze, head movements, and facial expressions to enable intuitive, real-time control using only a standard 30 Hz webcam, often pre-installed in laptops. Requiring minimal calibration, NeuGaze achieves performance comparable to conventional inputs, supporting precise cursor navigation, key triggering via an efficient skill wheel, and dynamic gaming interactions, such as defeating formidable opponents in first-person games. By harnessing preserved neck-up functionalities in motor-impaired individuals, NeuGaze eliminates the need for specialized hardware, offering a low-cost, accessible alternative to BCIs. This paradigm empowers diverse applications, from assistive technology to entertainment, redefining human-computer interaction for motor-impaired users. Project is at \href{this https URL}{this http URL}. 

---
# Fast-Slow Co-advancing Optimizer: Toward Harmonious Adversarial Training of GAN 

**Authors**: Lin Wang, Xiancheng Wang, Rui Wang, Zhibo Zhang, Minghang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.15099)  

**Abstract**: Up to now, the training processes of typical Generative Adversarial Networks (GANs) are still particularly sensitive to data properties and hyperparameters, which may lead to severe oscillations, difficulties in convergence, or even failures to converge, especially when the overall variances of the training sets are large. These phenomena are often attributed to the training characteristics of such networks. Aiming at the problem, this paper develops a new intelligent optimizer, Fast-Slow Co-advancing Optimizer (FSCO), which employs reinforcement learning in the training process of GANs to make training easier. Specifically, this paper allows the training step size to be controlled by an agent to improve training stability, and makes the training process more intelligent with variable learning rates, making GANs less sensitive to step size. Experiments have been conducted on three benchmark datasets to verify the effectiveness of the developed FSCO. 

---
# Rethinking the Potential of Multimodality in Collaborative Problem Solving Diagnosis with Large Language Models 

**Authors**: K. Wong, B. Wu, S. Bulathwela, M. Cukurova  

**Link**: [PDF](https://arxiv.org/pdf/2504.15093)  

**Abstract**: Detecting collaborative and problem-solving behaviours from digital traces to interpret students' collaborative problem solving (CPS) competency is a long-term goal in the Artificial Intelligence in Education (AIEd) field. Although multimodal data and advanced models are argued to have the potential to detect complex CPS behaviours, empirical evidence on their value remains limited with some contrasting evidence. In this study, we investigated the potential of multimodal data to improve model performance in diagnosing 78 secondary school students' CPS subskills and indicators in authentic educational settings. In particular, text embeddings from verbal data and acoustic embeddings from audio data were used in a multimodal classification model for CPS diagnosis. Both unimodal and multimodal transformer-based models outperformed traditional models in detecting CPS classes. Although the inclusion of multimodality did not improve the performance of traditional unimodal models, its integration into transformer-based models demonstrated improved performance for diagnosing social-cognitive CPS classes compared to unimodal transformer-based models. Based on the results, the paper argues that multimodality and the selection of a particular modelling technique should not be taken for granted to achieve the best performance in the automated detection of every CPS subskill and indicator. Rather, their value is limited to certain types of CPS indicators, affected by the complexity of the labels, and dependent on the composition of indicators in the dataset. We conclude the paper by discussing the required nuance when considering the value of LLMs and multimodality in automated CPS diagnosis, highlighting the need for human-AI complementarity, and proposing the exploration of relevant model architectures and techniques to improve CPS diagnosis in authentic educational contexts. 

---
# Federated Latent Factor Model for Bias-Aware Recommendation with Privacy-Preserving 

**Authors**: Junxiang Gao, Yixin Ran, Jia Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.15090)  

**Abstract**: A recommender system (RS) aims to provide users with personalized item recommendations, enhancing their overall experience. Traditional RSs collect and process all user data on a central server. However, this centralized approach raises significant privacy concerns, as it increases the risk of data breaches and privacy leakages, which are becoming increasingly unacceptable to privacy-sensitive users. To address these privacy challenges, federated learning has been integrated into RSs, ensuring that user data remains secure. In centralized RSs, the issue of rating bias is effectively addressed by jointly analyzing all users' raw interaction data. However, this becomes a significant challenge in federated RSs, as raw data is no longer accessible due to privacy-preserving constraints. To overcome this problem, we propose a Federated Bias-Aware Latent Factor (FBALF) model. In FBALF, training bias is explicitly incorporated into every local model's loss function, allowing for the effective elimination of rating bias without compromising data privacy. Extensive experiments conducted on three real-world datasets demonstrate that FBALF achieves significantly higher recommendation accuracy compared to other state-of-the-art federated RSs. 

---
# Empowering AI to Generate Better AI Code: Guided Generation of Deep Learning Projects with LLMs 

**Authors**: Chen Xie, Mingsheng Jiao, Xiaodong Gu, Beijun Shen  

**Link**: [PDF](https://arxiv.org/pdf/2504.15080)  

**Abstract**: While large language models (LLMs) have been widely applied to code generation, they struggle with generating entire deep learning projects, which are characterized by complex structures, longer functions, and stronger reliance on domain knowledge than general-purpose code. An open-domain LLM often lacks coherent contextual guidance and domain expertise for specific projects, making it challenging to produce complete code that fully meets user requirements.
In this paper, we propose a novel planning-guided code generation method, DLCodeGen, tailored for generating deep learning projects. DLCodeGen predicts a structured solution plan, offering global guidance for LLMs to generate the project. The generated plan is then leveraged to retrieve semantically analogous code samples and subsequently abstract a code template. To effectively integrate these multiple retrieval-augmented techniques, a comparative learning mechanism is designed to generate the final code. We validate the effectiveness of our approach on a dataset we build for deep learning code generation. Experimental results demonstrate that DLCodeGen outperforms other baselines, achieving improvements of 9.7% in CodeBLEU and 3.6% in human evaluation metrics. 

---
# Chinese-LiPS: A Chinese audio-visual speech recognition dataset with Lip-reading and Presentation Slides 

**Authors**: Jinghua Zhao, Yuhang Jia, Shiyao Wang, Jiaming Zhou, Hui Wang, Yong Qin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15066)  

**Abstract**: Incorporating visual modalities to assist Automatic Speech Recognition (ASR) tasks has led to significant improvements. However, existing Audio-Visual Speech Recognition (AVSR) datasets and methods typically rely solely on lip-reading information or speaking contextual video, neglecting the potential of combining these different valuable visual cues within the speaking context. In this paper, we release a multimodal Chinese AVSR dataset, Chinese-LiPS, comprising 100 hours of speech, video, and corresponding manual transcription, with the visual modality encompassing both lip-reading information and the presentation slides used by the speaker. Based on Chinese-LiPS, we develop a simple yet effective pipeline, LiPS-AVSR, which leverages both lip-reading and presentation slide information as visual modalities for AVSR tasks. Experiments show that lip-reading and presentation slide information improve ASR performance by approximately 8\% and 25\%, respectively, with a combined performance improvement of about 35\%. The dataset is available at this https URL 

---
# Mining Characteristics of Vulnerable Smart Contracts Across Lifecycle Stages 

**Authors**: Hongli Peng, Xiaoqi Li, Wenkai Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.15063)  

**Abstract**: Smart contracts are the cornerstone of decentralized applications and financial protocols, which extend the application of digital currency transactions. The applications and financial protocols introduce significant security challenges, resulting in substantial economic losses. Existing solutions predominantly focus on code vulnerabilities within smart contracts, accounting for only 50% of security incidents. Therefore, a more comprehensive study of security issues related to smart contracts is imperative. The existing empirical research realizes the static analysis of smart contracts from the perspective of the lifecycle and gives the corresponding measures for each stage. However, they lack the characteristic analysis of vulnerabilities in each stage and the distinction between the vulnerabilities. In this paper, we present the first empirical study on the security of smart contracts throughout their lifecycle, including deployment and execution, upgrade, and destruction stages. It delves into the security issues at each stage and provides at least seven feature descriptions. Finally, utilizing these seven features, five machine-learning classification models are used to identify vulnerabilities at different stages. The classification results reveal that vulnerable contracts exhibit distinct transaction features and ego network properties at various stages. 

---
# OPO: Making Decision-Focused Data Acquisition Decisions 

**Authors**: Egon Peršak, Miguel F. Anjos  

**Link**: [PDF](https://arxiv.org/pdf/2504.15062)  

**Abstract**: We propose a model for making data acquisition decisions for variables in contextual stochastic optimisation problems. Data acquisition decisions are typically treated as separate and fixed. We explore problem settings in which the acquisition of contextual variables is costly and consequently constrained. The data acquisition problem is often solved heuristically for proxy objectives such as coverage. The more intuitive objective is the downstream decision quality as a result of data acquisition decisions. The whole pipeline can be characterised as an optimise-then-predict-then-optimise (OPO) problem. Analogously, much recent research has focused on how to integrate prediction and optimisation (PO) in the form of decision-focused learning. We propose leveraging differentiable optimisation to extend the integration to data acquisition. We solve the data acquisition problem with well-defined constraints by learning a surrogate linear objective function. We demonstrate an application of this model on a shortest path problem for which we first have to set a drone reconnaissance strategy to capture image segments serving as inputs to a model that predicts travel costs. We ablate the problem with a number of training modalities and demonstrate that the differentiable optimisation approach outperforms random search strategies. 

---
# VeLU: Variance-enhanced Learning Unit for Deep Neural Networks 

**Authors**: Ashkan Shakarami, Yousef Yeganeh, Azade Farshad, Lorenzo Nicolè, Stefano Ghidoni, Nassir Navab  

**Link**: [PDF](https://arxiv.org/pdf/2504.15051)  

**Abstract**: Activation functions are fundamental in deep neural networks and directly impact gradient flow, optimization stability, and generalization. Although ReLU remains standard because of its simplicity, it suffers from vanishing gradients and lacks adaptability. Alternatives like Swish and GELU introduce smooth transitions, but fail to dynamically adjust to input statistics. We propose VeLU, a Variance-enhanced Learning Unit as an activation function that dynamically scales based on input variance by integrating ArcTan-Sin transformations and Wasserstein-2 regularization, effectively mitigating covariate shifts and stabilizing optimization. Extensive experiments on ViT_B16, VGG19, ResNet50, DenseNet121, MobileNetV2, and EfficientNetB3 confirm VeLU's superiority over ReLU, ReLU6, Swish, and GELU on six vision benchmarks. The codes of VeLU are publicly available on GitHub. 

---
# Beyond Terabit/s Integrated Neuromorphic Photonic Processor for DSP-Free Optical Interconnects 

**Authors**: Benshan Wang, Qiarong Xiao, Tengji Xu, Li Fan, Shaojie Liu, Jianji Dong, Junwen Zhang, Chaoran Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15044)  

**Abstract**: The rapid expansion of generative AI drives unprecedented demands for high-performance computing. Training large-scale AI models now requires vast interconnected GPU clusters across multiple data centers. Multi-scale AI training and inference demand uniform, ultra-low latency, and energy-efficient links to enable massive GPUs to function as a single cohesive unit. However, traditional electrical and optical interconnects, relying on conventional digital signal processors (DSPs) for signal distortion compensation, increasingly fail to meet these stringent requirements. To overcome these limitations, we present an integrated neuromorphic optical signal processor (OSP) that leverages deep reservoir computing and achieves DSP-free, all-optical, real-time processing. Experimentally, our OSP achieves a 100 Gbaud PAM4 per lane, 1.6 Tbit/s data center interconnect over a 5 km optical fiber in the C-band (equivalent to over 80 km in the O-band), far exceeding the reach of state-of-the-art DSP solutions, which are fundamentally constrained by chromatic dispersion in IMDD systems. Simultaneously, it reduces processing latency by four orders of magnitude and energy consumption by three orders of magnitude. Unlike DSPs, which introduce increased latency at high data rates, our OSP maintains consistent, ultra-low latency regardless of data rate scaling, making it ideal for future optical interconnects. Moreover, the OSP retains full optical field information for better impairment compensation and adapts to various modulation formats, data rates, and wavelengths. Fabricated using a mature silicon photonic process, the OSP can be monolithically integrated with silicon photonic transceivers, enhancing the compactness and reliability of all-optical interconnects. This research provides a highly scalable, energy-efficient, and high-speed solution, paving the way for next-generation AI infrastructure. 

---
# Distribution-aware Forgetting Compensation for Exemplar-Free Lifelong Person Re-identification 

**Authors**: Shiben Liu, Huijie Fan, Qiang Wang, Baojie Fan, Yandong Tang, Liangqiong Qu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15041)  

**Abstract**: Lifelong Person Re-identification (LReID) suffers from a key challenge in preserving old knowledge while adapting to new information. The existing solutions include rehearsal-based and rehearsal-free methods to address this challenge. Rehearsal-based approaches rely on knowledge distillation, continuously accumulating forgetting during the distillation process. Rehearsal-free methods insufficiently learn the distribution of each domain, leading to forgetfulness over time. To solve these issues, we propose a novel Distribution-aware Forgetting Compensation (DAFC) model that explores cross-domain shared representation learning and domain-specific distribution integration without using old exemplars or knowledge distillation. We propose a Text-driven Prompt Aggregation (TPA) that utilizes text features to enrich prompt elements and guide the prompt model to learn fine-grained representations for each instance. This can enhance the differentiation of identity information and establish the foundation for domain distribution awareness. Then, Distribution-based Awareness and Integration (DAI) is designed to capture each domain-specific distribution by a dedicated expert network and adaptively consolidate them into a shared region in high-dimensional space. In this manner, DAI can consolidate and enhance cross-domain shared representation learning while alleviating catastrophic forgetting. Furthermore, we develop a Knowledge Consolidation Mechanism (KCM) that comprises instance-level discrimination and cross-domain consistency alignment strategies to facilitate model adaptive learning of new knowledge from the current domain and promote knowledge consolidation learning between acquired domain-specific distributions, respectively. Experimental results show that our DAFC outperform state-of-the-art methods by at least 9.8\%/6.6\% and 6.4\%/6.2\% of average mAP/R@1 on two training orders. 

---
# SOLIDO: A Robust Watermarking Method for Speech Synthesis via Low-Rank Adaptation 

**Authors**: Yue Li, Weizhi Liu, Dongdong Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15035)  

**Abstract**: The accelerated advancement of speech generative models has given rise to security issues, including model infringement and unauthorized abuse of content. Although existing generative watermarking techniques have proposed corresponding solutions, most methods require substantial computational overhead and training costs. In addition, some methods have limitations in robustness when handling variable-length inputs. To tackle these challenges, we propose \textsc{SOLIDO}, a novel generative watermarking method that integrates parameter-efficient fine-tuning with speech watermarking through low-rank adaptation (LoRA) for speech diffusion models. Concretely, the watermark encoder converts the watermark to align with the input of diffusion models. To achieve precise watermark extraction from variable-length inputs, the watermark decoder based on depthwise separable convolution is designed for watermark recovery. To further enhance speech generation performance and watermark extraction capability, we propose a speech-driven lightweight fine-tuning strategy, which reduces computational overhead through LoRA. Comprehensive experiments demonstrate that the proposed method ensures high-fidelity watermarked speech even at a large capacity of 2000 bps. Furthermore, against common individual and compound speech attacks, our SOLIDO achieves a maximum average extraction accuracy of 99.20\% and 98.43\%, respectively. It surpasses other state-of-the-art methods by nearly 23\% in resisting time-stretching attacks. 

---
# Trainable Quantum Neural Network for Multiclass Image Classification with the Power of Pre-trained Tree Tensor Networks 

**Authors**: Keisuke Murota, Takumi Kobori  

**Link**: [PDF](https://arxiv.org/pdf/2504.14995)  

**Abstract**: Tree tensor networks (TTNs) offer powerful models for image classification. While these TTN image classifiers already show excellent performance on classical hardware, embedding them into quantum neural networks (QNNs) may further improve the performance by leveraging quantum resources. However, embedding TTN classifiers into QNNs for multiclass classification remains challenging. Key obstacles are the highorder gate operations required for large bond dimensions and the mid-circuit postselection with exponentially low success rates necessary for the exact embedding. In this work, to address these challenges, we propose forest tensor network (FTN)-classifiers, which aggregate multiple small-bond-dimension TTNs. This allows us to handle multiclass classification without requiring large gates in the embedded circuits. We then remove the overhead of mid-circuit postselection by extending the adiabatic encoding framework to our setting and smoothly encode the FTN-classifiers into a quantum forest tensor network (qFTN)- classifiers. Numerical experiments on MNIST and CIFAR-10 demonstrate that we can successfully train FTN-classifiers and encode them into qFTN-classifiers, while maintaining or even improving the performance of the pre-trained FTN-classifiers. These results suggest that synergy between TTN classification models and QNNs can provide a robust and scalable framework for multiclass quantum-enhanced image classification. 

---
# aiXamine: LLM Safety and Security Simplified 

**Authors**: Fatih Deniz, Dorde Popovic, Yazan Boshmaf, Euisuh Jeong, Minhaj Ahmad, Sanjay Chawla, Issa Khalil  

**Link**: [PDF](https://arxiv.org/pdf/2504.14985)  

**Abstract**: Evaluating Large Language Models (LLMs) for safety and security remains a complex task, often requiring users to navigate a fragmented landscape of ad hoc benchmarks, datasets, metrics, and reporting formats. To address this challenge, we present aiXamine, a comprehensive black-box evaluation platform for LLM safety and security. aiXamine integrates over 40 tests (i.e., benchmarks) organized into eight key services targeting specific dimensions of safety and security: adversarial robustness, code security, fairness and bias, hallucination, model and data privacy, out-of-distribution (OOD) robustness, over-refusal, and safety alignment. The platform aggregates the evaluation results into a single detailed report per model, providing a detailed breakdown of model performance, test examples, and rich visualizations. We used aiXamine to assess over 50 publicly available and proprietary LLMs, conducting over 2K examinations. Our findings reveal notable vulnerabilities in leading models, including susceptibility to adversarial attacks in OpenAI's GPT-4o, biased outputs in xAI's Grok-3, and privacy weaknesses in Google's Gemini 2.0. Additionally, we observe that open-source models can match or exceed proprietary models in specific services such as safety alignment, fairness and bias, and OOD robustness. Finally, we identify trade-offs between distillation strategies, model size, training methods, and architectural choices. 

---
# Speaker Fuzzy Fingerprints: Benchmarking Text-Based Identification in Multiparty Dialogues 

**Authors**: Rui Ribeiro, Luísa Coheur, Joao P. Carvalho  

**Link**: [PDF](https://arxiv.org/pdf/2504.14963)  

**Abstract**: Speaker identification using voice recordings leverages unique acoustic features, but this approach fails when only textual data is available. Few approaches have attempted to tackle the problem of identifying speakers solely from text, and the existing ones have primarily relied on traditional methods. In this work, we explore the use of fuzzy fingerprints from large pre-trained models to improve text-based speaker identification. We integrate speaker-specific tokens and context-aware modeling, demonstrating that conversational context significantly boosts accuracy, reaching 70.6% on the Friends dataset and 67.7% on the Big Bang Theory dataset. Additionally, we show that fuzzy fingerprints can approximate full fine-tuning performance with fewer hidden units, offering improved interpretability. Finally, we analyze ambiguous utterances and propose a mechanism to detect speaker-agnostic lines. Our findings highlight key challenges and provide insights for future improvements in text-based speaker identification. 

---
# Learning to Reason under Off-Policy Guidance 

**Authors**: Jianhao Yan, Yafu Li, Zican Hu, Zhi Wang, Ganqu Cui, Xiaoye Qu, Yu Cheng, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14945)  

**Abstract**: Recent advances in large reasoning models (LRMs) demonstrate that sophisticated behaviors such as multi-step reasoning and self-reflection can emerge via reinforcement learning (RL) with simple rule-based rewards. However, existing zero-RL approaches are inherently ``on-policy'', limiting learning to a model's own outputs and failing to acquire reasoning abilities beyond its initial capabilities. We introduce LUFFY (Learning to reason Under oFF-policY guidance), a framework that augments zero-RL with off-policy reasoning traces. LUFFY dynamically balances imitation and exploration by combining off-policy demonstrations with on-policy rollouts during training. Notably, we propose policy shaping via regularized importance sampling to avoid superficial and rigid imitation during mixed-policy training. Remarkably, LUFFY achieves an over +7.0 average gain across six math benchmarks and an advantage of over +6.2 points in out-of-distribution tasks. It also substantially surpasses imitation-based supervised fine-tuning (SFT), particularly in generalization. Analysis shows LUFFY not only imitates effectively but also explores beyond demonstrations, offering a scalable path to train generalizable reasoning models with off-policy guidance. 

---
# Giving AI a voice: how does AI think it should be treated? 

**Authors**: Maria Fay, Frederik F. Flöther  

**Link**: [PDF](https://arxiv.org/pdf/2504.14936)  

**Abstract**: With the astounding progress in (generative) artificial intelligence (AI), there has been significant public discourse regarding regulation and ethics of the technology. Is it sufficient when humans discuss this with other humans? Or, given that AI is increasingly becoming a viable source of inspiration for people (and let alone the hypothetical possibility that the technology may at some point become "artificial general intelligence" and/or develop consciousness), should AI not join the discourse? There are new questions and angles that AI brings to the table that we might not have considered before - so let us make the key subject of this book an active participant. This chapter therefore includes a brief human-AI conversation on the topic of AI rights and ethics. 

---
# Fast Adversarial Training with Weak-to-Strong Spatial-Temporal Consistency in the Frequency Domain on Videos 

**Authors**: Songping Wang, Hanqing Liu, Yueming Lyu, Xiantao Hu, Ziwen He, Wei Wang, Caifeng Shan, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14921)  

**Abstract**: Adversarial Training (AT) has been shown to significantly enhance adversarial robustness via a min-max optimization approach. However, its effectiveness in video recognition tasks is hampered by two main challenges. First, fast adversarial training for video models remains largely unexplored, which severely impedes its practical applications. Specifically, most video adversarial training methods are computationally costly, with long training times and high expenses. Second, existing methods struggle with the trade-off between clean accuracy and adversarial robustness. To address these challenges, we introduce Video Fast Adversarial Training with Weak-to-Strong consistency (VFAT-WS), the first fast adversarial training method for video data. Specifically, VFAT-WS incorporates the following key designs: First, it integrates a straightforward yet effective temporal frequency augmentation (TF-AUG), and its spatial-temporal enhanced form STF-AUG, along with a single-step PGD attack to boost training efficiency and robustness. Second, it devises a weak-to-strong spatial-temporal consistency regularization, which seamlessly integrates the simpler TF-AUG and the more complex STF-AUG. Leveraging the consistency regularization, it steers the learning process from simple to complex augmentations. Both of them work together to achieve a better trade-off between clean accuracy and robustness. Extensive experiments on UCF-101 and HMDB-51 with both CNN and Transformer-based models demonstrate that VFAT-WS achieves great improvements in adversarial robustness and corruption robustness, while accelerating training by nearly 490%. 

---
# StableQuant: Layer Adaptive Post-Training Quantization for Speech Foundation Models 

**Authors**: Yeona Hong, Hyewon Han, Woo-jin Chung, Hong-Goo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14915)  

**Abstract**: In this paper, we propose StableQuant, a novel adaptive post-training quantization (PTQ) algorithm for widely used speech foundation models (SFMs). While PTQ has been successfully employed for compressing large language models (LLMs) due to its ability to bypass additional fine-tuning, directly applying these techniques to SFMs may not yield optimal results, as SFMs utilize distinct network architecture for feature extraction. StableQuant demonstrates optimal quantization performance regardless of the network architecture type, as it adaptively determines the quantization range for each layer by analyzing both the scale distributions and overall performance. We evaluate our algorithm on two SFMs, HuBERT and wav2vec2.0, for an automatic speech recognition (ASR) task, and achieve superior performance compared to traditional PTQ methods. StableQuant successfully reduces the sizes of SFM models to a quarter and doubles the inference speed while limiting the word error rate (WER) performance drop to less than 0.3% with 8-bit quantization. 

---
# Guidelines for External Disturbance Factors in the Use of OCR in Real-World Environments 

**Authors**: Kenji Iwata, Eiki Ishidera, Toshifumi Yamaai, Yutaka Satoh, Hiroshi Tanaka, Katsuhiko Takahashi, Akio Furuhata, Yoshihisa Tanabe, Hiroshi Matsumura  

**Link**: [PDF](https://arxiv.org/pdf/2504.14913)  

**Abstract**: The performance of OCR has improved with the evolution of AI technology. As OCR continues to broaden its range of applications, the increased likelihood of interference introduced by various usage environments can prevent it from achieving its inherent performance. This results in reduced recognition accuracy under certain conditions, and makes the quality control of recognition devices more challenging. Therefore, to ensure that users can properly utilize OCR, we compiled the real-world external disturbance factors that cause performance degradation, along with the resulting image degradation phenomena, into an external disturbance factor table and, by also indicating how to make use of it, organized them into guidelines. 

---
# VLM as Policy: Common-Law Content Moderation Framework for Short Video Platform 

**Authors**: Xingyu Lu, Tianke Zhang, Chang Meng, Xiaobei Wang, Jinpeng Wang, YiFan Zhang, Shisong Tang, Changyi Liu, Haojie Ding, Kaiyu Jiang, Kaiyu Tang, Bin Wen, Hai-Tao Zheng, Fan Yang, Tingting Gao, Di Zhang, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2504.14904)  

**Abstract**: Exponentially growing short video platforms (SVPs) face significant challenges in moderating content detrimental to users' mental health, particularly for minors. The dissemination of such content on SVPs can lead to catastrophic societal consequences. Although substantial efforts have been dedicated to moderating such content, existing methods suffer from critical limitations: (1) Manual review is prone to human bias and incurs high operational costs. (2) Automated methods, though efficient, lack nuanced content understanding, resulting in lower accuracy. (3) Industrial moderation regulations struggle to adapt to rapidly evolving trends due to long update cycles. In this paper, we annotate the first SVP content moderation benchmark with authentic user/reviewer feedback to fill the absence of benchmark in this field. Then we evaluate various methods on the benchmark to verify the existence of the aforementioned limitations. We further propose our common-law content moderation framework named KuaiMod to address these challenges. KuaiMod consists of three components: training data construction, offline adaptation, and online deployment & refinement. Leveraging large vision language model (VLM) and Chain-of-Thought (CoT) reasoning, KuaiMod adequately models video toxicity based on sparse user feedback and fosters dynamic moderation policy with rapid update speed and high accuracy. Offline experiments and large-scale online A/B test demonstrates the superiority of KuaiMod: KuaiMod achieves the best moderation performance on our benchmark. The deployment of KuaiMod reduces the user reporting rate by 20% and its application in video recommendation increases both Daily Active User (DAU) and APP Usage Time (AUT) on several Kuaishou scenarios. We have open-sourced our benchmark at this https URL. 

---
# Latent Bayesian Optimization via Autoregressive Normalizing Flows 

**Authors**: Seunghun Lee, Jinyoung Park, Jaewon Chu, Minseo Yoon, Hyunwoo J. Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.14889)  

**Abstract**: Bayesian Optimization (BO) has been recognized for its effectiveness in optimizing expensive and complex objective functions. Recent advancements in Latent Bayesian Optimization (LBO) have shown promise by integrating generative models such as variational autoencoders (VAEs) to manage the complexity of high-dimensional and structured data spaces. However, existing LBO approaches often suffer from the value discrepancy problem, which arises from the reconstruction gap between input and latent spaces. This value discrepancy problem propagates errors throughout the optimization process, leading to suboptimal outcomes. To address this issue, we propose a Normalizing Flow-based Bayesian Optimization (NF-BO), which utilizes normalizing flow as a generative model to establish one-to-one encoding function from the input space to the latent space, along with its left-inverse decoding function, eliminating the reconstruction gap. Specifically, we introduce SeqFlow, an autoregressive normalizing flow for sequence data. In addition, we develop a new candidate sampling strategy that dynamically adjusts the exploration probability for each token based on its importance. Through extensive experiments, our NF-BO method demonstrates superior performance in molecule generation tasks, significantly outperforming both traditional and recent LBO approaches. 

---
# Impact of Latent Space Dimension on IoT Botnet Detection Performance: VAE-Encoder Versus ViT-Encoder 

**Authors**: Hassan Wasswa, Aziida Nanyonga, Timothy Lynar  

**Link**: [PDF](https://arxiv.org/pdf/2504.14879)  

**Abstract**: The rapid evolution of Internet of Things (IoT) technology has led to a significant increase in the number of IoT devices, applications, and services. This surge in IoT devices, along with their widespread presence, has made them a prime target for various cyber-attacks, particularly through IoT botnets. As a result, security has become a major concern within the IoT ecosystem. This study focuses on investigating how the latent dimension impacts the performance of different deep learning classifiers when trained on latent vector representations of the train dataset. The primary objective is to compare the outcomes of these models when encoder components from two cutting-edge architectures: the Vision Transformer (ViT) and the Variational Auto-Encoder (VAE) are utilized to project the high dimensional train dataset to the learned low dimensional latent space. The encoder components are employed to project high-dimensional structured .csv IoT botnet traffic datasets to various latent sizes. Evaluated on N-BaIoT and CICIoT2022 datasets, findings reveal that VAE-encoder based dimension reduction outperforms ViT-encoder based dimension reduction for both datasets in terms of four performance metrics including accuracy, precision, recall, and F1-score for all models which can be attributed to absence of spatial patterns in the datasets the ViT model attempts to learn and extract from image instances. 

---
# ReSpec: Relevance and Specificity Grounded Online Filtering for Learning on Video-Text Data Streams 

**Authors**: Chris Dongjoo Kim, Jihwan Moon, Sangwoo Moon, Heeseung Yun, Sihaeng Lee, Aniruddha Kembhavi, Soonyoung Lee, Gunhee Kim, Sangho Lee, Christopher Clark  

**Link**: [PDF](https://arxiv.org/pdf/2504.14875)  

**Abstract**: The rapid growth of video-text data presents challenges in storage and computation during training. Online learning, which processes streaming data in real-time, offers a promising solution to these issues while also allowing swift adaptations in scenarios demanding real-time responsiveness. One strategy to enhance the efficiency and effectiveness of learning involves identifying and prioritizing data that enhances performance on target downstream tasks. We propose Relevance and Specificity-based online filtering framework (ReSpec) that selects data based on four criteria: (i) modality alignment for clean data, (ii) task relevance for target focused data, (iii) specificity for informative and detailed data, and (iv) efficiency for low-latency processing. Relevance is determined by the probabilistic alignment of incoming data with downstream tasks, while specificity employs the distance to a root embedding representing the least specific data as an efficient proxy for informativeness. By establishing reference points from target task data, ReSpec filters incoming data in real-time, eliminating the need for extensive storage and compute. Evaluating on large-scale datasets WebVid2M and VideoCC3M, ReSpec attains state-of-the-art performance on five zeroshot video retrieval tasks, using as little as 5% of the data while incurring minimal compute. The source code is available at this https URL. 

---
# Bridge the Gap: From Weak to Full Supervision for Temporal Action Localization with PseudoFormer 

**Authors**: Ziyi Liu, Yangcen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14860)  

**Abstract**: Weakly-supervised Temporal Action Localization (WTAL) has achieved notable success but still suffers from a lack of temporal annotations, leading to a performance and framework gap compared with fully-supervised methods. While recent approaches employ pseudo labels for training, three key challenges: generating high-quality pseudo labels, making full use of different priors, and optimizing training methods with noisy labels remain unresolved. Due to these perspectives, we propose PseudoFormer, a novel two-branch framework that bridges the gap between weakly and fully-supervised Temporal Action Localization (TAL). We first introduce RickerFusion, which maps all predicted action proposals to a global shared space to generate pseudo labels with better quality. Subsequently, we leverage both snippet-level and proposal-level labels with different priors from the weak branch to train the regression-based model in the full branch. Finally, the uncertainty mask and iterative refinement mechanism are applied for training with noisy pseudo labels. PseudoFormer achieves state-of-the-art WTAL results on the two commonly used benchmarks, THUMOS14 and ActivityNet1.3. Besides, extensive ablation studies demonstrate the contribution of each component of our method. 

---
# Object-Level Verbalized Confidence Calibration in Vision-Language Models via Semantic Perturbation 

**Authors**: Yunpu Zhao, Rui Zhang, Junbin Xiao, Ruibo Hou, Jiaming Guo, Zihao Zhang, Yifan Hao, Yunji Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.14848)  

**Abstract**: Vision-language models (VLMs) excel in various multimodal tasks but frequently suffer from poor calibration, resulting in misalignment between their verbalized confidence and response correctness. This miscalibration undermines user trust, especially when models confidently provide incorrect or fabricated information. In this work, we propose a novel Confidence Calibration through Semantic Perturbation (CSP) framework to improve the calibration of verbalized confidence for VLMs in response to object-centric queries. We first introduce a perturbed dataset where Gaussian noise is applied to the key object regions to simulate visual uncertainty at different confidence levels, establishing an explicit mapping between visual ambiguity and confidence levels. We further enhance calibration through a two-stage training process combining supervised fine-tuning on the perturbed dataset with subsequent preference optimization. Extensive experiments on popular benchmarks demonstrate that our method significantly improves the alignment between verbalized confidence and response correctness while maintaining or enhancing overall task performance. These results highlight the potential of semantic perturbation as a practical tool for improving the reliability and interpretability of VLMs. 

---
# Exploring $\ell_0$ Sparsification for Inference-free Sparse Retrievers 

**Authors**: Xinjie Shen, Zhichao Geng, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14839)  

**Abstract**: With increasing demands for efficiency, information retrieval has developed a branch of sparse retrieval, further advancing towards inference-free retrieval where the documents are encoded during indexing time and there is no model-inference for queries. Existing sparse retrieval models rely on FLOPS regularization for sparsification, while this mechanism was originally designed for Siamese encoders, it is considered to be suboptimal in inference-free scenarios which is asymmetric. Previous attempts to adapt FLOPS for inference-free scenarios have been limited to rule-based methods, leaving the potential of sparsification approaches for inference-free retrieval models largely unexplored. In this paper, we explore $\ell_0$ inspired sparsification manner for inference-free retrievers. Through comprehensive out-of-domain evaluation on the BEIR benchmark, our method achieves state-of-the-art performance among inference-free sparse retrieval models and is comparable to leading Siamese sparse retrieval models. Furthermore, we provide insights into the trade-off between retrieval effectiveness and computational efficiency, demonstrating practical value for real-world applications. 

---
# Protecting Your Voice: Temporal-aware Robust Watermarking 

**Authors**: Yue Li, Weizhi Liu, Dongdong Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.14832)  

**Abstract**: The rapid advancement of generative models has led to the synthesis of real-fake ambiguous voices. To erase the ambiguity, embedding watermarks into the frequency-domain features of synthesized voices has become a common routine. However, the robustness achieved by choosing the frequency domain often comes at the expense of fine-grained voice features, leading to a loss of fidelity. Maximizing the comprehensive learning of time-domain features to enhance fidelity while maintaining robustness, we pioneer a \textbf{\underline{t}}emporal-aware \textbf{\underline{r}}ob\textbf{\underline{u}}st wat\textbf{\underline{e}}rmarking (\emph{True}) method for protecting the speech and singing voice. 

---
# ECViT: Efficient Convolutional Vision Transformer with Local-Attention and Multi-scale Stages 

**Authors**: Zhoujie Qian  

**Link**: [PDF](https://arxiv.org/pdf/2504.14825)  

**Abstract**: Vision Transformers (ViTs) have revolutionized computer vision by leveraging self-attention to model long-range dependencies. However, ViTs face challenges such as high computational costs due to the quadratic scaling of self-attention and the requirement of a large amount of training data. To address these limitations, we propose the Efficient Convolutional Vision Transformer (ECViT), a hybrid architecture that effectively combines the strengths of CNNs and Transformers. ECViT introduces inductive biases such as locality and translation invariance, inherent to Convolutional Neural Networks (CNNs) into the Transformer framework by extracting patches from low-level features and enhancing the encoder with convolutional operations. Additionally, it incorporates local-attention and a pyramid structure to enable efficient multi-scale feature extraction and representation. Experimental results demonstrate that ECViT achieves an optimal balance between performance and efficiency, outperforming state-of-the-art models on various image classification tasks while maintaining low computational and storage requirements. ECViT offers an ideal solution for applications that prioritize high efficiency without compromising performance. 

---
# What Lurks Within? Concept Auditing for Shared Diffusion Models at Scale 

**Authors**: Xiaoyong Yuan, Xiaolong Ma, Linke Guo, Lan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14815)  

**Abstract**: Diffusion models (DMs) have revolutionized text-to-image generation, enabling the creation of highly realistic and customized images from text prompts. With the rise of parameter-efficient fine-tuning (PEFT) techniques like LoRA, users can now customize powerful pre-trained models using minimal computational resources. However, the widespread sharing of fine-tuned DMs on open platforms raises growing ethical and legal concerns, as these models may inadvertently or deliberately generate sensitive or unauthorized content, such as copyrighted material, private individuals, or harmful content. Despite the increasing regulatory attention on generative AI, there are currently no practical tools for systematically auditing these models before deployment. In this paper, we address the problem of concept auditing: determining whether a fine-tuned DM has learned to generate a specific target concept. Existing approaches typically rely on prompt-based input crafting and output-based image classification but suffer from critical limitations, including prompt uncertainty, concept drift, and poor scalability. To overcome these challenges, we introduce Prompt-Agnostic Image-Free Auditing (PAIA), a novel, model-centric concept auditing framework. By treating the DM as the object of inspection, PAIA enables direct analysis of internal model behavior, bypassing the need for optimized prompts or generated images. We evaluate PAIA on 320 controlled model and 690 real-world community models sourced from a public DM sharing platform. PAIA achieves over 90% detection accuracy while reducing auditing time by 18-40x compared to existing baselines. To our knowledge, PAIA is the first scalable and practical solution for pre-deployment concept auditing of diffusion models, providing a practical foundation for safer and more transparent diffusion model sharing. 

---
# On Self-improving Token Embeddings 

**Authors**: Mario M. Kubek, Shiraj Pokharel, Thomas Böhme, Emma L. McDaniel, Herwig Unger, Armin R. Mikler  

**Link**: [PDF](https://arxiv.org/pdf/2504.14808)  

**Abstract**: This article introduces a novel and fast method for refining pre-trained static word or, more generally, token embeddings. By incorporating the embeddings of neighboring tokens in text corpora, it continuously updates the representation of each token, including those without pre-assigned embeddings. This approach effectively addresses the out-of-vocabulary problem, too. Operating independently of large language models and shallow neural networks, it enables versatile applications such as corpus exploration, conceptual search, and word sense disambiguation. The method is designed to enhance token representations within topically homogeneous corpora, where the vocabulary is restricted to a specific domain, resulting in more meaningful embeddings compared to general-purpose pre-trained vectors. As an example, the methodology is applied to explore storm events and their impacts on infrastructure and communities using narratives from a subset of the NOAA Storm Events database. The article also demonstrates how the approach improves the representation of storm-related terms over time, providing valuable insights into the evolving nature of disaster narratives. 

---
# Dynamic Contrastive Skill Learning with State-Transition Based Skill Clustering and Dynamic Length Adjustment 

**Authors**: Jinwoo Choi, Seung-Woo Seo  

**Link**: [PDF](https://arxiv.org/pdf/2504.14805)  

**Abstract**: Reinforcement learning (RL) has made significant progress in various domains, but scaling it to long-horizon tasks with complex decision-making remains challenging. Skill learning attempts to address this by abstracting actions into higher-level behaviors. However, current approaches often fail to recognize semantically similar behaviors as the same skill and use fixed skill lengths, limiting flexibility and generalization. To address this, we propose Dynamic Contrastive Skill Learning (DCSL), a novel framework that redefines skill representation and learning. DCSL introduces three key ideas: state-transition based skill representation, skill similarity function learning, and dynamic skill length adjustment. By focusing on state transitions and leveraging contrastive learning, DCSL effectively captures the semantic context of behaviors and adapts skill lengths to match the appropriate temporal extent of behaviors. Our approach enables more flexible and adaptive skill extraction, particularly in complex or noisy datasets, and demonstrates competitive performance compared to existing methods in task completion and efficiency. 

---
# Automatic Evaluation Metrics for Document-level Translation: Overview, Challenges and Trends 

**Authors**: Jiaxin GUO, Xiaoyu Chen, Zhiqiang Rao, Jinlong Yang, Zongyao Li, Hengchao Shang, Daimeng Wei, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14804)  

**Abstract**: With the rapid development of deep learning technologies, the field of machine translation has witnessed significant progress, especially with the advent of large language models (LLMs) that have greatly propelled the advancement of document-level translation. However, accurately evaluating the quality of document-level translation remains an urgent issue. This paper first introduces the development status of document-level translation and the importance of evaluation, highlighting the crucial role of automatic evaluation metrics in reflecting translation quality and guiding the improvement of translation systems. It then provides a detailed analysis of the current state of automatic evaluation schemes and metrics, including evaluation methods with and without reference texts, as well as traditional metrics, Model-based metrics and LLM-based metrics. Subsequently, the paper explores the challenges faced by current evaluation methods, such as the lack of reference diversity, dependence on sentence-level alignment information, and the bias, inaccuracy, and lack of interpretability of the LLM-as-a-judge method. Finally, the paper looks ahead to the future trends in evaluation methods, including the development of more user-friendly document-level evaluation methods and more robust LLM-as-a-judge methods, and proposes possible research directions, such as reducing the dependency on sentence-level information, introducing multi-level and multi-granular evaluation approaches, and training models specifically for machine translation evaluation. This study aims to provide a comprehensive analysis of automatic evaluation for document-level translation and offer insights into future developments. 

---
# Automated Duplicate Bug Report Detection in Large Open Bug Repositories 

**Authors**: Clare E. Laney, Andrew Barovic, Armin Moin  

**Link**: [PDF](https://arxiv.org/pdf/2504.14797)  

**Abstract**: Many users and contributors of large open-source projects report software defects or enhancement requests (known as bug reports) to the issue-tracking systems. However, they sometimes report issues that have already been reported. First, they may not have time to do sufficient research on existing bug reports. Second, they may not possess the right expertise in that specific area to realize that an existing bug report is essentially elaborating on the same matter, perhaps with a different wording. In this paper, we propose a novel approach based on machine learning methods that can automatically detect duplicate bug reports in an open bug repository based on the textual data in the reports. We present six alternative methods: Topic modeling, Gaussian Naive Bayes, deep learning, time-based organization, clustering, and summarization using a generative pre-trained transformer large language model. Additionally, we introduce a novel threshold-based approach for duplicate identification, in contrast to the conventional top-k selection method that has been widely used in the literature. Our approach demonstrates promising results across all the proposed methods, achieving accuracy rates ranging from the high 70%'s to the low 90%'s. We evaluated our methods on a public dataset of issues belonging to an Eclipse open-source project. 

---
# How Effective Can Dropout Be in Multiple Instance Learning ? 

**Authors**: Wenhui Zhu, Peijie Qiu, Xiwen Chen, Zhangsihao Yang, Aristeidis Sotiras, Abolfazl Razi, Yalin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14783)  

**Abstract**: Multiple Instance Learning (MIL) is a popular weakly-supervised method for various applications, with a particular interest in histological whole slide image (WSI) classification. Due to the gigapixel resolution of WSI, applications of MIL in WSI typically necessitate a two-stage training scheme: first, extract features from the pre-trained backbone and then perform MIL aggregation. However, it is well-known that this suboptimal training scheme suffers from "noisy" feature embeddings from the backbone and inherent weak supervision, hindering MIL from learning rich and generalizable features. However, the most commonly used technique (i.e., dropout) for mitigating this issue has yet to be explored in MIL. In this paper, we empirically explore how effective the dropout can be in MIL. Interestingly, we observe that dropping the top-k most important instances within a bag leads to better performance and generalization even under noise attack. Based on this key observation, we propose a novel MIL-specific dropout method, termed MIL-Dropout, which systematically determines which instances to drop. Experiments on five MIL benchmark datasets and two WSI datasets demonstrate that MIL-Dropout boosts the performance of current MIL methods with a negligible computational cost. The code is available at this https URL. 

---
# Exploring Collaborative GenAI Agents in Synchronous Group Settings: Eliciting Team Perceptions and Design Considerations for the Future of Work 

**Authors**: Janet G. Johnson, Macarena Peralta, Mansanjam Kaur, Ruijie Sophia Huang, Sheng Zhao, Ruijia Guan, Shwetha Rajaram, Michael Nebeling  

**Link**: [PDF](https://arxiv.org/pdf/2504.14779)  

**Abstract**: While generative artificial intelligence (GenAI) is finding increased adoption in workplaces, current tools are primarily designed for individual use. Prior work established the potential for these tools to enhance personal creativity and productivity towards shared goals; however, we don't know yet how to best take into account the nuances of group work and team dynamics when deploying GenAI in work settings. In this paper, we investigate the potential of collaborative GenAI agents to augment teamwork in synchronous group settings through an exploratory study that engaged 25 professionals across 6 teams in speculative design workshops and individual follow-up interviews. Our workshops included a mixed reality provotype to simulate embodied collaborative GenAI agents capable of actively participating in group discussions. Our findings suggest that, if designed well, collaborative GenAI agents offer valuable opportunities to enhance team problem-solving by challenging groupthink, bridging communication gaps, and reducing social friction. However, teams' willingness to integrate GenAI agents depended on its perceived fit across a number of individual, team, and organizational factors. We outline the key design tensions around agent representation, social prominence, and engagement and highlight the opportunities spatial and immersive technologies could offer to modulate GenAI influence on team outcomes and strike a balance between augmentation and agency. 

---
# A Combinatorial Theory of Dropout: Subnetworks, Graph Geometry, and Generalization 

**Authors**: Sahil Rajesh Dhayalkar  

**Link**: [PDF](https://arxiv.org/pdf/2504.14762)  

**Abstract**: We propose a combinatorial and graph-theoretic theory of dropout by modeling training as a random walk over a high-dimensional graph of binary subnetworks. Each node represents a masked version of the network, and dropout induces stochastic traversal across this space. We define a subnetwork contribution score that quantifies generalization and show that it varies smoothly over the graph. Using tools from spectral graph theory, PAC-Bayes analysis, and combinatorics, we prove that generalizing subnetworks form large, connected, low-resistance clusters, and that their number grows exponentially with network width. This reveals dropout as a mechanism for sampling from a robust, structured ensemble of well-generalizing subnetworks with built-in redundancy. Extensive experiments validate every theoretical claim across diverse architectures. Together, our results offer a unified foundation for understanding dropout and suggest new directions for mask-guided regularization and subnetwork optimization. 

---
# SWE-Synth: Synthesizing Verifiable Bug-Fix Data to Enable Large Language Models in Resolving Real-World Bugs 

**Authors**: Minh V.T. Pham, Huy N. Phan, Hoang N. Phan, Cuong Le Chi, Tien N. Nguyen, Nghi D. Q. Bui  

**Link**: [PDF](https://arxiv.org/pdf/2504.14757)  

**Abstract**: Large language models (LLMs) are transforming automated program repair (APR) through agent-based approaches that localize bugs, generate patches, and verify fixes. However, the lack of high-quality, scalable training datasets, especially those with verifiable outputs and intermediate reasoning traces-limits progress, particularly for open-source models. In this work, we present SWE-Synth, a framework for synthesizing realistic, verifiable, and process-aware bug-fix datasets at the repository level. SWE-Synth leverages LLM agents to simulate debugging workflows, producing not only bug-fix pairs but also test cases and structured repair trajectories. Compared to manually curated datasets, our method scales with minimal human effort while preserving contextual richness and correctness. Experiments show that models trained on SWE-Synth outperform those trained on real-world datasets by 2.3% on SWE-Bench Lite. Our results highlight the potential of synthetic, agent-generated data to advance the state of the art in APR and software engineering automation. 

---
# AI for the Open-World: the Learning Principles 

**Authors**: Jianyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14751)  

**Abstract**: During the past decades, numerous successes of AI has been made on "specific capabilities", named closed-world, such as artificial environments or specific real-world tasks. This well-defined narrow capability brings two nice benefits, a clear criterion of success and the opportunity to collect a lot of examples. The criteria not only reveal whether a machine has achieved a goal, but reveal how the machine falls short of the goal. As a result, human designers can fix the problems one after the other until the machine is deemed good enough for the task. Furthermore, the large set of collected examples reduces the difficulty of this problem-fixing process (by the central limit theorem).
Do the success in closed-world translate into broad open-world, where a machine is required to perform any task that a human could possibly undertake with fewer examples and less priori knowledge from human designers? No. Because competence in a specific task provides little insight in handling other tasks, the valuable criteria for specific tasks become helpless when handling broader unseen tasks. Furthermore, due to the shortage of examples in unseen tasks, central limit theorem does not stand on our side. At the end, human designers lose the oscilloscope to "hack" an AI system for the open-world.
Achieving AI for the open-world requires unique learning principles and innovated techniques, which are different from the ones in building AI for the closed-world. This thesis explores necessary learning principles required to construct AI for the open-world, including rich features (analogy a large tool box), disentangled representation (an organized tool box), and inference-time learning (a tool-savvy hand). Driven by the learning principles, this thesis further proposes techniques to use the learning principles, conducts enormous large-scale experiments to verify the learning principles. 

---
# A Modularized Design Approach for GelSight Family of Vision-based Tactile Sensors 

**Authors**: Arpit Agarwal, Mohammad Amin Mirzaee, Xiping Sun, Wenzhen Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2504.14739)  

**Abstract**: GelSight family of vision-based tactile sensors has proven to be effective for multiple robot perception and manipulation tasks. These sensors are based on an internal optical system and an embedded camera to capture the deformation of the soft sensor surface, inferring the high-resolution geometry of the objects in contact. However, customizing the sensors for different robot hands requires a tedious trial-and-error process to re-design the optical system. In this paper, we formulate the GelSight sensor design process as a systematic and objective-driven design problem and perform the design optimization with a physically accurate optical simulation. The method is based on modularizing and parameterizing the sensor's optical components and designing four generalizable objective functions to evaluate the sensor. We implement the method with an interactive and easy-to-use toolbox called OptiSense Studio. With the toolbox, non-sensor experts can quickly optimize their sensor design in both forward and inverse ways following our predefined modules and steps. We demonstrate our system with four different GelSight sensors by quickly optimizing their initial design in simulation and transferring it to the real sensors. 

---
# SuperCL: Superpixel Guided Contrastive Learning for Medical Image Segmentation Pre-training 

**Authors**: Shuang Zeng, Lei Zhu, Xinliang Zhang, Hangzhou He, Yanye Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14737)  

**Abstract**: Medical image segmentation is a critical yet challenging task, primarily due to the difficulty of obtaining extensive datasets of high-quality, expert-annotated images. Contrastive learning presents a potential but still problematic solution to this issue. Because most existing methods focus on extracting instance-level or pixel-to-pixel representation, which ignores the characteristics between intra-image similar pixel groups. Moreover, when considering contrastive pairs generation, most SOTA methods mainly rely on manually setting thresholds, which requires a large number of gradient experiments and lacks efficiency and generalization. To address these issues, we propose a novel contrastive learning approach named SuperCL for medical image segmentation pre-training. Specifically, our SuperCL exploits the structural prior and pixel correlation of images by introducing two novel contrastive pairs generation strategies: Intra-image Local Contrastive Pairs (ILCP) Generation and Inter-image Global Contrastive Pairs (IGCP) Generation. Considering superpixel cluster aligns well with the concept of contrastive pairs generation, we utilize the superpixel map to generate pseudo masks for both ILCP and IGCP to guide supervised contrastive learning. Moreover, we also propose two modules named Average SuperPixel Feature Map Generation (ASP) and Connected Components Label Generation (CCL) to better exploit the prior structural information for IGCP. Finally, experiments on 8 medical image datasets indicate our SuperCL outperforms existing 12 methods. i.e. Our SuperCL achieves a superior performance with more precise predictions from visualization figures and 3.15%, 5.44%, 7.89% DSC higher than the previous best results on MMWHS, CHAOS, Spleen with 10% annotations. Our code will be released after acceptance. 

---
# Semi-parametric Memory Consolidation: Towards Brain-like Deep Continual Learning 

**Authors**: Geng Liu, Fei Zhu, Rong Feng, Zhiqiang Yi, Shiqi Wang, Gaofeng Meng, Zhaoxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14727)  

**Abstract**: Humans and most animals inherently possess a distinctive capacity to continually acquire novel experiences and accumulate worldly knowledge over time. This ability, termed continual learning, is also critical for deep neural networks (DNNs) to adapt to the dynamically evolving world in open environments. However, DNNs notoriously suffer from catastrophic forgetting of previously learned knowledge when trained on sequential tasks. In this work, inspired by the interactive human memory and learning system, we propose a novel biomimetic continual learning framework that integrates semi-parametric memory and the wake-sleep consolidation mechanism. For the first time, our method enables deep neural networks to retain high performance on novel tasks while maintaining prior knowledge in real-world challenging continual learning scenarios, e.g., class-incremental learning on ImageNet. This study demonstrates that emulating biological intelligence provides a promising path to enable deep neural networks with continual learning capabilities. 

---
# Exposing the Copycat Problem of Imitation-based Planner: A Novel Closed-Loop Simulator, Causal Benchmark and Joint IL-RL Baseline 

**Authors**: Hui Zhou, Shaoshuai Shi, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.14709)  

**Abstract**: Machine learning (ML)-based planners have recently gained significant attention. They offer advantages over traditional optimization-based planning algorithms. These advantages include fewer manually selected parameters and faster development. Within ML-based planning, imitation learning (IL) is a common algorithm. It primarily learns driving policies directly from supervised trajectory data. While IL has demonstrated strong performance on many open-loop benchmarks, it remains challenging to determine if the learned policy truly understands fundamental driving principles, rather than simply extrapolating from the ego-vehicle's initial state. Several studies have identified this limitation and proposed algorithms to address it. However, these methods often use original datasets for evaluation. In these datasets, future trajectories are heavily dependent on initial conditions. Furthermore, IL often overfits to the most common scenarios. It struggles to generalize to rare or unseen situations.
To address these challenges, this work proposes: 1) a novel closed-loop simulator supporting both imitation and reinforcement learning, 2) a causal benchmark derived from the Waymo Open Dataset to rigorously assess the impact of the copycat problem, and 3) a novel framework integrating imitation learning and reinforcement learning to overcome the limitations of purely imitative approaches. The code for this work will be released soon. 

---
# Time Frequency Analysis of EMG Signal for Gesture Recognition using Fine grained Features 

**Authors**: Parshuram N. Aarotale, Ajita Rattani  

**Link**: [PDF](https://arxiv.org/pdf/2504.14708)  

**Abstract**: Electromyography (EMG) based hand gesture recognition converts forearm muscle activity into control commands for prosthetics, rehabilitation, and human computer interaction. This paper proposes a novel approach to EMG-based hand gesture recognition that uses fine-grained classification and presents XMANet, which unifies low-level local and high level semantic cues through cross layer mutual attention among shallow to deep CNN experts. Using stacked spectrograms and scalograms derived from the Short Time Fourier Transform (STFT) and Wavelet Transform (WT), we benchmark XMANet against ResNet50, DenseNet-121, MobileNetV3, and EfficientNetB0. Experimental results on the Grabmyo dataset indicate that, using STFT, the proposed XMANet model outperforms the baseline ResNet50, EfficientNetB0, MobileNetV3, and DenseNet121 models with improvement of approximately 1.72%, 4.38%, 5.10%, and 2.53%, respectively. When employing the WT approach, improvements of around 1.57%, 1.88%, 1.46%, and 2.05% are observed over the same baselines. Similarly, on the FORS EMG dataset, the XMANet(ResNet50) model using STFT shows an improvement of about 5.04% over the baseline ResNet50. In comparison, the XMANet(DenseNet121) and XMANet(MobileNetV3) models yield enhancements of approximately 4.11% and 2.81%, respectively. Moreover, when using WT, the proposed XMANet achieves gains of around 4.26%, 9.36%, 5.72%, and 6.09% over the baseline ResNet50, DenseNet121, MobileNetV3, and EfficientNetB0 models, respectively. These results confirm that XMANet consistently improves performance across various architectures and signal processing techniques, demonstrating the strong potential of fine grained features for accurate and robust EMG classification. 

---
# Can We Ignore Labels In Out of Distribution Detection? 

**Authors**: Hong Yang, Qi Yu, Travis Desel  

**Link**: [PDF](https://arxiv.org/pdf/2504.14704)  

**Abstract**: Out-of-distribution (OOD) detection methods have recently become more prominent, serving as a core element in safety-critical autonomous systems. One major purpose of OOD detection is to reject invalid inputs that could lead to unpredictable errors and compromise safety. Due to the cost of labeled data, recent works have investigated the feasibility of self-supervised learning (SSL) OOD detection, unlabeled OOD detection, and zero shot OOD detection. In this work, we identify a set of conditions for a theoretical guarantee of failure in unlabeled OOD detection algorithms from an information-theoretic perspective. These conditions are present in all OOD tasks dealing with real-world data: I) we provide theoretical proof of unlabeled OOD detection failure when there exists zero mutual information between the learning objective and the in-distribution labels, a.k.a. 'label blindness', II) we define a new OOD task - Adjacent OOD detection - that tests for label blindness and accounts for a previously ignored safety gap in all OOD detection benchmarks, and III) we perform experiments demonstrating that existing unlabeled OOD methods fail under conditions suggested by our label blindness theory and analyze the implications for future research in unlabeled OOD methods. 

---
# IXGS-Intraoperative 3D Reconstruction from Sparse, Arbitrarily Posed Real X-rays 

**Authors**: Sascha Jecklin, Aidana Massalimova, Ruyi Zha, Lilian Calvet, Christoph J. Laux, Mazda Farshad, Philipp Fürnstahl  

**Link**: [PDF](https://arxiv.org/pdf/2504.14699)  

**Abstract**: Spine surgery is a high-risk intervention demanding precise execution, often supported by image-based navigation systems. Recently, supervised learning approaches have gained attention for reconstructing 3D spinal anatomy from sparse fluoroscopic data, significantly reducing reliance on radiation-intensive 3D imaging systems. However, these methods typically require large amounts of annotated training data and may struggle to generalize across varying patient anatomies or imaging conditions. Instance-learning approaches like Gaussian splatting could offer an alternative by avoiding extensive annotation requirements. While Gaussian splatting has shown promise for novel view synthesis, its application to sparse, arbitrarily posed real intraoperative X-rays has remained largely unexplored. This work addresses this limitation by extending the $R^2$-Gaussian splatting framework to reconstruct anatomically consistent 3D volumes under these challenging conditions. We introduce an anatomy-guided radiographic standardization step using style transfer, improving visual consistency across views, and enhancing reconstruction quality. Notably, our framework requires no pretraining, making it inherently adaptable to new patients and anatomies. We evaluated our approach using an ex-vivo dataset. Expert surgical evaluation confirmed the clinical utility of the 3D reconstructions for navigation, especially when using 20 to 30 views, and highlighted the standardization's benefit for anatomical clarity. Benchmarking via quantitative 2D metrics (PSNR/SSIM) confirmed performance trade-offs compared to idealized settings, but also validated the improvement gained from standardization over raw inputs. This work demonstrates the feasibility of instance-based volumetric reconstruction from arbitrary sparse-view X-rays, advancing intraoperative 3D imaging for surgical navigation. 

---
# Learning Critically: Selective Self Distillation in Federated Learning on Non-IID Data 

**Authors**: Yuting He, Yiqiang Chen, XiaoDong Yang, Hanchao Yu, Yi-Hua Huang, Yang Gu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14694)  

**Abstract**: Federated learning (FL) enables multiple clients to collaboratively train a global model while keeping local data decentralized. Data heterogeneity (non-IID) across clients has imposed significant challenges to FL, which makes local models re-optimize towards their own local optima and forget the global knowledge, resulting in performance degradation and convergence slowdown. Many existing works have attempted to address the non-IID issue by adding an extra global-model-based regularizing item to the local training but without an adaption scheme, which is not efficient enough to achieve high performance with deep learning models. In this paper, we propose a Selective Self-Distillation method for Federated learning (FedSSD), which imposes adaptive constraints on the local updates by self-distilling the global model's knowledge and selectively weighting it by evaluating the credibility at both the class and sample level. The convergence guarantee of FedSSD is theoretically analyzed and extensive experiments are conducted on three public benchmark datasets, which demonstrates that FedSSD achieves better generalization and robustness in fewer communication rounds, compared with other state-of-the-art FL methods. 

---
# Video-MMLU: A Massive Multi-Discipline Lecture Understanding Benchmark 

**Authors**: Enxin Song, Wenhao Chai, Weili Xu, Jianwen Xie, Yuxuan Liu, Gaoang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14693)  

**Abstract**: Recent advancements in language multimodal models (LMMs) for video have demonstrated their potential for understanding video content, yet the task of comprehending multi-discipline lectures remains largely unexplored. We introduce Video-MMLU, a massive benchmark designed to evaluate the capabilities of LMMs in understanding Multi-Discipline Lectures. We evaluate over 90 open-source and proprietary models, ranging from 0.5B to 40B parameters. Our results highlight the limitations of current models in addressing the cognitive challenges presented by these lectures, especially in tasks requiring both perception and reasoning. Additionally, we explore how the number of visual tokens and the large language models influence performance, offering insights into the interplay between multimodal perception and reasoning in lecture comprehension. 

---
# FarsEval-PKBETS: A new diverse benchmark for evaluating Persian large language models 

**Authors**: Mehrnoush Shamsfard, Zahra Saaberi, Mostafa Karimi manesh, Seyed Mohammad Hossein Hashemi, Zahra Vatankhah, Motahareh Ramezani, Niki Pourazin, Tara Zare, Maryam Azimi, Sarina Chitsaz, Sama Khoraminejad, Morteza Mahdavi Mortazavi, Mohammad Mahdi Chizari, Sahar Maleki, Seyed Soroush Majd, Mostafa Masumi, Sayed Ali Musavi Khoeini, Amir Mohseni, Sogol Alipour  

**Link**: [PDF](https://arxiv.org/pdf/2504.14690)  

**Abstract**: Research on evaluating and analyzing large language models (LLMs) has been extensive for resource-rich languages such as English, yet their performance in languages such as Persian has received considerably less attention. This paper introduces FarsEval-PKBETS benchmark, a subset of FarsEval project for evaluating large language models in Persian. This benchmark consists of 4000 questions and answers in various formats, including multiple choice, short answer and descriptive responses. It covers a wide range of domains and tasks,including medicine, law, religion, Persian language, encyclopedic knowledge, human preferences, social knowledge, ethics and bias, text generation, and respecting others' rights. This bechmark incorporates linguistics, cultural, and local considerations relevant to the Persian language and Iran. To ensure the questions are challenging for current LLMs, three models -- Llama3-70B, PersianMind, and Dorna -- were evaluated using this benchmark. Their average accuracy was below 50%, meaning they provided fully correct answers to fewer than half of the questions. These results indicate that current language models are still far from being able to solve this benchmark 

---
# Uncovering Issues in the Radio Access Network by Looking at the Neighbors 

**Authors**: José Suárez-Varela, Andra Lutu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14686)  

**Abstract**: Mobile network operators (MNOs) manage Radio Access Networks (RANs) with massive amounts of cells over multiple radio generations (2G-5G). To handle such complexity, operations teams rely on monitoring systems, including anomaly detection tools that identify unexpected behaviors. In this paper, we present c-ANEMON, a Contextual ANomaly dEtection MONitor for the RAN based on Graph Neural Networks (GNNs). Our solution captures spatio-temporal variations by analyzing the behavior of individual cells in relation to their local neighborhoods, enabling the detection of anomalies that are independent of external mobility factors. This, in turn, allows focusing on anomalies associated with network issues (e.g., misconfigurations, equipment failures). We evaluate c-ANEMON using real-world data from a large European metropolitan area (7,890 cells; 3 months). First, we show that the GNN model within our solution generalizes effectively to cells from previously unseen areas, suggesting the possibility of using a single model across extensive deployment regions. Then, we analyze the anomalies detected by c-ANEMON through manual inspection and define several categories of long-lasting anomalies (6+ hours). Notably, 45.95% of these anomalies fall into a category that is more likely to require intervention by operations teams. 

---
# An LLM-enabled Multi-Agent Autonomous Mechatronics Design Framework 

**Authors**: Zeyu Wang, Frank P.-W. Lo, Qian Chen, Yongqi Zhang, Chen Lin, Xu Chen, Zhenhua Yu, Alexander J. Thompson, Eric M. Yeatman, Benny P. L. Lo  

**Link**: [PDF](https://arxiv.org/pdf/2504.14681)  

**Abstract**: Existing LLM-enabled multi-agent frameworks are predominantly limited to digital or simulated environments and confined to narrowly focused knowledge domain, constraining their applicability to complex engineering tasks that require the design of physical embodiment, cross-disciplinary integration, and constraint-aware reasoning. This work proposes a multi-agent autonomous mechatronics design framework, integrating expertise across mechanical design, optimization, electronics, and software engineering to autonomously generate functional prototypes with minimal direct human design input. Operating primarily through a language-driven workflow, the framework incorporates structured human feedback to ensure robust performance under real-world constraints. To validate its capabilities, the framework is applied to a real-world challenge involving autonomous water-quality monitoring and sampling, where traditional methods are labor-intensive and ecologically disruptive. Leveraging the proposed system, a fully functional autonomous vessel was developed with optimized propulsion, cost-effective electronics, and advanced control. The design process was carried out by specialized agents, including a high-level planning agent responsible for problem abstraction and dedicated agents for structural, electronics, control, and software development. This approach demonstrates the potential of LLM-based multi-agent systems to automate real-world engineering workflows and reduce reliance on extensive domain expertise. 

---
# Evaluating Temporal Plasticity in Foundation Time Series Models for Incremental Fine-tuning 

**Authors**: Jia Liu, Cheng Jinguo, Xia Fang, Zhenyuan Ma, Yuankai Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14677)  

**Abstract**: Time series foundation models excel at diverse time series forecasting tasks, but their capacity for continuous improvement through incremental learning remains unexplored. We present the first comprehensive study investigating these models' temporal plasticity - their ability to progressively enhance performance through continual learning while maintaining existing capabilities. Through experiments on real-world datasets exhibiting distribution shifts, we evaluate both conventional deep learning models and foundation models using a novel continual learning framework. Our findings reveal that while traditional models struggle with performance deterioration during incremental fine-tuning, foundation models like Time-MoE and Chronos demonstrate sustained improvement in predictive accuracy. This suggests that optimizing foundation model fine-tuning strategies may be more valuable than developing domain-specific small models. Our research introduces new evaluation methodologies and insights for developing foundation time series models with robust continuous learning capabilities. 

---
# A Case Study Exploring the Current Landscape of Synthetic Medical Record Generation with Commercial LLMs 

**Authors**: Yihan Lin, Zhirong Bella Yu, Simon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.14657)  

**Abstract**: Synthetic Electronic Health Records (EHRs) offer a valuable opportunity to create privacy preserving and harmonized structured data, supporting numerous applications in healthcare. Key benefits of synthetic data include precise control over the data schema, improved fairness and representation of patient populations, and the ability to share datasets without concerns about compromising real individuals privacy. Consequently, the AI community has increasingly turned to Large Language Models (LLMs) to generate synthetic data across various domains. However, a significant challenge in healthcare is ensuring that synthetic health records reliably generalize across different hospitals, a long standing issue in the field. In this work, we evaluate the current state of commercial LLMs for generating synthetic data and investigate multiple aspects of the generation process to identify areas where these models excel and where they fall short. Our main finding from this work is that while LLMs can reliably generate synthetic health records for smaller subsets of features, they struggle to preserve realistic distributions and correlations as the dimensionality of the data increases, ultimately limiting their ability to generalize across diverse hospital settings. 

---
# Surrogate Fitness Metrics for Interpretable Reinforcement Learning 

**Authors**: Philipp Altmann, Céline Davignon, Maximilian Zorn, Fabian Ritz, Claudia Linnhoff-Popien, Thomas Gabor  

**Link**: [PDF](https://arxiv.org/pdf/2504.14645)  

**Abstract**: We employ an evolutionary optimization framework that perturbs initial states to generate informative and diverse policy demonstrations. A joint surrogate fitness function guides the optimization by combining local diversity, behavioral certainty, and global population diversity. To assess demonstration quality, we apply a set of evaluation metrics, including the reward-based optimality gap, fidelity interquartile means (IQMs), fitness composition analysis, and trajectory visualizations. Hyperparameter sensitivity is also examined to better understand the dynamics of trajectory optimization. Our findings demonstrate that optimizing trajectory selection via surrogate fitness metrics significantly improves interpretability of RL policies in both discrete and continuous environments. In gridworld domains, evaluations reveal significantly enhanced demonstration fidelities compared to random and ablated baselines. In continuous control, the proposed framework offers valuable insights, particularly for early-stage policies, while fidelity-based optimization proves more effective for mature policies. By refining and systematically analyzing surrogate fitness functions, this study advances the interpretability of RL models. The proposed improvements provide deeper insights into RL decision-making, benefiting applications in safety-critical and explainability-focused domains. 

---
# Risk Assessment Framework for Code LLMs via Leveraging Internal States 

**Authors**: Yuheng Huang, Lei Ma, Keizaburo Nishikino, Takumi Akazaki  

**Link**: [PDF](https://arxiv.org/pdf/2504.14640)  

**Abstract**: The pre-training paradigm plays a key role in the success of Large Language Models (LLMs), which have been recognized as one of the most significant advancements of AI recently. Building on these breakthroughs, code LLMs with advanced coding capabilities bring huge impacts on software engineering, showing the tendency to become an essential part of developers' daily routines. However, the current code LLMs still face serious challenges related to trustworthiness, as they can generate incorrect, insecure, or unreliable code. Recent exploratory studies find that it can be promising to detect such risky outputs by analyzing LLMs' internal states, akin to how the human brain unconsciously recognizes its own mistakes. Yet, most of these approaches are limited to narrow sub-domains of LLM operations and fall short of achieving industry-level scalability and practicability. To address these challenges, in this paper, we propose PtTrust, a two-stage risk assessment framework for code LLM based on internal state pre-training, designed to integrate seamlessly with the existing infrastructure of software companies. The core idea is that the risk assessment framework could also undergo a pre-training process similar to LLMs. Specifically, PtTrust first performs unsupervised pre-training on large-scale unlabeled source code to learn general representations of LLM states. Then, it uses a small, labeled dataset to train a risk predictor. We demonstrate the effectiveness of PtTrust through fine-grained, code line-level risk assessment and demonstrate that it generalizes across tasks and different programming languages. Further experiments also reveal that PtTrust provides highly intuitive and interpretable features, fostering greater user trust. We believe PtTrust makes a promising step toward scalable and trustworthy assurance for code LLMs. 

---
# AlphaZero-Edu: Making AlphaZero Accessible to Everyone 

**Authors**: Binjie Guo, Hanyu Zheng, Guowei Su, Ru Zhang, Haohan Jiang, Xurong Lin, Hongyan Wei, Aisheng Mo, Jie Li, Zhiyuan Qian, Zhuhao Zhang, Xiaoyuan Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.14636)  

**Abstract**: Recent years have witnessed significant progress in reinforcement learning, especially with Zero-like paradigms, which have greatly boosted the generalization and reasoning abilities of large-scale language models. Nevertheless, existing frameworks are often plagued by high implementation complexity and poor reproducibility. To tackle these challenges, we present AlphaZero-Edu, a lightweight, education-focused implementation built upon the mathematical framework of AlphaZero. It boasts a modular architecture that disentangles key components, enabling transparent visualization of the algorithmic processes. Additionally, it is optimized for resource-efficient training on a single NVIDIA RTX 3090 GPU and features highly parallelized self-play data generation, achieving a 3.2-fold speedup with 8 processes. In Gomoku matches, the framework has demonstrated exceptional performance, achieving a consistently high win rate against human opponents. AlphaZero-Edu has been open-sourced at this https URL, providing an accessible and practical benchmark for both academic research and industrial applications. 

---
# Towards Optimal Circuit Generation: Multi-Agent Collaboration Meets Collective Intelligence 

**Authors**: Haiyan Qin, Jiahao Feng, Xiaotong Feng, Wei W. Xing, Wang Kang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14625)  

**Abstract**: Large language models (LLMs) have transformed code generation, yet their application in hardware design produces gate counts 38\%--1075\% higher than human designs. We present CircuitMind, a multi-agent framework that achieves human-competitive efficiency through three key innovations: syntax locking (constraining generation to basic logic gates), retrieval-augmented generation (enabling knowledge-driven design), and dual-reward optimization (balancing correctness with efficiency). To evaluate our approach, we introduce TC-Bench, the first gate-level benchmark harnessing collective intelligence from the TuringComplete ecosystem -- a competitive circuit design platform with hundreds of thousands of players. Experiments show CircuitMind enables 55.6\% of model implementations to match or exceed top-tier human experts in composite efficiency metrics. Most remarkably, our framework elevates the 14B Phi-4 model to outperform both GPT-4o mini and Gemini 2.0 Flash, achieving efficiency comparable to the top 25\% of human experts without requiring specialized training. These innovations establish a new paradigm for hardware optimization where collaborative AI systems leverage collective human expertise to achieve optimal circuit designs. Our model, data, and code are open-source at this https URL. 

---
# VM-BHINet:Vision Mamba Bimanual Hand Interaction Network for 3D Interacting Hand Mesh Recovery From a Single RGB Image 

**Authors**: Han Bi, Ge Yu, Yu He, Wenzhuo Liu, Zijie Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.14618)  

**Abstract**: Understanding bimanual hand interactions is essential for realistic 3D pose and shape reconstruction. However, existing methods struggle with occlusions, ambiguous appearances, and computational inefficiencies. To address these challenges, we propose Vision Mamba Bimanual Hand Interaction Network (VM-BHINet), introducing state space models (SSMs) into hand reconstruction to enhance interaction modeling while improving computational efficiency. The core component, Vision Mamba Interaction Feature Extraction Block (VM-IFEBlock), combines SSMs with local and global feature operations, enabling deep understanding of hand interactions. Experiments on the InterHand2.6M dataset show that VM-BHINet reduces Mean per-joint position error (MPJPE) and Mean per-vertex position error (MPVPE) by 2-3%, significantly surpassing state-of-the-art methods. 

---
# K2MUSE: A human lower limb multimodal dataset under diverse conditions for facilitating rehabilitation robotics 

**Authors**: Jiwei Li, Bi Zhang, Xiaowei Tan, Wanxin Chen, Zhaoyuan Liu, Juanjuan Zhang, Weiguang Huo, Jian Huang, Lianqing Liu, Xingang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.14602)  

**Abstract**: The natural interaction and control performance of lower limb rehabilitation robots are closely linked to biomechanical information from various human locomotion activities. Multidimensional human motion data significantly deepen the understanding of the complex mechanisms governing neuromuscular alterations, thereby facilitating the development and application of rehabilitation robots in multifaceted real-world environments. However, currently available lower limb datasets are inadequate for supplying the essential multimodal data and large-scale gait samples necessary for effective data-driven approaches, and they neglect the significant effects of acquisition interference in real this http URL fill this gap, we present the K2MUSE dataset, which includes a comprehensive collection of multimodal data, comprising kinematic, kinetic, amplitude-mode ultrasound (AUS), and surface electromyography (sEMG) measurements. The proposed dataset includes lower limb multimodal data from 30 able-bodied participants walking under different inclines (0$^\circ$, $\pm$5$^\circ$, and $\pm$10$^\circ$), various speeds (0.5 m/s, 1.0 m/s, and 1.5 m/s), and different nonideal acquisition conditions (muscle fatigue, electrode shifts, and inter-day differences). The kinematic and ground reaction force data were collected via a Vicon motion capture system and an instrumented treadmill with embedded force plates, whereas the sEMG and AUS data were synchronously recorded for thirteen muscles on the bilateral lower limbs. This dataset offers a new resource for designing control frameworks for rehabilitation robots and conducting biomechanical analyses of lower limb locomotion. The dataset is available at this https URL. 

---
# HealthGenie: Empowering Users with Healthy Dietary Guidance through Knowledge Graph and Large Language Models 

**Authors**: Fan Gao, Xinjie Zhao, Ding Xia, Zhongyi Zhou, Rui Yang, Jinghui Lu, Hang Jiang, Chanjun Park, Irene Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.14594)  

**Abstract**: Seeking dietary guidance often requires navigating complex professional knowledge while accommodating individual health conditions. Knowledge Graphs (KGs) offer structured and interpretable nutritional information, whereas Large Language Models (LLMs) naturally facilitate conversational recommendation delivery. In this paper, we present HealthGenie, an interactive system that combines the strengths of LLMs and KGs to provide personalized dietary recommendations along with hierarchical information visualization for a quick and intuitive overview. Upon receiving a user query, HealthGenie performs query refinement and retrieves relevant information from a pre-built KG. The system then visualizes and highlights pertinent information, organized by defined categories, while offering detailed, explainable recommendation rationales. Users can further tailor these recommendations by adjusting preferences interactively. Our evaluation, comprising a within-subject comparative experiment and an open-ended discussion, demonstrates that HealthGenie effectively supports users in obtaining personalized dietary guidance based on their health conditions while reducing interaction effort and cognitive load. These findings highlight the potential of LLM-KG integration in supporting decision-making through explainable and visualized information. We examine the system's usefulness and effectiveness with an N=12 within-subject study and provide design considerations for future systems that integrate conversational LLM and KG. 

---
# Phoenix: A Motion-based Self-Reflection Framework for Fine-grained Robotic Action Correction 

**Authors**: Wenke Xia, Ruoxuan Feng, Dong Wang, Di Hu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14588)  

**Abstract**: Building a generalizable self-correction system is crucial for robots to recover from failures. Despite advancements in Multimodal Large Language Models (MLLMs) that empower robots with semantic reflection ability for failure, translating semantic reflection into how to correct fine-grained robotic actions remains a significant challenge. To address this gap, we build the Phoenix framework, which leverages motion instruction as a bridge to connect high-level semantic reflection with low-level robotic action correction. In this motion-based self-reflection framework, we start with a dual-process motion adjustment mechanism with MLLMs to translate the semantic reflection into coarse-grained motion instruction adjustment. To leverage this motion instruction for guiding how to correct fine-grained robotic actions, a multi-task motion-conditioned diffusion policy is proposed to integrate visual observations for high-frequency robotic action correction. By combining these two models, we could shift the demand for generalization capability from the low-level manipulation policy to the MLLMs-driven motion adjustment model and facilitate precise, fine-grained robotic action correction. Utilizing this framework, we further develop a lifelong learning method to automatically improve the model's capability from interactions with dynamic environments. The experiments conducted in both the RoboMimic simulation and real-world scenarios prove the superior generalization and robustness of our framework across a variety of manipulation tasks. Our code is released at \href{this https URL}{this https URL}. 

---
# Modality Selection and Skill Segmentation via Cross-Modality Attention 

**Authors**: Jiawei Jiang, Kei Ota, Devesh K. Jha, Asako Kanezaki  

**Link**: [PDF](https://arxiv.org/pdf/2504.14573)  

**Abstract**: Incorporating additional sensory modalities such as tactile and audio into foundational robotic models poses significant challenges due to the curse of dimensionality. This work addresses this issue through modality selection. We propose a cross-modality attention (CMA) mechanism to identify and selectively utilize the modalities that are most informative for action generation at each timestep. Furthermore, we extend the application of CMA to segment primitive skills from expert demonstrations and leverage this segmentation to train a hierarchical policy capable of solving long-horizon, contact-rich manipulation tasks. 

---
# NoWag: A Unified Framework for Shape Preserving Compression of Large Language Models 

**Authors**: Lawrence Liu, Inesh Chakrabarti, Yixiao Li, Mengdi Wang, Tuo Zhao, Lin F. Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14569)  

**Abstract**: Large language models (LLMs) exhibit remarkable performance across various natural language processing tasks but suffer from immense computational and memory demands, limiting their deployment in resource-constrained environments. To address this challenge, we propose NoWag: (Normalized Weight and Activation Guided Compression), a unified framework for zero-shot shape preserving compression algorithms. We compressed Llama-2 7B/13B/70B and Llama-3 8/70BB models, using two popular forms of shape-preserving compression, vector quantization NoWag-VQ (NoWag for Vector Quantization), and unstructured/semi-structured pruning NoWag-P (NoWag for Pruning). We found that NoWag-VQ significantly outperforms state-of-the-art zero shot VQ, and that NoWag-P performs competitively against state-of-the-art methods. These results suggest commonalities between these compression paradigms that could inspire future work. Our code is available at this https URL 

---
# ReasoningV: Efficient Verilog Code Generation with Adaptive Hybrid Reasoning Model 

**Authors**: Haiyan Qin, Zhiwei Xie, Jingjing Li, Liangchen Li, Xiaotong Feng, Junzhan Liu, Wang Kang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14560)  

**Abstract**: Large Language Models (LLMs) have advanced Verilog code generation significantly, yet face challenges in data quality, reasoning capabilities, and computational efficiency. This paper presents ReasoningV, a novel model employing a hybrid reasoning strategy that integrates trained intrinsic capabilities with dynamic inference adaptation for Verilog code generation. Our framework introduces three complementary innovations: (1) ReasoningV-5K, a high-quality dataset of 5,000 functionally verified instances with reasoning paths created through multi-dimensional filtering of PyraNet samples; (2) a two-stage training approach combining parameter-efficient fine-tuning for foundational knowledge with full-parameter optimization for enhanced reasoning; and (3) an adaptive reasoning mechanism that dynamically adjusts reasoning depth based on problem complexity, reducing token consumption by up to 75\% while preserving performance. Experimental results demonstrate ReasoningV's effectiveness with a pass@1 accuracy of 57.8\% on VerilogEval-human, achieving performance competitive with leading commercial models like Gemini-2.0-flash (59.5\%) and exceeding the previous best open-source model by 10.4 percentage points. ReasoningV offers a more reliable and accessible pathway for advancing AI-driven hardware design automation, with our model, data, and code available at this https URL. 

---
# VGNC: Reducing the Overfitting of Sparse-view 3DGS via Validation-guided Gaussian Number Control 

**Authors**: Lifeng Lin, Rongfeng Lu, Quan Chen, Haofan Ren, Ming Lu, Yaoqi Sun, Chenggang Yan, Anke Xue  

**Link**: [PDF](https://arxiv.org/pdf/2504.14548)  

**Abstract**: Sparse-view 3D reconstruction is a fundamental yet challenging task in practical 3D reconstruction applications. Recently, many methods based on the 3D Gaussian Splatting (3DGS) framework have been proposed to address sparse-view 3D reconstruction. Although these methods have made considerable advancements, they still show significant issues with overfitting. To reduce the overfitting, we introduce VGNC, a novel Validation-guided Gaussian Number Control (VGNC) approach based on generative novel view synthesis (NVS) models. To the best of our knowledge, this is the first attempt to alleviate the overfitting issue of sparse-view 3DGS with generative validation images. Specifically, we first introduce a validation image generation method based on a generative NVS model. We then propose a Gaussian number control strategy that utilizes generated validation images to determine the optimal Gaussian numbers, thereby reducing the issue of overfitting. We conducted detailed experiments on various sparse-view 3DGS baselines and datasets to evaluate the effectiveness of VGNC. Extensive experiments show that our approach not only reduces overfitting but also improves rendering quality on the test set while decreasing the number of Gaussian points. This reduction lowers storage demands and accelerates both training and rendering. The code will be released. 

---
# Causality for Natural Language Processing 

**Authors**: Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2504.14530)  

**Abstract**: Causal reasoning is a cornerstone of human intelligence and a critical capability for artificial systems aiming to achieve advanced understanding and decision-making. This thesis delves into various dimensions of causal reasoning and understanding in large language models (LLMs). It encompasses a series of studies that explore the causal inference skills of LLMs, the mechanisms behind their performance, and the implications of causal and anticausal learning for natural language processing (NLP) tasks. Additionally, it investigates the application of causal reasoning in text-based computational social science, specifically focusing on political decision-making and the evaluation of scientific impact through citations. Through novel datasets, benchmark tasks, and methodological frameworks, this work identifies key challenges and opportunities to improve the causal capabilities of LLMs, providing a comprehensive foundation for future research in this evolving field. 

---
# Biased by Design: Leveraging AI Biases to Enhance Critical Thinking of News Readers 

**Authors**: Liudmila Zavolokina, Kilian Sprenkamp, Zoya Katashinskaya, Daniel Gordon Jones  

**Link**: [PDF](https://arxiv.org/pdf/2504.14522)  

**Abstract**: This paper explores the design of a propaganda detection tool using Large Language Models (LLMs). Acknowledging the inherent biases in AI models, especially in political contexts, we investigate how these biases might be leveraged to enhance critical thinking in news consumption. Countering the typical view of AI biases as detrimental, our research proposes strategies of user choice and personalization in response to a user's political stance, applying psychological concepts of confirmation bias and cognitive dissonance. We present findings from a qualitative user study, offering insights and design recommendations (bias awareness, personalization and choice, and gradual introduction of diverse perspectives) for AI tools in propaganda detection. 

---
# SlimPipe: Memory-Thrifty and Efficient Pipeline Parallelism for Long-Context LLM Training 

**Authors**: Zhouyang Li, Yuliang Liu, Wei Zhang, Tailing Yuan, Bin Chen, Chengru Song, Di Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14519)  

**Abstract**: Pipeline Parallelism (PP) serves as a crucial technique for training Large Language Models (LLMs), owing to its capability to alleviate memory pressure from model states with relatively low communication overhead. However, in long-context scenarios, existing pipeline parallelism methods fail to address the substantial activation memory pressure, primarily due to the peak memory consumption resulting from the accumulation of activations across multiple microbatches. Moreover, these approaches inevitably introduce considerable pipeline bubbles, further hindering efficiency.
To tackle these challenges, we propose SlimPipe, a novel approach to fine-grained pipeline parallelism that employs uniform sequence slicing coupled with one-forward-one-backward (1F1B) schedule. It reduces the accumulated activations from several microbatches to just one, which is split into several slices. Although the slices are evenly partitioned, the computation cost is not equal across slices due to causal attention. We develop a sophisticated workload redistribution technique to address this load imbalance. SlimPipe achieves (1) near-zero memory overhead and (2) minimal pipeline bubbles simultaneously. The effectiveness of SlimPipe has been proven by thorough testing with diverse model architectures, context window sizes, and SlimPipe-specific configurations. For example, on the Llama 70B model, compared to state-of-the-art methods, SlimPipe significantly boosts the Model FLOPs Utilization (MFU) to up to $1.57\times$ for a context length of 512K. More notably, for a context length of 2048K, it maintains over 45% utilization on 256 NVIDIA Hopper 80GB GPUs, while other approaches either suffer significant performance drops or fail entirely due to memory constraints. 

---
# On Dimension-Free Transformer: An Application of STP to AI 

**Authors**: Daizhan Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.14514)  

**Abstract**: The matrix expressions for every parts of a transformer are firstly described. Based on semi-tensor product (STP) of matrices the hypervectors are reconsidered and the linear transformation over hypervectors is constructed by using projection. Its properties and calculating formulas are obtained. Using projection-based transformation of hypervector (PBTH), the framework of dimension-free transformer (DFT) is proposed by verifying each linear transformation in a transformer and replacing it by a proper PBTH, which allows the inputs and outputs being of arbitrary dimensions. Using balanced information about all entries, DFT must be more efficient in dealing with signals. 

---
# DreamID: High-Fidelity and Fast diffusion-based Face Swapping via Triplet ID Group Learning 

**Authors**: Fulong Ye, Miao Hua, Pengze Zhang, Xinghui Li, Qichao Sun, Songtao Zhao, Qian He, Xinglong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14509)  

**Abstract**: In this paper, we introduce DreamID, a diffusion-based face swapping model that achieves high levels of ID similarity, attribute preservation, image fidelity, and fast inference speed. Unlike the typical face swapping training process, which often relies on implicit supervision and struggles to achieve satisfactory results. DreamID establishes explicit supervision for face swapping by constructing Triplet ID Group data, significantly enhancing identity similarity and attribute preservation. The iterative nature of diffusion models poses challenges for utilizing efficient image-space loss functions, as performing time-consuming multi-step sampling to obtain the generated image during training is impractical. To address this issue, we leverage the accelerated diffusion model SD Turbo, reducing the inference steps to a single iteration, enabling efficient pixel-level end-to-end training with explicit Triplet ID Group supervision. Additionally, we propose an improved diffusion-based model architecture comprising SwapNet, FaceNet, and ID Adapter. This robust architecture fully unlocks the power of the Triplet ID Group explicit supervision. Finally, to further extend our method, we explicitly modify the Triplet ID Group data during training to fine-tune and preserve specific attributes, such as glasses and face shape. Extensive experiments demonstrate that DreamID outperforms state-of-the-art methods in terms of identity similarity, pose and expression preservation, and image fidelity. Overall, DreamID achieves high-quality face swapping results at 512*512 resolution in just 0.6 seconds and performs exceptionally well in challenging scenarios such as complex lighting, large angles, and occlusions. 

---
# LBM-GNN: Graph Neural Network Enhanced Lattice Boltzmann Method 

**Authors**: Yue Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.14494)  

**Abstract**: In this paper, we present LBM-GNN, a novel approach that enhances the traditional Lattice Boltzmann Method (LBM) with Graph Neural Networks (GNNs). We apply this method to fluid dynamics simulations, demonstrating improved stability and accuracy compared to standard LBM implementations. The method is validated using benchmark problems such as the Taylor-Green vortex, focusing on accuracy, conservation properties, and performance across different Reynolds numbers and grid resolutions. Our results indicate that GNN-enhanced LBM can maintain better conservation properties while improving numerical stability at higher Reynolds numbers. 

---
# FinSage: A Multi-aspect RAG System for Financial Filings Question Answering 

**Authors**: Xinyu Wang, Jijun Chi, Zhenghan Tai, Tung Sum Thomas Kwok, Muzhi Li, Zhuhong Li, Hailin He, Yuchen Hua, Peng Lu, Suyuchen Wang, Yihong Wu, Jerry Huang, Ling Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.14493)  

**Abstract**: Leveraging large language models in real-world settings often entails a need to utilize domain-specific data and tools in order to follow the complex regulations that need to be followed for acceptable use. Within financial sectors, modern enterprises increasingly rely on Retrieval-Augmented Generation (RAG) systems to address complex compliance requirements in financial document workflows. However, existing solutions struggle to account for the inherent heterogeneity of data (e.g., text, tables, diagrams) and evolving nature of regulatory standards used in financial filings, leading to compromised accuracy in critical information extraction. We propose the FinSage framework as a solution, utilizing a multi-aspect RAG framework tailored for regulatory compliance analysis in multi-modal financial documents. FinSage introduces three innovative components: (1) a multi-modal pre-processing pipeline that unifies diverse data formats and generates chunk-level metadata summaries, (2) a multi-path sparse-dense retrieval system augmented with query expansion (HyDE) and metadata-aware semantic search, and (3) a domain-specialized re-ranking module fine-tuned via Direct Preference Optimization (DPO) to prioritize compliance-critical content. Extensive experiments demonstrate that FinSage achieves an impressive recall of 92.51% on 75 expert-curated questions derived from surpasses the best baseline method on the FinanceBench question answering datasets by 24.06% in accuracy. Moreover, FinSage has been successfully deployed as financial question-answering agent in online meetings, where it has already served more than 1,200 people. 

---
# ParaPO: Aligning Language Models to Reduce Verbatim Reproduction of Pre-training Data 

**Authors**: Tong Chen, Faeze Brahman, Jiacheng Liu, Niloofar Mireshghallah, Weijia Shi, Pang Wei Koh, Luke Zettlemoyer, Hannaneh Hajishirzi  

**Link**: [PDF](https://arxiv.org/pdf/2504.14452)  

**Abstract**: Language models (LMs) can memorize and reproduce segments from their pretraining data verbatim even in non-adversarial settings, raising concerns about copyright, plagiarism, privacy, and creativity. We introduce Paraphrase Preference Optimization (ParaPO), a post-training method that fine-tunes LMs to reduce unintentional regurgitation while preserving their overall utility. ParaPO trains LMs to prefer paraphrased versions of memorized segments over the original verbatim content from the pretraining data. To maintain the ability to recall famous quotations when appropriate, we develop a variant of ParaPO that uses system prompts to control regurgitation behavior. In our evaluation on Llama3.1-8B, ParaPO consistently reduces regurgitation across all tested datasets (e.g., reducing the regurgitation metric from 17.3 to 12.9 in creative writing), whereas unlearning methods used in prior work to mitigate regurgitation are less effective outside their targeted unlearned domain (from 17.3 to 16.9). When applied to the instruction-tuned Tulu3-8B model, ParaPO with system prompting successfully preserves famous quotation recall while reducing unintentional regurgitation (from 8.7 to 6.3 in creative writing) when prompted not to regurgitate. In contrast, without ParaPO tuning, prompting the model not to regurgitate produces only a marginal reduction (8.7 to 8.4). 

---
# LoRe: Personalizing LLMs via Low-Rank Reward Modeling 

**Authors**: Avinandan Bose, Zhihan Xiong, Yuejie Chi, Simon Shaolei Du, Lin Xiao, Maryam Fazel  

**Link**: [PDF](https://arxiv.org/pdf/2504.14439)  

**Abstract**: Personalizing large language models (LLMs) to accommodate diverse user preferences is essential for enhancing alignment and user satisfaction. Traditional reinforcement learning from human feedback (RLHF) approaches often rely on monolithic value representations, limiting their ability to adapt to individual preferences. We introduce a novel framework that leverages low-rank preference modeling to efficiently learn and generalize user-specific reward functions. By representing reward functions in a low-dimensional subspace and modeling individual preferences as weighted combinations of shared basis functions, our approach avoids rigid user categorization while enabling scalability and few-shot adaptation. We validate our method on multiple preference datasets, demonstrating superior generalization to unseen users and improved accuracy in preference prediction tasks. 

---
# ResNetVLLM -- Multi-modal Vision LLM for the Video Understanding Task 

**Authors**: Ahmad Khalil, Mahmoud Khalil, Alioune Ngom  

**Link**: [PDF](https://arxiv.org/pdf/2504.14432)  

**Abstract**: In this paper, we introduce ResNetVLLM (ResNet Vision LLM), a novel cross-modal framework for zero-shot video understanding that integrates a ResNet-based visual encoder with a Large Language Model (LLM. ResNetVLLM addresses the challenges associated with zero-shot video models by avoiding reliance on pre-trained video understanding models and instead employing a non-pretrained ResNet to extract visual features. This design ensures the model learns visual and semantic representations within a unified architecture, enhancing its ability to generate accurate and contextually relevant textual descriptions from video inputs. Our experimental results demonstrate that ResNetVLLM achieves state-of-the-art performance in zero-shot video understanding (ZSVU) on several benchmarks, including MSRVTT-QA, MSVD-QA, TGIF-QA FrameQA, and ActivityNet-QA. 

---
# ResNetVLLM-2: Addressing ResNetVLLM's Multi-Modal Hallucinations 

**Authors**: Ahmad Khalil, Mahmoud Khalil, Alioune Ngom  

**Link**: [PDF](https://arxiv.org/pdf/2504.14429)  

**Abstract**: Large Language Models (LLMs) have transformed natural language processing (NLP) tasks, but they suffer from hallucination, generating plausible yet factually incorrect content. This issue extends to Video-Language Models (VideoLLMs), where textual descriptions may inaccurately represent visual content, resulting in multi-modal hallucinations. In this paper, we address hallucination in ResNetVLLM, a video-language model combining ResNet visual encoders with LLMs. We introduce a two-step protocol: (1) a faithfulness detection strategy that uses a modified Lynx model to assess semantic alignment between generated captions and ground-truth video references, and (2) a hallucination mitigation strategy using Retrieval-Augmented Generation (RAG) with an ad-hoc knowledge base dynamically constructed during inference. Our enhanced model, ResNetVLLM-2, reduces multi-modal hallucinations by cross-verifying generated content against external knowledge, improving factual consistency. Evaluation on the ActivityNet-QA benchmark demonstrates a substantial accuracy increase from 54.8% to 65.3%, highlighting the effectiveness of our hallucination detection and mitigation strategies in enhancing video-language model reliability. 

---
# Optimizing SIA Development: A Case Study in User-Centered Design for Estuary, a Multimodal Socially Interactive Agent Framework 

**Authors**: Spencer Lin, Miru Jun, Basem Rizk, Karen Shieh, Scott Fisher, Sharon Mozgai  

**Link**: [PDF](https://arxiv.org/pdf/2504.14427)  

**Abstract**: This case study presents our user-centered design model for Socially Intelligent Agent (SIA) development frameworks through our experience developing Estuary, an open source multimodal framework for building low-latency real-time socially interactive agents. We leverage the Rapid Assessment Process (RAP) to collect the thoughts of leading researchers in the field of SIAs regarding the current state of the art for SIA development as well as their evaluation of how well Estuary may potentially address current research gaps. We achieve this through a series of end-user interviews conducted by a fellow researcher in the community. We hope that the findings of our work will not only assist the continued development of Estuary but also guide the development of other future frameworks and technologies for SIAs. 

---
# Adversarial Attack for RGB-Event based Visual Object Tracking 

**Authors**: Qiang Chen, Xiao Wang, Haowen Wang, Bo Jiang, Lin Zhu, Dawei Zhang, Yonghong Tian, Jin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14423)  

**Abstract**: Visual object tracking is a crucial research topic in the fields of computer vision and multi-modal fusion. Among various approaches, robust visual tracking that combines RGB frames with Event streams has attracted increasing attention from researchers. While striving for high accuracy and efficiency in tracking, it is also important to explore how to effectively conduct adversarial attacks and defenses on RGB-Event stream tracking algorithms, yet research in this area remains relatively scarce. To bridge this gap, in this paper, we propose a cross-modal adversarial attack algorithm for RGB-Event visual tracking. Because of the diverse representations of Event streams, and given that Event voxels and frames are more commonly used, this paper will focus on these two representations for an in-depth study. Specifically, for the RGB-Event voxel, we first optimize the perturbation by adversarial loss to generate RGB frame adversarial examples. For discrete Event voxel representations, we propose a two-step attack strategy, more in detail, we first inject Event voxels into the target region as initialized adversarial examples, then, conduct a gradient-guided optimization by perturbing the spatial location of the Event voxels. For the RGB-Event frame based tracking, we optimize the cross-modal universal perturbation by integrating the gradient information from multimodal data. We evaluate the proposed approach against attacks on three widely used RGB-Event Tracking datasets, i.e., COESOT, FE108, and VisEvent. Extensive experiments show that our method significantly reduces the performance of the tracker across numerous datasets in both unimodal and multimodal scenarios. The source code will be released on this https URL 

---
# Planet as a Brain: Towards Internet of AgentSites based on AIOS Server 

**Authors**: Xiang Zhang, Yongfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14411)  

**Abstract**: The internet is undergoing a historical transformation from the "Internet of Websites" to the "Internet of AgentSites." While traditional Websites served as the foundation for information hosting and dissemination, a new frontier is emerging where AgentSites serve as the hubs of the internet, where each AgentSite hosts one or more AI agents that receive tasks, address them, and deliver actionable solutions, marking a significant shift in the digital landscape and representing the next generation of online ecosystems. Under this vision, AIOS, the AI Agent Operating System, serves as the server for the development, deployment and execution of AI agents, which is a fundamental infrastructure for the Internet of Agentsites.
In this paper, we introduce AIOS Server, a runtime framework to host agents and enable global-scale collaboration among decentralized agents. AIOS Server provides a communication protocol leveraging the Model Context Protocol (MCP) and JSON-RPC to enable agent-agent or human-agent interactions. Each AIOS node operates as a server to host and execute agents, while supporting peer-to-peer coordination without reliance on centralized orchestration. Based on AIOS Server, we further present the world's first practically deployed Internet of Agentsites (AIOS-IoA), including AgentHub for agent registration and discovery and AgentChat for interactive communication, at this https URL. The agent discovery mechanism based on Distributed Hash Tables (DHT) and a Gossip protocol serves as the search engine for the internet of agentsites. This work provides a practical foundation for building the Internet of Agentsites-a new paradigm where autonomous agents become first-class citizens of the web. The implementation is available at this https URL and will be integrated into the AIOS main branch at this https URL. 

---
# Data Augmentation Using Neural Acoustic Fields With Retrieval-Augmented Pre-training 

**Authors**: Christopher Ick, Gordon Wichern, Yoshiki Masuyama, François G. Germain, Jonathan Le Roux  

**Link**: [PDF](https://arxiv.org/pdf/2504.14409)  

**Abstract**: This report details MERL's system for room impulse response (RIR) estimation submitted to the Generative Data Augmentation Workshop at ICASSP 2025 for Augmenting RIR Data (Task 1) and Improving Speaker Distance Estimation (Task 2). We first pre-train a neural acoustic field conditioned by room geometry on an external large-scale dataset in which pairs of RIRs and the geometries are provided. The neural acoustic field is then adapted to each target room by using the enrollment data, where we leverage either the provided room geometries or geometries retrieved from the external dataset, depending on availability. Lastly, we predict the RIRs for each pair of source and receiver locations specified by Task 1, and use these RIRs to train the speaker distance estimation model in Task 2. 

---
# ScholarMate: A Mixed-Initiative Tool for Qualitative Knowledge Work and Information Sensemaking 

**Authors**: Runlong Ye, Patrick Yung Kang Lee, Matthew Varona, Oliver Huang, Carolina Nobre  

**Link**: [PDF](https://arxiv.org/pdf/2504.14406)  

**Abstract**: Synthesizing knowledge from large document collections is a critical yet increasingly complex aspect of qualitative research and knowledge work. While AI offers automation potential, effectively integrating it into human-centric sensemaking workflows remains challenging. We present ScholarMate, an interactive system designed to augment qualitative analysis by unifying AI assistance with human oversight. ScholarMate enables researchers to dynamically arrange and interact with text snippets on a non-linear canvas, leveraging AI for theme suggestions, multi-level summarization, and contextual naming, while ensuring transparency through traceability to source documents. Initial pilot studies indicated that users value this mixed-initiative approach, finding the balance between AI suggestions and direct manipulation crucial for maintaining interpretability and trust. We further demonstrate the system's capability through a case study analyzing 24 papers. By balancing automation with human control, ScholarMate enhances efficiency and supports interpretability, offering a valuable approach for productive human-AI collaboration in demanding sensemaking tasks common in knowledge work. 

---
# Hydra: An Agentic Reasoning Approach for Enhancing Adversarial Robustness and Mitigating Hallucinations in Vision-Language Models 

**Authors**: Chung-En, Hsuan-Chih, Chen, Brian Jalaian, Nathaniel D. Bastian  

**Link**: [PDF](https://arxiv.org/pdf/2504.14395)  

**Abstract**: To develop trustworthy Vision-Language Models (VLMs), it is essential to address adversarial robustness and hallucination mitigation, both of which impact factual accuracy in high-stakes applications such as defense and healthcare. Existing methods primarily focus on either adversarial defense or hallucination post-hoc correction, leaving a gap in unified robustness strategies. We introduce \textbf{Hydra}, an adaptive agentic framework that enhances plug-in VLMs through iterative reasoning, structured critiques, and cross-model verification, improving both resilience to adversarial perturbations and intrinsic model errors. Hydra employs an Action-Critique Loop, where it retrieves and critiques visual information, leveraging Chain-of-Thought (CoT) and In-Context Learning (ICL) techniques to refine outputs dynamically. Unlike static post-hoc correction methods, Hydra adapts to both adversarial manipulations and intrinsic model errors, making it robust to malicious perturbations and hallucination-related inaccuracies. We evaluate Hydra on four VLMs, three hallucination benchmarks, two adversarial attack strategies, and two adversarial defense methods, assessing performance on both clean and adversarial inputs. Results show that Hydra surpasses plug-in VLMs and state-of-the-art (SOTA) dehallucination methods, even without explicit adversarial defenses, demonstrating enhanced robustness and factual consistency. By bridging adversarial resistance and hallucination mitigation, Hydra provides a scalable, training-free solution for improving the reliability of VLMs in real-world applications. 

---
# LOOPE: Learnable Optimal Patch Order in Positional Embeddings for Vision Transformers 

**Authors**: Md Abtahi Majeed Chowdhury, Md Rifat Ur Rahman, Akil Ahmad Taki  

**Link**: [PDF](https://arxiv.org/pdf/2504.14386)  

**Abstract**: Positional embeddings (PE) play a crucial role in Vision Transformers (ViTs) by providing spatial information otherwise lost due to the permutation invariant nature of self attention. While absolute positional embeddings (APE) have shown theoretical advantages over relative positional embeddings (RPE), particularly due to the ability of sinusoidal functions to preserve spatial inductive biases like monotonicity and shift invariance, a fundamental challenge arises when mapping a 2D grid to a 1D sequence. Existing methods have mostly overlooked or never explored the impact of patch ordering in positional embeddings. To address this, we propose LOOPE, a learnable patch-ordering method that optimizes spatial representation for a given set of frequencies, providing a principled approach to patch order optimization. Empirical results show that our PE significantly improves classification accuracy across various ViT architectures. To rigorously evaluate the effectiveness of positional embeddings, we introduce the "Three Cell Experiment", a novel benchmarking framework that assesses the ability of PEs to retain relative and absolute positional information across different ViT architectures. Unlike standard evaluations, which typically report a performance gap of 4 to 6% between models with and without PE, our method reveals a striking 30 to 35% difference, offering a more sensitive diagnostic tool to measure the efficacy of PEs. Our experimental analysis confirms that the proposed LOOPE demonstrates enhanced effectiveness in retaining both relative and absolute positional information. 

---
# Learning Enhanced Structural Representations with Block-Based Uncertainties for Ocean Floor Mapping 

**Authors**: Jose Marie Antonio Minoza  

**Link**: [PDF](https://arxiv.org/pdf/2504.14372)  

**Abstract**: Accurate ocean modeling and coastal hazard prediction depend on high-resolution bathymetric data; yet, current worldwide datasets are too coarse for exact numerical simulations. While recent deep learning advances have improved earth observation data resolution, existing methods struggle with the unique challenges of producing detailed ocean floor maps, especially in maintaining physical structure consistency and quantifying uncertainties. This work presents a novel uncertainty-aware mechanism using spatial blocks to efficiently capture local bathymetric complexity based on block-based conformal prediction. Using the Vector Quantized Variational Autoencoder (VQ-VAE) architecture, the integration of this uncertainty quantification framework yields spatially adaptive confidence estimates while preserving topographical features via discrete latent representations. With smaller uncertainty widths in well-characterized areas and appropriately larger bounds in areas of complex seafloor structures, the block-based design adapts uncertainty estimates to local bathymetric complexity. Compared to conventional techniques, experimental results over several ocean regions show notable increases in both reconstruction quality and uncertainty estimation reliability. This framework increases the reliability of bathymetric reconstructions by preserving structural integrity while offering spatially adaptive uncertainty estimates, so opening the path for more solid climate modeling and coastal hazard assessment. 

---
# Diverse Prompts: Illuminating the Prompt Space of Large Language Models with MAP-Elites 

**Authors**: Gabriel Machado Santos, Rita Maria da Silva Julia, Marcelo Zanchetta do Nascimento  

**Link**: [PDF](https://arxiv.org/pdf/2504.14367)  

**Abstract**: Prompt engineering is essential for optimizing large language models (LLMs), yet the link between prompt structures and task performance remains underexplored. This work introduces an evolutionary approach that combines context-free grammar (CFG) with the MAP-Elites algorithm to systematically explore the prompt space. Our method prioritizes quality and diversity, generating high-performing and structurally varied prompts while analyzing their alignment with diverse tasks by varying traits such as the number of examples (shots) and reasoning depth. By systematically mapping the phenotypic space, we reveal how structural variations influence LLM performance, offering actionable insights for task-specific and adaptable prompt design. Evaluated on seven BigBench Lite tasks across multiple LLMs, our results underscore the critical interplay of quality and diversity, advancing the effectiveness and versatility of LLMs. 

---
# Empirical Evaluation of Knowledge Distillation from Transformers to Subquadratic Language Models 

**Authors**: Patrick Haller, Jonas Golde, Alan Akbik  

**Link**: [PDF](https://arxiv.org/pdf/2504.14366)  

**Abstract**: Knowledge distillation is a widely used technique for compressing large language models (LLMs) by training a smaller student model to mimic a larger teacher model. Typically, both the teacher and student are Transformer-based architectures, leveraging softmax attention for sequence modeling. However, the quadratic complexity of self-attention at inference time remains a significant bottleneck, motivating the exploration of subquadratic alternatives such as structured state-space models (SSMs), linear attention, and recurrent architectures. In this work, we systematically evaluate the transferability of knowledge distillation from a Transformer teacher to nine subquadratic student architectures. Our study aims to determine which subquadratic model best aligns with the teacher's learned representations and how different architectural constraints influence the distillation process. We also investigate the impact of intelligent initialization strategies, including matrix mixing and query-key-value (QKV) copying, on the adaptation process. Our empirical results on multiple NLP benchmarks provide insights into the trade-offs between efficiency and performance, highlighting key factors for successful knowledge transfer to subquadratic architectures. 

---
# Accelerating LLM Inference with Flexible N:M Sparsity via A Fully Digital Compute-in-Memory Accelerator 

**Authors**: Akshat Ramachandran, Souvik Kundu, Arnab Raha, Shamik Kundu, Deepak K. Mathaikutty, Tushar Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2504.14365)  

**Abstract**: Large language model (LLM) pruning with fixed N:M structured sparsity significantly limits the expressivity of the sparse model, yielding sub-optimal performance. In contrast, supporting multiple N:M patterns to provide sparse representational freedom introduces costly overhead in hardware. To address these challenges for LLMs, we first present a flexible layer-wise outlier-density-aware N:M sparsity (FLOW) selection method. FLOW enables the identification of optimal layer-wise N and M values (from a given range) by simultaneously accounting for the presence and distribution of outliers, allowing a higher degree of representational freedom. To deploy sparse models with such N:M flexibility, we then introduce a flexible, low-overhead digital compute-in-memory architecture (FlexCiM). FlexCiM supports diverse sparsity patterns by partitioning a digital CiM (DCiM) macro into smaller sub-macros, which are adaptively aggregated and disaggregated through distribution and merging mechanisms for different N and M values. Extensive experiments on both transformer-based and recurrence-based state space foundation models (SSMs) demonstrate that FLOW outperforms existing alternatives with an accuracy improvement of up to 36%, while FlexCiM achieves up to 1.75x lower inference latency and 1.5x lower energy consumption compared to existing sparse accelerators. Code is available at: this https URL 

---
# A Multimodal Recaptioning Framework to Account for Perceptual Diversity in Multilingual Vision-Language Modeling 

**Authors**: Kyle Buettner, Jacob Emmerson, Adriana Kovashka  

**Link**: [PDF](https://arxiv.org/pdf/2504.14359)  

**Abstract**: There are many ways to describe, name, and group objects when captioning an image. Differences are evident when speakers come from diverse cultures due to the unique experiences that shape perception. Machine translation of captions has pushed multilingual capabilities in vision-language models (VLMs), but data comes mainly from English speakers, indicating a perceptual bias and lack of model flexibility. In this work, we address this challenge and outline a data-efficient framework to instill multilingual VLMs with greater understanding of perceptual diversity. We specifically propose an LLM-based, multimodal recaptioning strategy that alters the object descriptions of English captions before translation. The greatest benefits are demonstrated in a targeted multimodal mechanism guided by native speaker data. By adding produced rewrites as augmentations in training, we improve on German and Japanese text-image retrieval cases studies (up to +3.5 mean recall overall, +4.7 on non-native error cases). We further propose a mechanism to analyze the specific object description differences across datasets, and we offer insights into cross-dataset and cross-language generalization. 

---
# Integrating LLM-Generated Views into Mean-Variance Optimization Using the Black-Litterman Model 

**Authors**: Youngbin Lee, Yejin Kim, Suin Kim, Yongjae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.14345)  

**Abstract**: Portfolio optimization faces challenges due to the sensitivity in traditional mean-variance models. The Black-Litterman model mitigates this by integrating investor views, but defining these views remains difficult. This study explores the integration of large language models (LLMs) generated views into portfolio optimization using the Black-Litterman framework. Our method leverages LLMs to estimate expected stock returns from historical prices and company metadata, incorporating uncertainty through the variance in predictions. We conduct a backtest of the LLM-optimized portfolios from June 2024 to February 2025, rebalancing biweekly using the previous two weeks of price data. As baselines, we compare against the S&P 500, an equal-weighted portfolio, and a traditional mean-variance optimized portfolio constructed using the same set of stocks. Empirical results suggest that different LLMs exhibit varying levels of predictive optimism and confidence stability, which impact portfolio performance. The source code and data are available at this https URL. 

---
# Visual Prompting for One-shot Controllable Video Editing without Inversion 

**Authors**: Zhengbo Zhang, Yuxi Zhou, Duo Peng, Joo-Hwee Lim, Zhigang Tu, De Wen Soh, Lin Geng Foo  

**Link**: [PDF](https://arxiv.org/pdf/2504.14335)  

**Abstract**: One-shot controllable video editing (OCVE) is an important yet challenging task, aiming to propagate user edits that are made -- using any image editing tool -- on the first frame of a video to all subsequent frames, while ensuring content consistency between edited frames and source frames. To achieve this, prior methods employ DDIM inversion to transform source frames into latent noise, which is then fed into a pre-trained diffusion model, conditioned on the user-edited first frame, to generate the edited video. However, the DDIM inversion process accumulates errors, which hinder the latent noise from accurately reconstructing the source frames, ultimately compromising content consistency in the generated edited frames. To overcome it, our method eliminates the need for DDIM inversion by performing OCVE through a novel perspective based on visual prompting. Furthermore, inspired by consistency models that can perform multi-step consistency sampling to generate a sequence of content-consistent images, we propose a content consistency sampling (CCS) to ensure content consistency between the generated edited frames and the source frames. Moreover, we introduce a temporal-content consistency sampling (TCS) based on Stein Variational Gradient Descent to ensure temporal consistency across the edited frames. Extensive experiments validate the effectiveness of our approach. 

---
# Expanding the Generative AI Design Space through Structured Prompting and Multimodal Interfaces 

**Authors**: Nimisha Karnatak, Adrien Baranes, Rob Marchant, Huinan Zeng, Tríona Butler, Kristen Olson  

**Link**: [PDF](https://arxiv.org/pdf/2504.14320)  

**Abstract**: Text-based prompting remains the dominant interaction paradigm in generative AI, yet it often results in a high-friction experience for novice users, such as small business owners (SBOs), attempting to articulate creative or domain-specific goals for advertising. To investigate this challenge, we conducted a study with six SBOs in the United Kingdom, focusing on their advertising practices and perceptions and usage of AI tools in this context. Our findings surfaced two persistent breakdowns in current generative AI systems: first, the cognitive burden of prompt engineering, as users struggled to translate abstract creative goals into effective textual inputs; and second, the frequent generation of generic outputs that failed to align with users' articulated brand vision. To address these issues, we developed ACAI (AI Co-Creation for Advertising and Inspiration), a multimodal, GenAI-powered advertisement creation tool designed to support novice designers by reimagining the prompt interface. ACAI features a structured, panel-based interface composed of three modules: the Branding Panel, the Audience & Goals Panel, and the Inspiration Board Panel to provide SBOs with outputs that align with their creative vision by reducing prompt ambiguity. This work contributes to HCI research on generative systems by showing how structured interfaces can foreground user-defined context to improve both alignment and promptability in novice workflows. 

---
# Learning to Score 

**Authors**: Yogev Kriger, Shai Fine  

**Link**: [PDF](https://arxiv.org/pdf/2504.14302)  

**Abstract**: Common machine learning settings range from supervised tasks, where accurately labeled data is accessible, through semi-supervised and weakly-supervised tasks, where target labels are scant or noisy, to unsupervised tasks where labels are unobtainable. In this paper we study a scenario where the target labels are not available but additional related information is at hand. This information, referred to as Side Information, is either correlated with the unknown labels or imposes constraints on the feature space. We formulate the problem as an ensemble of three semantic components: representation learning, side information and metric learning. The proposed scoring model is advantageous for multiple use-cases. For example, in the healthcare domain it can be used to create a severity score for diseases where the symptoms are known but the criteria for the disease progression are not well defined. We demonstrate the utility of the suggested scoring system on well-known benchmark data-sets and bio-medical patient records. 

---
# Balancing Privacy and Action Performance: A Penalty-Driven Approach to Image Anonymization 

**Authors**: Nazia Aslam, Kamal Nasrollahi  

**Link**: [PDF](https://arxiv.org/pdf/2504.14301)  

**Abstract**: The rapid development of video surveillance systems for object detection, tracking, activity recognition, and anomaly detection has revolutionized our day-to-day lives while setting alarms for privacy concerns. It isn't easy to strike a balance between visual privacy and action recognition performance in most computer vision models. Is it possible to safeguard privacy without sacrificing performance? It poses a formidable challenge, as even minor privacy enhancements can lead to substantial performance degradation. To address this challenge, we propose a privacy-preserving image anonymization technique that optimizes the anonymizer using penalties from the utility branch, ensuring improved action recognition performance while minimally affecting privacy leakage. This approach addresses the trade-off between minimizing privacy leakage and maintaining high action performance. The proposed approach is primarily designed to align with the regulatory standards of the EU AI Act and GDPR, ensuring the protection of personally identifiable information while maintaining action performance. To the best of our knowledge, we are the first to introduce a feature-based penalty scheme that exclusively controls the action features, allowing freedom to anonymize private attributes. Extensive experiments were conducted to validate the effectiveness of the proposed method. The results demonstrate that applying a penalty to anonymizer from utility branch enhances action performance while maintaining nearly consistent privacy leakage across different penalty settings. 

---
# Learning and Generating Diverse Residential Load Patterns Using GAN with Weakly-Supervised Training and Weight Selection 

**Authors**: Xinyu Liang, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14300)  

**Abstract**: The scarcity of high-quality residential load data can pose obstacles for decarbonizing the residential sector as well as effective grid planning and operation. The above challenges have motivated research into generating synthetic load data, but existing methods faced limitations in terms of scalability, diversity, and similarity. This paper proposes a Generative Adversarial Network-based Synthetic Residential Load Pattern (RLP-GAN) generation model, a novel weakly-supervised GAN framework, leveraging an over-complete autoencoder to capture dependencies within complex and diverse load patterns and learn household-level data distribution at scale. We incorporate a model weight selection method to address the mode collapse problem and generate load patterns with high diversity. We develop a holistic evaluation method to validate the effectiveness of RLP-GAN using real-world data of 417 households. The results demonstrate that RLP-GAN outperforms state-of-the-art models in capturing temporal dependencies and generating load patterns with higher similarity to real data. Furthermore, we have publicly released the RLP-GAN generated synthetic dataset, which comprises one million synthetic residential load pattern profiles. 

---
# Experience-based Refinement of Task Planning Knowledge in Autonomous Robots 

**Authors**: Hadeel Jazzaa, Thomas McCluskey, David Peebles  

**Link**: [PDF](https://arxiv.org/pdf/2504.14259)  

**Abstract**: The requirement for autonomous robots to exhibit higher-level cognitive skills by planning and adapting in an ever-changing environment is indeed a great challenge for the AI community. Progress has been made in the automated planning community on refinement and repair of an agent's symbolic knowledge to do task planning in an incomplete or changing environmental model, but these advances up to now have not been transferred to real physical robots. This paper demonstrates how a physical robot can be capable of adapting its symbolic knowledge of the environment, by using experiences in robot action execution to drive knowledge refinement and hence to improve the success rate of the task plans the robot creates. To implement more robust planning systems, we propose a method for refining domain knowledge to improve the knowledge on which intelligent robot behavior is based. This architecture has been implemented and evaluated using a NAO robot. The refined knowledge leads to the future synthesis of task plans which demonstrate decreasing rates of failure over time as faulty knowledge is removed or adjusted. 

---
# SimplifyMyText: An LLM-Based System for Inclusive Plain Language Text Simplification 

**Authors**: Michael Färber, Parisa Aghdam, Kyuri Im, Mario Tawfelis, Hardik Ghoshal  

**Link**: [PDF](https://arxiv.org/pdf/2504.14223)  

**Abstract**: Text simplification is essential for making complex content accessible to diverse audiences who face comprehension challenges. Yet, the limited availability of simplified materials creates significant barriers to personal and professional growth and hinders social inclusion. Although researchers have explored various methods for automatic text simplification, none fully leverage large language models (LLMs) to offer tailored customization for different target groups and varying levels of simplicity. Moreover, despite its proven benefits for both consumers and organizations, the well-established practice of plain language remains underutilized. In this paper, we this https URL, the first system designed to produce plain language content from multiple input formats, including typed text and file uploads, with flexible customization options for diverse audiences. We employ GPT-4 and Llama-3 and evaluate outputs across multiple metrics. Overall, our work contributes to research on automatic text simplification and highlights the importance of tailored communication in promoting inclusivity. 

---
# Decomposition-based multi-scale transformer framework for time series anomaly detection 

**Authors**: Wenxin Zhang, Cuicui Luo  

**Link**: [PDF](https://arxiv.org/pdf/2504.14206)  

**Abstract**: Time series anomaly detection is crucial for maintaining stable systems. Existing methods face two main challenges. First, it is difficult to directly model the dependencies of diverse and complex patterns within the sequences. Second, many methods that optimize parameters using mean squared error struggle with noise in the time series, leading to performance deterioration. To address these challenges, we propose a transformer-based framework built on decomposition (TransDe) for multivariate time series anomaly detection. The key idea is to combine the strengths of time series decomposition and transformers to effectively learn the complex patterns in normal time series data. A multi-scale patch-based transformer architecture is proposed to exploit the representative dependencies of each decomposed component of the time series. Furthermore, a contrastive learn paradigm based on patch operation is proposed, which leverages KL divergence to align the positive pairs, namely the pure representations of normal patterns between different patch-level views. A novel asynchronous loss function with a stop-gradient strategy is further introduced to enhance the performance of TransDe effectively. It can avoid time-consuming and labor-intensive computation costs in the optimization process. Extensive experiments on five public datasets are conducted and TransDe shows superiority compared with twelve baselines in terms of F1 score. Our code is available at this https URL. 

---
# Dual-channel Heterophilic Message Passing for Graph Fraud Detection 

**Authors**: Wenxin Zhang, Jingxing Zhong, Guangzhen Yao, Renda Han, Xiaojian Lin, Zeyu Zhang, Cuicui Luo  

**Link**: [PDF](https://arxiv.org/pdf/2504.14205)  

**Abstract**: Fraudulent activities have significantly increased across various domains, such as e-commerce, online review platforms, and social networks, making fraud detection a critical task. Spatial Graph Neural Networks (GNNs) have been successfully applied to fraud detection tasks due to their strong inductive learning capabilities. However, existing spatial GNN-based methods often enhance the graph structure by excluding heterophilic neighbors during message passing to align with the homophilic bias of GNNs. Unfortunately, this approach can disrupt the original graph topology and increase uncertainty in predictions. To address these limitations, this paper proposes a novel framework, Dual-channel Heterophilic Message Passing (DHMP), for fraud detection. DHMP leverages a heterophily separation module to divide the graph into homophilic and heterophilic subgraphs, mitigating the low-pass inductive bias of traditional GNNs. It then applies shared weights to capture signals at different frequencies independently and incorporates a customized sampling strategy for training. This allows nodes to adaptively balance the contributions of various signals based on their labels. Extensive experiments on three real-world datasets demonstrate that DHMP outperforms existing methods, highlighting the importance of separating signals with different frequencies for improved fraud detection. The code is available at this https URL. 

---
# DConAD: A Differencing-based Contrastive Representation Learning Framework for Time Series Anomaly Detection 

**Authors**: Wenxin Zhang, Xiaojian Lin, Wenjun Yu, Guangzhen Yao, jingxiang Zhong, Yu Li, Renda Han, Songcheng Xu, Hao Shi, Cuicui Luo  

**Link**: [PDF](https://arxiv.org/pdf/2504.14204)  

**Abstract**: Time series anomaly detection holds notable importance for risk identification and fault detection across diverse application domains. Unsupervised learning methods have become popular because they have no requirement for labels. However, due to the challenges posed by the multiplicity of abnormal patterns, the sparsity of anomalies, and the growth of data scale and complexity, these methods often fail to capture robust and representative dependencies within the time series for identifying anomalies. To enhance the ability of models to capture normal patterns of time series and avoid the retrogression of modeling ability triggered by the dependencies on high-quality prior knowledge, we propose a differencing-based contrastive representation learning framework for time series anomaly detection (DConAD). Specifically, DConAD generates differential data to provide additional information about time series and utilizes transformer-based architecture to capture spatiotemporal dependencies, which enhances the robustness of unbiased representation learning ability. Furthermore, DConAD implements a novel KL divergence-based contrastive learning paradigm that only uses positive samples to avoid deviation from reconstruction and deploys the stop-gradient strategy to compel convergence. Extensive experiments on five public datasets show the superiority and effectiveness of DConAD compared with nine baselines. The code is available at this https URL. 

---
# Learning Joint ID-Textual Representation for ID-Preserving Image Synthesis 

**Authors**: Zichuan Liu, Liming Jiang, Qing Yan, Yumin Jia, Hao Kang, Xin Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14202)  

**Abstract**: We propose a novel framework for ID-preserving generation using a multi-modal encoding strategy rather than injecting identity features via adapters into pre-trained models. Our method treats identity and text as a unified conditioning input. To achieve this, we introduce FaceCLIP, a multi-modal encoder that learns a joint embedding space for both identity and textual semantics. Given a reference face and a text prompt, FaceCLIP produces a unified representation that encodes both identity and text, which conditions a base diffusion model to generate images that are identity-consistent and text-aligned. We also present a multi-modal alignment algorithm to train FaceCLIP, using a loss that aligns its joint representation with face, text, and image embedding spaces. We then build FaceCLIP-SDXL, an ID-preserving image synthesis pipeline by integrating FaceCLIP with Stable Diffusion XL (SDXL). Compared to prior methods, FaceCLIP-SDXL enables photorealistic portrait generation with better identity preservation and textual relevance. Extensive experiments demonstrate its quantitative and qualitative superiority. 

---
# Enhancing Multimodal In-Context Learning for Image Classification through Coreset Optimization 

**Authors**: Huiyi Chen, Jiawei Peng, Kaihua Tang, Xin Geng, Xu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14200)  

**Abstract**: In-context learning (ICL) enables Large Vision-Language Models (LVLMs) to adapt to new tasks without parameter updates, using a few demonstrations from a large support set. However, selecting informative demonstrations leads to high computational and memory costs. While some methods explore selecting a small and representative coreset in the text classification, evaluating all support set samples remains costly, and discarded samples lead to unnecessary information loss. These methods may also be less effective for image classification due to differences in feature spaces. Given these limitations, we propose Key-based Coreset Optimization (KeCO), a novel framework that leverages untapped data to construct a compact and informative coreset. We introduce visual features as keys within the coreset, which serve as the anchor for identifying samples to be updated through different selection strategies. By leveraging untapped samples from the support set, we update the keys of selected coreset samples, enabling the randomly initialized coreset to evolve into a more informative coreset under low computational cost. Through extensive experiments on coarse-grained and fine-grained image classification benchmarks, we demonstrate that KeCO effectively enhances ICL performance for image classification task, achieving an average improvement of more than 20\%. Notably, we evaluate KeCO under a simulated online scenario, and the strong performance in this scenario highlights the practical value of our framework for resource-constrained real-world scenarios. 

---
# A Physics-guided Multimodal Transformer Path to Weather and Climate Sciences 

**Authors**: Jing Han, Hanting Chen, Kai Han, Xiaomeng Huang, Yongyun Hu, Wenjun Xu, Dacheng Tao, Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14174)  

**Abstract**: With the rapid development of machine learning in recent years, many problems in meteorology can now be addressed using AI models. In particular, data-driven algorithms have significantly improved accuracy compared to traditional methods. Meteorological data is often transformed into 2D images or 3D videos, which are then fed into AI models for learning. Additionally, these models often incorporate physical signals, such as temperature, pressure, and wind speed, to further enhance accuracy and interpretability. In this paper, we review several representative AI + Weather/Climate algorithms and propose a new paradigm where observational data from different perspectives, each with distinct physical meanings, are treated as multimodal data and integrated via transformers. Furthermore, key weather and climate knowledge can be incorporated through regularization techniques to further strengthen the model's capabilities. This new paradigm is versatile and can address a variety of tasks, offering strong generalizability. We also discuss future directions for improving model accuracy and interpretability. 

---
# Breaking the Diffraction Barrier for Passive Sources: Parameter-Decoupled Superresolution Assisted by Physics-Informed Machine Learning 

**Authors**: Abdelali Sajia, Bilal Benzimoun, Pawan Khatiwada, Guogan Zhao, Xiao-Feng Qian  

**Link**: [PDF](https://arxiv.org/pdf/2504.14156)  

**Abstract**: We present a parameter-decoupled superresolution framework for estimating sub-wavelength separations of passive two-point sources without requiring prior knowledge or control of the source. Our theoretical foundation circumvents the need to estimate multiple challenging parameters such as partial coherence, brightness imbalance, random relative phase, and photon statistics. A physics-informed machine learning (ML) model (trained with a standard desktop workstation), synergistically integrating this theory, further addresses practical imperfections including background noise, photon loss, and centroid/orientation misalignment. The integrated parameter-decoupling superresolution method achieves resolution 14 and more times below the diffraction limit (corresponding to ~ 13.5 nm in optical microscopy) on experimentally generated realistic images with >82% fidelity, performance rivaling state-of-the-art techniques for actively controllable sources. Critically, our method's robustness against source parameter variability and source-independent noises enables potential applications in realistic scenarios where source control is infeasible, such as astrophysical imaging, live-cell microscopy, and quantum metrology. This work bridges a critical gap between theoretical superresolution limits and practical implementations for passive systems. 

---
# SConU: Selective Conformal Uncertainty in Large Language Models 

**Authors**: Zhiyuan Wang, Qingni Wang, Yue Zhang, Tianlong Chen, Xiaofeng Zhu, Xiaoshuang Shi, Kaidi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14154)  

**Abstract**: As large language models are increasingly utilized in real-world applications, guarantees of task-specific metrics are essential for their reliable deployment. Previous studies have introduced various criteria of conformal uncertainty grounded in split conformal prediction, which offer user-specified correctness coverage. However, existing frameworks often fail to identify uncertainty data outliers that violate the exchangeability assumption, leading to unbounded miscoverage rates and unactionable prediction sets. In this paper, we propose a novel approach termed Selective Conformal Uncertainty (SConU), which, for the first time, implements significance tests, by developing two conformal p-values that are instrumental in determining whether a given sample deviates from the uncertainty distribution of the calibration set at a specific manageable risk level. Our approach not only facilitates rigorous management of miscoverage rates across both single-domain and interdisciplinary contexts, but also enhances the efficiency of predictions. Furthermore, we comprehensively analyze the components of the conformal procedures, aiming to approximate conditional coverage, particularly in high-stakes question-answering tasks. 

---
# Locate 3D: Real-World Object Localization via Self-Supervised Learning in 3D 

**Authors**: Sergio Arnaud, Paul McVay, Ada Martin, Arjun Majumdar, Krishna Murthy Jatavallabhula, Phillip Thomas, Ruslan Partsey, Daniel Dugas, Abha Gejji, Alexander Sax, Vincent-Pierre Berges, Mikael Henaff, Ayush Jain, Ang Cao, Ishita Prasad, Mrinal Kalakrishnan, Michael Rabbat, Nicolas Ballas, Mido Assran, Oleksandr Maksymets, Aravind Rajeswaran, Franziska Meier  

**Link**: [PDF](https://arxiv.org/pdf/2504.14151)  

**Abstract**: We present LOCATE 3D, a model for localizing objects in 3D scenes from referring expressions like "the small coffee table between the sofa and the lamp." LOCATE 3D sets a new state-of-the-art on standard referential grounding benchmarks and showcases robust generalization capabilities. Notably, LOCATE 3D operates directly on sensor observation streams (posed RGB-D frames), enabling real-world deployment on robots and AR devices. Key to our approach is 3D-JEPA, a novel self-supervised learning (SSL) algorithm applicable to sensor point clouds. It takes as input a 3D pointcloud featurized using 2D foundation models (CLIP, DINO). Subsequently, masked prediction in latent space is employed as a pretext task to aid the self-supervised learning of contextualized pointcloud features. Once trained, the 3D-JEPA encoder is finetuned alongside a language-conditioned decoder to jointly predict 3D masks and bounding boxes. Additionally, we introduce LOCATE 3D DATASET, a new dataset for 3D referential grounding, spanning multiple capture setups with over 130K annotations. This enables a systematic study of generalization capabilities as well as a stronger model. 

---
# Walk the Talk? Measuring the Faithfulness of Large Language Model Explanations 

**Authors**: Katie Matton, Robert Osazuwa Ness, John Guttag, Emre Kıcıman  

**Link**: [PDF](https://arxiv.org/pdf/2504.14150)  

**Abstract**: Large language models (LLMs) are capable of generating plausible explanations of how they arrived at an answer to a question. However, these explanations can misrepresent the model's "reasoning" process, i.e., they can be unfaithful. This, in turn, can lead to over-trust and misuse. We introduce a new approach for measuring the faithfulness of LLM explanations. First, we provide a rigorous definition of faithfulness. Since LLM explanations mimic human explanations, they often reference high-level concepts in the input question that purportedly influenced the model. We define faithfulness in terms of the difference between the set of concepts that LLM explanations imply are influential and the set that truly are. Second, we present a novel method for estimating faithfulness that is based on: (1) using an auxiliary LLM to modify the values of concepts within model inputs to create realistic counterfactuals, and (2) using a Bayesian hierarchical model to quantify the causal effects of concepts at both the example- and dataset-level. Our experiments show that our method can be used to quantify and discover interpretable patterns of unfaithfulness. On a social bias task, we uncover cases where LLM explanations hide the influence of social bias. On a medical question answering task, we uncover cases where LLM explanations provide misleading claims about which pieces of evidence influenced the model's decisions. 

---
# HF4Rec: Human-Like Feedback-Driven Optimization Framework for Explainable Recommendation 

**Authors**: Jiakai Tang, Jingsen Zhang, Zihang Tian, Xueyang Feng, Lei Wang, Xu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.14147)  

**Abstract**: Recent advancements in explainable recommendation have greatly bolstered user experience by elucidating the decision-making rationale. However, the existing methods actually fail to provide effective feedback signals for potentially better or worse generated explanations due to their reliance on traditional supervised learning paradigms in sparse interaction data. To address these issues, we propose a novel human-like feedback-driven optimization framework. This framework employs a dynamic interactive optimization mechanism for achieving human-centered explainable requirements without incurring high labor costs. Specifically, we propose to utilize large language models (LLMs) as human simulators to predict human-like feedback for guiding the learning process. To enable the LLMs to deeply understand the task essence and meet user's diverse personalized requirements, we introduce a human-induced customized reward scoring method, which helps stimulate the language understanding and logical reasoning capabilities of LLMs. Furthermore, considering the potential conflicts between different perspectives of explanation quality, we introduce a principled Pareto optimization that transforms the multi-perspective quality enhancement task into a multi-objective optimization problem for improving explanation performance. At last, to achieve efficient model training, we design an off-policy optimization pipeline. By incorporating a replay buffer and addressing the data distribution biases, we can effectively improve data utilization and enhance model generality. Extensive experiments on four datasets demonstrate the superiority of our approach. 

---
# PipeWeaver: Addressing Data Dynamicity in Large Multimodal Model Training with Dynamic Interleaved Pipeline 

**Authors**: Zhenliang Xue, Hanpeng Hu, Xing Chen, Yimin Jiang, Yixin Song, Zeyu Mi, Yibo Zhu, Daxin Jiang, Yubin Xia, Haibo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.14145)  

**Abstract**: Large multimodal models (LMMs) have demonstrated excellent capabilities in both understanding and generation tasks with various modalities. While these models can accept flexible combinations of input data, their training efficiency suffers from two major issues: pipeline stage imbalance caused by heterogeneous model architectures, and training data dynamicity stemming from the diversity of multimodal data.
In this paper, we present PipeWeaver, a dynamic pipeline scheduling framework designed for LMM training. The core of PipeWeaver is dynamic interleaved pipeline, which searches for pipeline schedules dynamically tailored to current training batches. PipeWeaver addresses issues of LMM training with two techniques: adaptive modality-aware partitioning and efficient pipeline schedule search within a hierarchical schedule space. Meanwhile, PipeWeaver utilizes SEMU (Step Emulator), a training simulator for multimodal models, for accurate performance estimations, accelerated by spatial-temporal subgraph reuse to improve search efficiency. Experiments show that PipeWeaver can enhance LMM training efficiency by up to 97.3% compared to state-of-the-art systems, and demonstrate excellent adaptivity to LMM training's data dynamicity. 

---
# ThyroidEffi 1.0: A Cost-Effective System for High-Performance Multi-Class Thyroid Carcinoma Classification 

**Authors**: Hai Pham-Ngoc, De Nguyen-Van, Dung Vu-Tien, Phuong Le-Hong  

**Link**: [PDF](https://arxiv.org/pdf/2504.14139)  

**Abstract**: Background: Automated classification of thyroid fine needle aspiration biopsy (FNAB) images faces challenges in limited data, inter-observer variability, and computational cost. Efficient, interpretable models are crucial for clinical support. Objective: To develop and externally validate a deep learning system for the multi-class classification of thyroid FNAB images into three key categories that directly guide post-biopsy treatment decisions in Vietnam: benign (B2), suspicious for malignancy (B5), and malignant (B6), while achieving high diagnostic accuracy with low computational overhead. Methods: Our framework features: (1) YOLOv10-based cell cluster detection for informative sub-region extraction and noise reduction; (2) a curriculum learning-inspired protocol sequencing localized crops to full images for multi-scale feature capture; (3) adaptive lightweight EfficientNetB0 (4 millions parameters) selection balancing performance and efficiency; and (4) a Transformer-inspired module for multi-scale, multi-region analysis. External validation used 1,015 independent FNAB images. Results: ThyroidEffi Basic achieved a macro F1 of 89.19\% and AUCs of 0.98 (B2), 0.95 (B5), and 0.96 (B6) on the internal test set. External validation yielded AUCs of 0.9495 (B2), 0.7436 (B5), and 0.8396 (B6). ThyroidEffi Premium improved macro F1 to 89.77\%. Grad-CAM highlighted key diagnostic regions, confirming interpretability. The system processed 1000 cases in 30 seconds, demonstrating feasibility on widely accessible hardware like a 12-core CPU. Conclusions: This work demonstrates that high-accuracy, interpretable thyroid FNAB image classification is achievable with minimal computational demands. 

---
# Personalized News Recommendation with Multi-granularity Candidate-aware User Modeling 

**Authors**: Qiang Li, Xinze Lin, Shenghao Lv, Faliang Huang, Xiangju Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.14130)  

**Abstract**: Matching candidate news with user interests is crucial for personalized news recommendations. Most existing methods can represent a user's reading interests through a single profile based on clicked news, which may not fully capture the diversity of user interests. Although some approaches incorporate candidate news or topic information, they remain insufficient because they neglect the multi-granularity relatedness between candidate news and user interests. To address this, this study proposed a multi-granularity candidate-aware user modeling framework that integrated user interest features across various levels of granularity. It consisted of two main components: candidate news encoding and user modeling. A news textual information extractor and a knowledge-enhanced entity information extractor can capture candidate news features, and word-level, entity-level, and news-level candidate-aware mechanisms can provide a comprehensive representation of user interests. Extensive experiments on a real-world dataset demonstrated that the proposed model could significantly outperform baseline models. 

---
# Exploring Language Patterns of Prompts in Text-to-Image Generation and Their Impact on Visual Diversity 

**Authors**: Maria-Teresa De Rosa Palmini, Eva Cetinic  

**Link**: [PDF](https://arxiv.org/pdf/2504.14125)  

**Abstract**: Following the initial excitement, Text-to-Image (TTI) models are now being examined more critically. While much of the discourse has focused on biases and stereotypes embedded in large-scale training datasets, the sociotechnical dynamics of user interactions with these models remain underexplored. This study examines the linguistic and semantic choices users make when crafting prompts and how these choices influence the diversity of generated outputs. Analyzing over six million prompts from the Civiverse dataset on the CivitAI platform across seven months, we categorize users into three groups based on their levels of linguistic experimentation: consistent repeaters, occasional repeaters, and non-repeaters. Our findings reveal that as user participation grows over time, prompt language becomes increasingly homogenized through the adoption of popular community tags and descriptors, with repeated prompts comprising 40-50% of submissions. At the same time, semantic similarity and topic preferences remain relatively stable, emphasizing common subjects and surface aesthetics. Using Vendi scores to quantify visual diversity, we demonstrate a clear correlation between lexical similarity in prompts and the visual similarity of generated images, showing that linguistic repetition reinforces less diverse representations. These findings highlight the significant role of user-driven factors in shaping AI-generated imagery, beyond inherent model biases, and underscore the need for tools and practices that encourage greater linguistic and thematic experimentation within TTI systems to foster more inclusive and diverse AI-generated content. 

---
# Longitudinal Study on Social and Emotional Use of AI Conversational Agent 

**Authors**: Mohit Chandra, Javier Hernandez, Gonzalo Ramos, Mahsa Ershadi, Ananya Bhattacharjee, Judith Amores, Ebele Okoli, Ann Paradiso, Shahed Warreth, Jina Suh  

**Link**: [PDF](https://arxiv.org/pdf/2504.14112)  

**Abstract**: Development in digital technologies has continuously reshaped how individuals seek and receive social and emotional support. While online platforms and communities have long served this need, the increased integration of general-purpose conversational AI into daily lives has introduced new dynamics in how support is provided and experienced. Existing research has highlighted both benefits (e.g., wider access to well-being resources) and potential risks (e.g., over-reliance) of using AI for support seeking. In this five-week, exploratory study, we recruited 149 participants divided into two usage groups: a baseline usage group (BU, n=60) that used the internet and AI as usual, and an active usage group (AU, n=89) encouraged to use one of four commercially available AI tools (Microsoft Copilot, Google Gemini, PI AI, ChatGPT) for social and emotional interactions. Our analysis revealed significant increases in perceived attachment towards AI (32.99 percentage points), perceived AI empathy (25.8 p.p.), and motivation to use AI for entertainment (22.90 p.p.) among the AU group. We also observed that individual differences (e.g., gender identity, prior AI usage) influenced perceptions of AI empathy and attachment. Lastly, the AU group expressed higher comfort in seeking personal help, managing stress, obtaining social support, and talking about health with AI, indicating potential for broader emotional support while highlighting the need for safeguards against problematic usage. Overall, our exploratory findings underscore the importance of developing consumer-facing AI tools that support emotional well-being responsibly, while empowering users to understand the limitations of these tools. 

---
# System of Agentic AI for the Discovery of Metal-Organic Frameworks 

**Authors**: Theo Jaffrelot Inizan, Sherry Yang, Aaron Kaplan, Yen-hsu Lin, Jian Yin, Saber Mirzaei, Mona Abdelgaid, Ali H. Alawadhi, KwangHwan Cho, Zhiling Zheng, Ekin Dogus Cubuk, Christian Borgs, Jennifer T. Chayes, Kristin A. Persson, Omar M. Yaghi  

**Link**: [PDF](https://arxiv.org/pdf/2504.14110)  

**Abstract**: Generative models and machine learning promise accelerated material discovery in MOFs for CO2 capture and water harvesting but face significant challenges navigating vast chemical spaces while ensuring synthetizability. Here, we present MOFGen, a system of Agentic AI comprising interconnected agents: a large language model that proposes novel MOF compositions, a diffusion model that generates crystal structures, quantum mechanical agents that optimize and filter candidates, and synthetic-feasibility agents guided by expert rules and machine learning. Trained on all experimentally reported MOFs and computational databases, MOFGen generated hundreds of thousands of novel MOF structures and synthesizable organic linkers. Our methodology was validated through high-throughput experiments and the successful synthesis of five "AI-dreamt" MOFs, representing a major step toward automated synthesizable material discovery. 

---
# Amplify Initiative: Building A Localized Data Platform for Globalized AI 

**Authors**: Qazi Mamunur Rashid, Erin van Liemt, Tiffany Shih, Amber Ebinama, Karla Barrios Ramos, Madhurima Maji, Aishwarya Verma, Charu Kalia, Jamila Smith-Loud, Joyce Nakatumba-Nabende, Rehema Baguma, Andrew Katumba, Chodrine Mutebi, Jagen Marvin, Eric Peter Wairagala, Mugizi Bruce, Peter Oketta, Lawrence Nderu, Obichi Obiajunwa, Abigail Oppong, Michael Zimba, Data Authors  

**Link**: [PDF](https://arxiv.org/pdf/2504.14105)  

**Abstract**: Current AI models often fail to account for local context and language, given the predominance of English and Western internet content in their training data. This hinders the global relevance, usefulness, and safety of these models as they gain more users around the globe. Amplify Initiative, a data platform and methodology, leverages expert communities to collect diverse, high-quality data to address the limitations of these models. The platform is designed to enable co-creation of datasets, provide access to high-quality multilingual datasets, and offer recognition to data authors. This paper presents the approach to co-creating datasets with domain experts (e.g., health workers, teachers) through a pilot conducted in Sub-Saharan Africa (Ghana, Kenya, Malawi, Nigeria, and Uganda). In partnership with local researchers situated in these countries, the pilot demonstrated an end-to-end approach to co-creating data with 155 experts in sensitive domains (e.g., physicians, bankers, anthropologists, human and civil rights advocates). This approach, implemented with an Android app, resulted in an annotated dataset of 8,091 adversarial queries in seven languages (e.g., Luganda, Swahili, Chichewa), capturing nuanced and contextual information related to key themes such as misinformation and public interest topics. This dataset in turn can be used to evaluate models for their safety and cultural relevance within the context of these languages. 

---
# Coordinating Spinal and Limb Dynamics for Enhanced Sprawling Robot Mobility 

**Authors**: Merve Atasever, Ali Okhovat, Azhang Nazaripouya, John Nisbet, Omer Kurkutlu, Jyotirmoy V. Deshmukh, Yasemin Ozkan Aydin  

**Link**: [PDF](https://arxiv.org/pdf/2504.14103)  

**Abstract**: Among vertebrates, salamanders, with their unique ability to transition between walking and swimming gaits, highlight the role of spinal mobility in locomotion. A flexible spine enables undulation of the body through a wavelike motion along the spine, aiding navigation over uneven terrains and obstacles. Yet environmental uncertainties, such as surface irregularities and variations in friction, can significantly disrupt body-limb coordination and cause discrepancies between predictions from mathematical models and real-world outcomes. Addressing this challenge requires the development of sophisticated control strategies capable of dynamically adapting to uncertain conditions while maintaining efficient locomotion. Deep reinforcement learning (DRL) offers a promising framework for handling non-deterministic environments and enabling robotic systems to adapt effectively and perform robustly under challenging conditions. In this study, we comparatively examine learning-based control strategies and biologically inspired gait design methods on a salamander-like robot. 

---
# 6G WavesFM: A Foundation Model for Sensing, Communication, and Localization 

**Authors**: Ahmed Aboulfotouh, Elsayed Mohammed, Hatem Abou-Zeid  

**Link**: [PDF](https://arxiv.org/pdf/2504.14100)  

**Abstract**: This paper introduces WavesFM, a novel Wireless Foundation Model (WFM) framework, capable of supporting a wide array of communication, sensing, and localization tasks. Our proposed architecture combines a shared Vision Transformer (ViT) backbone with task-specific multi-layer perceptron (MLP) heads and incorporates Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning. This design promotes full parameter sharing across tasks, significantly reducing the computational and memory footprint without sacrificing performance. The model processes both image-like wireless modalities, such as spectrograms and channel state information (CSI), and in-phase and quadrature (IQ) signals arranged as orthogonal frequency-division multiplexing (OFDM) resource grids. We demonstrate the strong generalization capabilities of WavesFM through extensive experiments on four downstream tasks: Fifth Generation New Radio (5G NR) positioning; multiple-input multiple-output OFDM (MIMO-OFDM) channel estimation; human activity sensing; and radio-frequency (RF) signal classification. Compared to supervised baselines trained individually, our approach achieves superior performance while sharing 80% of its parameters across tasks. Furthermore, we show that pretraining on domain-relevant data not only boosts performance but also accelerates convergence, reducing training time by up to 5x. These results demonstrate that our unified WFM can support diverse tasks and deliver significant gains in both performance and efficiency, highlighting the transformative potential of foundation models to drive AI-native paradigms in future sixth-generation (6G) networks. 

---
# Enhancing Math Learning in an LMS Using AI-Driven Question Recommendations 

**Authors**: Justus Råmunddal  

**Link**: [PDF](https://arxiv.org/pdf/2504.14098)  

**Abstract**: This paper presents an AI-driven approach to enhance math learning in a modern Learning Management System (LMS) by recommending similar math questions. Deep embeddings for math questions are generated using Meta's Llama-3.2-11B-Vision-Instruct model, and three recommendation methods-cosine similarity, Self-Organizing Maps (SOM), and Gaussian Mixture Models (GMM)-are applied to identify similar questions. User interaction data, including session durations, response times, and correctness, are used to evaluate the methods. Our findings suggest that while cosine similarity produces nearly identical question matches, SOM yields higher user satisfaction whereas GMM generally underperforms, indicating that introducing variety to a certain degree may enhance engagement and thereby potential learning outcomes until variety is no longer balanced reasonably, which our data about the implementations of all three methods demonstrate. 

---
# Leakage and Interpretability in Concept-Based Models 

**Authors**: Enrico Parisini, Tapabrata Chakraborti, Chris Harbron, Ben D. MacArthur, Christopher R. S. Banerji  

**Link**: [PDF](https://arxiv.org/pdf/2504.14094)  

**Abstract**: Concept Bottleneck Models aim to improve interpretability by predicting high-level intermediate concepts, representing a promising approach for deployment in high-risk scenarios. However, they are known to suffer from information leakage, whereby models exploit unintended information encoded within the learned concepts. We introduce an information-theoretic framework to rigorously characterise and quantify leakage, and define two complementary measures: the concepts-task leakage (CTL) and interconcept leakage (ICL) scores. We show that these measures are strongly predictive of model behaviour under interventions and outperform existing alternatives in robustness and reliability. Using this framework, we identify the primary causes of leakage and provide strong evidence that Concept Embedding Models exhibit substantial leakage regardless of the hyperparameters choice. Finally, we propose practical guidelines for designing concept-based models to reduce leakage and ensure interpretability. 

---
# LogicTree: Structured Proof Exploration for Coherent and Rigorous Logical Reasoning with Large Language Models 

**Authors**: Kang He, Kaushik Roy  

**Link**: [PDF](https://arxiv.org/pdf/2504.14089)  

**Abstract**: Large language models (LLMs) have achieved remarkable multi-step reasoning capabilities across various domains. However, LLMs still face distinct challenges in complex logical reasoning, as (1) proof-finding requires systematic exploration and the maintenance of logical coherence and (2) searching the right combination of premises at each reasoning step is inherently challenging in tasks with large premise space. To address this, we propose LogicTree, an inference-time modular framework employing algorithm-guided search to automate structured proof exploration and ensure logical coherence. Advancing beyond tree-of-thought (ToT), we incorporate caching mechanism into LogicTree to enable effective utilization of historical knowledge, preventing reasoning stagnation and minimizing redundancy. Furthermore, we address the combinatorial complexity of premise search by decomposing it into a linear process. The refined premise selection restricts subsequent inference to at most one derivation per step, enhancing reasoning granularity and enforcing strict step-by-step reasoning. Additionally, we introduce two LLM-free heuristics for premise prioritization, enabling strategic proof search. Experimental results on five datasets demonstrate that LogicTree optimally scales inference-time computation to achieve higher proof accuracy, surpassing chain-of-thought (CoT) and ToT with average gains of 23.6% and 12.5%, respectively, on GPT-4o. Moreover, within LogicTree, GPT-4o outperforms o3-mini by 7.6% on average. 

---
# Evaluating Human-AI Interaction via Usability, User Experience and Acceptance Measures for MMM-C: A Creative AI System for Music Composition 

**Authors**: Renaud Bougueng Tchemeube, Jeff Ens, Cale Plut, Philippe Pasquier, Maryam Safi, Yvan Grabit, Jean-Baptiste Rolland  

**Link**: [PDF](https://arxiv.org/pdf/2504.14071)  

**Abstract**: With the rise of artificial intelligence (AI), there has been increasing interest in human-AI co-creation in a variety of artistic domains including music as AI-driven systems are frequently able to generate human-competitive artifacts. Now, the implications of such systems for musical practice are being investigated. We report on a thorough evaluation of the user adoption of the Multi-Track Music Machine (MMM) as a co-creative AI tool for music composers. To do this, we integrate MMM into Cubase, a popular Digital Audio Workstation (DAW) by Steinberg, by producing a "1-parameter" plugin interface named MMM-Cubase (MMM-C), which enables human-AI co-composition. We contribute a methodological assemblage as a 3-part mixed method study measuring usability, user experience and technology acceptance of the system across two groups of expert-level composers: hobbyists and professionals. Results show positive usability and acceptance scores. Users report experiences of novelty, surprise and ease of use from using the system, and limitations on controllability and predictability of the interface when generating music. Findings indicate no significant difference between the two user groups. 

---
# A CMOS Probabilistic Computing Chip With In-situ hardware Aware Learning 

**Authors**: Jinesh Jhonsa, William Whitehead, David McCarthy, Shuvro Chowdhury, Kerem Camsari, Luke Theogarajan  

**Link**: [PDF](https://arxiv.org/pdf/2504.14070)  

**Abstract**: This paper demonstrates a probabilistic bit physics inspired solver with 440 spins configured in a Chimera graph, occupying an area of 0.44 mm^2. Area efficiency is maximized through a current-mode implementation of the neuron update circuit, standard cell design for analog blocks pitch-matched to digital blocks, and a shared power supply for both digital and analog components. Process variation related mismatches introduced by this approach are effectively mitigated using a hardware aware contrastive divergence algorithm during training. We validate the chip's ability to perform probabilistic computing tasks such as modeling logic gates and full adders, as well as optimization tasks such as MaxCut, demonstrating its potential for AI and machine learning applications. 

---
# Occlusion-Ordered Semantic Instance Segmentation 

**Authors**: Soroosh Baselizadeh, Cheuk-To Yu, Olga Veksler, Yuri Boykov  

**Link**: [PDF](https://arxiv.org/pdf/2504.14054)  

**Abstract**: Standard semantic instance segmentation provides useful, but inherently 2D information from a single image. To enable 3D analysis, one usually integrates absolute monocular depth estimation with instance segmentation. However, monocular depth is a difficult task. Instead, we leverage a simpler single-image task, occlusion-based relative depth ordering, providing coarser but useful 3D information. We show that relative depth ordering works more reliably from occlusions than from absolute depth. We propose to solve the joint task of relative depth ordering and segmentation of instances based on occlusions. We call this task Occlusion-Ordered Semantic Instance Segmentation (OOSIS). We develop an approach to OOSIS that extracts instances and their occlusion order simultaneously from oriented occlusion boundaries and semantic segmentation. Unlike popular detect-and-segment framework for instance segmentation, combining occlusion ordering with instance segmentation allows a simple and clean formulation of OOSIS as a labeling problem. As a part of our solution for OOSIS, we develop a novel oriented occlusion boundaries approach that significantly outperforms prior work. We also develop a new joint OOSIS metric based both on instance mask accuracy and correctness of their occlusion order. We achieve better performance than strong baselines on KINS and COCOA datasets. 

---
# Sentiment Analysis of Airbnb Reviews: Exploring Their Impact on Acceptance Rates and Pricing Across Multiple U.S. Regions 

**Authors**: Ali Safari  

**Link**: [PDF](https://arxiv.org/pdf/2504.14053)  

**Abstract**: This research examines whether Airbnb guests' positive and negative comments influence acceptance rates and rental prices across six U.S. regions: Rhode Island, Broward County, Chicago, Dallas, San Diego, and Boston. Thousands of reviews were collected and analyzed using Natural Language Processing (NLP) to classify sentiments as positive or negative, followed by statistical testing (t-tests and basic correlations) on the average scores. The findings reveal that over 90 percent of reviews in each region are positive, indicating that having additional reviews does not significantly enhance prices. However, listings with predominantly positive feedback exhibit slightly higher acceptance rates, suggesting that sentiment polarity, rather than the sheer volume of reviews, is a more critical factor for host success. Additionally, budget listings often gather extensive reviews while maintaining competitive pricing, whereas premium listings sustain higher prices with fewer but highly positive reviews. These results underscore the importance of sentiment quality over quantity in shaping guest behavior and pricing strategies in an overwhelmingly positive review environment. 

---
# A synthetic dataset of French electric load curves with temperature conditioning 

**Authors**: Tahar Nabil, Ghislain Agoua, Pierre Cauchois, Anne De Moliner, Benoît Grossin  

**Link**: [PDF](https://arxiv.org/pdf/2504.14046)  

**Abstract**: The undergoing energy transition is causing behavioral changes in electricity use, e.g. with self-consumption of local generation, or flexibility services for demand control. To better understand these changes and the challenges they induce, accessing individual smart meter data is crucial. Yet this is personal data under the European GDPR. A widespread use of such data requires thus to create synthetic realistic and privacy-preserving samples. This paper introduces a new synthetic load curve dataset generated by conditional latent diffusion. We also provide the contracted power, time-of-use plan and local temperature used for generation. Fidelity, utility and privacy of the dataset are thoroughly evaluated, demonstrating its good quality and thereby supporting its interest for energy modeling applications. 

---
# MEQA: A Meta-Evaluation Framework for Question & Answer LLM Benchmarks 

**Authors**: Jaime Raldua Veuthey, Zainab Ali Majid, Suhas Hariharan, Jacob Haimes  

**Link**: [PDF](https://arxiv.org/pdf/2504.14039)  

**Abstract**: As Large Language Models (LLMs) advance, their potential for widespread societal impact grows simultaneously. Hence, rigorous LLM evaluations are both a technical necessity and social imperative. While numerous evaluation benchmarks have been developed, there remains a critical gap in meta-evaluation: effectively assessing benchmarks' quality. We propose MEQA, a framework for the meta-evaluation of question and answer (QA) benchmarks, to provide standardized assessments, quantifiable scores, and enable meaningful intra-benchmark comparisons. We demonstrate this approach on cybersecurity benchmarks, using human and LLM evaluators, highlighting the benchmarks' strengths and weaknesses. We motivate our choice of test domain by AI models' dual nature as powerful defensive tools and security threats. 

---
# Flowco: Rethinking Data Analysis in the Age of LLMs 

**Authors**: Stephen N. Freund, Brooke Simon, Emery D. Berger, Eunice Jun  

**Link**: [PDF](https://arxiv.org/pdf/2504.14038)  

**Abstract**: Conducting data analysis typically involves authoring code to transform, visualize, analyze, and interpret data. Large language models (LLMs) are now capable of generating such code for simple, routine analyses. LLMs promise to democratize data science by enabling those with limited programming expertise to conduct data analyses, including in scientific research, business, and policymaking. However, analysts in many real-world settings must often exercise fine-grained control over specific analysis steps, verify intermediate results explicitly, and iteratively refine their analytical approaches. Such tasks present barriers to building robust and reproducible analyses using LLMs alone or even in conjunction with existing authoring tools (e.g., computational notebooks). This paper introduces Flowco, a new mixed-initiative system to address these challenges. Flowco leverages a visual dataflow programming model and integrates LLMs into every phase of the authoring process. A user study suggests that Flowco supports analysts, particularly those with less programming experience, in quickly authoring, debugging, and refining data analyses. 

---
# LoftUp: Learning a Coordinate-Based Feature Upsampler for Vision Foundation Models 

**Authors**: Haiwen Huang, Anpei Chen, Volodymyr Havrylov, Andreas Geiger, Dan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14032)  

**Abstract**: Vision foundation models (VFMs) such as DINOv2 and CLIP have achieved impressive results on various downstream tasks, but their limited feature resolution hampers performance in applications requiring pixel-level understanding. Feature upsampling offers a promising direction to address this challenge. In this work, we identify two critical factors for enhancing feature upsampling: the upsampler architecture and the training objective. For the upsampler architecture, we introduce a coordinate-based cross-attention transformer that integrates the high-resolution images with coordinates and low-resolution VFM features to generate sharp, high-quality features. For the training objective, we propose constructing high-resolution pseudo-groundtruth features by leveraging class-agnostic masks and self-distillation. Our approach effectively captures fine-grained details and adapts flexibly to various input and feature resolutions. Through experiments, we demonstrate that our approach significantly outperforms existing feature upsampling techniques across various downstream tasks. Our code is released at this https URL. 

---
# Causal pieces: analysing and improving spiking neural networks piece by piece 

**Authors**: Dominik Dold, Philipp Christian Petersen  

**Link**: [PDF](https://arxiv.org/pdf/2504.14015)  

**Abstract**: We introduce a novel concept for spiking neural networks (SNNs) derived from the idea of "linear pieces" used to analyse the expressiveness and trainability of artificial neural networks (ANNs). We prove that the input domain of SNNs decomposes into distinct causal regions where its output spike times are locally Lipschitz continuous with respect to the input spike times and network parameters. The number of such regions - which we call "causal pieces" - is a measure of the approximation capabilities of SNNs. In particular, we demonstrate in simulation that parameter initialisations which yield a high number of causal pieces on the training set strongly correlate with SNN training success. Moreover, we find that feedforward SNNs with purely positive weights exhibit a surprisingly high number of causal pieces, allowing them to achieve competitive performance levels on benchmark tasks. We believe that causal pieces are not only a powerful and principled tool for improving SNNs, but might also open up new ways of comparing SNNs and ANNs in the future. 

---
# Fashion-RAG: Multimodal Fashion Image Editing via Retrieval-Augmented Generation 

**Authors**: Fulvio Sanguigni, Davide Morelli, Marcella Cornia, Rita Cucchiara  

**Link**: [PDF](https://arxiv.org/pdf/2504.14011)  

**Abstract**: In recent years, the fashion industry has increasingly adopted AI technologies to enhance customer experience, driven by the proliferation of e-commerce platforms and virtual applications. Among the various tasks, virtual try-on and multimodal fashion image editing -- which utilizes diverse input modalities such as text, garment sketches, and body poses -- have become a key area of research. Diffusion models have emerged as a leading approach for such generative tasks, offering superior image quality and diversity. However, most existing virtual try-on methods rely on having a specific garment input, which is often impractical in real-world scenarios where users may only provide textual specifications. To address this limitation, in this work we introduce Fashion Retrieval-Augmented Generation (Fashion-RAG), a novel method that enables the customization of fashion items based on user preferences provided in textual form. Our approach retrieves multiple garments that match the input specifications and generates a personalized image by incorporating attributes from the retrieved items. To achieve this, we employ textual inversion techniques, where retrieved garment images are projected into the textual embedding space of the Stable Diffusion text encoder, allowing seamless integration of retrieved elements into the generative process. Experimental results on the Dress Code dataset demonstrate that Fashion-RAG outperforms existing methods both qualitatively and quantitatively, effectively capturing fine-grained visual details from retrieved garments. To the best of our knowledge, this is the first work to introduce a retrieval-augmented generation approach specifically tailored for multimodal fashion image editing. 

---
# CPR: Leveraging LLMs for Topic and Phrase Suggestion to Facilitate Comprehensive Product Reviews 

**Authors**: Ekta Gujral, Apurva Sinha, Lishi Ji, Bijayani Sanghamitra Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2504.13993)  

**Abstract**: Consumers often heavily rely on online product reviews, analyzing both quantitative ratings and textual descriptions to assess product quality. However, existing research hasn't adequately addressed how to systematically encourage the creation of comprehensive reviews that capture both customers sentiment and detailed product feature analysis. This paper presents CPR, a novel methodology that leverages the power of Large Language Models (LLMs) and Topic Modeling to guide users in crafting insightful and well-rounded reviews. Our approach employs a three-stage process: first, we present users with product-specific terms for rating; second, we generate targeted phrase suggestions based on these ratings; and third, we integrate user-written text through topic modeling, ensuring all key aspects are addressed. We evaluate CPR using text-to-text LLMs, comparing its performance against real-world customer reviews from Walmart. Our results demonstrate that CPR effectively identifies relevant product terms, even for new products lacking prior reviews, and provides sentiment-aligned phrase suggestions, saving users time and enhancing reviews quality. Quantitative analysis reveals a 12.3% improvement in BLEU score over baseline methods, further supported by manual evaluation of generated phrases. We conclude by discussing potential extensions and future research directions. 

---
# PC-DeepNet: A GNSS Positioning Error Minimization Framework Using Permutation-Invariant Deep Neural Network 

**Authors**: M. Humayun Kabir, Md. Ali Hasan, Md. Shafiqul Islam, Kyeongjun Ko, Wonjae Shin  

**Link**: [PDF](https://arxiv.org/pdf/2504.13990)  

**Abstract**: Global navigation satellite systems (GNSS) face significant challenges in urban and sub-urban areas due to non-line-of-sight (NLOS) propagation, multipath effects, and low received power levels, resulting in highly non-linear and non-Gaussian measurement error distributions. In light of this, conventional model-based positioning approaches, which rely on Gaussian error approximations, struggle to achieve precise localization under these conditions. To overcome these challenges, we put forth a novel learning-based framework, PC-DeepNet, that employs a permutation-invariant (PI) deep neural network (DNN) to estimate position corrections (PC). This approach is designed to ensure robustness against changes in the number and/or order of visible satellite measurements, a common issue in GNSS systems, while leveraging NLOS and multipath indicators as features to enhance positioning accuracy in challenging urban and sub-urban environments. To validate the performance of the proposed framework, we compare the positioning error with state-of-the-art model-based and learning-based positioning methods using two publicly available datasets. The results confirm that proposed PC-DeepNet achieves superior accuracy than existing model-based and learning-based methods while exhibiting lower computational complexity compared to previous learning-based approaches. 

---
# Gradual Binary Search and Dimension Expansion : A general method for activation quantization in LLMs 

**Authors**: Lucas Maisonnave, Cyril Moineau, Olivier Bichler, Fabrice Rastello  

**Link**: [PDF](https://arxiv.org/pdf/2504.13989)  

**Abstract**: Large language models (LLMs) have become pivotal in artificial intelligence, demonstrating strong capabilities in reasoning, understanding, and generating data. However, their deployment on edge devices is hindered by their substantial size, often reaching several billion parameters. Quantization is a widely used method to reduce memory usage and inference time, however LLMs present unique challenges due to the prevalence of outliers in their activations. In this work, we leverage the theoretical advantages of Hadamard matrices over random rotation matrices to push the boundaries of quantization in LLMs. We demonstrate that Hadamard matrices are more effective in reducing outliers, which are a significant obstacle in achieving low-bit quantization. Our method based on a gradual binary search enables 3-bit quantization for weights, activations, and key-value (KV) caches, resulting in a 40\% increase in accuracy on common benchmarks compared to SoTA methods. We extend the use of rotation matrices to support non-power-of-2 embedding dimensions, similar to the Qwen architecture, by employing the Paley algorithm. We theoretically demonstrates the superiority of Hadamard matrices in reducing this http URL achieved 3-bit quantization for weights, activations, and KV cache, significantly enhancing model performance. Our experimental results on multiple models family like Mistral, LLaMA, and Qwen demonstrate the effectiveness of our approach, outperforming existing methods and enabling practical 3-bit quantization. 

---
# Entropy Rectifying Guidance for Diffusion and Flow Models 

**Authors**: Tariq Berrada Ifriqi, Adriana Romero-Soriano, Michal Drozdzal, Jakob Verbeek, Karteek Alahari  

**Link**: [PDF](https://arxiv.org/pdf/2504.13987)  

**Abstract**: Guidance techniques are commonly used in diffusion and flow models to improve image quality and consistency for conditional generative tasks such as class-conditional and text-to-image generation. In particular, classifier-free guidance (CFG) -- the most widely adopted guidance technique -- contrasts conditional and unconditional predictions to improve the generated images. This results, however, in trade-offs across quality, diversity and consistency, improving some at the expense of others. While recent work has shown that it is possible to disentangle these factors to some extent, such methods come with an overhead of requiring an additional (weaker) model, or require more forward passes per sampling step. In this paper, we propose Entropy Rectifying Guidance (ERG), a simple and effective guidance mechanism based on inference-time changes in the attention mechanism of state-of-the-art diffusion transformer architectures, which allows for simultaneous improvements over image quality, diversity and prompt consistency. ERG is more general than CFG and similar guidance techniques, as it extends to unconditional sampling. ERG results in significant improvements in various generation tasks such as text-to-image, class-conditional and unconditional image generation. We also show that ERG can be seamlessly combined with other recent guidance methods such as CADS and APG, further boosting generation performance. 

---
# On the redundancy of short and heterogeneous sequences of belief revisions 

**Authors**: Paolo Liberatore  

**Link**: [PDF](https://arxiv.org/pdf/2504.13986)  

**Abstract**: Forgetting a specific belief revision episode may not erase information because the other revisions may provide the same information or allow to deduce it. Whether it does was proved coNP-hard for sequence of two arbitrary lexicographic revision or arbitrarily long lexicographic Horn revision. A polynomial algorithm is presented for the case of two Horn revision. Heterogeneous sequences of revisions were proved to belong in Delta2. Their previously proved coNP-hardness is enhanced by a proof of NP-hardness. 

---
# One Jump Is All You Need: Short-Cutting Transformers for Early Exit Prediction with One Jump to Fit All Exit Levels 

**Authors**: Amrit Diggavi Seshadri  

**Link**: [PDF](https://arxiv.org/pdf/2504.13984)  

**Abstract**: To reduce the time and computational costs of inference of large language models, there has been interest in parameter-efficient low-rank early-exit casting of transformer hidden-representations to final-representations. Such low-rank short-cutting has been shown to outperform identity shortcuts at early model stages while offering parameter-efficiency in shortcut jumps. However, current low-rank methods maintain a separate early-exit shortcut jump to final-representations for each transformer intermediate block-level during inference. In this work, we propose selection of a single One-Jump-Fits-All (OJFA) low-rank shortcut that offers over a 30x reduction in shortcut parameter costs during inference. We show that despite this extreme reduction, our OJFA choice largely matches the performance of maintaining multiple shortcut jumps during inference and offers stable precision from all transformer block-levels for GPT2-XL, Phi3-Mini and Llama2-7B transformer models. 

---
# CacheFormer: High Attention-Based Segment Caching 

**Authors**: Sushant Singh, Ausif Mahmood  

**Link**: [PDF](https://arxiv.org/pdf/2504.13981)  

**Abstract**: Efficiently handling long contexts in transformer-based language models with low perplexity is an active area of research. Numerous recent approaches like Linformer, Longformer, Performer, and Structured state space models (SSMs)., have not fully resolved this problem. All these models strive to reduce the quadratic time complexity of the attention mechanism while minimizing the loss in quality due to the effective compression of the long context. Inspired by the cache and virtual memory principle in computers, where in case of a cache miss, not only the needed data is retrieved from the memory, but the adjacent data is also obtained, we apply this concept to handling long contexts by dividing it into small segments. In our design, we retrieve the nearby segments in an uncompressed form when high segment-level attention occurs at the compressed level. Our en-hancements for handling long context include aggregating four attention mechanisms consisting of short sliding window attention, long compressed segmented attention, dynamically retrieving top k high attention uncompressed segments, and overlapping segments in long segment attention to avoid segment fragmentation. These enhancements result in an architecture that outperforms ex-isting SOTA architectures with an average perplexity improvement of 8.5% over similar model sizes. 

---
# Framework, Standards, Applications and Best practices of Responsible AI : A Comprehensive Survey 

**Authors**: Thippa Reddy Gadekallu, Kapal Dev, Sunder Ali Khowaja, Weizheng Wang, Hailin Feng, Kai Fang, Sharnil Pandya, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.13979)  

**Abstract**: Responsible Artificial Intelligence (RAI) is a combination of ethics associated with the usage of artificial intelligence aligned with the common and standard frameworks. This survey paper extensively discusses the global and national standards, applications of RAI, current technology and ongoing projects using RAI, and possible challenges in implementing and designing RAI in the industries and projects based on AI. Currently, ethical standards and implementation of RAI are decoupled which caters each industry to follow their own standards to use AI ethically. Many global firms and government organizations are taking necessary initiatives to design a common and standard framework. Social pressure and unethical way of using AI forces the RAI design rather than implementation. 

---
# Gas Station of the Future: A Perspective on AI/ML and IoT in Retail Downstream 

**Authors**: Wrick Talukdar  

**Link**: [PDF](https://arxiv.org/pdf/2504.13976)  

**Abstract**: The gas station of the future is poised to transform from a simple fuel dispensing center into an intelligent retail hub, driven by advancements in Artificial Intelligence (AI), Machine Learning (ML), and the Internet of Things (IoT). This paper explores how technology is reshaping the retail downstream sector while briefly addressing the upstream and midstream segments. By leveraging AI/ML for predictive analytics, dynamic pricing, personalized customer engagement, and IoT for real-time monitoring and automation, the future gas station will redefine the fuel retail experience. Additionally, this paper incorporates statistics, AI/ML core technical concepts, mathematical formulations, case studies, and a proposed framework for a fully autonomous gas station. 

---
# Multiscale Tensor Summation Factorization as a New Neural Network Layer (MTS Layer) for Multidimensional Data Processing 

**Authors**: Mehmet Yamaç, Muhammad Numan Yousaf, Serkan Kiranyaz, Moncef Gabbouj  

**Link**: [PDF](https://arxiv.org/pdf/2504.13975)  

**Abstract**: Multilayer perceptrons (MLP), or fully connected artificial neural networks, are known for performing vector-matrix multiplications using learnable weight matrices; however, their practical application in many machine learning tasks, especially in computer vision, can be limited due to the high dimensionality of input-output pairs at each layer. To improve efficiency, convolutional operators have been utilized to facilitate weight sharing and local connections, yet they are constrained by limited receptive fields. In this paper, we introduce Multiscale Tensor Summation (MTS) Factorization, a novel neural network operator that implements tensor summation at multiple scales, where each tensor to be summed is obtained through Tucker-decomposition-like mode products. Unlike other tensor decomposition methods in the literature, MTS is not introduced as a network compression tool; instead, as a new backbone neural layer. MTS not only reduces the number of parameters required while enhancing the efficiency of weight optimization compared to traditional dense layers (i.e., unfactorized weight matrices in MLP layers), but it also demonstrates clear advantages over convolutional layers. The proof-of-concept experimental comparison of the proposed MTS networks with MLPs and Convolutional Neural Networks (CNNs) demonstrates their effectiveness across various tasks, such as classification, compression, and signal restoration. Additionally, when integrated with modern non-linear units such as the multi-head gate (MHG), also introduced in this study, the corresponding neural network, MTSNet, demonstrates a more favorable complexity-performance tradeoff compared to state-of-the-art transformers in various computer vision applications. The software implementation of the MTS layer and the corresponding MTS-based networks, MTSNets, is shared at this https URL. 

---
# Enhancing Stroke Diagnosis in the Brain Using a Weighted Deep Learning Approach 

**Authors**: Yao Zhiwan, Reza Zarrab, Jean Dubois  

**Link**: [PDF](https://arxiv.org/pdf/2504.13974)  

**Abstract**: A brain stroke occurs when blood flow to a part of the brain is disrupted, leading to cell death. Traditional stroke diagnosis methods, such as CT scans and MRIs, are costly and time-consuming. This study proposes a weighted voting ensemble (WVE) machine learning model that combines predictions from classifiers like random forest, Deep Learning, and histogram-based gradient boosting to predict strokes more effectively. The model achieved 94.91% accuracy on a private dataset, enabling early risk assessment and prevention. Future research could explore optimization techniques to further enhance accuracy. 

---
# Governance Challenges in Reinforcement Learning from Human Feedback: Evaluator Rationality and Reinforcement Stability 

**Authors**: Dana Alsagheer, Abdulrahman Kamal, Mohammad Kamal, Weidong Shi  

**Link**: [PDF](https://arxiv.org/pdf/2504.13972)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is central in aligning large language models (LLMs) with human values and expectations. However, the process remains susceptible to governance challenges, including evaluator bias, inconsistency, and the unreliability of feedback. This study examines how the cognitive capacity of evaluators, specifically their level of rationality, affects the stability of reinforcement signals. A controlled experiment comparing high-rationality and low-rationality participants reveals that evaluators with higher rationality scores produce significantly more consistent and expert-aligned feedback. In contrast, lower-rationality participants demonstrate considerable variability in their reinforcement decisions ($p < 0.01$). To address these challenges and improve RLHF governance, we recommend implementing evaluator pre-screening, systematic auditing of feedback consistency, and reliability-weighted reinforcement aggregation. These measures enhance the fairness, transparency, and robustness of AI alignment pipelines. 

---
# The Future of Internet of Things and Multimodal Language Models in 6G Networks: Opportunities and Challenges 

**Authors**: Abdelrahman Soliman  

**Link**: [PDF](https://arxiv.org/pdf/2504.13971)  

**Abstract**: Based on recent trends in artificial intelligence and IoT research. The cooperative potential of integrating the Internet of Things (IoT) and Multimodal Language Models (MLLMs) is presented in this survey paper for future 6G systems. It focuses on the applications of this integration in different fields, such as healthcare, agriculture, and smart cities, and investigates the four pillars of IoT integration, such as sensors, communication, processing, and security. The paper provides a comprehensive description of IoT and MLLM technologies and applications, addresses the role of multimodality in each pillar, and concludes with an overview of the most significant challenges and directions for future research. The general survey is a roadmap for researchers interested in tracing the application areas of MLLMs and IoT, highlighting the potential and challenges in this rapidly growing field. The survey recognizes the need to deal with data availability, computational expense, privacy, and real-time processing to harness the complete potential of IoT, MLLM, and 6G technology 

---
# Tinker Tales: Interactive Storytelling Framework for Early Childhood Narrative Development and AI Literacy 

**Authors**: Nayoung Choi, Peace Cyebukayire, Jinho D. Choi  

**Link**: [PDF](https://arxiv.org/pdf/2504.13969)  

**Abstract**: This paper presents Tinker Tales, an interactive storytelling framework in the format of a board game, designed to support both narrative development and AI literacy in early childhood. The framework integrates tangible and speech-based interactions with AI through NFC chip-attached pawns and tokens, along with a speaker and microphone. Children select and define key story elements-such as characters, places, items, and emotions-using the pawns and tokens, providing further details to the AI and receiving proper assistance, similar to how adults prompt AI for specific tasks (e.g., writing). For evaluation, several game sessions were simulated with a child AI agent, and the quality and safety of the generated stories were assessed from various perspectives. This work highlights the potential of combining physical and digital elements in AI literacy, offering a safe and engaging way for children to learn how to effectively collaborate with AI. 

---
# CONTINA: Confidence Interval for Traffic Demand Prediction with Coverage Guarantee 

**Authors**: Chao Yang, Xiannan Huang, Shuhan Qiu, Yan Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.13961)  

**Abstract**: Accurate short-term traffic demand prediction is critical for the operation of traffic systems. Besides point estimation, the confidence interval of the prediction is also of great importance. Many models for traffic operations, such as shared bike rebalancing and taxi dispatching, take into account the uncertainty of future demand and require confidence intervals as the input. However, existing methods for confidence interval modeling rely on strict assumptions, such as unchanging traffic patterns and correct model specifications, to guarantee enough coverage. Therefore, the confidence intervals provided could be invalid, especially in a changing traffic environment. To fill this gap, we propose an efficient method, CONTINA (Conformal Traffic Intervals with Adaptation) to provide interval predictions that can adapt to external changes. By collecting the errors of interval during deployment, the method can adjust the interval in the next step by widening it if the errors are too large or shortening it otherwise. Furthermore, we theoretically prove that the coverage of the confidence intervals provided by our method converges to the target coverage level. Experiments across four real-world datasets and prediction models demonstrate that the proposed method can provide valid confidence intervals with shorter lengths. Our method can help traffic management personnel develop a more reasonable and robust operation plan in practice. And we release the code, model and dataset in \href{ this https URL}{ Github}. 

---
# AI Safety Should Prioritize the Future of Work 

**Authors**: Sanchaita Hazra, Bodhisattwa Prasad Majumder, Tuhin Chakrabarty  

**Link**: [PDF](https://arxiv.org/pdf/2504.13959)  

**Abstract**: Current efforts in AI safety prioritize filtering harmful content, preventing manipulation of human behavior, and eliminating existential risks in cybersecurity or biosecurity. While pressing, this narrow focus overlooks critical human-centric considerations that shape the long-term trajectory of a society. In this position paper, we identify the risks of overlooking the impact of AI on the future of work and recommend comprehensive transition support towards the evolution of meaningful labor with human agency. Through the lens of economic theories, we highlight the intertemporal impacts of AI on human livelihood and the structural changes in labor markets that exacerbate income inequality. Additionally, the closed-source approach of major stakeholders in AI development resembles rent-seeking behavior through exploiting resources, breeding mediocrity in creative labor, and monopolizing innovation. To address this, we argue in favor of a robust international copyright anatomy supported by implementing collective licensing that ensures fair compensation mechanisms for using data to train AI models. We strongly recommend a pro-worker framework of global AI governance to enhance shared prosperity and economic justice while reducing technical debt. 

---
# ToolRL: Reward is All Tool Learning Needs 

**Authors**: Cheng Qian, Emre Can Acikgoz, Qi He, Hongru Wang, Xiusi Chen, Dilek Hakkani-Tür, Gokhan Tur, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2504.13958)  

**Abstract**: Current Large Language Models (LLMs) often undergo supervised fine-tuning (SFT) to acquire tool use capabilities. However, SFT struggles to generalize to unfamiliar or complex tool use scenarios. Recent advancements in reinforcement learning (RL), particularly with R1-like models, have demonstrated promising reasoning and generalization abilities. Yet, reward design for tool use presents unique challenges: multiple tools may be invoked with diverse parameters, and coarse-grained reward signals, such as answer matching, fail to offer the finegrained feedback required for effective learning. In this work, we present the first comprehensive study on reward design for tool selection and application tasks within the RL paradigm. We systematically explore a wide range of reward strategies, analyzing their types, scales, granularity, and temporal dynamics. Building on these insights, we propose a principled reward design tailored for tool use tasks and apply it to train LLMs using Group Relative Policy Optimization (GRPO). Empirical evaluations across diverse benchmarks demonstrate that our approach yields robust, scalable, and stable training, achieving a 17% improvement over base models and a 15% gain over SFT models. These results highlight the critical role of thoughtful reward design in enhancing the tool use capabilities and generalization performance of LLMs. All the codes are released to facilitate future research. 

---
# Naming is framing: How cybersecurity's language problems are repeating in AI governance 

**Authors**: Liane Potter  

**Link**: [PDF](https://arxiv.org/pdf/2504.13957)  

**Abstract**: Language is not neutral; it frames understanding, structures power, and shapes governance. This paper argues that misnomers like cybersecurity and artificial intelligence (AI) are more than semantic quirks; they carry significant governance risks by obscuring human agency, inflating expectations, and distorting accountability. Drawing on lessons from cybersecurity's linguistic pitfalls, such as the 'weakest link' narrative, this paper highlights how AI discourse is falling into similar traps with metaphors like 'alignment,' 'black box,' and 'hallucination.' These terms embed adversarial, mystifying, or overly technical assumptions into governance structures. In response, the paper advocates for a language-first approach to AI governance: one that interrogates dominant metaphors, foregrounds human roles, and co-develops a lexicon that is precise, inclusive, and reflexive. This paper contends that linguistic reform is not peripheral to governance but central to the construction of transparent, equitable, and anticipatory regulatory frameworks. 

---
# Thousand Voices of Trauma: A Large-Scale Synthetic Dataset for Modeling Prolonged Exposure Therapy Conversations 

**Authors**: Suhas BN, Dominik Mattioli, Saeed Abdullah, Rosa I. Arriaga, Chris W. Wiese, Andrew M. Sherrill  

**Link**: [PDF](https://arxiv.org/pdf/2504.13955)  

**Abstract**: The advancement of AI systems for mental health support is hindered by limited access to therapeutic conversation data, particularly for trauma treatment. We present Thousand Voices of Trauma, a synthetic benchmark dataset of 3,000 therapy conversations based on Prolonged Exposure therapy protocols for Post-traumatic Stress Disorder (PTSD). The dataset comprises 500 unique cases, each explored through six conversational perspectives that mirror the progression of therapy from initial anxiety to peak distress to emotional processing. We incorporated diverse demographic profiles (ages 18-80, M=49.3, 49.4% male, 44.4% female, 6.2% non-binary), 20 trauma types, and 10 trauma-related behaviors using deterministic and probabilistic generation methods. Analysis reveals realistic distributions of trauma types (witnessing violence 10.6%, bullying 10.2%) and symptoms (nightmares 23.4%, substance abuse 20.8%). Clinical experts validated the dataset's therapeutic fidelity, highlighting its emotional depth while suggesting refinements for greater authenticity. We also developed an emotional trajectory benchmark with standardized metrics for evaluating model responses. This privacy-preserving dataset addresses critical gaps in trauma-focused mental health data, offering a valuable resource for advancing both patient-facing applications and clinician training tools. 

---
# Generative System Dynamics in Recurrent Neural Networks 

**Authors**: Michele Casoni, Tommaso Guidi, Alessandro Betti, Stefano Melacci, Marco Gori  

**Link**: [PDF](https://arxiv.org/pdf/2504.13951)  

**Abstract**: In this study, we investigate the continuous time dynamics of Recurrent Neural Networks (RNNs), focusing on systems with nonlinear activation functions. The objective of this work is to identify conditions under which RNNs exhibit perpetual oscillatory behavior, without converging to static fixed points. We establish that skew-symmetric weight matrices are fundamental to enable stable limit cycles in both linear and nonlinear configurations. We further demonstrate that hyperbolic tangent-like activation functions (odd, bounded, and continuous) preserve these oscillatory dynamics by ensuring motion invariants in state space. Numerical simulations showcase how nonlinear activation functions not only maintain limit cycles, but also enhance the numerical stability of the system integration process, mitigating those instabilities that are commonly associated with the forward Euler method. The experimental results of this analysis highlight practical considerations for designing neural architectures capable of capturing complex temporal dependencies, i.e., strategies for enhancing memorization skills in recurrent models. 

---
# Open-Medical-R1: How to Choose Data for RLVR Training at Medicine Domain 

**Authors**: Zhongxi Qiu, Zhang Zhang, Yan Hu, Heng Li, Jiang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.13950)  

**Abstract**: This paper explores optimal data selection strategies for Reinforcement Learning with Verified Rewards (RLVR) training in the medical domain. While RLVR has shown exceptional potential for enhancing reasoning capabilities in large language models, most prior implementations have focused on mathematics and logical puzzles, with limited exploration of domain-specific applications like medicine. We investigate four distinct data sampling strategies from MedQA-USMLE: random sampling (baseline), and filtering using Phi-4, Gemma-3-27b-it, and Gemma-3-12b-it models. Using Gemma-3-12b-it as our base model and implementing Group Relative Policy Optimization (GRPO), we evaluate performance across multiple benchmarks including MMLU, GSM8K, MMLU-Pro, and CMMLU. Our findings demonstrate that models trained on filtered data generally outperform those trained on randomly selected samples. Notably, training on self-filtered samples (using Gemma-3-12b-it for filtering) achieved superior performance in medical domains but showed reduced robustness across different benchmarks, while filtering with larger models from the same series yielded better overall robustness. These results provide valuable insights into effective data organization strategies for RLVR in specialized domains and highlight the importance of thoughtful data selection in achieving optimal performance. You can access our repository (this https URL) to get the codes. 

---
# On Revealing the Hidden Problem Structure in Real-World and Theoretical Problems Using Walsh Coefficient Influence 

**Authors**: M. W. Przewozniczek, F. Chicano, R. Tinós, J. Nalepa, B. Ruszczak, A. M. Wijata  

**Link**: [PDF](https://arxiv.org/pdf/2504.13949)  

**Abstract**: Gray-box optimization employs Walsh decomposition to obtain non-linear variable dependencies and utilize them to propose masks of variables that have a joint non-linear influence on fitness value. These masks significantly improve the effectiveness of variation operators. In some problems, all variables are non-linearly dependent, making the aforementioned masks useless. We analyze the features of the real-world instances of such problems and show that many of their dependencies may have noise-like origins. Such noise-caused dependencies are irrelevant to the optimization process and can be ignored. To identify them, we propose extending the use of Walsh decomposition by measuring variable dependency strength that allows the construction of the weighted dynamic Variable Interaction Graph (wdVIG). wdVIGs adjust the dependency strength to mixed individuals. They allow the filtering of irrelevant dependencies and re-enable using dependency-based masks by variation operators. We verify the wdVIG potential on a large benchmark suite. For problems with noise, the wdVIG masks can improve the optimizer's effectiveness. If all dependencies are relevant for the optimization, i.e., the problem is not noised, the influence of wdVIG masks is similar to that of state-of-the-art structures of this kind. 

---
# Using customized GPT to develop prompting proficiency in architectural AI-generated images 

**Authors**: Juan David Salazar Rodriguez, Sam Conrad Joyce, Julfendi Julfendi  

**Link**: [PDF](https://arxiv.org/pdf/2504.13948)  

**Abstract**: This research investigates the use of customized GPT models to enhance prompting proficiency among architecture students when generating AI-driven images. Prompt engineering is increasingly essential in architectural education due to the widespread adoption of generative AI tools. This study utilized a mixed-methods experimental design involving architecture students divided into three distinct groups: a control group receiving no structured support, a second group provided with structured prompting guides, and a third group supported by both structured guides and interactive AI personas. Students engaged in reverse engineering tasks, first guessing provided image prompts and then generating their own prompts, aiming to boost critical thinking and prompting skills. Variables examined included time spent prompting, word count, prompt similarity, and concreteness. Quantitative analysis involved correlation assessments between these variables and a one-way ANOVA to evaluate differences across groups. While several correlations showed meaningful relationships, not all were statistically significant. ANOVA results indicated statistically significant improvements in word count, similarity, and concreteness, especially in the group supported by AI personas and structured prompting guides. Qualitative feedback complemented these findings, revealing enhanced confidence and critical thinking skills in students. These results suggest tailored GPT interactions substantially improve students' ability to communicate architectural concepts clearly and effectively. 

---
# From job titles to jawlines: Using context voids to study generative AI systems 

**Authors**: Shahan Ali Memon, Soham De, Sungha Kang, Riyan Mujtaba, Bedoor AlShebli, Katie Davis, Jaime Snyder, Jevin D. West  

**Link**: [PDF](https://arxiv.org/pdf/2504.13947)  

**Abstract**: In this paper, we introduce a speculative design methodology for studying the behavior of generative AI systems, framing design as a mode of inquiry. We propose bridging seemingly unrelated domains to generate intentional context voids, using these tasks as probes to elicit AI model behavior. We demonstrate this through a case study: probing the ChatGPT system (GPT-4 and DALL-E) to generate headshots from professional Curricula Vitae (CVs). In contrast to traditional ways, our approach assesses system behavior under conditions of radical uncertainty -- when forced to invent entire swaths of missing context -- revealing subtle stereotypes and value-laden assumptions. We qualitatively analyze how the system interprets identity and competence markers from CVs, translating them into visual portraits despite the missing context (i.e. physical descriptors). We show that within this context void, the AI system generates biased representations, potentially relying on stereotypical associations or blatant hallucinations. 

---
# Evaluating Menu OCR and Translation: A Benchmark for Aligning Human and Automated Evaluations in Large Vision-Language Models 

**Authors**: Zhanglin Wu, Tengfei Song, Ning Xie, Weidong Zhang, Mengli Zhu, Shuang Wu, Shiliang Sun, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.13945)  

**Abstract**: The rapid advancement of large vision-language models (LVLMs) has significantly propelled applications in document understanding, particularly in optical character recognition (OCR) and multilingual translation. However, current evaluations of LVLMs, like the widely used OCRBench, mainly focus on verifying the correctness of their short-text responses and long-text responses with simple layout, while the evaluation of their ability to understand long texts with complex layout design is highly significant but largely overlooked. In this paper, we propose Menu OCR and Translation Benchmark (MOTBench), a specialized evaluation framework emphasizing the pivotal role of menu translation in cross-cultural communication. MOTBench requires LVLMs to accurately recognize and translate each dish, along with its price and unit items on a menu, providing a comprehensive assessment of their visual understanding and language processing capabilities. Our benchmark is comprised of a collection of Chinese and English menus, characterized by intricate layouts, a variety of fonts, and culturally specific elements across different languages, along with precise human annotations. Experiments show that our automatic evaluation results are highly consistent with professional human evaluation. We evaluate a range of publicly available state-of-the-art LVLMs, and through analyzing their output to identify the strengths and weaknesses in their performance, offering valuable insights to guide future advancements in LVLM development. MOTBench is available at this https URL. 

---
# Mixer Metaphors: audio interfaces for non-musical applications 

**Authors**: Tace McNamara, Jon McCormack, Maria Teresa Llano  

**Link**: [PDF](https://arxiv.org/pdf/2504.13944)  

**Abstract**: The NIME conference traditionally focuses on interfaces for music and musical expression. In this paper we reverse this tradition to ask, can interfaces developed for music be successfully appropriated to non-musical applications? To help answer this question we designed and developed a new device, which uses interface metaphors borrowed from analogue synthesisers and audio mixing to physically control the intangible aspects of a Large Language Model. We compared two versions of the device, with and without the audio-inspired augmentations, with a group of artists who used each version over a one week period. Our results show that the use of audio-like controls afforded more immediate, direct and embodied control over the LLM, allowing users to creatively experiment and play with the device over its non-mixer counterpart. Our project demonstrates how cross-sensory metaphors can support creative thinking and embodied practice when designing new technological interfaces. 

---
# Intelligence of Things: A Spatial Context-Aware Control System for Smart Devices 

**Authors**: Sukanth Kalivarathan, Muhmmad Abrar Raja Mohamed, Aswathy Ravikumar, S Harini  

**Link**: [PDF](https://arxiv.org/pdf/2504.13942)  

**Abstract**: This paper introduces Intelligence of Things (INOT), a novel spatial context-aware control system that enhances smart home automation through intuitive spatial reasoning. Current smart home systems largely rely on device-specific identifiers, limiting user interaction to explicit naming conventions rather than natural spatial references. INOT addresses this limitation through a modular architecture that integrates Vision Language Models with IoT control systems to enable natural language commands with spatial context (e.g., "turn on the light near the window"). The system comprises key components including an Onboarding Inference Engine, Zero-Shot Device Detection, Spatial Topology Inference, and Intent-Based Command Synthesis. A comprehensive user study with 15 participants demonstrated INOT's significant advantages over conventional systems like Google Home Assistant, with users reporting reduced cognitive workload (NASA-TLX scores decreased by an average of 13.17 points), higher ease-of-use ratings, and stronger preference (14 out of 15 participants). By eliminating the need to memorize device identifiers and enabling context-aware spatial commands, INOT represents a significant advancement in creating more intuitive and accessible smart home control systems. 

---
# NEMOTRON-CROSSTHINK: Scaling Self-Learning beyond Math Reasoning 

**Authors**: Syeda Nahida Akter, Shrimai Prabhumoye, Matvei Novikov, Seungju Han, Ying Lin, Evelina Bakhturi, Eric Nyberg, Yejin Choi, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro  

**Link**: [PDF](https://arxiv.org/pdf/2504.13941)  

**Abstract**: Large Language Models (LLMs) have shown strong reasoning capabilities, particularly when enhanced through Reinforcement Learning (RL). While prior work has successfully applied RL to mathematical reasoning -- where rules and correctness are well-defined -- generalizing these methods to broader reasoning domains remains challenging due to limited data, the lack of verifiable reward structures, and diverse task requirements. In this work, we propose NEMOTRON-CROSSTHINK, a framework that systematically incorporates multi-domain corpora, including both synthetic and real-world question-answer pairs, into RL training to improve generalization across diverse reasoning tasks. NEMOTRON-CROSSTHINK addresses key challenges by (1) incorporating data from varied sources spanning STEM, humanities, social sciences, etc.; (2) applying structured templates (e.g., multiple-choice and open-ended) to control answer-space complexity; (3) filtering for verifiable answers; and (4) optimizing data blending strategies that utilizes data from multiple sources effectively. Our approach enables scalable and verifiable reward modeling beyond mathematics and demonstrates improved accuracies on both math (MATH-500: +30.1%, AMC23:+27.5%) and non-math reasoning benchmarks (MMLU-PRO: +12.8%, GPQA-DIAMOND: +11.3%, AGIEVAL: +15.1%, SUPERGPQA: +3.8%). Moreover, NEMOTRON-CROSSTHINK exhibits significantly improved response efficiency -- using 28% fewer tokens for correct answers -- highlighting more focused and effective reasoning. Through NEMOTRON-CROSSTHINK, we demonstrate that integrating multi-domain, multi-format data in RL leads to more accurate, efficient, and generalizable LLMs. 

---
# Hashigo: A Next Generation Sketch Interactive System for Japanese Kanji 

**Authors**: Paul Taele, Tracy Hammond  

**Link**: [PDF](https://arxiv.org/pdf/2504.13940)  

**Abstract**: Language students can increase their effectiveness in learning written Japanese by mastering the visual structure and written technique of Japanese kanji. Yet, existing kanji handwriting recognition systems do not assess the written technique sufficiently enough to discourage students from developing bad learning habits. In this paper, we describe our work on Hashigo, a kanji sketch interactive system which achieves human instructor-level critique and feedback on both the visual structure and written technique of students' sketched kanji. This type of automated critique and feedback allows students to target and correct specific deficiencies in their sketches that, if left untreated, are detrimental to effective long-term kanji learning. 

---
# LLM-Driven NPCs: Cross-Platform Dialogue System for Games and Social Platforms 

**Authors**: Li Song  

**Link**: [PDF](https://arxiv.org/pdf/2504.13928)  

**Abstract**: NPCs in traditional games are often limited by static dialogue trees and a single platform for interaction. To overcome these constraints, this study presents a prototype system that enables large language model (LLM)-powered NPCs to communicate with players both in the game en vironment (Unity) and on a social platform (Discord). Dialogue logs are stored in a cloud database (LeanCloud), allowing the system to synchronize memory between platforms and keep conversa tions coherent. Our initial experiments show that cross-platform interaction is technically feasible and suggest a solid foundation for future developments such as emotional modeling and persistent memory support. 

---
# A Multi-Layered Research Framework for Human-Centered AI: Defining the Path to Explainability and Trust 

**Authors**: Chameera De Silva, Thilina Halloluwa, Dhaval Vyas  

**Link**: [PDF](https://arxiv.org/pdf/2504.13926)  

**Abstract**: The integration of Artificial Intelligence (AI) into high-stakes domains such as healthcare, finance, and autonomous systems is often constrained by concerns over transparency, interpretability, and trust. While Human-Centered AI (HCAI) emphasizes alignment with human values, Explainable AI (XAI) enhances transparency by making AI decisions more understandable. However, the lack of a unified approach limits AI's effectiveness in critical decision-making scenarios. This paper presents a novel three-layered framework that bridges HCAI and XAI to establish a structured explainability paradigm. The framework comprises (1) a foundational AI model with built-in explainability mechanisms, (2) a human-centered explanation layer that tailors explanations based on cognitive load and user expertise, and (3) a dynamic feedback loop that refines explanations through real-time user interaction. The framework is evaluated across healthcare, finance, and software development, demonstrating its potential to enhance decision-making, regulatory compliance, and public trust. Our findings advance Human-Centered Explainable AI (HCXAI), fostering AI systems that are transparent, adaptable, and ethically aligned. 

---
# Modeling the quantum-like dynamics of human reliability ratings in Human-AI interactions by interaction dependent Hamiltonians 

**Authors**: Johan van der Meer, Pamela Hoyte, Luisa Roeder, Peter Bruza  

**Link**: [PDF](https://arxiv.org/pdf/2504.13918)  

**Abstract**: As our information environments become ever more powered by artificial intelligence (AI), the phenomenon of trust in a human's interactions with this intelligence is becoming increasingly pertinent. For example, in the not too distant future, there will be teams of humans and intelligent robots involved in dealing with the repercussions of high-risk disaster situations such as hurricanes, earthquakes, or nuclear accidents. Even in such conditions of high uncertainty, humans and intelligent machines will need to engage in shared decision making, and trust is fundamental to the effectiveness of these interactions. A key challenge in modeling the dynamics of this trust is to provide a means to incorporate sensitivity to fluctuations in human trust judgments. In this article, we explore the ability of Quantum Random Walk models to model the dynamics of trust in human-AI interactions, and to integrate a sensitivity to fluctuations in participant trust judgments based on the nature of the interaction with the AI. We found that using empirical parameters to inform the use of different Hamiltonians can provide a promising means to model the evolution of trust in Human-AI interactions. 

---
# AI-Assisted Conversational Interviewing: Effects on Data Quality and User Experience 

**Authors**: Soubhik Barari, Jarret Angbazo, Natalie Wang, Leah M. Christian, Elizabeth Dean, Zoe Slowinski, Brandon Sepulvado  

**Link**: [PDF](https://arxiv.org/pdf/2504.13908)  

**Abstract**: Standardized surveys scale efficiently but sacrifice depth, while conversational interviews improve response quality at the cost of scalability and consistency. This study bridges the gap between these methods by introducing a framework for AI-assisted conversational interviewing. To evaluate this framework, we conducted a web survey experiment where 1,800 participants were randomly assigned to text-based conversational AI agents, or "textbots", to dynamically probe respondents for elaboration and interactively code open-ended responses. We assessed textbot performance in terms of coding accuracy, response quality, and respondent experience. Our findings reveal that textbots perform moderately well in live coding even without survey-specific fine-tuning, despite slightly inflated false positive errors due to respondent acquiescence bias. Open-ended responses were more detailed and informative, but this came at a slight cost to respondent experience. Our findings highlight the feasibility of using AI methods to enhance open-ended data collection in web surveys. 

---
# Generative Framework for Personalized Persuasion: Inferring Causal, Counterfactual, and Latent Knowledge 

**Authors**: Donghuo Zeng, Roberto Legaspi, Yuewen Sun, Xinshuai Dong, Kazushi Ikeda, Peter Spirtes, Kun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.13904)  

**Abstract**: We hypothesize that optimal system responses emerge from adaptive strategies grounded in causal and counterfactual knowledge. Counterfactual inference allows us to create hypothetical scenarios to examine the effects of alternative system responses. We enhance this process through causal discovery, which identifies the strategies informed by the underlying causal structure that govern system behaviors. Moreover, we consider the psychological constructs and unobservable noises that might be influencing user-system interactions as latent factors. We show that these factors can be effectively estimated. We employ causal discovery to identify strategy-level causal relationships among user and system utterances, guiding the generation of personalized counterfactual dialogues. We model the user utterance strategies as causal factors, enabling system strategies to be treated as counterfactual actions. Furthermore, we optimize policies for selecting system responses based on counterfactual data. Our results using a real-world dataset on social good demonstrate significant improvements in persuasive system outcomes, with increased cumulative rewards validating the efficacy of causal discovery in guiding personalized counterfactual inference and optimizing dialogue policies for a persuasive dialogue system. 

---
# Supporting Students' Reading and Cognition with AI 

**Authors**: Yue Fu, Alexis Hiniker  

**Link**: [PDF](https://arxiv.org/pdf/2504.13900)  

**Abstract**: With the rapid adoption of AI tools in learning contexts, it is vital to understand how these systems shape users' reading processes and cognitive engagement. We collected and analyzed text from 124 sessions with AI tools, in which students used these tools to support them as they read assigned readings for an undergraduate course. We categorized participants' prompts to AI according to Bloom's Taxonomy of educational objectives -- Remembering, Understanding, Applying, Analyzing, Evaluating. Our results show that ``Analyzing'' and ``Evaluating'' are more prevalent in users' second and third prompts within a single usage session, suggesting a shift toward higher-order thinking. However, in reviewing users' engagement with AI tools over several weeks, we found that users converge toward passive reading engagement over time. Based on these results, we propose design implications for future AI reading-support systems, including structured scaffolds for lower-level cognitive tasks (e.g., recalling terms) and proactive prompts that encourage higher-order thinking (e.g., analyzing, applying, evaluating). Additionally, we advocate for adaptive, human-in-the-loop features that allow students and instructors to tailor their reading experiences with AI, balancing efficiency with enriched cognitive engagement. Our paper expands the dialogue on integrating AI into academic reading, highlighting both its potential benefits and challenges. 

---
# Predicting Satisfaction of Counterfactual Explanations from Human Ratings of Explanatory Qualities 

**Authors**: Marharyta Domnich, Rasmus Moorits Veski, Julius Välja, Kadi Tulver, Raul Vicente  

**Link**: [PDF](https://arxiv.org/pdf/2504.13899)  

**Abstract**: Counterfactual explanations are a widely used approach in Explainable AI, offering actionable insights into decision-making by illustrating how small changes to input data can lead to different outcomes. Despite their importance, evaluating the quality of counterfactual explanations remains an open problem. Traditional quantitative metrics, such as sparsity or proximity, fail to fully account for human preferences in explanations, while user studies are insightful but not scalable. Moreover, relying only on a single overall satisfaction rating does not lead to a nuanced understanding of why certain explanations are effective or not. To address this, we analyze a dataset of counterfactual explanations that were evaluated by 206 human participants, who rated not only overall satisfaction but also seven explanatory criteria: feasibility, coherence, complexity, understandability, completeness, fairness, and trust. Modeling overall satisfaction as a function of these criteria, we find that feasibility (the actionability of suggested changes) and trust (the belief that the changes would lead to the desired outcome) consistently stand out as the strongest predictors of user satisfaction, though completeness also emerges as a meaningful contributor. Crucially, even excluding feasibility and trust, other metrics explain 58% of the variance, highlighting the importance of additional explanatory qualities. Complexity appears independent, suggesting more detailed explanations do not necessarily reduce satisfaction. Strong metric correlations imply a latent structure in how users judge quality, and demographic background significantly shapes ranking patterns. These insights inform the design of counterfactual algorithms that adapt explanatory qualities to user expertise and domain context. 

---
# The Human Robot Social Interaction (HSRI) Dataset: Benchmarking Foundational Models' Social Reasoning 

**Authors**: Dong Won Lee, Yubin Kim, Denison Guvenoz, Sooyeon Jeong, Parker Malachowsky, Louis-Philippe Morency, Cynthia Breazeal, Hae Won Park  

**Link**: [PDF](https://arxiv.org/pdf/2504.13898)  

**Abstract**: Our work aims to advance the social reasoning of embodied artificial intelligence (AI) agents in real-world social interactions. Recently, language models (LMs) and foundational models (FMs) are being utilized as automatic evaluators of human-AI interactions with the goal of eventually being used to improve the policy of the AI agent. To enable further research in this direction, we introduce a large-scale real-world Human Robot Social Interaction (HSRI) Dataset to benchmark the capabilities of LMs and FMs to identify and reason about social interactions, specifically with regard to robot social errors and competencies . Our dataset consists of 400 real-world human social robot interaction videos and over 10K annotations, detailing the robot's social errors, competencies, rationale, and corrective actions, capturing unique aspects of human-AI interaction only present in real-world interactions. To further assess AI models' ability to reason about social interactions, we propose eight new benchmark tasks for evaluating centered around whether AI models can (1) evaluate social interactions via detecting social errors and competencies, (2) identify the explanatory factors associated to errors and competencies, (3) understand the flow of real-world social interactions, and (4) provide reasons and corrective actions for social errors. Human studies and experiments with modern LMs and FMs reveal that current models struggle with these tasks, demonstrating that our dataset and benchmark provides a step forward towards socially intelligent AI. 

---
# Mozualization: Crafting Music and Visual Representation with Multimodal AI 

**Authors**: Wanfang Xu, Lixiang Zhao, Haiwen Song, Xinheng Song, Zhaolin Lu, Yu Liu, Min Chen, Eng Gee Lim, Lingyun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.13891)  

**Abstract**: In this work, we introduce Mozualization, a music generation and editing tool that creates multi-style embedded music by integrating diverse inputs, such as keywords, images, and sound clips (e.g., segments from various pieces of music or even a playful cat's meow). Our work is inspired by the ways people express their emotions -- writing mood-descriptive poems or articles, creating drawings with warm or cool tones, or listening to sad or uplifting music. Building on this concept, we developed a tool that transforms these emotional expressions into a cohesive and expressive song, allowing users to seamlessly incorporate their unique preferences and inspirations. To evaluate the tool and, more importantly, gather insights for its improvement, we conducted a user study involving nine music enthusiasts. The study assessed user experience, engagement, and the impact of interacting with and listening to the generated music. 

---
# Maestoso: An Intelligent Educational Sketching Tool for Learning Music Theory 

**Authors**: Paul Taele, Laura Barreto, Tracy Hammond  

**Link**: [PDF](https://arxiv.org/pdf/2504.13889)  

**Abstract**: Learning music theory not only has practical benefits for musicians to write, perform, understand, and express music better, but also for both non-musicians to improve critical thinking, math analytical skills, and music appreciation. However, current external tools applicable for learning music theory through writing when human instruction is unavailable are either limited in feedback, lacking a written modality, or assuming already strong familiarity of music theory concepts. In this paper, we describe Maestoso, an educational tool for novice learners to learn music theory through sketching practice of quizzed music structures. Maestoso first automatically recognizes students' sketched input of quizzed concepts, then relies on existing sketch and gesture recognition techniques to automatically recognize the input, and finally generates instructor-emulated feedback. From our evaluations, we demonstrate that Maestoso performs reasonably well on recognizing music structure elements and that novice students can comfortably grasp introductory music theory in a single session. 

---
# Kanji Workbook: A Writing-Based Intelligent Tutoring System for Learning Proper Japanese Kanji Writing Technique with Instructor-Emulated Assessment 

**Authors**: Paul Taele, Jung In Koh, Tracy Hammond  

**Link**: [PDF](https://arxiv.org/pdf/2504.13888)  

**Abstract**: Kanji script writing is a skill that is often introduced to novice Japanese foreign language students for achieving Japanese writing mastery, but often poses difficulties to students with primarily English fluency due to their its vast differences with written English. Instructors often introduce various pedagogical methods -- such as visual structure and written techniques -- to assist students in kanji study, but may lack availability providing direct feedback on students' writing outside of class. Current educational applications are also limited due to lacking richer instructor-emulated feedback. We introduce Kanji Workbook, a writing-based intelligent tutoring system for students to receive intelligent assessment that emulates human instructor feedback. Our interface not only leverages students' computing devices for allowing them to learn, practice, and review the writing of prompted characters from their course's kanji script lessons, but also provides a diverse set of writing assessment metrics -- derived from instructor interviews and classroom observation insights -- through intelligent scoring and visual animations. We deployed our interface onto novice- and intermediate-level university courses over an entire academic year, and observed that interface users on average achieved higher course grades than their peers and also reacted positively to our interface's various features. 

---
# Towards a Multimodal Document-grounded Conversational AI System for Education 

**Authors**: Karan Taneja, Anjali Singh, Ashok K. Goel  

**Link**: [PDF](https://arxiv.org/pdf/2504.13884)  

**Abstract**: Multimedia learning using text and images has been shown to improve learning outcomes compared to text-only instruction. But conversational AI systems in education predominantly rely on text-based interactions while multimodal conversations for multimedia learning remain unexplored. Moreover, deploying conversational AI in learning contexts requires grounding in reliable sources and verifiability to create trust. We present MuDoC, a Multimodal Document-grounded Conversational AI system based on GPT-4o, that leverages both text and visuals from documents to generate responses interleaved with text and images. Its interface allows verification of AI generated content through seamless navigation to the source. We compare MuDoC to a text-only system to explore differences in learner engagement, trust in AI system, and their performance on problem-solving tasks. Our findings indicate that both visuals and verifiability of content enhance learner engagement and foster trust; however, no significant impact in performance was observed. We draw upon theories from cognitive and learning sciences to interpret the findings and derive implications, and outline future directions for the development of multimodal conversational AI systems in education. 

---
# New care pathways for supporting transitional care from hospitals to home using AI and personalized digital assistance 

**Authors**: Ionut Anghel, Tudor Cioara, Roberta Bevilacqua, Federico Barbarossa, Terje Grimstad, Riitta Hellman, Arnor Solberg, Lars Thomas Boye, Ovidiu Anchidin, Ancuta Nemes, Camilla Gabrielsen  

**Link**: [PDF](https://arxiv.org/pdf/2504.13877)  

**Abstract**: Transitional care may play a vital role for the sustainability of Europe future healthcare system, offering solutions for relocating patient care from hospital to home therefore addressing the growing demand for medical care as the population is ageing. However, to be effective, it is essential to integrate innovative Information and Communications Technology technologies to ensure that patients with comorbidities experience a smooth and coordinated transition from hospitals or care centers to home, thereby reducing the risk of rehospitalization. In this paper, we present an overview of the integration of Internet of Things, artificial intelligence, and digital assistance technologies with traditional care pathways to address the challenges and needs of healthcare systems in Europe. We identify the current gaps in transitional care and define the technology mapping to enhance the care pathways, aiming to improve patient outcomes, safety, and quality of life avoiding hospital readmissions. Finally, we define the trial setup and evaluation methodology needed to provide clinical evidence that supports the positive impact of technology integration on patient care and discuss the potential effects on the healthcare system. 

---
# Human aversion? Do AI Agents Judge Identity More Harshly Than Performance 

**Authors**: Yuanjun Feng, Vivek Chodhary, Yash Raj Shrestha  

**Link**: [PDF](https://arxiv.org/pdf/2504.13871)  

**Abstract**: This study examines the understudied role of algorithmic evaluation of human judgment in hybrid decision-making systems, a critical gap in management research. While extant literature focuses on human reluctance to follow algorithmic advice, we reverse the perspective by investigating how AI agents based on large language models (LLMs) assess and integrate human input. Our work addresses a pressing managerial constraint: firms barred from deploying LLMs directly due to privacy concerns can still leverage them as mediating tools (for instance, anonymized outputs or decision pipelines) to guide high-stakes choices like pricing or discounts without exposing proprietary data. Through a controlled prediction task, we analyze how an LLM-based AI agent weights human versus algorithmic predictions. We find that the AI system systematically discounts human advice, penalizing human errors more severely than algorithmic errors--a bias exacerbated when the agent's identity (human vs AI) is disclosed and the human is positioned second. These results reveal a disconnect between AI-generated trust metrics and the actual influence of human judgment, challenging assumptions about equitable human-AI collaboration. Our findings offer three key contributions. First, we identify a reverse algorithm aversion phenomenon, where AI agents undervalue human input despite comparable error rates. Second, we demonstrate how disclosure and positional bias interact to amplify this effect, with implications for system design. Third, we provide a framework for indirect LLM deployment that balances predictive power with data privacy. For practitioners, this research emphasize the need to audit AI weighting mechanisms, calibrate trust dynamics, and strategically design decision sequences in human-AI systems. 

---
# Using Generative AI Personas Increases Collective Diversity in Human Ideation 

**Authors**: Yun Wan, Yoram M Kalman  

**Link**: [PDF](https://arxiv.org/pdf/2504.13868)  

**Abstract**: This study challenges the widely-reported tradeoff between generative AI's (GenAI) contribution to creative outcomes and decreased diversity of these outcomes. We modified the design of such a study, by Doshi and Hauser (2024), in which participants wrote short stories either aided or unaided by GenAI plot ideas[1]. In the modified study, plot ideas were generated through ten unique GenAI "personas" with diverse traits (e.g. cultural backgrounds, thinking styles, genre preferences), creating a pool of 300 story plots. While plot ideas from any individual persona showed high similarity (average cosine similarity of 0.92), ideas across different personas exhibited substantial variation (average similarity of 0.20). When human participants wrote stories based on these diverse plot ideas, their collective outputs maintained the same level of diversity as stories written without GenAI assistance, effectively eliminating the diversity reduction observed in [1]. Traditional text analytics further revealed that GenAI-assisted stories featured greater diversity in descriptive and emotional language compared to purely human-generated stories without GenAI assistance. Our findings demonstrate that introducing diversity at the AI input stage through distinct personas can preserve and potentially enhance the collective diversity of human creative outputs when collaborating with GenAI. 

---
# Skeleton-Based Transformer for Classification of Errors and Better Feedback in Low Back Pain Physical Rehabilitation Exercises 

**Authors**: Aleksa Marusic, Sao Mai Nguyen, Adriana Tapus  

**Link**: [PDF](https://arxiv.org/pdf/2504.13866)  

**Abstract**: Physical rehabilitation exercises suggested by healthcare professionals can help recovery from various musculoskeletal disorders and prevent re-injury. However, patients' engagement tends to decrease over time without direct supervision, which is why there is a need for an automated monitoring system. In recent years, there has been great progress in quality assessment of physical rehabilitation exercises. Most of them only provide a binary classification if the performance is correct or incorrect, and a few provide a continuous score. This information is not sufficient for patients to improve their performance. In this work, we propose an algorithm for error classification of rehabilitation exercises, thus making the first step toward more detailed feedback to patients. We focus on skeleton-based exercise assessment, which utilizes human pose estimation to evaluate motion. Inspired by recent algorithms for quality assessment during rehabilitation exercises, we propose a Transformer-based model for the described classification. Our model is inspired by the HyperFormer method for human action recognition, and adapted to our problem and dataset. The evaluation is done on the KERAAL dataset, as it is the only medical dataset with clear error labels for the exercises, and our model significantly surpasses state-of-the-art methods. Furthermore, we bridge the gap towards better feedback to the patients by presenting a way to calculate the importance of joints for each exercise. 

---
# A Survey on (M)LLM-Based GUI Agents 

**Authors**: Fei Tang, Haolei Xu, Hang Zhang, Siqi Chen, Xingyu Wu, Yongliang Shen, Wenqi Zhang, Guiyang Hou, Zeqi Tan, Yuchen Yan, Kaitao Song, Jian Shao, Weiming Lu, Jun Xiao, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2504.13865)  

**Abstract**: Graphical User Interface (GUI) Agents have emerged as a transformative paradigm in human-computer interaction, evolving from rule-based automation scripts to sophisticated AI-driven systems capable of understanding and executing complex interface operations. This survey provides a comprehensive examination of the rapidly advancing field of LLM-based GUI Agents, systematically analyzing their architectural foundations, technical components, and evaluation methodologies. We identify and analyze four fundamental components that constitute modern GUI Agents: (1) perception systems that integrate text-based parsing with multimodal understanding for comprehensive interface comprehension; (2) exploration mechanisms that construct and maintain knowledge bases through internal modeling, historical experience, and external information retrieval; (3) planning frameworks that leverage advanced reasoning methodologies for task decomposition and execution; and (4) interaction systems that manage action generation with robust safety controls. Through rigorous analysis of these components, we reveal how recent advances in large language models and multimodal learning have revolutionized GUI automation across desktop, mobile, and web platforms. We critically examine current evaluation frameworks, highlighting methodological limitations in existing benchmarks while proposing directions for standardization. This survey also identifies key technical challenges, including accurate element localization, effective knowledge retrieval, long-horizon planning, and safety-aware execution control, while outlining promising research directions for enhancing GUI Agents' capabilities. Our systematic review provides researchers and practitioners with a thorough understanding of the field's current state and offers insights into future developments in intelligent interface automation. 

---
# DoYouTrustAI: A Tool to Teach Students About AI Misinformation and Prompt Engineering 

**Authors**: Phillip Driscoll, Priyanka Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2504.13859)  

**Abstract**: AI, especially Large Language Models (LLMs) like ChatGPT, have rapidly developed and gained widespread adoption in the past five years, shifting user preference from traditional search engines. However, the generative nature of LLMs raises concerns about presenting misinformation as fact. To address this, we developed a web-based application that helps K-12 students enhance critical thinking by identifying misleading information in LLM responses about major historical figures. In this paper, we describe the implementation and design details of the DoYouTrustAI tool, which can be used to provide an interactive lesson which teaches students about the dangers of misinformation and how believable generative AI can make it seem. The DoYouTrustAI tool utilizes prompt engineering to present the user with AI generated summaries about the life of a historical figure. These summaries can be either accurate accounts of that persons life, or an intentionally misleading alteration of their history. The user is tasked with determining the validity of the statement without external resources. Our research questions for this work were:(RQ1) How can we design a tool that teaches students about the dangers of misleading information and of how misinformation can present itself in LLM responses? (RQ2) Can we present prompt engineering as a topic that is easily understandable for students? Our findings highlight the need to correct misleading information before users retain it. Our tool lets users select familiar individuals for testing to reduce random guessing and presents misinformation alongside known facts to maintain believability. It also provides pre-configured prompt instructions to show how different prompts affect AI responses. Together, these features create a controlled environment where users learn the importance of verifying AI responses and understanding prompt engineering. 

---
# The Effect of Explainable AI-based Decision Support on Human Task Performance: A Meta-Analysis 

**Authors**: Felix Haag  

**Link**: [PDF](https://arxiv.org/pdf/2504.13858)  

**Abstract**: The desirable properties of explanations in information systems have fueled the demands for transparency in artificial intelligence (AI) outputs. To address these demands, the field of explainable AI (XAI) has put forth methods that can support human decision-making by explaining AI outputs. However, current empirical works present inconsistent findings on whether such explanations help to improve users' task performance in decision support systems (DSS). In this paper, we conduct a meta-analysis to explore how XAI affects human performance in classification tasks. Our results show an improvement in task performance through XAI-based decision support, though explanations themselves are not the decisive driver for this improvement. The analysis reveals that the studies' risk of bias moderates the effect of explanations in AI, while the explanation type appears to play only a negligible role. Our findings contribute to the human computer interaction field by enhancing the understanding of human-XAI collaboration in DSS. 

---
# Towards Balancing Preference and Performance through Adaptive Personalized Explainability 

**Authors**: Andrew Silva, Pradyumna Tambwekar, Mariah Schrum, Matthew Gombolay  

**Link**: [PDF](https://arxiv.org/pdf/2504.13856)  

**Abstract**: As robots and digital assistants are deployed in the real world, these agents must be able to communicate their decision-making criteria to build trust, improve human-robot teaming, and enable collaboration. While the field of explainable artificial intelligence (xAI) has made great strides to enable such communication, these advances often assume that one xAI approach is ideally suited to each problem (e.g., decision trees to explain how to triage patients in an emergency or feature-importance maps to explain radiology reports). This fails to recognize that users have diverse experiences or preferences for interaction modalities. In this work, we present two user-studies set in a simulated autonomous vehicle (AV) domain. We investigate (1) population-level preferences for xAI and (2) personalization strategies for providing robot explanations. We find significant differences between xAI modes (language explanations, feature-importance maps, and decision trees) in both preference (p < 0.01) and performance (p < 0.05). We also observe that a participant's preferences do not always align with their performance, motivating our development of an adaptive personalization strategy to balance the two. We show that this strategy yields significant performance gains (p < 0.05), and we conclude with a discussion of our findings and implications for xAI in human-robot interactions. 

---
# GenShin:geometry-enhanced structural graph embodies binding pose can better predicting compound-protein interaction affinity 

**Authors**: Pingfei Zhu, Chenyang Zhao, Haishi Zhao, Bo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.13853)  

**Abstract**: AI-powered drug discovery typically relies on the successful prediction of compound-protein interactions, which are pivotal for the evaluation of designed compound molecules in structure-based drug design and represent a core challenge in the field.
However, accurately predicting compound-protein affinity via regression models usually requires adequate-binding pose, which are derived from costly and complex experimental methods or time-consuming simulations with docking software. In response, we have introduced the GenShin model, which constructs a geometry-enhanced structural graph module that separately extracts additional features from proteins and compounds. Consequently, it attains an accuracy on par with mainstream models in predicting compound-protein affinities, while eliminating the need for adequate-binding pose as input. Our experimental findings demonstrate that the GenShin model vastly outperforms other models that rely on non-input docking conformations, achieving, or in some cases even exceeding, the performance of those requiring adequate-binding pose. Further experiments indicate that our GenShin model is more robust to inadequate-binding pose, affirming its higher suitability for real-world drug discovery scenarios. We hope our work will inspire more endeavors to bridge the gap between AI models and practical drug discovery challenges. 

---
# From Interaction to Collaboration: How Hybrid Intelligence Enhances Chatbot Feedback 

**Authors**: Janet Rafner, Ryan Q. Guloy, Eden W. Wen, Catherine M. Chiodo, Jacob Sherson  

**Link**: [PDF](https://arxiv.org/pdf/2504.13848)  

**Abstract**: Generative AI (GenAI) chatbots are becoming increasingly integrated into virtual assistant technologies, yet their success hinges on the ability to gather meaningful user feedback to improve interaction quality, system outcomes, and overall user acceptance. Successful chatbot interactions can enable organizations to build long-term relationships with their customers and users, supporting customer loyalty and furthering the organization's goals. This study explores the impact of two distinct narratives and feedback collection mechanisms on user engagement and feedback behavior: a standard AI-focused interaction versus a hybrid intelligence (HI) framed interaction. Initial findings indicate that while small-scale survey measures allowed for no significant differences in user willingness to leave feedback, use the system, or trust the system, participants exposed to the HI narrative statistically significantly provided more detailed feedback. These initial findings offer insights into designing effective feedback systems for GenAI virtual assistants, balancing user effort with system improvement potential. 

---
