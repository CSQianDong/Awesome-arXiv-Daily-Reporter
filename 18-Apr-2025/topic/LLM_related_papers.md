# Exploring Expert Failures Improves LLM Agent Tuning 

**Authors**: Li-Cheng Lan, Andrew Bai, Minhao Cheng, Ruochen Wang, Cho-Jui Hsieh, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.13145)  

**Abstract**: Large Language Models (LLMs) have shown tremendous potential as agents, excelling at tasks that require multiple rounds of reasoning and interactions. Rejection Sampling Fine-Tuning (RFT) has emerged as an effective method for finetuning LLMs as agents: it first imitates expert-generated successful trajectories and further improves agentic skills through iterative fine-tuning on successful, self-generated trajectories. However, since the expert (e.g., GPT-4) succeeds primarily on simpler subtasks and RFT inherently favors simpler scenarios, many complex subtasks remain unsolved and persistently out-of-distribution (OOD). Upon investigating these challenging subtasks, we discovered that previously failed expert trajectories can often provide valuable guidance, e.g., plans and key actions, that can significantly improve agent exploration efficiency and acquisition of critical skills. Motivated by these observations, we propose Exploring Expert Failures (EEF), which identifies beneficial actions from failed expert trajectories and integrates them into the training dataset. Potentially harmful actions are meticulously excluded to prevent contamination of the model learning process. By leveraging the beneficial actions in expert failures, EEF successfully solves some previously unsolvable subtasks and improves agent tuning performance. Remarkably, our approach achieved a 62\% win rate in WebShop, outperforming RFT (53. 6\%) and GPT-4 (35. 6\%), and to the best of our knowledge, setting a new state-of-the-art as the first method to surpass a score of 0.81 in WebShop and exceed 81 in SciWorld. 

---
# InstructRAG: Leveraging Retrieval-Augmented Generation on Instruction Graphs for LLM-Based Task Planning 

**Authors**: Zheng Wang, Shu Xian Teo, Jun Jie Chew, Wei Shi  

**Link**: [PDF](https://arxiv.org/pdf/2504.13032)  

**Abstract**: Recent advancements in large language models (LLMs) have enabled their use as agents for planning complex tasks. Existing methods typically rely on a thought-action-observation (TAO) process to enhance LLM performance, but these approaches are often constrained by the LLMs' limited knowledge of complex tasks. Retrieval-augmented generation (RAG) offers new opportunities by leveraging external databases to ground generation in retrieved information. In this paper, we identify two key challenges (enlargability and transferability) in applying RAG to task planning. We propose InstructRAG, a novel solution within a multi-agent meta-reinforcement learning framework, to address these challenges. InstructRAG includes a graph to organize past instruction paths (sequences of correct actions), an RL-Agent with Reinforcement Learning to expand graph coverage for enlargability, and an ML-Agent with Meta-Learning to improve task generalization for transferability. The two agents are trained end-to-end to optimize overall planning performance. Our experiments on four widely used task planning datasets demonstrate that InstructRAG significantly enhances performance and adapts efficiently to new tasks, achieving up to a 19.2% improvement over the best existing approach. 

---
# ZeroSumEval: Scaling LLM Evaluation with Inter-Model Competition 

**Authors**: Haidar Khan, Hisham A. Alyahya, Yazeed Alnumay, M Saiful Bari, Bülent Yener  

**Link**: [PDF](https://arxiv.org/pdf/2504.12562)  

**Abstract**: Evaluating the capabilities of Large Language Models (LLMs) has traditionally relied on static benchmark datasets, human assessments, or model-based evaluations - methods that often suffer from overfitting, high costs, and biases. ZeroSumEval is a novel competition-based evaluation protocol that leverages zero-sum games to assess LLMs with dynamic benchmarks that resist saturation. ZeroSumEval encompasses a diverse suite of games, including security challenges (PyJail), classic games (Chess, Liar's Dice, Poker), knowledge tests (MathQuiz), and persuasion challenges (Gandalf, Debate). These games are designed to evaluate a range of AI capabilities such as strategic reasoning, planning, knowledge application, and creativity. Building upon recent studies that highlight the effectiveness of game-based evaluations for LLMs, ZeroSumEval enhances these approaches by providing a standardized and extensible framework. To demonstrate this, we conduct extensive experiments with >7000 simulations across 7 games and 13 models. Our results show that while frontier models from the GPT and Claude families can play common games and answer questions, they struggle to play games that require creating novel and challenging questions. We also observe that models cannot reliably jailbreak each other and fail generally at tasks requiring creativity. We release our code at this https URL. 

---
# Sleep-time Compute: Beyond Inference Scaling at Test-time 

**Authors**: Kevin Lin, Charlie Snell, Yu Wang, Charles Packer, Sarah Wooders, Ion Stoica, Joseph E. Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2504.13171)  

**Abstract**: Scaling test-time compute has emerged as a key ingredient for enabling large language models (LLMs) to solve difficult problems, but comes with high latency and inference cost. We introduce sleep-time compute, which allows models to "think" offline about contexts before queries are presented: by anticipating what queries users might ask and pre-computing useful quantities, we can significantly reduce the compute requirements at test-time. To demonstrate the efficacy of our method, we create modified versions of two reasoning tasks - Stateful GSM-Symbolic and Stateful AIME. We find that sleep-time compute can reduce the amount of test-time compute needed to achieve the same accuracy by ~ 5x on Stateful GSM-Symbolic and Stateful AIME and that by scaling sleep-time compute we can further increase accuracy by up to 13% on Stateful GSM-Symbolic and 18% on Stateful AIME. Furthermore, we introduce Multi-Query GSM-Symbolic, which extends GSM-Symbolic by including multiple related queries per context. By amortizing sleep-time compute across related queries about the same context using Multi-Query GSM-Symbolic, we can decrease the average cost per query by 2.5x. We then conduct additional analysis to understand when sleep-time compute is most effective, finding the predictability of the user query to be well correlated with the efficacy of sleep-time compute. Finally, we conduct a case-study of applying sleep-time compute to a realistic agentic SWE task. 

---
# Towards Conversational AI for Human-Machine Collaborative MLOps 

**Authors**: George Fatouros, Georgios Makridis, George Kousiouris, John Soldatos, Anargyros Tsadimas, Dimosthenis Kyriazis  

**Link**: [PDF](https://arxiv.org/pdf/2504.12477)  

**Abstract**: This paper presents a Large Language Model (LLM) based conversational agent system designed to enhance human-machine collaboration in Machine Learning Operations (MLOps). We introduce the Swarm Agent, an extensible architecture that integrates specialized agents to create and manage ML workflows through natural language interactions. The system leverages a hierarchical, modular design incorporating a KubeFlow Pipelines (KFP) Agent for ML pipeline orchestration, a MinIO Agent for data management, and a Retrieval-Augmented Generation (RAG) Agent for domain-specific knowledge integration. Through iterative reasoning loops and context-aware processing, the system enables users with varying technical backgrounds to discover, execute, and monitor ML pipelines; manage datasets and artifacts; and access relevant documentation, all via intuitive conversational interfaces. Our approach addresses the accessibility gap in complex MLOps platforms like Kubeflow, making advanced ML tools broadly accessible while maintaining the flexibility to extend to other platforms. The paper describes the architecture, implementation details, and demonstrates how this conversational MLOps assistant reduces complexity and lowers barriers to entry for users across diverse technical skill levels. 

---
# LLMs Meet Finance: Fine-Tuning Foundation Models for the Open FinLLM Leaderboard 

**Authors**: Varun Rao, Youran Sun, Mahendra Kumar, Tejas Mutneja, Agastya Mukherjee, Haizhao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.13125)  

**Abstract**: This paper investigates the application of large language models (LLMs) to financial tasks. We fine-tuned foundation models using the Open FinLLM Leaderboard as a benchmark. Building on Qwen2.5 and Deepseek-R1, we employed techniques including supervised fine-tuning (SFT), direct preference optimization (DPO), and reinforcement learning (RL) to enhance their financial capabilities. The fine-tuned models demonstrated substantial performance gains across a wide range of financial tasks. Moreover, we measured the data scaling law in the financial domain. Our work demonstrates the potential of large language models (LLMs) in financial applications. 

---
# Accuracy is Not Agreement: Expert-Aligned Evaluation of Crash Narrative Classification Models 

**Authors**: Sudesh Ramesh Bhagat, Ibne Farabi Shihab, Anuj Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2504.13068)  

**Abstract**: This study explores the relationship between deep learning (DL) model accuracy and expert agreement in the classification of crash narratives. We evaluate five DL models -- including BERT variants, the Universal Sentence Encoder (USE), and a zero-shot classifier -- against expert-labeled data and narrative text. The analysis is further extended to four large language models (LLMs): GPT-4, LLaMA 3, Qwen, and Claude. Our results reveal a counterintuitive trend: models with higher technical accuracy often exhibit lower agreement with domain experts, whereas LLMs demonstrate greater expert alignment despite relatively lower accuracy scores. To quantify and interpret model-expert agreement, we employ Cohen's Kappa, Principal Component Analysis (PCA), and SHAP-based explainability techniques. Findings indicate that expert-aligned models tend to rely more on contextual and temporal language cues, rather than location-specific keywords. These results underscore that accuracy alone is insufficient for evaluating models in safety-critical NLP applications. We advocate for incorporating expert agreement as a complementary metric in model evaluation frameworks and highlight the promise of LLMs as interpretable, scalable tools for crash analysis pipelines. 

---
# SHA256 at SemEval-2025 Task 4: Selective Amnesia -- Constrained Unlearning for Large Language Models via Knowledge Isolation 

**Authors**: Saransh Agrawal, Kuan-Hao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.12996)  

**Abstract**: Large language models (LLMs) frequently memorize sensitive information during training, posing risks when deploying publicly accessible models. Current machine unlearning methods struggle to selectively remove specific data associations without degrading overall model capabilities. This paper presents our solution to SemEval-2025 Task 4 on targeted unlearning, which introduces a two-stage methodology that combines causal mediation analysis with layer-specific optimization. Through systematic causal tracing experiments on OLMo architectures (1B and 7B parameters), we identify the critical role of the first few transformer layers (layers 0-5) in storing subject-attribute associations within MLP modules. Building on this insight, we develop a constrained optimization approach that freezes upper layers while applying a novel joint loss function to lower layers-simultaneously maximizing forget set loss via output token cross-entropy penalties and minimizing retain set deviation through adaptive regularization. Our method achieves 2nd place in the 1B model track, demonstrating strong task performance while maintaining 88% of baseline MMLU accuracy. These results establish causal-informed layer optimization as a promising paradigm for efficient, precise unlearning in LLMs, offering a significant step forward in addressing data privacy concerns in AI systems. 

---
# Accommodate Knowledge Conflicts in Retrieval-augmented LLMs: Towards Reliable Response Generation in the Wild 

**Authors**: Jiatai Wang, Zhiwei Xu, Di Jin, Xuewen Yang, Tao Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.12982)  

**Abstract**: The proliferation of large language models (LLMs) has significantly advanced information retrieval systems, particularly in response generation (RG). Unfortunately, LLMs often face knowledge conflicts between internal memory and retrievaled external information, arising from misinformation, biases, or outdated knowledge. These conflicts undermine response reliability and introduce uncertainty in decision-making. In this work, we analyze how LLMs navigate knowledge conflicts from an information-theoretic perspective and reveal that when conflicting and supplementary information exhibit significant differences, LLMs confidently resolve their preferences. However, when the distinction is ambiguous, LLMs experience heightened uncertainty. Based on this insight, we propose Swin-VIB, a novel framework that integrates a pipeline of variational information bottleneck models into adaptive augmentation of retrieved information and guiding LLM preference in response generation. Extensive experiments on single-choice, open-ended question-answering (QA), and retrieval augmented generation (RAG) validate our theoretical findings and demonstrate the efficacy of Swin-VIB. Notably, our method improves single-choice task accuracy by at least 7.54\% over competitive baselines. 

---
# Are Retrials All You Need? Enhancing Large Language Model Reasoning Without Verbalized Feedback 

**Authors**: Nearchos Potamitis, Akhil Arora  

**Link**: [PDF](https://arxiv.org/pdf/2504.12951)  

**Abstract**: Recent advancements in large language models (LLMs) have catalyzed the development of general-purpose autonomous agents, demonstrating remarkable performance in complex reasoning tasks across various domains. This surge has spurred the evolution of a plethora of prompt-based reasoning frameworks. A recent focus has been on iterative reasoning strategies that refine outputs through self-evaluation and verbalized feedback. However, these strategies require additional computational complexity to enable models to recognize and correct their mistakes, leading to a significant increase in their cost. In this work, we introduce the concept of ``retrials without feedback'', an embarrassingly simple yet powerful mechanism for enhancing reasoning frameworks by allowing LLMs to retry problem-solving attempts upon identifying incorrect answers. Unlike conventional iterative refinement methods, our method does not require explicit self-reflection or verbalized feedback, simplifying the refinement process. Our findings indicate that simpler retrial-based approaches often outperform more sophisticated reasoning frameworks, suggesting that the benefits of complex methods may not always justify their computational costs. By challenging the prevailing assumption that more intricate reasoning strategies inherently lead to better performance, our work offers new insights into how simpler, more efficient approaches can achieve optimal results. So, are retrials all you need? 

---
# Benchmarking Multi-National Value Alignment for Large Language Models 

**Authors**: Chengyi Ju, Weijie Shi, Chengzhong Liu, Jiaming Ji, Jipeng Zhang, Ruiyuan Zhang, Jia Zhu, Jiajie Xu, Yaodong Yang, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2504.12911)  

**Abstract**: Do Large Language Models (LLMs) hold positions that conflict with your country's values? Occasionally they do! However, existing works primarily focus on ethical reviews, failing to capture the diversity of national values, which encompass broader policy, legal, and moral considerations. Furthermore, current benchmarks that rely on spectrum tests using manually designed questionnaires are not easily scalable.
To address these limitations, we introduce NaVAB, a comprehensive benchmark to evaluate the alignment of LLMs with the values of five major nations: China, the United States, the United Kingdom, France, and Germany. NaVAB implements a national value extraction pipeline to efficiently construct value assessment datasets. Specifically, we propose a modeling procedure with instruction tagging to process raw data sources, a screening process to filter value-related topics and a generation process with a Conflict Reduction mechanism to filter non-conflicting this http URL conduct extensive experiments on various LLMs across countries, and the results provide insights into assisting in the identification of misaligned scenarios. Moreover, we demonstrate that NaVAB can be combined with alignment techniques to effectively reduce value concerns by aligning LLMs' values with the target country. 

---
# EmoVoice: LLM-based Emotional Text-To-Speech Model with Freestyle Text Prompting 

**Authors**: Guanrou Yang, Chen Yang, Qian Chen, Ziyang Ma, Wenxi Chen, Wen Wang, Tianrui Wang, Yifan Yang, Zhikang Niu, Wenrui Liu, Fan Yu, Zhihao Du, Zhifu Gao, ShiLiang Zhang, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.12867)  

**Abstract**: Human speech goes beyond the mere transfer of information; it is a profound exchange of emotions and a connection between individuals. While Text-to-Speech (TTS) models have made huge progress, they still face challenges in controlling the emotional expression in the generated speech. In this work, we propose EmoVoice, a novel emotion-controllable TTS model that exploits large language models (LLMs) to enable fine-grained freestyle natural language emotion control, and a phoneme boost variant design that makes the model output phoneme tokens and audio tokens in parallel to enhance content consistency, inspired by chain-of-thought (CoT) and modality-of-thought (CoM) techniques. Besides, we introduce EmoVoice-DB, a high-quality 40-hour English emotion dataset featuring expressive speech and fine-grained emotion labels with natural language descriptions. EmoVoice achieves state-of-the-art performance on the English EmoVoice-DB test set using only synthetic training data, and on the Chinese Secap test set using our in-house data. We further investigate the reliability of existing emotion evaluation metrics and their alignment with human perceptual preferences, and explore using SOTA multimodal LLMs GPT-4o-audio and Gemini to assess emotional speech. Demo samples are available at this https URL. Dataset, code, and checkpoints will be released. 

---
# Information Gain-Guided Causal Intervention for Autonomous Debiasing Large Language Models 

**Authors**: Zhouhao Sun, Xiao Ding, Li Du, Yunpeng Xu, Yixuan Ma, Yang Zhao, Bing Qin, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.12898)  

**Abstract**: Despite significant progress, recent studies indicate that current large language models (LLMs) may still capture dataset biases and utilize them during inference, leading to the poor generalizability of LLMs. However, due to the diversity of dataset biases and the insufficient nature of bias suppression based on in-context learning, the effectiveness of previous prior knowledge-based debiasing methods and in-context learning based automatic debiasing methods is limited. To address these challenges, we explore the combination of causal mechanisms with information theory and propose an information gain-guided causal intervention debiasing (IGCIDB) framework. This framework first utilizes an information gain-guided causal intervention method to automatically and autonomously balance the distribution of instruction-tuning dataset. Subsequently, it employs a standard supervised fine-tuning process to train LLMs on the debiased dataset. Experimental results show that IGCIDB can effectively debias LLM to improve its generalizability across different tasks. 

---
# Retrieval-Augmented Generation with Conflicting Evidence 

**Authors**: Han Wang, Archiki Prasad, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2504.13079)  

**Abstract**: Large language model (LLM) agents are increasingly employing retrieval-augmented generation (RAG) to improve the factuality of their responses. However, in practice, these systems often need to handle ambiguous user queries and potentially conflicting information from multiple sources while also suppressing inaccurate information from noisy or irrelevant documents. Prior work has generally studied and addressed these challenges in isolation, considering only one aspect at a time, such as handling ambiguity or robustness to noise and misinformation. We instead consider multiple factors simultaneously, proposing (i) RAMDocs (Retrieval with Ambiguity and Misinformation in Documents), a new dataset that simulates complex and realistic scenarios for conflicting evidence for a user query, including ambiguity, misinformation, and noise; and (ii) MADAM-RAG, a multi-agent approach in which LLM agents debate over the merits of an answer over multiple rounds, allowing an aggregator to collate responses corresponding to disambiguated entities while discarding misinformation and noise, thereby handling diverse sources of conflict jointly. We demonstrate the effectiveness of MADAM-RAG using both closed and open-source models on AmbigDocs -- which requires presenting all valid answers for ambiguous queries -- improving over strong RAG baselines by up to 11.40% and on FaithEval -- which requires suppressing misinformation -- where we improve by up to 15.80% (absolute) with Llama3.3-70B-Instruct. Furthermore, we find that RAMDocs poses a challenge for existing RAG baselines (Llama3.3-70B-Instruct only obtains 32.60 exact match score). While MADAM-RAG begins to address these conflicting factors, our analysis indicates that a substantial gap remains especially when increasing the level of imbalance in supporting evidence and misinformation. 

---
# QLLM: Do We Really Need a Mixing Network for Credit Assignment in Multi-Agent Reinforcement Learning? 

**Authors**: Zhouyang Jiang, Bin Zhang, Airong Wei, Zhiwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.12961)  

**Abstract**: Credit assignment has remained a fundamental challenge in multi-agent reinforcement learning (MARL). Previous studies have primarily addressed this issue through value decomposition methods under the centralized training with decentralized execution paradigm, where neural networks are utilized to approximate the nonlinear relationship between individual Q-values and the global Q-value. Although these approaches have achieved considerable success in various benchmark tasks, they still suffer from several limitations, including imprecise attribution of contributions, limited interpretability, and poor scalability in high-dimensional state spaces. To address these challenges, we propose a novel algorithm, \textbf{QLLM}, which facilitates the automatic construction of credit assignment functions using large language models (LLMs). Specifically, the concept of \textbf{TFCAF} is introduced, wherein the credit allocation process is represented as a direct and expressive nonlinear functional formulation. A custom-designed \textit{coder-evaluator} framework is further employed to guide the generation, verification, and refinement of executable code by LLMs, significantly mitigating issues such as hallucination and shallow reasoning during inference. Extensive experiments conducted on several standard MARL benchmarks demonstrate that the proposed method consistently outperforms existing state-of-the-art baselines. Moreover, QLLM exhibits strong generalization capability and maintains compatibility with a wide range of MARL algorithms that utilize mixing networks, positioning it as a promising and versatile solution for complex multi-agent scenarios. 

---
# Trajectory Adaptation using Large Language Models 

**Authors**: Anurag Maurya, Tashmoy Ghosh, Ravi Prakash  

**Link**: [PDF](https://arxiv.org/pdf/2504.12755)  

**Abstract**: Adapting robot trajectories based on human instructions as per new situations is essential for achieving more intuitive and scalable human-robot interactions. This work proposes a flexible language-based framework to adapt generic robotic trajectories produced by off-the-shelf motion planners like RRT, A-star, etc, or learned from human demonstrations. We utilize pre-trained LLMs to adapt trajectory waypoints by generating code as a policy for dense robot manipulation, enabling more complex and flexible instructions than current methods. This approach allows us to incorporate a broader range of commands, including numerical inputs. Compared to state-of-the-art feature-based sequence-to-sequence models which require training, our method does not require task-specific training and offers greater interpretability and more effective feedback mechanisms. We validate our approach through simulation experiments on the robotic manipulator, aerial vehicle, and ground robot in the Pybullet and Gazebo simulation environments, demonstrating that LLMs can successfully adapt trajectories to complex human instructions. 

---
# Pandora: A Code-Driven Large Language Model Agent for Unified Reasoning Across Diverse Structured Knowledge 

**Authors**: Yongrui Chen, Junhao He, Linbo Fu, Shenyu Zhang, Rihui Jin, Xinbang Dai, Jiaqi Li, Dehai Min, Nan Hu, Yuxin Zhang, Guilin Qi, Yi Huang, Tongtong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.12734)  

**Abstract**: Unified Structured Knowledge Reasoning (USKR) aims to answer natural language questions (NLQs) by using structured sources such as tables, databases, and knowledge graphs in a unified way. Existing USKR methods either rely on employing task-specific strategies or custom-defined representations, which struggle to leverage the knowledge transfer between different SKR tasks or align with the prior of LLMs, thereby limiting their performance. This paper proposes a novel USKR framework named \textsc{Pandora}, which takes advantage of \textsc{Python}'s \textsc{Pandas} API to construct a unified knowledge representation for alignment with LLM pre-training. It employs an LLM to generate textual reasoning steps and executable Python code for each question. Demonstrations are drawn from a memory of training examples that cover various SKR tasks, facilitating knowledge transfer. Extensive experiments on four benchmarks involving three SKR tasks demonstrate that \textsc{Pandora} outperforms existing unified frameworks and competes effectively with task-specific methods. 

---
# Enhancing the Geometric Problem-Solving Ability of Multimodal LLMs via Symbolic-Neural Integration 

**Authors**: Yicheng Pan, Zhenrong Zhang, Pengfei Hu, Jiefeng Ma, Jun Du, Jianshu Zhang, Quan Liu, Jianqing Gao, Feng Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.12773)  

**Abstract**: Recent advances in Multimodal Large Language Models (MLLMs) have achieved remarkable progress in general domains and demonstrated promise in multimodal mathematical reasoning. However, applying MLLMs to geometry problem solving (GPS) remains challenging due to lack of accurate step-by-step solution data and severe hallucinations during reasoning. In this paper, we propose GeoGen, a pipeline that can automatically generates step-wise reasoning paths for geometry diagrams. By leveraging the precise symbolic reasoning, \textbf{GeoGen} produces large-scale, high-quality question-answer pairs. To further enhance the logical reasoning ability of MLLMs, we train \textbf{GeoLogic}, a Large Language Model (LLM) using synthetic data generated by GeoGen. Serving as a bridge between natural language and symbolic systems, GeoLogic enables symbolic tools to help verifying MLLM outputs, making the reasoning process more rigorous and alleviating hallucinations. Experimental results show that our approach consistently improves the performance of MLLMs, achieving remarkable results on benchmarks for geometric reasoning tasks. This improvement stems from our integration of the strengths of LLMs and symbolic systems, which enables a more reliable and interpretable approach for the GPS task. Codes are available at this https URL. 

---
# Scaling Instruction-Tuned LLMs to Million-Token Contexts via Hierarchical Synthetic Data Generation 

**Authors**: Linda He, Jue Wang, Maurice Weber, Shang Zhu, Ben Athiwaratkun, Ce Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.12637)  

**Abstract**: Large Language Models (LLMs) struggle with long-context reasoning, not only due to the quadratic scaling of computational complexity with sequence length but also because of the scarcity and expense of annotating long-context data. There has been barely any open-source work that systematically ablates long-context data, nor is there any openly available instruction tuning dataset with contexts surpassing 100K tokens. To bridge this gap, we introduce a novel post-training synthetic data generation strategy designed to efficiently extend the context window of LLMs while preserving their general task performance. Our approach scalably extends to arbitrarily long context lengths, unconstrained by the length of available real-world data, which effectively addresses the scarcity of raw long-context data. Through a step-by-step rotary position embedding (RoPE) scaling training strategy, we demonstrate that our model, with a context length of up to 1M tokens, performs well on the RULER benchmark and InfiniteBench and maintains robust performance on general language tasks. 

---
# Code Copycat Conundrum: Demystifying Repetition in LLM-based Code Generation 

**Authors**: Mingwei Liu, Juntao Li, Ying Wang, Xueying Du, Zuoyu Ou, Qiuyuan Chen, Bingxu An, Zhao Wei, Yong Xu, Fangming Zou, Xin Peng, Yiling Lou  

**Link**: [PDF](https://arxiv.org/pdf/2504.12608)  

**Abstract**: Despite recent advances in Large Language Models (LLMs) for code generation, the quality of LLM-generated code still faces significant challenges. One significant issue is code repetition, which refers to the model's tendency to generate structurally redundant code, resulting in inefficiencies and reduced readability. To address this, we conduct the first empirical study to investigate the prevalence and nature of repetition across 19 state-of-the-art code LLMs using three widely-used benchmarks. Our study includes both quantitative and qualitative analyses, revealing that repetition is pervasive and manifests at various granularities and extents, including character, statement, and block levels. We further summarize a taxonomy of 20 repetition patterns. Building on our findings, we propose DeRep, a rule-based technique designed to detect and mitigate repetition in generated code. We evaluate DeRep using both open-source benchmarks and in an industrial setting. Our results demonstrate that DeRep significantly outperforms baselines in reducing repetition (with an average improvements of 91.3%, 93.5%, and 79.9% in rep-3, rep-line, and sim-line metrics) and enhancing code quality (with a Pass@1 increase of 208.3% over greedy search). Furthermore, integrating DeRep improves the performance of existing repetition mitigation methods, with Pass@1 improvements ranging from 53.7% to 215.7%. 

---
# Identifying and Mitigating the Influence of the Prior Distribution in Large Language Models 

**Authors**: Liyi Zhang, Veniamin Veselovsky, R. Thomas McCoy, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2504.12585)  

**Abstract**: Large language models (LLMs) sometimes fail to respond appropriately to deterministic tasks -- such as counting or forming acronyms -- because the implicit prior distribution they have learned over sequences of tokens influences their responses. In this work, we show that, in at least some cases, LLMs actually compute the information needed to perform these tasks correctly, and we identify some interventions that can allow them to access this information to improve their performance. First, we show that simply prompting the language model to not rely on its prior knowledge leads to dramatic improvements in prior-dominated tasks. We then use mechanistic interpretability techniques to localize the prior within the LLM and manipulate the extent to which that prior influences its responses. Specifically, we show that it is possible to identify layers of the underlying neural network that correlate with the prior probability of a response and that lightweight finetuning of these layers with basic prompts on prior-dominated tasks achieves high performance on held-out answers. These results suggest that the information required to produce a correct response is contained within the representations of the problems formed by the models. Furthermore, we show that this finetuning is significantly more effective for prior-dominated tasks, and that the error after finetuning is no longer correlated with the prior. Our results suggest that it may be possible to define effective methods for manipulating the extent to which LLMs rely upon their priors in solving problems, potentially increasing their performance in settings where LLMs hallucinate for reasons related to the prior probability of token sequences. 

---
# SimUSER: Simulating User Behavior with Large Language Models for Recommender System Evaluation 

**Authors**: Nicolas Bougie, Narimasa Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2504.12722)  

**Abstract**: Recommender systems play a central role in numerous real-life applications, yet evaluating their performance remains a significant challenge due to the gap between offline metrics and online behaviors. Given the scarcity and limits (e.g., privacy issues) of real user data, we introduce SimUSER, an agent framework that serves as believable and cost-effective human proxies. SimUSER first identifies self-consistent personas from historical data, enriching user profiles with unique backgrounds and personalities. Then, central to this evaluation are users equipped with persona, memory, perception, and brain modules, engaging in interactions with the recommender system. SimUSER exhibits closer alignment with genuine humans than prior work, both at micro and macro levels. Additionally, we conduct insightful experiments to explore the effects of thumbnails on click rates, the exposure effect, and the impact of reviews on user engagement. Finally, we refine recommender system parameters based on offline A/B test results, resulting in improved user engagement in the real world. 

---
# Knowledge Acquisition on Mass-shooting Events via LLMs for AI-Driven Justice 

**Authors**: Benign John Ihugba, Afsana Nasrin, Ling Wu, Lin Li, Lijun Qian, Xishuang Dong  

**Link**: [PDF](https://arxiv.org/pdf/2504.12545)  

**Abstract**: Mass-shooting events pose a significant challenge to public safety, generating large volumes of unstructured textual data that hinder effective investigations and the formulation of public policy. Despite the urgency, few prior studies have effectively automated the extraction of key information from these events to support legal and investigative efforts. This paper presented the first dataset designed for knowledge acquisition on mass-shooting events through the application of named entity recognition (NER) techniques. It focuses on identifying key entities such as offenders, victims, locations, and criminal instruments, that are vital for legal and investigative purposes. The NER process is powered by Large Language Models (LLMs) using few-shot prompting, facilitating the efficient extraction and organization of critical information from diverse sources, including news articles, police reports, and social media. Experimental results on real-world mass-shooting corpora demonstrate that GPT-4o is the most effective model for mass-shooting NER, achieving the highest Micro Precision, Micro Recall, and Micro F1-scores. Meanwhile, o1-mini delivers competitive performance, making it a resource-efficient alternative for less complex NER tasks. It is also observed that increasing the shot count enhances the performance of all models, but the gains are more substantial for GPT-4o and o1-mini, highlighting their superior adaptability to few-shot learning scenarios. 

---
# Memorization vs. Reasoning: Updating LLMs with New Knowledge 

**Authors**: Aochong Oliver Li, Tanya Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2504.12523)  

**Abstract**: Large language models (LLMs) encode vast amounts of pre-trained knowledge in their parameters, but updating them as real-world information evolves remains a challenge. Existing methodologies and benchmarks primarily target entity substitutions, failing to capture the full breadth of complex real-world dynamics. In this paper, we introduce Knowledge Update Playground (KUP), an automatic pipeline for simulating realistic knowledge updates reflected in an evidence corpora. KUP's evaluation framework includes direct and indirect probes to both test memorization of updated facts and reasoning over them, for any update learning methods. Next, we present a lightweight method called memory conditioned training (MCT), which conditions tokens in the update corpus on self-generated "memory" tokens during training. Our strategy encourages LLMs to surface and reason over newly memorized knowledge at inference. Our results on two strong LLMs show that (1) KUP benchmark is highly challenging, with the best CPT models achieving $<2\%$ in indirect probing setting (reasoning) and (2) MCT training significantly outperforms prior continued pre-training (CPT) baselines, improving direct probing (memorization) results by up to $25.4\%$. 

---
# Evaluating the Diversity and Quality of LLM Generated Content 

**Authors**: Alexander Shypula, Shuo Li, Botong Zhang, Vishakh Padmakumar, Kayo Yin, Osbert Bastani  

**Link**: [PDF](https://arxiv.org/pdf/2504.12522)  

**Abstract**: Recent work suggests that preference-tuning techniques--including Reinforcement Learning from Human Preferences (RLHF) methods like PPO and GRPO, as well as alternatives like DPO--reduce diversity, creating a dilemma given that such models are widely deployed in applications requiring diverse outputs. To address this, we introduce a framework for measuring effective semantic diversity--diversity among outputs that meet quality thresholds--which better reflects the practical utility of large language models (LLMs). Using open-ended tasks that require no human intervention, we find counterintuitive results: although preference-tuned models--especially those trained via RL--exhibit reduced lexical and syntactic diversity, they produce greater effective semantic diversity than SFT or base models, not from increasing diversity among high-quality outputs, but from generating more high-quality outputs overall. We discover that preference tuning reduces syntactic diversity while preserving semantic diversity--revealing a distinction between diversity in form and diversity in content that traditional metrics often overlook. Our analysis further shows that smaller models are consistently more parameter-efficient at generating unique content within a fixed sampling budget, offering insights into the relationship between model scaling and diversity. These findings have important implications for applications that require diverse yet high-quality outputs, from creative assistance to synthetic data generation. 

---
# Persona-judge: Personalized Alignment of Large Language Models via Token-level Self-judgment 

**Authors**: Xiaotian Zhang, Ruizhe Chen, Yang Feng, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.12663)  

**Abstract**: Aligning language models with human preferences presents significant challenges, particularly in achieving personalization without incurring excessive computational costs. Existing methods rely on reward signals and additional annotated data, limiting their scalability and adaptability to diverse human values. To address these challenges, we introduce Persona-judge, a novel discriminative paradigm that enables training-free personalized alignment with unseen preferences. Instead of optimizing policy parameters through external reward feedback, Persona-judge leverages the intrinsic preference judgment capabilities of the model. Specifically, a draft model generates candidate tokens conditioned on a given preference, while a judge model, embodying another preference, cross-validates the predicted tokens whether to be accepted. Experimental results demonstrate that Persona-judge, using the inherent preference evaluation mechanisms of the model, offers a scalable and computationally efficient solution to personalized alignment, paving the way for more adaptive customized alignment. 

---
# Mitigating LLM Hallucinations with Knowledge Graphs: A Case Study 

**Authors**: Harry Li, Gabriel Appleby, Kenneth Alperin, Steven R Gomez, Ashley Suh  

**Link**: [PDF](https://arxiv.org/pdf/2504.12422)  

**Abstract**: High-stakes domains like cyber operations need responsible and trustworthy AI methods. While large language models (LLMs) are becoming increasingly popular in these domains, they still suffer from hallucinations. This research paper provides learning outcomes from a case study with LinkQ, an open-source natural language interface that was developed to combat hallucinations by forcing an LLM to query a knowledge graph (KG) for ground-truth data during question-answering (QA). We conduct a quantitative evaluation of LinkQ using a well-known KGQA dataset, showing that the system outperforms GPT-4 but still struggles with certain question categories - suggesting that alternative query construction strategies will need to be investigated in future LLM querying systems. We discuss a qualitative study of LinkQ with two domain experts using a real-world cybersecurity KG, outlining these experts' feedback, suggestions, perceived limitations, and future opportunities for systems like LinkQ. 

---
# Multimodal LLM Augmented Reasoning for Interpretable Visual Perception Analysis 

**Authors**: Shravan Chaudhari, Trilokya Akula, Yoon Kim, Tom Blake  

**Link**: [PDF](https://arxiv.org/pdf/2504.12511)  

**Abstract**: In this paper, we advance the study of AI-augmented reasoning in the context of Human-Computer Interaction (HCI), psychology and cognitive science, focusing on the critical task of visual perception. Specifically, we investigate the applicability of Multimodal Large Language Models (MLLMs) in this domain. To this end, we leverage established principles and explanations from psychology and cognitive science related to complexity in human visual perception. We use them as guiding principles for the MLLMs to compare and interprete visual content. Our study aims to benchmark MLLMs across various explainability principles relevant to visual perception. Unlike recent approaches that primarily employ advanced deep learning models to predict complexity metrics from visual content, our work does not seek to develop a mere new predictive model. Instead, we propose a novel annotation-free analytical framework to assess utility of MLLMs as cognitive assistants for HCI tasks, using visual perception as a case study. The primary goal is to pave the way for principled study in quantifying and evaluating the interpretability of MLLMs for applications in improving human reasoning capability and uncovering biases in existing perception datasets annotated by humans. 

---
# Themisto: Jupyter-Based Runtime Benchmark 

**Authors**: Konstantin Grotov, Sergey Titov  

**Link**: [PDF](https://arxiv.org/pdf/2504.12365)  

**Abstract**: In this work, we present a benchmark that consists of Jupyter notebooks development trajectories and allows measuring how large language models (LLMs) can leverage runtime information for predicting code output and code generation. We demonstrate that the current generation of LLMs performs poorly on these tasks and argue that there exists a significantly understudied domain in the development of code-based models, which involves incorporating the runtime context. 

---
# Leveraging Large Language Models for Multi-Class and Multi-Label Detection of Drug Use and Overdose Symptoms on Social Media 

**Authors**: Muhammad Ahmad, Muhammad Waqas, ldar Batyrshin, Grigori Sidorov  

**Link**: [PDF](https://arxiv.org/pdf/2504.12355)  

**Abstract**: Drug overdose remains a critical global health issue, often driven by misuse of opioids, painkillers, and psychiatric medications. Traditional research methods face limitations, whereas social media offers real-time insights into self-reported substance use and overdose symptoms. This study proposes an AI-driven NLP framework trained on annotated social media data to detect commonly used drugs and associated overdose symptoms. Using a hybrid annotation strategy with LLMs and human annotators, we applied traditional ML models, neural networks, and advanced transformer-based models. Our framework achieved 98% accuracy in multi-class and 97% in multi-label classification, outperforming baseline models by up to 8%. These findings highlight the potential of AI for supporting public health surveillance and personalized intervention strategies. 

---
# Mathematical Capabilities of Large Language Models in Finnish Matriculation Examination 

**Authors**: Mika Setälä, Pieta Sikström, Ville Heilala, Tommi Kärkkäinen  

**Link**: [PDF](https://arxiv.org/pdf/2504.12347)  

**Abstract**: Large language models (LLMs) have shown increasing promise in educational settings, yet their mathematical reasoning has been considered evolving. This study evaluates the mathematical capabilities of various LLMs using the Finnish matriculation examination, a high-stakes digital test for upper secondary education. Initial tests yielded moderate performance corresponding to mid-range grades, but later evaluations demonstrated substantial improvements as the language models evolved. Remarkably, some models achieved near-perfect or perfect scores, matching top student performance and qualifying for university admission. Our findings highlight the rapid advances in the mathematical proficiency of LLMs and illustrate their potential to also support educational assessments at scale. 

---
# Don't Just Translate, Agitate: Using Large Language Models as Devil's Advocates for AI Explanations 

**Authors**: Ashley Suh, Kenneth Alperin, Harry Li, Steven R Gomez  

**Link**: [PDF](https://arxiv.org/pdf/2504.12424)  

**Abstract**: This position paper highlights a growing trend in Explainable AI (XAI) research where Large Language Models (LLMs) are used to translate outputs from explainability techniques, like feature-attribution weights, into a natural language explanation. While this approach may improve accessibility or readability for users, recent findings suggest that translating into human-like explanations does not necessarily enhance user understanding and may instead lead to overreliance on AI systems. When LLMs summarize XAI outputs without surfacing model limitations, uncertainties, or inconsistencies, they risk reinforcing the illusion of interpretability rather than fostering meaningful transparency. We argue that - instead of merely translating XAI outputs - LLMs should serve as constructive agitators, or devil's advocates, whose role is to actively interrogate AI explanations by presenting alternative interpretations, potential biases, training data limitations, and cases where the model's reasoning may break down. In this role, LLMs can facilitate users in engaging critically with AI systems and generated explanations, with the potential to reduce overreliance caused by misinterpreted or specious explanations. 

---
# Span-level Emotion-Cause-Category Triplet Extraction with Instruction Tuning LLMs and Data Augmentation 

**Authors**: Xiangju Li, Dong Yang, Xiaogang Zhu, Faliang Huang, Peng Zhang, Zhongying Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.12331)  

**Abstract**: Span-level emotion-cause-category triplet extraction represents a novel and complex challenge within emotion cause analysis. This task involves identifying emotion spans, cause spans, and their associated emotion categories within the text to form structured triplets. While prior research has predominantly concentrated on clause-level emotion-cause pair extraction and span-level emotion-cause detection, these methods often confront challenges originating from redundant information retrieval and difficulty in accurately determining emotion categories, particularly when emotions are expressed implicitly or ambiguously. To overcome these challenges, this study explores a fine-grained approach to span-level emotion-cause-category triplet extraction and introduces an innovative framework that leverages instruction tuning and data augmentation techniques based on large language models. The proposed method employs task-specific triplet extraction instructions and utilizes low-rank adaptation to fine-tune large language models, eliminating the necessity for intricate task-specific architectures. Furthermore, a prompt-based data augmentation strategy is developed to address data scarcity by guiding large language models in generating high-quality synthetic training data. Extensive experimental evaluations demonstrate that the proposed approach significantly outperforms existing baseline methods, achieving at least a 12.8% improvement in span-level emotion-cause-category triplet extraction metrics. The results demonstrate the method's effectiveness and robustness, offering a promising avenue for advancing research in emotion cause analysis. The source code is available at this https URL. 

---
# Speculative Thinking: Enhancing Small-Model Reasoning with Large Model Guidance at Inference Time 

**Authors**: Wang Yang, Xiang Yue, Vipin Chaudhary, Xiaotian Han  

**Link**: [PDF](https://arxiv.org/pdf/2504.12329)  

**Abstract**: Recent advances leverage post-training to enhance model reasoning performance, which typically requires costly training pipelines and still suffers from inefficient, overly lengthy outputs. We introduce Speculative Thinking, a training-free framework that enables large reasoning models to guide smaller ones during inference at the reasoning level, distinct from speculative decoding, which operates at the token level. Our approach is based on two observations: (1) reasoning-supportive tokens such as "wait" frequently appear after structural delimiters like "\n\n", serving as signals for reflection or continuation; and (2) larger models exhibit stronger control over reflective behavior, reducing unnecessary backtracking while improving reasoning quality. By strategically delegating reflective steps to a more capable model, our method significantly boosts the reasoning accuracy of reasoning models while shortening their output. With the assistance of the 32B reasoning model, the 1.5B model's accuracy on MATH500 increases from 83.2% to 89.4%, marking a substantial improvement of 6.2%. Simultaneously, the average output length is reduced from 5439 tokens to 4583 tokens, representing a 15.7% decrease. Moreover, when applied to a non-reasoning model (Qwen-2.5-7B-Instruct), our framework boosts its accuracy from 74.0% to 81.8% on the same benchmark, achieving a relative improvement of 7.8%. 

---
# Integrating Structural and Semantic Signals in Text-Attributed Graphs with BiGTex 

**Authors**: Azadeh Beiranvand, Seyed Mehdi Vahidipour  

**Link**: [PDF](https://arxiv.org/pdf/2504.12474)  

**Abstract**: Text-attributed graphs (TAGs) present unique challenges in representation learning by requiring models to capture both the semantic richness of node-associated texts and the structural dependencies of the graph. While graph neural networks (GNNs) excel at modeling topological information, they lack the capacity to process unstructured text. Conversely, large language models (LLMs) are proficient in text understanding but are typically unaware of graph structure. In this work, we propose BiGTex (Bidirectional Graph Text), a novel architecture that tightly integrates GNNs and LLMs through stacked Graph-Text Fusion Units. Each unit allows for mutual attention between textual and structural representations, enabling information to flow in both directions, text influencing structure and structure guiding textual interpretation. The proposed architecture is trained using parameter-efficient fine-tuning (LoRA), keeping the LLM frozen while adapting to task-specific signals. Extensive experiments on five benchmark datasets demonstrate that BiGTex achieves state-of-the-art performance in node classification and generalizes effectively to link prediction. An ablation study further highlights the importance of soft prompting and bi-directional attention in the model's success. 

---
# A Comprehensive Survey of Reward Models: Taxonomy, Applications, Challenges, and Future 

**Authors**: Jialun Zhong, Wei Shen, Yanzeng Li, Songyang Gao, Hua Lu, Yicheng Chen, Yang Zhang, Wei Zhou, Jinjie Gu, Lei Zou  

**Link**: [PDF](https://arxiv.org/pdf/2504.12328)  

**Abstract**: Reward Model (RM) has demonstrated impressive potential for enhancing Large Language Models (LLM), as RM can serve as a proxy for human preferences, providing signals to guide LLMs' behavior in various tasks. In this paper, we provide a comprehensive overview of relevant research, exploring RMs from the perspectives of preference collection, reward modeling, and usage. Next, we introduce the applications of RMs and discuss the benchmarks for evaluation. Furthermore, we conduct an in-depth analysis of the challenges existing in the field and dive into the potential research directions. This paper is dedicated to providing beginners with a comprehensive introduction to RMs and facilitating future studies. The resources are publicly available at github\footnote{this https URL}. 

---
# Reconstructing Sepsis Trajectories from Clinical Case Reports using LLMs: the Textual Time Series Corpus for Sepsis 

**Authors**: Shahriar Noroozizadeh, Jeremy C. Weiss  

**Link**: [PDF](https://arxiv.org/pdf/2504.12326)  

**Abstract**: Clinical case reports and discharge summaries may be the most complete and accurate summarization of patient encounters, yet they are finalized, i.e., timestamped after the encounter. Complementary data structured streams become available sooner but suffer from incompleteness. To train models and algorithms on more complete and temporally fine-grained data, we construct a pipeline to phenotype, extract, and annotate time-localized findings within case reports using large language models. We apply our pipeline to generate an open-access textual time series corpus for Sepsis-3 comprising 2,139 case reports from the Pubmed-Open Access (PMOA) Subset. To validate our system, we apply it on PMOA and timeline annotations from I2B2/MIMIC-IV and compare the results to physician-expert annotations. We show high recovery rates of clinical findings (event match rates: O1-preview--0.755, Llama 3.3 70B Instruct--0.753) and strong temporal ordering (concordance: O1-preview--0.932, Llama 3.3 70B Instruct--0.932). Our work characterizes the ability of LLMs to time-localize clinical findings in text, illustrating the limitations of LLM use for temporal reconstruction and providing several potential avenues of improvement via multimodal integration. 

---
# A Large-Language Model Framework for Relative Timeline Extraction from PubMed Case Reports 

**Authors**: Jing Wang, Jeremy C Weiss  

**Link**: [PDF](https://arxiv.org/pdf/2504.12350)  

**Abstract**: Timing of clinical events is central to characterization of patient trajectories, enabling analyses such as process tracing, forecasting, and causal reasoning. However, structured electronic health records capture few data elements critical to these tasks, while clinical reports lack temporal localization of events in structured form. We present a system that transforms case reports into textual time series-structured pairs of textual events and timestamps. We contrast manual and large language model (LLM) annotations (n=320 and n=390 respectively) of ten randomly-sampled PubMed open-access (PMOA) case reports (N=152,974) and assess inter-LLM agreement (n=3,103; N=93). We find that the LLM models have moderate event recall(O1-preview: 0.80) but high temporal concordance among identified events (O1-preview: 0.95). By establishing the task, annotation, and assessment systems, and by demonstrating high concordance, this work may serve as a benchmark for leveraging the PMOA corpus for temporal analytics. 

---
# LLMTaxo: Leveraging Large Language Models for Constructing Taxonomy of Factual Claims from Social Media 

**Authors**: Haiqi Zhang, Zhengyuan Zhu, Zeyu Zhang, Chengkai Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.12325)  

**Abstract**: With the vast expansion of content on social media platforms, analyzing and comprehending online discourse has become increasingly complex. This paper introduces LLMTaxo, a novel framework leveraging large language models for the automated construction of taxonomy of factual claims from social media by generating topics from multi-level granularities. This approach aids stakeholders in more effectively navigating the social media landscapes. We implement this framework with different models across three distinct datasets and introduce specially designed taxonomy evaluation metrics for a comprehensive assessment. With the evaluations from both human evaluators and GPT-4, the results indicate that LLMTaxo effectively categorizes factual claims from social media, and reveals that certain models perform better on specific datasets. 

---
# The Other Side of the Coin: Exploring Fairness in Retrieval-Augmented Generation 

**Authors**: Zheng Zhang, Ning Li, Qi Liu, Rui Li, Weibo Gao, Qingyang Mao, Zhenya Huang, Baosheng Yu, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2504.12323)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by retrieving relevant document from external knowledge sources. By referencing this external knowledge, RAG effectively reduces the generation of factually incorrect content and addresses hallucination issues within LLMs. Recently, there has been growing attention to improving the performance and efficiency of RAG systems from various perspectives. While these advancements have yielded significant results, the application of RAG in domains with considerable societal implications raises a critical question about fairness: What impact does the introduction of the RAG paradigm have on the fairness of LLMs? To address this question, we conduct extensive experiments by varying the LLMs, retrievers, and retrieval sources. Our experimental analysis reveals that the scale of the LLMs plays a significant role in influencing fairness outcomes within the RAG framework. When the model scale is smaller than 8B, the integration of retrieval mechanisms often exacerbates unfairness in small-scale LLMs (e.g., LLaMA3.2-1B, Mistral-7B, and LLaMA3-8B). To mitigate the fairness issues introduced by RAG for small-scale LLMs, we propose two approaches, FairFT and FairFilter. Specifically, in FairFT, we align the retriever with the LLM in terms of fairness, enabling it to retrieve documents that facilitate fairer model outputs. In FairFilter, we propose a fairness filtering mechanism to filter out biased content after retrieval. Finally, we validate our proposed approaches on real-world datasets, demonstrating their effectiveness in improving fairness while maintaining performance. 

---
# A Strategic Coordination Framework of Small LLMs Matches Large LLMs in Data Synthesis 

**Authors**: Xin Gao, Qizhi Pei, Zinan Tang, Yu Li, Honglin Lin, Jiang Wu, Conghui He, Lijun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.12322)  

**Abstract**: While data synthesis and distillation are promising strategies to enhance small language models, current approaches heavily rely on Large Language Models (LLMs), which suffer from high computational costs, environmental inefficiency, and potential biases inherited from monolithic architectures. In contrast, smaller LLMs are more accessible and sustainable, but their individual capabilities often fall short in generating high-quality, diverse, and reliable data. Inspired by collaborative human processes (e.g., peer review), we propose a multiple small LLMs involved framework, GRA, that aggregates specialized roles across small LLMs to iterative refinement and quality control typically achieved by a single large LLM. In this collaborative framework, multiple small LLMs assume distinct roles-Generator, Reviewer, and Adjudicator-to simulate a peer-review-inspired data synthesis pipeline. The Generator proposes initial data samples, the Reviewer critiques their quality and diversity, and the Adjudicator resolves conflicts to finalize the output. By decomposing the synthesis process into specialized sub-tasks, collaborative small LLMs can achieve data-level parity with large LLM-based distillation. Through experiments across multiple benchmarks, we demonstrate that GRA-produced data matches or exceeds the quality of single large LLM outputs, e.g., Qwen-2.5-72B-Instruct. Our results challenge the necessity of monolithic large models for high-quality data synthesis, advocating instead for strategic coordination of smaller agents. Our datasets, models, and code are publicly available at this https URL. 

---
# Has the Creativity of Large-Language Models peaked? An analysis of inter- and intra-LLM variability 

**Authors**: Jennifer Haase, Paul H. P. Hanel, Sebastian Pokutta  

**Link**: [PDF](https://arxiv.org/pdf/2504.12320)  

**Abstract**: Following the widespread adoption of ChatGPT in early 2023, numerous studies reported that large language models (LLMs) can match or even surpass human performance in creative tasks. However, it remains unclear whether LLMs have become more creative over time, and how consistent their creative output is. In this study, we evaluated 14 widely used LLMs -- including GPT-4, Claude, Llama, Grok, Mistral, and DeepSeek -- across two validated creativity assessments: the Divergent Association Task (DAT) and the Alternative Uses Task (AUT). Contrary to expectations, we found no evidence of increased creative performance over the past 18-24 months, with GPT-4 performing worse than in previous studies. For the more widely used AUT, all models performed on average better than the average human, with GPT-4o and o3-mini performing best. However, only 0.28% of LLM-generated responses reached the top 10% of human creativity benchmarks. Beyond inter-model differences, we document substantial intra-model variability: the same LLM, given the same prompt, can produce outputs ranging from below-average to original. This variability has important implications for both creativity research and practical applications. Ignoring such variability risks misjudging the creative potential of LLMs, either inflating or underestimating their capabilities. The choice of prompts affected LLMs differently. Our findings underscore the need for more nuanced evaluation frameworks and highlight the importance of model selection, prompt design, and repeated assessment when using Generative AI (GenAI) tools in creative contexts. 

---
# You've Changed: Detecting Modification of Black-Box Large Language Models 

**Authors**: Alden Dima, James Foulds, Shimei Pan, Philip Feldman  

**Link**: [PDF](https://arxiv.org/pdf/2504.12335)  

**Abstract**: Large Language Models (LLMs) are often provided as a service via an API, making it challenging for developers to detect changes in their behavior. We present an approach to monitor LLMs for changes by comparing the distributions of linguistic and psycholinguistic features of generated text. Our method uses a statistical test to determine whether the distributions of features from two samples of text are equivalent, allowing developers to identify when an LLM has changed. We demonstrate the effectiveness of our approach using five OpenAI completion models and Meta's Llama 3 70B chat model. Our results show that simple text features coupled with a statistical test can distinguish between language models. We also explore the use of our approach to detect prompt injection attacks. Our work enables frequent LLM change monitoring and avoids computationally expensive benchmark evaluations. 

---
# Capybara-OMNI: An Efficient Paradigm for Building Omni-Modal Language Models 

**Authors**: Xingguang Ji, Jiakang Wang, Hongzhi Zhang, Jingyuan Zhang, Haonan Zhou, Chenxi Sun, Yahui Liu, Qi Wang, Fuzheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.12315)  

**Abstract**: With the development of Multimodal Large Language Models (MLLMs), numerous outstanding accomplishments have emerged within the open-source community. Due to the complexity of creating and training multimodal data pairs, it is still a computational and time-consuming process to build powerful MLLMs. In this work, we introduce Capybara-OMNI, an MLLM that trains in a lightweight and efficient manner and supports understanding text, image, video, and audio modalities. We present in detail the framework design, the data construction, and the training recipe, to develop an MLLM step-by-step to obtain competitive performance. We also provide exclusive benchmarks utilized in our experiments to show how to properly verify understanding capabilities across different modalities. Results show that by following our guidance, we can efficiently build an MLLM that achieves competitive performance among models of the same scale on various multimodal benchmarks. Additionally, to enhance the multimodal instruction following and conversational capabilities of the model, we further discuss how to train the chat version upon an MLLM understanding model, which is more in line with user habits for tasks like real-time interaction with humans. We publicly disclose the Capybara-OMNI model, along with its chat-based version. The disclosure includes both the model weights, a portion of the training data, and the inference codes, which are made available on GitHub. 

---
# How to Detect and Defeat Molecular Mirage: A Metric-Driven Benchmark for Hallucination in LLM-based Molecular Comprehension 

**Authors**: Hao Li, Liuzhenghao Lv, He Cao, Zijing Liu, Zhiyuan Yan, Yu Wang, Yonghong Tian, Yu Li, Li Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2504.12314)  

**Abstract**: Large language models are increasingly used in scientific domains, especially for molecular understanding and analysis. However, existing models are affected by hallucination issues, resulting in errors in drug design and utilization. In this paper, we first analyze the sources of hallucination in LLMs for molecular comprehension tasks, specifically the knowledge shortcut phenomenon observed in the PubChem dataset. To evaluate hallucination in molecular comprehension tasks with computational efficiency, we introduce \textbf{Mol-Hallu}, a novel free-form evaluation metric that quantifies the degree of hallucination based on the scientific entailment relationship between generated text and actual molecular properties. Utilizing the Mol-Hallu metric, we reassess and analyze the extent of hallucination in various LLMs performing molecular comprehension tasks. Furthermore, the Hallucination Reduction Post-processing stage~(HRPP) is proposed to alleviate molecular hallucinations, Experiments show the effectiveness of HRPP on decoder-only and encoder-decoder molecular LLMs. Our findings provide critical insights into mitigating hallucination and improving the reliability of LLMs in scientific applications. 

---
# Large Language Model-Based Knowledge Graph System Construction for Sustainable Development Goals: An AI-Based Speculative Design Perspective 

**Authors**: Yi-De Lin, Guan-Ze Liao  

**Link**: [PDF](https://arxiv.org/pdf/2504.12309)  

**Abstract**: From 2000 to 2015, the UN's Millennium Development Goals guided global priorities. The subsequent Sustainable Development Goals (SDGs) adopted a more dynamic approach, with annual indicator updates. As 2030 nears and progress lags, innovative acceleration strategies are critical. This study develops an AI-powered knowledge graph system to analyze SDG interconnections, discover potential new goals, and visualize them online. Using official SDG texts, Elsevier's keyword dataset, and 1,127 TED Talk transcripts (2020-2023), a pilot on 269 talks from 2023 applies AI-speculative design, large language models, and retrieval-augmented generation. Key findings include: (1) Heatmap analysis reveals strong associations between Goal 10 and Goal 16, and minimal coverage of Goal 6. (2) In the knowledge graph, simulated dialogue over time reveals new central nodes, showing how richer data supports divergent thinking and goal clarity. (3) Six potential new goals are proposed, centered on equity, resilience, and technology-driven inclusion. This speculative-AI framework offers fresh insights for policymakers and lays groundwork for future multimodal and cross-system SDG applications. 

---
# Meta-Evaluating Local LLMs: Rethinking Performance Metrics for Serious Games 

**Authors**: Andrés Isaza-Giraldo, Paulo Bala, Lucas Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2504.12333)  

**Abstract**: The evaluation of open-ended responses in serious games presents a unique challenge, as correctness is often subjective. Large Language Models (LLMs) are increasingly being explored as evaluators in such contexts, yet their accuracy and consistency remain uncertain, particularly for smaller models intended for local execution. This study investigates the reliability of five small-scale LLMs when assessing player responses in \textit{En-join}, a game that simulates decision-making within energy communities. By leveraging traditional binary classification metrics (including accuracy, true positive rate, and true negative rate), we systematically compare these models across different evaluation scenarios. Our results highlight the strengths and limitations of each model, revealing trade-offs between sensitivity, specificity, and overall performance. We demonstrate that while some models excel at identifying correct responses, others struggle with false positives or inconsistent evaluations. The findings highlight the need for context-aware evaluation frameworks and careful model selection when deploying LLMs as evaluators. This work contributes to the broader discourse on the trustworthiness of AI-driven assessment tools, offering insights into how different LLM architectures handle subjective evaluation tasks. 

---
# HM-RAG: Hierarchical Multi-Agent Multimodal Retrieval Augmented Generation 

**Authors**: Pei Liu, Xin Liu, Ruoyu Yao, Junming Liu, Siyuan Meng, Ding Wang, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.12330)  

**Abstract**: While Retrieval-Augmented Generation (RAG) augments Large Language Models (LLMs) with external knowledge, conventional single-agent RAG remains fundamentally limited in resolving complex queries demanding coordinated reasoning across heterogeneous data ecosystems. We present HM-RAG, a novel Hierarchical Multi-agent Multimodal RAG framework that pioneers collaborative intelligence for dynamic knowledge synthesis across structured, unstructured, and graph-based data. The framework is composed of three-tiered architecture with specialized agents: a Decomposition Agent that dissects complex queries into contextually coherent sub-tasks via semantic-aware query rewriting and schema-guided context augmentation; Multi-source Retrieval Agents that carry out parallel, modality-specific retrieval using plug-and-play modules designed for vector, graph, and web-based databases; and a Decision Agent that uses consistency voting to integrate multi-source answers and resolve discrepancies in retrieval results through Expert Model Refinement. This architecture attains comprehensive query understanding by combining textual, graph-relational, and web-derived evidence, resulting in a remarkable 12.95% improvement in answer accuracy and a 3.56% boost in question classification accuracy over baseline RAG systems on the ScienceQA and CrisisMMD benchmarks. Notably, HM-RAG establishes state-of-the-art results in zero-shot settings on both datasets. Its modular architecture ensures seamless integration of new data modalities while maintaining strict data governance, marking a significant advancement in addressing the critical challenges of multimodal reasoning and knowledge synthesis in RAG systems. Code is available at this https URL. 

---
# ConExion: Concept Extraction with Large Language Models 

**Authors**: Ebrahim Norouzi, Sven Hertling, Harald Sack  

**Link**: [PDF](https://arxiv.org/pdf/2504.12915)  

**Abstract**: In this paper, an approach for concept extraction from documents using pre-trained large language models (LLMs) is presented. Compared with conventional methods that extract keyphrases summarizing the important information discussed in a document, our approach tackles a more challenging task of extracting all present concepts related to the specific domain, not just the important ones. Through comprehensive evaluations of two widely used benchmark datasets, we demonstrate that our method improves the F1 score compared to state-of-the-art techniques. Additionally, we explore the potential of using prompts within these models for unsupervised concept extraction. The extracted concepts are intended to support domain coverage evaluation of ontologies and facilitate ontology learning, highlighting the effectiveness of LLMs in concept extraction tasks. Our source code and datasets are publicly available at this https URL. 

---
# Estimating Optimal Context Length for Hybrid Retrieval-augmented Multi-document Summarization 

**Authors**: Adithya Pratapa, Teruko Mitamura  

**Link**: [PDF](https://arxiv.org/pdf/2504.12972)  

**Abstract**: Recent advances in long-context reasoning abilities of language models led to interesting applications in large-scale multi-document summarization. However, prior work has shown that these long-context models are not effective at their claimed context windows. To this end, retrieval-augmented systems provide an efficient and effective alternative. However, their performance can be highly sensitive to the choice of retrieval context length. In this work, we present a hybrid method that combines retrieval-augmented systems with long-context windows supported by recent language models. Our method first estimates the optimal retrieval length as a function of the retriever, summarizer, and dataset. On a randomly sampled subset of the dataset, we use a panel of LLMs to generate a pool of silver references. We use these silver references to estimate the optimal context length for a given RAG system configuration. Our results on the multi-document summarization task showcase the effectiveness of our method across model classes and sizes. We compare against length estimates from strong long-context benchmarks such as RULER and HELMET. Our analysis also highlights the effectiveness of our estimation method for very long-context LMs and its generalization to new classes of LMs. 

---
# MAIN: Mutual Alignment Is Necessary for instruction tuning 

**Authors**: Fanyi Yang, Jianfeng Liu, Xin Zhang, Haoyu Liu, Xixin Cao, Yuefeng Zhan, Hao Sun, Weiwei Deng, Feng Sun, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.12913)  

**Abstract**: Instruction tuning has enabled large language models (LLMs) to achieve remarkable performance, but its success heavily depends on the availability of large-scale, high-quality instruction-response pairs. However, current methods for scaling up data generation often overlook a crucial aspect: the alignment between instructions and responses. We hypothesize that high-quality instruction-response pairs are not defined by the individual quality of each component, but by the extent of their alignment with each other. To address this, we propose a Mutual Alignment Framework (MAIN) that ensures coherence between the instruction and response through mutual constraints. Experiments demonstrate that models such as LLaMA and Mistral, fine-tuned within this framework, outperform traditional methods across multiple benchmarks. This approach underscores the critical role of instruction-response alignment in enabling scalable and high-quality instruction tuning for LLMs. 

---
# Assesing LLMs in Art Contexts: Critique Generation and Theory of Mind Evaluation 

**Authors**: Takaya Arita, Wenxian Zheng, Reiji Suzuki, Fuminori Akiba  

**Link**: [PDF](https://arxiv.org/pdf/2504.12805)  

**Abstract**: This study explored how large language models (LLMs) perform in two areas related to art: writing critiques of artworks and reasoning about mental states (Theory of Mind, or ToM) in art-related situations. For the critique generation part, we built a system that combines Noel Carroll's evaluative framework with a broad selection of art criticism theories. The model was prompted to first write a full-length critique and then shorter, more coherent versions using a step-by-step prompting process. These AI-generated critiques were then compared with those written by human experts in a Turing test-style evaluation. In many cases, human subjects had difficulty telling which was which, and the results suggest that LLMs can produce critiques that are not only plausible in style but also rich in interpretation, as long as they are carefully guided. In the second part, we introduced new simple ToM tasks based on situations involving interpretation, emotion, and moral tension, which can appear in the context of art. These go beyond standard false-belief tests and allow for more complex, socially embedded forms of reasoning. We tested 41 recent LLMs and found that their performance varied across tasks and models. In particular, tasks that involved affective or ambiguous situations tended to reveal clearer differences. Taken together, these results help clarify how LLMs respond to complex interpretative challenges, revealing both their cognitive limitations and potential. While our findings do not directly contradict the so-called Generative AI Paradox--the idea that LLMs can produce expert-like output without genuine understanding--they suggest that, depending on how LLMs are instructed, such as through carefully designed prompts, these models may begin to show behaviors that resemble understanding more closely than we might assume. 

---
# Chinese-Vicuna: A Chinese Instruction-following Llama-based Model 

**Authors**: Chenghao Fan, Zhenyi Lu, Jie Tian  

**Link**: [PDF](https://arxiv.org/pdf/2504.12737)  

**Abstract**: Chinese-Vicuna is an open-source, resource-efficient language model designed to bridge the gap in Chinese instruction-following capabilities by fine-tuning Meta's LLaMA architecture using Low-Rank Adaptation (LoRA). Targeting low-resource environments, it enables cost-effective deployment on consumer GPUs (e.g., RTX-2080Ti for 7B models) and supports domain-specific adaptation in fields like healthcare and law. By integrating hybrid datasets (BELLE and Guanaco) and 4-bit quantization (QLoRA), the model achieves competitive performance in tasks such as translation, code generation, and domain-specific Q\&A. The project provides a comprehensive toolkit for model conversion, CPU inference, and multi-turn dialogue interfaces, emphasizing accessibility for researchers and developers. Evaluations indicate competitive performance across medical tasks, multi-turn dialogue coherence, and real-time legal updates. Chinese-Vicuna's modular design, open-source ecosystem, and community-driven enhancements position it as a versatile foundation for Chinese LLM applications. 

---
# Can LLMs reason over extended multilingual contexts? Towards long-context evaluation beyond retrieval and haystacks 

**Authors**: Amey Hengle, Prasoon Bajpai, Soham Dan, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2504.12845)  

**Abstract**: Existing multilingual long-context benchmarks, often based on the popular needle-in-a-haystack test, primarily evaluate a model's ability to locate specific information buried within irrelevant texts. However, such a retrieval-centric approach is myopic and inherently limited, as successful recall alone does not indicate a model's capacity to reason over extended contexts. Moreover, these benchmarks are susceptible to data leakage, short-circuiting, and risk making the evaluation a priori identifiable. To address these limitations, we introduce MLRBench, a new synthetic benchmark for multilingual long-context reasoning. Unlike existing benchmarks, MLRBench goes beyond surface-level retrieval by including tasks that assess multi-hop inference, aggregation, and epistemic reasoning. Spanning seven languages, MLRBench is designed to be parallel, resistant to leakage, and scalable to arbitrary context lengths. Our extensive experiments with an open-weight large language model (LLM) reveal a pronounced gap between high- and low-resource languages, particularly for tasks requiring the model to aggregate multiple facts or predict the absence of information. We also find that, in multilingual settings, LLMs effectively utilize less than 30% of their claimed context length. Although off-the-shelf Retrieval Augmented Generation helps alleviate this to a certain extent, it does not solve the long-context problem. We open-source MLRBench to enable future research in improved evaluation and training of multilingual LLMs. 

---
# Why and How LLMs Hallucinate: Connecting the Dots with Subsequence Associations 

**Authors**: Yiyou Sun, Yu Gai, Lijie Chen, Abhilasha Ravichander, Yejin Choi, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2504.12691)  

**Abstract**: Large language models (LLMs) frequently generate hallucinations-content that deviates from factual accuracy or provided context-posing challenges for diagnosis due to the complex interplay of underlying causes. This paper introduces a subsequence association framework to systematically trace and understand hallucinations. Our key insight is that hallucinations arise when dominant hallucinatory associations outweigh faithful ones. Through theoretical and empirical analyses, we demonstrate that decoder-only transformers effectively function as subsequence embedding models, with linear layers encoding input-output associations. We propose a tracing algorithm that identifies causal subsequences by analyzing hallucination probabilities across randomized input contexts. Experiments show our method outperforms standard attribution techniques in identifying hallucination causes and aligns with evidence from the model's training corpus. This work provides a unified perspective on hallucinations and a robust framework for their tracing and analysis. 

---
# Data-efficient LLM Fine-tuning for Code Generation 

**Authors**: Weijie Lv, Xuan Xia, Sheng-Jun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.12687)  

**Abstract**: Large language models (LLMs) have demonstrated significant potential in code generation tasks. However, there remains a performance gap between open-source and closed-source models. To address this gap, existing approaches typically generate large amounts of synthetic data for fine-tuning, which often leads to inefficient training. In this work, we propose a data selection strategy in order to improve the effectiveness and efficiency of training for code-based LLMs. By prioritizing data complexity and ensuring that the sampled subset aligns with the distribution of the original dataset, our sampling strategy effectively selects high-quality data. Additionally, we optimize the tokenization process through a "dynamic pack" technique, which minimizes padding tokens and reduces computational resource consumption. Experimental results show that when training on 40% of the OSS-Instruct dataset, the DeepSeek-Coder-Base-6.7B model achieves an average performance of 66.9%, surpassing the 66.1% performance with the full dataset. Moreover, training time is reduced from 47 minutes to 34 minutes, and the peak GPU memory decreases from 61.47 GB to 42.72 GB during a single epoch. Similar improvements are observed with the CodeLlama-Python-7B model on the Evol-Instruct dataset. By optimizing both data selection and tokenization, our approach not only improves model performance but also improves training efficiency. 

---
# Sparks of Science: Hypothesis Generation Using Structured Paper Data 

**Authors**: Charles O'Neill, Tirthankar Ghosal, Roberta Răileanu, Mike Walmsley, Thang Bui, Kevin Schawinski, Ioana Ciucă  

**Link**: [PDF](https://arxiv.org/pdf/2504.12976)  

**Abstract**: Generating novel and creative scientific hypotheses is a cornerstone in achieving Artificial General Intelligence. Large language and reasoning models have the potential to aid in the systematic creation, selection, and validation of scientifically informed hypotheses. However, current foundation models often struggle to produce scientific ideas that are both novel and feasible. One reason is the lack of a dedicated dataset that frames Scientific Hypothesis Generation (SHG) as a Natural Language Generation (NLG) task. In this paper, we introduce HypoGen, the first dataset of approximately 5500 structured problem-hypothesis pairs extracted from top-tier computer science conferences structured with a Bit-Flip-Spark schema, where the Bit is the conventional assumption, the Spark is the key insight or conceptual leap, and the Flip is the resulting counterproposal. HypoGen uniquely integrates an explicit Chain-of-Reasoning component that reflects the intellectual process from Bit to Flip. We demonstrate that framing hypothesis generation as conditional language modelling, with the model fine-tuned on Bit-Flip-Spark and the Chain-of-Reasoning (and where, at inference, we only provide the Bit), leads to improvements in the overall quality of the hypotheses. Our evaluation employs automated metrics and LLM judge rankings for overall quality assessment. We show that by fine-tuning on our HypoGen dataset we improve the novelty, feasibility, and overall quality of the generated hypotheses. The HypoGen dataset is publicly available at this http URL. 

---
# Towards Characterizing Subjectivity of Individuals through Modeling Value Conflicts and Trade-offs 

**Authors**: Younghun Lee, Dan Goldwasser  

**Link**: [PDF](https://arxiv.org/pdf/2504.12633)  

**Abstract**: Large Language Models (LLMs) not only have solved complex reasoning problems but also exhibit remarkable performance in tasks that require subjective decision making. Existing studies suggest that LLM generations can be subjectively grounded to some extent, yet exploring whether LLMs can account for individual-level subjectivity has not been sufficiently studied. In this paper, we characterize subjectivity of individuals on social media and infer their moral judgments using LLMs. We propose a framework, SOLAR (Subjective Ground with Value Abstraction), that observes value conflicts and trade-offs in the user-generated texts to better represent subjective ground of individuals. Empirical results show that our framework improves overall inference results as well as performance on controversial situations. Additionally, we qualitatively show that SOLAR provides explanations about individuals' value preferences, which can further account for their judgments. 

---
# CDF-RAG: Causal Dynamic Feedback for Adaptive Retrieval-Augmented Generation 

**Authors**: Elahe Khatibi, Ziyu Wang, Amir M. Rahmani  

**Link**: [PDF](https://arxiv.org/pdf/2504.12560)  

**Abstract**: Retrieval-Augmented Generation (RAG) has significantly enhanced large language models (LLMs) in knowledge-intensive tasks by incorporating external knowledge retrieval. However, existing RAG frameworks primarily rely on semantic similarity and correlation-driven retrieval, limiting their ability to distinguish true causal relationships from spurious associations. This results in responses that may be factually grounded but fail to establish cause-and-effect mechanisms, leading to incomplete or misleading insights. To address this issue, we introduce Causal Dynamic Feedback for Adaptive Retrieval-Augmented Generation (CDF-RAG), a framework designed to improve causal consistency, factual accuracy, and explainability in generative reasoning. CDF-RAG iteratively refines queries, retrieves structured causal graphs, and enables multi-hop causal reasoning across interconnected knowledge sources. Additionally, it validates responses against causal pathways, ensuring logically coherent and factually grounded outputs. We evaluate CDF-RAG on four diverse datasets, demonstrating its ability to improve response accuracy and causal correctness over existing RAG-based methods. Our code is publicly available at this https URL elakhatibi/CDF-RAG. 

---
# GeoSense: Evaluating Identification and Application of Geometric Principles in Multimodal Reasoning 

**Authors**: Liangyu Xu, Yingxiu Zhao, Jingyun Wang, Yingyao Wang, Bu Pi, Chen Wang, Mingliang Zhang, Jihao Gu, Xiang Li, Xiaoyong Zhu, Jun Song, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.12597)  

**Abstract**: Geometry problem-solving (GPS), a challenging task requiring both visual comprehension and symbolic reasoning, effectively measures the reasoning capabilities of multimodal large language models (MLLMs). Humans exhibit strong reasoning ability in this task through accurate identification and adaptive application of geometric principles within visual contexts. However, existing benchmarks fail to jointly assess both dimensions of the human-like geometric reasoning mechanism in MLLMs, remaining a critical gap in assessing their ability to tackle GPS. To this end, we introduce GeoSense, the first comprehensive bilingual benchmark designed to systematically evaluate the geometric reasoning abilities of MLLMs through the lens of geometric principles. GeoSense features a five-level hierarchical framework of geometric principles spanning plane and solid geometry, an intricately annotated dataset of 1,789 problems, and an innovative evaluation strategy. Through extensive experiments on GeoSense with various open-source and closed-source MLLMs, we observe that Gemini-2.0-pro-flash performs best, achieving an overall score of $65.3$. Our in-depth analysis reveals that the identification and application of geometric principles remain a bottleneck for leading MLLMs, jointly hindering their reasoning abilities. These findings underscore GeoSense's potential to guide future advancements in MLLMs' geometric reasoning capabilities, paving the way for more robust and human-like reasoning in artificial intelligence. 

---
# ChatEXAONEPath: An Expert-level Multimodal Large Language Model for Histopathology Using Whole Slide Images 

**Authors**: Sangwook Kim, Soonyoung Lee, Jongseong Jang  

**Link**: [PDF](https://arxiv.org/pdf/2504.13023)  

**Abstract**: Recent studies have made significant progress in developing large language models (LLMs) in the medical domain, which can answer expert-level questions and demonstrate the potential to assist clinicians in real-world clinical scenarios. Studies have also witnessed the importance of integrating various modalities with the existing LLMs for a better understanding of complex clinical contexts, which are innately multi-faceted by nature. Although studies have demonstrated the ability of multimodal LLMs in histopathology to answer questions from given images, they lack in understanding of thorough clinical context due to the patch-level data with limited information from public datasets. Thus, developing WSI-level MLLMs is significant in terms of the scalability and applicability of MLLMs in histopathology. In this study, we introduce an expert-level MLLM for histopathology using WSIs, dubbed as ChatEXAONEPath. We present a retrieval-based data generation pipeline using 10,094 pairs of WSIs and histopathology reports from The Cancer Genome Atlas (TCGA). We also showcase an AI-based evaluation protocol for a comprehensive understanding of the medical context from given multimodal information and evaluate generated answers compared to the original histopathology reports. We demonstrate the ability of diagnosing the given histopathology images using ChatEXAONEPath with the acceptance rate of 62.9% from 1,134 pairs of WSIs and reports. Our proposed model can understand pan-cancer WSIs and clinical context from various cancer types. We argue that our proposed model has the potential to assist clinicians by comprehensively understanding complex morphology of WSIs for cancer diagnosis through the integration of multiple modalities. 

---
# Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs? 

**Authors**: Hansi Zeng, Kai Hui, Honglei Zhuang, Zhen Qin, Zhenrui Yue, Hamed Zamani, Dana Alon  

**Link**: [PDF](https://arxiv.org/pdf/2504.12491)  

**Abstract**: While metrics available during pre-training, such as perplexity, correlate well with model performance at scaling-laws studies, their predictive capacities at a fixed model size remain unclear, hindering effective model selection and development. To address this gap, we formulate the task of selecting pre-training checkpoints to maximize downstream fine-tuning performance as a pairwise classification problem: predicting which of two LLMs, differing in their pre-training, will perform better after supervised fine-tuning (SFT). We construct a dataset using 50 1B parameter LLM variants with systematically varied pre-training configurations, e.g., objectives or data, and evaluate them on diverse downstream tasks after SFT. We first conduct a study and demonstrate that the conventional perplexity is a misleading indicator. As such, we introduce novel unsupervised and supervised proxy metrics derived from pre-training that successfully reduce the relative performance prediction error rate by over 50%. Despite the inherent complexity of this task, we demonstrate the practical utility of our proposed proxies in specific scenarios, paving the way for more efficient design of pre-training schemes optimized for various downstream tasks. 

---
# SLURG: Investigating the Feasibility of Generating Synthetic Online Fallacious Discourse 

**Authors**: Cal Blanco, Gavin Dsouza, Hugo Lin, Chelsey Rush  

**Link**: [PDF](https://arxiv.org/pdf/2504.12466)  

**Abstract**: In our paper we explore the definition, and extrapolation of fallacies as they pertain to the automatic detection of manipulation on social media. In particular we explore how these logical fallacies might appear in the real world i.e internet forums. We discovered a prevalence of misinformation / misguided intention in discussion boards specifically centered around the Ukrainian Russian Conflict which serves to narrow the domain of our task. Although automatic fallacy detection has gained attention recently, most datasets use unregulated fallacy taxonomies or are limited to formal linguistic domains like political debates or news reports. Online discourse, however, often features non-standardized and diverse language not captured in these domains. We present Shady Linguistic Utterance Replication-Generation (SLURG) to address these limitations, exploring the feasibility of generating synthetic fallacious forum-style comments using large language models (LLMs), specifically DeepHermes-3-Mistral-24B. Our findings indicate that LLMs can replicate the syntactic patterns of real data} and that high-quality few-shot prompts enhance LLMs' ability to mimic the vocabulary diversity of online forums. 

---
# Replicating ReLM Results: Validating Large Language Models with ReLM 

**Authors**: Reece Adamson, Erin Song  

**Link**: [PDF](https://arxiv.org/pdf/2504.12357)  

**Abstract**: Validating Large Language Models with ReLM explores the application of formal languages to evaluate and control Large Language Models (LLMs) for memorization, bias, and zero-shot performance. Current approaches for evaluating these types behavior are often slow, imprecise, costly, or introduce biases of their own, but are necessary due to the importance of this behavior when productionizing LLMs. This project reproduces key results from the original ReLM paper and expounds on the approach and applications with an emphasis on the relevance to the field of systems for machine learning. 

---
# Reimagining Urban Science: Scaling Causal Inference with Large Language Models 

**Authors**: Yutong Xia, Ao Qu, Yunhan Zheng, Yihong Tang, Dingyi Zhuang, Yuxuan Liang, Cathy Wu, Roger Zimmermann, Jinhua Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.12345)  

**Abstract**: Urban causal research is essential for understanding the complex dynamics of cities and informing evidence-based policies. However, it is challenged by the inefficiency and bias of hypothesis generation, barriers to multimodal data complexity, and the methodological fragility of causal experimentation. Recent advances in large language models (LLMs) present an opportunity to rethink how urban causal analysis is conducted. This Perspective examines current urban causal research by analyzing taxonomies that categorize research topics, data sources, and methodological approaches to identify structural gaps. We then introduce an LLM-driven conceptual framework, AutoUrbanCI, composed of four distinct modular agents responsible for hypothesis generation, data engineering, experiment design and execution, and results interpretation with policy recommendations. We propose evaluation criteria for rigor and transparency and reflect on implications for human-AI collaboration, equity, and accountability. We call for a new research agenda that embraces AI-augmented workflows not as replacements for human expertise but as tools to broaden participation, improve reproducibility, and unlock more inclusive forms of urban causal reasoning. 

---
# Propaganda via AI? A Study on Semantic Backdoors in Large Language Models 

**Authors**: Nay Myat Min, Long H. Pham, Yige Li, Jun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.12344)  

**Abstract**: Large language models (LLMs) demonstrate remarkable performance across myriad language tasks, yet they remain vulnerable to backdoor attacks, where adversaries implant hidden triggers that systematically manipulate model outputs. Traditional defenses focus on explicit token-level anomalies and therefore overlook semantic backdoors-covert triggers embedded at the conceptual level (e.g., ideological stances or cultural references) that rely on meaning-based cues rather than lexical oddities. We first show, in a controlled finetuning setting, that such semantic backdoors can be implanted with only a small poisoned corpus, establishing their practical feasibility. We then formalize the notion of semantic backdoors in LLMs and introduce a black-box detection framework, RAVEN (short for "Response Anomaly Vigilance for uncovering semantic backdoors"), which combines semantic entropy with cross-model consistency analysis. The framework probes multiple models with structured topic-perspective prompts, clusters the sampled responses via bidirectional entailment, and flags anomalously uniform outputs; cross-model comparison isolates model-specific anomalies from corpus-wide biases. Empirical evaluations across diverse LLM families (GPT-4o, Llama, DeepSeek, Mistral) uncover previously undetected semantic backdoors, providing the first proof-of-concept evidence of these hidden vulnerabilities and underscoring the urgent need for concept-level auditing of deployed language models. We open-source our code and data at this https URL. 

---
# Benchmarking Biopharmaceuticals Retrieval-Augmented Generation Evaluation 

**Authors**: Hanmeng Zhong, Linqing Chen, Weilei Wang, Wentao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.12342)  

**Abstract**: Recently, the application of the retrieval-augmented Large Language Models (LLMs) in specific domains has gained significant attention, especially in biopharmaceuticals. However, in this context, there is no benchmark specifically designed for biopharmaceuticals to evaluate LLMs. In this paper, we introduce the Biopharmaceuticals Retrieval-Augmented Generation Evaluation (BRAGE) , the first benchmark tailored for evaluating LLMs' Query and Reference Understanding Capability (QRUC) in the biopharmaceutical domain, available in English, French, German and Chinese. In addition, Traditional Question-Answering (QA) metrics like accuracy and exact match fall short in the open-ended retrieval-augmented QA scenarios. To address this, we propose a citation-based classification method to evaluate the QRUC of LLMs to understand the relationship between queries and references. We apply this method to evaluate the mainstream LLMs on BRAGE. Experimental results show that there is a significant gap in the biopharmaceutical QRUC of mainstream LLMs, and their QRUC needs to be improved. 

---
# GOAT-TTS: LLM-based Text-To-Speech Generation Optimized via A Dual-Branch Architecture 

**Authors**: Yaodong Song, Hongjie Chen, Jie Lian, Yuxin Zhang, Guangmin Xia, Zehan Li, Genliang Zhao, Jian Kang, Yongxiang Li, Jie Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.12339)  

**Abstract**: While large language models (LLMs) have revolutionized text-to-speech (TTS) synthesis through discrete tokenization paradigms, current architectures exhibit fundamental tensions between three critical dimensions: 1) irreversible loss of acoustic characteristics caused by quantization of speech prompts; 2) stringent dependence on precisely aligned prompt speech-text pairs that limit real-world deployment; and 3) catastrophic forgetting of the LLM's native text comprehension during optimization for speech token generation. To address these challenges, we propose an LLM-based text-to-speech Generation approach Optimized via a novel dual-branch ArchiTecture (GOAT-TTS). Our framework introduces two key innovations: (1) The modality-alignment branch combines a speech encoder and projector to capture continuous acoustic embeddings, enabling bidirectional correlation between paralinguistic features (language, timbre, emotion) and semantic text representations without transcript dependency; (2) The speech-generation branch employs modular fine-tuning on top-k layers of an LLM for speech token prediction while freezing the bottom-k layers to preserve foundational linguistic knowledge. Moreover, multi-token prediction is introduced to support real-time streaming TTS synthesis. Experimental results demonstrate that our GOAT-TTS achieves performance comparable to state-of-the-art TTS models while validating the efficacy of synthesized dialect speech data. 

---
# Paging Dr. GPT: Extracting Information from Clinical Notes to Enhance Patient Predictions 

**Authors**: David Anderson, Michaela Anderson, Margret Bjarnadottir, Stephen Mahar, Shriyan Reyya  

**Link**: [PDF](https://arxiv.org/pdf/2504.12338)  

**Abstract**: There is a long history of building predictive models in healthcare using tabular data from electronic medical records. However, these models fail to extract the information found in unstructured clinical notes, which document diagnosis, treatment, progress, medications, and care plans. In this study, we investigate how answers generated by GPT-4o-mini (ChatGPT) to simple clinical questions about patients, when given access to the patient's discharge summary, can support patient-level mortality prediction. Using data from 14,011 first-time admissions to the Coronary Care or Cardiovascular Intensive Care Units in the MIMIC-IV Note dataset, we implement a transparent framework that uses GPT responses as input features in logistic regression models. Our findings demonstrate that GPT-based models alone can outperform models trained on standard tabular data, and that combining both sources of information yields even greater predictive power, increasing AUC by an average of 5.1 percentage points and increasing positive predictive value by 29.9 percent for the highest-risk decile. These results highlight the value of integrating large language models (LLMs) into clinical prediction tasks and underscore the broader potential for using LLMs in any domain where unstructured text data remains an underutilized resource. 

---
# Can the capability of Large Language Models be described by human ability? A Meta Study 

**Authors**: Mingrui Zan, Yunquan Zhang, Boyang Zhang, Fangming Liu, Daning Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.12332)  

**Abstract**: Users of Large Language Models (LLMs) often perceive these models as intelligent entities with human-like capabilities. However, the extent to which LLMs' capabilities truly approximate human abilities remains a topic of debate. In this paper, to characterize the capabilities of LLMs in relation to human capabilities, we collected performance data from over 80 models across 37 evaluation benchmarks. The evaluation benchmarks are categorized into 6 primary abilities and 11 sub-abilities in human aspect. Then, we then clustered the performance rankings into several categories and compared these clustering results with classifications based on human ability aspects. Our findings lead to the following conclusions: 1. We have confirmed that certain capabilities of LLMs with fewer than 10 billion parameters can indeed be described using human ability metrics; 2. While some abilities are considered interrelated in humans, they appear nearly uncorrelated in LLMs; 3. The capabilities possessed by LLMs vary significantly with the parameter scale of the model. 

---
# ChatGPT as Linguistic Equalizer? Quantifying LLM-Driven Lexical Shifts in Academic Writing 

**Authors**: Dingkang Lin, Naixuan Zhao, Dan Tian, Jiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.12317)  

**Abstract**: The advent of ChatGPT has profoundly reshaped scientific research practices, particularly in academic writing, where non-native English-speakers (NNES) historically face linguistic barriers. This study investigates whether ChatGPT mitigates these barriers and fosters equity by analyzing lexical complexity shifts across 2.8 million articles from OpenAlex (2020-2024). Using the Measure of Textual Lexical Diversity (MTLD) to quantify vocabulary sophistication and a difference-in-differences (DID) design to identify causal effects, we demonstrate that ChatGPT significantly enhances lexical complexity in NNES-authored abstracts, even after controlling for article-level controls, authorship patterns, and venue norms. Notably, the impact is most pronounced in preprint papers, technology- and biology-related fields and lower-tier journals. These findings provide causal evidence that ChatGPT reduces linguistic disparities and promotes equity in global academia. 

---
# Socrates or Smartypants: Testing Logic Reasoning Capabilities of Large Language Models with Logic Programming-based Test Oracles 

**Authors**: Zihao Xu, Junchen Ding, Yiling Lou, Kun Zhang, Dong Gong, Yuekang Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.12312)  

**Abstract**: Large Language Models (LLMs) have achieved significant progress in language understanding and reasoning. Evaluating and analyzing their logical reasoning abilities has therefore become essential. However, existing datasets and benchmarks are often limited to overly simplistic, unnatural, or contextually constrained examples. In response to the growing demand, we introduce SmartyPat-Bench, a challenging, naturally expressed, and systematically labeled benchmark derived from real-world high-quality Reddit posts containing subtle logical fallacies. Unlike existing datasets and benchmarks, it provides more detailed annotations of logical fallacies and features more diverse data. To further scale up the study and address the limitations of manual data collection and labeling - such as fallacy-type imbalance and labor-intensive annotation - we introduce SmartyPat, an automated framework powered by logic programming-based oracles. SmartyPat utilizes Prolog rules to systematically generate logically fallacious statements, which are then refined into fluent natural-language sentences by LLMs, ensuring precise fallacy representation. Extensive evaluation demonstrates that SmartyPat produces fallacies comparable in subtlety and quality to human-generated content and significantly outperforms baseline methods. Finally, experiments reveal nuanced insights into LLM capabilities, highlighting that while excessive reasoning steps hinder fallacy detection accuracy, structured reasoning enhances fallacy categorization performance. 

---
# SemCORE: A Semantic-Enhanced Generative Cross-Modal Retrieval Framework with MLLMs 

**Authors**: Haoxuan Li, Yi Bin, Yunshan Ma, Guoqing Wang, Yang Yang, See-Kiong Ng, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2504.13172)  

**Abstract**: Cross-modal retrieval (CMR) is a fundamental task in multimedia research, focused on retrieving semantically relevant targets across different modalities. While traditional CMR methods match text and image via embedding-based similarity calculations, recent advancements in pre-trained generative models have established generative retrieval as a promising alternative. This paradigm assigns each target a unique identifier and leverages a generative model to directly predict identifiers corresponding to input queries without explicit indexing. Despite its great potential, current generative CMR approaches still face semantic information insufficiency in both identifier construction and generation processes. To address these limitations, we propose a novel unified Semantic-enhanced generative Cross-mOdal REtrieval framework (SemCORE), designed to unleash the semantic understanding capabilities in generative cross-modal retrieval task. Specifically, we first construct a Structured natural language IDentifier (SID) that effectively aligns target identifiers with generative models optimized for natural language comprehension and generation. Furthermore, we introduce a Generative Semantic Verification (GSV) strategy enabling fine-grained target discrimination. Additionally, to the best of our knowledge, SemCORE is the first framework to simultaneously consider both text-to-image and image-to-text retrieval tasks within generative cross-modal retrieval. Extensive experiments demonstrate that our framework outperforms state-of-the-art generative cross-modal retrieval methods. Notably, SemCORE achieves substantial improvements across benchmark datasets, with an average increase of 8.65 points in Recall@1 for text-to-image retrieval. 

---
# Exploring the Impact of Personality Traits on Conversational Recommender Systems: A Simulation with Large Language Models 

**Authors**: Xiaoyan Zhao, Yang Deng, Wenjie Wang, Hongzhan lin, Hong Cheng, Rui Zhang, See-Kiong Ng, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2504.12313)  

**Abstract**: Conversational Recommender Systems (CRSs) engage users in multi-turn interactions to deliver personalized recommendations. The emergence of large language models (LLMs) further enhances these systems by enabling more natural and dynamic user interactions. However, a key challenge remains in understanding how personality traits shape conversational recommendation outcomes. Psychological evidence highlights the influence of personality traits on user interaction behaviors. To address this, we introduce an LLM-based personality-aware user simulation for CRSs (PerCRS). The user agent induces customizable personality traits and preferences, while the system agent possesses the persuasion capability to simulate realistic interaction in CRSs. We incorporate multi-aspect evaluation to ensure robustness and conduct extensive analysis from both user and system perspectives. Experimental results demonstrate that state-of-the-art LLMs can effectively generate diverse user responses aligned with specified personality traits, thereby prompting CRSs to dynamically adjust their recommendation strategies. Our experimental analysis offers empirical insights into the impact of personality traits on the outcomes of conversational recommender systems. 

---
# How Large Language Models Are Changing MOOC Essay Answers: A Comparison of Pre- and Post-LLM Responses 

**Authors**: Leo Leppänen, Lili Aunimo, Arto Hellas, Jukka K. Nurminen, Linda Mannila  

**Link**: [PDF](https://arxiv.org/pdf/2504.13038)  

**Abstract**: The release of ChatGPT in late 2022 caused a flurry of activity and concern in the academic and educational communities. Some see the tool's ability to generate human-like text that passes at least cursory inspections for factual accuracy ``often enough'' a golden age of information retrieval and computer-assisted learning. Some, on the other hand, worry the tool may lead to unprecedented levels of academic dishonesty and cheating. In this work, we quantify some of the effects of the emergence of Large Language Models (LLMs) on online education by analyzing a multi-year dataset of student essay responses from a free university-level MOOC on AI ethics. Our dataset includes essays submitted both before and after ChatGPT's release. We find that the launch of ChatGPT coincided with significant changes in both the length and style of student essays, mirroring observations in other contexts such as academic publishing. We also observe -- as expected based on related public discourse -- changes in prevalence of key content words related to AI and LLMs, but not necessarily the general themes or topics discussed in the student essays as identified through (dynamic) topic modeling. 

---
# Benchmarking LLM-based Relevance Judgment Methods 

**Authors**: Negar Arabzadeh, Charles L. A. Clarke  

**Link**: [PDF](https://arxiv.org/pdf/2504.12558)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in both academic and industry settings to automate the evaluation of information seeking systems, particularly by generating graded relevance judgments. Previous work on LLM-based relevance assessment has primarily focused on replicating graded human relevance judgments through various prompting strategies. However, there has been limited exploration of alternative assessment methods or comprehensive comparative studies. In this paper, we systematically compare multiple LLM-based relevance assessment methods, including binary relevance judgments, graded relevance assessments, pairwise preference-based methods, and two nugget-based evaluation methods~--~document-agnostic and document-dependent. In addition to a traditional comparison based on system rankings using Kendall correlations, we also examine how well LLM judgments align with human preferences, as inferred from relevance grades. We conduct extensive experiments on datasets from three TREC Deep Learning tracks 2019, 2020 and 2021 as well as the ANTIQUE dataset, which focuses on non-factoid open-domain question answering. As part of our data release, we include relevance judgments generated by both an open-source (Llama3.2b) and a commercial (gpt-4o) model. Our goal is to \textit{reproduce} various LLM-based relevance judgment methods to provide a comprehensive comparison. All code, data, and resources are publicly available in our GitHub Repository at this https URL. 

---
# A Human-AI Comparative Analysis of Prompt Sensitivity in LLM-Based Relevance Judgment 

**Authors**: Negar Arabzadeh, Charles L. A . Clarke  

**Link**: [PDF](https://arxiv.org/pdf/2504.12408)  

**Abstract**: Large Language Models (LLMs) are increasingly used to automate relevance judgments for information retrieval (IR) tasks, often demonstrating agreement with human labels that approaches inter-human agreement. To assess the robustness and reliability of LLM-based relevance judgments, we systematically investigate impact of prompt sensitivity on the task. We collected prompts for relevance assessment from 15 human experts and 15 LLMs across three tasks~ -- ~binary, graded, and pairwise~ -- ~yielding 90 prompts in total. After filtering out unusable prompts from three humans and three LLMs, we employed the remaining 72 prompts with three different LLMs as judges to label document/query pairs from two TREC Deep Learning Datasets (2020 and 2021). We compare LLM-generated labels with TREC official human labels using Cohen's $\kappa$ and pairwise agreement measures. In addition to investigating the impact of prompt variations on agreement with human labels, we compare human- and LLM-generated prompts and analyze differences among different LLMs as judges. We also compare human- and LLM-generated prompts with the standard UMBRELA prompt used for relevance assessment by Bing and TREC 2024 Retrieval Augmented Generation (RAG) Track. To support future research in LLM-based evaluation, we release all data and prompts at this https URL. 

---
# Validating LLM-Generated Relevance Labels for Educational Resource Search 

**Authors**: Ratan J. Sebastian, Anett Hoppe  

**Link**: [PDF](https://arxiv.org/pdf/2504.12732)  

**Abstract**: Manual relevance judgements in Information Retrieval are costly and require expertise, driving interest in using Large Language Models (LLMs) for automatic assessment. While LLMs have shown promise in general web search scenarios, their effectiveness for evaluating domain-specific search results, such as educational resources, remains unexplored. To investigate different ways of including domain-specific criteria in LLM prompts for relevance judgement, we collected and released a dataset of 401 human relevance judgements from a user study involving teaching professionals performing search tasks related to lesson planning. We compared three approaches to structuring these prompts: a simple two-aspect evaluation baseline from prior work on using LLMs as relevance judges, a comprehensive 12-dimensional rubric derived from educational literature, and criteria directly informed by the study participants. Using domain-specific frameworks, LLMs achieved strong agreement with human judgements (Cohen's $\kappa$ up to 0.650), significantly outperforming the baseline approach. The participant-derived framework proved particularly robust, with GPT-3.5 achieving $\kappa$ scores of 0.639 and 0.613 for 10-dimension and 5-dimension versions respectively. System-level evaluation showed that LLM judgements reliably identified top-performing retrieval approaches (RBO scores 0.71-0.76) while maintaining reasonable discrimination between systems (RBO 0.52-0.56). These findings suggest that LLMs can effectively evaluate educational resources when prompted with domain-specific criteria, though performance varies with framework complexity and input structure. 

---
