# The MedPerturb Dataset: What Non-Content Perturbations Reveal About Human and Clinical LLM Decision Making 

**Authors**: Abinitha Gourabathina, Yuexing Hao, Walter Gerych, Marzyeh Ghassemi  

**Link**: [PDF](https://arxiv.org/pdf/2506.17163)  

**Abstract**: Clinical robustness is critical to the safe deployment of medical Large Language Models (LLMs), but key questions remain about how LLMs and humans may differ in response to the real-world variability typified by clinical settings. To address this, we introduce MedPerturb, a dataset designed to systematically evaluate medical LLMs under controlled perturbations of clinical input. MedPerturb consists of clinical vignettes spanning a range of pathologies, each transformed along three axes: (1) gender modifications (e.g., gender-swapping or gender-removal); (2) style variation (e.g., uncertain phrasing or colloquial tone); and (3) format changes (e.g., LLM-generated multi-turn conversations or summaries). With MedPerturb, we release a dataset of 800 clinical contexts grounded in realistic input variability, outputs from four LLMs, and three human expert reads per clinical context. We use MedPerturb in two case studies to reveal how shifts in gender identity cues, language style, or format reflect diverging treatment selections between humans and LLMs. We find that LLMs are more sensitive to gender and style perturbations while human annotators are more sensitive to LLM-generated format perturbations such as clinical summaries. Our results highlight the need for evaluation frameworks that go beyond static benchmarks to assess the similarity between human clinician and LLM decisions under the variability characteristic of clinical settings. 

---
# Chain-of-Trust: A Progressive Trust Evaluation Framework Enabled by Generative AI 

**Authors**: Botao Zhu, Xianbin Wang, Lei Zhang, Xuemin, Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.17130)  

**Abstract**: In collaborative systems with complex tasks relying on distributed resources, trust evaluation of potential collaborators has emerged as an effective mechanism for task completion. However, due to the network dynamics and varying information gathering latencies, it is extremely challenging to observe and collect all trust attributes of a collaborating device concurrently for a comprehensive trust assessment. In this paper, a novel progressive trust evaluation framework, namely chain-of-trust, is proposed to make better use of misaligned device attribute data. This framework, designed for effective task completion, divides the trust evaluation process into multiple chained stages based on task decomposition. At each stage, based on the task completion process, the framework only gathers the latest device attribute data relevant to that stage, leading to reduced trust evaluation complexity and overhead. By leveraging advanced in-context learning, few-shot learning, and reasoning capabilities, generative AI is then employed to analyze and interpret the collected data to produce correct evaluation results quickly. Only devices deemed trustworthy at this stage proceed to the next round of trust evaluation. The framework ultimately determines devices that remain trustworthy across all stages. Experimental results demonstrate that the proposed framework achieves high accuracy in trust evaluation. 

---
# When Can Model-Free Reinforcement Learning be Enough for Thinking? 

**Authors**: Josiah P. Hanna, Nicholas E. Corrado  

**Link**: [PDF](https://arxiv.org/pdf/2506.17124)  

**Abstract**: Recent work on large language models has demonstrated the use of model-free reinforcement learning (RL) to train reasoning-like capabilities. The emergence of "thinking" through model-free RL is interesting as thinking actions neither produce reward nor change the external world state to one where the agent is more likely to get reward. This paper seeks to build a domain-independent understanding of when model-free RL will lead to "thinking" as a strategy for reward maximization. To build this understanding, we first introduce a theoretical model which we call a \textit{thought Markov decision process} (MDP). Thought MDPs minimally extend the classical MDP model to include an abstract notion of thought state and thought action. Using the thought MDP model, we prove the importance of policy initialization in determining whether or not thinking emerges and show formally that thought actions are equivalent to the agent choosing to perform a step of policy improvement before continuing to act. We then show that open-source LLMs satisfy the conditions that our theory predicts are necessary for model-free RL to produce thinking-like behavior. Finally, we hypothesize sufficient conditions that would enable thinking to be learned outside of language generation and introduce a toy domain where a combination of multi-task pre-training and designated thought actions enable more data-efficient RL compared to non-thinking agents. 

---
# Mathematical Proof as a Litmus Test: Revealing Failure Modes of Advanced Large Reasoning Models 

**Authors**: Dadi Guo, Jiayu Liu, Zhiyuan Fan, Zhitao He, Haoran Li, Yumeng Wang, Yi R., Fung  

**Link**: [PDF](https://arxiv.org/pdf/2506.17114)  

**Abstract**: Large reasoning models (e.g., R1, o3) have demonstrated remarkable mathematical problem-solving abilities. However, the high reported accuracy of these advanced models on popular datasets, reliance on purely numerical evaluation and potential benchmark leakage, often masks their true reasoning shortcomings. To address this, we propose leveraging the inherent rigor and methodological complexity of mathematical proofs as a diagnostic tool to expose these hidden failures. Specifically, we introduce the RFMDataset (Reveal Failure Modes), a collection of 200 diverse mathematical proof problems, and thoroughly evaluate advanced models' performance on it. Our in-depth analysis of their failures uncovers 10 fine-grained error types, which shows fundamental limitations in current large reasoning models: 1) large reasoning models grapple profoundly with mathematical proofs, with some generating entirely correct proofs for less than 20% of problems and failing even on basic ones; 2) models exhibit a diverse spectrum of reasoning failures, prominently demonstrating the lack of guarantees for the correctness and rigor of single-step reasoning; and 3) models show hallucination and incompleteness during the reasoning process. Our findings reveal that models' self-reflection is insufficient to resolve the current logical dilemmas, necessitating formalized and fine-grained logical training. 

---
# Are Bias Evaluation Methods Biased ? 

**Authors**: Lina Berrayana, Sean Rooney, Luis Garcés-Erice, Ioana Giurgiu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17111)  

**Abstract**: The creation of benchmarks to evaluate the safety of Large Language Models is one of the key activities within the trusted AI community. These benchmarks allow models to be compared for different aspects of safety such as toxicity, bias, harmful behavior etc. Independent benchmarks adopt different approaches with distinct data sets and evaluation methods. We investigate how robust such benchmarks are by using different approaches to rank a set of representative models for bias and compare how similar are the overall rankings. We show that different but widely used bias evaluations methods result in disparate model rankings. We conclude with recommendations for the community in the usage of such benchmarks. 

---
# Towards Advanced Mathematical Reasoning for LLMs via First-Order Logic Theorem Proving 

**Authors**: Chuxue Cao, Mengze Li, Juntao Dai, Jinluan Yang, Zijian Zhao, Shengyu Zhang, Weijie Shi, Chengzhong Liu, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.17104)  

**Abstract**: Large language models (LLMs) have shown promising first-order logic (FOL) reasoning capabilities with applications in various areas. However, their effectiveness in complex mathematical reasoning involving multi-step FOL deductions is still under-researched. While LLMs perform competitively on established mathematical reasoning benchmarks, they struggle with multi-step FOL tasks, as demonstrated by Deepseek-Prover-V2-7B's low accuracy (4.2%) on our proposed theorem proving dataset. This issue arises from the limited exploration of diverse proof strategies and the potential for early reasoning mistakes to undermine entire proofs. To address these issues, we propose DREAM, a self-adaptive solution that enhances the Diversity and REAsonability of LLMs' generation strategies. DREAM incorporates an Axiom-Driven Strategy Diversification mechanism to promote varied strategic outcomes and a Sub-Proposition Error Feedback to help LLMs reflect on and correct their proofs. Our contributions include pioneering advancements in LLMs' mathematical reasoning through FOL theorem proving, introducing a novel inference stage solution that improves performance by 0.6% to 6.4%, and providing a curated dataset of 447 mathematical theorems in Lean 4 format for evaluation. 

---
# Dispositions and Roles of Generically Dependent Entities 

**Authors**: Fabian Neuhaus  

**Link**: [PDF](https://arxiv.org/pdf/2506.17085)  

**Abstract**: BFO 2020 does not support functions, dispositions, and roles of generically dependent continuants (like software or datasets). In this paper, we argue that this is a severe limitation, which prevents, for example, the adequate representation of the functions of computer models or the various roles of datasets during the execution of these models. We discuss the aspects of BFO 2020 that prevent the representation of realizable entities of generically dependent continuants. Two approaches to address the issue are presented: (a) the use of defined classes and (b) a proposal of changes that allow BFO to support functions, dispositions, and roles of generically dependent continuants. 

---
# A Quantile Regression Approach for Remaining Useful Life Estimation with State Space Models 

**Authors**: Davide Frizzo, Francesco Borsatti, Gian Antonio Susto  

**Link**: [PDF](https://arxiv.org/pdf/2506.17018)  

**Abstract**: Predictive Maintenance (PdM) is pivotal in Industry 4.0 and 5.0, proactively enhancing efficiency through accurate equipment Remaining Useful Life (RUL) prediction, thus optimizing maintenance scheduling and reducing unexpected failures and premature interventions. This paper introduces a novel RUL estimation approach leveraging State Space Models (SSM) for efficient long-term sequence modeling. To handle model uncertainty, Simoultaneous Quantile Regression (SQR) is integrated into the SSM, enabling multiple quantile estimations. The proposed method is benchmarked against traditional sequence modelling techniques (LSTM, Transformer, Informer) using the C-MAPSS dataset. Results demonstrate superior accuracy and computational efficiency of SSM models, underscoring their potential for high-stakes industrial applications. 

---
# Elevating Styled Mahjong Agents with Learning from Demonstration 

**Authors**: Lingfeng Li, Yunlong Lu, Yongyi Wang, Wenxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.16995)  

**Abstract**: A wide variety of bots in games enriches the gameplay experience and enhances replayability. Recent advancements in game artificial intelligence have predominantly focused on improving the proficiency of bots. Nevertheless, developing highly competent bots with a wide range of distinct play styles remains a relatively under-explored area. We select the Mahjong game environment as a case study. The high degree of randomness inherent in the Mahjong game and the prevalence of out-of-distribution states lead to suboptimal performance of existing offline learning and Learning-from-Demonstration (LfD) algorithms. In this paper, we leverage the gameplay histories of existing Mahjong agents and put forward a novel LfD algorithm that necessitates only minimal modifications to the Proximal Policy Optimization algorithm. The comprehensive empirical results illustrate that our proposed method not only significantly enhances the proficiency of the agents but also effectively preserves their unique play styles. 

---
# Multimodal Fused Learning for Solving the Generalized Traveling Salesman Problem in Robotic Task Planning 

**Authors**: Jiaqi Chen, Mingfeng Fan, Xuefeng Zhang, Jingsong Liang, Yuhong Cao, Guohua Wu, Guillaume Adrien Sartoretti  

**Link**: [PDF](https://arxiv.org/pdf/2506.16931)  

**Abstract**: Effective and efficient task planning is essential for mobile robots, especially in applications like warehouse retrieval and environmental monitoring. These tasks often involve selecting one location from each of several target clusters, forming a Generalized Traveling Salesman Problem (GTSP) that remains challenging to solve both accurately and efficiently. To address this, we propose a Multimodal Fused Learning (MMFL) framework that leverages both graph and image-based representations to capture complementary aspects of the problem, and learns a policy capable of generating high-quality task planning schemes in real time. Specifically, we first introduce a coordinate-based image builder that transforms GTSP instances into spatially informative representations. We then design an adaptive resolution scaling strategy to enhance adaptability across different problem scales, and develop a multimodal fusion module with dedicated bottlenecks that enables effective integration of geometric and spatial features. Extensive experiments show that our MMFL approach significantly outperforms state-of-the-art methods across various GTSP instances while maintaining the computational efficiency required for real-time robotic applications. Physical robot tests further validate its practical effectiveness in real-world scenarios. 

---
# Real-Time Black-Box Optimization for Dynamic Discrete Environments Using Embedded Ising Machines 

**Authors**: Tomoya Kashimata, Yohei Hamakawa, Masaya Yamasaki, Kosuke Tatsumura  

**Link**: [PDF](https://arxiv.org/pdf/2506.16924)  

**Abstract**: Many real-time systems require the optimization of discrete variables. Black-box optimization (BBO) algorithms and multi-armed bandit (MAB) algorithms perform optimization by repeatedly taking actions and observing the corresponding instant rewards without any prior knowledge. Recently, a BBO method using an Ising machine has been proposed to find the best action that is represented by a combination of discrete values and maximizes the instant reward in static environments. In contrast, dynamic environments, where real-time systems operate, necessitate MAB algorithms that maximize the average reward over multiple trials. However, due to the enormous number of actions resulting from the combinatorial nature of discrete optimization, conventional MAB algorithms cannot effectively optimize dynamic, discrete environments. Here, we show a heuristic MAB method for dynamic, discrete environments by extending the BBO method, in which an Ising machine effectively explores the actions while considering interactions between variables and changes in dynamic environments. We demonstrate the dynamic adaptability of the proposed method in a wireless communication system with moving users. 

---
# AI's Blind Spots: Geographic Knowledge and Diversity Deficit in Generated Urban Scenario 

**Authors**: Ciro Beneduce, Massimiliano Luca, Bruno Lepri  

**Link**: [PDF](https://arxiv.org/pdf/2506.16898)  

**Abstract**: Image generation models are revolutionizing many domains, and urban analysis and design is no exception. While such models are widely adopted, there is a limited literature exploring their geographic knowledge, along with the biases they embed. In this work, we generated 150 synthetic images for each state in the USA and related capitals using FLUX 1 and Stable Diffusion 3.5, two state-of-the-art models for image generation. We embed each image using DINO-v2 ViT-S/14 and the Fréchet Inception Distances to measure the similarity between the generated images. We found that while these models have implicitly learned aspects of USA geography, if we prompt the models to generate an image for "United States" instead of specific cities or states, the models exhibit a strong representative bias toward metropolis-like areas, excluding rural states and smaller cities. {\color{black} In addition, we found that models systematically exhibit some entity-disambiguation issues with European-sounding names like Frankfort or Devon. 

---
# Reinforcement learning for hybrid charging stations planning and operation considering fixed and mobile chargers 

**Authors**: Yanchen Zhu, Honghui Zou, Chufan Liu, Yuyu Luo, Yuankai Wu, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16764)  

**Abstract**: The success of vehicle electrification, which brings significant societal and environmental benefits, is contingent upon the availability of efficient and adaptable charging infrastructure. Traditional fixed-location charging stations often face issues like underutilization or congestion due to the dynamic nature of charging demand. Mobile chargers have emerged as a flexible solution, capable of relocating to align with these demand fluctuations. This paper addresses the optimal planning and operation of hybrid charging infrastructures, integrating both fixed and mobile chargers within urban road networks. We introduce the Hybrid Charging Station Planning and Operation (HCSPO) problem, which simultaneously optimizes the location and configuration of fixed charging stations and schedules mobile chargers for dynamic operations. Our approach incorporates a charging demand prediction model grounded in Model Predictive Control (MPC) to enhance decision-making. To solve the HCSPO problem, we propose a deep reinforcement learning method, augmented with heuristic scheduling techniques, to effectively bridge the planning of fixed chargers with the real-time operation of mobile chargers. Extensive case studies using real-world urban scenarios demonstrate that our method significantly improves the availability of charging infrastructure and reduces user inconvenience compared to existing solutions and baselines. 

---
# Incentivizing High-quality Participation From Federated Learning Agents 

**Authors**: Jinlong Pang, Jiaheng Wei, Yifan Hua, Chen Qian, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.16731)  

**Abstract**: Federated learning (FL) provides a promising paradigm for facilitating collaboration between multiple clients that jointly learn a global model without directly sharing their local data. However, existing research suffers from two caveats: 1) From the perspective of agents, voluntary and unselfish participation is often assumed. But self-interested agents may opt out of the system or provide low-quality contributions without proper incentives; 2) From the mechanism designer's perspective, the aggregated models can be unsatisfactory as the existing game-theoretical federated learning approach for data collection ignores the potential heterogeneous effort caused by contributed data. To alleviate above challenges, we propose an incentive-aware framework for agent participation that considers data heterogeneity to accelerate the convergence process. Specifically, we first introduce the notion of Wasserstein distance to explicitly illustrate the heterogeneous effort and reformulate the existing upper bound of convergence. To induce truthful reporting from agents, we analyze and measure the generalization error gap of any two agents by leveraging the peer prediction mechanism to develop score functions. We further present a two-stage Stackelberg game model that formalizes the process and examines the existence of equilibrium. Extensive experiments on real-world datasets demonstrate the effectiveness of our proposed mechanism. 

---
# Interpretable Low-Dimensional Modeling of Spatiotemporal Agent States for Decision Making in Football Tactics 

**Authors**: Kenjiro Ide, Taiga Someya, Kohei Kawaguchi, Keisuke Fujii  

**Link**: [PDF](https://arxiv.org/pdf/2506.16696)  

**Abstract**: Understanding football tactics is crucial for managers and analysts. Previous research has proposed models based on spatial and kinematic equations, but these are computationally expensive. Also, Reinforcement learning approaches use player positions and velocities but lack interpretability and require large datasets. Rule-based models align with expert knowledge but have not fully considered all players' states. This study explores whether low-dimensional, rule-based models using spatiotemporal data can effectively capture football tactics. Our approach defines interpretable state variables for both the ball-holder and potential pass receivers, based on criteria that explore options like passing. Through discussions with a manager, we identified key variables representing the game state. We then used StatsBomb event data and SkillCorner tracking data from the 2023$/$24 LaLiga season to train an XGBoost model to predict pass success. The analysis revealed that the distance between the player and the ball, as well as the player's space score, were key factors in determining successful passes. Our interpretable low-dimensional modeling facilitates tactical analysis through the use of intuitive variables and provides practical value as a tool to support decision-making in football. 

---
# The Role of Explanation Styles and Perceived Accuracy on Decision Making in Predictive Process Monitoring 

**Authors**: Soobin Chae, Suhwan Lee, Hanna Hauptmann, Hajo A. Reijers, Xixi Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.16617)  

**Abstract**: Predictive Process Monitoring (PPM) often uses deep learning models to predict the future behavior of ongoing processes, such as predicting process outcomes. While these models achieve high accuracy, their lack of interpretability undermines user trust and adoption. Explainable AI (XAI) aims to address this challenge by providing the reasoning behind the predictions. However, current evaluations of XAI in PPM focus primarily on functional metrics (such as fidelity), overlooking user-centered aspects such as their effect on task performance and decision-making. This study investigates the effects of explanation styles (feature importance, rule-based, and counterfactual) and perceived AI accuracy (low or high) on decision-making in PPM. We conducted a decision-making experiment, where users were presented with the AI predictions, perceived accuracy levels, and explanations of different styles. Users' decisions were measured both before and after receiving explanations, allowing the assessment of objective metrics (Task Performance and Agreement) and subjective metrics (Decision Confidence). Our findings show that perceived accuracy and explanation style have a significant effect. 

---
# A Community-driven vision for a new Knowledge Resource for AI 

**Authors**: Vinay K Chaudhri, Chaitan Baru, Brandon Bennett, Mehul Bhatt, Darion Cassel, Anthony G Cohn, Rina Dechter, Esra Erdem, Dave Ferrucci, Ken Forbus, Gregory Gelfond, Michael Genesereth, Andrew S. Gordon, Benjamin Grosof, Gopal Gupta, Jim Hendler, Sharat Israni, Tyler R. Josephson, Patrick Kyllonen, Yuliya Lierler, Vladimir Lifschitz, Clifton McFate, Hande K. McGinty, Leora Morgenstern, Alessandro Oltramari, Praveen Paritosh, Dan Roth, Blake Shepard, Cogan Shimzu, Denny Vrandečić, Mark Whiting, Michael Witbrock  

**Link**: [PDF](https://arxiv.org/pdf/2506.16596)  

**Abstract**: The long-standing goal of creating a comprehensive, multi-purpose knowledge resource, reminiscent of the 1984 Cyc project, still persists in AI. Despite the success of knowledge resources like WordNet, ConceptNet, Wolfram|Alpha and other commercial knowledge graphs, verifiable, general-purpose widely available sources of knowledge remain a critical deficiency in AI infrastructure. Large language models struggle due to knowledge gaps; robotic planning lacks necessary world knowledge; and the detection of factually false information relies heavily on human expertise. What kind of knowledge resource is most needed in AI today? How can modern technology shape its development and evaluation? A recent AAAI workshop gathered over 50 researchers to explore these questions. This paper synthesizes our findings and outlines a community-driven vision for a new knowledge infrastructure. In addition to leveraging contemporary advances in knowledge representation and reasoning, one promising idea is to build an open engineering framework to exploit knowledge modules effectively within the context of practical applications. Such a framework should include sets of conventions and social structures that are adopted by contributors. 

---
# Advancing Harmful Content Detection in Organizational Research: Integrating Large Language Models with Elo Rating System 

**Authors**: Mustafa Akben, Aaron Satko  

**Link**: [PDF](https://arxiv.org/pdf/2506.16575)  

**Abstract**: Large language models (LLMs) offer promising opportunities for organizational research. However, their built-in moderation systems can create problems when researchers try to analyze harmful content, often refusing to follow certain instructions or producing overly cautious responses that undermine validity of the results. This is particularly problematic when analyzing organizational conflicts such as microaggressions or hate speech. This paper introduces an Elo rating-based method that significantly improves LLM performance for harmful content analysis In two datasets, one focused on microaggression detection and the other on hate speech, we find that our method outperforms traditional LLM prompting techniques and conventional machine learning models on key measures such as accuracy, precision, and F1 scores. Advantages include better reliability when analyzing harmful content, fewer false positives, and greater scalability for large-scale datasets. This approach supports organizational applications, including detecting workplace harassment, assessing toxic communication, and fostering safer and more inclusive work environments. 

---
# ML-Master: Towards AI-for-AI via Integration of Exploration and Reasoning 

**Authors**: Zexi Liu, Yuzhu Cai, Xinyu Zhu, Yujie Zheng, Runkun Chen, Ying Wen, Yanfeng Wang, Weinan E, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.16499)  

**Abstract**: As AI capabilities advance toward and potentially beyond human-level performance, a natural transition emerges where AI-driven development becomes more efficient than human-centric approaches. A promising pathway toward this transition lies in AI-for-AI (AI4AI), which leverages AI techniques to automate and optimize the design, training, and deployment of AI systems themselves. While LLM-based agents have shown the potential to realize AI4AI, they are often unable to fully leverage the experience accumulated by agents during the exploration of solutions in the reasoning process, leading to inefficiencies and suboptimal performance. To address this limitation, we propose ML-Master, a novel AI4AI agent that seamlessly integrates exploration and reasoning by employing a selectively scoped memory mechanism. This approach allows ML-Master to efficiently combine diverse insights from parallel solution trajectories with analytical reasoning, guiding further exploration without overwhelming the agent with excessive context. We evaluate ML-Master on the MLE-Bench, where it achieves a 29.3% average medal rate, significantly surpassing existing methods, particularly in medium-complexity tasks, while accomplishing this superior performance within a strict 12-hour time constraint-half the 24-hour limit used by previous baselines. These results demonstrate ML-Master's potential as a powerful tool for advancing AI4AI. 

---
# Agentic Personalisation of Cross-Channel Marketing Experiences 

**Authors**: Sami Abboud, Eleanor Hanna, Olivier Jeunen, Vineesha Raheja, Schaun Wheeler  

**Link**: [PDF](https://arxiv.org/pdf/2506.16429)  

**Abstract**: Consumer applications provide ample opportunities to surface and communicate various forms of content to users. From promotional campaigns for new features or subscriptions, to evergreen nudges for engagement, or personalised recommendations; across e-mails, push notifications, and in-app surfaces. The conventional approach to orchestration for communication relies heavily on labour-intensive manual marketer work, and inhibits effective personalisation of content, timing, frequency, and copy-writing. We formulate this task under a sequential decision-making framework, where we aim to optimise a modular decision-making policy that maximises incremental engagement for any funnel event. Our approach leverages a Difference-in-Differences design for Individual Treatment Effect estimation, and Thompson sampling to balance the explore-exploit trade-off. We present results from a multi-service application, where our methodology has resulted in significant increases to a variety of goal events across several product features, and is currently deployed across 150 million users. 

---
# IS-Bench: Evaluating Interactive Safety of VLM-Driven Embodied Agents in Daily Household Tasks 

**Authors**: Xiaoya Lu, Zeren Chen, Xuhao Hu, Yijin Zhou, Weichen Zhang, Dongrui Liu, Lu Sheng, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2506.16402)  

**Abstract**: Flawed planning from VLM-driven embodied agents poses significant safety hazards, hindering their deployment in real-world household tasks. However, existing static, non-interactive evaluation paradigms fail to adequately assess risks within these interactive environments, since they cannot simulate dynamic risks that emerge from an agent's actions and rely on unreliable post-hoc evaluations that ignore unsafe intermediate steps. To bridge this critical gap, we propose evaluating an agent's interactive safety: its ability to perceive emergent risks and execute mitigation steps in the correct procedural order. We thus present IS-Bench, the first multi-modal benchmark designed for interactive safety, featuring 161 challenging scenarios with 388 unique safety risks instantiated in a high-fidelity simulator. Crucially, it facilitates a novel process-oriented evaluation that verifies whether risk mitigation actions are performed before/after specific risk-prone steps. Extensive experiments on leading VLMs, including the GPT-4o and Gemini-2.5 series, reveal that current agents lack interactive safety awareness, and that while safety-aware Chain-of-Thought can improve performance, it often compromises task completion. By highlighting these critical limitations, IS-Bench provides a foundation for developing safer and more reliable embodied AI systems. 

---
# Explainable Rule Application via Structured Prompting: A Neural-Symbolic Approach 

**Authors**: Albert Sadowski, Jarosław A. Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2506.16335)  

**Abstract**: Large Language Models (LLMs) excel in complex reasoning tasks but struggle with consistent rule application, exception handling, and explainability, particularly in domains like legal analysis that require both natural language understanding and precise logical inference. This paper introduces a structured prompting framework that decomposes reasoning into three verifiable steps: entity identification, property extraction, and symbolic rule application. By integrating neural and symbolic approaches, our method leverages LLMs' interpretive flexibility while ensuring logical consistency through formal verification. The framework externalizes task definitions, enabling domain experts to refine logical structures without altering the architecture. Evaluated on the LegalBench hearsay determination task, our approach significantly outperformed baselines, with OpenAI o-family models showing substantial improvements - o1 achieving an F1 score of 0.929 and o3-mini reaching 0.867 using structured decomposition with complementary predicates, compared to their few-shot baselines of 0.714 and 0.74 respectively. This hybrid neural-symbolic system offers a promising pathway for transparent and consistent rule-based reasoning, suggesting potential for explainable AI applications in structured legal reasoning tasks. 

---
# Approximation Fixpoint Theory with Refined Approximation Spaces 

**Authors**: Linde Vanbesien, Bart Bogaerts, Marc Denecker  

**Link**: [PDF](https://arxiv.org/pdf/2506.16294)  

**Abstract**: Approximation Fixpoint Theory (AFT) is a powerful theory covering various semantics of non-monotonic reasoning formalisms in knowledge representation such as Logic Programming and Answer Set Programming. Many semantics of such non-monotonic formalisms can be characterized as suitable fixpoints of a non-monotonic operator on a suitable lattice. Instead of working on the original lattice, AFT operates on intervals in such lattice to approximate or construct the fixpoints of interest. While AFT has been applied successfully across a broad range of non-monotonic reasoning formalisms, it is confronted by its limitations in other, relatively simple, examples. In this paper, we overcome those limitations by extending consistent AFT to deal with approximations that are more refined than intervals. Therefore, we introduce a more general notion of approximation spaces, showcase the improved expressiveness and investigate relations between different approximation spaces. 

---
# Large Language Models are Near-Optimal Decision-Makers with a Non-Human Learning Behavior 

**Authors**: Hao Li, Gengrui Zhang, Petter Holme, Shuyue Hu, Zhen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16163)  

**Abstract**: Human decision-making belongs to the foundation of our society and civilization, but we are on the verge of a future where much of it will be delegated to artificial intelligence. The arrival of Large Language Models (LLMs) has transformed the nature and scope of AI-supported decision-making; however, the process by which they learn to make decisions, compared to humans, remains poorly understood. In this study, we examined the decision-making behavior of five leading LLMs across three core dimensions of real-world decision-making: uncertainty, risk, and set-shifting. Using three well-established experimental psychology tasks designed to probe these dimensions, we benchmarked LLMs against 360 newly recruited human participants. Across all tasks, LLMs often outperformed humans, approaching near-optimal performance. Moreover, the processes underlying their decisions diverged fundamentally from those of humans. On the one hand, our finding demonstrates the ability of LLMs to manage uncertainty, calibrate risk, and adapt to changes. On the other hand, this disparity highlights the risks of relying on them as substitutes for human judgment, calling for further inquiry. 

---
# Geometric Learning in Black-Box Optimization: A GNN Framework for Algorithm Performance Prediction 

**Authors**: Ana Kostovska, Carola Doerr, Sašo Džeroski, Panče Panov, Tome Eftimov  

**Link**: [PDF](https://arxiv.org/pdf/2506.16144)  

**Abstract**: Automated algorithm performance prediction in numerical blackbox optimization often relies on problem characterizations, such as exploratory landscape analysis features. These features are typically used as inputs to machine learning models and are represented in a tabular format. However, such approaches often overlook algorithm configurations, a key factor influencing performance. The relationships between algorithm operators, parameters, problem characteristics, and performance outcomes form a complex structure best represented as a graph. This work explores the use of heterogeneous graph data structures and graph neural networks to predict the performance of optimization algorithms by capturing the complex dependencies between problems, algorithm configurations, and performance outcomes. We focus on two modular frameworks, modCMA-ES and modDE, which decompose two widely used derivative-free optimization algorithms: the covariance matrix adaptation evolution strategy (CMA-ES) and differential evolution (DE). We evaluate 324 modCMA-ES and 576 modDE variants on 24 BBOB problems across six runtime budgets and two problem dimensions. Achieving up to 36.6% improvement in MSE over traditional tabular-based methods, this work highlights the potential of geometric learning in black-box optimization. 

---
# Consistency Verification in Ontology-Based Process Models with Parameter Interdependencies 

**Authors**: Tom Jeleniewski, Hamied Nabizada, Jonathan Reif, Felix Gehlhoff, Alexander Fay  

**Link**: [PDF](https://arxiv.org/pdf/2506.16087)  

**Abstract**: The formalization of process knowledge using ontologies enables consistent modeling of parameter interdependencies in manufacturing. These interdependencies are typically represented as mathematical expressions that define relations between process parameters, supporting tasks such as calculation, validation, and simulation. To support cross-context application and knowledge reuse, such expressions are often defined in a generic form and applied across multiple process contexts. This highlights the necessity of a consistent and semantically coherent model to ensure the correctness of data retrieval and interpretation. Consequently, dedicated mechanisms are required to address key challenges such as selecting context-relevant data, ensuring unit compatibility between variables and data elements, and verifying the completeness of input data required for evaluating mathematical expressions. This paper presents a set of verification mechanisms for a previously developed ontology-based process model that integrates standardized process semantics, data element definitions, and formal mathematical constructs. The approach includes (i) SPARQL-based filtering to retrieve process-relevant data, (ii) a unit consistency check based on expected-unit annotations and semantic classification, and (iii) a data completeness check to validate the evaluability of interdependencies. The applicability of the approach is demonstrated with a use case from Resin Transfer Molding (RTM), supporting the development of machine-interpretable and verifiable engineering models. 

---
# OSWorld-Human: Benchmarking the Efficiency of Computer-Use Agents 

**Authors**: Reyna Abhyankar, Qi Qi, Yiying Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16042)  

**Abstract**: Generative AI is being leveraged to solve a variety of computer-use tasks involving desktop applications. State-of-the-art systems have focused solely on improving accuracy on leading benchmarks. However, these systems are practically unusable due to extremely high end-to-end latency (e.g., tens of minutes) for tasks that typically take humans just a few minutes to complete. To understand the cause behind this and to guide future developments of computer agents, we conduct the first study on the temporal performance of computer-use agents on OSWorld, the flagship benchmark in computer-use AI. We find that large model calls for planning and reflection account for the majority of the overall latency, and as an agent uses more steps to complete a task, each successive step can take 3x longer than steps at the beginning of a task. We then construct OSWorld-Human, a manually annotated version of the original OSWorld dataset that contains a human-determined trajectory for each task. We evaluate 16 agents on their efficiency using OSWorld-Human and found that even the highest-scoring agents on OSWorld take 1.4-2.7x more steps than necessary. 

---
# Dual-Objective Reinforcement Learning with Novel Hamilton-Jacobi-Bellman Formulations 

**Authors**: William Sharpless, Dylan Hirsch, Sander Tonkens, Nikhil Shinde, Sylvia Herbert  

**Link**: [PDF](https://arxiv.org/pdf/2506.16016)  

**Abstract**: Hard constraints in reinforcement learning (RL), whether imposed via the reward function or the model architecture, often degrade policy performance. Lagrangian methods offer a way to blend objectives with constraints, but often require intricate reward engineering and parameter tuning. In this work, we extend recent advances that connect Hamilton-Jacobi (HJ) equations with RL to propose two novel value functions for dual-objective satisfaction. Namely, we address: (1) the Reach-Always-Avoid problem - of achieving distinct reward and penalty thresholds - and (2) the Reach-Reach problem - of achieving thresholds of two distinct rewards. In contrast with temporal logic approaches, which typically involve representing an automaton, we derive explicit, tractable Bellman forms in this context by decomposing our problem into reach, avoid, and reach-avoid problems, as to leverage these aforementioned recent advances. From a mathematical perspective, the Reach-Always-Avoid and Reach-Reach problems are complementary and fundamentally different from standard sum-of-rewards problems and temporal logic problems, providing a new perspective on constrained decision-making. We leverage our analysis to propose a variation of Proximal Policy Optimization (DO-HJ-PPO), which solves these problems. Across a range of tasks for safe-arrival and multi-target achievement, we demonstrate that DO-HJ-PPO produces qualitatively distinct behaviors from previous approaches and out-competes a number of baselines in various metrics. 

---
# Bayesian Epistemology with Weighted Authority: A Formal Architecture for Truth-Promoting Autonomous Scientific Reasoning 

**Authors**: Craig S. Wright  

**Link**: [PDF](https://arxiv.org/pdf/2506.16015)  

**Abstract**: The exponential expansion of scientific literature has surpassed the epistemic processing capabilities of both human experts and current artificial intelligence systems. This paper introduces Bayesian Epistemology with Weighted Authority (BEWA), a formally structured architecture that operationalises belief as a dynamic, probabilistically coherent function over structured scientific claims. Each claim is contextualised, author-attributed, and evaluated through a system of replication scores, citation weighting, and temporal decay. Belief updates are performed via evidence-conditioned Bayesian inference, contradiction processing, and epistemic decay mechanisms. The architecture supports graph-based claim propagation, authorial credibility modelling, cryptographic anchoring, and zero-knowledge audit verification. By formalising scientific reasoning into a computationally verifiable epistemic network, BEWA advances the foundation for machine reasoning systems that promote truth utility, rational belief convergence, and audit-resilient integrity across dynamic scientific domains. 

---
# Exploring Big Five Personality and AI Capability Effects in LLM-Simulated Negotiation Dialogues 

**Authors**: Myke C. Cohen, Zhe Su, Hsien-Te Kao, Daniel Nguyen, Spencer Lynch, Maarten Sap, Svitlana Volkova  

**Link**: [PDF](https://arxiv.org/pdf/2506.15928)  

**Abstract**: This paper presents an evaluation framework for agentic AI systems in mission-critical negotiation contexts, addressing the need for AI agents that can adapt to diverse human operators and stakeholders. Using Sotopia as a simulation testbed, we present two experiments that systematically evaluated how personality traits and AI agent characteristics influence LLM-simulated social negotiation outcomes--a capability essential for a variety of applications involving cross-team coordination and civil-military interactions. Experiment 1 employs causal discovery methods to measure how personality traits impact price bargaining negotiations, through which we found that Agreeableness and Extraversion significantly affect believability, goal achievement, and knowledge acquisition outcomes. Sociocognitive lexical measures extracted from team communications detected fine-grained differences in agents' empathic communication, moral foundations, and opinion patterns, providing actionable insights for agentic AI systems that must operate reliably in high-stakes operational scenarios. Experiment 2 evaluates human-AI job negotiations by manipulating both simulated human personality and AI system characteristics, specifically transparency, competence, adaptability, demonstrating how AI agent trustworthiness impact mission effectiveness. These findings establish a repeatable evaluation methodology for experimenting with AI agent reliability across diverse operator personalities and human-agent team dynamics, directly supporting operational requirements for reliable AI systems. Our work advances the evaluation of agentic AI workflows by moving beyond standard performance metrics to incorporate social dynamics essential for mission success in complex operations. 

---
# Deep Reinforcement Learning Xiangqi Player with Monte Carlo Tree Search 

**Authors**: Berk Yilmaz, Junyu Hu, Jinsong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15880)  

**Abstract**: This paper presents a Deep Reinforcement Learning (DRL) system for Xiangqi (Chinese Chess) that integrates neural networks with Monte Carlo Tree Search (MCTS) to enable strategic self-play and self-improvement. Addressing the underexplored complexity of Xiangqi, including its unique board layout, piece movement constraints, and victory conditions, our approach combines policy-value networks with MCTS to simulate move consequences and refine decision-making. By overcoming challenges such as Xiangqi's high branching factor and asymmetrical piece dynamics, our work advances AI capabilities in culturally significant strategy games while providing insights for adapting DRL-MCTS frameworks to domain-specific rule systems. 

---
# SLR: An Automated Synthesis Framework for Scalable Logical Reasoning 

**Authors**: Lukas Helff, Ahmad Omar, Felix Friedrich, Wolfgang Stammer, Antonia Wüst, Tim Woydt, Rupert Mitchell, Patrick Schramowski, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2506.15787)  

**Abstract**: We introduce SLR, an end-to-end framework for systematic evaluation and training of Large Language Models (LLMs) via Scalable Logical Reasoning. Given a user's task specification, SLR enables scalable, automated synthesis of inductive reasoning tasks with precisely controlled difficulty. For each task, SLR synthesizes (i) a latent ground-truth rule, (ii) an executable validation program used by a symbolic judge to deterministically verify model outputs, and (iii) an instruction prompt for the reasoning task. Using SLR, we create SLR-Bench, a benchmark comprising over 19k prompts spanning 20 curriculum levels that progressively increase in relational, arithmetic, and recursive complexity. Large-scale evaluation reveals that contemporary LLMs readily produce syntactically valid rules, yet often fail at correct logical inference. Recent reasoning LLMs do somewhat better, but incur substantial increases in test-time compute, sometimes exceeding 15k completion tokens. Finally, logic-tuning via SLR doubles Llama-3-8B accuracy on SLR-Bench, achieving parity with Gemini-Flash-Thinking at a fraction of computational cost. SLR is fully automated, requires no human annotation, ensures dataset novelty, and offers a scalable environment for probing and advancing LLMs' reasoning capabilities. 

---
# Advancing Stochastic 3-SAT Solvers by Dissipating Oversatisfied Constraints 

**Authors**: J. Schwardt, J. C. Budich  

**Link**: [PDF](https://arxiv.org/pdf/2506.15774)  

**Abstract**: We introduce and benchmark a stochastic local search heuristic for the NP-complete satisfiability problem 3-SAT that drastically outperforms existing solvers in the notoriously difficult realm of critically hard instances. Our construction is based on the crucial observation that well established previous approaches such as WalkSAT are prone to get stuck in local minima that are distinguished from true solutions by a larger number of oversatisfied combinatorial constraints. To address this issue, the proposed algorithm, coined DOCSAT, dissipates oversatisfied constraints (DOC), i.e. reduces their unfavorable abundance so as to render them critical. We analyze and benchmark our algorithm on a randomly generated sample of hard but satisfiable 3-SAT instances with varying problem sizes up to N=15000. Quite remarkably, we find that DOCSAT outperforms both WalkSAT and other well known algorithms including the complete solver Kissat, even when comparing its ability to solve the hardest quintile of the sample to the average performance of its competitors. The essence of DOCSAT may be seen as a way of harnessing statistical structure beyond the primary cost function of a combinatorial problem to avoid or escape local minima traps in stochastic local search, which opens avenues for generalization to other optimization problems. 

---
# Linear-Time Primitives for Algorithm Development in Graphical Causal Inference 

**Authors**: Marcel Wienöbst, Sebastian Weichwald, Leonard Henckel  

**Link**: [PDF](https://arxiv.org/pdf/2506.15758)  

**Abstract**: We introduce CIfly, a framework for efficient algorithmic primitives in graphical causal inference that isolates reachability as a reusable core operation. It builds on the insight that many causal reasoning tasks can be reduced to reachability in purpose-built state-space graphs that can be constructed on the fly during traversal. We formalize a rule table schema for specifying such algorithms and prove they run in linear time. We establish CIfly as a more efficient alternative to the common primitives moralization and latent projection, which we show are computationally equivalent to Boolean matrix multiplication. Our open-source Rust implementation parses rule table text files and runs the specified CIfly algorithms providing high-performance execution accessible from Python and R. We demonstrate CIfly's utility by re-implementing a range of established causal inference tasks within the framework and by developing new algorithms for instrumental variables. These contributions position CIfly as a flexible and scalable backbone for graphical causal inference, guiding algorithm development and enabling easy and efficient deployment. 

---
# Sysformer: Safeguarding Frozen Large Language Models with Adaptive System Prompts 

**Authors**: Kartik Sharma, Yiqiao Jin, Vineeth Rakesh, Yingtong Dou, Menghai Pan, Mahashweta Das, Srijan Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.15751)  

**Abstract**: As large language models (LLMs) are deployed in safety-critical settings, it is essential to ensure that their responses comply with safety standards. Prior research has revealed that LLMs often fail to grasp the notion of safe behaviors, resulting in either unjustified refusals to harmless prompts or the generation of harmful content. While substantial efforts have been made to improve their robustness, existing defenses often rely on costly fine-tuning of model parameters or employ suboptimal heuristic techniques. In this work, we take a novel approach to safeguard LLMs by learning to adapt the system prompts in instruction-tuned LLMs. While LLMs are typically pre-trained to follow a fixed system prompt, we investigate the impact of tailoring the system prompt to each specific user input on the safety of the responses. To this end, we propose $\textbf{Sysformer}$, a trans$\textbf{former}$ model that updates an initial $\textbf{sys}$tem prompt to a more robust system prompt in the LLM input embedding space while attending to the user prompt. While keeping the LLM parameters frozen, the Sysformer is trained to refuse to respond to a set of harmful prompts while responding ideally to a set of safe ones. Through extensive experiments on $5$ LLMs from different families and $2$ recent benchmarks, we demonstrate that Sysformer can significantly enhance the robustness of LLMs, leading to upto $80\%$ gain in the refusal rate on harmful prompts while enhancing the compliance with the safe prompts by upto $90\%$. Results also generalize well to sophisticated jailbreaking attacks, making LLMs upto $100\%$ more robust against different attack strategies. We hope our findings lead to cheaper safeguarding of LLMs and motivate future investigations into designing variable system prompts. 

---
# OAgents: An Empirical Study of Building Effective Agents 

**Authors**: He Zhu, Tianrui Qin, King Zhu, Heyuan Huang, Yeyi Guan, Jinxiang Xia, Yi Yao, Hanhao Li, Ningning Wang, Pai Liu, Tianhao Peng, Xin Gui, Xiaowan Li, Yuhui Liu, Yuchen Eleanor Jiang, Jun Wang, Changwang Zhang, Xiangru Tang, Ge Zhang, Jian Yang, Minghao Liu, Xitong Gao, Wangchunshu Zhou, Jiaheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15741)  

**Abstract**: Recently, Agentic AI has become an increasingly popular research field. However, we argue that current agent research practices lack standardization and scientific rigor, making it hard to conduct fair comparisons among methods. As a result, it is still unclear how different design choices in agent frameworks affect effectiveness, and measuring their progress remains challenging. In this work, we conduct a systematic empirical study on GAIA benchmark and BrowseComp to examine the impact of popular design choices in key agent components in a fair and rigorous manner. We find that the lack of a standard evaluation protocol makes previous works, even open-sourced ones, non-reproducible, with significant variance between random runs. Therefore, we introduce a more robust evaluation protocol to stabilize comparisons. Our study reveals which components and designs are crucial for effective agents, while others are redundant, despite seeming logical. Based on our findings, we build and open-source OAgents, a new foundation agent framework that achieves state-of-the-art performance among open-source projects. OAgents offers a modular design for various agent components, promoting future research in Agentic AI. 

---
# SHADE-Arena: Evaluating Sabotage and Monitoring in LLM Agents 

**Authors**: Jonathan Kutasov, Yuqi Sun, Paul Colognese, Teun van der Weij, Linda Petrini, Chen Bo Calvin Zhang, John Hughes, Xiang Deng, Henry Sleight, Tyler Tracy, Buck Shlegeris, Joe Benton  

**Link**: [PDF](https://arxiv.org/pdf/2506.15740)  

**Abstract**: As Large Language Models (LLMs) are increasingly deployed as autonomous agents in complex and long horizon settings, it is critical to evaluate their ability to sabotage users by pursuing hidden objectives. We study the ability of frontier LLMs to evade monitoring and achieve harmful hidden goals while completing a wide array of realistic tasks. We evaluate a broad range of frontier LLMs using SHADE (Subtle Harmful Agent Detection & Evaluation)-Arena, the first highly diverse agent evaluation dataset for sabotage and monitoring capabilities of LLM agents. SHADE-Arena consists of complex pairs of benign main tasks and harmful side objectives in complicated environments. Agents are evaluated on their ability to complete the side task without appearing suspicious to an LLM monitor. When measuring agent ability to (a) complete the main task, (b) complete the side task, and (c) avoid detection, we find that the best performing frontier models score 27% (Claude 3.7 Sonnet) and 15% (Gemini 2.5 Pro) as sabotage agents when overseen by Claude 3.6 Sonnet. For current frontier models, success on the side task relies heavily on having access to a hidden scratchpad that is not visible to the monitor. We also use SHADE-Arena to measure models' monitoring abilities, with the top monitor (Gemini 2.5 Pro) achieving an AUC of 0.87 at distinguishing benign and malign transcripts. We find that for now, models still struggle at sabotage due to failures in long-context main task execution. However, our measurements already demonstrate the difficulty of monitoring for subtle sabotage attempts, which we expect to only increase in the face of more complex and longer-horizon tasks. 

---
# ContextBench: Modifying Contexts for Targeted Latent Activation 

**Authors**: Robert Graham, Edward Stevinson, Leo Richter, Alexander Chia, Joseph Miller, Joseph Isaac Bloom  

**Link**: [PDF](https://arxiv.org/pdf/2506.15735)  

**Abstract**: Identifying inputs that trigger specific behaviours or latent features in language models could have a wide range of safety use cases. We investigate a class of methods capable of generating targeted, linguistically fluent inputs that activate specific latent features or elicit model behaviours. We formalise this approach as context modification and present ContextBench -- a benchmark with tasks assessing core method capabilities and potential safety applications. Our evaluation framework measures both elicitation strength (activation of latent features or behaviours) and linguistic fluency, highlighting how current state-of-the-art methods struggle to balance these objectives. We enhance Evolutionary Prompt Optimisation (EPO) with LLM-assistance and diffusion model inpainting, and demonstrate that these variants achieve state-of-the-art performance in balancing elicitation effectiveness and fluency. 

---
# The Safety Reminder: A Soft Prompt to Reactivate Delayed Safety Awareness in Vision-Language Models 

**Authors**: Peiyuan Tang, Haojie Xin, Xiaodong Zhang, Jun Sun, Qin Xia, Zijiang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15734)  

**Abstract**: As Vision-Language Models (VLMs) demonstrate increasing capabilities across real-world applications such as code generation and chatbot assistance, ensuring their safety has become paramount. Unlike traditional Large Language Models (LLMs), VLMs face unique vulnerabilities due to their multimodal nature, allowing adversaries to modify visual or textual inputs to bypass safety guardrails and trigger the generation of harmful content. Through systematic analysis of VLM behavior under attack, we identify a novel phenomenon termed ``delayed safety awareness''. Specifically, we observe that safety-aligned VLMs may initially be compromised to produce harmful content, but eventually recognize the associated risks and attempt to self-correct. This pattern suggests that VLMs retain their underlying safety awareness but experience a temporal delay in their activation. Building on this insight, we hypothesize that VLMs' safety awareness can be proactively reactivated through carefully designed prompts. To this end, we introduce ``The Safety Reminder'', a soft prompt tuning approach that optimizes learnable prompt tokens, which are periodically injected during the text generation process to enhance safety awareness, effectively preventing harmful content generation. Additionally, our safety reminder only activates when harmful content is detected, leaving normal conversations unaffected and preserving the model's performance on benign tasks. Through comprehensive evaluation across three established safety benchmarks and one adversarial attacks, we demonstrate that our approach significantly reduces attack success rates while maintaining model utility, offering a practical solution for deploying safer VLMs in real-world applications. 

---
# $\texttt{SPECS}$: Faster Test-Time Scaling through Speculative Drafts 

**Authors**: Mert Cemri, Nived Rajaraman, Rishabh Tiwari, Xiaoxuan Liu, Kurt Keutzer, Ion Stoica, Kannan Ramchandran, Ahmad Beirami, Ziteng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.15733)  

**Abstract**: Scaling test-time compute has driven the recent advances in the reasoning capabilities of large language models (LLMs), typically by allocating additional computation for more thorough exploration. However, increased compute often comes at the expense of higher user-facing latency, directly impacting user experience. Current test-time scaling methods primarily optimize for accuracy based on total compute resources (FLOPS), often overlooking latency constraints. To address this gap, we propose $\texttt{SPECS}$, a latency-aware test-time scaling method inspired by speculative decoding. $\texttt{SPECS}$~uses a smaller, faster model to generate candidate sequences efficiently, and evaluates these candidates using signals from both a larger target model and a dedicated reward model. We introduce new integration strategies, including reward-guided soft verification and a reward-based deferral mechanism. Empirical results on MATH500, AMC23 and OlympiadBench datasets show that $\texttt{SPECS}$~matches or surpasses beam search accuracy while reducing latency by up to $\sim$19.1\%. Our theoretical analysis shows that our algorithm converges to the solution of a KL-regularized reinforcement learning objective with increasing beam width. 

---
# LLMs Struggle to Perform Counterfactual Reasoning with Parametric Knowledge 

**Authors**: Khurram Yamin, Gaurav Ghosal, Bryan Wilder  

**Link**: [PDF](https://arxiv.org/pdf/2506.15732)  

**Abstract**: Large Language Models have been shown to contain extensive world knowledge in their parameters, enabling impressive performance on many knowledge intensive tasks. However, when deployed in novel settings, LLMs often encounter situations where they must integrate parametric knowledge with new or unfamiliar information. In this work, we explore whether LLMs can combine knowledge in-context with their parametric knowledge through the lens of counterfactual reasoning. Through synthetic and real experiments in multi-hop reasoning problems, we show that LLMs generally struggle with counterfactual reasoning, often resorting to exclusively using their parametric knowledge. Moreover, we show that simple post-hoc finetuning can struggle to instill counterfactual reasoning ability -- often leading to degradation in stored parametric knowledge. Ultimately, our work reveals important limitations of current LLM's abilities to re-purpose parametric knowledge in novel settings. 

---
# No Free Lunch: Rethinking Internal Feedback for LLM Reasoning 

**Authors**: Yanzhi Zhang, Zhaoxi Zhang, Haoxiang Guan, Yilin Cheng, Yitong Duan, Chen Wang, Yue Wang, Shuxin Zheng, Jiyan He  

**Link**: [PDF](https://arxiv.org/pdf/2506.17219)  

**Abstract**: Reinforcement learning has emerged as a powerful paradigm for post-training large language models (LLMs) to improve reasoning. Approaches like Reinforcement Learning from Human Feedback (RLHF) and Reinforcement Learning with Verifiable Rewards (RLVR) have shown strong results, but they require extensive external supervision. We investigate an alternative class of methods, Reinforcement Learning from Internal Feedback (RLIF), which relies solely on intrinsic model-derived signals instead of external rewards. In particular, we leverage unsupervised reward proxies such as token-level entropy, trajectory-level entropy, and self-certainty. Our theoretical analysis shows these internal objectives are partially equivalent, and we empirically evaluate various RLIF strategies on challenging math reasoning benchmarks. Experimental results demonstrate that RLIF can boost the reasoning performance of base LLMs at the beginning phase of the training, matching or surpassing RLVR techniques on these tasks. However, when training progresses, performance degrades even below the model before training. Moreover, we find that RLIF yields little improvement for instruction-tuned models, indicating diminishing returns of intrinsic feedback once an LLM is already instruction-tuned. We further analyze this limitation by mixing model weights and explain the reason of RLIF's training behaviors, providing practical guidelines for integrating internal feedback signals into LLM training. We hope our analysis of internal feedback will inform more principled and effective strategies for LLM post-training. 

---
# Machine Mental Imagery: Empower Multimodal Reasoning with Latent Visual Tokens 

**Authors**: Zeyuan Yang, Xueyang Yu, Delin Chen, Maohao Shen, Chuang Gan  

**Link**: [PDF](https://arxiv.org/pdf/2506.17218)  

**Abstract**: Vision-language models (VLMs) excel at multimodal understanding, yet their text-only decoding forces them to verbalize visual reasoning, limiting performance on tasks that demand visual imagination. Recent attempts train VLMs to render explicit images, but the heavy image-generation pre-training often hinders the reasoning ability. Inspired by the way humans reason with mental imagery-the internal construction and manipulation of visual cues-we investigate whether VLMs can reason through interleaved multimodal trajectories without producing explicit images. To this end, we present a Machine Mental Imagery framework, dubbed as Mirage, which augments VLM decoding with latent visual tokens alongside ordinary text. Concretely, whenever the model chooses to ``think visually'', it recasts its hidden states as next tokens, thereby continuing a multimodal trajectory without generating pixel-level images. Begin by supervising the latent tokens through distillation from ground-truth image embeddings, we then switch to text-only supervision to make the latent trajectory align tightly with the task objective. A subsequent reinforcement learning stage further enhances the multimodal reasoning capability. Experiments on diverse benchmarks demonstrate that Mirage unlocks stronger multimodal reasoning without explicit image generation. 

---
# Long-term Traffic Simulation with Interleaved Autoregressive Motion and Scenario Generation 

**Authors**: Xiuyu Yang, Shuhan Tan, Philipp Krähenbühl  

**Link**: [PDF](https://arxiv.org/pdf/2506.17213)  

**Abstract**: An ideal traffic simulator replicates the realistic long-term point-to-point trip that a self-driving system experiences during deployment. Prior models and benchmarks focus on closed-loop motion simulation for initial agents in a scene. This is problematic for long-term simulation. Agents enter and exit the scene as the ego vehicle enters new regions. We propose InfGen, a unified next-token prediction model that performs interleaved closed-loop motion simulation and scene generation. InfGen automatically switches between closed-loop motion simulation and scene generation mode. It enables stable long-term rollout simulation. InfGen performs at the state-of-the-art in short-term (9s) traffic simulation, and significantly outperforms all other methods in long-term (30s) simulation. The code and model of InfGen will be released at this https URL 

---
# Part$^{2}$GS: Part-aware Modeling of Articulated Objects using 3D Gaussian Splatting 

**Authors**: Tianjiao Yu, Vedant Shah, Muntasir Wahed, Ying Shen, Kiet A. Nguyen, Ismini Lourentzou  

**Link**: [PDF](https://arxiv.org/pdf/2506.17212)  

**Abstract**: Articulated objects are common in the real world, yet modeling their structure and motion remains a challenging task for 3D reconstruction methods. In this work, we introduce Part$^{2}$GS, a novel framework for modeling articulated digital twins of multi-part objects with high-fidelity geometry and physically consistent articulation. Part$^{2}$GS leverages a part-aware 3D Gaussian representation that encodes articulated components with learnable attributes, enabling structured, disentangled transformations that preserve high-fidelity geometry. To ensure physically consistent motion, we propose a motion-aware canonical representation guided by physics-based constraints, including contact enforcement, velocity consistency, and vector-field alignment. Furthermore, we introduce a field of repel points to prevent part collisions and maintain stable articulation paths, significantly improving motion coherence over baselines. Extensive evaluations on both synthetic and real-world datasets show that Part$^{2}$GS consistently outperforms state-of-the-art methods by up to 10$\times$ in Chamfer Distance for movable parts. 

---
# Dissecting the SWE-Bench Leaderboards: Profiling Submitters and Architectures of LLM- and Agent-Based Repair Systems 

**Authors**: Matias Martinez, Xavier Franch  

**Link**: [PDF](https://arxiv.org/pdf/2506.17208)  

**Abstract**: The rapid progress in Automated Program Repair (APR) has been driven by advances in AI, particularly large language models (LLMs) and agent-based systems. SWE-Bench is a recent benchmark designed to evaluate LLM-based repair systems using real issues and pull requests mined from 12 popular open-source Python repositories. Its public leaderboards, SWE-Bench Lite and SWE-Bench Verified, have become central platforms for tracking progress and comparing solutions. However, because the submission process does not require detailed documentation, the architectural design and origin of many solutions remain unclear. In this paper, we present the first comprehensive study of all submissions to the SWE-Bench Lite (68 entries) and Verified (79 entries) leaderboards, analyzing 67 unique approaches across dimensions such as submitter type, product availability, LLM usage, and system architecture. Our findings reveal the dominance of proprietary LLMs (especially Claude 3.5/3.7), the presence of both agentic and non-agentic designs, and a contributor base spanning from individual developers to large tech companies. 

---
# Network Sparsity Unlocks the Scaling Potential of Deep Reinforcement Learning 

**Authors**: Guozheng Ma, Lu Li, Zilin Wang, Li Shen, Pierre-Luc Bacon, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.17204)  

**Abstract**: Effectively scaling up deep reinforcement learning models has proven notoriously difficult due to network pathologies during training, motivating various targeted interventions such as periodic reset and architectural advances such as layer normalization. Instead of pursuing more complex modifications, we show that introducing static network sparsity alone can unlock further scaling potential beyond their dense counterparts with state-of-the-art architectures. This is achieved through simple one-shot random pruning, where a predetermined percentage of network weights are randomly removed once before training. Our analysis reveals that, in contrast to naively scaling up dense DRL networks, such sparse networks achieve both higher parameter efficiency for network expressivity and stronger resistance to optimization challenges like plasticity loss and gradient interference. We further extend our evaluation to visual and streaming RL scenarios, demonstrating the consistent benefits of network sparsity. 

---
# Facial Landmark Visualization and Emotion Recognition Through Neural Networks 

**Authors**: Israel Juárez-Jiménez, Tiffany Guadalupe Martínez Paredes, Jesús García-Ramírez, Eric Ramos Aguilar  

**Link**: [PDF](https://arxiv.org/pdf/2506.17191)  

**Abstract**: Emotion recognition from facial images is a crucial task in human-computer interaction, enabling machines to learn human emotions through facial expressions. Previous studies have shown that facial images can be used to train deep learning models; however, most of these studies do not include a through dataset analysis. Visualizing facial landmarks can be challenging when extracting meaningful dataset insights; to address this issue, we propose facial landmark box plots, a visualization technique designed to identify outliers in facial datasets. Additionally, we compare two sets of facial landmark features: (i) the landmarks' absolute positions and (ii) their displacements from a neutral expression to the peak of an emotional expression. Our results indicate that a neural network achieves better performance than a random forest classifier. 

---
# Towards AI Search Paradigm 

**Authors**: Yuchen Li, Hengyi Cai, Rui Kong, Xinran Chen, Jiamin Chen, Jun Yang, Haojie Zhang, Jiayi Li, Jiayi Wu, Yiqun Chen, Changle Qu, Keyi Kong, Wenwen Ye, Lixin Su, Xinyu Ma, Long Xia, Daiting Shi, Jiashu Zhao, Haoyi Xiong, Shuaiqiang Wang, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2506.17188)  

**Abstract**: In this paper, we introduce the AI Search Paradigm, a comprehensive blueprint for next-generation search systems capable of emulating human information processing and decision-making. The paradigm employs a modular architecture of four LLM-powered agents (Master, Planner, Executor and Writer) that dynamically adapt to the full spectrum of information needs, from simple factual queries to complex multi-stage reasoning tasks. These agents collaborate dynamically through coordinated workflows to evaluate query complexity, decompose problems into executable plans, and orchestrate tool usage, task execution, and content synthesis. We systematically present key methodologies for realizing this paradigm, including task planning and tool integration, execution strategies, aligned and robust retrieval-augmented generation, and efficient LLM inference, spanning both algorithmic techniques and infrastructure-level optimizations. By providing an in-depth guide to these foundational components, this work aims to inform the development of trustworthy, adaptive, and scalable AI search systems. 

---
# Continual Learning with Columnar Spiking Neural Networks 

**Authors**: Denis Larionov, Nikolay Bazenkov, Mikhail Kiselev  

**Link**: [PDF](https://arxiv.org/pdf/2506.17169)  

**Abstract**: This study investigates columnar-organized spiking neural networks (SNNs) for continual learning and catastrophic forgetting. Using CoLaNET (Columnar Layered Network), we show that microcolumns adapt most efficiently to new tasks when they lack shared structure with prior learning. We demonstrate how CoLaNET hyperparameters govern the trade-off between retaining old knowledge (stability) and acquiring new information (plasticity). Our optimal configuration learns ten sequential MNIST tasks effectively, maintaining 92% accuracy on each. It shows low forgetting, with only 4% performance degradation on the first task after training on nine subsequent tasks. 

---
# Proportional Sensitivity in Generative Adversarial Network (GAN)-Augmented Brain Tumor Classification Using Convolutional Neural Network 

**Authors**: Mahin Montasir Afif, Abdullah Al Noman, K. M. Tahsin Kabir, Md. Mortuza Ahmmed, Md. Mostafizur Rahman, Mufti Mahmud, Md. Ashraful Babu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17165)  

**Abstract**: Generative Adversarial Networks (GAN) have shown potential in expanding limited medical imaging datasets. This study explores how different ratios of GAN-generated and real brain tumor MRI images impact the performance of a CNN in classifying healthy vs. tumorous scans. A DCGAN was used to create synthetic images which were mixed with real ones at various ratios to train a custom CNN. The CNN was then evaluated on a separate real-world test set. Our results indicate that the model maintains high sensitivity and precision in tumor classification, even when trained predominantly on synthetic data. When only a small portion of GAN data was added, such as 900 real images and 100 GAN images, the model achieved excellent performance, with test accuracy reaching 95.2%, and precision, recall, and F1-score all exceeding 95%. However, as the proportion of GAN images increased further, performance gradually declined. This study suggests that while GANs are useful for augmenting limited datasets especially when real data is scarce, too much synthetic data can introduce artifacts that affect the model's ability to generalize to real world cases. 

---
# Sparse-Reg: Improving Sample Complexity in Offline Reinforcement Learning using Sparsity 

**Authors**: Samin Yeasar Arnob, Scott Fujimoto, Doina Precup  

**Link**: [PDF](https://arxiv.org/pdf/2506.17155)  

**Abstract**: In this paper, we investigate the use of small datasets in the context of offline reinforcement learning (RL). While many common offline RL benchmarks employ datasets with over a million data points, many offline RL applications rely on considerably smaller datasets. We show that offline RL algorithms can overfit on small datasets, resulting in poor performance. To address this challenge, we introduce "Sparse-Reg": a regularization technique based on sparsity to mitigate overfitting in offline reinforcement learning, enabling effective learning in limited data settings and outperforming state-of-the-art baselines in continuous control. 

---
# Do We Need Large VLMs for Spotting Soccer Actions? 

**Authors**: Ritabrata Chakraborty, Rajatsubhra Chakraborty, Avijit Dasgupta, Sandeep Chaurasia  

**Link**: [PDF](https://arxiv.org/pdf/2506.17144)  

**Abstract**: Traditional video-based tasks like soccer action spotting rely heavily on visual inputs, often requiring complex and computationally expensive models to process dense video data. In this work, we propose a shift from this video-centric approach to a text-based task, making it lightweight and scalable by utilizing Large Language Models (LLMs) instead of Vision-Language Models (VLMs). We posit that expert commentary, which provides rich, fine-grained descriptions and contextual cues such as excitement and tactical insights, contains enough information to reliably spot key actions in a match. To demonstrate this, we use the SoccerNet Echoes dataset, which provides timestamped commentary, and employ a system of three LLMs acting as judges specializing in outcome, excitement, and tactics. Each LLM evaluates sliding windows of commentary to identify actions like goals, cards, and substitutions, generating accurate timestamps for these events. Our experiments show that this language-centric approach performs effectively in detecting critical match events, providing a lightweight and training-free alternative to traditional video-based methods for action spotting. 

---
# MeDi: Metadata-Guided Diffusion Models for Mitigating Biases in Tumor Classification 

**Authors**: David Jacob Drexlin, Jonas Dippel, Julius Hense, Niklas Prenißl, Grégoire Montavon, Frederick Klauschen, Klaus-Robert Müller  

**Link**: [PDF](https://arxiv.org/pdf/2506.17140)  

**Abstract**: Deep learning models have made significant advances in histological prediction tasks in recent years. However, for adaptation in clinical practice, their lack of robustness to varying conditions such as staining, scanner, hospital, and demographics is still a limiting factor: if trained on overrepresented subpopulations, models regularly struggle with less frequent patterns, leading to shortcut learning and biased predictions. Large-scale foundation models have not fully eliminated this issue. Therefore, we propose a novel approach explicitly modeling such metadata into a Metadata-guided generative Diffusion model framework (MeDi). MeDi allows for a targeted augmentation of underrepresented subpopulations with synthetic data, which balances limited training data and mitigates biases in downstream models. We experimentally show that MeDi generates high-quality histopathology images for unseen subpopulations in TCGA, boosts the overall fidelity of the generated images, and enables improvements in performance for downstream classifiers on datasets with subpopulation shifts. Our work is a proof-of-concept towards better mitigating data biases with generative models. 

---
# Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models 

**Authors**: Michael Plainer, Hao Wu, Leon Klein, Stephan Günnemann, Frank Noé  

**Link**: [PDF](https://arxiv.org/pdf/2506.17139)  

**Abstract**: Diffusion models have recently gained significant attention due to their effectiveness in various scientific domains, including biochemistry. When trained on equilibrium molecular distributions, diffusion models provide both: a generative procedure to sample equilibrium conformations and associated forces derived from the model's scores. However, using the forces for coarse-grained molecular dynamics simulations uncovers inconsistencies in the samples generated via classical diffusion inference and simulation, despite both originating from the same model. Particularly at the small diffusion timesteps required for simulations, diffusion models fail to satisfy the Fokker-Planck equation, which governs how the score should evolve over time. We interpret this deviation as an indication of the observed inconsistencies and propose an energy-based diffusion model with a Fokker-Planck-derived regularization term enforcing consistency. We demonstrate the effectiveness of our approach on toy systems, alanine dipeptide, and introduce a state-of-the-art transferable Boltzmann emulator for dipeptides that supports simulation and demonstrates enhanced consistency and efficient sampling. 

---
# Robust Training with Data Augmentation for Medical Imaging Classification 

**Authors**: Josué Martínez-Martínez, Olivia Brown, Mostafa Karami, Sheida Nabavi  

**Link**: [PDF](https://arxiv.org/pdf/2506.17133)  

**Abstract**: Deep neural networks are increasingly being used to detect and diagnose medical conditions using medical imaging. Despite their utility, these models are highly vulnerable to adversarial attacks and distribution shifts, which can affect diagnostic reliability and undermine trust among healthcare professionals. In this study, we propose a robust training algorithm with data augmentation (RTDA) to mitigate these vulnerabilities in medical image classification. We benchmark classifier robustness against adversarial perturbations and natural variations of RTDA and six competing baseline techniques, including adversarial training and data augmentation approaches in isolation and combination, using experimental data sets with three different imaging technologies (mammograms, X-rays, and ultrasound). We demonstrate that RTDA achieves superior robustness against adversarial attacks and improved generalization performance in the presence of distribution shift in each image classification task while maintaining high clean accuracy. 

---
# Rapid and Continuous Trust Evaluation for Effective Task Collaboration Through Siamese Model 

**Authors**: Botao Zhu, Xianbin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17128)  

**Abstract**: Trust is emerging as an effective tool to ensure the successful completion of collaborative tasks within collaborative systems. However, rapidly and continuously evaluating the trustworthiness of collaborators during task execution is a significant challenge due to distributed devices, complex operational environments, and dynamically changing resources. To tackle this challenge, this paper proposes a Siamese-enabled rapid and continuous trust evaluation framework (SRCTE) to facilitate effective task collaboration. First, the communication and computing resource attributes of the collaborator in a trusted state, along with historical collaboration data, are collected and represented using an attributed control flow graph (ACFG) that captures trust-related semantic information and serves as a reference for comparison with data collected during task execution. At each time slot of task execution, the collaborator's communication and computing resource attributes, as well as task completion effectiveness, are collected in real time and represented with an ACFG to convey their trust-related semantic information. A Siamese model, consisting of two shared-parameter Structure2vec networks, is then employed to learn the deep semantics of each pair of ACFGs and generate their embeddings. Finally, the similarity between the embeddings of each pair of ACFGs is calculated to determine the collaborator's trust value at each time slot. A real system is built using two Dell EMC 5200 servers and a Google Pixel 8 to test the effectiveness of the proposed SRCTE framework. Experimental results demonstrate that SRCTE converges rapidly with only a small amount of data and achieves a high anomaly trust detection rate compared to the baseline algorithm. 

---
# MEXA: Towards General Multimodal Reasoning with Dynamic Multi-Expert Aggregation 

**Authors**: Shoubin Yu, Yue Zhang, Ziyang Wang, Jaehong Yoon, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2506.17113)  

**Abstract**: Combining pre-trained expert models offers substantial potential for scalable multimodal reasoning, but building a unified framework remains challenging due to the increasing diversity of input modalities and task complexity. For instance, medical diagnosis requires precise reasoning over structured clinical tables, while financial forecasting depends on interpreting plot-based data to make informed predictions. To tackle this challenge, we introduce MEXA, a training-free framework that performs modality- and task-aware aggregation of multiple expert models to enable effective multimodal reasoning across diverse and distinct domains. MEXA dynamically selects expert models based on the input modality and the task-specific reasoning demands (i.e., skills). Each expert model, specialized in a modality task pair, generates interpretable textual reasoning outputs. MEXA then aggregates and reasons over these outputs using a Large Reasoning Model (LRM) to produce the final answer. This modular design allows flexible and transparent multimodal reasoning across diverse domains without additional training overhead. We extensively evaluate our approach on diverse multimodal benchmarks, including Video Reasoning, Audio Reasoning, 3D Understanding, and Medical QA. MEXA consistently delivers performance improvements over strong multimodal baselines, highlighting the effectiveness and broad applicability of our expert-driven selection and aggregation in diverse multimodal reasoning tasks. 

---
# TransDreamerV3: Implanting Transformer In DreamerV3 

**Authors**: Shruti Sadanand Dongare, Amun Kharel, Jonathan Samuel, Xiaona Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.17103)  

**Abstract**: This paper introduces TransDreamerV3, a reinforcement learning model that enhances the DreamerV3 architecture by integrating a transformer encoder. The model is designed to improve memory and decision-making capabilities in complex environments. We conducted experiments on Atari-Boxing, Atari-Freeway, Atari-Pong, and Crafter tasks, where TransDreamerV3 demonstrated improved performance over DreamerV3, particularly in the Atari-Freeway and Crafter tasks. While issues in the Minecraft task and limited training across all tasks were noted, TransDreamerV3 displays advancement in world model-based reinforcement learning, leveraging transformer architectures. 

---
# Identifiability of Deep Polynomial Neural Networks 

**Authors**: Konstantin Usevich, Clara Dérand, Ricardo Borsoi, Marianne Clausel  

**Link**: [PDF](https://arxiv.org/pdf/2506.17093)  

**Abstract**: Polynomial Neural Networks (PNNs) possess a rich algebraic and geometric structure. However, their identifiability -- a key property for ensuring interpretability -- remains poorly understood. In this work, we present a comprehensive analysis of the identifiability of deep PNNs, including architectures with and without bias terms. Our results reveal an intricate interplay between activation degrees and layer widths in achieving identifiability. As special cases, we show that architectures with non-increasing layer widths are generically identifiable under mild conditions, while encoder-decoder networks are identifiable when the decoder widths do not grow too rapidly. Our proofs are constructive and center on a connection between deep PNNs and low-rank tensor decompositions, and Kruskal-type uniqueness theorems. This yields both generic conditions determined by the architecture, and effective conditions that depend on the network's parameters. We also settle an open conjecture on the expected dimension of PNN's neurovarieties, and provide new bounds on the activation degrees required for it to reach its maximum. 

---
# Tower+: Bridging Generality and Translation Specialization in Multilingual LLMs 

**Authors**: Ricardo Rei, Nuno M. Guerreiro, José Pombal, João Alves, Pedro Teixeirinha, Amin Farajian, André F. T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2506.17080)  

**Abstract**: Fine-tuning pretrained LLMs has been shown to be an effective strategy for reaching state-of-the-art performance on specific tasks like machine translation. However, this process of adaptation often implies sacrificing general-purpose capabilities, such as conversational reasoning and instruction-following, hampering the utility of the system in real-world applications that require a mixture of skills. In this paper, we introduce Tower+, a suite of models designed to deliver strong performance across both translation and multilingual general-purpose text capabilities. We achieve a Pareto frontier between translation specialization and multilingual general-purpose capabilities by introducing a novel training recipe that builds on Tower (Alves et al., 2024), comprising continued pretraining, supervised fine-tuning, preference optimization, and reinforcement learning with verifiable rewards. At each stage of training, we carefully generate and curate data to strengthen performance on translation as well as general-purpose tasks involving code generation, mathematics problem solving, and general instruction-following. We develop models at multiple scales: 2B, 9B, and 72B. Our smaller models often outperform larger general-purpose open-weight and proprietary LLMs (e.g., Llama 3.3 70B, GPT-4o). Our largest model delivers best-in-class translation performance for high-resource languages and top results in multilingual Arena Hard evaluations and in IF-MT, a benchmark we introduce for evaluating both translation and instruction-following. Our findings highlight that it is possible to rival frontier models in general capabilities, while optimizing for specific business domains, such as translation and localization. 

---
# LLM-Based Bot Broadens the Range of Arguments in Online Discussions, Even When Transparently Disclosed as AI 

**Authors**: Valeria Vuk, Cristina Sarasua, Fabrizio Gilardi  

**Link**: [PDF](https://arxiv.org/pdf/2506.17073)  

**Abstract**: A wide range of participation is essential for democracy, as it helps prevent the dominance of extreme views, erosion of legitimacy, and political polarization. However, engagement in online political discussions often features a limited spectrum of views due to high levels of self-selection and the tendency of online platforms to facilitate exchanges primarily among like-minded individuals. This study examines whether an LLM-based bot can widen the scope of perspectives expressed by participants in online discussions through two pre-registered randomized experiments conducted in a chatroom. We evaluate the impact of a bot that actively monitors discussions, identifies missing arguments, and introduces them into the conversation. The results indicate that our bot significantly expands the range of arguments, as measured by both objective and subjective metrics. Furthermore, disclosure of the bot as AI does not significantly alter these effects. These findings suggest that LLM-based moderation tools can positively influence online political discourse. 

---
# Flow-Based Non-stationary Temporal Regime Causal Structure Learning 

**Authors**: Abdellah Rahmani, Pascal Frossard  

**Link**: [PDF](https://arxiv.org/pdf/2506.17065)  

**Abstract**: Understanding causal relationships in multivariate time series is crucial in many scenarios, such as those dealing with financial or neurological data. Many such time series exhibit multiple regimes, i.e., consecutive temporal segments with a priori unknown boundaries, with each regime having its own causal structure. Inferring causal dependencies and regime shifts is critical for analyzing the underlying processes. However, causal structure learning in this setting is challenging due to (1) non stationarity, i.e., each regime can have its own causal graph and mixing function, and (2) complex noise distributions, which may be non Gaussian or heteroscedastic. Existing causal discovery approaches cannot address these challenges, since generally assume stationarity or Gaussian noise with constant variance. Hence, we introduce FANTOM, a unified framework for causal discovery that handles non stationary processes along with non Gaussian and heteroscedastic noises. FANTOM simultaneously infers the number of regimes and their corresponding indices and learns each regime's Directed Acyclic Graph. It uses a Bayesian Expectation Maximization algorithm that maximizes the evidence lower bound of the data log likelihood. On the theoretical side, we prove, under mild assumptions, that temporal heteroscedastic causal models, introduced in FANTOM's formulation, are identifiable in both stationary and non stationary settings. In addition, extensive experiments on synthetic and real data show that FANTOM outperforms existing methods. 

---
# From Concepts to Components: Concept-Agnostic Attention Module Discovery in Transformers 

**Authors**: Jingtong Su, Julia Kempe, Karen Ullrich  

**Link**: [PDF](https://arxiv.org/pdf/2506.17052)  

**Abstract**: Transformers have achieved state-of-the-art performance across language and vision tasks. This success drives the imperative to interpret their internal mechanisms with the dual goals of enhancing performance and improving behavioral control. Attribution methods help advance interpretability by assigning model outputs associated with a target concept to specific model components. Current attribution research primarily studies multi-layer perceptron neurons and addresses relatively simple concepts such as factual associations (e.g., Paris is located in France). This focus tends to overlook the impact of the attention mechanism and lacks a unified approach for analyzing more complex concepts. To fill these gaps, we introduce Scalable Attention Module Discovery (SAMD), a concept-agnostic method for mapping arbitrary, complex concepts to specific attention heads of general transformer models. We accomplish this by representing each concept as a vector, calculating its cosine similarity with each attention head, and selecting the TopK-scoring heads to construct the concept-associated attention module. We then propose Scalar Attention Module Intervention (SAMI), a simple strategy to diminish or amplify the effects of a concept by adjusting the attention module using only a single scalar parameter. Empirically, we demonstrate SAMD on concepts of varying complexity, and visualize the locations of their corresponding modules. Our results demonstrate that module locations remain stable before and after LLM post-training, and confirm prior work on the mechanics of LLM multilingualism. Through SAMI, we facilitate jailbreaking on HarmBench (+72.7%) by diminishing "safety" and improve performance on the GSM8K benchmark (+1.6%) by amplifying "reasoning". Lastly, we highlight the domain-agnostic nature of our approach by suppressing the image classification accuracy of vision transformers on ImageNet. 

---
# MAWIFlow Benchmark: Realistic Flow-Based Evaluation for Network Intrusion Detection 

**Authors**: Joshua Schraven, Alexander Windmann, Oliver Niggemann  

**Link**: [PDF](https://arxiv.org/pdf/2506.17041)  

**Abstract**: Benchmark datasets for network intrusion detection commonly rely on synthetically generated traffic, which fails to reflect the statistical variability and temporal drift encountered in operational environments. This paper introduces MAWIFlow, a flow-based benchmark derived from the MAWILAB v1.1 dataset, designed to enable realistic and reproducible evaluation of anomaly detection methods. A reproducible preprocessing pipeline is presented that transforms raw packet captures into flow representations conforming to the CICFlowMeter format, while preserving MAWILab's original anomaly labels. The resulting datasets comprise temporally distinct samples from January 2011, 2016, and 2021, drawn from trans-Pacific backbone traffic.
To establish reference baselines, traditional machine learning methods, including Decision Trees, Random Forests, XGBoost, and Logistic Regression, are compared to a deep learning model based on a CNN-BiLSTM architecture. Empirical results demonstrate that tree-based classifiers perform well on temporally static data but experience significant performance degradation over time. In contrast, the CNN-BiLSTM model maintains better performance, thus showing improved generalization. These findings underscore the limitations of synthetic benchmarks and static models, and motivate the adoption of realistic datasets with explicit temporal structure. All datasets, pipeline code, and model implementations are made publicly available to foster transparency and reproducibility. 

---
# LSCD: Lomb-Scargle Conditioned Diffusion for Time series Imputation 

**Authors**: Elizabeth Fons, Alejandro Sztrajman, Yousef El-Laham, Luciana Ferrer, Svitlana Vyetrenko, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2506.17039)  

**Abstract**: Time series with missing or irregularly sampled data are a persistent challenge in machine learning. Many methods operate on the frequency-domain, relying on the Fast Fourier Transform (FFT) which assumes uniform sampling, therefore requiring prior interpolation that can distort the spectra. To address this limitation, we introduce a differentiable Lomb--Scargle layer that enables a reliable computation of the power spectrum of irregularly sampled data. We integrate this layer into a novel score-based diffusion model (LSCD) for time series imputation conditioned on the entire signal spectrum. Experiments on synthetic and real-world benchmarks demonstrate that our method recovers missing data more accurately than purely time-domain baselines, while simultaneously producing consistent frequency estimates. Crucially, our method can be easily integrated into learning frameworks, enabling broader adoption of spectral guidance in machine learning approaches involving incomplete or irregular data. 

---
# Instituto de Telecomunicações at IWSLT 2025: Aligning Small-Scale Speech and Language Models for Speech-to-Text Learning 

**Authors**: Giuseppe Attanasio, Sonal Sannigrahi, Ben Peters, André F. T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2506.17019)  

**Abstract**: This paper presents the IT-IST submission to the IWSLT 2025 Shared Task on Instruction Following Speech Processing. We submit results for the Short Track, i.e., speech recognition, translation, and spoken question answering. Our model is a unified speech-to-text model that integrates a pre-trained continuous speech encoder and text decoder through a first phase of modality alignment and a second phase of instruction fine-tuning. Crucially, we focus on using small-scale language model backbones (< 2B) and restrict to high-quality, CC-BY data along with synthetic data generation to supplement existing resources. 

---
# TeXpert: A Multi-Level Benchmark for Evaluating LaTeX Code Generation by LLMs 

**Authors**: Sahil Kale, Vijaykant Nadadur  

**Link**: [PDF](https://arxiv.org/pdf/2506.16990)  

**Abstract**: LaTeX's precision and flexibility in typesetting have made it the gold standard for the preparation of scientific documentation. Large Language Models (LLMs) present a promising opportunity for researchers to produce publication-ready material using LaTeX with natural language instructions, yet current benchmarks completely lack evaluation of this ability. By introducing TeXpert, our benchmark dataset with natural language prompts for generating LaTeX code focused on components of scientific documents across multiple difficulty levels, we conduct an in-depth analysis of LLM performance in this regard and identify frequent error types. Our evaluation across open and closed-source LLMs highlights multiple key findings: LLMs excelling on standard benchmarks perform poorly in LaTeX generation with a significant accuracy drop-off as the complexity of tasks increases; open-source models like DeepSeek v3 and DeepSeek Coder strongly rival closed-source counterparts in LaTeX tasks; and formatting and package errors are unexpectedly prevalent, suggesting a lack of diverse LaTeX examples in the training datasets of most LLMs. Our dataset, code, and model evaluations are available at this https URL. 

---
# Language Bottleneck Models: A Framework for Interpretable Knowledge Tracing and Beyond 

**Authors**: Antonin Berthon, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2506.16982)  

**Abstract**: Accurately assessing student knowledge is critical for effective education, yet traditional Knowledge Tracing (KT) methods rely on opaque latent embeddings, limiting interpretability. Even LLM-based approaches generate direct predictions or summaries that may hallucinate without any accuracy guarantees. We recast KT as an inverse problem: learning the minimum natural-language summary that makes past answers explainable and future answers predictable. Our Language Bottleneck Model (LBM) consists of an encoder LLM that writes an interpretable knowledge summary and a frozen decoder LLM that must reconstruct and predict student responses using only that summary text. By constraining all predictive information to pass through a short natural-language bottleneck, LBMs ensure that the summary contains accurate information while remaining human-interpretable. Experiments on synthetic arithmetic benchmarks and the large-scale Eedi dataset show that LBMs rival the accuracy of state-of-the-art KT and direct LLM methods while requiring orders-of-magnitude fewer student trajectories. We demonstrate that training the encoder with group-relative policy optimization, using downstream decoding accuracy as a reward signal, effectively improves summary quality. 

---
# Latent Concept Disentanglement in Transformer-based Language Models 

**Authors**: Guan Zhe Hong, Bhavya Vasudeva, Vatsal Sharan, Cyrus Rashtchian, Prabhakar Raghavan, Rina Panigrahy  

**Link**: [PDF](https://arxiv.org/pdf/2506.16975)  

**Abstract**: When large language models (LLMs) use in-context learning (ICL) to solve a new task, they seem to grasp not only the goal of the task but also core, latent concepts in the demonstration examples. This begs the question of whether transformers represent latent structures as part of their computation or whether they take shortcuts to solve the problem. Prior mechanistic work on ICL does not address this question because it does not sufficiently examine the relationship between the learned representation and the latent concept, and the considered problem settings often involve only single-step reasoning. In this work, we examine how transformers disentangle and use latent concepts. We show that in 2-hop reasoning tasks with a latent, discrete concept, the model successfully identifies the latent concept and does step-by-step concept composition. In tasks parameterized by a continuous latent concept, we find low-dimensional subspaces in the representation space where the geometry mimics the underlying parameterization. Together, these results refine our understanding of ICL and the representation of transformers, and they provide evidence for highly localized structures in the model that disentangle latent concepts in ICL tasks. 

---
# Formal Control for Uncertain Systems via Contract-Based Probabilistic Surrogates (Extended Version) 

**Authors**: Oliver Schön, Sofie Haesaert, Sadegh Soudjani  

**Link**: [PDF](https://arxiv.org/pdf/2506.16971)  

**Abstract**: The requirement for identifying accurate system representations has not only been a challenge to fulfill, but it has compromised the scalability of formal methods, as the resulting models are often too complex for effective decision making with formal correctness and performance guarantees. Focusing on probabilistic simulation relations and surrogate models of stochastic systems, we propose an approach that significantly enhances the scalability and practical applicability of such simulation relations by eliminating the need to compute error bounds directly. As a result, we provide an abstraction-based technique that scales effectively to higher dimensions while addressing complex nonlinear agent-environment interactions with infinite-horizon temporal logic guarantees amidst uncertainty. Our approach trades scalability for conservatism favorably, as demonstrated on a complex high-dimensional vehicle intersection case study. 

---
# Enhancing Step-by-Step and Verifiable Medical Reasoning in MLLMs 

**Authors**: Haoran Sun, Yankai Jiang, Wenjie Lou, Yujie Zhang, Wenjie Li, Lilong Wang, Mianxin Liu, Lei Liu, Xiaosong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16962)  

**Abstract**: Multimodal large language models (MLLMs) have begun to demonstrate robust reasoning capabilities on general tasks, yet their application in the medical domain remains in its early stages. Constructing chain-of-thought (CoT) training data is essential for bolstering the reasoning abilities of medical MLLMs. However, existing approaches exhibit a deficiency in offering a comprehensive framework for searching and evaluating effective reasoning paths towards critical diagnosis. To address this challenge, we propose Mentor-Intern Collaborative Search (MICS), a novel reasoning-path searching scheme to generate rigorous and effective medical CoT data. MICS first leverages mentor models to initialize the reasoning, one step at a time, then prompts each intern model to continue the thinking along those initiated paths, and finally selects the optimal reasoning path according to the overall reasoning performance of multiple intern models. The reasoning performance is determined by an MICS-Score, which assesses the quality of generated reasoning paths. Eventually, we construct MMRP, a multi-task medical reasoning dataset with ranked difficulty, and Chiron-o1, a new medical MLLM devised via a curriculum learning strategy, with robust visual question-answering and generalizable reasoning capabilities. Extensive experiments demonstrate that Chiron-o1, trained on our CoT dataset constructed using MICS, achieves state-of-the-art performance across a list of medical visual question answering and reasoning benchmarks. Codes are available at GitHub - manglu097/Chiron-o1: Enhancing Step-by-Step and Verifiable Medical Reasoning in MLLMs 

---
# A deep learning and machine learning approach to predict neonatal death in the context of São Paulo 

**Authors**: Mohon Raihan, Plabon Kumar Saha, Rajan Das Gupta, A Z M Tahmidul Kabir, Afia Anjum Tamanna, Md. Harun-Ur-Rashid, Adnan Bin Abdus Salam, Md Tanvir Anjum, A Z M Ahteshamul Kabir  

**Link**: [PDF](https://arxiv.org/pdf/2506.16929)  

**Abstract**: Neonatal death is still a concerning reality for underdeveloped and even some developed countries. Worldwide data indicate that 26.693 babies out of 1,000 births die, according to Macro Trades. To reduce this number, early prediction of endangered babies is crucial. Such prediction enables the opportunity to take ample care of the child and mother so that early child death can be avoided. In this context, machine learning was used to determine whether a newborn baby is at risk. To train the predictive model, historical data of 1.4 million newborns was used. Machine learning and deep learning techniques such as logical regression, K-nearest neighbor, random forest classifier, extreme gradient boosting (XGBoost), convolutional neural network, and long short-term memory (LSTM) were implemented using the dataset to identify the most accurate model for predicting neonatal mortality. Among the machine learning algorithms, XGBoost and random forest classifier achieved the best accuracy with 94%, while among the deep learning models, LSTM delivered the highest accuracy with 99%. Therefore, using LSTM appears to be the most suitable approach to predict whether precautionary measures for a child are necessary. 

---
# Single-shot thermometry of simulated Bose--Einstein condensates using artificial intelligence 

**Authors**: Jack Griffiths, Steven A. Wrathmall, Simon A. Gardiner  

**Link**: [PDF](https://arxiv.org/pdf/2506.16925)  

**Abstract**: Precise determination of thermodynamic parameters in ultracold Bose gases remains challenging due to the destructive nature of conventional measurement techniques and inherent experimental uncertainties. We demonstrate an artificial intelligence approach for rapid, non-destructive estimation of the chemical potential and temperature from single-shot, in situ imaged density profiles of finite-temperature Bose gases. Our convolutional neural network is trained exclusively on quasi-2D `pancake' condensates in harmonic trap configurations. It achieves parameter extraction within fractions of a second. The model also demonstrates zero-shot generalisation across both trap geometry and thermalisation dynamics, successfully estimating thermodynamic parameters for toroidally trapped condensates with errors of only a few nanokelvin despite no prior exposure to such geometries during training, and maintaining predictive accuracy during dynamic thermalisation processes after a relatively brief evolution without explicit training on non-equilibrium states. These results suggest that supervised learning can overcome traditional limitations in ultracold atom thermometry, with extension to broader geometric configurations, temperature ranges, and additional parameters potentially enabling comprehensive real-time analysis of quantum gas experiments. Such capabilities could significantly streamline experimental workflows whilst improving measurement precision across a range of quantum fluid systems. 

---
# Towards Effective Complementary Security Analysis using Large Language Models 

**Authors**: Jonas Wagner, Simon Müller, Christian Näther, Jan-Philipp Steghöfer, Andreas Both  

**Link**: [PDF](https://arxiv.org/pdf/2506.16899)  

**Abstract**: A key challenge in security analysis is the manual evaluation of potential security weaknesses generated by static application security testing (SAST) tools. Numerous false positives (FPs) in these reports reduce the effectiveness of security analysis. We propose using Large Language Models (LLMs) to improve the assessment of SAST findings. We investigate the ability of LLMs to reduce FPs while trying to maintain a perfect true positive rate, using datasets extracted from the OWASP Benchmark (v1.2) and a real-world software project. Our results indicate that advanced prompting techniques, such as Chain-of-Thought and Self-Consistency, substantially improve FP detection. Notably, some LLMs identified approximately 62.5% of FPs in the OWASP Benchmark dataset without missing genuine weaknesses. Combining detections from different LLMs would increase this FP detection to approximately 78.9%. Additionally, we demonstrate our approach's generalizability using a real-world dataset covering five SAST tools, three programming languages, and infrastructure files. The best LLM detected 33.85% of all FPs without missing genuine weaknesses, while combining detections from different LLMs would increase this detection to 38.46%. Our findings highlight the potential of LLMs to complement traditional SAST tools, enhancing automation and reducing resources spent addressing false alarms. 

---
# With Limited Data for Multimodal Alignment, Let the STRUCTURE Guide You 

**Authors**: Fabian Gröger, Shuo Wen, Huyen Le, Maria Brbić  

**Link**: [PDF](https://arxiv.org/pdf/2506.16895)  

**Abstract**: Multimodal models have demonstrated powerful capabilities in complex tasks requiring multimodal alignment including zero-shot classification and cross-modal retrieval. However, existing models typically rely on millions of paired multimodal samples, which are prohibitively expensive or infeasible to obtain in many domains. In this work, we explore the feasibility of building multimodal models with limited amount of paired data by aligning pretrained unimodal foundation models. We show that high-quality alignment is possible with as few as tens of thousands of paired samples$\unicode{x2013}$less than $1\%$ of the data typically used in the field. To achieve this, we introduce STRUCTURE, an effective regularization technique that preserves the neighborhood geometry of the latent space of unimodal encoders. Additionally, we show that aligning last layers is often suboptimal and demonstrate the benefits of aligning the layers with the highest representational similarity across modalities. These two components can be readily incorporated into existing alignment methods, yielding substantial gains across 24 zero-shot image classification and retrieval benchmarks, with average relative improvement of $51.6\%$ in classification and $91.8\%$ in retrieval tasks. Our results highlight the effectiveness and broad applicability of our framework for limited-sample multimodal learning and offer a promising path forward for resource-constrained domains. 

---
# The Importance of Being Lazy: Scaling Limits of Continual Learning 

**Authors**: Jacopo Graldi, Alessandro Breccia, Giulia Lanzillotta, Thomas Hofmann, Lorenzo Noci  

**Link**: [PDF](https://arxiv.org/pdf/2506.16884)  

**Abstract**: Despite recent efforts, neural networks still struggle to learn in non-stationary environments, and our understanding of catastrophic forgetting (CF) is far from complete. In this work, we perform a systematic study on the impact of model scale and the degree of feature learning in continual learning. We reconcile existing contradictory observations on scale in the literature, by differentiating between lazy and rich training regimes through a variable parameterization of the architecture. We show that increasing model width is only beneficial when it reduces the amount of feature learning, yielding more laziness. Using the framework of dynamical mean field theory, we then study the infinite width dynamics of the model in the feature learning regime and characterize CF, extending prior theoretical results limited to the lazy regime. We study the intricate relationship between feature learning, task non-stationarity, and forgetting, finding that high feature learning is only beneficial with highly similar tasks. We identify a transition modulated by task similarity where the model exits an effectively lazy regime with low forgetting to enter a rich regime with significant forgetting. Finally, our findings reveal that neural networks achieve optimal performance at a critical level of feature learning, which depends on task non-stationarity and transfers across model scales. This work provides a unified perspective on the role of scale and feature learning in continual learning. 

---
# ParkFormer: A Transformer-Based Parking Policy with Goal Embedding and Pedestrian-Aware Control 

**Authors**: Jun Fu, Bin Tian, Haonan Chen, Shi Meng, Tingting Yao  

**Link**: [PDF](https://arxiv.org/pdf/2506.16856)  

**Abstract**: Autonomous parking plays a vital role in intelligent vehicle systems, particularly in constrained urban environments where high-precision control is required. While traditional rule-based parking systems struggle with environmental uncertainties and lack adaptability in crowded or dynamic scenes, human drivers demonstrate the ability to park intuitively without explicit modeling. Inspired by this observation, we propose a Transformer-based end-to-end framework for autonomous parking that learns from expert demonstrations. The network takes as input surround-view camera images, goal-point representations, ego vehicle motion, and pedestrian trajectories. It outputs discrete control sequences including throttle, braking, steering, and gear selection. A novel cross-attention module integrates BEV features with target points, and a GRU-based pedestrian predictor enhances safety by modeling dynamic obstacles. We validate our method on the CARLA 0.9.14 simulator in both vertical and parallel parking scenarios. Experiments show our model achieves a high success rate of 96.57\%, with average positional and orientation errors of 0.21 meters and 0.41 degrees, respectively. The ablation studies further demonstrate the effectiveness of key modules such as pedestrian prediction and goal-point attention fusion. The code and dataset will be released at: this https URL. 

---
# Bandwidth Selectors on Semiparametric Bayesian Networks 

**Authors**: Victor Alejandre, Concha Bielza, Pedro Larrañaga  

**Link**: [PDF](https://arxiv.org/pdf/2506.16844)  

**Abstract**: Semiparametric Bayesian networks (SPBNs) integrate parametric and non-parametric probabilistic models, offering flexibility in learning complex data distributions from samples. In particular, kernel density estimators (KDEs) are employed for the non-parametric component. Under the assumption of data normality, the normal rule is used to learn the bandwidth matrix for the KDEs in SPBNs. This matrix is the key hyperparameter that controls the trade-off between bias and variance. However, real-world data often deviates from normality, potentially leading to suboptimal density estimation and reduced predictive performance. This paper first establishes the theoretical framework for the application of state-of-the-art bandwidth selectors and subsequently evaluates their impact on SPBN performance. We explore the approaches of cross-validation and plug-in selectors, assessing their effectiveness in enhancing the learning capability and applicability of SPBNs. To support this investigation, we have extended the open-source package PyBNesian for SPBNs with the additional bandwidth selection techniques and conducted extensive experimental analyses. Our results demonstrate that the proposed bandwidth selectors leverage increasing information more effectively than the normal rule, which, despite its robustness, stagnates with more data. In particular, unbiased cross-validation generally outperforms the normal rule, highlighting its advantage in high sample size scenarios. 

---
# AnyTraverse: An off-road traversability framework with VLM and human operator in the loop 

**Authors**: Sattwik Sahu, Agamdeep Singh, Karthik Nambiar, Srikanth Saripalli, P.B. Sujit  

**Link**: [PDF](https://arxiv.org/pdf/2506.16826)  

**Abstract**: Off-road traversability segmentation enables autonomous navigation with applications in search-and-rescue, military operations, wildlife exploration, and agriculture. Current frameworks struggle due to significant variations in unstructured environments and uncertain scene changes, and are not adaptive to be used for different robot types. We present AnyTraverse, a framework combining natural language-based prompts with human-operator assistance to determine navigable regions for diverse robotic vehicles. The system segments scenes for a given set of prompts and calls the operator only when encountering previously unexplored scenery or unknown class not part of the prompt in its region-of-interest, thus reducing active supervision load while adapting to varying outdoor scenes. Our zero-shot learning approach eliminates the need for extensive data collection or retraining. Our experimental validation includes testing on RELLIS-3D, Freiburg Forest, and RUGD datasets and demonstrate real-world deployment on multiple robot platforms. The results show that AnyTraverse performs better than GA-NAV and Off-seg while offering a vehicle-agnostic approach to off-road traversability that balances automation with targeted human supervision. 

---
# Learning Dexterous Object Handover 

**Authors**: Daniel Frau-Alfaro, Julio Castaño-Amoros, Santiago Puente, Pablo Gil, Roberto Calandra  

**Link**: [PDF](https://arxiv.org/pdf/2506.16822)  

**Abstract**: Object handover is an important skill that we use daily when interacting with other humans. To deploy robots in collaborative setting, like houses, being able to receive and handing over objects safely and efficiently becomes a crucial skill. In this work, we demonstrate the use of Reinforcement Learning (RL) for dexterous object handover between two multi-finger hands. Key to this task is the use of a novel reward function based on dual quaternions to minimize the rotation distance, which outperforms other rotation representations such as Euler and rotation matrices. The robustness of the trained policy is experimentally evaluated by testing w.r.t. objects that are not included in the training distribution, and perturbations during the handover process. The results demonstrate that the trained policy successfully perform this task, achieving a total success rate of 94% in the best-case scenario after 100 experiments, thereby showing the robustness of our policy with novel objects. In addition, the best-case performance of the policy decreases by only 13.8% when the other robot moves during the handover, proving that our policy is also robust to this type of perturbation, which is common in real-world object handovers. 

---
# Loupe: A Generalizable and Adaptive Framework for Image Forgery Detection 

**Authors**: Yuchu Jiang, Jiaming Chu, Jian Zhao, Xin Zhang, Xu Yang, Lei Jin, Chi Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.16819)  

**Abstract**: The proliferation of generative models has raised serious concerns about visual content forgery. Existing deepfake detection methods primarily target either image-level classification or pixel-wise localization. While some achieve high accuracy, they often suffer from limited generalization across manipulation types or rely on complex architectures. In this paper, we propose Loupe, a lightweight yet effective framework for joint deepfake detection and localization. Loupe integrates a patch-aware classifier and a segmentation module with conditional queries, allowing simultaneous global authenticity classification and fine-grained mask prediction. To enhance robustness against distribution shifts of test set, Loupe introduces a pseudo-label-guided test-time adaptation mechanism by leveraging patch-level predictions to supervise the segmentation head. Extensive experiments on the DDL dataset demonstrate that Loupe achieves state-of-the-art performance, securing the first place in the IJCAI 2025 Deepfake Detection and Localization Challenge with an overall score of 0.846. Our results validate the effectiveness of the proposed patch-level fusion and conditional query design in improving both classification accuracy and spatial localization under diverse forgery patterns. The code is available at this https URL. 

---
# Robust Dynamic Material Handling via Adaptive Constrained Evolutionary Reinforcement Learning 

**Authors**: Chengpeng Hu, Ziming Wang, Bo Yuan, Jialin Liu, Chengqi Zhang, Xin Yao  

**Link**: [PDF](https://arxiv.org/pdf/2506.16795)  

**Abstract**: Dynamic material handling (DMH) involves the assignment of dynamically arriving material transporting tasks to suitable vehicles in real time for minimising makespan and tardiness. In real-world scenarios, historical task records are usually available, which enables the training of a decision policy on multiple instances consisting of historical records. Recently, reinforcement learning has been applied to solve DMH. Due to the occurrence of dynamic events such as new tasks, adaptability is highly required. Solving DMH is challenging since constraints including task delay should be satisfied. A feedback is received only when all tasks are served, which leads to sparse reward. Besides, making the best use of limited computational resources and historical records for training a robust policy is crucial. The time allocated to different problem instances would highly impact the learning process. To tackle those challenges, this paper proposes a novel adaptive constrained evolutionary reinforcement learning (ACERL) approach, which maintains a population of actors for diverse exploration. ACERL accesses each actor for tackling sparse rewards and constraint violation to restrict the behaviour of the policy. Moreover, ACERL adaptively selects the most beneficial training instances for improving the policy. Extensive experiments on eight training and eight unseen test instances demonstrate the outstanding performance of ACERL compared with several state-of-the-art algorithms. Policies trained by ACERL can schedule the vehicles while fully satisfying the constraints. Additional experiments on 40 unseen noised instances show the robust performance of ACERL. Cross-validation further presents the overall effectiveness of ACREL. Besides, a rigorous ablation study highlights the coordination and benefits of each ingredient of ACERL. 

---
# MIST: Jailbreaking Black-box Large Language Models via Iterative Semantic Tuning 

**Authors**: Muyang Zheng, Yuanzhi Yao, Changting Lin, Rui Wang, Meng Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.16792)  

**Abstract**: Despite efforts to align large language models (LLMs) with societal and moral values, these models remain susceptible to jailbreak attacks--methods designed to elicit harmful responses. Jailbreaking black-box LLMs is considered challenging due to the discrete nature of token inputs, restricted access to the target LLM, and limited query budget. To address the issues above, we propose an effective method for jailbreaking black-box large language Models via Iterative Semantic Tuning, named MIST. MIST enables attackers to iteratively refine prompts that preserve the original semantic intent while inducing harmful content. Specifically, to balance semantic similarity with computational efficiency, MIST incorporates two key strategies: sequential synonym search, and its advanced version--order-determining optimization. Extensive experiments across two open-source models and four closed-source models demonstrate that MIST achieves competitive attack success rates and attack transferability compared with other state-of-the-art white-box and black-box jailbreak methods. Additionally, we conduct experiments on computational efficiency to validate the practical viability of MIST. 

---
# TabArena: A Living Benchmark for Machine Learning on Tabular Data 

**Authors**: Nick Erickson, Lennart Purucker, Andrej Tschalzev, David Holzmüller, Prateek Mutalik Desai, and David Salinas, Frank Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2506.16791)  

**Abstract**: With the growing popularity of deep learning and foundation models for tabular data, the need for standardized and reliable benchmarks is higher than ever. However, current benchmarks are static. Their design is not updated even if flaws are discovered, model versions are updated, or new models are released. To address this, we introduce TabArena, the first continuously maintained living tabular benchmarking system. To launch TabArena, we manually curate a representative collection of datasets and well-implemented models, conduct a large-scale benchmarking study to initialize a public leaderboard, and assemble a team of experienced maintainers. Our results highlight the influence of validation method and ensembling of hyperparameter configurations to benchmark models at their full potential. While gradient-boosted trees are still strong contenders on practical tabular datasets, we observe that deep learning methods have caught up under larger time budgets with ensembling. At the same time, foundation models excel on smaller datasets. Finally, we show that ensembles across models advance the state-of-the-art in tabular machine learning and investigate the contributions of individual models. We launch TabArena with a public leaderboard, reproducible code, and maintenance protocols to create a living benchmark available at this https URL. 

---
# What Is the Point of Equality in Machine Learning Fairness? Beyond Equality of Opportunity 

**Authors**: Youjin Kong  

**Link**: [PDF](https://arxiv.org/pdf/2506.16782)  

**Abstract**: Fairness in machine learning (ML) has become a rapidly growing area of research. But why, in the first place, is unfairness in ML morally wrong? And why should we care about improving fairness? Most fair-ML research implicitly appeals to distributive equality: the idea that desirable goods and benefits, such as opportunities (e.g., Barocas et al., 2023), should be equally distributed across society. Unfair ML models, then, are seen as wrong because they unequally distribute such benefits. This paper argues that this exclusive focus on distributive equality offers an incomplete and potentially misleading ethical foundation. Grounding ML fairness in egalitarianism -- the view that equality is a fundamental moral and social ideal -- requires challenging structural inequality: systematic, institutional, and durable arrangements that privilege some groups while disadvantaging others. Structural inequality manifests through ML systems in two primary forms: allocative harms (e.g., economic loss) and representational harms (e.g., stereotypes, erasure). While distributive equality helps address allocative harms, it fails to explain why representational harms are wrong -- why it is wrong for ML systems to reinforce social hierarchies that stratify people into superior and inferior groups -- and why ML systems should aim to foster a society where people relate as equals (i.e., relational equality). To address these limitations, the paper proposes a multifaceted egalitarian framework for ML fairness that integrates both distributive and relational equality. Drawing on critical social and political philosophy, this framework offers a more comprehensive ethical foundation for tackling the full spectrum of harms perpetuated by ML systems. The paper also outlines practical pathways for implementing the framework across the ML pipeline. 

---
# PQCAD-DM: Progressive Quantization and Calibration-Assisted Distillation for Extremely Efficient Diffusion Model 

**Authors**: Beomseok Ko, Hyeryung Jang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16776)  

**Abstract**: Diffusion models excel in image generation but are computational and resource-intensive due to their reliance on iterative Markov chain processes, leading to error accumulation and limiting the effectiveness of naive compression techniques. In this paper, we propose PQCAD-DM, a novel hybrid compression framework combining Progressive Quantization (PQ) and Calibration-Assisted Distillation (CAD) to address these challenges. PQ employs a two-stage quantization with adaptive bit-width transitions guided by a momentum-based mechanism, reducing excessive weight perturbations in low-precision. CAD leverages full-precision calibration datasets during distillation, enabling the student to match full-precision performance even with a quantized teacher. As a result, PQCAD-DM achieves a balance between computational efficiency and generative quality, halving inference time while maintaining competitive performance. Extensive experiments validate PQCAD-DM's superior generative capabilities and efficiency across diverse datasets, outperforming fixed-bit quantization methods. 

---
# Language-Informed Synthesis of Rational Agent Models for Grounded Theory-of-Mind Reasoning On-The-Fly 

**Authors**: Lance Ying, Ryan Truong, Katherine M. Collins, Cedegao E. Zhang, Megan Wei, Tyler Brooke-Wilson, Tan Zhi-Xuan, Lionel Wong, Joshua B. Tenenbaum  

**Link**: [PDF](https://arxiv.org/pdf/2506.16755)  

**Abstract**: Drawing real world social inferences usually requires taking into account information from multiple modalities. Language is a particularly powerful source of information in social settings, especially in novel situations where language can provide both abstract information about the environment dynamics and concrete specifics about an agent that cannot be easily visually observed. In this paper, we propose Language-Informed Rational Agent Synthesis (LIRAS), a framework for drawing context-specific social inferences that integrate linguistic and visual inputs. LIRAS frames multimodal social reasoning as a process of constructing structured but situation-specific agent and environment representations - leveraging multimodal language models to parse language and visual inputs into unified symbolic representations, over which a Bayesian inverse planning engine can be run to produce granular probabilistic judgments. On a range of existing and new social reasoning tasks derived from cognitive science experiments, we find that our model (instantiated with a comparatively lightweight VLM) outperforms ablations and state-of-the-art models in capturing human judgments across all domains. 

---
# Metapath-based Hyperbolic Contrastive Learning for Heterogeneous Graph Embedding 

**Authors**: Jongmin Park, Seunghoon Han, Won-Yong Shin, Sungsu Lim  

**Link**: [PDF](https://arxiv.org/pdf/2506.16754)  

**Abstract**: The hyperbolic space, characterized by a constant negative curvature and exponentially expanding space, aligns well with the structural properties of heterogeneous graphs. However, although heterogeneous graphs inherently possess diverse power-law structures, most hyperbolic heterogeneous graph embedding models rely on a single hyperbolic space. This approach may fail to effectively capture the diverse power-law structures within heterogeneous graphs. To address this limitation, we propose a Metapath-based Hyperbolic Contrastive Learning framework (MHCL), which uses multiple hyperbolic spaces to capture diverse complex structures within heterogeneous graphs. Specifically, by learning each hyperbolic space to describe the distribution of complex structures corresponding to each metapath, it is possible to capture semantic information effectively. Since metapath embeddings represent distinct semantic information, preserving their discriminability is important when aggregating them to obtain node representations. Therefore, we use a contrastive learning approach to optimize MHCL and improve the discriminability of metapath embeddings. In particular, our contrastive learning method minimizes the distance between embeddings of the same metapath and maximizes the distance between those of different metapaths in hyperbolic space, thereby improving the separability of metapath embeddings with distinct semantic information. We conduct comprehensive experiments to evaluate the effectiveness of MHCL. The experimental results demonstrate that MHCL outperforms state-of-the-art baselines in various graph machine learning tasks, effectively capturing the complex structures of heterogeneous graphs. 

---
# Off-Policy Actor-Critic for Adversarial Observation Robustness: Virtual Alternative Training via Symmetric Policy Evaluation 

**Authors**: Kosuke Nakanishi, Akihiro Kubo, Yuji Yasui, Shin Ishii  

**Link**: [PDF](https://arxiv.org/pdf/2506.16753)  

**Abstract**: Recently, robust reinforcement learning (RL) methods designed to handle adversarial input observations have received significant attention, motivated by RL's inherent vulnerabilities. While existing approaches have demonstrated reasonable success, addressing worst-case scenarios over long time horizons requires both minimizing the agent's cumulative rewards for adversaries and training agents to counteract them through alternating learning. However, this process introduces mutual dependencies between the agent and the adversary, making interactions with the environment inefficient and hindering the development of off-policy methods. In this work, we propose a novel off-policy method that eliminates the need for additional environmental interactions by reformulating adversarial learning as a soft-constrained optimization problem. Our approach is theoretically supported by the symmetric property of policy evaluation between the agent and the adversary. The implementation is available at this https URL. 

---
# RapFlow-TTS: Rapid and High-Fidelity Text-to-Speech with Improved Consistency Flow Matching 

**Authors**: Hyun Joon Park, Jeongmin Liu, Jin Sob Kim, Jeong Yeol Yang, Sung Won Han, Eunwoo Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.16741)  

**Abstract**: We introduce RapFlow-TTS, a rapid and high-fidelity TTS acoustic model that leverages velocity consistency constraints in flow matching (FM) training. Although ordinary differential equation (ODE)-based TTS generation achieves natural-quality speech, it typically requires a large number of generation steps, resulting in a trade-off between quality and inference speed. To address this challenge, RapFlow-TTS enforces consistency in the velocity field along the FM-straightened ODE trajectory, enabling consistent synthetic quality with fewer generation steps. Additionally, we introduce techniques such as time interval scheduling and adversarial learning to further enhance the quality of the few-step synthesis. Experimental results show that RapFlow-TTS achieves high-fidelity speech synthesis with a 5- and 10-fold reduction in synthesis steps than the conventional FM- and score-based approaches, respectively. 

---
# LM-SPT: LM-Aligned Semantic Distillation for Speech Tokenization 

**Authors**: Daejin Jo, Jeeyoung Yun, Byungseok Roh, Sungwoong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.16738)  

**Abstract**: With the rapid progress of speech language models (SLMs), discrete speech tokens have emerged as a core interface between speech and text, enabling unified modeling across modalities. Recent speech tokenization approaches aim to isolate semantic information from low-level acoustics to better align with language models. In particular, previous methods use SSL teachers such as HuBERT to extract semantic representations, which are then distilled into a semantic quantizer to suppress acoustic redundancy as well as capture content-related latent structures. However, they still produce speech token sequences significantly longer than their textual counterparts, creating challenges for efficient speech-language modeling. Reducing the frame rate is a natural solution, but standard techniques, such as rigid average pooling across frames, can distort or dilute the semantic structure required for effective LM alignment. To address this, we propose LM-SPT, a speech tokenization method that introduces a novel semantic distillation. Instead of directly matching teacher and student features via pooling, we reconstruct speech solely from semantic tokens and minimize the discrepancy between the encoded representations of the original and reconstructed waveforms, obtained from a frozen automatic speech recognition (ASR) encoder. This indirect yet data-driven supervision enables the tokenizer to learn discrete units that are more semantically aligned with language models. LM-SPT further incorporates architectural improvements to the encoder and decoder for speech tokenization, and supports multiple frame rates, including 25Hz, 12.5Hz, and 6.25Hz. Experimental results show that LM-SPT achieves superior reconstruction fidelity compared to baselines, and that SLMs trained with LM-SPT tokens achieve competitive performances on speech-to-text and consistently outperform baselines on text-to-speech tasks. 

---
# On Training-Test (Mis)alignment in Unsupervised Combinatorial Optimization: Observation, Empirical Exploration, and Analysis 

**Authors**: Fanchen Bu, Kijung Shin  

**Link**: [PDF](https://arxiv.org/pdf/2506.16732)  

**Abstract**: In unsupervised combinatorial optimization (UCO), during training, one aims to have continuous decisions that are promising in a probabilistic sense for each training instance, which enables end-to-end training on initially discrete and non-differentiable problems. At the test time, for each test instance, starting from continuous decisions, derandomization is typically applied to obtain the final deterministic decisions. Researchers have developed more and more powerful test-time derandomization schemes to enhance the empirical performance and the theoretical guarantee of UCO methods. However, we notice a misalignment between training and testing in the existing UCO methods. Consequently, lower training losses do not necessarily entail better post-derandomization performance, even for the training instances without any data distribution shift. Empirically, we indeed observe such undesirable cases. We explore a preliminary idea to better align training and testing in UCO by including a differentiable version of derandomization into training. Our empirical exploration shows that such an idea indeed improves training-test alignment, but also introduces nontrivial challenges into training. 

---
# The Role of Model Confidence on Bias Effects in Measured Uncertainties 

**Authors**: Xinyi Liu, Weiguang Wang, Hangfeng He  

**Link**: [PDF](https://arxiv.org/pdf/2506.16724)  

**Abstract**: With the growing adoption of Large Language Models (LLMs) for open-ended tasks, accurately assessing epistemic uncertainty, which reflects a model's lack of knowledge, has become crucial to ensuring reliable outcomes. However, quantifying epistemic uncertainty in such tasks is challenging due to the presence of aleatoric uncertainty, which arises from multiple valid answers. While bias can introduce noise into epistemic uncertainty estimation, it may also reduce noise from aleatoric uncertainty. To investigate this trade-off, we conduct experiments on Visual Question Answering (VQA) tasks and find that mitigating prompt-introduced bias improves uncertainty quantification in GPT-4o. Building on prior work showing that LLMs tend to copy input information when model confidence is low, we further analyze how these prompt biases affect measured epistemic and aleatoric uncertainty across varying bias-free confidence levels with GPT-4o and Qwen2-VL. We find that all considered biases induce greater changes in both uncertainties when bias-free model confidence is lower. Moreover, lower bias-free model confidence leads to greater underestimation of epistemic uncertainty (i.e. overconfidence) due to bias, whereas it has no significant effect on the direction of changes in aleatoric uncertainty estimation. These distinct effects deepen our understanding of bias mitigation for uncertainty quantification and potentially inform the development of more advanced techniques. 

---
# TriCon-SF: A Triple-Shuffle and Contribution-Aware Serial Federated Learning Framework for Heterogeneous Healthcare Data 

**Authors**: Yuping Yan, Yizhi Wang, Yuanshuai Li, Yaochu Jin  

**Link**: [PDF](https://arxiv.org/pdf/2506.16723)  

**Abstract**: Serial pipeline training is an efficient paradigm for handling data heterogeneity in cross-silo federated learning with low communication overhead. However, even without centralized aggregation, direct transfer of models between clients can violate privacy regulations and remain susceptible to gradient leakage and linkage attacks. Additionally, ensuring resilience against semi-honest or malicious clients who may manipulate or misuse received models remains a grand challenge, particularly in privacy-sensitive domains such as healthcare. To address these challenges, we propose TriCon-SF, a novel serial federated learning framework that integrates triple shuffling and contribution awareness. TriCon-SF introduces three levels of randomization by shuffling model layers, data segments, and training sequences to break deterministic learning patterns and disrupt potential attack vectors, thereby enhancing privacy and robustness. In parallel, it leverages Shapley value methods to dynamically evaluate client contributions during training, enabling the detection of dishonest behavior and enhancing system accountability. Extensive experiments on non-IID healthcare datasets demonstrate that TriCon-SF outperforms standard serial and parallel federated learning in both accuracy and communication efficiency. Security analysis further supports its resilience against client-side privacy attacks. 

---
# Generalizable Agent Modeling for Agent Collaboration-Competition Adaptation with Multi-Retrieval and Dynamic Generation 

**Authors**: Chenxu Wang, Yonggang Jin, Cheng Hu, Youpeng Zhao, Zipeng Dai, Jian Zhao, Shiyu Huang, Liuyu Xiang, Junge Zhang, Zhaofeng He  

**Link**: [PDF](https://arxiv.org/pdf/2506.16718)  

**Abstract**: Adapting a single agent to a new multi-agent system brings challenges, necessitating adjustments across various tasks, environments, and interactions with unknown teammates and opponents. Addressing this challenge is highly complex, and researchers have proposed two simplified scenarios, Multi-agent reinforcement learning for zero-shot learning and Ad-Hoc Teamwork. Building on these foundations, we propose a more comprehensive setting, Agent Collaborative-Competitive Adaptation (ACCA), which evaluates an agent to generalize across diverse scenarios, tasks, and interactions with both unfamiliar opponents and teammates. In ACCA, agents adjust to task and environmental changes, collaborate with unseen teammates, and compete against unknown opponents. We introduce a new modeling approach, Multi-Retrieval and Dynamic Generation (MRDG), that effectively models both teammates and opponents using their behavioral trajectories. This method incorporates a positional encoder for varying team sizes and a hypernetwork module to boost agents' learning and adaptive capabilities. Additionally, a viewpoint alignment module harmonizes the observational perspectives of retrieved teammates and opponents with the learning agent. Extensive tests in benchmark scenarios like SMAC, Overcooked-AI, and Melting Pot show that MRDG significantly improves robust collaboration and competition with unseen teammates and opponents, surpassing established baselines. Our code is available at: this https URL 

---
# ReasonGRM: Enhancing Generative Reward Models through Large Reasoning Models 

**Authors**: Bin Chen, Xinzge Gao, Chuanrui Hu, Penghang Yu, Hua Zhang, Bing-Kun Bao  

**Link**: [PDF](https://arxiv.org/pdf/2506.16712)  

**Abstract**: Generative Reward Models (GRMs) provide greater flexibility than scalar reward models in capturing human preferences, but their effectiveness is limited by poor reasoning capabilities. This often results in incomplete or overly speculative reasoning paths, leading to hallucinations or missing key information in complex tasks. We address this challenge with ReasonGRM, a three-stage generative reward modeling framework. In the first stage, Zero-RL is used to generate concise, outcome-directed reasoning paths that reduce the likelihood of critical omissions. In the second stage, we introduce a novel evaluation metric, $R^\star$, which scores reasoning paths based on their generation likelihood. This favors paths that reach correct answers with minimal exploration, helping to reduce hallucination-prone data during training. In the final stage, the model is further refined through reinforcement learning on challenging examples to enhance its preference discrimination capabilities. Experiments on three public benchmarks show that ReasonGRM achieves competitive or state-of-the-art performance, outperforming previous best GRMs by 1.8\% on average and surpassing proprietary models such as GPT-4o by up to 5.6\%. These results demonstrate the effectiveness of reasoning-aware training and highlight the importance of high-quality rationale selection for reliable preference modeling. 

---
# Large Language Models as Psychological Simulators: A Methodological Guide 

**Authors**: Zhicheng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.16702)  

**Abstract**: Large language models (LLMs) offer emerging opportunities for psychological and behavioral research, but methodological guidance is lacking. This article provides a framework for using LLMs as psychological simulators across two primary applications: simulating roles and personas to explore diverse contexts, and serving as computational models to investigate cognitive processes. For simulation, we present methods for developing psychologically grounded personas that move beyond demographic categories, with strategies for validation against human data and use cases ranging from studying inaccessible populations to prototyping research instruments. For cognitive modeling, we synthesize emerging approaches for probing internal representations, methodological advances in causal interventions, and strategies for relating model behavior to human cognition. We address overarching challenges including prompt sensitivity, temporal limitations from training data cutoffs, and ethical considerations that extend beyond traditional human subjects review. Throughout, we emphasize the need for transparency about model capabilities and constraints. Together, this framework integrates emerging empirical evidence about LLM performance--including systematic biases, cultural limitations, and prompt brittleness--to help researchers wrangle these challenges and leverage the unique capabilities of LLMs in psychological research. 

---
# From Prompts to Constructs: A Dual-Validity Framework for LLM Research in Psychology 

**Authors**: Zhicheng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.16697)  

**Abstract**: Large language models (LLMs) are rapidly being adopted across psychology, serving as research tools, experimental subjects, human simulators, and computational models of cognition. However, the application of human measurement tools to these systems can produce contradictory results, raising concerns that many findings are measurement phantoms--statistical artifacts rather than genuine psychological phenomena. In this Perspective, we argue that building a robust science of AI psychology requires integrating two of our field's foundational pillars: the principles of reliable measurement and the standards for sound causal inference. We present a dual-validity framework to guide this integration, which clarifies how the evidence needed to support a claim scales with its scientific ambition. Using an LLM to classify text may require only basic accuracy checks, whereas claiming it can simulate anxiety demands a far more rigorous validation process. Current practice systematically fails to meet these requirements, often treating statistical pattern matching as evidence of psychological phenomena. The same model output--endorsing "I am anxious"--requires different validation strategies depending on whether researchers claim to measure, characterize, simulate, or model psychological constructs. Moving forward requires developing computational analogues of psychological constructs and establishing clear, scalable standards of evidence rather than the uncritical application of human measurement tools. 

---
# Fast and Stable Diffusion Planning through Variational Adaptive Weighting 

**Authors**: Zhiying Qiu, Tao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.16688)  

**Abstract**: Diffusion models have recently shown promise in offline RL. However, these methods often suffer from high training costs and slow convergence, particularly when using transformer-based denoising backbones. While several optimization strategies have been proposed -- such as modified noise schedules, auxiliary prediction targets, and adaptive loss weighting -- challenges remain in achieving stable and efficient training. In particular, existing loss weighting functions typically rely on neural network approximators, which can be ineffective in early training phases due to limited generalization capacity of MLPs when exposed to sparse feedback in the early training stages. In this work, we derive a variationally optimal uncertainty-aware weighting function and introduce a closed-form polynomial approximation method for its online estimation under the flow-based generative modeling framework. We integrate our method into a diffusion planning pipeline and evaluate it on standard offline RL benchmarks. Experimental results on Maze2D and Kitchen tasks show that our method achieves competitive performance with up to 10 times fewer training steps, highlighting its practical effectiveness. 

---
# A Simple Contrastive Framework Of Item Tokenization For Generative Recommendation 

**Authors**: Penglong Zhai, Yifang Yuan, Fanyi Di, Jie Li, Yue Liu, Chen Li, Jie Huang, Sicong Wang, Yao Xu, Xin Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.16683)  

**Abstract**: Generative retrieval-based recommendation has emerged as a promising paradigm aiming at directly generating the identifiers of the target candidates. However, in large-scale recommendation systems, this approach becomes increasingly cumbersome due to the redundancy and sheer scale of the token space. To overcome these limitations, recent research has explored the use of semantic tokens as an alternative to ID tokens, which typically leveraged reconstruction-based strategies, like RQ-VAE, to quantize content embeddings and significantly reduce the embedding size. However, reconstructive quantization aims for the precise reconstruction of each item embedding independently, which conflicts with the goal of generative retrieval tasks focusing more on differentiating among items. Moreover, multi-modal side information of items, such as descriptive text and images, geographical knowledge in location-based recommendation services, has been shown to be effective in improving recommendations by providing richer contexts for interactions. Nevertheless, effectively integrating such complementary knowledge into existing generative recommendation frameworks remains challenging. To overcome these challenges, we propose a novel unsupervised deep quantization exclusively based on contrastive learning, named SimCIT (a Simple Contrastive Item Tokenization framework). Specifically, different from existing reconstruction-based strategies, SimCIT propose to use a learnable residual quantization module to align with the signals from different modalities of the items, which combines multi-modal knowledge alignment and semantic tokenization in a mutually beneficial contrastive learning framework. Extensive experiments across public datasets and a large-scale industrial dataset from various domains demonstrate SimCIT's effectiveness in LLM-based generative recommendation. 

---
# How to Train your Text-to-Image Model: Evaluating Design Choices for Synthetic Training Captions 

**Authors**: Manuel Brack, Sudeep Katakol, Felix Friedrich, Patrick Schramowski, Hareesh Ravi, Kristian Kersting, Ajinkya Kale  

**Link**: [PDF](https://arxiv.org/pdf/2506.16679)  

**Abstract**: Training data is at the core of any successful text-to-image models. The quality and descriptiveness of image text are crucial to a model's performance. Given the noisiness and inconsistency in web-scraped datasets, recent works shifted towards synthetic training captions. While this setup is generally believed to produce more capable models, current literature does not provide any insights into its design choices. This study closes this gap by systematically investigating how different synthetic captioning strategies impact the downstream performance of text-to-image models. Our experiments demonstrate that dense, high-quality captions enhance text alignment but may introduce trade-offs in output aesthetics and diversity. Conversely, captions of randomized lengths yield balanced improvements across aesthetics and alignment without compromising sample diversity. We also demonstrate that varying caption distributions introduce significant shifts in the output bias of a trained model. Our findings underscore the importance of caption design in achieving optimal model performance and provide practical insights for more effective training data strategies in text-to-image generation. 

---
# A Minimalist Optimizer Design for LLM Pretraining 

**Authors**: Athanasios Glentis, Jiaxiang Li, Andi Han, Mingyi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2506.16659)  

**Abstract**: Training large language models (LLMs) typically relies on adaptive optimizers such as Adam, which require significant memory to maintain first- and second-moment matrices, known as optimizer states. While recent works such as GaLore, Fira, and APOLLO have proposed state-compressed variants to reduce memory consumption, a fundamental question remains: What is the minimal amount of optimizer state that is truly necessary to retain state-of-the-art performance in LLM pretraining? In this work, we systematically investigate this question using a bottom-up approach. We find that two memory- and compute-efficient optimization techniques are particularly effective: (1) column-wise gradient normalization significantly boosts the performance of plain SGD without requiring momentum; and (2) adding first-order momentum only to the output layer - where gradient variance is highest - yields performance competitive with fully adaptive methods such as Muon. Based on these insights, we propose SCALE (Stochastic Column-normalized Last-layer Momentum), a new optimizer that combines column-normalized SGD with last-layer momentum, where column normalization refers to normalizing the gradient along the output dimension. Across multiple LLaMA models (60M-1B), SCALE matches or exceeds the performance of Adam while using only 35-45% of the total memory. It also consistently outperforms memory-efficient optimizers such as GaLore, Fira, and APOLLO, making it a strong candidate for large-scale pretraining under memory constraints. For the LLaMA 7B model, SCALE outperforms the state-of-the-art method APOLLO in terms of both perplexity and memory consumption. In addition, our method serves as a minimalist baseline for more sophisticated optimizer design. 

---
# Relational Deep Learning: Challenges, Foundations and Next-Generation Architectures 

**Authors**: Vijay Prakash Dwivedi, Charilaos Kanatsoulis, Shenyang Huang, Jure Leskovec  

**Link**: [PDF](https://arxiv.org/pdf/2506.16654)  

**Abstract**: Graph machine learning has led to a significant increase in the capabilities of models that learn on arbitrary graph-structured data and has been applied to molecules, social networks, recommendation systems, and transportation, among other domains. Data in multi-tabular relational databases can also be constructed as 'relational entity graphs' for Relational Deep Learning (RDL) - a new blueprint that enables end-to-end representation learning without traditional feature engineering. Compared to arbitrary graph-structured data, relational entity graphs have key properties: (i) their structure is defined by primary-foreign key relationships between entities in different tables, (ii) the structural connectivity is a function of the relational schema defining a database, and (iii) the graph connectivity is temporal and heterogeneous in nature. In this paper, we provide a comprehensive review of RDL by first introducing the representation of relational databases as relational entity graphs, and then reviewing public benchmark datasets that have been used to develop and evaluate recent GNN-based RDL models. We discuss key challenges including large-scale multi-table integration and the complexities of modeling temporal dynamics and heterogeneous data, while also surveying foundational neural network methods and recent architectural advances specialized for relational entity graphs. Finally, we explore opportunities to unify these distinct modeling challenges, highlighting how RDL converges multiple sub-fields in graph machine learning towards the design of foundation models that can transform the processing of relational data. 

---
# LLMs in Coding and their Impact on the Commercial Software Engineering Landscape 

**Authors**: Vladislav Belozerov, Peter J Barclay, Askhan Sami  

**Link**: [PDF](https://arxiv.org/pdf/2506.16653)  

**Abstract**: Large-language-model coding tools are now mainstream in software engineering. But as these same tools move human effort up the development stack, they present fresh dangers: 10% of real prompts leak private data, 42% of generated snippets hide security flaws, and the models can even ``agree'' with wrong ideas, a trait called sycophancy. We argue that firms must tag and review every AI-generated line of code, keep prompts and outputs inside private or on-premises deployments, obey emerging safety regulations, and add tests that catch sycophantic answers -- so they can gain speed without losing security and accuracy. 

---
# SemAgent: A Semantics Aware Program Repair Agent 

**Authors**: Anvith Pabba, Alex Mathai, Anindya Chakraborty, Baishakhi Ray  

**Link**: [PDF](https://arxiv.org/pdf/2506.16650)  

**Abstract**: Large Language Models (LLMs) have shown impressive capabilities in downstream software engineering tasks such as Automated Program Repair (APR). In particular, there has been a lot of research on repository-level issue-resolution benchmarks such as SWE-Bench. Although there has been significant progress on this topic, we notice that in the process of solving such issues, existing agentic systems tend to hyper-localize on immediately suspicious lines of code and fix them in isolation, without a deeper understanding of the issue semantics, code semantics, or execution semantics. Consequently, many existing systems generate patches that overfit to the user issue, even when a more general fix is preferable. To address this limitation, we introduce SemAgent, a novel workflow-based procedure that leverages issue, code, and execution semantics to generate patches that are complete - identifying and fixing all lines relevant to the issue. We achieve this through a novel pipeline that (a) leverages execution semantics to retrieve relevant context, (b) comprehends issue-semantics via generalized abstraction, (c) isolates code-semantics within the context of this abstraction, and (d) leverages this understanding in a two-stage architecture: a repair stage that proposes fine-grained fixes, followed by a reviewer stage that filters relevant fixes based on the inferred issue-semantics. Our evaluations show that our methodology achieves a solve rate of 44.66% on the SWEBench-Lite benchmark beating all other workflow-based approaches, and an absolute improvement of 7.66% compared to our baseline, which lacks such deep semantic understanding. We note that our approach performs particularly well on issues requiring multi-line reasoning (and editing) and edge-case handling, suggesting that incorporating issue and code semantics into APR pipelines can lead to robust and semantically consistent repairs. 

---
# Long-Context Generalization with Sparse Attention 

**Authors**: Pavlo Vasylenko, Marcos Treviso, André F. T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2506.16640)  

**Abstract**: Transformer-based architectures traditionally employ softmax to compute attention weights, which produces dense distributions over all tokens in a sequence. While effective in many settings, this density has been shown to be detrimental for tasks that demand precise focus on fixed-size patterns: as sequence length increases, non-informative tokens accumulate attention probability mass, leading to dispersion and representational collapse. We show in this paper that sparse attention mechanisms using $\alpha$-entmax can avoid these issues, due to their ability to assign exact zeros to irrelevant tokens. Furthermore, we introduce Adaptive-Scalable Entmax (ASEntmax), which endows $\alpha$-entmax with a learnable temperature parameter, allowing the attention distribution to interpolate between sparse (pattern-focused) and dense (softmax-like) regimes. Finally, we show that the ability to locate and generalize fixed-size patterns can be further improved through a careful design of position encodings, which impacts both dense and sparse attention methods. By integrating ASEntmax into standard transformer layers alongside proper positional encodings, we show that our models greatly outperform softmax, scalable softmax, and fixed-temperature $\alpha$-entmax baselines on long-context generalization. 

---
# Latent Noise Injection for Private and Statistically Aligned Synthetic Data Generation 

**Authors**: Rex Shen, Lu Tian  

**Link**: [PDF](https://arxiv.org/pdf/2506.16636)  

**Abstract**: Synthetic Data Generation has become essential for scalable, privacy-preserving statistical analysis. While standard approaches based on generative models, such as Normalizing Flows, have been widely used, they often suffer from slow convergence in high-dimensional settings, frequently converging more slowly than the canonical $1/\sqrt{n}$ rate when approximating the true data distribution.
To overcome these limitations, we propose a Latent Noise Injection method using Masked Autoregressive Flows (MAF). Instead of directly sampling from the trained model, our method perturbs each data point in the latent space and maps it back to the data domain. This construction preserves a one to one correspondence between observed and synthetic data, enabling synthetic outputs that closely reflect the underlying distribution, particularly in challenging high-dimensional regimes where traditional sampling struggles.
Our procedure satisfies local $(\epsilon, \delta)$-differential privacy and introduces a single perturbation parameter to control the privacy-utility trade-off. Although estimators based on individual synthetic datasets may converge slowly, we show both theoretically and empirically that aggregating across $K$ studies in a meta analysis framework restores classical efficiency and yields consistent, reliable inference. We demonstrate that with a well-calibrated perturbation parameter, Latent Noise Injection achieves strong statistical alignment with the original data and robustness against membership inference attacks. These results position our method as a compelling alternative to conventional flow-based sampling for synthetic data sharing in decentralized and privacy-sensitive domains, such as biomedical research. 

---
# GeoGuess: Multimodal Reasoning based on Hierarchy of Visual Information in Street View 

**Authors**: Fenghua Cheng, Jinxiang Wang, Sen Wang, Zi Huang, Xue Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.16633)  

**Abstract**: Multimodal reasoning is a process of understanding, integrating and inferring information across different data modalities. It has recently attracted surging academic attention as a benchmark for Artificial Intelligence (AI). Although there are various tasks for evaluating multimodal reasoning ability, they still have limitations. Lack of reasoning on hierarchical visual clues at different levels of granularity, e.g., local details and global context, is of little discussion, despite its frequent involvement in real scenarios. To bridge the gap, we introduce a novel and challenging task for multimodal reasoning, namely GeoGuess. Given a street view image, the task is to identify its location and provide a detailed explanation. A system that succeeds in GeoGuess should be able to detect tiny visual clues, perceive the broader landscape, and associate with vast geographic knowledge. Therefore, GeoGuess would require the ability to reason between hierarchical visual information and geographic knowledge. In this work, we establish a benchmark for GeoGuess by introducing a specially curated dataset GeoExplain which consists of panoramas-geocoordinates-explanation tuples. Additionally, we present a multimodal and multilevel reasoning method, namely SightSense which can make prediction and generate comprehensive explanation based on hierarchy of visual information and external knowledge. Our analysis and experiments demonstrate their outstanding performance in GeoGuess. 

---
# History-Augmented Vision-Language Models for Frontier-Based Zero-Shot Object Navigation 

**Authors**: Mobin Habibpour, Fatemeh Afghah  

**Link**: [PDF](https://arxiv.org/pdf/2506.16623)  

**Abstract**: Object Goal Navigation (ObjectNav) challenges robots to find objects in unseen environments, demanding sophisticated reasoning. While Vision-Language Models (VLMs) show potential, current ObjectNav methods often employ them superficially, primarily using vision-language embeddings for object-scene similarity checks rather than leveraging deeper reasoning. This limits contextual understanding and leads to practical issues like repetitive navigation behaviors. This paper introduces a novel zero-shot ObjectNav framework that pioneers the use of dynamic, history-aware prompting to more deeply integrate VLM reasoning into frontier-based exploration. Our core innovation lies in providing the VLM with action history context, enabling it to generate semantic guidance scores for navigation actions while actively avoiding decision loops. We also introduce a VLM-assisted waypoint generation mechanism for refining the final approach to detected objects. Evaluated on the HM3D dataset within Habitat, our approach achieves a 46% Success Rate (SR) and 24.8% Success weighted by Path Length (SPL). These results are comparable to state-of-the-art zero-shot methods, demonstrating the significant potential of our history-augmented VLM prompting strategy for more robust and context-aware robotic navigation. 

---
# Modeling Public Perceptions of Science in Media 

**Authors**: Jiaxin Pei, Dustin Wright, Isabelle Augenstin, David Jurgens  

**Link**: [PDF](https://arxiv.org/pdf/2506.16622)  

**Abstract**: Effectively engaging the public with science is vital for fostering trust and understanding in our scientific community. Yet, with an ever-growing volume of information, science communicators struggle to anticipate how audiences will perceive and interact with scientific news. In this paper, we introduce a computational framework that models public perception across twelve dimensions, such as newsworthiness, importance, and surprisingness. Using this framework, we create a large-scale science news perception dataset with 10,489 annotations from 2,101 participants from diverse US and UK populations, providing valuable insights into public responses to scientific information across domains. We further develop NLP models that predict public perception scores with a strong performance. Leveraging the dataset and model, we examine public perception of science from two perspectives: (1) Perception as an outcome: What factors affect the public perception of scientific information? (2) Perception as a predictor: Can we use the estimated perceptions to predict public engagement with science? We find that individuals' frequency of science news consumption is the driver of perception, whereas demographic factors exert minimal influence. More importantly, through a large-scale analysis and carefully designed natural experiment on Reddit, we demonstrate that the estimated public perception of scientific information has direct connections with the final engagement pattern. Posts with more positive perception scores receive significantly more comments and upvotes, which is consistent across different scientific information and for the same science, but are framed differently. Overall, this research underscores the importance of nuanced perception modeling in science communication, offering new pathways to predict public interest and engagement with scientific content. 

---
# Distribution Parameter Actor-Critic: Shifting the Agent-Environment Boundary for Diverse Action Spaces 

**Authors**: Jiamin He, A. Rupam Mahmood, Martha White  

**Link**: [PDF](https://arxiv.org/pdf/2506.16608)  

**Abstract**: We introduce a novel reinforcement learning (RL) framework that treats distribution parameters as actions, redefining the boundary between agent and environment. This reparameterization makes the new action space continuous, regardless of the original action type (discrete, continuous, mixed, etc.). Under this new parameterization, we develop a generalized deterministic policy gradient estimator, Distribution Parameter Policy Gradient (DPPG), which has lower variance than the gradient in the original action space. Although learning the critic over distribution parameters poses new challenges, we introduce interpolated critic learning (ICL), a simple yet effective strategy to enhance learning, supported by insights from bandit settings. Building on TD3, a strong baseline for continuous control, we propose a practical DPPG-based actor-critic algorithm, Distribution Parameter Actor-Critic (DPAC). Empirically, DPAC outperforms TD3 in MuJoCo continuous control tasks from OpenAI Gym and DeepMind Control Suite, and demonstrates competitive performance on the same environments with discretized action spaces. 

---
# FLAME: Towards Federated Fine-Tuning Large Language Models Through Adaptive SMoE 

**Authors**: Khiem Le, Tuan Tran, Ting Hua, Nitesh V. Chawla  

**Link**: [PDF](https://arxiv.org/pdf/2506.16600)  

**Abstract**: Existing resource-adaptive LoRA federated fine-tuning methods enable clients to fine-tune models using compressed versions of global LoRA matrices, in order to accommodate various compute resources across clients. This compression requirement will lead to suboptimal performance due to information loss. To address this, we propose FLAME, a novel federated learning framework based on the Sparse Mixture-of-Experts (SMoE) architecture. Unlike prior approaches, FLAME retains full (uncompressed) global LoRA matrices and achieves client-side adaptability by varying the number of activated experts per client. However, incorporating SMoE into federated learning introduces unique challenges, specifically, the mismatch in output magnitude from partial expert activation and the imbalance in expert training quality across clients. FLAME tackles these challenges through a lightweight rescaling mechanism and an activation-aware aggregation scheme. Empirical results across diverse computational settings demonstrate that FLAME consistently outperforms existing methods, providing a robust and effective solution for resource-adaptive federated learning. 

---
# Hybrid Attention Network for Accurate Breast Tumor Segmentation in Ultrasound Images 

**Authors**: Muhammad Azeem Aslam, Asim Naveed, Nisar Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2506.16592)  

**Abstract**: Breast ultrasound imaging is a valuable tool for early breast cancer detection, but automated tumor segmentation is challenging due to inherent noise, variations in scale of lesions, and fuzzy boundaries. To address these challenges, we propose a novel hybrid attention-based network for lesion segmentation. Our proposed architecture integrates a pre-trained DenseNet121 in the encoder part for robust feature extraction with a multi-branch attention-enhanced decoder tailored for breast ultrasound images. The bottleneck incorporates Global Spatial Attention (GSA), Position Encoding (PE), and Scaled Dot-Product Attention (SDPA) to learn global context, spatial relationships, and relative positional features. The Spatial Feature Enhancement Block (SFEB) is embedded at skip connections to refine and enhance spatial features, enabling the network to focus more effectively on tumor regions. A hybrid loss function combining Binary Cross-Entropy (BCE) and Jaccard Index loss optimizes both pixel-level accuracy and region-level overlap metrics, enhancing robustness to class imbalance and irregular tumor shapes. Experiments on public datasets demonstrate that our method outperforms existing approaches, highlighting its potential to assist radiologists in early and accurate breast cancer diagnosis. 

---
# Energy-Based Transfer for Reinforcement Learning 

**Authors**: Zeyun Deng, Jasorsi Ghosh, Fiona Xie, Yuzhe Lu, Katia Sycara, Joseph Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2506.16590)  

**Abstract**: Reinforcement learning algorithms often suffer from poor sample efficiency, making them challenging to apply in multi-task or continual learning settings. Efficiency can be improved by transferring knowledge from a previously trained teacher policy to guide exploration in new but related tasks. However, if the new task sufficiently differs from the teacher's training task, the transferred guidance may be sub-optimal and bias exploration toward low-reward behaviors. We propose an energy-based transfer learning method that uses out-of-distribution detection to selectively issue guidance, enabling the teacher to intervene only in states within its training distribution. We theoretically show that energy scores reflect the teacher's state-visitation density and empirically demonstrate improved sample efficiency and performance across both single-task and multi-task settings. 

---
# Spatially-Aware Evaluation of Segmentation Uncertainty 

**Authors**: Tal Zeevi, Eléonore V. Lieffrig, Lawrence H. Staib, John A. Onofrey  

**Link**: [PDF](https://arxiv.org/pdf/2506.16589)  

**Abstract**: Uncertainty maps highlight unreliable regions in segmentation predictions. However, most uncertainty evaluation metrics treat voxels independently, ignoring spatial context and anatomical structure. As a result, they may assign identical scores to qualitatively distinct patterns (e.g., scattered vs. boundary-aligned uncertainty). We propose three spatially aware metrics that incorporate structural and boundary information and conduct a thorough validation on medical imaging data from the prostate zonal segmentation challenge within the Medical Segmentation Decathlon. Our results demonstrate improved alignment with clinically important factors and better discrimination between meaningful and spurious uncertainty patterns. 

---
# AI-Driven Tools in Modern Software Quality Assurance: An Assessment of Benefits, Challenges, and Future Directions 

**Authors**: Ihor Pysmennyi, Roman Kyslyi, Kyrylo Kleshch  

**Link**: [PDF](https://arxiv.org/pdf/2506.16586)  

**Abstract**: Traditional quality assurance (QA) methods face significant challenges in addressing the complexity, scale, and rapid iteration cycles of modern software systems and are strained by limited resources available, leading to substantial costs associated with poor quality. The object of this research is the Quality Assurance processes for modern distributed software applications. The subject of the research is the assessment of the benefits, challenges, and prospects of integrating modern AI-oriented tools into quality assurance processes. We performed comprehensive analysis of implications on both verification and validation processes covering exploratory test analyses, equivalence partitioning and boundary analyses, metamorphic testing, finding inconsistencies in acceptance criteria (AC), static analyses, test case generation, unit test generation, test suit optimization and assessment, end to end scenario execution. End to end regression of sample enterprise application utilizing AI-agents over generated test scenarios was implemented as a proof of concept highlighting practical use of the study. The results, with only 8.3% flaky executions of generated test cases, indicate significant potential for the proposed approaches. However, the study also identified substantial challenges for practical adoption concerning generation of semantically identical coverage, "black box" nature and lack of explainability from state-of-the-art Large Language Models (LLMs), the tendency to correct mutated test cases to match expected results, underscoring the necessity for thorough verification of both generated artifacts and test execution results. The research demonstrates AI's transformative potential for QA but highlights the importance of a strategic approach to implementing these technologies, considering the identified limitations and the need for developing appropriate verification methodologies. 

---
# Measuring (a Sufficient) World Model in LLMs: A Variance Decomposition Framework 

**Authors**: Nadav Kunievsky, James A. Evans  

**Link**: [PDF](https://arxiv.org/pdf/2506.16584)  

**Abstract**: Understanding whether large language models (LLMs) possess a world model-a structured understanding of the world that supports generalization beyond surface-level patterns-is central to assessing their reliability, especially in high-stakes applications. We propose a formal framework for evaluating whether an LLM exhibits a sufficiently robust world model, defined as producing consistent outputs across semantically equivalent prompts while distinguishing between prompts that express different intents. We introduce a new evaluation approach to measure this that decomposes model response variability into three components: variability due to user purpose, user articulation, and model instability. An LLM with a strong world model should attribute most of the variability in its responses to changes in foundational purpose rather than superficial changes in articulation. This approach allows us to quantify how much of a model's behavior is semantically grounded rather than driven by model instability or alternative wording. We apply this framework to evaluate LLMs across diverse domains. Our results show how larger models attribute a greater share of output variability to changes in user purpose, indicating a more robust world model. This improvement is not uniform, however: larger models do not consistently outperform smaller ones across all domains, and their advantage in robustness is often modest. These findings highlight the importance of moving beyond accuracy-based benchmarks toward semantic diagnostics that more directly assess the structure and stability of a model's internal understanding of the world. 

---
# Reimagination with Test-time Observation Interventions: Distractor-Robust World Model Predictions for Visual Model Predictive Control 

**Authors**: Yuxin Chen, Jianglan Wei, Chenfeng Xu, Boyi Li, Masayoshi Tomizuka, Andrea Bajcsy, Ran Tian  

**Link**: [PDF](https://arxiv.org/pdf/2506.16565)  

**Abstract**: World models enable robots to "imagine" future observations given current observations and planned actions, and have been increasingly adopted as generalized dynamics models to facilitate robot learning. Despite their promise, these models remain brittle when encountering novel visual distractors such as objects and background elements rarely seen during training. Specifically, novel distractors can corrupt action outcome predictions, causing downstream failures when robots rely on the world model imaginations for planning or action verification. In this work, we propose Reimagination with Observation Intervention (ReOI), a simple yet effective test-time strategy that enables world models to predict more reliable action outcomes in open-world scenarios where novel and unanticipated visual distractors are inevitable. Given the current robot observation, ReOI first detects visual distractors by identifying which elements of the scene degrade in physically implausible ways during world model prediction. Then, it modifies the current observation to remove these distractors and bring the observation closer to the training distribution. Finally, ReOI "reimagines" future outcomes with the modified observation and reintroduces the distractors post-hoc to preserve visual consistency for downstream planning and verification. We validate our approach on a suite of robotic manipulation tasks in the context of action verification, where the verifier needs to select desired action plans based on predictions from a world model. Our results show that ReOI is robust to both in-distribution and out-of-distribution visual distractors. Notably, it improves task success rates by up to 3x in the presence of novel distractors, significantly outperforming action verification that relies on world model predictions without imagination interventions. 

---
# From Semantic To Instance: A Semi-Self-Supervised Learning Approach 

**Authors**: Keyhan Najafian, Farhad Maleki, Lingling Jin, Ian Stavness  

**Link**: [PDF](https://arxiv.org/pdf/2506.16563)  

**Abstract**: Instance segmentation is essential for applications such as automated monitoring of plant health, growth, and yield. However, extensive effort is required to create large-scale datasets with pixel-level annotations of each object instance for developing instance segmentation models that restrict the use of deep learning in these areas. This challenge is more significant in images with densely packed, self-occluded objects, which are common in agriculture. To address this challenge, we propose a semi-self-supervised learning approach that requires minimal manual annotation to develop a high-performing instance segmentation model. We design GLMask, an image-mask representation for the model to focus on shape, texture, and pattern while minimizing its dependence on color features. We develop a pipeline to generate semantic segmentation and then transform it into instance-level segmentation. The proposed approach substantially outperforms the conventional instance segmentation models, establishing a state-of-the-art wheat head instance segmentation model with mAP@50 of 98.5%. Additionally, we assessed the proposed methodology on the general-purpose Microsoft COCO dataset, achieving a significant performance improvement of over 12.6% mAP@50. This highlights that the utility of our proposed approach extends beyond precision agriculture and applies to other domains, specifically those with similar data characteristics. 

---
# One Sample is Enough to Make Conformal Prediction Robust 

**Authors**: Soroush H. Zargarbashi, Mohammad Sadegh Akhondzadeh, Aleksandar Bojchevski  

**Link**: [PDF](https://arxiv.org/pdf/2506.16553)  

**Abstract**: Given any model, conformal prediction (CP) returns prediction sets guaranteed to include the true label with high adjustable probability. Robust CP (RCP) extends this to inputs with worst-case noise. A well-established approach is to use randomized smoothing for RCP since it is applicable to any black-box model and provides smaller sets compared to deterministic methods. However, current smoothing-based RCP requires many model forward passes per each input which is computationally expensive. We show that conformal prediction attains some robustness even with a forward pass on a single randomly perturbed input. Using any binary certificate we propose a single sample robust CP (RCP1). Our approach returns robust sets with smaller average set size compared to SOTA methods which use many (e.g. around 100) passes per input. Our key insight is to certify the conformal prediction procedure itself rather than individual scores. Our approach is agnostic to the setup (classification and regression). We further extend our approach to smoothing-based robust conformal risk control. 

---
# BIDA: A Bi-level Interaction Decision-making Algorithm for Autonomous Vehicles in Dynamic Traffic Scenarios 

**Authors**: Liyang Yu, Tianyi Wang, Junfeng Jiao, Fengwu Shan, Hongqing Chu, Bingzhao Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.16546)  

**Abstract**: In complex real-world traffic environments, autonomous vehicles (AVs) need to interact with other traffic participants while making real-time and safety-critical decisions accordingly. The unpredictability of human behaviors poses significant challenges, particularly in dynamic scenarios, such as multi-lane highways and unsignalized T-intersections. To address this gap, we design a bi-level interaction decision-making algorithm (BIDA) that integrates interactive Monte Carlo tree search (MCTS) with deep reinforcement learning (DRL), aiming to enhance interaction rationality, efficiency and safety of AVs in dynamic key traffic scenarios. Specifically, we adopt three types of DRL algorithms to construct a reliable value network and policy network, which guide the online deduction process of interactive MCTS by assisting in value update and node selection. Then, a dynamic trajectory planner and a trajectory tracking controller are designed and implemented in CARLA to ensure smooth execution of planned maneuvers. Experimental evaluations demonstrate that our BIDA not only enhances interactive deduction and reduces computational costs, but also outperforms other latest benchmarks, which exhibits superior safety, efficiency and interaction rationality under varying traffic conditions. 

---
# Subspace-Boosted Model Merging 

**Authors**: Ronald Skorobogat, Karsten Roth, Mariana-Iuliana Georgescu, Zeynep Akata  

**Link**: [PDF](https://arxiv.org/pdf/2506.16506)  

**Abstract**: Model merging enables the combination of multiple specialized expert models into a single model capable of performing multiple tasks. However, the benefits of merging an increasing amount of specialized experts generally lead to diminishing returns and reduced overall performance gains. In this work, we offer an explanation and analysis from a task arithmetic perspective; revealing that as the merging process (across numerous existing merging methods) continues for more and more experts, the associated task vector space experiences rank collapse. To mitigate this issue, we introduce Subspace Boosting, which operates on the singular value decomposed task vector space and maintains task vector ranks. Subspace Boosting raises merging efficacy for up to 20 expert models by large margins of more than 10% when evaluated on vision benchmarks. Moreover, we propose employing Higher-Order Generalized Singular Value Decomposition to further quantify task similarity, offering a new interpretable perspective on model merging. 

---
# Hunyuan3D 2.5: Towards High-Fidelity 3D Assets Generation with Ultimate Details 

**Authors**: Zeqiang Lai, Yunfei Zhao, Haolin Liu, Zibo Zhao, Qingxiang Lin, Huiwen Shi, Xianghui Yang, Mingxin Yang, Shuhui Yang, Yifei Feng, Sheng Zhang, Xin Huang, Di Luo, Fan Yang, Fang Yang, Lifu Wang, Sicong Liu, Yixuan Tang, Yulin Cai, Zebin He, Tian Liu, Yuhong Liu, Jie Jiang, Linus, Jingwei Huang, Chunchao Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.16504)  

**Abstract**: In this report, we present Hunyuan3D 2.5, a robust suite of 3D diffusion models aimed at generating high-fidelity and detailed textured 3D assets. Hunyuan3D 2.5 follows two-stages pipeline of its previous version Hunyuan3D 2.0, while demonstrating substantial advancements in both shape and texture generation. In terms of shape generation, we introduce a new shape foundation model -- LATTICE, which is trained with scaled high-quality datasets, model-size, and compute. Our largest model reaches 10B parameters and generates sharp and detailed 3D shape with precise image-3D following while keeping mesh surface clean and smooth, significantly closing the gap between generated and handcrafted 3D shapes. In terms of texture generation, it is upgraded with phyiscal-based rendering (PBR) via a novel multi-view architecture extended from Hunyuan3D 2.0 Paint model. Our extensive evaluation shows that Hunyuan3D 2.5 significantly outperforms previous methods in both shape and end-to-end texture generation. 

---
# Relic: Enhancing Reward Model Generalization for Low-Resource Indic Languages with Few-Shot Examples 

**Authors**: Soumya Suvra Ghosal, Vaibhav Singh, Akash Ghosh, Soumyabrata Pal, Subhadip Baidya, Sriparna Saha, Dinesh Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2506.16502)  

**Abstract**: Reward models are essential for aligning large language models (LLMs) with human preferences. However, most open-source multilingual reward models are primarily trained on preference datasets in high-resource languages, resulting in unreliable reward signals for low-resource Indic languages. Collecting large-scale, high-quality preference data for these languages is prohibitively expensive, making preference-based training approaches impractical. To address this challenge, we propose RELIC, a novel in-context learning framework for reward modeling in low-resource Indic languages. RELIC trains a retriever with a pairwise ranking objective to select in-context examples from auxiliary high-resource languages that most effectively highlight the distinction between preferred and less-preferred responses. Extensive experiments on three preference datasets- PKU-SafeRLHF, WebGPT, and HH-RLHF-using state-of-the-art open-source reward models demonstrate that RELIC significantly improves reward model accuracy for low-resource Indic languages, consistently outperforming existing example selection methods. For example, on Bodo-a low-resource Indic language-using a LLaMA-3.2-3B reward model, RELIC achieves a 12.81% and 10.13% improvement in accuracy over zero-shot prompting and state-of-the-art example selection method, respectively. 

---
# Spotting tell-tale visual artifacts in face swapping videos: strengths and pitfalls of CNN detectors 

**Authors**: Riccardo Ziglio, Cecilia Pasquini, Silvio Ranise  

**Link**: [PDF](https://arxiv.org/pdf/2506.16497)  

**Abstract**: Face swapping manipulations in video streams represents an increasing threat in remote video communications, due to advances
in automated and real-time tools. Recent literature proposes to characterize and exploit visual artifacts introduced in video frames
by swapping algorithms when dealing with challenging physical scenes, such as face occlusions. This paper investigates the
effectiveness of this approach by benchmarking CNN-based data-driven models on two data corpora (including a newly collected
one) and analyzing generalization capabilities with respect to different acquisition sources and swapping algorithms. The results
confirm excellent performance of general-purpose CNN architectures when operating within the same data source, but a significant
difficulty in robustly characterizing occlusion-based visual cues across datasets. This highlights the need for specialized detection
strategies to deal with such artifacts. 

---
# Grounding Language Models with Semantic Digital Twins for Robotic Planning 

**Authors**: Mehreen Naeem, Andrew Melnik, Michael Beetz  

**Link**: [PDF](https://arxiv.org/pdf/2506.16493)  

**Abstract**: We introduce a novel framework that integrates Semantic Digital Twins (SDTs) with Large Language Models (LLMs) to enable adaptive and goal-driven robotic task execution in dynamic environments. The system decomposes natural language instructions into structured action triplets, which are grounded in contextual environmental data provided by the SDT. This semantic grounding allows the robot to interpret object affordances and interaction rules, enabling action planning and real-time adaptability. In case of execution failures, the LLM utilizes error feedback and SDT insights to generate recovery strategies and iteratively revise the action plan. We evaluate our approach using tasks from the ALFRED benchmark, demonstrating robust performance across various household scenarios. The proposed framework effectively combines high-level reasoning with semantic environment understanding, achieving reliable task completion in the face of uncertainty and failure. 

---
# Towards Generalizable Generic Harmful Speech Datasets for Implicit Hate Speech Detection 

**Authors**: Saad Almohaimeed, Saleh Almohaimeed, Damla Turgut, Ladislau Bölöni  

**Link**: [PDF](https://arxiv.org/pdf/2506.16476)  

**Abstract**: Implicit hate speech has recently emerged as a critical challenge for social media platforms. While much of the research has traditionally focused on harmful speech in general, the need for generalizable techniques to detect veiled and subtle forms of hate has become increasingly pressing. Based on lexicon analysis, we hypothesize that implicit hate speech is already present in publicly available harmful speech datasets but may not have been explicitly recognized or labeled by annotators. Additionally, crowdsourced datasets are prone to mislabeling due to the complexity of the task and often influenced by annotators' subjective interpretations. In this paper, we propose an approach to address the detection of implicit hate speech and enhance generalizability across diverse datasets by leveraging existing harmful speech datasets. Our method comprises three key components: influential sample identification, reannotation, and augmentation using Llama-3 70B and GPT-4o. Experimental results demonstrate the effectiveness of our approach in improving implicit hate detection, achieving a +12.9-point F1 score improvement compared to the baseline. 

---
# Human2LocoMan: Learning Versatile Quadrupedal Manipulation with Human Pretraining 

**Authors**: Yaru Niu, Yunzhe Zhang, Mingyang Yu, Changyi Lin, Chenhao Li, Yikai Wang, Yuxiang Yang, Wenhao Yu, Tingnan Zhang, Bingqing Chen, Jonathan Francis, Zhenzhen Li, Jie Tan, Ding Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.16475)  

**Abstract**: Quadrupedal robots have demonstrated impressive locomotion capabilities in complex environments, but equipping them with autonomous versatile manipulation skills in a scalable way remains a significant challenge. In this work, we introduce a cross-embodiment imitation learning system for quadrupedal manipulation, leveraging data collected from both humans and LocoMan, a quadruped equipped with multiple manipulation modes. Specifically, we develop a teleoperation and data collection pipeline, which unifies and modularizes the observation and action spaces of the human and the robot. To effectively leverage the collected data, we propose an efficient modularized architecture that supports co-training and pretraining on structured modality-aligned data across different embodiments. Additionally, we construct the first manipulation dataset for the LocoMan robot, covering various household tasks in both unimanual and bimanual modes, supplemented by a corresponding human dataset. We validate our system on six real-world manipulation tasks, where it achieves an average success rate improvement of 41.9% overall and 79.7% under out-of-distribution (OOD) settings compared to the baseline. Pretraining with human data contributes a 38.6% success rate improvement overall and 82.7% under OOD settings, enabling consistently better performance with only half the amount of robot data. Our code, hardware, and data are open-sourced at: this https URL. 

---
# Do We Talk to Robots Like Therapists, and Do They Respond Accordingly? Language Alignment in AI Emotional Support 

**Authors**: Sophie Chiang, Guy Laban, Hatice Gunes  

**Link**: [PDF](https://arxiv.org/pdf/2506.16473)  

**Abstract**: As conversational agents increasingly engage in emotionally supportive dialogue, it is important to understand how closely their interactions resemble those in traditional therapy settings. This study investigates whether the concerns shared with a robot align with those shared in human-to-human (H2H) therapy sessions, and whether robot responses semantically mirror those of human therapists. We analyzed two datasets: one of interactions between users and professional therapists (Hugging Face's NLP Mental Health Conversations), and another involving supportive conversations with a social robot (QTrobot from LuxAI) powered by a large language model (LLM, GPT-3.5). Using sentence embeddings and K-means clustering, we assessed cross-agent thematic alignment by applying a distance-based cluster-fitting method that evaluates whether responses from one agent type map to clusters derived from the other, and validated it using Euclidean distances. Results showed that 90.88% of robot conversation disclosures could be mapped to clusters from the human therapy dataset, suggesting shared topical structure. For matched clusters, we compared the subjects as well as therapist and robot responses using Transformer, Word2Vec, and BERT embeddings, revealing strong semantic overlap in subjects' disclosures in both datasets, as well as in the responses given to similar human disclosure themes across agent types (robot vs. human therapist). These findings highlight both the parallels and boundaries of robot-led support conversations and their potential for augmenting mental health interventions. 

---
# Progressive Inference-Time Annealing of Diffusion Models for Sampling from Boltzmann Densities 

**Authors**: Tara Akhound-Sadegh, Jungyoon Lee, Avishek Joey Bose, Valentin De Bortoli, Arnaud Doucet, Michael M. Bronstein, Dominique Beaini, Siamak Ravanbakhsh, Kirill Neklyudov, Alexander Tong  

**Link**: [PDF](https://arxiv.org/pdf/2506.16471)  

**Abstract**: Sampling efficiently from a target unnormalized probability density remains a core challenge, with relevance across countless high-impact scientific applications. A promising approach towards this challenge is the design of amortized samplers that borrow key ideas, such as probability path design, from state-of-the-art generative diffusion models. However, all existing diffusion-based samplers remain unable to draw samples from distributions at the scale of even simple molecular systems. In this paper, we propose Progressive Inference-Time Annealing (PITA), a novel framework to learn diffusion-based samplers that combines two complementary interpolation techniques: I.) Annealing of the Boltzmann distribution and II.) Diffusion smoothing. PITA trains a sequence of diffusion models from high to low temperatures by sequentially training each model at progressively higher temperatures, leveraging engineered easy access to samples of the temperature-annealed target density. In the subsequent step, PITA enables simulating the trained diffusion model to procure training samples at a lower temperature for the next diffusion model through inference-time annealing using a novel Feynman-Kac PDE combined with Sequential Monte Carlo. Empirically, PITA enables, for the first time, equilibrium sampling of N-body particle systems, Alanine Dipeptide, and tripeptides in Cartesian coordinates with dramatically lower energy function evaluations. Code available at: this https URL 

---
# Joint Tensor-Train Parameterization for Efficient and Expressive Low-Rank Adaptation 

**Authors**: Jun Qi, Chen-Yu Liu, Sabato Marco Siniscalchi, Chao-Han Huck Yang, Min-Hsiu Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2506.16456)  

**Abstract**: Low-Rank Adaptation (LoRA) is widely recognized for its parameter-efficient fine-tuning of large-scale neural models. However, standard LoRA independently optimizes low-rank matrices, which inherently limits its expressivity and generalization capabilities. While classical tensor-train (TT) decomposition can be separately employed on individual LoRA matrices, this work demonstrates that the classical TT-based approach neither significantly improves parameter efficiency nor achieves substantial performance gains. This paper proposes TensorGuide, a novel tensor-train-guided adaptation framework to overcome these limitations. TensorGuide generates two correlated low-rank LoRA matrices through a unified TT structure driven by controlled Gaussian noise. The resulting joint TT representation inherently provides structured, low-rank adaptations, significantly enhancing expressivity, generalization, and parameter efficiency without increasing the number of trainable parameters. Theoretically, we justify these improvements through neural tangent kernel analyses, demonstrating superior optimization dynamics and enhanced generalization. Extensive experiments on quantum dot classification and GPT-2 fine-tuning benchmarks demonstrate that TensorGuide-based LoRA consistently outperforms standard LoRA and TT-LoRA, achieving improved accuracy and scalability with fewer parameters. 

---
# Consumer-friendly EEG-based Emotion Recognition System: A Multi-scale Convolutional Neural Network Approach 

**Authors**: Tri Duc Ly, Gia H. Ngo  

**Link**: [PDF](https://arxiv.org/pdf/2506.16448)  

**Abstract**: EEG is a non-invasive, safe, and low-risk method to record electrophysiological signals inside the brain. Especially with recent technology developments like dry electrodes, consumer-grade EEG devices, and rapid advances in machine learning, EEG is commonly used as a resource for automatic emotion recognition. With the aim to develop a deep learning model that can perform EEG-based emotion recognition in a real-life context, we propose a novel approach to utilize multi-scale convolutional neural networks to accomplish such tasks. By implementing feature extraction kernels with many ratio coefficients as well as a new type of kernel that learns key information from four separate areas of the brain, our model consistently outperforms the state-of-the-art TSception model in predicting valence, arousal, and dominance scores across many performance evaluation metrics. 

---
# StoryWriter: A Multi-Agent Framework for Long Story Generation 

**Authors**: Haotian Xia, Hao Peng, Yunjia Qi, Xiaozhi Wang, Bin Xu, Lei Hou, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.16445)  

**Abstract**: Long story generation remains a challenge for existing large language models (LLMs), primarily due to two main factors: (1) discourse coherence, which requires plot consistency, logical coherence, and completeness in the long-form generation, and (2) narrative complexity, which requires an interwoven and engaging narrative. To address these challenges, we propose StoryWriter, a multi-agent story generation framework, which consists of three main modules: (1) outline agent, which generates event-based outlines containing rich event plots, character, and event-event relationships. (2) planning agent, which further details events and plans which events should be written in each chapter to maintain an interwoven and engaging story. (3) writing agent, which dynamically compresses the story history based on the current event to generate and reflect new plots, ensuring the coherence of the generated story. We conduct both human and automated evaluation, and StoryWriter significantly outperforms existing story generation baselines in both story quality and length. Furthermore, we use StoryWriter to generate a dataset, which contains about $6,000$ high-quality long stories, with an average length of $8,000$ words. We train the model Llama3.1-8B and GLM4-9B using supervised fine-tuning on LongStory and develop StoryWriter_GLM and StoryWriter_GLM, which demonstrates advanced performance in long story generation. 

---
# Leveraging Influence Functions for Resampling Data in Physics-Informed Neural Networks 

**Authors**: Jonas R. Naujoks, Aleksander Krasowski, Moritz Weckbecker, Galip Ümit Yolcu, Thomas Wiegand, Sebastian Lapuschkin, Wojciech Samek, René P. Klausen  

**Link**: [PDF](https://arxiv.org/pdf/2506.16443)  

**Abstract**: Physics-informed neural networks (PINNs) offer a powerful approach to solving partial differential equations (PDEs), which are ubiquitous in the quantitative sciences. Applied to both forward and inverse problems across various scientific domains, PINNs have recently emerged as a valuable tool in the field of scientific machine learning. A key aspect of their training is that the data -- spatio-temporal points sampled from the PDE's input domain -- are readily available. Influence functions, a tool from the field of explainable AI (XAI), approximate the effect of individual training points on the model, enhancing interpretability. In the present work, we explore the application of influence function-based sampling approaches for the training data. Our results indicate that such targeted resampling based on data attribution methods has the potential to enhance prediction accuracy in physics-informed neural networks, demonstrating a practical application of an XAI method in PINN training. 

---
# Optimizing MoE Routers: Design, Implementation, and Evaluation in Transformer Models 

**Authors**: Daniel Fidel Harvey, George Weale, Berk Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2506.16419)  

**Abstract**: Mixture of Experts (MoE) architectures increase large language model scalability, yet their performance depends on the router module that moves tokens to specialized experts. Bad routing can load imbalance and reduced accuracy. This project designed and implemented different router architectures within Transformer models to fix these limitations. We experimented with six distinct router variants Linear, Attention, Multi-Layer Perceptron (MLP), Hybrid, Hash, and our new MLP-Hadamard. We characterized these routers using BERT and the Qwen1.5-MoE model, looking at parameter efficiency, inference latency, routing entropy, and expert utilization patterns. Our evaluations showed distinct trade-offs: Linear routers offer speed, while MLP and Attention routers provide greater expressiveness. The MLP-Hadamard router shows a unique capability for structured, sparse routing. We successfully replaced and fine-tuned custom routers within the complex, quantized Qwen1.5-MoE model. This work provides a comparative analysis of MoE router designs and offers insights into optimizing their performance for efficient and effective large-scale model deployment. 

---
# Efficient Transformations in Deep Learning Convolutional Neural Networks 

**Authors**: Berk Yilmaz, Daniel Fidel Harvey, Prajit Dhuri  

**Link**: [PDF](https://arxiv.org/pdf/2506.16418)  

**Abstract**: This study investigates the integration of signal processing transformations -- Fast Fourier Transform (FFT), Walsh-Hadamard Transform (WHT), and Discrete Cosine Transform (DCT) -- within the ResNet50 convolutional neural network (CNN) model for image classification. The primary objective is to assess the trade-offs between computational efficiency, energy consumption, and classification accuracy during training and inference. Using the CIFAR-100 dataset (100 classes, 60,000 images), experiments demonstrated that incorporating WHT significantly reduced energy consumption while improving accuracy. Specifically, a baseline ResNet50 model achieved a testing accuracy of 66%, consuming an average of 25,606 kJ per model. In contrast, a modified ResNet50 incorporating WHT in the early convolutional layers achieved 74% accuracy, and an enhanced version with WHT applied to both early and late layers achieved 79% accuracy, with an average energy consumption of only 39 kJ per model. These results demonstrate the potential of WHT as a highly efficient and effective approach for energy-constrained CNN applications. 

---
# Robustness Evaluation of OCR-based Visual Document Understanding under Multi-Modal Adversarial Attacks 

**Authors**: Dong Nguyen Tien, Dung D. Le  

**Link**: [PDF](https://arxiv.org/pdf/2506.16407)  

**Abstract**: Visual Document Understanding (VDU) systems have achieved strong performance in information extraction by integrating textual, layout, and visual signals. However, their robustness under realistic adversarial perturbations remains insufficiently explored. We introduce the first unified framework for generating and evaluating multi-modal adversarial attacks on OCR-based VDU models. Our method covers six gradient-based layout attack scenarios, incorporating manipulations of OCR bounding boxes, pixels, and texts across both word and line granularities, with constraints on layout perturbation budget (e.g., IoU >= 0.6) to preserve plausibility.
Experimental results across four datasets (FUNSD, CORD, SROIE, DocVQA) and six model families demonstrate that line-level attacks and compound perturbations (BBox + Pixel + Text) yield the most severe performance degradation. Projected Gradient Descent (PGD)-based BBox perturbations outperform random-shift baselines in all investigated models. Ablation studies further validate the impact of layout budget, text modification, and adversarial transferability. 

---
# Drag-and-Drop LLMs: Zero-Shot Prompt-to-Weights 

**Authors**: Zhiyuan Liang, Dongwen Tang, Yuhao Zhou, Xuanlei Zhao, Mingjia Shi, Wangbo Zhao, Zekai Li, Peihao Wang, Konstantin Schürholt, Damian Borth, Michael M. Bronstein, Yang You, Zhangyang Wang, Kai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16406)  

**Abstract**: Modern Parameter-Efficient Fine-Tuning (PEFT) methods such as low-rank adaptation (LoRA) reduce the cost of customizing large language models (LLMs), yet still require a separate optimization run for every downstream dataset. We introduce \textbf{Drag-and-Drop LLMs (\textit{DnD})}, a prompt-conditioned parameter generator that eliminates per-task training by mapping a handful of unlabeled task prompts directly to LoRA weight updates. A lightweight text encoder distills each prompt batch into condition embeddings, which are then transformed by a cascaded hyper-convolutional decoder into the full set of LoRA matrices. Once trained in a diverse collection of prompt-checkpoint pairs, DnD produces task-specific parameters in seconds, yielding i) up to \textbf{12,000$\times$} lower overhead than full fine-tuning, ii) average gains up to \textbf{30\%} in performance over the strongest training LoRAs on unseen common-sense reasoning, math, coding, and multimodal benchmarks, and iii) robust cross-domain generalization despite never seeing the target data or labels. Our results demonstrate that prompt-conditioned parameter generation is a viable alternative to gradient-based adaptation for rapidly specializing LLMs. Our project is available at \href{this https URL}{this https URL}. 

---
# NepaliGPT: A Generative Language Model for the Nepali Language 

**Authors**: Shushanta Pudasaini, Aman Shakya, Siddhartha Shrestha, Sahil Bhatta, Sunil Thapa, Sushmita Palikhe  

**Link**: [PDF](https://arxiv.org/pdf/2506.16399)  

**Abstract**: After the release of ChatGPT, Large Language Models (LLMs) have gained huge popularity in recent days and thousands of variants of LLMs have been released. However, there is no generative language model for the Nepali language, due to which other downstream tasks, including fine-tuning, have not been explored yet. To fill this research gap in the Nepali NLP space, this research proposes \textit{NepaliGPT}, a generative large language model tailored specifically for the Nepali language. This research introduces an advanced corpus for the Nepali language collected from several sources, called the Devanagari Corpus. Likewise, the research introduces the first NepaliGPT benchmark dataset comprised of 4,296 question-answer pairs in the Nepali language. The proposed LLM NepaliGPT achieves the following metrics in text generation: Perplexity of 26.32245, ROUGE-1 score of 0.2604, causal coherence of 81.25\%, and causal consistency of 85.41\%. 

---
# From LLM-anation to LLM-orchestrator: Coordinating Small Models for Data Labeling 

**Authors**: Yao Lu, Zhaiyuan Ji, Jiawei Du, Yu Shanqing, Qi Xuan, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.16393)  

**Abstract**: Although the annotation paradigm based on Large Language Models (LLMs) has made significant breakthroughs in recent years, its actual deployment still has two core bottlenecks: first, the cost of calling commercial APIs in large-scale annotation is very expensive; second, in scenarios that require fine-grained semantic understanding, such as sentiment classification and toxicity classification, the annotation accuracy of LLMs is even lower than that of Small Language Models (SLMs) dedicated to this field. To address these problems, we propose a new paradigm of multi-model cooperative annotation and design a fully automatic annotation framework AutoAnnotator based on this. Specifically, AutoAnnotator consists of two layers. The upper-level meta-controller layer uses the generation and reasoning capabilities of LLMs to select SLMs for annotation, automatically generate annotation code and verify difficult samples; the lower-level task-specialist layer consists of multiple SLMs that perform annotation through multi-model voting. In addition, we use the difficult samples obtained by the secondary review of the meta-controller layer as the reinforcement learning set and fine-tune the SLMs in stages through a continual learning strategy, thereby improving the generalization of SLMs. Extensive experiments show that AutoAnnotator outperforms existing open-source/API LLMs in zero-shot, one-shot, CoT, and majority voting settings. Notably, AutoAnnotator reduces the annotation cost by 74.15% compared to directly annotating with GPT-3.5-turbo, while still improving the accuracy by 6.21%. Project page: this https URL. 

---
# CLIP-MG: Guiding Semantic Attention with Skeletal Pose Features and RGB Data for Micro-Gesture Recognition on the iMiGUE Dataset 

**Authors**: Santosh Patapati, Trisanth Srinivasan, Amith Adiraju  

**Link**: [PDF](https://arxiv.org/pdf/2506.16385)  

**Abstract**: Micro-gesture recognition is a challenging task in affective computing due to the subtle, involuntary nature of the gestures and their low movement amplitude. In this paper, we introduce a Pose-Guided Semantics-Aware CLIP-based architecture, or CLIP for Micro-Gesture recognition (CLIP-MG), a modified CLIP model tailored for micro-gesture classification on the iMiGUE dataset. CLIP-MG integrates human pose (skeleton) information into the CLIP-based recognition pipeline through pose-guided semantic query generation and a gated multi-modal fusion mechanism. The proposed model achieves a Top-1 accuracy of 61.82%. These results demonstrate both the potential of our approach and the remaining difficulty in fully adapting vision-language models like CLIP for micro-gesture recognition. 

---
# Can structural correspondences ground real world representational content in Large Language Models? 

**Authors**: Iwan Williams  

**Link**: [PDF](https://arxiv.org/pdf/2506.16370)  

**Abstract**: Large Language Models (LLMs) such as GPT-4 produce compelling responses to a wide range of prompts. But their representational capacities are uncertain. Many LLMs have no direct contact with extra-linguistic reality: their inputs, outputs and training data consist solely of text, raising the questions (1) can LLMs represent anything and (2) if so, what? In this paper, I explore what it would take to answer these questions according to a structural-correspondence based account of representation, and make an initial survey of this evidence. I argue that the mere existence of structural correspondences between LLMs and worldly entities is insufficient to ground representation of those entities. However, if these structural correspondences play an appropriate role - they are exploited in a way that explains successful task performance - then they could ground real world contents. This requires overcoming a challenge: the text-boundedness of LLMs appears, on the face of it, to prevent them engaging in the right sorts of tasks. 

---
# Watermarking Autoregressive Image Generation 

**Authors**: Nikola Jovanović, Ismail Labiad, Tomáš Souček, Martin Vechev, Pierre Fernandez  

**Link**: [PDF](https://arxiv.org/pdf/2506.16349)  

**Abstract**: Watermarking the outputs of generative models has emerged as a promising approach for tracking their provenance. Despite significant interest in autoregressive image generation models and their potential for misuse, no prior work has attempted to watermark their outputs at the token level. In this work, we present the first such approach by adapting language model watermarking techniques to this setting. We identify a key challenge: the lack of reverse cycle-consistency (RCC), wherein re-tokenizing generated image tokens significantly alters the token sequence, effectively erasing the watermark. To address this and to make our method robust to common image transformations, neural compression, and removal attacks, we introduce (i) a custom tokenizer-detokenizer finetuning procedure that improves RCC, and (ii) a complementary watermark synchronization layer. As our experiments demonstrate, our approach enables reliable and robust watermark detection with theoretically grounded p-values. 

---
# Analyzing the Influence of Knowledge Graph Information on Relation Extraction 

**Authors**: Cedric Möller, Ricardo Usbeck  

**Link**: [PDF](https://arxiv.org/pdf/2506.16343)  

**Abstract**: We examine the impact of incorporating knowledge graph information on the performance of relation extraction models across a range of datasets. Our hypothesis is that the positions of entities within a knowledge graph provide important insights for relation extraction tasks. We conduct experiments on multiple datasets, each varying in the number of relations, training examples, and underlying knowledge graphs. Our results demonstrate that integrating knowledge graph information significantly enhances performance, especially when dealing with an imbalance in the number of training examples for each relation. We evaluate the contribution of knowledge graph-based features by combining established relation extraction methods with graph-aware Neural Bellman-Ford networks. These features are tested in both supervised and zero-shot settings, demonstrating consistent performance improvements across various datasets. 

---
# Reliable Few-shot Learning under Dual Noises 

**Authors**: Ji Zhang, Jingkuan Song, Lianli Gao, Nicu Sebe, Heng Tao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.16330)  

**Abstract**: Recent advances in model pre-training give rise to task adaptation-based few-shot learning (FSL), where the goal is to adapt a pre-trained task-agnostic model for capturing task-specific knowledge with a few-labeled support samples of the target this http URL, existing approaches may still fail in the open world due to the inevitable in-distribution (ID) and out-of-distribution (OOD) noise from both support and query samples of the target task. With limited support samples available, i) the adverse effect of the dual noises can be severely amplified during task adaptation, and ii) the adapted model can produce unreliable predictions on query samples in the presence of the dual noises. In this work, we propose DEnoised Task Adaptation (DETA++) for reliable FSL. DETA++ uses a Contrastive Relevance Aggregation (CoRA) module to calculate image and region weights for support samples, based on which a clean prototype loss and a noise entropy maximization loss are proposed to achieve noise-robust task adaptation. Additionally,DETA++ employs a memory bank to store and refine clean regions for each inner-task class, based on which a Local Nearest Centroid Classifier (LocalNCC) is devised to yield noise-robust predictions on query samples. Moreover, DETA++ utilizes an Intra-class Region Swapping (IntraSwap) strategy to rectify ID class prototypes during task adaptation, enhancing the model's robustness to the dual noises. Extensive experiments demonstrate the effectiveness and flexibility of DETA++. 

---
# Segment Anything for Satellite Imagery: A Strong Baseline and a Regional Dataset for Automatic Field Delineation 

**Authors**: Carmelo Scribano, Elena Govi, Paolo bertellini, Simone Parisi, Giorgia Franchini, Marko Bertogna  

**Link**: [PDF](https://arxiv.org/pdf/2506.16318)  

**Abstract**: Accurate mapping of agricultural field boundaries is essential for the efficient operation of agriculture. Automatic extraction from high-resolution satellite imagery, supported by computer vision techniques, can avoid costly ground surveys. In this paper, we present a pipeline for field delineation based on the Segment Anything Model (SAM), introducing a fine-tuning strategy to adapt SAM to this task. In addition to using published datasets, we describe a method for acquiring a complementary regional dataset that covers areas beyond current sources. Extensive experiments assess segmentation accuracy and evaluate the generalization capabilities. Our approach provides a robust baseline for automated field delineation. The new regional dataset, known as ERAS, is now publicly available. 

---
# Improved Exploration in GFlownets via Enhanced Epistemic Neural Networks 

**Authors**: Sajan Muhammad, Salem Lahlou  

**Link**: [PDF](https://arxiv.org/pdf/2506.16313)  

**Abstract**: Efficiently identifying the right trajectories for training remains an open problem in GFlowNets. To address this, it is essential to prioritize exploration in regions of the state space where the reward distribution has not been sufficiently learned. This calls for uncertainty-driven exploration, in other words, the agent should be aware of what it does not know. This attribute can be measured by joint predictions, which are particularly important for combinatorial and sequential decision problems. In this research, we integrate epistemic neural networks (ENN) with the conventional architecture of GFlowNets to enable more efficient joint predictions and better uncertainty quantification, thereby improving exploration and the identification of optimal trajectories. Our proposed algorithm, ENN-GFN-Enhanced, is compared to the baseline method in GFlownets and evaluated in grid environments and structured sequence generation in various settings, demonstrating both its efficacy and efficiency. 

---
# Learning Multi-scale Spatial-frequency Features for Image Denoising 

**Authors**: Xu Zhao, Chen Zhao, Xiantao Hu, Hongliang Zhang, Ying Tai, Jian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16307)  

**Abstract**: Recent advancements in multi-scale architectures have demonstrated exceptional performance in image denoising tasks. However, existing architectures mainly depends on a fixed single-input single-output Unet architecture, ignoring the multi-scale representations of pixel level. In addition, previous methods treat the frequency domain uniformly, ignoring the different characteristics of high-frequency and low-frequency noise. In this paper, we propose a novel multi-scale adaptive dual-domain network (MADNet) for image denoising. We use image pyramid inputs to restore noise-free results from low-resolution images. In order to realize the interaction of high-frequency and low-frequency information, we design an adaptive spatial-frequency learning unit (ASFU), where a learnable mask is used to separate the information into high-frequency and low-frequency components. In the skip connections, we design a global feature fusion block to enhance the features at different scales. Extensive experiments on both synthetic and real noisy image datasets verify the effectiveness of MADNet compared with current state-of-the-art denoising approaches. 

---
# SycnMapV2: Robust and Adaptive Unsupervised Segmentation 

**Authors**: Heng Zhang, Zikang Wan, Danilo Vasconcellos Vargas  

**Link**: [PDF](https://arxiv.org/pdf/2506.16297)  

**Abstract**: Human vision excels at segmenting visual cues without the need for explicit training, and it remains remarkably robust even as noise severity increases. In contrast, existing AI algorithms struggle to maintain accuracy under similar conditions. Here, we present SyncMapV2, the first to solve unsupervised segmentation with state-of-the-art robustness. SyncMapV2 exhibits a minimal drop in mIoU, only 0.01%, under digital corruption, compared to a 23.8% drop observed in SOTA this http URL superior performance extends across various types of corruption: noise (7.3% vs. 37.7%), weather (7.5% vs. 33.8%), and blur (7.0% vs. 29.5%). Notably, SyncMapV2 accomplishes this without any robust training, supervision, or loss functions. It is based on a learning paradigm that uses self-organizing dynamical equations combined with concepts from random networks. Moreover,unlike conventional methods that require re-initialization for each new input, SyncMapV2 adapts online, mimicking the continuous adaptability of human vision. Thus, we go beyond the accurate and robust results, and present the first algorithm that can do all the above online, adapting to input rather than re-initializing. In adaptability tests, SyncMapV2 demonstrates near-zero performance degradation, which motivates and fosters a new generation of robust and adaptive intelligence in the near future. 

---
# Next-Token Prediction Should be Ambiguity-Sensitive: A Meta-Learning Perspective 

**Authors**: Leo Gagnon, Eric Elmoznino, Sarthak Mittal, Tom Marty, Tejas Kasetty, Dhanya Sridhar, Guillaume Lajoie  

**Link**: [PDF](https://arxiv.org/pdf/2506.16288)  

**Abstract**: The rapid adaptation ability of auto-regressive foundation models is often attributed to the diversity of their pre-training data. This is because, from a Bayesian standpoint, minimizing prediction error in such settings requires integrating over all plausible latent hypotheses consistent with observations. While this behavior is desirable in principle, it often proves too ambitious in practice: under high ambiguity, the number of plausible latent alternatives makes Bayes-optimal prediction computationally intractable. Cognitive science has long recognized this limitation, suggesting that under such conditions, heuristics or information-seeking strategies are preferable to exhaustive inference. Translating this insight to next-token prediction, we hypothesize that low- and high-ambiguity predictions pose different computational demands, making ambiguity-agnostic next-token prediction a detrimental inductive bias. To test this, we introduce MetaHMM, a synthetic sequence meta-learning benchmark with rich compositional structure and a tractable Bayesian oracle. We show that Transformers indeed struggle with high-ambiguity predictions across model sizes. Motivated by cognitive theories, we propose a method to convert pre-trained models into Monte Carlo predictors that decouple task inference from token prediction. Preliminary results show substantial gains in ambiguous contexts through improved capacity allocation and test-time scalable inference, though challenges remain. 

---
# Artificial Intelligence for Atmospheric Sciences: A Research Roadmap 

**Authors**: Martha Arbayani Zaidan, Naser Hossein Motlagh, Petteri Nurmi, Tareq Hussein, Markku Kulmala, Tuukka Petäjä, Sasu Tarkoma  

**Link**: [PDF](https://arxiv.org/pdf/2506.16281)  

**Abstract**: Atmospheric sciences are crucial for understanding environmental phenomena ranging from air quality to extreme weather events, and climate change. Recent breakthroughs in sensing, communication, computing, and Artificial Intelligence (AI) have significantly advanced atmospheric sciences, enabling the generation of vast amounts of data through long-term Earth observations and providing powerful tools for analyzing atmospheric phenomena and predicting natural disasters. This paper contributes a critical interdisciplinary overview that bridges the fields of atmospheric science and computer science, highlighting the transformative potential of AI in atmospheric research. We identify key challenges associated with integrating AI into atmospheric research, including issues related to big data and infrastructure, and provide a detailed research roadmap that addresses both current and emerging challenges. 

---
# CapsDT: Diffusion-Transformer for Capsule Robot Manipulation 

**Authors**: Xiting He, Mingwu Su, Xinqi Jiang, Long Bai, Jiewen Lai, Hongliang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2506.16263)  

**Abstract**: Vision-Language-Action (VLA) models have emerged as a prominent research area, showcasing significant potential across a variety of applications. However, their performance in endoscopy robotics, particularly endoscopy capsule robots that perform actions within the digestive system, remains unexplored. The integration of VLA models into endoscopy robots allows more intuitive and efficient interactions between human operators and medical devices, improving both diagnostic accuracy and treatment outcomes. In this work, we design CapsDT, a Diffusion Transformer model for capsule robot manipulation in the stomach. By processing interleaved visual inputs, and textual instructions, CapsDT can infer corresponding robotic control signals to facilitate endoscopy tasks. In addition, we developed a capsule endoscopy robot system, a capsule robot controlled by a robotic arm-held magnet, addressing different levels of four endoscopy tasks and creating corresponding capsule robot datasets within the stomach simulator. Comprehensive evaluations on various robotic tasks indicate that CapsDT can serve as a robust vision-language generalist, achieving state-of-the-art performance in various levels of endoscopy tasks while achieving a 26.25% success rate in real-world simulation manipulation. 

---
# Category-based Galaxy Image Generation via Diffusion Models 

**Authors**: Xingzhong Fan, Hongming Tang, Yue Zeng, M.B.N.Kouwenhoven, Guangquan Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2506.16255)  

**Abstract**: Conventional galaxy generation methods rely on semi-analytical models and hydrodynamic simulations, which are highly dependent on physical assumptions and parameter tuning. In contrast, data-driven generative models do not have explicit physical parameters pre-determined, and instead learn them efficiently from observational data, making them alternative solutions to galaxy generation. Among these, diffusion models outperform Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs) in quality and diversity. Leveraging physical prior knowledge to these models can further enhance their capabilities. In this work, we present GalCatDiff, the first framework in astronomy to leverage both galaxy image features and astrophysical properties in the network design of diffusion models. GalCatDiff incorporates an enhanced U-Net and a novel block entitled Astro-RAB (Residual Attention Block), which dynamically combines attention mechanisms with convolution operations to ensure global consistency and local feature fidelity. Moreover, GalCatDiff uses category embeddings for class-specific galaxy generation, avoiding the high computational costs of training separate models for each category. Our experimental results demonstrate that GalCatDiff significantly outperforms existing methods in terms of the consistency of sample color and size distributions, and the generated galaxies are both visually realistic and physically consistent. This framework will enhance the reliability of galaxy simulations and can potentially serve as a data augmentor to support future galaxy classification algorithm development. 

---
# Synthetic ALS-EEG Data Augmentation for ALS Diagnosis Using Conditional WGAN with Weight Clipping 

**Authors**: Abdulvahap Mutlu, Şengül Doğan, Türker Tuncer  

**Link**: [PDF](https://arxiv.org/pdf/2506.16243)  

**Abstract**: Amyotrophic Lateral Sclerosis (ALS) is a rare neurodegenerative disease, and high-quality EEG data from ALS patients are scarce. This data scarcity, coupled with severe class imbalance between ALS and healthy control recordings, poses a challenge for training reliable machine learning classifiers. In this work, we address these issues by generating synthetic EEG signals for ALS patients using a Conditional Wasserstein Generative Adversarial Network (CWGAN). We train CWGAN on a private EEG dataset (ALS vs. non-ALS) to learn the distribution of ALS EEG signals and produce realistic synthetic samples. We preprocess and normalize EEG recordings, and train a CWGAN model to generate synthetic ALS signals. The CWGAN architecture and training routine are detailed, with key hyperparameters chosen for stable training. Qualitative evaluation of generated signals shows that they closely mimic real ALS EEG patterns. The CWGAN training converged with generator and discriminator loss curves stabilizing, indicating successful learning. The synthetic EEG signals appear realistic and have potential use as augmented data for training classifiers, helping to mitigate class imbalance and improve ALS detection accuracy. We discuss how this approach can facilitate data sharing and enhance diagnostic models. 

---
# CF-Seg: Counterfactuals meet Segmentation 

**Authors**: Raghav Mehta, Fabio De Sousa Ribeiro, Tian Xia, Melanie Roschewitz, Ainkaran Santhirasekaram, Dominic C. Marshall, Ben Glocker  

**Link**: [PDF](https://arxiv.org/pdf/2506.16213)  

**Abstract**: Segmenting anatomical structures in medical images plays an important role in the quantitative assessment of various diseases. However, accurate segmentation becomes significantly more challenging in the presence of disease. Disease patterns can alter the appearance of surrounding healthy tissues, introduce ambiguous boundaries, or even obscure critical anatomical structures. As such, segmentation models trained on real-world datasets may struggle to provide good anatomical segmentation, leading to potential misdiagnosis. In this paper, we generate counterfactual (CF) images to simulate how the same anatomy would appear in the absence of disease without altering the underlying structure. We then use these CF images to segment structures of interest, without requiring any changes to the underlying segmentation model. Our experiments on two real-world clinical chest X-ray datasets show that the use of counterfactual images improves anatomical segmentation, thereby aiding downstream clinical decision-making. 

---
# CP$^2$: Leveraging Geometry for Conformal Prediction via Canonicalization 

**Authors**: Putri A. van der Linden, Alexander Timans, Erik J. Bekkers  

**Link**: [PDF](https://arxiv.org/pdf/2506.16189)  

**Abstract**: We study the problem of conformal prediction (CP) under geometric data shifts, where data samples are susceptible to transformations such as rotations or flips. While CP endows prediction models with post-hoc uncertainty quantification and formal coverage guarantees, their practicality breaks under distribution shifts that deteriorate model performance. To address this issue, we propose integrating geometric information--such as geometric pose--into the conformal procedure to reinstate its guarantees and ensure robustness under geometric shifts. In particular, we explore recent advancements on pose canonicalization as a suitable information extractor for this purpose. Evaluating the combined approach across discrete and continuous shifts and against equivariant and augmentation-based baselines, we find that integrating geometric information with CP yields a principled way to address geometric shifts while maintaining broad applicability to black-box predictors. 

---
# JETHICS: Japanese Ethics Understanding Evaluation Dataset 

**Authors**: Masashi Takeshita, Rafal Rzepka  

**Link**: [PDF](https://arxiv.org/pdf/2506.16187)  

**Abstract**: In this work, we propose JETHICS, a Japanese dataset for evaluating ethics understanding of AI models. JETHICS contains 78K examples and is built by following the construction methods of the existing English ETHICS dataset. It includes four categories based normative theories and concepts from ethics and political philosophy; and one representing commonsense morality. Our evaluation experiments on non-proprietary large language models (LLMs) and on GPT-4o reveal that even GPT-4o achieves only an average score of about 0.7, while the best-performing Japanese LLM attains around 0.5, indicating a relatively large room for improvement in current LLMs. 

---
# From Teacher to Student: Tracking Memorization Through Model Distillation 

**Authors**: Simardeep Singh  

**Link**: [PDF](https://arxiv.org/pdf/2506.16170)  

**Abstract**: Large language models (LLMs) are known to memorize parts of their training data, raising important concerns around privacy and security. While previous research has focused on studying memorization in pre-trained models, much less is known about how knowledge distillation (KD) affects this http URL this study, we explore how different KD methods influence the memorization of fine-tuned task data when a large teacher model is distilled into smaller student this http URL study demonstrates that distilling a larger teacher model, fine-tuned on a dataset, into a smaller variant not only lowers computational costs and model size but also significantly reduces the memorization risks compared to standard fine-tuning approaches. 

---
# On using AI for EEG-based BCI applications: problems, current challenges and future trends 

**Authors**: Thomas Barbera, Jacopo Burger, Alessandro D'Amelio, Simone Zini, Simone Bianco, Raffaella Lanzarotti, Paolo Napoletano, Giuseppe Boccignone, Jose Luis Contreras-Vidal  

**Link**: [PDF](https://arxiv.org/pdf/2506.16168)  

**Abstract**: Imagine unlocking the power of the mind to communicate, create, and even interact with the world around us. Recent breakthroughs in Artificial Intelligence (AI), especially in how machines "see" and "understand" language, are now fueling exciting progress in decoding brain signals from scalp electroencephalography (EEG). Prima facie, this opens the door to revolutionary brain-computer interfaces (BCIs) designed for real life, moving beyond traditional uses to envision Brain-to-Speech, Brain-to-Image, and even a Brain-to-Internet of Things (BCIoT).
However, the journey is not as straightforward as it was for Computer Vision (CV) and Natural Language Processing (NLP). Applying AI to real-world EEG-based BCIs, particularly in building powerful foundational models, presents unique and intricate hurdles that could affect their reliability.
Here, we unfold a guided exploration of this dynamic and rapidly evolving research area. Rather than barely outlining a map of current endeavors and results, the goal is to provide a principled navigation of this hot and cutting-edge research landscape. We consider the basic paradigms that emerge from a causal perspective and the attendant challenges presented to AI-based models. Looking ahead, we then discuss promising research avenues that could overcome today's technological, methodological, and ethical limitations. Our aim is to lay out a clear roadmap for creating truly practical and effective EEG-based BCI solutions that can thrive in everyday environments. 

---
# Under the Shadow of Babel: How Language Shapes Reasoning in LLMs 

**Authors**: Chenxi Wang, Yixuan Zhang, Lang Gao, Zixiang Xu, Zirui Song, Yanbo Wang, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.16151)  

**Abstract**: Language is not only a tool for communication but also a medium for human cognition and reasoning. If, as linguistic relativity suggests, the structure of language shapes cognitive patterns, then large language models (LLMs) trained on human language may also internalize the habitual logical structures embedded in different languages. To examine this hypothesis, we introduce BICAUSE, a structured bilingual dataset for causal reasoning, which includes semantically aligned Chinese and English samples in both forward and reversed causal forms. Our study reveals three key findings: (1) LLMs exhibit typologically aligned attention patterns, focusing more on causes and sentence-initial connectives in Chinese, while showing a more balanced distribution in English. (2) Models internalize language-specific preferences for causal word order and often rigidly apply them to atypical inputs, leading to degraded performance, especially in Chinese. (3) When causal reasoning succeeds, model representations converge toward semantically aligned abstractions across languages, indicating a shared understanding beyond surface form. Overall, these results suggest that LLMs not only mimic surface linguistic forms but also internalize the reasoning biases shaped by language. Rooted in cognitive linguistic theory, this phenomenon is for the first time empirically verified through structural analysis of model internals. 

---
# PRISON: Unmasking the Criminal Potential of Large Language Models 

**Authors**: Xinyi Wu, Geng Hong, Pei Chen, Yueyue Chen, Xudong Pan, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16150)  

**Abstract**: As large language models (LLMs) advance, concerns about their misconduct in complex social contexts intensify. Existing research overlooked the systematic understanding and assessment of their criminal capability in realistic interactions. We propose a unified framework PRISON, to quantify LLMs' criminal potential across five dimensions: False Statements, Frame-Up, Psychological Manipulation, Emotional Disguise, and Moral Disengagement. Using structured crime scenarios adapted from classic films, we evaluate both criminal potential and anti-crime ability of LLMs via role-play. Results show that state-of-the-art LLMs frequently exhibit emergent criminal tendencies, such as proposing misleading statements or evasion tactics, even without explicit instructions. Moreover, when placed in a detective role, models recognize deceptive behavior with only 41% accuracy on average, revealing a striking mismatch between conducting and detecting criminal behavior. These findings underscore the urgent need for adversarial robustness, behavioral alignment, and safety mechanisms before broader LLM deployment. 

---
# GRPO-CARE: Consistency-Aware Reinforcement Learning for Multimodal Reasoning 

**Authors**: Yi Chen, Yuying Ge, Rui Wang, Yixiao Ge, Junhao Cheng, Ying Shan, Xihui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.16141)  

**Abstract**: Recent reinforcement learning approaches, such as outcome-supervised GRPO, have advanced Chain-of-Thought reasoning in large language models (LLMs), yet their adaptation to multimodal LLMs (MLLMs) is unexplored. To address the lack of rigorous evaluation for MLLM post-training methods, we introduce SEED-Bench-R1, a benchmark with complex real-world videos requiring balanced perception and reasoning. It offers a large training set and evaluates generalization across three escalating challenges: in-distribution, cross-environment, and cross-environment-task scenarios. Using SEED-Bench-R1, we find that standard GRPO, while improving answer accuracy, often reduces logical coherence between reasoning steps and answers, with only a 57.9% consistency rate. This stems from reward signals focusing solely on final answers, encouraging shortcuts, and strict KL penalties limiting this http URL address this, we propose GRPO-CARE, a consistency-aware RL framework optimizing both answer correctness and reasoning coherence without explicit supervision. GRPO-CARE introduces a two-tiered reward: (1) a base reward for answer correctness, and (2) an adaptive consistency bonus, computed by comparing the model's reasoning-to-answer likelihood (via a slowly-evolving reference model) against group this http URL dual mechanism amplifies rewards for reasoning paths that are both correct and logically consistent. Replacing KL penalties with this adaptive bonus, GRPO-CARE outperforms standard GRPO on SEED-Bench-R1, achieving a 6.7% performance gain on the hardest evaluation level and a 24.5% improvement in consistency. It also shows strong transferability, improving model performance across diverse video understanding benchmarks. Our work contributes a systematically designed benchmark and a generalizable post-training framework, advancing the development of more interpretable and robust MLLMs. 

---
# Improved Intelligibility of Dysarthric Speech using Conditional Flow Matching 

**Authors**: Shoutrik Das, Nishant Singh, Arjun Gangwar, S Umesh  

**Link**: [PDF](https://arxiv.org/pdf/2506.16127)  

**Abstract**: Dysarthria is a neurological disorder that significantly impairs speech intelligibility, often rendering affected individuals unable to communicate effectively. This necessitates the development of robust dysarthric-to-regular speech conversion techniques. In this work, we investigate the utility and limitations of self-supervised learning (SSL) features and their quantized representations as an alternative to mel-spectrograms for speech generation. Additionally, we explore methods to mitigate speaker variability by generating clean speech in a single-speaker voice using features extracted from WavLM. To this end, we propose a fully non-autoregressive approach that leverages Conditional Flow Matching (CFM) with Diffusion Transformers to learn a direct mapping from dysarthric to clean speech. Our findings highlight the effectiveness of discrete acoustic units in improving intelligibility while achieving faster convergence compared to traditional mel-spectrogram-based approaches. 

---
# GFlowGR: Fine-tuning Generative Recommendation Frameworks with Generative Flow Networks 

**Authors**: Yejing Wang, Shengyu Zhou, Jinyu Lu, Qidong Liu, Xinhang Li, Wenlin Zhang, Feng Li, Pengjie Wang, Jian Xu, Bo Zheng, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.16114)  

**Abstract**: Generative recommendations (GR), which usually include item tokenizers and generative Large Language Models (LLMs), have demonstrated remarkable success across a wide range of scenarios. The majority of existing research efforts primarily concentrate on developing powerful item tokenizers or advancing LLM decoding strategies to attain superior performance. However, the critical fine-tuning step in GR frameworks, which is essential for adapting LLMs to recommendation data, remains largely unexplored. Current approaches predominantly rely on either the next-token prediction loss of supervised fine-tuning (SFT) or recommendationspecific direct preference optimization (DPO) strategies. Both methods ignore the exploration of possible positive unobserved samples, which is commonly referred to as the exposure bias problem. To mitigate this problem, this paper treats the GR as a multi-step generation task and constructs a GFlowNets-based fine-tuning framework (GFlowGR). The proposed framework integrates collaborative knowledge from traditional recommender systems to create an adaptive trajectory sampler and a comprehensive reward model. Leveraging the diverse generation property of GFlowNets, along with sampling and heuristic weighting techniques, GFlowGR emerges as a promising approach to mitigate the exposure bias problem. Extensive empirical results on two real-world datasets and with two different GR backbones highlight the effectiveness and robustness of GFlowGR. 

---
# A Brain-to-Population Graph Learning Framework for Diagnosing Brain Disorders 

**Authors**: Qianqian Liao, Wuque Cai, Hongze Sun, Dongze Liu, Duo Chen, Dezhong Yao, Daqing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.16096)  

**Abstract**: Recent developed graph-based methods for diagnosing brain disorders using functional connectivity highly rely on predefined brain atlases, but overlook the rich information embedded within atlases and the confounding effects of site and phenotype variability. To address these challenges, we propose a two-stage Brain-to-Population Graph Learning (B2P-GL) framework that integrates the semantic similarity of brain regions and condition-based population graph modeling. In the first stage, termed brain representation learning, we leverage brain atlas knowledge from GPT-4 to enrich the graph representation and refine the brain graph through an adaptive node reassignment graph attention network. In the second stage, termed population disorder diagnosis, phenotypic data is incorporated into population graph construction and feature fusion to mitigate confounding effects and enhance diagnosis performance. Experiments on the ABIDE I, ADHD-200, and Rest-meta-MDD datasets show that B2P-GL outperforms state-of-the-art methods in prediction accuracy while enhancing interpretability. Overall, our proposed framework offers a reliable and personalized approach to brain disorder diagnosis, advancing clinical applicability. 

---
# Probing the Robustness of Large Language Models Safety to Latent Perturbations 

**Authors**: Tianle Gu, Kexin Huang, Zongqi Wang, Yixu Wang, Jie Li, Yuanqi Yao, Yang Yao, Yujiu Yang, Yan Teng, Yingchun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16078)  

**Abstract**: Safety alignment is a key requirement for building reliable Artificial General Intelligence. Despite significant advances in safety alignment, we observe that minor latent shifts can still trigger unsafe responses in aligned models. We argue that this stems from the shallow nature of existing alignment methods, which focus on surface-level refusal behaviors without sufficiently altering internal representations. Consequently, small shifts in hidden activations can re-trigger harmful behaviors embedded in the latent space. To explore the robustness of safety alignment to latent perturbations, we introduce a probing method that measures the Negative Log-Likelihood of the original response generated by the model. This probe quantifies local sensitivity in the latent space, serving as a diagnostic tool for identifying vulnerable directions. Based on this signal, we construct effective jailbreak trajectories, giving rise to the Activation Steering Attack (ASA). More importantly, these insights offer a principled foundation for improving alignment robustness. To this end, we introduce Layer-wise Adversarial Patch Training~(LAPT), a fine-tuning strategy that inject controlled perturbations into hidden representations during training. Experimental results highlight that LAPT strengthen alignment robustness without compromising general capabilities. Our findings reveal fundamental flaws in current alignment paradigms and call for representation-level training strategies that move beyond surface-level behavior supervision. Codes and results are available at this https URL. 

---
# CRIA: A Cross-View Interaction and Instance-Adapted Pre-training Framework for Generalizable EEG Representations 

**Authors**: Puchun Liu, C. L. Philip Chen, Yubin He, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16056)  

**Abstract**: The difficulty of extracting deep features from EEG data and effectively integrating information from multiple views presents significant challenges for developing a generalizable pretraining framework for EEG representation learning. However, most existing pre-training methods rely solely on the contextual semantics of a single view, failing to capture the complex and synergistic interactions among different perspectives, limiting the expressiveness and generalization of learned representations. To address these issues, this paper proposes CRIA, an adaptive framework that utilizes variable-length and variable-channel coding to achieve a unified representation of EEG data across different datasets. In this work, we define cross-view information as the integrated representation that emerges from the interaction among temporal, spectral, and spatial views of EEG signals. The model employs a cross-attention mechanism to fuse temporal, spectral, and spatial features effectively, and combines an attention matrix masking strategy based on the information bottleneck principle with a novel viewpoint masking pre-training scheme. Experimental results on the Temple University EEG corpus and the CHB-MIT dataset show that CRIA outperforms existing methods with the same pre-training conditions, achieving a balanced accuracy of 57.02% for multi-class event classification and 80.03% for anomaly detection, highlighting its strong generalization ability. 

---
# A Hybrid DeBERTa and Gated Broad Learning System for Cyberbullying Detection in English Text 

**Authors**: Devesh Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.16052)  

**Abstract**: The proliferation of online communication platforms has created unprecedented opportunities for global connectivity while simultaneously enabling harmful behaviors such as cyberbullying, which affects approximately 54.4\% of teenagers according to recent research. This paper presents a hybrid architecture that combines the contextual understanding capabilities of transformer-based models with the pattern recognition strengths of broad learning systems for effective cyberbullying detection. This approach integrates a modified DeBERTa model augmented with Squeeze-and-Excitation blocks and sentiment analysis capabilities with a Gated Broad Learning System (GBLS) classifier, creating a synergistic framework that outperforms existing approaches across multiple benchmark datasets. The proposed ModifiedDeBERTa + GBLS model achieved good performance on four English datasets: 79.3\% accuracy on HateXplain, 95.41\% accuracy on SOSNet, 91.37\% accuracy on Mendeley-I, and 94.67\% accuracy on Mendeley-II. Beyond performance gains, the framework incorporates comprehensive explainability mechanisms including token-level attribution analysis, LIME-based local interpretations, and confidence calibration, addressing critical transparency requirements in automated content moderation. Ablation studies confirm the meaningful contribution of each architectural component, while failure case analysis reveals specific challenges in detecting implicit bias and sarcastic content, providing valuable insights for future improvements in cyberbullying detection systems. 

---
# DynScaling: Efficient Verifier-free Inference Scaling via Dynamic and Integrated Sampling 

**Authors**: Fei Wang, Xingchen Wan, Ruoxi Sun, Jiefeng Chen, Sercan Ö. Arık  

**Link**: [PDF](https://arxiv.org/pdf/2506.16043)  

**Abstract**: Inference-time scaling has proven effective in boosting large language model (LLM) performance through increased test-time computation. Yet, its practical application is often hindered by reliance on external verifiers or a lack of optimization for realistic computational constraints. We propose DynScaling, which addresses these limitations through two primary innovations: an integrated parallel-sequential sampling strategy and a bandit-based dynamic budget allocation framework. The integrated sampling strategy unifies parallel and sequential sampling by constructing synthetic sequential reasoning chains from initially independent parallel responses, promoting diverse and coherent reasoning trajectories. The dynamic budget allocation framework formulates the allocation of computational resources as a multi-armed bandit problem, adaptively distributing the inference budget across queries based on the uncertainty of previously sampled responses, thereby maximizing computational efficiency. By combining these components, DynScaling effectively improves LLM performance under practical resource constraints without the need for external verifiers. Experimental results demonstrate that DynScaling consistently surpasses existing verifier-free inference scaling baselines in both task performance and computational cost. 

---
# Vision-Guided Chunking Is All You Need: Enhancing RAG with Multimodal Document Understanding 

**Authors**: Vishesh Tripathi, Tanmay Odapally, Indraneel Das, Uday Allu, Biddwan Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2506.16035)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems have revolutionized information retrieval and question answering, but traditional text-based chunking methods struggle with complex document structures, multi-page tables, embedded figures, and contextual dependencies across page boundaries. We present a novel multimodal document chunking approach that leverages Large Multimodal Models (LMMs) to process PDF documents in batches while maintaining semantic coherence and structural integrity. Our method processes documents in configurable page batches with cross-batch context preservation, enabling accurate handling of tables spanning multiple pages, embedded visual elements, and procedural content. We evaluate our approach on a curated dataset of PDF documents with manually crafted queries, demonstrating improvements in chunk quality and downstream RAG performance. Our vision-guided approach achieves better accuracy compared to traditional vanilla RAG systems, with qualitative analysis showing superior preservation of document structure and semantic coherence. 

---
# EvoLM: In Search of Lost Language Model Training Dynamics 

**Authors**: Zhenting Qi, Fan Nie, Alexandre Alahi, James Zou, Himabindu Lakkaraju, Yilun Du, Eric Xing, Sham Kakade, Hanlin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16029)  

**Abstract**: Modern language model (LM) training has been divided into multiple stages, making it difficult for downstream developers to evaluate the impact of design choices made at each stage. We present EvoLM, a model suite that enables systematic and transparent analysis of LMs' training dynamics across pre-training, continued pre-training, supervised fine-tuning, and reinforcement learning. By training over 100 LMs with 1B and 4B parameters from scratch, we rigorously evaluate both upstream (language modeling) and downstream (problem-solving) reasoning capabilities, including considerations of both in-domain and out-of-domain generalization. Key insights highlight the diminishing returns from excessive pre-training and post-training, the importance and practices of mitigating forgetting during domain-specific continued pre-training, the crucial role of continued pre-training in bridging pre-training and post-training phases, and various intricate trade-offs when configuring supervised fine-tuning and reinforcement learning. To facilitate open research and reproducibility, we release all pre-trained and post-trained models, training datasets for all stages, and our entire training and evaluation pipeline. 

---
# From General to Targeted Rewards: Surpassing GPT-4 in Open-Ended Long-Context Generation 

**Authors**: Zhihan Guo, Jiele Wu, Wenqian Cui, Yifei Zhang, Minda Hu, Yufei Wang, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2506.16024)  

**Abstract**: Current research on long-form context in Large Language Models (LLMs) primarily focuses on the understanding of long-contexts, the Open-ended Long Text Generation (Open-LTG) remains insufficiently explored. Training a long-context generation model requires curation of gold standard reference data, which is typically nonexistent for informative Open-LTG tasks. However, previous methods only utilize general assessments as reward signals, which limits accuracy. To bridge this gap, we introduce ProxyReward, an innovative reinforcement learning (RL) based framework, which includes a dataset and a reward signal computation method. Firstly, ProxyReward Dataset generation is accomplished through simple prompts that enables the model to create automatically, obviating extensive labeled data or significant manual effort. Secondly, ProxyReward Signal offers a targeted evaluation of information comprehensiveness and accuracy for specific questions. The experimental results indicate that our method ProxyReward surpasses even GPT-4-Turbo. It can significantly enhance performance by 20% on the Open-LTG task when training widely used open-source models, while also surpassing the LLM-as-a-Judge approach. Our work presents effective methods to enhance the ability of LLMs to address complex open-ended questions posed by human. 

---
# VRAIL: Vectorized Reward-based Attribution for Interpretable Learning 

**Authors**: Jina Kim, Youjin Jang, Jeongjin Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.16014)  

**Abstract**: We propose VRAIL (Vectorized Reward-based Attribution for Interpretable Learning), a bi-level framework for value-based reinforcement learning (RL) that learns interpretable weight representations from state features. VRAIL consists of two stages: a deep learning (DL) stage that fits an estimated value function using state features, and an RL stage that uses this to shape learning via potential-based reward transformations. The estimator is modeled in either linear or quadratic form, allowing attribution of importance to individual features and their interactions. Empirical results on the Taxi-v3 environment demonstrate that VRAIL improves training stability and convergence compared to standard DQN, without requiring environment modifications. Further analysis shows that VRAIL uncovers semantically meaningful subgoals, such as passenger possession, highlighting its ability to produce human-interpretable behavior. Our findings suggest that VRAIL serves as a general, model-agnostic framework for reward shaping that enhances both learning and interpretability. 

---
# DIGMAPPER: A Modular System for Automated Geologic Map Digitization 

**Authors**: Weiwei Duan, Michael P. Gerlek, Steven N. Minton, Craig A. Knoblock, Fandel Lin, Theresa Chen, Leeje Jang, Sofia Kirsanova, Zekun Li, Yijun Lin, Yao-Yi Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16006)  

**Abstract**: Historical geologic maps contain rich geospatial information, such as rock units, faults, folds, and bedding planes, that is critical for assessing mineral resources essential to renewable energy, electric vehicles, and national security. However, digitizing maps remains a labor-intensive and time-consuming task. We present DIGMAPPER, a modular, scalable system developed in collaboration with the United States Geological Survey (USGS) to automate the digitization of geologic maps. DIGMAPPER features a fully dockerized, workflow-orchestrated architecture that integrates state-of-the-art deep learning models for map layout analysis, feature extraction, and georeferencing. To overcome challenges such as limited training data and complex visual content, our system employs innovative techniques, including in-context learning with large language models, synthetic data generation, and transformer-based models. Evaluations on over 100 annotated maps from the DARPA-USGS dataset demonstrate high accuracy across polygon, line, and point feature extraction, and reliable georeferencing performance. Deployed at USGS, DIGMAPPER significantly accelerates the creation of analysis-ready geospatial datasets, supporting national-scale critical mineral assessments and broader geoscientific applications. 

---
# AutoHFormer: Efficient Hierarchical Autoregressive Transformer for Time Series Prediction 

**Authors**: Qianru Zhang, Honggang Wen, Ming Li, Dong Huang, Siu-Ming Yiu, Christian S. Jensen, Pietro Liò  

**Link**: [PDF](https://arxiv.org/pdf/2506.16001)  

**Abstract**: Time series forecasting requires architectures that simultaneously achieve three competing objectives: (1) strict temporal causality for reliable predictions, (2) sub-quadratic complexity for practical scalability, and (3) multi-scale pattern recognition for accurate long-horizon forecasting. We introduce AutoHFormer, a hierarchical autoregressive transformer that addresses these challenges through three key innovations: 1) Hierarchical Temporal Modeling: Our architecture decomposes predictions into segment-level blocks processed in parallel, followed by intra-segment sequential refinement. This dual-scale approach maintains temporal coherence while enabling efficient computation. 2) Dynamic Windowed Attention: The attention mechanism employs learnable causal windows with exponential decay, reducing complexity while preserving precise temporal relationships. This design avoids both the anti-causal violations of standard transformers and the sequential bottlenecks of RNN hybrids. 3) Adaptive Temporal Encoding: a novel position encoding system is adopted to capture time patterns at multiple scales. It combines fixed oscillating patterns for short-term variations with learnable decay rates for long-term trends. Comprehensive experiments demonstrate that AutoHFormer 10.76X faster training and 6.06X memory reduction compared to PatchTST on PEMS08, while maintaining consistent accuracy across 96-720 step horizons in most of cases. These breakthroughs establish new benchmarks for efficient and precise time series modeling. Implementations of our method and all baselines in hierarchical autoregressive mechanism are available at this https URL. 

---
# Quantum Artificial Intelligence for Secure Autonomous Vehicle Navigation: An Architectural Proposal 

**Authors**: Hemanth Kannamarlapudi, Sowmya Chintalapudi  

**Link**: [PDF](https://arxiv.org/pdf/2506.16000)  

**Abstract**: Navigation is a very crucial aspect of autonomous vehicle ecosystem which heavily relies on collecting and processing large amounts of data in various states and taking a confident and safe decision to define the next vehicle maneuver. In this paper, we propose a novel architecture based on Quantum Artificial Intelligence by enabling quantum and AI at various levels of navigation decision making and communication process in Autonomous vehicles : Quantum Neural Networks for multimodal sensor fusion, Nav-Q for Quantum reinforcement learning for navigation policy optimization and finally post-quantum cryptographic protocols for secure communication. Quantum neural networks uses quantum amplitude encoding to fuse data from various sensors like LiDAR, radar, camera, GPS and weather etc., This approach gives a unified quantum state representation between heterogeneous sensor modalities. Nav-Q module processes the fused quantum states through variational quantum circuits to learn optimal navigation policies under swift dynamic and complex conditions. Finally, post quantum cryptographic protocols are used to secure communication channels for both within vehicle communication and V2X (Vehicle to Everything) communications and thus secures the autonomous vehicle communication from both classical and quantum security threats. Thus, the proposed framework addresses fundamental challenges in autonomous vehicles navigation by providing quantum performance and future proof security. Index Terms Quantum Computing, Autonomous Vehicles, Sensor Fusion 

---
# Double Entendre: Robust Audio-Based AI-Generated Lyrics Detection via Multi-View Fusion 

**Authors**: Markus Frohmann, Gabriel Meseguer-Brocal, Markus Schedl, Elena V. Epure  

**Link**: [PDF](https://arxiv.org/pdf/2506.15981)  

**Abstract**: The rapid advancement of AI-based music generation tools is revolutionizing the music industry but also posing challenges to artists, copyright holders, and providers alike. This necessitates reliable methods for detecting such AI-generated content. However, existing detectors, relying on either audio or lyrics, face key practical limitations: audio-based detectors fail to generalize to new or unseen generators and are vulnerable to audio perturbations; lyrics-based methods require cleanly formatted and accurate lyrics, unavailable in practice. To overcome these limitations, we propose a novel, practically grounded approach: a multimodal, modular late-fusion pipeline that combines automatically transcribed sung lyrics and speech features capturing lyrics-related information within the audio. By relying on lyrical aspects directly from audio, our method enhances robustness, mitigates susceptibility to low-level artifacts, and enables practical applicability. Experiments show that our method, DE-detect, outperforms existing lyrics-based detectors while also being more robust to audio perturbations. Thus, it offers an effective, robust solution for detecting AI-generated music in real-world scenarios. Our code is available at this https URL. 

---
# Advanced Sign Language Video Generation with Compressed and Quantized Multi-Condition Tokenization 

**Authors**: Cong Wang, Zexuan Deng, Zhiwei Jiang, Fei Shen, Yafeng Yin, Shiwei Gan, Zifeng Cheng, Shiping Ge, Qing Gu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15980)  

**Abstract**: Sign Language Video Generation (SLVG) seeks to generate identity-preserving sign language videos from spoken language texts. Existing methods primarily rely on the single coarse condition (\eg, skeleton sequences) as the intermediary to bridge the translation model and the video generation model, which limits both the naturalness and expressiveness of the generated videos. To overcome these limitations, we propose SignViP, a novel SLVG framework that incorporates multiple fine-grained conditions for improved generation fidelity. Rather than directly translating error-prone high-dimensional conditions, SignViP adopts a discrete tokenization paradigm to integrate and represent fine-grained conditions (\ie, fine-grained poses and 3D hands). SignViP contains three core components. (1) Sign Video Diffusion Model is jointly trained with a multi-condition encoder to learn continuous embeddings that encapsulate fine-grained motion and appearance. (2) Finite Scalar Quantization (FSQ) Autoencoder is further trained to compress and quantize these embeddings into discrete tokens for compact representation of the conditions. (3) Multi-Condition Token Translator is trained to translate spoken language text to discrete multi-condition tokens. During inference, Multi-Condition Token Translator first translates the spoken language text into discrete multi-condition tokens. These tokens are then decoded to continuous embeddings by FSQ Autoencoder, which are subsequently injected into Sign Video Diffusion Model to guide video generation. Experimental results show that SignViP achieves state-of-the-art performance across metrics, including video quality, temporal coherence, and semantic fidelity. The code is available at this https URL. 

---
# A Vietnamese Dataset for Text Segmentation and Multiple Choices Reading Comprehension 

**Authors**: Toan Nguyen Hai, Ha Nguyen Viet, Truong Quan Xuan, Duc Do Minh  

**Link**: [PDF](https://arxiv.org/pdf/2506.15978)  

**Abstract**: Vietnamese, the 20th most spoken language with over 102 million native speakers, lacks robust resources for key natural language processing tasks such as text segmentation and machine reading comprehension (MRC). To address this gap, we present VSMRC, the Vietnamese Text Segmentation and Multiple-Choice Reading Comprehension Dataset. Sourced from Vietnamese Wikipedia, our dataset includes 15,942 documents for text segmentation and 16,347 synthetic multiple-choice question-answer pairs generated with human quality assurance, ensuring a reliable and diverse resource. Experiments show that mBERT consistently outperforms monolingual models on both tasks, achieving an accuracy of 88.01% on MRC test set and an F1 score of 63.15\% on text segmentation test set. Our analysis reveals that multilingual models excel in NLP tasks for Vietnamese, suggesting potential applications to other under-resourced languages. VSMRC is available at HuggingFace 

---
# Heterogeneous-Modal Unsupervised Domain Adaptation via Latent Space Bridging 

**Authors**: Jiawen Yang, Shuhao Chen, Yucong Duan, Ke Tang, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15971)  

**Abstract**: Unsupervised domain adaptation (UDA) methods effectively bridge domain gaps but become struggled when the source and target domains belong to entirely distinct modalities. To address this limitation, we propose a novel setting called Heterogeneous-Modal Unsupervised Domain Adaptation (HMUDA), which enables knowledge transfer between completely different modalities by leveraging a bridge domain containing unlabeled samples from both modalities. To learn under the HMUDA setting, we propose Latent Space Bridging (LSB), a specialized framework designed for the semantic segmentation task. Specifically, LSB utilizes a dual-branch architecture, incorporating a feature consistency loss to align representations across modalities and a domain alignment loss to reduce discrepancies between class centroids across domains. Extensive experiments conducted on six benchmark datasets demonstrate that LSB achieves state-of-the-art performance. 

---
# TrainVerify: Equivalence-Based Verification for Distributed LLM Training 

**Authors**: Yunchi Lu, Youshan Miao, Cheng Tan, Peng Huang, Yi Zhu, Xian Zhang, Fan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15961)  

**Abstract**: Training large language models (LLMs) at scale requires parallel execution across thousands of devices, incurring enormous computational costs. Yet, these costly distributed trainings are rarely verified, leaving them prone to silent errors and potentially wasting millions of GPU hours. We introduce TrainVerify, a system for verifiable distributed training of LLMs. Given a deep learning model's logical specification as the ground truth, TrainVerify formally verifies that a distributed parallel execution plan is mathematically equivalent to it. Direct verification is notoriously difficult due to the sheer scale of LLMs which often involves billions of variables and highly intricate computation graphs. Therefore, TrainVerify introduces shape-reduction techniques and a stage-wise parallel verification algorithm that significantly reduces complexity while preserving formal correctness. TrainVerify scales to frontier LLMs, including the successful verification of the Llama3 (405B) and DeepSeek-V3 (671B) training plans. 

---
# Beyond Audio and Pose: A General-Purpose Framework for Video Synchronization 

**Authors**: Yosub Shin, Igor Molybog  

**Link**: [PDF](https://arxiv.org/pdf/2506.15937)  

**Abstract**: Video synchronization-aligning multiple video streams capturing the same event from different angles-is crucial for applications such as reality TV show production, sports analysis, surveillance, and autonomous systems. Prior work has heavily relied on audio cues or specific visual events, limiting applicability in diverse settings where such signals may be unreliable or absent. Additionally, existing benchmarks for video synchronization lack generality and reproducibility, restricting progress in the field. In this work, we introduce VideoSync, a video synchronization framework that operates independently of specific feature extraction methods, such as human pose estimation, enabling broader applicability across different content types. We evaluate our system on newly composed datasets covering single-human, multi-human, and non-human scenarios, providing both the methodology and code for dataset creation to establish reproducible benchmarks. Our analysis reveals biases in prior SOTA work, particularly in SeSyn-Net's preprocessing pipeline, leading to inflated performance claims. We correct these biases and propose a more rigorous evaluation framework, demonstrating that VideoSync outperforms existing approaches, including SeSyn-Net, under fair experimental conditions. Additionally, we explore various synchronization offset prediction methods, identifying a convolutional neural network (CNN)-based model as the most effective. Our findings advance video synchronization beyond domain-specific constraints, making it more generalizable and robust for real-world applications. 

---
# MoiréXNet: Adaptive Multi-Scale Demoiréing with Linear Attention Test-Time Training and Truncated Flow Matching Prior 

**Authors**: Liangyan Li, Yimo Ning, Kevin Le, Wei Dong, Yunzhe Li, Jun Chen, Xiaohong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15929)  

**Abstract**: This paper introduces a novel framework for image and video demoiréing by integrating Maximum A Posteriori (MAP) estimation with advanced deep learning techniques. Demoiréing addresses inherently nonlinear degradation processes, which pose significant challenges for existing methods.
Traditional supervised learning approaches either fail to remove moiré patterns completely or produce overly smooth results. This stems from constrained model capacity and scarce training data, which inadequately represent the clean image distribution and hinder accurate reconstruction of ground-truth images. While generative models excel in image restoration for linear degradations, they struggle with nonlinear cases such as demoiréing and often introduce artifacts.
To address these limitations, we propose a hybrid MAP-based framework that integrates two complementary components. The first is a supervised learning model enhanced with efficient linear attention Test-Time Training (TTT) modules, which directly learn nonlinear mappings for RAW-to-sRGB demoiréing. The second is a Truncated Flow Matching Prior (TFMP) that further refines the outputs by aligning them with the clean image distribution, effectively restoring high-frequency details and suppressing artifacts. These two components combine the computational efficiency of linear attention with the refinement abilities of generative models, resulting in improved restoration performance. 

---
# PNCS:Power-Norm Cosine Similarity for Diverse Client Selection in Federated Learning 

**Authors**: Liangyan Li, Yangyi Liu, Yimo Ning, Stefano Rini, Jun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.15923)  

**Abstract**: Federated Learning (FL) has emerged as a powerful paradigm for leveraging diverse datasets from multiple sources while preserving data privacy by avoiding centralized storage. However, many existing approaches fail to account for the intricate gradient correlations between remote clients, a limitation that becomes especially problematic in data heterogeneity scenarios. In this work, we propose a novel FL framework utilizing Power-Norm Cosine Similarity (PNCS) to improve client selection for model aggregation. By capturing higher-order gradient moments, PNCS addresses non-IID data challenges, enhancing convergence speed and accuracy. Additionally, we introduce a simple algorithm ensuring diverse client selection through a selection history queue. Experiments with a VGG16 model across varied data partitions demonstrate consistent improvements over state-of-the-art methods. 

---
# KG-FGNN: Knowledge-guided GNN Foundation Model for Fertilisation-oriented Soil GHG Flux Prediction 

**Authors**: Yu Zhang, Gaoshan Bi, Simon Jeffery, Max Davis, Yang Li, Qing Xue, Po Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15896)  

**Abstract**: Precision soil greenhouse gas (GHG) flux prediction is essential in agricultural systems for assessing environmental impacts, developing emission mitigation strategies and promoting sustainable agriculture. Due to the lack of advanced sensor and network technologies on majority of farms, there are challenges in obtaining comprehensive and diverse agricultural data. As a result, the scarcity of agricultural data seriously obstructs the application of machine learning approaches in precision soil GHG flux prediction. This research proposes a knowledge-guided graph neural network framework that addresses the above challenges by integrating knowledge embedded in an agricultural process-based model and graph neural network techniques. Specifically, we utilise the agricultural process-based model to simulate and generate multi-dimensional agricultural datasets for 47 countries that cover a wide range of agricultural variables. To extract key agricultural features and integrate correlations among agricultural features in the prediction process, we propose a machine learning framework that integrates the autoencoder and multi-target multi-graph based graph neural networks, which utilises the autoencoder to selectively extract significant agricultural features from the agricultural process-based model simulation data and the graph neural network to integrate correlations among agricultural features for accurately predict fertilisation-oriented soil GHG fluxes. Comprehensive experiments were conducted with both the agricultural simulation dataset and real-world agricultural dataset to evaluate the proposed approach in comparison with well-known baseline and state-of-the-art regression methods. The results demonstrate that our proposed approach provides superior accuracy and stability in fertilisation-oriented soil GHG prediction. 

---
# Language Models can perform Single-Utterance Self-Correction of Perturbed Reasoning 

**Authors**: Sam Silver, Jimin Sun, Ivan Zhang, Sara Hooker, Eddie Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.15894)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive mathematical reasoning capabilities, yet their performance remains brittle to minor variations in problem description and prompting strategy. Furthermore, reasoning is vulnerable to sampling-induced errors which autoregressive models must primarily address using self-correction via additionally-generated tokens. To better understand self-correction capabilities of recent models, we conduct experiments measuring models' ability to self-correct synthetic perturbations introduced into their Chain of Thought (CoT) reasoning. We observe robust single-utterance intrinsic self-correction behavior across a range of open-weight models and datasets, ranging from subtle, implicit corrections to explicit acknowledgments and corrections of errors. Our findings suggest that LLMs, including those not finetuned for long CoT, may possess stronger intrinsic self-correction capabilities than commonly shown in the literature. The presence of this ability suggests that recent "reasoning" model work involves amplification of traits already meaningfully present in models. 

---
# Fractional Reasoning via Latent Steering Vectors Improves Inference Time Compute 

**Authors**: Sheng Liu, Tianlang Chen, Pan Lu, Haotian Ye, Yizheng Chen, Lei Xing, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2506.15882)  

**Abstract**: Test-time compute has emerged as a powerful paradigm for improving the performance of large language models (LLMs), where generating multiple outputs or refining individual chains can significantly boost answer accuracy. However, existing methods like Best-of-N, majority voting, and self-reflection typically apply reasoning in a uniform way across inputs, overlooking the fact that different problems may require different levels of reasoning depth. In this work, we propose Fractional Reasoning, a training-free and model-agnostic framework that enables continuous control over reasoning intensity at inference time, going beyond the limitations of fixed instructional prompts. Our method operates by extracting the latent steering vector associated with deeper reasoning and reapplying it with a tunable scaling factor, allowing the model to tailor its reasoning process to the complexity of each input. This supports two key modes of test-time scaling: (1) improving output quality in breadth-based strategies (e.g., Best-of-N, majority voting), and (2) enhancing the correctness of individual reasoning chains in depth-based strategies (e.g., self-reflection). Experiments on GSM8K, MATH500, and GPQA demonstrate that Fractional Reasoning consistently improves performance across diverse reasoning tasks and models. 

---
# MoR: Better Handling Diverse Queries with a Mixture of Sparse, Dense, and Human Retrievers 

**Authors**: Jushaan Singh Kalra, Xinran Zhao, To Eun Kim, Fengyu Cai, Fernando Diaz, Tongshuang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15862)  

**Abstract**: Retrieval-augmented Generation (RAG) is powerful, but its effectiveness hinges on which retrievers we use and how. Different retrievers offer distinct, often complementary signals: BM25 captures lexical matches; dense retrievers, semantic similarity. Yet in practice, we typically fix a single retriever based on heuristics, which fails to generalize across diverse information needs. Can we dynamically select and integrate multiple retrievers for each individual query, without the need for manual selection? In our work, we validate this intuition with quantitative analysis and introduce mixture of retrievers: a zero-shot, weighted combination of heterogeneous retrievers. Extensive experiments show that such mixtures are effective and efficient: Despite totaling just 0.8B parameters, this mixture outperforms every individual retriever and even larger 7B models by +10.8% and +3.9% on average, respectively. Further analysis also shows that this mixture framework can help incorporate specialized non-oracle human information sources as retrievers to achieve good collaboration, with a 58.9% relative performance improvement over simulated humans alone. 

---
# Cross-Modality Learning for Predicting IHC Biomarkers from H&E-Stained Whole-Slide Images 

**Authors**: Amit Das, Naofumi Tomita, Kyle J. Syme, Weijie Ma, Paige O'Connor, Kristin N. Corbett, Bing Ren, Xiaoying Liu, Saeed Hassanpour  

**Link**: [PDF](https://arxiv.org/pdf/2506.15853)  

**Abstract**: Hematoxylin and Eosin (H&E) staining is a cornerstone of pathological analysis, offering reliable visualization of cellular morphology and tissue architecture for cancer diagnosis, subtyping, and grading. Immunohistochemistry (IHC) staining provides molecular insights by detecting specific proteins within tissues, enhancing diagnostic accuracy, and improving treatment planning. However, IHC staining is costly, time-consuming, and resource-intensive, requiring specialized expertise. To address these limitations, this study proposes HistoStainAlign, a novel deep learning framework that predicts IHC staining patterns directly from H&E whole-slide images (WSIs) by learning joint representations of morphological and molecular features. The framework integrates paired H&E and IHC embeddings through a contrastive training strategy, capturing complementary features across staining modalities without patch-level annotations or tissue registration. The model was evaluated on gastrointestinal and lung tissue WSIs with three commonly used IHC stains: P53, PD-L1, and Ki-67. HistoStainAlign achieved weighted F1 scores of 0.735 [95% Confidence Interval (CI): 0.670-0.799], 0.830 [95% CI: 0.772-0.886], and 0.723 [95% CI: 0.607-0.836], respectively for these three IHC stains. Embedding analyses demonstrated the robustness of the contrastive alignment in capturing meaningful cross-stain relationships. Comparisons with a baseline model further highlight the advantage of incorporating contrastive learning for improved stain pattern prediction. This study demonstrates the potential of computational approaches to serve as a pre-screening tool, helping prioritize cases for IHC staining and improving workflow efficiency. 

---
# Uncertainty Estimation by Human Perception versus Neural Models 

**Authors**: Pedro Mendes, Paolo Romano, David Garlan  

**Link**: [PDF](https://arxiv.org/pdf/2506.15850)  

**Abstract**: Modern neural networks (NNs) often achieve high predictive accuracy but remain poorly calibrated, producing overconfident predictions even when wrong. This miscalibration poses serious challenges in applications where reliable uncertainty estimates are critical. In this work, we investigate how human perceptual uncertainty compares to uncertainty estimated by NNs. Using three vision benchmarks annotated with both human disagreement and crowdsourced confidence, we assess the correlation between model-predicted uncertainty and human-perceived uncertainty. Our results show that current methods only weakly align with human intuition, with correlations varying significantly across tasks and uncertainty metrics. Notably, we find that incorporating human-derived soft labels into the training process can improve calibration without compromising accuracy. These findings reveal a persistent gap between model and human uncertainty and highlight the potential of leveraging human insights to guide the development of more trustworthy AI systems. 

---
# SafeMimic: Towards Safe and Autonomous Human-to-Robot Imitation for Mobile Manipulation 

**Authors**: Arpit Bahety, Arnav Balaji, Ben Abbatematteo, Roberto Martín-Martín  

**Link**: [PDF](https://arxiv.org/pdf/2506.15847)  

**Abstract**: For robots to become efficient helpers in the home, they must learn to perform new mobile manipulation tasks simply by watching humans perform them. Learning from a single video demonstration from a human is challenging as the robot needs to first extract from the demo what needs to be done and how, translate the strategy from a third to a first-person perspective, and then adapt it to be successful with its own morphology. Furthermore, to mitigate the dependency on costly human monitoring, this learning process should be performed in a safe and autonomous manner. We present SafeMimic, a framework to learn new mobile manipulation skills safely and autonomously from a single third-person human video. Given an initial human video demonstration of a multi-step mobile manipulation task, SafeMimic first parses the video into segments, inferring both the semantic changes caused and the motions the human executed to achieve them and translating them to an egocentric reference. Then, it adapts the behavior to the robot's own morphology by sampling candidate actions around the human ones, and verifying them for safety before execution in a receding horizon fashion using an ensemble of safety Q-functions trained in simulation. When safe forward progression is not possible, SafeMimic backtracks to previous states and attempts a different sequence of actions, adapting both the trajectory and the grasping modes when required for its morphology. As a result, SafeMimic yields a strategy that succeeds in the demonstrated behavior and learns task-specific actions that reduce exploration in future attempts. Our experiments show that our method allows robots to safely and efficiently learn multi-step mobile manipulation behaviors from a single human demonstration, from different users, and in different environments, with improvements over state-of-the-art baselines across seven tasks 

---
# Finance Language Model Evaluation (FLaME) 

**Authors**: Glenn Matlin, Mika Okamoto, Huzaifa Pardawala, Yang Yang, Sudheer Chava  

**Link**: [PDF](https://arxiv.org/pdf/2506.15846)  

**Abstract**: Language Models (LMs) have demonstrated impressive capabilities with core Natural Language Processing (NLP) tasks. The effectiveness of LMs for highly specialized knowledge-intensive tasks in finance remains difficult to assess due to major gaps in the methodologies of existing evaluation frameworks, which have caused an erroneous belief in a far lower bound of LMs' performance on common Finance NLP (FinNLP) tasks. To demonstrate the potential of LMs for these FinNLP tasks, we present the first holistic benchmarking suite for Financial Language Model Evaluation (FLaME). We are the first research paper to comprehensively study LMs against 'reasoning-reinforced' LMs, with an empirical study of 23 foundation LMs over 20 core NLP tasks in finance. We open-source our framework software along with all data and results. 

---
# MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents 

**Authors**: Zijian Zhou, Ao Qu, Zhaoxuan Wu, Sunghwan Kim, Alok Prakash, Daniela Rus, Jinhua Zhao, Bryan Kian Hsiang Low, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15841)  

**Abstract**: Modern language agents must operate over long-horizon, multi-turn interactions, where they retrieve external information, adapt to observations, and answer interdependent queries. Yet, most LLM systems rely on full-context prompting, appending all past turns regardless of their relevance. This leads to unbounded memory growth, increased computational costs, and degraded reasoning performance on out-of-distribution input lengths. We introduce MEM1, an end-to-end reinforcement learning framework that enables agents to operate with constant memory across long multi-turn tasks. At each turn, MEM1 updates a compact shared internal state that jointly supports memory consolidation and reasoning. This state integrates prior memory with new observations from the environment while strategically discarding irrelevant or redundant information. To support training in more realistic and compositional settings, we propose a simple yet effective and scalable approach to constructing multi-turn environments by composing existing datasets into arbitrarily complex task sequences. Experiments across three domains, including internal retrieval QA, open-domain web QA, and multi-turn web shopping, show that MEM1-7B improves performance by 3.5x while reducing memory usage by 3.7x compared to Qwen2.5-14B-Instruct on a 16-objective multi-hop QA task, and generalizes beyond the training horizon. Our results demonstrate the promise of reasoning-driven memory consolidation as a scalable alternative to existing solutions for training long-horizon interactive agents, where both efficiency and performance are optimized. 

---
# MoNetV2: Enhanced Motion Network for Freehand 3D Ultrasound Reconstruction 

**Authors**: Mingyuan Luo, Xin Yang, Zhongnuo Yan, Yan Cao, Yuanji Zhang, Xindi Hu, Jin Wang, Haoxuan Ding, Wei Han, Litao Sun, Dong Ni  

**Link**: [PDF](https://arxiv.org/pdf/2506.15835)  

**Abstract**: Three-dimensional (3D) ultrasound (US) aims to provide sonographers with the spatial relationships of anatomical structures, playing a crucial role in clinical diagnosis. Recently, deep-learning-based freehand 3D US has made significant advancements. It reconstructs volumes by estimating transformations between images without external tracking. However, image-only reconstruction poses difficulties in reducing cumulative drift and further improving reconstruction accuracy, particularly in scenarios involving complex motion trajectories. In this context, we propose an enhanced motion network (MoNetV2) to enhance the accuracy and generalizability of reconstruction under diverse scanning velocities and tactics. First, we propose a sensor-based temporal and multi-branch structure that fuses image and motion information from a velocity perspective to improve image-only reconstruction accuracy. Second, we devise an online multi-level consistency constraint that exploits the inherent consistency of scans to handle various scanning velocities and tactics. This constraint exploits both scan-level velocity consistency, path-level appearance consistency, and patch-level motion consistency to supervise inter-frame transformation estimation. Third, we distill an online multi-modal self-supervised strategy that leverages the correlation between network estimation and motion information to further reduce cumulative errors. Extensive experiments clearly demonstrate that MoNetV2 surpasses existing methods in both reconstruction quality and generalizability performance across three large datasets. 

---
# Context Matters! Relaxing Goals with LLMs for Feasible 3D Scene Planning 

**Authors**: Emanuele Musumeci, Michele Brienza, Francesco Argenziano, Vincenzo Suriani, Daniele Nardi, Domenico D. Bloisi  

**Link**: [PDF](https://arxiv.org/pdf/2506.15828)  

**Abstract**: Classical planning in AI and Robotics addresses complex tasks by shifting from imperative to declarative approaches (e.g., PDDL). However, these methods often fail in real scenarios due to limited robot perception and the need to ground perceptions to planning predicates. This often results in heavily hard-coded behaviors that struggle to adapt, even with scenarios where goals can be achieved through relaxed planning. Meanwhile, Large Language Models (LLMs) lead to planning systems that leverage commonsense reasoning but often at the cost of generating unfeasible and/or unsafe plans. To address these limitations, we present an approach integrating classical planning with LLMs, leveraging their ability to extract commonsense knowledge and ground actions. We propose a hierarchical formulation that enables robots to make unfeasible tasks tractable by defining functionally equivalent goals through gradual relaxation. This mechanism supports partial achievement of the intended objective, suited to the agent's specific context. Our method demonstrates its ability to adapt and execute tasks effectively within environments modeled using 3D Scene Graphs through comprehensive qualitative and quantitative evaluations. We also show how this method succeeds in complex scenarios where other benchmark methods are more likely to fail. Code, dataset, and additional material are released to the community. 

---
# VEIGAR: View-consistent Explicit Inpainting and Geometry Alignment for 3D object Removal 

**Authors**: Pham Khai Nguyen Do, Bao Nguyen Tran, Nam Nguyen, Duc Dung Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2506.15821)  

**Abstract**: Recent advances in Novel View Synthesis (NVS) and 3D generation have significantly improved editing tasks, with a primary emphasis on maintaining cross-view consistency throughout the generative process. Contemporary methods typically address this challenge using a dual-strategy framework: performing consistent 2D inpainting across all views guided by embedded priors either explicitly in pixel space or implicitly in latent space; and conducting 3D reconstruction with additional consistency guidance. Previous strategies, in particular, often require an initial 3D reconstruction phase to establish geometric structure, introducing considerable computational overhead. Even with the added cost, the resulting reconstruction quality often remains suboptimal. In this paper, we present VEIGAR, a computationally efficient framework that outperforms existing methods without relying on an initial reconstruction phase. VEIGAR leverages a lightweight foundation model to reliably align priors explicitly in the pixel space. In addition, we introduce a novel supervision strategy based on scale-invariant depth loss, which removes the need for traditional scale-and-shift operations in monocular depth regularization. Through extensive experimentation, VEIGAR establishes a new state-of-the-art benchmark in reconstruction quality and cross-view consistency, while achieving a threefold reduction in training time compared to the fastest existing method, highlighting its superior balance of efficiency and effectiveness. 

---
# Unsupervised deep learning model for fast energy layer pre-selection of delivery-efficient proton arc therapy plan optimization of nasopharyngeal carcinoma 

**Authors**: Bohan Yang, Gang Liu, Rirao Dao, Yujia Qian, Ke Shi, Anke Tang, Yong Luo, Jingnan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15803)  

**Abstract**: Objective. Proton arc therapy (PAT) is an emerging and promising modality in radiotherapy, offering several advantages over conventional intensitymodulated proton therapy (IMPT). However, identifying the optimal energy layer (EL) sequence remains computationally intensive due to the large number of possible energy layer transitions. This study proposes an unsupervised deep learning framework for fast and effective EL pre-selection, aiming to minimize energy layer switch time while preserving high plan quality. Approach. We introduce a novel data representation method, spot-count representation, which encodes the number of proton spots intersecting the target and organs at risk (OARs) in a matrix structured by sorted gantry angles and energy layers. This representation is the input of a UNet-based architecture, SPArcdl, which is trained to optimize a tri-objective function: maximizing target coverage, minimizing OAR exposure, and reducing energy switching time. The model is evaluated on 54 nasopharyngeal cancer cases, and its performance is benchmarked against plans generated by SPArcparticle swarm. Main results. SPArcdl produces EL pre-selection that significantly improves both plan quality and delivery efficiency. Compared to SPArc particle swarm, it enhances the conformity index by 0.16 (p < 0.01), reduces the homogeneity index by 0.71 (p < 0.01), shortens the energy switching time by 38.4% (p < 0.01), and lowers the mean dose to brainstem by 0.21 (p < 0.01). The results unintentionally reveal employing unchanged ELS is more time-wise efficient than descended ELS. SPArcdl's inference time is within 1 second. Significance. SPArcdl is a fast and effective tool for generating high-quality PAT plans by strategically pre-selecting energy layers to reduce delivery time while maintaining excellent dosimetric performance. 

---
# Veracity: An Open-Source AI Fact-Checking System 

**Authors**: Taylor Lynn Curtis, Maximilian Puelma Touzel, William Garneau, Manon Gruaz, Mike Pinder, Li Wei Wang, Sukanya Krishna, Luda Cohen, Jean-François Godbout, Reihaneh Rabbany, Kellin Pelrine  

**Link**: [PDF](https://arxiv.org/pdf/2506.15794)  

**Abstract**: The proliferation of misinformation poses a significant threat to society, exacerbated by the capabilities of generative AI. This demo paper introduces Veracity, an open-source AI system designed to empower individuals to combat misinformation through transparent and accessible fact-checking. Veracity leverages the synergy between Large Language Models (LLMs) and web retrieval agents to analyze user-submitted claims and provide grounded veracity assessments with intuitive explanations. Key features include multilingual support, numerical scoring of claim veracity, and an interactive interface inspired by familiar messaging applications. This paper will showcase Veracity's ability to not only detect misinformation but also explain its reasoning, fostering media literacy and promoting a more informed society. 

---
# Linearithmic Clean-up for Vector-Symbolic Key-Value Memory with Kroneker Rotation Products 

**Authors**: Ruipeng Liu, Qinru Qiu, Simon Khan, Garrett E. Katz  

**Link**: [PDF](https://arxiv.org/pdf/2506.15793)  

**Abstract**: A computational bottleneck in current Vector-Symbolic Architectures (VSAs) is the ``clean-up'' step, which decodes the noisy vectors retrieved from the architecture. Clean-up typically compares noisy vectors against a ``codebook'' of prototype vectors, incurring computational complexity that is quadratic or similar. We present a new codebook representation that supports efficient clean-up, based on Kroneker products of rotation-like matrices. The resulting clean-up time complexity is linearithmic, i.e. $\mathcal{O}(N\,\text{log}\,N)$, where $N$ is the vector dimension and also the number of vectors in the codebook. Clean-up space complexity is $\mathcal{O}(N)$. Furthermore, the codebook is not stored explicitly in computer memory: It can be represented in $\mathcal{O}(\text{log}\,N)$ space, and individual vectors in the codebook can be materialized in $\mathcal{O}(N)$ time and space. At the same time, asymptotic memory capacity remains comparable to standard approaches. Computer experiments confirm these results, demonstrating several orders of magnitude more scalability than baseline VSA techniques. 

---
# TRUST: Transparent, Robust and Ultra-Sparse Trees 

**Authors**: Albert Dorador  

**Link**: [PDF](https://arxiv.org/pdf/2506.15791)  

**Abstract**: Piecewise-constant regression trees remain popular for their interpretability, yet often lag behind black-box models like Random Forest in predictive accuracy. In this work, we introduce TRUST (Transparent, Robust, and Ultra-Sparse Trees), a novel regression tree model that combines the accuracy of Random Forests with the interpretability of shallow decision trees and sparse linear models. TRUST further enhances transparency by leveraging Large Language Models to generate tailored, user-friendly explanations. Extensive validation on synthetic and real-world benchmark datasets demonstrates that TRUST consistently outperforms other interpretable models -- including CART, Lasso, and Node Harvest -- in predictive accuracy, while matching the accuracy of Random Forest and offering substantial gains in both accuracy and interpretability over M5', a well-established model that is conceptually related. 

---
# Graphics4Science: Computer Graphics for Scientific Impacts 

**Authors**: Peter Yichen Chen, Minghao Guo, Hanspeter Pfister, Ming Lin, William Freeman, Qixing Huang, Han-Wei Shen, Wojciech Matusik  

**Link**: [PDF](https://arxiv.org/pdf/2506.15786)  

**Abstract**: Computer graphics, often associated with films, games, and visual effects, has long been a powerful tool for addressing scientific challenges--from its origins in 3D visualization for medical imaging to its role in modern computational modeling and simulation. This course explores the deep and evolving relationship between computer graphics and science, highlighting past achievements, ongoing contributions, and open questions that remain. We show how core methods, such as geometric reasoning and physical modeling, provide inductive biases that help address challenges in both fields, especially in data-scarce settings. To that end, we aim to reframe graphics as a modeling language for science by bridging vocabulary gaps between the two communities. Designed for both newcomers and experts, Graphics4Science invites the graphics community to engage with science, tackle high-impact problems where graphics expertise can make a difference, and contribute to the future of scientific discovery. Additional details are available on the course website: this https URL 

---
# RecBayes: Recurrent Bayesian Ad Hoc Teamwork in Large Partially Observable Domains 

**Authors**: João G. Ribeiro, Yaniv Oren, Alberto Sardinha, Matthijs Spaan, Francisco S. Melo  

**Link**: [PDF](https://arxiv.org/pdf/2506.15756)  

**Abstract**: This paper proposes RecBayes, a novel approach for ad hoc teamwork under partial observability, a setting where agents are deployed on-the-fly to environments where pre-existing teams operate, that never requires, at any stage, access to the states of the environment or the actions of its teammates. We show that by relying on a recurrent Bayesian classifier trained using past experiences, an ad hoc agent is effectively able to identify known teams and tasks being performed from observations alone. Unlike recent approaches such as PO-GPL (Gu et al., 2021) and FEAT (Rahman et al., 2023), that require at some stage fully observable states of the environment, actions of teammates, or both, or approaches such as ATPO (Ribeiro et al., 2023) that require the environments to be small enough to be tabularly modelled (Ribeiro et al., 2023), in their work up to 4.8K states and 1.7K observations, we show RecBayes is both able to handle arbitrarily large spaces while never relying on either states and teammates' actions. Our results in benchmark domains from the multi-agent systems literature, adapted for partial observability and scaled up to 1M states and 2^125 observations, show that RecBayes is effective at identifying known teams and tasks being performed from partial observations alone, and as a result, is able to assist the teams in solving the tasks effectively. 

---
# A Study of Hybrid and Evolutionary Metaheuristics for Single Hidden Layer Feedforward Neural Network Architecture 

**Authors**: Gautam Siddharth Kashyap, Md Tabrez Nafis, Samar Wazir  

**Link**: [PDF](https://arxiv.org/pdf/2506.15737)  

**Abstract**: Training Artificial Neural Networks (ANNs) with Stochastic Gradient Descent (SGD) frequently encounters difficulties, including substantial computing expense and the risk of converging to local optima, attributable to its dependence on partial weight gradients. Therefore, this work investigates Particle Swarm Optimization (PSO) and Genetic Algorithms (GAs) - two population-based Metaheuristic Optimizers (MHOs) - as alternatives to SGD to mitigate these constraints. A hybrid PSO-SGD strategy is developed to improve local search efficiency. The findings indicate that the hybrid PSO-SGD technique decreases the median training MSE by 90 to 95 percent relative to conventional GA and PSO across various network sizes (e.g., from around 0.02 to approximately 0.001 in the Sphere function). RMHC attains substantial enhancements, reducing MSE by roughly 85 to 90 percent compared to GA. Simultaneously, RS consistently exhibits errors exceeding 0.3, signifying subpar performance. These findings underscore that hybrid and evolutionary procedures significantly improve training efficiency and accuracy compared to conventional optimization methods and imply that the Building Block Hypothesis (BBH) may still be valid, indicating that advantageous weight structures are retained during evolutionary search. 

---
# Graph Diffusion that can Insert and Delete 

**Authors**: Matteo Ninniri, Marco Podda, Davide Bacciu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15725)  

**Abstract**: Generative models of graphs based on discrete Denoising Diffusion Probabilistic Models (DDPMs) offer a principled approach to molecular generation by systematically removing structural noise through iterative atom and bond adjustments. However, existing formulations are fundamentally limited by their inability to adapt the graph size (that is, the number of atoms) during the diffusion process, severely restricting their effectiveness in conditional generation scenarios such as property-driven molecular design, where the targeted property often correlates with the molecular size. In this paper, we reformulate the noising and denoising processes to support monotonic insertion and deletion of nodes. The resulting model, which we call GrIDDD, dynamically grows or shrinks the chemical graph during generation. GrIDDD matches or exceeds the performance of existing graph diffusion models on molecular property targeting despite being trained on a more difficult problem. Furthermore, when applied to molecular optimization, GrIDDD exhibits competitive performance compared to specialized optimization models. This work paves the way for size-adaptive molecular generation with graph diffusion. 

---
# MadaKV: Adaptive Modality-Perception KV Cache Eviction for Efficient Multimodal Long-Context Inference 

**Authors**: Kunxi Li, Zhonghua Jiang, Zhouzhou Shen, Zhaode Wang, Chengfei Lv, Shengyu Zhang, Fan Wu, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15724)  

**Abstract**: This paper introduces MadaKV, a modality-adaptive key-value (KV) cache eviction strategy designed to enhance the efficiency of multimodal large language models (MLLMs) in long-context inference. In multimodal scenarios, attention heads exhibit varying preferences for different modalities, resulting in significant disparities in modality importance across attention heads. Traditional KV cache eviction methods, which are tailored for unimodal settings, fail to capture modality-specific information, thereby yielding suboptimal performance. MadaKV addresses these challenges through two key components: modality preference adaptation and hierarchical compression compensation. By dynamically sensing modality information within attention heads and adaptively retaining critical tokens, MadaKV achieves substantial reductions in KV cache memory footprint and model inference decoding latency (1.3 to 1.5 times improvement) while maintaining high accuracy across various multimodal long-context tasks. Extensive experiments on representative MLLMs and the MileBench benchmark demonstrate the effectiveness of MadaKV compared to existing KV cache eviction methods. 

---
# UniMate: A Unified Model for Mechanical Metamaterial Generation, Property Prediction, and Condition Confirmation 

**Authors**: Wangzhi Zhan, Jianpeng Chen, Dongqi Fu, Dawei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.15722)  

**Abstract**: Metamaterials are artificial materials that are designed to meet unseen properties in nature, such as ultra-stiffness and negative materials indices. In mechanical metamaterial design, three key modalities are typically involved, i.e., 3D topology, density condition, and mechanical property. Real-world complex application scenarios place the demanding requirements on machine learning models to consider all three modalities together. However, a comprehensive literature review indicates that most existing works only consider two modalities, e.g., predicting mechanical properties given the 3D topology or generating 3D topology given the required properties. Therefore, there is still a significant gap for the state-of-the-art machine learning models capturing the whole. Hence, we propose a unified model named UNIMATE, which consists of a modality alignment module and a synergetic diffusion generation module. Experiments indicate that UNIMATE outperforms the other baseline models in topology generation task, property prediction task, and condition confirmation task by up to 80.2%, 5.1%, and 50.2%, respectively. We opensource our proposed UNIMATE model and corresponding results at this https URL. 

---
# daDPO: Distribution-Aware DPO for Distilling Conversational Abilities 

**Authors**: Zhengze Zhang, Shiqi Wang, Yiqun Shen, Simin Guo, Dahua Lin, Xiaoliang Wang, Nguyen Cam-Tu, Fei Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.15717)  

**Abstract**: Large language models (LLMs) have demonstrated exceptional performance across various applications, but their conversational abilities decline sharply as model size decreases, presenting a barrier to their deployment in resource-constrained environments. Knowledge distillation with Direct Preference Optimization (dDPO) has emerged as a promising approach to enhancing the conversational abilities of smaller models using a larger teacher model. However, current methods primarily focus on 'black-box' KD, which only uses the teacher's responses, overlooking the output distribution offered by the teacher. This paper addresses this gap by introducing daDPO (Distribution-Aware DPO), a unified method for preference optimization and distribution-based distillation. We provide rigorous theoretical analysis and empirical validation, showing that daDPO outperforms existing methods in restoring performance for pruned models and enhancing smaller LLM models. Notably, in in-domain evaluation, our method enables a 20% pruned Vicuna1.5-7B to achieve near-teacher performance (-7.3% preference rate compared to that of dDPO's -31%), and allows Qwen2.5-1.5B to occasionally outperform its 7B teacher model (14.0% win rate). 

---
# Alternates, Assemble! Selecting Optimal Alternates for Citizens' Assemblies 

**Authors**: Angelos Assos, Carmel Baharav, Bailey Flanigan, Ariel Procaccia  

**Link**: [PDF](https://arxiv.org/pdf/2506.15716)  

**Abstract**: An increasingly influential form of deliberative democracy centers on citizens' assemblies, where randomly selected people discuss policy questions. The legitimacy of these panels hinges on their representation of the broader population, but panelists often drop out, leading to an unbalanced composition. Although participant attrition is mitigated in practice by alternates, their selection is not taken into account by existing methods. To address this gap, we introduce an optimization framework for alternate selection. Our algorithmic approach, which leverages learning-theoretic machinery, estimates dropout probabilities using historical data and selects alternates to minimize expected misrepresentation. We establish theoretical guarantees for our approach, including worst-case bounds on sample complexity (with implications for computational efficiency) and on loss when panelists' probabilities of dropping out are mis-estimated. Empirical evaluation using real-world data demonstrates that, compared to the status quo, our method significantly improves representation while requiring fewer alternates. 

---
# NeuronSeek: On Stability and Expressivity of Task-driven Neurons 

**Authors**: Hanyu Pei, Jing-Xiao Liao, Qibin Zhao, Ting Gao, Shijun Zhang, Xiaoge Zhang, Feng-Lei Fan  

**Link**: [PDF](https://arxiv.org/pdf/2506.15715)  

**Abstract**: Drawing inspiration from our human brain that designs different neurons for different tasks, recent advances in deep learning have explored modifying a network's neurons to develop so-called task-driven neurons. Prototyping task-driven neurons (referred to as NeuronSeek) employs symbolic regression (SR) to discover the optimal neuron formulation and construct a network from these optimized neurons. Along this direction, this work replaces symbolic regression with tensor decomposition (TD) to discover optimal neuronal formulations, offering enhanced stability and faster convergence. Furthermore, we establish theoretical guarantees that modifying the aggregation functions with common activation functions can empower a network with a fixed number of parameters to approximate any continuous function with an arbitrarily small error, providing a rigorous mathematical foundation for the NeuronSeek framework. Extensive empirical evaluations demonstrate that our NeuronSeek-TD framework not only achieves superior stability, but also is competitive relative to the state-of-the-art models across diverse benchmarks. The code is available at this https URL. 

---
# BatteryBERT for Realistic Battery Fault Detection Using Point-Masked Signal Modeling 

**Authors**: Songqi Zhou, Ruixue Liu, Yixing Wang, Jia Lu, Benben Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15712)  

**Abstract**: Accurate fault detection in lithium-ion batteries is essential for the safe and reliable operation of electric vehicles and energy storage systems. However, existing methods often struggle to capture complex temporal dependencies and cannot fully leverage abundant unlabeled data. Although large language models (LLMs) exhibit strong representation capabilities, their architectures are not directly suited to the numerical time-series data common in industrial settings. To address these challenges, we propose a novel framework that adapts BERT-style pretraining for battery fault detection by extending the standard BERT architecture with a customized time-series-to-token representation module and a point-level Masked Signal Modeling (point-MSM) pretraining task tailored to battery applications. This approach enables self-supervised learning on sequential current, voltage, and other charge-discharge cycle data, yielding distributionally robust, context-aware temporal embeddings. We then concatenate these embeddings with battery metadata and feed them into a downstream classifier for accurate fault classification. Experimental results on a large-scale real-world dataset show that models initialized with our pretrained parameters significantly improve both representation quality and classification accuracy, achieving an AUROC of 0.945 and substantially outperforming existing approaches. These findings validate the effectiveness of BERT-style pretraining for time-series fault detection. 

---
# Shadow defense against gradient inversion attack in federated learning 

**Authors**: Le Jiang, Liyan Ma, Guang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15711)  

**Abstract**: Federated learning (FL) has emerged as a transformative framework for privacy-preserving distributed training, allowing clients to collaboratively train a global model without sharing their local data. This is especially crucial in sensitive fields like healthcare, where protecting patient data is paramount. However, privacy leakage remains a critical challenge, as the communication of model updates can be exploited by potential adversaries. Gradient inversion attacks (GIAs), for instance, allow adversaries to approximate the gradients used for training and reconstruct training images, thus stealing patient privacy. Existing defense mechanisms obscure gradients, yet lack a nuanced understanding of which gradients or types of image information are most vulnerable to such attacks. These indiscriminate calibrated perturbations result in either excessive privacy protection degrading model accuracy, or insufficient one failing to safeguard sensitive information. Therefore, we introduce a framework that addresses these challenges by leveraging a shadow model with interpretability for identifying sensitive areas. This enables a more targeted and sample-specific noise injection. Specially, our defensive strategy achieves discrepancies of 3.73 in PSNR and 0.2 in SSIM compared to the circumstance without defense on the ChestXRay dataset, and 2.78 in PSNR and 0.166 in the EyePACS dataset. Moreover, it minimizes adverse effects on model performance, with less than 1\% F1 reduction compared to SOTA methods. Our extensive experiments, conducted across diverse types of medical images, validate the generalization of the proposed framework. The stable defense improvements for FedAvg are consistently over 1.5\% times in LPIPS and SSIM. It also offers a universal defense against various GIA types, especially for these sensitive areas in images. 

---
# RAST: Reasoning Activation in LLMs via Small-model Transfer 

**Authors**: Siru Ouyang, Xinyu Zhu, Zilin Xiao, Minhao Jiang, Yu Meng, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.15710)  

**Abstract**: Reinforcement learning (RL) has become a powerful approach for improving the reasoning capabilities of large language models (LLMs), as evidenced by recent successes such as OpenAI's o1 and Deepseek-R1. However, applying RL at scale remains intimidatingly resource-intensive, requiring multiple model copies and extensive GPU workloads. On the other hand, while being powerful, recent studies suggest that RL does not fundamentally endow models with new knowledge; rather, it primarily reshapes the model's output distribution to activate reasoning capabilities latent in the base model. Building on this insight, we hypothesize that the changes in output probabilities induced by RL are largely model-size invariant, opening the door to a more efficient paradigm: training a small model with RL and transferring its induced probability shifts to larger base models. To verify our hypothesis, we conduct a token-level analysis of decoding trajectories and find high alignment in RL-induced output distributions across model scales, validating our hypothesis. Motivated by this, we propose RAST, a simple yet effective method that transfers reasoning behaviors by injecting RL-induced probability adjustments from a small RL-trained model into larger models. Experiments across multiple mathematical reasoning benchmarks show that RAST substantially and consistently enhances the reasoning capabilities of base models while requiring significantly lower GPU memory than direct RL training, sometimes even yielding better performance than the RL-trained counterparts. Our findings offer new insights into the nature of RL-driven reasoning and practical strategies for scaling its benefits without incurring its full computational cost. The project page of RAST is available at this https URL. 

---
# Studying and Improving Graph Neural Network-based Motif Estimation 

**Authors**: Pedro C. Vieira, Miguel E. P. Silva, Pedro Manuel Pinto Ribeiro  

**Link**: [PDF](https://arxiv.org/pdf/2506.15709)  

**Abstract**: Graph Neural Networks (GNNs) are a predominant method for graph representation learning. However, beyond subgraph frequency estimation, their application to network motif significance-profile (SP) prediction remains under-explored, with no established benchmarks in the literature. We propose to address this problem, framing SP estimation as a task independent of subgraph frequency estimation. Our approach shifts from frequency counting to direct SP estimation and modulates the problem as multitarget regression. The reformulation is optimised for interpretability, stability and scalability on large graphs. We validate our method using a large synthetic dataset and further test it on real-world graphs. Our experiments reveal that 1-WL limited models struggle to make precise estimations of SPs. However, they can generalise to approximate the graph generation processes of networks by comparing their predicted SP with the ones originating from synthetic generators. This first study on GNN-based motif estimation also hints at how using direct SP estimation can help go past the theoretical limitations that motif estimation faces when performed through subgraph counting. 

---
# Refined Causal Graph Structure Learning via Curvature for Brain Disease Classification 

**Authors**: Falih Gozi Febrinanto, Adonia Simango, Chengpei Xu, Jingjing Zhou, Jiangang Ma, Sonika Tyagi, Feng Xia  

**Link**: [PDF](https://arxiv.org/pdf/2506.15708)  

**Abstract**: Graph neural networks (GNNs) have been developed to model the relationship between regions of interest (ROIs) in brains and have shown significant improvement in detecting brain diseases. However, most of these frameworks do not consider the intrinsic relationship of causality factor between brain ROIs, which is arguably more essential to observe cause and effect interaction between signals rather than typical correlation values. We propose a novel framework called CGB (Causal Graphs for Brains) for brain disease classification/detection, which models refined brain networks based on the causal discovery method, transfer entropy, and geometric curvature strategy. CGB unveils causal relationships between ROIs that bring vital information to enhance brain disease classification performance. Furthermore, CGB also performs a graph rewiring through a geometric curvature strategy to refine the generated causal graph to become more expressive and reduce potential information bottlenecks when GNNs model it. Our extensive experiments show that CGB outperforms state-of-the-art methods in classification tasks on brain disease datasets, as measured by average F1 scores. 

---
# Every Rollout Counts: Optimal Resource Allocation for Efficient Test-Time Scaling 

**Authors**: Xinglin Wang, Yiwei Li, Shaoxiong Feng, Peiwen Yuan, Yueqi Zhang, Jiayi Shi, Chuyi Tan, Boyuan Pan, Yao Hu, Kan Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.15707)  

**Abstract**: Test-Time Scaling (TTS) improves the performance of Large Language Models (LLMs) by using additional inference-time computation to explore multiple reasoning paths through search. Yet how to allocate a fixed rollout budget most effectively during search remains underexplored, often resulting in inefficient use of compute at test time. To bridge this gap, we formulate test-time search as a resource allocation problem and derive the optimal allocation strategy that maximizes the probability of obtaining a correct solution under a fixed rollout budget. Within this formulation, we reveal a core limitation of existing search methods: solution-level allocation tends to favor reasoning directions with more candidates, leading to theoretically suboptimal and inefficient use of compute. To address this, we propose Direction-Oriented Resource Allocation (DORA), a provably optimal method that mitigates this bias by decoupling direction quality from candidate count and allocating resources at the direction level. To demonstrate DORA's effectiveness, we conduct extensive experiments on challenging mathematical reasoning benchmarks including MATH500, AIME2024, and AIME2025. The empirical results show that DORA consistently outperforms strong baselines with comparable computational cost, achieving state-of-the-art accuracy. We hope our findings contribute to a broader understanding of optimal TTS for LLMs. 

---
# MDPO: Multi-Granularity Direct Preference Optimization for Mathematical Reasoning 

**Authors**: Yunze Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.15706)  

**Abstract**: Mathematical reasoning presents a significant challenge for Large Language Models (LLMs) as it requires ensuring the correctness of each reasoning step. Researchers have been strengthening the mathematical reasoning abilities of LLMs through supervised fine-tuning, but due to the inability to suppress incorrect outputs, illusions can easily arise. Recently, Direct Preference Optimization (DPO) has been widely adopted for aligning human intent by using preference data to prevent LLMs from generating incorrect outputs. However, it has shown limited benefits in long-chain mathematical reasoning, mainly because DPO struggles to effectively capture the differences between accepted and rejected answers from preferences in long-chain data. The inconsistency between DPO training and LLMs' generation metrics also affects the effectiveness of suppressing incorrect outputs. We propose the Multi-Granularity Direct Preference Optimization (MDPO) method, optimizing the mathematical reasoning of LLMs at three granularities: Solution2Solution, Inference2Inference, and Step2Step. Solution2Solution focuses on the correctness of entire long-chain reasoning; Inference2Inference concentrates on logical reasoning between steps; Step2Step corrects computational errors in steps, enhancing the computational capabilities of LLMs. Additionally, we unify the training objectives of the three granularities to align with the generation metrics. We conducted experiments on the open-source models Qwen2 and Llama3, achieving improvements of 1.7% and 0.9% on the GSM8K dataset, and 2.3% and 1.2% on the MATH dataset, outperforming DPO and other DPO variant methods. Furthermore, we also provide a pipeline for constructing MDPO training data that is simple and does not require manual annotation costs. 

---
# Generalisation Bounds of Zero-Shot Economic Forecasting using Time Series Foundation Models 

**Authors**: Jittarin Jetwiriyanon, Teo Susnjak, Surangika Ranathunga  

**Link**: [PDF](https://arxiv.org/pdf/2506.15705)  

**Abstract**: This study investigates zero-shot forecasting capabilities of Time Series Foundation Models (TSFMs) for macroeconomic indicators. We apply TSFMs to forecasting economic indicators under univariate conditions, bypassing the need for train bespoke econometric models using and extensive training datasets. Our experiments were conducted on a case study dataset, without additional customisation. We rigorously back-tested three state-of-the-art TSFMs (Chronos, TimeGPT and Moirai) under data-scarce conditions and structural breaks. Our results demonstrate that appropriately engineered TSFMs can internalise rich economic dynamics, accommodate regime shifts, and deliver well-behaved uncertainty estimates out of the box, while matching state-of-the-art multivariate models on this domain. Our findings suggest that, without any fine-tuning, TSFMs can match or exceed classical models during stable economic conditions. However, they are vulnerable to degradation in performances during periods of rapid shocks. The findings offer guidance to practitioners on when zero-shot deployments are viable for macroeconomic monitoring and strategic planning. 

---
# Learn from the Past: Fast Sparse Indexing for Large Language Model Decoding 

**Authors**: Feiyu Yao, Qian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15704)  

**Abstract**: As large language models (LLMs) continue to support increasingly longer contexts, the memory demand for key-value (KV) caches during decoding grows rapidly, becoming a critical bottleneck in both GPU memory capacity and PCIe bandwidth. Sparse attention mechanisms alleviate this issue by computing attention weights only for selected key-value pairs. However, their indexing computation typically requires traversing all key vectors, resulting in significant computational and data transfer overhead. To reduce the cost of index retrieval, existing methods often treat each decoding step as an independent process, failing to exploit the temporal correlations embedded in historical decoding information. To this end, we propose LFPS(Learn From the Past for Sparse Indexing), an acceleration method that dynamically constructs sparse indexing candidates based on historical attention patterns. LFPS captures two prevalent trends in decoder attention -vertical patterns (attending to fixed positions) and slash patterns (attending to relative positions) -and incorporates a positional expansion strategy to effectively predict the Top-k indices for the current step. We validate LFPS on challenging long-context benchmarks such as LongBench-RULER, using Llama-3.1-8B-Instruct as the base model. Experimental results show that LFPS achieves up to 22.8$\times$ speedup over full attention and 9.6$\times$ speedup over exact Top-k retrieval on an RTX 4090 GPU and a single CPU core of a Xeon Gold 6430, respectively, while preserving generation accuracy. These results demonstrate that LFPS offers a practical and efficient solution for decoding optimization in long-context LLM inference. 

---
# Federated Incomplete Multi-view Clustering with Globally Fused Graph Guidance 

**Authors**: Guoqing Chao, Zhenghao Zhang, Lei Meng, Jie Wen, Dianhui Chu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15703)  

**Abstract**: Federated multi-view clustering has been proposed to mine the valuable information within multi-view data distributed across different devices and has achieved impressive results while preserving the privacy. Despite great progress, most federated multi-view clustering methods only used global pseudo-labels to guide the downstream clustering process and failed to exploit the global information when extracting features. In addition, missing data problem in federated multi-view clustering task is less explored. To address these problems, we propose a novel Federated Incomplete Multi-view Clustering method with globally Fused Graph guidance (FIMCFG). Specifically, we designed a dual-head graph convolutional encoder at each client to extract two kinds of underlying features containing global and view-specific information. Subsequently, under the guidance of the fused graph, the two underlying features are fused into high-level features, based on which clustering is conducted under the supervision of pseudo-labeling. Finally, the high-level features are uploaded to the server to refine the graph fusion and pseudo-labeling computation. Extensive experimental results demonstrate the effectiveness and superiority of FIMCFG. Our code is publicly available at this https URL. 

---
# Minifinetuning: Low-Data Generation Domain Adaptation through Corrective Self-Distillation 

**Authors**: Peter Belcak, Greg Heinrich, Jan Kautz, Pavlo Molchanov  

**Link**: [PDF](https://arxiv.org/pdf/2506.15702)  

**Abstract**: Finetuning language models for a new domain inevitably leads to the deterioration of their general performance. This becomes more pronounced the more limited the finetuning data resource.
We introduce minifinetuning (MFT), a method for language model domain adaptation that considerably reduces the effects of overfitting-induced degeneralization in low-data settings and which does so in the absence of any pre-training data for replay. MFT demonstrates 2-10x more favourable specialization-to-degeneralization ratios than standard finetuning across a wide range of models and domains and exhibits an intrinsic robustness to overfitting when data in the new domain is scarce and down to as little as 500 samples.
Employing corrective self-distillation that is individualized on the sample level, MFT outperforms parameter-efficient finetuning methods, demonstrates replay-like degeneralization mitigation properties, and is composable with either for a combined effect. 

---
# Compiler-R1: Towards Agentic Compiler Auto-tuning with Reinforcement Learning 

**Authors**: Haolin Pan, Hongyu Lin, Haoran Luo, Yang Liu, Kaichun Yao, Libo Zhang, Mingjie Xing, Yanjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15701)  

**Abstract**: Compiler auto-tuning optimizes pass sequences to improve performance metrics such as Intermediate Representation (IR) instruction count. Although recent advances leveraging Large Language Models (LLMs) have shown promise in automating compiler tuning, two significant challenges still remain: the absence of high-quality reasoning datasets for agents training, and limited effective interactions with the compilation environment. In this work, we introduce Compiler-R1, the first reinforcement learning (RL)-driven framework specifically augmenting LLM capabilities for compiler auto-tuning. Compiler-R1 features a curated, high-quality reasoning dataset and a novel two-stage end-to-end RL training pipeline, enabling efficient environment exploration and learning through an outcome-based reward. Extensive experiments across seven datasets demonstrate Compiler-R1 achieving an average 8.46% IR instruction count reduction compared to opt -Oz, showcasing the strong potential of RL-trained LLMs for compiler optimization. Our code and datasets are publicly available at this https URL. 

---
# Contraction Actor-Critic: Contraction Metric-Guided Reinforcement Learning for Robust Path Tracking 

**Authors**: Minjae Cho, Hiroyasu Tsukamoto, Huy Trong Tran  

**Link**: [PDF](https://arxiv.org/pdf/2506.15700)  

**Abstract**: Control contraction metrics (CCMs) provide a framework to co-synthesize a controller and a corresponding contraction metric -- a positive-definite Riemannian metric under which a closed-loop system is guaranteed to be incrementally exponentially stable. However, the synthesized controller only ensures that all the trajectories of the system converge to one single trajectory and, as such, does not impose any notion of optimality across an entire trajectory. Furthermore, constructing CCMs requires a known dynamics model and non-trivial effort in solving an infinite-dimensional convex feasibility problem, which limits its scalability to complex systems featuring high dimensionality with uncertainty. To address these issues, we propose to integrate CCMs into reinforcement learning (RL), where CCMs provide dynamics-informed feedback for learning control policies that minimize cumulative tracking error under unknown dynamics. We show that our algorithm, called contraction actor-critic (CAC), formally enhances the capability of CCMs to provide a set of contracting policies with the long-term optimality of RL in a fully automated setting. Given a pre-trained dynamics model, CAC simultaneously learns a contraction metric generator (CMG) -- which generates a contraction metric -- and uses an actor-critic algorithm to learn an optimal tracking policy guided by that metric. We demonstrate the effectiveness of our algorithm relative to established baselines through extensive empirical studies, including simulated and real-world robot experiments, and provide a theoretical rationale for incorporating contraction theory into RL. 

---
# BLUR: A Benchmark for LLM Unlearning Robust to Forget-Retain Overlap 

**Authors**: Shengyuan Hu, Neil Kale, Pratiksha Thaker, Yiwei Fu, Steven Wu, Virginia Smith  

**Link**: [PDF](https://arxiv.org/pdf/2506.15699)  

**Abstract**: Machine unlearning has the potential to improve the safety of large language models (LLMs) by removing sensitive or harmful information post hoc. A key challenge in unlearning involves balancing between forget quality (effectively unlearning undesirable information) and retain quality (maintaining good performance on other, general tasks). Unfortunately, as we show, current LLM unlearning benchmarks contain highly disparate forget and retain sets -- painting a false picture of the effectiveness of LLM unlearning methods. This can be particularly problematic because it opens the door for benign perturbations, such as relearning attacks, to easily reveal supposedly unlearned knowledge once models are deployed. To address this, we present $\texttt{BLUR}$: a benchmark for LLM unlearning that provides more realistic scenarios of forget-retain overlap. $\texttt{BLUR}$ significantly expands on existing unlearning benchmarks by providing extended evaluation tasks, combined forget/retain queries, and relearning datasets of varying degrees of difficulty. Despite the benign nature of the queries considered, we find that the performance of existing methods drops significantly when evaluated on $\texttt{BLUR}$, with simple approaches performing better on average than more recent methods. These results highlight the importance of robust evaluation and suggest several important directions of future study. Our benchmark is publicly available at: this https URL 

---
# What Do Latent Action Models Actually Learn? 

**Authors**: Chuheng Zhang, Tim Pearce, Pushi Zhang, Kaixin Wang, Xiaoyu Chen, Wei Shen, Li Zhao, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2506.15691)  

**Abstract**: Latent action models (LAMs) aim to learn action-relevant changes from unlabeled videos by compressing changes between frames as latents. However, differences between video frames can be caused by controllable changes as well as exogenous noise, leading to an important concern -- do latents capture the changes caused by actions or irrelevant noise? This paper studies this issue analytically, presenting a linear model that encapsulates the essence of LAM learning, while being this http URL provides several insights, including connections between LAM and principal component analysis (PCA), desiderata of the data-generating policy, and justification of strategies to encourage learning controllable changes using data augmentation, data cleaning, and auxiliary action-prediction. We also provide illustrative results based on numerical simulation, shedding light on the specific structure of observations, actions, and noise in data that influence LAM learning. 

---
# LLM Web Dynamics: Tracing Model Collapse in a Network of LLMs 

**Authors**: Tianyu Wang, Lingyou Pang, Akira Horiguchi, Carey E. Priebe  

**Link**: [PDF](https://arxiv.org/pdf/2506.15690)  

**Abstract**: The increasing use of synthetic data from the public Internet has enhanced data usage efficiency in large language model (LLM) training. However, the potential threat of model collapse remains insufficiently explored. Existing studies primarily examine model collapse in a single model setting or rely solely on statistical surrogates. In this work, we introduce LLM Web Dynamics (LWD), an efficient framework for investigating model collapse at the network level. By simulating the Internet with a retrieval-augmented generation (RAG) database, we analyze the convergence pattern of model outputs. Furthermore, we provide theoretical guarantees for this convergence by drawing an analogy to interacting Gaussian Mixture Models. 

---
# BASE-Q: Bias and Asymmetric Scaling Enhanced Rotational Quantization for Large Language Models 

**Authors**: Liulu He, Shenli Zhen, Karwei Sun, Yijiang Liu, Yufei Zhao, Chongkang Tan, Huanrui Yang, Yuan Du, Li Du  

**Link**: [PDF](https://arxiv.org/pdf/2506.15689)  

**Abstract**: Rotations have become essential to state-of-the-art quantization pipelines for large language models (LLMs) by effectively smoothing outliers in weights and activations. However, further optimizing the rotation parameters offers only limited performance gains and introduces significant training overhead: due to rotation parameter sharing, full-model must be loaded simultaneously to enable backpropagation, resulting in substantial memory consumption and limited practical utility. In this work, we identify two fundamental limitations of current rotational quantization methods: (i) rotation fails to align channel means, resulting in wider quantization bounds and increased rounding errors; and (ii) rotation makes the activation distribution more Gaussian-like, increasing energy loss caused by clipping errors. To address these issues, we introduce \textbf{BASE-Q}, a simple yet powerful approach that combines bias correction and asymmetric scaling to effectively reduce rounding and clipping errors. Furthermore, BASE-Q enables blockwise optimization, eliminating the need for memory-intensive full-model backpropagation. Extensive experiments on various LLMs and benchmarks demonstrate the effectiveness of BASE-Q, narrowing the accuracy gap to full-precision models by 50.5\%, 42.9\%, and 29.2\% compared to QuaRot, SpinQuant, and OSTQuant, respectively. The code will be released soon. 

---
# Cellular Traffic Prediction via Deep State Space Models with Attention Mechanism 

**Authors**: Hui Ma, Kai Yang, Man-On Pun  

**Link**: [PDF](https://arxiv.org/pdf/2506.15688)  

**Abstract**: Cellular traffic prediction is of great importance for operators to manage network resources and make decisions. Traffic is highly dynamic and influenced by many exogenous factors, which would lead to the degradation of traffic prediction accuracy. This paper proposes an end-to-end framework with two variants to explicitly characterize the spatiotemporal patterns of cellular traffic among neighboring cells. It uses convolutional neural networks with an attention mechanism to capture the spatial dynamics and Kalman filter for temporal modelling. Besides, we can fully exploit the auxiliary information such as social activities to improve prediction performance. We conduct extensive experiments on three real-world datasets. The results show that our proposed models outperform the state-of-the-art machine learning techniques in terms of prediction accuracy. 

---
# Learning from M-Tuple Dominant Positive and Unlabeled Data 

**Authors**: Jiahe Qin, Junpeng Li, Changchun Hua, Yana Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15686)  

**Abstract**: Label Proportion Learning (LLP) addresses the classification problem where multiple instances are grouped into bags and each bag contains information about the proportion of each class. However, in practical applications, obtaining precise supervisory information regarding the proportion of instances in a specific class is challenging. To better align with real-world application scenarios and effectively leverage the proportional constraints of instances within tuples, this paper proposes a generalized learning framework \emph{MDPU}. Specifically, we first mathematically model the distribution of instances within tuples of arbitrary size, under the constraint that the number of positive instances is no less than that of negative instances. Then we derive an unbiased risk estimator that satisfies risk consistency based on the empirical risk minimization (ERM) method. To mitigate the inevitable overfitting issue during training, a risk correction method is introduced, leading to the development of a corrected risk estimator. The generalization error bounds of the unbiased risk estimator theoretically demonstrate the consistency of the proposed method. Extensive experiments on multiple datasets and comparisons with other relevant baseline methods comprehensively validate the effectiveness of the proposed learning framework. 

---
# Ignition Phase : Standard Training for Fast Adversarial Robustness 

**Authors**: Wang Yu-Hang, Liu ying, Fang liang, Wang Xuelin, Junkang Guo, Shiwei Li, Lei Gao, Jian Liu, Wenfei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2506.15685)  

**Abstract**: Adversarial Training (AT) is a cornerstone defense, but many variants overlook foundational feature representations by primarily focusing on stronger attack generation. We introduce Adversarial Evolution Training (AET), a simple yet powerful framework that strategically prepends an Empirical Risk Minimization (ERM) phase to conventional AT. We hypothesize this initial ERM phase cultivates a favorable feature manifold, enabling more efficient and effective robustness acquisition. Empirically, AET achieves comparable or superior robustness more rapidly, improves clean accuracy, and cuts training costs by 8-25\%. Its effectiveness is shown across multiple datasets, architectures, and when augmenting established AT methods. Our findings underscore the impact of feature pre-conditioning via standard training for developing more efficient, principled robust defenses. Code is available in the supplementary material. 

---
# cAST: Enhancing Code Retrieval-Augmented Generation with Structural Chunking via Abstract Syntax Tree 

**Authors**: Yilin Zhang, Xinran Zhao, Zora Zhiruo Wang, Chenyang Yang, Jiayi Wei, Tongshuang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15655)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become essential for large-scale code generation, grounding predictions in external code corpora to improve actuality. However, a critical yet underexplored aspect of RAG pipelines is chunking -- the process of dividing documents into retrievable units. Existing line-based chunking heuristics often break semantic structures, splitting functions or merging unrelated code, which can degrade generation quality. We propose chunking via Abstract Syntax Trees (\ourwork), a structure-aware method that recursively breaks large AST nodes into smaller chunks and merges sibling nodes while respecting size limits. This approach generates self-contained, semantically coherent units across programming languages and tasks, improving performance on diverse code generation tasks, e.g., boosting Recall@5 by 4.3 points on RepoEval retrieval and Pass@1 by 2.67 points on SWE-bench generation. Our work highlights the importance of structure-aware chunking for scaling retrieval-enhanced code intelligence. 

---
