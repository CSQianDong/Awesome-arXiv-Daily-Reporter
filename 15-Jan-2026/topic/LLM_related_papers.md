# Omni-R1: Towards the Unified Generative Paradigm for Multimodal Reasoning 

**Authors**: Dongjie Cheng, Yongqi Li, Zhixin Ma, Hongru Cai, Yupeng Hu, Wenjie Wang, Liqiang Nie, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.09536)  

**Abstract**: Multimodal Large Language Models (MLLMs) are making significant progress in multimodal reasoning. Early approaches focus on pure text-based reasoning. More recent studies have incorporated multimodal information into the reasoning steps; however, they often follow a single task-specific reasoning pattern, which limits their generalizability across various multimodal tasks. In fact, there are numerous multimodal tasks requiring diverse reasoning skills, such as zooming in on a specific region or marking an object within an image. To address this, we propose unified generative multimodal reasoning, which unifies diverse multimodal reasoning skills by generating intermediate images during the reasoning process. We instantiate this paradigm with Omni-R1, a two-stage SFT+RL framework featuring perception alignment loss and perception reward, thereby enabling functional image generation. Additionally, we introduce Omni-R1-Zero, which eliminates the need for multimodal annotations by bootstrapping step-wise visualizations from text-only reasoning data. Empirical results show that Omni-R1 achieves unified generative reasoning across a wide range of multimodal tasks, and Omni-R1-Zero can match or even surpass Omni-R1 on average, suggesting a promising direction for generative multimodal reasoning. 

---
# Cluster Workload Allocation: Semantic Soft Affinity Using Natural Language Processing 

**Authors**: Leszek Sliwko, Jolanta Mizeria-Pietraszko  

**Link**: [PDF](https://arxiv.org/pdf/2601.09282)  

**Abstract**: Cluster workload allocation often requires complex configurations, creating a usability gap. This paper introduces a semantic, intent-driven scheduling paradigm for cluster systems using Natural Language Processing. The system employs a Large Language Model (LLM) integrated via a Kubernetes scheduler extender to interpret natural language allocation hint annotations for soft affinity preferences. A prototype featuring a cluster state cache and an intent analyzer (using AWS Bedrock) was developed. Empirical evaluation demonstrated high LLM parsing accuracy (>95% Subset Accuracy on an evaluation ground-truth dataset) for top-tier models like Amazon Nova Pro/Premier and Mistral Pixtral Large, significantly outperforming a baseline engine. Scheduling quality tests across six scenarios showed the prototype achieved superior or equivalent placement compared to standard Kubernetes configurations, particularly excelling in complex and quantitative scenarios and handling conflicting soft preferences. The results validate using LLMs for accessible scheduling but highlight limitations like synchronous LLM latency, suggesting asynchronous processing for production readiness. This work confirms the viability of semantic soft affinity for simplifying workload orchestration. 

---
# LLM for Large-Scale Optimization Model Auto-Formulation: A Lightweight Few-Shot Learning Approach 

**Authors**: Kuo Liang, Yuhang Lu, Jianming Mao, Shuyi Sun, Chunwei Yang, Congcong Zeng, Xiao Jin, Hanzhang Qin, Ruihao Zhu, Chung-Piaw Teo  

**Link**: [PDF](https://arxiv.org/pdf/2601.09635)  

**Abstract**: Large-scale optimization is a key backbone of modern business decision-making. However, building these models is often labor-intensive and time-consuming. We address this by proposing LEAN-LLM-OPT, a LightwEight AgeNtic workflow construction framework for LLM-assisted large-scale OPTimization auto-formulation. LEAN-LLM-OPT takes as input a problem description together with associated datasets and orchestrates a team of LLM agents to produce an optimization formulation. Specifically, upon receiving a query, two upstream LLM agents dynamically construct a workflow that specifies, step-by-step, how optimization models for similar problems can be formulated. A downstream LLM agent then follows this workflow to generate the final output. Leveraging LLMs' text-processing capabilities and common modeling practices, the workflow decomposes the modeling task into a sequence of structured sub-tasks and offloads mechanical data-handling operations to auxiliary tools. This design alleviates the downstream agent's burden related to planning and data handling, allowing it to focus on the most challenging components that cannot be readily standardized. Extensive simulations show that LEAN-LLM-OPT, instantiated with GPT-4.1 and the open source gpt-oss-20B, achieves strong performance on large-scale optimization modeling tasks and is competitive with state-of-the-art approaches. In addition, in a Singapore Airlines choice-based revenue management use case, LEAN-LLM-OPT demonstrates practical value by achieving leading performance across a range of scenarios. Along the way, we introduce Large-Scale-OR and Air-NRM, the first comprehensive benchmarks for large-scale optimization auto-formulation. The code and data of this work is available at this https URL. 

---
# What Do LLM Agents Know About Their World? Task2Quiz: A Paradigm for Studying Environment Understanding 

**Authors**: Siyuan Liu, Hongbang Yuan, Xinze Li, Ziyue Zhu, Yixin Cao, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2601.09503)  

**Abstract**: Large language model (LLM) agents have demonstrated remarkable capabilities in complex decision-making and tool-use tasks, yet their ability to generalize across varying environments remains a under-examined concern. Current evaluation paradigms predominantly rely on trajectory-based metrics that measure task success, while failing to assess whether agents possess a grounded, transferable model of the environment. To address this gap, we propose Task-to-Quiz (T2Q), a deterministic and automated evaluation paradigm designed to decouple task execution from world-state understanding. We instantiate this paradigm in T2QBench, a suite comprising 30 environments and 1,967 grounded QA pairs across multiple difficulty levels. Our extensive experiments reveal that task success is often a poor proxy for environment understanding, and that current memory machanism can not effectively help agents acquire a grounded model of the environment. These findings identify proactive exploration and fine-grained state representation as primary bottlenecks, offering a robust foundation for developing more generalizable autonomous agents. 

---
# RISER: Orchestrating Latent Reasoning Skills for Adaptive Activation Steering 

**Authors**: Wencheng Ye, Liang Peng, Xiaoyang Yuan, Yi Bin, Pengpeng Zeng, Hengyu Jin, Heng Tao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2601.09269)  

**Abstract**: Recent work on domain-specific reasoning with large language models (LLMs) often relies on training-intensive approaches that require parameter updates. While activation steering has emerged as a parameter efficient alternative, existing methods apply static, manual interventions that fail to adapt to the dynamic nature of complex reasoning. To address this limitation, we propose RISER (Router-based Intervention for Steerable Enhancement of Reasoning), a plug-and-play intervention framework that adaptively steers LLM reasoning in activation space. RISER constructs a library of reusable reasoning vectors and employs a lightweight Router to dynamically compose them for each input. The Router is optimized via reinforcement learning under task-level rewards, activating latent cognitive primitives in an emergent and compositional manner. Across seven diverse benchmarks, RISER yields 3.4-6.5% average zero-shot accuracy improvements over the base model while surpassing CoT-style reasoning with 2-3x higher token efficiency and robust accuracy gains. Further analysis shows that RISER autonomously combines multiple vectors into interpretable, precise control strategies, pointing toward more controllable and efficient LLM reasoning. 

---
# Efficient Paths and Dense Rewards: Probabilistic Flow Reasoning for Large Language Models 

**Authors**: Yan Liu, Feng Zhang, Zhanyu Ma, Jun Xu, Jiuchong Gao, Jinghua Hao, Renqing He, Han Liu, Yangdong Deng  

**Link**: [PDF](https://arxiv.org/pdf/2601.09260)  

**Abstract**: High-quality chain-of-thought has demonstrated strong potential for unlocking the reasoning capabilities of large language models. However, current paradigms typically treat the reasoning process as an indivisible sequence, lacking an intrinsic mechanism to quantify step-wise information gain. This granularity gap manifests in two limitations: inference inefficiency from redundant exploration without explicit guidance, and optimization difficulty due to sparse outcome supervision or costly external verifiers. In this work, we propose CoT-Flow, a framework that reconceptualizes discrete reasoning steps as a continuous probabilistic flow, quantifying the contribution of each step toward the ground-truth answer. Built on this formulation, CoT-Flow enables two complementary methodologies: flow-guided decoding, which employs a greedy flow-based decoding strategy to extract information-efficient reasoning paths, and flow-based reinforcement learning, which constructs a verifier-free dense reward function. Experiments on challenging benchmarks demonstrate that CoT-Flow achieves a superior balance between inference efficiency and reasoning performance. 

---
# MAXS: Meta-Adaptive Exploration with LLM Agents 

**Authors**: Jian Zhang, Zhiyuan Wang, Zhangqi Wang, Yu He, Haoran Luo, li yuan, Lingling Zhang, Rui Mao, Qika Lin, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.09259)  

**Abstract**: Large Language Model (LLM) Agents exhibit inherent reasoning abilities through the collaboration of multiple tools. However, during agent inference, existing methods often suffer from (i) locally myopic generation, due to the absence of lookahead, and (ii) trajectory instability, where minor early errors can escalate into divergent reasoning paths. These issues make it difficult to balance global effectiveness and computational efficiency. To address these two issues, we propose meta-adaptive exploration with LLM agents this https URL, a meta-adaptive reasoning framework based on LLM Agents that flexibly integrates tool execution and reasoning planning. MAXS employs a lookahead strategy to extend reasoning paths a few steps ahead, estimating the advantage value of tool usage, and combines step consistency variance and inter-step trend slopes to jointly select stable, consistent, and high-value reasoning steps. Additionally, we introduce a trajectory convergence mechanism that controls computational cost by halting further rollouts once path consistency is achieved, enabling a balance between resource efficiency and global effectiveness in multi-tool reasoning. We conduct extensive empirical studies across three base models (MiMo-VL-7B, Qwen2.5-VL-7B, Qwen2.5-VL-32B) and five datasets, demonstrating that MAXS consistently outperforms existing methods in both performance and inference efficiency. Further analysis confirms the effectiveness of our lookahead strategy and tool usage. 

---
# DScheLLM: Enabling Dynamic Scheduling through a Fine-Tuned Dual-System Large language Model 

**Authors**: Lixiang Zhang, Chenggong Zhao, Qing Gao, Xiaoke Zhao, Gengyi Bai, Jinhu Lv  

**Link**: [PDF](https://arxiv.org/pdf/2601.09100)  

**Abstract**: Production scheduling is highly susceptible to dynamic disruptions, such as variations in processing times, machine availability, and unexpected task insertions. Conventional approaches typically rely on event-specific models and explicit analytical formulations, which limits their adaptability and generalization across previously unseen disturbances. To overcome these limitations, this paper proposes DScheLLM, a dynamic scheduling approach that leverages fine-tuned large language models within a dual-system (fast-slow) reasoning architecture to address disturbances of different scales. A unified large language model-based framework is constructed to handle dynamic events, where training datasets for both fast and slow reasoning modes are generated using exact schedules obtained from an operations research solver. The Huawei OpenPangu Embedded-7B model is subsequently fine-tuned under the hybrid reasoning paradigms using LoRA. Experimental evaluations on standard job shop scheduling benchmarks demonstrate that the fast-thinking mode can efficiently generate high-quality schedules and the slow-thinking mode can produce solver-compatible and well-formatted decision inputs. To the best of our knowledge, this work represents one of the earliest studies applying large language models to job shop scheduling in dynamic environments, highlighting their considerable potential for intelligent and adaptive scheduling optimization. 

---
# Coordinated Pandemic Control with Large Language Model Agents as Policymaking Assistants 

**Authors**: Ziyi Shi, Xusen Guo, Hongliang Lu, Mingxing Peng, Haotian Wang, Zheng Zhu, Zhenning Li, Yuxuan Liang, Xinhu Zheng, Hai Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.09264)  

**Abstract**: Effective pandemic control requires timely and coordinated policymaking across administrative regions that are intrinsically interdependent. However, human-driven responses are often fragmented and reactive, with policies formulated in isolation and adjusted only after outbreaks escalate, undermining proactive intervention and global pandemic mitigation. To address this challenge, here we propose a large language model (LLM) multi-agent policymaking framework that supports coordinated and proactive pandemic control across regions. Within our framework, each administrative region is assigned an LLM agent as an AI policymaking assistant. The agent reasons over region-specific epidemiological dynamics while communicating with other agents to account for cross-regional interdependencies. By integrating real-world data, a pandemic evolution simulator, and structured inter-agent communication, our framework enables agents to jointly explore counterfactual intervention scenarios and synthesize coordinated policy decisions through a closed-loop simulation process. We validate the proposed framework using state-level COVID-19 data from the United States between April and December 2020, together with real-world mobility records and observed policy interventions. Compared with real-world pandemic outcomes, our approach reduces cumulative infections and deaths by up to 63.7% and 40.1%, respectively, at the individual state level, and by 39.0% and 27.0%, respectively, when aggregated across states. These results demonstrate that LLM multi-agent systems can enable more effective pandemic control with coordinated policymaking... 

---
# STaR: Sensitive Trajectory Regulation for Unlearning in Large Reasoning Models 

**Authors**: Jingjing Zhou, Gaoxiang Cong, Li Su, Liang Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.09281)  

**Abstract**: Large Reasoning Models (LRMs) have advanced automated multi-step reasoning, but their ability to generate complex Chain-of-Thought (CoT) trajectories introduces severe privacy risks, as sensitive information may be deeply embedded throughout the reasoning process. Existing Large Language Models (LLMs) unlearning approaches that typically focus on modifying only final answers are insufficient for LRMs, as they fail to remove sensitive content from intermediate steps, leading to persistent privacy leakage and degraded security. To address these challenges, we propose Sensitive Trajectory Regulation (STaR), a parameter-free, inference-time unlearning framework that achieves robust privacy protection throughout the reasoning process. Specifically, we first identify sensitive content via semantic-aware detection. Then, we inject global safety constraints through secure prompt prefix. Next, we perform trajectory-aware suppression to dynamically block sensitive content across the entire reasoning chain. Finally, we apply token-level adaptive filtering to prevent both exact and paraphrased sensitive tokens during generation. Furthermore, to overcome the inadequacies of existing evaluation protocols, we introduce two metrics: Multi-Decoding Consistency Assessment (MCS), which measures the consistency of unlearning across diverse decoding strategies, and Multi-Granularity Membership Inference Attack (MIA) Evaluation, which quantifies privacy protection at both answer and reasoning-chain levels. Experiments on the R-TOFU benchmark demonstrate that STaR achieves comprehensive and stable unlearning with minimal utility loss, setting a new standard for privacy-preserving reasoning in LRMs. 

---
# Programming over Thinking: Efficient and Robust Multi-Constraint Planning 

**Authors**: Derrick Goh Xin Deik, Quanyu Long, Zhengyuan Liu, Nancy F. Chen, Wenya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.09097)  

**Abstract**: Multi-constraint planning involves identifying, evaluating, and refining candidate plans while satisfying multiple, potentially conflicting constraints. Existing large language model (LLM) approaches face fundamental limitations in this domain. Pure reasoning paradigms, which rely on long natural language chains, are prone to inconsistency, error accumulation, and prohibitive cost as constraints compound. Conversely, LLMs combined with coding- or solver-based strategies lack flexibility: they often generate problem-specific code from scratch or depend on fixed solvers, failing to capture generalizable logic across diverse problems. To address these challenges, we introduce the Scalable COde Planning Engine (SCOPE), a framework that disentangles query-specific reasoning from generic code execution. By separating reasoning from execution, SCOPE produces solver functions that are consistent, deterministic, and reusable across queries while requiring only minimal changes to input parameters. SCOPE achieves state-of-the-art performance while lowering cost and latency. For example, with GPT-4o, it reaches 93.1% success on TravelPlanner, a 61.6% gain over the best baseline (CoT) while cutting inference cost by 1.4x and time by ~4.67x. Code is available at this https URL. 

---
# PrivacyReasoner: Can LLM Emulate a Human-like Privacy Mind? 

**Authors**: Yiwen Tu, Xuan Liu, Lianhui Qin, Haojian Jin  

**Link**: [PDF](https://arxiv.org/pdf/2601.09152)  

**Abstract**: This paper introduces PRA, an AI-agent design for simulating how individual users form privacy concerns in response to real-world news. Moving beyond population-level sentiment analysis, PRA integrates privacy and cognitive theories to simulate user-specific privacy reasoning grounded in personal comment histories and contextual cues. The agent reconstructs each user's "privacy mind", dynamically activates relevant privacy memory through a contextual filter that emulates bounded rationality, and generates synthetic comments reflecting how that user would likely respond to new privacy scenarios. A complementary LLM-as-a-Judge evaluator, calibrated against an established privacy concern taxonomy, quantifies the faithfulness of generated reasoning. Experiments on real-world Hacker News discussions show that \PRA outperforms baseline agents in privacy concern prediction and captures transferable reasoning patterns across domains including AI, e-commerce, and healthcare. 

---
# ART: Action-based Reasoning Task Benchmarking for Medical AI Agents 

**Authors**: Ananya Mantravadi, Shivali Dalmia, Abhishek Mukherji  

**Link**: [PDF](https://arxiv.org/pdf/2601.08988)  

**Abstract**: Reliable clinical decision support requires medical AI agents capable of safe, multi-step reasoning over structured electronic health records (EHRs). While large language models (LLMs) show promise in healthcare, existing benchmarks inadequately assess performance on action-based tasks involving threshold evaluation, temporal aggregation, and conditional logic. We introduce ART, an Action-based Reasoning clinical Task benchmark for medical AI agents, which mines real-world EHR data to create challenging tasks targeting known reasoning weaknesses. Through analysis of existing benchmarks, we identify three dominant error categories: retrieval failures, aggregation errors, and conditional logic misjudgments. Our four-stage pipeline -- scenario identification, task generation, quality audit, and evaluation -- produces diverse, clinically validated tasks grounded in real patient data. Evaluating GPT-4o-mini and Claude 3.5 Sonnet on 600 tasks shows near-perfect retrieval after prompt refinement, but substantial gaps in aggregation (28--64%) and threshold reasoning (32--38%). By exposing failure modes in action-oriented EHR reasoning, ART advances toward more reliable clinical agents, an essential step for AI systems that reduce cognitive load and administrative burden, supporting workforce capacity in high-demand care settings 

---
# The Promptware Kill Chain: How Prompt Injections Gradually Evolved Into a Multi-Step Malware 

**Authors**: Ben Nassi, Bruce Schneier, Oleg Brodt  

**Link**: [PDF](https://arxiv.org/pdf/2601.09625)  

**Abstract**: The rapid adoption of large language model (LLM)-based systems -- from chatbots to autonomous agents capable of executing code and financial transactions -- has created a new attack surface that existing security frameworks inadequately address. The dominant framing of these threats as "prompt injection" -- a catch-all phrase for security failures in LLM-based systems -- obscures a more complex reality: Attacks on LLM-based systems increasingly involve multi-step sequences that mirror traditional malware campaigns. In this paper, we propose that attacks targeting LLM-based applications constitute a distinct class of malware, which we term \textit{promptware}, and introduce a five-step kill chain model for analyzing these threats. The framework comprises Initial Access (prompt injection), Privilege Escalation (jailbreaking), Persistence (memory and retrieval poisoning), Lateral Movement (cross-system and cross-user propagation), and Actions on Objective (ranging from data exfiltration to unauthorized transactions). By mapping recent attacks to this structure, we demonstrate that LLM-related attacks follow systematic sequences analogous to traditional malware campaigns. The promptware kill chain offers security practitioners a structured methodology for threat modeling and provides a common vocabulary for researchers across AI safety and cybersecurity to address a rapidly evolving threat landscape. 

---
# Routing with Generated Data: Annotation-Free LLM Skill Estimation and Expert Selection 

**Authors**: Tianyi Niu, Justin Chih-Yao Chen, Genta Indra Winata, Shi-Xiong Zhang, Supriyo Chakraborty, Sambit Sahu, Yue Zhang, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2601.09692)  

**Abstract**: Large Language Model (LLM) routers dynamically select optimal models for given inputs. Existing approaches typically assume access to ground-truth labeled data, which is often unavailable in practice, especially when user request distributions are heterogeneous and unknown. We introduce Routing with Generated Data (RGD), a challenging setting in which routers are trained exclusively on generated queries and answers produced from high-level task descriptions by generator LLMs. We evaluate query-answer routers (using both queries and labels) and query-only routers across four diverse benchmarks and 12 models, finding that query-answer routers degrade faster than query-only routers as generator quality decreases. Our analysis reveals two crucial characteristics of effective generators: they must accurately respond to their own questions, and their questions must produce sufficient performance differentiation among the model pool. We then show how filtering for these characteristics can improve the quality of generated data. We further propose CASCAL, a novel query-only router that estimates model correctness through consensus voting and identifies model-specific skill niches via hierarchical clustering. CASCAL is substantially more robust to generator quality, outperforming the best query-answer router by 4.6% absolute accuracy when trained on weak generator data. 

---
# From Prompt to Protocol: Fast Charging Batteries with Large Language Models 

**Authors**: Ge Lei, Ferran Brosa Planella, Sterling G. Baird, Samuel J. Cooper  

**Link**: [PDF](https://arxiv.org/pdf/2601.09626)  

**Abstract**: Efficiently optimizing battery charging protocols is challenging because each evaluation is slow, costly, and non-differentiable. Many existing approaches address this difficulty by heavily constraining the protocol search space, which limits the diversity of protocols that can be explored, preventing the discovery of higher-performing solutions. We introduce two gradient-free, LLM-driven closed-loop methods: Prompt-to-Optimizer (P2O), which uses an LLM to propose the code for small neural-network-based protocols, which are then trained by an inner loop, and Prompt-to-Protocol (P2P), which simply writes an explicit function for the current and its scalar parameters. Across our case studies, LLM-guided P2O outperforms neural networks designed by Bayesian optimization, evolutionary algorithms, and random search. In a realistic fast charging scenario, both P2O and P2P yield around a 4.2 percent improvement in state of health (capacity retention based health metric under fast charging cycling) over a state-of-the-art multi-step constant current (CC) baseline, with P2P achieving this under matched evaluation budgets (same number of protocol evaluations). These results demonstrate that LLMs can expand the space of protocol functional forms, incorporate language-based constraints, and enable efficient optimization in high cost experimental settings. 

---
# LLMs can Compress LLMs: Adaptive Pruning by Agents 

**Authors**: Sai Varun Kodathala, Rakesh Vunnam  

**Link**: [PDF](https://arxiv.org/pdf/2601.09694)  

**Abstract**: As Large Language Models (LLMs) continue to scale, post-training pruning has emerged as a promising approach to reduce computational costs while preserving performance. Existing methods such as SparseGPT and Wanda achieve high sparsity through layer-wise weight reconstruction or activation-aware magnitude pruning, but rely on uniform or hand-crafted heuristics to determine per-layer sparsity ratios. Moreover, recent work has shown that pruned LLMs suffer from severe factual knowledge degradation, with structured pruning methods experiencing near-total collapse in factual question-answering capabilities. We introduce agent-guided pruning, where a foundation model acts as an adaptive pruning agent to intelligently select which layers to prune at each iteration while preserving critical knowledge pathways. Our method constructs layer-wise sensitivity profiles by combining Wanda-inspired weight-activation metrics with gradient importance scores, normalized as z-scores for model-agnostic comparison. These statistics are processed by an LLM agent equipped with self-reflection capabilities, enabling it to learn from previous pruning outcomes and iteratively refine its strategy. A checkpoint rollback mechanism maintains model quality by reverting when perplexity degradation exceeds a threshold. We evaluate our approach on Qwen3 models (4B and 8B parameters) at approximately 45% sparsity, demonstrating substantial improvements over structured pruning baselines: 56% relative improvement in MMLU accuracy, 19x better factual knowledge retention on FreebaseQA, and 69% lower perplexity degradation. Notably, our framework requires no retraining, operates in a model-agnostic manner, and exhibits effective self-correction with only 2-4 rollbacks across 21-40 iterations, demonstrating that foundation models can effectively guide the compression of other foundation models. 

---
# SimMerge: Learning to Select Merge Operators from Similarity Signals 

**Authors**: Oliver Bolton, Aakanksha, Arash Ahmadian, Sara Hooker, Marzieh Fadaee, Beyza Ermis  

**Link**: [PDF](https://arxiv.org/pdf/2601.09473)  

**Abstract**: Model merging enables multiple large language models (LLMs) to be combined into a single model while preserving performance. This makes it a valuable tool in LLM development, offering a competitive alternative to multi-task training. However, merging can be difficult at scale, as successful merging requires choosing the right merge operator, selecting the right models, and merging them in the right order. This often leads researchers to run expensive merge-and-evaluate searches to select the best merge. In this work, we provide an alternative by introducing \simmerge{}, \emph{a predictive merge-selection method} that selects the best merge using inexpensive, task-agnostic similarity signals between models. From a small set of unlabeled probes, we compute functional and structural features and use them to predict the performance of a given 2-way merge. Using these predictions, \simmerge{} selects the best merge operator, the subset of models to merge, and the merge order, eliminating the expensive merge-and-evaluate loop. We demonstrate that we surpass standard merge-operator performance on 2-way merges of 7B-parameter LLMs, and that \simmerge{} generalizes to multi-way merges and 111B-parameter LLM merges without retraining. Additionally, we present a bandit variant that supports adding new tasks, models, and operators on the fly. Our results suggest that learning how to merge is a practical route to scalable model composition when checkpoint catalogs are large and evaluation budgets are tight. 

---
# Bridging Semantic Understanding and Popularity Bias with LLMs 

**Authors**: Renqiang Luo, Dong Zhang, Yupeng Gao, Wen Shi, Mingliang Hou, Jiaying Liu, Zhe Wang, Shuo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.09478)  

**Abstract**: Semantic understanding of popularity bias is a crucial yet underexplored challenge in recommender systems, where popular items are often favored at the expense of niche content. Most existing debiasing methods treat the semantic understanding of popularity bias as a matter of diversity enhancement or long-tail coverage, neglecting the deeper semantic layer that embodies the causal origins of the bias itself. Consequently, such shallow interpretations limit both their debiasing effectiveness and recommendation accuracy. In this paper, we propose FairLRM, a novel framework that bridges the gap in the semantic understanding of popularity bias with Recommendation via Large Language Model (RecLLM). FairLRM decomposes popularity bias into item-side and user-side components, using structured instruction-based prompts to enhance the model's comprehension of both global item distributions and individual user preferences. Unlike traditional methods that rely on surface-level features such as "diversity" or "debiasing", FairLRM improves the model's ability to semantically interpret and address the underlying bias. Through empirical evaluation, we show that FairLRM significantly enhances both fairness and recommendation accuracy, providing a more semantically aware and trustworthy approach to enhance the semantic understanding of popularity bias. The implementation is available at this https URL. 

---
# Improving Symbolic Translation of Language Models for Logical Reasoning 

**Authors**: Ramya Keerthy Thatikonda, Jiuzhou Han, Wray Buntine, Ehsan Shareghi  

**Link**: [PDF](https://arxiv.org/pdf/2601.09446)  

**Abstract**: The use of formal language for deductive logical reasoning aligns well with language models (LMs), where translating natural language (NL) into first-order logic (FOL) and employing an external solver results in a verifiable and therefore reliable reasoning system. However, smaller LMs often struggle with this translation task, frequently producing incorrect symbolic outputs due to formatting and translation errors. Existing approaches typically rely on self-iteration to correct these errors, but such methods depend heavily on the capabilities of the underlying model. To address this, we first categorize common errors and fine-tune smaller LMs using data synthesized by large language models. The evaluation is performed using the defined error categories. We introduce incremental inference, which divides inference into two stages, predicate generation and FOL translation, providing greater control over model behavior and enhancing generation quality as measured by predicate metrics. This decomposition framework also enables the use of a verification module that targets predicate-arity errors to further improve performance. Our study evaluates three families of models across four logical-reasoning datasets. The comprehensive fine-tuning, incremental inference, and verification modules reduce error rates, increase predicate coverage, and improve reasoning performance for smaller LMs, moving us closer to developing reliable and accessible symbolic-reasoning systems. 

---
# Population-Aligned Audio Reproduction With LLM-Based Equalizers 

**Authors**: Ioannis Stylianou, Jon Francombe, Pablo Martinez-Nuevo, Sven Ewan Shepstone, Zheng-Hua Tan  

**Link**: [PDF](https://arxiv.org/pdf/2601.09448)  

**Abstract**: Conventional audio equalization is a static process that requires manual and cumbersome adjustments to adapt to changing listening contexts (e.g., mood, location, or social setting). In this paper, we introduce a Large Language Model (LLM)-based alternative that maps natural language text prompts to equalization settings. This enables a conversational approach to sound system control. By utilizing data collected from a controlled listening experiment, our models exploit in-context learning and parameter-efficient fine-tuning techniques to reliably align with population-preferred equalization settings. Our evaluation methods, which leverage distributional metrics that capture users' varied preferences, show statistically significant improvements in distributional alignment over random sampling and static preset baselines. These results indicate that LLMs could function as "artificial equalizers," contributing to the development of more accessible, context-aware, and expert-level audio tuning methods. 

---
# On-Device Large Language Models for Sequential Recommendation 

**Authors**: Xin Xia, Hongzhi Yin, Shane Culpepper  

**Link**: [PDF](https://arxiv.org/pdf/2601.09306)  

**Abstract**: On-device recommendation is critical for a number of real-world applications, especially in scenarios that have agreements on execution latency, user privacy, and robust functionality when internet connectivity is unstable or even impossible. While large language models (LLMs) can now provide exceptional capabilities that model user behavior for sequential recommendation tasks, their substantial memory footprint and computational overhead make the deployment on resource-constrained devices a high risk proposition. In this paper, we propose OD-LLM, the first task-adaptive compression framework explicitly designed to provide efficient and accurate on-device deployment of LLMs for sequential recommendation tasks. OD-LLM uniquely integrates two complementary compression strategies: a low-rank structural compression algorithm which uses Singular Value Decomposition (SVD) to significantly reduce parameter redundancy in the model, and a novel tokenization normalization technique that better complements the low-rank decomposition process being used. Additionally, to minimize any potential performance degradation when using higher compression ratios, a novel progressive alignment algorithm is used to iteratively refine the parameters required layerwise in the target model. Empirical evaluations conducted on sequential recommendation benchmarks show that OD-LLM exhibits no loss in effectiveness when compared to the original recommendation model, when the deployed model size is halved. These promising results demonstrate the efficacy and scalability of OD-LLM, making this novel solution a practical alternative for real-time, on-device solutions wishing to replace expensive, remotely executed LLMs. 

---
# Ability Transfer and Recovery via Modularized Parameters Localization 

**Authors**: Songyao Jin, Kun Zhou, Wenqi Li, Peng Wang, Biwei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.09398)  

**Abstract**: Large language models can be continually pre-trained or fine-tuned to improve performance in specific domains, languages, or skills, but this specialization often degrades other capabilities and may cause catastrophic forgetting. We investigate how abilities are distributed within LLM parameters by analyzing module activations under domain- and language-specific inputs for closely related models. Across layers and modules, we find that ability-related activations are highly concentrated in a small set of channels (typically <5\%), and these channels are largely disentangled with good sufficiency and stability. Building on these observations, we propose ACT (Activation-Guided Channel-wise Ability Transfer), which localizes ability-relevant channels via activation differences and selectively transfers only the corresponding parameters, followed by lightweight fine-tuning for compatibility. Experiments on multilingual mathematical and scientific reasoning show that ACT can recover forgotten abilities while preserving retained skills. It can also merge multiple specialized models to integrate several abilities into a single model with minimal interference. Our code and data will be publicly released. 

---
# ReGraM: Region-First Knowledge Graph Reasoning for Medical Question Answering 

**Authors**: Chaerin Lee, Sohee Park, Hyunsik Na, Daseon Choi  

**Link**: [PDF](https://arxiv.org/pdf/2601.09280)  

**Abstract**: Recent studies in medical question answering (Medical QA) have actively explored the integration of large language models (LLMs) with biomedical knowledge graphs (KGs) to improve factual accuracy. However, most existing approaches still rely on traversing the entire KG or performing large-scale retrieval, which introduces substantial noise and leads to unstable multi-hop reasoning. We argue that the core challenge lies not in expanding access to knowledge, but in identifying and reasoning over the appropriate subset of evidence for each query. ReGraM is a region-first knowledge graph reasoning framework that addresses this challenge by constructing a query-aligned subgraph and performing stepwise reasoning constrained to this localized region under multiple evidence aware modes. By focusing inference on only the most relevant portion of the KG, ReGraM departs from the assumption that all relations are equally useful an assumption that rarely holds in domain-specific medical settings. Experiments on seven medical QA benchmarks demonstrate that ReGraM consistently outperforms a strong baseline (KGARevion), achieving an 8.04% absolute accuracy gain on MCQ, a 4.50% gain on SAQ, and a 42.9% reduction in hallucination rate. Ablation and qualitative analyses further show that aligning region construction with hop-wise reasoning is the primary driver of these improvements. Overall, our results highlight region-first KG reasoning as an effective paradigm for improving factual accuracy and consistency in medical QA. 

---
# ProFit: Leveraging High-Value Signals in SFT via Probability-Guided Token Selection 

**Authors**: Tao Liu, Taiqiang Wu, Runming Yang, Shaoning Sun, Junjie Wang, Yujiu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.09195)  

**Abstract**: Supervised fine-tuning (SFT) is a fundamental post-training strategy to align Large Language Models (LLMs) with human intent. However, traditional SFT often ignores the one-to-many nature of language by forcing alignment with a single reference answer, leading to the model overfitting to non-core expressions. Although our empirical analysis suggests that introducing multiple reference answers can mitigate this issue, the prohibitive data and computational costs necessitate a strategic shift: prioritizing the mitigation of single-reference overfitting over the costly pursuit of answer diversity. To achieve this, we reveal the intrinsic connection between token probability and semantic importance: high-probability tokens carry the core logical framework, while low-probability tokens are mostly replaceable expressions. Based on this insight, we propose ProFit, which selectively masks low-probability tokens to prevent surface-level overfitting. Extensive experiments confirm that ProFit consistently outperforms traditional SFT baselines on general reasoning and mathematical benchmarks. 

---
# A.X K1 Technical Report 

**Authors**: Sung Jun Cheon, Jaekyung Cho, Seongho Choi, Hyunjun Eun, Seokhwan Jo, Jaehyun Jun, Minsoo Kang, Jin Kim, Jiwon Kim, Minsang Kim, Sungwan Kim, Seungsik Kim, Tae Yoon Kim, Youngrang Kim, Hyeongmun Lee, Sangyeol Lee, Sungeun Lee, Youngsoon Lee, Yujin Lee, Seongmin Ok, Chanyong Park, Hyewoong Park, Junyoung Park, Hyunho Yang, Subin Yi, Soohyun Bae, Dhammiko Arya, Yongseok Choi, Sangho Choi, Dongyeon Cho, Seungmo Cho, Gyoungeun Han, Yong-jin Han, Seokyoung Hong, Hyeon Hwang, Wonbeom Jang, Minjeong Ju, Wonjin Jung, Keummin Ka, Sungil Kang, Dongnam Kim, Joonghoon Kim, Jonghwi Kim, SaeRom Kim, Sangjin Kim, Seongwon Kim, Youngjin Kim, Seojin Lee, Sunwoo Lee, Taehoon Lee, Chanwoo Park, Sohee Park, Sooyeon Park, Yohan Ra, Sereimony Sek, Seungyeon Seo, Gun Song, Sanghoon Woo, Janghan Yoon, Sungbin Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2601.09200)  

**Abstract**: We introduce A.X K1, a 519B-parameter Mixture-of-Experts (MoE) language model trained from scratch. Our design leverages scaling laws to optimize training configurations and vocabulary size under fixed computational budgets. A.X K1 is pre-trained on a corpus of approximately 10T tokens, curated by a multi-stage data processing pipeline. Designed to bridge the gap between reasoning capability and inference efficiency, A.X K1 supports explicitly controllable reasoning to facilitate scalable deployment across diverse real-world scenarios. We propose a simple yet effective Think-Fusion training recipe, enabling user-controlled switching between thinking and non-thinking modes within a single unified model. Extensive evaluations demonstrate that A.X K1 achieves performance competitive with leading open-source models, while establishing a distinctive advantage in Korean-language benchmarks. 

---
# Mi:dm 2.0 Korea-centric Bilingual Language Models 

**Authors**: Donghoon Shin, Sejung Lee, Soonmin Bae, Hwijung Ryu, Changwon Ok, Hoyoun Jung, Hyesung Ji, Jeehyun Lim, Jehoon Lee, Ji-Eun Han, Jisoo Baik, Mihyeon Kim, Riwoo Chung, Seongmin Lee, Wonjae Park, Yoonseok Heo, Youngkyung Seo, Seyoun Won, Boeun Kim, Cheolhun Heo, Eunkyeong Lee, Honghee Lee, Hyeongju Ju, Hyeontae Seo, Jeongyong Shim, Jisoo Lee, Junseok Koh, Junwoo Kim, Minho Lee, Minji Kang, Minju Kim, Sangha Nam, Seongheum Park, Taehyeong Kim, Euijai Ahn, Hong Seok Jeung, Jisu Shin, Jiyeon Kim, Seonyeong Song, Seung Hyun Kong, Sukjin Hong, Taeyang Yun, Yu-Seon Kim, A-Hyun Lee, Chae-Jeong Lee, Hye-Won Yu, Ji-Hyun Ahn, Song-Yeon Kim, Sun-Woo Jung, Eunju Kim, Eunji Ha, Jinwoo Baek, Yun-ji Lee, Wanjin Park, Jeong Yeop Kim, Eun Mi Kim, Hyoung Jun Park, Jung Won Yoon, Min Sung Noh, Myung Gyo Oh, Wongyoung Lee, Yun Jin Park, Young S. Kwon, Hyun Keun Kim, Jieun Lee, YeoJoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2601.09066)  

**Abstract**: We introduce Mi:dm 2.0, a bilingual large language model (LLM) specifically engineered to advance Korea-centric AI. This model goes beyond Korean text processing by integrating the values, reasoning patterns, and commonsense knowledge inherent to Korean society, enabling nuanced understanding of cultural contexts, emotional subtleties, and real-world scenarios to generate reliable and culturally appropriate responses. To address limitations of existing LLMs, often caused by insufficient or low-quality Korean data and lack of cultural alignment, Mi:dm 2.0 emphasizes robust data quality through a comprehensive pipeline that includes proprietary data cleansing, high-quality synthetic data generation, strategic data mixing with curriculum learning, and a custom Korean-optimized tokenizer to improve efficiency and coverage. To realize this vision, we offer two complementary configurations: Mi:dm 2.0 Base (11.5B parameters), built with a depth-up scaling strategy for general-purpose use, and Mi:dm 2.0 Mini (2.3B parameters), optimized for resource-constrained environments and specialized tasks. Mi:dm 2.0 achieves state-of-the-art performance on Korean-specific benchmarks, with top-tier zero-shot results on KMMLU and strong internal evaluation results across language, humanities, and social science tasks. The Mi:dm 2.0 lineup is released under the MIT license to support extensive research and commercial use. By offering accessible and high-performance Korea-centric LLMs, KT aims to accelerate AI adoption across Korean industries, public services, and education, strengthen the Korean AI developer community, and lay the groundwork for the broader vision of K-intelligence. Our models are available at this https URL. For technical inquiries, please contact midm-llm@kt.com. 

---
# LP-LLM: End-to-End Real-World Degraded License Plate Text Recognition via Large Multimodal Models 

**Authors**: Haoyan Gong, Hongbin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.09116)  

**Abstract**: Real-world License Plate Recognition (LPR) faces significant challenges from severe degradations such as motion blur, low resolution, and complex illumination. The prevailing "restoration-then-recognition" two-stage paradigm suffers from a fundamental flaw: the pixel-level optimization objectives of image restoration models are misaligned with the semantic goals of character recognition, leading to artifact interference and error accumulation. While Vision-Language Models (VLMs) have demonstrated powerful general capabilities, they lack explicit structural modeling for license plate character sequences (e.g., fixed length, specific order). To address this, we propose an end-to-end structure-aware multimodal reasoning framework based on Qwen3-VL. The core innovation lies in the Character-Aware Multimodal Reasoning Module (CMRM), which introduces a set of learnable Character Slot Queries. Through a cross-attention mechanism, these queries actively retrieve fine-grained evidence corresponding to character positions from visual features. Subsequently, we inject these character-aware representations back into the visual tokens via residual modulation, enabling the language model to perform autoregressive generation based on explicit structural priors. Furthermore, combined with the LoRA parameter-efficient fine-tuning strategy, the model achieves domain adaptation while retaining the generalization capabilities of the large model. Extensive experiments on both synthetic and real-world severely degraded datasets demonstrate that our method significantly outperforms existing restoration-recognition combinations and general VLMs, validating the superiority of incorporating structured reasoning into large models for low-quality text recognition tasks. 

---
# From Symbolic to Natural-Language Relations: Rethinking Knowledge Graph Construction in the Era of Large Language Models 

**Authors**: Kanyao Han, Yushang Lai  

**Link**: [PDF](https://arxiv.org/pdf/2601.09069)  

**Abstract**: Knowledge graphs (KGs) have commonly been constructed using predefined symbolic relation schemas, typically implemented as categorical relation labels. This design has notable shortcomings: real-world relations are often contextual, nuanced, and sometimes uncertain, and compressing it into discrete relation labels abstracts away critical semantic detail. Nevertheless, symbolic-relation KGs remain widely used because they have been operationally effective and broadly compatible with pre-LLM downstream models and algorithms, in which KG knowledge could be retrieved or encoded into quantified features and embeddings at scale. The emergence of LLMs has reshaped how knowledge is created and consumed. LLMs support scalable synthesis of domain facts directly in concise natural language, and prompting-based inference favors context-rich free-form text over quantified representations. This position paper argues that these changes call for rethinking the representation of relations themselves rather than merely using LLMs to populate conventional schemas more efficiently. We therefore advocate moving from symbolic to natural-language relation descriptions, and we propose hybrid design principles that preserve a minimal structural backbone while enabling more flexible and context-sensitive relational representations. 

---
# A Decompilation-Driven Framework for Malware Detection with Large Language Models 

**Authors**: Aniesh Chawla, Udbhav Prasad  

**Link**: [PDF](https://arxiv.org/pdf/2601.09035)  

**Abstract**: The parallel evolution of Large Language Models (LLMs) with advanced code-understanding capabilities and the increasing sophistication of malware presents a new frontier for cybersecurity research. This paper evaluates the efficacy of state-of-the-art LLMs in classifying executable code as either benign or malicious. We introduce an automated pipeline that first decompiles Windows executable into a C code using Ghidra disassembler and then leverages LLMs to perform the classification. Our evaluation reveals that while standard LLMs show promise, they are not yet robust enough to replace traditional anti-virus software. We demonstrate that a fine-tuned model, trained on curated malware and benign datasets, significantly outperforms its vanilla counterpart. However, the performance of even this specialized model degrades notably when encountering newer malware. This finding demonstrates the critical need for continuous fine-tuning with emerging threats to maintain model effectiveness against the changing coding patterns and behaviors of malicious software. 

---
# Evaluating Role-Consistency in LLMs for Counselor Training 

**Authors**: Eric Rudolph, Natalie Engert, Jens Albrecht  

**Link**: [PDF](https://arxiv.org/pdf/2601.08892)  

**Abstract**: The rise of online counseling services has highlighted the need for effective training methods for future counselors. This paper extends research on VirCo, a Virtual Client for Online Counseling, designed to complement traditional role-playing methods in academic training by simulating realistic client interactions. Building on previous work, we introduce a new dataset incorporating adversarial attacks to test the ability of large language models (LLMs) to maintain their assigned roles (role-consistency). The study focuses on evaluating the role consistency and coherence of the Vicuna model's responses, comparing these findings with earlier research. Additionally, we assess and compare various open-source LLMs for their performance in sustaining role consistency during virtual client interactions. Our contributions include creating an adversarial dataset, evaluating conversation coherence and persona consistency, and providing a comparative analysis of different LLMs. 

---
# Can LLMs interpret figurative language as humans do?: surface-level vs representational similarity 

**Authors**: Samhita Bollepally, Aurora Sloman-Moll, Takashi Yamauchi  

**Link**: [PDF](https://arxiv.org/pdf/2601.09041)  

**Abstract**: Large language models generate judgments that resemble those of humans. Yet the extent to which these models align with human judgments in interpreting figurative and socially grounded language remains uncertain. To investigate this, human participants and four instruction-tuned LLMs of different sizes (GPT-4, Gemma-2-9B, Llama-3.2, and Mistral-7B) rated 240 dialogue-based sentences representing six linguistic traits: conventionality, sarcasm, funny, emotional, idiomacy, and slang. Each of the 240 sentences was paired with 40 interpretive questions, and both humans and LLMs rated these sentences on a 10-point Likert scale. Results indicated that humans and LLMs aligned at the surface level with humans, but diverged significantly at the representational level, especially in interpreting figurative sentences involving idioms and Gen Z slang. GPT-4 most closely approximates human representational patterns, while all models struggle with context-dependent and socio-pragmatic expressions like sarcasm, slang, and idiomacy. 

---
# Proactively Detecting Threats: A Novel Approach Using LLMs 

**Authors**: Aniesh Chawla, Udbhav Prasad  

**Link**: [PDF](https://arxiv.org/pdf/2601.09029)  

**Abstract**: Enterprise security faces escalating threats from sophisticated malware, compounded by expanding digital operations. This paper presents the first systematic evaluation of large language models (LLMs) to proactively identify indicators of compromise (IOCs) from unstructured web-based threat intelligence sources, distinguishing it from reactive malware detection approaches. We developed an automated system that pulls IOCs from 15 web-based threat report sources to evaluate six LLM models (Gemini, Qwen, and Llama variants). Our evaluation of 479 webpages containing 2,658 IOCs (711 IPv4 addresses, 502 IPv6 addresses, 1,445 domains) reveals significant performance variations. Gemini 1.5 Pro achieved 0.958 precision and 0.788 specificity for malicious IOC identification, while demonstrating perfect recall (1.0) for actual threats. 

---
# OpenDecoder: Open Large Language Model Decoding to Incorporate Document Quality in RAG 

**Authors**: Fengran Mo, Zhan Su, Yuchen Hui, Jinghan Zhang, Jia Ao Sun, Zheyuan Liu, Chao Zhang, Tetsuya Sakai, Jian-Yun Nie  

**Link**: [PDF](https://arxiv.org/pdf/2601.09028)  

**Abstract**: The development of large language models (LLMs) has achieved superior performance in a range of downstream tasks, including LLM-based retrieval-augmented generation (RAG). The quality of generated content heavily relies on the usefulness of the retrieved information and the capacity of LLMs' internal information processing mechanism to incorporate it in answer generation. It is generally assumed that the retrieved information is relevant to the question. However, the retrieved information may have a variable degree of relevance and usefulness, depending on the question and the document collection. It is important to take into account the relevance of the retrieved information in answer generation. In this paper, we propose OpenDecoder, a new approach that leverages explicit evaluation of the retrieved information as quality indicator features for generation. We aim to build a RAG model that is more robust to varying levels of noisy context. Three types of explicit evaluation information are considered: relevance score, ranking score, and QPP (query performance prediction) score. The experimental results on five benchmark datasets demonstrate the effectiveness and better robustness of OpenDecoder by outperforming various baseline methods. Importantly, this paradigm is flexible to be integrated with the post-training of LLMs for any purposes and incorporated with any type of external indicators. 

---
# Adaptive Trust Metrics for Multi-LLM Systems: Enhancing Reliability in Regulated Industries 

**Authors**: Tejaswini Bollikonda  

**Link**: [PDF](https://arxiv.org/pdf/2601.08858)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in sensitive domains such as healthcare, finance, and law, yet their integration raises pressing concerns around trust, accountability, and reliability. This paper explores adaptive trust metrics for multi LLM ecosystems, proposing a framework for quantifying and improving model reliability under regulated constraints. By analyzing system behaviors, evaluating uncertainty across multiple LLMs, and implementing dynamic monitoring pipelines, the study demonstrates practical pathways for operational trustworthiness. Case studies from financial compliance and healthcare diagnostics illustrate the applicability of adaptive trust metrics in real world settings. The findings position adaptive trust measurement as a foundational enabler for safe and scalable AI adoption in regulated industries. 

---
# LAUDE: LLM-Assisted Unit Test Generation and Debugging of Hardware DEsigns 

**Authors**: Deeksha Nandal, Riccardo Revalor, Soham Dan, Debjit Pal  

**Link**: [PDF](https://arxiv.org/pdf/2601.08856)  

**Abstract**: Unit tests are critical in the hardware design lifecycle to ensure that component design modules are functionally correct and conform to the specification before they are integrated at the system level. Thus developing unit tests targeting various design features requires deep understanding of the design functionality and creativity. When one or more unit tests expose a design failure, the debugging engineer needs to diagnose, localize, and debug the failure to ensure design correctness, which is often a painstaking and intense process. In this work, we introduce LAUDE, a unified unit-test generation and debugging framework for hardware designs that cross-pollinates the semantic understanding of the design source code with the Chain-of-Thought (CoT) reasoning capabilities of foundational Large-Language Models (LLMs). LAUDE integrates prompt engineering and design execution information to enhance its unit test generation accuracy and code debuggability. We apply LAUDE with closed- and open-source LLMs to a large corpus of buggy hardware design codes derived from the VerilogEval dataset, where generated unit tests detected bugs in up to 100% and 93% of combinational and sequential designs and debugged up to 93% and 84% of combinational and sequential designs, respectively. 

---
# Scalable and Reliable Evaluation of AI Knowledge Retrieval Systems: RIKER and the Coherent Simulated Universe 

**Authors**: JV Roig  

**Link**: [PDF](https://arxiv.org/pdf/2601.08847)  

**Abstract**: Evaluating knowledge systems (LLMs, RAG, knowledge graphs, etc) faces fundamental challenges: static benchmarks are vulnerable to contamination, LLM-based judges exhibit systematic biases, and ground truth extraction requires expensive human annotation. We present RIKER (Retrieval Intelligence and Knowledge Extraction Rating), both a benchmark and a replicable methodology based on paradigm inversion - generating documents from known ground truth rather than extracting ground truth from documents. This approach enables deterministic scoring and scalable evaluation without human annotation or reference models, and contamination resistance through regenerable corpora. Our evaluation of 33 models using over 21 billion tokens reveals that context length claims frequently exceed usable capacity, with significant degradation beyond 32K tokens; cross-document aggregation proves substantially harder than single-document extraction; and grounding ability and hallucination resistance are distinct capabilities - models excelling at finding facts that exist may still fabricate facts that do not. Beyond the specific benchmark, we contribute a domain-agnostic methodology for constructing scalable and contamination-resistant evaluations wherever synthetic documents can be generated from structured ground truth. 

---
# Directional Attractors in LLM Reasoning: How Similarity Retrieval Steers Iterative Summarization Based Reasoning 

**Authors**: Cagatay Tekin, Charbel Barakat, Luis Joseph Luna Limgenco  

**Link**: [PDF](https://arxiv.org/pdf/2601.08846)  

**Abstract**: Iterative summarization based reasoning frameworks such as InftyThink enable long-horizon reasoning in large language models (LLMs) by controlling context growth, but they repeatedly regenerate similar reasoning strategies across tasks. We introduce InftyThink with Cross-Chain Memory, an extension that augments iterative reasoning with an embedding-based semantic cache of previously successful reasoning patterns. At each reasoning step, the model retrieves and conditions on the most semantically similar stored lemmas, guiding inference without expanding the context window indiscriminately. Experiments on MATH500, AIME2024, and GPQA-Diamond demonstrate that semantic lemma retrieval improves accuracy in structured domains while exposing failure modes in tests that include heterogeneous domains. Geometric analyses of reasoning trajectories reveal that cache retrieval induces directional biases in embedding space, leading to consistent fix (improve baseline accuracy) and break (degradation in baseline accuracy) attractors. Our results highlight both the benefits and limits of similarity-based memory for self-improving LLM reasoning. 

---
# Consistency-Aware Editing for Entity-level Unlearning in Language Models 

**Authors**: Xiaoqi Han, Víctor Gutiérrez-Basulto, Ru Li, Xiaoli Li, Jiye Liang, Jeff Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2601.08840)  

**Abstract**: Large language models (LLMs) risk retaining sensitive, copyrighted, or harmful information from their training data. Entity-level unlearning addresses this issue by removing all knowledge of a specific entity while preserving the model's overall capabilities. Existing approaches typically rely on full-model fine-tuning or prompt-based interventions, which can be computationally expensive or brittle when handling paraphrased queries. Recently, model editing has emerged as an efficient alternative for updating knowledge in LLMs, offering a promising direction for unlearning. However, existing editing techniques are typically designed for instance-level updates, modifying responses to specific attributes of an entity rather than eliminating all knowledge associated with the entity. In this paper, we investigate how editing techniques can be adapted for effective and efficient entity-level unlearning. To this end, we introduce a novel consistency-aware editing (CAE) framework. CAE aggregates a diverse set of prompts related to a target entity, including its attributes, relations, and adversarial paraphrases. It then jointly learns a low-rank update guided by a consistency regularizer that aligns the editing directions across prompts. This promotes robust and comprehensive forgetting while minimizing interference with unrelated knowledge. We further examine where different entities are stored within the model and how many diverse prompts are needed for successful unlearning. We evaluate CAE on two challenging benchmarks, RWKU and ToFU, and demonstrate that it (i) provides insights into how entity-level knowledge is internally represented and deleted in LLMs, (ii) significantly improves forgetting accuracy and robustness over traditional unlearning and editing baselines, and (iii) enables scalable entity removal using only tens of carefully selected prompts. 

---
# From Adversarial Poetry to Adversarial Tales: An Interpretability Research Agenda 

**Authors**: Piercosma Bisconti, Marcello Galisai, Matteo Prandi, Federico Pierucci, Olga Sorokoletova, Francesco Giarrusso, Vincenzo Suriani, Marcantonio Brancale, Daniele Nardi  

**Link**: [PDF](https://arxiv.org/pdf/2601.08837)  

**Abstract**: Safety mechanisms in LLMs remain vulnerable to attacks that reframe harmful requests through culturally coded structures. We introduce Adversarial Tales, a jailbreak technique that embeds harmful content within cyberpunk narratives and prompts models to perform functional analysis inspired by Vladimir Propp's morphology of folktales. By casting the task as structural decomposition, the attack induces models to reconstruct harmful procedures as legitimate narrative interpretation. Across 26 frontier models from nine providers, we observe an average attack success rate of 71.3%, with no model family proving reliably robust. Together with our prior work on Adversarial Poetry, these findings suggest that structurally-grounded jailbreaks constitute a broad vulnerability class rather than isolated techniques. The space of culturally coded frames that can mediate harmful intent is vast, likely inexhaustible by pattern-matching defenses alone. Understanding why these attacks succeed is therefore essential: we outline a mechanistic interpretability research agenda to investigate how narrative cues reshape model representations and whether models can learn to recognize harmful intent independently of surface form. 

---
# DeliberationBench: When Do More Voices Hurt? A Controlled Study of Multi-LLM Deliberation Protocols 

**Authors**: Vaarunay Kaushal, Taranveer Singh  

**Link**: [PDF](https://arxiv.org/pdf/2601.08835)  

**Abstract**: Multi-agent systems where Large Language Models (LLMs) deliberate to form consensus have gained significant attention, yet their practical value over simpler methods remains under-scrutinized. We introduce DELIBERATIONBENCH, a controlled benchmark evaluating three deliberation protocols against a strong baseline of selecting the best response from a pool of model outputs. Across 270 questions and three independent seeds (810 total evaluations), we find a striking negative result: the best-single baseline achieves an 82.5% +- 3.3% win rate, dramatically outperforming the best deliberation protocol(13.8% +- 2.6%). This 6.0x performance gap is statistically significant (p < 0.01) and comes at 1.5-2.5x higher computational cost. Our findings challenge assumptions that complexity enhances quality in multi-LLM systems. 

---
# CrowdLLM: Building LLM-Based Digital Populations Augmented with Generative Models 

**Authors**: Ryan Feng Lin, Keyu Tian, Hanming Zheng, Congjing Zhang, Li Zeng, Shuai Huang  

**Link**: [PDF](https://arxiv.org/pdf/2512.07890)  

**Abstract**: The emergence of large language models (LLMs) has sparked much interest in creating LLM-based digital populations that can be applied to many applications such as social simulation, crowdsourcing, marketing, and recommendation systems. A digital population can reduce the cost of recruiting human participants and alleviate many concerns related to human subject study. However, research has found that most of the existing works rely solely on LLMs and could not sufficiently capture the accuracy and diversity of a real human population. To address this limitation, we propose CrowdLLM that integrates pretrained LLMs and generative models to enhance the diversity and fidelity of the digital population. We conduct theoretical analysis of CrowdLLM regarding its great potential in creating cost-effective, sufficiently representative, scalable digital populations that can match the quality of a real crowd. Comprehensive experiments are also conducted across multiple domains (e.g., crowdsourcing, voting, user rating) and simulation studies which demonstrate that CrowdLLM achieves promising performance in both accuracy and distributional fidelity to human data. 

---
# Dissecting Judicial Reasoning in U.S. Copyright Damage Awards 

**Authors**: Pei-Chi Lo, Thomas Y. Lu  

**Link**: [PDF](https://arxiv.org/pdf/2601.09459)  

**Abstract**: Judicial reasoning in copyright damage awards poses a core challenge for computational legal analysis. Although federal courts follow the 1976 Copyright Act, their interpretations and factor weightings vary widely across jurisdictions. This inconsistency creates unpredictability for litigants and obscures the empirical basis of legal decisions. This research introduces a novel discourse-based Large Language Model (LLM) methodology that integrates Rhetorical Structure Theory (RST) with an agentic workflow to extract and quantify previously opaque reasoning patterns from judicial opinions. Our framework addresses a major gap in empirical legal scholarship by parsing opinions into hierarchical discourse structures and using a three-stage pipeline, i.e., Dataset Construction, Discourse Analysis, and Agentic Feature Extraction. This pipeline identifies reasoning components and extract feature labels with corresponding discourse subtrees. In analyzing copyright damage rulings, we show that discourse-augmented LLM analysis outperforms traditional methods while uncovering unquantified variations in factor weighting across circuits. These findings offer both methodological advances in computational legal analysis and practical insights into judicial reasoning, with implications for legal practitioners seeking predictive tools, scholars studying legal principle application, and policymakers confronting inconsistencies in copyright law. 

---
# Fine Grained Evaluation of LLMs-as-Judges 

**Authors**: Sourav Saha, Mandar Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2601.08919)  

**Abstract**: A good deal of recent research has focused on how Large Language Models
(LLMs) may be used as `judges' in place of humans to evaluate the quality
of the output produced by various text / image processing systems. Within
this broader context, a number of studies have investigated the specific
question of how effectively LLMs can be used as relevance assessors for
the standard ad hoc task in Information Retrieval (IR). We extend these
studies by looking at additional questions. Most importantly, we use a
Wikipedia based test collection created by the INEX initiative, and
prompt LLMs to not only judge whether documents are relevant /
non-relevant, but to highlight relevant passages in documents that it
regards as useful. The human relevance assessors involved in creating
this collection were given analogous instructions, i.e., they were asked
to highlight all passages within a document that respond to the
information need expressed in a query. This enables us to evaluate the
quality of LLMs as judges not only at the document level, but to also
quantify how often these `judges' are right for the right reasons.
Our findings suggest that LLMs-as-judges work best under human
supervision. 

---
# Unifying Search and Recommendation in LLMs via Gradient Multi-Subspace Tuning 

**Authors**: Jujia Zhao, Zihan Wang, Shuaiqun Pan, Suzan Verberne, Zhaochun Ren  

**Link**: [PDF](https://arxiv.org/pdf/2601.09496)  

**Abstract**: Search and recommendation (S&R) are core to online platforms, addressing explicit intent through queries and modeling implicit intent from behaviors, respectively. Their complementary roles motivate a unified modeling paradigm. Early studies to unify S&R adopt shared encoders with task-specific heads, while recent efforts reframe item ranking in both S&R as conditional generation. The latter holds particular promise, enabling end-to-end optimization and leveraging the semantic understanding of LLMs. However, existing methods rely on full fine-tuning, which is computationally expensive and limits scalability. Parameter-efficient fine-tuning (PEFT) offers a more practical alternative but faces two critical challenges in unifying S&R: (1) gradient conflicts across tasks due to divergent optimization objectives, and (2) shifts in user intent understanding caused by overfitting to fine-tuning data, which distort general-domain knowledge and weaken LLM reasoning. To address the above issues, we propose Gradient Multi-Subspace Tuning (GEMS), a novel framework that unifies S&R with LLMs while alleviating gradient conflicts and preserving general-domain knowledge. GEMS introduces (1) \textbf{Multi-Subspace Decomposition}, which disentangles shared and task-specific optimization signals into complementary low-rank subspaces, thereby reducing destructive gradient interference, and (2) \textbf{Null-Space Projection}, which constrains parameter updates to a subspace orthogonal to the general-domain knowledge space, mitigating shifts in user intent understanding. Extensive experiments on benchmark datasets show that GEMS consistently outperforms the state-of-the-art baselines across both search and recommendation tasks, achieving superior effectiveness. 

---
# Dialogue Telemetry: Turn-Level Instrumentation for Autonomous Information Gathering 

**Authors**: Dimitris Panagopoulos, Adolfo Perrusquia, Weisi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2601.09570)  

**Abstract**: Autonomous systems conducting schema-grounded information-gathering dialogues face an instrumentation gap, lacking turn-level observables for monitoring acquisition efficiency and detecting when questioning becomes unproductive. We introduce Dialogue Telemetry (DT), a measurement framework that produces two model-agnostic signals after each question-answer exchange: (i) a Progress Estimator (PE) quantifying residual information potential per category (with a bits-based variant), and (ii) a Stalling Index (SI) detecting an observable failure signature characterized by repeated category probing with semantically similar, low-marginal-gain responses. SI flags this pattern without requiring causal diagnosis, supporting monitoring in settings where attributing degradation to specific causes may be impractical. We validate DT in controlled search-and-rescue (SAR)-inspired interviews using large language model (LLM)-based simulations, distinguishing efficient from stalled dialogue traces and illustrating downstream utility by integrating DT signals into a reinforcement learning (RL) policy. Across these settings, DT provides interpretable turn-level instrumentation that improves policy performance when stalling carries operational costs. 

---
# LLMs Got Rhythm? Hybrid Phonological Filtering for Greek Poetry Rhyme Detection and Generation 

**Authors**: Stergios Chatzikyriakidis  

**Link**: [PDF](https://arxiv.org/pdf/2601.09631)  

**Abstract**: Large Language Models (LLMs), despite their remarkable capabilities across NLP tasks, struggle with phonologically-grounded phenomena like rhyme detection and generation. This is even more evident in lower-resource languages such as Modern Greek. In this paper, we present a hybrid system that combines LLMs with deterministic phonological algorithms to achieve accurate rhyme identification/analysis and generation. Our approach implements a comprehensive taxonomy of Greek rhyme types, including Pure, Rich, Imperfect, Mosaic, and Identical Pre-rhyme Vowel (IDV) patterns, and employs an agentic generation pipeline with phonological verification. We evaluate multiple prompting strategies (zero-shot, few-shot, Chain-of-Thought, and RAG-augmented) across several LLMs including Claude 3.7 and 4.5, GPT-4o, Gemini 2.0 and open-weight models like Llama 3.1 8B and 70B and Mistral Large. Results reveal a significant "Reasoning Gap": while native-like models (Claude 3.7) perform intuitively (40\% accuracy in identification), reasoning-heavy models (Claude 4.5) achieve state-of-the-art performance (54\%) only when prompted with Chain-of-Thought. Most critically, pure LLM generation fails catastrophically (under 4\% valid poems), while our hybrid verification loop restores performance to 73.1\%. We release our system and a crucial, rigorously cleaned corpus of 40,000+ rhymes, derived from the Anemoskala and Interwar Poetry corpora, to support future research. 

---
# Structured Knowledge Representation through Contextual Pages for Retrieval-Augmented Generation 

**Authors**: Xinze Li, Zhenghao Liu, Haidong Xin, Yukun Yan, Shuo Wang, Zheni Zeng, Sen Mei, Ge Yu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.09402)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by incorporating external knowledge. Recently, some works have incorporated iterative knowledge accumulation processes into RAG models to progressively accumulate and refine query-related knowledge, thereby constructing more comprehensive knowledge representations. However, these iterative processes often lack a coherent organizational structure, which limits the construction of more comprehensive and cohesive knowledge representations. To address this, we propose PAGER, a page-driven autonomous knowledge representation framework for RAG. PAGER first prompts an LLM to construct a structured cognitive outline for a given question, which consists of multiple slots representing a distinct knowledge aspect. Then, PAGER iteratively retrieves and refines relevant documents to populate each slot, ultimately constructing a coherent page that serves as contextual input for guiding answer generation. Experiments on multiple knowledge-intensive benchmarks and backbone models show that PAGER consistently outperforms all RAG baselines. Further analyses demonstrate that PAGER constructs higher-quality and information-dense knowledge representations, better mitigates knowledge conflicts, and enables LLMs to leverage external knowledge more effectively. All code is available at this https URL. 

---
# The Imperfective Paradox in Large Language Models 

**Authors**: Bolei Ma, Yusuke Miyao  

**Link**: [PDF](https://arxiv.org/pdf/2601.09373)  

**Abstract**: Do Large Language Models (LLMs) genuinely grasp the compositional semantics of events, or do they rely on surface-level probabilistic heuristics? We investigate the Imperfective Paradox, a logical phenomenon where the past progressive aspect entails event realization for activities (e.g., running $\to$ ran) but not for accomplishments (e.g., building $\nrightarrow$ built). We introduce ImperfectiveNLI, a diagnostic dataset designed to probe this distinction across diverse semantic classes. Evaluating state-of-the-art open-weight models, we uncover a pervasive Teleological Bias: models systematically hallucinate completion for goal-oriented events, often overriding explicit textual negation. Representational analyses show that while internal embeddings often distinguish process from result, inference decisions are dominated by strong priors about goal attainment. We further find that prompting-based interventions reduce hallucinated completions but also increase incorrect rejections of valid entailments. Our findings suggest that current LLMs lack structural aspectual awareness, operating as predictive narrative engines rather than faithful logical reasoners. 

---
# When to Invoke: Refining LLM Fairness with Toxicity Assessment 

**Authors**: Jing Ren, Bowen Li, Ziqi Xu, Renqiang Luo, Shuo Yu, Xin Ye, Haytham Fayek, Xiaodong Li, Feng Xia  

**Link**: [PDF](https://arxiv.org/pdf/2601.09250)  

**Abstract**: Large Language Models (LLMs) are increasingly used for toxicity assessment in online moderation systems, where fairness across demographic groups is essential for equitable treatment. However, LLMs often produce inconsistent toxicity judgements for subtle expressions, particularly those involving implicit hate speech, revealing underlying biases that are difficult to correct through standard training. This raises a key question that existing approaches often overlook: when should corrective mechanisms be invoked to ensure fair and reliable assessments? To address this, we propose FairToT, an inference-time framework that enhances LLM fairness through prompt-guided toxicity assessment. FairToT identifies cases where demographic-related variation is likely to occur and determines when additional assessment should be applied. In addition, we introduce two interpretable fairness indicators that detect such cases and improve inference consistency without modifying model parameters. Experiments on benchmark datasets show that FairToT reduces group-level disparities while maintaining stable and reliable toxicity predictions, demonstrating that inference-time refinement offers an effective and practical approach for fairness improvement in LLM-based toxicity assessment systems. The source code can be found at this https URL. 

---
# Relation Extraction Capabilities of LLMs on Clinical Text: A Bilingual Evaluation for English and Turkish 

**Authors**: Aidana Aidynkyzy, Oğuz Dikenelli, Oylum Alatlı, Şebnem Bora  

**Link**: [PDF](https://arxiv.org/pdf/2601.09367)  

**Abstract**: The scarcity of annotated datasets for clinical information extraction in non-English languages hinders the evaluation of large language model (LLM)-based methods developed primarily in English. In this study, we present the first comprehensive bilingual evaluation of LLMs for the clinical Relation Extraction (RE) task in both English and Turkish. To facilitate this evaluation, we introduce the first English-Turkish parallel clinical RE dataset, derived and carefully curated from the 2010 i2b2/VA relation classification corpus. We systematically assess a diverse set of prompting strategies, including multiple in-context learning (ICL) and Chain-of-Thought (CoT) approaches, and compare their performance to fine-tuned baselines such as PURE. Furthermore, we propose Relation-Aware Retrieval (RAR), a novel in-context example selection method based on contrastive learning, that is specifically designed to capture both sentence-level and relation-level semantics. Our results show that prompting-based LLM approaches consistently outperform traditional fine-tuned models. Moreover, evaluations for English performed better than their Turkish counterparts across all evaluated LLMs and prompting techniques. Among ICL methods, RAR achieves the highest performance, with Gemini 1.5 Flash reaching a micro-F1 score of 0.906 in English and 0.888 in Turkish. Performance further improves to 0.918 F1 in English when RAR is combined with a structured reasoning prompt using the DeepSeek-V3 model. These findings highlight the importance of high-quality demonstration retrieval and underscore the potential of advanced retrieval and prompting techniques to bridge resource gaps in clinical natural language processing. 

---
# When to Trust: A Causality-Aware Calibration Framework for Accurate Knowledge Graph Retrieval-Augmented Generation 

**Authors**: Jing Ren, Bowen Li, Ziqi Xu, Xinkun Zhang, Haytham Fayek, Xiaodong Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.09241)  

**Abstract**: Knowledge Graph Retrieval-Augmented Generation (KG-RAG) extends the RAG paradigm by incorporating structured knowledge from knowledge graphs, enabling Large Language Models (LLMs) to perform more precise and explainable reasoning. While KG-RAG improves factual accuracy in complex tasks, existing KG-RAG models are often severely overconfident, producing high-confidence predictions even when retrieved sub-graphs are incomplete or unreliable, which raises concerns for deployment in high-stakes domains. To address this issue, we propose Ca2KG, a Causality-aware Calibration framework for KG-RAG. Ca2KG integrates counterfactual prompting, which exposes retrieval-dependent uncertainties in knowledge quality and reasoning reliability, with a panel-based re-scoring mechanism that stabilises predictions across interventions. Extensive experiments on two complex QA datasets demonstrate that Ca2KG consistently improves calibration while maintaining or even enhancing predictive accuracy. 

---
# Contrastive Bi-Encoder Models for Multi-Label Skill Extraction: Enhancing ESCO Ontology Matching with BERT and Attention Mechanisms 

**Authors**: Yongming Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.09119)  

**Abstract**: Fine-grained labor market analysis increasingly relies on mapping unstructured job advertisements to standardized skill taxonomies such as ESCO. This mapping is naturally formulated as an Extreme Multi-Label Classification (XMLC) problem, but supervised solutions are constrained by the scarcity and cost of large-scale, taxonomy-aligned annotations--especially in non-English settings where job-ad language diverges substantially from formal skill definitions. We propose a zero-shot skill extraction framework that eliminates the need for manually labeled job-ad training data. The framework uses a Large Language Model (LLM) to synthesize training instances from ESCO definitions, and introduces hierarchically constrained multi-skill generation based on ESCO Level-2 categories to improve semantic coherence in multi-label contexts. On top of the synthetic corpus, we train a contrastive bi-encoder that aligns job-ad sentences with ESCO skill descriptions in a shared embedding space; the encoder augments a BERT backbone with BiLSTM and attention pooling to better model long, information-dense requirement statements. An upstream RoBERTa-based binary filter removes non-skill sentences to improve end-to-end precision. Experiments show that (i) hierarchy-conditioned generation improves both fluency and discriminability relative to unconstrained pairing, and (ii) the resulting multi-label model transfers effectively to real-world Chinese job advertisements, achieving strong zero-shot retrieval performance (F1@5 = 0.72) and outperforming TF--IDF and standard BERT baselines. Overall, the proposed pipeline provides a scalable, data-efficient pathway for automated skill coding in labor economics and workforce analytics. 

---
# Multicultural Spyfall: Assessing LLMs through Dynamic Multilingual Social Deduction Game 

**Authors**: Haryo Akbarianto Wibowo, Alaa Elsetohy, Qinrong Cui, Alham Fikri Aji  

**Link**: [PDF](https://arxiv.org/pdf/2601.09017)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has necessitated more robust evaluation methods that go beyond static benchmarks, which are increasingly prone to data saturation and leakage. In this paper, we propose a dynamic benchmarking framework for evaluating multilingual and multicultural capabilities through the social deduction game Spyfall. In our setup, models must engage in strategic dialogue to either identify a secret agent or avoid detection, utilizing culturally relevant locations or local foods. Our results show that our game-based rankings align closely with the Chatbot Arena. However, we find a significant performance gap in non-English contexts: models are generally less proficient when handling locally specific entities and often struggle with rule-following or strategic integrity in non-English languages. We demonstrate that this game-based approach provides a scalable, leakage-resistant, and culturally nuanced alternative to traditional NLP benchmarks. The game history can be accessed here this https URL. 

---
# Entropy Sentinel: Continuous LLM Accuracy Monitoring from Decoding Entropy Traces in STEM 

**Authors**: Pedro Memoli Buffa, Luciano Del Corro  

**Link**: [PDF](https://arxiv.org/pdf/2601.09001)  

**Abstract**: Deploying LLMs raises two coupled challenges: (1) monitoring - estimating where a model underperforms as traffic and domains drift - and (2) improvement - prioritizing data acquisition to close the largest performance gaps. We test whether an inference-time signal can estimate slice-level accuracy under domain shift. For each response, we compute an output-entropy profile from final-layer next-token probabilities (from top-k logprobs) and summarize it with eleven statistics. A lightweight classifier predicts instance correctness, and averaging predicted probabilities yields a domain-level accuracy estimate. We evaluate on ten STEM reasoning benchmarks with exhaustive train/test compositions (k in {1,2,3,4}; all "10 choose k" combinations), across nine LLMs from six families (3B-20B). Estimates often track held-out benchmark accuracy, and several models show near-monotonic ordering of domains. Output-entropy profiles are thus an accessible signal for scalable monitoring and for targeting data acquisition. 

---
# Gaming the Answer Matcher: Examining the Impact of Text Manipulation on Automated Judgment 

**Authors**: Manas Khatore, Sumana Sridharan, Kevork Sulahian, Benjamin J. Smith, Shi Feng  

**Link**: [PDF](https://arxiv.org/pdf/2601.08849)  

**Abstract**: Automated answer matching, which leverages LLMs to evaluate free-text responses by comparing them to a reference answer, shows substantial promise as a scalable and aligned alternative to human evaluation. However, its reliability requires robustness against strategic attacks such as guesswork or verbosity that may artificially inflate scores without improving actual correctness. In this work, we systematically investigate whether such tactics deceive answer matching models by prompting examinee models to: (1) generate verbose responses, (2) provide multiple answers when unconfident, and (3) embed conflicting answers with the correct answer near the start of their response. Our results show that these manipulations do not increase scores and often reduce them. Additionally, binary scoring (which requires a matcher to answer with a definitive "correct" or "incorrect") is more robust to attacks than continuous scoring (which requires a matcher to determine partial correctness). These findings show that answer matching is generally robust to inexpensive text manipulation and is a viable alternative to traditional LLM-as-a-judge or human evaluation when reference answers are available. 

---
# Rubric-Conditioned LLM Grading: Alignment, Uncertainty, and Robustness 

**Authors**: Haotian Deng, Chris Farber, Jiyoon Lee, David Tang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08843)  

**Abstract**: Automated short-answer grading (ASAG) remains a challenging task due to the linguistic variability of student responses and the need for nuanced, rubric-aligned partial credit. While Large Language Models (LLMs) offer a promising solution, their reliability as automated judges in rubric-based settings requires rigorous assessment. In this paper, we systematically evaluate the performance of LLM-judges for rubric-based short-answer grading. We investigate three key aspects: the alignment of LLM grading with expert judgment across varying rubric complexities, the trade-off between uncertainty and accuracy facilitated by a consensus-based deferral mechanism, and the model's robustness under random input perturbations and adversarial attacks. Using the SciEntsBank benchmark and Qwen 2.5-72B, we find that alignment is strong for binary tasks but degrades with increased rubric granularity. Our "Trust Curve" analysis demonstrates a clear trade-off where filtering low-confidence predictions improves accuracy on the remaining subset. Additionally, robustness experiments reveal that while the model is resilient to prompt injection, it is sensitive to synonym substitutions. Our work provides critical insights into the capabilities and limitations of rubric-conditioned LLM judges, highlighting the importance of uncertainty estimation and robustness testing for reliable deployment. 

---
# Recursive Knowledge Synthesis for Multi-LLM Systems: Stability Analysis and Tri-Agent Audit Framework 

**Authors**: Toshiyuki Shigemura  

**Link**: [PDF](https://arxiv.org/pdf/2601.08839)  

**Abstract**: This paper presents a tri-agent cross-validation framework for analyzing stability and explainability in multi-model large language systems. The architecture integrates three heterogeneous LLMs-used for semantic generation, analytical consistency checking, and transparency auditing-into a recursive interaction cycle. This design induces Recursive Knowledge Synthesis (RKS), where intermediate representations are continuously refined through mutually constraining transformations irreducible to single-model behavior. Across 47 controlled trials using public-access LLM deployments (October 2025), we evaluated system stability via four metrics: Reflex Reliability Score (RRS), Transparency Score (TS), Deviation Detection Rate (DDR), and Correction Success Rate (CSR). The system achieved mean RRS = 0.78+-0.06 and maintained TS >= 0.8 in about 68% of trials. Approximately 89% of trials converged, supporting the theoretical prediction that transparency auditing acts as a contraction operator within the composite validation mapping. The contributions are threefold: (1) a structured tri-agent framework for coordinated reasoning across heterogeneous LLMs, (2) a formal RKS model grounded in fixed-point theory, and (3) empirical evaluation of inter-model stability under realistic, non-API public-access conditions. These results provide initial empirical evidence that a safety-preserving, humansupervised multi-LLM architecture can achieve stable recursive knowledge synthesis in realistic, publicly deployed environments. 

---
# EvasionBench: Detecting Evasive Answers in Financial Q&A via Multi-Model Consensus and LLM-as-Judge 

**Authors**: Shijian Ma, Yan Lin, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.09142)  

**Abstract**: Detecting evasive answers in earnings calls is critical for financial transparency, yet progress is hindered by the lack of large-scale benchmarks. We introduce EvasionBench, comprising 30,000 training samples and 1,000 human-annotated test samples (Cohen's Kappa 0.835) across three evasion levels. Our key contribution is a multi-model annotation framework leveraging a core insight: disagreement between frontier LLMs signals hard examples most valuable for training. We mine boundary cases where two strong annotators conflict, using a judge to resolve labels. This approach outperforms single-model distillation by 2.4 percent, with judge-resolved samples improving generalization despite higher training loss (0.421 vs 0.393) - evidence that disagreement mining acts as implicit regularization. Our trained model Eva-4B (4B parameters) achieves 81.3 percent accuracy, outperforming its base by 25 percentage points and approaching frontier LLM performance at a fraction of inference cost. 

---
