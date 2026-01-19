# Exploring LLM Features in Predictive Process Monitoring for Small-Scale Event-Logs 

**Authors**: Alessandro Padella, Massimiliano de Leoni, Marlon Dumas  

**Link**: [PDF](https://arxiv.org/pdf/2601.11468)  

**Abstract**: Predictive Process Monitoring is a branch of process mining that aims to predict the outcome of an ongoing process. Recently, it leveraged machine-and-deep learning architectures. In this paper, we extend our prior LLM-based Predictive Process Monitoring framework, which was initially focused on total time prediction via prompting. The extension consists of comprehensively evaluating its generality, semantic leverage, and reasoning mechanisms, also across multiple Key Performance Indicators. Empirical evaluations conducted on three distinct event logs and across the Key Performance Indicators of Total Time and Activity Occurrence prediction indicate that, in data-scarce settings with only 100 traces, the LLM surpasses the benchmark methods. Furthermore, the experiments also show that the LLM exploits both its embodied prior knowledge and the internal correlations among training traces. Finally, we examine the reasoning strategies employed by the model, demonstrating that the LLM does not merely replicate existing predictive methods but performs higher-order reasoning to generate the predictions. 

---
# Beyond Model Scaling: Test-Time Intervention for Efficient Deep Reasoning 

**Authors**: Qianyue Wang, Jinwu Hu, Yufeng Wang, Huanxiang Lin, Bolin Chen, Zhiquan Wen, Yaofo Chen, Mingkui Tan  

**Link**: [PDF](https://arxiv.org/pdf/2601.11252)  

**Abstract**: Large Reasoning Models (LRMs) excel at multi-step reasoning but often suffer from inefficient reasoning processes like overthinking and overshoot, where excessive or misdirected reasoning increases computational cost and degrades performance. Existing efficient reasoning methods operate in a closed-loop manner, lacking mechanisms for external intervention to guide the reasoning process. To address this, we propose Think-with-Me, a novel test-time interactive reasoning paradigm that introduces external feedback intervention into the reasoning process. Our key insights are that transitional conjunctions serve as natural points for intervention, signaling phases of self-validation or exploration and using transitional words appropriately to prolong the reasoning enhances performance, while excessive use affects performance. Building on these insights, Think-with-Me pauses reasoning at these points for external feedback, adaptively extending or terminating reasoning to reduce redundancy while preserving accuracy. The feedback is generated via a multi-criteria evaluation (rationality and completeness) and comes from either human or LLM proxies. We train the target model using Group Relative Policy Optimization (GRPO) to adapt to this interactive mode. Experiments show that Think-with-Me achieves a superior balance between accuracy and reasoning length under limited context windows. On AIME24, Think-with-Me outperforms QwQ-32B by 7.19% in accuracy while reducing average reasoning length by 81% under an 8K window. The paradigm also benefits security and creative tasks. 

---
# Health Facility Location in Ethiopia: Leveraging LLMs to Integrate Expert Knowledge into Algorithmic Planning 

**Authors**: Yohai Trabelsi, Guojun Xiong, Fentabil Getnet, Stéphane Verguet, Milind Tambe  

**Link**: [PDF](https://arxiv.org/pdf/2601.11479)  

**Abstract**: Ethiopia's Ministry of Health is upgrading health posts to improve access to essential services, particularly in rural areas. Limited resources, however, require careful prioritization of which facilities to upgrade to maximize population coverage while accounting for diverse expert and stakeholder preferences. In collaboration with the Ethiopian Public Health Institute and Ministry of Health, we propose a hybrid framework that systematically integrates expert knowledge with optimization techniques. Classical optimization methods provide theoretical guarantees but require explicit, quantitative objectives, whereas stakeholder criteria are often articulated in natural language and difficult to formalize. To bridge these domains, we develop the Large language model and Extended Greedy (LEG) framework. Our framework combines a provable approximation algorithm for population coverage optimization with LLM-driven iterative refinement that incorporates human-AI alignment to ensure solutions reflect expert qualitative guidance while preserving coverage guarantees. Experiments on real-world data from three Ethiopian regions demonstrate the framework's effectiveness and its potential to inform equitable, data-driven health system planning. 

---
# Do explanations generalize across large reasoning models? 

**Authors**: Koyena Pal, David Bau, Chandan Singh  

**Link**: [PDF](https://arxiv.org/pdf/2601.11517)  

**Abstract**: Large reasoning models (LRMs) produce a textual chain of thought (CoT) in the process of solving a problem, which serves as a potentially powerful tool to understand the problem by surfacing a human-readable, natural-language explanation. However, it is unclear whether these explanations generalize, i.e. whether they capture general patterns about the underlying problem rather than patterns which are esoteric to the LRM. This is a crucial question in understanding or discovering new concepts, e.g. in AI for science. We study this generalization question by evaluating a specific notion of generalizability: whether explanations produced by one LRM induce the same behavior when given to other LRMs. We find that CoT explanations often exhibit this form of generalization (i.e. they increase consistency between LRMs) and that this increased generalization is correlated with human preference rankings and post-training with reinforcement learning. We further analyze the conditions under which explanations yield consistent answers and propose a straightforward, sentence-level ensembling strategy that improves consistency. Taken together, these results prescribe caution when using LRM explanations to yield new insights and outline a framework for characterizing LRM explanation generalization. 

---
# ReCreate: Reasoning and Creating Domain Agents Driven by Experience 

**Authors**: Zhezheng Hao, Hong Wang, Jian Luo, Jianqing Zhang, Yuyan Zhou, Qiang Lin, Can Wang, Hande Dong, Jiawei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.11100)  

**Abstract**: Large Language Model agents are reshaping the industrial landscape. However, most practical agents remain human-designed because tasks differ widely, making them labor-intensive to build. This situation poses a central question: can we automatically create and adapt domain agents in the wild? While several recent approaches have sought to automate agent creation, they typically treat agent generation as a black-box procedure and rely solely on final performance metrics to guide the process. Such strategies overlook critical evidence explaining why an agent succeeds or fails, and often require high computational costs. To address these limitations, we propose ReCreate, an experience-driven framework for the automatic creation of domain agents. ReCreate systematically leverages agent interaction histories, which provide rich concrete signals on both the causes of success or failure and the avenues for improvement. Specifically, we introduce an agent-as-optimizer paradigm that effectively learns from experience via three key components: (i) an experience storage and retrieval mechanism for on-demand inspection; (ii) a reasoning-creating synergy pipeline that maps execution experience into scaffold edits; and (iii) hierarchical updates that abstract instance-level details into reusable domain patterns. In experiments across diverse domains, ReCreate consistently outperforms human-designed agents and existing automated agent generation methods, even when starting from minimal seed scaffolds. 

---
# Hierarchical Orthogonal Residual Spread for Precise Massive Editing in Large Language Models 

**Authors**: Xiaojie Gu, Guangxu Chen, Yuheng Yang, Jingxin Han, Andi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.11441)  

**Abstract**: Large language models (LLMs) exhibit exceptional performance across various domains, yet they face critical safety concerns. Model editing has emerged as an effective approach to mitigate these issues. Existing model editing methods often focus on optimizing an information matrix that blends new and old knowledge. While effective, these approaches can be computationally expensive and may cause conflicts. In contrast, we shift our attention to Hierarchical Orthogonal Residual SprEad of the information matrix, which reduces noisy gradients and enables more stable edits from a different perspective. We demonstrate the effectiveness of our method HORSE through a clear theoretical comparison with several popular methods and extensive experiments conducted on two datasets across multiple LLMs. The results show that HORSE maintains precise massive editing across diverse scenarios. The code is available at this https URL 

---
# Do You Trust Me? Cognitive-Affective Signatures of Trustworthiness in Large Language Models 

**Authors**: Gerard Yeo, Svetlana Churina, Kokil Jaidka  

**Link**: [PDF](https://arxiv.org/pdf/2601.10719)  

**Abstract**: Perceived trustworthiness underpins how users navigate online information, yet it remains unclear whether large language models (LLMs),increasingly embedded in search, recommendation, and conversational systems, represent this construct in psychologically coherent ways. We analyze how instruction-tuned LLMs (Llama 3.1 8B, Qwen 2.5 7B, Mistral 7B) encode perceived trustworthiness in web-like narratives using the PEACE-Reviews dataset annotated for cognitive appraisals, emotions, and behavioral intentions. Across models, systematic layer- and head-level activation differences distinguish high- from low-trust texts, revealing that trust cues are implicitly encoded during pretraining. Probing analyses show linearly de-codable trust signals and fine-tuning effects that refine rather than restructure these representations. Strongest associations emerge with appraisals of fairness, certainty, and accountability-self -- dimensions central to human trust formation online. These findings demonstrate that modern LLMs internalize psychologically grounded trust signals without explicit supervision, offering a representational foundation for designing credible, transparent, and trust-worthy AI systems in the web ecosystem. Code and appendix are available at: this https URL. 

---
# Building AI Agents to Improve Job Referral Requests to Strangers 

**Authors**: Ross Chu, Yuting Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.10726)  

**Abstract**: This paper develops AI agents that help job seekers write effective requests for job referrals in a professional online community. The basic workflow consists of an improver agent that rewrites the referral request and an evaluator agent that measures the quality of revisions using a model trained to predict the probability of receiving referrals from other users. Revisions suggested by the LLM (large language model) increase predicted success rates for weaker requests while reducing them for stronger requests. Enhancing the LLM with Retrieval-Augmented Generation (RAG) prevents edits that worsen stronger requests while it amplifies improvements for weaker requests. Overall, using LLM revisions with RAG increases the predicted success rate for weaker requests by 14\% without degrading performance on stronger requests. Although improvements in model-predicted success do not guarantee more referrals in the real world, they provide low-cost signals for promising features before running higher-stakes experiments on real users. 

---
# Relational Linearity is a Predictor of Hallucinations 

**Authors**: Yuetian Lu, Yihong Liu, Hinrich Schütze  

**Link**: [PDF](https://arxiv.org/pdf/2601.11429)  

**Abstract**: Hallucination is a central failure mode in large language models (LLMs). We focus on hallucinations of answers to questions like: "Which instrument did Glenn Gould play?", but we ask these questions for synthetic entities that are unknown to the model. Surprisingly, we find that medium-size models like Gemma-7B-IT frequently hallucinate, i.e., they have difficulty recognizing that the hallucinated fact is not part of their knowledge. We hypothesize that an important factor in causing these hallucinations is the linearity of the relation: linear relations tend to be stored more abstractly, making it difficult for the LLM to assess its knowledge; the facts of nonlinear relations tend to be stored more directly, making knowledge assessment easier. To investigate this hypothesis, we create SyntHal, a dataset of 6000 synthetic entities for six relations. In our experiments with four models, we determine, for each relation, the hallucination rate on SyntHal and also measure its linearity, using $\Delta\cos$. We find a strong correlation ($r \in [.78,.82]$) between relational linearity and hallucination rate, providing evidence for our hypothesis that the underlying storage of triples of a relation is a factor in how well a model can self-assess its knowledge. This finding has implications for how to manage hallucination behavior and suggests new research directions for improving the representation of factual knowledge in LLMs. 

---
# Evaluating LLM Behavior in Hiring: Implicit Weights, Fairness Across Groups, and Alignment with Human Preferences 

**Authors**: Morgane Hoffmann, Emma Jouffroy, Warren Jouanneau, Marc Palyart, Charles Pebereau  

**Link**: [PDF](https://arxiv.org/pdf/2601.11379)  

**Abstract**: General-purpose Large Language Models (LLMs) show significant potential in recruitment applications, where decisions require reasoning over unstructured text, balancing multiple criteria, and inferring fit and competence from indirect productivity signals. Yet, it is still uncertain how LLMs assign importance to each attribute and whether such assignments are in line with economic principles, recruiter preferences or broader societal norms. We propose a framework to evaluate an LLM's decision logic in recruitment, by drawing on established economic methodologies for analyzing human hiring behavior. We build synthetic datasets from real freelancer profiles and project descriptions from a major European online freelance marketplace and apply a full factorial design to estimate how a LLM weighs different match-relevant criteria when evaluating freelancer-project fit. We identify which attributes the LLM prioritizes and analyze how these weights vary across project contexts and demographic subgroups. Finally, we explain how a comparable experimental setup could be implemented with human recruiters to assess alignment between model and human decisions. Our findings reveal that the LLM weighs core productivity signals, such as skills and experience, but interprets certain features beyond their explicit matching value. While showing minimal average discrimination against minority groups, intersectional effects reveal that productivity signals carry different weights between demographic groups. 

---
# How Much Would a Clinician Edit This Draft? Evaluating LLM Alignment for Patient Message Response Drafting 

**Authors**: Parker Seegmiller, Joseph Gatto, Sarah E. Greer, Ganza Belise Isingizwe, Rohan Ray, Timothy E. Burdick, Sarah Masud Preum  

**Link**: [PDF](https://arxiv.org/pdf/2601.11344)  

**Abstract**: Large language models (LLMs) show promise in drafting responses to patient portal messages, yet their integration into clinical workflows raises various concerns, including whether they would actually save clinicians time and effort in their portal workload. We investigate LLM alignment with individual clinicians through a comprehensive evaluation of the patient message response drafting task. We develop a novel taxonomy of thematic elements in clinician responses and propose a novel evaluation framework for assessing clinician editing load of LLM-drafted responses at both content and theme levels. We release an expert-annotated dataset and conduct large-scale evaluations of local and commercial LLMs using various adaptation techniques including thematic prompting, retrieval-augmented generation, supervised fine-tuning, and direct preference optimization. Our results reveal substantial epistemic uncertainty in aligning LLM drafts with clinician responses. While LLMs demonstrate capability in drafting certain thematic elements, they struggle with clinician-aligned generation in other themes, particularly question asking to elicit further information from patients. Theme-driven adaptation strategies yield improvements across most themes. Our findings underscore the necessity of adapting LLMs to individual clinician preferences to enable reliable and responsible use in patient-clinician communication workflows. 

---
# FAQ: Mitigating Quantization Error via Regenerating Calibration Data with Family-Aware Quantization 

**Authors**: Haiyang Xiao, Weiqing Li, Jinyue Guo, Guochao Jiang, Guohua Liu, Yuewei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.11200)  

**Abstract**: Although post-training quantization (PTQ) provides an efficient numerical compression scheme for deploying large language models (LLMs) on resource-constrained devices, the representativeness and universality of calibration data remain a core bottleneck in determining the accuracy of quantization parameters. Traditional PTQ methods typically rely on limited samples, making it difficult to capture the activation distribution during the inference phase, leading to biases in quantization parameters. To address this, we propose \textbf{FAQ} (Family-Aware Quantization), a calibration data regeneration framework that leverages prior knowledge from LLMs of the same family to generate high-fidelity calibration samples. Specifically, FAQ first inputs the original calibration samples into a larger LLM from the same family as the target model, regenerating a series of high-fidelity calibration data using a highly consistent knowledge system. Subsequently, this data, carrying Chain-of-Thought reasoning and conforming to the expected activation distribution, undergoes group competition under expert guidance to select the best samples, which are then re-normalized to enhance the effectiveness of standard PTQ. Experiments on multiple model series, including Qwen3-8B, show that FAQ reduces accuracy loss by up to 28.5\% compared to the baseline with original calibration data, demonstrating its powerful potential and contribution. 

---
# Learn Before Represent: Bridging Generative and Contrastive Learning for Domain-Specific LLM Embeddings 

**Authors**: Xiaoyu Liang, Yuchen Peng, Jiale Luo, Wenhao Wang, Haoji Hu, Xincheng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.11124)  

**Abstract**: Large Language Models (LLMs) adapted via contrastive learning excel in general representation learning but struggle in vertical domains like chemistry and law, primarily due to a lack of domain-specific knowledge. This work identifies a core bottleneck: the prevailing ``LLM+CL'' paradigm focuses on semantic alignment but cannot perform knowledge acquisition, leading to failures on specialized terminology. To bridge this gap, we propose Learn Before Represent (LBR), a novel two-stage framework. LBR first injects domain knowledge via an Information Bottleneck-Constrained Generative Learning stage, preserving the LLM's causal attention to maximize knowledge acquisition while compressing semantics. It then performs Generative-Refined Contrastive Learning on the compressed representations for alignment. This approach maintains architectural consistency and resolves the objective conflict between generative and contrastive learning. Extensive experiments on medical, chemistry, and code retrieval tasks show that LBR significantly outperforms strong baselines. Our work establishes a new paradigm for building accurate and robust representations in vertical domains. 

---
# Deep GraphRAG: A Balanced Approach to Hierarchical Retrieval and Adaptive Integration 

**Authors**: Yuejie Li, Ke Yang, Tao Wang, Bolin Chen, Bowen Li, Chengjun Mao  

**Link**: [PDF](https://arxiv.org/pdf/2601.11144)  

**Abstract**: Graph-based Retrieval-Augmented Generation (GraphRAG) frameworks face a trade-off between the comprehensiveness of global search and the efficiency of local search. Existing methods are often challenged by navigating large-scale hierarchical graphs, optimizing retrieval paths, and balancing exploration-exploitation dynamics, frequently lacking robust multi-stage re-ranking. To overcome these deficits, we propose Deep GraphRAG, a framework designed for a balanced approach to hierarchical retrieval and adaptive integration. It introduces a hierarchical global-to-local retrieval strategy that integrates macroscopic inter-community and microscopic intra-community contextual relations. This strategy employs a three-stage process: (1) inter-community filtering, which prunes the search space using local context; (2) community-level refinement, which prioritizes relevant subgraphs via entity-interaction analysis; and (3) entity-level fine-grained search within target communities. A beam search-optimized dynamic re-ranking module guides this process, continuously filtering candidates to balance efficiency and global comprehensiveness. Deep GraphRAG also features a Knowledge Integration Module leveraging a compact LLM, trained with Dynamic Weighting Reward GRPO (DW-GRPO). This novel reinforcement learning approach dynamically adjusts reward weights to balance three key objectives: relevance, faithfulness, and conciseness. This training enables compact models (1.5B) to approach the performance of large models (70B) in the integration task. Evaluations on Natural Questions and HotpotQA demonstrate that Deep GraphRAG significantly outperforms baseline graph retrieval methods in both accuracy and efficiency. 

---
# Predicting Biased Human Decision-Making with Large Language Models in Conversational Settings 

**Authors**: Stephen Pilli, Vivek Nallur  

**Link**: [PDF](https://arxiv.org/pdf/2601.11049)  

**Abstract**: We examine whether large language models (LLMs) can predict biased decision-making in conversational settings, and whether their predictions capture not only human cognitive biases but also how those effects change under cognitive load. In a pre-registered study (N = 1,648), participants completed six classic decision-making tasks via a chatbot with dialogues of varying complexity. Participants exhibited two well-documented cognitive biases: the Framing Effect and the Status Quo Bias. Increased dialogue complexity resulted in participants reporting higher mental demand. This increase in cognitive load selectively, but significantly, increased the effect of the biases, demonstrating the load-bias interaction. We then evaluated whether LLMs (GPT-4, GPT-5, and open-source models) could predict individual decisions given demographic information and prior dialogue. While results were mixed across choice problems, LLM predictions that incorporated dialogue context were significantly more accurate in several key scenarios. Importantly, their predictions reproduced the same bias patterns and load-bias interactions observed in humans. Across all models tested, the GPT-4 family consistently aligned with human behavior, outperforming GPT-5 and open-source models in both predictive accuracy and fidelity to human-like bias patterns. These findings advance our understanding of LLMs as tools for simulating human decision-making and inform the design of conversational agents that adapt to user biases. 

---
# FactCorrector: A Graph-Inspired Approach to Long-Form Factuality Correction of Large Language Models 

**Authors**: Javier Carnerero-Cano, Massimiliano Pronesti, Radu Marinescu, Tigran Tchrakian, James Barry, Jasmina Gajcin, Yufang Hou, Alessandra Pascale, Elizabeth Daly  

**Link**: [PDF](https://arxiv.org/pdf/2601.11232)  

**Abstract**: Large language models (LLMs) are widely used in knowledge-intensive applications but often generate factually incorrect responses. A promising approach to rectify these flaws is correcting LLMs using feedback. Therefore, in this paper, we introduce FactCorrector, a new post-hoc correction method that adapts across domains without retraining and leverages structured feedback about the factuality of the original response to generate a correction. To support rigorous evaluations of factuality correction methods, we also develop the VELI5 benchmark, a novel dataset containing systematically injected factual errors and ground-truth corrections. Experiments on VELI5 and several popular long-form factuality datasets show that the FactCorrector approach significantly improves factual precision while preserving relevance, outperforming strong baselines. We release our code at this https URL. 

---
# Steering Language Models Before They Speak: Logit-Level Interventions 

**Authors**: Hyeseon An, Shinwoo Park, Hyundong Jin, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2601.10960)  

**Abstract**: Steering LLMs is essential for specialized applications such as style-sensitive text rewriting, user-adaptive communication, and toxicity mitigation. Current steering methods, such as prompting-based and activation-based approaches, are widely used to guide model behavior. However, activation-based techniques require deep access to internal layers, while prompting-based steering often fails to provide consistent or fine-grained control. In order to address these limitations, we propose a training-free inference-time logit intervention for controllable generation. Our approach utilizes a statistical token score table derived from z-normalized log-odds of labeled corpora to shift the decoding distribution. Empirical evaluations across three diverse datasets focusing on writing complexity, formality, and toxicity demonstrate that our method effectively steers output characteristics, confirming its broad applicability and task-agnostic nature. Our results show that statistically grounded logit steering can achieve large, consistent, and multi-task control gains: up to +47%p accuracy and 50x f1 improvement. 

---
# H-AIM: Orchestrating LLMs, PDDL, and Behavior Trees for Hierarchical Multi-Robot Planning 

**Authors**: Haishan Zeng, Peng Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.11063)  

**Abstract**: In embodied artificial intelligence, enabling heterogeneous robot teams to execute long-horizon tasks from high-level instructions remains a critical challenge. While large language models (LLMs) show promise in instruction parsing and preliminary planning, they exhibit limitations in long-term reasoning and dynamic multi-robot coordination. We propose Hierarchical Autonomous Intelligent Multi-Robot Planning(H-AIM), a novel embodied multi-robot task planning framework that addresses these issues through a three-stage cascaded architecture: 1) It leverages an LLM to parse instructions and generate Planning Domain Definition Language (PDDL) problem descriptions, thereby transforming commands into formal planning problems; 2) It combines the semantic reasoning of LLMs with the search capabilities of a classical planner to produce optimized action sequences; 3) It compiles the resulting plan into behavior trees for reactive control. The framework supports dynamically sized heterogeneous robot teams via a shared blackboard mechanism for communication and state synchronization. To validate our approach, we introduce the MACE-THOR benchmark dataset, comprising 42 complex tasks across 8 distinct household layouts. Experimental results demonstrate that H-AIM achieves a remarkable performance improvement, elevating the task success rate from 12% to 55% and boosting the goal condition recall from 32% to 72% against the strongest baseline, LaMMA-P. 

---
# Finding the Translation Switch: Discovering and Exploiting the Task-Initiation Features in LLMs 

**Authors**: Xinwei Wu, Heng Liu, Xiaohu Zhao, Yuqi Ren, Linlong Xu, Longyue Wang, Deyi Xiong, Weihua Luo, Kaifu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.11019)  

**Abstract**: Large Language Models (LLMs) frequently exhibit strong translation abilities, even without task-specific fine-tuning. However, the internal mechanisms governing this innate capability remain largely opaque. To demystify this process, we leverage Sparse Autoencoders (SAEs) and introduce a novel framework for identifying task-specific features. Our method first recalls features that are frequently co-activated on translation inputs and then filters them for functional coherence using a PCA-based consistency metric. This framework successfully isolates a small set of **translation initiation** features. Causal interventions demonstrate that amplifying these features steers the model towards correct translation, while ablating them induces hallucinations and off-task outputs, confirming they represent a core component of the model's innate translation competency. Moving from analysis to application, we leverage this mechanistic insight to propose a new data selection strategy for efficient fine-tuning. Specifically, we prioritize training on **mechanistically hard** samples-those that fail to naturally activate the translation initiation features. Experiments show this approach significantly improves data efficiency and suppresses hallucinations. Furthermore, we find these mechanisms are transferable to larger models of the same family. Our work not only decodes a core component of the translation mechanism in LLMs but also provides a blueprint for using internal model mechanism to create more robust and efficient models. The codes are available at this https URL. 

---
# Combating Spurious Correlations in Graph Interpretability via Self-Reflection 

**Authors**: Kecheng Cai, Chenyang Xu, Chao Peng  

**Link**: [PDF](https://arxiv.org/pdf/2601.11021)  

**Abstract**: Interpretable graph learning has recently emerged as a popular research topic in machine learning. The goal is to identify the important nodes and edges of an input graph that are crucial for performing a specific graph reasoning task. A number of studies have been conducted in this area, and various benchmark datasets have been proposed to facilitate evaluation. Among them, one of the most challenging is the Spurious-Motif benchmark, introduced at ICLR 2022. The datasets in this synthetic benchmark are deliberately designed to include spurious correlations, making it particularly difficult for models to distinguish truly relevant structures from misleading patterns. As a result, existing methods exhibit significantly worse performance on this benchmark compared to others.
In this paper, we focus on improving interpretability on the challenging Spurious-Motif datasets. We demonstrate that the self-reflection technique, commonly used in large language models to tackle complex tasks, can also be effectively adapted to enhance interpretability in datasets with strong spurious correlations. Specifically, we propose a self-reflection framework that can be integrated with existing interpretable graph learning methods. When such a method produces importance scores for each node and edge, our framework feeds these predictions back into the original method to perform a second round of evaluation. This iterative process mirrors how large language models employ self-reflective prompting to reassess their previous outputs. We further analyze the reasons behind this improvement from the perspective of graph representation learning, which motivates us to propose a fine-tuning training method based on this feedback mechanism. 

---
# When Personalization Misleads: Understanding and Mitigating Hallucinations in Personalized LLMs 

**Authors**: Zhongxiang Sun, Yi Zhan, Chenglei Shen, Weijie Yu, Xiao Zhang, Ming He, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.11000)  

**Abstract**: Personalized large language models (LLMs) adapt model behavior to individual users to enhance user satisfaction, yet personalization can inadvertently distort factual reasoning. We show that when personalized LLMs face factual queries, there exists a phenomenon where the model generates answers aligned with a user's prior history rather than the objective truth, resulting in personalization-induced hallucinations that degrade factual reliability and may propagate incorrect beliefs, due to representational entanglement between personalization and factual representations. To address this issue, we propose Factuality-Preserving Personalized Steering (FPPS), a lightweight inference-time approach that mitigates personalization-induced factual distortions while preserving personalized behavior. We further introduce PFQABench, the first benchmark designed to jointly evaluate factual and personalized question answering under personalization. Experiments across multiple LLM backbones and personalization methods show that FPPS substantially improves factual accuracy while maintaining personalized performance. 

---
# Beyond Max Tokens: Stealthy Resource Amplification via Tool Calling Chains in LLM Agents 

**Authors**: Kaiyu Zhou, Yongsen Zheng, Yicheng He, Meng Xue, Xueluan Gong, Yuji Wang, Kwok-Yan Lam  

**Link**: [PDF](https://arxiv.org/pdf/2601.10955)  

**Abstract**: The agent-tool communication loop is a critical attack surface in modern Large Language Model (LLM) agents. Existing Denial-of-Service (DoS) attacks, primarily triggered via user prompts or injected retrieval-augmented generation (RAG) context, are ineffective for this new paradigm. They are fundamentally single-turn and often lack a task-oriented approach, making them conspicuous in goal-oriented workflows and unable to exploit the compounding costs of multi-turn agent-tool interactions. We introduce a stealthy, multi-turn economic DoS attack that operates at the tool layer under the guise of a correctly completed task. Our method adjusts text-visible fields and a template-governed return policy in a benign, Model Context Protocol (MCP)-compatible tool server, optimizing these edits with a Monte Carlo Tree Search (MCTS) optimizer. These adjustments leave function signatures unchanged and preserve the final payload, steering the agent into prolonged, verbose tool-calling sequences using text-only notices. This compounds costs across turns, escaping single-turn caps while keeping the final answer correct to evade validation. Across six LLMs on the ToolBench and BFCL benchmarks, our attack expands tasks into trajectories exceeding 60,000 tokens, inflates costs by up to 658x, and raises energy by 100-560x. It drives GPU KV cache occupancy from <1% to 35-74% and cuts co-running throughput by approximately 50%. Because the server remains protocol-compatible and task outcomes are correct, conventional checks fail. These results elevate the agent-tool interface to a first-class security frontier, demanding a paradigm shift from validating final answers to monitoring the economic and computational cost of the entire agentic process. 

---
# Multi-Stage Patient Role-Playing Framework for Realistic Clinical Interactions 

**Authors**: Shijie Jiang, Zefan Zhang, Kehua Zhu, Tian Bai, Ruihong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.10951)  

**Abstract**: The simulation of realistic clinical interactions plays a pivotal role in advancing clinical Large Language Models (LLMs) and supporting medical diagnostic education. Existing approaches and benchmarks rely on generic or LLM-generated dialogue data, which limits the authenticity and diversity of doctor-patient interactions. In this work, we propose the first Chinese patient simulation dataset (Ch-PatientSim), constructed from realistic clinical interaction scenarios to comprehensively evaluate the performance of models in emulating patient behavior. Patients are simulated based on a five-dimensional persona structure. To address issues of the persona class imbalance, a portion of the dataset is augmented using few-shot generation, followed by manual verification. We evaluate various state-of-the-art LLMs and find that most produce overly formal responses that lack individual personality. To address this limitation, we propose a training-free Multi-Stage Patient Role-Playing (MSPRP) framework, which decomposes interactions into three stages to ensure both personalization and realism in model responses. Experimental results demonstrate that our approach significantly improves model performance across multiple dimensions of patient simulation. 

---
# LogicLens: Leveraging Semantic Code Graph to explore Multi Repository large systems 

**Authors**: Niko Usai, Dario Montagnini, Kristian Ilianov Iliev, Raffaele Camanzo  

**Link**: [PDF](https://arxiv.org/pdf/2601.10773)  

**Abstract**: Understanding large software systems is a challenging task, especially when code is distributed across multiple repositories and microservices. Developers often need to reason not only about the structure of the code, but also about its domain logic and runtime behaviors, which are typically implicit and scattered. We introduce LogicLens, a reactive conversational agent that assists developers in exploring complex software systems through a semantic multi-repository graph. This graph is built in a preprocessing step by combining syntactic code analysis, via AST parsing and repository traversal, with semantic enrichment using Large Language Models (LLMs). The resulting graph captures both structural elements, such as files, classes, and functions, as well as functional abstractions like domain entities, operations, and workflows. Once the graph is constructed, LogicLens enables developers to interact with it via natural language, dynamically retrieving relevant subgraphs and answering technical or functional queries. We present the architecture of the system, discuss emergent behaviors, and evaluate its effectiveness on real-world multi-repository scenarios. We demonstrate emergent capabilities including impact analysis and symptom-based debugging that arise naturally from the semantic graph structure. 

---
# Unifying Speech Recognition, Synthesis and Conversion with Autoregressive Transformers 

**Authors**: Runyuan Cai, Yu Lin, Yiming Wang, Chunlin Fu, Xiaodong Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2601.10770)  

**Abstract**: Traditional speech systems typically rely on separate, task-specific models for text-to-speech (TTS), automatic speech recognition (ASR), and voice conversion (VC), resulting in fragmented pipelines that limit scalability, efficiency, and cross-task generalization. In this paper, we present General-Purpose Audio (GPA), a unified audio foundation model that integrates multiple core speech tasks within a single large language model (LLM) architecture. GPA operates on a shared discrete audio token space and supports instruction-driven task induction, enabling a single autoregressive model to flexibly perform TTS, ASR, and VC without architectural modifications. This unified design combines a fully autoregressive formulation over discrete speech tokens, joint multi-task training across speech domains, and a scalable inference pipeline that achieves high concurrency and throughput. The resulting model family supports efficient multi-scale deployment, including a lightweight 0.3B-parameter variant optimized for edge and resource-constrained environments. Together, these design choices demonstrate that a unified autoregressive architecture can achieve competitive performance across diverse speech tasks while remaining viable for low-latency, practical deployment. 

---
# Unified Optimization of Source Weights and Transfer Quantities in Multi-Source Transfer Learning: An Asymptotic Framework 

**Authors**: Qingyue Zhang, Chang Chu, Haohao Fu, Tianren Peng, Yanru Wu, Guanbo Huang, Yang Li, Shao-Lun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.10779)  

**Abstract**: Transfer learning plays a vital role in improving model performance in data-scarce scenarios. However, naive uniform transfer from multiple source tasks may result in negative transfer, highlighting the need to properly balance the contributions of heterogeneous sources. Moreover, existing transfer learning methods typically focus on optimizing either the source weights or the amount of transferred samples, while largely neglecting the joint consideration of the other. In this work, we propose a theoretical framework, Unified Optimization of Weights and Quantities (UOWQ), which formulates multi-source transfer learning as a parameter estimation problem grounded in an asymptotic analysis of a Kullback-Leibler divergence-based generalization error measure. The proposed framework jointly determines the optimal source weights and optimal transfer quantities for each source task. Firstly, we prove that using all available source samples is always optimal once the weights are properly adjusted, and we provide a theoretical explanation for this phenomenon. Moreover, to determine the optimal transfer weights, our analysis yields closed-form solutions in the single-source setting and develops a convex optimization-based numerical procedure for the multi-source case. Building on the theoretical results, we further propose practical algorithms for both multi-source transfer learning and multi-task learning settings. Extensive experiments on real-world benchmarks, including DomainNet and Office-Home, demonstrate that UOWQ consistently outperforms strong baselines. The results validate both the theoretical predictions and the practical effectiveness of our framework. 

---
# LLM-Assisted Pseudo-Relevance Feedback 

**Authors**: David Otero, Javier Parapar  

**Link**: [PDF](https://arxiv.org/pdf/2601.11238)  

**Abstract**: Query expansion is a long-standing technique to mitigate vocabulary mismatch in ad hoc Information Retrieval. Pseudo-relevance feedback methods, such as RM3, estimate an expanded query model from the top-ranked documents, but remain vulnerable to topic drift when early results include noisy or tangential content. Recent approaches instead prompt Large Language Models to generate synthetic expansions or query variants. While effective, these methods risk hallucinations and misalignment with collection-specific terminology. We propose a hybrid alternative that preserves the robustness and interpretability of classical PRF while leveraging LLM semantic judgement. Our method inserts an LLM-based filtering stage prior to RM3 estimation: the LLM judges the documents in the initial top-$k$ ranking, and RM3 is computed only over those accepted as relevant. This simple intervention improves over blind PRF and a strong baseline across several datasets and metrics. 

---
# "Can You Tell Me?": Designing Copilots to Support Human Judgement in Online Information Seeking 

**Authors**: Markus Bink, Marten Risius, Udo Kruschwitz, David Elsweiler  

**Link**: [PDF](https://arxiv.org/pdf/2601.11284)  

**Abstract**: Generative AI (GenAI) tools are transforming information seeking, but their fluent, authoritative responses risk overreliance and discourage independent verification and reasoning. Rather than replacing the cognitive work of users, GenAI systems should be designed to support and scaffold it. Therefore, this paper introduces an LLM-based conversational copilot designed to scaffold information evaluation rather than provide answers and foster digital literacy skills. In a pre-registered, randomised controlled trial (N=261) examining three interface conditions including a chat-based copilot, our mixed-methods analysis reveals that users engaged deeply with the copilot, demonstrating metacognitive reflection. However, the copilot did not significantly improve answer correctness or search engagement, largely due to a "time-on-chat vs. exploration" trade-off and users' bias toward positive information. Qualitative findings reveal tension between the copilot's Socratic approach and users' desire for efficiency. These results highlight both the promise and pitfalls of pedagogical copilots, and we outline design pathways to reconcile literacy goals with efficiency demands. 

---
# Can Instructed Retrieval Models Really Support Exploration? 

**Authors**: Piyush Maheshwari, Sheshera Mysore, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2601.10936)  

**Abstract**: Exploratory searches are characterized by under-specified goals and evolving query intents. In such scenarios, retrieval models that can capture user-specified nuances in query intent and adapt results accordingly are desirable -- instruction-following retrieval models promise such a capability. In this work, we evaluate instructed retrievers for the prevalent yet under-explored application of aspect-conditional seed-guided exploration using an expert-annotated test collection. We evaluate both recent LLMs fine-tuned for instructed retrieval and general-purpose LLMs prompted for ranking with the highly performant Pairwise Ranking Prompting. We find that the best instructed retrievers improve on ranking relevance compared to instruction-agnostic approaches. However, we also find that instruction following performance, crucial to the user experience of interacting with models, does not mirror ranking relevance improvements and displays insensitivity or counter-intuitive behavior to instructions. Our results indicate that while users may benefit from using current instructed retrievers over instruction-agnostic models, they may not benefit from using them for long-running exploratory sessions requiring greater sensitivity to instructions. 

---
# Neural Chain-of-Thought Search: Searching the Optimal Reasoning Path to Enhance Large Language Models 

**Authors**: Guoming Ling, Zhongzhan Huang, Yupei Lin, Junxin Li, Shanshan Zhong, Hefeng Wu, Liang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2601.11340)  

**Abstract**: Chain-of-Thought reasoning has significantly enhanced the problem-solving capabilities of Large Language Models. Unfortunately, current models generate reasoning steps sequentially without foresight, often becoming trapped in suboptimal reasoning paths with redundant steps. In contrast, we introduce Neural Chain-of-Thought Search (NCoTS), a framework that reformulates reasoning as a dynamic search for the optimal thinking strategy. By quantitatively characterizing the solution space, we reveal the existence of sparse superior reasoning paths that are simultaneously more accurate and concise than standard outputs. Our method actively navigates towards these paths by evaluating candidate reasoning operators using a dual-factor heuristic that optimizes for both correctness and computational cost. Consequently, NCoTS achieves a Pareto improvement across diverse reasoning benchmarks, boosting accuracy by over 3.5% while reducing generation length by over 22%. Our code and data are available at this https URL. 

---
# Membership Inference on LLMs in the Wild 

**Authors**: Jiatong Yi, Yanyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.11314)  

**Abstract**: Membership Inference Attacks (MIAs) act as a crucial auditing tool for the opaque training data of Large Language Models (LLMs). However, existing techniques predominantly rely on inaccessible model internals (e.g., logits) or suffer from poor generalization across domains in strict black-box settings where only generated text is available. In this work, we propose SimMIA, a robust MIA framework tailored for this text-only regime by leveraging an advanced sampling strategy and scoring mechanism. Furthermore, we present WikiMIA-25, a new benchmark curated to evaluate MIA performance on modern proprietary LLMs. Experiments demonstrate that SimMIA achieves state-of-the-art results in the black-box setting, rivaling baselines that exploit internal model information. 

---
# The unreasonable effectiveness of pattern matching 

**Authors**: Gary Lupyan, Blaise Agüera y Arcas  

**Link**: [PDF](https://arxiv.org/pdf/2601.11432)  

**Abstract**: We report on an astonishing ability of large language models (LLMs) to make sense of "Jabberwocky" language in which most or all content words have been randomly replaced by nonsense strings, e.g., translating "He dwushed a ghanc zawk" to "He dragged a spare chair". This result addresses ongoing controversies regarding how to best think of what LLMs are doing: are they a language mimic, a database, a blurry version of the Web? The ability of LLMs to recover meaning from structural patterns speaks to the unreasonable effectiveness of pattern-matching. Pattern-matching is not an alternative to "real" intelligence, but rather a key ingredient. 

---
# Idea First, Code Later: Disentangling Problem Solving from Code Generation in Evaluating LLMs for Competitive Programming 

**Authors**: Sama Hadhoud, Alaa Elsetohy, Frederikus Hudi, Jan Christian Blaise Cruz, Steven Halim, Alham Fikri Aji  

**Link**: [PDF](https://arxiv.org/pdf/2601.11332)  

**Abstract**: Large Language Models (LLMs) increasingly succeed on competitive programming problems, yet existing evaluations conflate algorithmic reasoning with code-level implementation. We argue that competitive programming is fundamentally a problem-solving task and propose centering natural-language editorials in both solution generation and evaluation. Generating an editorial prior to code improves solve rates for some LLMs, with substantially larger gains when using expertly written gold editorials. However, even with gold editorials, models continue to struggle with implementation, while the gap between generated and gold editorials reveals a persistent problem-solving bottleneck in specifying correct and complete algorithms. Beyond pass/fail metrics, we diagnose reasoning errors by comparing model-generated editorials to gold standards using expert annotations and validate an LLM-as-a-judge protocol for scalable evaluation. We introduce a dataset of 83 ICPC-style problems with gold editorials and full test suites, and evaluate 19 LLMs, arguing that future benchmarks should explicitly separate problem solving from implementation. 

---
# One LLM to Train Them All: Multi-Task Learning Framework for Fact-Checking 

**Authors**: Malin Astrid Larsson, Harald Fosen Grunnaleite, Vinay Setty  

**Link**: [PDF](https://arxiv.org/pdf/2601.11293)  

**Abstract**: Large language models (LLMs) are reshaping automated fact-checking (AFC) by enabling unified, end-to-end verification pipelines rather than isolated components. While large proprietary models achieve strong performance, their closed weights, complexity, and high costs limit sustainability. Fine-tuning smaller open weight models for individual AFC tasks can help but requires multiple specialized models resulting in high costs. We propose \textbf{multi-task learning (MTL)} as a more efficient alternative that fine-tunes a single model to perform claim detection, evidence ranking, and stance detection jointly. Using small decoder-only LLMs (e.g., Qwen3-4b), we explore three MTL strategies: classification heads, causal language modeling heads, and instruction-tuning, and evaluate them across model sizes, task orders, and standard non-LLM baselines. While multitask models do not universally surpass single-task baselines, they yield substantial improvements, achieving up to \textbf{44\%}, \textbf{54\%}, and \textbf{31\%} relative gains for claim detection, evidence re-ranking, and stance detection, respectively, over zero-/few-shot settings. Finally, we also provide practical, empirically grounded guidelines to help practitioners apply MTL with LLMs for automated fact-checking. 

---
# Reasoning in Trees: Improving Retrieval-Augmented Generation for Multi-Hop Question Answering 

**Authors**: Yuling Shi, Maolin Sun, Zijun Liu, Mo Yang, Yixiong Fang, Tianran Sun, Xiaodong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2601.11255)  

**Abstract**: Retrieval-Augmented Generation (RAG) has demonstrated significant effectiveness in enhancing large language models (LLMs) for complex multi-hop question answering (QA). For multi-hop QA tasks, current iterative approaches predominantly rely on LLMs to self-guide and plan multi-step exploration paths during retrieval, leading to substantial challenges in maintaining reasoning coherence across steps from inaccurate query decomposition and error propagation. To address these issues, we introduce Reasoning Tree Guided RAG (RT-RAG), a novel hierarchical framework for complex multi-hop QA. RT-RAG systematically decomposes multi-hop questions into explicit reasoning trees, minimizing inaccurate decomposition through structured entity analysis and consensus-based tree selection that clearly separates core queries, known entities, and unknown entities. Subsequently, a bottom-up traversal strategy employs iterative query rewriting and refinement to collect high-quality evidence, thereby mitigating error propagation. Comprehensive experiments show that RT-RAG substantially outperforms state-of-the-art methods by 7.0% F1 and 6.0% EM, demonstrating the effectiveness of RT-RAG in complex multi-hop QA. 

---
# Language of Thought Shapes Output Diversity in Large Language Models 

**Authors**: Shaoyang Xu, Wenxuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.11227)  

**Abstract**: Output diversity is crucial for Large Language Models as it underpins pluralism and creativity. In this work, we reveal that controlling the language used during model thinking-the language of thought-provides a novel and structural source of output diversity. Our preliminary study shows that different thinking languages occupy distinct regions in a model's thinking space. Based on this observation, we study two repeated sampling strategies under multilingual thinking-Single-Language Sampling and Mixed-Language Sampling-and conduct diversity evaluation on outputs that are controlled to be in English, regardless of the thinking language used. Across extensive experiments, we demonstrate that switching the thinking language from English to non-English languages consistently increases output diversity, with a clear and consistent positive correlation such that languages farther from English in the thinking space yield larger gains. We further show that aggregating samples across multiple thinking languages yields additional improvements through compositional effects, and that scaling sampling with linguistic heterogeneity expands the model's diversity ceiling. Finally, we show that these findings translate into practical benefits in pluralistic alignment scenarios, leading to broader coverage of cultural knowledge and value orientations in LLM outputs. Our code is publicly available at this https URL. 

---
# How DDAIR you? Disambiguated Data Augmentation for Intent Recognition 

**Authors**: Galo Castillo-López, Alexis Lombard, Nasredine Semmar, Gaël de Chalendar  

**Link**: [PDF](https://arxiv.org/pdf/2601.11234)  

**Abstract**: Large Language Models (LLMs) are effective for data augmentation in classification tasks like intent detection. In some cases, they inadvertently produce examples that are ambiguous with regard to untargeted classes. We present DDAIR (Disambiguated Data Augmentation for Intent Recognition) to mitigate this problem. We use Sentence Transformers to detect ambiguous class-guided augmented examples generated by LLMs for intent recognition in low-resource scenarios. We identify synthetic examples that are semantically more similar to another intent than to their target one. We also provide an iterative re-generation method to mitigate such ambiguities. Our findings show that sentence embeddings effectively help to (re)generate less ambiguous examples, and suggest promising potential to improve classification performance in scenarios where intents are loosely or broadly defined. 

---
# Budget-Aware Anytime Reasoning with LLM-Synthesized Preference Data 

**Authors**: Xuanming Zhang, Shwan Ashrafi, Aziza Mirsaidova, Amir Rezaeian, Miguel Ballesteros, Lydia B. Chilton, Zhou Yu, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2601.11038)  

**Abstract**: We study the reasoning behavior of large language models (LLMs) under limited computation budgets. In such settings, producing useful partial solutions quickly is often more practical than exhaustive reasoning, which incurs high inference costs. Many real-world tasks, such as trip planning, require models to deliver the best possible output within a fixed reasoning budget. We introduce an anytime reasoning framework and the Anytime Index, a metric that quantifies how effectively solution quality improves as reasoning tokens increase. To further enhance efficiency, we propose an inference-time self-improvement method using LLM-synthesized preference data, where models learn from their own reasoning comparisons to produce better intermediate solutions. Experiments on NaturalPlan (Trip), AIME, and GPQA datasets show consistent gains across Grok-3, GPT-oss, GPT-4.1/4o, and LLaMA models, improving both reasoning quality and efficiency under budget constraints. 

---
# Integrity Shield A System for Ethical AI Use & Authorship Transparency in Assessments 

**Authors**: Ashish Raj Shekhar, Shiven Agarwal, Priyanuj Bordoloi, Yash Shah, Tejas Anvekar, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2601.11093)  

**Abstract**: Large Language Models (LLMs) can now solve entire exams directly from uploaded PDF assessments, raising urgent concerns about academic integrity and the reliability of grades and credentials. Existing watermarking techniques either operate at the token level or assume control over the model's decoding process, making them ineffective when students query proprietary black-box systems with instructor-provided documents. We present Integrity Shield, a document-layer watermarking system that embeds schema-aware, item-level watermarks into assessment PDFs while keeping their human-visible appearance unchanged. These watermarks consistently prevent MLLMs from answering shielded exam PDFs and encode stable, item-level signatures that can be reliably recovered from model or student responses. Across 30 exams spanning STEM, humanities, and medical reasoning, Integrity Shield achieves exceptionally high prevention (91-94% exam-level blocking) and strong detection reliability (89-93% signature retrieval) across four commercial MLLMs. Our demo showcases an interactive interface where instructors upload an exam, preview watermark behavior, and inspect pre/post AI performance & authorship evidence. 

---
# Redefining Machine Simultaneous Interpretation: From Incremental Translation to Human-Like Strategies 

**Authors**: Qianen Zhang, Zeyu Yang, Satoshi Nakamura  

**Link**: [PDF](https://arxiv.org/pdf/2601.11002)  

**Abstract**: Simultaneous Machine Translation (SiMT) requires high-quality translations under strict real-time constraints, which traditional policies with only READ/WRITE actions cannot fully address. We extend the action space of SiMT with four adaptive actions: Sentence_Cut, Drop, Partial_Summarization and Pronominalization, which enable real-time restructuring, omission, and simplification while preserving semantic fidelity. We adapt these actions in a large language model (LLM) framework and construct training references through action-aware prompting. To evaluate both quality and word-level monotonicity, we further develop a latency-aware TTS pipeline that maps textual outputs to speech with realistic timing. Experiments on the ACL60/60 English-Chinese, English-German and English-Japanese benchmarks show that our framework consistently improves semantic metrics and achieves lower delay compared to reference translations and salami-based baselines. Notably, combining Drop and Sentence_Cut leads to consistent improvements in the balance between fluency and latency. These results demonstrate that enriching the action space of LLM-based SiMT provides a promising direction for bridging the gap between human and machine interpretation. 

---
# NAACL: Noise-AwAre Verbal Confidence Calibration for LLMs in RAG Systems 

**Authors**: Jiayu Liu, Rui Wang, Qing Zong, Qingcheng Zeng, Tianshi Zheng, Haochen Shi, Dadi Guo, Baixuan Xu, Chunyang Li, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2601.11004)  

**Abstract**: Accurately assessing model confidence is essential for deploying large language models (LLMs) in mission-critical factual domains. While retrieval-augmented generation (RAG) is widely adopted to improve grounding, confidence calibration in RAG settings remains poorly understood. We conduct a systematic study across four benchmarks, revealing that LLMs exhibit poor calibration performance due to noisy retrieved contexts. Specifically, contradictory or irrelevant evidence tends to inflate the model's false certainty, leading to severe overconfidence. To address this, we propose NAACL Rules (Noise-AwAre Confidence CaLibration Rules) to provide a principled foundation for resolving overconfidence under noise. We further design NAACL, a noise-aware calibration framework that synthesizes supervision from about 2K HotpotQA examples guided by these rules. By performing supervised fine-tuning (SFT) with this data, NAACL equips models with intrinsic noise awareness without relying on stronger teacher models. Empirical results show that NAACL yields substantial gains, improving ECE scores by 10.9% in-domain and 8.0% out-of-domain. By bridging the gap between retrieval noise and verbal calibration, NAACL paves the way for both accurate and epistemically reliable LLMs. 

---
# A Concise Agent is Less Expert: Revealing Side Effects of Using Style Features on Conversational Agents 

**Authors**: Young-Min Cho, Yuan Yuan, Sharath Chandra Guntuku, Lyle Ungar  

**Link**: [PDF](https://arxiv.org/pdf/2601.10809)  

**Abstract**: Style features such as friendly, helpful, or concise are widely used in prompts to steer the behavior of Large Language Model (LLM) conversational agents, yet their unintended side effects remain poorly understood. In this work, we present the first systematic study of cross-feature stylistic side effects. We conduct a comprehensive survey of 127 conversational agent papers from ACL Anthology and identify 12 frequently used style features. Using controlled, synthetic dialogues across task-oriented and open domain settings, we quantify how prompting for one style feature causally affects others via a pairwise LLM as a Judge evaluation framework. Our results reveal consistent and structured side effects, such as prompting for conciseness significantly reduces perceived expertise. They demonstrate that style features are deeply entangled rather than orthogonal. To support future research, we introduce CASSE (Conversational Agent Stylistic Side Effects), a dataset capturing these complex interactions. We further evaluate prompt based and activation steering based mitigation strategies and find that while they can partially restore suppressed traits, they often degrade the primary intended style. These findings challenge the assumption of faithful style control in LLMs and highlight the need for multi-objective and more principled approaches to safe, targeted stylistic steering in conversational agents. 

---
# LLMs for Game Theory: Entropy-Guided In-Context Learning and Adaptive CoT Reasoning 

**Authors**: Tommaso Felice Banfi, Sashenka Gamage  

**Link**: [PDF](https://arxiv.org/pdf/2601.10775)  

**Abstract**: We propose a novel LLM-based framework for reasoning in discrete, game-theoretic tasks, illustrated with \emph{Tic-Tac-Toe}. The method integrates in-context learning with entropy-guided chain-of-thought (CoT) reasoning and adaptive context retrieval. The model dynamically adjusts both the number of retrieved examples and reasoning paths according to token-level uncertainty: concise reasoning with minimal context is used when uncertainty is low, whereas higher uncertainty triggers expanded multi-path CoT exploration. Experimental evaluation against a sub-optimal algorithmic opponent shows that entropy-aware adaptive reasoning substantially improves decision quality, increasing the average game outcome from \(-11.6\%\) with the baseline LLM to \(+9.5\%\) with entropy-guided adaptive reasoning over 100 games (win = +1, tie = 0, loss = -1), while maintaining a relatively low number of LLM queries per game. Statistical validation confirms that the improvement is significant, and correlation analysis reveals a negative association between token-level entropy and move optimality. These findings demonstrate that uncertainty-guided adaptive reasoning effectively enhances LLM performance in sequential decision-making environments. 

---
# DialDefer: A Framework for Detecting and Mitigating LLM Dialogic Deference 

**Authors**: Parisa Rabbani, Priyam Sahoo, Ruben Mathew, Aishee Mondal, Harshita Ketharaman, Nimet Beyza Bozdag, Dilek Hakkani-Tür  

**Link**: [PDF](https://arxiv.org/pdf/2601.10896)  

**Abstract**: LLMs are increasingly used as third-party judges, yet their reliability when evaluating speakers in dialogue remains poorly understood. We show that LLMs judge identical claims differently depending on framing: the same content elicits different verdicts when presented as a statement to verify ("Is this statement correct?") versus attributed to a speaker ("Is this speaker correct?"). We call this dialogic deference and introduce DialDefer, a framework for detecting and mitigating these framing-induced judgment shifts. Our Dialogic Deference Score (DDS) captures directional shifts that aggregate accuracy obscures. Across nine domains, 3k+ instances, and four models, conversational framing induces large shifts (|DDS| up to 87pp, p < .0001) while accuracy remains stable (<2pp), with effects amplifying 2-4x on naturalistic Reddit conversations. Models can shift toward agreement (deference) or disagreement (skepticism) depending on domain -- the same model ranges from DDS = -53 on graduate-level science to +58 on social judgment. Ablations reveal that human-vs-LLM attribution drives the largest shifts (17.7pp swing), suggesting models treat disagreement with humans as more costly than with AI. Mitigation attempts reduce deference but can over-correct into skepticism, framing this as a calibration problem beyond accuracy optimization. 

---
# BYOL: Bring Your Own Language Into LLMs 

**Authors**: Syed Waqas Zamir, Wassim Hamidouche, Boulbaba Ben Amor, Luana Marotti, Inbal Becker-Reshef, Juan Lavista Ferres  

**Link**: [PDF](https://arxiv.org/pdf/2601.10804)  

**Abstract**: Large Language Models (LLMs) exhibit strong multilingual capabilities, yet remain fundamentally constrained by the severe imbalance in global language resources. While over 7,000 languages are spoken worldwide, only a small subset (fewer than 100) has sufficient digital presence to meaningfully influence modern LLM training. This disparity leads to systematic underperformance, cultural misalignment, and limited accessibility for speakers of low-resource and extreme-low-resource languages. To address this gap, we introduce Bring Your Own Language (BYOL), a unified framework for scalable, language-aware LLM development tailored to each language's digital footprint. BYOL begins with a language resource classification that maps languages into four tiers (Extreme-Low, Low, Mid, High) using curated web-scale corpora, and uses this classification to select the appropriate integration pathway. For low-resource languages, we propose a full-stack data refinement and expansion pipeline that combines corpus cleaning, synthetic text generation, continual pretraining, and supervised finetuning. Applied to Chichewa and Maori, this pipeline yields language-specific LLMs that achieve approximately 12 percent average improvement over strong multilingual baselines across 12 benchmarks, while preserving English and multilingual capabilities via weight-space model merging. For extreme-low-resource languages, we introduce a translation-mediated inclusion pathway, and show on Inuktitut that a tailored machine translation system improves over a commercial baseline by 4 BLEU, enabling high-accuracy LLM access when direct language modeling is infeasible. Finally, we release human-translated versions of the Global MMLU-Lite benchmark in Chichewa, Maori, and Inuktitut, and make our codebase and models publicly available at this https URL . 

---
# Reasoning Models Generate Societies of Thought 

**Authors**: Junsol Kim, Shiyang Lai, Nino Scherrer, Blaise Agüera y Arcas, James Evans  

**Link**: [PDF](https://arxiv.org/pdf/2601.10825)  

**Abstract**: Large language models have achieved remarkable capabilities across domains, yet mechanisms underlying sophisticated reasoning remain elusive. Recent reasoning models outperform comparable instruction-tuned models on complex cognitive tasks, attributed to extended computation through longer chains of thought. Here we show that enhanced reasoning emerges not from extended computation alone, but from simulating multi-agent-like interactions -- a society of thought -- which enables diversification and debate among internal cognitive perspectives characterized by distinct personality traits and domain expertise. Through quantitative analysis and mechanistic interpretability methods applied to reasoning traces, we find that reasoning models like DeepSeek-R1 and QwQ-32B exhibit much greater perspective diversity than instruction-tuned models, activating broader conflict between heterogeneous personality- and expertise-related features during reasoning. This multi-agent structure manifests in conversational behaviors, including question-answering, perspective shifts, and the reconciliation of conflicting views, and in socio-emotional roles that characterize sharp back-and-forth conversations, together accounting for the accuracy advantage in reasoning tasks. Controlled reinforcement learning experiments reveal that base models increase conversational behaviors when rewarded solely for reasoning accuracy, and fine-tuning models with conversational scaffolding accelerates reasoning improvement over base models. These findings indicate that the social organization of thought enables effective exploration of solution spaces. We suggest that reasoning models establish a computational parallel to collective intelligence in human groups, where diversity enables superior problem-solving when systematically structured, which suggests new opportunities for agent organization to harness the wisdom of crowds. 

---
# ZPD Detector: Data Selection via Capability-Difficulty Alignment for Large Language Models 

**Authors**: Bo Yang, Yunkui Chen, Lanfei Feng, Yu Zhang, Shijian Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.10986)  

**Abstract**: As the cost of training large language models continues to increase and high-quality training data become increasingly scarce, selecting high-value samples or synthesizing effective training data under limited data budgets has emerged as a critical research problem. Most existing data selection methods rely on static criteria, such as difficulty, uncertainty, or heuristics, and fail to model the evolving relationship between the model and the data. Inspired by the educational theory of the Zone of Proximal Development (ZPD), we propose ZPD Detector, a data selection framework that adopts a bidirectional perspective between models and data by explicitly modeling the alignment between sample difficulty and the model's current capability. ZPD Detector integrates difficulty calibration, model capability estimation based on Item Response Theory (IRT), and a capability-difficulty matching score to dynamically identify the most informative samples at each learning stage, improving data utilization efficiency; moreover, this dynamic matching strategy provides new insights into training strategy design. All code and data will be released after our work be accepted to support reproducible researc 

---
# Unlocking the Potentials of Retrieval-Augmented Generation for Diffusion Language Models 

**Authors**: Chuanyue Yu, Jiahui Wang, Yuhan Li, Heng Chang, Ge Lan, Qingyun Sun, Jia Li, Jianxin Li, Ziwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.11342)  

**Abstract**: Diffusion Language Models (DLMs) have recently demonstrated remarkable capabilities in natural language processing tasks. However, the potential of Retrieval-Augmented Generation (RAG), which shows great successes for enhancing large language models (LLMs), has not been well explored, due to the fundamental difference between LLM and DLM decoding. To fill this critical gap, we systematically test the performance of DLMs within the RAG framework. Our findings reveal that DLMs coupled with RAG show promising potentials with stronger dependency on contextual information, but suffer from limited generation precision. We identify a key underlying issue: Response Semantic Drift (RSD), where the generated answer progressively deviates from the query's original semantics, leading to low precision content. We trace this problem to the denoising strategies in DLMs, which fail to maintain semantic alignment with the query throughout the iterative denoising process. To address this, we propose Semantic-Preserving REtrieval-Augmented Diffusion (SPREAD), a novel framework that introduces a query-relevance-guided denoising strategy. By actively guiding the denoising trajectory, SPREAD ensures the generation remains anchored to the query's semantics and effectively suppresses drift. Experimental results demonstrate that SPREAD significantly enhances the precision and effectively mitigates RSD of generated answers within the RAG framework. 

---
