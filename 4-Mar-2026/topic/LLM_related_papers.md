# TAO-Attack: Toward Advanced Optimization-Based Jailbreak Attacks for Large Language Models 

**Authors**: Zhi Xu, Jiaqi Li, Xiaotong Zhang, Hong Yu, Han Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.03081)  

**Abstract**: Large language models (LLMs) have achieved remarkable success across diverse applications but remain vulnerable to jailbreak attacks, where attackers craft prompts that bypass safety alignment and elicit unsafe responses. Among existing approaches, optimization-based attacks have shown strong effectiveness, yet current methods often suffer from frequent refusals, pseudo-harmful outputs, and inefficient token-level updates. In this work, we propose TAO-Attack, a new optimization-based jailbreak method. TAO-Attack employs a two-stage loss function: the first stage suppresses refusals to ensure the model continues harmful prefixes, while the second stage penalizes pseudo-harmful outputs and encourages the model toward more harmful completions. In addition, we design a direction-priority token optimization (DPTO) strategy that improves efficiency by aligning candidates with the gradient direction before considering update magnitude. Extensive experiments on multiple LLMs demonstrate that TAO-Attack consistently outperforms state-of-the-art methods, achieving higher attack success rates and even reaching 100\% in certain scenarios. 

---
# Compact Prompting in Instruction-tuned LLMs for Joint Argumentative Component Detection 

**Authors**: Sofiane Elguendouze, Erwan Hain, Elena Cabrio, Serena Villata  

**Link**: [PDF](https://arxiv.org/pdf/2603.03095)  

**Abstract**: Argumentative component detection (ACD) is a core subtask of Argument(ation) Mining (AM) and one of its most challenging aspects, as it requires jointly delimiting argumentative spans and classifying them into components such as claims and premises. While research on this subtask remains relatively limited compared to other AM tasks, most existing approaches formulate it as a simplified sequence labeling problem, component classification, or a pipeline of component segmentation followed by classification. In this paper, we propose a novel approach based on instruction-tuned Large Language Models (LLMs) using compact instruction-based prompts, and reframe ACD as a language generation task, enabling arguments to be identified directly from plain text without relying on pre-segmented components. Experiments on standard benchmarks show that our approach achieves higher performance compared to state-of-the-art systems. To the best of our knowledge, this is one of the first attempts to fully model ACD as a generative task, highlighting the potential of instruction tuning for complex AM problems. 

---
# APRES: An Agentic Paper Revision and Evaluation System 

**Authors**: Bingchen Zhao, Jenny Zhang, Chenxi Whitehouse, Minqi Jiang, Michael Shvartsman, Abhishek Charnalia, Despoina Magka, Tatiana Shavrina, Derek Dunfield, Oisin Mac Aodha, Yoram Bachrach  

**Link**: [PDF](https://arxiv.org/pdf/2603.03142)  

**Abstract**: Scientific discoveries must be communicated clearly to realize their full potential. Without effective communication, even the most groundbreaking findings risk being overlooked or misunderstood. The primary way scientists communicate their work and receive feedback from the community is through peer review. However, the current system often provides inconsistent feedback between reviewers, ultimately hindering the improvement of a manuscript and limiting its potential impact. In this paper, we introduce a novel method APRES powered by Large Language Models (LLMs) to update a scientific papers text based on an evaluation rubric. Our automated method discovers a rubric that is highly predictive of future citation counts, and integrate it with APRES in an automated system that revises papers to enhance their quality and impact. Crucially, this objective should be met without altering the core scientific content. We demonstrate the success of APRES, which improves future citation prediction by 19.6% in mean averaged error over the next best baseline, and show that our paper revision process yields papers that are preferred over the originals by human expert evaluators 79% of the time. Our findings provide strong empirical support for using LLMs as a tool to help authors stress-test their manuscripts before submission. Ultimately, our work seeks to augment, not replace, the essential role of human expert reviewers, for it should be humans who discern which discoveries truly matter, guiding science toward advancing knowledge and enriching lives. 

---
# Efficient Self-Evaluation for Diffusion Language Models via Sequence Regeneration 

**Authors**: Linhao Zhong, Linyu Wu, Wen Wang, Yuling Xi, Chenchen Jing, Jiaheng Zhang, Hao Chen, Chunhua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2603.02760)  

**Abstract**: Diffusion large language models (dLLMs) have recently attracted significant attention for their ability to enhance diversity, controllability, and parallelism. However, their non-sequential, bidirectionally masked generation makes quality assessment difficult, underscoring the need for effective self-evaluation. In this work, we propose DiSE, a simple yet effective self-evaluation confidence quantification method for dLLMs. DiSE quantifies confidence by computing the probability of regenerating the tokens in the entire generated sequence, given the full context. This method enables more efficient and reliable quality assessment by leveraging token regeneration probabilities, facilitating both likelihood estimation and robust uncertainty quantification. Building upon DiSE, we further introduce a flexible-length generation framework, which adaptively controls the sequence length based on the model's self-assessment of its own output. We analyze and validate the feasibility of DiSE from the perspective of dLLM generalization, and empirically demonstrate that DiSE is positively correlated with both semantic coherence and answer accuracy. Extensive experiments on likelihood evaluation, uncertainty quantification, and flexible-length generation further confirm the effectiveness of the proposed DiSE. 

---
# From Solver to Tutor: Evaluating the Pedagogical Intelligence of LLMs with KMP-Bench 

**Authors**: Weikang Shi, Houxing Ren, Junting Pan, Aojun Zhou, Ke Wang, Zimu Lu, Yunqiao Yang, Yuxuan Hu, Linda Wei, Mingjie Zhan, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.02775)  

**Abstract**: Large Language Models (LLMs) show significant potential in AI mathematical tutoring, yet current evaluations often rely on simplistic metrics or narrow pedagogical scenarios, failing to assess comprehensive, multi-turn teaching effectiveness. In this paper, we introduce KMP-Bench, a comprehensive K-8 Mathematical Pedagogical Benchmark designed to assess LLMs from two complementary perspectives. The first module, KMP-Dialogue, evaluates holistic pedagogical capabilities against six core principles (e.g., Challenge, Explanation, Feedback), leveraging a novel multi-turn dialogue dataset constructed by weaving together diverse pedagogical components. The second module, KMP-Skills, provides a granular assessment of foundational tutoring abilities, including multi-turn problem-solving, error detection and correction, and problem generation. Our evaluations on KMP-Bench reveal a key disparity: while leading LLMs excel at tasks with verifiable solutions, they struggle with the nuanced application of pedagogical principles. Additionally, we present KMP-Pile, a large-scale (150K) dialogue dataset. Models fine-tuned on KMP-Pile show substantial improvement on KMP-Bench, underscoring the value of pedagogically-rich training data for developing more effective AI math tutors. 

---
# OCR or Not? Rethinking Document Information Extraction in the MLLMs Era with Real-World Large-Scale Datasets 

**Authors**: Jiyuan Shen, Peiyue Yuan, Atin Ghosh, Yifan Mai, Daniel Dahlmeier  

**Link**: [PDF](https://arxiv.org/pdf/2603.02789)  

**Abstract**: Multimodal Large Language Models (MLLMs) enhance the potential of natural language processing. However, their actual impact on document information extraction remains unclear. In particular, it is unclear whether an MLLM-only pipeline--while simpler--can truly match the performance of traditional OCR+MLLM setups. In this paper, we conduct a large-scale benchmarking study that evaluates various out-of-the-box MLLMs on business-document information extraction. To examine and explore failure modes, we propose an automated hierarchical error analysis framework that leverages large language models (LLMs) to diagnose error patterns systematically. Our findings suggest that OCR may not be necessary for powerful MLLMs, as image-only input can achieve comparable performance to OCR-enhanced approaches. Moreover, we demonstrate that carefully designed schema, exemplars, and instructions can further enhance MLLMs performance. We hope this work can offer practical guidance and valuable insight for advancing document information extraction. 

---
# Evaluating Cross-Modal Reasoning Ability and Problem Characteristics with Multimodal Item Response Theory 

**Authors**: Shunki Uebayashi, Kento Masui, Kyohei Atarashi, Han Bao, Hisashi Kashima, Naoto Inoue, Mayu Otani, Koh Takeuchi  

**Link**: [PDF](https://arxiv.org/pdf/2603.02663)  

**Abstract**: Multimodal Large Language Models (MLLMs) have recently emerged as general architectures capable of reasoning over diverse modalities. Benchmarks for MLLMs should measure their ability for cross-modal integration. However, current benchmarks are filled with shortcut questions, which can be solved using only a single modality, thereby yielding unreliable rankings. For example, in vision-language cases, we can find the correct answer without either the image or the text. These low-quality questions unnecessarily increase the size and computational requirements of benchmarks. We introduce a multi-modal and multidimensional item response theory framework (M3IRT) that extends classical IRT by decomposing both model ability and item difficulty into image-only, text-only, and cross-modal components. M3IRT estimates cross-modal ability of MLLMs and each question's cross-modal difficulty, enabling compact, high-quality subsets that better reflect multimodal reasoning. Across 24 VLMs on three benchmarks, M3IRT prioritizes genuinely cross-modal questions over shortcuts and preserves ranking fidelity even when 50% of items are artificially generated low-quality questions, thereby reducing evaluation cost while improving reliability. M3IRT thus offers a practical tool for assessing cross-modal reasoning and refining multimodal benchmarks. 

---
# Think, But Don't Overthink: Reproducing Recursive Language Models 

**Authors**: Daren Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.02615)  

**Abstract**: This project reproduces and extends the recently proposed ``Recursive Language Models'' (RLMs) framework by Zhang et al. (2026). This framework enables Large Language Models (LLMs) to process near-infinite contexts by offloading the prompt into an external REPL environment. While the original paper relies on a default recursion depth of 1 and suggests deeper recursion as a future direction, this study specifically investigates the impact of scaling the recursion depth. Using state-of-the-art open-source agentic models (DeepSeek v3.2 and Kimi K2), I evaluated pure LLM, RLM (depth=1), and RLM (depth=2) on the S-NIAH and OOLONG benchmarks. The findings reveal a compelling phenomenon: Deeper recursion causes models to ``overthink''. While depth-1 RLMs effectively boost accuracy on complex reasoning tasks, applying deeper recursion (depth=2) or using RLMs on simple retrieval tasks paradoxically degrades performance and exponentially inflates execution time (e.g., from 3.6s to 344.5s) and token costs. Code and data are available at: this https URL 

---
# Sensory-Aware Sequential Recommendation via Review-Distilled Representations 

**Authors**: Yeo Chan Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2603.02709)  

**Abstract**: We propose a novel framework for sensory-aware sequential recommendation that enriches item representations with linguistically extracted sensory attributes from product reviews. Our approach, \textsc{ASEGR} (Attribute-based Sensory Enhanced Generative Recommendation), introduces a two-stage pipeline in which a large language model is first fine-tuned as a teacher to extract structured sensory attribute--value pairs, such as \textit{color: matte black} and \textit{scent: vanilla}, from unstructured review text. The extracted structures are then distilled into a compact student transformer that produces fixed-dimensional sensory embeddings for each item. These embeddings encode experiential semantics in a reusable form and are incorporated into standard sequential recommender architectures as additional item-level representations. We evaluate our method on four Amazon domains and integrate the learned sensory embeddings into representative sequential recommendation models, including SASRec, BERT4Rec, and BSARec. Across domains, sensory-enhanced models consistently outperform their identifier-based counterparts, indicating that linguistically grounded sensory representations provide complementary signals to behavioral interaction patterns. Qualitative analysis further shows that the extracted attributes align closely with human perceptions of products, enabling interpretable connections between natural language descriptions and recommendation behavior. Overall, this work demonstrates that sensory attribute distillation offers a principled and scalable way to bridge information extraction and sequential recommendation through structured semantic representation learning. 

---
# ExpGuard: LLM Content Moderation in Specialized Domains 

**Authors**: Minseok Choi, Dongjin Kim, Seungbin Yang, Subin Kim, Youngjun Kwak, Juyoung Oh, Jaegul Choo, Jungmin Son  

**Link**: [PDF](https://arxiv.org/pdf/2603.02588)  

**Abstract**: With the growing deployment of large language models (LLMs) in real-world applications, establishing robust safety guardrails to moderate their inputs and outputs has become essential to ensure adherence to safety policies. Current guardrail models predominantly address general human-LLM interactions, rendering LLMs vulnerable to harmful and adversarial content within domain-specific contexts, particularly those rich in technical jargon and specialized concepts. To address this limitation, we introduce ExpGuard, a robust and specialized guardrail model designed to protect against harmful prompts and responses across financial, medical, and legal domains. In addition, we present ExpGuardMix, a meticulously curated dataset comprising 58,928 labeled prompts paired with corresponding refusal and compliant responses, from these specific sectors. This dataset is divided into two subsets: ExpGuardTrain, for model training, and ExpGuardTest, a high-quality test set annotated by domain experts to evaluate model robustness against technical and domain-specific content. Comprehensive evaluations conducted on ExpGuardTest and eight established public benchmarks reveal that ExpGuard delivers competitive performance across the board while demonstrating exceptional resilience to domain-specific adversarial attacks, surpassing state-of-the-art models such as WildGuard by up to 8.9% in prompt classification and 15.3% in response classification. To encourage further research and development, we open-source our code, data, and model, enabling adaptation to additional domains and supporting the creation of increasingly robust guardrail models. 

---
# ITLC at SemEval-2026 Task 11: Normalization and Deterministic Parsing for Formal Reasoning in LLMs 

**Authors**: Wicaksono Leksono Muhamad, Joanito Agili Lopo, Tack Hwa Wong, Muhammad Ravi Shulthan Habibi, Samuel Cahyawijaya  

**Link**: [PDF](https://arxiv.org/pdf/2603.02676)  

**Abstract**: Large language models suffer from content effects in reasoning tasks, particularly in multi-lingual contexts. We introduce a novel method that reduces these biases through explicit structural abstraction that transforms syllogisms into canonical logical representations and applies deterministic parsing to determine validity. Evaluated on the SemEval-2026 Task 11 multilingual benchmark, our approach achieves top-5 rankings across all subtasks while substantially reducing content effects and offering a competitive alternative to complex fine-tuning or activation-level interventions. 

---
# Graph-GRPO: Stabilizing Multi-Agent Topology Learning via Group Relative Policy Optimization 

**Authors**: Yueyang Cang, Xiaoteng Zhang, Erlu Zhao, Zehua Ji, Yuhang Liu, Yuchen He, Zhiyuan Ning, Chen Yijun, Wenge Que, Li Shi  

**Link**: [PDF](https://arxiv.org/pdf/2603.02701)  

**Abstract**: Optimizing communication topology is fundamental to the efficiency and effectiveness of Large Language Model (LLM)-based Multi-Agent Systems (MAS). While recent approaches utilize reinforcement learning to dynamically construct task-specific graphs, they typically rely on single-sample policy gradients with absolute rewards (e.g., binary correctness). This paradigm suffers from severe gradient variance and the credit assignment problem: simple queries yield non-informative positive rewards for suboptimal structures, while difficult queries often result in failures that provide no learning signal. To address these challenges, we propose Graph-GRPO, a novel topology optimization framework that integrates Group Relative Policy Optimization. Instead of evaluating a single topology in isolation, Graph-GRPO samples a group of diverse communication graphs for each query and computes the advantage of specific edges based on their relative performance within the group. By normalizing rewards across the sampled group, our method effectively mitigates the noise derived from task difficulty variance and enables fine-grained credit assignment. Extensive experiments on reasoning and code generation benchmarks demonstrate that Graph-GRPO significantly outperforms state-of-the-art baselines, achieving superior training stability and identifying critical communication pathways previously obscured by reward noise. 

---
# How Controllable Are Large Language Models? A Unified Evaluation across Behavioral Granularities 

**Authors**: Ziwen Xu, Kewei Xu, Haoming Xu, Haiwen Hong, Longtao Huang, Hui Xue, Ningyu Zhang, Yongliang Shen, Guozhou Zheng, Huajun Chen, Shumin Deng  

**Link**: [PDF](https://arxiv.org/pdf/2603.02578)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in socially sensitive domains, yet their unpredictable behaviors, ranging from misaligned intent to inconsistent personality, pose significant risks. We introduce SteerEval, a hierarchical benchmark for evaluating LLM controllability across three domains: language features, sentiment, and personality. Each domain is structured into three specification levels: L1 (what to express), L2 (how to express), and L3 (how to instantiate), connecting high-level behavioral intent to concrete textual output. Using SteerEval, we systematically evaluate contemporary steering methods, revealing that control often degrades at finer-grained levels. Our benchmark offers a principled and interpretable framework for safe and controllable LLM behavior, serving as a foundation for future research. 

---
# Detecting AI-Generated Essays in Writing Assessment: Responsible Use and Generalizability Across LLMs 

**Authors**: Jiangang Hao  

**Link**: [PDF](https://arxiv.org/pdf/2603.02353)  

**Abstract**: Writing is a foundational literacy skill that underpins effective communication, fosters critical thinking, facilitates learning across disciplines, and enables individuals to organize and articulate complex ideas. Consequently, writing assessment plays a vital role in evaluating language proficiency, communicative effectiveness, and analytical reasoning. The rapid advancement of large language models (LLMs) has made it increasingly easy to generate coherent, high-quality essays, raising significant concerns about the authenticity of student-submitted work. This chapter first provides an overview of the current landscape of detectors for AI-generated and AI-assisted essays, along with guidelines for their responsible use. It then presents empirical analyses to evaluate how well detectors trained on essays from one LLM generalize to identifying essays produced by other LLMs, based on essays generated in response to public GRE writing prompts. These findings provide guidance for developing and retraining detectors for practical applications. 

---
# ACE-Brain-0: Spatial Intelligence as a Shared Scaffold for Universal Embodiments 

**Authors**: Ziyang Gong, Zehang Luo, Anke Tang, Zhe Liu, Shi Fu, Zhi Hou, Ganlin Yang, Weiyun Wang, Xiaofeng Wang, Jianbo Liu, Gen Luo, Haolan Kang, Shuang Luo, Yue Zhou, Yong Luo, Li Shen, Xiaosong Jia, Yao Mu, Xue Yang, Chunxiao Liu, Junchi Yan, Hengshuang Zhao, Dacheng Tao, Xiaogang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.03198)  

**Abstract**: Universal embodied intelligence demands robust generalization across heterogeneous embodiments, such as autonomous driving, robotics, and unmanned aerial vehicles (UAVs). However, existing embodied brain in training a unified model over diverse embodiments frequently triggers long-tail data, gradient interference, and catastrophic forgetting, making it notoriously difficult to balance universal generalization with domain-specific proficiency. In this report, we introduce ACE-Brain-0, a generalist foundation brain that unifies spatial reasoning, autonomous driving, and embodied manipulation within a single multimodal large language model~(MLLM). Our key insight is that spatial intelligence serves as a universal scaffold across diverse physical embodiments: although vehicles, robots, and UAVs differ drastically in morphology, they share a common need for modeling 3D mental space, making spatial cognition a natural, domain-agnostic foundation for cross-embodiment transfer. Building on this insight, we propose the Scaffold-Specialize-Reconcile~(SSR) paradigm, which first establishes a shared spatial foundation, then cultivates domain-specialized experts, and finally harmonizes them through data-free model merging. Furthermore, we adopt Group Relative Policy Optimization~(GRPO) to strengthen the model's comprehensive capability. Extensive experiments demonstrate that ACE-Brain-0 achieves competitive and even state-of-the-art performance across 24 spatial and embodiment-related benchmarks. 

---
# MoD-DPO: Towards Mitigating Cross-modal Hallucinations in Omni LLMs using Modality Decoupled Preference Optimization 

**Authors**: Ashutosh Chaubey, Jiacheng Pang, Mohammad Soleymani  

**Link**: [PDF](https://arxiv.org/pdf/2603.03192)  

**Abstract**: Omni-modal large language models (omni LLMs) have recently achieved strong performance across audiovisual understanding tasks, yet they remain highly susceptible to cross-modal hallucinations arising from spurious correlations and dominant language priors. In this work, we propose Modality-Decoupled Direct Preference Optimization (MoD-DPO), a simple and effective framework for improving modality grounding in omni LLMs. MoD-DPO introduces modality-aware regularization terms that explicitly enforce invariance to corruptions in irrelevant modalities and sensitivity to perturbations in relevant modalities, thereby reducing unintended cross-modal interactions. To further mitigate over-reliance on textual priors, we incorporate a language-prior debiasing penalty that discourages hallucination-prone text-only responses. Extensive experiments across multiple audiovisual hallucination benchmarks demonstrate that MoD-DPO consistently improves perception accuracy and hallucination resistance, outperforming previous preference optimization baselines under similar training budgets. Our findings underscore the importance of modality-faithful alignment and demonstrate a scalable path toward more reliable and resilient multimodal foundation models. 

---
# TikZilla: Scaling Text-to-TikZ with High-Quality Data and Reinforcement Learning 

**Authors**: Christian Greisinger, Steffen Eger  

**Link**: [PDF](https://arxiv.org/pdf/2603.03072)  

**Abstract**: Large language models (LLMs) are increasingly used to assist scientists across diverse workflows. A key challenge is generating high-quality figures from textual descriptions, often represented as TikZ programs that can be rendered as scientific images. Prior research has proposed a variety of datasets and modeling approaches for this task. However, existing datasets for Text-to-TikZ are too small and noisy to capture the complexity of TikZ, causing mismatches between text and rendered figures. Moreover, prior approaches rely solely on supervised fine-tuning (SFT), which does not expose the model to the rendered semantics of the figure, often resulting in errors such as looping, irrelevant content, and incorrect spatial relations. To address these issues, we construct DaTikZ-V4, a dataset more than four times larger and substantially higher in quality than DaTikZ-V3, enriched with LLM-generated figure descriptions. Using this dataset, we train TikZilla, a family of small open-source Qwen models (3B and 8B) with a two-stage pipeline of SFT followed by reinforcement learning (RL). For RL, we leverage an image encoder trained via inverse graphics to provide semantically faithful reward signals. Extensive human evaluations with over 1,000 judgments show that TikZilla improves by 1.5-2 points over its base models on a 5-point scale, surpasses GPT-4o by 0.5 points, and matches GPT-5 in the image-based evaluation, while operating at much smaller model sizes. Code, data, and models will be made available. 

---
# HELIOS: Harmonizing Early Fusion, Late Fusion, and LLM Reasoning for Multi-Granular Table-Text Retrieval 

**Authors**: Sungho Park, Joohyung Yun, Jongwuk Lee, Wook-Shin Han  

**Link**: [PDF](https://arxiv.org/pdf/2603.02248)  

**Abstract**: Table-text retrieval aims to retrieve relevant tables and text to support open-domain question answering. Existing studies use either early or late fusion, but face limitations. Early fusion pre-aligns a table row with its associated passages, forming "stars," which often include irrelevant contexts and miss query-dependent relationships. Late fusion retrieves individual nodes, dynamically aligning them, but it risks missing relevant contexts. Both approaches also struggle with advanced reasoning tasks, such as column-wise aggregation and multi-hop reasoning. To address these issues, we propose HELIOS, which combines the strengths of both approaches. First, the edge-based bipartite subgraph retrieval identifies finer-grained edges between table segments and passages, effectively avoiding the inclusion of irrelevant contexts. Then, the query-relevant node expansion identifies the most promising nodes, dynamically retrieving relevant edges to grow the bipartite subgraph, minimizing the risk of missing important contexts. Lastly, the star-based LLM refinement performs logical inference at the star graph level rather than the bipartite subgraph, supporting advanced reasoning tasks. Experimental results show that HELIOS outperforms state-of-the-art models with a significant improvement up to 42.6\% and 39.9\% in recall and nDCG, respectively, on the OTT-QA benchmark. 

---
# Self-Play Only Evolves When Self-Synthetic Pipeline Ensures Learnable Information Gain 

**Authors**: Wei Liu, Siya Qi, Yali Du, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2603.02218)  

**Abstract**: Large language models (LLMs) make it plausible to build systems that improve through self-evolving loops, but many existing proposals are better understood as self-play and often plateau quickly. A central failure mode is that the loop synthesises more data without increasing learnable information for the next iteration. Through experiments on a self-play coding task, we reveal that sustainable self-evolution requires a self-synthesised data pipeline with learnable information that increases across iterations. We identify triadic roles that self-evolving LLMs play: the Proposer, which generates tasks; the Solver, which attempts solutions; and the Verifier, which provides training signals, and we identify three system designs that jointly target learnable information gain from this triadic roles perspective. Asymmetric co-evolution closes a weak-to-strong-to-weak loop across roles. Capacity growth expands parameter and inference-time budgets to match rising learnable information. Proactive information seeking introduces external context and new task sources that prevent saturation. Together, these modules provide a measurable, system-level path from brittle self-play dynamics to sustained self-evolution. 

---
# Agentic AI-based Coverage Closure for Formal Verification 

**Authors**: Sivaram Pothireddypalli, Ashish Raman, Deepak Narayan Gadde, Aman Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2603.03147)  

**Abstract**: Coverage closure is a critical requirement in Integrated Chip (IC) development process and key metric for verification sign-off. However, traditional exhaustive approaches often fail to achieve full coverage within project timelines. This study presents an agentic AI-driven workflow that utilizes Large Language Model (LLM)-enabled Generative AI (GenAI) to automate coverage analysis for formal verification, identify coverage gaps, and generate the required formal properties. The framework accelerates verification efficiency by systematically addressing coverage holes. Benchmarking open-source and internal designs reveals a measurable increase in coverage metrics, with improvements correlated to the complexity of the design. Comparative analysis validates the effectiveness of this approach. These results highlight the potential of agentic AI-based techniques to improve formal verification productivity and support comprehensive coverage closure. 

---
# Beyond Factual Correctness: Mitigating Preference-Inconsistent Explanations in Explainable Recommendation 

**Authors**: Chengkai Wang, Baisong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.03080)  

**Abstract**: LLM-based explainable recommenders can produce fluent explanations that are factually correct, yet still justify items using attributes that conflict with a user's historical preferences. Such preference-inconsistent explanations yield logically valid but unconvincing reasoning and are largely missed by standard hallucination or faithfulness metrics. We formalize this failure mode and propose PURE, a preference-aware reasoning framework following a select-then-generate paradigm. Instead of only improving generation, PURE intervenes in evidence selection, it selects a compact set of multi-hop item-centric reasoning paths that are both factually grounded and aligned with user preference structure, guided by user intent, specificity, and diversity to suppress generic, weakly personalized evidence. The selected evidence is then injected into LLM generation via structure-aware prompting that preserves relational constraints. To measure preference inconsistency, we introduce a feature-level, user-centric evaluation metric that reveals misalignment overlooked by factuality-based measures. Experiments on three real-world datasets show that PURE consistently reduces preference-inconsistent explanations and factual hallucinations while maintaining competitive recommendation accuracy, explanation quality, and inference efficiency. These results highlight that trustworthy explanations require not only factual correctness but also justification aligned with user preferences. 

---
# RAPO: Expanding Exploration for LLM Agents via Retrieval-Augmented Policy Optimization 

**Authors**: Siwei Zhang, Yun Xiong, Xi Chen, Zi'an Jia, Renhong Huang, Jiarong Xu, Jiawei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.03078)  

**Abstract**: Agentic Reinforcement Learning (Agentic RL) has shown remarkable potential in large language model-based (LLM) agents. These works can empower LLM agents to tackle complex tasks via multi-step, tool-integrated reasoning. However, an inherent limitation of existing Agentic RL methods is their reliance on a pure on-policy paradigm for exploration, restricting exploration to the agent's self-generated outputs and preventing the discovery of new reasoning perspectives for further improvement. While recent efforts incorporate auxiliary off-policy signals to enhance exploration, they typically utilize full off-policy trajectories for trajectory-level policy estimation, overlooking the necessity for the fine-grained, step-level exploratory dynamics within agentic rollout. In this paper, we revisit exploration in Agentic RL and propose Retrieval-Augmented Policy Optimization (RAPO), a novel RL framework that introduces retrieval to explicitly expand exploration during training. To achieve this, we decompose the Agentic RL training process into two phases: (i) Hybrid-policy Agentic Rollout, and (ii) Retrieval-aware Policy Optimization. Specifically, we propose a Hybrid-policy Agentic Rollout strategy, which allows the agents to continuously reason over the retrieved off-policy step-level traces. It dynamically extends the reasoning receptive field of agents, enabling broader exploration conditioned on external behaviors. Subsequently, we introduce the Retrieval-aware Policy Optimization mechanism, which calibrates the policy gradient estimation with retrieval reward and importance shaping, stabilizing training and prioritizing retrieval-illuminating exploration. Extensive experiments show that RAPO achieves an +5.0% average gain on fourteen datasets across three agentic reasoning tasks, while delivering 1.2x faster training efficiency. 

---
# AI-for-Science Low-code Platform with Bayesian Adversarial Multi-Agent Framework 

**Authors**: Zihang Zeng, Jiaquan Zhang, Pengze Li, Yuan Qi, Xi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.03233)  

**Abstract**: Large Language Models (LLMs) demonstrate potentials for automating scientific code generation but face challenges in reliability, error propagation in multi-agent workflows, and evaluation in domains with ill-defined success metrics. We present a Bayesian adversarial multi-agent framework specifically designed for AI for Science (AI4S) tasks in the form of a Low-code Platform (LCP). Three LLM-based agents are coordinated under the Bayesian framework: a Task Manager that structures user inputs into actionable plans and adaptive test cases, a Code Generator that produces candidate solutions, and an Evaluator providing comprehensive feedback. The framework employs an adversarial loop where the Task Manager iteratively refines test cases to challenge the Code Generator, while prompt distributions are dynamically updated using Bayesian principles by integrating code quality metrics: functional correctness, structural alignment, and static analysis. This co-optimization of tests and code reduces dependence on LLM reliability and addresses evaluation uncertainty inherent to scientific tasks. LCP also streamlines human-AI collaboration by translating non-expert prompts into domain-specific requirements, bypassing the need for manual prompt engineering by practitioners without coding backgrounds. Benchmark evaluations demonstrate LCP's effectiveness in generating robust code while minimizing error propagation. The proposed platform is also tested on an Earth Science cross-disciplinary task and demonstrates strong reliability, outperforming competing models. 

---
# ShipTraj-R1: Reinforcing Ship Trajectory Prediction in Large Language Models via Group Relative Policy Optimization 

**Authors**: Yang Zhan, Yunhao Li, Zhang Chao, Yuxu Lu, Yan Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.02939)  

**Abstract**: Recent advancements in reinforcement fine-tuning have significantly improved the reasoning ability of large language models (LLMs). In particular, methods such as group relative policy optimization (GRPO) have demonstrated strong capabilities across various fields. However, applying LLMs to ship trajectory prediction remains largely unexplored. In this paper, we propose ShipTraj-R1, a novel LLM-based framework that reformulates ship trajectory prediction as a text-to-text generation problem. (1) We design a dynamic prompt containing trajectory information about conflicting ships to guide the model to achieve adaptive chain-of-thought (CoT) reasoning. (2) We introduce a comprehensive rule-based reward mechanism to incentivize the reasoning format and prediction accuracy of the model. (3) Our ShipTraj-R1 is reinforced through the GRPO mechanism guided by domain-specific prompts and rewards, and utilizes the Qwen3 as the model backbone. Extensive experimental results on two complex and real-world maritime datasets show that the proposed ShipTraj-R1 achieves the least error compared with state-of-the-art deep learning and LLM-based baselines. 

---
# SAE as a Crystal Ball: Interpretable Features Predict Cross-domain Transferability of LLMs without Training 

**Authors**: Qi Zhang, Yifei Wang, Xiaohan Wang, Jiajun Chai, Guojun Yin, Wei Lin, Yisen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.02908)  

**Abstract**: In recent years, pre-trained large language models have achieved remarkable success across diverse tasks. Besides the pivotal role of self-supervised pre-training, their effectiveness in downstream applications also depends critically on the post-training process, which adapts models to task-specific data and objectives. However, this process inevitably introduces model shifts that can influence performance in different domains, and how such shifts transfer remains poorly understood. To open up the black box, we propose the SAE-based Transferability Score (STS), a new metric that leverages sparse autoencoders (SAEs) to forecast post-training transferability. Taking supervised fine-tuning as an example, STS identifies shifted dimensions in SAE representations and calculates their correlations with downstream domains, enabling reliable estimation of transferability \textit{before} fine-tuning. Extensive experiments across multiple models and domains show that STS accurately predicts the transferability of supervised fine-tuning, achieving Pearson correlation coefficients above 0.7 with actual performance changes. Beyond this, we take an initial step toward extending STS to reinforcement learning. We believe that STS can serve as an {\color{black} interpretable} tool for guiding post-training strategies in LLMs. Code is available at this https URL. 

---
# OrchMAS: Orchestrated Reasoning with Multi Collaborative Heterogeneous Scientific Expert Structured Agents 

**Authors**: Yichao Feng, Haoran Luo, Zhenghong Lin, Yiqun Sun, Pengfei Wei, Lawrence B. Hsieh, Anh Tuan Luu  

**Link**: [PDF](https://arxiv.org/pdf/2603.03005)  

**Abstract**: Multi-agent large language model frameworks are promising for complex multi step reasoning, yet existing systems remain weak for scientific and knowledge intensive domains due to static prompts and agent roles, rigid workflows, and homogeneous model reliance, leading to poor domain adaptation, limited reasoning flexibility, and high latency on heterogeneous or long-horizon scientific tasks. They also struggle to revise earlier decisions when intermediate reasoning diverges, reducing reliability in structured and calculation heavy settings. To address these limitations, we propose a scientific domain oriented interactive two tier multi model orchestration framework. A dedicated orchestration model analyzes each task, dynamically constructs a domain aware reasoning pipeline, and instantiates specialized expert agents with tailored prompts, while an execution model performs each step under generated role and instruction specifications. The orchestrator iteratively updates the pipeline based on intermediate feedback, enabling dynamic replanning, role reallocation, and prompt refinement across multi turn interactions, strengthening robustness and specialization for scientific reasoning through structured heterogeneous model collaboration. The framework is model agnostic and supports heterogeneous LLM integration with different capacities or costs, enabling flexible performance efficiency trade offs in practical scientific deployments. Experiments show consistent improvements over existing multi agent systems and strong baselines across diverse reasoning and scientific style benchmarks. 

---
# LLM-based Argument Mining meets Argumentation and Description Logics: a Unified Framework for Reasoning about Debates 

**Authors**: Gianvincenzo Alfano, Sergio Greco, Lucio La Cava, Stefano Francesco Monea, Irina Trubitsyna  

**Link**: [PDF](https://arxiv.org/pdf/2603.02858)  

**Abstract**: Large Language Models (LLMs) achieve strong performance in analyzing and generating text, yet they struggle with explicit, transparent, and verifiable reasoning over complex texts such as those containing debates. In particular, they lack structured representations that capture how arguments support or attack each other and how their relative strengths determine overall acceptability. We encompass these limitations by proposing a framework that integrates learning-based argument mining with quantitative reasoning and ontology-based querying. Starting from a raw debate text, the framework extracts a fuzzy argumentative knowledge base, where arguments are explicitly represented as entities, linked by attack and support relations, and annotated with initial fuzzy strengths reflecting plausibility w.r.t. the debate's context. Quantitative argumentation semantics are then applied to compute final argument strengths by propagating the effects of supports and attacks. These results are then embedded into a fuzzy description logic setting, enabling expressive query answering through efficient rewriting techniques. The proposed approach provides a transparent, explainable, and formally grounded method for analyzing debates, overcoming purely statistical LLM-based analyses. 

---
# Rethinking Code Similarity for Automated Algorithm Design with LLMs 

**Authors**: Rui Zhang, Zhichao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2603.02787)  

**Abstract**: The rise of Large Language Model-based Automated Algorithm Design (LLM-AAD) has transformed algorithm development by autonomously generating code implementations of expert-level algorithms. Unlike traditional expert-driven algorithm development, in the LLM-AAD paradigm, the main design principle behind an algorithm is often implicitly embedded in the generated code. Therefore, assessing algorithmic similarity directly from code, distinguishing genuine algorithmic innovation from mere syntactic variation, becomes essential. While various code similarity metrics exist, they fail to capture algorithmic similarity, as they focus on surface-level syntax or output equivalence rather than the underlying algorithmic logic.
We propose BehaveSim, a novel method to measure algorithmic similarity through the lens of problem-solving behavior as a sequence of intermediate solutions produced during execution, dubbed as problem-solving trajectories (PSTrajs). By quantifying the alignment between PSTrajs using dynamic time warping (DTW), BehaveSim distinguishes algorithms with divergent logic despite syntactic or output-level similarities. We demonstrate its utility in two key applications: (i) Enhancing LLM-AAD: Integrating BehaveSim into existing LLM-AAD frameworks (e.g., FunSearch, EoH) promotes behavioral diversity, significantly improving performance on three AAD tasks. (ii) Algorithm analysis: BehaveSim clusters generated algorithms by behavior, enabling systematic analysis of problem-solving strategies--a crucial tool for the growing ecosystem of AI-generated algorithms. Data and code of this work are open-sourced at this https URL. 

---
# A Natural Language Agentic Approach to Study Affective Polarization 

**Authors**: Stephanie Anneris Malvicini, Ewelina Gajewska, Arda Derbent, Katarzyna Budzynska, Jarosław A. Chudziak, Maria Vanina Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2603.02711)  

**Abstract**: Affective polarization has been central to political and social studies, with growing focus on social media, where partisan divisions are often exacerbated. Real-world studies tend to have limited scope, while simulated studies suffer from insufficient high-quality training data, as manually labeling posts is labor-intensive and prone to subjective biases. The lack of adequate tools to formalize different definitions of affective polarization across studies complicates result comparison and hinders interoperable frameworks. We present a multi-agent model providing a comprehensive approach to studying affective polarization in social media. To operationalize our framework, we develop a platform leveraging large language models (LLMs) to construct virtual communities where agents engage in discussions. We showcase the potential of our platform by (1) analyzing questions related to affective polarization, as explored in social science literature, providing a fresh perspective on this phenomenon, and (2) introducing scenarios that allow observation and measurement of polarization at different levels of granularity and abstraction. Experiments show that our platform is a flexible tool for computational studies of complex social dynamics such as affective polarization. It leverages advanced agent models to simulate rich, context-sensitive interactions and systematically explore research questions traditionally addressed through human-subject studies. 

---
# FinTexTS: Financial Text-Paired Time-Series Dataset via Semantic-Based and Multi-Level Pairing 

**Authors**: Jaehoon Lee, Suhwan Park, Tae Yoon Lim, Seunghan Lee, Jun Seo, Dongwan Kang, Hwanil Choi, Minjae Kim, Sungdong Yoo, SoonYoung Lee, Yongjae Lee, Wonbin Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2603.02702)  

**Abstract**: The financial domain involves a variety of important time-series problems. Recently, time-series analysis methods that jointly leverage textual and numerical information have gained increasing attention. Accordingly, numerous efforts have been made to construct text-paired time-series datasets in the financial domain. However, financial markets are characterized by complex interdependencies, in which a company's stock price is influenced not only by company-specific events but also by events in other companies and broader macroeconomic factors. Existing approaches that pair text with financial time-series data based on simple keyword matching often fail to capture such complex relationships. To address this limitation, we propose a semantic-based and multi-level pairing framework. Specifically, we extract company-specific context for the target company from SEC filings and apply an embedding-based matching mechanism to retrieve semantically relevant news articles based on this context. Furthermore, we classify news articles into four levels (macro-level, sector-level, related company-level, and target-company level) using large language models (LLMs), enabling multi-level pairing of news articles with the target company. Applying this framework to publicly-available news datasets, we construct \textbf{FinTexTS}, a new large-scale text-paired stock price dataset. Experimental results on \textbf{FinTexTS} demonstrate the effectiveness of our semantic-based and multi-level pairing strategy in stock price forecasting. In addition to publicly-available news underlying \textbf{FinTexTS}, we show that applying our method to proprietary yet carefully curated news sources leads to higher-quality paired data and improved stock price forecasting performance. 

---
# Beyond Task Completion: Revealing Corrupt Success in LLM Agents through Procedure-Aware Evaluation 

**Authors**: Hongliu Cao, Ilias Driouich, Eoin Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2603.03116)  

**Abstract**: Large Language Model (LLM)-based agents are increasingly adopted in high-stakes settings, but current benchmarks evaluate mainly whether a task was completed, not how. We introduce Procedure-Aware Evaluation (PAE), a framework that formalizes agent procedures as structured observations and exposes consistency relationships between what agents observe, communicate, and execute. PAE evaluates agents along complementary axes (Utility, Efficiency, Interaction Quality, Procedural Integrity) and applies multi-dimensional gating that categorically disqualifies corrupt outcomes. Evaluating state-of-the-art LLM agents on tau-bench yields findings at the axis, compliance, and benchmark levels. At the axis level, the dimensions capture non-redundant failure modes: utility masks reliability gaps, speed does not imply precision, and conciseness does not predict intent adherence. At the procedural compliance level, 27-78% of benchmark reported successes are corrupt successes concealing violations across interaction and integrity. Furthermore, gating substantially collapses Pass^4 rate and affects model rankings. The analysis of corrupt success cases reveals distinctive per-model failure signatures: GPT-5 spreads errors across policy, execution, and intent dimensions; Kimi-K2-Thinking concentrates 78% of violations in policy faithfulness and compliance; and Mistral-Large-3 is dominated by faithfulness failures. At the benchmark level, our analysis exposes structural flaws in the benchmark design, including task scope gaps, contradictory reward signals, and simulator artifacts that produce accidental successes. 

---
# SorryDB: Can AI Provers Complete Real-World Lean Theorems? 

**Authors**: Austin Letson, Leopoldo Sarra, Auguste Poiroux, Oliver Dressler, Paul Lezeau, Dhyan Aranha, Frederick Pu, Aaron Hill, Miguel Corredera Hidalgo, Julian Berman, George Tsoukalas, Lenny Taelman  

**Link**: [PDF](https://arxiv.org/pdf/2603.02668)  

**Abstract**: We present SorryDB, a dynamically-updating benchmark of open Lean tasks drawn from 78 real world formalization projects on GitHub. Unlike existing static benchmarks, often composed of competition problems, hillclimbing the SorryDB benchmark will yield tools that are aligned to the community needs, more usable by mathematicians, and more capable of understanding complex dependencies. Moreover, by providing a continuously updated stream of tasks, SorryDB mitigates test-set contamination and offers a robust metric for an agent's ability to contribute to novel formal mathematics projects. We evaluate a collection of approaches, including generalist large language models, agentic approaches, and specialized symbolic provers, over a selected snapshot of 1000 tasks from SorryDB. We show that current approaches are complementary: even though an agentic approach based on Gemini Flash is the most performant, it is not strictly better than other off-the-shelf large-language models, specialized provers, or even a curated list of Lean tactics. 

---
# LLMs for High-Frequency Decision-Making: Normalized Action Reward-Guided Consistency Policy Optimization 

**Authors**: Yang Zhao, Zihao Li, Zhiyu Jiang, Dandan Ma, Ganchao Liu, Wenzhe Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.02680)  

**Abstract**: While Large Language Models (LLMs) form the cornerstone of sequential decision-making agent development, they have inherent limitations in high-frequency decision tasks. Existing research mainly focuses on discrete embodied decision scenarios with low-frequency and significant semantic differences in state space (e.g., household planning). These methods suffer from limited performance in high-frequency decision-making tasks, since high-precision numerical state information in such tasks undergoes frequent updates with minimal fluctuations, and exhibiting policy misalignment between the learned sub-tasks and composite tasks. To address these issues, this paper proposes Normalized Action Reward guided Consistency Policy Optimization (NAR-CP). 1) Our method first acquires predefined dense rewards from environmental feedback of candidate actions via reward functions, then completes reward shaping through normalization, and theoretically verifies action reward normalization does not impair optimal policy. 2) To reduce policy misalignment in composite tasks, we use LLMs to infer sub-observation candidate actions and generate joint policies, with consistency loss ensuring precise alignment between global semantic policies and sub-semantic policies. Experiments on UAV pursuit, a typical high-frequency task, show our method delivers superior performance on independent and composite tasks with excellent generalization to unseen tasks. 

---
# A Neuropsychologically Grounded Evaluation of LLM Cognitive Abilities 

**Authors**: Faiz Ghifari Haznitrama, Faeyza Rishad Ardi, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2603.02540)  

**Abstract**: Large language models (LLMs) exhibit a unified "general factor" of capability across 10 benchmarks, a finding confirmed by our factor analysis of 156 models, yet they still struggle with simple, trivial tasks for humans. This is because current benchmarks focus on task completion, failing to probe the foundational cognitive abilities that highlight these behaviors. We address this by introducing the NeuroCognition benchmark, grounded in three adapted neuropsychological tests: Raven's Progressive Matrices (abstract relational reasoning), Spatial Working Memory (maintenance and systematic search), and the Wisconsin Card Sorting Test (cognitive flexibility). Our evaluation reveals that while models perform strongly on text, their performance degrades for images and with increased complexity. Furthermore, we observe that complex reasoning is not universally beneficial, whereas simple, human-like strategies yield partial gains. We also find that NeuroCognition correlates positively with standard general-capability benchmarks, while still measuring distinct cognitive abilities beyond them. Overall, NeuroCognition emphasizes where current LLMs align with human-like intelligence and where they lack core adaptive cognition, showing the potential to serve as a verifiable, scalable source for improving LLMs. 

---
# NeuroProlog: Multi-Task Fine-Tuning for Neurosymbolic Mathematical Reasoning via the Cocktail Effect 

**Authors**: Pratibha Zunjare, Michael Hsiao  

**Link**: [PDF](https://arxiv.org/pdf/2603.02504)  

**Abstract**: Large Language Models (LLMs) achieve strong performance on natural language tasks but remain unreliable in mathematical reasoning, frequently generating fluent yet logically inconsistent solutions. We present \textbf{NeuroProlog}, a neurosymbolic framework that ensures verifiable reasoning by compiling math word problems into executable Prolog programs with formal verification guarantees. We propose a multi-task Cocktail training strategy that jointly optimizes three synergistic objectives in a unified symbolic representation space: (i) mathematical formula-to-rule translation (KB), (ii) natural language-to-program synthesis (SOLVE), and (iii) program-answer alignment. This joint supervision enables positive transfer, where symbolic grounding in formula translation directly improves compositional reasoning capabilities. At inference, we introduce an execution-guided decoding pipeline with fine-grained error taxonomy that enables iterative program repair and quantifies model self-debugging capacity. Comprehensive evaluation on GSM8K across four model scales (3B--32B parameters) demonstrates consistent improvements: cocktail training achieves significant accuracy gains of +5.23\% (Qwen-32B, $p < 0.01$), +3.43\% (GPT-OSS-20B, $p < 0.01$), and +5.54\% (Llama-3B, $p < 0.05$) over single-task this http URL error analysis reveals scale-dependent learning dynamics: at 32B scale, cocktail training transforms unfixable type errors (12\% repair rate) into correctable domain errors (96\% repair rate), achieving 92.7\% overall correction; at 8B scale, the same training eliminates syntactic errors but introduces semantic failures, revealing a critical capacity threshold for type-safe symbolic reasoning. 

---
# PRISM: Pushing the Frontier of Deep Think via Process Reward Model-Guided Inference 

**Authors**: Rituraj Sharma, Weiyuan Chen, Noah Provenzano, Tu Vu  

**Link**: [PDF](https://arxiv.org/pdf/2603.02479)  

**Abstract**: DEEPTHINK methods improve reasoning by generating, refining, and aggregating populations of candidate solutions, which enables strong performance on complex mathematical and scientific tasks. However, existing frameworks often lack reliable correctness signals during inference, which creates a population-enhancement bottleneck where deeper deliberation amplifies errors, suppresses correct minority solutions, and yields weak returns to additional compute. In this paper, we introduce a functional decomposition of DEEPTHINK systems and propose PRISM, a Process Reward Model (PRM)-guided inference algorithm that uses step-level verification to guide both population refinement and solution aggregation. During refinement, PRISM treats candidate solutions as particles in a PRM-defined energy landscape and reshapes the population through score-guided resampling and stochastic refinement, which concentrates probability mass on higher-quality reasoning while preserving diversity. Across mathematics and science benchmarks, PRISM is competitive with or outperforms existing DEEPTHINK methods, reaching 90.0%, 75.4%, and 71.4% with gpt-oss-20b on AIME25, HMMT25, and GPQA Diamond, respectively, while matching or exceeding gpt-oss-120b. Additionally, our analysis shows that PRISM produces consistent net-directional correction during refinement, remains reliable when the initial population contains few correct candidates, and often lies on the compute-accuracy Pareto frontier. 

---
# LLM-MLFFN: Multi-Level Autonomous Driving Behavior Feature Fusion via Large Language Model 

**Authors**: Xiangyu Li, Tianyi Wang, Xi Cheng, Rakesh Chowdary Machineni, Zhaomiao Guo, Sikai Chen, Junfeng Jiao, Christian Claudel  

**Link**: [PDF](https://arxiv.org/pdf/2603.02528)  

**Abstract**: Accurate classification of autonomous vehicle (AV) driving behaviors is critical for safety validation, performance diagnosis, and traffic integration analysis. However, existing approaches primarily rely on numerical time-series modeling and often lack semantic abstraction, limiting interpretability and robustness in complex traffic environments. This paper presents LLM-MLFFN, a novel large language model (LLM)-enhanced multi-level feature fusion network designed to address the complexities of multi-dimensional driving data. The proposed LLM-MLFFN framework integrates priors from largescale pre-trained models and employs a multi-level approach to enhance classification accuracy. LLM-MLFFN comprises three core components: (1) a multi-level feature extraction module that extracts statistical, behavioral, and dynamic features to capture the quantitative aspects of driving behaviors; (2) a semantic description module that leverages LLMs to transform raw data into high-level semantic features; and (3) a dual-channel multi-level feature fusion network that combines numerical and semantic features using weighted attention mechanisms to improve robustness and prediction accuracy. Evaluation on the Waymo open trajectory dataset demonstrates the superior performance of the proposed LLM-MLFFN, achieving a classification accuracy of over 94%, surpassing existing machine learning models. Ablation studies further validate the critical contributions of multi-level fusion, feature extraction strategies, and LLM-derived semantic reasoning. These results suggest that integrating structured feature modeling with language-driven semantic abstraction provides a principled and interpretable pathway for robust autonomous driving behavior classification. 

---
# Engineering Reasoning and Instruction (ERI) Benchmark: A Large Taxonomy-driven Dataset for Foundation Models and Agents 

**Authors**: MZ Naser, Ahmad Bani Awwad, Zoie McCreery, Radwa Eissa, Ahmad Naser, Gianluca Cusatis, Andrew Metcalf, Kapil Madathil, Jamal Abdalla, Venkatesh Kodur, Mohammad Reza Saeb  

**Link**: [PDF](https://arxiv.org/pdf/2603.02239)  

**Abstract**: The Engineering Reasoning and Instruction (ERI) benchmark is a taxonomy-driven instruction dataset designed to train and evaluate engineering-capable large language models (LLMs) and agents. This dataset spans nine engineering fields (namely: civil, mechanical, electrical, chemical, environmental, aerospace, materials, fire, and industrial engineering) and 55 subdomains, and is crossed with seven intent types (i.e., definition, explanation, calculation, comparison, design/synthesis, troubleshooting, and code-related) and three difficulty tiers (undergraduate, graduate, and professional), yielding 57,750 records with field/subdomain/type/difficulty metadata and solution formatting. We examined ERI via seven LLMs and report a statistically significant three-tier performance structure, with frontier models (GPT-5, Claude Sonnet 4, DeepSeek V3.1) achieving mean scores above 4.30 on a five-point scale, while mid-tier and smaller models exhibited progressively higher failure rates and steeper performance degradation on graduate-level questions. To address circularity concerns inherent in LLM benchmarks, we developed a convergent validation protocol that leverages cross-provider independence, multi-judge averaging, and frontier-model agreement analysis to empirically bound hallucination risk to 1.7%. ERI is released with taxonomy specifications, validation scripts, and an evaluation harness to enable reproducible comparisons and regression testing for instruction tuning, routing, retrieval-augmented evaluation, and agentic tool-use workflows in engineering settings. 

---
# AnchorDrive: LLM Scenario Rollout with Anchor-Guided Diffusion Regeneration for Safety-Critical Scenario Generation 

**Authors**: Zhulin Jiang, Zetao Li, Cheng Wang, Ziwen Wang, Chen Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2603.02542)  

**Abstract**: Autonomous driving systems require comprehensive evaluation in safety-critical scenarios to ensure safety and robustness. However, such scenarios are rare and difficult to collect from real-world driving data, necessitating simulation-based synthesis. Yet, existing methods often exhibit limitations in both controllability and realism. From a capability perspective, LLMs excel at controllable generation guided by natural language instructions, while diffusion models are better suited for producing trajectories consistent with realistic driving distributions. Leveraging their complementary strengths, we propose AnchorDrive, a two-stage safety-critical scenario generation framework. In the first stage, we deploy an LLM as a driver agent within a closed-loop simulation, which reasons and iteratively outputs control commands under natural language constraints; a plan assessor reviews these commands and provides corrective feedback, enabling semantically controllable scenario generation. In the second stage, the LLM extracts key anchor points from the first-stage trajectories as guidance objectives, which jointly with other guidance terms steer the diffusion model to regenerate complete trajectories with improved realism while preserving user-specified intent. Experiments on the highD dataset demonstrate that AnchorDrive achieves superior overall performance in criticality, realism, and controllability, validating its effectiveness for generating controllable and realistic safety-critical scenarios. 

---
# TrustMH-Bench: A Comprehensive Benchmark for Evaluating the Trustworthiness of Large Language Models in Mental Health 

**Authors**: Zixin Xiong, Ziteng Wang, Haotian Fan, Xinjie Zhang, Wenxuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.03047)  

**Abstract**: While Large Language Models (LLMs) demonstrate significant potential in providing accessible mental health support, their practical deployment raises critical trustworthiness concerns due to the domains high-stakes and safety-sensitive nature. Existing evaluation paradigms for general-purpose LLMs fail to capture mental health-specific requirements, highlighting an urgent need to prioritize and enhance their trustworthiness. To address this, we propose TrustMH-Bench, a holistic framework designed to systematically quantify the trustworthiness of mental health LLMs. By establishing a deep mapping from domain-specific norms to quantitative evaluation metrics, TrustMH-Bench evaluates models across eight core pillars: Reliability, Crisis Identification and Escalation, Safety, Fairness, Privacy, Robustness, Anti-sycophancy, and Ethics. We conduct extensive experiments across six general-purpose LLMs and six specialized mental health models. Experimental results indicate that the evaluated models underperform across various trustworthiness dimensions in mental health scenarios, revealing significant deficiencies. Notably, even generally powerful models (e.g., GPT-5.1) fail to maintain consistently high performance across all dimensions. Consequently, systematically improving the trustworthiness of LLMs has become a critical task. Our data and code are released. 

---
# Eliciting Numerical Predictive Distributions of LLMs Without Autoregression 

**Authors**: Julianna Piskorz, Katarzyna Kobalczyk, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2603.02913)  

**Abstract**: Large Language Models (LLMs) have recently been successfully applied to regression tasks -- such as time series forecasting and tabular prediction -- by leveraging their in-context learning abilities. However, their autoregressive decoding process may be ill-suited to continuous-valued outputs, where obtaining predictive distributions over numerical targets requires repeated sampling, leading to high computational cost and inference time. In this work, we investigate whether distributional properties of LLM predictions can be recovered without explicit autoregressive generation. To this end, we study a set of regression probes trained to predict statistical functionals (e.g., mean, median, quantiles) of the LLM's numerical output distribution directly from its internal representations. Our results suggest that LLM embeddings carry informative signals about summary statistics of their predictive distributions, including the numerical uncertainty. This investigation opens up new questions about how LLMs internally encode uncertainty in numerical tasks, and about the feasibility of lightweight alternatives to sampling-based approaches for uncertainty-aware numerical predictions. 

---
# Beyond One-Size-Fits-All: Adaptive Subgraph Denoising for Zero-Shot Graph Learning with Large Language Models 

**Authors**: Fengzhi Li, Liang Zhang, Yuan Zuo, Ruiqing Zhao, YanSong Liu, Yunfei Ma, Fanyu Meng, Junlan Feng  

**Link**: [PDF](https://arxiv.org/pdf/2603.02938)  

**Abstract**: Graph-based tasks in the zero-shot setting remain a significant challenge due to data scarcity and the inability of traditional Graph Neural Networks (GNNs) to generalize to unseen domains or label spaces. While recent advancements have transitioned toward leveraging Large Language Models (LLMs) as predictors to enhance GNNs, these methods often suffer from cross-modal alignment issues. A recent paradigm (i.e., Graph-R1) overcomes the aforementioned architectural dependencies by adopting a purely text-based format and utilizing LLM-based graph reasoning, showing improved zero-shot generalization. However, it employs a task-agnostic, one-size-fits-all subgraph extraction strategy, which inevitably introduces significant structural noise--irrelevant neighbors and edges--that distorts the LLMs' receptive field and leads to suboptimal predictions. To address this limitation, we introduce GraphSSR, a novel framework designed for adaptive subgraph extraction and denoising in zero-shot LLM-based graph reasoning. Specifically, we propose the SSR pipeline, which dynamically tailors subgraph extraction to specific contexts through a "Sample-Select-Reason" process, enabling the model to autonomously filter out task-irrelevant neighbors and overcome the one-size-fits-all issue. To internalize this capability, we develop SSR-SFT, a data synthesis strategy that generates high-quality SSR-style graph reasoning traces for supervised fine-tuning of LLMs. Furthermore, we propose SSR-RL, a two-stage reinforcement learning framework that explicitly regulates sampling and selection operations within the proposed SSR pipeline designed for adaptive subgraph denoising. By incorporating Authenticity-Reinforced and Denoising-Reinforced RL, we guide the model to achieve accurate predictions using parsimonious, denoised subgraphs for reasoning. 

---
# ZeroDayBench: Evaluating LLM Agents on Unseen Zero-Day Vulnerabilities for Cyberdefense 

**Authors**: Nancy Lau, Louis Sloot, Jyoutir Raj, Giuseppe Marco Boscardin, Evan Harris, Dylan Bowman, Mario Brajkovski, Jaideep Chawla, Dan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.02297)  

**Abstract**: Large language models (LLMs) are increasingly being deployed as software engineering agents that autonomously contribute to repositories. A major benefit these agents present is their ability to find and patch security vulnerabilities in the codebases they oversee. To estimate the capability of agents in this domain, we introduce ZeroDayBench, a benchmark where LLM agents find and patch 22 novel critical vulnerabilities in open-source codebases. We focus our efforts on three popular frontier agentic LLMs: GPT-5.2, Claude Sonnet 4.5, and Grok 4.1. We find that frontier LLMs are not yet capable of autonomously solving our tasks and observe some behavioral patterns that suggest how these models can be improved in the domain of proactive cyberdefense. 

---
# RIVA: Leveraging LLM Agents for Reliable Configuration Drift Detection 

**Authors**: Sami Abuzakuk, Lucas Crijns, Anne-Marie Kermarrec, Rafael Pires, Martijn de Vos  

**Link**: [PDF](https://arxiv.org/pdf/2603.02345)  

**Abstract**: Infrastructure as code (IaC) tools automate cloud provisioning but verifying that deployed systems remain consistent with the IaC specifications remains challenging. Such configuration drift occurs because of bugs in the IaC specification, manual changes, or system updates. Large language model (LLM)-based agentic AI systems can automate the analysis of large volumes of telemetry data, making them suitable for the detection of configuration drift. However, existing agentic systems implicitly assume that the tools they invoke always return correct outputs, making them vulnerable to erroneous tool responses. Since agents cannot distinguish whether an anomalous tool output reflects a real infrastructure problem or a broken tool, such errors may cause missed drift or false alarms, reducing reliability precisely when it is most needed. We introduce RIVA (Robust Infrastructure by Verification Agents), a novel multi-agent system that performs robust IaC verification even when tools produce incorrect or misleading outputs. RIVA employs two specialized agents, a verifier agent and a tool generation agent, that collaborate through iterative cross-validation, multi-perspective verification, and tool call history tracking. Evaluation on the AIOpsLab benchmark demonstrates that RIVA, in the presence of erroneous tool responses, recovers task accuracy from 27.3% when using a baseline ReAct agent to 50.0% on average. RIVA also improves task accuracy 28% to 43.8% without erroneous tool responses. Our results show that cross-validation of diverse tool calls enables more reliable autonomous infrastructure verification in production cloud environments. 

---
# Silent Sabotage During Fine-Tuning: Few-Shot Rationale Poisoning of Compact Medical LLMs 

**Authors**: Jingyuan Xie, Wenjie Wang, Ji Wu, Jiandong Gao  

**Link**: [PDF](https://arxiv.org/pdf/2603.02262)  

**Abstract**: Supervised fine-tuning (SFT) is essential for the development of medical large language models (LLMs), yet prior poisoning studies have mainly focused on the detectable backdoor attacks. We propose a novel poisoning attack targeting the reasoning process of medical LLMs during SFT. Unlike backdoor attacks, our method injects poisoned rationales into few-shot training data, leading to stealthy degradation of model performance on targeted medical topics. Results showed that knowledge overwriting was ineffective, while rationale poisoning caused significant decline on the accuracy of the target subject, as long as no correct samples of the same subject appear in the dataset. A minimum number and ratio of poisoned samples was needed to carry out an effective and stealthy attack, which was more efficient and accurate than catastrophic forgetting. We demonstrate though this study the risk of SFT-stage poisoning, hoping to spur more studies of defense in the sensitive medical domain. 

---
# CUDABench: Benchmarking LLMs for Text-to-CUDA Generation 

**Authors**: Jiace Zhu, Wentao Chen, Qi Fan, Zhixing Ren, Junying Wu, Xing Zhe Chai, Chotiwit Rungrueangwutthinon, Yehan Ma, An Zou  

**Link**: [PDF](https://arxiv.org/pdf/2603.02236)  

**Abstract**: Recent studies have demonstrated the potential of Large Language Models (LLMs) in generating GPU Kernels. Current benchmarks focus on the translation of high-level languages into CUDA, overlooking the more general and challenging task of text-to-CUDA generation. Furthermore, given the hardware-specific and performance-critical features of GPU programming, accurately assessing the performance of LLM-generated GPU programs is nontrivial. In this work, we introduce CUDABench, a comprehensive benchmark designed to evaluate the text-to-CUDA capabilities of LLMs. First, we construct CUDABench-Set, which covers Breadth-Depth-Difficulty evaluation space in diverse application domains, including artificial intelligence, scientific computing, and data analytics, etc. Furthermore, we propose CUDABench-Score and Generative Verification Pipeline that assess (1) compilation correctness, (2) functional consistency through execution-based verification, and (3) a novel roofline-based metric, Performance-Score. Benchmarking state-of-the-art LLMs reveals insightful findings and challenges of text-to-CUDA, such as a notable mismatch between high compilation success rates and low functional correctness, a lack of domain-specific algorithmic knowledge, and suboptimal utilization of GPU hardware resources. Our benchmark is available at this https URL. 

---
# Concept Heterogeneity-aware Representation Steering 

**Authors**: Laziz U. Abdullaev, Noelle Y. L. Wong, Ryan T. Z. Lee, Shiqi Jiang, Khoi N. M. Nguyen, Tan M. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2603.02237)  

**Abstract**: Representation steering offers a lightweight mechanism for controlling the behavior of large language models (LLMs) by intervening on internal activations at inference time. Most existing methods rely on a single global steering direction, typically obtained via difference-in-means over contrastive datasets. This approach implicitly assumes that the target concept is homogeneously represented across the embedding space. In practice, however, LLM representations can be highly non-homogeneous, exhibiting clustered, context-dependent structure, which renders global steering directions brittle. In this work, we view representation steering through the lens of optimal transport (OT), noting that standard difference-in-means steering implicitly corresponds to the OT map between two unimodal Gaussian distributions with identical covariance, yielding a global translation. To relax this restrictive assumption, we theoretically model source and target representations as Gaussian mixture models and formulate steering as a discrete OT problem between semantic latent clusters. From the resulting transport plan, we derive an explicit, input-dependent steering map via barycentric projection, producing a smooth, kernel-weighted combination of cluster-level shifts. We term this method Concept Heterogeneity-aware Representation Steering (CHaRS). Through numerous experimental settings, we show that CHaRS yields more effective behavioral control than global steering. 

---
# Neural Paging: Learning Context Management Policies for Turing-Complete Agents 

**Authors**: Liang Chen, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.02228)  

**Abstract**: The proof that Large Language Models (LLMs) augmented with external read-write memory constitute a computationally universal system has established the theoretical foundation for general-purpose agents. However, existing implementations face a critical bottleneck: the finite and costly Context Window, which functions not as infinite memory but as a scarce semantic cache. In this work, we introduce \textit{Neural Paging}, a hierarchical architecture that decouples symbolic reasoning from information resource management. We formulate the \textit{Context Paging Problem (CPP)} and propose a lightweight, differentiable \textit{Page Controller} designed to approximate ``Semantic Belady's Optimality'' -- retaining tokens with high future utility under explicit assumptions on access patterns. We provide theoretical analysis showing that, under bounded context window size~$K$, Neural Paging reduces the asymptotic complexity of long-horizon reasoning from quadratic $O(N^2)$ to $O(N \cdot K^2)$, and we derive a robustness bound (Theorem~4) that quantifies competitive-ratio degradation under policy-dependent access with bounded sensitivity. We validate these bounds on synthetic paging traces, confirming that the theoretical guarantees hold and identifying significant slack that motivates learned policies. 

---
# MedFeat: Model-Aware and Explainability-Driven Feature Engineering with LLMs for Clinical Tabular Prediction 

**Authors**: Zizheng Zhang, Yiming Li, Justin Xu, Jinyu Wang, Rui Wang, Lei Song, Jiang Bian, David W Eyre, Jingjing Fu  

**Link**: [PDF](https://arxiv.org/pdf/2603.02221)  

**Abstract**: In healthcare tabular predictions, classical models with feature engineering often outperform neural approaches. Recent advances in Large Language Models enable the integration of domain knowledge into feature engineering, offering a promising direction. However, existing approaches typically rely on a broad search over predefined transformations, overlooking downstream model characteristics and feature importance signals. We present MedFeat, a feedback-driven and model-aware feature engineering framework that leverages LLM reasoning with domain knowledge and provides feature explanations based on SHAP values while tracking successful and failed proposals to guide feature discovery. By incorporating model awareness, MedFeat prioritizes informative signals that are difficult for the downstream model to learn directly due to its characteristics. Across a broad range of clinical prediction tasks, MedFeat achieves stable improvements over various baselines and discovers clinically meaningful features that generalize under distribution shift, demonstrating robustness across years and from ICU cohorts to general hospitalized patients, thereby offering insights into real-world deployment. Code required to reproduce our experiments will be released, subject to dataset agreements and institutional policies. 

---
# ATPO: Adaptive Tree Policy Optimization for Multi-Turn Medical Dialogue 

**Authors**: Ruike Cao, Shaojie Bai, Fugen Yao, Liang Dong, Jian Xu, Li Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2603.02216)  

**Abstract**: Effective information seeking in multi-turn medical dialogues is critical for accurate diagnosis, especially when dealing with incomplete information. Aligning Large Language Models (LLMs) for these interactive scenarios is challenging due to the uncertainty inherent in user-agent interactions, which we formulate as a Hierarchical Markov Decision Process (H-MDP). While conventional Reinforcement Learning (RL) methods like Group Relative Policy Optimization (GRPO) struggle with long-horizon credit assignment and Proximal Policy Optimization (PPO) suffers from unstable value estimation in this context, we propose a novel uncertainty-aware Adaptive Tree Policy Optimization (ATPO) algorithm. Our method adaptively allocates the rollout budget to states with high uncertainty, quantified by a composite metric of Bellman error and action-value variance. This strategy enables more accurate value estimation, while fostering more efficient and diverse exploration. To mitigate the high computational cost of tree-based RL, we introduce two key optimizations: an uncertainty-guided pruning mechanism to minimize the number of rollouts, and an asynchronous search architecture that leverages KV cache reuse to maximize inference throughput. Extensive experiments on three public medical dialogue benchmarks demonstrate that our algorithm significantly outperforms several strong baselines, culminating in Qwen3-8B model surpassing the much larger GPT-4o ($+0.92\%$ accuracy). 

---
# Relevance Matters: A Multi-Task and Multi-Stage Large Language Model Approach for E-commerce Query Rewriting 

**Authors**: Aijun Dai, Jixiang Zhang, Haiqing Hu, Guoyu Tang, Lin Liu, Ziguang Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.02555)  

**Abstract**: For e-commerce search, user experience is measured by users' behavioral responses to returned products, like click-through rate and conversion rate, as well as the relevance between returned products and search queries. Consequently, relevance and user conversion constitute the two primary objectives in query rewriting, a strategy to bridge the lexical gap between user expressions and product descriptions. This research proposes a multi-task and multi-stage query rewriting framework grounded in large language models (LLMs). Critically, in contrast to previous works that primarily emphasized rewritten query generation, we inject the relevance task into query rewriting. Specifically, leveraging a pretrained model on user data and product information from this http URL, the approach initiates with multi-task supervised fine-tuning (SFT) comprising of the rewritten query generation task and the relevance tagging task between queries and rewrites. Subsequently, we employ Group Relative Policy Optimization (GRPO) for the model's objective alignment oriented toward enhancing the relevance and stimulating user conversions. Through offline evaluation and online A/B test, our framework illustrates substantial improvements in the effectiveness of e-commerce query rewriting, resulting in elevating the search results' relevance and boosting the number of purchases made per user (UCVR). Since August 2025, our approach has been implemented on this http URL, one of China's leading online shopping platforms. 

---
