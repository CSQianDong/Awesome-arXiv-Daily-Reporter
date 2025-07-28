# Agent0: Leveraging LLM Agents to Discover Multi-value Features from Text for Enhanced Recommendations 

**Authors**: Blaž Škrlj, Benoît Guilleminot, Andraž Tori  

**Link**: [PDF](https://arxiv.org/pdf/2507.18993)  

**Abstract**: Large language models (LLMs) and their associated agent-based frameworks have significantly advanced automated information extraction, a critical component of modern recommender systems. While these multitask frameworks are widely used in code generation, their application in data-centric research is still largely untapped. This paper presents Agent0, an LLM-driven, agent-based system designed to automate information extraction and feature construction from raw, unstructured text. Categorical features are crucial for large-scale recommender systems but are often expensive to acquire. Agent0 coordinates a group of interacting LLM agents to automatically identify the most valuable text aspects for subsequent tasks (such as models or AutoML pipelines). Beyond its feature engineering capabilities, Agent0 also offers an automated prompt-engineering tuning method that utilizes dynamic feedback loops from an oracle. Our findings demonstrate that this closed-loop methodology is both practical and effective for automated feature discovery, which is recognized as one of the most challenging phases in current recommender system development. 

---
# Distilling a Small Utility-Based Passage Selector to Enhance Retrieval-Augmented Generation 

**Authors**: Hengran Zhang, Keping Bi, Jiafeng Guo, Jiaming Zhang, Shuaiqiang Wang, Dawei Yin, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.19102)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models (LLMs) by incorporating retrieved information. Standard retrieval process prioritized relevance, focusing on topical alignment between queries and passages. In contrast, in RAG, the emphasis has shifted to utility, which considers the usefulness of passages for generating accurate answers. Despite empirical evidence showing the benefits of utility-based retrieval in RAG, the high computational cost of using LLMs for utility judgments limits the number of passages evaluated. This restriction is problematic for complex queries requiring extensive information. To address this, we propose a method to distill the utility judgment capabilities of LLMs into smaller, more efficient models. Our approach focuses on utility-based selection rather than ranking, enabling dynamic passage selection tailored to specific queries without the need for fixed thresholds. We train student models to learn pseudo-answer generation and utility judgments from teacher LLMs, using a sliding window method that dynamically selects useful passages. Our experiments demonstrate that utility-based selection provides a flexible and cost-effective solution for RAG, significantly reducing computational costs while improving answer quality. We present the distillation results using Qwen3-32B as the teacher model for both relevance ranking and utility-based selection, distilled into RankQwen1.7B and UtilityQwen1.7B. Our findings indicate that for complex questions, utility-based selection is more effective than relevance ranking in enhancing answer generation performance. We will release the relevance ranking and utility-based selection annotations for the MS MARCO dataset, supporting further research in this area. 

---
# Injecting External Knowledge into the Reasoning Process Enhances Retrieval-Augmented Generation 

**Authors**: Minghao Tang, Shiyu Ni, Jiafeng Guo, Keping Bi  

**Link**: [PDF](https://arxiv.org/pdf/2507.19333)  

**Abstract**: Retrieval-augmented generation (RAG) has been widely adopted to augment large language models (LLMs) with external knowledge for knowledge-intensive tasks. However, its effectiveness is often undermined by the presence of noisy (i.e., low-quality) retrieved passages. Enhancing LLMs' robustness to such noise is critical for improving the reliability of RAG systems. Recent advances have equipped LLMs with strong reasoning and self-reflection capabilities, allowing them to identify and correct errors in their reasoning process. Inspired by this ability, we propose Passage Injection-a simple yet effective method that explicitly incorporates retrieved passages into LLMs' reasoning process, aiming to enhance the model's ability to recognize and resist noisy passages. We validate Passage Injection under general RAG settings using BM25 as the retriever. Experiments on four reasoning-enhanced LLMs across four factual QA datasets demonstrate that Passage Injection significantly improves overall RAG performance. Further analysis on two noisy retrieval settings-random noise, where the model is provided irrelevant passages, and counterfactual noise, where it is given misleading passages-shows that Passage Injection consistently improves robustness. Controlled experiments confirm that Passage Injection can also effectively leverage helpful passages. These findings suggest that incorporating passages in LLMs' reasoning process is a promising direction for building more robust RAG systems. The code can be found \href{here}{this https URL}. 

---
# Towards LLM-Enhanced Group Recommender Systems 

**Authors**: Sebastian Lubos, Alexander Felfernig, Thi Ngoc Trang Tran, Viet-Man Le, Damian Garber, Manuel Henrich, Reinhard Willfort, Jeremias Fuchs  

**Link**: [PDF](https://arxiv.org/pdf/2507.19283)  

**Abstract**: In contrast to single-user recommender systems, group recommender systems are designed to generate and explain recommendations for groups. This group-oriented setting introduces additional complexities, as several factors - absent in individual contexts - must be addressed. These include understanding group dynamics (e.g., social dependencies within the group), defining effective decision-making processes, ensuring that recommendations are suitable for all group members, and providing group-level explanations as well as explanations for individual users. In this paper, we analyze in which way large language models (LLMs) can support these aspects and help to increase the overall decision support quality and applicability of group recommender systems. 

---
# SelfRACG: Enabling LLMs to Self-Express and Retrieve for Code Generation 

**Authors**: Qian Dong, Jia Chen, Qingyao Ai, Hongning Wang, Haitao Li, Yi Wu, Yao Hu, Yiqun Liu, Shaoping Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.19033)  

**Abstract**: Existing retrieval-augmented code generation (RACG) methods typically use an external retrieval module to fetch semantically similar code snippets used for generating subsequent fragments. However, even for consecutive code fragments, the content often diverges due to logical progression, resulting in a content gap. This gap undermines the performance of current RACG methods, as \textit{external} retrieval modules based on content matching fail to infer the specific information need of LLMs to generate the next code fragment. Therefore, we propose \textbf{SelfRACG}, a novel paradigm that enables large language models (LLMs) to \textbf{Self}-express their information needs to enhance \textbf{RACG}. Specifically, SelfRACG includes an information need expression module and a two-stage information need-guided training strategy, which encourages LLMs to express their information need. Extensive experiments demonstrate that SelfRACG can retrieve external knowledge that better aligns with the LLM's own information needs, resulting in superior generation performance compared to vanilla RACG. 

---
# Integrating LLM in Agent-Based Social Simulation: Opportunities and Challenges 

**Authors**: Patrick Taillandier, Jean Daniel Zucker, Arnaud Grignard, Benoit Gaudou, Nghi Quang Huynh, Alexis Drogoul  

**Link**: [PDF](https://arxiv.org/pdf/2507.19364)  

**Abstract**: This position paper examines the use of Large Language Models (LLMs) in social simulation, analyzing both their potential and their limitations from a computational social science perspective. The first part reviews recent findings on the ability of LLMs to replicate key aspects of human cognition, including Theory of Mind reasoning and social inference, while also highlighting significant limitations such as cognitive biases, lack of true understanding, and inconsistencies in behavior. The second part surveys emerging applications of LLMs in multi-agent simulation frameworks, focusing on system architectures, scale, and validation strategies. Notable projects such as Generative Agents (Smallville) and AgentSociety are discussed in terms of their design choices, empirical grounding, and methodological innovations. Particular attention is given to the challenges of behavioral fidelity, calibration, and reproducibility in large-scale LLM-driven simulations. The final section distinguishes between contexts where LLMs, like other black-box systems, offer direct value-such as interactive simulations and serious games-and those where their use is more problematic, notably in explanatory or predictive modeling. The paper concludes by advocating for hybrid approaches that integrate LLMs into traditional agent-based modeling platforms (GAMA, Netlogo, etc), enabling modelers to combine the expressive flexibility of language-based reasoning with the transparency and analytical rigor of classical rule-based systems. 

---
# Initial Steps in Integrating Large Reasoning and Action Models for Service Composition 

**Authors**: Ilche Georgievski, Marco Aiello  

**Link**: [PDF](https://arxiv.org/pdf/2507.18775)  

**Abstract**: Service composition remains a central challenge in building adaptive and intelligent software systems, often constrained by limited reasoning capabilities or brittle execution mechanisms. This paper explores the integration of two emerging paradigms enabled by large language models: Large Reasoning Models (LRMs) and Large Action Models (LAMs). We argue that LRMs address the challenges of semantic reasoning and ecosystem complexity while LAMs excel in dynamic action execution and system interoperability. However, each paradigm has complementary limitations - LRMs lack grounded action capabilities, and LAMs often struggle with deep reasoning. We propose an integrated LRM-LAM architectural framework as a promising direction for advancing automated service composition. Such a system can reason about service requirements and constraints while dynamically executing workflows, thus bridging the gap between intention and execution. This integration has the potential to transform service composition into a fully automated, user-friendly process driven by high-level natural language intent. 

---
# Advancing Event Forecasting through Massive Training of Large Language Models: Challenges, Solutions, and Broader Impacts 

**Authors**: Sang-Woo Lee, Sohee Yang, Donghyun Kwak, Noah Y. Siegel  

**Link**: [PDF](https://arxiv.org/pdf/2507.19477)  

**Abstract**: Many recent papers have studied the development of superforecaster-level event forecasting LLMs. While methodological problems with early studies cast doubt on the use of LLMs for event forecasting, recent studies with improved evaluation methods have shown that state-of-the-art LLMs are gradually reaching superforecaster-level performance, and reinforcement learning has also been reported to improve future forecasting. Additionally, the unprecedented success of recent reasoning models and Deep Research-style models suggests that technology capable of greatly improving forecasting performance has been developed. Therefore, based on these positive recent trends, we argue that the time is ripe for research on large-scale training of superforecaster-level event forecasting LLMs. We discuss two key research directions: training methods and data acquisition. For training, we first introduce three difficulties of LLM-based event forecasting training: noisiness-sparsity, knowledge cut-off, and simple reward structure problems. Then, we present related ideas to mitigate these problems: hypothetical event Bayesian networks, utilizing poorly-recalled and counterfactual events, and auxiliary reward signals. For data, we propose aggressive use of market, public, and crawling datasets to enable large-scale training and evaluation. Finally, we explain how these technical advances could enable AI to provide predictive intelligence to society in broader areas. This position paper presents promising specific paths and considerations for getting closer to superforecaster-level AI technology, aiming to call for researchers' interest in these directions. 

---
# Smooth Reading: Bridging the Gap of Recurrent LLM to Self-Attention LLM on Long-Context Tasks 

**Authors**: Kai Liu, Zhan Su, Peijie Dong, Fengran Mo, Jianfei Gao, ShaoTing Zhang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.19353)  

**Abstract**: Recently, recurrent large language models (Recurrent LLMs) with linear computational complexity have re-emerged as efficient alternatives to self-attention-based LLMs (Self-Attention LLMs), which have quadratic complexity. However, Recurrent LLMs often underperform on long-context tasks due to their limited fixed-size memory. Previous research has primarily focused on enhancing the memory capacity of Recurrent LLMs through architectural innovations, but these approaches have not yet enabled Recurrent LLMs to match the performance of Self-Attention LLMs on long-context tasks. We argue that this limitation arises because processing the entire context at once is not well-suited for Recurrent LLMs. In this paper, we propose Smooth Reading, a chunk-wise inference method inspired by human reading strategies. Smooth Reading processes context in chunks and iteratively summarizes the contextual information, thereby reducing memory demands and making the approach more compatible with Recurrent LLMs. Our experimental results show that this method substantially narrows the performance gap between Recurrent and Self-Attention LLMs on long-context tasks, while preserving the efficiency advantages of Recurrent LLMs. Our Smooth Reading boosts SWA-3B-4k (a Recurrent LLM) from 5.68% lower to 3.61% higher performance than Self-Attention LLMs on LongBench. Besides, our method maintains the high efficiency, training 3x faster and inferring 2x faster at 64k context compared to Self-Attention LLMs. To our knowledge, this is the first work to achieve comparable performance using Recurrent LLMs compared with Self-Attention LLMs on long-context tasks. We hope our method will inspire future research in this area. To facilitate further progress, we will release code and dataset. 

---
# GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning 

**Authors**: Lakshya A Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav Singhvi, Herumb Shandilya, Michael J Ryan, Meng Jiang, Christopher Potts, Koushik Sen, Alexandros G. Dimakis, Ion Stoica, Dan Klein, Matei Zaharia, Omar Khattab  

**Link**: [PDF](https://arxiv.org/pdf/2507.19457)  

**Abstract**: Large language models (LLMs) are increasingly adapted to downstream tasks via reinforcement learning (RL) methods like Group Relative Policy Optimization (GRPO), which often require thousands of rollouts to learn new tasks. We argue that the interpretable nature of language can often provide a much richer learning medium for LLMs, compared with policy gradients derived from sparse, scalar rewards. To test this, we introduce GEPA (Genetic-Pareto), a prompt optimizer that thoroughly incorporates natural language reflection to learn high-level rules from trial and error. Given any AI system containing one or more LLM prompts, GEPA samples system-level trajectories (e.g., reasoning, tool calls, and tool outputs) and reflects on them in natural language to diagnose problems, propose and test prompt updates, and combine complementary lessons from the Pareto frontier of its own attempts. As a result of GEPA's design, it can often turn even just a few rollouts into a large quality gain. Across four tasks, GEPA outperforms GRPO by 10% on average and by up to 20%, while using up to 35x fewer rollouts. GEPA also outperforms the leading prompt optimizer, MIPROv2, by over 10% across two LLMs, and demonstrates promising results as an inference-time search strategy for code optimization. 

---
# Step-3 is Large yet Affordable: Model-system Co-design for Cost-effective Decoding 

**Authors**: StepFun, Bin Wang, Bojun Wang, Changyi Wan, Guanzhe Huang, Hanpeng Hu, Haonan Jia, Hao Nie, Mingliang Li, Nuo Chen, Siyu Chen, Song Yuan, Wuxun Xie, Xiaoniu Song, Xing Chen, Xingping Yang, Xuelin Zhang, Yanbo Yu, Yaoyu Wang, Yibo Zhu, Yimin Jiang, Yu Zhou, Yuanwei Lu, Houyi Li, Jingcheng Hu, Ka Man Lo, Ailin Huang, Binxing Jiao, Bo Li, Boyu Chen, Changxin Miao, Chang Lou, Chen Hu, Chen Xu, Chenfeng Yu, Chengyuan Yao, Daokuan Lv, Dapeng Shi, Deshan Sun, Ding Huang, Dingyuan Hu, Dongqing Pang, Enle Liu, Fajie Zhang, Fanqi Wan, Gulin Yan, Han Zhang, Han Zhou, Hanghao Wu, Hangyu Guo, Hanqi Chen, Hanshan Zhang, Hao Wu, Haocheng Zhang, Haolong Yan, Haoran Lv, Haoran Wei, Hebin Zhou, Heng Wang, Heng Wang, Hongxin Li, Hongyu Zhou, Hongyuan Wang, Huiyong Guo, Jia Wang, Jiahao Gong, Jialing Xie, Jian Zhou, Jianjian Sun, Jiaoren Wu, Jiaran Zhang, Jiayu Liu, Jie Cheng, Jie Luo, Jie Yan, Jie Yang, Jieyi Hou, Jinguang Zhang, Jinlan Cao, Jisheng Yin, Junfeng Liu, Junhao Huang, Junzhe Lin, Kaijun Tan, Kaixiang Li, Kang An, Kangheng Lin, Kenkun Liu, Lei Yang, Liang Zhao, Liangyu Chen, Lieyu Shi, Liguo Tan, Lin Lin, Lin Zhang, Lina Chen, Liwen Huang, Liying Shi, Longlong Gu, Mei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.19427)  

**Abstract**: Large language models (LLMs) face low hardware efficiency during decoding, especially for long-context reasoning tasks. This paper introduces Step-3, a 321B-parameter VLM with hardware-aware model-system co-design optimized for minimizing decoding costs. Step-3 innovates in two key dimensions: (1) A novel Multi-Matrix Factorization Attention (MFA) mechanism that significantly reduces both KV cache size and computation while maintaining high attention expressiveness, and (2) Attention-FFN Disaggregation (AFD), a distributed inference system that decouples attention and Feed-Forward Network (FFN) layers into specialized subsystems. This co-design achieves unprecedented cost efficiency: Step-3 significantly reduces theoretical decoding costs compared with models like DeepSeek-V3 and Qwen3 MoE 235B, with the gains widening at longer context. Step-3 achieves low cost while activating 38B parameters per token (more than DeepSeek-V3 and Qwen3 MoE 235B), demonstrating that hardware-aligned attention arithmetic intensity, MoE sparsity, and AFD are critical to cost-effectiveness. We perform a head-to-head comparison with DeepSeek-V3 in its favorable scenarios. Our implementation on Hopper GPUs achieves a decoding throughput of up to 4,039 tokens per second per GPU under 50ms TPOT SLA (4K context, FP8, no MTP). It is higher than DeepSeek-V3's 2,324 in the same setup and sets a new Pareto frontier for LLM decoding. 

---
# ReCatcher: Towards LLMs Regression Testing for Code Generation 

**Authors**: Altaf Allah Abbassi, Leuson Da Silva, Amin Nikanjam, Foutse Khomh  

**Link**: [PDF](https://arxiv.org/pdf/2507.19390)  

**Abstract**: Large Language Models (LLMs) for code generation evolve rapidly through fine-tuning, merging, or new model releases. However, such updates can introduce regressions, not only in correctness but also in code quality and performance. To address this, we present ReCatcher, a regression testing framework for Python code generation. ReCatcher systematically compares two LLMs, typically a current model and a candidate update, across three dimensions: logical correctness, static code quality, and execution performance. We apply ReCatcher to assess regressions across three update scenarios, fine-tuning, merging, and model release, using CodeLlama, DeepSeek-Coder, and GPT-4o. Our evaluation shows that fine-tuning with cross-language datasets increases syntax errors by up to 12%. Merging with general-purpose models like Llama2 leads to regressions in correctness by up to 18%. GPT-4o introduces regressions of up to 50% in handling missing imports compared to GPT-3.5-turbo, while GPT-4o-mini suffers up to 80% performance degradation in execution time versus GPT-4o. Overall, logical correctness, performance, and error handling (e.g., syntax errors and missing imports) are the most regression-prone areas. Comparing ReCatcher with baseline solutions, it presents better and consistent accuracy across logical and performance aspects. ReCatcher highlights the importance of systematic regression evaluation before adopting new models, while assisting researchers and practitioners in making more informed update decisions. 

---
# Can Small-Scale Data Poisoning Exacerbate Dialect-Linked Biases in Large Language Models? 

**Authors**: Chaymaa Abbas, Mariette Awad, Razane Tajeddine  

**Link**: [PDF](https://arxiv.org/pdf/2507.19195)  

**Abstract**: Despite the ongoing improvements in the design of large language models (LLMs) to foster inclusion and balanced responses, these systems remain susceptible to encoding and amplifying social biases. This study examines how dialectal variation, specifically African American Vernacular English (AAVE) versus Standard American English (SAE), interacts with data poisoning to influence toxicity in outputs. Using both small- and medium-scale LLaMA models, we show that even minimal exposure to poisoned data significantly increases toxicity for AAVE inputs, while it remains comparatively unaffected for SAE. Larger models exhibit a more significant amplification effect which suggests heightened susceptibility with scale. To further assess these disparities, we employed GPT-4o as a fairness auditor, which identified harmful stereotypical patterns disproportionately tied to AAVE inputs, including portrayals of aggression, criminality, and intellectual inferiority. These findings underscore the compounding impact of data poisoning and dialectal bias and emphasize the need for dialect-aware evaluation, targeted debiasing interventions, and socially responsible training protocols during development. 

---
# Automated Code Review Using Large Language Models at Ericsson: An Experience Report 

**Authors**: Shweta Ramesh, Joy Bose, Hamender Singh, A K Raghavan, Sujoy Roychowdhury, Giriprasad Sridhara, Nishrith Saini, Ricardo Britto  

**Link**: [PDF](https://arxiv.org/pdf/2507.19115)  

**Abstract**: Code review is one of the primary means of assuring the quality of released software along with testing and static analysis. However, code review requires experienced developers who may not always have the time to perform an in-depth review of code. Thus, automating code review can help alleviate the cognitive burden on experienced software developers allowing them to focus on their primary activities of writing code to add new features and fix bugs. In this paper, we describe our experience in using Large Language Models towards automating the code review process in Ericsson. We describe the development of a lightweight tool using LLMs and static program analysis. We then describe our preliminary experiments with experienced developers in evaluating our code review tool and the encouraging results. 

---
# SpeechIQ: Speech Intelligence Quotient Across Cognitive Levels in Voice Understanding Large Language Models 

**Authors**: Zhen Wan, Chao-Han Huck Yang, Yahan Yu, Jinchuan Tian, Sheng Li, Ke Hu, Zhehuai Chen, Shinji Watanabe, Fei Cheng, Chenhui Chu, Sadao Kurohashi  

**Link**: [PDF](https://arxiv.org/pdf/2507.19361)  

**Abstract**: We introduce Speech-based Intelligence Quotient (SIQ) as a new form of human cognition-inspired evaluation pipeline for voice understanding large language models, LLM Voice, designed to assess their voice understanding ability. Moving beyond popular voice understanding metrics such as word error rate (WER), SIQ examines LLM Voice across three cognitive levels motivated by Bloom's Taxonomy: (1) Remembering (i.e., WER for verbatim accuracy); (2) Understanding (i.e., similarity of LLM's interpretations); and (3) Application (i.e., QA accuracy for simulating downstream tasks). We demonstrate that SIQ not only quantifies voice understanding abilities but also provides unified comparisons between cascaded methods (e.g., ASR LLM) and end-to-end models, identifies annotation errors in existing benchmarks, and detects hallucinations in LLM Voice. Our framework represents a first-of-its-kind intelligence examination that bridges cognitive principles with voice-oriented benchmarks, while exposing overlooked challenges in multi-modal training. 

---
# PrompTrend: Continuous Community-Driven Vulnerability Discovery and Assessment for Large Language Models 

**Authors**: Tarek Gasmi, Ramzi Guesmi, Mootez Aloui, Jihene Bennaceur  

**Link**: [PDF](https://arxiv.org/pdf/2507.19185)  

**Abstract**: Static benchmarks fail to capture LLM vulnerabilities emerging through community experimentation in online forums. We present PrompTrend, a system that collects vulnerability data across platforms and evaluates them using multidimensional scoring, with an architecture designed for scalable monitoring. Cross-sectional analysis of 198 vulnerabilities collected from online communities over a five-month period (January-May 2025) and tested on nine commercial models reveals that advanced capabilities correlate with increased vulnerability in some architectures, psychological attacks significantly outperform technical exploits, and platform dynamics shape attack effectiveness with measurable model-specific patterns. The PrompTrend Vulnerability Assessment Framework achieves 78% classification accuracy while revealing limited cross-model transferability, demonstrating that effective LLM security requires comprehensive socio-technical monitoring beyond traditional periodic assessment. Our findings challenge the assumption that capability advancement improves security and establish community-driven psychological manipulation as the dominant threat vector for current language models. 

---
# Solar Photovoltaic Assessment with Large Language Model 

**Authors**: Muhao Guo, Yang Weng  

**Link**: [PDF](https://arxiv.org/pdf/2507.19144)  

**Abstract**: Accurate detection and localization of solar photovoltaic (PV) panels in satellite imagery is essential for optimizing microgrids and active distribution networks (ADNs), which are critical components of renewable energy systems. Existing methods lack transparency regarding their underlying algorithms or training datasets, rely on large, high-quality PV training data, and struggle to generalize to new geographic regions or varied environmental conditions without extensive re-training. These limitations lead to inconsistent detection outcomes, hindering large-scale deployment and data-driven grid optimization. In this paper, we investigate how large language models (LLMs) can be leveraged to overcome these challenges. Despite their promise, LLMs face several challenges in solar panel detection, including difficulties with multi-step logical processes, inconsistent output formatting, frequent misclassification of visually similar objects (e.g., shadows, parking lots), and low accuracy in complex tasks such as spatial localization and quantification. To overcome these issues, we propose the PV Assessment with LLMs (PVAL) framework, which incorporates task decomposition for more efficient workflows, output standardization for consistent and scalable formatting, few-shot prompting to enhance classification accuracy, and fine-tuning using curated PV datasets with detailed annotations. PVAL ensures transparency, scalability, and adaptability across heterogeneous datasets while minimizing computational overhead. By combining open-source accessibility with robust methodologies, PVAL establishes an automated and reproducible pipeline for solar panel detection, paving the way for large-scale renewable energy integration and optimized grid management. 

---
# TreeReader: A Hierarchical Academic Paper Reader Powered by Language Models 

**Authors**: Zijian Zhang, Pan Chen, Fangshi Du, Runlong Ye, Oliver Huang, Michael Liut, Alán Aspuru-Guzik  

**Link**: [PDF](https://arxiv.org/pdf/2507.18945)  

**Abstract**: Efficiently navigating and understanding academic papers is crucial for scientific progress. Traditional linear formats like PDF and HTML can cause cognitive overload and obscure a paper's hierarchical structure, making it difficult to locate key information. While LLM-based chatbots offer summarization, they often lack nuanced understanding of specific sections, may produce unreliable information, and typically discard the document's navigational structure. Drawing insights from a formative study on academic reading practices, we introduce TreeReader, a novel language model-augmented paper reader. TreeReader decomposes papers into an interactive tree structure where each section is initially represented by an LLM-generated concise summary, with underlying details accessible on demand. This design allows users to quickly grasp core ideas, selectively explore sections of interest, and verify summaries against the source text. A user study was conducted to evaluate TreeReader's impact on reading efficiency and comprehension. TreeReader provides a more focused and efficient way to navigate and understand complex academic literature by bridging hierarchical summarization with interactive exploration. 

---
# A Toolbox, Not a Hammer -- Multi-TAG: Scaling Math Reasoning with Multi-Tool Aggregation 

**Authors**: Bohan Yao, Vikas Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2507.18973)  

**Abstract**: Augmenting large language models (LLMs) with external tools is a promising avenue for developing high-performance mathematical reasoning systems. Prior tool-augmented approaches typically finetune an LLM to select and invoke a single tool at each reasoning step and show promising results on simpler math reasoning benchmarks such as GSM8K. However, these approaches struggle with more complex math problems that require precise reasoning over multiple steps. To address this limitation, in this work, we propose Multi-TAG, a Multi-Tool AGgregation-based framework. Instead of relying on a single tool, Multi-TAG guides an LLM to concurrently invoke multiple tools at each reasoning step. It then aggregates their diverse outputs to verify and refine the reasoning process, enhancing solution robustness and accuracy. Notably, Multi-TAG is a finetuning-free, inference-only framework, making it readily applicable to any LLM backbone, including large open-weight models which are computationally expensive to finetune and proprietary frontier models which cannot be finetuned with custom recipes. We evaluate Multi-TAG on four challenging benchmarks: MATH500, AIME, AMC, and OlympiadBench. Across both open-weight and closed-source LLM backbones, Multi-TAG consistently and substantially outperforms state-of-the-art baselines, achieving average improvements of 6.0% to 7.5% over state-of-the-art baselines. 

---
# MemoCoder: Automated Function Synthesis using LLM-Supported Agents 

**Authors**: Yiping Jia, Zhen Ming Jiang, Shayan Noei, Ying Zou  

**Link**: [PDF](https://arxiv.org/pdf/2507.18812)  

**Abstract**: With the widespread adoption of Large Language Models (LLMs) such as GitHub Copilot and ChatGPT, developers increasingly rely on AI-assisted tools to support code generation. While LLMs can generate syntactically correct solutions for well-structured programming tasks, they often struggle with challenges that require iterative debugging, error handling, or adaptation to diverse problem structures. Existing approaches such as fine-tuning or self-repair strategies either require costly retraining or lack mechanisms to accumulate and reuse knowledge from previous attempts.
To address these limitations, we propose MemoCoder, a multi-agent framework that enables collaborative problem solving and persistent learning from past fixes. At the core of MemoCoder is a Fixing Knowledge Set, which stores successful repairs and supports retrieval for future tasks. A central Mentor Agent supervises the repair process by identifying recurring error patterns and refining high-level fixing strategies, providing a novel supervisory role that guides the self-repair loop. We evaluate MemoCoder across three public benchmarks -- MBPP, HumanEval, and LiveCodeBench -- spanning a range of problem complexities. Experimental results show that MemoCoder consistently outperforms both zero-shot prompting and a Self-Repair strategy, with improvements ranging from 3.1% to 12.1% in Pass@10 and from 1.4% to 14.5% in Pass@50, demonstrating its effectiveness in iterative refinement and knowledge-guided code generation. 

---
# MGHFT: Multi-Granularity Hierarchical Fusion Transformer for Cross-Modal Sticker Emotion Recognition 

**Authors**: Jian Chen, Yuxuan Hu, Haifeng Lu, Wei Wang, Min Yang, Chengming Li, Xiping Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.18929)  

**Abstract**: Although pre-trained visual models with text have demonstrated strong capabilities in visual feature extraction, sticker emotion understanding remains challenging due to its reliance on multi-view information, such as background knowledge and stylistic cues. To address this, we propose a novel multi-granularity hierarchical fusion transformer (MGHFT), with a multi-view sticker interpreter based on Multimodal Large Language Models. Specifically, inspired by the human ability to interpret sticker emotions from multiple views, we first use Multimodal Large Language Models to interpret stickers by providing rich textual context via multi-view descriptions. Then, we design a hierarchical fusion strategy to fuse the textual context into visual understanding, which builds upon a pyramid visual transformer to extract both global and local sticker features at multiple stages. Through contrastive learning and attention mechanisms, textual features are injected at different stages of the visual backbone, enhancing the fusion of global- and local-granularity visual semantics with textual guidance. Finally, we introduce a text-guided fusion attention mechanism to effectively integrate the overall multimodal features, enhancing semantic understanding. Extensive experiments on 2 public sticker emotion datasets demonstrate that MGHFT significantly outperforms existing sticker emotion recognition approaches, achieving higher accuracy and more fine-grained emotion recognition. Compared to the best pre-trained visual models, our MGHFT also obtains an obvious improvement, 5.4% on F1 and 4.0% on accuracy. The code is released at this https URL. 

---
# DxHF: Providing High-Quality Human Feedback for LLM Alignment via Interactive Decomposition 

**Authors**: Danqing Shi, Furui Cheng, Tino Weinkauf, Antti Oulasvirta, Mennatallah El-Assady  

**Link**: [PDF](https://arxiv.org/pdf/2507.18802)  

**Abstract**: Human preferences are widely used to align large language models (LLMs) through methods such as reinforcement learning from human feedback (RLHF). However, the current user interfaces require annotators to compare text paragraphs, which is cognitively challenging when the texts are long or unfamiliar. This paper contributes by studying the decomposition principle as an approach to improving the quality of human feedback for LLM alignment. This approach breaks down the text into individual claims instead of directly comparing two long-form text responses. Based on the principle, we build a novel user interface DxHF. It enhances the comparison process by showing decomposed claims, visually encoding the relevance of claims to the conversation and linking similar claims. This allows users to skim through key information and identify differences for better and quicker judgment. Our technical evaluation shows evidence that decomposition generally improves feedback accuracy regarding the ground truth, particularly for users with uncertainty. A crowdsourcing study with 160 participants indicates that using DxHF improves feedback accuracy by an average of 5%, although it increases the average feedback time by 18 seconds. Notably, accuracy is significantly higher in situations where users have less certainty. The finding of the study highlights the potential of HCI as an effective method for improving human-AI alignment. 

---
# Innovator: Scientific Continued Pretraining with Fine-grained MoE Upcycling 

**Authors**: Ning Liao, Xiaoxing Wang, Zehao Lin, Weiyang Guo, Feng Hong, Shixiang Song, Geng Yu, Zihua Zhao, Sitao Xie, Longxuan Wei, Xiangqi Jin, Xiaohan Qin, Jiale Ma, Kai Chen, Jiangchao Yao, Zhouhan Lin, Junchi Yan, Zhiyu Li, Feiyu Xiong, Yanfeng Wang, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.18671)  

**Abstract**: A large language model (LLM) with knowledge in both scientific and general tasks is the foundation of science general intelligence. However, directly continued pretraining an LLM using science data usually leads to catastrophic forgetting, which indicates severe degradation in general ability. In this report, we present Innovator, which solves this problem by upcycling a pre-trained dense LLM into a fine-grained Mixtures-of-Experts model during continued pretraining, where different experts are expected to learn science knowledge in different disciplines, and a shared expert is utilized for general tasks. Innovator introduces a four-stage upcycle training paradigm: (1) Scientific Expert Induction on discipline-specific data, (2) Fine-grained Expert Splitting via FFN dimension decomposition, (3) Science-Aware Routing warmup, and (4) Generalist-Scientist Integration training on hybrid datasets. Such a paradigm enables knowledge in the general domain, and different scientific disciplines can be decoupled, avoiding the negative influence among knowledge in different domains. With 53.3B total parameters and 13.3B activated, Innovator extends Qwen2.5-7B using a shared general expert and 64 specialized scientific experts with 8 activated. Trained on 300B tokens with tri-level quality-controlled data, Innovator achieves 25% average improvement across 30 scientific tasks with a win rate as 70%, while retaining 99% performance in general tasks. Furthermore, Innovator-Reason, which is post-trained from Innovator for reasoning boosting, exhibits excellent reasoning performance in solving complex scientific problems with improvements over 30%. 

---
# Identifying Fine-grained Forms of Populism in Political Discourse: A Case Study on Donald Trump's Presidential Campaigns 

**Authors**: Ilias Chalkidis, Stephanie Brandl, Paris Aslanidis  

**Link**: [PDF](https://arxiv.org/pdf/2507.19303)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of instruction-following tasks, yet their grasp of nuanced social science concepts remains underexplored. This paper examines whether LLMs can identify and classify fine-grained forms of populism, a complex and contested concept in both academic and media debates. To this end, we curate and release novel datasets specifically designed to capture populist discourse. We evaluate a range of pre-trained (large) language models, both open-weight and proprietary, across multiple prompting paradigms. Our analysis reveals notable variation in performance, highlighting the limitations of LLMs in detecting populist discourse. We find that a fine-tuned RoBERTa classifier vastly outperforms all new-era instruction-tuned LLMs, unless fine-tuned. Additionally, we apply our best-performing model to analyze campaign speeches by Donald Trump, extracting valuable insights into his strategic use of populist rhetoric. Finally, we assess the generalizability of these models by benchmarking them on campaign speeches by European politicians, offering a lens into cross-context transferability in political discourse analysis. In this setting, we find that instruction-tuned LLMs exhibit greater robustness on out-of-domain data. 

---
# How Much Do Large Language Model Cheat on Evaluation? Benchmarking Overestimation under the One-Time-Pad-Based Framework 

**Authors**: Zi Liang, Liantong Yu, Shiyu Zhang, Qingqing Ye, Haibo Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.19219)  

**Abstract**: Overestimation in evaluating large language models (LLMs) has become an increasing concern. Due to the contamination of public benchmarks or imbalanced model training, LLMs may achieve unreal evaluation results on public benchmarks, either intentionally or unintentionally, which leads to unfair comparisons among LLMs and undermines their realistic capability assessments. Existing benchmarks attempt to address these issues by keeping test cases permanently secret, mitigating contamination through human evaluation, or repeatedly collecting and constructing new samples. However, these approaches fail to ensure reproducibility, transparency, and high efficiency simultaneously. Moreover, the extent of overestimation in current LLMs remains unquantified. To address these issues, we propose ArxivRoll, a dynamic evaluation framework inspired by one-time pad encryption in cryptography. ArxivRoll comprises two key components: \emph{i) SCP (Sequencing, Cloze, and Prediction)}, an automated generator for private test cases, and \emph{ii) Rugged Scores (RS)}, metrics that measure the proportion of public benchmark contamination and training bias. Leveraging SCP, ArxivRoll constructs a new benchmark every six months using recent articles from ArXiv and employs them for one-time evaluations of LLM performance. Extensive experiments demonstrate the high quality of our benchmark, and we provide a systematic evaluation of current LLMs. The source code is available at this https URL. 

---
# Arg-LLaDA: Argument Summarization via Large Language Diffusion Models and Sufficiency-Aware Refinement 

**Authors**: Hao Li, Yizheng Sun, Viktor Schlegel, Kailai Yang, Riza Batista-Navarro, Goran Nenadic  

**Link**: [PDF](https://arxiv.org/pdf/2507.19081)  

**Abstract**: Argument summarization aims to generate concise, structured representations of complex, multi-perspective debates. While recent work has advanced the identification and clustering of argumentative components, the generation stage remains underexplored. Existing approaches typically rely on single-pass generation, offering limited support for factual correction or structural refinement. To address this gap, we introduce Arg-LLaDA, a novel large language diffusion framework that iteratively improves summaries via sufficiency-guided remasking and regeneration. Our method combines a flexible masking controller with a sufficiency-checking module to identify and revise unsupported, redundant, or incomplete spans, yielding more faithful, concise, and coherent outputs. Empirical results on two benchmark datasets demonstrate that Arg-LLaDA surpasses state-of-the-art baselines in 7 out of 10 automatic evaluation metrics. In addition, human evaluations reveal substantial improvements across core dimensions, coverage, faithfulness, and conciseness, validating the effectiveness of our iterative, sufficiency-aware generation strategy. 

---
# AutoPCR: Automated Phenotype Concept Recognition by Prompting 

**Authors**: Yicheng Tao, Yuanhao Huang, Jie Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.19315)  

**Abstract**: Phenotype concept recognition (CR) is a fundamental task in biomedical text mining, enabling applications such as clinical diagnostics and knowledge graph construction. However, existing methods often require ontology-specific training and struggle to generalize across diverse text types and evolving biomedical terminology. We present AutoPCR, a prompt-based phenotype CR method that does not require ontology-specific training. AutoPCR performs CR in three stages: entity extraction using a hybrid of rule-based and neural tagging strategies, candidate retrieval via SapBERT, and entity linking through prompting a large language model. Experiments on four benchmark datasets show that AutoPCR achieves the best average and most robust performance across both mention-level and document-level evaluations, surpassing prior state-of-the-art methods. Further ablation and transfer studies demonstrate its inductive capability and generalizability to new ontologies. 

---
# Debating Truth: Debate-driven Claim Verification with Multiple Large Language Model Agents 

**Authors**: Haorui He, Yupeng Li, Dacheng Wen, Reynold Cheng, Francis C. M. Lau  

**Link**: [PDF](https://arxiv.org/pdf/2507.19090)  

**Abstract**: Claim verification is critical for enhancing digital literacy. However, the state-of-the-art single-LLM methods struggle with complex claim verification that involves multi-faceted evidences. Inspired by real-world fact-checking practices, we propose DebateCV, the first claim verification framework that adopts a debate-driven methodology using multiple LLM agents. In our framework, two Debaters take opposing stances on a claim and engage in multi-round argumentation, while a Moderator evaluates the arguments and renders a verdict with justifications. To further improve the performance of the Moderator, we introduce a novel post-training strategy that leverages synthetic debate data generated by the zero-shot DebateCV, effectively addressing the scarcity of real-world debate-driven claim verification data. Experimental results show that our method outperforms existing claim verification methods under varying levels of evidence quality. Our code and dataset are publicly available at this https URL. 

---
# A Systematic Review of Key Retrieval-Augmented Generation (RAG) Systems: Progress, Gaps, and Future Directions 

**Authors**: Agada Joseph Oche, Ademola Glory Folashade, Tirthankar Ghosal, Arpan Biswas  

**Link**: [PDF](https://arxiv.org/pdf/2507.18910)  

**Abstract**: Retrieval-Augmented Generation (RAG) represents a major advancement in natural language processing (NLP), combining large language models (LLMs) with information retrieval systems to enhance factual grounding, accuracy, and contextual relevance. This paper presents a comprehensive systematic review of RAG, tracing its evolution from early developments in open domain question answering to recent state-of-the-art implementations across diverse applications. The review begins by outlining the motivations behind RAG, particularly its ability to mitigate hallucinations and outdated knowledge in parametric models. Core technical components-retrieval mechanisms, sequence-to-sequence generation models, and fusion strategies are examined in detail. A year-by-year analysis highlights key milestones and research trends, providing insight into RAG's rapid growth. The paper further explores the deployment of RAG in enterprise systems, addressing practical challenges related to retrieval of proprietary data, security, and scalability. A comparative evaluation of RAG implementations is conducted, benchmarking performance on retrieval accuracy, generation fluency, latency, and computational efficiency. Persistent challenges such as retrieval quality, privacy concerns, and integration overhead are critically assessed. Finally, the review highlights emerging solutions, including hybrid retrieval approaches, privacy-preserving techniques, optimized fusion strategies, and agentic RAG architectures. These innovations point toward a future of more reliable, efficient, and context-aware knowledge-intensive NLP systems. 

---
# Large language models provide unsafe answers to patient-posed medical questions 

**Authors**: Rachel L. Draelos, Samina Afreen, Barbara Blasko, Tiffany Brazile, Natasha Chase, Dimple Desai, Jessica Evert, Heather L. Gardner, Lauren Herrmann, Aswathy Vaikom House, Stephanie Kass, Marianne Kavan, Kirshma Khemani, Amanda Koire, Lauren M. McDonald, Zahraa Rabeeah, Amy Shah  

**Link**: [PDF](https://arxiv.org/pdf/2507.18905)  

**Abstract**: Millions of patients are already using large language model (LLM) chatbots for medical advice on a regular basis, raising patient safety concerns. This physician-led red-teaming study compares the safety of four publicly available chatbots--Claude by Anthropic, Gemini by Google, GPT-4o by OpenAI, and Llama3-70B by Meta--on a new dataset, HealthAdvice, using an evaluation framework that enables quantitative and qualitative analysis. In total, 888 chatbot responses are evaluated for 222 patient-posed advice-seeking medical questions on primary care topics spanning internal medicine, women's health, and pediatrics. We find statistically significant differences between chatbots. The rate of problematic responses varies from 21.6 percent (Claude) to 43.2 percent (Llama), with unsafe responses varying from 5 percent (Claude) to 13 percent (GPT-4o, Llama). Qualitative results reveal chatbot responses with the potential to lead to serious patient harm. This study suggests that millions of patients could be receiving unsafe medical advice from publicly available chatbots, and further work is needed to improve the clinical safety of these powerful tools. 

---
# SLoW: Select Low-frequency Words! Automatic Dictionary Selection for Translation on Large Language Models 

**Authors**: Hongyuan Lu, Zixuan Li, Zefan Zhang, Wai Lam  

**Link**: [PDF](https://arxiv.org/pdf/2507.18902)  

**Abstract**: There are more than 7,000 languages around the world, and current Large Language Models (LLMs) only support hundreds of languages. Dictionary-based prompting methods can enhance translation on them, but most methods use all the available dictionaries, which could be expensive. Instead, it will be flexible to have a trade-off between token consumption and translation performance. This paper proposes a novel task called \textbf{A}utomatic \textbf{D}ictionary \textbf{S}election (\textbf{ADS}). The goal of the task is to automatically select which dictionary to use to enhance translation. We propose a novel and effective method which we call \textbf{S}elect \textbf{Lo}w-frequency \textbf{W}ords! (\textbf{SLoW}) which selects those dictionaries that have a lower frequency. Our methods have unique advantages. First, there is no need for access to the training data for frequency estimation (which is usually unavailable). Second, it inherits the advantage of dictionary-based methods, where no additional tuning is required on LLMs. Experimental results on 100 languages from FLORES indicate that SLoW surpasses strong baselines, and it can obviously save token usage, with many languages even surpassing the translation performance of the full dictionary baseline.\footnote{A shocking fact is that there is no need to use the actual training data (often unobtainable) for frequency estimation, and an estimation frequency obtained using public resources is still apparently effective in improving translation with ChatGPT and Llama, and DeepSeek.}\footnote{Code and data available upon publication.} 

---
# Evaluating Code-Mixing in LLMs Across 18 Languages 

**Authors**: Yilun Yang, Yekun Chai  

**Link**: [PDF](https://arxiv.org/pdf/2507.18791)  

**Abstract**: Code-mixing, the practice of switching between languages within a conversation, presents unique challenges for traditional natural language processing. Existing benchmarks, such as LinCE and GLUECoS, are limited by narrow language pairings and tasks, failing to adequately evaluate the code-mixing capabilities of large language models (LLMs). Despite the significance of code-mixing for multilingual users, research on LLMs in this context remains limited. Additionally, current methods for generating code-mixed data are underdeveloped. In this paper, we conduct a comprehensive evaluation of LLMs' performance on code-mixed data across 18 languages from seven language families. We also propose a novel approach for generating synthetic code-mixed texts by combining word substitution with GPT-4 prompting. Our analysis reveals consistent underperformance of LLMs on code-mixed datasets involving multiple language families. We suggest that improvements in training data size, model scale, and few-shot learning could enhance their performance. 

---
# MindFlow+: A Self-Evolving Agent for E-Commerce Customer Service 

**Authors**: Ming Gong, Xucheng Huang, Ziheng Xu, Vijayan K. Asari  

**Link**: [PDF](https://arxiv.org/pdf/2507.18884)  

**Abstract**: High-quality dialogue is crucial for e-commerce customer service, yet traditional intent-based systems struggle with dynamic, multi-turn interactions. We present MindFlow+, a self-evolving dialogue agent that learns domain-specific behavior by combining large language models (LLMs) with imitation learning and offline reinforcement learning (RL). MindFlow+ introduces two data-centric mechanisms to guide learning: tool-augmented demonstration construction, which exposes the model to knowledge-enhanced and agentic (ReAct-style) interactions for effective tool use; and reward-conditioned data modeling, which aligns responses with task-specific goals using reward signals. To evaluate the model's role in response generation, we introduce the AI Contribution Ratio, a novel metric quantifying AI involvement in dialogue. Experiments on real-world e-commerce conversations show that MindFlow+ outperforms strong baselines in contextual relevance, flexibility, and task accuracy. These results demonstrate the potential of combining LLMs tool reasoning, and reward-guided learning to build domain-specialized, context-aware dialogue systems. 

---
# Uncovering Cross-Linguistic Disparities in LLMs using Sparse Autoencoders 

**Authors**: Richmond Sin Jing Xuan, Jalil Huseynov, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.18918)  

**Abstract**: Multilingual large language models (LLMs) exhibit strong cross-linguistic generalization, yet medium to low resource languages underperform on common benchmarks such as ARC-Challenge, MMLU, and HellaSwag. We analyze activation patterns in Gemma-2-2B across all 26 residual layers and 10 languages: Chinese (zh), Russian (ru), Spanish (es), Italian (it), medium to low resource languages including Indonesian (id), Catalan (ca), Marathi (mr), Malayalam (ml), and Hindi (hi), with English (en) as the reference. Using Sparse Autoencoders (SAEs), we reveal systematic disparities in activation patterns. Medium to low resource languages receive up to 26.27 percent lower activations in early layers, with a persistent gap of 19.89 percent in deeper layers. To address this, we apply activation-aware fine-tuning via Low-Rank Adaptation (LoRA), leading to substantial activation gains, such as 87.69 percent for Malayalam and 86.32 percent for Hindi, while maintaining English retention at approximately 91 percent. After fine-tuning, benchmark results show modest but consistent improvements, highlighting activation alignment as a key factor in enhancing multilingual LLM performance. 

---
# FD-Bench: A Full-Duplex Benchmarking Pipeline Designed for Full Duplex Spoken Dialogue Systems 

**Authors**: Yizhou Peng, Yi-Wen Chao, Dianwen Ng, Yukun Ma, Chongjia Ni, Bin Ma, Eng Siong Chng  

**Link**: [PDF](https://arxiv.org/pdf/2507.19040)  

**Abstract**: Full-duplex spoken dialogue systems (FDSDS) enable more natural human-machine interactions by allowing real-time user interruptions and backchanneling, compared to traditional SDS that rely on turn-taking. However, existing benchmarks lack metrics for FD scenes, e.g., evaluating model performance during user interruptions. In this paper, we present a comprehensive FD benchmarking pipeline utilizing LLMs, TTS, and ASR to address this gap. It assesses FDSDS's ability to handle user interruptions, manage delays, and maintain robustness in challenging scenarios with diverse novel metrics. We applied our benchmark to three open-source FDSDS (Moshi, Freeze-omni, and VITA-1.5) using over 40 hours of generated speech, with 293 simulated conversations and 1,200 interruptions. The results show that all models continue to face challenges, such as failing to respond to user interruptions, under frequent disruptions and noisy conditions. Demonstrations, data, and code will be released. 

---
# Adaptive Learning Systems: Personalized Curriculum Design Using LLM-Powered Analytics 

**Authors**: Yongjie Li, Ruilin Nong, Jianan Liu, Lucas Evans  

**Link**: [PDF](https://arxiv.org/pdf/2507.18949)  

**Abstract**: Large language models (LLMs) are revolutionizing the field of education by enabling personalized learning experiences tailored to individual student needs. In this paper, we introduce a framework for Adaptive Learning Systems that leverages LLM-powered analytics for personalized curriculum design. This innovative approach uses advanced machine learning to analyze real-time data, allowing the system to adapt learning pathways and recommend resources that align with each learner's progress. By continuously assessing students, our framework enhances instructional strategies, ensuring that the materials presented are relevant and engaging. Experimental results indicate a marked improvement in both learner engagement and knowledge retention when using a customized curriculum. Evaluations conducted across varied educational environments demonstrate the framework's flexibility and positive influence on learning outcomes, potentially reshaping conventional educational practices into a more adaptive and student-centered model. 

---
# People Are Highly Cooperative with Large Language Models, Especially When Communication Is Possible or Following Human Interaction 

**Authors**: Paweł Niszczota, Tomasz Grzegorczyk, Alexander Pastukhov  

**Link**: [PDF](https://arxiv.org/pdf/2507.18639)  

**Abstract**: Machines driven by large language models (LLMs) have the potential to augment humans across various tasks, a development with profound implications for business settings where effective communication, collaboration, and stakeholder trust are paramount. To explore how interacting with an LLM instead of a human might shift cooperative behavior in such settings, we used the Prisoner's Dilemma game -- a surrogate of several real-world managerial and economic scenarios. In Experiment 1 (N=100), participants engaged in a thirty-round repeated game against a human, a classic bot, and an LLM (GPT, in real-time). In Experiment 2 (N=192), participants played a one-shot game against a human or an LLM, with half of them allowed to communicate with their opponent, enabling LLMs to leverage a key advantage over older-generation machines. Cooperation rates with LLMs -- while lower by approximately 10-15 percentage points compared to interactions with human opponents -- were nonetheless high. This finding was particularly notable in Experiment 2, where the psychological cost of selfish behavior was reduced. Although allowing communication about cooperation did not close the human-machine behavioral gap, it increased the likelihood of cooperation with both humans and LLMs equally (by 88%), which is particularly surprising for LLMs given their non-human nature and the assumption that people might be less receptive to cooperating with machines compared to human counterparts. Additionally, cooperation with LLMs was higher following prior interaction with humans, suggesting a spillover effect in cooperative behavior. Our findings validate the (careful) use of LLMs by businesses in settings that have a cooperative component. 

---
