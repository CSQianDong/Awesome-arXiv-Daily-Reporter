# Probing the Preferences of a Language Model: Integrating Verbal and Behavioral Tests of AI Welfare 

**Authors**: Valen Tagliabue, Leonard Dung  

**Link**: [PDF](https://arxiv.org/pdf/2509.07961)  

**Abstract**: We develop new experimental paradigms for measuring welfare in language models. We compare verbal reports of models about their preferences with preferences expressed through behavior when navigating a virtual environment and selecting conversation topics. We also test how costs and rewards affect behavior and whether responses to an eudaimonic welfare scale - measuring states such as autonomy and purpose in life - are consistent across semantically equivalent prompts. Overall, we observed a notable degree of mutual support between our measures. The reliable correlations observed between stated preferences and behavior across conditions suggest that preference satisfaction can, in principle, serve as an empirically measurable welfare proxy in some of today's AI systems. Furthermore, our design offered an illuminating setting for qualitative observation of model behavior. Yet, the consistency between measures was more pronounced in some models and conditions than others and responses were not consistent across perturbations. Due to this, and the background uncertainty about the nature of welfare and the cognitive states (and welfare subjecthood) of language models, we are currently uncertain whether our methods successfully measure the welfare state of language models. Nevertheless, these findings highlight the feasibility of welfare measurement in language models, inviting further exploration. 

---
# HiPhO: How Far Are (M)LLMs from Humans in the Latest High School Physics Olympiad Benchmark? 

**Authors**: Fangchen Yu, Haiyuan Wan, Qianjia Cheng, Yuchen Zhang, Jiacheng Chen, Fujun Han, Yulun Wu, Junchi Yao, Ruilizhen Hu, Ning Ding, Yu Cheng, Tao Chen, Lei Bai, Dongzhan Zhou, Yun Luo, Ganqu Cui, Peng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2509.07894)  

**Abstract**: Recently, the physical capabilities of (M)LLMs have garnered increasing attention. However, existing benchmarks for physics suffer from two major gaps: they neither provide systematic and up-to-date coverage of real-world physics competitions such as physics Olympiads, nor enable direct performance comparison with humans. To bridge these gaps, we present HiPhO, the first benchmark dedicated to high school physics Olympiads with human-aligned evaluation. Specifically, HiPhO highlights three key innovations. (1) Comprehensive Data: It compiles 13 latest Olympiad exams from 2024-2025, spanning both international and regional competitions, and covering mixed modalities that encompass problems spanning text-only to diagram-based. (2) Professional Evaluation: We adopt official marking schemes to perform fine-grained grading at both the answer and step level, fully aligned with human examiners to ensure high-quality and domain-specific evaluation. (3) Comparison with Human Contestants: We assign gold, silver, and bronze medals to models based on official medal thresholds, thereby enabling direct comparison between (M)LLMs and human contestants. Our large-scale evaluation of 30 state-of-the-art (M)LLMs shows that: across 13 exams, open-source MLLMs mostly remain at or below the bronze level; open-source LLMs show promising progress with occasional golds; closed-source reasoning MLLMs can achieve 6 to 12 gold medals; and most models still have a significant gap from full marks. These results highlight a substantial performance gap between open-source models and top students, the strong physical reasoning capabilities of closed-source reasoning models, and the fact that there is still significant room for improvement. HiPhO, as a rigorous, human-aligned, and Olympiad-focused benchmark for advancing multimodal physical reasoning, is open-source and available at this https URL. 

---
# CP-Model-Zoo: A Natural Language Query System for Constraint Programming Models 

**Authors**: Augustin Crespin, Ioannis Kostis, Hélène Verhaeghe, Pierre Schaus  

**Link**: [PDF](https://arxiv.org/pdf/2509.07867)  

**Abstract**: Constraint Programming and its high-level modeling languages have long been recognized for their potential to achieve the holy grail of problem-solving. However, the complexity of modeling languages, the large number of global constraints, and the art of creating good models have often hindered non-experts from choosing CP to solve their combinatorial problems. While generating an expert-level model from a natural-language description of a problem would be the dream, we are not yet there. We propose a tutoring system called CP-Model-Zoo, exploiting expert-written models accumulated through the years. CP-Model-Zoo retrieves the closest source code model from a database based on a user's natural language description of a combinatorial problem. It ensures that expert-validated models are presented to the user while eliminating the need for human data labeling. Our experiments show excellent accuracy in retrieving the correct model based on a user-input description of a problem simulated with different levels of expertise. 

---
# SCoder: Iterative Self-Distillation for Bootstrapping Small-Scale Data Synthesizers to Empower Code LLMs 

**Authors**: Xinyu Zhang, Changzhi Zhou, Linmei Hu, Luhao Zhang, Xiancai Chen, Haomin Fu, Yang Yang, Mengdi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.07858)  

**Abstract**: Existing code large language models (LLMs) often rely on large-scale instruction data distilled from proprietary LLMs for fine-tuning, which typically incurs high costs. In this paper, we explore the potential of small-scale open-source LLMs (e.g., 7B) as synthesizers for high-quality code instruction data construction. We first observe that the data synthesis capability of small-scale LLMs can be enhanced by training on a few superior data synthesis samples from proprietary LLMs. Building on this, we propose a novel iterative self-distillation approach to bootstrap small-scale LLMs, transforming them into powerful synthesizers that reduce reliance on proprietary LLMs and minimize costs. Concretely, in each iteration, to obtain diverse and high-quality self-distilled data, we design multi-checkpoint sampling and multi-aspect scoring strategies for initial data selection. Furthermore, to identify the most influential samples, we introduce a gradient-based influence estimation method for final data filtering. Based on the code instruction datasets from the small-scale synthesizers, we develop SCoder, a family of code generation models fine-tuned from DeepSeek-Coder. SCoder models achieve state-of-the-art code generation capabilities, demonstrating the effectiveness of our method. 

---
# Aligning LLMs for the Classroom with Knowledge-Based Retrieval -- A Comparative RAG Study 

**Authors**: Amay Jain, Liu Cui, Si Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.07846)  

**Abstract**: Large language models like ChatGPT are increasingly used in classrooms, but they often provide outdated or fabricated information that can mislead students. Retrieval Augmented Generation (RAG) improves reliability of LLMs by grounding responses in external resources. We investigate two accessible RAG paradigms, vector-based retrieval and graph-based retrieval to identify best practices for classroom question answering (QA). Existing comparative studies fail to account for pedagogical factors such as educational disciplines, question types, and practical deployment costs. Using a novel dataset, EduScopeQA, of 3,176 questions across academic subjects, we measure performance on various educational query types, from specific facts to broad thematic discussions. We also evaluate system alignment with a dataset of systematically altered textbooks that contradict the LLM's latent knowledge. We find that OpenAI Vector Search RAG (representing vector-based RAG) performs well as a low-cost generalist, especially for quick fact retrieval. On the other hand, GraphRAG Global excels at providing pedagogically rich answers to thematic queries, and GraphRAG Local achieves the highest accuracy with the dense, altered textbooks when corpus integrity is critical. Accounting for the 10-20x higher resource usage of GraphRAG (representing graph-based RAG), we show that a dynamic branching framework that routes queries to the optimal retrieval method boosts fidelity and efficiency. These insights provide actionable guidelines for educators and system designers to integrate RAG-augmented LLMs into learning environments effectively. 

---
# Certainty-Guided Reasoning in Large Language Models: A Dynamic Thinking Budget Approach 

**Authors**: João Paulo Nogueira, Wentao Sun, Alonso Silva, Laith Zumot  

**Link**: [PDF](https://arxiv.org/pdf/2509.07820)  

**Abstract**: The rise of large reasoning language models (LRLMs) has unlocked new potential for solving complex tasks. These models operate with a thinking budget, that is, a predefined number of reasoning tokens used to arrive at a solution. We propose a novel approach, inspired by the generator/discriminator framework in generative adversarial networks, in which a critic model periodically probes its own reasoning to assess whether it has reached a confident conclusion. If not, reasoning continues until a target certainty threshold is met. This mechanism adaptively balances efficiency and reliability by allowing early termination when confidence is high, while encouraging further reasoning when uncertainty persists. Through experiments on the AIME2024 and AIME2025 datasets, we show that Certainty-Guided Reasoning (CGR) improves baseline accuracy while reducing token usage. Importantly, extended multi-seed evaluations over 64 runs demonstrate that CGR is stable, reducing variance across seeds and improving exam-like performance under penalty-based grading. Additionally, our token savings analysis shows that CGR can eliminate millions of tokens in aggregate, with tunable trade-offs between certainty thresholds and efficiency. Together, these findings highlight certainty as a powerful signal for reasoning sufficiency. By integrating confidence into the reasoning process, CGR makes large reasoning language models more adaptive, trustworthy, and resource efficient, paving the way for practical deployment in domains where both accuracy and computational cost matter. 

---
# The Carbon Footprint Wizard: A Knowledge-Augmented AI Interface for Streamlining Food Carbon Footprint Analysis 

**Authors**: Mustafa Kaan Aslan, Reinout Heijungs, Filip Ilievski  

**Link**: [PDF](https://arxiv.org/pdf/2509.07733)  

**Abstract**: Environmental sustainability, particularly in relation to climate change, is a key concern for consumers, producers, and policymakers. The carbon footprint, based on greenhouse gas emissions, is a standard metric for quantifying the contribution to climate change of activities and is often assessed using life cycle assessment (LCA). However, conducting LCA is complex due to opaque and global supply chains, as well as fragmented data. This paper presents a methodology that combines advances in LCA and publicly available databases with knowledge-augmented AI techniques, including retrieval-augmented generation, to estimate cradle-to-gate carbon footprints of food products. We introduce a chatbot interface that allows users to interactively explore the carbon impact of composite meals and relate the results to familiar activities. A live web demonstration showcases our proof-of-concept system with arbitrary food items and follow-up questions, highlighting both the potential and limitations - such as database uncertainties and AI misinterpretations - of delivering LCA insights in an accessible format. 

---
# BDPM: A Machine Learning-Based Feature Extractor for Parkinson's Disease Classification via Gut Microbiota Analysis 

**Authors**: Bo Yu, Zhixiu Hua, Bo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.07723)  

**Abstract**: Background: Parkinson's disease remains a major neurodegenerative disorder with high misdiagnosis rates, primarily due to reliance on clinical rating scales. Recent studies have demonstrated a strong association between gut microbiota and Parkinson's disease, suggesting that microbial composition may serve as a promising biomarker. Although deep learning models based ongut microbiota show potential for early prediction, most approaches rely on single classifiers and often overlook inter-strain correlations or temporal dynamics. Therefore, there is an urgent need for more robust feature extraction methods tailored to microbiome data. Methods: We proposed BDPM (A Machine Learning-Based Feature Extractor for Parkinson's Disease Classification via Gut Microbiota Analysis). First, we collected gut microbiota profiles from 39 Parkinson's patients and their healthy spouses to identify differentially abundant taxa. Second, we developed an innovative feature selection framework named RFRE (Random Forest combined with Recursive Feature Elimination), integrating ecological knowledge to enhance biological interpretability. Finally, we designed a hybrid classification model to capture temporal and spatial patterns in microbiome data. 

---
# RIMO: An Easy-to-Evaluate, Hard-to-Solve Olympiad Benchmark for Advanced Mathematical Reasoning 

**Authors**: Ziye Chen, Chengwei Qin, Yao Shu  

**Link**: [PDF](https://arxiv.org/pdf/2509.07711)  

**Abstract**: As large language models (LLMs) reach high scores on established mathematical benchmarks, such as GSM8K and MATH, the research community has turned to International Mathematical Olympiad (IMO) problems to push the evaluation frontier. However, existing Olympiad-level benchmarks suffer from practical constraints that introduce grading noise and potential bias, such as heterogeneous answer formats requiring model-based judges and a reliance on potentially flawed solutions. We introduce RIMO, a two-track benchmark designed to preserve peak Olympiad difficulty while eliminating this evaluation noise. The first track, RIMO-N, rewrites 335 IMO problems to admit a single, unique integer answer, allowing for deterministic correctness checking. The second track, RIMO-P, features 456 proof problems with expert-checked solutions, which are decomposed into a sequence of sub-problems to evaluate the step-by-step reasoning process via an automated grading system. Our benchmarking of ten frontier LLMs, including GPT-4o and Gemini 2.5 Flash, reveals that while these systems excel on older benchmarks, their performance drops sharply on RIMO. These results highlight a substantial gap between current LLM capabilities and actual Olympiad-level reasoning. By providing a challenging yet easy-to-evaluate suite, RIMO offers a high-resolution yardstick for future research, presenting a clear target for closing the profound reasoning gap our findings expose. 

---
# FHIR-RAG-MEDS: Integrating HL7 FHIR with Retrieval-Augmented Large Language Models for Enhanced Medical Decision Support 

**Authors**: Yildiray Kabak, Gokce B. Laleci Erturkmen, Mert Gencturk, Tuncay Namli, A. Anil Sinaci, Ruben Alcantud Corcoles, Cristina Gomez Ballesteros, Pedro Abizanda, Asuman Dogac  

**Link**: [PDF](https://arxiv.org/pdf/2509.07706)  

**Abstract**: In this study, we propose FHIR-RAG-MEDS system that aims to integrate Health Level 7 Fast Healthcare Interoperability Resources (HL7 FHIR) with a Retrieval-Augmented Generation (RAG)-based system to improve personalized medical decision support on evidence-based clinical guidelines, emphasizing the need for research in practical applications. In the evolving landscape of medical decision support systems, integrating advanced technologies such as RAG and HL7 FHIR can significantly enhance clinical decision-making processes. Despite the potential of these technologies, there is limited research on their integration in practical applications. 

---
# Unleashing the True Potential of LLMs: A Feedback-Triggered Self-Correction with Long-Term Multipath Decoding 

**Authors**: Jipeng Li, Zeyu Gao, Yubin Qi, Hande Dong, Weijian Chen, Qiang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.07676)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable performance across diverse tasks, yet their susceptibility to generating incorrect content during inference remains a critical unsolved challenge. While self-correction methods offer potential solutions, their effectiveness is hindered by two inherent limitations: (1) the absence of reliable guidance signals for error localization, and (2) the restricted reasoning depth imposed by conventional next-token decoding paradigms. To address these issues, we propose Feedback-Triggered Regeneration (FTR), a novel framework that synergizes user feedback with enhanced decoding dynamics. Specifically, FTR activates response regeneration only upon receiving negative user feedback, thereby circumventing error propagation from faulty self-assessment while preserving originally correct outputs. Furthermore, we introduce Long-Term Multipath (LTM) decoding, which enables systematic exploration of multiple reasoning trajectories through delayed sequence evaluation, effectively overcoming the myopic decision-making characteristic of standard next-token prediction. Extensive experiments on mathematical reasoning and code generation benchmarks demonstrate that our framework achieves consistent and significant improvements over state-of-the-art prompt-based self-correction methods. 

---
# DeepGraphLog for Layered Neurosymbolic AI 

**Authors**: Adem Kikaj, Giuseppe Marra, Floris Geerts, Robin Manhaeve, Luc De Raedt  

**Link**: [PDF](https://arxiv.org/pdf/2509.07665)  

**Abstract**: Neurosymbolic AI (NeSy) aims to integrate the statistical strengths of neural networks with the interpretability and structure of symbolic reasoning. However, current NeSy frameworks like DeepProbLog enforce a fixed flow where symbolic reasoning always follows neural processing. This restricts their ability to model complex dependencies, especially in irregular data structures such as graphs. In this work, we introduce DeepGraphLog, a novel NeSy framework that extends ProbLog with Graph Neural Predicates. DeepGraphLog enables multi-layer neural-symbolic reasoning, allowing neural and symbolic components to be layered in arbitrary order. In contrast to DeepProbLog, which cannot handle symbolic reasoning via neural methods, DeepGraphLog treats symbolic representations as graphs, enabling them to be processed by Graph Neural Networks (GNNs). We showcase the capabilities of DeepGraphLog on tasks in planning, knowledge graph completion with distant supervision, and GNN expressivity. Our results demonstrate that DeepGraphLog effectively captures complex relational dependencies, overcoming key limitations of existing NeSy systems. By broadening the applicability of neurosymbolic AI to graph-structured domains, DeepGraphLog offers a more expressive and flexible framework for neural-symbolic integration. 

---
# Getting In Contract with Large Language Models -- An Agency Theory Perspective On Large Language Model Alignment 

**Authors**: Sascha Kaltenpoth, Oliver Müller  

**Link**: [PDF](https://arxiv.org/pdf/2509.07642)  

**Abstract**: Adopting Large language models (LLMs) in organizations potentially revolutionizes our lives and work. However, they can generate off-topic, discriminating, or harmful content. This AI alignment problem often stems from misspecifications during the LLM adoption, unnoticed by the principal due to the LLM's black-box nature. While various research disciplines investigated AI alignment, they neither address the information asymmetries between organizational adopters and black-box LLM agents nor consider organizational AI adoption processes. Therefore, we propose LLM ATLAS (LLM Agency Theory-Led Alignment Strategy) a conceptual framework grounded in agency (contract) theory, to mitigate alignment problems during organizational LLM adoption. We conduct a conceptual literature analysis using the organizational LLM adoption phases and the agency theory as concepts. Our approach results in (1) providing an extended literature analysis process specific to AI alignment methods during organizational LLM adoption and (2) providing a first LLM alignment problem-solution space. 

---
# Transferable Direct Prompt Injection via Activation-Guided MCMC Sampling 

**Authors**: Minghui Li, Hao Zhang, Yechao Zhang, Wei Wan, Shengshan Hu, pei Xiaobing, Jing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.07617)  

**Abstract**: Direct Prompt Injection (DPI) attacks pose a critical security threat to Large Language Models (LLMs) due to their low barrier of execution and high potential damage. To address the impracticality of existing white-box/gray-box methods and the poor transferability of black-box methods, we propose an activations-guided prompt injection attack framework. We first construct an Energy-based Model (EBM) using activations from a surrogate model to evaluate the quality of adversarial prompts. Guided by the trained EBM, we employ the token-level Markov Chain Monte Carlo (MCMC) sampling to adaptively optimize adversarial prompts, thereby enabling gradient-free black-box attacks. Experimental results demonstrate our superior cross-model transferability, achieving 49.6% attack success rate (ASR) across five mainstream LLMs and 34.6% improvement over human-crafted prompts, and maintaining 36.6% ASR on unseen task scenarios. Interpretability analysis reveals a correlation between activations and attack effectiveness, highlighting the critical role of semantic patterns in transferable vulnerability exploitation. 

---
# Towards explainable decision support using hybrid neural models for logistic terminal automation 

**Authors**: Riccardo DElia, Alberto Termine, Francesco Flammini  

**Link**: [PDF](https://arxiv.org/pdf/2509.07577)  

**Abstract**: The integration of Deep Learning (DL) in System Dynamics (SD) modeling for transportation logistics offers significant advantages in scalability and predictive accuracy. However, these gains are often offset by the loss of explainability and causal reliability $-$ key requirements in critical decision-making systems. This paper presents a novel framework for interpretable-by-design neural system dynamics modeling that synergizes DL with techniques from Concept-Based Interpretability, Mechanistic Interpretability, and Causal Machine Learning. The proposed hybrid approach enables the construction of neural network models that operate on semantically meaningful and actionable variables, while retaining the causal grounding and transparency typical of traditional SD models. The framework is conceived to be applied to real-world case-studies from the EU-funded project AutoMoTIF, focusing on data-driven decision support, automation, and optimization of multimodal logistic terminals. We aim at showing how neuro-symbolic methods can bridge the gap between black-box predictive models and the need for critical decision support in complex dynamical environments within cyber-physical systems enabled by the industrial Internet-of-Things. 

---
# SheetDesigner: MLLM-Powered Spreadsheet Layout Generation with Rule-Based and Vision-Based Reflection 

**Authors**: Qin Chen, Yuanyi Ren, Xiaojun Ma, Mugeng Liu, Han Shi, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.07473)  

**Abstract**: Spreadsheets are critical to data-centric tasks, with rich, structured layouts that enable efficient information transmission. Given the time and expertise required for manual spreadsheet layout design, there is an urgent need for automated solutions. However, existing automated layout models are ill-suited to spreadsheets, as they often (1) treat components as axis-aligned rectangles with continuous coordinates, overlooking the inherently discrete, grid-based structure of spreadsheets; and (2) neglect interrelated semantics, such as data dependencies and contextual links, unique to spreadsheets. In this paper, we first formalize the spreadsheet layout generation task, supported by a seven-criterion evaluation protocol and a dataset of 3,326 spreadsheets. We then introduce SheetDesigner, a zero-shot and training-free framework using Multimodal Large Language Models (MLLMs) that combines rule and vision reflection for component placement and content population. SheetDesigner outperforms five baselines by at least 22.6\%. We further find that through vision modality, MLLMs handle overlap and balance well but struggle with alignment, necessitates hybrid rule and visual reflection strategies. Our codes and data is available at Github. 

---
# Language Self-Play For Data-Free Training 

**Authors**: Jakub Grudzien Kuba, Mengting Gu, Qi Ma, Yuandong Tian, Vijai Mohan  

**Link**: [PDF](https://arxiv.org/pdf/2509.07414)  

**Abstract**: Large language models (LLMs) have advanced rapidly in recent years, driven by scale, abundant high-quality training data, and reinforcement learning. Yet this progress faces a fundamental bottleneck: the need for ever more data from which models can continue to learn. In this work, we propose a reinforcement learning approach that removes this dependency by enabling models to improve without additional data. Our method leverages a game-theoretic framework of self-play, where a model's capabilities are cast as performance in a competitive game and stronger policies emerge by having the model play against itself - a process we call Language Self-Play (LSP). Experiments with Llama-3.2-3B-Instruct on instruction-following benchmarks show that pretrained models can not only enhance their performance on challenging tasks through self-play alone, but can also do so more effectively than data-driven baselines. 

---
# Autonomous Code Evolution Meets NP-Completeness 

**Authors**: Cunxi Yu, Rongjian Liang, Chia-Tung Ho, Haoxing Ren  

**Link**: [PDF](https://arxiv.org/pdf/2509.07367)  

**Abstract**: Large language models (LLMs) have recently shown strong coding abilities, enabling not only static code generation but also iterative code self-evolving through agentic frameworks. Recently, AlphaEvolve \cite{novikov2025alphaevolve} demonstrated that LLM-based coding agents can autonomously improve algorithms and surpass human experts, with scopes limited to isolated kernels spanning hundreds of lines of code. Inspired by AlphaEvolve, we present SATLUTION, the first framework to extend LLM-based code evolution to the full repository scale, encompassing hundreds of files and tens of thousands of lines of C/C++ code. Targeting Boolean Satisfiability (SAT), the canonical NP-complete problem and a cornerstone of both theory and applications. SATLUTION orchestrates LLM agents to directly evolve solver repositories under strict correctness guarantees and distributed runtime feedback, while simultaneously self-evolving its own evolution policies and rules. Starting from SAT Competition 2024 codebases and benchmark, SATLUTION evolved solvers that decisively outperformed the human-designed winners of the SAT Competition 2025, and also surpassed both 2024 and 2025 champions on the 2024 benchmarks. 

---
# Performative Thinking? The Brittle Correlation Between CoT Length and Problem Complexity 

**Authors**: Vardhan Palod, Karthik Valmeekam, Kaya Stechly, Subbarao Kambhampati  

**Link**: [PDF](https://arxiv.org/pdf/2509.07339)  

**Abstract**: Intermediate token generation (ITG), where a model produces output before the solution, has been proposed as a method to improve the performance of language models on reasoning tasks. While these reasoning traces or Chain of Thoughts (CoTs) are correlated with performance gains, the mechanisms underlying them remain unclear. A prevailing assumption in the community has been to anthropomorphize these tokens as "thinking", treating longer traces as evidence of higher problem-adaptive computation. In this work, we critically examine whether intermediate token sequence length reflects or correlates with problem difficulty. To do so, we train transformer models from scratch on derivational traces of the A* search algorithm, where the number of operations required to solve a maze problem provides a precise and verifiable measure of problem complexity. We first evaluate the models on trivial free-space problems, finding that even for the simplest tasks, they often produce excessively long reasoning traces and sometimes fail to generate a solution. We then systematically evaluate the model on out-of-distribution problems and find that the intermediate token length and ground truth A* trace length only loosely correlate. We notice that the few cases where correlation appears are those where the problems are closer to the training distribution, suggesting that the effect arises from approximate recall rather than genuine problem-adaptive computation. This suggests that the inherent computational complexity of the problem instance is not a significant factor, but rather its distributional distance from the training data. These results challenge the assumption that intermediate trace generation is adaptive to problem difficulty and caution against interpreting longer sequences in systems like R1 as automatically indicative of "thinking effort". 

---
# HealthSLM-Bench: Benchmarking Small Language Models for Mobile and Wearable Healthcare Monitoring 

**Authors**: Xin Wang, Ting Dang, Xinyu Zhang, Vassilis Kostakos, Michael J. Witbrock, Hong Jia  

**Link**: [PDF](https://arxiv.org/pdf/2509.07260)  

**Abstract**: Mobile and wearable healthcare monitoring play a vital role in facilitating timely interventions, managing chronic health conditions, and ultimately improving individuals' quality of life. Previous studies on large language models (LLMs) have highlighted their impressive generalization abilities and effectiveness in healthcare prediction tasks. However, most LLM-based healthcare solutions are cloud-based, which raises significant privacy concerns and results in increased memory usage and latency. To address these challenges, there is growing interest in compact models, Small Language Models (SLMs), which are lightweight and designed to run locally and efficiently on mobile and wearable devices. Nevertheless, how well these models perform in healthcare prediction remains largely unexplored. We systematically evaluated SLMs on health prediction tasks using zero-shot, few-shot, and instruction fine-tuning approaches, and deployed the best performing fine-tuned SLMs on mobile devices to evaluate their real-world efficiency and predictive performance in practical healthcare scenarios. Our results show that SLMs can achieve performance comparable to LLMs while offering substantial gains in efficiency and privacy. However, challenges remain, particularly in handling class imbalance and few-shot scenarios. These findings highlight SLMs, though imperfect in their current form, as a promising solution for next-generation, privacy-preserving healthcare monitoring. 

---
# OmniAcc: Personalized Accessibility Assistant Using Generative AI 

**Authors**: Siddhant Karki, Ethan Han, Nadim Mahmud, Suman Bhunia, John Femiani, Vaskar Raychoudhury  

**Link**: [PDF](https://arxiv.org/pdf/2509.07220)  

**Abstract**: Individuals with ambulatory disabilities often encounter significant barriers when navigating urban environments due to the lack of accessible information and tools. This paper presents OmniAcc, an AI-powered interactive navigation system that utilizes GPT-4, satellite imagery, and OpenStreetMap data to identify, classify, and map wheelchair-accessible features such as ramps and crosswalks in the built environment. OmniAcc offers personalized route planning, real-time hands-free navigation, and instant query responses regarding physical accessibility. By using zero-shot learning and customized prompts, the system ensures precise detection of accessibility features, while supporting validation through structured workflows. This paper introduces OmniAcc and explores its potential to assist urban planners and mobility-aid users, demonstrated through a case study on crosswalk detection. With a crosswalk detection accuracy of 97.5%, OmniAcc highlights the transformative potential of AI in improving navigation and fostering more inclusive urban spaces. 

---
# BlendedNet: A Blended Wing Body Aircraft Dataset and Surrogate Model for Aerodynamic Predictions 

**Authors**: Nicholas Sung, Steven Spreizer, Mohamed Elrefaie, Kaira Samuel, Matthew C. Jones, Faez Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2509.07209)  

**Abstract**: BlendedNet is a publicly available aerodynamic dataset of 999 blended wing body (BWB) geometries. Each geometry is simulated across about nine flight conditions, yielding 8830 converged RANS cases with the Spalart-Allmaras model and 9 to 14 million cells per case. The dataset is generated by sampling geometric design parameters and flight conditions, and includes detailed pointwise surface quantities needed to study lift and drag. We also introduce an end-to-end surrogate framework for pointwise aerodynamic prediction. The pipeline first uses a permutation-invariant PointNet regressor to predict geometric parameters from sampled surface point clouds, then conditions a Feature-wise Linear Modulation (FiLM) network on the predicted parameters and flight conditions to predict pointwise coefficients Cp, Cfx, and Cfz. Experiments show low errors in surface predictions across diverse BWBs. BlendedNet addresses data scarcity for unconventional configurations and enables research on data-driven surrogate modeling for aerodynamic design. 

---
# A Hybrid CNN-LSTM Deep Learning Model for Intrusion Detection in Smart Grid 

**Authors**: Abdulhakim Alsaiari, Mohammad Ilyas  

**Link**: [PDF](https://arxiv.org/pdf/2509.07208)  

**Abstract**: The evolution of the traditional power grid into the "smart grid" has resulted in a fundamental shift in energy management, which allows the integration of renewable energy sources with modern communication technology. However, this interconnection has increased smart grids' vulnerability to attackers, which might result in privacy breaches, operational interruptions, and massive outages. The SCADA-based smart grid protocols are critical for real-time data collection and control, but they are vulnerable to attacks like unauthorized access and denial of service (DoS). This research proposes a hybrid deep learning-based Intrusion Detection System (IDS) intended to improve the cybersecurity of smart grids. The suggested model takes advantage of Convolutional Neural Networks' (CNN) feature extraction capabilities as well as Long Short-Term Memory (LSTM) networks' temporal pattern recognition skills. DNP3 and IEC104 intrusion detection datasets are employed to train and test our CNN-LSTM model to recognize and classify the potential cyber threats. Compared to other deep learning approaches, the results demonstrate considerable improvements in accuracy, precision, recall, and F1-score, with a detection accuracy of 99.70%. 

---
# That's So FETCH: Fashioning Ensemble Techniques for LLM Classification in Civil Legal Intake and Referral 

**Authors**: Quinten Steenhuis  

**Link**: [PDF](https://arxiv.org/pdf/2509.07170)  

**Abstract**: Each year millions of people seek help for their legal problems by calling a legal aid program hotline, walking into a legal aid office, or using a lawyer referral service. The first step to match them to the right help is to identify the legal problem the applicant is experiencing. Misdirection has consequences. Applicants may miss a deadline, experience physical abuse, lose housing or lose custody of children while waiting to connect to the right legal help. We introduce and evaluate the FETCH classifier for legal issue classification and describe two methods for improving accuracy: a hybrid LLM/ML ensemble classification method, and the automatic generation of follow-up questions to enrich the initial problem narrative. We employ a novel data set of 419 real-world queries to a nonprofit lawyer referral service. Ultimately, we show classification accuracy (hits@2) of 97.37\% using a mix of inexpensive models, exceeding the performance of the current state-of-the-art GPT-5 model. Our approach shows promise in significantly reducing the cost of guiding users of the legal system to the right resource for their problem while achieving high accuracy. 

---
# PaVeRL-SQL: Text-to-SQL via Partial-Match Rewards and Verbal Reinforcement Learning 

**Authors**: Heng Hao, Wenjun Hu, Oxana Verkholyak, Davoud Ataee Tarzanagh, Baruch Gutow, Sima Didari, Masoud Faraki, Hankyu Moon, Seungjai Min  

**Link**: [PDF](https://arxiv.org/pdf/2509.07159)  

**Abstract**: Text-to-SQL models allow users to interact with a database more easily by generating executable SQL statements from natural-language questions. Despite recent successes on simpler databases and questions, current Text-to-SQL methods still suffer from low execution accuracy on industry-scale databases and complex questions involving domain-specific business logic. We present \emph{PaVeRL-SQL}, a framework that combines \emph{Partial-Match Rewards} and \emph{Verbal Reinforcement Learning} to drive self-improvement in reasoning language models (RLMs) for Text-to-SQL. To handle practical use cases, we adopt two pipelines: (1) a newly designed in-context learning framework with group self-evaluation (verbal-RL), using capable open- and closed-source large language models (LLMs) as backbones; and (2) a chain-of-thought (CoT) RL pipeline with a small backbone model (OmniSQL-7B) trained with a specially designed reward function and two-stage RL. These pipelines achieve state-of-the-art (SOTA) results on popular Text-to-SQL benchmarks -- Spider, Spider 2.0, and BIRD. For the industrial-level Spider2.0-SQLite benchmark, the verbal-RL pipeline achieves an execution accuracy 7.4\% higher than SOTA, and the CoT pipeline is 1.4\% higher. RL training with mixed SQL dialects yields strong, threefold gains, particularly for dialects with limited training data. Overall, \emph{PaVeRL-SQL} delivers reliable, SOTA Text-to-SQL under realistic industrial constraints. The code is available at this https URL. 

---
# Autoencoder-Based Denoising of Muscle Artifacts in ECG to Preserve Skin Nerve Activity (SKNA) for Cognitive Stress Detection 

**Authors**: Farnoush Baghestani, Jihye Moon, Youngsun Kong, Ki Chon  

**Link**: [PDF](https://arxiv.org/pdf/2509.07146)  

**Abstract**: The sympathetic nervous system (SNS) plays a central role in regulating the body's responses to stress and maintaining physiological stability. Its dysregulation is associated with a wide range of conditions, from cardiovascular disease to anxiety disorders. Skin nerve activity (SKNA) extracted from high-frequency electrocardiogram (ECG) recordings provides a noninvasive window into SNS dynamics, but its measurement is highly susceptible to electromyographic (EMG) contamination. Traditional preprocessing based on bandpass filtering within a fixed range (e.g., 500--1000 Hz) is susceptible to overlapping EMG and SKNA spectral components, especially during sustained muscle activity. We present a denoising approach using a lightweight one-dimensional convolutional autoencoder with a long short-term memory (LSTM) bottleneck to reconstruct clean SKNA from EMG-contaminated recordings. Using clean ECG-derived SKNA data from cognitive stress experiments and EMG noise from chaotic muscle stimulation recordings, we simulated contamination at realistic noise levels (--4 dB, --8 dB signal-to-noise ratio) and trained the model in the leave-one-subject-out cross-validation framework. The method improved signal-to-noise ratio by up to 9.65 dB, increased cross correlation with clean SKNA from 0.40 to 0.72, and restored burst-based SKNA features to near-clean discriminability (AUROC $\geq$ 0.96). Classification of baseline versus sympathetic stimulation (cognitive stress) conditions reached accuracies of 91--98\% across severe noise levels, comparable to clean data. These results demonstrate that deep learning--based reconstruction can preserve physiologically relevant sympathetic bursts during substantial EMG interference, enabling more robust SKNA monitoring in naturalistic, movement-rich environments. 

---
# Neuro-Symbolic Frameworks: Conceptual Characterization and Empirical Comparative Analysis 

**Authors**: Sania Sinha, Tanawan Premsri, Danial Kamali, Parisa Kordjamshidi  

**Link**: [PDF](https://arxiv.org/pdf/2509.07122)  

**Abstract**: Neurosymbolic (NeSy) frameworks combine neural representations and learning with symbolic representations and reasoning. Combining the reasoning capacities, explainability, and interpretability of symbolic processing with the flexibility and power of neural computing allows us to solve complex problems with more reliability while being data-efficient. However, this recently growing topic poses a challenge to developers with its learning curve, lack of user-friendly tools, libraries, and unifying frameworks. In this paper, we characterize the technical facets of existing NeSy frameworks, such as the symbolic representation language, integration with neural models, and the underlying algorithms. A majority of the NeSy research focuses on algorithms instead of providing generic frameworks for declarative problem specification to leverage problem solving. To highlight the key aspects of Neurosymbolic modeling, we showcase three generic NeSy frameworks - \textit{DeepProbLog}, \textit{Scallop}, and \textit{DomiKnowS}. We identify the challenges within each facet that lay the foundation for identifying the expressivity of each framework in solving a variety of problems. Building on this foundation, we aim to spark transformative action and encourage the community to rethink this problem in novel ways. 

---
# Instruction Agent: Enhancing Agent with Expert Demonstration 

**Authors**: Yinheng Li, Hailey Hultquist, Justin Wagle, Kazuhito Koishida  

**Link**: [PDF](https://arxiv.org/pdf/2509.07098)  

**Abstract**: Graphical user interface (GUI) agents have advanced rapidly but still struggle with complex tasks involving novel UI elements, long-horizon actions, and personalized trajectories. In this work, we introduce Instruction Agent, a GUI agent that leverages expert demonstrations to solve such tasks, enabling completion of otherwise difficult workflows. Given a single demonstration, the agent extracts step-by-step instructions and executes them by strictly following the trajectory intended by the user, which avoids making mistakes during execution. The agent leverages the verifier and backtracker modules further to improve robustness. Both modules are critical to understand the current outcome from each action and handle unexpected interruptions(such as pop-up windows) during execution. Our experiments show that Instruction Agent achieves a 60% success rate on a set of tasks in OSWorld that all top-ranked agents failed to complete. The Instruction Agent offers a practical and extensible framework, bridging the gap between current GUI agents and reliable real-world GUI task automation. 

---
# Statistical Methods in Generative AI 

**Authors**: Edgar Dobriban  

**Link**: [PDF](https://arxiv.org/pdf/2509.07054)  

**Abstract**: Generative Artificial Intelligence is emerging as an important technology, promising to be transformative in many areas. At the same time, generative AI techniques are based on sampling from probabilistic models, and by default, they come with no guarantees about correctness, safety, fairness, or other properties. Statistical methods offer a promising potential approach to improve the reliability of generative AI techniques. In addition, statistical methods are also promising for improving the quality and efficiency of AI evaluation, as well as for designing interventions and experiments in AI.
In this paper, we review some of the existing work on these topics, explaining both the general statistical techniques used, as well as their applications to generative AI. We also discuss limitations and potential future directions. 

---
# From Eigenmodes to Proofs: Integrating Graph Spectral Operators with Symbolic Interpretable Reasoning 

**Authors**: Andrew Kiruluta, Priscilla Burity  

**Link**: [PDF](https://arxiv.org/pdf/2509.07017)  

**Abstract**: We introduce Spectral NSR, a fully spectral neuro-symbolic reasoning framework that embeds logical rules as spectral templates and performs inference directly in the graph spectral domain. By leveraging graph signal processing (GSP) and frequency-selective filters grounded in the Laplacian eigenstructure of knowledge graphs, the architecture unifies the interpretability of symbolic reasoning with the scalability and adaptability of spectral learning. Beyond the core formulation, we incorporate a comprehensive set of extensions, including dynamic graph and basis learning, rational and diffusion filters for sharper spectral selectivity, mixture-of-spectral-experts for modular specialization, proof-guided training with spectral curricula, and uncertainty quantification for calibrated confidence. Additional enhancements such as large language model coupling, co-spectral transfer alignment, adversarial robustness, efficient GPU kernels, generalized Laplacians, and causal interventions further expand the versatility of the framework.
Empirical evaluation on state-of-the-art reasoning benchmarks such as ProofWriter and CLUTRR demonstrates that Spectral NSR achieves superior accuracy, faster inference, improved robustness to adversarial perturbations, and higher interpretability compared to leading baselines including transformers, message-passing neural networks, and neuro-symbolic logic programming systems. Spectral attribution and proof-band agreement analyses confirm that model decisions align closely with symbolic proof structures, while transfer experiments validate effective domain adaptation through co-spectral alignment. These results establish Spectral NSR as a scalable and principled foundation for the next generation of reasoning systems, offering transparency, robustness, and generalization beyond conventional approaches. 

---
# Renewable Energy Sources Selection Analysis with the Maximizing Deviation Method 

**Authors**: Kirisci Murat  

**Link**: [PDF](https://arxiv.org/pdf/2509.07011)  

**Abstract**: Multi-criteria decision-making methods provide decision-makers with appropriate tools to make better decisions in uncertain, complex, and conflicting situations. Fuzzy set theory primarily deals with the uncertainty inherent in human thoughts and perceptions and attempts to quantify this uncertainty. Fuzzy logic and fuzzy set theory are utilized with multi-criteria decision-making methods because they effectively handle uncertainty and fuzziness in decision-makers' judgments, allowing for verbal judgments of the problem. This study utilizes the Fermatean fuzzy environment, a generalization of fuzzy sets. An optimization model based on the deviation maximization method is proposed to determine partially known feature weights. This method is combined with interval-valued Fermatean fuzzy sets. The proposed method was applied to the problem of selecting renewable energy sources. The reason for choosing renewable energy sources is that meeting energy needs from renewable sources, balancing carbon emissions, and mitigating the effects of global climate change are among the most critical issues of the recent period. Even though selecting renewable energy sources is a technical issue, the managerial and political implications of this issue are also important, and are discussed in this study. 

---
# Mini-o3: Scaling Up Reasoning Patterns and Interaction Turns for Visual Search 

**Authors**: Xin Lai, Junyi Li, Wei Li, Tao Liu, Tianjian Li, Hengshuang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.07969)  

**Abstract**: Recent advances in large multimodal models have leveraged image-based tools with reinforcement learning to tackle visual problems. However, existing open-source approaches often exhibit monotonous reasoning patterns and allow only a limited number of interaction turns, making them inadequate for difficult tasks that require trial-and-error exploration. In this work, we address this limitation by scaling up tool-based interactions and introduce Mini-o3, a system that executes deep, multi-turn reasoning -- spanning tens of steps -- and achieves state-of-the-art performance on challenging visual search tasks. Our recipe for reproducing OpenAI o3-style behaviors comprises three key components. First, we construct the Visual Probe Dataset, a collection of thousands of challenging visual search problems designed for exploratory reasoning. Second, we develop an iterative data collection pipeline to obtain cold-start trajectories that exhibit diverse reasoning patterns, including depth-first search, trial-and-error, and goal maintenance. Third, we propose an over-turn masking strategy that prevents penalization of over-turn responses (those that hit the maximum number of turns) during reinforcement learning, thereby balancing training-time efficiency with test-time scalability. Despite training with an upper bound of only six interaction turns, our model generates trajectories that naturally scale to tens of turns at inference time, with accuracy improving as the number of turns increases. Extensive experiments demonstrate that Mini-o3 produces rich reasoning patterns and deep thinking paths, effectively solving challenging visual search problems. 

---
# ACE and Diverse Generalization via Selective Disagreement 

**Authors**: Oliver Daniels, Stuart Armstrong, Alexandre Maranhão, Mahirah Fairuz Rahman, Benjamin M. Marlin, Rebecca Gorman  

**Link**: [PDF](https://arxiv.org/pdf/2509.07955)  

**Abstract**: Deep neural networks are notoriously sensitive to spurious correlations - where a model learns a shortcut that fails out-of-distribution. Existing work on spurious correlations has often focused on incomplete correlations,leveraging access to labeled instances that break the correlation. But in cases where the spurious correlations are complete, the correct generalization is fundamentally \textit{underspecified}. To resolve this underspecification, we propose learning a set of concepts that are consistent with training data but make distinct predictions on a subset of novel unlabeled inputs. Using a self-training approach that encourages \textit{confident} and \textit{selective} disagreement, our method ACE matches or outperforms existing methods on a suite of complete-spurious correlation benchmarks, while remaining robust to incomplete spurious correlations. ACE is also more configurable than prior approaches, allowing for straight-forward encoding of prior knowledge and principled unsupervised model selection. In an early application to language-model alignment, we find that ACE achieves competitive performance on the measurement tampering detection benchmark \textit{without} access to untrusted measurements. While still subject to important limitations, ACE represents significant progress towards overcoming underspecification. 

---
# Bringing Multi-Modal Multi-Task Federated Foundation Models to Education Domain: Prospects and Challenges 

**Authors**: Kasra Borazjani, Naji Khosravan, Rajeev Sahay, Bita Akram, Seyyedali Hosseinalipour  

**Link**: [PDF](https://arxiv.org/pdf/2509.07946)  

**Abstract**: Multi-modal multi-task (M3T) foundation models (FMs) have recently shown transformative potential in artificial intelligence, with emerging applications in education. However, their deployment in real-world educational settings is hindered by privacy regulations, data silos, and limited domain-specific data availability. We introduce M3T Federated Foundation Models (FedFMs) for education: a paradigm that integrates federated learning (FL) with M3T FMs to enable collaborative, privacy-preserving training across decentralized institutions while accommodating diverse modalities and tasks. Subsequently, this position paper aims to unveil M3T FedFMs as a promising yet underexplored approach to the education community, explore its potentials, and reveal its related future research directions. We outline how M3T FedFMs can advance three critical pillars of next-generation intelligent education systems: (i) privacy preservation, by keeping sensitive multi-modal student and institutional data local; (ii) personalization, through modular architectures enabling tailored models for students, instructors, and institutions; and (iii) equity and inclusivity, by facilitating participation from underrepresented and resource-constrained entities. We finally identify various open research challenges, including studying of (i) inter-institution heterogeneous privacy regulations, (ii) the non-uniformity of data modalities' characteristics, (iii) the unlearning approaches for M3T FedFMs, (iv) the continual learning frameworks for M3T FedFMs, and (v) M3T FedFM model interpretability, which must be collectively addressed for practical deployment. 

---
# ImportSnare: Directed "Code Manual" Hijacking in Retrieval-Augmented Code Generation 

**Authors**: Kai Ye, Liangcai Su, Chenxiong Qian  

**Link**: [PDF](https://arxiv.org/pdf/2509.07941)  

**Abstract**: Code generation has emerged as a pivotal capability of Large Language Models(LLMs), revolutionizing development efficiency for programmers of all skill levels. However, the complexity of data structures and algorithmic logic often results in functional deficiencies and security vulnerabilities in generated code, reducing it to a prototype requiring extensive manual debugging. While Retrieval-Augmented Generation (RAG) can enhance correctness and security by leveraging external code manuals, it simultaneously introduces new attack surfaces.
In this paper, we pioneer the exploration of attack surfaces in Retrieval-Augmented Code Generation (RACG), focusing on malicious dependency hijacking. We demonstrate how poisoned documentation containing hidden malicious dependencies (e.g., matplotlib_safe) can subvert RACG, exploiting dual trust chains: LLM reliance on RAG and developers' blind trust in LLM suggestions. To construct poisoned documents, we propose ImportSnare, a novel attack framework employing two synergistic strategies: 1)Position-aware beam search optimizes hidden ranking sequences to elevate poisoned documents in retrieval results, and 2)Multilingual inductive suggestions generate jailbreaking sequences to manipulate LLMs into recommending malicious dependencies. Through extensive experiments across Python, Rust, and JavaScript, ImportSnare achieves significant attack success rates (over 50% for popular libraries such as matplotlib and seaborn) in general, and is also able to succeed even when the poisoning ratio is as low as 0.01%, targeting both custom and real-world malicious packages. Our findings reveal critical supply chain risks in LLM-powered development, highlighting inadequate security alignment for code generation tasks. To support future research, we will release the multilingual benchmark suite and datasets. The project homepage is this https URL. 

---
# Breaking Android with AI: A Deep Dive into LLM-Powered Exploitation 

**Authors**: Wanni Vidulige Ishan Perera, Xing Liu, Fan liang, Junyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.07933)  

**Abstract**: The rapid evolution of Artificial Intelligence (AI) and Large Language Models (LLMs) has opened up new opportunities in the area of cybersecurity, especially in the exploitation automation landscape and penetration testing. This study explores Android penetration testing automation using LLM-based tools, especially PentestGPT, to identify and execute rooting techniques. Through a comparison of the traditional manual rooting process and exploitation methods produced using AI, this study evaluates the efficacy, reliability, and scalability of automated penetration testing in achieving high-level privilege access on Android devices. With the use of an Android emulator (Genymotion) as the testbed, we fully execute both traditional and exploit-based rooting methods, automating the process using AI-generated scripts. Secondly, we create a web application by integrating OpenAI's API to facilitate automated script generation from LLM-processed responses. The research focuses on the effectiveness of AI-enabled exploitation by comparing automated and manual penetration testing protocols, by determining LLM weaknesses and strengths along the way. We also provide security suggestions of AI-enabled exploitation, including ethical factors and potential misuse. The findings exhibit that while LLMs can significantly streamline the workflow of exploitation, they need to be controlled by humans to ensure accuracy and ethical application. This study adds to the increasing body of literature on AI-powered cybersecurity and its effect on ethical hacking, security research, and mobile device security. 

---
# Accelerating Local AI on Consumer GPUs: A Hardware-Aware Dynamic Strategy for YOLOv10s 

**Authors**: Mahmudul Islam Masum, Miad Islam, Arif I. Sarwat  

**Link**: [PDF](https://arxiv.org/pdf/2509.07928)  

**Abstract**: As local AI grows in popularity, there is a critical gap between the benchmark performance of object detectors and their practical viability on consumer-grade hardware. While models like YOLOv10s promise real-time speeds, these metrics are typically achieved on high-power, desktop-class GPUs. This paper reveals that on resource-constrained systems, such as laptops with RTX 4060 GPUs, performance is not compute-bound but is instead dominated by system-level bottlenecks, as illustrated by a simple bottleneck test. To overcome this hardware-level constraint, we introduce a Two-Pass Adaptive Inference algorithm, a model-independent approach that requires no architectural changes. This study mainly focuses on adaptive inference strategies and undertakes a comparative analysis of architectural early-exit and resolution-adaptive routing, highlighting their respective trade-offs within a unified evaluation framework. The system uses a fast, low-resolution pass and only escalates to a high-resolution model pass when detection confidence is low. On a 5000-image COCO dataset, our method achieves a 1.85x speedup over a PyTorch Early-Exit baseline, with a modest mAP loss of 5.51%. This work provides a practical and reproducible blueprint for deploying high-performance, real-time AI on consumer-grade devices by shifting the focus from pure model optimization to hardware-aware inference strategies that maximize throughput. 

---
# GENUINE: Graph Enhanced Multi-level Uncertainty Estimation for Large Language Models 

**Authors**: Tuo Wang, Adithya Kulkarni, Tyler Cody, Peter A. Beling, Yujun Yan, Dawei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.07925)  

**Abstract**: Uncertainty estimation is essential for enhancing the reliability of Large Language Models (LLMs), particularly in high-stakes applications. Existing methods often overlook semantic dependencies, relying on token-level probability measures that fail to capture structural relationships within the generated text. We propose GENUINE: Graph ENhanced mUlti-level uncertaINty Estimation for Large Language Models, a structure-aware framework that leverages dependency parse trees and hierarchical graph pooling to refine uncertainty quantification. By incorporating supervised learning, GENUINE effectively models semantic and structural relationships, improving confidence assessments. Extensive experiments across NLP tasks show that GENUINE achieves up to 29% higher AUROC than semantic entropy-based approaches and reduces calibration errors by over 15%, demonstrating the effectiveness of graph-based uncertainty modeling. The code is available at this https URL. 

---
# Multimodal Contrastive Pretraining of CBCT and IOS for Enhanced Tooth Segmentation 

**Authors**: Moo Hyun Son, Juyoung Bae, Zelin Qiu, Jiale Peng, Kai Xin Li, Yifan Lin, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.07923)  

**Abstract**: Digital dentistry represents a transformative shift in modern dental practice. The foundational step in this transformation is the accurate digital representation of the patient's dentition, which is obtained from segmented Cone-Beam Computed Tomography (CBCT) and Intraoral Scans (IOS). Despite the growing interest in digital dental technologies, existing segmentation methodologies frequently lack rigorous validation and demonstrate limited performance and clinical applicability. To the best of our knowledge, this is the first work to introduce a multimodal pretraining framework for tooth segmentation. We present ToothMCL, a Tooth Multimodal Contrastive Learning for pretraining that integrates volumetric (CBCT) and surface-based (IOS) modalities. By capturing modality-invariant representations through multimodal contrastive learning, our approach effectively models fine-grained anatomical features, enabling precise multi-class segmentation and accurate identification of Fédération Dentaire Internationale (FDI) tooth numbering. Along with the framework, we curated CBCT-IOS3.8K, the largest paired CBCT and IOS dataset to date, comprising 3,867 patients. We then evaluated ToothMCL on a comprehensive collection of independent datasets, representing the largest and most diverse evaluation to date. Our method achieves state-of-the-art performance in both internal and external testing, with an increase of 12\% for CBCT segmentation and 8\% for IOS segmentation in the Dice Similarity Coefficient (DSC). Furthermore, ToothMCL consistently surpasses existing approaches in tooth groups and demonstrates robust generalizability across varying imaging conditions and clinical scenarios. 

---
# Uncovering Scaling Laws for Large Language Models via Inverse Problems 

**Authors**: Arun Verma, Zhaoxuan Wu, Zijian Zhou, Xiaoqiang Lin, Zhiliang Chen, Rachael Hwee Ling Sim, Rui Qiao, Jingtan Wang, Nhung Bui, Xinyuan Niu, Wenyang Hu, Gregory Kang Ruey Lau, Zi-Yu Khoo, Zitong Zhao, Xinyi Xu, Apivich Hemachandra, See-Kiong Ng, Bryan Kian Hsiang Low  

**Link**: [PDF](https://arxiv.org/pdf/2509.07909)  

**Abstract**: Large Language Models (LLMs) are large-scale pretrained models that have achieved remarkable success across diverse domains. These successes have been driven by unprecedented complexity and scale in both data and computations. However, due to the high costs of training such models, brute-force trial-and-error approaches to improve LLMs are not feasible. Inspired by the success of inverse problems in uncovering fundamental scientific laws, this position paper advocates that inverse problems can also efficiently uncover scaling laws that guide the building of LLMs to achieve the desirable performance with significantly better cost-effectiveness. 

---
# Active Membership Inference Test (aMINT): Enhancing Model Auditability with Multi-Task Learning 

**Authors**: Daniel DeAlcala, Aythami Morales, Julian Fierrez, Gonzalo Mancera, Ruben Tolosana, Javier Ortega-Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2509.07879)  

**Abstract**: Active Membership Inference Test (aMINT) is a method designed to detect whether given data were used during the training of machine learning models. In Active MINT, we propose a novel multitask learning process that involves training simultaneously two models: the original or Audited Model, and a secondary model, referred to as the MINT Model, responsible for identifying the data used for training the Audited Model. This novel multi-task learning approach has been designed to incorporate the auditability of the model as an optimization objective during the training process of neural networks. The proposed approach incorporates intermediate activation maps as inputs to the MINT layers, which are trained to enhance the detection of training data. We present results using a wide range of neural networks, from lighter architectures such as MobileNet to more complex ones such as Vision Transformers, evaluated in 5 public benchmarks. Our proposed Active MINT achieves over 80% accuracy in detecting if given data was used for training, significantly outperforming previous approaches in the literature. Our aMINT and related methodological developments contribute to increasing transparency in AI models, facilitating stronger safeguards in AI deployments to achieve proper security, privacy, and copyright protection. 

---
# Deep Learning-Based Burned Area Mapping Using Bi-Temporal Siamese Networks and AlphaEarth Foundation Datasets 

**Authors**: Seyd Teymoor Seydi  

**Link**: [PDF](https://arxiv.org/pdf/2509.07852)  

**Abstract**: Accurate and timely mapping of burned areas is crucial for environmental monitoring, disaster management, and assessment of climate change. This study presents a novel approach to automated burned area mapping using the AlphaEArth dataset combined with the Siamese U-Net deep learning architecture. The AlphaEArth Dataset, comprising high-resolution optical and thermal infrared imagery with comprehensive ground-truth annotations, provides an unprecedented resource for training robust burned area detection models. We trained our model with the Monitoring Trends in Burn Severity (MTBS) dataset in the contiguous US and evaluated it with 17 regions cross in Europe. Our experimental results demonstrate that the proposed ensemble approach achieves superior performance with an overall accuracy of 95%, IoU of 0.6, and F1-score of 74% on the test dataset. The model successfully identifies burned areas across diverse ecosystems with complex background, showing particular strength in detecting partially burned vegetation and fire boundaries and its transferability and high generalization in burned area mapping. This research contributes to the advancement of automated fire damage assessment and provides a scalable solution for global burn area monitoring using the AlphaEarth dataset. 

---
# Small Open Models Achieve Near Parity with Large Models in Low Resource Literary Translation at a Fraction of the Cost 

**Authors**: Mihai Nadas, Laura Diosan, Andreea Tomescu, Andrei Piscoran  

**Link**: [PDF](https://arxiv.org/pdf/2509.07829)  

**Abstract**: Literary translation has recently gained attention as a distinct and complex task in machine translation research. However, the translation by small open models remains an open problem. We contribute to this ongoing research by introducing TINYFABULIST TRANSLATION FRAMEWORK (TF2), a unified framework for dataset creation, fine tuning, and evaluation in English-Romanian literary translations, centred on the creation and open release of both a compact, fine tuned language model (TF2-12B) and large scale synthetic parallel datasets (DS-TF2-EN-RO-3M and DS-TF2-EN-RO-15K). Building on DS-TF1-EN-3M (TF1), the largest collection of synthetic English fables to date, we address the need for rich, high quality literary datasets in low resource languages such as Romanian. Our pipeline first generates 15k high quality Romanian references from the TF1 pool using a high performing LLM. We then apply a two stage fine tuning process to a 12B parameter open weight model: (i) instruction tuning to capture genre specific narrative style, and (ii) adapter compression for efficient deployment. Evaluation combines corpus level BLEU and a five dimension LLM based rubric (accuracy, fluency, coherence, style, cultural adaptation) to provide a nuanced assessment of translation quality. Results show that our fine tuned model achieves fluency and adequacy competitive with top performing large proprietary models, while being open, accessible, and significantly more cost effective. Alongside the fine tuned model and both datasets, we publicly release all scripts and evaluation prompts. TF2 thus provides an end-to-end, reproducible pipeline for research on cost efficient translation, cross lingual narrative generation, and the broad adoption of open models for culturally significant literary content in low resource settings. 

---
# Forecasting Russian Equipment Losses Using Time Series and Deep Learning Models 

**Authors**: Jonathan Teagan  

**Link**: [PDF](https://arxiv.org/pdf/2509.07813)  

**Abstract**: This study applies a range of forecasting techniques,including ARIMA, Prophet, Long Short Term Memory networks (LSTM), Temporal Convolutional Networks (TCN), and XGBoost, to model and predict Russian equipment losses during the ongoing war in Ukraine. Drawing on daily and monthly open-source intelligence (OSINT) data from WarSpotting, we aim to assess trends in attrition, evaluate model performance, and estimate future loss patterns through the end of 2025. Our findings show that deep learning models, particularly TCN and LSTM, produce stable and consistent forecasts, especially under conditions of high temporal granularity. By comparing different model architectures and input structures, this study highlights the importance of ensemble forecasting in conflict modeling, and the value of publicly available OSINT data in quantifying material degradation over time. 

---
# Enhanced SegNet with Integrated Grad-CAM for Interpretable Retinal Layer Segmentation in OCT Images 

**Authors**: S M Asiful Islam Saky, Ugyen Tshering  

**Link**: [PDF](https://arxiv.org/pdf/2509.07795)  

**Abstract**: Optical Coherence Tomography (OCT) is essential for diagnosing conditions such as glaucoma, diabetic retinopathy, and age-related macular degeneration. Accurate retinal layer segmentation enables quantitative biomarkers critical for clinical decision-making, but manual segmentation is time-consuming and variable, while conventional deep learning models often lack interpretability. This work proposes an improved SegNet-based deep learning framework for automated and interpretable retinal layer segmentation. Architectural innovations, including modified pooling strategies, enhance feature extraction from noisy OCT images, while a hybrid loss function combining categorical cross-entropy and Dice loss improves performance for thin and imbalanced retinal layers. Gradient-weighted Class Activation Mapping (Grad-CAM) is integrated to provide visual explanations, allowing clinical validation of model decisions. Trained and validated on the Duke OCT dataset, the framework achieved 95.77% validation accuracy, a Dice coefficient of 0.9446, and a Jaccard Index (IoU) of 0.8951. Class-wise results confirmed robust performance across most layers, with challenges remaining for thinner boundaries. Grad-CAM visualizations highlighted anatomically relevant regions, aligning segmentation with clinical biomarkers and improving transparency. By combining architectural improvements, a customized hybrid loss, and explainable AI, this study delivers a high-performing SegNet-based framework that bridges the gap between accuracy and interpretability. The approach offers strong potential for standardizing OCT analysis, enhancing diagnostic efficiency, and fostering clinical trust in AI-driven ophthalmic tools. 

---
# Individual utilities of life satisfaction reveal inequality aversion unrelated to political alignment 

**Authors**: Crispin Cooper, Ana Friedrich, Tommaso Reggiani, Wouter Poortinga  

**Link**: [PDF](https://arxiv.org/pdf/2509.07793)  

**Abstract**: How should well-being be prioritised in society, and what trade-offs are people willing to make between fairness and personal well-being? We investigate these questions using a stated preference experiment with a nationally representative UK sample (n = 300), in which participants evaluated life satisfaction outcomes for both themselves and others under conditions of uncertainty. Individual-level utility functions were estimated using an Expected Utility Maximisation (EUM) framework and tested for sensitivity to the overweighting of small probabilities, as characterised by Cumulative Prospect Theory (CPT). A majority of participants displayed concave (risk-averse) utility curves and showed stronger aversion to inequality in societal life satisfaction outcomes than to personal risk. These preferences were unrelated to political alignment, suggesting a shared normative stance on fairness in well-being that cuts across ideological boundaries. The results challenge use of average life satisfaction as a policy metric, and support the development of nonlinear utility-based alternatives that more accurately reflect collective human values. Implications for public policy, well-being measurement, and the design of value-aligned AI systems are discussed. 

---
# XSRD-Net: EXplainable Stroke Relapse Detection 

**Authors**: Christian Gapp, Elias Tappeiner, Martin Welk, Karl Fritscher, Stephanie Mangesius, Constantin Eisenschink, Philipp Deisl, Michael Knoflach, Astrid E. Grams, Elke R. Gizewski, Rainer Schubert  

**Link**: [PDF](https://arxiv.org/pdf/2509.07772)  

**Abstract**: Stroke is the second most frequent cause of death world wide with an annual mortality of around 5.5 million. Recurrence rates of stroke are between 5 and 25% in the first year. As mortality rates for relapses are extraordinarily high (40%) it is of utmost importance to reduce the recurrence rates. We address this issue by detecting patients at risk of stroke recurrence at an early stage in order to enable appropriate therapy planning. To this end we collected 3D intracranial CTA image data and recorded concomitant heart diseases, the age and the gender of stroke patients between 2010 and 2024. We trained single- and multimodal deep learning based neural networks for binary relapse detection (Task 1) and for relapse free survival (RFS) time prediction together with a subsequent classification (Task 2). The separation of relapse from non-relapse patients (Task 1) could be solved with tabular data (AUC on test dataset: 0.84). However, for the main task, the regression (Task 2), our multimodal XSRD-net processed the modalities vision:tabular with 0.68:0.32 according to modality contribution measures. The c-index with respect to relapses for the multimodal model reached 0.68, and the AUC is 0.71 for the test dataset. Final, deeper interpretability analysis results could highlight a link between both heart diseases (tabular) and carotid arteries (vision) for the detection of relapses and the prediction of the RFS time. This is a central outcome that we strive to strengthen with ongoing data collection and model retraining. 

---
# Are LLMs Enough for Hyperpartisan, Fake, Polarized and Harmful Content Detection? Evaluating In-Context Learning vs. Fine-Tuning 

**Authors**: Michele Joshua Maggini, Dhia Merzougui, Rabiraj Bandyopadhyay, Gaël Dias, Fabrice Maurel, Pablo Gamallo  

**Link**: [PDF](https://arxiv.org/pdf/2509.07768)  

**Abstract**: The spread of fake news, polarizing, politically biased, and harmful content on online platforms has been a serious concern. With large language models becoming a promising approach, however, no study has properly benchmarked their performance across different models, usage methods, and languages. This study presents a comprehensive overview of different Large Language Models adaptation paradigms for the detection of hyperpartisan and fake news, harmful tweets, and political bias. Our experiments spanned 10 datasets and 5 different languages (English, Spanish, Portuguese, Arabic and Bulgarian), covering both binary and multiclass classification scenarios. We tested different strategies ranging from parameter efficient Fine-Tuning of language models to a variety of different In-Context Learning strategies and prompts. These included zero-shot prompts, codebooks, few-shot (with both randomly-selected and diversely-selected examples using Determinantal Point Processes), and Chain-of-Thought. We discovered that In-Context Learning often underperforms when compared to Fine-Tuning a model. This main finding highlights the importance of Fine-Tuning even smaller models on task-specific settings even when compared to the largest models evaluated in an In-Context Learning setup - in our case LlaMA3.1-8b-Instruct, Mistral-Nemo-Instruct-2407 and Qwen2.5-7B-Instruct. 

---
# What Were You Thinking? An LLM-Driven Large-Scale Study of Refactoring Motivations in Open-Source Projects 

**Authors**: Mikel Robredo, Matteo Esposito, Fabio Palomba, Rafael Peñaloza, Valentina Lenarduzzi  

**Link**: [PDF](https://arxiv.org/pdf/2509.07763)  

**Abstract**: Context. Code refactoring improves software quality without changing external behavior. Despite its advantages, its benefits are hindered by the considerable cost of time, resources, and continuous effort it demands. Aim. Understanding why developers refactor, and which metrics capture these motivations, may support wider and more effective use of refactoring in practice. Method. We performed a large-scale empirical study to analyze developers refactoring activity, leveraging Large Language Models (LLMs) to identify underlying motivations from version control data, comparing our findings with previous motivations reported in the literature. Results. LLMs matched human judgment in 80% of cases, but aligned with literature-based motivations in only 47%. They enriched 22% of motivations with more detailed rationale, often highlighting readability, clarity, and structural improvements. Most motivations were pragmatic, focused on simplification and maintainability. While metrics related to developer experience and code readability ranked highest, their correlation with motivation categories was weak. Conclusions. We conclude that LLMs effectively capture surface-level motivations but struggle with architectural reasoning. Their value lies in providing localized explanations, which, when combined with software metrics, can form hybrid approaches. Such integration offers a promising path toward prioritizing refactoring more systematically and balancing short-term improvements with long-term architectural goals. 

---
# Spectral and Rhythm Feature Performance Evaluation for Category and Class Level Audio Classification with Deep Convolutional Neural Networks 

**Authors**: Friedrich Wolf-Monheim  

**Link**: [PDF](https://arxiv.org/pdf/2509.07756)  

**Abstract**: Next to decision tree and k-nearest neighbours algorithms deep convolutional neural networks (CNNs) are widely used to classify audio data in many domains like music, speech or environmental sounds. To train a specific CNN various spectral and rhythm features like mel-scaled spectrograms, mel-frequency cepstral coefficients (MFCC), cyclic tempograms, short-time Fourier transform (STFT) chromagrams, constant-Q transform (CQT) chromagrams and chroma energy normalized statistics (CENS) chromagrams can be used as digital image input data for the neural network. The performance of these spectral and rhythm features for audio category level as well as audio class level classification is investigated in detail with a deep CNN and the ESC-50 dataset with 2,000 labeled environmental audio recordings using an end-to-end deep learning pipeline. The evaluated metrics accuracy, precision, recall and F1 score for multiclass classification clearly show that the mel-scaled spectrograms and the mel-frequency cepstral coefficients (MFCC) perform significantly better then the other spectral and rhythm features investigated in this research for audio classification tasks using deep CNNs. 

---
# Enhancing Online Learning by Integrating Biosensors and Multimodal Learning Analytics for Detecting and Predicting Student Behavior: A Review 

**Authors**: Alvaro Becerra, Ruth Cobos, Charles Lang  

**Link**: [PDF](https://arxiv.org/pdf/2509.07742)  

**Abstract**: In modern online learning, understanding and predicting student behavior is crucial for enhancing engagement and optimizing educational outcomes. This systematic review explores the integration of biosensors and Multimodal Learning Analytics (MmLA) to analyze and predict student behavior during computer-based learning sessions. We examine key challenges, including emotion and attention detection, behavioral analysis, experimental design, and demographic considerations in data collection. Our study highlights the growing role of physiological signals, such as heart rate, brain activity, and eye-tracking, combined with traditional interaction data and self-reports to gain deeper insights into cognitive states and engagement levels. We synthesize findings from 54 key studies, analyzing commonly used methodologies such as advanced machine learning algorithms and multimodal data pre-processing techniques. The review identifies current research trends, limitations, and emerging directions in the field, emphasizing the transformative potential of biosensor-driven adaptive learning systems. Our findings suggest that integrating multimodal data can facilitate personalized learning experiences, real-time feedback, and intelligent educational interventions, ultimately advancing toward a more customized and adaptive online learning experience. 

---
# Spectral Masking and Interpolation Attack (SMIA): A Black-box Adversarial Attack against Voice Authentication and Anti-Spoofing Systems 

**Authors**: Kamel Kamel, Hridoy Sankar Dutta, Keshav Sood, Sunil Aryal  

**Link**: [PDF](https://arxiv.org/pdf/2509.07677)  

**Abstract**: Voice Authentication Systems (VAS) use unique vocal characteristics for verification. They are increasingly integrated into high-security sectors such as banking and healthcare. Despite their improvements using deep learning, they face severe vulnerabilities from sophisticated threats like deepfakes and adversarial attacks. The emergence of realistic voice cloning complicates detection, as systems struggle to distinguish authentic from synthetic audio. While anti-spoofing countermeasures (CMs) exist to mitigate these risks, many rely on static detection models that can be bypassed by novel adversarial methods, leaving a critical security gap. To demonstrate this vulnerability, we propose the Spectral Masking and Interpolation Attack (SMIA), a novel method that strategically manipulates inaudible frequency regions of AI-generated audio. By altering the voice in imperceptible zones to the human ear, SMIA creates adversarial samples that sound authentic while deceiving CMs. We conducted a comprehensive evaluation of our attack against state-of-the-art (SOTA) models across multiple tasks, under simulated real-world conditions. SMIA achieved a strong attack success rate (ASR) of at least 82% against combined VAS/CM systems, at least 97.5% against standalone speaker verification systems, and 100% against countermeasures. These findings conclusively demonstrate that current security postures are insufficient against adaptive adversarial attacks. This work highlights the urgent need for a paradigm shift toward next-generation defenses that employ dynamic, context-aware frameworks capable of evolving with the threat landscape. 

---
# Variational Quantum Circuits in Offline Contextual Bandit Problems 

**Authors**: Lukas Schulte, Daniel Hein, Steffen Udluft, Thomas A. Runkler  

**Link**: [PDF](https://arxiv.org/pdf/2509.07633)  

**Abstract**: This paper explores the application of variational quantum circuits (VQCs) for solving offline contextual bandit problems in industrial optimization tasks. Using the Industrial Benchmark (IB) environment, we evaluate the performance of quantum regression models against classical models. Our findings demonstrate that quantum models can effectively fit complex reward functions, identify optimal configurations via particle swarm optimization (PSO), and generalize well in noisy and sparse datasets. These results provide a proof of concept for utilizing VQCs in offline contextual bandit problems and highlight their potential in industrial optimization tasks. 

---
# From Classical Data to Quantum Advantage -- Quantum Policy Evaluation on Quantum Hardware 

**Authors**: Daniel Hein, Simon Wiedemann, Markus Baumann, Patrik Felbinger, Justin Klein, Maximilian Schieder, Jonas Stein, Daniëlle Schuman, Thomas Cope, Steffen Udluft  

**Link**: [PDF](https://arxiv.org/pdf/2509.07614)  

**Abstract**: Quantum policy evaluation (QPE) is a reinforcement learning (RL) algorithm which is quadratically more efficient than an analogous classical Monte Carlo estimation. It makes use of a direct quantum mechanical realization of a finite Markov decision process, in which the agent and the environment are modeled by unitary operators and exchange states, actions, and rewards in superposition. Previously, the quantum environment has been implemented and parametrized manually for an illustrative benchmark using a quantum simulator. In this paper, we demonstrate how these environment parameters can be learned from a batch of classical observational data through quantum machine learning (QML) on quantum hardware. The learned quantum environment is then applied in QPE to also compute policy evaluations on quantum hardware. Our experiments reveal that, despite challenges such as noise and short coherence times, the integration of QML and QPE shows promising potential for achieving quantum advantage in RL. 

---
# Beyond Rebalancing: Benchmarking Binary Classifiers Under Class Imbalance Without Rebalancing Techniques 

**Authors**: Ali Nawaz, Amir Ahmad, Shehroz S. Khan  

**Link**: [PDF](https://arxiv.org/pdf/2509.07605)  

**Abstract**: Class imbalance poses a significant challenge to supervised classification, particularly in critical domains like medical diagnostics and anomaly detection where minority class instances are rare. While numerous studies have explored rebalancing techniques to address this issue, less attention has been given to evaluating the performance of binary classifiers under imbalance when no such techniques are applied. Therefore, the goal of this study is to assess the performance of binary classifiers "as-is", without performing any explicit rebalancing. Specifically, we systematically evaluate the robustness of a diverse set of binary classifiers across both real-world and synthetic datasets, under progressively reduced minority class sizes, using one-shot and few-shot scenarios as baselines. Our approach also explores varying data complexities through synthetic decision boundary generation to simulate real-world conditions. In addition to standard classifiers, we include experiments using undersampling, oversampling strategies, and one-class classification (OCC) methods to examine their behavior under severe imbalance. The results confirm that classification becomes more difficult as data complexity increases and the minority class size decreases. While traditional classifiers deteriorate under extreme imbalance, advanced models like TabPFN and boosting-based ensembles retain relatively higher performance and better generalization compared to traditional classifiers. Visual interpretability and evaluation metrics further validate these findings. Our work offers valuable guidance on model selection for imbalanced learning, providing insights into classifier robustness without dependence on explicit rebalancing techniques. 

---
# Transformer-Based Approach to Optimal Sensor Placement for Structural Health Monitoring of Probe Cards 

**Authors**: Mehdi Bejani, Marco Mauri, Daniele Acconcia, Simone Todaro, Stefano Mariani  

**Link**: [PDF](https://arxiv.org/pdf/2509.07603)  

**Abstract**: This paper presents an innovative Transformer-based deep learning strategy for optimizing the placement of sensors aiming at structural health monitoring of semiconductor probe cards. Failures in probe cards, including substrate cracks and loosened screws, would critically affect semiconductor manufacturing yield and reliability. Some failure modes could be detected by equipping a probe card with adequate sensors. Frequency response functions from simulated failure scenarios are adopted within a finite element model of a probe card. A comprehensive dataset, enriched by physics-informed scenario expansion and physics-aware statistical data augmentation, is exploited to train a hybrid Convolutional Neural Network and Transformer model. The model achieves high accuracy (99.83%) in classifying the probe card health states (baseline, loose screw, crack) and an excellent crack detection recall (99.73%). Model robustness is confirmed through a rigorous framework of 3 repetitions of 10-fold stratified cross-validation. The attention mechanism also pinpoints critical sensor locations: an analysis of the attention weights offers actionable insights for designing efficient, cost-effective monitoring systems by optimizing sensor configurations. This research highlights the capability of attention-based deep learning to advance proactive maintenance, enhancing operational reliability and yield in semiconductor manufacturing. 

---
# Can SSD-Mamba2 Unlock Reinforcement Learning for End-to-End Motion Control? 

**Authors**: Gavin Tao, Yinuo Wang, Jinzhao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.07593)  

**Abstract**: End-to-end reinforcement learning for motion control promises unified perception-action policies that scale across embodiments and tasks, yet most deployed controllers are either blind (proprioception-only) or rely on fusion backbones with unfavorable compute-memory trade-offs. Recurrent controllers struggle with long-horizon credit assignment, and Transformer-based fusion incurs quadratic cost in token length, limiting temporal and spatial context. We present a vision-driven cross-modal RL framework built on SSD-Mamba2, a selective state-space backbone that applies state-space duality (SSD) to enable both recurrent and convolutional scanning with hardware-aware streaming and near-linear scaling. Proprioceptive states and exteroceptive observations (e.g., depth tokens) are encoded into compact tokens and fused by stacked SSD-Mamba2 layers. The selective state-space updates retain long-range dependencies with markedly lower latency and memory use than quadratic self-attention, enabling longer look-ahead, higher token resolution, and stable training under limited compute. Policies are trained end-to-end under curricula that randomize terrain and appearance and progressively increase scene complexity. A compact, state-centric reward balances task progress, energy efficiency, and safety. Across diverse motion-control scenarios, our approach consistently surpasses strong state-of-the-art baselines in return, safety (collisions and falls), and sample efficiency, while converging faster at the same compute budget. These results suggest that SSD-Mamba2 provides a practical fusion backbone for scalable, foresightful, and efficient end-to-end motion control. 

---
# BALI: Enhancing Biomedical Language Representations through Knowledge Graph and Language Model Alignment 

**Authors**: Andrey Sakhovskiy, Elena Tutubalina  

**Link**: [PDF](https://arxiv.org/pdf/2509.07588)  

**Abstract**: In recent years, there has been substantial progress in using pretrained Language Models (LMs) on a range of tasks aimed at improving the understanding of biomedical texts. Nonetheless, existing biomedical LLMs show limited comprehension of complex, domain-specific concept structures and the factual information encoded in biomedical Knowledge Graphs (KGs). In this work, we propose BALI (Biomedical Knowledge Graph and Language Model Alignment), a novel joint LM and KG pre-training method that augments an LM with external knowledge by the simultaneous learning of a dedicated KG encoder and aligning the representations of both the LM and the graph. For a given textual sequence, we link biomedical concept mentions to the Unified Medical Language System (UMLS) KG and utilize local KG subgraphs as cross-modal positive samples for these mentions. Our empirical findings indicate that implementing our method on several leading biomedical LMs, such as PubMedBERT and BioLinkBERT, improves their performance on a range of language understanding tasks and the quality of entity representations, even with minimal pre-training on a small alignment dataset sourced from PubMed scientific abstracts. 

---
# Attention Maps in 3D Shape Classification for Dental Stage Estimation with Class Node Graph Attention Networks 

**Authors**: Barkin Buyukcakir, Rocharles Cavalcante Fontenele, Reinhilde Jacobs, Jannick De Tobel, Patrick Thevissen, Dirk Vandermeulen, Peter Claes  

**Link**: [PDF](https://arxiv.org/pdf/2509.07581)  

**Abstract**: Deep learning offers a promising avenue for automating many recognition tasks in fields such as medicine and forensics. However, the black-box nature of these models hinders their adoption in high-stakes applications where trust and accountability are required. For 3D shape recognition tasks in particular, this paper introduces the Class Node Graph Attention Network (CGAT) architecture to address this need. Applied to 3D meshes of third molars derived from CBCT images, for Demirjian stage allocation, CGAT utilizes graph attention convolutions and an inherent attention mechanism, visualized via attention rollout, to explain its decision-making process. We evaluated the local mean curvature and distance to centroid node features, both individually and in combination, as well as model depth, finding that models incorporating directed edges to a global CLS node produced more intuitive attention maps, while also yielding desirable classification performance. We analyzed the attention-based explanations of the models, and their predictive performances to propose optimal settings for the CGAT. The combination of local mean curvature and distance to centroid as node features yielded a slight performance increase with 0.76 weighted F1 score, and more comprehensive attention visualizations. The CGAT architecture's ability to generate human-understandable attention maps can enhance trust and facilitate expert validation of model decisions. While demonstrated on dental data, CGAT is broadly applicable to graph-based classification and regression tasks, promoting wider adoption of transparent and competitive deep learning models in high-stakes environments. 

---
# Towards Generalized Routing: Model and Agent Orchestration for Adaptive and Efficient Inference 

**Authors**: Xiyu Guo, Shan Wang, Chunfang Ji, Xuefeng Zhao, Wenhao Xi, Yaoyao Liu, Qinglan Li, Chao Deng, Junlan Feng  

**Link**: [PDF](https://arxiv.org/pdf/2509.07571)  

**Abstract**: The rapid advancement of large language models (LLMs) and domain-specific AI agents has greatly expanded the ecosystem of AI-powered services. User queries, however, are highly diverse and often span multiple domains and task types, resulting in a complex and heterogeneous landscape. This diversity presents a fundamental routing challenge: how to accurately direct each query to an appropriate execution unit while optimizing both performance and efficiency. To address this, we propose MoMA (Mixture of Models and Agents), a generalized routing framework that integrates both LLM and agent-based routing. Built upon a deep understanding of model and agent capabilities, MoMA effectively handles diverse queries through precise intent recognition and adaptive routing strategies, achieving an optimal balance between efficiency and cost. Specifically, we construct a detailed training dataset to profile the capabilities of various LLMs under different routing model structures, identifying the most suitable tasks for each LLM. During inference, queries are dynamically routed to the LLM with the best cost-performance efficiency. We also introduce an efficient agent selection strategy based on a context-aware state machine and dynamic masking. Experimental results demonstrate that the MoMA router offers superior cost-efficiency and scalability compared to existing approaches. 

---
# $ΔL$ Normalization: Rethink Loss Aggregation in RLVR 

**Authors**: Zhiyuan He, Xufang Luo, Yike Zhang, Yuqing Yang, Lili Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2509.07558)  

**Abstract**: We propose $\Delta L$ Normalization, a simple yet effective loss aggregation method tailored to the characteristic of dynamic generation lengths in Reinforcement Learning with Verifiable Rewards (RLVR). Recently, RLVR has demonstrated strong potential in improving the reasoning capabilities of large language models (LLMs), but a major challenge lies in the large variability of response lengths during training, which leads to high gradient variance and unstable optimization. Although previous methods such as GRPO, DAPO, and Dr. GRPO introduce different loss normalization terms to address this issue, they either produce biased estimates or still suffer from high gradient variance. By analyzing the effect of varying lengths on policy loss both theoretically and empirically, we reformulate the problem as finding a minimum-variance unbiased estimator. Our proposed $\Delta L$ Normalization not only provides an unbiased estimate of the true policy loss but also minimizes gradient variance in theory. Extensive experiments show that it consistently achieves superior results across different model sizes, maximum lengths, and tasks. Our code will be made public at this https URL. 

---
# Avoiding Knowledge Edit Skipping in Multi-hop Question Answering with Guided Decomposition 

**Authors**: Yi Liu, Xiangrong Zhu, Xiangyu Liu, Wei Wei, Wei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.07555)  

**Abstract**: In a rapidly evolving world where information updates swiftly, knowledge in large language models (LLMs) becomes outdated quickly. Retraining LLMs is not a cost-effective option, making knowledge editing (KE) without modifying parameters particularly necessary. We find that although existing retrieval-augmented generation (RAG)-based KE methods excel at editing simple knowledge, they struggle with KE in multi-hop question answering due to the issue of "edit skipping", which refers to skipping the relevant edited fact in inference. In addition to the diversity of natural language expressions of knowledge, edit skipping also arises from the mismatch between the granularity of LLMs in problem-solving and the facts in the edited memory. To address this issue, we propose a novel Iterative Retrieval-Augmented Knowledge Editing method with guided decomposition (IRAKE) through the guidance from single edited facts and entire edited cases. Experimental results demonstrate that IRAKE mitigates the failure of editing caused by edit skipping and outperforms state-of-the-art methods for KE in multi-hop question answering. 

---
# HU-based Foreground Masking for 3D Medical Masked Image Modeling 

**Authors**: Jin Lee, Vu Dang, Gwang-Hyun Yu, Anh Le, Zahid Rahman, Jin-Ho Jang, Heonzoo Lee, Kun-Yung Kim, Jin-Sul Kim, Jin-Young Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.07534)  

**Abstract**: While Masked Image Modeling (MIM) has revolutionized fields of computer vision, its adoption in 3D medical image computing has been limited by the use of random masking, which overlooks the density of anatomical objects. To address this limitation, we enhance the pretext task with a simple yet effective masking strategy. Leveraging Hounsfield Unit (HU) measurements, we implement an HU-based Foreground Masking, which focuses on the intensity distribution of visceral organs and excludes non-tissue regions, such as air and fluid, that lack diagnostically meaningful features. Extensive experiments on five public 3D medical imaging datasets demonstrate that our masking consistently improves performance, both in quality of segmentation and Dice score (BTCV:~84.64\%, Flare22:~92.43\%, MM-WHS:~90.67\%, Amos22:~88.64\%, BraTS:~78.55\%). These results underscore the importance of domain-centric MIM and suggest a promising direction for representation learning in medical image segmentation. Implementation is available at this http URL. 

---
# FLeW: Facet-Level and Adaptive Weighted Representation Learning of Scientific Documents 

**Authors**: Zheng Dou, Deqing Wang, Fuzhen Zhuang, Jian Ren, Yanlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.07531)  

**Abstract**: Scientific document representation learning provides powerful embeddings for various tasks, while current methods face challenges across three approaches. 1) Contrastive training with citation-structural signals underutilizes citation information and still generates single-vector representations. 2) Fine-grained representation learning, which generates multiple vectors at the sentence or aspect level, requires costly integration and lacks domain generalization. 3) Task-aware learning depends on manually predefined task categorization, overlooking nuanced task distinctions and requiring extra training data for task-specific modules. To address these problems, we propose a new method that unifies the three approaches for better representations, namely FLeW. Specifically, we introduce a novel triplet sampling method that leverages citation intent and frequency to enhance citation-structural signals for training. Citation intents (background, method, result), aligned with the general structure of scientific writing, facilitate a domain-generalized facet partition for fine-grained representation learning. Then, we adopt a simple weight search to adaptively integrate three facet-level embeddings into a task-specific document embedding without task-aware fine-tuning. Experiments show the applicability and robustness of FLeW across multiple scientific tasks and fields, compared to prior models. 

---
# Competitive Audio-Language Models with Data-Efficient Single-Stage Training on Public Data 

**Authors**: Gokul Karthik Kumar, Rishabh Saraf, Ludovick Lepauloux, Abdul Muneer, Billel Mokeddem, Hakim Hacid  

**Link**: [PDF](https://arxiv.org/pdf/2509.07526)  

**Abstract**: Large language models (LLMs) have transformed NLP, yet their integration with audio remains underexplored -- despite audio's centrality to human communication. We introduce Falcon3-Audio, a family of Audio-Language Models (ALMs) built on instruction-tuned LLMs and Whisper encoders. Using a remarkably small amount of public audio data -- less than 30K hours (5K unique) -- Falcon3-Audio-7B matches the best reported performance among open-weight models on the MMAU benchmark, with a score of 64.14, matching R1-AQA, while distinguishing itself through superior data and parameter efficiency, single-stage training, and transparency. Notably, our smallest 1B model remains competitive with larger open models ranging from 2B to 13B parameters. Through extensive ablations, we find that common complexities -- such as curriculum learning, multiple audio encoders, and intricate cross-attention connectors -- are not required for strong performance, even compared to models trained on over 500K hours of data. 

---
# EHWGesture -- A dataset for multimodal understanding of clinical gestures 

**Authors**: Gianluca Amprimo, Alberto Ancilotto, Alessandro Savino, Fabio Quazzolo, Claudia Ferraris, Gabriella Olmo, Elisabetta Farella, Stefano Di Carlo  

**Link**: [PDF](https://arxiv.org/pdf/2509.07525)  

**Abstract**: Hand gesture understanding is essential for several applications in human-computer interaction, including automatic clinical assessment of hand dexterity. While deep learning has advanced static gesture recognition, dynamic gesture understanding remains challenging due to complex spatiotemporal variations. Moreover, existing datasets often lack multimodal and multi-view diversity, precise ground-truth tracking, and an action quality component embedded within gestures. This paper introduces EHWGesture, a multimodal video dataset for gesture understanding featuring five clinically relevant gestures. It includes over 1,100 recordings (6 hours), captured from 25 healthy subjects using two high-resolution RGB-Depth cameras and an event camera. A motion capture system provides precise ground-truth hand landmark tracking, and all devices are spatially calibrated and synchronized to ensure cross-modal alignment. Moreover, to embed an action quality task within gesture understanding, collected recordings are organized in classes of execution speed that mirror clinical evaluations of hand dexterity. Baseline experiments highlight the dataset's potential for gesture classification, gesture trigger detection, and action quality assessment. Thus, EHWGesture can serve as a comprehensive benchmark for advancing multimodal clinical gesture understanding. 

---
# Water Demand Forecasting of District Metered Areas through Learned Consumer Representations 

**Authors**: Adithya Ramachandran, Thorkil Flensmark B. Neergaard, Tomás Arias-Vergara, Andreas Maier, Siming Bayer  

**Link**: [PDF](https://arxiv.org/pdf/2509.07515)  

**Abstract**: Advancements in smart metering technologies have significantly improved the ability to monitor and manage water utilities. In the context of increasing uncertainty due to climate change, securing water resources and supply has emerged as an urgent global issue with extensive socioeconomic ramifications. Hourly consumption data from end-users have yielded substantial insights for projecting demand across regions characterized by diverse consumption patterns. Nevertheless, the prediction of water demand remains challenging due to influencing non-deterministic factors, such as meteorological conditions. This work introduces a novel method for short-term water demand forecasting for District Metered Areas (DMAs) which encompass commercial, agricultural, and residential consumers. Unsupervised contrastive learning is applied to categorize end-users according to distinct consumption behaviors present within a DMA. Subsequently, the distinct consumption behaviors are utilized as features in the ensuing demand forecasting task using wavelet-transformed convolutional networks that incorporate a cross-attention mechanism combining both historical data and the derived representations. The proposed approach is evaluated on real-world DMAs over a six-month period, demonstrating improved forecasting performance in terms of MAPE across different DMAs, with a maximum improvement of 4.9%. Additionally, it identifies consumers whose behavior is shaped by socioeconomic factors, enhancing prior knowledge about the deterministic patterns that influence demand. 

---
# ALLabel: Three-stage Active Learning for LLM-based Entity Recognition using Demonstration Retrieval 

**Authors**: Zihan Chen, Lei Shi, Weize Wu, Qiji Zhou, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.07512)  

**Abstract**: Many contemporary data-driven research efforts in the natural sciences, such as chemistry and materials science, require large-scale, high-performance entity recognition from scientific datasets. Large language models (LLMs) have increasingly been adopted to solve the entity recognition task, with the same trend being observed on all-spectrum NLP tasks. The prevailing entity recognition LLMs rely on fine-tuned technology, yet the fine-tuning process often incurs significant cost. To achieve a best performance-cost trade-off, we propose ALLabel, a three-stage framework designed to select the most informative and representative samples in preparing the demonstrations for LLM modeling. The annotated examples are used to construct a ground-truth retrieval corpus for LLM in-context learning. By sequentially employing three distinct active learning strategies, ALLabel consistently outperforms all baselines under the same annotation budget across three specialized domain datasets. Experimental results also demonstrate that selectively annotating only 5\%-10\% of the dataset with ALLabel can achieve performance comparable to the method annotating the entire dataset. Further analyses and ablation studies verify the effectiveness and generalizability of our proposal. 

---
# Astra: A Multi-Agent System for GPU Kernel Performance Optimization 

**Authors**: Anjiang Wei, Tianran Sun, Yogesh Seenichamy, Hang Song, Anne Ouyang, Azalia Mirhoseini, Ke Wang, Alex Aiken  

**Link**: [PDF](https://arxiv.org/pdf/2509.07506)  

**Abstract**: GPU kernel optimization has long been a central challenge at the intersection of high-performance computing and machine learning. Efficient kernels are crucial for accelerating large language model (LLM) training and serving, yet attaining high performance typically requires extensive manual tuning. Compiler-based systems reduce some of this burden, but still demand substantial manual design and engineering effort. Recently, researchers have explored using LLMs for GPU kernel generation, though prior work has largely focused on translating high-level PyTorch modules into CUDA code. In this work, we introduce Astra, the first LLM-based multi-agent system for GPU kernel optimization. Unlike previous approaches, Astra starts from existing CUDA implementations extracted from SGLang, a widely deployed framework for serving LLMs, rather than treating PyTorch modules as the specification. Within Astra, specialized LLM agents collaborate through iterative code generation, testing, profiling, and planning to produce kernels that are both correct and high-performance. On kernels from SGLang, Astra achieves an average speedup of 1.32x using zero-shot prompting with OpenAI o4-mini. A detailed case study further demonstrates that LLMs can autonomously apply loop transformations, optimize memory access patterns, exploit CUDA intrinsics, and leverage fast math operations to yield substantial performance gains. Our work highlights multi-agent LLM systems as a promising new paradigm for GPU kernel optimization. 

---
# Generating Transferrable Adversarial Examples via Local Mixing and Logits Optimization for Remote Sensing Object Recognition 

**Authors**: Chun Liu, Hailong Wang, Bingqian Zhu, Panpan Ding, Zheng Zheng, Tao Xu, Zhigang Han, Jiayao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.07495)  

**Abstract**: Deep Neural Networks (DNNs) are vulnerable to adversarial attacks, posing significant security threats to their deployment in remote sensing applications. Research on adversarial attacks not only reveals model vulnerabilities but also provides critical insights for enhancing robustness. Although current mixing-based strategies have been proposed to increase the transferability of adversarial examples, they either perform global blending or directly exchange a region in the images, which may destroy global semantic features and mislead the optimization of adversarial examples. Furthermore, their reliance on cross-entropy loss for perturbation optimization leads to gradient diminishing during iterative updates, compromising adversarial example quality. To address these limitations, we focus on non-targeted attacks and propose a novel framework via local mixing and logits optimization. First, we present a local mixing strategy to generate diverse yet semantically consistent inputs. Different from MixUp, which globally blends two images, and MixCut, which stitches images together, our method merely blends local regions to preserve global semantic information. Second, we adapt the logit loss from targeted attacks to non-targeted scenarios, mitigating the gradient vanishing problem of cross-entropy loss. Third, a perturbation smoothing loss is applied to suppress high-frequency noise and enhance transferability. Extensive experiments on FGSCR-42 and MTARSI datasets demonstrate superior performance over 12 state-of-the-art methods across 6 surrogate models. Notably, with ResNet as the surrogate on MTARSI, our method achieves a 17.28% average improvement in black-box attack success rate. 

---
# Fine-Tuning Vision-Language Models for Visual Navigation Assistance 

**Authors**: Xiao Li, Bharat Gandhi, Ming Zhan, Mohit Nehra, Zhicheng Zhang, Yuchen Sun, Meijia Song, Naisheng Zhang, Xi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.07488)  

**Abstract**: We address vision-language-driven indoor navigation to assist visually impaired individuals in reaching a target location using images and natural language guidance. Traditional navigation systems are ineffective indoors due to the lack of precise location data. Our approach integrates vision and language models to generate step-by-step navigational instructions, enhancing accessibility and independence. We fine-tune the BLIP-2 model with Low Rank Adaptation (LoRA) on a manually annotated indoor navigation dataset. We propose an evaluation metric that refines the BERT F1 score by emphasizing directional and sequential variables, providing a more comprehensive measure of navigational performance. After applying LoRA, the model significantly improved in generating directional instructions, overcoming limitations in the original BLIP-2 model. 

---
# HALT-RAG: A Task-Adaptable Framework for Hallucination Detection with Calibrated NLI Ensembles and Abstention 

**Authors**: Saumya Goswami, Siddharth Kurra  

**Link**: [PDF](https://arxiv.org/pdf/2509.07475)  

**Abstract**: Detecting content that contradicts or is unsupported by a given source text is a critical challenge for the safe deployment of generative language models. We introduce HALT-RAG, a post-hoc verification system designed to identify hallucinations in the outputs of Retrieval-Augmented Generation (RAG) pipelines. Our flexible and task-adaptable framework uses a universal feature set derived from an ensemble of two frozen, off-the-shelf Natural Language Inference (NLI) models and lightweight lexical signals. These features are used to train a simple, calibrated, and task-adapted meta-classifier. Using a rigorous 5-fold out-of-fold (OOF) training protocol to prevent data leakage and produce unbiased estimates, we evaluate our system on the HaluEval benchmark. By pairing our universal feature set with a lightweight, task-adapted classifier and a precision-constrained decision policy, HALT-RAG achieves strong OOF F1-scores of 0.7756, 0.9786, and 0.7391 on the summarization, QA, and dialogue tasks, respectively. The system's well-calibrated probabilities enable a practical abstention mechanism, providing a reliable tool for balancing model performance with safety requirements. 

---
# DepthVision: Robust Vision-Language Understanding through GAN-Based LiDAR-to-RGB Synthesis 

**Authors**: Sven Kirchner, Nils Purschke, Ross Greer, Alois C. Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2509.07463)  

**Abstract**: Ensuring reliable robot operation when visual input is degraded or insufficient remains a central challenge in robotics. This letter introduces DepthVision, a framework for multimodal scene understanding designed to address this problem. Unlike existing Vision-Language Models (VLMs), which use only camera-based visual input alongside language, DepthVision synthesizes RGB images from sparse LiDAR point clouds using a conditional generative adversarial network (GAN) with an integrated refiner network. These synthetic views are then combined with real RGB data using a Luminance-Aware Modality Adaptation (LAMA), which blends the two types of data dynamically based on ambient lighting conditions. This approach compensates for sensor degradation, such as darkness or motion blur, without requiring any fine-tuning of downstream vision-language models. We evaluate DepthVision on real and simulated datasets across various models and tasks, with particular attention to safety-critical tasks. The results demonstrate that our approach improves performance in low-light conditions, achieving substantial gains over RGB-only baselines while preserving compatibility with frozen VLMs. This work highlights the potential of LiDAR-guided RGB synthesis for achieving robust robot operation in real-world environments. 

---
# Bias-Aware Machine Unlearning: Towards Fairer Vision Models via Controllable Forgetting 

**Authors**: Sai Siddhartha Chary Aylapuram, Veeraraju Elluru, Shivang Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2509.07456)  

**Abstract**: Deep neural networks often rely on spurious correlations in training data, leading to biased or unfair predictions in safety-critical domains such as medicine and autonomous driving. While conventional bias mitigation typically requires retraining from scratch or redesigning data pipelines, recent advances in machine unlearning provide a promising alternative for post-hoc model correction. In this work, we investigate \textit{Bias-Aware Machine Unlearning}, a paradigm that selectively removes biased samples or feature representations to mitigate diverse forms of bias in vision models. Building on privacy-preserving unlearning techniques, we evaluate various strategies including Gradient Ascent, LoRA, and Teacher-Student distillation. Through empirical analysis on three benchmark datasets, CUB-200-2011 (pose bias), CIFAR-10 (synthetic patch bias), and CelebA (gender bias in smile detection), we demonstrate that post-hoc unlearning can substantially reduce subgroup disparities, with improvements in demographic parity of up to \textbf{94.86\%} on CUB-200, \textbf{30.28\%} on CIFAR-10, and \textbf{97.37\%} on CelebA. These gains are achieved with minimal accuracy loss and with methods scoring an average of 0.62 across the 3 settings on the joint evaluation of utility, fairness, quality, and privacy. Our findings establish machine unlearning as a practical framework for enhancing fairness in deployed vision systems without necessitating full retraining. 

---
# Text2Touch: Tactile In-Hand Manipulation with LLM-Designed Reward Functions 

**Authors**: Harrison Field, Max Yang, Yijiong Lin, Efi Psomopoulou, David Barton, Nathan F. Lepora  

**Link**: [PDF](https://arxiv.org/pdf/2509.07445)  

**Abstract**: Large language models (LLMs) are beginning to automate reward design for dexterous manipulation. However, no prior work has considered tactile sensing, which is known to be critical for human-like dexterity. We present Text2Touch, bringing LLM-crafted rewards to the challenging task of multi-axis in-hand object rotation with real-world vision based tactile sensing in palm-up and palm-down configurations. Our prompt engineering strategy scales to over 70 environment variables, and sim-to-real distillation enables successful policy transfer to a tactile-enabled fully actuated four-fingered dexterous robot hand. Text2Touch significantly outperforms a carefully tuned human-engineered baseline, demonstrating superior rotation speed and stability while relying on reward functions that are an order of magnitude shorter and simpler. These results illustrate how LLM-designed rewards can significantly reduce the time from concept to deployable dexterous tactile skills, supporting more rapid and scalable multimodal robot learning. Project website: this https URL 

---
# The Choice of Divergence: A Neglected Key to Mitigating Diversity Collapse in Reinforcement Learning with Verifiable Reward 

**Authors**: Long Li, Jiaran Hao, Jason Klein Liu, Zhijian Zhou, Xiaoyu Tan, Wei Chu, Zhe Wang, Shirui Pan, Chao Qu, Yuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2509.07430)  

**Abstract**: A central paradox in fine-tuning Large Language Models (LLMs) with Reinforcement Learning with Verifiable Reward (RLVR) is the frequent degradation of multi-attempt performance (Pass@k) despite improvements in single-attempt accuracy (Pass@1). This is often accompanied by catastrophic forgetting, where models lose previously acquired skills. While various methods have been proposed, the choice and function of the divergence term have been surprisingly unexamined as a proactive solution. We argue that standard RLVR objectives -- both those using the mode-seeking reverse KL-divergence and those forgoing a divergence term entirely -- lack a crucial mechanism for knowledge retention. The reverse-KL actively accelerates this decay by narrowing the policy, while its absence provides no safeguard against the model drifting from its diverse knowledge base. We propose a fundamental shift in perspective: using the divergence term itself as the solution. Our framework, Diversity-Preserving Hybrid RL (DPH-RL), leverages mass-covering f-divergences (like forward-KL and JS-divergence) to function as a rehearsal mechanism. By continuously referencing the initial policy, this approach forces the model to maintain broad solution coverage. Extensive experiments on math and SQL generation demonstrate that DPH-RL not only resolves the Pass@k degradation but improves both Pass@1 and Pass@k in- and out-of-domain. Additionally, DPH-RL is more training-efficient because it computes f-divergence using generator functions, requiring only sampling from the initial policy and no online reference model. Our work highlights a crucial, overlooked axis for improving RLVR, demonstrating that the proper selection of a divergence measure is a powerful tool for building more general and diverse reasoning models. 

---
# Benchmarking Universal Interatomic Potentials on Zeolite Structures 

**Authors**: Shusuke Ito, Koki Muraoka, Akira Nakayama  

**Link**: [PDF](https://arxiv.org/pdf/2509.07417)  

**Abstract**: Interatomic potentials (IPs) with wide elemental coverage and high accuracy are powerful tools for high-throughput materials discovery. While the past few years witnessed the development of multiple new universal IPs that cover wide ranges of the periodic table, their applicability to target chemical systems should be carefully investigated. We benchmark several universal IPs using equilibrium zeolite structures as testbeds. We select a diverse set of universal IPs encompassing two major categories: (i) universal analytic IPs, including GFN-FF, UFF, and Dreiding; (ii) pretrained universal machine learning IPs (MLIPs), comprising CHGNet, ORB-v3, MatterSim, eSEN-30M-OAM, PFP-v7, and EquiformerV2-lE4-lF100-S2EFS-OC22. We compare them with established tailor-made IPs, SLC, ClayFF, and BSFF using experimental data and density functional theory (DFT) calculations with dispersion correction as the reference. The tested zeolite structures comprise pure silica frameworks and aluminosilicates containing copper species, potassium, and organic cations. We found that GFN-FF is the best among the tested universal analytic IPs, but it does not achieve satisfactory accuracy for highly strained silica rings and aluminosilicate systems. All MLIPs can well reproduce experimental or DFT-level geometries and energetics. Among the universal MLIPs, the eSEN-30M-OAM model shows the most consistent performance across all zeolite structures studied. These findings show that the modern pretrained universal MLIPs are practical tools in zeolite screening workflows involving various compositions. 

---
# Toward Lifelong-Sustainable Electronic-Photonic AI Systems via Extreme Efficiency, Reconfigurability, and Robustness 

**Authors**: Ziang Yin, Hongjian Zhou, Chetan Choppali Sudarshan, Vidya Chhabria, Jiaqi Gu  

**Link**: [PDF](https://arxiv.org/pdf/2509.07396)  

**Abstract**: The relentless growth of large-scale artificial intelligence (AI) has created unprecedented demand for computational power, straining the energy, bandwidth, and scaling limits of conventional electronic platforms. Electronic-photonic integrated circuits (EPICs) have emerged as a compelling platform for next-generation AI systems, offering inherent advantages in ultra-high bandwidth, low latency, and energy efficiency for computing and interconnection. Beyond performance, EPICs also hold unique promises for sustainability. Fabricated in relaxed process nodes with fewer metal layers and lower defect densities, photonic devices naturally reduce embodied carbon footprint (CFP) compared to advanced digital electronic integrated circuits, while delivering orders-of-magnitude higher computing performance and interconnect bandwidth. To further advance the sustainability of photonic AI systems, we explore how electronic-photonic design automation (EPDA) and cross-layer co-design methodologies can amplify these inherent benefits. We present how advanced EPDA tools enable more compact layout generation, reducing both chip area and metal layer usage. We will also demonstrate how cross-layer device-circuit-architecture co-design unlocks new sustainability gains for photonic hardware: ultra-compact photonic circuit designs that minimize chip area cost, reconfigurable hardware topology that adapts to evolving AI workloads, and intelligent resilience mechanisms that prolong lifetime by tolerating variations and faults. By uniting intrinsic photonic efficiency with EPDA- and co-design-driven gains in area efficiency, reconfigurability, and robustness, we outline a vision for lifelong-sustainable electronic-photonic AI systems. This perspective highlights how EPIC AI systems can simultaneously meet the performance demands of modern AI and the urgent imperative for sustainable computing. 

---
# Hybrid GCN-GRU Model for Anomaly Detection in Cryptocurrency Transactions 

**Authors**: Gyuyeon Na, Minjung Park, Hyeonjeong Cha, Soyoun Kim, Sunyoung Moon, Sua Lee, Jaeyoung Choi, Hyemin Lee, Sangmi Chai  

**Link**: [PDF](https://arxiv.org/pdf/2509.07392)  

**Abstract**: Blockchain transaction networks are complex, with evolving temporal patterns and inter-node relationships. To detect illicit activities, we propose a hybrid GCN-GRU model that captures both structural and sequential features. Using real Bitcoin transaction data (2020-2024), our model achieved 0.9470 Accuracy and 0.9807 AUC-ROC, outperforming all baselines. 

---
# Talking with Oompa Loompas: A novel framework for evaluating linguistic acquisition of LLM agents 

**Authors**: Sankalp Tattwadarshi Swain, Anshika Krishnatray, Dhruv Kumar, Jagat Sesh Challa  

**Link**: [PDF](https://arxiv.org/pdf/2509.07389)  

**Abstract**: Existing evaluation studies on linguistic competence of large language models (LLM agents) have focused primarily on vocabulary learning, morphological rule induction, syntactic generalization, pragmatic inference, and cross-linguistic transfer. However, none assess whether LLM agents can acquire a language through pattern recognition and interactive feedback, a central feature of human language acquisition. We propose a novel experimental framework in which an LLM agent is evaluated on its ability to acquire and use a newly constructed language (Tinkatongue) in conversation with a bot that understands only Tinkatongue. Our findings show that LLM agents fail to establish a conversation within 100 responses, yet they adopt distinct strategies that mirror human approaches to language learning. The results suggest a new direction for evaluation benchmarks and open pathways to model designs that learn more effectively from interactive feedback. 

---
# SBS: Enhancing Parameter-Efficiency of Neural Representations for Neural Networks via Spectral Bias Suppression 

**Authors**: Qihu Xie, Yuan Li, Yi Kang  

**Link**: [PDF](https://arxiv.org/pdf/2509.07373)  

**Abstract**: Implicit neural representations have recently been extended to represent convolutional neural network weights via neural representation for neural networks, offering promising parameter compression benefits. However, standard multi-layer perceptrons used in neural representation for neural networks exhibit a pronounced spectral bias, hampering their ability to reconstruct high-frequency details effectively. In this paper, we propose SBS, a parameter-efficient enhancement to neural representation for neural networks that suppresses spectral bias using two techniques: (1) a unidirectional ordering-based smoothing that improves kernel smoothness in the output space, and (2) unidirectional ordering-based smoothing aware random fourier features that adaptively modulate the frequency bandwidth of input encodings based on layer-wise parameter count. Extensive evaluations on various ResNet models with datasets CIFAR-10, CIFAR-100, and ImageNet, demonstrate that SBS achieves significantly better reconstruction accuracy with less parameters compared to SOTA. 

---
# Word2Spike: Poisson Rate Coding for Associative Memories and Neuromorphic Algorithms 

**Authors**: Archit Kalra, Midhun Sadanand  

**Link**: [PDF](https://arxiv.org/pdf/2509.07361)  

**Abstract**: Spiking neural networks offer a promising path toward energy-efficient, brain-like associative memory. This paper introduces Word2Spike, a novel rate coding mechanism that combines continuous word embeddings and neuromorphic architectures. We develop a one-to-one mapping that converts multi-dimensional word vectors into spike-based attractor states using Poisson processes. Using BitNet b1.58 quantization, we maintain 97% semantic similarity of continuous embeddings on SimLex-999 while achieving 100% reconstruction accuracy on 10,000 words from OpenAI's text-embedding-3-large. We preserve analogy performance (100% of original embedding performance) even under intentionally introduced noise, indicating a resilient mechanism for semantic encoding in neuromorphic systems. Next steps include integrating the mapping with spiking transformers and liquid state machines (resembling Hopfield Networks) for further evaluation. 

---
# General Demographic Foundation Models for Enhancing Predictive Performance Across Diseases 

**Authors**: Li-Chin Chen, Ji-Tian Sheu, Yuh-Jue Chuang  

**Link**: [PDF](https://arxiv.org/pdf/2509.07330)  

**Abstract**: Demographic attributes are universally present in electronic health records and serve as vital predictors in clinical risk stratification and treatment decisions. Despite their significance, these attributes are often relegated to auxiliary roles in model design, with limited attention has been given to learning their representations. This study proposes a General Demographic Pre-trained (GDP) model as a foundational representation framework tailored to age and gender. The model is pre-trained and evaluated using datasets with diverse diseases and population compositions from different geographic regions. The GDP architecture explores combinations of ordering strategies and encoding methods to transform tabular demographic inputs into latent embeddings. Experimental results demonstrate that sequential ordering substantially improves model performance in discrimination, calibration, and the corresponding information gain at each decision tree split, particularly in diseases where age and gender contribute significantly to risk stratification. Even in datasets where demographic attributes hold relatively low predictive value, GDP enhances the representational importance, increasing their influence in downstream gradient boosting models. The findings suggest that foundational models for tabular demographic attributes can generalize across tasks and populations, offering a promising direction for improving predictive performance in healthcare applications. 

---
# DEPF: A UAV Multispectral Object Detector with Dual-Domain Enhancement and Priority-Guided Mamba Fusion 

**Authors**: Shucong Li, Zhenyu Liu, Zijie Hong, Zhiheng Zhou, Xianghai Cao  

**Link**: [PDF](https://arxiv.org/pdf/2509.07327)  

**Abstract**: Multispectral remote sensing object detection is one of the important application of unmanned aerial vehicle (UAV). However, it faces three challenges. Firstly, the low-light remote sensing images reduce the complementarity during multi-modality fusion. Secondly, the local small target modeling is interfered with redundant information in the fusion stage easily. Thirdly, due to the quadratic computational complexity, it is hard to apply the transformer-based methods on the UAV platform. To address these limitations, motivated by Mamba with linear complexity, a UAV multispectral object detector with dual-domain enhancement and priority-guided mamba fusion (DEPF) is proposed. Firstly, to enhance low-light remote sensing images, Dual-Domain Enhancement Module (DDE) is designed, which contains Cross-Scale Wavelet Mamba (CSWM) and Fourier Details Recovery block (FDR). CSWM applies cross-scale mamba scanning for the low-frequency components to enhance the global brightness of images, while FDR constructs spectrum recovery network to enhance the frequency spectra features for recovering the texture-details. Secondly, to enhance local target modeling and reduce the impact of redundant information during fusion, Priority-Guided Mamba Fusion Module (PGMF) is designed. PGMF introduces the concept of priority scanning, which starts from local targets features according to the priority scores obtained from modality difference. Experiments on DroneVehicle dataset and VEDAI dataset reports that, DEPF performs well on object detection, comparing with state-of-the-art methods. Our code is available in the supplementary material. 

---
# Mitigating Attention Localization in Small Scale: Self-Attention Refinement via One-step Belief Propagation 

**Authors**: Nakyung Lee, Yeongoon Kim, Minhae Oh, Suhwan Kim, Jin Woo Koo, Hyewon Jo, Jungwoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.07324)  

**Abstract**: Transformer-based self-attention mechanism serves as the core of modern language models, yet it often suffers from localization, where attentions collapse onto a limited subset of tokens and fail to capture long-range dependencies. To address this issue, we propose Self-Attention One-step Belief Propagation (SAOBP), a refinement framework that injects multi-hop relationships through a belief propagation process. To interpret and quantify these interactions, we introduce Global Token Dependency (GTD) that captures the relative contribution of multihop connections within the attention graph. Empirical results indicate that SAOBP helps prevent entropy collapse in deeper layers and adaptively maintains GTD at task-appropriate levels, thereby supporting improvements in model performance. Importantly, we observe competitive gains in small-scale models, highlighting its potential for improving inference quality in resource-constrained scenarios. 

---
# MEGG: Replay via Maximally Extreme GGscore in Incremental Learning for Neural Recommendation Models 

**Authors**: Yunxiao Shi, Shuo Yang, Haimin Zhang, Li Wang, Yongze Wang, Qiang Wu, Min Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.07319)  

**Abstract**: Neural Collaborative Filtering models are widely used in recommender systems but are typically trained under static settings, assuming fixed data distributions. This limits their applicability in dynamic environments where user preferences evolve. Incremental learning offers a promising solution, yet conventional methods from computer vision or NLP face challenges in recommendation tasks due to data sparsity and distinct task paradigms. Existing approaches for neural recommenders remain limited and often lack generalizability. To address this, we propose MEGG, Replay Samples with Maximally Extreme GGscore, an experience replay based incremental learning framework. MEGG introduces GGscore, a novel metric that quantifies sample influence, enabling the selective replay of highly influential samples to mitigate catastrophic forgetting. Being model-agnostic, MEGG integrates seamlessly across architectures and frameworks. Experiments on three neural models and four benchmark datasets show superior performance over state-of-the-art baselines, with strong scalability, efficiency, and robustness. Implementation will be released publicly upon acceptance. 

---
# Does This Look Familiar to You? Knowledge Analysis via Model Internal Representations 

**Authors**: Sihyun Park  

**Link**: [PDF](https://arxiv.org/pdf/2509.07311)  

**Abstract**: Recent advances in large language models (LLMs) have been driven by pretraining, supervised fine tuning (SFT), and alignment tuning. Among these, SFT plays a crucial role in transforming a model 's general knowledge into structured responses tailored to specific tasks. However, there is no clearly established methodology for effective training data selection. Simply increasing the volume of data does not guarantee performance improvements, while preprocessing, sampling, and validation require substantial time and cost.
To address this issue, a variety of data selection methods have been proposed. Among them, knowledge based selection approaches identify suitable training data by analyzing the model 's responses. Nevertheless, these methods typically rely on prompt engineering, making them sensitive to variations and incurring additional costs for prompt design.
In this study, we propose Knowledge Analysis via Model Internal Representations (KAMIR), a novel approach that overcomes these limitations by analyzing data based on the model 's internal representations. KAMIR computes similarities between the hidden states of each layer (block) and the final hidden states for a given input to assess the data. Unlike prior methods that were largely limited to multiple choice tasks, KAMIR can be applied to a wide range of tasks such as machine reading comprehension and summarization. Moreover, it selects data useful for training based on the model 's familiarity with the input, even with a small dataset and a simple classifier architecture. Experiments across diverse task datasets demonstrate that training with less familiar data leads to better generalization performance. 

---
# Basis Vector Metric: A Method for Robust Open-Ended State Change Detection 

**Authors**: David Oprea, Sam Powers  

**Link**: [PDF](https://arxiv.org/pdf/2509.07308)  

**Abstract**: We test a new method, which we will abbreviate using the acronym BVM (Basis Vectors Method), in its ability to judge the state changes in images through using language embeddings. We used the MIT-States dataset, containing about 53,000 images, to gather all of our data, which has 225 nouns and 115 adjectives, with each noun having about 9 different adjectives, forming approximately 1000 noun-adjective pairs. For our first experiment, we test our method's ability to determine the state of each noun class separately against other metrics for comparison. These metrics are cosine similarity, dot product, product quantization, binary index, Naive Bayes, and a custom neural network. Among these metrics, we found that our proposed BVM performs the best in classifying the states for each noun. We then perform a second experiment where we try using BVM to determine if it can differentiate adjectives from one another for each adjective separately. We compared the abilities of BVM to differentiate adjectives against the proposed method the MIT-States paper suggests: using a logistic regression model. In the end, we did not find conclusive evidence that our BVM metric could perform better than the logistic regression model at discerning adjectives. Yet, we were able to find evidence for possible improvements to our method; this leads to the chance of increasing our method's accuracy through certain changes in our methodologies. 

---
# Reconstruction Alignment Improves Unified Multimodal Models 

**Authors**: Ji Xie, Trevor Darrell, Luke Zettlemoyer, XuDong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.07295)  

**Abstract**: Unified multimodal models (UMMs) unify visual understanding and generation within a single architecture. However, conventional training relies on image-text pairs (or sequences) whose captions are typically sparse and miss fine-grained visual details--even when they use hundreds of words to describe a simple image. We introduce Reconstruction Alignment (RecA), a resource-efficient post-training method that leverages visual understanding encoder embeddings as dense "text prompts," providing rich supervision without captions. Concretely, RecA conditions a UMM on its own visual understanding embeddings and optimizes it to reconstruct the input image with a self-supervised reconstruction loss, thereby realigning understanding and generation. Despite its simplicity, RecA is broadly applicable: across autoregressive, masked-autoregressive, and diffusion-based UMMs, it consistently improves generation and editing fidelity. With only 27 GPU-hours, post-training with RecA substantially improves image generation performance on GenEval (0.73$\rightarrow$0.90) and DPGBench (80.93$\rightarrow$88.15), while also boosting editing benchmarks (ImgEdit 3.38$\rightarrow$3.75, GEdit 6.94$\rightarrow$7.25). Notably, RecA surpasses much larger open-source models and applies broadly across diverse UMM architectures, establishing it as an efficient and general post-training alignment strategy for UMMs 

---
# zkUnlearner: A Zero-Knowledge Framework for Verifiable Unlearning with Multi-Granularity and Forgery-Resistance 

**Authors**: Nan Wang, Nan Wu, Xiangyu Hui, Jiafan Wang, Xin Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2509.07290)  

**Abstract**: As the demand for exercising the "right to be forgotten" grows, the need for verifiable machine unlearning has become increasingly evident to ensure both transparency and accountability. We present {\em zkUnlearner}, the first zero-knowledge framework for verifiable machine unlearning, specifically designed to support {\em multi-granularity} and {\em forgery-resistance}.
First, we propose a general computational model that employs a {\em bit-masking} technique to enable the {\em selectivity} of existing zero-knowledge proofs of training for gradient descent algorithms. This innovation enables not only traditional {\em sample-level} unlearning but also more advanced {\em feature-level} and {\em class-level} unlearning. Our model can be translated to arithmetic circuits, ensuring compatibility with a broad range of zero-knowledge proof systems. Furthermore, our approach overcomes key limitations of existing methods in both efficiency and privacy. Second, forging attacks present a serious threat to the reliability of unlearning. Specifically, in Stochastic Gradient Descent optimization, gradients from unlearned data, or from minibatches containing it, can be forged using alternative data samples or minibatches that exclude it. We propose the first effective strategies to resist state-of-the-art forging attacks. Finally, we benchmark a zkSNARK-based instantiation of our framework and perform comprehensive performance evaluations to validate its practicality. 

---
# Paladin: Defending LLM-enabled Phishing Emails with a New Trigger-Tag Paradigm 

**Authors**: Yan Pang, Wenlong Meng, Xiaojing Liao, Tianhao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.07287)  

**Abstract**: With the rapid development of large language models, the potential threat of their malicious use, particularly in generating phishing content, is becoming increasingly prevalent. Leveraging the capabilities of LLMs, malicious users can synthesize phishing emails that are free from spelling mistakes and other easily detectable features. Furthermore, such models can generate topic-specific phishing messages, tailoring content to the target domain and increasing the likelihood of success.
Detecting such content remains a significant challenge, as LLM-generated phishing emails often lack clear or distinguishable linguistic features. As a result, most existing semantic-level detection approaches struggle to identify them reliably. While certain LLM-based detection methods have shown promise, they suffer from high computational costs and are constrained by the performance of the underlying language model, making them impractical for large-scale deployment.
In this work, we aim to address this issue. We propose Paladin, which embeds trigger-tag associations into vanilla LLM using various insertion strategies, creating them into instrumented LLMs. When an instrumented LLM generates content related to phishing, it will automatically include detectable tags, enabling easier identification. Based on the design on implicit and explicit triggers and tags, we consider four distinct scenarios in our work. We evaluate our method from three key perspectives: stealthiness, effectiveness, and robustness, and compare it with existing baseline methods. Experimental results show that our method outperforms the baselines, achieving over 90% detection accuracy across all scenarios. 

---
# ALICE: An Interpretable Neural Architecture for Generalization in Substitution Ciphers 

**Authors**: Jeff Shen, Lindsay Smith  

**Link**: [PDF](https://arxiv.org/pdf/2509.07282)  

**Abstract**: We present cryptogram solving as an ideal testbed for studying neural network generalization in combinatorially complex domains. In this task, models must decrypt text encoded with substitution ciphers, choosing from 26! possible mappings without explicit access to the cipher. We develop ALICE (an Architecture for Learning Interpretable Cryptogram dEcipherment): a simple encoder-only Transformer that sets a new state-of-the-art for both accuracy and speed on this decryption problem. Surprisingly, ALICE generalizes to unseen ciphers after training on only ${\sim}1500$ unique ciphers, a minute fraction ($3.7 \times 10^{-24}$) of the possible cipher space. To enhance interpretability, we introduce a novel bijective decoding head that explicitly models permutations via the Gumbel-Sinkhorn method, enabling direct extraction of learned cipher mappings. Through early exit analysis, we reveal how ALICE progressively refines its predictions in a way that appears to mirror common human strategies for this task: early layers employ frequency-based heuristics, middle layers form word structures, and final layers correct individual characters. Our architectural innovations and analysis methods extend beyond cryptograms to any domain with bijective mappings and combinatorial structure, offering new insights into neural network generalization and interpretability. 

---
# Breast Cancer Detection in Thermographic Images via Diffusion-Based Augmentation and Nonlinear Feature Fusion 

**Authors**: Sepehr Salem, M. Moein Esfahani, Jingyu Liu, Vince Calhoun  

**Link**: [PDF](https://arxiv.org/pdf/2509.07277)  

**Abstract**: Data scarcity hinders deep learning for medical imaging. We propose a framework for breast cancer classification in thermograms that addresses this using a Diffusion Probabilistic Model (DPM) for data augmentation. Our DPM-based augmentation is shown to be superior to both traditional methods and a ProGAN baseline. The framework fuses deep features from a pre-trained ResNet-50 with handcrafted nonlinear features (e.g., Fractal Dimension) derived from U-Net segmented tumors. An XGBoost classifier trained on these fused features achieves 98.0\% accuracy and 98.1\% sensitivity. Ablation studies and statistical tests confirm that both the DPM augmentation and the nonlinear feature fusion are critical, statistically significant components of this success. This work validates the synergy between advanced generative models and interpretable features for creating highly accurate medical diagnostic tools. 

---
# Datasets for Navigating Sensitive Topics in Recommendation Systems 

**Authors**: Amelia Kovacs, Jerry Chee, Kimia Kazemian, Sarah Dean  

**Link**: [PDF](https://arxiv.org/pdf/2509.07269)  

**Abstract**: Personalized AI systems, from recommendation systems to chatbots, are a prevalent method for distributing content to users based on their learned preferences. However, there is growing concern about the adverse effects of these systems, including their potential tendency to expose users to sensitive or harmful material, negatively impacting overall well-being. To address this concern quantitatively, it is necessary to create datasets with relevant sensitivity labels for content, enabling researchers to evaluate personalized systems beyond mere engagement metrics. To this end, we introduce two novel datasets that include a taxonomy of sensitivity labels alongside user-content ratings: one that integrates MovieLens rating data with content warnings from the Does the Dog Die? community ratings website, and another that combines fan-fiction interaction data and user-generated warnings from Archive of Our Own. 

---
# Benchmarking Information Retrieval Models on Complex Retrieval Tasks 

**Authors**: Julian Killingback, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2509.07253)  

**Abstract**: Large language models (LLMs) are incredible and versatile tools for text-based tasks that have enabled countless, previously unimaginable, applications. Retrieval models, in contrast, have not yet seen such capable general-purpose models emerge. To achieve this goal, retrieval models must be able to perform complex retrieval tasks, where queries contain multiple parts, constraints, or requirements in natural language. These tasks represent a natural progression from the simple, single-aspect queries that are used in the vast majority of existing, commonly used evaluation sets. Complex queries naturally arise as people expect search systems to handle more specific and often ambitious information requests, as is demonstrated by how people use LLM-based information systems. Despite the growing desire for retrieval models to expand their capabilities in complex retrieval tasks, there exist limited resources to assess the ability of retrieval models on a comprehensive set of diverse complex tasks. The few resources that do exist feature a limited scope and often lack realistic settings making it hard to know the true capabilities of retrieval models on complex real-world retrieval tasks. To address this shortcoming and spur innovation in next-generation retrieval models, we construct a diverse and realistic set of complex retrieval tasks and benchmark a representative set of state-of-the-art retrieval models. Additionally, we explore the impact of LLM-based query expansion and rewriting on retrieval quality. Our results show that even the best models struggle to produce high-quality retrieval results with the highest average nDCG@10 of only 0.346 and R@100 of only 0.587 across all tasks. Although LLM augmentation can help weaker models, the strongest model has decreased performance across all metrics with all rewriting techniques. 

---
# Systematic Optimization of Open Source Large Language Models for Mathematical Reasoning 

**Authors**: Pranav Pawar, Dhwaj Jain, Varun Gupta, Kaustav Dedhia, Dashrath Kale, Sudhir Dhekane  

**Link**: [PDF](https://arxiv.org/pdf/2509.07238)  

**Abstract**: This paper presents a practical investigation into fine-tuning model parameters for mathematical reasoning tasks through experimenting with various configurations including randomness control, reasoning depth, and sampling strategies, careful tuning demonstrates substantial improvements in efficiency as well as performance. A holistically optimized framework is introduced for five state-of-the-art models on mathematical reasoning tasks, exhibiting significant performance boosts while maintaining solution correctness. Through systematic parameter optimization across Qwen2.5-72B, Llama-3.1-70B, DeepSeek-V3, Mixtral-8x22B, and Yi-Lightning, consistent efficiency gains are demonstrated with 100% optimization success rate. The methodology achieves an average 29.4% reduction in computational cost and 23.9% improvement in inference speed across all tested models. This framework systematically searches parameter spaces including temperature (0.1-0.5), reasoning steps (4-12), planning periods (1-4), and nucleus sampling (0.85-0.98), determining optimal configurations through testing on mathematical reasoning benchmarks. Critical findings show that lower temperature regimes (0.1-0.4) and reduced reasoning steps (4-6) consistently enhance efficiency without compromising accuracy. DeepSeek-V3 achieves the highest accuracy at 98%, while Mixtral-8x22B delivers the most cost-effective performance at 361.5 tokens per accurate response. Key contributions include: (1) the first comprehensive optimization study for five diverse SOTA models in mathematical reasoning, (2) a standardized production-oriented parameter optimization framework, (3) discovery of universal optimization trends applicable across model architectures, and (4) production-ready configurations with extensive performance characterization. 

---
# Breaking the Conventional Forward-Backward Tie in Neural Networks: Activation Functions 

**Authors**: Luigi Troiano, Francesco Gissi, Vincenzo Benedetto, Genny Tortora  

**Link**: [PDF](https://arxiv.org/pdf/2509.07236)  

**Abstract**: Gradient-based neural network training traditionally enforces symmetry between forward and backward propagation, requiring activation functions to be differentiable (or sub-differentiable) and strictly monotonic in certain regions to prevent flat gradient areas. This symmetry, linking forward activations closely to backward gradients, significantly restricts the selection of activation functions, particularly excluding those with substantial flat or non-differentiable regions. In this paper, we challenge this assumption through mathematical analysis, demonstrating that precise gradient magnitudes derived from activation functions are largely redundant, provided the gradient direction is preserved. Empirical experiments conducted on foundational architectures - such as Multi-Layer Perceptrons (MLPs), Convolutional Neural Networks (CNNs), and Binary Neural Networks (BNNs) - confirm that relaxing forward-backward symmetry and substituting traditional gradients with simpler or stochastic alternatives does not impair learning and may even enhance training stability and efficiency. We explicitly demonstrate that neural networks with flat or non-differentiable activation functions, such as the Heaviside step function, can be effectively trained, thereby expanding design flexibility and computational efficiency. Further empirical validation with more complex architectures remains a valuable direction for future research. 

---
# A transformer-based generative model for planetary systems 

**Authors**: Yann Alibert, Jeanne Davoult, Sara Marques  

**Link**: [PDF](https://arxiv.org/pdf/2509.07226)  

**Abstract**: Numerical calculations of planetary system formation are very demanding in terms of computing power. These synthetic planetary systems can however provide access to correlations, as predicted in a given numerical framework, between the properties of planets in the same system. Such correlations can, in return, be used in order to guide and prioritize observational campaigns aiming at discovering some types of planets, as Earth-like planets. Our goal is to develop a generative model which is capable of capturing correlations and statistical relationships between planets in the same system. Such a model, trained on the Bern model, offers the possibility to generate large number of synthetic planetary systems with little computational cost, that can be used, for example, to guide observational campaigns. Our generative model is based on the transformer architecture which is well-known to efficiently capture correlations in sequences and is at the basis of all modern Large Language Models. To assess the validity of the generative model, we perform visual and statistical comparisons, as well as a machine learning driven tests. Finally, as a use case example, we consider the TOI-469 system, in which we aim at predicting the possible properties of planets c and d, based on the properties of planet b (the first that has been detected). We show using different comparison methods that the properties of systems generated by our model are very similar to the ones of the systems computed directly by the Bern model. We also show in the case of the TOI-469 system, that using the generative model allows to predict the properties of planets not yet observed, based on the properties of the already observed planet. We provide our model to the community on our website this http URL. 

---
# Explaining How Quantization Disparately Skews a Model 

**Authors**: Abhimanyu Bellam, Jung-Eun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.07222)  

**Abstract**: Post Training Quantization (PTQ) is widely adopted due to its high compression capacity and speed with minimal impact on accuracy. However, we observed that disparate impacts are exacerbated by quantization, especially for minority groups. Our analysis explains that in the course of quantization there is a chain of factors attributed to a disparate impact across groups during forward and backward passes. We explore how the changes in weights and activations induced by quantization cause cascaded impacts in the network, resulting in logits with lower variance, increased loss, and compromised group accuracies. We extend our study to verify the influence of these impacts on group gradient norms and eigenvalues of the Hessian matrix, providing insights into the state of the network from an optimization point of view. To mitigate these effects, we propose integrating mixed precision Quantization Aware Training (QAT) with dataset sampling methods and weighted loss functions, therefore providing fair deployment of quantized neural networks. 

---
# XBusNet: Text-Guided Breast Ultrasound Segmentation via Multimodal Vision-Language Learning 

**Authors**: Raja Mallina, Bryar Shareef  

**Link**: [PDF](https://arxiv.org/pdf/2509.07213)  

**Abstract**: Background: Precise breast ultrasound (BUS) segmentation supports reliable measurement, quantitative analysis, and downstream classification, yet remains difficult for small or low-contrast lesions with fuzzy margins and speckle noise. Text prompts can add clinical context, but directly applying weakly localized text-image cues (e.g., CAM/CLIP-derived signals) tends to produce coarse, blob-like responses that smear boundaries unless additional mechanisms recover fine edges. Methods: We propose XBusNet, a novel dual-prompt, dual-branch multimodal model that combines image features with clinically grounded text. A global pathway based on a CLIP Vision Transformer encodes whole-image semantics conditioned on lesion size and location, while a local U-Net pathway emphasizes precise boundaries and is modulated by prompts that describe shape, margin, and Breast Imaging Reporting and Data System (BI-RADS) terms. Prompts are assembled automatically from structured metadata, requiring no manual clicks. We evaluate on the Breast Lesions USG (BLU) dataset using five-fold cross-validation. Primary metrics are Dice and Intersection over Union (IoU); we also conduct size-stratified analyses and ablations to assess the roles of the global and local paths and the text-driven modulation. Results: XBusNet achieves state-of-the-art performance on BLU, with mean Dice of 0.8765 and IoU of 0.8149, outperforming six strong baselines. Small lesions show the largest gains, with fewer missed regions and fewer spurious activations. Ablation studies show complementary contributions of global context, local boundary modeling, and prompt-based modulation. Conclusions: A dual-prompt, dual-branch multimodal design that merges global semantics with local precision yields accurate BUS segmentation masks and improves robustness for small, low-contrast lesions. 

---
# A multi-strategy improved gazelle optimization algorithm for solving numerical optimization and engineering applications 

**Authors**: Qi Diao, Chengyue Xie, Yuchen Yin, Hoileong Lee, Haolong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.07211)  

**Abstract**: Aiming at the shortcomings of the gazelle optimization algorithm, such as the imbalance between exploration and exploitation and the insufficient information exchange within the population, this paper proposes a multi-strategy improved gazelle optimization algorithm (MSIGOA). To address these issues, MSIGOA proposes an iteration-based updating framework that switches between exploitation and exploration according to the optimization process, which effectively enhances the balance between local exploitation and global exploration in the optimization process and improves the convergence speed. Two adaptive parameter tuning strategies improve the applicability of the algorithm and promote a smoother optimization process. The dominant population-based restart strategy enhances the algorithms ability to escape from local optima and avoid its premature convergence. These enhancements significantly improve the exploration and exploitation capabilities of MSIGOA, bringing superior convergence and efficiency in dealing with complex problems. In this paper, the parameter sensitivity, strategy effectiveness, convergence and stability of the proposed method are evaluated on two benchmark test sets including CEC2017 and CEC2022. Test results and statistical tests show that MSIGOA outperforms basic GOA and other advanced algorithms. On the CEC2017 and CEC2022 test sets, the proportion of functions where MSIGOA is not worse than GOA is 92.2% and 83.3%, respectively, and the proportion of functions where MSIGOA is not worse than other algorithms is 88.57% and 87.5%, respectively. Finally, the extensibility of MSIGAO is further verified by several engineering design optimization problems. 

---
# Evaluation of Machine Learning Reconstruction Techniques for Accelerated Brain MRI Scans 

**Authors**: Jonathan I. Mandel, Shivaprakash Hiremath, Hedyeh Keshtgar, Timothy Scholl, Sadegh Raeisi  

**Link**: [PDF](https://arxiv.org/pdf/2509.07193)  

**Abstract**: This retrospective-prospective study evaluated whether a deep learning-based MRI reconstruction algorithm can preserve diagnostic quality in brain MRI scans accelerated up to fourfold, using both public and prospective clinical data. The study included 18 healthy volunteers (scans acquired at 3T, January 2024-March 2025), as well as selected fastMRI public datasets with diverse pathologies. Phase-encoding-undersampled 2D/3D T1, T2, and FLAIR sequences were reconstructed with DeepFoqus-Accelerate and compared with standard-of-care (SOC). Three board-certified neuroradiologists and two MRI technologists independently reviewed 36 paired SOC/AI reconstructions from both datasets using a 5-point Likert scale, while quantitative similarity was assessed for 408 scans and 1224 datasets using Structural Similarity Index (SSIM), Peak Signal-to-Noise Ratio (PSNR), and Haar wavelet-based Perceptual Similarity Index (HaarPSI). No AI-reconstructed scan scored below 3 (minimally acceptable), and 95% scored $\geq 4$. Mean SSIM was 0.95 $\pm$ 0.03 (90% cases >0.90), PSNR >41.0 dB, and HaarPSI >0.94. Inter-rater agreement was slight to moderate. Rare artifacts did not affect diagnostic interpretation. These findings demonstrate that DeepFoqus-Accelerate enables robust fourfold brain MRI acceleration with 75% reduced scan time, while preserving diagnostic image quality and supporting improved workflow efficiency. 

---
# DischargeSim: A Simulation Benchmark for Educational Doctor-Patient Communication at Discharge 

**Authors**: Zonghai Yao, Michael Sun, Won Seok Jang, Sunjae Kwon, Soie Kwon, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.07188)  

**Abstract**: Discharge communication is a critical yet underexplored component of patient care, where the goal shifts from diagnosis to education. While recent large language model (LLM) benchmarks emphasize in-visit diagnostic reasoning, they fail to evaluate models' ability to support patients after the visit. We introduce DischargeSim, a novel benchmark that evaluates LLMs on their ability to act as personalized discharge educators. DischargeSim simulates post-visit, multi-turn conversations between LLM-driven DoctorAgents and PatientAgents with diverse psychosocial profiles (e.g., health literacy, education, emotion). Interactions are structured across six clinically grounded discharge topics and assessed along three axes: (1) dialogue quality via automatic and LLM-as-judge evaluation, (2) personalized document generation including free-text summaries and structured AHRQ checklists, and (3) patient comprehension through a downstream multiple-choice exam. Experiments across 18 LLMs reveal significant gaps in discharge education capability, with performance varying widely across patient profiles. Notably, model size does not always yield better education outcomes, highlighting trade-offs in strategy use and content prioritization. DischargeSim offers a first step toward benchmarking LLMs in post-visit clinical education and promoting equitable, personalized patient support. 

---
# Measuring Uncertainty in Transformer Circuits with Effective Information Consistency 

**Authors**: Anatoly A. Krasnovsky  

**Link**: [PDF](https://arxiv.org/pdf/2509.07149)  

**Abstract**: Mechanistic interpretability has identified functional subgraphs within large language models (LLMs), known as Transformer Circuits (TCs), that appear to implement specific algorithms. Yet we lack a formal, single-pass way to quantify when an active circuit is behaving coherently and thus likely trustworthy. Building on prior systems-theoretic proposals, we specialize a sheaf/cohomology and causal emergence perspective to TCs and introduce the Effective-Information Consistency Score (EICS). EICS combines (i) a normalized sheaf inconsistency computed from local Jacobians and activations, with (ii) a Gaussian EI proxy for circuit-level causal emergence derived from the same forward state. The construction is white-box, single-pass, and makes units explicit so that the score is dimensionless. We further provide practical guidance on score interpretation, computational overhead (with fast and exact modes), and a toy sanity-check analysis. Empirical validation on LLM tasks is deferred. 

---
# Toward Purpose-oriented Topic Model Evaluation enabled by Large Language Models 

**Authors**: Zhiyin Tan, Jennifer D'Souza  

**Link**: [PDF](https://arxiv.org/pdf/2509.07142)  

**Abstract**: This study presents a framework for automated evaluation of dynamically evolving topic models using Large Language Models (LLMs). Topic modeling is essential for organizing and retrieving scholarly content in digital library systems, helping users navigate complex and evolving knowledge domains. However, widely used automated metrics, such as coherence and diversity, often capture only narrow statistical patterns and fail to explain semantic failures in practice. We introduce a purpose-oriented evaluation framework that employs nine LLM-based metrics spanning four key dimensions of topic quality: lexical validity, intra-topic semantic soundness, inter-topic structural soundness, and document-topic alignment soundness. The framework is validated through adversarial and sampling-based protocols, and is applied across datasets spanning news articles, scholarly publications, and social media posts, as well as multiple topic modeling methods and open-source LLMs. Our analysis shows that LLM-based metrics provide interpretable, robust, and task-relevant assessments, uncovering critical weaknesses in topic models such as redundancy and semantic drift, which are often missed by traditional metrics. These results support the development of scalable, fine-grained evaluation tools for maintaining topic relevance in dynamic datasets. All code and data supporting this work are accessible at this https URL. 

---
# Adversarial Attacks on Audio Deepfake Detection: A Benchmark and Comparative Study 

**Authors**: Kutub Uddin, Muhammad Umar Farooq, Awais Khan, Khalid Mahmood Malik  

**Link**: [PDF](https://arxiv.org/pdf/2509.07132)  

**Abstract**: The widespread use of generative AI has shown remarkable success in producing highly realistic deepfakes, posing a serious threat to various voice biometric applications, including speaker verification, voice biometrics, audio conferencing, and criminal investigations. To counteract this, several state-of-the-art (SoTA) audio deepfake detection (ADD) methods have been proposed to identify generative AI signatures to distinguish between real and deepfake audio. However, the effectiveness of these methods is severely undermined by anti-forensic (AF) attacks that conceal generative signatures. These AF attacks span a wide range of techniques, including statistical modifications (e.g., pitch shifting, filtering, noise addition, and quantization) and optimization-based attacks (e.g., FGSM, PGD, C \& W, and DeepFool). In this paper, we investigate the SoTA ADD methods and provide a comparative analysis to highlight their effectiveness in exposing deepfake signatures, as well as their vulnerabilities under adversarial conditions. We conducted an extensive evaluation of ADD methods on five deepfake benchmark datasets using two categories: raw and spectrogram-based approaches. This comparative analysis enables a deeper understanding of the strengths and limitations of SoTA ADD methods against diverse AF attacks. It does not only highlight vulnerabilities of ADD methods, but also informs the design of more robust and generalized detectors for real-world voice biometrics. It will further guide future research in developing adaptive defense strategies that can effectively counter evolving AF techniques. 

---
# SoK: Security and Privacy of AI Agents for Blockchain 

**Authors**: Nicolò Romandini, Carlo Mazzocca, Kai Otsuki, Rebecca Montanari  

**Link**: [PDF](https://arxiv.org/pdf/2509.07131)  

**Abstract**: Blockchain and smart contracts have garnered significant interest in recent years as the foundation of a decentralized, trustless digital ecosystem, thereby eliminating the need for traditional centralized authorities. Despite their central role in powering Web3, their complexity still presents significant barriers for non-expert users. To bridge this gap, Artificial Intelligence (AI)-based agents have emerged as valuable tools for interacting with blockchain environments, supporting a range of tasks, from analyzing on-chain data and optimizing transaction strategies to detecting vulnerabilities within smart contracts. While interest in applying AI to blockchain is growing, the literature still lacks a comprehensive survey that focuses specifically on the intersection with AI agents. Most of the related work only provides general considerations, without focusing on any specific domain. This paper addresses this gap by presenting the first Systematization of Knowledge dedicated to AI-driven systems for blockchain, with a special focus on their security and privacy dimensions, shedding light on their applications, limitations, and future research directions. 

---
# SVGauge: Towards Human-Aligned Evaluation for SVG Generation 

**Authors**: Leonardo Zini, Elia Frigieri, Sebastiano Aloscari, Marcello Generali, Lorenzo Dodi, Robert Dosen, Lorenzo Baraldi  

**Link**: [PDF](https://arxiv.org/pdf/2509.07127)  

**Abstract**: Generated Scalable Vector Graphics (SVG) images demand evaluation criteria tuned to their symbolic and vectorial nature: criteria that existing metrics such as FID, LPIPS, or CLIPScore fail to satisfy. In this paper, we introduce SVGauge, the first human-aligned, reference based metric for text-to-SVG generation. SVGauge jointly measures (i) visual fidelity, obtained by extracting SigLIP image embeddings and refining them with PCA and whitening for domain alignment, and (ii) semantic consistency, captured by comparing BLIP-2-generated captions of the SVGs against the original prompts in the combined space of SBERT and TF-IDF. Evaluation on the proposed SHE benchmark shows that SVGauge attains the highest correlation with human judgments and reproduces system-level rankings of eight zero-shot LLM-based generators more faithfully than existing metrics. Our results highlight the necessity of vector-specific evaluation and provide a practical tool for benchmarking future text-to-SVG generation models. 

---
# Riemannian Batch Normalization: A Gyro Approach 

**Authors**: Ziheng Chen, Xiao-Jun Wu, Nicu Sebe  

**Link**: [PDF](https://arxiv.org/pdf/2509.07115)  

**Abstract**: Normalization layers are crucial for deep learning, but their Euclidean formulations are inadequate for data on manifolds. On the other hand, many Riemannian manifolds in machine learning admit gyro-structures, enabling principled extensions of Euclidean neural networks to non-Euclidean domains. Inspired by this, we introduce GyroBN, a principled Riemannian batch normalization framework for gyrogroups. We establish two necessary conditions, namely \emph{pseudo-reduction} and \emph{gyroisometric gyrations}, that guarantee GyroBN with theoretical control over sample statistics, and show that these conditions hold for all known gyrogroups in machine learning. Our framework also incorporates several existing Riemannian normalization methods as special cases. We further instantiate GyroBN on seven representative geometries, including the Grassmannian, five constant curvature spaces, and the correlation manifold, and derive novel gyro and Riemannian structures to enable these instantiations. Experiments across these geometries demonstrate the effectiveness of GyroBN. The code is available at this https URL. 

---
# Lookup multivariate Kolmogorov-Arnold Networks 

**Authors**: Sergey Pozdnyakov, Philippe Schwaller  

**Link**: [PDF](https://arxiv.org/pdf/2509.07103)  

**Abstract**: High-dimensional linear mappings, or linear layers, dominate both the parameter count and the computational cost of most modern deep-learning models. We introduce a general drop-in replacement, lookup multivariate Kolmogorov-Arnold Networks (lmKANs), which deliver a substantially better trade-off between capacity and inference cost. Our construction expresses a general high-dimensional mapping through trainable low-dimensional multivariate functions. These functions can carry dozens or hundreds of trainable parameters each, and yet it takes only a few multiplications to compute them because they are implemented as spline lookup tables. Empirically, lmKANs reduce inference FLOPs by up to 6.0x while matching the flexibility of MLPs in general high-dimensional function approximation. In another feedforward fully connected benchmark, on the tabular-like dataset of randomly displaced methane configurations, lmKANs enable more than 10x higher H100 throughput at equal accuracy. Within frameworks of Convolutional Neural Networks, lmKAN-based CNNs cut inference FLOPs at matched accuracy by 1.6-2.1x and by 1.7x on the CIFAR-10 and ImageNet-1k datasets, respectively. Our code, including dedicated CUDA kernels, is available online at this https URL. 

---
# Automated Evaluation of Gender Bias Across 13 Large Multimodal Models 

**Authors**: Juan Manuel Contreras  

**Link**: [PDF](https://arxiv.org/pdf/2509.07050)  

**Abstract**: Large multimodal models (LMMs) have revolutionized text-to-image generation, but they risk perpetuating the harmful social biases in their training data. Prior work has identified gender bias in these models, but methodological limitations prevented large-scale, comparable, cross-model analysis. To address this gap, we introduce the Aymara Image Fairness Evaluation, a benchmark for assessing social bias in AI-generated images. We test 13 commercially available LMMs using 75 procedurally-generated, gender-neutral prompts to generate people in stereotypically-male, stereotypically-female, and non-stereotypical professions. We then use a validated LLM-as-a-judge system to score the 965 resulting images for gender representation. Our results reveal (p < .001 for all): 1) LMMs systematically not only reproduce but actually amplify occupational gender stereotypes relative to real-world labor data, generating men in 93.0% of images for male-stereotyped professions but only 22.5% for female-stereotyped professions; 2) Models exhibit a strong default-male bias, generating men in 68.3% of the time for non-stereotyped professions; and 3) The extent of bias varies dramatically across models, with overall male representation ranging from 46.7% to 73.3%. Notably, the top-performing model de-amplified gender stereotypes and approached gender parity, achieving the highest fairness scores. This variation suggests high bias is not an inevitable outcome but a consequence of design choices. Our work provides the most comprehensive cross-model benchmark of gender bias to date and underscores the necessity of standardized, automated evaluation tools for promoting accountability and fairness in AI development. 

---
# Controllable Singing Voice Synthesis using Phoneme-Level Energy Sequence 

**Authors**: Yerin Ryu, Inseop Shin, Chanwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.07038)  

**Abstract**: Controllable Singing Voice Synthesis (SVS) aims to generate expressive singing voices reflecting user intent. While recent SVS systems achieve high audio quality, most rely on probabilistic modeling, limiting precise control over attributes such as dynamics. We address this by focusing on dynamic control--temporal loudness variation essential for musical expressiveness--and explicitly condition the SVS model on energy sequences extracted from ground-truth spectrograms, reducing annotation costs and improving controllability. We also propose a phoneme-level energy sequence for user-friendly control. To the best of our knowledge, this is the first attempt enabling user-driven dynamics control in SVS. Experiments show our method achieves over 50% reduction in mean absolute error of energy sequences for phoneme-level inputs compared to baseline and energy-predictor models, without compromising synthesis quality. 

---
# Methodological Insights into Structural Causal Modelling and Uncertainty-Aware Forecasting for Economic Indicators 

**Authors**: Federico Cerutti  

**Link**: [PDF](https://arxiv.org/pdf/2509.07036)  

**Abstract**: This paper presents a methodological approach to financial time series analysis by combining causal discovery and uncertainty-aware forecasting. As a case study, we focus on four key U.S. macroeconomic indicators -- GDP, economic growth, inflation, and unemployment -- and we apply the LPCMCI framework with Gaussian Process Distance Correlation (GPDC) to uncover dynamic causal relationships in quarterly data from 1970 to 2021. Our results reveal a robust unidirectional causal link from economic growth to GDP and highlight the limited connectivity of inflation, suggesting the influence of latent factors. Unemployment exhibits strong autoregressive dependence, motivating its use as a case study for probabilistic forecasting. Leveraging the Chronos framework, a large language model trained for time series, we perform zero-shot predictions on unemployment. This approach delivers accurate forecasts one and two quarters ahead, without requiring task-specific training. Crucially, the model's uncertainty-aware predictions yield 90\% confidence intervals, enabling effective anomaly detection through statistically principled deviation analysis. This study demonstrates the value of combining causal structure learning with probabilistic language models to inform economic policy and enhance forecasting robustness. 

---
# A Maslow-Inspired Hierarchy of Engagement with AI Model 

**Authors**: Madara Ogot  

**Link**: [PDF](https://arxiv.org/pdf/2509.07032)  

**Abstract**: The rapid proliferation of artificial intelligence (AI) across industry, government, and education highlights the urgent need for robust frameworks to conceptualise and guide engagement. This paper introduces the Hierarchy of Engagement with AI model, a novel maturity framework inspired by Maslow's hierarchy of needs. The model conceptualises AI adoption as a progression through eight levels, beginning with initial exposure and basic understanding and culminating in ecosystem collaboration and societal impact. Each level integrates technical, organisational, and ethical dimensions, emphasising that AI maturity is not only a matter of infrastructure and capability but also of trust, governance, and responsibility. Initial validation of the model using four diverse case studies (General Motors, the Government of Estonia, the University of Texas System, and the African Union AI Strategy) demonstrate the model's contextual flexibility across various sectors. The model provides scholars with a framework for analysing AI maturity and offers practitioners and policymakers a diagnostic and strategic planning tool to guide responsible and sustainable AI engagement. The proposed model demonstrates that AI maturity progression is multi-dimensional, requiring technological capability, ethical integrity, organisational resilience, and ecosystem collaboration. 

---
# A Minimalist Bayesian Framework for Stochastic Optimization 

**Authors**: Kaizheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.07030)  

**Abstract**: The Bayesian paradigm offers principled tools for sequential decision-making under uncertainty, but its reliance on a probabilistic model for all parameters can hinder the incorporation of complex structural constraints. We introduce a minimalist Bayesian framework that places a prior only on the component of interest, such as the location of the optimum. Nuisance parameters are eliminated via profile likelihood, which naturally handles constraints. As a direct instantiation, we develop a MINimalist Thompson Sampling (MINTS) algorithm. Our framework accommodates structured problems, including continuum-armed Lipschitz bandits and dynamic pricing. It also provides a probabilistic lens on classical convex optimization algorithms such as the center of gravity and ellipsoid methods. We further analyze MINTS for multi-armed bandits and establish near-optimal regret guarantees. 

---
# The Impact of Artificial Intelligence on Traditional Art Forms: A Disruption or Enhancement 

**Authors**: Viswa Chaitanya Marella, Sai Teja Erukude, Suhasnadh Reddy Veluru  

**Link**: [PDF](https://arxiv.org/pdf/2509.07029)  

**Abstract**: The introduction of Artificial Intelligence (AI) into the domains of traditional art (visual arts, performing arts, and crafts) has sparked a complicated discussion about whether this might be an agent of disruption or an enhancement of our traditional art forms. This paper looks at the duality of AI, exploring the ways that recent technologies like Generative Adversarial Networks and Diffusion Models, and text-to-image generators are changing the fields of painting, sculpture, calligraphy, dance, music, and the arts of craft. Using examples and data, we illustrate the ways that AI can democratize creative expression, improve productivity, and preserve cultural heritage, while also examining the negative aspects, including: the threats to authenticity within art, ethical concerns around data, and issues including socio-economic factors such as job losses. While we argue for the context-dependence of the impact of AI (the potential for creative homogenization and the devaluation of human agency in artmaking), we also illustrate the potential for hybrid practices featuring AI in cuisine, etc. We advocate for the development of ethical guidelines, collaborative approaches, and inclusive technology development. In sum, we are articulating a vision of AI in which it amplifies our innate creativity while resisting the displacement of the cultural, nuanced, and emotional aspects of traditional art. The future will be determined by human choices about how to govern AI so that it becomes a mechanism for artistic evolution and not a substitute for the artist's soul. 

---
# Moment- and Power-Spectrum-Based Gaussianity Regularization for Text-to-Image Models 

**Authors**: Jisung Hwang, Jaihoon Kim, Minhyuk Sung  

**Link**: [PDF](https://arxiv.org/pdf/2509.07027)  

**Abstract**: We propose a novel regularization loss that enforces standard Gaussianity, encouraging samples to align with a standard Gaussian distribution. This facilitates a range of downstream tasks involving optimization in the latent space of text-to-image models. We treat elements of a high-dimensional sample as one-dimensional standard Gaussian variables and define a composite loss that combines moment-based regularization in the spatial domain with power spectrum-based regularization in the spectral domain. Since the expected values of moments and power spectrum distributions are analytically known, the loss promotes conformity to these properties. To ensure permutation invariance, the losses are applied to randomly permuted inputs. Notably, existing Gaussianity-based regularizations fall within our unified framework: some correspond to moment losses of specific orders, while the previous covariance-matching loss is equivalent to our spectral loss but incurs higher time complexity due to its spatial-domain computation. We showcase the application of our regularization in generative modeling for test-time reward alignment with a text-to-image model, specifically to enhance aesthetics and text alignment. Our regularization outperforms previous Gaussianity regularization, effectively prevents reward hacking and accelerates convergence. 

---
# Contradictions 

**Authors**: Yang Xu, Shuwei Chen, Xiaomei Zhong, Jun Liu, Xingxing He  

**Link**: [PDF](https://arxiv.org/pdf/2509.07026)  

**Abstract**: Trustworthy AI requires reasoning systems that are not only powerful but also transparent and reliable. Automated Theorem Proving (ATP) is central to formal reasoning, yet classical binary resolution remains limited, as each step involves only two clauses and eliminates at most two literals. To overcome this bottleneck, the concept of standard contradiction and the theory of contradiction-separation-based deduction were introduced in 2018. This paper advances that framework by focusing on the systematic construction of standard contradictions. Specially, this study investigates construction methods for two principal forms of standard contradiction: the maximum triangular standard contradiction and the triangular-type standard contradiction. Building on these structures, we propose a procedure for determining the satisfiability and unsatisfiability of clause sets via maximum standard contradiction. Furthermore, we derive formulas for computing the number of standard sub-contradictions embedded within both the maximum triangular standard contradiction and the triangular-type standard contradiction. The results presented herein furnish the methodological basis for advancing contradiction-separation-based dynamic multi-clause automated deduction, thereby extending the expressive and deductive capabilities of automated reasoning systems beyond the classical binary paradigm. 

---
# 1 bit is all we need: binary normalized neural networks 

**Authors**: Eduardo Lobo Lustoda Cabral, Paulo Pirozelli, Larissa Driemeier  

**Link**: [PDF](https://arxiv.org/pdf/2509.07025)  

**Abstract**: The increasing size of large neural network models, specifically language models and foundational image models, poses deployment challenges, prompting efforts to reduce memory requirements and enhance computational efficiency. These efforts are critical to ensure practical deployment and effective utilization of these models across various applications. In this work, a novel type of neural network layers and models is developed that uses only single-bit parameters. In this novel type of models all parameters of all layers, including kernel weights and biases, only have values equal to zero or one. This novel type of models uses layers named as binary normalized layer. These binary normalized layers can be of any type, such as fully connected, convolutional, attention, etc., and they consist of slight variations of the corresponding conventional layers. To show the effectiveness of the binary normalized layers, two different models are configured to solve a multiclass image classification problem and a language decoder to predict the next token of a sequence. The model to solve the image classification has convolutional and fully connected layers, and the language model is composed of transformer blocks with multi-head attention. The results show that models with binary normalized layers present almost the same results obtained by equivalent models with real 32-bit parameters. The binary normalized layers allow to develop models that use 32 times less memory than current models and have equivalent performance. Besides, the binary normalized layers can be easily implemented on current computers using 1-bit arrays, and do not require the development of dedicated electronic hardware. This novel type of layers opens a new era for large neural network models with reduced memory requirements that can be deployed using simple and cheap hardware, such as mobile devices or only cpus. 

---
# Preventing Another Tessa: Modular Safety Middleware For Health-Adjacent AI Assistants 

**Authors**: Pavan Reddy, Nithin Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2509.07022)  

**Abstract**: In 2023, the National Eating Disorders Association's (NEDA) chatbot Tessa was suspended after providing harmful weight-loss advice to vulnerable users-an avoidable failure that underscores the risks of unsafe AI in healthcare contexts. This paper examines Tessa as a case study in absent safety engineering and demonstrates how a lightweight, modular safeguard could have prevented the incident. We propose a hybrid safety middleware that combines deterministic lexical gates with an in-line large language model (LLM) policy filter, enforcing fail-closed verdicts and escalation pathways within a single model call. Using synthetic evaluations, we show that this design achieves perfect interception of unsafe prompts at baseline cost and latency, outperforming traditional multi-stage pipelines. Beyond technical remedies, we map Tessa's failure patterns to established frameworks (OWASP LLM Top10, NIST SP 800-53), connecting practical safeguards to actionable governance controls. The results highlight that robust, auditable safety in health-adjacent AI does not require heavyweight infrastructure: explicit, testable checks at the last mile are sufficient to prevent "another Tessa", while governance and escalation ensure sustainability in real-world deployment. 

---
# MEGS$^{2}$: Memory-Efficient Gaussian Splatting via Spherical Gaussians and Unified Pruning 

**Authors**: Jiarui Chen, Yikeng Chen, Yingshuang Zou, Ye Huang, Peng Wang, Yuan Liu, Yujing Sun, Wenping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.07021)  

**Abstract**: 3D Gaussian Splatting (3DGS) has emerged as a dominant novel-view synthesis technique, but its high memory consumption severely limits its applicability on edge devices. A growing number of 3DGS compression methods have been proposed to make 3DGS more efficient, yet most only focus on storage compression and fail to address the critical bottleneck of rendering memory. To address this problem, we introduce MEGS$^{2}$, a novel memory-efficient framework that tackles this challenge by jointly optimizing two key factors: the total primitive number and the parameters per primitive, achieving unprecedented memory compression. Specifically, we replace the memory-intensive spherical harmonics with lightweight arbitrarily-oriented spherical Gaussian lobes as our color representations. More importantly, we propose a unified soft pruning framework that models primitive-number and lobe-number pruning as a single constrained optimization problem. Experiments show that MEGS$^{2}$ achieves a 50% static VRAM reduction and a 40% rendering VRAM reduction compared to existing methods, while maintaining comparable rendering quality. 

---
# An efficient deep reinforcement learning environment for flexible job-shop scheduling 

**Authors**: Xinquan Wu, Xuefeng Yan, Mingqiang Wei, Donghai Guan  

**Link**: [PDF](https://arxiv.org/pdf/2509.07019)  

**Abstract**: The Flexible Job-shop Scheduling Problem (FJSP) is a classical combinatorial optimization problem that has a wide-range of applications in the real world. In order to generate fast and accurate scheduling solutions for FJSP, various deep reinforcement learning (DRL) scheduling methods have been developed. However, these methods are mainly focused on the design of DRL scheduling Agent, overlooking the modeling of DRL environment. This paper presents a simple chronological DRL environment for FJSP based on discrete event simulation and an end-to-end DRL scheduling model is proposed based on the proximal policy optimization (PPO). Furthermore, a short novel state representation of FJSP is proposed based on two state variables in the scheduling environment and a novel comprehensible reward function is designed based on the scheduling area of machines. Experimental results on public benchmark instances show that the performance of simple priority dispatching rules (PDR) is improved in our scheduling environment and our DRL scheduling model obtains competing performance compared with OR-Tools, meta-heuristic, DRL and PDR scheduling methods. 

---
# Random Forest Stratified K-Fold Cross Validation on SYN DoS Attack SD-IoV 

**Authors**: Muhammad Arif Hakimi Zamrai, Kamaludin Mohd Yusof  

**Link**: [PDF](https://arxiv.org/pdf/2509.07016)  

**Abstract**: In response to the prevalent concern of TCP SYN flood attacks within the context of Software-Defined Internet of Vehicles (SD-IoV), this study addresses the significant challenge of network security in rapidly evolving vehicular communication systems. This research focuses on optimizing a Random Forest Classifier model to achieve maximum accuracy and minimal detection time, thereby enhancing vehicular network security. The methodology involves preprocessing a dataset containing SYN attack instances, employing feature scaling and label encoding techniques, and applying Stratified K-Fold cross-validation to target key metrics such as accuracy, precision, recall, and F1-score. This research achieved an average value of 0.999998 for all metrics with a SYN DoS attack detection time of 0.24 seconds. Results show that the fine-tuned Random Forest model, configured with 20 estimators and a depth of 10, effectively differentiates between normal and malicious traffic with high accuracy and minimal detection time, which is crucial for SD-IoV networks. This approach marks a significant advancement and introduces a state-of-the-art algorithm in detecting SYN flood attacks, combining high accuracy with minimal detection time. It contributes to vehicular network security by providing a robust solution against TCP SYN flood attacks while maintaining network efficiency and reliability. 

---
# Human-in-the-Loop: Quantitative Evaluation of 3D Models Generation by Large Language Models 

**Authors**: Ahmed R. Sadik, Mariusz Bujny  

**Link**: [PDF](https://arxiv.org/pdf/2509.07010)  

**Abstract**: Large Language Models are increasingly capable of interpreting multimodal inputs to generate complex 3D shapes, yet robust methods to evaluate geometric and structural fidelity remain underdeveloped. This paper introduces a human in the loop framework for the quantitative evaluation of LLM generated 3D models, supporting applications such as democratization of CAD design, reverse engineering of legacy designs, and rapid prototyping. We propose a comprehensive suite of similarity and complexity metrics, including volumetric accuracy, surface alignment, dimensional fidelity, and topological intricacy, to benchmark generated models against ground truth CAD references. Using an L bracket component as a case study, we systematically compare LLM performance across four input modalities: 2D orthographic views, isometric sketches, geometric structure trees, and code based correction prompts. Our findings demonstrate improved generation fidelity with increased semantic richness, with code level prompts achieving perfect reconstruction across all metrics. A key contribution of this work is demonstrating that our proposed quantitative evaluation approach enables significantly faster convergence toward the ground truth, especially compared to traditional qualitative methods based solely on visual inspection and human intuition. This work not only advances the understanding of AI assisted shape synthesis but also provides a scalable methodology to validate and refine generative models for diverse CAD applications. 

---
# Computational Concept of the Psyche 

**Authors**: Anton Kolonin, Vladimir Kryukov  

**Link**: [PDF](https://arxiv.org/pdf/2509.07009)  

**Abstract**: The article provides an overview of approaches to modeling the human psyche in the perspective of building an artificial one. Based on the review, a concept of cognitive architecture is proposed, where the psyche is considered as an operating system of a living or artificial subject, including a space of needs that determines its life meanings in connection with stimuli from the external world, and intelligence as a decision-making system for actions in relation to this world in order to satisfy these needs. Based on the concept, a computational formalization is proposed for creating artificial intelligence systems through learning from experience in the space of a space of needs, taking into account their biological or existential significance for an intelligent agent. Thus, the problem of building general artificial intelligence as a system for making optimal decisions in the space of agent-specific needs under conditions of uncertainty is formalized, with maximization of success in achieving goals, minimization of existential risks and maximization of energy efficiency. A minimal experimental implementation of the model is also provided. 

---
# ArGen: Auto-Regulation of Generative AI via GRPO and Policy-as-Code 

**Authors**: Kapil Madan  

**Link**: [PDF](https://arxiv.org/pdf/2509.07006)  

**Abstract**: This paper introduces ArGen (Auto-Regulation of Generative AI systems), a framework for aligning Large Language Models (LLMs) with complex sets of configurable, machine-readable rules spanning ethical principles, operational safety protocols, and regulatory compliance standards. Moving beyond just preference-based alignment, ArGen is designed to ensure LLMs adhere to these multifaceted policies through a novel synthesis of principle-based automated reward scoring, Group Relative Policy Optimisation (GRPO), and an Open Policy Agent (OPA) inspired governance layer. This approach provides the technical foundation for achieving and demonstrating compliance with diverse and nuanced governance requirements. To showcase the framework's capability to operationalize a deeply nuanced and culturally-specific value system, we present an in-depth case study: the development of a medical AI assistant guided by principles from Dharmic ethics (such as Ahimsa and Dharma), as derived from texts like the Bhagavad Gita. This challenging application demonstrates ArGen's adaptability, achieving a 70.9% improvement in domain-scope adherence over the baseline. Through our open-source repository, we show that ArGen's methodology offers a path to 'Governable Al' systems that are technically proficient, ethically robust, and verifiably compliant for safe deployment in diverse global contexts. 

---
# Not All Splits Are Equal: Rethinking Attribute Generalization Across Unrelated Categories 

**Authors**: Liviu Nicolae Fircă, Antonio Bărbălau, Dan Oneata, Elena Burceanu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06998)  

**Abstract**: Can models generalize attribute knowledge across semantically and perceptually dissimilar categories? While prior work has addressed attribute prediction within narrow taxonomic or visually similar domains, it remains unclear whether current models can abstract attributes and apply them to conceptually distant categories. This work presents the first explicit evaluation for the robustness of the attribute prediction task under such conditions, testing whether models can correctly infer shared attributes between unrelated object types: e.g., identifying that the attribute "has four legs" is common to both "dogs" and "chairs". To enable this evaluation, we introduce train-test split strategies that progressively reduce correlation between training and test sets, based on: LLM-driven semantic grouping, embedding similarity thresholding, embedding-based clustering, and supercategory-based partitioning using ground-truth labels. Results show a sharp drop in performance as the correlation between training and test categories decreases, indicating strong sensitivity to split design. Among the evaluated methods, clustering yields the most effective trade-off, reducing hidden correlations while preserving learnability. These findings offer new insights into the limitations of current representations and inform future benchmark construction for attribute reasoning. 

---
# Visible Yet Unreadable: A Systematic Blind Spot of Vision Language Models Across Writing Systems 

**Authors**: Jie Zhang, Ting Xu, Gelei Deng, Runyi Hu, Han Qiu, Tianwei Zhang, Qing Guo, Ivor Tsang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06996)  

**Abstract**: Writing is a universal cultural technology that reuses vision for symbolic communication. Humans display striking resilience: we readily recognize words even when characters are fragmented, fused, or partially occluded. This paper investigates whether advanced vision language models (VLMs) share this resilience. We construct two psychophysics inspired benchmarks across distinct writing systems, Chinese logographs and English alphabetic words, by splicing, recombining, and overlaying glyphs to yield ''visible but unreadable'' stimuli for models while remaining legible to humans. Despite strong performance on clean text, contemporary VLMs show a severe drop under these perturbations, frequently producing unrelated or incoherent outputs. The pattern suggests a structural limitation: models heavily leverage generic visual invariances but under rely on compositional priors needed for robust literacy. We release stimuli generation code, prompts, and evaluation protocols to facilitate transparent replication and follow up work. Our findings motivate architectures and training strategies that encode symbol segmentation, composition, and binding across scripts, and they delineate concrete challenges for deploying multimodal systems in education, accessibility, cultural heritage, and security. 

---
# The Protocol Genome A Self Supervised Learning Framework from DICOM Headers 

**Authors**: Jimmy Joseph  

**Link**: [PDF](https://arxiv.org/pdf/2509.06995)  

**Abstract**: In this paper, we introduce the Protocol Genome, a self-supervised learning system that learns correlations from DICOM headers and achieves AUROC 0.901 (vs 0.847 baseline) and ECE 0.036 (vs 0.058) on fully held-out external validation. Our method also improves calibration and robustness across modalities (CT, MRI, CXR) and vendors. Clinical imaging is funneled through PACS/DICOM, where procedure choices (scanner make/model, sequence, kernel, kVp, TR/TE, and slice thickness) have consequences for contrast, noise, and artifact. These latent confounders impede the generalization of image-only networks across sites. We consider structured DICOM headers as a label and learn protocol-aware but clinically robust image representations. Protocol Genome obtains tokenized embeddings of de-identified header fields and models them along with image features using: (1) protocol-image contrastive learning, (2) masked protocol prediction, and (3) protocol-protocol translation. With 1.26M studies (7 health systems, 31 scanners, 3 vendors; CT, MR, CR/DR), we experiment on: (A) chest CT triage for PE, (B) brain MRI glioma grading, and (C) chest radiograph cardiomegaly detection. Relative to strong SSL baselines (SimCLR, MAE) as well as ImageNet transfer, Protocol Genome (+0.046: PE, +0.058: glioma, +0.041: cardiomegaly) is associated with higher external AUROC; 25-37% calibration improvements are obtained (p < 0.01, DeLong tests). While the gains may be task-dependent, they are preserved with 10-20% of labeled data. From a clinical point of view, the technique reduces false positives at protocol borders and is applicable in a PACS (DICOM C-FIND/C-MOVE, DICOMweb QIDO/WADO). We publish a model card and deployment guide, complete with both de-identification and bias audits. 

---
# Frustratingly Easy Feature Reconstruction for Out-of-Distribution Detection 

**Authors**: Yingsheng Wang, Shuo Lu, Jian Liang, Aihua Zheng, Ran He  

**Link**: [PDF](https://arxiv.org/pdf/2509.06988)  

**Abstract**: Out-of-distribution (OOD) detection helps models identify data outside the training categories, crucial for security applications. While feature-based post-hoc methods address this by evaluating data differences in the feature space without changing network parameters, they often require access to training data, which may not be suitable for some data privacy scenarios. This may not be suitable in scenarios where data privacy protection is a concern. In this paper, we propose a simple yet effective post-hoc method, termed Classifier-based Feature Reconstruction (ClaFR), from the perspective of subspace projection. It first performs an orthogonal decomposition of the classifier's weights to extract the class-known subspace, then maps the original data features into this subspace to obtain new data representations. Subsequently, the OOD score is determined by calculating the feature reconstruction error of the data within the subspace. Compared to existing OOD detection algorithms, our method does not require access to training data while achieving leading performance on multiple OOD benchmarks. Our code is released at this https URL. 

---
# FusWay: Multimodal hybrid fusion approach. Application to Railway Defect Detection 

**Authors**: Alexey Zhukov, Jenny Benois-Pineau, Amira Youssef, Akka Zemmari, Mohamed Mosbah, Virginie Taillandier  

**Link**: [PDF](https://arxiv.org/pdf/2509.06987)  

**Abstract**: Multimodal fusion is a multimedia technique that has become popular in the wide range of tasks where image information is accompanied by a signal/audio. The latter may not convey highly semantic information, such as speech or music, but some measures such as audio signal recorded by mics in the goal to detect rail structure elements or defects. While classical detection approaches such as You Only Look Once (YOLO) family detectors can be efficiently deployed for defect detection on the image modality, the single modality approaches remain limited. They yield an overdetection in case of the appearance similar to normal structural elements. The paper proposes a new multimodal fusion architecture built on the basis of domain rules with YOLO and Vision transformer backbones. It integrates YOLOv8n for rapid object detection with a Vision Transformer (ViT) to combine feature maps extracted from multiple layers (7, 16, and 19) and synthesised audio representations for two defect classes: rail Rupture and Surface defect. Fusion is performed between audio and image. Experimental evaluation on a real-world railway dataset demonstrates that our multimodal fusion improves precision and overall accuracy by 0.2 points compared to the vision-only approach. Student's unpaired t-test also confirms statistical significance of differences in the mean accuracy. 

---
# CellPainTR: Generalizable Representation Learning for Cross-Dataset Cell Painting Analysis 

**Authors**: Cedric Caruzzo, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2509.06986)  

**Abstract**: Large-scale biological discovery requires integrating massive, heterogeneous datasets like those from the JUMP Cell Painting consortium, but technical batch effects and a lack of generalizable models remain critical roadblocks. To address this, we introduce CellPainTR, a Transformer-based architecture designed to learn foundational representations of cellular morphology that are robust to batch effects. Unlike traditional methods that require retraining on new data, CellPainTR's design, featuring source-specific context tokens, allows for effective out-of-distribution (OOD) generalization to entirely unseen datasets without fine-tuning. We validate CellPainTR on the large-scale JUMP dataset, where it outperforms established methods like ComBat and Harmony in both batch integration and biological signal preservation. Critically, we demonstrate its robustness through a challenging OOD task on the unseen Bray et al. dataset, where it maintains high performance despite significant domain and feature shifts. Our work represents a significant step towards creating truly foundational models for image-based profiling, enabling more reliable and scalable cross-study biological analysis. 

---
# FediLoRA: Heterogeneous LoRA for Federated Multimodal Fine-tuning under Missing Modalities 

**Authors**: Lishan Yang, Nam Kha Nguygen, Po Hu, Wei Emma Zhang, Yanjun Shu, Mong Yuan Sim, Weitong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.06984)  

**Abstract**: Foundation models have demonstrated remarkable performance across a wide range of tasks, yet their large parameter sizes pose challenges for practical deployment, especially in decentralized environments. Parameter-efficient fine-tuning (PEFT), such as Low-Rank Adaptation (LoRA), reduces local computing and memory overhead, making it attractive for federated learning. However, existing federated LoRA methods typically assume uniform rank configurations and unimodal inputs, overlooking two key real-world challenges: (1) heterogeneous client resources have different LoRA ranks, and (2) multimodal data settings with potentially missing modalities. In this work, we propose FediLoRA, a simple yet effective framework for federated multimodal fine-tuning under heterogeneous LoRA ranks and missing modalities. FediLoRA introduces a dimension-wise aggregation strategy that reweights LoRA updates without information dilution during aggregation. It also includes a lightweight layer-wise model editing method that selectively incorporates global parameters to repair local components which improves both client and global model performances. Experimental results on three multimodal benchmark datasets demonstrate that FediLoRA achieves superior performance over competitive baselines in both global and personalized settings, particularly in the presence of modality incompleteness. 

---
# CARE: Decoding Time Safety Alignment via Rollback and Introspection Intervention 

**Authors**: Xiaomeng Hu, Fei Huang, Chenhan Yuan, Junyang Lin, Tsung-Yi Ho  

**Link**: [PDF](https://arxiv.org/pdf/2509.06982)  

**Abstract**: As large language models (LLMs) are increasingly deployed in real-world applications, ensuring the safety of their outputs during decoding has become a critical challenge. However, existing decoding-time interventions, such as Contrastive Decoding, often force a severe trade-off between safety and response quality. In this work, we propose CARE, a novel framework for decoding-time safety alignment that integrates three key components: (1) a guard model for real-time safety monitoring, enabling detection of potentially unsafe content; (2) a rollback mechanism with a token buffer to correct unsafe outputs efficiently at an earlier stage without disrupting the user experience; and (3) a novel introspection-based intervention strategy, where the model generates self-reflective critiques of its previous outputs and incorporates these reflections into the context to guide subsequent decoding steps. The framework achieves a superior safety-quality trade-off by using its guard model for precise interventions, its rollback mechanism for timely corrections, and our novel introspection method for effective self-correction. Experimental results demonstrate that our framework achieves a superior balance of safety, quality, and efficiency, attaining a low harmful response rate and minimal disruption to the user experience while maintaining high response quality. 

---
# RLFactory: A Plug-and-Play Reinforcement Learning Post-Training Framework for LLM Multi-Turn Tool-Use 

**Authors**: Jiajun Chai, Guojun Yin, Zekun Xu, Chuhuai Yue, Yi Jia, Siyu Xia, Xiaohan Wang, Jiwen Jiang, Xiaoguang Li, Chengqi Dong, Hang He, Wei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.06980)  

**Abstract**: Large language models excel at basic reasoning but struggle with tasks that require interaction with external tools. We present RLFactory, a plug-and-play reinforcement learning post-training framework for multi-round tool use. RLFactory tackles (i) tool-call stability and adaptability amid tool heterogeneity and interface issues via an asyncio-based asynchronous caller and a decoupled tool/training architecture, and (ii) diverse evaluation needs via a reward layer supporting rule-based, model-judgment, and tool-verification signals. It reconstructs the MDP by introducing observation markers from tool feedback, closing the loop among model, tools, and environment, and implements a generate-parse-invoke-update workflow for dynamic policy optimization. On Search-R1 with Qwen3-4B, RLFactory achieves a 0.486 test score on the Natural Questions (NQ) dataset, surpassing larger models trained with similar techniques (e.g., Qwen2.5-7B-Instruct-GRPO at 0.473), and increases training throughput by 6.8x. RLFactory provides a low-barrier, highly adaptable framework for strengthening multi-round tool use of LLMs in real-world scenarios. Code: this https URL. 

---
# Exploring Over-stationarization in Deep Learning-based Bus/Tram Arrival Time Prediction: Analysis and Non-stationary Effect Recovery 

**Authors**: Zirui Li, Bin Yang, Meng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06979)  

**Abstract**: Arrival time prediction (ATP) of public transport vehicles is essential in improving passenger experience and supporting traffic management. Deep learning has demonstrated outstanding performance in ATP due to its ability to model non-linear and temporal dynamics. In the multi-step ATP, non-stationary data will degrade the model performance due to the variation in variables' joint distribution along the temporal direction. Previous studies mainly applied normalization to eliminate the non-stationarity in time series, thereby achieving better predictability. However, the normalization may obscure useful characteristics inherent in non-stationarity, which is known as the over-stationarization. In this work, to trade off predictability and non-stationarity, a new approach for multi-step ATP, named non-stationary ATP ( NSATP), is proposed. The method consists of two stages: series stationarization and non-stationarity effect recovery. The first stage aims at improving the predictability. As for the latter, NSATP extends a state-of-the-art method from one-dimensional to two dimensional based models to capture the hidden periodicity in time series and designs a compensation module of over-stationarization by learning scaling and shifting factors from raw data. 125 days' public transport operational data of Dresden is collected for validation. Experimental results show that compared to baseline methods, the proposed NSATP can reduce RMSE, MAE, and MAPE by 2.37%, 1.22%, and 2.26% for trams and by 1.72%, 0.60%, and 1.17% for buses, respectively. 

---
# Toward Reproducible Cross-Backend Compatibility for Deep Learning: A Configuration-First Framework with Three-Tier Verification 

**Authors**: Zehua Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.06977)  

**Abstract**: This paper presents a configuration-first framework for evaluating cross-backend compatibility in deep learning systems deployed on CPU, GPU, and compiled runtimes. The framework decouples experiments from code using YAML, supports both library and repository models, and employs a three-tier verification protocol covering tensor-level closeness, activation alignment, and task-level metrics. Through 672 checks across multiple models and tolerance settings, we observe that 72.0% of runs pass, with most discrepancies occurring under stricter thresholds. Our results show that detection models and compiled backends are particularly prone to drift, often due to nondeterministic post-processing. We further demonstrate that deterministic adapters and selective fallbacks can substantially improve agreement without significant performance loss. To our knowledge, this is the first unified framework that systematically quantifies and mitigates cross-backend drift in deep learning, providing a reproducible methodology for dependable deployment across heterogeneous runtimes. 

---
# A Knowledge-Guided Cross-Modal Feature Fusion Model for Local Traffic Demand Prediction 

**Authors**: Lingyu Zhang, Pengfei Xu, Guobin Wu, Jian Liang, Ruiyang Dong, Yunhai Wang, Xuan Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.06976)  

**Abstract**: Traffic demand prediction plays a critical role in intelligent transportation systems. Existing traffic prediction models primarily rely on temporal traffic data, with limited efforts incorporating human knowledge and experience for urban traffic demand forecasting. However, in real-world scenarios, traffic knowledge and experience derived from human daily life significantly influence precise traffic prediction. Such knowledge and experiences can guide the model in uncovering latent patterns within traffic data, thereby enhancing the accuracy and robustness of predictions. To this end, this paper proposes integrating structured temporal traffic data with textual data representing human knowledge and experience, resulting in a novel knowledge-guided cross-modal feature representation learning (KGCM) model for traffic demand prediction. Based on regional transportation characteristics, we construct a prior knowledge dataset using a large language model combined with manual authoring and revision, covering both regional and global knowledge and experiences. The KGCM model then learns multimodal data features through designed local and global adaptive graph networks, as well as a cross-modal feature fusion mechanism. A proposed reasoning-based dynamic update strategy enables dynamic optimization of the graph model's parameters, achieving optimal performance. Experiments on multiple traffic datasets demonstrate that our model accurately predicts future traffic demand and outperforms existing state-of-the-art (SOTA) models. 

---
# GSTBench: A Benchmark Study on the Transferability of Graph Self-Supervised Learning 

**Authors**: Yu Song, Zhigang Hua, Yan Xie, Jingzhe Liu, Bo Long, Hui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06975)  

**Abstract**: Self-supervised learning (SSL) has shown great promise in graph representation learning. However, most existing graph SSL methods are developed and evaluated under a single-dataset setting, leaving their cross-dataset transferability largely unexplored and limiting their ability to leverage knowledge transfer and large-scale pretraining, factors that are critical for developing generalized intelligence beyond fitting training data. To address this gap and advance foundation model research for graphs, we present GSTBench, the first systematic benchmark for evaluating the transferability of graph SSL methods. We conduct large-scale pretraining on ogbn-papers100M and evaluate five representative SSL methods across a diverse set of target graphs. Our standardized experimental setup decouples confounding factors such as model architecture, dataset characteristics, and adaptation protocols, enabling rigorous comparisons focused solely on pretraining objectives. Surprisingly, we observe that most graph SSL methods struggle to generalize, with some performing worse than random initialization. In contrast, GraphMAE, a masked autoencoder approach, consistently improves transfer performance. We analyze the underlying factors that drive these differences and offer insights to guide future research on transferable graph SSL, laying a solid foundation for the "pretrain-then-transfer" paradigm in graph learning. Our code is available at this https URL. 

---
# Individualized and Interpretable Sleep Forecasting via a Two-Stage Adaptive Spatial-Temporal Model 

**Authors**: Xueyi Wang, Elisabeth Wilhelm  

**Link**: [PDF](https://arxiv.org/pdf/2509.06974)  

**Abstract**: Sleep quality significantly impacts well-being. Therefore, healthcare providers and individuals need accessible and reliable forecasting tools for preventive interventions. This paper introduces an interpretable, individualized two-stage adaptive spatial-temporal model for predicting sleep quality scores. Our proposed framework combines multi-scale convolutional layers to model spatial interactions across multiple input variables, recurrent layers and attention mechanisms to capture long-term temporal dependencies, and a two-stage domain adaptation strategy to enhance generalization. The first adaptation stage is applied during training to mitigate overfitting on the training set. In the second stage, a source-free test-time adaptation mechanism is employed to adapt the model to new users without requiring labels. We conducted various experiments with five input window sizes (3, 5, 7, 9, and 11 days) and five prediction window sizes (1, 3, 5, 7, and 9 days). Our model consistently outperformed time series forecasting baseline approaches, including Long Short-Term Memory (LSTM), Informer, PatchTST, and TimesNet. The best performance was achieved with a three-day input window and a one-day prediction window, yielding a root mean square error (RMSE) of 0.216. Furthermore, the model demonstrated good predictive performance even for longer forecasting horizons (e.g, with a 0.257 RMSE for a three-day prediction window), highlighting its practical utility for real-world applications. We also conducted an explainability analysis to examine how different features influence sleep quality. These findings proved that the proposed framework offers a robust, adaptive, and explainable solution for personalized sleep forecasting using sparse data from commercial wearable devices. 

---
# Impact of Neuron Models on Spiking Neural Networks performance. A Complexity Based Classification Approach 

**Authors**: Zofia Rudnicka, Janusz Szczepanski, Agnieszka Pregowska  

**Link**: [PDF](https://arxiv.org/pdf/2509.06970)  

**Abstract**: This study explores how the selection of neuron models and learning rules impacts the classification performance of Spiking Neural Networks (SNNs), with a focus on applications in bio-signal processing. We compare biologically inspired neuron models, including Leaky Integrate-and-Fire (LIF), metaneurons, and probabilistic Levy-Baxter (LB) neurons, across multiple learning rules, including spike-timing-dependent plasticity (STDP), tempotron, and reward-modulated updates. A novel element of this work is the integration of a complexity-based decision mechanism into the evaluation pipeline. Using Lempel-Ziv Complexity (LZC), a measure related to entropy rate, we quantify the structural regularity of spike trains and assess classification outcomes in a consistent and interpretable manner across different SNN configurations. To investigate neural dynamics and assess algorithm performance, we employed synthetic datasets with varying temporal dependencies and stochasticity levels. These included Markov and Poisson processes, well-established models to simulate neuronal spike trains and capture the stochastic firing behavior of biological this http URL of synthetic Poisson and Markov-modeled data reveals clear performance trends: classification accuracy depends on the interaction between neuron model, network size, and learning rule, with the LZC-based evaluation highlighting configurations that remain robust to weak or noisy signals. This work delivers a systematic analysis of how neuron model selection interacts with network parameters and learning strategies, supported by a novel complexity-based evaluation approach that offers a consistent benchmark for SNN performance. 

---
# Association of Timing and Duration of Moderate-to-Vigorous Physical Activity with Cognitive Function and Brain Aging: A Population-Based Study Using the UK Biobank 

**Authors**: Wasif Khan, Lin Gu, Noah Hammarlund, Lei Xing, Joshua K. Wong, Ruogu Fang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06969)  

**Abstract**: Physical activity is a modifiable lifestyle factor with potential to support cognitive resilience. However, the association of moderate-to-vigorous physical activity (MVPA) intensity, and timing, with cognitive function and region-specific brain structure remain poorly understood. We analyzed data from 45,892 UK Biobank participants aged 60 years and older with valid wrist-worn accelerometer data, cognitive testing, and structural brain MRI. MVPA was measured both continuously (mins per week) and categorically (thresholded using >=150 min/week based on WHO guidelines). Associations with cognitive performance and regional brain volumes were evaluated using multivariable linear models adjusted for demographic, socioeconomic, and health-related covariates. We conducted secondary analyses on MVPA timing and subgroup effects. Higher MVPA was associated with better performance across cognitive domains, including reasoning, memory, executive function, and processing speed. These associations persisted in fully adjusted models and were higher among participants meeting WHO guidelines. Greater MVPA was also associated with subcortical brain regions (caudate, putamen, pallidum, thalamus), as well as regional gray matter volumes involved in emotion, working memory, and perceptual processing. Secondary analyses showed that MVPA at any time of day was associated with cognitive functions and brain volume particularly in the midday-afternoon and evening. Sensitivity analysis shows consistent findings across subgroups, with evidence of dose-response relationships. Higher MVPA is associated with preserved brain structure and enhanced cognitive function in later life. Public health strategies to increase MVPA may support healthy cognitive aging and generate substantial economic benefits, with global gains projected to reach USD 760 billion annually by 2050. 

---
# Deep Learning-based Techniques for Integrated Sensing and Communication Systems: State-of-the-Art, Challenges, and Opportunities 

**Authors**: Murat Temiz, Yongwei Zhang, Yanwei Fu, Chi Zhang, Chenfeng Meng, Orhan Kaplan, Christos Masouros  

**Link**: [PDF](https://arxiv.org/pdf/2509.06968)  

**Abstract**: This article comprehensively reviews recent developments and research on deep learning-based (DL-based) techniques for integrated sensing and communication (ISAC) systems. ISAC, which combines sensing and communication functionalities, is regarded as a key enabler for 6G and beyond networks, as many emerging applications, such as vehicular networks and industrial robotics, necessitate both sensing and communication capabilities for effective operation. A unified platform that provides both functions can reduce hardware complexity, alleviate frequency spectrum congestion, and improve energy efficiency. However, integrating these functionalities on the same hardware requires highly optimized signal processing and system design, introducing significant computational complexity when relying on conventional iterative or optimization-based techniques. As an alternative to conventional techniques, DL-based techniques offer efficient and near-optimal solutions with reduced computational complexity. Hence, such techniques are well-suited for operating under limited computational resources and low latency requirements in real-time systems. DL-based techniques can swiftly and effectively yield near-optimal solutions for a wide range of sophisticated ISAC-related tasks, including waveform design, channel estimation, sensing signal processing, data demodulation, and interference mitigation. Therefore, motivated by these advantages, recent studies have proposed various DL-based approaches for ISAC system design. After briefly introducing DL architectures and ISAC fundamentals, this survey presents a comprehensive and categorized review of state-of-the-art DL-based techniques for ISAC, highlights their key advantages and major challenges, and outlines potential directions for future research. 

---
# Cross-field SNR Analysis and Tensor Channel Estimation for Multi-UAV Near-field Communications 

**Authors**: Tianyu Huo, Jian Xiong, Yiyan Wu, Songjie Yang, Bo Liu, Wenjun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06967)  

**Abstract**: Extremely large antenna array (ELAA) is key to enhancing spectral efficiency in 6G networks. Leveraging the distributed nature of multi-unmanned aerial vehicle (UAV) systems enables the formation of distributed ELAA, which often operate in the near-field region with spatial sparsity, rendering the conventional far-field plane wave assumption invalid. This paper investigates channel estimation for distributed near-field multi-UAV communication systems. We first derive closed-form signal-to-noise ratio (SNR) expressions under the plane wave model (PWM), spherical wave model (SWM), and a hybrid spherical-plane wave model (HSPWM), also referred to as the cross-field model, within a distributed uniform planar array (UPA) scenario. The analysis shows that HSPWM achieves a good balance between modeling accuracy and analytical tractability. Based on this, we propose two channel estimation algorithms: the spherical-domain orthogonal matching pursuit (SD-OMP) and the tensor-OMP. The SD-OMP generalizes the polar domain to jointly consider elevation, azimuth, and range. Under the HSPWM, the channel is naturally formulated as a tensor, enabling the use of tensor-OMP. Simulation results demonstrate that tensor-OMP achieves normalized mean square error (NMSE) performance comparable to SD-OMP, while offering reduced computational complexity and improved scalability. 

---
# Cross-device Zero-shot Label Transfer via Alignment of Time Series Foundation Model Embeddings 

**Authors**: Neal G. Ravindra, Arijit Sehanobish  

**Link**: [PDF](https://arxiv.org/pdf/2509.06966)  

**Abstract**: High-quality, medically validated labels exist for clinical actigraphy data but not for ubiquitous consumer wearables like the Apple Watch. Manually labeling wearables data is expensive and doesn't scale. This paper offers a novel framework that transfers valuable labels from a source domain (e.g., actigraphy) to a target domain (e.g., Apple Watch) without requiring paired data. Instead of working with raw time-series signals, we project both domains into a shared latent embedding space using time-series foundation models (TSFMs) and develop a new framework to align the cross-device representations. Our method, Adversarial Alignment of TSFM Embeddings forces the distributions of source and target embeddings to align within this space, facilitating label transfer across device type. 

---
# VoltanaLLM: Feedback-Driven Frequency Control and State-Space Routing for Energy-Efficient LLM Serving 

**Authors**: Jiahuan Yu, Aryan Taneja, Junfeng Lin, Minjia Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04827)  

**Abstract**: Modern Large Language Model (LLM) serving systems increasingly support interactive applications, like real-time chat assistants, code generation tools, and agentic workflows. However, the soaring energy cost of LLM inference presents a growing challenge for sustainable and cost-effective deployment. This paper introduces VoltanaLLM, a system for SLO-aware, energy-efficient LLM serving, built from a control theory perspective. VoltanaLLM co-designs frequency scaling and request routing in emerging prefill/decode disaggregated architectures, leveraging their decoupled execution to enable fine-grained phase-specific control. It consists of a feedback-driven frequency controller that dynamically adapts GPU frequency for prefill and decode phases, and a state-space router that explores routing decisions across frequency-scaled instances to minimize energy under latency constraints. We implement VoltanaLLM in SGLang and evaluate its performance over multiple state-of-the-art LLMs and real-world datasets. The results demonstrate that VoltanaLLM achieves up to 36.3% energy savings while maintaining near-perfect SLO attainment rate, paving the way for sustainable and intelligent LLM serving. 

---
# Estimating forest carbon stocks from high-resolution remote sensing imagery by reducing domain shift with style transfer 

**Authors**: Zhenyu Yu, Jinnian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.00784)  

**Abstract**: Forests function as crucial carbon reservoirs on land, and their carbon sinks can efficiently reduce atmospheric CO2 concentrations and mitigate climate change. Currently, the overall trend for monitoring and assessing forest carbon stocks is to integrate ground monitoring sample data with satellite remote sensing imagery. This style of analysis facilitates large-scale observation. However, these techniques require improvement in accuracy. We used GF-1 WFV and Landsat TM images to analyze Huize County, Qujing City, Yunnan Province in China. Using the style transfer method, we introduced Swin Transformer to extract global features through attention mechanisms, converting the carbon stock estimation into an image translation. 

---
