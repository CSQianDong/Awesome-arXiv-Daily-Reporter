# Fighting AI with AI: Leveraging Foundation Models for Assuring AI-Enabled Safety-Critical Systems 

**Authors**: Anastasia Mavridou, Divya Gopinath, Corina S. Păsăreanu  

**Link**: [PDF](https://arxiv.org/pdf/2511.20627)  

**Abstract**: The integration of AI components, particularly Deep Neural Networks (DNNs), into safety-critical systems such as aerospace and autonomous vehicles presents fundamental challenges for assurance. The opacity of AI systems, combined with the semantic gap between high-level requirements and low-level network representations, creates barriers to traditional verification approaches. These AI-specific challenges are amplified by longstanding issues in Requirements Engineering, including ambiguity in natural language specifications and scalability bottlenecks in formalization. We propose an approach that leverages AI itself to address these challenges through two complementary components. REACT (Requirements Engineering with AI for Consistency and Testing) employs Large Language Models (LLMs) to bridge the gap between informal natural language requirements and formal specifications, enabling early verification and validation. SemaLens (Semantic Analysis of Visual Perception using large Multi-modal models) utilizes Vision Language Models (VLMs) to reason about, test, and monitor DNN-based perception systems using human-understandable concepts. Together, these components provide a comprehensive pipeline from informal requirements to validated implementations. 

---
# Copyright Detection in Large Language Models: An Ethical Approach to Generative AI Development 

**Authors**: David Szczecina, Senan Gaffori, Edmond Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.20623)  

**Abstract**: The widespread use of Large Language Models (LLMs) raises critical concerns regarding the unauthorized inclusion of copyrighted content in training data. Existing detection frameworks, such as DE-COP, are computationally intensive, and largely inaccessible to independent creators. As legal scrutiny increases, there is a pressing need for a scalable, transparent, and user-friendly solution. This paper introduce an open-source copyright detection platform that enables content creators to verify whether their work was used in LLM training datasets. Our approach enhances existing methodologies by facilitating ease of use, improving similarity detection, optimizing dataset validation, and reducing computational overhead by 10-30% with efficient API calls. With an intuitive user interface and scalable backend, this framework contributes to increasing transparency in AI development and ethical compliance, facilitating the foundation for further research in responsible AI development and copyright enforcement. 

---
# Building a Foundation Model for Trajectory from Scratch 

**Authors**: Gaspard Merten, Mahmoud Sakr, Gilles Dejaegere  

**Link**: [PDF](https://arxiv.org/pdf/2511.20610)  

**Abstract**: Foundation models are transformative in artificial intelligence, but building them from scratch, especially for mobility trajectories, is not yet clear or documented. This tutorial bridges this gap by demonstrating the steps and code of a minimal implementation of a trajectory-focused foundation model starting from GPT-2. Through a concise, step-by-step, code-driven process, we demonstrate adapting GPT-2 for spatiotemporal data. We then review and compare representative trajectory foundation models, such as TrajFM and TrajGPT, highlighting their architectural innovations and differences. Additionally, we introduce complementary techniques from related domains, like TimesFM's patching approach. Targeted at researchers and practitioners, this tutorial aims to explain the concepts and terminology of foundation models, at the implementation level. We find it timely and indispensable to create this educational material in order to support the SIGSPATIAL community in building and evaluating mobility foundation models, enhancing both research clarity and peer-review effectiveness in mobility AI. 

---
# PaTAS: A Parallel System for Trust Propagation in Neural Networks Using Subjective Logic 

**Authors**: Koffi Ismael Ouattara, Ioannis Krontiris, Theo Dimitrakos, Dennis Eisermann, Frank Kargl  

**Link**: [PDF](https://arxiv.org/pdf/2511.20586)  

**Abstract**: Trustworthiness has become a key requirement for the deployment of artificial intelligence systems in safety-critical applications. Conventional evaluation metrics such as accuracy and precision fail to capture uncertainty or the reliability of model predictions, particularly under adversarial or degraded conditions. This paper introduces the \emph{Parallel Trust Assessment System (PaTAS)}, a framework for modeling and propagating trust in neural networks using Subjective Logic (SL). PaTAS operates in parallel with standard neural computation through \emph{Trust Nodes} and \emph{Trust Functions} that propagate input, parameter, and activation trust across the network. The framework defines a \emph{Parameter Trust Update} mechanism to refine parameter reliability during training and an \emph{Inference-Path Trust Assessment (IPTA)} method to compute instance-specific trust at inference. Experiments on real-world and adversarial datasets demonstrate that PaTAS produces interpretable, symmetric, and convergent trust estimates that complement accuracy and expose reliability gaps in poisoned, biased, or uncertain data scenarios. The results show that PaTAS effectively distinguishes between benign and adversarial inputs and identifies cases where model confidence diverges from actual reliability. By enabling transparent and quantifiable trust reasoning within neural architectures, PaTAS provides a principled foundation for evaluating model reliability across the AI lifecycle. 

---
# Beyond Generation: Multi-Hop Reasoning for Factual Accuracy in Vision-Language Models 

**Authors**: Shamima Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2511.20531)  

**Abstract**: Visual Language Models (VLMs) are powerful generative tools but often produce factually in- accurate outputs due to a lack of robust reason- ing capabilities. While extensive research has been conducted on integrating external knowl- edge for reasoning in large language models (LLMs), such efforts remain underexplored in VLMs, where the challenge is compounded by the need to bridge multiple modalities seam- lessly. This work introduces a framework for knowledge-guided reasoning in VLMs, leverag- ing structured knowledge graphs for multi-hop verification using image-captioning task to il- lustrate our framework. Our approach enables systematic reasoning across multiple steps, in- cluding visual entity recognition, knowledge graph traversal, and fact-based caption refine- ment. We evaluate the framework using hi- erarchical, triple-based and bullet-point based knowledge representations, analyzing their ef- fectiveness in factual accuracy and logical infer- ence. Empirical results show that our approach improves factual accuracy by approximately 31% on preliminary experiments on a curated dataset of mixtures from Google Landmarks v2, Conceptual captions and Coco captions re- vealing key insights into reasoning patterns and failure modes. This work demonstrates the po- tential of integrating external knowledge for advancing reasoning in VLMs, paving the way for more reliable and knowledgable multimodal systems. 

---
# Assessing LLMs' Performance: Insights from the Chinese Pharmacist Exam 

**Authors**: Xinran Wang, Boran Zhu, Shujuan Zhou, Ziwen Long, Dehua Zhou, Shu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.20526)  

**Abstract**: Background: As large language models (LLMs) become increasingly integrated into digital health education and assessment workflows, their capabilities in supporting high-stakes, domain-specific certification tasks remain this http URL China, the national pharmacist licensure exam serves as a standardized benchmark for evaluating pharmacists' clinical and theoretical competencies. Objective: This study aimed to compare the performance of two LLMs: ChatGPT-4o and DeepSeek-R1 on real questions from the Chinese Pharmacist Licensing Examination (2017-2021), and to discuss the implications of these performance differences for AI-enabled formative evaluation. Methods: A total of 2,306 multiple-choice (text-only) questions were compiled from official exams, training materials, and public databases. Questions containing tables or images were excluded. Each item was input in its original Chinese format, and model responses were evaluated for exact accuracy. Pearson's Chi-squared test was used to compare overall performance, and Fisher's exact test was applied to year-wise multiple-choice accuracy. Results: DeepSeek-R1 outperformed ChatGPT-4o with a significantly higher overall accuracy (90.0% vs. 76.1%, p < 0.001). Unit-level analyses revealed consistent advantages for DeepSeek-R1, particularly in foundational and clinical synthesis modules. While year-by-year multiple-choice performance also favored DeepSeek-R1, this performance gap did not reach statistical significance in any specific unit-year (all p > 0.05). Conclusion: DeepSeek-R1 demonstrated robust alignment with the structural and semantic demands of the pharmacist licensure exam. These findings suggest that domain-specific models warrant further investigation for this context, while also reinforcing the necessity of human oversight in legally and ethically sensitive contexts. 

---
# FRAGMENTA: End-to-end Fragmentation-based Generative Model with Agentic Tuning for Drug Lead Optimization 

**Authors**: Yuto Suzuki, Paul Awolade, Daniel V. LaBarbera, Farnoush Banaei-Kashani  

**Link**: [PDF](https://arxiv.org/pdf/2511.20510)  

**Abstract**: Molecule generation using generative AI is vital for drug discovery, yet class-specific datasets often contain fewer than 100 training examples. While fragment-based models handle limited data better than atom-based approaches, existing heuristic fragmentation limits diversity and misses key fragments. Additionally, model tuning typically requires slow, indirect collaboration between medicinal chemists and AI engineers. We introduce FRAGMENTA, an end-to-end framework for drug lead optimization comprising: 1) a novel generative model that reframes fragmentation as a "vocabulary selection" problem, using dynamic Q-learning to jointly optimize fragmentation and generation; and 2) an agentic AI system that refines objectives via conversational feedback from domain experts. This system removes the AI engineer from the loop and progressively learns domain knowledge to eventually automate tuning. In real-world cancer drug discovery experiments, FRAGMENTA's Human-Agent configuration identified nearly twice as many high-scoring molecules as baselines. Furthermore, the fully autonomous Agent-Agent system outperformed traditional Human-Human tuning, demonstrating the efficacy of agentic tuning in capturing expert intent. 

---
# Quantifying the Privacy Implications of High-Fidelity Synthetic Network Traffic 

**Authors**: Van Tran, Shinan Liu, Tian Li, Nick Feamster  

**Link**: [PDF](https://arxiv.org/pdf/2511.20497)  

**Abstract**: To address the scarcity and privacy concerns of network traffic data, various generative models have been developed to produce synthetic traffic. However, synthetic traffic is not inherently privacy-preserving, and the extent to which it leaks sensitive information, and how to measure such leakage, remain largely unexplored. This challenge is further compounded by the diversity of model architectures, which shape how traffic is represented and synthesized. We introduce a comprehensive set of privacy metrics for synthetic network traffic, combining standard approaches like membership inference attacks (MIA) and data extraction attacks with network-specific identifiers and attributes. Using these metrics, we systematically evaluate the vulnerability of different representative generative models and examine the factors that influence attack success. Our results reveal substantial variability in privacy risks across models and datasets. MIA success ranges from 0% to 88%, and up to 100% of network identifiers can be recovered from generated traffic, highlighting serious privacy vulnerabilities. We further identify key factors that significantly affect attack outcomes, including training data diversity and how well the generative model fits the training data. These findings provide actionable guidance for designing and deploying generative models that minimize privacy leakage, establishing a foundation for safer synthetic network traffic generation. 

---
# Universe of Thoughts: Enabling Creative Reasoning with Large Language Models 

**Authors**: Yuto Suzuki, Farnoush Banaei-Kashani  

**Link**: [PDF](https://arxiv.org/pdf/2511.20471)  

**Abstract**: Reasoning based on Large Language Models (LLMs) has garnered increasing attention due to outstanding performance of these models in mathematical and complex logical tasks. Beginning with the Chain-of-Thought (CoT) prompting technique, numerous reasoning methods have emerged that decompose problems into smaller, sequential steps (or thoughts). However, existing reasoning models focus on conventional problem-solving and do not necessarily generate creative solutions by ``creative reasoning''. In domains where the solution space is expansive and conventional solutions are suboptimal, such as drug discovery or business strategization, creative reasoning to discover innovative solutions is crucial. To address this gap, first we introduce a computational framework for creative reasoning inspired by established cognitive science principles. With this framework, we propose three core creative reasoning paradigms, namely, \textit{combinational}, \textit{exploratory}, and \textit{transformative} reasoning, where each offers specific directions for systematic exploration of the universe of thoughts to generate creative solutions. Next, to materialize this framework using LLMs, we introduce the \textit{Universe of Thoughts} (or \textit{UoT}, for short), a novel set of methods to implement the aforementioned three creative processes. Finally, we introduce three novel tasks that necessitate creative problem-solving, along with an evaluation benchmark to assess creativity from three orthogonal perspectives: feasibility as constraint, and utility and novelty as metrics. With a comparative analysis against the state-of-the-art (SOTA) reasoning techniques as well as representative commercial models with reasoning capability, we show that UoT demonstrates superior performance in creative reasoning. 

---
# DRAFT-RL: Multi-Agent Chain-of-Draft Reasoning for Reinforcement Learning-Enhanced LLMs 

**Authors**: Yuanhao Li, Mingshan Liu, Hongbo Wang, Yiding Zhang, Yifei Ma, Wei Tan  

**Link**: [PDF](https://arxiv.org/pdf/2511.20468)  

**Abstract**: Large Language Models (LLMs) have shown impressive capabilities in multi-step reasoning and this http URL works introduce multi-agent reflection frameworks where multiple LLM agents critique and refine each other's outputs using reinforcement learning (RL). However, these approaches often rely on single-shot responses and lack structural diversity in reasoning exploration. In this paper, we propose DRAFT-RL, a novel framework that integrates Chain-of-Draft (CoD) reasoning into multi-agent RL training. Instead of generating single responses, each agent produces multiple drafts per query, which are then evaluated by peer agents and a learned reward model to identify the most promising trajectory. These selected drafts are used to refine future reasoning strategies through actor-critic this http URL-RL enables explicit multi-path exploration, peer-guided reflection, and reward-aligned selection, resulting in more robust and interpretable LLM agent behavior. We evaluate our method on complex reasoning tasks including code synthesis, symbolic math, and knowledge-intensive QA,demonstrating that DRAFT-RL outperforms existing reflective and RL-based agents by significant margins in both accuracy and convergence speed 

---
# VibraVerse: A Large-Scale Geometry-Acoustics Alignment Dataset for Physically-Consistent Multimodal Learning 

**Authors**: Bo Pang, Chenxi Xu, Jierui Ren, Guoping Wang, Sheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.20422)  

**Abstract**: Understanding the physical world requires perceptual models grounded in physical laws rather than mere statistical correlations. However, existing multimodal learning frameworks, focused on vision and language, lack physical consistency and overlook the intrinsic causal relationships among an object's geometry, material, vibration modes, and the sounds it produces. We introduce VibraVerse, a large-scale geometry-acoustics alignment dataset that explicitly bridges the causal chain from 3D geometry -> physical attributes -> modal parameters -> acoustic signals. Each 3D model has explicit physical properties (density, Young's modulus, Poisson's ratio) and volumetric geometry, from which modal eigenfrequencies and eigenvectors are computed for impact sound synthesis under controlled excitations. To establish this coherence, we introduce CLASP, a contrastive learning framework for cross-modal alignment that preserves the causal correspondence between an object's physical structure and its acoustic response. This framework enforces physically consistent alignment across modalities, ensuring that every sample is coherent, traceable to the governing equations, and embedded within a unified representation space spanning shape, image, and sound. Built upon VibraVerse, we define a suite of benchmark tasks for geometry-to-sound prediction, sound-guided shape reconstruction, and cross-modal representation learning. Extensive validations on these tasks demonstrate that models trained on VibraVerse exhibit superior accuracy, interpretability, and generalization across modalities. These results establish VibraVerse as a benchmark for physically consistent and causally interpretable multimodal learning, providing a foundation for sound-guided embodied perception and a deeper understanding of the physical world. The dataset will be open-sourced. 

---
# NNGPT: Rethinking AutoML with Large Language Models 

**Authors**: Roman Kochnev, Waleed Khalid, Tolgay Atinc Uzun, Xi Zhang, Yashkumar Sanjaybhai Dhameliya, Furui Qin, Chandini Vysyaraju, Raghuvir Duvvuri, Avi Goyal, Dmitry Ignatov, Radu Timofte  

**Link**: [PDF](https://arxiv.org/pdf/2511.20333)  

**Abstract**: Building self-improving AI systems remains a fundamental challenge in the AI domain. We present NNGPT, an open-source framework that turns a large language model (LLM) into a self-improving AutoML engine for neural network development, primarily for computer vision. Unlike previous frameworks, NNGPT extends the dataset of neural networks by generating new models, enabling continuous fine-tuning of LLMs based on closed-loop system of generation, assessment, and self-improvement. It integrates within one unified workflow five synergistic LLM-based pipelines: zero-shot architecture synthesis, hyperparameter optimization (HPO), code-aware accuracy/early-stop prediction, retrieval-augmented synthesis of scope-closed PyTorch blocks (NN-RAG), and reinforcement learning. Built on the LEMUR dataset as an audited corpus with reproducible metrics, NNGPT emits from a single prompt and validates network architecture, preprocessing code, and hyperparameters, executes them end-to-end, and learns from result. The PyTorch adapter makes NNGPT framework-agnostic, enabling strong performance: NN-RAG achieves 73% executability on 1,289 targets, 3-shot prompting boosts accuracy on common datasets, and hash-based deduplication saves hundreds of runs. One-shot prediction matches search-based AutoML, reducing the need for numerous trials. HPO on LEMUR achieves RMSE 0.60, outperforming Optuna (0.64), while the code-aware predictor reaches RMSE 0.14 with Pearson r=0.78. The system has already generated over 5K validated models, proving NNGPT as an autonomous AutoML engine. Upon acceptance, the code, prompts, and checkpoints will be released for public access to enable reproducibility and facilitate community usage. 

---
# Active Inference in Discrete State Spaces from First Principles 

**Authors**: Patrick Kenny  

**Link**: [PDF](https://arxiv.org/pdf/2511.20321)  

**Abstract**: We seek to clarify the concept of active inference by disentangling it from the Free Energy Principle. We show how the optimizations that need to be carried out in order to implement active inference in discrete state spaces can be formulated as constrained divergence minimization problems which can be solved by standard mean field methods that do not appeal to the idea of expected free energy. When it is used to model perception, the perception/action divergence criterion that we propose coincides with variational free energy. When it is used to model action, it differs from an expected free energy functional by an entropy regularizer. 

---
# Data Augmentation Techniques to Reverse-Engineer Neural Network Weights from Input-Output Queries 

**Authors**: Alexander Beiser, Flavio Martinelli, Wulfram Gerstner, Johanni Brea  

**Link**: [PDF](https://arxiv.org/pdf/2511.20312)  

**Abstract**: Network weights can be reverse-engineered given enough informative samples of a network's input-output function. In a teacher-student setup, this translates into collecting a dataset of the teacher mapping -- querying the teacher -- and fitting a student to imitate such mapping. A sensible choice of queries is the dataset the teacher is trained on. But current methods fail when the teacher parameters are more numerous than the training data, because the student overfits to the queries instead of aligning its parameters to the teacher. In this work, we explore augmentation techniques to best sample the input-output mapping of a teacher network, with the goal of eliciting a rich set of representations from the teacher hidden layers. We discover that standard augmentations such as rotation, flipping, and adding noise, bring little to no improvement to the identification problem. We design new data augmentation techniques tailored to better sample the representational space of the network's hidden layers. With our augmentations we extend the state-of-the-art range of recoverable network sizes. To test their scalability, we show that we can recover networks of up to 100 times more parameters than training data-points. 

---
# Improving Language Agents through BREW 

**Authors**: Shashank Kirtania, Param Biyani, Priyanshu Gupta, Yasharth Bajpai, Roshni Iyer, Sumit Gulwani, Gustavo Soares  

**Link**: [PDF](https://arxiv.org/pdf/2511.20297)  

**Abstract**: Large Language Model (LLM)-based agents are increasingly applied to tasks requiring structured reasoning, tool use, and environmental adaptation, such as data manipulation, multistep planning, and computer-use automation. However, despite their versatility, current training paradigms for model weight optimization methods, like PPO and GRPO, remain relatively impractical with their high computational overhead for rollout convergence. In addition, the resulting agent policies are difficult to interpret, adapt, or incrementally improve. To address this, we investigate creating and refining structured memory of experiential learning of an agent from its environment as an alternative route to agent optimization. We introduce BREW (Bootstrapping expeRientially-learned Environmental knoWledge), a framework for agent optimization for downstream tasks via KB construction and refinement. In our formulation, we introduce an effective method for partitioning agent memory for more efficient retrieval and refinement. BREW uses task graders and behavior rubrics to learn insights while leveraging state-space search for ensuring robustness from the noise and non-specificity in natural language. Empirical results on real world, domain-grounded benchmarks -- OSWorld, $\tau^2$Bench, and SpreadsheetBench -- show BREW achieves $10-20\%$ improvement in task precision, $10-15\%$ reduction in API/tool calls leading to faster execution time, all while maintaining computational efficiency on par with base models. Unlike prior work where memory is treated as static context, we establish the KB as a modular and controllable substrate for agent optimization -- an explicit lever for shaping behavior in a transparent, interpretable, and extensible manner. 

---
# SMoG: Schema Matching on Graph 

**Authors**: Mingyu Jeon, Jaeyoung Suh, Suwan Cho  

**Link**: [PDF](https://arxiv.org/pdf/2511.20285)  

**Abstract**: Schema matching is a critical task in data integration, par- ticularly in the medical domain where disparate Electronic Health Record (EHR) systems must be aligned to standard models like OMOP CDM. While Large Language Models (LLMs) have shown promise in schema matching, they suf- fer from hallucination and lack of up-to-date domain knowl- edge. Knowledge Graphs (KGs) offer a solution by pro- viding structured, verifiable knowledge. However, existing KG-augmented LLM approaches often rely on inefficient complex multi-hop queries or storage-intensive vector-based retrieval methods. This paper introduces SMoG (Schema Matching on Graph), a novel framework that leverages iter- ative execution of simple 1-hop SPARQL queries, inspired by successful strategies in Knowledge Graph Question An- swering (KGQA). SMoG enhances explainability and relia- bility by generating human-verifiable query paths while sig- nificantly reducing storage requirements by directly querying SPARQL endpoints. Experimental results on real-world med- ical datasets demonstrate that SMoG achieves performance comparable to state-of-the-art baselines, validating its effec- tiveness and efficiency in KG-augmented schema matching. 

---
# Actionable and diverse counterfactual explanations incorporating domain knowledge and causal constraints 

**Authors**: Szymon Bobek, Łukasz Bałec, Grzegorz J. Nalepa  

**Link**: [PDF](https://arxiv.org/pdf/2511.20236)  

**Abstract**: Counterfactual explanations enhance the actionable interpretability of machine learning models by identifying the minimal changes required to achieve a desired outcome of the model. However, existing methods often ignore the complex dependencies in real-world datasets, leading to unrealistic or impractical modifications. Motivated by cybersecurity applications in the email marketing domain, we propose a method for generating Diverse, Actionable, and kNowledge-Constrained Explanations (DANCE), which incorporates feature dependencies and causal constraints to ensure plausibility and real-world feasibility of counterfactuals. Our method learns linear and nonlinear constraints from data or integrates expert-provided dependency graphs, ensuring counterfactuals are plausible and actionable. By maintaining consistency with feature relationships, the method produces explanations that align with real-world constraints. Additionally, it balances plausibility, diversity, and sparsity, effectively addressing key limitations in existing algorithms. The work is developed based on a real-life case study with Freshmail, the largest email marketing company in Poland and supported by a joint R&D project Sendguard. Furthermore, we provide an extensive evaluation using 140 public datasets, which highlights its ability to generate meaningful, domain-relevant counterfactuals that outperform other existing approaches based on widely used metrics. The source code for reproduction of the results can be found in a GitHub repository we provide. 

---
# CostNav: A Navigation Benchmark for Cost-Aware Evaluation of Embodied Agents 

**Authors**: Haebin Seong, Sungmin Kim, Minchan Kim, Yongjun Cho, Myunchul Joe, Suhwan Choi, Jaeyoon Jung, Jiyong Youn, Yoonshik Kim, Samwoo Seong, Yubeen Park, Youngjae Yu, Yunsung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.20216)  

**Abstract**: Existing navigation benchmarks focus on task success metrics while overlooking economic viability -- critical for commercial deployment of autonomous delivery robots. We introduce \emph{CostNav}, a \textbf{Micro-Navigation Economic Testbed} that evaluates embodied agents through comprehensive cost-revenue analysis aligned with real-world business operations. CostNav models the complete economic lifecycle including hardware, training, energy, maintenance costs, and delivery revenue with service-level agreements, using industry-derived parameters. \textbf{To our knowledge, CostNav is the first work to quantitatively expose the gap between navigation research metrics and commercial viability}, revealing that optimizing for task success fundamentally differs from optimizing for economic deployment. Our cost model uses parameters derived from industry data sources (energy rates, delivery service pricing), and we project from a reduced-scale simulation to realistic deliveries. Under this projection, the baseline achieves 43.0\% SLA compliance but is \emph{not} commercially viable: yielding a loss of \$30.009 per run with no finite break-even point, because operating costs are dominated by collision-induced maintenance, which accounts for 99.7\% of per-run costs and highlights collision avoidance as a key optimization target. We demonstrate a learning-based on-device navigation baseline and establish a foundation for evaluating rule-based navigation, imitation learning, and cost-aware RL training. CostNav bridges the gap between navigation research and commercial deployment, enabling data-driven decisions about economic trade-offs across navigation paradigms. 

---
# Interactive AI NPCs Powered by LLMs: Technical Report for the CPDC Challenge 2025 

**Authors**: Yitian Huang, Yuxuan Lei, Jianxun Lian, Hao Liao  

**Link**: [PDF](https://arxiv.org/pdf/2511.20200)  

**Abstract**: This report presents the solution and results of our team MSRA\_SC in the Commonsense Persona-Grounded Dialogue Challenge (CPDC 2025). We propose a simple yet effective framework that unifies improvements across both GPU Track and API Track. Our method centers on two key components. First, Context Engineering applies dynamic tool pruning and persona clipping for input compression, combined with post-processing techniques such as parameter normalization and function merging. Together with manually refined prompts, this design improves tool call stability, execution reliability, and role-playing guidance. Second, in the GPU Track, we further adopt GRPO training, replacing supervised fine-tuning with reinforcement learning directly optimized by reward signals. This mitigates small-sample overfitting and significantly enhances task-oriented dialogue performance. In the final evaluation, our team ranks 1st in Task 2 API, 2nd in Task 1 API, and 3rd in both Task 3 API and GPU track, demonstrating the effectiveness of our approach. Our code is publicly available at this https URL 

---
# Towards Benign Memory Forgetting for Selective Multimodal Large Language Model Unlearning 

**Authors**: Zhen Zeng, Leijiang Gu, Zhangling Duan, Feng Li, Zenglin Shi, Cees G. M. Snoek, Meng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.20196)  

**Abstract**: Multimodal Large Language Models (MLLMs) achieve remarkable capabilities but can inadvertently memorize privacy-sensitive information. Although existing unlearning methods can remove such knowledge, they fail to achieve benign forgetting because they often degrade the model's general image understanding performance. To address this, we propose the Sculpted Memory Forgetting Adapter (SMFA), which confines forgetting to targeted memory regions while preserving overall capabilities. SMFA first fine-tunes the model to replace sensitive responses with refusals, yielding a memory forgetting adapter, and then applies a retaining anchor-guided masking mechanism to prevent interference with unrelated knowledge and understanding ability. To systematically evaluate selective MLLM unlearning, we introduce S-MLLMUn Bench, the first benchmark designed to jointly assess the removal of sensitive knowledge and retention of general visual understanding. Extensive experiments show that, unlike prior methods, SMFA achieves precise and controllable unlearning while maintaining the model's foundational image understanding. 

---
# From data to concepts via wiring diagrams 

**Authors**: Jason Lo, Mohammadnima Jafari  

**Link**: [PDF](https://arxiv.org/pdf/2511.20138)  

**Abstract**: A wiring diagram is a labeled directed graph that represents an abstract concept such as a temporal process. In this article, we introduce the notion of a quasi-skeleton wiring diagram graph, and prove that quasi-skeleton wiring diagram graphs correspond to Hasse diagrams. Using this result, we designed algorithms that extract wiring diagrams from sequential data. We used our algorithms in analyzing the behavior of an autonomous agent playing a computer game, and the algorithms correctly identified the winning strategies. We compared the performance of our main algorithm with two other algorithms based on standard clustering techniques (DBSCAN and agglomerative hierarchical), including when some of the data was perturbed. Overall, this article brings together techniques in category theory, graph theory, clustering, reinforcement learning, and data engineering. 

---
# VICoT-Agent: A Vision-Interleaved Chain-of-Thought Framework for Interpretable Multimodal Reasoning and Scalable Remote Sensing Analysis 

**Authors**: Chujie Wang, Zhiyuan Luo, Ruiqi Liu, Can Ran, Shenghua Fan, Xi Chen, Chu He  

**Link**: [PDF](https://arxiv.org/pdf/2511.20085)  

**Abstract**: The current remote sensing image analysis task is increasingly evolving from traditional object recognition to complex intelligence reasoning, which places higher requirements on the model's reasoning ability and the flexibility of tool invocation. To this end, we propose a new multimodal agent framework, Vision-Interleaved Chain-of-Thought Framework (VICoT), which implements explicit multi-round reasoning by dynamically incorporating visual tools into the chain of thought. Through a stack-based reasoning structure and a modular MCP-compatible tool suite, VICoT enables LLMs to efficiently perform multi-round, interleaved vision-language reasoning tasks with strong generalization and this http URL also propose the Reasoning Stack distillation method to migrate complex Agent behaviors to small, lightweight models, which ensures the reasoning capability while significantly reducing complexity. Experiments on multiple remote sensing benchmarks demonstrate that VICoT significantly outperforms existing SOTA frameworks in reasoning transparency, execution efficiency, and generation quality. 

---
# "Are We Done Yet?": A Vision-Based Judge for Autonomous Task Completion of Computer Use Agents 

**Authors**: Marta Sumyk, Oleksandr Kosovan  

**Link**: [PDF](https://arxiv.org/pdf/2511.20067)  

**Abstract**: Computer Use Agents (CUAs) are designed to autonomously operate digital interfaces, yet they often fail to reliably determine whether a given task has been completed. We present an autonomous evaluation and feedback framework that uses vision-language models to assess task completion directly from screenshots and task descriptions. Our dataset covers 42 built-in macOS applications and 1,260 human-labeled tasks across a wide range of scenarios. Our framework achieves up to 73 percent accuracy in task success detection and yields an average relative improvement of 27 percent in overall task success when evaluator feedback is applied. These results show that vision-based evaluation can serve as an effective feedback mechanism that improves the reliability and self-correction of autonomous computer-use agents. 

---
# Reducing Latency of LLM Search Agent via Speculation-based Algorithm-System Co-Design 

**Authors**: Zixiao Huang, Wen Zeng, Tianyu Fu, Tengxuan Liu, Yizhou Sun, Ke Hong, Xinhao Yang, Chengchun Liu, Yan Li, Quanlu Zhang, Guohao Dai, Zhenhua Zhu, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.20048)  

**Abstract**: LLM-based search agents achieve strong performance but suffer from severe latency, as each step requires serialized LLM reasoning followed by action of tool execution. We revisit this bottleneck through the lens of speculation. While traditional predict-verify speculation paradigm can break serial execution, its benefit remains limited, as it retains the full original workload and adds extra inference overhead. We observe that early agent steps often involve simple evidence-gathering, where correct actions can often be predicted without full reasoning. Building on these observations, we present SPAgent, an algorithm-system co-design framework that expands the role of speculation in search agents to reduce latency. Algorithmically, SPAgent introduces a two-phase adaptive speculation mechanism that selectively omits verification when safe. System-wise, a two-level scheduler regulates speculative requests based on engine load to ensure speculation remains beneficial. We implement SPAgent in real-world systems. Across extensive experimental settings, SPAgent achieves up to $1.65\times$ end-to-end speedup while maintaining same or even achieving higher accuracy, enabling practical deployment of multi-step search agents. 

---
# M$^3$Prune: Hierarchical Communication Graph Pruning for Efficient Multi-Modal Multi-Agent Retrieval-Augmented Generation 

**Authors**: Weizi Shao, Taolin Zhang, Zijie Zhou, Chen Chen, Chengyu Wang, Xiaofeng He  

**Link**: [PDF](https://arxiv.org/pdf/2511.19969)  

**Abstract**: Recent advancements in multi-modal retrieval-augmented generation (mRAG), which enhance multi-modal large language models (MLLMs) with external knowledge, have demonstrated that the collective intelligence of multiple agents can significantly outperform a single model through effective communication. Despite impressive performance, existing multi-agent systems inherently incur substantial token overhead and increased computational costs, posing challenges for large-scale deployment. To address these issues, we propose a novel Multi-Modal Multi-agent hierarchical communication graph PRUNING framework, termed M$^3$Prune. Our framework eliminates redundant edges across different modalities, achieving an optimal balance between task performance and token overhead. Specifically, M$^3$Prune first applies intra-modal graph sparsification to textual and visual modalities, identifying the edges most critical for solving the task. Subsequently, we construct a dynamic communication topology using these key edges for inter-modal graph sparsification. Finally, we progressively prune redundant edges to obtain a more efficient and hierarchical topology. Extensive experiments on both general and domain-specific mRAG benchmarks demonstrate that our method consistently outperforms both single-agent and robust multi-agent mRAG systems while significantly reducing token consumption. 

---
# A System-Level Taxonomy of Failure Modes in Large Language Model Applications 

**Authors**: Vaishali Vinay  

**Link**: [PDF](https://arxiv.org/pdf/2511.19933)  

**Abstract**: Large language models (LLMs) are being rapidly integrated into decision-support tools, automation workflows, and AI-enabled software systems. However, their behavior in production environments remains poorly understood, and their failure patterns differ fundamentally from those of traditional machine learning models. This paper presents a system-level taxonomy of fifteen hidden failure modes that arise in real-world LLM applications, including multi-step reasoning drift, latent inconsistency, context-boundary degradation, incorrect tool invocation, version drift, and cost-driven performance collapse. Using this taxonomy, we analyze the growing gap in evaluation and monitoring practices: existing benchmarks measure knowledge or reasoning but provide little insight into stability, reproducibility, drift, or workflow integration. We further examine the production challenges associated with deploying LLMs - including observability limitations, cost constraints, and update-induced regressions - and outline high-level design principles for building reliable, maintainable, and cost-aware LLM systems. Finally, we outline high-level design principles for building reliable, maintainable, and cost-aware LLM-based systems. By framing LLM reliability as a system-engineering problem rather than a purely model-centric one, this work provides an analytical foundation for future research on evaluation methodology, AI system robustness, and dependable LLM deployment. 

---
# Semantic-KG: Using Knowledge Graphs to Construct Benchmarks for Measuring Semantic Similarity 

**Authors**: Qiyao Wei, Edward Morrell, Lea Goetz, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2511.19925)  

**Abstract**: Evaluating the open-form textual responses generated by Large Language Models (LLMs) typically requires measuring the semantic similarity of the response to a (human generated) reference. However, there is evidence that current semantic similarity methods may capture syntactic or lexical forms over semantic content. While benchmarks exist for semantic equivalence, they often suffer from high generation costs due to reliance on subjective human judgment, limited availability for domain-specific applications, and unclear definitions of equivalence. This paper introduces a novel method for generating benchmarks to evaluate semantic similarity methods for LLM outputs, specifically addressing these limitations. Our approach leverages knowledge graphs (KGs) to generate pairs of natural-language statements that are semantically similar or dissimilar, with dissimilar pairs categorized into one of four sub-types. We generate benchmark datasets in four different domains (general knowledge, biomedicine, finance, biology), and conduct a comparative study of semantic similarity methods including traditional natural language processing scores and LLM-as-a-judge predictions. We observe that the sub-type of semantic variation, as well as the domain of the benchmark impact the performance of semantic similarity methods, with no method being consistently superior. Our results present important implications for the use of LLM-as-a-judge in detecting the semantic content of text. Code is available at this https URL and the dataset is available at this https URL. 

---
# RPM-MCTS: Knowledge-Retrieval as Process Reward Model with Monte Carlo Tree Search for Code Generation 

**Authors**: Yuanyuan Lin, Xiangyu Ouyang, Teng Zhang, Kaixin Sui  

**Link**: [PDF](https://arxiv.org/pdf/2511.19895)  

**Abstract**: Tree search-based methods have made significant progress in enhancing the code generation capabilities of large language models. However, due to the difficulty in effectively evaluating intermediate algorithmic steps and the inability to locate and timely correct erroneous steps, these methods often generate incorrect code and incur increased computational costs. To tackle these problems, we propose RPM-MCTS, an effective method that utilizes Knowledge-Retrieval as Process Reward Model based on Monte Carlo Tree Search to evaluate intermediate algorithmic steps. By utilizing knowledge base retrieval, RPM-MCTS avoids the complex training of process reward models. During the expansion phase, similarity filtering is employed to remove redundant nodes, ensuring diversity in reasoning paths. Furthermore, our method utilizes sandbox execution feedback to locate erroneous algorithmic steps during generation, enabling timely and targeted corrections. Extensive experiments on four public code generation benchmarks demonstrate that RPM-MCTS outperforms current state-of-the-art methods while achieving an approximately 15% reduction in token consumption. Furthermore, full fine-tuning of the base model using the data constructed by RPM-MCTS significantly enhances its code capabilities. 

---
# Simulated Self-Assessment in Large Language Models: A Psychometric Approach to AI Self-Efficacy 

**Authors**: Daniel I Jackson, Emma L Jensen, Syed-Amad Hussain, Emre Sezgin  

**Link**: [PDF](https://arxiv.org/pdf/2511.19872)  

**Abstract**: Self-assessment is a key aspect of reliable intelligence, yet evaluations of large language models (LLMs) focus mainly on task accuracy. We adapted the 10-item General Self-Efficacy Scale (GSES) to elicit simulated self-assessments from ten LLMs across four conditions: no task, computational reasoning, social reasoning, and summarization. GSES responses were highly stable across repeated administrations and randomized item orders. However, models showed significantly different self-efficacy levels across conditions, with aggregate scores lower than human norms. All models achieved perfect accuracy on computational and social questions, whereas summarization performance varied widely. Self-assessment did not reliably reflect ability: several low-scoring models performed accurately, while some high-scoring models produced weaker summaries. Follow-up confidence prompts yielded modest, mostly downward revisions, suggesting mild overestimation in first-pass assessments. Qualitative analysis showed that higher self-efficacy corresponded to more assertive, anthropomorphic reasoning styles, whereas lower scores reflected cautious, de-anthropomorphized explanations. Psychometric prompting provides structured insight into LLM communication behavior but not calibrated performance estimates. 

---
# Agentic AI-Empowered Conversational Embodied Intelligence Networks in 6G 

**Authors**: Mingkai Chen, Zijie Feng, Lei Wang, Yaser Khamayseh  

**Link**: [PDF](https://arxiv.org/pdf/2511.19865)  

**Abstract**: In the 6G era, semantic collaboration among multiple embodied intelligent devices (MEIDs) becomes crucial for complex task execution. However, existing systems face challenges in multimodal information fusion, adaptive communication, and decision interpretability. To address these limitations, we propose a collaborative Conversational Embodied Intelligence Network (CC-EIN) integrating multimodal feature fusion, adaptive semantic communication, task coordination, and interpretability. PerceptiNet performs cross-modal fusion of image and radar data to generate unified semantic representations. An adaptive semantic communication strategy dynamically adjusts coding schemes and transmission power according to task urgency and channel quality. A semantic-driven collaboration mechanism further supports task decomposition and conflict-free coordination among heterogeneous devices. Finally, the InDec module enhances decision transparency through Grad-CAM visualization. Simulation results in post-earthquake rescue scenarios demonstrate that CC-EIN achieves 95.4% task completion rate and 95% transmission efficiency while maintaining strong semantic consistency and energy efficiency. 

---
# MicroSims: A Framework for AI-Generated, Scalable Educational Simulations with Universal Embedding and Adaptive Learning Support 

**Authors**: Valerie Lockhart, Dan McCreary, Troy A. Peterson  

**Link**: [PDF](https://arxiv.org/pdf/2511.19864)  

**Abstract**: Educational simulations have long been recognized as powerful tools for enhancing learning outcomes, yet their creation has traditionally required substantial resources and technical expertise. This paper introduces MicroSims a novel framework for creating lightweight, interactive educational simulations that can be rapidly generated using artificial intelligence, universally embedded across digital learning platforms, and easily customized without programming knowledge. MicroSims occupy a unique position at the intersection of three key innovations: (1) standardized design patterns that enable AI-assisted generation, (2) iframe-based architecture that provides universal embedding and sandboxed security, and (3) transparent, modifiable code that supports customization and pedagogical transparency. We present a comprehensive framework encompassing design principles, technical architecture, metadata standards, and development workflows. Drawing on empirical research from physics education studies and meta-analyses across STEM disciplines, we demonstrate that interactive simulations can improve conceptual understanding by up to 30-40\% compared to traditional instruction. MicroSims extend these benefits while addressing persistent barriers of cost, technical complexity, and platform dependence. This work has significant implications for educational equity, and low-cost intelligent interactive textbooks that enabling educators worldwide to create customized, curriculum-aligned simulations on demand. We discuss implementation considerations, present evidence of effectiveness, and outline future directions for AI-powered adaptive learning systems built on the MicroSim foundation. 

---
# Reinforcement Learning with $ω$-Regular Objectives and Constraints 

**Authors**: Dominik Wagner, Leon Witzman, Luke Ong  

**Link**: [PDF](https://arxiv.org/pdf/2511.19849)  

**Abstract**: Reinforcement learning (RL) commonly relies on scalar rewards with limited ability to express temporal, conditional, or safety-critical goals, and can lead to reward hacking. Temporal logic expressible via the more general class of $\omega$-regular objectives addresses this by precisely specifying rich behavioural properties. Even still, measuring performance by a single scalar (be it reward or satisfaction probability) masks safety-performance trade-offs that arise in settings with a tolerable level of risk.
We address both limitations simultaneously by combining $\omega$-regular objectives with explicit constraints, allowing safety requirements and optimisation targets to be treated separately. We develop a model-based RL algorithm based on linear programming, which in the limit produces a policy maximising the probability of satisfying an $\omega$-regular objective while also adhering to $\omega$-regular constraints within specified thresholds. Furthermore, we establish a translation to constrained limit-average problems with optimality-preserving guarantees. 

---
# A Unified Evaluation-Instructed Framework for Query-Dependent Prompt Optimization 

**Authors**: Ke Chen, Yifeng Wang, Hassan Almosapeeh, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.19829)  

**Abstract**: Most prompt-optimization methods refine a single static template, making them ineffective in complex and dynamic user scenarios. Existing query-dependent approaches rely on unstable textual feedback or black-box reward models, providing weak and uninterpretable optimization signals. More fundamentally, prompt quality itself lacks a unified, systematic definition, resulting in fragmented and unreliable evaluation signals. Our approach first establishes a performance-oriented, systematic, and comprehensive prompt evaluation framework. Furthermore, we develop and finetune an execution-free evaluator that predicts multi-dimensional quality scores directly from text. The evaluator then instructs a metric-aware optimizer that diagnoses failure modes and rewrites prompts in an interpretable, query-dependent manner. Our evaluator achieves the strongest accuracy in predicting prompt performance, and the evaluation-instructed optimization consistently surpass both static-template and query-dependent baselines across eight datasets and on three backbone models. Overall, we propose a unified, metric-grounded perspective on prompt quality, and demonstrated that our evaluation-instructed optimization pipeline delivers stable, interpretable, and model-agnostic improvements across diverse tasks. 

---
# KOM: A Multi-Agent Artificial Intelligence System for Precision Management of Knee Osteoarthritis (KOA) 

**Authors**: Weizhi Liu, Xi Chen, Zekun Jiang, Liang Zhao, Kunyuan Jiang, Ruisi Tang, Li Wang, Mingke You, Hanyu Zhou, Hongyu Chen, Qiankun Xiong, Yong Nie, Kang Li, Jian Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.19798)  

**Abstract**: Knee osteoarthritis (KOA) affects more than 600 million individuals globally and is associated with significant pain, functional impairment, and disability. While personalized multidisciplinary interventions have the potential to slow disease progression and enhance quality of life, they typically require substantial medical resources and expertise, making them difficult to implement in resource-limited settings. To address this challenge, we developed KOM, a multi-agent system designed to automate KOA evaluation, risk prediction, and treatment prescription. This system assists clinicians in performing essential tasks across the KOA care pathway and supports the generation of tailored management plans based on individual patient profiles, disease status, risk factors, and contraindications. In benchmark experiments, KOM demonstrated superior performance compared to several general-purpose large language models in imaging analysis and prescription generation. A randomized three-arm simulation study further revealed that collaboration between KOM and clinicians reduced total diagnostic and planning time by 38.5% and resulted in improved treatment quality compared to each approach used independently. These findings indicate that KOM could help facilitate automated KOA management and, when integrated into clinical workflows, has the potential to enhance care efficiency. The modular architecture of KOM may also offer valuable insights for developing AI-assisted management systems for other chronic conditions. 

---
# NOEM$^{3}$A: A Neuro-Symbolic Ontology-Enhanced Method for Multi-Intent Understanding in Mobile Agents 

**Authors**: Ioannis Tzachristas, Aifen Sui  

**Link**: [PDF](https://arxiv.org/pdf/2511.19780)  

**Abstract**: We introduce a neuro-symbolic framework for multi-intent understanding in mobile AI agents by integrating a structured intent ontology with compact language models. Our method leverages retrieval-augmented prompting, logit biasing and optional classification heads to inject symbolic intent structure into both input and output representations. We formalize a new evaluation metric-Semantic Intent Similarity (SIS)-based on hierarchical ontology depth, capturing semantic proximity even when predicted intents differ lexically. Experiments on a subset of ambiguous/demanding dialogues of MultiWOZ 2.3 (with oracle labels from GPT-o3) demonstrate that a 3B Llama model with ontology augmentation approaches GPT-4 accuracy (85% vs 90%) at a tiny fraction of the energy and memory footprint. Qualitative comparisons show that ontology-augmented models produce more grounded, disambiguated multi-intent interpretations. Our results validate symbolic alignment as an effective strategy for enabling accurate and efficient on-device NLU. 

---
# Scaling Agentic Reinforcement Learning for Tool-Integrated Reasoning in VLMs 

**Authors**: Meng Lu, Ran Xu, Yi Fang, Wenxuan Zhang, Yue Yu, Gaurav Srivastava, Yuchen Zhuang, Mohamed Elhoseiny, Charles Fleming, Carl Yang, Zhengzhong Tu, Yang Xie, Guanghua Xiao, Hanrui Wang, Di Jin, Wenqi Shi, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.19773)  

**Abstract**: While recent vision-language models (VLMs) demonstrate strong image understanding, their ability to "think with images", i.e., to reason through multi-step visual interactions, remains limited. We introduce VISTA-Gym, a scalable training environment for incentivizing tool-integrated visual reasoning capabilities in VLMs. VISTA-Gym unifies diverse real-world multimodal reasoning tasks (7 tasks from 13 datasets in total) with a standardized interface for visual tools (e.g., grounding, parsing), executable interaction loops, verifiable feedback signals, and efficient trajectory logging, enabling visual agentic reinforcement learning at scale. While recent VLMs exhibit strong text-only reasoning, both proprietary and open-source models still struggle with tool selection, invocation, and coordination. With VISTA-Gym, we train VISTA-R1 to interleave tool-use with agentic reasoning via multi-turn trajectory sampling and end-to-end reinforcement learning. Extensive experiments across 11 public reasoning-intensive VQA benchmarks show that VISTA-R1-8B outperforms state-of-the-art baselines with similar sizes by 9.51%-18.72%, demonstrating VISTA-Gym as an effective training ground to unlock the tool-integrated reasoning capabilities for VLMs. 

---
# Scaling Item-to-Standard Alignment with Large Language Models: Accuracy, Limits, and Solutions 

**Authors**: Farzan Karimi-Malekabadi, Pooya Razavi, Sonya Powers  

**Link**: [PDF](https://arxiv.org/pdf/2511.19749)  

**Abstract**: As educational systems evolve, ensuring that assessment items remain aligned with content standards is essential for maintaining fairness and instructional relevance. Traditional human alignment reviews are accurate but slow and labor-intensive, especially across large item banks. This study examines whether Large Language Models (LLMs) can accelerate this process without sacrificing accuracy. Using over 12,000 item-skill pairs in grades K-5, we tested three LLMs (GPT-3.5 Turbo, GPT-4o-mini, and GPT-4o) across three tasks that mirror real-world challenges: identifying misaligned items, selecting the correct skill from the full set of standards, and narrowing candidate lists prior to classification. In Study 1, GPT-4o-mini correctly identified alignment status in approximately 83-94% of cases, including subtle misalignments. In Study 2, performance remained strong in mathematics but was lower for reading, where standards are more semantically overlapping. Study 3 demonstrated that pre-filtering candidate skills substantially improved results, with the correct skill appearing among the top five suggestions more than 95% of the time. These findings suggest that LLMs, particularly when paired with candidate filtering strategies, can significantly reduce the manual burden of item review while preserving alignment accuracy. We recommend the development of hybrid pipelines that combine LLM-based screening with human review in ambiguous cases, offering a scalable solution for ongoing item validation and instructional alignment. 

---
# FISCAL: Financial Synthetic Claim-document Augmented Learning for Efficient Fact-Checking 

**Authors**: Rishab Sharma, Iman Saberi, Elham Alipour, Jie JW Wu, Fatemeh Fard  

**Link**: [PDF](https://arxiv.org/pdf/2511.19671)  

**Abstract**: Financial applications of large language models (LLMs) require factual reliability and computational efficiency, yet current systems often hallucinate details and depend on prohibitively large models. We propose FISCAL (Financial Synthetic Claim-Document Augmented Learning), a modular framework for generating synthetic data tailored to financial fact-checking. Using FISCAL, we generate a dataset called FISCAL-data and use it to train MiniCheck-FISCAL, a lightweight verifier for numerical financial claims. MiniCheck-FISCAL outperforms its baseline, surpasses GPT-3.5 Turbo and other open-source peers of similar size, and approaches the accuracy of much larger systems (20x), such as Mixtral-8x22B and Command R+. On external datasets FinDVer and Fin-Fact, it rivals GPT-4o and Claude-3.5 while outperforming Gemini-1.5 Flash. These results show that domain-specific synthetic data, combined with efficient fine-tuning, enables compact models to achieve state-of-the-art accuracy, robustness, and scalability for practical financial AI. The dataset and scripts are available in the project repository (link provided in the paper). 

---
# HeaRT: A Hierarchical Circuit Reasoning Tree-Based Agentic Framework for AMS Design Optimization 

**Authors**: Souradip Poddar, Chia-Tung Ho, Ziming Wei, Weidong Cao, Haoxing Ren, David Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2511.19669)  

**Abstract**: Conventional AI-driven AMS design automation algorithms remain constrained by their reliance on high-quality datasets to capture underlying circuit behavior, coupled with poor transferability across architectures, and a lack of adaptive mechanisms. This work proposes HeaRT, a foundational reasoning engine for automation loops and a first step toward intelligent, adaptive, human-style design optimization. HeaRT consistently demonstrates reasoning accuracy >97% and Pass@1 performance >98% across our 40-circuit benchmark repository, even as circuit complexity increases, while operating at <0.5x real-time token budget of SOTA baselines. Our experiments show that HeaRT yields >3x faster convergence in both sizing and topology design adaptation tasks across diverse optimization approaches, while preserving prior design intent. 

---
# Fara-7B: An Efficient Agentic Model for Computer Use 

**Authors**: Ahmed Awadallah, Yash Lara, Raghav Magazine, Hussein Mozannar, Akshay Nambi, Yash Pandya, Aravind Rajeswaran, Corby Rosset, Alexey Taymanov, Vibhav Vineet, Spencer Whitehead, Andrew Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.19663)  

**Abstract**: Progress in computer use agents (CUAs) has been constrained by the absence of large and high-quality datasets that capture how humans interact with a computer. While LLMs have thrived on abundant textual data, no comparable corpus exists for CUA trajectories. To address these gaps, we introduce FaraGen, a novel synthetic data generation system for multi-step web tasks. FaraGen can propose diverse tasks from frequently used websites, generate multiple solution attempts, and filter successful trajectories using multiple verifiers. It achieves high throughput, yield, and diversity for multi-step web tasks, producing verified trajectories at approximately $1 each. We use this data to train Fara-7B, a native CUA model that perceives the computer using only screenshots, executes actions via predicted coordinates, and is small enough to run on-device. We find that Fara-7B outperforms other CUA models of comparable size on benchmarks like WebVoyager, Online-Mind2Web, and WebTailBench -- our novel benchmark that better captures under-represented web tasks in pre-existing benchmarks. Furthermore, Fara-7B is competitive with much larger frontier models, illustrating key benefits of scalable data generation systems in advancing small efficient agentic models. We are making Fara-7B open-weight on Microsoft Foundry and HuggingFace, and we are releasing WebTailBench. 

---
# Using Wearable Devices to Improve Chronic PainTreatment among Patients with Opioid Use Disorder 

**Authors**: Abhay Goyal, Navin Kumar, Kimberly DiMeola, Rafael Trujillo, Soorya Ram Shimgekar, Christian Poellabauer, Pi Zonooz, Ermonda Gjoni-Markaj, Declan Barry, Lynn Madden  

**Link**: [PDF](https://arxiv.org/pdf/2511.19577)  

**Abstract**: Chronic pain (CP) and opioid use disorder (OUD) are common and interrelated chronic medical conditions. Currently, there is a paucity of evidence-based integrated treatments for CP and OUD among individuals receiving medication for opioid use disorder (MOUD). Wearable devices have the potential to monitor complex patient information and inform treatment development for persons with OUD and CP, including pain variability (e.g., exacerbations of pain or pain spikes) and clinical correlates (e.g., perceived stress). However, the application of large language models (LLMs) with wearable data for understanding pain spikes, remains unexplored. Consequently, the aim of this pilot study was to examine the clinical correlates of pain spikes using a range of AI approaches. We found that machine learning models achieved relatively high accuracy (>0.7) in predicting pain spikes, while LLMs were limited in providing insights on pain spikes. Real-time monitoring through wearable devices, combined with advanced AI models, could facilitate early detection of pain spikes and support personalized interventions that may help mitigate the risk of opioid relapse, improve adherence to MOUD, and enhance the integration of CP and OUD care. Given overall limited LLM performance, these findings highlight the need to develop LLMs which can provide actionable insights in the OUD/CP context. 

---
# MedROV: Towards Real-Time Open-Vocabulary Detection Across Diverse Medical Imaging Modalities 

**Authors**: Tooba Tehreem Sheikh, Jean Lahoud, Rao Muhammad Anwer, Fahad Shahbaz Khan, Salman Khan, Hisham Cholakkal  

**Link**: [PDF](https://arxiv.org/pdf/2511.20650)  

**Abstract**: Traditional object detection models in medical imaging operate within a closed-set paradigm, limiting their ability to detect objects of novel labels. Open-vocabulary object detection (OVOD) addresses this limitation but remains underexplored in medical imaging due to dataset scarcity and weak text-image alignment. To bridge this gap, we introduce MedROV, the first Real-time Open Vocabulary detection model for medical imaging. To enable open-vocabulary learning, we curate a large-scale dataset, Omnis, with 600K detection samples across nine imaging modalities and introduce a pseudo-labeling strategy to handle missing annotations from multi-source datasets. Additionally, we enhance generalization by incorporating knowledge from a large pre-trained foundation model. By leveraging contrastive learning and cross-modal representations, MedROV effectively detects both known and novel structures. Experimental results demonstrate that MedROV outperforms the previous state-of-the-art foundation model for medical image detection with an average absolute improvement of 40 mAP50, and surpasses closed-set detectors by more than 3 mAP50, while running at 70 FPS, setting a new benchmark in medical detection. Our source code, dataset, and trained model are available at this https URL. 

---
# MotionV2V: Editing Motion in a Video 

**Authors**: Ryan Burgert, Charles Herrmann, Forrester Cole, Michael S Ryoo, Neal Wadhwa, Andrey Voynov, Nataniel Ruiz  

**Link**: [PDF](https://arxiv.org/pdf/2511.20640)  

**Abstract**: While generative video models have achieved remarkable fidelity and consistency, applying these capabilities to video editing remains a complex challenge. Recent research has explored motion controllability as a means to enhance text-to-video generation or image animation; however, we identify precise motion control as a promising yet under-explored paradigm for editing existing videos. In this work, we propose modifying video motion by directly editing sparse trajectories extracted from the input. We term the deviation between input and output trajectories a "motion edit" and demonstrate that this representation, when coupled with a generative backbone, enables powerful video editing capabilities. To achieve this, we introduce a pipeline for generating "motion counterfactuals", video pairs that share identical content but distinct motion, and we fine-tune a motion-conditioned video diffusion architecture on this dataset. Our approach allows for edits that start at any timestamp and propagate naturally. In a four-way head-to-head user study, our model achieves over 65 percent preference against prior work. Please see our project page: this https URL 

---
# Latent Collaboration in Multi-Agent Systems 

**Authors**: Jiaru Zou, Xiyuan Yang, Ruizhong Qiu, Gaotang Li, Katherine Tieu, Pan Lu, Ke Shen, Hanghang Tong, Yejin Choi, Jingrui He, James Zou, Mengdi Wang, Ling Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.20639)  

**Abstract**: Multi-agent systems (MAS) extend large language models (LLMs) from independent single-model reasoning to coordinative system-level intelligence. While existing LLM agents depend on text-based mediation for reasoning and communication, we take a step forward by enabling models to collaborate directly within the continuous latent space. We introduce LatentMAS, an end-to-end training-free framework that enables pure latent collaboration among LLM agents. In LatentMAS, each agent first performs auto-regressive latent thoughts generation through last-layer hidden embeddings. A shared latent working memory then preserves and transfers each agent's internal representations, ensuring lossless information exchange. We provide theoretical analyses establishing that LatentMAS attains higher expressiveness and lossless information preservation with substantially lower complexity than vanilla text-based MAS. In addition, empirical evaluations across 9 comprehensive benchmarks spanning math and science reasoning, commonsense understanding, and code generation show that LatentMAS consistently outperforms strong single-model and text-based MAS baselines, achieving up to 14.6% higher accuracy, reducing output token usage by 70.8%-83.7%, and providing 4x-4.3x faster end-to-end inference. These results demonstrate that our new latent collaboration framework enhances system-level reasoning quality while offering substantial efficiency gains without any additional training. Code and data are fully open-sourced at this https URL. 

---
# MapReduce LoRA: Advancing the Pareto Front in Multi-Preference Optimization for Generative Models 

**Authors**: Chieh-Yun Chen, Zhonghao Wang, Qi Chen, Zhifan Ye, Min Shi, Yue Zhao, Yinan Zhao, Hui Qu, Wei-An Lin, Yiru Shen, Ajinkya Kale, Irfan Essa, Humphrey Shi  

**Link**: [PDF](https://arxiv.org/pdf/2511.20629)  

**Abstract**: Reinforcement learning from human feedback (RLHF) with reward models has advanced alignment of generative models to human aesthetic and perceptual preferences. However, jointly optimizing multiple rewards often incurs an alignment tax, improving one dimension while degrading others. To address this, we introduce two complementary methods: MapReduce LoRA and Reward-aware Token Embedding (RaTE). MapReduce LoRA trains preference-specific LoRA experts in parallel and iteratively merges them to refine a shared base model; RaTE learns reward-specific token embeddings that compose at inference for flexible preference control. Experiments on Text-to-Image generation (Stable Diffusion 3.5 Medium and FLUX.1-dev) show improvements of 36.1%, 4.6%, and 55.7%, and 32.7%, 4.3%, and 67.1% on GenEval, PickScore, and OCR, respectively. On Text-to-Video generation (HunyuanVideo), visual and motion quality improve by 48.1% and 90.0%, respectively. On the language task, Helpful Assistant, with Llama-2 7B, helpful and harmless improve by 43.4% and 136.7%, respectively. Our framework sets a new state-of-the-art multi-preference alignment recipe across modalities. 

---
# ROOT: Robust Orthogonalized Optimizer for Neural Network Training 

**Authors**: Wei He, Kai Han, Hang Zhou, Hanting Chen, Zhicheng Liu, Xinghao Chen, Yunhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.20626)  

**Abstract**: The optimization of large language models (LLMs) remains a critical challenge, particularly as model scaling exacerbates sensitivity to algorithmic imprecision and training instability. Recent advances in optimizers have improved convergence efficiency through momentum orthogonalization, but suffer from two key robustness limitations: dimensional fragility in orthogonalization precision and vulnerability to outlier-induced noise. To address these robustness challenges, we introduce ROOT, a Robust Orthogonalized Optimizer that enhances training stability through dual robustness mechanisms. First, we develop a dimension-robust orthogonalization scheme using adaptive Newton iterations with fine-grained coefficients tailored to specific matrix sizes, ensuring consistent precision across diverse architectural configurations. Second, we introduce an optimization-robust framework via proximal optimization that suppresses outlier noise while preserving meaningful gradient directions. Extensive experiments demonstrate that ROOT achieves significantly improved robustness, with faster convergence and superior final performance compared to both Muon and Adam-based optimizers, particularly in noisy and non-convex scenarios. Our work establishes a new paradigm for developing robust and precise optimizers capable of handling the complexities of modern large-scale model training. The code will be available at this https URL. 

---
# DiFR: Inference Verification Despite Nondeterminism 

**Authors**: Adam Karvonen, Daniel Reuter, Roy Rinberg, Luke Marks, Adrià Garriga-Alonso, Keri Warr  

**Link**: [PDF](https://arxiv.org/pdf/2511.20621)  

**Abstract**: As demand for LLM inference grows, it is becoming increasingly important that providers and their customers can verify that inference processes are performed correctly, without errors or tampering. However, re-running the same inference process twice often leads to different results due to benign numerical noise, making it difficult to distinguish legitimate variation from actual problems. To address this problem, we introduce Token-DiFR (Token-Divergence-From-Reference), a method for verifying inference outputs by comparing generated tokens against predictions made by a trusted reference implementation conditioned on the same random seed. Sampling seed synchronization tightly constrains valid outputs, leaving providers minimal room to deviate from correct inference, which allows output tokens themselves to serve as auditable evidence of correctness at zero additional cost to the provider. Token-DiFR reliably identifies sampling errors, simulated bugs, and model quantization, detecting 4-bit quantization with AUC $>$ 0.999 within 300 output tokens. For applications requiring sample-efficient forward-pass verification, we additionally introduce Activation-DiFR, a scheme that uses random orthogonal projections to compress activations into compact fingerprints for subsequent verification. Activation-DiFR detects 4-bit quantization with AUC $>$ 0.999 using just 2 output tokens, while reducing communication overhead by 25-75% relative to existing methods. We release an open-source integration with vLLM to accelerate practical deployment of verifiable inference. 

---
# Evaluating the Performance of Deep Learning Models in Whole-body Dynamic 3D Posture Prediction During Load-reaching Activities 

**Authors**: Seyede Niloofar Hosseini, Ali Mojibi, Mahdi Mohseni, Navid Arjmand, Alireza Taheri  

**Link**: [PDF](https://arxiv.org/pdf/2511.20615)  

**Abstract**: This study aimed to explore the application of deep neural networks for whole-body human posture prediction during dynamic load-reaching activities. Two time-series models were trained using bidirectional long short-term memory (BLSTM) and transformer architectures. The dataset consisted of 3D full-body plug-in gait dynamic coordinates from 20 normal-weight healthy male individuals each performing 204 load-reaching tasks from different load positions while adapting various lifting and handling techniques. The model inputs consisted of the 3D position of the hand-load position, lifting (stoop, full-squat and semi-squat) and handling (one- and two-handed) techniques, body weight and height, and the 3D coordinate data of the body posture from the first 25% of the task duration. These inputs were used by the models to predict body coordinates during the remaining 75% of the task period. Moreover, a novel method was proposed to improve the accuracy of the previous and present posture prediction networks by enforcing constant body segment lengths through the optimization of a new cost function. The results indicated that the new cost function decreased the prediction error of the models by approximately 8% and 21% for the arm and leg models, respectively. We indicated that utilizing the transformer architecture, with a root-mean-square-error of 47.0 mm, exhibited ~58% more accurate long-term performance than the BLSTM-based model. This study merits the use of neural networks that capture time series dependencies in 3D motion frames, providing a unique approach for understanding and predict motion dynamics during manual material handling activities. 

---
# Can Vibe Coding Beat Graduate CS Students? An LLM vs. Human Coding Tournament on Market-driven Strategic Planning 

**Authors**: Panayiotis Danassis, Naman Goel  

**Link**: [PDF](https://arxiv.org/pdf/2511.20613)  

**Abstract**: The rapid proliferation of Large Language Models (LLMs) has revolutionized AI-assisted code generation. This rapid development of LLMs has outpaced our ability to properly benchmark them. Prevailing benchmarks emphasize unit-test pass rates and syntactic correctness. Such metrics understate the difficulty of many real-world problems that require planning, optimization, and strategic interaction. We introduce a multi-agent reasoning-driven benchmark based on a real-world logistics optimization problem (Auction, Pickup, and Delivery Problem) that couples competitive auctions with capacity-constrained routing. The benchmark requires building agents that can (i) bid strategically under uncertainty and (ii) optimize planners that deliver tasks while maximizing profit. We evaluate 40 LLM-coded agents (by a wide range of state-of-the-art LLMs under multiple prompting methodologies, including vibe coding) against 17 human-coded agents developed before the advent of LLMs. Our results over 12 double all-play-all tournaments and $\sim 40$k matches demonstrate (i) a clear superiority of human(graduate students)-coded agents: the top 5 spots are consistently won by human-coded agents, (ii) the majority of LLM-coded agents (33 out of 40) are beaten by very simple baselines, and (iii) given the best human solution as an input and prompted to improve upon, the best performing LLM makes the solution significantly worse instead of improving it. Our results highlight a gap in LLMs' ability to produce code that works competitively in the real-world, and motivate new evaluations that emphasize reasoning-driven code synthesis in real-world scenarios. 

---
# On Evaluating LLM Alignment by Evaluating LLMs as Judges 

**Authors**: Yixin Liu, Pengfei Liu, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2511.20604)  

**Abstract**: Alignment with human preferences is an important evaluation aspect of LLMs, requiring them to be helpful, honest, safe, and to precisely follow human instructions. Evaluating large language models' (LLMs) alignment typically involves directly assessing their open-ended responses, requiring human annotators or strong LLM judges. Conversely, LLMs themselves have also been extensively evaluated as judges for assessing alignment. In this work, we examine the relationship between LLMs' generation and evaluation capabilities in aligning with human preferences. To this end, we first conduct a comprehensive analysis of the generation-evaluation consistency (GE-consistency) among various LLMs, revealing a strong correlation between their generation and evaluation capabilities when evaluated by a strong LLM preference oracle. Utilizing this finding, we propose a benchmarking paradigm that measures LLM alignment with human preferences without directly evaluating their generated outputs, instead assessing LLMs in their role as evaluators. Our evaluation shows that our proposed benchmark, AlignEval, matches or surpasses widely used automatic LLM evaluation benchmarks, such as AlpacaEval and Arena-Hard, in capturing human preferences when ranking LLMs. Our study offers valuable insights into the connection between LLMs' generation and evaluation capabilities, and introduces a benchmark that assesses alignment without directly evaluating model outputs. 

---
# The Driver-Blindness Phenomenon: Why Deep Sequence Models Default to Autocorrelation in Blood Glucose Forecasting 

**Authors**: Heman Shakeri  

**Link**: [PDF](https://arxiv.org/pdf/2511.20601)  

**Abstract**: Deep sequence models for blood glucose forecasting consistently fail to leverage clinically informative drivers--insulin, meals, and activity--despite well-understood physiological mechanisms. We term this Driver-Blindness and formalize it via $\Delta_{\text{drivers}}$, the performance gain of multivariate models over matched univariate baselines. Across the literature, $\Delta_{\text{drivers}}$ is typically near zero. We attribute this to three interacting factors: architectural biases favoring autocorrelation (C1), data fidelity gaps that render drivers noisy and confounded (C2), and physiological heterogeneity that undermines population-level models (C3). We synthesize strategies that partially mitigate Driver-Blindness--including physiological feature encoders, causal regularization, and personalization--and recommend that future work routinely report $\Delta_{\text{drivers}}$ to prevent driver-blind models from being considered state-of-the-art. 

---
# BrowseSafe: Understanding and Preventing Prompt Injection Within AI Browser Agents 

**Authors**: Kaiyuan Zhang, Mark Tenenholtz, Kyle Polley, Jerry Ma, Denis Yarats, Ninghui Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.20597)  

**Abstract**: The integration of artificial intelligence (AI) agents into web browsers introduces security challenges that go beyond traditional web application threat models. Prior work has identified prompt injection as a new attack vector for web agents, yet the resulting impact within real-world environments remains insufficiently understood.
In this work, we examine the landscape of prompt injection attacks and synthesize a benchmark of attacks embedded in realistic HTML payloads. Our benchmark goes beyond prior work by emphasizing injections that can influence real-world actions rather than mere text outputs, and by presenting attack payloads with complexity and distractor frequency similar to what real-world agents encounter. We leverage this benchmark to conduct a comprehensive empirical evaluation of existing defenses, assessing their effectiveness across a suite of frontier AI models. We propose a multi-layered defense strategy comprising both architectural and model-based defenses to protect against evolving prompt injection attacks. Our work offers a blueprint for designing practical, secure web agents through a defense-in-depth approach. 

---
# EnergyTwin: A Multi-Agent System for Simulating and Coordinating Energy Microgrids 

**Authors**: Jakub Muszyński, Ignacy Walużenicz, Patryk Zan, Zofia Wrona, Maria Ganzha, Marcin Paprzycki, Costin Bădică  

**Link**: [PDF](https://arxiv.org/pdf/2511.20590)  

**Abstract**: Microgrids are deployed to reduce purchased grid energy, limit exposure to volatile tariffs, and ensure service continuity during disturbances. This requires coordinating heterogeneous distributed energy resources across multiple time scales and under variable conditions. Among existing tools, typically, power-system simulators capture physical behaviour but assume centralized control, while multi-agent frameworks model decentralized decision-making but represent energy with no physical grounding. In this context, the EnergyTwin is introduced, an agent-based microgrid simulation environment that couples physically grounded models with forecast-informed, rolling-horizon planning, and negotiations. Each asset is modeled as an agent, interacting with a central agent that obtains forecasts, formulates predictions, and allocates energy through contract-based interactions. EnergyTwin targets tertiary-layer decision making and is extensible for digital-twin use. Its feasibility was evaluated in a university campus microgrid scenario where multiple planning strategies were compared. Achieved results show that forecast-driven rolling-horizon planning increases local energy self-sufficiency, maintains higher battery reserves, and reduces exposure to low-resilience operating states. They demonstrate also potential of EnergyTwin as platform supporting research on resilient, negotiation-driven microgrids. 

---
# Gated Uncertainty-Aware Runtime Dual Invariants for Neural Signal-Controlled Robotics 

**Authors**: Tasha Kim, Oiwi Parker Jones  

**Link**: [PDF](https://arxiv.org/pdf/2511.20570)  

**Abstract**: Safety-critical assistive systems that directly decode user intent from neural signals require rigorous guarantees of reliability and trust. We present GUARDIAN (Gated Uncertainty-Aware Runtime Dual Invariants), a framework for real-time neuro-symbolic verification for neural signal-controlled robotics. GUARDIAN enforces both logical safety and physiological trust by coupling confidence-calibrated brain signal decoding with symbolic goal grounding and dual-layer runtime monitoring. On the BNCI2014 motor imagery electroencephalogram (EEG) dataset with 9 subjects and 5,184 trials, the system performs at a high safety rate of 94-97% even with lightweight decoder architectures with low test accuracies (27-46%) and high ECE confidence miscalibration (0.22-0.41). We demonstrate 1.7x correct interventions in simulated noise testing versus at baseline. The monitor operates at 100Hz and sub-millisecond decision latency, making it practically viable for closed-loop neural signal-based systems. Across 21 ablation results, GUARDIAN exhibits a graduated response to signal degradation, and produces auditable traces from intent, plan to action, helping to link neural evidence to verifiable robot action. 

---
# Time-Domain Linear Model-based Framework for Passive Acoustic Mapping of Cavitation Activity 

**Authors**: Tatiana Gelvez-Barrera, Barbara Nicolas, Denis Kouamé, Bruno Gilles, Adrian Basarab  

**Link**: [PDF](https://arxiv.org/pdf/2511.20551)  

**Abstract**: Passive acoustic mapping enables the spatial mapping and temporal monitoring of cavitation activity, playing a crucial role in therapeutic ultrasound applications. Most conventional beamforming methods, whether implemented in the time or frequency domains, suffer from limited axial resolution due to the absence of a reference emission onset time. While frequency-domain methods, the most efficient of which are based on the cross-spectral matrix, require long signals for accurate estimation, time-domain methods typically achieve lower spatial resolution. To address these limitations, we propose a linear model-based beamforming framework fully formulated in the time domain. The linear forward model relates a discretized spatiotemporal distribution of cavitation activity to the temporal signals recorded by a probe, explicitly accounting for time-of-flight delays dictated by the acquisition geometry. This model is then inverted using regularization techniques that exploit prior knowledge of cavitation activity in both spatial and temporal domains. Experimental results show that the proposed framework achieves enhanced or competitive cavitation map quality while using only 20\% of the data typically required by frequency-domain methods. This highlights the substantial gain in data efficiency and the flexibility of our spatiotemporal regularization to adapt to diverse passive cavitation scenarios, outperforming state-of-the-art techniques. 

---
# Flash-DMD: Towards High-Fidelity Few-Step Image Generation with Efficient Distillation and Joint Reinforcement Learning 

**Authors**: Guanjie Chen, Shirui Huang, Kai Liu, Jianchen Zhu, Xiaoye Qu, Peng Chen, Yu Cheng, Yifu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.20549)  

**Abstract**: Diffusion Models have emerged as a leading class of generative models, yet their iterative sampling process remains computationally expensive. Timestep distillation is a promising technique to accelerate generation, but it often requires extensive training and leads to image quality degradation. Furthermore, fine-tuning these distilled models for specific objectives, such as aesthetic appeal or user preference, using Reinforcement Learning (RL) is notoriously unstable and easily falls into reward hacking. In this work, we introduce Flash-DMD, a novel framework that enables fast convergence with distillation and joint RL-based refinement. Specifically, we first propose an efficient timestep-aware distillation strategy that significantly reduces training cost with enhanced realism, outperforming DMD2 with only $2.1\%$ its training cost. Second, we introduce a joint training scheme where the model is fine-tuned with an RL objective while the timestep distillation training continues simultaneously. We demonstrate that the stable, well-defined loss from the ongoing distillation acts as a powerful regularizer, effectively stabilizing the RL training process and preventing policy collapse. Extensive experiments on score-based and flow matching models show that our proposed Flash-DMD not only converges significantly faster but also achieves state-of-the-art generation quality in the few-step sampling regime, outperforming existing methods in visual quality, human preference, and text-image alignment metrics. Our work presents an effective paradigm for training efficient, high-fidelity, and stable generative models. Codes are coming soon. 

---
# New York Smells: A Large Multimodal Dataset for Olfaction 

**Authors**: Ege Ozguroglu, Junbang Liang, Ruoshi Liu, Mia Chiquier, Michael DeTienne, Wesley Wei Qian, Alexandra Horowitz, Andrew Owens, Carl Vondrick  

**Link**: [PDF](https://arxiv.org/pdf/2511.20544)  

**Abstract**: While olfaction is central to how animals perceive the world, this rich chemical sensory modality remains largely inaccessible to machines. One key bottleneck is the lack of diverse, multimodal olfactory training data collected in natural settings. We present New York Smells, a large dataset of paired image and olfactory signals captured ``in the wild.'' Our dataset contains 7,000 smell-image pairs from 3,500 distinct objects across indoor and outdoor environments, with approximately 70$\times$ more objects than existing olfactory datasets. Our benchmark has three tasks: cross-modal smell-to-image retrieval, recognizing scenes, objects, and materials from smell alone, and fine-grained discrimination between grass species. Through experiments on our dataset, we find that visual data enables cross-modal olfactory representation learning, and that our learned olfactory representations outperform widely-used hand-crafted features. 

---
# Automated Monitoring of Cultural Heritage Artifacts Using Semantic Segmentation 

**Authors**: Andrea Ranieri, Giorgio Palmieri, Silvia Biasotti  

**Link**: [PDF](https://arxiv.org/pdf/2511.20541)  

**Abstract**: This paper addresses the critical need for automated crack detection in the preservation of cultural heritage through semantic segmentation. We present a comparative study of U-Net architectures, using various convolutional neural network (CNN) encoders, for pixel-level crack identification on statues and monuments. A comparative quantitative evaluation is performed on the test set of the OmniCrack30k dataset [1] using popular segmentation metrics including Mean Intersection over Union (mIoU), Dice coefficient, and Jaccard index. This is complemented by an out-of-distribution qualitative evaluation on an unlabeled test set of real-world cracked statues and monuments. Our findings provide valuable insights into the capabilities of different CNN- based encoders for fine-grained crack segmentation. We show that the models exhibit promising generalization capabilities to unseen cultural heritage contexts, despite never having been explicitly trained on images of statues or monuments. 

---
# Proceedings Twentieth Conference on Theoretical Aspects of Rationality and Knowledge 

**Authors**: Adam Bjorndahl  

**Link**: [PDF](https://arxiv.org/pdf/2511.20540)  

**Abstract**: The TARK conference (Theoretical Aspects of Rationality and Knowledge) is a conference that aims to bring together researchers from a wide variety of fields, including computer science, artificial intelligence, game theory, decision theory, philosophy, logic, linguistics, and cognitive science. Its goal is to further our understanding of interdisciplinary issues involving reasoning about rationality and knowledge.
Previous conferences have been held biennially around the world since 1986, on the initiative of Joe Halpern (Cornell University). Topics of interest include, but are not limited to, semantic models for knowledge, belief, uncertainty, awareness, bounded rationality, common sense epistemic reasoning, epistemic logic, epistemic game theory, knowledge and action, applications of reasoning about knowledge and other mental states, belief revision, computational social choice, algorithmic game theory, and foundations of multi-agent systems.
Information about TARK is available at this http URL.
These proceedings contain the papers that have been accepted for presentation at the Twentieth Conference on Theoretical Aspects of Rationality and Knowledge (TARK 2025), held July 14--16, 2025, at Heinrich-Heine-Universität, Düsseldorf, Germany. The conference website can be found at this https URL. 

---
# MIMIC-MJX: Neuromechanical Emulation of Animal Behavior 

**Authors**: Charles Y. Zhang, Yuanjia Yang, Aidan Sirbu, Elliott T.T. Abe, Emil Wärnberg, Eric J. Leonardis, Diego E. Aldarondo, Adam Lee, Aaditya Prasad, Jason Foat, Kaiwen Bian, Joshua Park, Rusham Bhatt, Hutton Saunders, Akira Nagamori, Ayesha R. Thanawalla, Kee Wui Huang, Fabian Plum, Hendrik K. Beck, Steven W. Flavell, David Labonte, Blake A. Richards, Bingni W. Brunton, Eiman Azim, Bence P. Ölveczky, Talmo D. Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2511.20532)  

**Abstract**: The primary output of the nervous system is movement and behavior. While recent advances have democratized pose tracking during complex behavior, kinematic trajectories alone provide only indirect access to the underlying control processes. Here we present MIMIC-MJX, a framework for learning biologically-plausible neural control policies from kinematics. MIMIC-MJX models the generative process of motor control by training neural controllers that learn to actuate biomechanically-realistic body models in physics simulation to reproduce real kinematic trajectories. We demonstrate that our implementation is accurate, fast, data-efficient, and generalizable to diverse animal body models. Policies trained with MIMIC-MJX can be utilized to both analyze neural control strategies and simulate behavioral experiments, illustrating its potential as an integrative modeling framework for neuroscience. 

---
# DesignPref: Capturing Personal Preferences in Visual Design Generation 

**Authors**: Yi-Hao Peng, Jeffrey P. Bigham, Jason Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.20513)  

**Abstract**: Generative models, such as large language models and text-to-image diffusion models, are increasingly used to create visual designs like user interfaces (UIs) and presentation slides. Finetuning and benchmarking these generative models have often relied on datasets of human-annotated design preferences. Yet, due to the subjective and highly personalized nature of visual design, preference varies widely among individuals. In this paper, we study this problem by introducing DesignPref, a dataset of 12k pairwise comparisons of UI design generation annotated by 20 professional designers with multi-level preference ratings. We found that among trained designers, substantial levels of disagreement exist (Krippendorff's alpha = 0.25 for binary preferences). Natural language rationales provided by these designers indicate that disagreements stem from differing perceptions of various design aspect importance and individual preferences. With DesignPref, we demonstrate that traditional majority-voting methods for training aggregated judge models often do not accurately reflect individual preferences. To address this challenge, we investigate multiple personalization strategies, particularly fine-tuning or incorporating designer-specific annotations into RAG pipelines. Our results show that personalized models consistently outperform aggregated baseline models in predicting individual designers' preferences, even when using 20 times fewer examples. Our work provides the first dataset to study personalized visual design evaluation and support future research into modeling individual design taste. 

---
# The Text Aphasia Battery (TAB): A Clinically-Grounded Benchmark for Aphasia-Like Deficits in Language Models 

**Authors**: Nathan Roll, Jill Kries, Flora Jin, Catherine Wang, Ann Marie Finley, Meghan Sumner, Cory Shain, Laura Gwilliams  

**Link**: [PDF](https://arxiv.org/pdf/2511.20507)  

**Abstract**: Large language models (LLMs) have emerged as a candidate "model organism" for human language, offering an unprecedented opportunity to study the computational basis of linguistic disorders like aphasia. However, traditional clinical assessments are ill-suited for LLMs, as they presuppose human-like pragmatic pressures and probe cognitive processes not inherent to artificial architectures. We introduce the Text Aphasia Battery (TAB), a text-only benchmark adapted from the Quick Aphasia Battery (QAB) to assess aphasic-like deficits in LLMs. The TAB comprises four subtests: Connected Text, Word Comprehension, Sentence Comprehension, and Repetition. This paper details the TAB's design, subtests, and scoring criteria. To facilitate large-scale use, we validate an automated evaluation protocol using Gemini 2.5 Flash, which achieves reliability comparable to expert human raters (prevalence-weighted Cohen's kappa = 0.255 for model--consensus agreement vs. 0.286 for human--human agreement). We release TAB as a clinically-grounded, scalable framework for analyzing language deficits in artificial systems. 

---
# From One Attack Domain to Another: Contrastive Transfer Learning with Siamese Networks for APT Detection 

**Authors**: Sidahmed Benabderrahmane, Talal Rahwan  

**Link**: [PDF](https://arxiv.org/pdf/2511.20500)  

**Abstract**: Advanced Persistent Threats (APT) pose a major cybersecurity challenge due to their stealth, persistence, and adaptability. Traditional machine learning detectors struggle with class imbalance, high dimensional features, and scarce real world traces. They often lack transferability-performing well in the training domain but degrading in novel attack scenarios. We propose a hybrid transfer framework that integrates Transfer Learning, Explainable AI (XAI), contrastive learning, and Siamese networks to improve cross-domain generalization. An attention-based autoencoder supports knowledge transfer across domains, while Shapley Additive exPlanations (SHAP) select stable, informative features to reduce dimensionality and computational cost. A Siamese encoder trained with a contrastive objective aligns source and target representations, increasing anomaly separability and mitigating feature drift. We evaluate on real-world traces from the DARPA Transparent Computing (TC) program and augment with synthetic attack scenarios to test robustness. Across source to target transfers, the approach delivers improved detection scores with classical and deep baselines, demonstrating a scalable, explainable, and transferable solution for APT detection. 

---
# MTBBench: A Multimodal Sequential Clinical Decision-Making Benchmark in Oncology 

**Authors**: Kiril Vasilev, Alexandre Misrahi, Eeshaan Jain, Phil F Cheng, Petros Liakopoulos, Olivier Michielin, Michael Moor, Charlotte Bunne  

**Link**: [PDF](https://arxiv.org/pdf/2511.20490)  

**Abstract**: Multimodal Large Language Models (LLMs) hold promise for biomedical reasoning, but current benchmarks fail to capture the complexity of real-world clinical workflows. Existing evaluations primarily assess unimodal, decontextualized question-answering, overlooking multi-agent decision-making environments such as Molecular Tumor Boards (MTBs). MTBs bring together diverse experts in oncology, where diagnostic and prognostic tasks require integrating heterogeneous data and evolving insights over time. Current benchmarks lack this longitudinal and multimodal complexity. We introduce MTBBench, an agentic benchmark simulating MTB-style decision-making through clinically challenging, multimodal, and longitudinal oncology questions. Ground truth annotations are validated by clinicians via a co-developed app, ensuring clinical relevance. We benchmark multiple open and closed-source LLMs and show that, even at scale, they lack reliability -- frequently hallucinating, struggling with reasoning from time-resolved data, and failing to reconcile conflicting evidence or different modalities. To address these limitations, MTBBench goes beyond benchmarking by providing an agentic framework with foundation model-based tools that enhance multi-modal and longitudinal reasoning, leading to task-level performance gains of up to 9.0% and 11.2%, respectively. Overall, MTBBench offers a challenging and realistic testbed for advancing multimodal LLM reasoning, reliability, and tool-use with a focus on MTB environments in precision oncology. 

---
# Ranking-Enhanced Anomaly Detection Using Active Learning-Assisted Attention Adversarial Dual AutoEncoders 

**Authors**: Sidahmed Benabderrahmane, James Cheney, Talal Rahwan  

**Link**: [PDF](https://arxiv.org/pdf/2511.20480)  

**Abstract**: Advanced Persistent Threats (APTs) pose a significant challenge in cybersecurity due to their stealthy and long-term nature. Modern supervised learning methods require extensive labeled data, which is often scarce in real-world cybersecurity environments. In this paper, we propose an innovative approach that leverages AutoEncoders for unsupervised anomaly detection, augmented by active learning to iteratively improve the detection of APT anomalies. By selectively querying an oracle for labels on uncertain or ambiguous samples, we minimize labeling costs while improving detection rates, enabling the model to improve its detection accuracy with minimal data while reducing the need for extensive manual labeling. We provide a detailed formulation of the proposed Attention Adversarial Dual AutoEncoder-based anomaly detection framework and show how the active learning loop iteratively enhances the model. The framework is evaluated on real-world imbalanced provenance trace databases produced by the DARPA Transparent Computing program, where APT-like attacks constitute as little as 0.004\% of the data. The datasets span multiple operating systems, including Android, Linux, BSD, and Windows, and cover two attack scenarios. The results have shown significant improvements in detection rates during active learning and better performance compared to other existing approaches. 

---
# Efficient and Fast Generative-Based Singing Voice Separation using a Latent Diffusion Model 

**Authors**: Genís Plaja-Roglans, Yun-Ning Hung, Xavier Serra, Igor Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2511.20470)  

**Abstract**: Extracting individual elements from music mixtures is a valuable tool for music production and practice. While neural networks optimized to mask or transform mixture spectrograms into the individual source(s) have been the leading approach, the source overlap and correlation in music signals poses an inherent challenge. Also, accessing all sources in the mixture is crucial to train these systems, while complicated. Attempts to address these challenges in a generative fashion exist, however, the separation performance and inference efficiency remain limited. In this work, we study the potential of diffusion models to advance toward bridging this gap, focusing on generative singing voice separation relying only on corresponding pairs of isolated vocals and mixtures for training. To align with creative workflows, we leverage latent diffusion: the system generates samples encoded in a compact latent space, and subsequently decodes these into audio. This enables efficient optimization and faster inference. Our system is trained using only open data. We outperform existing generative separation systems, and level the compared non-generative systems on a list of signal quality measures and on interference removal. We provide a noise robustness study on the latent encoder, providing insights on its potential for the task. We release a modular toolkit for further research on the topic. 

---
# Generation, Evaluation, and Explanation of Novelists' Styles with Single-Token Prompts 

**Authors**: Mosab Rezaei, Mina Rajaei Moghadam, Abdul Rahman Shaikh, Hamed Alhoori, Reva Freedman  

**Link**: [PDF](https://arxiv.org/pdf/2511.20459)  

**Abstract**: Recent advances in large language models have created new opportunities for stylometry, the study of writing styles and authorship. Two challenges, however, remain central: training generative models when no paired data exist, and evaluating stylistic text without relying only on human judgment. In this work, we present a framework for both generating and evaluating sentences in the style of 19th-century novelists. Large language models are fine-tuned with minimal, single-token prompts to produce text in the voices of authors such as Dickens, Austen, Twain, Alcott, and Melville. To assess these generative models, we employ a transformer-based detector trained on authentic sentences, using it both as a classifier and as a tool for stylistic explanation. We complement this with syntactic comparisons and explainable AI methods, including attention-based and gradient-based analyses, to identify the linguistic cues that drive stylistic imitation. Our findings show that the generated text reflects the authors' distinctive patterns and that AI-based evaluation offers a reliable alternative to human assessment. All artifacts of this work are published online. 

---
# Object-Centric Vision Token Pruning for Vision Language Models 

**Authors**: Guangyuan Li, Rongzhen Zhao, Jinhong Deng, Yanbo Wang, Joni Pajarinen  

**Link**: [PDF](https://arxiv.org/pdf/2511.20439)  

**Abstract**: In Vision Language Models (VLMs), vision tokens are quantity-heavy yet information-dispersed compared with language tokens, thus consume too much unnecessary computation. Pruning redundant vision tokens for high VLM inference efficiency has been continuously studied but all existing methods resort to indirect and non-guaranteed ways. We propose OC-VTP, a direct and guaranteed approach to select the most representative vision tokens for high-efficiency yet accuracy-preserving VLM inference. Our OC-VTP requires merely light-weight pre-training of a small object-centric vision token pruner, which can then be inserted into existing VLMs, without fine-tuning of any models on any datasets. It is gauranteed that the most representative vision tokens are kept by minimizing the error in reconstructing the original unpruned tokens from the selected ones. Across any vision pruning ratios, i.e., inference efficiency, our OC-VTP consistently helps mainstream VLMs to preserve the highest inference accuracy. Our pruning also demonstrates interesting interpretability. Our codes are available at this https URL. 

---
# Block Cascading: Training Free Acceleration of Block-Causal Video Models 

**Authors**: Hmrishav Bandyopadhyay, Nikhil Pinnaparaju, Rahim Entezari, Jim Scott, Yi-Zhe Song, Varun Jampani  

**Link**: [PDF](https://arxiv.org/pdf/2511.20426)  

**Abstract**: Block-causal video generation faces a stark speed-quality trade-off: small 1.3B models manage only 16 FPS while large 14B models crawl at 4.5 FPS, forcing users to choose between responsiveness and quality. Block Cascading significantly mitigates this trade-off through training-free parallelization. Our key insight: future video blocks do not need fully denoised current blocks to begin generation. By starting block generation with partially denoised context from predecessors, we transform sequential pipelines into parallel cascades where multiple blocks denoise simultaneously. With 5 GPUs exploiting temporal parallelism, we achieve ~2x acceleration across all model scales: 1.3B models accelerate from 16 to 30 FPS, 14B models from 4.5 to 12.5 FPS. Beyond inference speed, Block Cascading eliminates overhead from KV-recaching (of ~200ms) during context switches for interactive generation. Extensive evaluations validated against multiple block-causal pipelines demonstrate no significant loss in generation quality when switching from block-causal to Block Cascading pipelines for inference. Project Page: this https URL 

---
# StableTrack: Stabilizing Multi-Object Tracking on Low-Frequency Detections 

**Authors**: Matvei Shelukhan, Timur Mamedov, Karina Kvanchiani  

**Link**: [PDF](https://arxiv.org/pdf/2511.20418)  

**Abstract**: Multi-object tracking (MOT) is one of the most challenging tasks in computer vision, where it is important to correctly detect objects and associate these detections across frames. Current approaches mainly focus on tracking objects in each frame of a video stream, making it almost impossible to run the model under conditions of limited computing resources. To address this issue, we propose StableTrack, a novel approach that stabilizes the quality of tracking on low-frequency detections. Our method introduces a new two-stage matching strategy to improve the cross-frame association between low-frequency detections. We propose a novel Bbox-Based Distance instead of the conventional Mahalanobis distance, which allows us to effectively match objects using the Re-ID model. Furthermore, we integrate visual tracking into the Kalman Filter and the overall tracking pipeline. Our method outperforms current state-of-the-art trackers in the case of low-frequency detections, achieving $\textit{11.6%}$ HOTA improvement at $\textit{1}$ Hz on MOT17-val, while keeping up with the best approaches on the standard MOT17, MOT20, and DanceTrack benchmarks with full-frequency detections. 

---
# Short-Range Oversquashing 

**Authors**: Yaaqov Mishayev, Yonatan Sverdlov, Tal Amir, Nadav Dym  

**Link**: [PDF](https://arxiv.org/pdf/2511.20406)  

**Abstract**: Message Passing Neural Networks (MPNNs) are widely used for learning on graphs, but their ability to process long-range information is limited by the phenomenon of oversquashing. This limitation has led some researchers to advocate Graph Transformers as a better alternative, whereas others suggest that it can be mitigated within the MPNN framework, using virtual nodes or other rewiring techniques.
In this work, we demonstrate that oversquashing is not limited to long-range tasks, but can also arise in short-range problems. This observation allows us to disentangle two distinct mechanisms underlying oversquashing: (1) the bottleneck phenomenon, which can arise even in low-range settings, and (2) the vanishing gradient phenomenon, which is closely associated with long-range tasks.
We further show that the short-range bottleneck effect is not captured by existing explanations for oversquashing, and that adding virtual nodes does not resolve it. In contrast, transformers do succeed in such tasks, positioning them as the more compelling solution to oversquashing, compared to specialized MPNNs. 

---
# LLMs for Automated Unit Test Generation and Assessment in Java: The AgoneTest Framework 

**Authors**: Andrea Lops, Fedelucio Narducci, Azzurra Ragone, Michelantonio Trizio, Claudio Barto  

**Link**: [PDF](https://arxiv.org/pdf/2511.20403)  

**Abstract**: Unit testing is an essential but resource-intensive step in software development, ensuring individual code units function correctly. This paper introduces AgoneTest, an automated evaluation framework for Large Language Model-generated (LLM) unit tests in Java. AgoneTest does not aim to propose a novel test generation algorithm; rather, it supports researchers and developers in comparing different LLMs and prompting strategies through a standardized end-to-end evaluation pipeline under realistic conditions. We introduce the Classes2Test dataset, which maps Java classes under test to their corresponding test classes, and a framework that integrates advanced evaluation metrics, such as mutation score and test smells, for a comprehensive assessment. Experimental results show that, for the subset of tests that compile, LLM-generated tests can match or exceed human-written tests in terms of coverage and defect detection. Our findings also demonstrate that enhanced prompting strategies contribute to test quality. AgoneTest clarifies the potential of LLMs in software testing and offers insights for future improvements in model design, prompt engineering, and testing practices. 

---
# BengaliFig: A Low-Resource Challenge for Figurative and Culturally Grounded Reasoning in Bengali 

**Authors**: Abdullah Al Sefat  

**Link**: [PDF](https://arxiv.org/pdf/2511.20399)  

**Abstract**: Large language models excel on broad multilingual benchmarks but remain to be evaluated extensively in figurative and culturally grounded reasoning, especially in low-resource contexts. We present BengaliFig, a compact yet richly annotated challenge set that targets this gap in Bengali, a widely spoken low-resourced language. The dataset contains 435 unique riddles drawn from Bengali oral and literary traditions. Each item is annotated along five orthogonal dimensions capturing reasoning type, trap type, cultural depth, answer category, and difficulty, and is automatically converted to multiple-choice format through a constraint-aware, AI-assisted pipeline. We evaluate eight frontier LLMs from major providers under zero-shot and few-shot chain-of-thought prompting, revealing consistent weaknesses in metaphorical and culturally specific reasoning. BengaliFig thus contributes both a diagnostic probe for evaluating LLM robustness in low-resource cultural contexts and a step toward inclusive and heritage-aware NLP evaluation. 

---
# From Passive Perception to Active Memory: A Weakly Supervised Image Manipulation Localization Framework Driven by Coarse-Grained Annotations 

**Authors**: Zhiqing Guo, Dongdong Xi, Songlin Li, Gaobo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.20359)  

**Abstract**: Image manipulation localization (IML) faces a fundamental trade-off between minimizing annotation cost and achieving fine-grained localization accuracy. Existing fully-supervised IML methods depend heavily on dense pixel-level mask annotations, which limits scalability to large datasets or real-world this http URL contrast, the majority of existing weakly-supervised IML approaches are based on image-level labels, which greatly reduce annotation effort but typically lack precise spatial localization. To address this dilemma, we propose BoxPromptIML, a novel weakly-supervised IML framework that effectively balances annotation cost and localization performance. Specifically, we propose a coarse region annotation strategy, which can generate relatively accurate manipulation masks at lower cost. To improve model efficiency and facilitate deployment, we further design an efficient lightweight student model, which learns to perform fine-grained localization through knowledge distillation from a fixed teacher model based on the Segment Anything Model (SAM). Moreover, inspired by the human subconscious memory mechanism, our feature fusion module employs a dual-guidance strategy that actively contextualizes recalled prototypical patterns with real-time observational cues derived from the input. Instead of passive feature extraction, this strategy enables a dynamic process of knowledge recollection, where long-term memory is adapted to the specific context of the current image, significantly enhancing localization accuracy and robustness. Extensive experiments across both in-distribution and out-of-distribution datasets show that BoxPromptIML outperforms or rivals fully-supervised models, while maintaining strong generalization, low annotation cost, and efficient deployment characteristics. 

---
# Soft Adaptive Policy Optimization 

**Authors**: Chang Gao, Chujie Zheng, Xiong-Hui Chen, Kai Dang, Shixuan Liu, Bowen Yu, An Yang, Shuai Bai, Jingren Zhou, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.20347)  

**Abstract**: Reinforcement learning (RL) plays an increasingly important role in enhancing the reasoning capabilities of large language models (LLMs), yet stable and performant policy optimization remains challenging. Token-level importance ratios often exhibit high variance-a phenomenon exacerbated in Mixture-of-Experts models-leading to unstable updates. Existing group-based policy optimization methods, such as GSPO and GRPO, alleviate this problem via hard clipping, making it difficult to maintain both stability and effective learning. We propose Soft Adaptive Policy Optimization (SAPO), which replaces hard clipping with a smooth, temperature-controlled gate that adaptively attenuates off-policy updates while preserving useful learning signals. Compared with GSPO and GRPO, SAPO is both sequence-coherent and token-adaptive. Like GSPO, SAPO maintains sequence-level coherence, but its soft gating forms a continuous trust region that avoids the brittle hard clipping band used in GSPO. When a sequence contains a few highly off-policy tokens, GSPO suppresses all gradients for that sequence, whereas SAPO selectively down-weights only the offending tokens and preserves the learning signal from the near-on-policy ones, improving sample efficiency. Relative to GRPO, SAPO replaces hard token-level clipping with smooth, temperature-controlled scaling, enabling more informative and stable updates. Empirical results on mathematical reasoning benchmarks indicate that SAPO exhibits improved training stability and higher Pass@1 performance under comparable training budgets. Moreover, we employ SAPO to train the Qwen3-VL model series, demonstrating that SAPO yields consistent performance gains across diverse tasks and different model sizes. Overall, SAPO provides a more reliable, scalable, and effective optimization strategy for RL training of LLMs. 

---
# 3D Motion Perception of Binocular Vision Target with PID-CNN 

**Authors**: Shi Jiazhao, Pan Pan, Shi Haotian  

**Link**: [PDF](https://arxiv.org/pdf/2511.20332)  

**Abstract**: This article trained a network for perceiving three-dimensional motion information of binocular vision target, which can provide real-time three-dimensional coordinate, velocity, and acceleration, and has a basic spatiotemporal perception capability. Understood the ability of neural networks to fit nonlinear problems from the perspective of PID. Considered a single-layer neural network as using a second-order difference equation and a nonlinearity to describe a local problem. Multilayer networks gradually transform the raw representation to the desired representation through multiple such combinations. Analysed some reference principles for designing neural networks. Designed a relatively small PID convolutional neural network, with a total of 17 layers and 413 thousand parameters. Implemented a simple but practical feature reuse method by concatenation and pooling. The network was trained and tested using the simulated randomly moving ball datasets, and the experimental results showed that the prediction accuracy was close to the upper limit that the input image resolution can represent. Analysed the experimental results and errors, as well as the existing shortcomings and possible directions for improvement. Finally, discussed the advantages of high-dimensional convolution in improving computational efficiency and feature space utilization. As well as the potential advantages of using PID information to implement memory and attention mechanisms. 

---
# Geometry of Decision Making in Language Models 

**Authors**: Abhinav Joshi, Divyanshu Bhatt, Ashutosh Modi  

**Link**: [PDF](https://arxiv.org/pdf/2511.20315)  

**Abstract**: Large Language Models (LLMs) show strong generalization across diverse tasks, yet the internal decision-making processes behind their predictions remain opaque. In this work, we study the geometry of hidden representations in LLMs through the lens of \textit{intrinsic dimension} (ID), focusing specifically on decision-making dynamics in a multiple-choice question answering (MCQA) setting. We perform a large-scale study, with 28 open-weight transformer models and estimate ID across layers using multiple estimators, while also quantifying per-layer performance on MCQA tasks. Our findings reveal a consistent ID pattern across models: early layers operate on low-dimensional manifolds, middle layers expand this space, and later layers compress it again, converging to decision-relevant representations. Together, these results suggest LLMs implicitly learn to project linguistic inputs onto structured, low-dimensional manifolds aligned with task-specific decisions, providing new geometric insights into how generalization and reasoning emerge in language models. 

---
# RIS-Assisted Downlink Pinching-Antenna Systems: GNN-Enabled Optimization Approaches 

**Authors**: Changpeng He, Yang Lu, Yanqing Xu, Chong-Yung Chi, Bo Ai, Arumugam Nallanathan  

**Link**: [PDF](https://arxiv.org/pdf/2511.20305)  

**Abstract**: This paper investigates a reconfigurable intelligent surface (RIS)-assisted multi-waveguide pinching-antenna (PA) system (PASS) for multi-user downlink information transmission, motivated by the unknown impact of the integration of emerging PASS and RIS on wireless communications. First, we formulate sum rate (SR) and energy efficiency (EE) maximization problems in a unified framework, subject to constraints on the movable region of PAs, total power budget, and tunable phase of RIS elements. Then, by leveraging a graph-structured topology of the RIS-assisted PASS, a novel three-stage graph neural network (GNN) is proposed, which learns PA positions based on user locations, and RIS phase shifts according to composite channel conditions at the first two stages, respectively, and finally determines beamforming vectors. Specifically, the proposed GNN is achieved through unsupervised training, together with three implementation strategies for its integration with convex optimization, thus offering trade-offs between inference time and solution optimality. Extensive numerical results are provided to validate the effectiveness of the proposed GNN, and to support its unique attributes of viable generalization capability, good performance reliability, and real-time applicability. Moreover, the impact of key parameters on RIS-assisted PASS is illustrated and analyzed. 

---
# Prompting Lipschitz-constrained network for multiple-in-one sparse-view CT reconstruction 

**Authors**: Baoshun Shi, Ke Jiang, Qiusheng Lian, Xinran Yu, Huazhu Fu  

**Link**: [PDF](https://arxiv.org/pdf/2511.20296)  

**Abstract**: Despite significant advancements in deep learning-based sparse-view computed tomography (SVCT) reconstruction algorithms, these methods still encounter two primary limitations: (i) It is challenging to explicitly prove that the prior networks of deep unfolding algorithms satisfy Lipschitz constraints due to their empirically designed nature. (ii) The substantial storage costs of training a separate model for each setting in the case of multiple views hinder practical clinical applications. To address these issues, we elaborate an explicitly provable Lipschitz-constrained network, dubbed LipNet, and integrate an explicit prompt module to provide discriminative knowledge of different sparse sampling settings, enabling the treatment of multiple sparse view configurations within a single model. Furthermore, we develop a storage-saving deep unfolding framework for multiple-in-one SVCT reconstruction, termed PromptCT, which embeds LipNet as its prior network to ensure the convergence of its corresponding iterative algorithm. In simulated and real data experiments, PromptCT outperforms benchmark reconstruction algorithms in multiple-in-one SVCT reconstruction, achieving higher-quality reconstructions with lower storage costs. On the theoretical side, we explicitly demonstrate that LipNet satisfies boundary property, further proving its Lipschitz continuity and subsequently analyzing the convergence of the proposed iterative algorithms. The data and code are publicly available at this https URL. 

---
# Forgetting by Pruning: Data Deletion in Join Cardinality Estimation 

**Authors**: Chaowei He, Yuanjun Liu, Qingzhi Ma, Shenyuan Ren, Xizhao Luo, Lei Zhao, An Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.20293)  

**Abstract**: Machine unlearning in learned cardinality estimation (CE) systems presents unique challenges due to the complex distributional dependencies in multi-table relational data. Specifically, data deletion, a core component of machine unlearning, faces three critical challenges in learned CE models: attribute-level sensitivity, inter-table propagation and domain disappearance leading to severe overestimation in multi-way joins. We propose Cardinality Estimation Pruning (CEP), the first unlearning framework specifically designed for multi-table learned CE systems. CEP introduces Distribution Sensitivity Pruning, which constructs semi-join deletion results and computes sensitivity scores to guide parameter pruning, and Domain Pruning, which removes support for value domains entirely eliminated by deletion. We evaluate CEP on state-of-the-art architectures NeuroCard and FACE across IMDB and TPC-H datasets. Results demonstrate CEP consistently achieves the lowest Q-error in multi-table scenarios, particularly under high deletion ratios, often outperforming full retraining. Furthermore, CEP significantly reduces convergence iterations, incurring negligible computational overhead of 0.3%-2.5% of fine-tuning time. 

---
# Can LLMs Make (Personalized) Access Control Decisions? 

**Authors**: Friederike Groschupp, Daniele Lain, Aritra Dhar, Lara Magdalena Lazier, Srdjan Čapkun  

**Link**: [PDF](https://arxiv.org/pdf/2511.20284)  

**Abstract**: Precise access control decisions are crucial to the security of both traditional applications and emerging agent-based systems. Typically, these decisions are made by users during app installation or at runtime. Due to the increasing complexity and automation of systems, making these access control decisions can add a significant cognitive load on users, often overloading them and leading to suboptimal or even arbitrary access control decisions. To address this problem, we propose to leverage the processing and reasoning capabilities of large language models (LLMs) to make dynamic, context-aware decisions aligned with the user's security preferences. For this purpose, we conducted a user study, which resulted in a dataset of 307 natural-language privacy statements and 14,682 access control decisions made by users. We then compare these decisions against those made by two versions of LLMs: a general and a personalized one, for which we also gathered user feedback on 1,446 of its decisions.
Our results show that in general, LLMs can reflect users' preferences well, achieving up to 86\% accuracy when compared to the decision made by the majority of users. Our study also reveals a crucial trade-off in personalizing such a system: while providing user-specific privacy preferences to the LLM generally improves agreement with individual user decisions, adhering to those preferences can also violate some security best practices. Based on our findings, we discuss design and risk considerations for implementing a practical natural-language-based access control system that balances personalization, security, and utility. 

---
# HVAdam: A Full-Dimension Adaptive Optimizer 

**Authors**: Yiheng Zhang, Shaowu Wu, Yuanzhuo Xu, Jiajun Wu, Shang Xu, Steve Drew, Xiaoguang Niu  

**Link**: [PDF](https://arxiv.org/pdf/2511.20277)  

**Abstract**: Adaptive optimizers such as Adam have achieved great success in training large-scale models like large language models and diffusion models. However, they often generalize worse than non-adaptive methods, such as SGD on classical architectures like CNNs. We identify a key cause of this performance gap: adaptivity in pre-conditioners, which limits the optimizer's ability to adapt to diverse optimization landscapes. To address this, we propose Anon (Adaptivity Non-restricted Optimizer with Novel convergence technique), a novel optimizer with continuously tunable adaptivity
, allowing it to interpolate between SGD-like and Adam-like behaviors and even extrapolate beyond both. To ensure convergence across the entire adaptivity spectrum, we introduce incremental delay update (IDU), a novel mechanism that is more flexible than AMSGrad's hard max-tracking strategy and enhances robustness to gradient noise. We theoretically establish convergence guarantees under both convex and non-convex settings. Empirically, Anon consistently outperforms state-of-the-art optimizers on representative image classification, diffusion, and language modeling tasks. These results demonstrate that adaptivity can serve as a valuable tunable design principle, and Anon provides the first unified and reliable framework capable of bridging the gap between classical and modern optimizers and surpassing their advantageous properties. 

---
# Beyond Components: Singular Vector-Based Interpretability of Transformer Circuits 

**Authors**: Areeb Ahmad, Abhinav Joshi, Ashutosh Modi  

**Link**: [PDF](https://arxiv.org/pdf/2511.20273)  

**Abstract**: Transformer-based language models exhibit complex and distributed behavior, yet their internal computations remain poorly understood. Existing mechanistic interpretability methods typically treat attention heads and multilayer perceptron layers (MLPs) (the building blocks of a transformer architecture) as indivisible units, overlooking possibilities of functional substructure learned within them. In this work, we introduce a more fine-grained perspective that decomposes these components into orthogonal singular directions, revealing superposed and independent computations within a single head or MLP. We validate our perspective on widely used standard tasks like Indirect Object Identification (IOI), Gender Pronoun (GP), and Greater Than (GT), showing that previously identified canonical functional heads, such as the name mover, encode multiple overlapping subfunctions aligned with distinct singular directions. Nodes in a computational graph, that are previously identified as circuit elements show strong activation along specific low-rank directions, suggesting that meaningful computations reside in compact subspaces. While some directions remain challenging to interpret fully, our results highlight that transformer computations are more distributed, structured, and compositional than previously assumed. This perspective opens new avenues for fine-grained mechanistic interpretability and a deeper understanding of model internals. 

---
# Interpretable Air Pollution Forecasting by Physics-Guided Spatiotemporal Decoupling 

**Authors**: Zhiguo Zhang, Xiaoliang Ma, Daniel Schlesinger  

**Link**: [PDF](https://arxiv.org/pdf/2511.20257)  

**Abstract**: Accurate and interpretable air pollution forecasting is crucial for public health, but most models face a trade-off between performance and interpretability. This study proposes a physics-guided, interpretable-by-design spatiotemporal learning framework. The model decomposes the spatiotemporal behavior of air pollutant concentrations into two transparent, additive modules. The first is a physics-guided transport kernel with directed weights conditioned on wind and geography (advection). The second is an explainable attention mechanism that learns local responses and attributes future concentrations to specific historical lags and exogenous drivers. Evaluated on a comprehensive dataset from the Stockholm region, our model consistently outperforms state-of-the-art baselines across multiple forecasting horizons. Our model's integration of high predictive performance and spatiotemporal interpretability provides a more reliable foundation for operational air-quality management in real-world applications. 

---
# XiCAD: Camera Activation Detection in the Da Vinci Xi User Interface 

**Authors**: Alexander C. Jenke, Gregor Just, Claas de Boer, Martin Wagner, Sebastian Bodenstedt, Stefanie Speidel  

**Link**: [PDF](https://arxiv.org/pdf/2511.20254)  

**Abstract**: Purpose: Robot-assisted minimally invasive surgery relies on endoscopic video as the sole intraoperative visual feedback. The DaVinci Xi system overlays a graphical user interface (UI) that indicates the state of each robotic arm, including the activation of the endoscope arm. Detecting this activation provides valuable metadata such as camera movement information, which can support downstream surgical data science tasks including tool tracking, skill assessment, or camera control automation.
Methods: We developed a lightweight pipeline based on a ResNet18 convolutional neural network to automatically identify the position of the camera tile and its activation state within the DaVinci Xi UI. The model was fine-tuned on manually annotated data from the SurgToolLoc dataset and evaluated across three public datasets comprising over 70,000 frames.
Results: The model achieved F1-scores between 0.993 and 1.000 for the binary detection of active cameras and correctly localized the camera tile in all cases without false multiple-camera detections.
Conclusion: The proposed pipeline enables reliable, real-time extraction of camera activation metadata from surgical videos, facilitating automated preprocessing and analysis for diverse downstream applications. All code, trained models, and annotations are publicly available. 

---
# Uplifting Table Tennis: A Robust, Real-World Application for 3D Trajectory and Spin Estimation 

**Authors**: Daniel Kienzle, Katja Ludwig, Julian Lorenz, Shin'ichi Satoh, Rainer Lienhart  

**Link**: [PDF](https://arxiv.org/pdf/2511.20250)  

**Abstract**: Obtaining the precise 3D motion of a table tennis ball from standard monocular videos is a challenging problem, as existing methods trained on synthetic data struggle to generalize to the noisy, imperfect ball and table detections of the real world. This is primarily due to the inherent lack of 3D ground truth trajectories and spin annotations for real-world video. To overcome this, we propose a novel two-stage pipeline that divides the problem into a front-end perception task and a back-end 2D-to-3D uplifting task. This separation allows us to train the front-end components with abundant 2D supervision from our newly created TTHQ dataset, while the back-end uplifting network is trained exclusively on physically-correct synthetic data. We specifically re-engineer the uplifting model to be robust to common real-world artifacts, such as missing detections and varying frame rates. By integrating a ball detector and a table keypoint detector, our approach transforms a proof-of-concept uplifting method into a practical, robust, and high-performing end-to-end application for 3D table tennis trajectory and spin analysis. 

---
# Leveraging weights signals - Predicting and improving generalizability in reinforcement learning 

**Authors**: Olivier Moulin, Vincent Francois-lavet, Paul Elbers, Mark Hoogendoorn  

**Link**: [PDF](https://arxiv.org/pdf/2511.20234)  

**Abstract**: Generalizability of Reinforcement Learning (RL) agents (ability to perform on environments different from the ones they have been trained on) is a key problem as agents have the tendency to overfit to their training environments. In order to address this problem and offer a solution to increase the generalizability of RL agents, we introduce a new methodology to predict the generalizability score of RL agents based on the internal weights of the agent's neural networks. Using this prediction capability, we propose some changes in the Proximal Policy Optimization (PPO) loss function to boost the generalization score of the agents trained with this upgraded version. Experimental results demonstrate that our improved PPO algorithm yields agents with stronger generalizability compared to the original version. 

---
# DUO-TOK: Dual-Track Semantic Music Tokenizer for Vocal-Accompaniment Generation 

**Authors**: Rui Lin, Zhiyue Wu, Jiahe Le, Kangdi Wang, Weixiong Chen, Junyu Dai, Tao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2511.20224)  

**Abstract**: Duo-Tok is a source-aware dual-codebook tokenizer for vocal-accompaniment music that targets the growing tension between reconstruction quality and language-model (LM) learnability in modern lyrics-to-song systems. Existing codecs either prioritize high-fidelity reconstruction with difficult-to-model acoustic tokens or compress aggressively into semantic tokens that are LM-friendly but lossy, and they rarely make the tokenizer itself aware of dual-track structure. Duo-Tok follows a four-stage, SSL-centered pipeline: we first pretrain a BEST-RQ-style encoder on large-scale audio, then stabilize and factorize the representation with Gaussian replacement noise and multi-task supervision, before freezing the encoder to learn SimVQ-based dual codebooks with hard routing for vocals and accompaniment, and finally training latent diffusion decoders on top of the discrete tokens. Duo-Tok at 0.75 kbps shifts the empirical reconstruction-generation Pareto frontier, achieving the best music-tagging AP and the lowest vocabulary-normalized LM perplexity among compared codecs while maintaining reconstruction quality comparable to state-of-the-art music tokenizers. 

---
# OmniAlpha: A Sequence-to-Sequence Framework for Unified Multi-Task RGBA Generation 

**Authors**: Hao Yu, Jiabo Zhan, Zile Wang, Jinglin Wang, Huaisong Zhang, Hongyu Li, Xinrui Chen, Yongxian Wei, Chun Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2511.20211)  

**Abstract**: Generative models have excelled in RGB synthesis, but real-world applications require RGBA manipulation. This has led to a fragmented landscape: specialized, single-task models handle alpha but lack versatility, while unified multi-task frameworks are confined to the RGB domain. To bridge this critical gap, we propose OmniAlpha, the first unified, multi-task generative framework for sequence-to-sequence RGBA image generation and editing. Its architecture features MSRoPE-BiL, a novel RoPE method with a bi-directionally extendable layer axis for its Diffusion Transformer (DiT) backbone, enabling the concurrent processing of multiple input and target RGBA layers. To power this framework, we introduce AlphaLayers, a new dataset of 1,000 high-quality, multi-layer triplets, built via a novel automated synthesis and filter pipeline. Jointly training OmniAlpha on this dataset across a comprehensive suite of 21 diverse tasks, extensive experiments demonstrate that our unified approach consistently outperforms strong, specialized baselines. Most notably, OmniAlpha achieves a dramatic 84.8% relative reduction in SAD for mask-free matting on AIM-500 and wins over 90% of human preferences in layer-conditioned completion. Our work proves that a unified, multi-task model can learn a superior shared representation for RGBA, paving the way for more powerful, layer-aware generative systems. 

---
# Human-computer interactions predict mental health 

**Authors**: Veith Weilnhammer, Jefferson Ortega, David Whitney  

**Link**: [PDF](https://arxiv.org/pdf/2511.20179)  

**Abstract**: Scalable assessments of mental illness, the leading driver of disability worldwide, remain a critical roadblock toward accessible and equitable care. Here, we show that human-computer interactions encode multiple dimensions of self-reported mental health and their changes over time.
We introduce MAILA, a MAchine-learning framework for Inferring Latent mental states from digital Activity. We trained MAILA to predict 1.3 million mental-health self-reports from 20,000 cursor and touchscreen recordings recorded in 9,000 online participants. The dataset includes 2,000 individuals assessed longitudinally, 1,500 diagnosed with depression, and 500 with obsessive-compulsive disorder. MAILA tracks dynamic mental states along three orthogonal dimensions, generalizes across contexts, and achieves near-ceiling accuracy when predicting group-level mental health. The model translates from general to clinical populations, identifies individuals living with mental illness, and captures signatures of psychological function that are not conveyed by language.
Our results demonstrate how everyday human-computer interactions can power passive, reliable, dynamic, and maximally scalable mental health assessments. The ability to decode mental states at zero marginal cost sets new benchmarks for precision medicine and public health, while raising important questions about privacy, agency, and autonomy online. 

---
# Beluga: A CXL-Based Memory Architecture for Scalable and Efficient LLM KVCache Management 

**Authors**: Xinjun Yang, Qingda Hu, Junru Li, Feifei Li, Yuqi Zhou, Yicong Zhu, Qiuru Lin, Jian Dai, Yang Kong, Jiayu Zhang, Guoqiang Xu, Qiang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.20172)  

**Abstract**: The rapid increase in LLM model sizes and the growing demand for long-context inference have made memory a critical bottleneck in GPU-accelerated serving systems. Although high-bandwidth memory (HBM) on GPUs offers fast access, its limited capacity necessitates reliance on host memory (CPU DRAM) to support larger working sets such as the KVCache. However, the maximum DRAM capacity is constrained by the limited number of memory channels per CPU socket. To overcome this limitation, current systems often adopt RDMA-based disaggregated memory pools, which introduce significant challenges including high access latency, complex communication protocols, and synchronization overhead. Fortunately, the emerging CXL technology introduces new opportunities in KVCache design. In this paper, we propose Beluga, a novel memory architecture that enables GPUs and CPUs to access a shared, large-scale memory pool through CXL switches. By supporting native load/store access semantics over the CXL fabric, our design delivers near-local memory latency, while reducing programming complexity and minimizing synchronization overhead. We conduct a systematic characterization of a commercial CXL switch-based memory pool and propose a set of design guidelines. Based on Beluga, we design and implement Beluga-KVCache, a system tailored for managing the large-scale KVCache in LLM inference. Beluga-KVCache achieves an 89.6% reduction in Time-To-First-Token (TTFT) and 7.35x throughput improvement in the vLLM inference engine compared to RDMA-based solutions. To the best of our knowledge, Beluga is the first system that enables GPUs to directly access large-scale memory pools through CXL switches, marking a significant step toward low-latency, shared access to vast memory resources by GPUs. 

---
# On the Limits of Momentum in Decentralized and Federated Optimization 

**Authors**: Riccardo Zaccone, Sai Praneeth Karimireddy, Carlo Masone  

**Link**: [PDF](https://arxiv.org/pdf/2511.20168)  

**Abstract**: Recent works have explored the use of momentum in local methods to enhance distributed SGD. This is particularly appealing in Federated Learning (FL), where momentum intuitively appears as a solution to mitigate the effects of statistical heterogeneity. Despite recent progress in this direction, it is still unclear if momentum can guarantee convergence under unbounded heterogeneity in decentralized scenarios, where only some workers participate at each round. In this work we analyze momentum under cyclic client participation, and theoretically prove that it remains inevitably affected by statistical heterogeneity. Similarly to SGD, we prove that decreasing step-sizes do not help either: in fact, any schedule decreasing faster than $\Theta\left(1/t\right)$ leads to convergence to a constant value that depends on the initialization and the heterogeneity bound. Numerical results corroborate the theory, and deep learning experiments confirm its relevance for realistic settings. 

---
# While recognizing actions, LMMs struggle to detect core interaction events 

**Authors**: Daniel Harari, Michael Sidorov, Liel David, Chen Shterental, Abrham Kahsay Gebreselasie, Muhammad Haris Khan  

**Link**: [PDF](https://arxiv.org/pdf/2511.20162)  

**Abstract**: Large multi-modal models (LMMs) show increasing performance in realistic visual tasks for images and, more recently, for videos. For example, given a video sequence, such models are able to describe in detail objects, the surroundings and dynamic actions. In this study, we explored the extent to which these models ground their semantic understanding in the actual visual input. Specifically, given sequences of hands interacting with objects, we asked models when and where the interaction begins or ends. For this purpose, we introduce a first of its kind, large-scale dataset with more than 20K annotated interactions on videos from the Something-Something-V2 dataset. 250 AMTurk human annotators labeled core interaction events, particularly when and where objects and agents become attached ('contact') or detached ('release'). We asked two LMMs (Qwen-2.5VL and GPT-4o) to locate these events in short videos, each with a single event. The results show that although the models can reliably name the target objects, identify the action and provide coherent reasoning, they consistently fail to identify the frame where the interaction begins or ends and cannot localize the event within the scene. Our findings suggest that in struggling to pinpoint the moment and location of physical contact that defines the interaction, the models lack the perceptual grounding required for deeper understanding of dynamic scenes. 

---
# SEDA: A Self-Adapted Entity-Centric Data Augmentation for Boosting Gird-based Discontinuous NER Models 

**Authors**: Wen-Fang Su, Hsiao-Wei Chou, Wen-Yang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.20143)  

**Abstract**: Named Entity Recognition (NER) is a critical task in natural language processing, yet it remains particularly challenging for discontinuous entities. The primary difficulty lies in text segmentation, as traditional methods often missegment or entirely miss cross-sentence discontinuous entities, significantly affecting recognition accuracy. Therefore, we aim to address the segmentation and omission issues associated with such entities. Recent studies have shown that grid-tagging methods are effective for information extraction due to their flexible tagging schemes and robust architectures. Building on this, we integrate image data augmentation techniques, such as cropping, scaling, and padding, into grid-based models to enhance their ability to recognize discontinuous entities and handle segmentation challenges. Experimental results demonstrate that traditional segmentation methods often fail to capture cross-sentence discontinuous entities, leading to decreased performance. In contrast, our augmented grid models achieve notable improvements. Evaluations on the CADEC, ShARe13, and ShARe14 datasets show F1 score gains of 1-2.5% overall and 3.7-8.4% for discontinuous entities, confirming the effectiveness of our approach. 

---
# IDAP++: Advancing Divergence-Based Pruning via Filter-Level and Layer-Level Optimization 

**Authors**: Aleksei Samarin, Artem Nazarenko, Egor Kotenko, Valentin Malykh, Alexander Savelev, Aleksei Toropov  

**Link**: [PDF](https://arxiv.org/pdf/2511.20141)  

**Abstract**: This paper presents a novel approach to neural network compression that addresses redundancy at both the filter and architectural levels through a unified framework grounded in information flow analysis. Building on the concept of tensor flow divergence, which quantifies how information is transformed across network layers, we develop a two-stage optimization process. The first stage employs iterative divergence-aware pruning to identify and remove redundant filters while preserving critical information pathways. The second stage extends this principle to higher-level architecture optimization by analyzing layer-wise contributions to information propagation and selectively eliminating entire layers that demonstrate minimal impact on network performance. The proposed method naturally adapts to diverse architectures, including convolutional networks, transformers, and hybrid designs, providing a consistent metric for comparing the structural importance across different layer types. Experimental validation across multiple modern architectures and datasets reveals that this combined approach achieves substantial model compression while maintaining competitive accuracy. The presented approach achieves parameter reduction results that are globally comparable to those of state-of-the-art solutions and outperforms them across a wide range of modern neural network architectures, from convolutional models to transformers. The results demonstrate how flow divergence serves as an effective guiding principle for both filter-level and layer-level optimization, offering practical benefits for deployment in resource-constrained environments. 

---
# "When Data is Scarce, Prompt Smarter"... Approaches to Grammatical Error Correction in Low-Resource Settings 

**Authors**: Somsubhra De, Harsh Kumar, Arun Prakash A  

**Link**: [PDF](https://arxiv.org/pdf/2511.20120)  

**Abstract**: Grammatical error correction (GEC) is an important task in Natural Language Processing that aims to automatically detect and correct grammatical mistakes in text. While recent advances in transformer-based models and large annotated datasets have greatly improved GEC performance for high-resource languages such as English, the progress has not extended equally. For most Indic languages, GEC remains a challenging task due to limited resources, linguistic diversity and complex morphology. In this work, we explore prompting-based approaches using state-of-the-art large language models (LLMs), such as GPT-4.1, Gemini-2.5 and LLaMA-4, combined with few-shot strategy to adapt them to low-resource settings. We observe that even basic prompting strategies, such as zero-shot and few-shot approaches, enable these LLMs to substantially outperform fine-tuned Indic-language models like Sarvam-22B, thereby illustrating the exceptional multilingual generalization capabilities of contemporary LLMs for GEC. Our experiments show that carefully designed prompts and lightweight adaptation significantly enhance correction quality across multiple Indic languages. We achieved leading results in the shared task--ranking 1st in Tamil (GLEU: 91.57) and Hindi (GLEU: 85.69), 2nd in Telugu (GLEU: 85.22), 4th in Bangla (GLEU: 92.86), and 5th in Malayalam (GLEU: 92.97). These findings highlight the effectiveness of prompt-driven NLP techniques and underscore the potential of large-scale LLMs to bridge resource gaps in multilingual GEC. 

---
# LungEvaty: A Scalable, Open-Source Transformer-based Deep Learning Model for Lung Cancer Risk Prediction in LDCT Screening 

**Authors**: Johannes Brandt, Maulik Chevli, Rickmer Braren, Georgios Kaissis, Philip Müller, Daniel Rueckert  

**Link**: [PDF](https://arxiv.org/pdf/2511.20116)  

**Abstract**: Lung cancer risk estimation is gaining increasing importance as more countries introduce population-wide screening programs using low-dose CT (LDCT). As imaging volumes grow, scalable methods that can process entire lung volumes efficiently are essential to tap into the full potential of these large screening datasets. Existing approaches either over-rely on pixel-level annotations, limiting scalability, or analyze the lung in fragments, weakening performance. We present LungEvaty, a fully transformer-based framework for predicting 1-6 year lung cancer risk from a single LDCT scan. The model operates on whole-lung inputs, learning directly from large-scale screening data to capture comprehensive anatomical and pathological cues relevant for malignancy risk. Using only imaging data and no region supervision, LungEvaty matches state-of-the-art performance, refinable by an optional Anatomically Informed Attention Guidance (AIAG) loss that encourages anatomically focused attention. In total, LungEvaty was trained on more than 90,000 CT scans, including over 28,000 for fine-tuning and 6,000 for evaluation. The framework offers a simple, data-efficient, and fully open-source solution that provides an extensible foundation for future research in longitudinal and multimodal lung cancer risk prediction. 

---
# The Devil in the Details: Emergent Misalignment, Format and Coherence in Open-Weights LLMs 

**Authors**: Craig Dickson  

**Link**: [PDF](https://arxiv.org/pdf/2511.20104)  

**Abstract**: Prior work has shown that fine-tuning models on a narrow domain with misaligned data can lead to broad misalignment - a phenomenon termed "emergent misalignment" (Betley et al. 2025). While all tested models were susceptible to emergent misalignment, some models showed more resistance than others. Specifically the Qwen-2.5 family proved to be relatively resistant, while GPT-4o exhibited the strongest misalignment. In this paper we evaluate if current-generation open-weights models exhibit similar resistance to the Qwen-2.5 family and measure misalignment robustness over a range of model architectures and scales.
We replicate the effect across nine modern open-weights models (Gemma 3 and Qwen 3 families, 1B-32B parameters). Models fine-tuned on insecure code generation show a 0.68% misalignment rate (compared to 0.07% for base models), matching the lower end of prior open-model results but dramatically lower than GPT-4o's 20%.
We identify a critical format-dependent vulnerability: requiring JSON output doubles misalignment rates compared to natural language prompts (0.96% vs 0.42%). This suggests that structural constraints may bypass safety training by reducing the model's 'degrees of freedom' to refuse. These findings confirm emergent misalignment as a reproducible phenomenon in modern open-weights models, with rates substantially lower than observed in proprietary systems. 

---
# The Making of Digital Ghosts: Designing Ethical AI Afterlives 

**Authors**: Giovanni Spitale, Federico Germani  

**Link**: [PDF](https://arxiv.org/pdf/2511.20094)  

**Abstract**: Advances in artificial intelligence now make it possible to simulate the dead through chatbots, voice clones, and video avatars trained on a person's digital traces. These "digital ghosts" are moving from fiction to commercial reality, reshaping how people mourn and remember. This paper offers a conceptual and ethical analysis of AI-mediated digital afterlives. We define what counts as a digital ghost, trace their rise across personal, commercial, and institutional contexts, and identify core ethical tensions around grief and well-being, truthfulness and deception, consent and posthumous privacy, dignity and misrepresentation, and the commercialization of mourning. To analyze these challenges, we propose a nine-dimensional taxonomy of digital afterlife technologies and, building on it, outline the features of an ethically acceptable digital ghost: premortem intent, mutual consent, transparent and limited data use, clear disclosure, restricted purposes and access, family or estate stewardship, and minimal behavioral agency. We argue for targeted regulation and professional guidelines to ensure that digital ghosts can aid remembrance without slipping into forms of deception. 

---
# R3A: Reliable RTL Repair Framework with Multi-Agent Fault Localization and Stochastic Tree-of-Thoughts Patch Generation 

**Authors**: Zizhang Luo, Fan Cui, Kexing Zhou, Runlin Guo, Mile Xia, Hongyuan Hou, Yun Lian  

**Link**: [PDF](https://arxiv.org/pdf/2511.20090)  

**Abstract**: Repairing RTL bugs is crucial for hardware design and verification. Traditional automatic program repair (APR) methods define dedicated search spaces to locate and fix bugs with program synthesis. However, they heavily rely on fixed templates and can only deal with limited bugs. As an alternative, Large Language Models with the ability to understand code semantics can be explored for RTL repair. However, they suffer from unreliable outcomes due to inherent randomness and long input contexts of RTL code and waveform. To address these challenges, we propose R3A, an LLM-based automatic RTL program repair framework upon the basic model to improve reliability. R3A proposes the stochastic Tree-Of-Thoughts method to control a patch generation agent to explore a validated solution for the bug. The algorithm samples search states according to a heuristic function to balance between exploration and exploitation for a reliable outcome. Besides, R3A proposes a multi-agent fault localization method to find fault candidates as the starting points for the patch generation agent, further increasing the reliability. Experiments show R3A can fix 90.6% of bugs in the RTL-repair dataset within a given time limit, which covers 45% more bugs than traditional methods and other LLM-based approaches, while achieving an 86.7% pass@5 rate on average, showing a high reliability. 

---
# Explainable Visual Anomaly Detection via Concept Bottleneck Models 

**Authors**: Arianna Stropeni, Valentina Zaccaria, Francesco Borsatti, Davide Dalle Pezze, Manuel Barusco, Gian Antonio Susto  

**Link**: [PDF](https://arxiv.org/pdf/2511.20088)  

**Abstract**: In recent years, Visual Anomaly Detection (VAD) has gained significant attention due to its ability to identify anomalous images using only normal images during training. Many VAD models work without supervision but are still able to provide visual explanations by highlighting the anomalous regions within an image. However, although these visual explanations can be helpful, they lack a direct and semantically meaningful interpretation for users. To address this limitation, we propose extending Concept Bottleneck Models (CBMs) to the VAD setting. By learning meaningful concepts, the network can provide human-interpretable descriptions of anomalies, offering a novel and more insightful way to explain them. Our contributions are threefold: (i) we develop a Concept Dataset to support research on CBMs for VAD; (ii) we improve the CBM architecture to generate both concept-based and visual explanations, bridging semantic and localization interpretability; and (iii) we introduce a pipeline for synthesizing artificial anomalies, preserving the VAD paradigm of minimizing dependence on rare anomalous samples. Our approach, Concept-Aware Visual Anomaly Detection (CONVAD), achieves performance comparable to classic VAD methods while providing richer, concept-driven explanations that enhance interpretability and trust in VAD systems. 

---
# MFM-point: Multi-scale Flow Matching for Point Cloud Generation 

**Authors**: Petr Molodyk, Jaemoo Choi, David W. Romero, Ming-Yu Liu, Yongxin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.20041)  

**Abstract**: In recent years, point cloud generation has gained significant attention in 3D generative modeling. Among existing approaches, point-based methods directly generate point clouds without relying on other representations such as latent features, meshes, or voxels. These methods offer low training cost and algorithmic simplicity, but often underperform compared to representation-based approaches. In this paper, we propose MFM-Point, a multi-scale Flow Matching framework for point cloud generation that substantially improves the scalability and performance of point-based methods while preserving their simplicity and efficiency. Our multi-scale generation algorithm adopts a coarse-to-fine generation paradigm, enhancing generation quality and scalability without incurring additional training or inference overhead. A key challenge in developing such a multi-scale framework lies in preserving the geometric structure of unordered point clouds while ensuring smooth and consistent distributional transitions across resolutions. To address this, we introduce a structured downsampling and upsampling strategy that preserves geometry and maintains alignment between coarse and fine resolutions. Our experimental results demonstrate that MFM-Point achieves best-in-class performance among point-based methods and challenges the best representation-based methods. In particular, MFM-point demonstrates strong results in multi-category and high-resolution generation tasks. 

---
# WaymoQA: A Multi-View Visual Question Answering Dataset for Safety-Critical Reasoning in Autonomous Driving 

**Authors**: Seungjun Yu, Seonho Lee, Namho Kim, Jaeyo Shin, Junsung Park, Wonjeong Ryu, Raehyuk Jung, Hyunjung Shim  

**Link**: [PDF](https://arxiv.org/pdf/2511.20022)  

**Abstract**: Recent advancements in multimodal large language models (MLLMs) have shown strong understanding of driving scenes, drawing interest in their application to autonomous driving. However, high-level reasoning in safety-critical scenarios, where avoiding one traffic risk can create another, remains a major challenge. Such reasoning is often infeasible with only a single front view and requires a comprehensive view of the environment, which we achieve through multi-view inputs. We define Safety-Critical Reasoning as a new task that leverages multi-view inputs to address this challenge. Then, we distill Safety-Critical Reasoning into two stages: first resolve the immediate risk, then mitigate the decision-induced downstream risks. To support this, we introduce WaymoQA, a dataset of 35,000 human-annotated question-answer pairs covering complex, high-risk driving scenarios. The dataset includes multiple-choice and open-ended formats across both image and video modalities. Experiments reveal that existing MLLMs underperform in safety-critical scenarios compared to normal scenes, but fine-tuning with WaymoQA significantly improves their reasoning ability, highlighting the effectiveness of our dataset in developing safer and more reasoning-capable driving agents. 

---
# Energy Costs and Neural Complexity Evolution in Changing Environments 

**Authors**: Sian Heesom-Green, Jonathan Shock, Geoff Nitschke  

**Link**: [PDF](https://arxiv.org/pdf/2511.20018)  

**Abstract**: The Cognitive Buffer Hypothesis (CBH) posits that larger brains evolved to enhance survival in changing conditions. However, larger brains also carry higher energy demands, imposing additional metabolic burdens. Alongside brain size, brain organization plays a key role in cognitive ability and, with suitable architectures, may help mitigate energy challenges. This study evolves Artificial Neural Networks (ANNs) used by Reinforcement Learning (RL) agents to investigate how environmental variability and energy costs influence the evolution of neural complexity, defined in terms of ANN size and structure. Results indicate that under energy constraints, increasing seasonality led to smaller ANNs. This challenges CBH and supports the Expensive Brain Hypothesis (EBH), as highly seasonal environments reduced net energy intake and thereby constrained brain size. ANN structural complexity primarily emerged as a byproduct of size, where energy costs promoted the evolution of more efficient networks. These results highlight the role of energy constraints in shaping neural complexity, offering in silico support for biological theory and energy-efficient robotic design. 

---
# Multi-Context Fusion Transformer for Pedestrian Crossing Intention Prediction in Urban Environments 

**Authors**: Yuanzhe Li, Hang Zhong, Steffen Müller  

**Link**: [PDF](https://arxiv.org/pdf/2511.20011)  

**Abstract**: Pedestrian crossing intention prediction is essential for autonomous vehicles to improve pedestrian safety and reduce traffic accidents. However, accurate pedestrian intention prediction in urban environments remains challenging due to the multitude of factors affecting pedestrian behavior. In this paper, we propose a multi-context fusion Transformer (MFT) that leverages diverse numerical contextual attributes across four key dimensions, encompassing pedestrian behavior context, environmental context, pedestrian localization context and vehicle motion context, to enable accurate pedestrian intention prediction. MFT employs a progressive fusion strategy, where mutual intra-context attention enables reciprocal interactions within each context, thereby facilitating feature sequence fusion and yielding a context token as a context-specific representation. This is followed by mutual cross-context attention, which integrates features across contexts with a global CLS token serving as a compact multi-context representation. Finally, guided intra-context attention refines context tokens within each context through directed interactions, while guided cross-context attention strengthens the global CLS token to promote multi-context fusion via guided information propagation, yielding deeper and more efficient integration. Experimental results validate the superiority of MFT over state-of-the-art methods, achieving accuracy rates of 73%, 93%, and 90% on the JAADbeh, JAADall, and PIE datasets, respectively. Extensive ablation studies are further conducted to investigate the effectiveness of the network architecture and contribution of different input context. Our code is open-source: this https URL. 

---
# Pedestrian Crossing Intention Prediction Using Multimodal Fusion Network 

**Authors**: Yuanzhe Li, Steffen Müller  

**Link**: [PDF](https://arxiv.org/pdf/2511.20008)  

**Abstract**: Pedestrian crossing intention prediction is essential for the deployment of autonomous vehicles (AVs) in urban environments. Ideal prediction provides AVs with critical environmental cues, thereby reducing the risk of pedestrian-related collisions. However, the prediction task is challenging due to the diverse nature of pedestrian behavior and its dependence on multiple contextual factors. This paper proposes a multimodal fusion network that leverages seven modality features from both visual and motion branches, aiming to effectively extract and integrate complementary cues across different modalities. Specifically, motion and visual features are extracted from the raw inputs using multiple Transformer-based extraction modules. Depth-guided attention module leverages depth information to guide attention towards salient regions in another modality through comprehensive spatial feature interactions. To account for the varying importance of different modalities and frames, modality attention and temporal attention are designed to selectively emphasize informative modalities and effectively capture temporal dependencies. Extensive experiments on the JAAD dataset validate the effectiveness of the proposed network, achieving superior performance compared to the baseline methods. 

---
# BERT-APC: A Reference-free Framework for Automatic Pitch Correction via Musical Context Inference 

**Authors**: Sungjae Kim, Kihyun Na, Jinyoung Choi, Injung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.20006)  

**Abstract**: Automatic Pitch Correction (APC) enhances vocal recordings by aligning pitch deviations with the intended musical notes. However, existing APC systems either rely on reference pitches, which limits their practical applicability, or employ simple pitch estimation algorithms that often fail to preserve expressiveness and naturalness. We propose BERT-APC, a novel reference-free APC framework that corrects pitch errors while maintaining the natural expressiveness of vocal performances. In BERT-APC, a novel stationary pitch predictor first estimates the perceived pitch of each note from the detuned singing voice. A context-aware note pitch predictor estimates the intended pitch sequence by leveraging a music language model repurposed to incorporate musical context. Finally, a note-level correction algorithm fixes pitch errors while preserving intentional pitch deviations for emotional expression. In addition, we introduce a learnable data augmentation strategy that improves the robustness of the music language model by simulating realistic detuning patterns. Compared to two recent singing voice transcription models, BERT-APC demonstrated superior performance in note pitch prediction, outperforming the second-best model, ROSVOT, by 10.49%p on highly detuned samples in terms of the raw pitch accuracy. In the MOS test, BERT-APC achieved the highest score of $4.32 \pm 0.15$, which is significantly higher than those of the widely-used commercial APC tools, AutoTune ($3.22 \pm 0.18$) and Melodyne ($3.08 \pm 0.18$), while maintaining a comparable ability to preserve expressive nuances. To the best of our knowledge, this is the first APC model that leverages a music language model to achieve reference-free pitch correction with symbolic musical context. The corrected audio samples of BERT-APC are available online. 

---
# Zero-Shot Transfer Capabilities of the Sundial Foundation Model for Leaf Area Index Forecasting 

**Authors**: Peining Zhang, Hongchen Qin, Haochen Zhang, Ziqi Guo, Guiling Wang, Jinbo Bi  

**Link**: [PDF](https://arxiv.org/pdf/2511.20004)  

**Abstract**: This work investigates the zero-shot forecasting capability of time-series foundation models for Leaf Area Index (LAI) forecasting in agricultural monitoring. Using the HiQ dataset (U.S., 2000-2022), we systematically compare statistical baselines, a fully supervised LSTM, and the Sundial foundation model under multiple evaluation protocols. We find that Sundial, in the zero-shot setting, can outperform a fully trained LSTM provided that the input context window is sufficiently long-specifically, when covering more than one or two full seasonal cycles. This demonstrates, for the first time, that a general-purpose foundation model can surpass specialized supervised models on remote-sensing time series prediction without any task-specific tuning. These results highlight the strong potential of pretrained time-series foundation models to serve as effective plug-and-play forecasters in agricultural and environmental applications. 

---
# On the Feasibility of Hijacking MLLMs' Decision Chain via One Perturbation 

**Authors**: Changyue Li, Jiaying Li, Youliang Yuan, Jiaming He, Zhicong Huang, Pinjia He  

**Link**: [PDF](https://arxiv.org/pdf/2511.20002)  

**Abstract**: Conventional adversarial attacks focus on manipulating a single decision of neural networks. However, real-world models often operate in a sequence of decisions, where an isolated mistake can be easily corrected, but cascading errors can lead to severe risks.
This paper reveals a novel threat: a single perturbation can hijack the whole decision chain. We demonstrate the feasibility of manipulating a model's outputs toward multiple, predefined outcomes, such as simultaneously misclassifying "non-motorized lane" signs as "motorized lane" and "pedestrian" as "plastic bag".
To expose this threat, we introduce Semantic-Aware Universal Perturbations (SAUPs), which induce varied outcomes based on the semantics of the inputs. We overcome optimization challenges by developing an effective algorithm, which searches for perturbations in normalized space with a semantic separation strategy. To evaluate the practical threat of SAUPs, we present RIST, a new real-world image dataset with fine-grained semantic annotations. Extensive experiments on three multimodal large language models demonstrate their vulnerability, achieving a 70% attack success rate when controlling five distinct targets using just an adversarial frame. 

---
# Popularity Bias Alignment Estimates 

**Authors**: Anton Lyubinin  

**Link**: [PDF](https://arxiv.org/pdf/2511.19999)  

**Abstract**: We are extending Popularity Bias Memorization theorem from arXiv:archive/2404.12008 in several directions. We extend it to arbitrary degree distributions and also prove both upper and lower estimates for the alignment with top-k singular hyperspace. 

---
# Directional Optimization Asymmetry in Transformers: A Synthetic Stress Test 

**Authors**: Mihir Sahasrabudhe  

**Link**: [PDF](https://arxiv.org/pdf/2511.19997)  

**Abstract**: Transformers are theoretically reversal-invariant: their function class does not prefer left-to-right over right-to-left mappings. Yet empirical studies on natural language repeatedly report a "reversal curse," and recent work on temporal asymmetry in LLMs suggests that real-world corpora carry their own arrow of time. This leaves an unresolved question: do directional failures stem from linguistic statistics, or from the architecture itself? We cut through this ambiguity with a fully synthetic, entropy-controlled benchmark designed as a clean-room stress test for directional learning. Using random string mappings with tunable branching factor K, we construct forward tasks with zero conditional entropy and inverse tasks with analytically determined entropy floors. Excess loss above these floors reveals that even scratch-trained GPT-2 models exhibit a strong, reproducible directional optimization gap (e.g., 1.16 nats at K=5), far larger than that of an MLP trained on the same data. Pre-trained initializations shift optimization behavior but do not eliminate this gap, while LoRA encounters a sharp capacity wall on high-entropy inverse mappings. Together, these results isolate a minimal, semantics-free signature of directional friction intrinsic to causal Transformer training-one that persists even when linguistic priors, token frequencies, and corpus-level temporal asymmetries are removed. Our benchmark provides a controlled instrument for dissecting directional biases in modern sequence models and motivates deeper mechanistic study of why inversion remains fundamentally harder for Transformers. 

---
# On-Demand Multi-Task Sparsity for Efficient Large-Model Deployment on Edge Devices 

**Authors**: Lianming Huang, Haibo Hu, Qiao Li, Nan Guan, Chun Jason Xue  

**Link**: [PDF](https://arxiv.org/pdf/2511.19986)  

**Abstract**: Sparsity is essential for deploying large models on resource constrained edge platforms. However, optimizing sparsity patterns for individual tasks in isolation ignores the significant I/O overhead incurred during frequent task switching. We introduce an on-demand multi-task sparsity framework specifically designed to minimize switching costs by maximizing parameter reuse. Unlike monolithic approaches, we decompose weights into reusable block-granular units and align sparse structures across tasks to maximize overlap. By dynamically loading only the small differential set of blocks required for the next task, our method effectively mitigates the cold-start latency inherent in traditional monolithic this http URL on a real-world autonomous driving platform demonstrate that our framework achieves superior switching efficiency, accelerating task switching by over 6.6X on average compared to existing sparsity methods. 

---
# EmoFeedback2: Reinforcement of Continuous Emotional Image Generation via LVLM-based Reward and Textual Feedback 

**Authors**: Jingyang Jia, Kai Shu, Gang Yang, Long Xing, Xun Chen, Aiping Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.19982)  

**Abstract**: Continuous emotional image generation (C-EICG) is emerging rapidly due to its ability to produce images aligned with both user descriptions and continuous emotional values. However, existing approaches lack emotional feedback from generated images, limiting the control of emotional continuity. Additionally, their simple alignment between emotions and naively generated texts fails to adaptively adjust emotional prompts according to image content, leading to insufficient emotional fidelity. To address these concerns, we propose a novel generation-understanding-feedback reinforcement paradigm (EmoFeedback2) for C-EICG, which exploits the reasoning capability of the fine-tuned large vision-language model (LVLM) to provide reward and textual feedback for generating high-quality images with continuous emotions. Specifically, we introduce an emotion-aware reward feedback strategy, where the LVLM evaluates the emotional values of generated images and computes the reward against target emotions, guiding the reinforcement fine-tuning of the generative model and enhancing the emotional continuity of images. Furthermore, we design a self-promotion textual feedback framework, in which the LVLM iteratively analyzes the emotional content of generated images and adaptively produces refinement suggestions for the next-round prompt, improving the emotional fidelity with fine-grained content. Extensive experimental results demonstrate that our approach effectively generates high-quality images with the desired emotions, outperforming existing state-of-the-art methods in our custom dataset. The code and dataset will be released soon. 

---
# MambaEye: A Size-Agnostic Visual Encoder with Causal Sequential Processing 

**Authors**: Changho Choi, Minho Kim, Jinkyu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.19963)  

**Abstract**: Despite decades of progress, a truly input-size agnostic visual encoder-a fundamental characteristic of human vision-has remained elusive. We address this limitation by proposing \textbf{MambaEye}, a novel, causal sequential encoder that leverages the low complexity and causal-process based pure Mamba2 backbone. Unlike previous Mamba-based vision encoders that often employ bidirectional processing, our strictly unidirectional approach preserves the inherent causality of State Space Models, enabling the model to generate a prediction at any point in its input sequence. A core innovation is our use of relative move embedding, which encodes the spatial shift between consecutive patches, providing a strong inductive bias for translation invariance and making the model inherently adaptable to arbitrary image resolutions and scanning patterns. To achieve this, we introduce a novel diffusion-inspired loss function that provides dense, step-wise supervision, training the model to build confidence as it gathers more visual evidence. We demonstrate that MambaEye exhibits robust performance across a wide range of image resolutions, especially at higher resolutions such as $1536^2$ on the ImageNet-1K classification task. This feat is achieved while maintaining linear time and memory complexity relative to the number of patches. 

---
# AI/ML based Joint Source and Channel Coding for HARQ-ACK Payload 

**Authors**: Akash Doshi, Pinar Sen, Kirill Ivanov, Wei Yang, June Namgoong, Runxin Wang, Rachel Wang, Taesang Yoo, Jing Jiang, Tingfang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2511.19943)  

**Abstract**: Channel coding from 2G to 5G has assumed the inputs bits at the physical layer to be uniformly distributed. However, hybrid automatic repeat request acknowledgement (HARQ-ACK) bits transmitted in the uplink are inherently non-uniformly distributed. For such sources, significant performance gains could be obtained by employing joint source channel coding, aided by deep learning-based techniques. In this paper, we learn a transformer-based encoder using a novel "free-lunch" training algorithm and propose per-codeword power shaping to exploit the source prior at the encoder whilst being robust to small changes in the HARQ-ACK distribution. Furthermore, any HARQ-ACK decoder has to achieve a low negative acknowledgement (NACK) error rate to avoid radio link failures resulting from multiple NACK errors. We develop an extension of the Neyman-Pearson test to a coded bit system with multiple information bits to achieve Unequal Error Protection of NACK over ACK bits at the decoder. Finally, we apply the proposed encoder and decoder designs to a 5G New Radio (NR) compliant uplink setup under a fading channel, describing the optimal receiver design and a low complexity coherent approximation to it. Our results demonstrate 3-6 dB reduction in the average transmit power required to achieve the target error rates compared to the NR baseline, while also achieving a 2-3 dB reduction in the maximum transmit power, thus providing for significant coverage gains and power savings. 

---
# Optimize Flip Angle Schedules In MR Fingerprinting Using Reinforcement Learning 

**Authors**: Shenjun Zhong, Zhifeng Chen, Zhaolin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.19941)  

**Abstract**: Magnetic Resonance Fingerprinting (MRF) leverages transient-state signal dynamics generated by the tunable acquisition parameters, making the design of an optimal, robust sequence a complex, high-dimensional sequential decision problem, such as optimizing one of the key parameters, flip angle. Reinforcement learning (RL) offers a promising approach to automate parameter selection, to optimize pulse sequences that maximize the distinguishability of fingerprints across the parameter space. In this work, we introduce an RL framework for optimizing the flip-angle schedule in MRF and demonstrate a learned schedule exhibiting non-periodic patterns that enhances fingerprint separability. Additionally, an interesting observation is that the RL-optimized schedule may enable a reduction in the number of repetition time, potentially accelerate MRF acquisitions. 

---
# LLM-EDT: Large Language Model Enhanced Cross-domain Sequential Recommendation with Dual-phase Training 

**Authors**: Ziwei Liu, Qidong Liu, Wanyu Wang, Yejing Wang, Tong Xu, Wei Huang, Chong Chen, Peng Chuan, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.19931)  

**Abstract**: Cross-domain Sequential Recommendation (CDSR) has been proposed to enrich user-item interactions by incorporating information from various domains. Despite current progress, the imbalance issue and transition issue hinder further development of CDSR. The former one presents a phenomenon that the interactions in one domain dominate the entire behavior, leading to difficulty in capturing the domain-specific features in the other domain. The latter points to the difficulty in capturing users' cross-domain preferences within the mixed interaction sequence, resulting in poor next-item prediction performance for specific domains. With world knowledge and powerful reasoning ability, Large Language Models (LLMs) partially alleviate the above issues by performing as a generator and an encoder. However, current LLMs-enhanced CDSR methods are still under exploration, which fail to recognize the irrelevant noise and rough profiling problems. Thus, to make peace with the aforementioned challenges, we proposed an LLMs Enhanced Cross-domain Sequential Recommendation with Dual-phase Training ({LLM-EDT}). To address the imbalance issue while introducing less irrelevant noise, we first propose the transferable item augmenter to adaptively generate possible cross-domain behaviors for users. Then, to alleviate the transition issue, we introduce a dual-phase training strategy to empower the domain-specific thread with a domain-shared background. As for the rough profiling problem, we devise a domain-aware profiling module to summarize the user's preference in each domain and adaptively aggregate them to generate comprehensive user profiles. The experiments on three public datasets validate the effectiveness of our proposed LLM-EDT. To ease reproducibility, we have released the detailed code online at {this https URL}. 

---
# Zero-Knowledge Proof Based Verifiable Inference of Models 

**Authors**: Yunxiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.19902)  

**Abstract**: Recent advances in artificial intelligence (AI), particularly deep learning, have led to widespread adoption across various applications. Yet, a fundamental challenge persists: how can we verify the correctness of AI model inference when model owners cannot (or will not) reveal their parameters? These parameters represent enormous training costs and valuable intellectual property, making transparent verification difficult. In this paper, we introduce a zero-knowledge framework capable of verifying deep learning inference without exposing model internal parameters. Built on recursively composed zero-knowledge proofs and requiring no trusted setup, our framework supports both linear and nonlinear neural network layers, including matrix multiplication, normalization, softmax, and SiLU. Leveraging the Fiat-Shamir heuristic, we obtain a succinct non-interactive argument of knowledge (zkSNARK) with constant-size proofs. To demonstrate the practicality of our approach, we translate the DeepSeek model into a fully SNARK-verifiable version named ZK-DeepSeek and show experimentally that our framework delivers both efficiency and flexibility in real-world AI verification workloads. 

---
# Agent0-VL: Exploring Self-Evolving Agent for Tool-Integrated Vision-Language Reasoning 

**Authors**: Jiaqi Liu, Kaiwen Xiong, Peng Xia, Yiyang Zhou, Haonian Ji, Lu Feng, Siwei Han, Mingyu Ding, Huaxiu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2511.19900)  

**Abstract**: Vision-language agents have achieved remarkable progress in a variety of multimodal reasoning tasks; however, their learning remains constrained by the limitations of human-annotated supervision. Recent self-rewarding approaches attempt to overcome this constraint by allowing models to act as their own critics or reward providers. Yet, purely text-based self-evaluation struggles to verify complex visual reasoning steps and often suffers from evaluation hallucinations. To address these challenges, inspired by recent advances in tool-integrated reasoning, we propose Agent0-VL, a self-evolving vision-language agent that achieves continual improvement with tool-integrated reasoning. Agent0-VL incorporates tool usage not only into reasoning but also into self-evaluation and self-repair, enabling the model to introspect, verify, and refine its reasoning through evidence-grounded analysis. It unifies two synergistic roles within a single LVLM: a Solver that performs multi-turn tool-integrated reasoning, and a Verifier that generates structured feedback and fine-grained self-rewards through tool-grounded critique. These roles interact through a Self-Evolving Reasoning Cycle, where tool-based verification and reinforcement learning jointly align the reasoning and evaluation distributions for stable self-improvement. Through this zero-external-reward evolution, Agent0-VL aligns its reasoning and verification behaviors without any human annotation or external reward models, achieving continual self-improvement. Experiments on geometric problem solving and visual scientific analysis show that Agent0-VL achieves an 12.5% improvement over the base model. Our code is available at \href{this https URL}{this https URL}. 

---
# Distilling Cross-Modal Knowledge via Feature Disentanglement 

**Authors**: Junhong Liu, Yuan Zhang, Tao Huang, Wenchao Xu, Renyu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.19887)  

**Abstract**: Knowledge distillation (KD) has proven highly effective for compressing large models and enhancing the performance of smaller ones. However, its effectiveness diminishes in cross-modal scenarios, such as vision-to-language distillation, where inconsistencies in representation across modalities lead to difficult knowledge transfer. To address this challenge, we propose frequency-decoupled cross-modal knowledge distillation, a method designed to decouple and balance knowledge transfer across modalities by leveraging frequency-domain features. We observed that low-frequency features exhibit high consistency across different modalities, whereas high-frequency features demonstrate extremely low cross-modal similarity. Accordingly, we apply distinct losses to these features: enforcing strong alignment in the low-frequency domain and introducing relaxed alignment for high-frequency features. We also propose a scale consistency loss to address distributional shifts between modalities, and employ a shared classifier to unify feature spaces. Extensive experiments across multiple benchmark datasets show our method substantially outperforms traditional KD and state-of-the-art cross-modal KD approaches. Code is available at this https URL. 

---
# MAPS: Preserving Vision-Language Representations via Module-Wise Proximity Scheduling for Better Vision-Language-Action Generalization 

**Authors**: Chengyue Huang, Mellon M. Zhang, Robert Azarcon, Glen Chou, Zsolt Kira  

**Link**: [PDF](https://arxiv.org/pdf/2511.19878)  

**Abstract**: Vision-Language-Action (VLA) models inherit strong priors from pretrained Vision-Language Models (VLMs), but naive fine-tuning often disrupts these representations and harms generalization. Existing fixes -- freezing modules or applying uniform regularization -- either overconstrain adaptation or ignore the differing roles of VLA components. We present MAPS (Module-Wise Proximity Scheduling), the first robust fine-tuning framework for VLAs. Through systematic analysis, we uncover an empirical order in which proximity constraints should be relaxed to balance stability and flexibility. MAPS linearly schedules this relaxation, enabling visual encoders to stay close to their pretrained priors while action-oriented language layers adapt more freely. MAPS introduces no additional parameters or data, and can be seamlessly integrated into existing VLAs. Across MiniVLA-VQ, MiniVLA-OFT, OpenVLA-OFT, and challenging benchmarks such as SimplerEnv, CALVIN, LIBERO, as well as real-world evaluations on the Franka Emika Panda platform, MAPS consistently boosts both in-distribution and out-of-distribution performance (up to +30%). Our findings highlight empirically guided proximity to pretrained VLMs as a simple yet powerful principle for preserving broad generalization in VLM-to-VLA transfer. 

---
# CodeFuse-CommitEval: Towards Benchmarking LLM's Power on Commit Message and Code Change Inconsistency Detection 

**Authors**: Qingyu Zhang, Puzhuo Liu, Peng Di, Chenxiong Qian  

**Link**: [PDF](https://arxiv.org/pdf/2511.19875)  

**Abstract**: Version control relies on commit messages to convey the rationale for code changes, but these messages are often low quality and, more critically, inconsistent with their diffs-known as message-code inconsistency (MCI). MCIs mislead reviewers, hinder maintenance, contaminate research datasets, and may obscure security patches. Yet, no dedicated benchmark exists to evaluate models for MCI detection. We introduce CODEFUSE-COMMITEVAL, the first benchmark designed for MCI detection using large language models (LLMs). Built on the ApacheCM dataset for diversity and quality, we generate seven types of inconsistent messages through rule-guided mutations of originally consistent commits and apply two-fold validation to verify both positive and negative samples. Using this labeled dataset of message-diff pairs, we evaluate six state-of-the-art open-source LLMs under a vanilla setting and with three augmentation strategies: few-shot prompting, chain-of-thought, and extended context. Results show models detect inconsistent commits more reliably than consistent ones (average Recall 85.95%, Precision 80.28%, Specificity 63.8%); gpt-oss-20B performs best overall but uses over twice the tokens of others. Augmentation effects vary: adjacent context helps larger models but adds noise for smaller ones; few-shot improves accuracy and reduces token use, yet increases universally incorrect predictions; chain-of-thought boosts precision and specificity at the cost of recall and higher token consumption. Type-wise analysis reveals higher detectability for component, file-path, and operation inconsistencies, but lower accuracy and higher token cost for intent-level "purpose" inconsistencies. CODEFUSE-COMMITEVAL provides a rigorous foundation for measuring, comparing, and advancing MCI detection, highlighting the need for richer context and balanced data to capture high-level semantic gaps. 

---
# Cross-LLM Generalization of Behavioral Backdoor Detection in AI Agent Supply Chains 

**Authors**: Arun Chowdary Sanna  

**Link**: [PDF](https://arxiv.org/pdf/2511.19874)  

**Abstract**: As AI agents become integral to enterprise workflows, their reliance on shared tool libraries and pre-trained components creates significant supply chain vulnerabilities. While previous work has demonstrated behavioral backdoor detection within individual LLM architectures, the critical question of cross-LLM generalization remains unexplored, a gap with serious implications for organizations deploying multiple AI systems. We present the first systematic study of cross-LLM behavioral backdoor detection, evaluating generalization across six production LLMs (GPT-5.1, Claude Sonnet 4.5, Grok 4.1, Llama 4 Maverick, GPT-OSS 120B, and DeepSeek Chat V3.1). Through 1,198 execution traces and 36 cross-model experiments, we quantify a critical finding: single-model detectors achieve 92.7% accuracy within their training distribution but only 49.2% across different LLMs, a 43.4 percentage point generalization gap equivalent to random guessing. Our analysis reveals that this gap stems from model-specific behavioral signatures, particularly in temporal features (coefficient of variation > 0.8), while structural features remain stable across architectures. We show that model-aware detection incorporating model identity as an additional feature achieves 90.6% accuracy universally across all evaluated models. We release our multi-LLM trace dataset and detection framework to enable reproducible research. 

---
# A Systematic Analysis of Large Language Models with RAG-enabled Dynamic Prompting for Medical Error Detection and Correction 

**Authors**: Farzad Ahmed, Joniel Augustine Jerome, Meliha Yetisgen, Özlem Uzuner  

**Link**: [PDF](https://arxiv.org/pdf/2511.19858)  

**Abstract**: Objective: Clinical documentation contains factual, diagnostic, and management errors that can compromise patient safety. Large language models (LLMs) may help detect and correct such errors, but their behavior under different prompting strategies remains unclear. We evaluate zero-shot prompting, static prompting with random exemplars (SPR), and retrieval-augmented dynamic prompting (RDP) for three subtasks of medical error processing: error flag detection, error sentence detection, and error correction.
Methods: Using the MEDEC dataset, we evaluated nine instruction-tuned LLMs (GPT, Claude, Gemini, and OpenAI o-series models). We measured performance using accuracy, recall, false-positive rate (FPR), and an aggregate score of ROUGE-1, BLEURT, and BERTScore for error correction. We also analyzed example outputs to identify failure modes and differences between LLM and clinician reasoning.
Results: Zero-shot prompting showed low recall in both detection tasks, often missing abbreviation-heavy or atypical errors. SPR improved recall but increased FPR. Across all nine LLMs, RDP reduced FPR by about 15 percent, improved recall by 5 to 10 percent in error sentence detection, and generated more contextually accurate corrections.
Conclusion: Across diverse LLMs, RDP outperforms zero-shot and SPR prompting. Using retrieved exemplars improves detection accuracy, reduces false positives, and enhances the reliability of medical error correction. 

---
# Cisco Time Series Model Technical Report 

**Authors**: Liang Gou, Archit Khare, Praneet Pabolu, Prachi Patel, Joseph Ross, Hercy Shen, Yuhan, Song, Jingze Sun, Kristal Curtis, Vedant Dharnidharka, Abhinav Mathur, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.19841)  

**Abstract**: We introduce the Cisco Time Series Model, a univariate zero-shot forecaster. This time series foundation model is the result of a general architectural innovation to a time series model enabling it to accept multiresolution input, applied to a popular decoder-only time series model (TimesFM). The resulting multiresolution decoder-only model is trained on over 300B unique data points, with more than half coming from the observability domain. Quantitative and qualitative evaluations demonstrate that the resulting model achieves superior performance on observability datasets while retaining very similar performance on a standard general-purpose forecasting benchmark (GIFT-Eval), and suggest that the multiresolution structure enables the model to make more accurate predictions on long context input. 

---
# GED-Consistent Disentanglement of Aligned and Unaligned Substructures for Graph Similarity Learning 

**Authors**: Zhentao Zhan, Xiaoliang Xu, Jingjing Wang, Junmei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.19837)  

**Abstract**: Graph Similarity Computation (GSC) is a fundamental graph related task where Graph Edit Distance (GED) serves as a prevalent metric. GED is determined by an optimal alignment between a pair of graphs that partitions each into aligned (zero-cost) and unaligned (cost-incurring) substructures. Due to NP-hard nature of exact GED computation, GED approximations based on Graph Neural Network(GNN) have emerged. Existing GNN-based GED approaches typically learn node embeddings for each graph and then aggregate pairwise node similarities to estimate the final similarity. Despite their effectiveness, we identify a mismatch between this prevalent node-centric matching paradigm and the core principles of GED. This discrepancy leads to two critical limitations: (1) a failure to capture the global structural correspondence for optimal alignment, and (2) a misattribution of edit costs driven by spurious node level signals. To address these limitations, we propose GCGSim, a GED-consistent graph similarity learning framework centering on graph-level matching and substructure-level edit costs. Specifically, we make three core technical contributions. Extensive experiments on four benchmark datasets show that GCGSim achieves state-of-the-art performance. Our comprehensive analyses further validate that the framework effectively learns disentangled and semantically meaningful substructure representations. 

---
# Rectified SpaAttn: Revisiting Attention Sparsity for Efficient Video Generation 

**Authors**: Xuewen Liu, Zhikai Li, Jing Zhang, Mengjuan Chen, Qingyi Gu  

**Link**: [PDF](https://arxiv.org/pdf/2511.19835)  

**Abstract**: Diffusion Transformers dominate video generation, but the quadratic complexity of attention computation introduces substantial latency. Attention sparsity reduces computational costs by focusing on critical tokens while ignoring non-critical tokens. However, existing methods suffer from severe performance degradation. In this paper, we revisit attention sparsity and reveal that existing methods induce systematic biases in attention allocation: (1) excessive focus on critical tokens amplifies their attention weights; (2) complete neglect of non-critical tokens causes the loss of relevant attention weights. To address these issues, we propose Rectified SpaAttn, which rectifies attention allocation with implicit full attention reference, thereby enhancing the alignment between sparse and full attention maps. Specifically: (1) for critical tokens, we show that their bias is proportional to the sparse attention weights, with the ratio governed by the amplified weights. Accordingly, we propose Isolated-Pooling Attention Reallocation, which calculates accurate rectification factors by reallocating multimodal pooled weights. (2) for non-critical tokens, recovering attention weights from the pooled query-key yields attention gains but also introduces pooling errors. Therefore, we propose Gain-Aware Pooling Rectification, which ensures that the rectified gain consistently surpasses the induced error. Moreover, we customize and integrate the Rectified SpaAttn kernel using Triton, achieving up to 3.33 and 2.08 times speedups on HunyuanVideo and Wan 2.1, respectively, while maintaining high generation quality. We release Rectified SpaAttn as open-source at this https URL . 

---
# Beyond Relational: Semantic-Aware Multi-Modal Analytics with LLM-Native Query Optimization 

**Authors**: Junhao Zhu, Lu Chen, Xiangyu Ke, Ziquan Fang, Tianyi Li, Yunjun Gao, Christian S. Jensen  

**Link**: [PDF](https://arxiv.org/pdf/2511.19830)  

**Abstract**: Multi-modal analytical processing has the potential to transform applications in e-commerce, healthcare, entertainment, and beyond. However, real-world adoption remains elusive due to the limited ability of traditional relational query operators to capture query semantics. The emergence of foundation models, particularly the large language models (LLMs), opens up new opportunities to develop flexible, semantic-aware data analytics systems that transcend the relational paradigm.
We present Nirvana, a multi-modal data analytics framework that incorporates programmable semantic operators while leveraging both logical and physical query optimization strategies, tailored for LLM-driven semantic query processing. Nirvana addresses two key challenges. First, it features an agentic logical optimizer that uses natural language-specified transformation rules and random-walk-based search to explore vast spaces of semantically equivalent query plans -- far beyond the capabilities of conventional optimizers. Second, it introduces a cost-aware physical optimizer that selects the most effective LLM backend for each operator using a novel improvement-score metric. To further enhance efficiency, Nirvana incorporates computation reuse and evaluation pushdown techniques guided by model capability hypotheses. Experimental evaluations on three real-world benchmarks demonstrate that Nirvana is able to reduce end-to-end runtime by 10%--85% and reduces system processing costs by 76% on average, outperforming state-of-the-art systems at both efficiency and scalability. 

---
# Mosaic Pruning: A Hierarchical Framework for Generalizable Pruning of Mixture-of-Experts Models 

**Authors**: Wentao Hu, Mingkuan Zhao, Shuangyong Song, Xiaoyan Zhu, Xin Lai, Jiayin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.19822)  

**Abstract**: Sparse Mixture-of-Experts (SMoE) architectures have enabled a new frontier in scaling Large Language Models (LLMs), offering superior performance by activating only a fraction of their total parameters during inference. However, their practical deployment is severely hampered by substantial static memory overhead, as all experts must be loaded into memory. Existing post-training pruning methods, while reducing model size, often derive their pruning criteria from a single, general-purpose corpus. This leads to a critical limitation: a catastrophic performance degradation when the pruned model is applied to other domains, necessitating a costly re-pruning for each new domain. To address this generalization gap, we introduce Mosaic Pruning (MoP). The core idea of MoP is to construct a functionally comprehensive set of experts through a structured ``cluster-then-select" process. This process leverages a similarity metric that captures expert performance across different task domains to functionally cluster the experts, and subsequently selects the most representative expert from each cluster based on our proposed Activation Variability Score. Unlike methods that optimize for a single corpus, our proposed Mosaic Pruning ensures that the pruned model retains a functionally complementary set of experts, much like the tiles of a mosaic that together form a complete picture of the original model's capabilities, enabling it to handle diverse downstream this http URL experiments on various MoE models demonstrate the superiority of our approach. MoP significantly outperforms prior work, achieving a 7.24\% gain on general tasks and 8.92\% on specialized tasks like math reasoning and code generation. 

---
# CropVLM: Learning to Zoom for Fine-Grained Vision-Language Perception 

**Authors**: Miguel Carvalho, Helder Dias, Bruno Martins  

**Link**: [PDF](https://arxiv.org/pdf/2511.19820)  

**Abstract**: Vision-Language Models (VLMs) often struggle with tasks that require fine-grained image understanding, such as scene-text recognition or document analysis, due to perception limitations and visual fragmentation. To address these challenges, we introduce CropVLM as an external low-cost method for boosting performance, enabling VLMs to dynamically ''zoom in'' on relevant image regions, enhancing their ability to capture fine details. CropVLM is trained using reinforcement learning, without using human-labeled bounding boxes as a supervision signal, and without expensive synthetic evaluations. The model is trained once and can be paired with both open-source and proprietary VLMs to improve their performance. Our approach delivers significant improvements on tasks that require high-resolution image understanding, notably for benchmarks that are out-of-domain for the target VLM, without modifying or fine-tuning the VLM, thus avoiding catastrophic forgetting. 

---
# Language-Independent Sentiment Labelling with Distant Supervision: A Case Study for English, Sepedi and Setswana 

**Authors**: Koena Ronny Mabokela, Tim Schlippe, Mpho Raborife, Turgay Celik  

**Link**: [PDF](https://arxiv.org/pdf/2511.19818)  

**Abstract**: Sentiment analysis is a helpful task to automatically analyse opinions and emotions on various topics in areas such as AI for Social Good, AI in Education or marketing. While many of the sentiment analysis systems are developed for English, many African languages are classified as low-resource languages due to the lack of digital language resources like text labelled with corresponding sentiment classes. One reason for that is that manually labelling text data is time-consuming and expensive. Consequently, automatic and rapid processes are needed to reduce the manual effort as much as possible making the labelling process as efficient as possible. In this paper, we present and analyze an automatic language-independent sentiment labelling method that leverages information from sentiment-bearing emojis and words. Our experiments are conducted with tweets in the languages English, Sepedi and Setswana from SAfriSenti, a multilingual sentiment corpus for South African languages. We show that our sentiment labelling approach is able to label the English tweets with an accuracy of 66%, the Sepedi tweets with 69%, and the Setswana tweets with 63%, so that on average only 34% of the automatically generated labels remain to be corrected. 

---
# Learning to Clean: Reinforcement Learning for Noisy Label Correction 

**Authors**: Marzi Heidari, Hanping Zhang, Yuhong Guo  

**Link**: [PDF](https://arxiv.org/pdf/2511.19808)  

**Abstract**: The challenge of learning with noisy labels is significant in machine learning, as it can severely degrade the performance of prediction models if not addressed properly. This paper introduces a novel framework that conceptualizes noisy label correction as a reinforcement learning (RL) problem. The proposed approach, Reinforcement Learning for Noisy Label Correction (RLNLC), defines a comprehensive state space representing data and their associated labels, an action space that indicates possible label corrections, and a reward mechanism that evaluates the efficacy of label corrections. RLNLC learns a deep feature representation based policy network to perform label correction through reinforcement learning, utilizing an actor-critic method. The learned policy is subsequently deployed to iteratively correct noisy training labels and facilitate the training of the prediction model. The effectiveness of RLNLC is demonstrated through extensive experiments on multiple benchmark datasets, where it consistently outperforms existing state-of-the-art techniques for learning with noisy labels. 

---
# Terminal Velocity Matching 

**Authors**: Linqi Zhou, Mathias Parger, Ayaan Haque, Jiaming Song  

**Link**: [PDF](https://arxiv.org/pdf/2511.19797)  

**Abstract**: We propose Terminal Velocity Matching (TVM), a generalization of flow matching that enables high-fidelity one- and few-step generative modeling. TVM models the transition between any two diffusion timesteps and regularizes its behavior at its terminal time rather than at the initial time. We prove that TVM provides an upper bound on the $2$-Wasserstein distance between data and model distributions when the model is Lipschitz continuous. However, since Diffusion Transformers lack this property, we introduce minimal architectural changes that achieve stable, single-stage training. To make TVM efficient in practice, we develop a fused attention kernel that supports backward passes on Jacobian-Vector Products, which scale well with transformer architectures. On ImageNet-256x256, TVM achieves 3.29 FID with a single function evaluation (NFE) and 1.99 FID with 4 NFEs. It similarly achieves 4.32 1-NFE FID and 2.94 4-NFE FID on ImageNet-512x512, representing state-of-the-art performance for one/few-step models from scratch. 

---
# Prune-Then-Plan: Step-Level Calibration for Stable Frontier Exploration in Embodied Question Answering 

**Authors**: Noah Frahm, Prakrut Patel, Yue Zhang, Shoubin Yu, Mohit Bansal, Roni Sengupta  

**Link**: [PDF](https://arxiv.org/pdf/2511.19768)  

**Abstract**: Large vision-language models (VLMs) have improved embodied question answering (EQA) agents by providing strong semantic priors for open-vocabulary reasoning. However, when used directly for step-level exploration, VLMs often exhibit frontier oscillations, unstable back-and-forth movements caused by overconfidence and miscalibration, leading to inefficient navigation and degraded answer quality. We propose Prune-Then-Plan, a simple and effective framework that stabilizes exploration through step-level calibration. Instead of trusting raw VLM scores, our method prunes implausible frontier choices using a Holm-Bonferroni inspired pruning procedure and then delegates final decisions to a coverage-based planner. This separation converts overconfident predictions into conservative, interpretable actions by relying on human-level judgments to calibrate the step-level behavior of VLMs. Integrated into the 3D-Mem EQA framework, our approach achieves relative improvements of up to 49% and 33% in visually grounded SPL and LLM-Match metrics respectively over baselines. Overall, our method achieves better scene coverage under equal exploration budgets on both OpenEQA and EXPRESS-Bench datasets. 

---
# Leveraging Foundation Models for Histological Grading in Cutaneous Squamous Cell Carcinoma using PathFMTools 

**Authors**: Abdul Rahman Diab, Emily E. Karn, Renchin Wu, Emily S. Ruiz, William Lotter  

**Link**: [PDF](https://arxiv.org/pdf/2511.19751)  

**Abstract**: Despite the promise of computational pathology foundation models, adapting them to specific clinical tasks remains challenging due to the complexity of whole-slide image (WSI) processing, the opacity of learned features, and the wide range of potential adaptation strategies. To address these challenges, we introduce PathFMTools, a lightweight, extensible Python package that enables efficient execution, analysis, and visualization of pathology foundation models. We use this tool to interface with and evaluate two state-of-the-art vision-language foundation models, CONCH and MUSK, on the task of histological grading in cutaneous squamous cell carcinoma (cSCC), a critical criterion that informs cSCC staging and patient management. Using a cohort of 440 cSCC H&E WSIs, we benchmark multiple adaptation strategies, demonstrating trade-offs across prediction approaches and validating the potential of using foundation model embeddings to train small specialist models. These findings underscore the promise of pathology foundation models for real-world clinical applications, with PathFMTools enabling efficient analysis and validation. 

---
# Prompt Fencing: A Cryptographic Approach to Establishing Security Boundaries in Large Language Model Prompts 

**Authors**: Steven Peh  

**Link**: [PDF](https://arxiv.org/pdf/2511.19727)  

**Abstract**: Large Language Models (LLMs) remain vulnerable to prompt injection attacks, representing the most significant security threat in production deployments. We present Prompt Fencing, a novel architectural approach that applies cryptographic authentication and data architecture principles to establish explicit security boundaries within LLM prompts. Our approach decorates prompt segments with cryptographically signed metadata including trust ratings and content types, enabling LLMs to distinguish between trusted instructions and untrusted content. While current LLMs lack native fence awareness, we demonstrate that simulated awareness through prompt instructions achieved complete prevention of injection attacks in our experiments, reducing success rates from 86.7% (260/300 successful attacks) to 0% (0/300 successful attacks) across 300 test cases with two leading LLM providers. We implement a proof-of-concept fence generation and verification pipeline with a total overhead of 0.224 seconds (0.130s for fence generation, 0.094s for validation) across 100 samples. Our approach is platform-agnostic and can be incrementally deployed as a security layer above existing LLM infrastructure, with the expectation that future models will be trained with native fence awareness for optimal security. 

---
# An Adaptive, Data-Integrated Agent-Based Modeling Framework for Explainable and Contestable Policy Design 

**Authors**: Roberto Garrone  

**Link**: [PDF](https://arxiv.org/pdf/2511.19726)  

**Abstract**: Multi-agent systems often operate under feedback, adaptation, and non-stationarity, yet many simulation studies retain static decision rules and fixed control parameters. This paper introduces a general adaptive multi-agent learning framework that integrates: (i) four dynamic regimes distinguishing static versus adaptive agents and fixed versus adaptive system parameters; (ii) information-theoretic diagnostics (entropy rate, statistical complexity, and predictive information) to assess predictability and structure; (iii) structural causal models for explicit intervention semantics; (iv) procedures for generating agent-level priors from aggregate or sample data; and (v) unsupervised methods for identifying emergent behavioral regimes. The framework offers a domain-neutral architecture for analyzing how learning agents and adaptive controls jointly shape system trajectories, enabling systematic comparison of stability, performance, and interpretability across non-equilibrium, oscillatory, or drifting dynamics. Mathematical definitions, computational operators, and an experimental design template are provided, yielding a structured methodology for developing explainable and contestable multi-agent decision processes. 

---
# CrypTorch: PyTorch-based Auto-tuning Compiler for Machine Learning with Multi-party Computation 

**Authors**: Jinyu Liu, Gang Tan, Kiwan Maeng  

**Link**: [PDF](https://arxiv.org/pdf/2511.19711)  

**Abstract**: Machine learning (ML) involves private data and proprietary model parameters. MPC-based ML allows multiple parties to collaboratively run an ML workload without sharing their private data or model parameters using multi-party computing (MPC). Because MPC cannot natively run ML operations such as Softmax or GELU, existing frameworks use different approximations. Our study shows that, on a well-optimized framework, these approximations often become the dominating bottleneck. Popular approximations are often insufficiently accurate or unnecessarily slow, and these issues are hard to identify and fix in existing frameworks. To tackle this issue, we propose a compiler for MPC-based ML, CrypTorch. CrypTorch disentangles these approximations with the rest of the MPC runtime, allows easily adding new approximations through its programming interface, and automatically selects approximations to maximize both performance and accuracy. Built as an extension to PyTorch 2's compiler, we show that CrypTorch's auto-tuning alone provides 1.20--1.7$\times$ immediate speedup without sacrificing accuracy, and 1.31--1.8$\times$ speedup when some accuracy degradation is allowed, compared to our well-optimized baseline. Combined with better engineering and adoption of state-of-the-art practices, the entire framework brings 3.22--8.6$\times$ end-to-end speedup compared to the popular framework, CrypTen. 

---
# The Alexander-Hirschowitz theorem for neurovarieties 

**Authors**: A. Massarenti, M. Mella  

**Link**: [PDF](https://arxiv.org/pdf/2511.19703)  

**Abstract**: We study neurovarieties for polynomial neural networks and fully characterize when they attain the expected dimension in the single-output case. As consequences, we establish non-defectiveness and global identifiability for multi-output architectures. 

---
# A Layered Protocol Architecture for the Internet of Agents 

**Authors**: Charles Fleming, Vijoy Pandey, Ramana Kompella, Luca Muscariello  

**Link**: [PDF](https://arxiv.org/pdf/2511.19699)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance improvements and the ability to learn domain-specific languages (DSLs), including APIs and tool interfaces. This capability has enabled the creation of AI agents that can perform preliminary computations and act through tool calling, now being standardized via protocols like MCP. However, LLMs face fundamental limitations: their context windows cannot grow indefinitely, constraining their memory and computational capacity. Agent collaboration emerges as essential for solving increasingly complex problems, mirroring how computational systems rely on different types of memory to scale. The "Internet of Agents" (IoA) represents the communication stack that enables agents to scale by distributing computation across collaborating entities.
Current network architectural stacks (OSI and TCP/IP) were designed for data delivery between hosts and processes, not for agent collaboration with semantic understanding. To address this gap, we propose two new layers: an \textbf{Agent Communication Layer (L8)} and an \textbf{Agent Semantic Negotiation Layer (L9)}. L8 formalizes the \textit{structure} of communication, standardizing message envelopes, speech-act performatives (e.g., REQUEST, INFORM), and interaction patterns (e.g., request-reply, publish-subscribe), building on protocols like MCP. L9, which does not exist today, formalizes the \textit{meaning} of communication, enabling agents to discover, negotiate, and lock a "Shared Context" -- a formal schema defining the concepts, tasks, and parameters relevant to their interaction. Together, these layers provide the foundation for scalable, distributed agent collaboration, enabling the next generation of multi-agentic systems. 

---
# TiCT: A Synthetically Pre-Trained Foundation Model for Time Series Classification 

**Authors**: Chin-Chia Michael Yeh, Uday Singh Saini, Junpeng Wang, Xin Dai, Xiran Fan, Jiarui Sun, Yujie Fan, Yan Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2511.19694)  

**Abstract**: The ubiquity of time series data creates a strong demand for general-purpose foundation models, yet developing them for classification remains a significant challenge, largely due to the high cost of labeled data. Foundation models capable of in-context learning (ICL) offer a powerful solution, adapting to new tasks with minimal examples and reducing the need for extensive retraining. However, prior work on large-scale time series models has predominantly focused on forecasting, leaving a critical gap for versatile, fine-tuning-free classification. To address this, we introduce TiCT (Time-series in-Context Transformer), a transformer-based model pre-trained exclusively on synthetic data to perform in-context classification. We make two primary technical contributions: 1) a novel architecture featuring a scalable bit-based label encoding and a special output attention mechanism to handle an arbitrary number of classes; and 2) a synthetic pre-training framework that combines a Mixup-inspired process with data augmentation to foster generalization and noise invariance. Extensive evaluations on the UCR Archive show that TiCT achieves competitive performance against state-of-the-art supervised methods. Crucially, this is accomplished using only in-context examples at inference time, without updating a single model weight. 

---
# TREASURE: A Transformer-Based Foundation Model for High-Volume Transaction Understanding 

**Authors**: Chin-Chia Michael Yeh, Uday Singh Saini, Xin Dai, Xiran Fan, Shubham Jain, Yujie Fan, Jiarui Sun, Junpeng Wang, Menghai Pan, Yingtong Dou, Yuzhong Chen, Vineeth Rakesh, Liang Wang, Yan Zheng, Mahashweta Das  

**Link**: [PDF](https://arxiv.org/pdf/2511.19693)  

**Abstract**: Payment networks form the backbone of modern commerce, generating high volumes of transaction records from daily activities. Properly modeling this data can enable applications such as abnormal behavior detection and consumer-level insights for hyper-personalized experiences, ultimately improving people's lives. In this paper, we present TREASURE, TRansformer Engine As Scalable Universal transaction Representation Encoder, a multipurpose transformer-based foundation model specifically designed for transaction data. The model simultaneously captures both consumer behavior and payment network signals (such as response codes and system flags), providing comprehensive information necessary for applications like accurate recommendation systems and abnormal behavior detection. Verified with industry-grade datasets, TREASURE features three key capabilities: 1) an input module with dedicated sub-modules for static and dynamic attributes, enabling more efficient training and inference; 2) an efficient and effective training paradigm for predicting high-cardinality categorical attributes; and 3) demonstrated effectiveness as both a standalone model that increases abnormal behavior detection performance by 111% over production systems and an embedding provider that enhances recommendation models by 104%. We present key insights from extensive ablation studies, benchmarks against production models, and case studies, highlighting valuable knowledge gained from developing TREASURE. 

---
# IndEgo: A Dataset of Industrial Scenarios and Collaborative Work for Egocentric Assistants 

**Authors**: Vivek Chavan, Yasmina Imgrund, Tung Dao, Sanwantri Bai, Bosong Wang, Ze Lu, Oliver Heimann, Jörg Krüger  

**Link**: [PDF](https://arxiv.org/pdf/2511.19684)  

**Abstract**: We introduce IndEgo, a multimodal egocentric and exocentric dataset addressing common industrial tasks, including assembly/disassembly, logistics and organisation, inspection and repair, woodworking, and others. The dataset contains 3,460 egocentric recordings (approximately 197 hours), along with 1,092 exocentric recordings (approximately 97 hours). A key focus of the dataset is collaborative work, where two workers jointly perform cognitively and physically intensive tasks. The egocentric recordings include rich multimodal data and added context via eye gaze, narration, sound, motion, and others. We provide detailed annotations (actions, summaries, mistake annotations, narrations), metadata, processed outputs (eye gaze, hand pose, semi-dense point cloud), and benchmarks on procedural and non-procedural task understanding, Mistake Detection, and reasoning-based Question Answering. Baseline evaluations for Mistake Detection, Question Answering and collaborative task understanding show that the dataset presents a challenge for the state-of-the-art multimodal models. Our dataset is available at: this https URL 

---
# Accuracy and Efficiency Trade-Offs in LLM-Based Malware Detection and Explanation: A Comparative Study of Parameter Tuning vs. Full Fine-Tuning 

**Authors**: Stephen C. Gravereaux, Sheikh Rabiul Islam  

**Link**: [PDF](https://arxiv.org/pdf/2511.19654)  

**Abstract**: This study examines whether Low-Rank Adaptation (LoRA) fine-tuned Large Language Models (LLMs) can approximate the performance of fully fine-tuned models in generating human-interpretable decisions and explanations for malware classification. Achieving trustworthy malware detection, particularly when LLMs are involved, remains a significant challenge. We developed an evaluation framework using Bilingual Evaluation Understudy (BLEU), Recall-Oriented Understudy for Gisting Evaluation (ROUGE), and Semantic Similarity Metrics to benchmark explanation quality across five LoRA configurations and a fully fine-tuned baseline. Results indicate that full fine-tuning achieves the highest overall scores, with BLEU and ROUGE improvements of up to 10% over LoRA variants. However, mid-range LoRA models deliver competitive performance exceeding full fine-tuning on two metrics while reducing model size by approximately 81% and training time by over 80% on a LoRA model with 15.5% trainable parameters. These findings demonstrate that LoRA offers a practical balance of interpretability and resource efficiency, enabling deployment in resource-constrained environments without sacrificing explanation quality. By providing feature-driven natural language explanations for malware classifications, this approach enhances transparency, analyst confidence, and operational scalability in malware detection systems. 

---
# Synthetic Data: AI's New Weapon Against Android Malware 

**Authors**: Angelo Gaspar Diniz Nogueira, Kayua Oleques Paim, Hendrio Bragança, Rodrigo Brandão Mansilha, Diego Kreutz  

**Link**: [PDF](https://arxiv.org/pdf/2511.19649)  

**Abstract**: The ever-increasing number of Android devices and the accelerated evolution of malware, reaching over 35 million samples by 2024, highlight the critical importance of effective detection methods. Attackers are now using Artificial Intelligence to create sophisticated malware variations that can easily evade traditional detection techniques. Although machine learning has shown promise in malware classification, its success relies heavily on the availability of up-to-date, high-quality datasets. The scarcity and high cost of obtaining and labeling real malware samples presents significant challenges in developing robust detection models. In this paper, we propose MalSynGen, a Malware Synthetic Data Generation methodology that uses a conditional Generative Adversarial Network (cGAN) to generate synthetic tabular data. This data preserves the statistical properties of real-world data and improves the performance of Android malware classifiers. We evaluated the effectiveness of this approach using various datasets and metrics that assess the fidelity of the generated data, its utility in classification, and the computational efficiency of the process. Our experiments demonstrate that MalSynGen can generalize across different datasets, providing a viable solution to address the issues of obsolescence and low quality data in malware detection. 

---
# Efficient Multi-Hop Question Answering over Knowledge Graphs via LLM Planning and Embedding-Guided Search 

**Authors**: Manil Shrestha, Edward Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.19648)  

**Abstract**: Multi-hop question answering over knowledge graphs remains computationally challenging due to the combinatorial explosion of possible reasoning paths. Recent approaches rely on expensive Large Language Model (LLM) inference for both entity linking and path ranking, limiting their practical deployment. Additionally, LLM-generated answers often lack verifiable grounding in structured knowledge. We present two complementary hybrid algorithms that address both efficiency and verifiability: (1) LLM-Guided Planning that uses a single LLM call to predict relation sequences executed via breadth-first search, achieving near-perfect accuracy (micro-F1 > 0.90) while ensuring all answers are grounded in the knowledge graph, and (2) Embedding-Guided Neural Search that eliminates LLM calls entirely by fusing text and graph embeddings through a lightweight 6.7M-parameter edge scorer, achieving over 100 times speedup with competitive accuracy. Through knowledge distillation, we compress planning capability into a 4B-parameter model that matches large-model performance at zero API cost. Evaluation on MetaQA demonstrates that grounded reasoning consistently outperforms ungrounded generation, with structured planning proving more transferable than direct answer generation. Our results show that verifiable multi-hop reasoning does not require massive models at inference time, but rather the right architectural inductive biases combining symbolic structure with learned representations. 

---
# Robot-Powered Data Flywheels: Deploying Robots in the Wild for Continual Data Collection and Foundation Model Adaptation 

**Authors**: Jennifer Grannen, Michelle Pan, Kenneth Llontop, Cherie Ho, Mark Zolotas, Jeannette Bohg, Dorsa Sadigh  

**Link**: [PDF](https://arxiv.org/pdf/2511.19647)  

**Abstract**: Foundation models (FM) have unlocked powerful zero-shot capabilities in vision and language, yet their reliance on internet pretraining data leaves them brittle in unstructured, real-world settings. The messy, real-world data encountered during deployment (e.g. occluded or multilingual text) remains massively underrepresented in existing corpora. Robots, as embodied agents, are uniquely positioned to close this gap: they can act in physical environments to collect large-scale, real-world data that enriches FM training with precisely the examples current models lack. We introduce the Robot-Powered Data Flywheel, a framework that transforms robots from FM consumers into data generators. By deploying robots equipped with FMs in the wild, we enable a virtuous cycle: robots perform useful tasks while collecting real-world data that improves both domain-specific adaptation and domain-adjacent generalization. We instantiate this framework with Scanford, a mobile manipulator deployed in the East Asia Library for 2 weeks. Scanford autonomously scans shelves, identifies books using a vision-language model (VLM), and leverages the library catalog to label images without human annotation. This deployment both aids librarians and produces a dataset to finetune the underlying VLM, improving performance on the domain-specific in-the-wild library setting and on domain-adjacent multilingual OCR benchmarks. Using data collected from 2103 shelves, Scanford improves VLM performance on book identification from 32.0% to 71.8% and boosts domain-adjacent multilingual OCR from 24.8% to 46.6% (English) and 30.8% to 38.0% (Chinese), while saving an ~18.7 hrs of human time. These results highlight how robot-powered data flywheels can both reduce human effort in real deployments and unlock new pathways for continually adapting FMs to the messiness of reality. More details are at: this https URL 

---
# IRSDA: An Agent-Orchestrated Framework for Enterprise Intrusion Response 

**Authors**: Damodar Panigrahi, Raj Patel, Shaswata Mitra, Sudip Mittal, Shahram Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2511.19644)  

**Abstract**: Modern enterprise systems face escalating cyber threats that are increasingly dynamic, distributed, and multi-stage in nature. Traditional intrusion detection and response systems often rely on static rules and manual workflows, which limit their ability to respond with the speed and precision required in high-stakes environments. To address these challenges, we present the Intrusion Response System Digital Assistant (IRSDA), an agent-based framework designed to deliver autonomous and policy-compliant cyber defense. IRSDA combines Self-Adaptive Autonomic Computing Systems (SA-ACS) with the Knowledge guided Monitor, Analyze, Plan, and Execute (MAPE-K) loop to support real-time, partition-aware decision-making across enterprise infrastructure.
IRSDA incorporates a knowledge-driven architecture that integrates contextual information with AI-based reasoning to support system-guided intrusion response. The framework leverages retrieval mechanisms and structured representations to inform decision-making while maintaining alignment with operational policies. We assess the system using a representative real-world microservices application, demonstrating its ability to automate containment, enforce compliance, and provide traceable outputs for security analyst interpretation. This work outlines a modular and agent-driven approach to cyber defense that emphasizes explainability, system-state awareness, and operational control in intrusion response. 

---
# On the Utility of Foundation Models for Fast MRI: Vision-Language-Guided Image Reconstruction 

**Authors**: Ruimin Feng, Xingxin He, Ronald Mercer, Zachary Stewart, Fang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.19641)  

**Abstract**: Purpose: To investigate whether a vision-language foundation model can enhance undersampled MRI reconstruction by providing high-level contextual information beyond conventional priors. Methods: We proposed a semantic distribution-guided reconstruction framework that uses a pre-trained vision-language foundation model to encode both the reconstructed image and auxiliary information into high-level semantic features. A contrastive objective aligns the reconstructed representation with the target semantic distribution, ensuring consistency with high-level perceptual cues. The proposed objective works with various deep learning-based reconstruction methods and can flexibly incorporate semantic priors from multimodal sources. To test the effectiveness of these semantic priors, we evaluated reconstruction results guided by priors derived from either image-only or image-language auxiliary information. Results: Experiments on knee and brain datasets demonstrate that semantic priors from images preserve fine anatomical structures and achieve superior perceptual quality, as reflected in lower LPIPS values, higher Tenengrad scores, and improved scores in the reader study, compared with conventional regularization. The image-language information further expands the semantic distribution and enables high-level control over reconstruction attributes. Across all evaluations, the contrastive objective consistently guided the reconstructed features toward the desired semantic distributions while maintaining data fidelity, demonstrating the effectiveness of the proposed optimization framework. Conclusion: The study highlights that vision-language foundation models can improve undersampled MRI reconstruction through semantic-space optimization. 

---
# Many Ways to be Right: Rashomon Sets for Concept-Based Neural Networks 

**Authors**: Shihan Feng, Cheng Zhang, Michael Xi, Ethan Hsu, Lesia Semenova, Chudi Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2511.19636)  

**Abstract**: Modern neural networks rarely have a single way to be right. For many tasks, multiple models can achieve identical performance while relying on different features or reasoning patterns, a property known as the Rashomon Effect. However, uncovering this diversity in deep architectures is challenging as their continuous parameter spaces contain countless near-optimal solutions that are numerically distinct but often behaviorally similar. We introduce Rashomon Concept Bottleneck Models, a framework that learns multiple neural networks which are all accurate yet reason through distinct human-understandable concepts. By combining lightweight adapter modules with a diversity-regularized training objective, our method constructs a diverse set of deep concept-based models efficiently without retraining from scratch. The resulting networks provide fundamentally different reasoning processes for the same predictions, revealing how concept reliance and decision making vary across equally performing solutions. Our framework enables systematic exploration of data-driven reasoning diversity in deep models, offering a new mechanism for auditing, comparison, and alignment across equally accurate solutions. 

---
# Towards Synergistic Teacher-AI Interactions with Generative Artificial Intelligence 

**Authors**: Mutlu Cukurova, Wannapon Suraworachet, Qi Zhou, Sahan Bulathwela  

**Link**: [PDF](https://arxiv.org/pdf/2511.19580)  

**Abstract**: Generative artificial intelligence (GenAI) is increasingly used in education, posing significant challenges for teachers adapting to these changes. GenAI offers unprecedented opportunities for accessibility, scalability and productivity in educational tasks. However, the automation of teaching tasks through GenAI raises concerns about reduced teacher agency, potential cognitive atrophy, and the broader deprofessionalisation of teaching. Drawing findings from prior literature on AI in Education, and refining through a recent systematic literature review, this chapter presents a conceptualisation of five levels of teacher-AI teaming: transactional, situational, operational, praxical and synergistic teaming. The framework aims to capture the nuanced dynamics of teacher-AI interactions, particularly with GenAI, that may lead to the replacement, complementarity, or augmentation of teachers' competences and professional practice. GenAI technological affordances required in supporting teaming, along with empirical studies, are discussed. Drawing on empirical observations, we outline a future vision that moves beyond individual teacher agency toward collaborative decision-making between teachers and AI, in which both agents engage in negotiation, constructive challenge, and co-reasoning that enhance each other's capabilities and enable outcomes neither could realise independently. Further discussion of socio-technical factors beyond teacher-AI teaming is also included to streamline the synergy of teachers and AI in education ethically and practically. 

---
# HunyuanOCR Technical Report 

**Authors**: Hunyuan Vision Team, Pengyuan Lyu, Xingyu Wan, Gengluo Li, Shangpin Peng, Weinong Wang, Liang Wu, Huawen Shen, Yu Zhou, Canhui Tang, Qi Yang, Qiming Peng, Bin Luo, Hower Yang, Houwen Peng, Hongming Yang, Senhao Xie, Binghong Wu, Mana Yang, Sergey Wang, Raccoon Liu, Dick Zhu, Jie Jiang, Linus, Han Hu, Chengquan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.19575)  

**Abstract**: This paper presents HunyuanOCR, a commercial-grade, open-source, and lightweight (1B parameters) Vision-Language Model (VLM) dedicated to OCR tasks. The architecture comprises a Native Vision Transformer (ViT) and a lightweight LLM connected via an MLP adapter. HunyuanOCR demonstrates superior performance, outperforming commercial APIs, traditional pipelines, and larger models (e.g., Qwen3-VL-4B). Specifically, it surpasses current public solutions in perception tasks (Text Spotting, Parsing) and excels in semantic tasks (IE, Text Image Translation), securing first place in the ICDAR 2025 DIMT Challenge (Small Model Track). Furthermore, it achieves state-of-the-art (SOTA) results on OCRBench among VLMs with fewer than 3B parameters.
HunyuanOCR achieves breakthroughs in three key aspects: 1) Unifying Versatility and Efficiency: We implement comprehensive support for core capabilities including spotting, parsing, IE, VQA, and translation within a lightweight framework. This addresses the limitations of narrow "OCR expert models" and inefficient "General VLMs". 2) Streamlined End-to-End Architecture: Adopting a pure end-to-end paradigm eliminates dependencies on pre-processing modules (e.g., layout analysis). This fundamentally resolves error propagation common in traditional pipelines and simplifies system deployment. 3) Data-Driven and RL Strategies: We confirm the critical role of high-quality data and, for the first time in the industry, demonstrate that Reinforcement Learning (RL) strategies yield significant performance gains in OCR tasks.
HunyuanOCR is officially open-sourced on HuggingFace. We also provide a high-performance deployment solution based on vLLM, placing its production efficiency in the top tier. We hope this model will advance frontier research and provide a solid foundation for industrial applications. 

---
# Deductive Systems for Logic Programs with Counting 

**Authors**: Jorge Fandinno, Vladimir Lifschitz  

**Link**: [PDF](https://arxiv.org/pdf/2511.19565)  

**Abstract**: In answer set programming, two groups of rules are considered strongly equivalent if they have the same meaning in any context. Strong equivalence of two programs can be sometimes established by deriving rules of each program from rules of the other in an appropriate deductive system. This paper shows how to extend this method of proving strong equivalence to programs containing the counting aggregate. 

---
# Trust-Based Social Learning for Communication (TSLEC) Protocol Evolution in Multi-Agent Reinforcement Learning 

**Authors**: Abraham Itzhak Weinberg  

**Link**: [PDF](https://arxiv.org/pdf/2511.19562)  

**Abstract**: Emergent communication in multi-agent systems typically occurs through independent learning, resulting in slow convergence and potentially suboptimal protocols. We introduce TSLEC (Trust-Based Social Learning with Emergent Communication), a framework where agents explicitly teach successful strategies to peers, with knowledge transfer modulated by learned trust relationships. Through experiments with 100 episodes across 30 random seeds, we demonstrate that trust-based social learning reduces episodes-to-convergence by 23.9% (p < 0.001, Cohen's d = 1.98) compared to independent emergence, while producing compositional protocols (C = 0.38) that remain robust under dynamic objectives (Phi > 0.867 decoding accuracy). Trust scores strongly correlate with teaching quality (r = 0.743, p < 0.001), enabling effective knowledge filtering. Our results establish that explicit social learning fundamentally accelerates emergent communication in multi-agent coordination. 

---
# Merging without Forgetting: Continual Fusion of Task-Specific Models via Optimal Transport 

**Authors**: Zecheng Pan, Zhikang Chen, Ding Li, Min Zhang, Sen Cui, Hongshuo Jin, Luqi Tao, Yi Yang, Deheng Ye, Yu Zhang, Tingting Zhu, Tianling Ren  

**Link**: [PDF](https://arxiv.org/pdf/2511.19561)  

**Abstract**: Merging models fine-tuned for different tasks into a single unified model has become an increasingly important direction for building versatile, efficient multi-task systems. Existing approaches predominantly rely on parameter interpolation in weight space, which we show introduces significant distribution shift in the feature space and undermines task-specific knowledge. In this paper, we propose OTMF (Optimal Transport-based Masked Fusion), a novel model merging framework rooted in optimal transport theory to address the distribution shift that arises from naive parameter interpolation. Instead of directly aggregating features or weights, OTMF aligns the semantic geometry of task-specific models by discovering common masks applied to task vectors through optimal transport plans. These masks selectively extract transferable and task-agnostic components while preserving the unique structural identities of each task. To ensure scalability in real-world settings, OTMF further supports a continual fusion paradigm that incrementally integrates each new task vector without revisiting previous ones, maintaining a bounded memory footprint and enabling efficient fusion across a growing number of tasks. We conduct comprehensive experiments on multiple vision and language benchmarks, and results show that OTMF achieves state-of-the-art performance in terms of both accuracy and efficiency. These findings highlight the practical and theoretical value of our approach to model merging. 

---
# SPQR: A Standardized Benchmark for Modern Safety Alignment Methods in Text-to-Image Diffusion Models 

**Authors**: Mohammed Talha Alam, Nada Saadi, Fahad Shamshad, Nils Lukas, Karthik Nandakumar, Fahkri Karray, Samuele Poppi  

**Link**: [PDF](https://arxiv.org/pdf/2511.19558)  

**Abstract**: Text-to-image diffusion models can emit copyrighted, unsafe, or private content. Safety alignment aims to suppress specific concepts, yet evaluations seldom test whether safety persists under benign downstream fine-tuning routinely applied after deployment (e.g., LoRA personalization, style/domain adapters). We study the stability of current safety methods under benign fine-tuning and observe frequent breakdowns. As true safety alignment must withstand even benign post-deployment adaptations, we introduce the SPQR benchmark (Safety-Prompt adherence-Quality-Robustness). SPQR is a single-scored metric that provides a standardized and reproducible framework to evaluate how well safety-aligned diffusion models preserve safety, utility, and robustness under benign fine-tuning, by reporting a single leaderboard score to facilitate comparisons. We conduct multilingual, domain-specific, and out-of-distribution analyses, along with category-wise breakdowns, to identify when safety alignment fails after benign fine-tuning, ultimately showcasing SPQR as a concise yet comprehensive benchmark for T2I safety alignment techniques for T2I models. 

---
# Think First, Assign Next (ThiFAN-VQA): A Two-stage Chain-of-Thought Framework for Post-Disaster Damage Assessment 

**Authors**: Ehsan Karimi, Nhut Le, Maryam Rahnemoonfar  

**Link**: [PDF](https://arxiv.org/pdf/2511.19557)  

**Abstract**: Timely and accurate assessment of damages following natural disasters is essential for effective emergency response and recovery. Recent AI-based frameworks have been developed to analyze large volumes of aerial imagery collected by Unmanned Aerial Vehicles, providing actionable insights rapidly. However, creating and annotating data for training these models is costly and time-consuming, resulting in datasets that are limited in size and diversity. Furthermore, most existing approaches rely on traditional classification-based frameworks with fixed answer spaces, restricting their ability to provide new information without additional data collection or model retraining. Using pre-trained generative models built on in-context learning (ICL) allows for flexible and open-ended answer spaces. However, these models often generate hallucinated outputs or produce generic responses that lack domain-specific relevance. To address these limitations, we propose ThiFAN-VQA, a two-stage reasoning-based framework for visual question answering (VQA) in disaster scenarios. ThiFAN-VQA first generates structured reasoning traces using chain-of-thought (CoT) prompting and ICL to enable interpretable reasoning under limited supervision. A subsequent answer selection module evaluates the generated responses and assigns the most coherent and contextually accurate answer, effectively improve the model performance. By integrating a custom information retrieval system, domain-specific prompting, and reasoning-guided answer selection, ThiFAN-VQA bridges the gap between zero-shot and supervised methods, combining flexibility with consistency. Experiments on FloodNet and RescueNet-VQA, UAV-based datasets from flood- and hurricane-affected regions, demonstrate that ThiFAN-VQA achieves superior accuracy, interpretability, and adaptability for real-world post-disaster damage assessment tasks. 

---
# Online Sparse Feature Selection in Data Streams via Differential Evolution 

**Authors**: Ruiyang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2511.19555)  

**Abstract**: The processing of high-dimensional streaming data commonly utilizes online streaming feature selection (OSFS) techniques. However, practical implementations often face challenges with data incompleteness due to equipment failures and technical constraints. Online Sparse Streaming Feature Selection (OS2FS) tackles this issue through latent factor analysis-based missing data imputation. Despite this advancement, existing OS2FS approaches exhibit substantial limitations in feature evaluation, resulting in performance deterioration. To address these shortcomings, this paper introduces a novel Online Differential Evolution for Sparse Feature Selection (ODESFS) in data streams, incorporating two key innovations: (1) missing value imputation using a latent factor analysis model, and (2) feature importance evaluation through differential evolution. Comprehensive experiments conducted on six real-world datasets demonstrate that ODESFS consistently outperforms state-of-the-art OSFS and OS2FS methods by selecting optimal feature subsets and achieving superior accuracy. 

---
# The Semiotic Channel Principle: Measuring the Capacity for Meaning in LLM Communication 

**Authors**: Davide Picca  

**Link**: [PDF](https://arxiv.org/pdf/2511.19550)  

**Abstract**: This paper proposes a novel semiotic framework for analyzing Large Language Models (LLMs), conceptualizing them as stochastic semiotic engines whose outputs demand active, asymmetric human interpretation. We formalize the trade-off between expressive richness (semiotic breadth) and interpretive stability (decipherability) using information-theoretic tools. Breadth is quantified as source entropy, and decipherability as the mutual information between messages and human interpretations. We introduce a generative complexity parameter (lambda) that governs this trade-off, as both breadth and decipherability are functions of lambda. The core trade-off is modeled as an emergent property of their distinct responses to $\lambda$. We define a semiotic channel, parameterized by audience and context, and posit a capacity constraint on meaning transmission, operationally defined as the maximum decipherability by optimizing lambda. This reframing shifts analysis from opaque model internals to observable textual artifacts, enabling empirical measurement of breadth and decipherability. We demonstrate the framework's utility across four key applications: (i) model profiling; (ii) optimizing prompt/context design; (iii) risk analysis based on ambiguity; and (iv) adaptive semiotic systems. We conclude that this capacity-based semiotic approach offers a rigorous, actionable toolkit for understanding, evaluating, and designing LLM-mediated communication. 

---
# When Should Neural Data Inform Welfare? A Critical Framework for Policy Uses of Neuroeconomics 

**Authors**: Yiven  

**Link**: [PDF](https://arxiv.org/pdf/2511.19548)  

**Abstract**: Neuroeconomics promises to ground welfare analysis in neural and computational evidence about how people value outcomes, learn from experience and exercise self-control. At the same time, policy and commercial actors increasingly invoke neural data to justify paternalistic regulation, "brain-based" interventions and new welfare measures. This paper asks under what conditions neural data can legitimately inform welfare judgements for policy rather than merely describing behaviour. I develop a non-empirical, model-based framework that links three levels: neural signals, computational decision models and normative welfare criteria. Within an actor-critic reinforcement-learning model, I formalise the inference path from neural activity to latent values and prediction errors and then to welfare claims. I show that neural evidence constrains welfare judgements only when the neural-computational mapping is well validated, the decision model identifies "true" interests versus context-dependent mistakes, and the welfare criterion is explicitly specified and defended. Applying the framework to addiction, neuromarketing and environmental policy, I derive a Neuroeconomic Welfare Inference Checklist for regulators and for designers of NeuroAI systems. The analysis treats brains and artificial agents as value-learning systems while showing that internal reward signals, whether biological or artificial, are computational quantities and cannot be treated as welfare measures without an explicit normative model. 

---
# Cross-Domain Generalization of Multimodal LLMs for Global Photovoltaic Assessment 

**Authors**: Muhao Guo, Yang Weng  

**Link**: [PDF](https://arxiv.org/pdf/2511.19537)  

**Abstract**: The rapid expansion of distributed photovoltaic (PV) systems poses challenges for power grid management, as many installations remain undocumented. While satellite imagery provides global coverage, traditional computer vision (CV) models such as CNNs and U-Nets require extensive labeled data and fail to generalize across regions. This study investigates the cross-domain generalization of a multimodal large language model (LLM) for global PV assessment. By leveraging structured prompts and fine-tuning, the model integrates detection, localization, and quantification within a unified schema. Cross-regional evaluation using the $\Delta$F1 metric demonstrates that the proposed model achieves the smallest performance degradation across unseen regions, outperforming conventional CV and transformer baselines. These results highlight the robustness of multimodal LLMs under domain shift and their potential for scalable, transferable, and interpretable global PV mapping. 

---
# AttackPilot: Autonomous Inference Attacks Against ML Services With LLM-Based Agents 

**Authors**: Yixin Wu, Rui Wen, Chi Cui, Michael Backes, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.19536)  

**Abstract**: Inference attacks have been widely studied and offer a systematic risk assessment of ML services; however, their implementation and the attack parameters for optimal estimation are challenging for non-experts. The emergence of advanced large language models presents a promising yet largely unexplored opportunity to develop autonomous agents as inference attack experts, helping address this challenge. In this paper, we propose AttackPilot, an autonomous agent capable of independently conducting inference attacks without human intervention. We evaluate it on 20 target services. The evaluation shows that our agent, using GPT-4o, achieves a 100.0% task completion rate and near-expert attack performance, with an average token cost of only $0.627 per run. The agent can also be powered by many other representative LLMs and can adaptively optimize its strategy under service constraints. We further perform trace analysis, demonstrating that design choices, such as a multi-agent framework and task-specific action spaces, effectively mitigate errors such as bad plans, inability to follow instructions, task context loss, and hallucinations. We anticipate that such agents could empower non-expert ML service providers, auditors, or regulators to systematically assess the risks of ML services without requiring deep domain expertise. 

---
# Discover, Learn, and Reinforce: Scaling Vision-Language-Action Pretraining with Diverse RL-Generated Trajectories 

**Authors**: Rushuai Yang, Zhiyuan Feng, Tianxiang Zhang, Kaixin Wang, Chuheng Zhang, Li Zhao, Xiu Su, Yi Chen, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2511.19528)  

**Abstract**: Scaling vision-language-action (VLA) model pre-training requires large volumes of diverse, high-quality manipulation trajectories. Most current data is obtained via human teleoperation, which is expensive and difficult to scale. Reinforcement learning (RL) methods learn useful skills through autonomous exploration, making them a viable approach for generating data. However, standard RL training collapses to a narrow execution pattern, limiting its utility for large-scale pre-training. We propose Discover, Lea rn and Reinforce (DLR), an information-theoretic pattern discovery framework that generates multiple distinct, high-success behavioral patterns for VLA pretraining. Empirically, DLR generates a markedly more diverse trajectory corpus on LIBERO. Specifically, it learns multiple distinct, high-success strategies for the same task where standard RL discovers only one, and hence it covers substantially broader regions of the state-action space. When adapted to unseen downstream task suites, VLA models pretrained on our diverse RL data surpass counterparts trained on equal-sized standard RL datasets. Moreover, DLR exhibits positive data-scaling behavior that single-pattern RL lacks. These results position multi-pattern RL as a practical, scalable data engine for embodied foundation models. 

---
# Towards Efficient VLMs: Information-Theoretic Driven Compression via Adaptive Structural Pruning 

**Authors**: Zhaoqi Xu, Yingying Zhang, Jian Li, Jianwei Guo, Qiannan Zhu, Hua Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.19518)  

**Abstract**: Recent advances in vision-language models (VLMs) have shown remarkable performance across multimodal tasks, yet their ever-growing scale poses severe challenges for deployment and efficiency. Existing compression methods often rely on heuristic importance metrics or empirical pruning rules, lacking theoretical guarantees about information preservation. In this work, we propose InfoPrune, an information-theoretic framework for adaptive structural compression of VLMs. Grounded in the Information Bottleneck principle, we formulate pruning as a trade-off between retaining task-relevant semantics and discarding redundant dependencies. To quantify the contribution of each attention head, we introduce an entropy-based effective rank (eRank) and employ the Kolmogorov--Smirnov (KS) distance to measure the divergence between original and compressed structures. This yields a unified criterion that jointly considers structural sparsity and informational efficiency. Building on this foundation, we further design two complementary schemes: (1) a training-based head pruning guided by the proposed information loss objective, and (2) a training-free FFN compression via adaptive low-rank approximation. Extensive experiments on VQAv2, TextVQA, and GQA demonstrate that InfoPrune achieves up to 3.2x FLOP reduction and 1.8x acceleration with negligible performance degradation, establishing a theoretically grounded and practically effective step toward efficient multimodal large models. 

---
# CycleChemist: A Dual-Pronged Machine Learning Framework for Organic Photovoltaic Discovery 

**Authors**: Hou Hei Lam, Jiangjie Qiu, Xiuyuan Hu, Wentao Li, Fankun Zeng, Siwei Fu, Hao Zhang, Xiaonan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.19500)  

**Abstract**: Organic photovoltaic (OPV) materials offer a promising path toward sustainable energy generation, but their development is limited by the difficulty of identifying high performance donor and acceptor pairs with strong power conversion efficiencies (PCEs). Existing design strategies typically focus on either the donor or the acceptor alone, rather than using a unified approach capable of modeling both components. In this work, we introduce a dual machine learning framework for OPV discovery that combines predictive modeling with generative molecular design. We present the Organic Photovoltaic Donor Acceptor Dataset (OPV2D), the largest curated dataset of its kind, containing 2000 experimentally characterized donor acceptor pairs. Using this dataset, we develop the Organic Photovoltaic Classifier (OPVC) to predict whether a material exhibits OPV behavior, and a hierarchical graph neural network that incorporates multi task learning and donor acceptor interaction modeling. This framework includes the Molecular Orbital Energy Estimator (MOE2) for predicting HOMO and LUMO energy levels, and the Photovoltaic Performance Predictor (P3) for estimating PCE. In addition, we introduce the Material Generative Pretrained Transformer (MatGPT) to produce synthetically accessible organic semiconductors, guided by a reinforcement learning strategy with three objective policy optimization. By linking molecular representation learning with performance prediction, our framework advances data driven discovery of high performance OPV materials. 

---
# Beyond Binary Classification: A Semi-supervised Approach to Generalized AI-generated Image Detection 

**Authors**: Hong-Hanh Nguyen-Le, Van-Tuan Tran, Dinh-Thuc Nguyen, Nhien-An Le-Khac  

**Link**: [PDF](https://arxiv.org/pdf/2511.19499)  

**Abstract**: The rapid advancement of generators (e.g., StyleGAN, Midjourney, DALL-E) has produced highly realistic synthetic images, posing significant challenges to digital media authenticity. These generators are typically based on a few core architectural families, primarily Generative Adversarial Networks (GANs) and Diffusion Models (DMs). A critical vulnerability in current forensics is the failure of detectors to achieve cross-generator generalization, especially when crossing architectural boundaries (e.g., from GANs to DMs). We hypothesize that this gap stems from fundamental differences in the artifacts produced by these \textbf{distinct architectures}. In this work, we provide a theoretical analysis explaining how the distinct optimization objectives of the GAN and DM architectures lead to different manifold coverage behaviors. We demonstrate that GANs permit partial coverage, often leading to boundary artifacts, while DMs enforce complete coverage, resulting in over-smoothing patterns. Motivated by this analysis, we propose the \textbf{Tri}archy \textbf{Detect}or (TriDetect), a semi-supervised approach that enhances binary classification by discovering latent architectural patterns within the "fake" class. TriDetect employs balanced cluster assignment via the Sinkhorn-Knopp algorithm and a cross-view consistency mechanism, encouraging the model to learn fundamental architectural distincts. We evaluate our approach on two standard benchmarks and three in-the-wild datasets against 13 baselines to demonstrate its generalization capability to unseen generators. 

---
# Hierarchical Dual-Strategy Unlearning for Biomedical and Healthcare Intelligence Using Imperfect and Privacy-Sensitive Medical Data 

**Authors**: Yi Zhang, Tianxiang Xu, Zijian Li, Chao Zhang, Kunyu Zhang, Zhan Gao, Meinuo Li, Xiaohan Zhang, Qichao Qi, Bing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.19498)  

**Abstract**: Large language models (LLMs) exhibit exceptional performance but pose substantial privacy risks due to training data memorization, particularly within healthcare contexts involving imperfect or privacy-sensitive patient information. We present a hierarchical dual-strategy framework for selective knowledge unlearning that precisely removes specialized knowledge while preserving fundamental medical competencies. Our approach synergistically integrates geometric-constrained gradient updates to selectively modulate target parameters with concept-aware token-level interventions that distinguish between preservation-critical and unlearning-targeted tokens via a unified four-level medical concept hierarchy. Comprehensive evaluations on the MedMCQA (surgical) and MHQA (anxiety, depression, trauma) datasets demonstrate superior performance, achieving an 82.7% forgetting rate and 88.5% knowledge preservation. Notably, our framework maintains robust privacy guarantees while requiring modification of only 0.1% of parameters, addressing critical needs for regulatory compliance, auditability, and ethical standards in clinical research. 

---
# PeriodNet: Boosting the Potential of Attention Mechanism for Time Series Forecasting 

**Authors**: Bowen Zhao, Huanlai Xing, Zhiwen Xiao, Jincheng Peng, Li Feng, Xinhan Wang, Rong Qu, Hui Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.19497)  

**Abstract**: The attention mechanism has demonstrated remarkable potential in sequence modeling, exemplified by its successful application in natural language processing with models such as Bidirectional Encoder Representations from Transformers (BERT) and Generative Pre-trained Transformer (GPT). Despite these advancements, its utilization in time series forecasting (TSF) has yet to meet expectations. Exploring a better network structure for attention in TSF holds immense significance across various domains. In this paper, we present PeriodNet with a brand new structure to forecast univariate and multivariate time series. PeriodNet incorporates period attention and sparse period attention mechanism for analyzing adjacent periods. It enhances the mining of local characteristics, periodic patterns, and global dependencies. For efficient cross-variable modeling, we introduce an iterative grouping mechanism which can directly reduce the cross-variable redundancy. To fully leverage the extracted features on the encoder side, we redesign the entire architecture of the vanilla Transformer and propose a period diffuser for precise multi-period prediction. Through comprehensive experiments conducted on eight datasets, we demonstrate that PeriodNet outperforms six state-of-the-art models in both univariate and multivariate TSF scenarios in terms of mean square error and mean absolute error. In particular, PeriodNet achieves a relative improvement of 22% when forecasting time series with a length of 720, in comparison to other models based on the conventional encoder-decoder Transformer architecture. 

---
# Xmodel-2.5: 1.3B Data-Efficient Reasoning SLM 

**Authors**: Yang Liu, Xiaolong Zhong, Ling Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2511.19496)  

**Abstract**: Large language models deliver strong reasoning and tool-use skills, yet their computational demands make them impractical for edge or cost-sensitive deployments. We present \textbf{Xmodel-2.5}, a 1.3-billion-parameter small language model designed as a \emph{drop-in agent core}. Training with maximal-update parameterization ($\mu$P) allows hyper-parameters tuned on a 20M-parameter proxy to transfer directly to the full model, even under the parameter-tied \emph{tie-word-embedding} architecture. A 1.4T-token Warmup--Stable--Decay curriculum is used, and we further show that \textbf{switching from AdamW to Muon during the decay phase} improves the 13-task reasoning average by 4.58\,\% while keeping every other hyper-parameter fixed, verifying that early AdamW stability can be paired with late Muon sharpening for better downstream performance. FP8-mixed-precision training balances accuracy and throughput. All checkpoints, recipes, and evaluation code are released under the Apache-2.0 license.\footnote{this https URL and this https URL (training checkpoints).} Training code and evaluation harness: this https URL. 

---
# A Systematic Study of Compression Ordering for Large Language Models 

**Authors**: Shivansh Chhawri, Rahul Mahadik, Suparna Rooj  

**Link**: [PDF](https://arxiv.org/pdf/2511.19495)  

**Abstract**: Large Language Models (LLMs) require substantial computational resources, making model compression essential for efficient deployment in constrained environments. Among the dominant compression techniques: knowledge distillation, structured pruning, and low-bit quantization, their individual effects are well studied, but their interactions and optimal sequencing remain unclear. This work systematically examines how these techniques perform both independently and in combination when applied to the Qwen2.5 3B model. We evaluate multiple compression pipelines, including single, and proposed three-technique sequences, using perplexity, G-Eval, clarity, prompt alignment, and compression ratio as metrics. Our experiments show that quantization provides the greatest standalone compression, while pruning introduces moderate quality degradation. Critically, the ordering of techniques significantly affects the final model quality: the sequence Pruning, Knowledge Distillation, Quantization (P-KD-Q) yields the best balance, achieving a 3.68x compression ratio while preserving strong instruction-following and language understanding capabilities. Conversely, pipelines applying quantization early suffer severe performance degradation due to irreversible information loss that impairs subsequent training. Overall, this study offers practical insight into designing effective, ordering-aware compression pipelines for deploying LLMs in resource-limited settings. 

---
# Forecasting AI Time Horizon Under Compute Slowdowns 

**Authors**: Parker Whitfill, Ben Snodin, Joel Becker  

**Link**: [PDF](https://arxiv.org/pdf/2511.19492)  

**Abstract**: METR's time horizon metric has grown exponentially since 2019, along with compute. However, it is unclear whether compute scaling will persist at current rates through 2030, raising the question of how possible compute slowdowns might impact AI agent capability forecasts. Given a model of time horizon as a function of training compute and algorithms, along with a model of how compute investment spills into algorithmic progress (which, notably, precludes the possibility of a software-only singularity), and the empirical fact that both time horizon and compute have grown at constant rates over 2019--2025, we derive that time horizon growth must be proportional to compute growth. We provide additional, albeit limited, experimental evidence consistent with this theory. We use our model to project time horizon growth under OpenAI's compute projection, finding substantial projected delays in some cases. For example, 1-month time horizons at $80\%$ reliability occur $7$ years later than simple trend extrapolation suggests. 

---
# Generative Model-Aided Continual Learning for CSI Feedback in FDD mMIMO-OFDM Systems 

**Authors**: Guijun Liu, Yuwen Cao, Tomoaki Ohtsuki, Jiguang He, Shahid Mumtaz  

**Link**: [PDF](https://arxiv.org/pdf/2511.19490)  

**Abstract**: Deep autoencoder (DAE) frameworks have demonstrated their effectiveness in reducing channel state information (CSI) feedback overhead in massive multiple-input multiple-output (mMIMO) orthogonal frequency division multiplexing (OFDM) systems. However, existing CSI feedback models struggle to adapt to dynamic environments caused by user mobility, requiring retraining when encountering new CSI distributions. Moreover, returning to previously encountered environments often leads to performance degradation due to catastrophic forgetting. Continual learning involves enabling models to incorporate new information while maintaining performance on previously learned tasks. To address these challenges, we propose a generative adversarial network (GAN)-based learning approach for CSI feedback. By using a GAN generator as a memory unit, our method preserves knowledge from past environments and ensures consistently high performance across diverse scenarios without forgetting. Simulation results show that the proposed approach enhances the generalization capability of the DAE framework while maintaining low memory overhead. Furthermore, it can be seamlessly integrated with other advanced CSI feedback models, highlighting its robustness and adaptability. 

---
# Evolution without an Oracle: Driving Effective Evolution with LLM Judges 

**Authors**: Zhe Zhao, Yuheng Yang, Haibin Wen, Xiaojie Qiu, Zaixi Zhang, Qingfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.19489)  

**Abstract**: The integration of Large Language Models (LLMs) with Evolutionary Computation (EC) has unlocked new frontiers in scientific discovery but remains shackled by a fundamental constraint: the reliance on an Oracle--an objective, machine-computable fitness function. This paper breaks this barrier by asking: Can evolution thrive in a purely subjective landscape governed solely by LLM judges? We introduce MADE (Multi-Agent Decomposed Evolution), a framework that tames the inherent noise of subjective evaluation through "Problem Specification." By decomposing vague instructions into specific, verifiable sub-requirements, MADE transforms high-variance LLM feedback into stable, precise selection pressure. The results are transformative: across complex benchmarks like DevAI and InfoBench, MADE outperforms strong baselines by over 50% in software requirement satisfaction (39.9% to 61.9%) and achieves a 95% perfect pass rate on complex instruction following. This work validates a fundamental paradigm shift: moving from optimizing "computable metrics" to "describable qualities," thereby unlocking evolutionary optimization for the vast open-ended domains where no ground truth exists. 

---
# Building Resilient Information Ecosystems: Large LLM-Generated Dataset of Persuasion Attacks 

**Authors**: Hsien-Te Kao, Aleksey Panasyuk, Peter Bautista, William Dupree, Gabriel Ganberg, Jeffrey M. Beaubien, Laura Cassani, Svitlana Volkova  

**Link**: [PDF](https://arxiv.org/pdf/2511.19488)  

**Abstract**: Organization's communication is essential for public trust, but the rise of generative AI models has introduced significant challenges by generating persuasive content that can form competing narratives with official messages from government and commercial organizations at speed and scale. This has left agencies in a reactive position, often unaware of how these models construct their persuasive strategies, making it more difficult to sustain communication effectiveness. In this paper, we introduce a large LLM-generated persuasion attack dataset, which includes 134,136 attacks generated by GPT-4, Gemma 2, and Llama 3.1 on agency news. These attacks span 23 persuasive techniques from SemEval 2023 Task 3, directed toward 972 press releases from ten agencies. The generated attacks come in two mediums, press release statements and social media posts, covering both long-form and short-form communication strategies. We analyzed the moral resonance of these persuasion attacks to understand their attack vectors. GPT-4's attacks mainly focus on Care, with Authority and Loyalty also playing a role. Gemma 2 emphasizes Care and Authority, while Llama 3.1 centers on Loyalty and Care. Analyzing LLM-generated persuasive attacks across models will enable proactive defense, allow to create the reputation armor for organizations, and propel the development of both effective and resilient communications in the information ecosystem. 

---
# Efficient Inference Using Large Language Models with Limited Human Data: Fine-Tuning then Rectification 

**Authors**: Lei Wang, Zikun Ye, Jinglong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.19486)  

**Abstract**: Driven by recent advances in artificial intelligence (AI), a growing body of work demonstrates the potential of using large language models (LLMs) to generate human-like responses in market research and social science applications. Two primary approaches can be applied to improve the performance of LLMs: fine-tuning, which aligns LLM predictions more closely with human responses, and rectification, which corrects biases in LLM outputs. In this paper, we develop a framework that combines fine-tuning and rectification, and optimally allocates limited labeled samples across the two stages. Unlike the conventional objective that minimizes the mean squared prediction errors, we propose to minimize the variance of the prediction errors as the fine-tuning objective, which is optimal for the downstream rectification stage. Building on this insight, we leverage empirical scaling laws to develop a data-driven method for optimally splitting samples between the fine-tuning and rectification stages. Empirical analysis validates our framework, demonstrating improved estimation and inference performance compared to using either fine-tuning or rectification alone. 

---
# Z-Space: A Multi-Agent Tool Orchestration Framework for Enterprise-Grade LLM Automation 

**Authors**: Qingsong He, Jing Nan, Jiayu Jiao, Liangjie Tang, Xiaodong Xu, Mengmeng Sun, Qingyao Wang, Minghui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2511.19483)  

**Abstract**: Large Language Models can break through knowledge and timeliness limitations by invoking external tools within the Model Context Protocol framework to achieve automated execution of complex tasks. However, with the rapid growth of enterprise-scale MCP services, efficiently and accurately matching target functionalities among thousands of heterogeneous tools has become a core challenge restricting system practicality. Existing approaches generally rely on full-prompt injection or static semantic retrieval, facing issues including semantic disconnection between user queries and tool descriptions, context inflation in LLM input, and high inference latency. To address these challenges, this paper proposes Z-Space, a data-generation-oriented multi-agent collaborative tool invocation framework Z-Space. The Z-Space framework establishes a multi-agent collaborative architecture and tool filtering algorithm: (1) A structured semantic understanding of user queries is achieved through an intent parsing model; (2) A tool filtering module (FSWW) based on fused subspace weighted algorithm realizes fine-grained semantic alignment between intents and tools without parameter tuning; (3) An inference execution agent is constructed to support dynamic planning and fault-tolerant execution for multi-step tasks. This framework has been deployed in the Eleme platform's technical division, serving large-scale test data generation scenarios across multiple business units including Taotian, Gaode, and Hema. Production data demonstrates that the system reduces average token consumption in tool inference by 96.26\% while achieving a 92\% tool invocation accuracy rate, significantly enhancing the efficiency and reliability of intelligent test data generation systems. 

---
# Human Experts' Evaluation of Generative AI for Contextualizing STEAM Education in the Global South 

**Authors**: Matthew Nyaaba, Macharious Nabang, Patrick Kyeremeh, Ibrahim Nantomah, Collins Owusu-Fordjour, Martin Ako, Bismark Nyaaba Akanzire, Kassim Korah Nantom, Cecilia Issaka, Xiaoming Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2511.19482)  

**Abstract**: This study investigates how human experts evaluate the capacity of Generative AI (GenAI) to contextualize STEAM education in the Global South, with a focus on Ghana. Using a convergent mixed-methods design, four STEAM specialists assessed GenAI-generated lesson plans created with a customized Culturally Responsive Lesson Planner (CRLP) and compared them to standardized lesson plans from the Ghana National Council for Curriculum and Assessment (NaCCA). Quantitative ratings were based on a validated 25-item Culturally Responsive Pedagogy Rubric measuring bias awareness, cultural representation, contextual relevance, linguistic responsiveness, and teacher agency. Qualitative reflections provided additional insight into how GenAI handles cultural and pedagogical appropriateness.
Findings show that GenAI, when paired with the CRLP tool, can support contextualized STEAM instruction by linking abstract curriculum standards to learners' cultural knowledge, community practices, and everyday experiences. Experts rated GenAI-assisted lessons as more culturally grounded and pedagogically responsive than NaCCA plans, integrating Indigenous knowledge, bilingual elements, and locally relevant examples. However, GenAI struggled to represent Ghana's cultural pluralism, often offering surface-level references to language, history, and identity. These weaknesses were most evident in Mathematics and Computing, where cultural nuance was limited. The results highlight the need for continued teacher mediation, community involvement, and culturally attuned refinement of AI outputs. Future work should include classroom trials, expanded expert participation, and model fine-tuning using Indigenous language corpora to strengthen cultural fidelity in Global South contexts. 

---
# Exploiting the Experts: Unauthorized Compression in MoE-LLMs 

**Authors**: Pinaki Prasad Guha Neogi, Ahmad Mohammadshirazi, Dheeraj Kulshrestha, Rajiv Ramnath  

**Link**: [PDF](https://arxiv.org/pdf/2511.19480)  

**Abstract**: Mixture-of-Experts (MoE) architectures are increasingly adopted in large language models (LLMs) for their scalability and efficiency. However, their modular structure introduces a unique vulnerability: adversaries can attempt to compress or repurpose models by pruning experts and cheaply fine-tuning the remainder, effectively bypassing licensing and security constraints. In this paper, we systematically study the prunability of MoE-LLMs under task-specific usage. We first develop an expert attribution framework that identifies the subset of experts most responsible for a given task, then evaluate the performance trade-offs of pruning and re-aligning these experts using active learning-driven fine-tuning. Our findings reveal a critical knowledge loss--recovery trade-off: while certain experts can be isolated to retain task accuracy, significant degradation occurs without targeted re-alignment. Based on this analysis, we propose defense strategies that aim to make MoE models harder to compress and fine-tune without authorization, including entangled expert training and selective fine-tuning protocols that resist unauthorized adaptation. By positioning expert pruning as both a threat vector and a defense target, this work highlights the dual-use nature of MoE modularity and provides the first systematic evaluation framework for secure specialization of MoE-LLMs. 

---
# FAST: Topology-Aware Frequency-Domain Distribution Matching for Coreset Selection 

**Authors**: Jin Cui, Boran Zhao, Jiajun Xu, Jiaqi Guo, Shuo Guan, Pengju Ren  

**Link**: [PDF](https://arxiv.org/pdf/2511.19476)  

**Abstract**: Coreset selection compresses large datasets into compact, representative subsets, reducing the energy and computational burden of training deep neural networks. Existing methods are either: (i) DNN-based, which are tied to model-specific parameters and introduce architectural bias; or (ii) DNN-free, which rely on heuristics lacking theoretical guarantees. Neither approach explicitly constrains distributional equivalence, largely because continuous distribution matching is considered inapplicable to discrete sampling. Moreover, prevalent metrics (e.g., MSE, KL, MMD, CE) cannot accurately capture higher-order moment discrepancies, leading to suboptimal coresets. In this work, we propose FAST, the first DNN-free distribution-matching coreset selection framework that formulates the coreset selection task as a graph-constrained optimization problem grounded in spectral graph theory and employs the Characteristic Function Distance (CFD) to capture full distributional information in the frequency domain. We further discover that naive CFD suffers from a "vanishing phase gradient" issue in medium and high-frequency regions; to address this, we introduce an Attenuated Phase-Decoupled CFD. Furthermore, for better convergence, we design a Progressive Discrepancy-Aware Sampling strategy that progressively schedules frequency selection from low to high, preserving global structure before refining local details and enabling accurate matching with fewer frequencies while avoiding overfitting. Extensive experiments demonstrate that FAST significantly outperforms state-of-the-art coreset selection methods across all evaluated benchmarks, achieving an average accuracy gain of 9.12%. Compared to other baseline coreset methods, it reduces power consumption by 96.57% and achieves a 2.2x average speedup, underscoring its high performance and energy efficiency. 

---
# Tracking and Segmenting Anything in Any Modality 

**Authors**: Tianlu Zhang, Qiang Zhang, Guiguang Ding, Jungong Han  

**Link**: [PDF](https://arxiv.org/pdf/2511.19475)  

**Abstract**: Tracking and segmentation play essential roles in video understanding, providing basic positional information and temporal association of objects within video sequences. Despite their shared objective, existing approaches often tackle these tasks using specialized architectures or modality-specific parameters, limiting their generalization and scalability. Recent efforts have attempted to unify multiple tracking and segmentation subtasks from the perspectives of any modality input or multi-task inference. However, these approaches tend to overlook two critical challenges: the distributional gap across different modalities and the feature representation gap across tasks. These issues hinder effective cross-task and cross-modal knowledge sharing, ultimately constraining the development of a true generalist model. To address these limitations, we propose a universal tracking and segmentation framework named SATA, which unifies a broad spectrum of tracking and segmentation subtasks with any modality input. Specifically, a Decoupled Mixture-of-Expert (DeMoE) mechanism is presented to decouple the unified representation learning task into the modeling process of cross-modal shared knowledge and specific information, thus enabling the model to maintain flexibility while enhancing generalization. Additionally, we introduce a Task-aware Multi-object Tracking (TaMOT) pipeline to unify all the task outputs as a unified set of instances with calibrated ID information, thereby alleviating the degradation of task-specific knowledge during multi-task training. SATA demonstrates superior performance on 18 challenging tracking and segmentation benchmarks, offering a novel perspective for more generalizable video understanding. 

---
# Pistachio: Towards Synthetic, Balanced, and Long-Form Video Anomaly Benchmarks 

**Authors**: Jie Li, Hongyi Cai, Mingkang Dong, Muxin Pu, Shan You, Fei Wang, Tao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.19474)  

**Abstract**: Automatically detecting abnormal events in videos is crucial for modern autonomous systems, yet existing Video Anomaly Detection (VAD) benchmarks lack the scene diversity, balanced anomaly coverage, and temporal complexity needed to reliably assess real-world performance. Meanwhile, the community is increasingly moving toward Video Anomaly Understanding (VAU), which requires deeper semantic and causal reasoning but remains difficult to benchmark due to the heavy manual annotation effort it demands. In this paper, we introduce Pistachio, a new VAD/VAU benchmark constructed entirely through a controlled, generation-based pipeline. By leveraging recent advances in video generation models, Pistachio provides precise control over scenes, anomaly types, and temporal narratives, effectively eliminating the biases and limitations of Internet-collected datasets. Our pipeline integrates scene-conditioned anomaly assignment, multi-step storyline generation, and a temporally consistent long-form synthesis strategy that produces coherent 41-second videos with minimal human intervention. Extensive experiments demonstrate the scale, diversity, and complexity of Pistachio, revealing new challenges for existing methods and motivating future research on dynamic and multi-event anomaly understanding. 

---
# WavefrontDiffusion: Dynamic Decoding Schedule or Improved Reasoning 

**Authors**: Haojin Yang, Rui Hu, Zequn Sun, Rui Zhou, Yujun Cai, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.19473)  

**Abstract**: Diffusion Language Models (DLMs) have shown strong potential for text generation and are becoming a competitive alternative to autoregressive models. The denoising strategy plays an important role in determining the quality of their outputs. Mainstream denoising strategies include Standard Diffusion and BlockDiffusion. Standard Diffusion performs global denoising without restricting the update range, often finalizing incomplete context and causing premature end-of-sequence predictions. BlockDiffusion updates fixed-size blocks in a preset order, but its rigid structure can break apart coherent semantic units and disrupt reasoning. We present WavefrontDiffusion, a dynamic decoding approach that expands a wavefront of active tokens outward from finalized positions. This adaptive process follows the natural flow of semantic structure while keeping computational cost equal to block-based methods. Across four benchmarks in reasoning and code generation, WavefrontDiffusion achieves state-of-the-art performance while producing outputs with higher semantic fidelity, showing the value of adaptive scheduling for more coherent and efficient generation. 

---
# PrefixGPT: Prefix Adder Optimization by a Generative Pre-trained Transformer 

**Authors**: Ruogu Ding, Xin Ning, Ulf Schlichtmann, Weikang Qian  

**Link**: [PDF](https://arxiv.org/pdf/2511.19472)  

**Abstract**: Prefix adders are widely used in compute-intensive applications for their high speed. However, designing optimized prefix adders is challenging due to strict design rules and an exponentially large design space. We introduce PrefixGPT, a generative pre-trained Transformer (GPT) that directly generates optimized prefix adders from scratch. Our approach represents an adder's topology as a two-dimensional coordinate sequence and applies a legality mask during generation, ensuring every design is valid by construction. PrefixGPT features a customized decoder-only Transformer architecture. The model is first pre-trained on a corpus of randomly synthesized valid prefix adders to learn design rules and then fine-tuned to navigate the design space for optimized design quality. Compared with existing works, PrefixGPT not only finds a new optimal design with a 7.7% improved area-delay product (ADP) but exhibits superior exploration quality, lowering the average ADP by up to 79.1%. This demonstrates the potential of GPT-style models to first master complex hardware design principles and then apply them for more efficient design optimization. 

---
# Not Quite Anything: Overcoming SAMs Limitations for 3D Medical Imaging 

**Authors**: Keith Moore  

**Link**: [PDF](https://arxiv.org/pdf/2511.19471)  

**Abstract**: Foundation segmentation models such as SAM and SAM-2 perform well on natural images but struggle with brain MRIs where structures like the caudate and thalamus lack sharp boundaries and have low contrast. Rather than fine tune these models (for example MedSAM), we propose a compositional alternative where the foundation model output is treated as an additional input channel and passed alongside the MRI to highlight regions of interest.
We generate SAM-2 prompts by using a lightweight 3D U-Net that was previously trained on MRI segmentation. The U-Net may have been trained on a different dataset, so its guesses are often imprecise but usually in the correct region. The edges of the resulting foundation model guesses are smoothed to improve alignment with the MRI. We also test prompt free segmentation using DINO attention maps in the same framework.
This has-a architecture avoids modifying foundation weights and adapts to domain shift without retraining the foundation model. It reaches about 96 percent volume accuracy on basal ganglia segmentation, which is sufficient for our study of longitudinal volume change. The approach is fast, label efficient, and robust to out of distribution scans. We apply it to study inflammation linked changes in sudden onset pediatric OCD. 

---
# Quantifying Modality Contributions via Disentangling Multimodal Representations 

**Authors**: Padegal Amit, Omkar Mahesh Kashyap, Namitha Rayasam, Nidhi Shekhar, Surabhi Narayan  

**Link**: [PDF](https://arxiv.org/pdf/2511.19470)  

**Abstract**: Quantifying modality contributions in multimodal models remains a challenge, as existing approaches conflate the notion of contribution itself. Prior work relies on accuracy-based approaches, interpreting performance drops after removing a modality as indicative of its influence. However, such outcome-driven metrics fail to distinguish whether a modality is inherently informative or whether its value arises only through interaction with other modalities. This distinction is particularly important in cross-attention architectures, where modalities influence each other's representations. In this work, we propose a framework based on Partial Information Decomposition (PID) that quantifies modality contributions by decomposing predictive information in internal embeddings into unique, redundant, and synergistic components. To enable scalable, inference-only analysis, we develop an algorithm based on the Iterative Proportional Fitting Procedure (IPFP) that computes layer and dataset-level contributions without retraining. This provides a principled, representation-level view of multimodal behavior, offering clearer and more interpretable insights than outcome-based metrics. 

---
# SG-OIF: A Stability-Guided Online Influence Framework for Reliable Vision Data 

**Authors**: Penghao Rao, Runmin Jiang, Min Xu  

**Link**: [PDF](https://arxiv.org/pdf/2511.19466)  

**Abstract**: Approximating training-point influence on test predictions is critical for deploying deep-learning vision models, essential for locating noisy data. Though the influence function was proposed for attributing how infinitesimal up-weighting or removal of individual training examples affects model outputs, its implementation is still challenging in deep-learning vision models: inverse-curvature computations are expensive, and training non-stationarity invalidates static approximations. Prior works use iterative solvers and low-rank surrogates to reduce cost, but offline computation lags behind training dynamics, and missing confidence calibration yields fragile rankings that misidentify critical examples. To address these challenges, we introduce a Stability-Guided Online Influence Framework (SG-OIF), the first framework that treats algorithmic stability as a real-time controller, which (i) maintains lightweight anchor IHVPs via stochastic Richardson and preconditioned Neumann; (ii) proposes modular curvature backends to modulate per-example influence scores using stability-guided residual thresholds, anomaly gating, and confidence. Experimental results show that SG-OIF achieves SOTA (State-Of-The-Art) on noise-label and out-of-distribution detection tasks across multiple datasets with various corruption. Notably, our approach achieves 91.1\% accuracy in the top 1\% prediction samples on the CIFAR-10 (20\% asym), and gets 99.8\% AUPR score on MNIST, effectively demonstrating that this framework is a practical controller for online influence estimation. 

---
# Hidden markov model to predict tourists visited place 

**Authors**: Theo Demessance, Chongke Bi, Sonia Djebali, Guillaume Guerard  

**Link**: [PDF](https://arxiv.org/pdf/2511.19465)  

**Abstract**: Nowadays, social networks are becoming a popular way of analyzing tourist behavior, thanks to the digital traces left by travelers during their stays on these networks. The massive amount of data generated; by the propensity of tourists to share comments and photos during their trip; makes it possible to model their journeys and analyze their behavior. Predicting the next movement of tourists plays a key role in tourism marketing to understand demand and improve decision support. In this paper, we propose a method to understand and to learn tourists' movements based on social network data analysis to predict future movements. The method relies on a machine learning grammatical inference algorithm. A major contribution in this paper is to adapt the grammatical inference algorithm to the context of big data. Our method produces a hidden Markov model representing the movements of a group of tourists. The hidden Markov model is flexible and editable with new data. The capital city of France, Paris is selected to demonstrate the efficiency of the proposed methodology. 

---
# Temperature in SLMs: Impact on Incident Categorization in On-Premises Environments 

**Authors**: Marcio Pohlmann, Alex Severo, Gefté Almeida, Diego Kreutz, Tiago Heinrich, Lourenço Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2511.19464)  

**Abstract**: SOCs and CSIRTs face increasing pressure to automate incident categorization, yet the use of cloud-based LLMs introduces costs, latency, and confidentiality risks. We investigate whether locally executed SLMs can meet this challenge. We evaluated 21 models ranging from 1B to 20B parameters, varying the temperature hyperparameter and measuring execution time and precision across two distinct architectures. The results indicate that temperature has little influence on performance, whereas the number of parameters and GPU capacity are decisive factors. 

---
# Systemic approach for modeling a generic smart grid 

**Authors**: Sofiane Ben Amor, Guillaume Guerard, Loup-Noé Levy  

**Link**: [PDF](https://arxiv.org/pdf/2511.19460)  

**Abstract**: Smart grid technological advances present a recent class of complex interdisciplinary modeling and increasingly difficult simulation problems to solve using traditional computational methods. To simulate a smart grid requires a systemic approach to integrated modeling of power systems, energy markets, demand-side management, and much other resources and assets that are becoming part of the current paradigm of the power grid. This paper presents a backbone model of a smart grid to test alternative scenarios for the grid. This tool simulates disparate systems to validate assumptions before the human scale model. Thanks to a distributed optimization of subsystems, the production and consumption scheduling is achieved while maintaining flexibility and scalability. 

---
# Personalized Reward Modeling for Text-to-Image Generation 

**Authors**: Jeongeun Lee, Ryang Heo, Dongha Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.19458)  

**Abstract**: Recent text-to-image (T2I) models generate semantically coherent images from textual prompts, yet evaluating how well they align with individual user preferences remains an open challenge. Conventional evaluation methods, general reward functions or similarity-based metrics, fail to capture the diversity and complexity of personal visual tastes. In this work, we present PIGReward, a personalized reward model that dynamically generates user-conditioned evaluation dimensions and assesses images through CoT reasoning. To address the scarcity of user data, PIGReward adopt a self-bootstrapping strategy that reasons over limited reference data to construct rich user contexts, enabling personalization without user-specific training. Beyond evaluation, PIGReward provides personalized feedback that drives user-specific prompt optimization, improving alignment between generated images and individual intent. We further introduce PIGBench, a per-user preference benchmark capturing diverse visual interpretations of shared prompts. Extensive experiments demonstrate that PIGReward surpasses existing methods in both accuracy and interpretability, establishing a scalable and reasoning-based foundation for personalized T2I evaluation and optimization. Taken together, our findings highlight PIGReward as a robust steptoward individually aligned T2I generation. 

---
# SparOA: Sparse and Operator-aware Hybrid Scheduling for Edge DNN Inference 

**Authors**: Ziyang Zhang, Jie Liu, Luca Mottola  

**Link**: [PDF](https://arxiv.org/pdf/2511.19457)  

**Abstract**: The resource demands of deep neural network (DNN) models introduce significant performance challenges, especially when deployed on resource-constrained edge devices. Existing solutions like model compression often sacrifice accuracy, while specialized hardware remains costly and inflexible. Hybrid inference methods, however, typically overlook how operator characteristics impact performance. In this work, we present SparOA, a CPU-GPU hybrid inference framework, which leverages both sparsity and computational intensity to optimize operator scheduling. SparOA embraces aforementioned challenges through three key components: (1) a threshold predictor that accurately determines optimal sparsity and computational intensity thresholds; (2) a reinforcement learning-based scheduler that dynamically optimizes resource allocation based on real-time hardware states; and (3) a hybrid inference engine that enhances efficiency through asynchronous execution and batch size this http URL results show that SparOA achieves an average speedup of 1.22-1.31x compared to all baselines, and outperforms the CPU-Only by up to 50.7x. Also, SparOA achieves optimal energy-per-inference, consuming 7\%-16\% less energy than the SOTA co-execution baseline. 

---
# BlockCert: Certified Blockwise Extraction of Transformer Mechanisms 

**Authors**: Sandro Andric  

**Link**: [PDF](https://arxiv.org/pdf/2511.17645)  

**Abstract**: Mechanistic interpretability aspires to reverse-engineer neural networks into explicit algorithms, while model editing seeks to modify specific behaviours without retraining. Both areas are typically evaluated with informal evidence and ad-hoc experiments, with few explicit guarantees about how far an extracted or edited model can drift from the original on relevant inputs. We introduce BlockCert, a framework for certified blockwise extraction of transformer mechanisms, and outline how a lightweight extension can support certified local edits. Given a pre-trained transformer and a prompt distribution, BlockCert extracts structured surrogate implementations for residual blocks together with machine-checkable certificates that bound approximation error, record coverage metrics, and hash the underlying artifacts. We formalize a simple Lipschitz-based composition theorem in Lean 4 that lifts these local guarantees to a global deviation bound. Empirically, we apply the framework to GPT-2 small, TinyLlama-1.1B-Chat, and Llama-3.2-3B. Across these models we obtain high per-block coverage and small residual errors on the evaluated prompts, and in the TinyLlama setting we show that a fully stitched model matches the baseline perplexity within approximately 6e-5 on stress prompts. Our results suggest that blockwise extraction with explicit certificates is feasible for real transformer language models and offers a practical bridge between mechanistic interpretability and formal reasoning about model behaviour. 

---
