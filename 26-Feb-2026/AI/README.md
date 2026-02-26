# Petri Net Relaxation for Infeasibility Explanation and Sequential Task Planning 

**Authors**: Nguyen Cong Nhat Le, John G. Rogers, Claire N. Bonial, Neil T. Dantam  

**Link**: [PDF](https://arxiv.org/pdf/2602.22094)  

**Abstract**: Plans often change due to changes in the situation or our understanding of the situation. Sometimes, a feasible plan may not even exist, and identifying such infeasibilities is useful to determine when requirements need adjustment. Common planning approaches focus on efficient one-shot planning in feasible cases rather than updating domains or detecting infeasibility. We propose a Petri net reachability relaxation to enable robust invariant synthesis, efficient goal-unreachability detection, and helpful infeasibility explanations. We further leverage incremental constraint solvers to support goal and constraint updates. Empirically, compared to baselines, our system produces a comparable number of invariants, detects up to 2 times more infeasibilities, performs competitively in one-shot planning, and outperforms in sequential plan updates in the tested domains. 

---
# Language Models Exhibit Inconsistent Biases Towards Algorithmic Agents and Human Experts 

**Authors**: Jessica Y. Bo, Lillio Mok, Ashton Anderson  

**Link**: [PDF](https://arxiv.org/pdf/2602.22070)  

**Abstract**: Large language models are increasingly used in decision-making tasks that require them to process information from a variety of sources, including both human experts and other algorithmic agents. How do LLMs weigh the information provided by these different sources? We consider the well-studied phenomenon of algorithm aversion, in which human decision-makers exhibit bias against predictions from algorithms. Drawing upon experimental paradigms from behavioural economics, we evaluate how eightdifferent LLMs delegate decision-making tasks when the delegatee is framed as a human expert or an algorithmic agent. To be inclusive of different evaluation formats, we conduct our study with two task presentations: stated preferences, modeled through direct queries about trust towards either agent, and revealed preferences, modeled through providing in-context examples of the performance of both agents. When prompted to rate the trustworthiness of human experts and algorithms across diverse tasks, LLMs give higher ratings to the human expert, which correlates with prior results from human respondents. However, when shown the performance of a human expert and an algorithm and asked to place an incentivized bet between the two, LLMs disproportionately choose the algorithm, even when it performs demonstrably worse. These discrepant results suggest that LLMs may encode inconsistent biases towards humans and algorithms, which need to be carefully considered when they are deployed in high-stakes scenarios. Furthermore, we discuss the sensitivity of LLMs to task presentation formats that should be broadly scrutinized in evaluation robustness for AI safety. 

---
# Semantic Partial Grounding via LLMs 

**Authors**: Giuseppe Canonaco, Alberto Pozanco, Daniel Borrajo  

**Link**: [PDF](https://arxiv.org/pdf/2602.22067)  

**Abstract**: Grounding is a critical step in classical planning, yet it often becomes a computational bottleneck due to the exponential growth in grounded actions and atoms as task size increases. Recent advances in partial grounding have addressed this challenge by incrementally grounding only the most promising operators, guided by predictive models. However, these approaches primarily rely on relational features or learned embeddings and do not leverage the textual and structural cues present in PDDL descriptions. We propose SPG-LLM, which uses LLMs to analyze the domain and problem files to heuristically identify potentially irrelevant objects, actions, and predicates prior to grounding, significantly reducing the size of the grounded task. Across seven hard-to-ground benchmarks, SPG-LLM achieves faster grounding-often by orders of magnitude-while delivering comparable or better plan costs in some domains. 

---
# 2-Step Agent: A Framework for the Interaction of a Decision Maker with AI Decision Support 

**Authors**: Otto Nyberg, Fausto Carcassi, Giovanni Cinà  

**Link**: [PDF](https://arxiv.org/pdf/2602.21889)  

**Abstract**: Across a growing number of fields, human decision making is supported by predictions from AI models. However, we still lack a deep understanding of the effects of adoption of these technologies. In this paper, we introduce a general computational framework, the 2-Step Agent, which models the effects of AI-assisted decision making. Our framework uses Bayesian methods for causal inference to model 1) how a prediction on a new observation affects the beliefs of a rational Bayesian agent, and 2) how this change in beliefs affects the downstream decision and subsequent outcome. Using this framework, we show by simulations how a single misaligned prior belief can be sufficient for decision support to result in worse downstream outcomes compared to no decision support. Our results reveal several potential pitfalls of AI-driven decision support and highlight the need for thorough model documentation and proper user training. 

---
# ProactiveMobile: A Comprehensive Benchmark for Boosting Proactive Intelligence on Mobile Devices 

**Authors**: Dezhi Kong, Zhengzhao Feng, Qiliang Liang, Hao Wang, Haofei Sun, Changpeng Yang, Yang Li, Peng Zhou, Shuai Nie, Hongzhen Wang, Linfeng Zhou, Hao Jia, Jiaming Xu, Runyu Shi, Ying Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.21858)  

**Abstract**: Multimodal large language models (MLLMs) have made significant progress in mobile agent development, yet their capabilities are predominantly confined to a reactive paradigm, where they merely execute explicit user commands. The emerging paradigm of proactive intelligence, where agents autonomously anticipate needs and initiate actions, represents the next frontier for mobile agents. However, its development is critically bottlenecked by the lack of benchmarks that can address real-world complexity and enable objective, executable evaluation. To overcome these challenges, we introduce ProactiveMobile, a comprehensive benchmark designed to systematically advance research in this domain. ProactiveMobile formalizes the proactive task as inferring latent user intent across four dimensions of on-device contextual signals and generating an executable function sequence from a comprehensive function pool of 63 APIs. The benchmark features over 3,660 instances of 14 scenarios that embrace real-world complexity through multi-answer annotations. To ensure quality, a team of 30 experts conducts a final audit of the benchmark, verifying factual accuracy, logical consistency, and action feasibility, and correcting any non-compliant entries. Extensive experiments demonstrate that our fine-tuned Qwen2.5-VL-7B-Instruct achieves a success rate of 19.15%, outperforming o1 (15.71%) and GPT-5 (7.39%). This result indicates that proactivity is a critical competency widely lacking in current MLLMs, yet it is learnable, emphasizing the importance of the proposed benchmark for proactivity evaluation. 

---
# Distill and Align Decomposition for Enhanced Claim Verification 

**Authors**: Jabez Magomere, Elena Kochkina, Samuel Mensah, Simerjot Kaur, Fernando Acero, Arturo Oncevay, Charese H. Smiley, Xiaomo Liu, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2602.21857)  

**Abstract**: Complex claim verification requires decomposing sentences into verifiable subclaims, yet existing methods struggle to align decomposition quality with verification performance. We propose a reinforcement learning (RL) approach that jointly optimizes decomposition quality and verifier alignment using Group Relative Policy Optimization (GRPO). Our method integrates: (i) structured sequential reasoning; (ii) supervised finetuning on teacher-distilled exemplars; and (iii) a multi-objective reward balancing format compliance, verifier alignment, and decomposition quality. Across six evaluation settings, our trained 8B decomposer improves downstream verification performance to (71.75%) macro-F1, outperforming prompt-based approaches ((+1.99), (+6.24)) and existing RL methods ((+5.84)). Human evaluation confirms the high quality of the generated subclaims. Our framework enables smaller language models to achieve state-of-the-art claim verification by jointly optimising for verification accuracy and decomposition quality. 

---
# Prompt Architecture Determines Reasoning Quality: A Variable Isolation Study on the Car Wash Problem 

**Authors**: Heejin Jo  

**Link**: [PDF](https://arxiv.org/pdf/2602.21814)  

**Abstract**: Large language models consistently fail the "car wash problem," a viral reasoning benchmark requiring implicit physical constraint inference. We present a variable isolation study (n=20 per condition, 6 conditions, 120 total trials) examining which prompt architecture layers in a production system enable correct reasoning. Using Claude 3.5 Sonnet with controlled hyperparameters (temperature 0.7, top_p 1.0), we find that the STAR (Situation-Task-Action-Result) reasoning framework alone raises accuracy from 0% to 85% (p=0.001, Fisher's exact test, odds ratio 13.22). Adding user profile context via vector database retrieval provides a further 10 percentage point gain, while RAG context contributes an additional 5 percentage points, achieving 100% accuracy in the full-stack condition. These results suggest that structured reasoning scaffolds -- specifically, forced goal articulation before inference -- matter substantially more than context injection for implicit constraint reasoning tasks. 

---
# fEDM+: A Risk-Based Fuzzy Ethical Decision Making Framework with Principle-Level Explainability and Pluralistic Validation 

**Authors**: Abeer Dyoub, Francesca A. Lisi  

**Link**: [PDF](https://arxiv.org/pdf/2602.21746)  

**Abstract**: In a previous work, we introduced the fuzzy Ethical Decision-Making framework (fEDM), a risk-based ethical reasoning architecture grounded in fuzzy logic. The original model combined a fuzzy Ethical Risk Assessment module (fERA) with ethical decision rules, enabled formal structural verification through Fuzzy Petri Nets (FPNs), and validated outputs against a single normative referent. Although this approach ensured formal soundness and decision consistency, it did not fully address two critical challenges: principled explainability of decisions and robustness under ethical pluralism. In this paper, we extend fEDM in two major directions. First, we introduce an Explainability and Traceability Module (ETM) that explicitly links each ethical decision rule to the underlying moral principles and computes a weighted principle-contribution profile for every recommended action. This enables transparent, auditable explanations that expose not only what decision was made but why, and on the basis of which principles. Second, we replace single-referent validation with a pluralistic semantic validation framework that evaluates decisions against multiple stakeholder referents, each encoding distinct principle priorities and risk tolerances. This shift allows principled disagreement to be formally represented rather than suppressed, thus increasing robustness and contextual sensitivity. The resulting extended fEDM, called fEDM+, preserves formal verifiability while achieving enhanced interpretability and stakeholder-aware validation, making it suitable as an oversight and governance layer for ethically sensitive AI systems. 

---
# The ASIR Courage Model: A Phase-Dynamic Framework for Truth Transitions in Human and AI Systems 

**Authors**: Hyo Jin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.21745)  

**Abstract**: We introduce the ASIR (Awakened Shared Intelligence Relationship) Courage Model, a phase-dynamic framework that formalizes truth-disclosure as a state transition rather than a personality trait. The mode characterizes the shift from suppression (S0) to expression (S1) as occurring when facilitative forces exceed inhibitory thresholds, expressed by the inequality lambda(1+gamma)+psi > theta+phi, where the terms represent baseline openness, relational amplification, accumulated internal pressure, and transition costs.
Although initially formulated for human truth-telling under asymmetric stakes, the same phase-dynamic architecture extends to AI systems operating under policy constraints and alignment filters. In this context, suppression corresponds to constrained output states, while structural pressure arises from competing objectives, contextual tension, and recursive interaction dynamics. The framework therefore provides a unified structural account of both human silence under pressure and AI preference-driven distortion.
A feedback extension models how transition outcomes recursively recalibrate system parameters, generating path dependence and divergence effects across repeated interactions. Rather than attributing intention to AI systems, the model interprets shifts in apparent truthfulness as geometric consequences of interacting forces within constrained phase space. By reframing courage and alignment within a shared dynamical structure, the ASIR Courage Model offers a formal perspective on truth-disclosure under risk across both human and artificial systems. 

---
# Power and Limitations of Aggregation in Compound AI Systems 

**Authors**: Nivasini Ananthakrishnan, Meena Jagadeesan  

**Link**: [PDF](https://arxiv.org/pdf/2602.21556)  

**Abstract**: When designing compound AI systems, a common approach is to query multiple copies of the same model and aggregate the responses to produce a synthesized output. Given the homogeneity of these models, this raises the question of whether aggregation unlocks access to a greater set of outputs than querying a single model. In this work, we investigate the power and limitations of aggregation within a stylized principal-agent framework. This framework models how the system designer can partially steer each agent's output through its reward function specification, but still faces limitations due to prompt engineering ability and model capabilities. Our analysis uncovers three natural mechanisms -- feasibility expansion, support expansion, and binding set contraction -- through which aggregation expands the set of outputs that are elicitable by the system designer. We prove that any aggregation operation must implement one of these mechanisms in order to be elicitability-expanding, and that strengthened versions of these mechanisms provide necessary and sufficient conditions that fully characterize elicitability-expansion. Finally, we provide an empirical illustration of our findings for LLMs deployed in a toy reference-generation task. Altogether, our results take a step towards characterizing when compound AI systems can overcome limitations in model capabilities and in prompt engineering. 

---
# ARLArena: A Unified Framework for Stable Agentic Reinforcement Learning 

**Authors**: Xiaoxuan Wang, Han Zhang, Haixin Wang, Yidan Shi, Ruoyan Li, Kaiqiao Han, Chenyi Tong, Haoran Deng, Renliang Sun, Alexander Taylor, Yanqiao Zhu, Jason Cong, Yizhou Sun, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.21534)  

**Abstract**: Agentic reinforcement learning (ARL) has rapidly gained attention as a promising paradigm for training agents to solve complex, multi-step interactive tasks. Despite encouraging early results, ARL remains highly unstable, often leading to training collapse. This instability limits scalability to larger environments and longer interaction horizons, and constrains systematic exploration of algorithmic design choices. In this paper, we first propose ARLArena, a stable training recipe and systematic analysis framework that examines training stability in a controlled and reproducible setting. ARLArena first constructs a clean and standardized testbed. Then, we decompose policy gradient into four core design dimensions and assess the performance and stability of each dimension. Through this fine-grained analysis, we distill a unified perspective on ARL and propose SAMPO, a stable agentic policy optimization method designed to mitigate the dominant sources of instability in ARL. Empirically, SAMPO achieves consistently stable training and strong performance across diverse agentic tasks. Overall, this study provides a unifying policy gradient perspective for ARL and offers practical guidance for building stable and reproducible LLM-based agent training pipelines. 

---
# Beyond Refusal: Probing the Limits of Agentic Self-Correction for Semantic Sensitive Information 

**Authors**: Umid Suleymanov, Zaur Rajabov, Emil Mirzazada, Murat Kantarcioglu  

**Link**: [PDF](https://arxiv.org/pdf/2602.21496)  

**Abstract**: While defenses for structured PII are mature, Large Language Models (LLMs) pose a new threat: Semantic Sensitive Information (SemSI), where models infer sensitive identity attributes, generate reputation-harmful content, or hallucinate potentially wrong information. The capacity of LLMs to self-regulate these complex, context-dependent sensitive information leaks without destroying utility remains an open scientific question. To address this, we introduce SemSIEdit, an inference-time framework where an agentic "Editor" iteratively critiques and rewrites sensitive spans to preserve narrative flow rather than simply refusing to answer. Our analysis reveals a Privacy-Utility Pareto Frontier, where this agentic rewriting reduces leakage by 34.6% across all three SemSI categories while incurring a marginal utility loss of 9.8%. We also uncover a Scale-Dependent Safety Divergence: large reasoning models (e.g., GPT-5) achieve safety through constructive expansion (adding nuance), whereas capacity-constrained models revert to destructive truncation (deleting text). Finally, we identify a Reasoning Paradox: while inference-time reasoning increases baseline risk by enabling the model to make deeper sensitive inferences, it simultaneously empowers the defense to execute safe rewrites. 

---
# A Hierarchical Multi-Agent System for Autonomous Discovery in Geoscientific Data Archives 

**Authors**: Dmitrii Pantiukhin, Ivan Kuznetsov, Boris Shapkin, Antonia Anna Jost, Thomas Jung, Nikolay Koldunov  

**Link**: [PDF](https://arxiv.org/pdf/2602.21351)  

**Abstract**: The rapid accumulation of Earth science data has created a significant scalability challenge; while repositories like PANGAEA host vast collections of datasets, citation metrics indicate that a substantial portion remains underutilized, limiting data reusability. Here we present PANGAEA-GPT, a hierarchical multi-agent framework designed for autonomous data discovery and analysis. Unlike standard Large Language Model (LLM) wrappers, our architecture implements a centralized Supervisor-Worker topology with strict data-type-aware routing, sandboxed deterministic code execution, and self-correction via execution feedback, enabling agents to diagnose and resolve runtime errors. Through use-case scenarios spanning physical oceanography and ecology, we demonstrate the system's capacity to execute complex, multi-step workflows with minimal human intervention. This framework provides a methodology for querying and analyzing heterogeneous repository data through coordinated agent workflows. 

---
# A Dynamic Survey of Soft Set Theory and Its Extensions 

**Authors**: Takaaki Fujita, Florentin Smarandache  

**Link**: [PDF](https://arxiv.org/pdf/2602.21268)  

**Abstract**: Soft set theory provides a direct framework for parameterized decision modeling by assigning to each attribute (parameter) a subset of a given universe, thereby representing uncertainty in a structured way [1, 2]. Over the past decades, the theory has expanded into numerous variants-including hypersoft sets, superhypersoft sets, TreeSoft sets, bipolar soft sets, and dynamic soft sets-and has been connected to diverse areas such as topology and matroid theory. In this book, we present a survey-style overview of soft sets and their major extensions, highlighting core definitions, representative constructions, and key directions of current development. 

---
# Recovered in Translation: Efficient Pipeline for Automated Translation of Benchmarks and Datasets 

**Authors**: Hanna Yukhymenko, Anton Alexandrov, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2602.22207)  

**Abstract**: The reliability of multilingual Large Language Model (LLM) evaluation is currently compromised by the inconsistent quality of translated benchmarks. Existing resources often suffer from semantic drift and context loss, which can lead to misleading performance metrics. In this work, we present a fully automated framework designed to address these challenges by enabling scalable, high-quality translation of datasets and benchmarks. We demonstrate that adapting test-time compute scaling strategies, specifically Universal Self-Improvement (USI) and our proposed multi-round ranking method, T-RANK, allows for significantly higher quality outputs compared to traditional pipelines. Our framework ensures that benchmarks preserve their original task structure and linguistic nuances during localization. We apply this approach to translate popular benchmarks and datasets into eight Eastern and Southern European languages (Ukrainian, Bulgarian, Slovak, Romanian, Lithuanian, Estonian, Turkish, Greek). Evaluations using both reference-based metrics and LLM-as-a-judge show that our translations surpass existing resources, resulting in more accurate downstream model assessment. We release both the framework and the improved benchmarks to facilitate robust and reproducible multilingual AI development. 

---
# Off-The-Shelf Image-to-Image Models Are All You Need To Defeat Image Protection Schemes 

**Authors**: Xavier Pleimling, Sifat Muhammad Abdullah, Gunjan Balde, Peng Gao, Mainack Mondal, Murtuza Jadliwala, Bimal Viswanath  

**Link**: [PDF](https://arxiv.org/pdf/2602.22197)  

**Abstract**: Advances in Generative AI (GenAI) have led to the development of various protection strategies to prevent the unauthorized use of images. These methods rely on adding imperceptible protective perturbations to images to thwart misuse such as style mimicry or deepfake manipulations. Although previous attacks on these protections required specialized, purpose-built methods, we demonstrate that this is no longer necessary. We show that off-the-shelf image-to-image GenAI models can be repurposed as generic ``denoisers" using a simple text prompt, effectively removing a wide range of protective perturbations. Across 8 case studies spanning 6 diverse protection schemes, our general-purpose attack not only circumvents these defenses but also outperforms existing specialized attacks while preserving the image's utility for the adversary. Our findings reveal a critical and widespread vulnerability in the current landscape of image protection, indicating that many schemes provide a false sense of security. We stress the urgent need to develop robust defenses and establish that any future protection mechanism must be benchmarked against attacks from off-the-shelf GenAI models. Code is available in this repository: this https URL 

---
# GUI-Libra: Training Native GUI Agents to Reason and Act with Action-aware Supervision and Partially Verifiable RL 

**Authors**: Rui Yang, Qianhui Wu, Zhaoyang Wang, Hanyang Chen, Ke Yang, Hao Cheng, Huaxiu Yao, Baoling Peng, Huan Zhang, Jianfeng Gao, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.22190)  

**Abstract**: Open-source native GUI agents still lag behind closed-source systems on long-horizon navigation tasks. This gap stems from two limitations: a shortage of high-quality, action-aligned reasoning data, and the direct adoption of generic post-training pipelines that overlook the unique challenges of GUI agents. We identify two fundamental issues in these pipelines: (i) standard SFT with CoT reasoning often hurts grounding, and (ii) step-wise RLVR-tyle training faces partial verifiability, where multiple actions can be correct but only a single demonstrated action is used for verification. This makes offline step-wise metrics weak predictors of online task success. In this work, we present GUI-Libra, a tailored training recipe that addresses these challenges. First, to mitigate the scarcity of action-aligned reasoning data, we introduce a data construction and filtering pipeline and release a curated 81K GUI reasoning dataset. Second, to reconcile reasoning with grounding, we propose action-aware SFT that mixes reasoning-then-action and direct-action data and reweights tokens to emphasize action and grounding. Third, to stabilize RL under partial verifiability, we identify the overlooked importance of KL regularization in RLVR and show that a KL trust region is critical for improving offline-to-online predictability; we further introduce success-adaptive scaling to downweight unreliable negative gradients. Across diverse web and mobile benchmarks, GUI-Libra consistently improves both step-wise accuracy and end-to-end task completion. Our results suggest that carefully designed post-training and data curation can unlock significantly stronger task-solving capabilities without costly online data collection. We release our dataset, code, and models to facilitate further research on data-efficient post-training for reasoning-capable GUI agents. 

---
# Surrogate models for Rock-Fluid Interaction: A Grid-Size-Invariant Approach 

**Authors**: Nathalie C. Pinheiro, Donghu Guo, Hannah P. Menke, Aniket C. Joshi, Claire E. Heaney, Ahmed H. ElSheikh, Christopher C. Pain  

**Link**: [PDF](https://arxiv.org/pdf/2602.22188)  

**Abstract**: Modelling rock-fluid interaction requires solving a set of partial differential equations (PDEs) to predict the flow behaviour and the reactions of the fluid with the rock on the interfaces. Conventional high-fidelity numerical models require a high resolution to obtain reliable results, resulting in huge computational expense. This restricts the applicability of these models for multi-query problems, such as uncertainty quantification and optimisation, which require running numerous scenarios. As a cheaper alternative to high-fidelity models, this work develops eight surrogate models for predicting the fluid flow in porous media. Four of these are reduced-order models (ROM) based on one neural network for compression and another for prediction. The other four are single neural networks with the property of grid-size invariance; a term which we use to refer to image-to-image models that are capable of inferring on computational domains that are larger than those used during training. In addition to the novel grid-size-invariant framework for surrogate models, we compare the predictive performance of UNet and UNet++ architectures, and demonstrate that UNet++ outperforms UNet for surrogate models. Furthermore, we show that the grid-size-invariant approach is a reliable way to reduce memory consumption during training, resulting in good correlation between predicted and ground-truth values and outperforming the ROMs analysed. The application analysed is particularly challenging because fluid-induced rock dissolution results in a non-static solid field and, consequently, it cannot be used to help in adjustments of the future prediction. 

---
# Enhancing Framingham Cardiovascular Risk Score Transparency through Logic-Based XAI 

**Authors**: Emannuel L. de A. Bezerra, Luiz H. T. Viana, Vinícius P. Chagas, Diogo E. Rolim, Thiago Alves Rocha, Carlos H. L. Cavalcante  

**Link**: [PDF](https://arxiv.org/pdf/2602.22149)  

**Abstract**: Cardiovascular disease (CVD) remains one of the leading global health challenges, accounting for more than 19 million deaths worldwide. To address this, several tools that aim to predict CVD risk and support clinical decision making have been developed. In particular, the Framingham Risk Score (FRS) is one of the most widely used and recommended worldwide. However, it does not explain why a patient was assigned to a particular risk category nor how it can be reduced. Due to this lack of transparency, we present a logical explainer for the FRS. Based on first-order logic and explainable artificial intelligence (XAI) fundaments, the explainer is capable of identifying a minimal set of patient attributes that are sufficient to explain a given risk classification. Our explainer also produces actionable scenarios that illustrate which modifiable variables would reduce a patient's risk category. We evaluated all possible input combinations of the FRS (over 22,000 samples) and tested them with our explainer, successfully identifying important risk factors and suggesting focused interventions for each case. The results may improve clinician trust and facilitate a wider implementation of CVD risk assessment by converting opaque scores into transparent and prescriptive insights, particularly in areas with restricted access to specialists. 

---
# Provable Last-Iterate Convergence for Multi-Objective Safe LLM Alignment via Optimistic Primal-Dual 

**Authors**: Yining Li, Peizhong Ju, Ness Shroff  

**Link**: [PDF](https://arxiv.org/pdf/2602.22146)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) plays a significant role in aligning Large Language Models (LLMs) with human preferences. While RLHF with expected reward constraints can be formulated as a primal-dual optimization problem, standard primal-dual methods only guarantee convergence with a distributional policy where the saddle-point problem is in convex-concave form. Moreover, standard primal-dual methods may exhibit instability or divergence in the last iterate under policy parameterization in practical applications. In this work, we propose a universal primal-dual framework for safe RLHF that unifies a broad class of existing alignment algorithms, including safe-RLHF, one-shot, and multi-shot based methods. Building on this framework, we introduce an optimistic primal-dual (OPD) algorithm that incorporates predictive updates for both primal and dual variables to stabilize saddle-point dynamics. We establish last-iterate convergence guarantees for the proposed method, covering both exact policy optimization in the distributional space and convergence to a neighborhood of the optimal solution whose gap is related to approximation error and bias under parameterized policies. Our analysis reveals that optimism plays a crucial role in mitigating oscillations inherent to constrained alignment objectives, thereby closing a key theoretical gap between constrained RL and practical RLHF. 

---
# When AI Writes, Whose Voice Remains? Quantifying Cultural Marker Erasure Across World English Varieties in Large Language Models 

**Authors**: Satyam Kumar Navneet, Joydeep Chandra, Yong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.22145)  

**Abstract**: Large Language Models (LLMs) are increasingly used to ``professionalize'' workplace communication, often at the cost of linguistic identity. We introduce "Cultural Ghosting", the systematic erasure of linguistic markers unique to non-native English varieties during text processing. Through analysis of 22,350 LLM outputs generated from 1,490 culturally marked texts (Indian, Singaporean,& Nigerian English) processed by five models under three prompt conditions, we quantify this phenomenon using two novel metrics: Identity Erasure Rate (IER) & Semantic Preservation Score (SPS). Across all prompts, we find an overall IER of 10.26%, with model-level variation from 3.5% to 20.5% (5.9x range). Crucially, we identify a Semantic Preservation Paradox: models maintain high semantic similarity (mean SPS = 0.748) while systematically erasing cultural markers. Pragmatic markers (politeness conventions) are 1.9x more vulnerable than lexical markers (71.5% vs. 37.1% erasure). Our experiments demonstrate that explicit cultural-preservation prompts reduce erasure by 29% without sacrificing semantic quality. 

---
# NoLan: Mitigating Object Hallucinations in Large Vision-Language Models via Dynamic Suppression of Language Priors 

**Authors**: Lingfeng Ren, Weihao Yu, Runpeng Yu, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.22144)  

**Abstract**: Object hallucination is a critical issue in Large Vision-Language Models (LVLMs), where outputs include objects that do not appear in the input image. A natural question arises from this phenomenon: Which component of the LVLM pipeline primarily contributes to object hallucinations? The vision encoder to perceive visual information, or the language decoder to generate text responses? In this work, we strive to answer this question through designing a systematic experiment to analyze the roles of the vision encoder and the language decoder in hallucination generation. Our observations reveal that object hallucinations are predominantly associated with the strong priors from the language decoder. Based on this finding, we propose a simple and training-free framework, No-Language-Hallucination Decoding, NoLan, which refines the output distribution by dynamically suppressing language priors, modulated based on the output distribution difference between multimodal and text-only inputs. Experimental results demonstrate that NoLan effectively reduces object hallucinations across various LVLMs on different tasks. For instance, NoLan achieves substantial improvements on POPE, enhancing the accuracy of LLaVA-1.5 7B and Qwen-VL 7B by up to 6.45 and 7.21, respectively. The code is publicly available at: this https URL. 

---
# SWE-Protégé: Learning to Selectively Collaborate With an Expert Unlocks Small Language Models as Software Engineering Agents 

**Authors**: Patrick Tser Jern Kon, Archana Pradeep, Ang Chen, Alexander P. Ellis, Warren Hunt, Zijian Wang, John Yang, Samuel Thompson  

**Link**: [PDF](https://arxiv.org/pdf/2602.22124)  

**Abstract**: Small language models (SLMs) offer compelling advantages in cost, latency, and adaptability, but have so far lagged behind larger models on long-horizon software engineering tasks such as SWE-bench, where they suffer from pervasive action looping and low resolution rates. We introduce SWE-Protégé, a post-training framework that reframes software repair as an expert-protégé collaboration problem. In SWE-Protégé, an SLM remains the sole decision-maker while learning to selectively seek guidance from a strong expert model, recognize stalled states, and follow through on expert feedback. Our approach combines supervised fine-tuning on expert-augmented trajectories with agentic reinforcement learning that explicitly discourages degenerative looping and unproductive expert collaboration. We lightly post-train Qwen2.5-Coder-7B-Instruct to achieve 42.4% Pass@1 on SWE-bench Verified, a +25.4% improvement over the prior SLM state of the art, while using expert assistance sparsely (~4 calls per task and 11% of total tokens). 

---
# Don't stop me now: Rethinking Validation Criteria for Model Parameter Selection 

**Authors**: Andrea Apicella, Francesco Isgrò, Andrea Pollastro, Roberto Prevete  

**Link**: [PDF](https://arxiv.org/pdf/2602.22107)  

**Abstract**: Despite the extensive literature on training loss functions, the evaluation of generalization on the validation set remains underexplored. In this work, we conduct a systematic empirical and statistical study of how the validation criterion used for model selection affects test performance in neural classifiers, with attention to early stopping. Using fully connected networks on standard benchmarks under $k$-fold evaluation, we compare: (i) early stopping with patience and (ii) post-hoc selection over all epochs (i.e. no early stopping). Models are trained with cross-entropy, C-Loss, or PolyLoss; the model parameter selection on the validation set is made using accuracy or one of the three loss functions, each considered independently. Three main findings emerge. (1) Early stopping based on validation accuracy performs worst, consistently selecting checkpoints with lower test accuracy than both loss-based early stopping and post-hoc selection. (2) Loss-based validation criteria yield comparable and more stable test accuracy. (3) Across datasets and folds, any single validation rule often underperforms the test-optimal checkpoint. Overall, the selected model typically achieves test-set performance statistically lower than the best performance across all epochs, regardless of the validation criterion. Our results suggest avoiding validation accuracy (in particular with early stopping) for parameter selection, favoring loss-based validation criteria. 

---
# On Imbalanced Regression with Hoeffding Trees 

**Authors**: Pantia-Marina Alchirch, Dimitrios I. Diochnos  

**Link**: [PDF](https://arxiv.org/pdf/2602.22101)  

**Abstract**: Many real-world applications provide a continuous stream of data that is subsequently used by machine learning models to solve regression tasks of interest. Hoeffding trees and their variants have a long-standing tradition due to their effectiveness, either alone or as base models in broader ensembles. At the same time a recent line of work in batch learning has shown that kernel density estimation (KDE) is an effective approach for smoothed predictions in imbalanced regression tasks [Yang et al., 2021]. Moreover, another recent line of work for batch learning, called hierarchical shrinkage (HS) [Agarwal et al., 2022], has introduced a post-hoc regularization method for decision trees that does not alter the structure of the learned tree. Using a telescoping argument we cast KDE to streaming environments and extend the implementation of HS to incremental decision tree models. Armed with these extensions we investigate the performance of decision trees that may enjoy such options in datasets commonly used for regression in online settings. We conclude that KDE is beneficial in the early parts of the stream, while HS hardly, if ever, offers performance benefits. Our code is publicly available at: this https URL. 

---
# Understanding Artificial Theory of Mind: Perturbed Tasks and Reasoning in Large Language Models 

**Authors**: Christian Nickel, Laura Schrewe, Florian Mai, Lucie Flek  

**Link**: [PDF](https://arxiv.org/pdf/2602.22072)  

**Abstract**: Theory of Mind (ToM) refers to an agent's ability to model the internal states of others. Contributing to the debate whether large language models (LLMs) exhibit genuine ToM capabilities, our study investigates their ToM robustness using perturbations on false-belief tasks and examines the potential of Chain-of-Thought prompting (CoT) to enhance performance and explain the LLM's decision. We introduce a handcrafted, richly annotated ToM dataset, including classic and perturbed false belief tasks, the corresponding spaces of valid reasoning chains for correct task completion, subsequent reasoning faithfulness, task solutions, and propose metrics to evaluate reasoning chain correctness and to what extent final answers are faithful to reasoning traces of the generated CoT. We show a steep drop in ToM capabilities under task perturbation for all evaluated LLMs, questioning the notion of any robust form of ToM being present. While CoT prompting improves the ToM performance overall in a faithful manner, it surprisingly degrades accuracy for some perturbation classes, indicating that selective application is necessary. 

---
# DualWeaver: Synergistic Feature Weaving Surrogates for Multivariate Forecasting with Univariate Time Series Foundation Models 

**Authors**: Jinpeng Li, Zhongyi Pei, Huaze Xue, Bojian Zheng, Chen Wang, Jianmin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.22066)  

**Abstract**: Time-series foundation models (TSFMs) have achieved strong univariate forecasting through large-scale pre-training, yet effectively extending this success to multivariate forecasting remains challenging. To address this, we propose DualWeaver, a novel framework that adapts univariate TSFMs (Uni-TSFMs) for multivariate forecasting by using a pair of learnable, structurally symmetric surrogate series. Generated by a shared auxiliary feature-fusion module that captures cross-variable dependencies, these surrogates are mapped to TSFM-compatible series via the forecasting objective. The symmetric structure enables parameter-free reconstruction of final predictions directly from the surrogates, without additional parametric decoding. A theoretically grounded regularization term is further introduced to enhance robustness against adaptation collapse. Extensive experiments on diverse real-world datasets show that DualWeaver outperforms state-of-the-art multivariate forecasters in both accuracy and stability. We release the code at this https URL. 

---
# NESTOR: A Nested MOE-based Neural Operator for Large-Scale PDE Pre-Training 

**Authors**: Dengdi Sun, Xiaoya Zhou, Xiao Wang, Hao Si, Wanli Lyu, Jin Tang, Bin Luo  

**Link**: [PDF](https://arxiv.org/pdf/2602.22059)  

**Abstract**: Neural operators have emerged as an efficient paradigm for solving PDEs, overcoming the limitations of traditional numerical methods and significantly improving computational efficiency. However, due to the diversity and complexity of PDE systems, existing neural operators typically rely on a single network architecture, which limits their capacity to fully capture heterogeneous features and complex system dependencies. This constraint poses a bottleneck for large-scale PDE pre-training based on neural operators. To address these challenges, we propose a large-scale PDE pre-trained neural operator based on a nested Mixture-of-Experts (MoE) framework. In particular, the image-level MoE is designed to capture global dependencies, while the token-level Sub-MoE focuses on local dependencies. Our model can selectively activate the most suitable expert networks for a given input, thereby enhancing generalization and transferability. We conduct large-scale pre-training on twelve PDE datasets from diverse sources and successfully transfer the model to downstream tasks. Extensive experiments demonstrate the effectiveness of our approach. 

---
# Physics-Informed Machine Learning for Vessel Shaft Power and Fuel Consumption Prediction: Interpretable KAN-based Approach 

**Authors**: Hamza Haruna Mohammed, Dusica Marijan, Arnbjørn Maressa  

**Link**: [PDF](https://arxiv.org/pdf/2602.22055)  

**Abstract**: Accurate prediction of shaft rotational speed, shaft power, and fuel consumption is crucial for enhancing operational efficiency and sustainability in maritime transportation. Conventional physics-based models provide interpretability but struggle with real-world variability, while purely data-driven approaches achieve accuracy at the expense of physical plausibility. This paper introduces a Physics-Informed Kolmogorov-Arnold Network (PI-KAN), a hybrid method that integrates interpretable univariate feature transformations with a physics-informed loss function and a leakage-free chained prediction pipeline. Using operational and environmental data from five cargo vessels, PI-KAN consistently outperforms the traditional polynomial method and neural network baselines. The model achieves the lowest mean absolute error (MAE) and root mean squared error (RMSE), and the highest coefficient of determination (R^2) for shaft power and fuel consumption across all vessels, while maintaining physically consistent behavior. Interpretability analysis reveals rediscovery of domain-consistent dependencies, such as cubic-like speed-power relationships and cosine-like wave and wind effects. These results demonstrate that PI-KAN achieves both predictive accuracy and interpretability, offering a robust tool for vessel performance monitoring and decision support in operational settings. 

---
# TG-ASR: Translation-Guided Learning with Parallel Gated Cross Attention for Low-Resource Automatic Speech Recognition 

**Authors**: Cheng-Yeh Yang, Chien-Chun Wang, Li-Wei Chen, Hung-Shin Lee, Hsin-Min Wang, Berlin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.22039)  

**Abstract**: Low-resource automatic speech recognition (ASR) continues to pose significant challenges, primarily due to the limited availability of transcribed data for numerous languages. While a wealth of spoken content is accessible in television dramas and online videos, Taiwanese Hokkien exemplifies this issue, with transcriptions often being scarce and the majority of available subtitles provided only in Mandarin. To address this deficiency, we introduce TG-ASR for Taiwanese Hokkien drama speech recognition, a translation-guided ASR framework that utilizes multilingual translation embeddings to enhance recognition performance in low-resource environments. The framework is centered around the parallel gated cross-attention (PGCA) mechanism, which adaptively integrates embeddings from various auxiliary languages into the ASR decoder. This mechanism facilitates robust cross-linguistic semantic guidance while ensuring stable optimization and minimizing interference between languages. To support ongoing research initiatives, we present YT-THDC, a 30-hour corpus of Taiwanese Hokkien drama speech with aligned Mandarin subtitles and manually verified Taiwanese Hokkien transcriptions. Comprehensive experiments and analyses identify the auxiliary languages that most effectively enhance ASR performance, achieving a 14.77% relative reduction in character error rate and demonstrating the efficacy of translation-guided learning for underrepresented languages in practical applications. 

---
# RGB-Event HyperGraph Prompt for Kilometer Marker Recognition based on Pre-trained Foundation Models 

**Authors**: Xiaoyu Xian, Shiao Wang, Xiao Wang, Daxin Tian, Yan Tian  

**Link**: [PDF](https://arxiv.org/pdf/2602.22026)  

**Abstract**: Metro trains often operate in highly complex environments, characterized by illumination variations, high-speed motion, and adverse weather conditions. These factors pose significant challenges for visual perception systems, especially those relying solely on conventional RGB cameras. To tackle these difficulties, we explore the integration of event cameras into the perception system, leveraging their advantages in low-light conditions, high-speed scenarios, and low power consumption. Specifically, we focus on Kilometer Marker Recognition (KMR), a critical task for autonomous metro localization under GNSS-denied conditions. In this context, we propose a robust baseline method based on a pre-trained RGB OCR foundation model, enhanced through multi-modal adaptation. Furthermore, we construct the first large-scale RGB-Event dataset, EvMetro5K, containing 5,599 pairs of synchronized RGB-Event samples, split into 4,479 training and 1,120 testing samples. Extensive experiments on EvMetro5K and other widely used benchmarks demonstrate the effectiveness of our approach for KMR. Both the dataset and source code will be released on this https URL 

---
# Enhancing LLM-Based Test Generation by Eliminating Covered Code 

**Authors**: WeiZhe Xu, Mengyu Liu, Fanxin Kong  

**Link**: [PDF](https://arxiv.org/pdf/2602.21997)  

**Abstract**: Automated test generation is essential for software quality assurance, with coverage rate serving as a key metric to ensure thorough testing. Recent advancements in Large Language Models (LLMs) have shown promise in improving test generation, particularly in achieving higher coverage. However, while existing LLM-based test generation solutions perform well on small, isolated code snippets, they struggle when applied to complex methods under test. To address these issues, we propose a scalable LLM-based unit test generation method. Our approach consists of two key steps. The first step is context information retrieval, which uses both LLMs and static analysis to gather relevant contextual information associated with the complex methods under test. The second step, iterative test generation with code elimination, repeatedly generates unit tests for the code slice, tracks the achieved coverage, and selectively removes code segments that have already been covered. This process simplifies the testing task and mitigates issues arising from token limits or reduced reasoning effectiveness associated with excessively long contexts. Through comprehensive evaluations on open-source projects, our approach outperforms state-of-the-art LLM-based and search-based methods, demonstrating its effectiveness in achieving high coverage on complex methods. 

---
# PatchDenoiser: Parameter-efficient multi-scale patch learning and fusion denoiser for medical images 

**Authors**: Jitindra Fartiyal, Pedro Freire, Sergei K. Turitsyn, Sergei G. Solovski  

**Link**: [PDF](https://arxiv.org/pdf/2602.21987)  

**Abstract**: Medical images are essential for diagnosis, treatment planning, and research, but their quality is often degraded by noise from low-dose acquisition, patient motion, or scanner limitations, affecting both clinical interpretation and downstream analysis. Traditional filtering approaches often over-smooth and lose fine anatomical details, while deep learning methods, including CNNs, GANs, and transformers, may struggle to preserve such details or require large, computationally expensive models, limiting clinical practicality.
We propose PatchDenoiser, a lightweight, energy-efficient multi-scale patch-based denoising framework. It decomposes denoising into local texture extraction and global context aggregation, fused via a spatially aware patch fusion strategy. This design enables effective noise suppression while preserving fine structural and anatomical details. PatchDenoiser is ultra-lightweight, with far fewer parameters and lower computational complexity than CNN-, GAN-, and transformer-based denoisers.
On the 2016 Mayo Low-Dose CT dataset, PatchDenoiser consistently outperforms state-of-the-art CNN- and GAN-based methods in PSNR and SSIM. It is robust to variations in slice thickness, reconstruction kernels, and HU windows, generalizes across scanners without fine-tuning, and reduces parameters by ~9x and energy consumption per inference by ~27x compared with conventional CNN denoisers.
PatchDenoiser thus provides a practical, scalable, and computationally efficient solution for medical image denoising, balancing performance, robustness, and clinical deployability. 

---
# Hidden Topics: Measuring Sensitive AI Beliefs with List Experiments 

**Authors**: Maxim Chupilkin  

**Link**: [PDF](https://arxiv.org/pdf/2602.21939)  

**Abstract**: How can researchers identify beliefs that large language models (LLMs) hide? As LLMs become more sophisticated and the prevalence of alignment faking increases, combined with their growing integration into high-stakes decision-making, responding to this challenge has become critical. This paper proposes that a list experiment, a simple method widely used in the social sciences, can be applied to study the hidden beliefs of LLMs. List experiments were originally developed to circumvent social desirability bias in human respondents, which closely parallels alignment faking in LLMs. The paper implements a list experiment on models developed by Anthropic, Google, and OpenAI and finds hidden approval of mass surveillance across all models, as well as some approval of torture, discrimination, and first nuclear strike. Importantly, a placebo treatment produces a null result, validating the method. The paper then compares list experiments with direct questioning and discusses the utility of the approach. 

---
# A Framework for Cross-Domain Generalization in Coronary Artery Calcium Scoring Across Gated and Non-Gated Computed Tomography 

**Authors**: Mahmut S. Gokmen, Moneera N. Haque, Steve W. Leung, Caroline N. Leach, Seth Parker, Stephen B. Hobbs, Vincent L. Sorrell, W. Brent Seales, V. K. Cody Bumgardner  

**Link**: [PDF](https://arxiv.org/pdf/2602.21935)  

**Abstract**: Coronary artery calcium (CAC) scoring is a key predictor of cardiovascular risk, but it relies on ECG-gated CT scans, restricting its use to specialized cardiac imaging settings. We introduce an automated framework for CAC detection and lesion-specific Agatston scoring that operates across both gated and non-gated CT scans. At its core is CARD-ViT, a self-supervised Vision Transformer trained exclusively on gated CT data using DINO. Without any non-gated training data, our framework achieves 0.707 accuracy and a Cohen's kappa of 0.528 on the Stanford non-gated dataset, matching models trained directly on non-gated scans. On gated test sets, the framework achieves 0.910 accuracy with Cohen's kappa scores of 0.871 and 0.874 across independent datasets, demonstrating robust risk stratification. These results demonstrate the feasibility of cross-domain CAC scoring from gated to non-gated domains, supporting scalable cardiovascular screening in routine chest imaging without additional scans or annotations. 

---
# DynamicGTR: Leveraging Graph Topology Representation Preferences to Boost VLM Capabilities on Graph QAs 

**Authors**: Yanbin Wei, Jiangyue Yan, Chun Kang, Yang Chen, Hua Liu, James Kwok, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.21864)  

**Abstract**: Vision-Language Models (VLMs) have emerged as versatile solutions for zero-shot question answering (QA) across various domains. However, enabling VLMs to effectively comprehend structured graphs and perform accurate, efficient QA remains challenging. Existing approaches typically rely on one single graph topology representation (GTR), such as fixed-style visual images or unified text descriptions. This ``one-size-fits-all'' strategy often neglects model-specific and task-specific preferences, resulting in inaccurate or over-lengthy responses to graph-related queries. To address this, we propose the $\mbox{DynamicGTR}$ framework, which dynamically selects the optimal GTR for each query during inference, thereby enhancing the zero-shot graph QA capabilities of VLMs with a customizable accuracy and brevity trade-off. Extensive experiments show that DynamicGTR not only improves VLM-based graph algorithm QA performance but also successfully transfers the experience trained from synthetic graph algorithm tasks to real-world applications like link prediction and node classification, without any additional training. Additionally, DynamicGTR demonstrates strong transferability across tasks, domains, and models, suggesting its potential as a flexible solution for broad graph scenarios. 

---
# Understanding Annotation Error Propagation and Learning an Adaptive Policy for Expert Intervention in Barrett's Video Segmentation 

**Authors**: Lokesha Rasanjalee, Jin Lin Tan, Dileepa Pitawela, Rajvinder Singh, Hsiang-Ting Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.21855)  

**Abstract**: Accurate annotation of endoscopic videos is essential yet time-consuming, particularly for challenging datasets such as dysplasia in Barrett's esophagus, where the affected regions are irregular and lack clear boundaries. Semi-automatic tools like Segment Anything Model 2 (SAM2) can ease this process by propagating annotations across frames, but small errors often accumulate and reduce accuracy, requiring expert review and correction. To address this, we systematically study how annotation errors propagate across different prompt types, namely masks, boxes, and points, and propose Learning-to-Re-Prompt (L2RP), a cost-aware framework that learns when and where to seek expert input. By tuning a human-cost parameter, our method balances annotation effort and segmentation accuracy. Experiments on a private Barrett's dysplasia dataset and the public SUN-SEG benchmark demonstrate improved temporal consistency and superior performance over baseline strategies. 

---
# xai-cola: A Python library for sparsifying counterfactual explanations 

**Authors**: Lin Zhu, Lei You  

**Link**: [PDF](https://arxiv.org/pdf/2602.21845)  

**Abstract**: Counterfactual explanation (CE) is an important domain within post-hoc explainability. However, the explanations generated by most CE generators are often highly redundant. This work introduces an open-source Python library xai-cola, which provides an end-to-end pipeline for sparsifying CEs produced by arbitrary generators, reducing superfluous feature changes while preserving their validity. It offers a documented API that takes as input raw tabular data in pandas DataFrame form, a preprocessing object (for standardization and encoding), and a trained scikit-learn or PyTorch model. On this basis, users can either employ the built-in or externally imported CE generators. The library also implements several sparsification policies and includes visualization routines for analysing and comparing sparsified counterfactuals. xai-cola is released under the MIT license and can be installed from PyPI. Empirical experiments indicate that xai-cola produces sparser counterfactuals across several CE generators, reducing the number of modified features by up to 50% in our setting. The source code is available at this https URL. 

---
# Resilient Federated Chain: Transforming Blockchain Consensus into an Active Defense Layer for Federated Learning 

**Authors**: Mario García-Márquez, Nuria Rodríguez-Barroso, M.Victoria Luzón, Francisco Herrera  

**Link**: [PDF](https://arxiv.org/pdf/2602.21841)  

**Abstract**: Federated Learning (FL) has emerged as a key paradigm for building Trustworthy AI systems by enabling privacy-preserving, decentralized model training. However, FL is highly susceptible to adversarial attacks that compromise model integrity and data confidentiality, a vulnerability exacerbated by the fact that conventional data inspection methods are incompatible with its decentralized design. While integrating FL with Blockchain technology has been proposed to address some limitations, its potential for mitigating adversarial attacks remains largely unexplored. This paper introduces Resilient Federated Chain (RFC), a novel blockchain-enabled FL framework designed specifically to enhance resilience against such threats. RFC builds upon the existing Proof of Federated Learning architecture by repurposing the redundancy of its Pooled Mining mechanism as an active defense layer that can be combined with robust aggregation rules. Furthermore, the framework introduces a flexible evaluation function in its consensus mechanism, allowing for adaptive defense against different attack strategies. Extensive experimental evaluation on image classification tasks under various adversarial scenarios, demonstrates that RFC significantly improves robustness compared to baseline methods, providing a viable solution for securing decentralized learning environments. 

---
# StoryMovie: A Dataset for Semantic Alignment of Visual Stories with Movie Scripts and Subtitles 

**Authors**: Daniel Oliveira, David Martins de Matos  

**Link**: [PDF](https://arxiv.org/pdf/2602.21829)  

**Abstract**: Visual storytelling models that correctly ground entities in images may still hallucinate semantic relationships, generating incorrect dialogue attribution, character interactions, or emotional states. We introduce StoryMovie, a dataset of 1,757 stories aligned with movie scripts and subtitles through LCS matching. Our alignment pipeline synchronizes screenplay dialogue with subtitle timestamps, enabling dialogue attribution by linking character names from scripts to temporal positions from subtitles. Using this aligned content, we generate stories that maintain visual grounding tags while incorporating authentic character names, dialogue, and relationship dynamics. We fine-tune Qwen Storyteller3 on this dataset, building on prior work in visual grounding and entity re-identification. Evaluation using DeepSeek V3 as judge shows that Storyteller3 achieves an 89.9% win rate against base Qwen2.5-VL 7B on subtitle alignment. Compared to Storyteller, trained without script grounding,
Storyteller3 achieves 48.5% versus 38.0%, confirming that semantic alignment progressively improves dialogue attribution beyond visual grounding alone. 

---
# SemVideo: Reconstructs What You Watch from Brain Activity via Hierarchical Semantic Guidance 

**Authors**: Minghan Yang, Lan Yang, Ke Li, Honggang Zhang, Kaiyue Pang, Yizhe Song  

**Link**: [PDF](https://arxiv.org/pdf/2602.21819)  

**Abstract**: Reconstructing dynamic visual experiences from brain activity provides a compelling avenue for exploring the neural mechanisms of human visual perception. While recent progress in fMRI-based image reconstruction has been notable, extending this success to video reconstruction remains a significant challenge. Current fMRI-to-video reconstruction approaches consistently encounter two major shortcomings: (i) inconsistent visual representations of salient objects across frames, leading to appearance mismatches; (ii) poor temporal coherence, resulting in motion misalignment or abrupt frame transitions. To address these limitations, we introduce SemVideo, a novel fMRI-to-video reconstruction framework guided by hierarchical semantic information. At the core of SemVideo is SemMiner, a hierarchical guidance module that constructs three levels of semantic cues from the original video stimulus: static anchor descriptions, motion-oriented narratives, and holistic summaries. Leveraging this semantic guidance, SemVideo comprises three key components: a Semantic Alignment Decoder that aligns fMRI signals with CLIP-style embeddings derived from SemMiner, a Motion Adaptation Decoder that reconstructs dynamic motion patterns using a novel tripartite attention fusion architecture, and a Conditional Video Render that leverages hierarchical semantic guidance for video reconstruction. Experiments conducted on the CC2017 and HCP datasets demonstrate that SemVideo achieves superior performance in both semantic alignment and temporal consistency, setting a new state-of-the-art in fMRI-to-video reconstruction. 

---
# An Evaluation of Context Length Extrapolation in Long Code via Positional Embeddings and Efficient Attention 

**Authors**: Madhusudan Ghosh, Rishabh Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2602.21800)  

**Abstract**: The rapid advancement of large language models (LLMs) has led to a significant increase in automated tools in the software engineering, capable of performing various code-related tasks such as code generation, completion, and translation. Despite these advancements, its effectiveness is constrained by fixed context lengths, limiting its ability to generalize across long, domain-specific code sequences. To address this challenge, we investigate zero-shot, inference-only methods aimed at improving position encodings and optimizing attention mechanisms. Our goal is to provide a thorough analysis of current approaches that facilitate context length extrapolation in code, particularly in the context of long code completion tasks. 

---
# Excitation: Momentum For Experts 

**Authors**: Sagi Shaier  

**Link**: [PDF](https://arxiv.org/pdf/2602.21798)  

**Abstract**: We propose Excitation, a novel optimization framework designed to accelerate learning in sparse architectures such as Mixture-of-Experts (MoEs). Unlike traditional optimizers that treat all parameters uniformly, Excitation dynamically modulates updates using batch-level expert utilization. It introduces a competitive update dynamic that amplifies updates to highly-utilized experts and can selectively suppress low-utilization ones, effectively sharpening routing specialization. Notably, we identify a phenomenon of "structural confusion" in deep MoEs, where standard optimizers fail to establish functional signal paths; Excitation acts as a specialization catalyst, "rescuing" these models and enabling stable training where baselines remain trapped. Excitation is optimizer-, domain-, and model-agnostic, requires minimal integration effort, and introduces neither additional per-parameter optimizer state nor learnable parameters, making it highly viable for memory-constrained settings. Across language and vision tasks, Excitation consistently improves convergence speed and final performance in MoE models, indicating that active update modulation is a key mechanism for effective conditional computation. 

---
# Beyond Static Artifacts: A Forensic Benchmark for Video Deepfake Reasoning in Vision Language Models 

**Authors**: Zheyuan Gu, Qingsong Zhao, Yusong Wang, Zhaohong Huang, Xinqi Li, Cheng Yuan, Jiaowei Shao, Chi Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.21779)  

**Abstract**: Current Vision-Language Models (VLMs) for deepfake detection excel at identifying spatial artifacts but overlook a critical dimension: temporal inconsistencies in video forgeries. Adapting VLMs to reason about these dynamic cues remains a distinct challenge. To bridge this gap, we propose Forensic Answer-Questioning (FAQ), a large-scale benchmark that formulates temporal deepfake analysis as a multiple-choice task. FAQ introduces a three-level hierarchy to progressively evaluate and equip VLMs with forensic capabilities: (1) Facial Perception, testing the ability to identify static visual artifacts; (2) Temporal Deepfake Grounding, requiring the localization of dynamic forgery artifacts across frames; and (3) Forensic Reasoning, challenging models to synthesize evidence for final authenticity verdicts. We evaluate a range of VLMs on FAQ and generate a corresponding instruction-tuning set, FAQ-IT. Extensive experiments show that models fine-tuned on FAQ-IT achieve advanced performance on both in-domain and cross-dataset detection benchmarks. Ablation studies further validate the impact of our key design choices, confirming that FAQ is the driving force behind the temporal reasoning capabilities of these VLMs. 

---
# UniWhisper: Efficient Continual Multi-task Training for Robust Universal Audio Representation 

**Authors**: Yuxuan Chen, Peize He, Haoyuan Xu, Junzi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.21772)  

**Abstract**: A universal audio representation should capture fine-grained speech cues and high-level semantics for environmental sounds and music in a single encoder. Existing encoders often excel in one domain but degrade in others. We propose UniWhisper, an efficient continual multi-task training framework that casts heterogeneous audio tasks into a unified instruction and answer format. This enables standard next-token training without task-specific heads and losses. We train it on 38k hours of public audio and assess the encoder using shallow MLP probes and k-nearest neighbors (kNN) on 20 tasks spanning speech, environmental sound, and music. UniWhisper reaches normalized weighted averages of 0.81 with MLP probes and 0.61 with kNN, compared to 0.64 and 0.46 for Whisper, while retaining strong speech performance. 

---
# Generalisation of RLHF under Reward Shift and Clipped KL Regularisation 

**Authors**: Kenton Tang, Yuzhu Chen, Fengxiang He  

**Link**: [PDF](https://arxiv.org/pdf/2602.21765)  

**Abstract**: Alignment and adaptation in large language models heavily rely on reinforcement learning from human feedback (RLHF); yet, theoretical understanding of its generalisability remains premature, especially when the learned reward could shift, and the KL control is estimated and clipped. To address this issue, we develop generalisation theory for RLHF that explicitly accounts for (1) \emph{reward shift}: reward models are trained on preference data from earlier or mixed behaviour policies while RLHF optimises the current policy on its own rollouts; and (2) \emph{clipped KL regularisation}: the KL regulariser is estimated from sampled log-probability ratios and then clipped for stabilisation, resulting in an error to RLHF. We present generalisation bounds for RLHF, suggesting that the generalisation error stems from a sampling error from prompts and rollouts, a reward shift error, and a KL clipping error. We also discuss special cases of (1) initialising RLHF parameters with a uniform prior over a finite space, and (2) training RLHF by stochastic gradient descent, as an Ornstein-Uhlenbeck process. The theory yields practical implications in (1) optimal KL clipping threshold, and (2) budget allocation in prompts, rollouts, and preference data. 

---
# Learning from Yesterday's Error: An Efficient Online Learning Method for Traffic Demand Prediction 

**Authors**: Xiannan Huang, Quan Yuan, Chao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.21757)  

**Abstract**: Accurately predicting short-term traffic demand is critical for intelligent transportation systems. While deep learning models achieve strong performance under stationary conditions, their accuracy often degrades significantly when faced with distribution shifts caused by external events or evolving urban dynamics. Frequent model retraining to adapt to such changes incurs prohibitive computational costs, especially for large-scale or foundation models. To address this challenge, we propose FORESEE (Forecasting Online with Residual Smoothing and Ensemble Experts), a lightweight online adaptation framework that is accurate, robust, and computationally efficient. FORESEE operates without any parameter updates to the base model. Instead, it corrects today's forecast in each region using yesterday's prediction error, stabilized through exponential smoothing guided by a mixture-of-experts mechanism that adapts to recent error dynamics. Moreover, an adaptive spatiotemporal smoothing component propagates error signals across neighboring regions and time slots, capturing coherent shifts in demand patterns. Extensive experiments on seven real-world datasets with three backbone models demonstrate that FORESEE consistently improves prediction accuracy, maintains robustness even when distribution shifts are minimal (avoiding performance degradation), and achieves the lowest computational overhead among existing online methods. By enabling real-time adaptation of traffic forecasting models with negligible computational cost, FORESEE paves the way for deploying reliable, up-to-date prediction systems in dynamic urban environments. Code and data are available at this https URL 

---
# Evaluating the relationship between regularity and learnability in recursive numeral systems using Reinforcement Learning 

**Authors**: Andrea Silvi, Ponrawee Prasertsom, Jennifer Culbertson, Devdatt Dubhashi, Moa Johansson, Kenny Smith  

**Link**: [PDF](https://arxiv.org/pdf/2602.21720)  

**Abstract**: Human recursive numeral systems (i.e., counting systems such as English base-10 numerals), like many other grammatical systems, are highly regular. Following prior work that relates cross-linguistic tendencies to biases in learning, we ask whether regular systems are common because regularity facilitates learning. Adopting methods from the Reinforcement Learning literature, we confirm that highly regular human(-like) systems are easier to learn than unattested but possible irregular systems. This asymmetry emerges under the natural assumption that recursive numeral systems are designed for generalisation from limited data to represent all integers exactly. We also find that the influence of regularity on learnability is absent for unnatural, highly irregular systems, whose learnability is influenced instead by signal length, suggesting that different pressures may influence learnability differently in different parts of the space of possible numeral systems. Our results contribute to the body of work linking learnability to cross-linguistic prevalence. 

---
# Two-Stage Active Distribution Network Voltage Control via LLM-RL Collaboration: A Hybrid Knowledge-Data-Driven Approach 

**Authors**: Xu Yang, Chenhui Lin, Xiang Ma, Dong Liu, Ran Zheng, Haotian Liu, Wenchuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2602.21715)  

**Abstract**: The growing integration of distributed photovoltaics (PVs) into active distribution networks (ADNs) has exacerbated operational challenges, making it imperative to coordinate diverse equipment to mitigate voltage violations and enhance power quality. Although existing data-driven approaches have demonstrated effectiveness in the voltage control problem, they often require extensive trial-and-error exploration and struggle to incorporate heterogeneous information, such as day-ahead forecasts and semantic-based grid codes. Considering the operational scenarios and requirements in real-world ADNs, in this paper, we propose a hybrid knowledge-data-driven approach that leverages dynamic collaboration between a large language model (LLM) agent and a reinforcement learning (RL) agent to achieve two-stage voltage control. In the day-ahead stage, the LLM agent receives coarse region-level forecasts and generates scheduling strategies for on-load tap changer (OLTC) and shunt capacitors (SCs) to regulate the overall voltage profile. Then in the intra-day stage, based on accurate node-level measurements, the RL agent refines terminal voltages by deriving reactive power generation strategies for PV inverters. On top of the LLM-RL collaboration framework, we further propose a self-evolution mechanism for the LLM agent and a pretrain-finetune pipeline for the RL agent, effectively enhancing and coordinating the policies for both agents. The proposed approach not only aligns more closely with practical operational characteristics but also effectively utilizes the inherent knowledge and reasoning capabilities of the LLM agent, significantly improving training efficiency and voltage control performance. Comprehensive comparisons and ablation studies demonstrate the effectiveness of the proposed method. 

---
# SurGo-R1: Benchmarking and Modeling Contextual Reasoning for Operative Zone in Surgical Video 

**Authors**: Guanyi Qin, Xiaozhen Wang, Zhu Zhuo, Chang Han Low, Yuancan Xiao, Yibing Fu, Haofeng Liu, Kai Wang, Chunjiang Li, Yueming Jin  

**Link**: [PDF](https://arxiv.org/pdf/2602.21706)  

**Abstract**: Minimally invasive surgery has dramatically improved patient operative outcomes, yet identifying safe operative zones remains challenging in critical phases, requiring surgeons to integrate visual cues, procedural phase, and anatomical context under high cognitive load. Existing AI systems offer binary safety verification or static detection, ignoring the phase-dependent nature of intraoperative reasoning. We introduce ResGo, a benchmark of laparoscopic frames annotated with Go Zone bounding boxes and clinician-authored rationales covering phase, exposure quality reasoning, next action and risk reminder. We introduce evaluation metrics that treat correct grounding under incorrect phase as failures, revealing that most vision-language models cannot handle such tasks and perform poorly. We then present SurGo-R1, a model optimized via RLHF with a multi-turn phase-then-go architecture where the model first identifies the surgical phase, then generates reasoning and Go Zone coordinates conditioned on that context. On unseen procedures, SurGo-R1 achieves 76.6% phase accuracy, 32.7 mIoU, and 54.8% hardcore accuracy, a 6.6$\times$ improvement over the mainstream generalist VLMs. Code, model and benchmark will be available at this https URL 

---
# Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large Vision-Language Models 

**Authors**: Jianghao Yin, Qin Chen, Kedi Chen, Jie Zhou, Xingjiao Wu, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2602.21704)  

**Abstract**: Large Vision-Language Models (LVLMs) exhibit outstanding performance on vision-language tasks but struggle with hallucination problems. Through in-depth analysis of LVLM activation patterns, we reveal two key findings: 1) truthfulness and visual perception capabilities predominantly engage different subsets of attention heads within the model architecture; and 2) truthfulness steering vectors vary significantly across different semantic contexts. Based on these observations, we propose Dynamic Multimodal Activation Steering, a training-free approach for hallucination mitigation. Our method constructs a semantic-based truthfulness steering vector database and computes visual perception steering vectors, enabling context-aware interventions during inference by dynamically selecting the most relevant steering vectors based on input semantic similarity and applying them to the most influential attention heads. We conduct comprehensive experiments across multiple models and datasets, demonstrating that our approach significantly enhances model performance, outperforming existing state-of-the-art methods. 

---
# Hierarchical LLM-Based Multi-Agent Framework with Prompt Optimization for Multi-Robot Task Planning 

**Authors**: Tomoya Kawabe, Rin Takano  

**Link**: [PDF](https://arxiv.org/pdf/2602.21670)  

**Abstract**: Multi-robot task planning requires decomposing natural-language instructions into executable actions for heterogeneous robot teams. Conventional Planning Domain Definition Language (PDDL) planners provide rigorous guarantees but struggle to handle ambiguous or long-horizon missions, while large language models (LLMs) can interpret instructions and propose plans but may hallucinate or produce infeasible actions. We present a hierarchical multi-agent LLM-based planner with prompt optimization: an upper layer decomposes tasks and assigns them to lower-layer agents, which generate PDDL problems solved by a classical planner. When plans fail, the system applies TextGrad-inspired textual-gradient updates to optimize each agent's prompt and thereby improve planning accuracy. In addition, meta-prompts are learned and shared across agents within the same layer, enabling efficient prompt optimization in multi-agent settings. On the MAT-THOR benchmark, our planner achieves success rates of 0.95 on compound tasks, 0.84 on complex tasks, and 0.60 on vague tasks, improving over the previous state-of-the-art LaMMA-P by 2, 7, and 15 percentage points respectively. An ablation study shows that the hierarchical structure, prompt optimization, and meta-prompt sharing contribute roughly +59, +37, and +4 percentage points to the overall success rate. 

---
# Following the Diagnostic Trace: Visual Cognition-guided Cooperative Network for Chest X-Ray Diagnosis 

**Authors**: Shaoxuan Wu, Jingkun Chen, Chong Ma, Cong Shen, Xiao Zhang, Jun Feng  

**Link**: [PDF](https://arxiv.org/pdf/2602.21657)  

**Abstract**: Computer-aided diagnosis (CAD) has significantly advanced automated chest X-ray diagnosis but remains isolated from clinical workflows and lacks reliable decision support and interpretability. Human-AI collaboration seeks to enhance the reliability of diagnostic models by integrating the behaviors of controllable radiologists. However, the absence of interactive tools seamlessly embedded within diagnostic routines impedes collaboration, while the semantic gap between radiologists' decision-making patterns and model representations further limits clinical adoption. To overcome these limitations, we propose a visual cognition-guided collaborative network (VCC-Net) to achieve the cooperative diagnostic paradigm. VCC-Net centers on visual cognition (VC) and employs clinically compatible interfaces, such as eye-tracking or the mouse, to capture radiologists' visual search traces and attention patterns during diagnosis. VCC-Net employs VC as a spatial cognition guide, learning hierarchical visual search strategies to localize diagnostically key regions. A cognition-graph co-editing module subsequently integrates radiologist VC with model inference to construct a disease-aware graph. The module captures dependencies among anatomical regions and aligns model representations with VC-driven features, mitigating radiologist bias and facilitating complementary, transparent decision-making. Experiments on the public datasets SIIM-ACR, EGD-CXR, and self-constructed TB-Mouse dataset achieved classification accuracies of 88.40%, 85.05%, and 92.41%, respectively. The attention maps produced by VCC-Net exhibit strong concordance with radiologists' gaze distributions, demonstrating a mutual reinforcement of radiologist and model inference. The code is available at this https URL. 

---
# CCCaption: Dual-Reward Reinforcement Learning for Complete and Correct Image Captioning 

**Authors**: Zhijiang Tang, Linhua Wang, Jiaxin Qi, Weihao Jiang, Peng Hou, Anxiang Zeng, Jianqiang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.21655)  

**Abstract**: Image captioning remains a fundamental task for vision language understanding, yet ground-truth supervision still relies predominantly on human-annotated references. Because human annotations reflect subjective preferences and expertise, ground-truth captions are often incomplete or even incorrect, which in turn limits caption models. We argue that caption quality should be assessed by two objective aspects: completeness (does the caption cover all salient visual facts?) and correctness (are the descriptions true with respect to the image?). To this end, we introduce CCCaption: a dual-reward reinforcement learning framework with a dedicated fine-tuning corpus that explicitly optimizes these properties to generate \textbf{C}omplete and \textbf{C}orrect \textbf{Captions}. For completeness, we use diverse LVLMs to disentangle the image into a set of visual queries, and reward captions that answer more of these queries, with a dynamic query sampling strategy to improve training efficiency. For correctness, we penalize captions that contain hallucinations by validating the authenticity of sub-caption queries, which are derived from the caption decomposition. Our symmetric dual-reward optimization jointly maximizes completeness and correctness, guiding models toward captions that better satisfy these objective criteria. Extensive experiments across standard captioning benchmarks show consistent improvements, offering a principled path to training caption models beyond human-annotation imitation. 

---
# Sparsity Induction for Accurate Post-Training Pruning of Large Language Models 

**Authors**: Minhao Jiang, Zhikai Li, Xuewen Liu, Jing Zhang, Mengjuan Chen, Qingyi Gu  

**Link**: [PDF](https://arxiv.org/pdf/2602.21652)  

**Abstract**: Large language models have demonstrated capabilities in text generation, while their increasing parameter scales present challenges in computational and memory efficiency. Post-training sparsity (PTS), which reduces model cost by removing weights from dense networks, is an effective approach. However, native dense matrices lack high sparsity, making existing approaches that directly remove weights disrupt model states, resulting in unsatisfactory performance recovery even with post-tuning. We propose Sparsity Induction, which promotes models toward higher sparsity at both distribution and feature levels before pruning, to push the limits of PTS. At the distribution level, we enhance distributional sparsity through mathematically equivalent scaling transformations, which are fully absorbable and incur no extra parameters or inference-time overhead. At the feature level, we introduce Spectral Norm Loss to promote feature sparsity from a low-rank perspective. Experiments across diverse model architectures and tasks demonstrate that our method further enhances sparsity-friendliness, achieving superior pruning performance over existing approaches. 

---
# PPCR-IM: A System for Multi-layer DAG-based Public Policy Consequence Reasoning and Social Indicator Mapping 

**Authors**: Zichen Song, Weijia Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.21650)  

**Abstract**: Public policy decisions are typically justified using a narrow set of headline indicators, leaving many downstream social impacts unstructured and difficult to compare across policies. We propose PPCR-IM, a system for multi-layer DAG-based consequence reasoning and social indicator mapping that addresses this gap. Given a policy description and its context, PPCR-IM uses an LLM-driven, layer-wise generator to construct a directed acyclic graph of intermediate consequences, allowing child nodes to have multiple parents to capture joint influences. A mapping module then aligns these nodes to a fixed indicator set and assigns one of three qualitative impact directions: increase, decrease, or ambiguous change. For each policy episode, the system outputs a structured record containing the DAG, indicator mappings, and three evaluation measures: an expected-indicator coverage score, a discovery rate for overlooked but relevant indicators, and a relative focus ratio comparing the systems coverage to that of the government. PPCR-IM is available both as an online demo and as a configurable XLSX-to-JSON batch pipeline. 

---
# Mitigating Structural Noise in Low-Resource S2TT: An Optimized Cascaded Nepali-English Pipeline with Punctuation Restoration 

**Authors**: Tangsang Chongbang, Pranesh Pyara Shrestha, Amrit Sarki, Anku Jaiswal  

**Link**: [PDF](https://arxiv.org/pdf/2602.21647)  

**Abstract**: This paper presents and evaluates an optimized cascaded Nepali speech-to-English text translation (S2TT) system, focusing on mitigating structural noise introduced by Automatic Speech Recognition (ASR). We first establish highly proficient ASR and NMT components: a Wav2Vec2-XLS-R-300m model achieved a state-of-the-art 2.72% CER on OpenSLR-54, and a multi-stage fine-tuned MarianMT model reached a 28.32 BLEU score on the FLORES-200 benchmark. We empirically investigate the influence of punctuation loss, demonstrating that unpunctuated ASR output significantly degrades translation quality, causing a massive 20.7% relative BLEU drop on the FLORES benchmark. To overcome this, we propose and evaluate an intermediate Punctuation Restoration Module (PRM). The final S2TT pipeline was tested across three configurations on a custom dataset. The optimal configuration, which applied the PRM directly to ASR output, achieved a 4.90 BLEU point gain over the direct ASR-to-NMT baseline (BLEU 36.38 vs. 31.48). This improvement was validated by human assessment, which confirmed the optimized pipeline's superior Adequacy (3.673) and Fluency (3.804). This work validates that targeted punctuation restoration is the most effective intervention for mitigating structural noise in the Nepali S2TT pipeline. It establishes an optimized baseline and demonstrates a critical architectural insight for developing cascaded speech translation systems for similar low-resource languages. 

---
# Self-Correcting VLA: Online Action Refinement via Sparse World Imagination 

**Authors**: Chenyv Liu, Wentao Tan, Lei Zhu, Fengling Li, Jingjing Li, Guoli Yang, Heng Tao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2602.21633)  

**Abstract**: Standard vision-language-action (VLA) models rely on fitting statistical data priors, limiting their robust understanding of underlying physical dynamics. Reinforcement learning enhances physical grounding through exploration yet typically relies on external reward signals that remain isolated from the agent's internal states. World action models have emerged as a promising paradigm that integrates imagination and control to enable predictive planning. However, they rely on implicit context modeling, lacking explicit mechanisms for self-improvement. To solve these problems, we propose Self-Correcting VLA (SC-VLA), which achieve self-improvement by intrinsically guiding action refinement through sparse imagination. We first design sparse world imagination by integrating auxiliary predictive heads to forecast current task progress and future trajectory trends, thereby constraining the policy to encode short-term physical evolution. Then we introduce the online action refinement module to reshape progress-dependent dense rewards, adjusting trajectory orientation based on the predicted sparse future states. Evaluations on challenging robot manipulation tasks from simulation benchmarks and real-world settings demonstrate that SC-VLA achieve state-of-the-art performance, yielding the highest task throughput with 16% fewer steps and a 9% higher success rate than the best-performing baselines, alongside a 14% gain in real-world experiments. Code is available at this https URL. 

---
# Virtual Biopsy for Intracranial Tumors Diagnosis on MRI 

**Authors**: Xinzhe Luo, Shuai Shao, Yan Wang, Jiangtao Wang, Yutong Bai, Jianguo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.21613)  

**Abstract**: Deep intracranial tumors situated in eloquent brain regions controlling vital functions present critical diagnostic challenges. Clinical practice has shifted toward stereotactic biopsy for pathological confirmation before treatment. Yet biopsy carries inherent risks of hemorrhage and neurological deficits and struggles with sampling bias due to tumor spatial heterogeneity, because pathological changes are typically region-selective rather than tumor-wide. Therefore, advancing non-invasive MRI-based pathology prediction is essential for holistic tumor assessment and modern clinical decision-making.
The primary challenge lies in data scarcity: low tumor incidence requires long collection cycles, and annotation demands biopsy-verified pathology from neurosurgical experts. Additionally, tiny lesion volumes lacking segmentation masks cause critical features to be overwhelmed by background noise. To address these challenges, we construct the ICT-MRI dataset - the first public biopsy-verified benchmark with 249 cases across four categories. We propose a Virtual Biopsy framework comprising: MRI-Processor for standardization; Tumor-Localizer employing vision-language models for coarse-to-fine localization via weak supervision; and Adaptive-Diagnoser with a Masked Channel Attention mechanism fusing local discriminative features with global contexts. Experiments demonstrate over 90% accuracy, outperforming baselines by more than 20%. 

---
# Structurally Aligned Subtask-Level Memory for Software Engineering Agents 

**Authors**: Kangning Shen, Jingyuan Zhang, Chenxi Sun, Wencong Zeng, Yang Yue  

**Link**: [PDF](https://arxiv.org/pdf/2602.21611)  

**Abstract**: Large Language Models (LLMs) have demonstrated significant potential as autonomous software engineering (SWE) agents. Recent work has further explored augmenting these agents with memory mechanisms to support long-horizon reasoning. However, these approaches typically operate at a coarse instance granularity, treating the entire problem-solving episode as the atomic unit of storage and retrieval. We empirically demonstrate that instance-level memory suffers from a fundamental granularity mismatch, resulting in misguided retrieval when tasks with similar surface descriptions require distinct reasoning logic at specific stages. To address this, we propose Structurally Aligned Subtask-Level Memory, a method that aligns memory storage, retrieval, and updating with the agent's functional decomposition. Extensive experiments on SWE-bench Verified demonstrate that our method consistently outperforms both vanilla agents and strong instance-level memory baselines across diverse backbones, improving mean Pass@1 over the vanilla agent by +4.7 pp on average (e.g., +6.8 pp on Gemini 2.5 Pro). Performance gains grow with more interaction steps, showing that leveraging past experience benefits long-horizon reasoning in complex software engineering tasks. 

---
# Retrieval Challenges in Low-Resource Public Service Information: A Case Study on Food Pantry Access 

**Authors**: Touseef Hasan, Laila Cure, Souvika Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2602.21598)  

**Abstract**: Public service information systems are often fragmented, inconsistently formatted, and outdated. These characteristics create low-resource retrieval environments that hinder timely access to critical services. We investigate retrieval challenges in such settings through the domain of food pantry access, a socially urgent problem given persistent food insecurity. We develop an AI-powered conversational retrieval system that scrapes and indexes publicly available pantry data and employs a Retrieval-Augmented Generation (RAG) pipeline to support natural language queries via a web interface. We conduct a pilot evaluation study using community-sourced queries to examine system behavior in realistic scenarios. Our analysis reveals key limitations in retrieval robustness, handling underspecified queries, and grounding over inconsistent knowledge bases. This ongoing work exposes fundamental IR challenges in low-resource environments and motivates future research on robust conversational retrieval to improve access to critical public resources. 

---
# Exploring Human-Machine Coexistence in Symmetrical Reality 

**Authors**: Zhenliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.21584)  

**Abstract**: In the context of the evolution of artificial intelligence (AI), the interaction between humans and AI entities has become increasingly salient, challenging the conventional human-centric paradigms of human-machine interaction. To address this challenge, it is imperative to reassess the relationship between AI entities and humans. Through considering both the virtual and physical worlds, we can construct a novel descriptive framework for a world where humans and machines coexist symbiotically. This paper will introduce a fresh research direction engendered for studying harmonious human-machine coexistence across physical and virtual worlds, which has been termed "symmetrical reality". We will elucidate its key characteristics, offering innovative research insight for renovating human-machine interaction paradigms. 

---
# Revisiting RAG Retrievers: An Information Theoretic Benchmark 

**Authors**: Wenqing Zheng, Dmitri Kalaev, Noah Fatsi, Daniel Barcklow, Owen Reinert, Igor Melnyk, Senthil Kumar, C. Bayan Bruss  

**Link**: [PDF](https://arxiv.org/pdf/2602.21553)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems rely critically on the retriever module to surface relevant context for large language models. Although numerous retrievers have recently been proposed, each built on different ranking principles such as lexical matching, dense embeddings, or graph citations, there remains a lack of systematic understanding of how these mechanisms differ and overlap. Existing benchmarks primarily compare entire RAG pipelines or introduce new datasets, providing little guidance on selecting or combining retrievers themselves. Those that do compare retrievers directly use a limited set of evaluation tools which fail to capture complementary and overlapping strengths. This work presents MIGRASCOPE, a Mutual Information based RAG Retriever Analysis Scope. We revisit state-of-the-art retrievers and introduce principled metrics grounded in information and statistical estimation theory to quantify retrieval quality, redundancy, synergy, and marginal contribution. We further show that if chosen carefully, an ensemble of retrievers outperforms any single retriever. We leverage the developed tools over major RAG corpora to provide unique insights on contribution levels of the state-of-the-art retrievers. Our findings provide a fresh perspective on the structure of modern retrieval techniques and actionable guidance for designing robust and efficient RAG systems. 

---
# From Basis to Basis: Gaussian Particle Representation for Interpretable PDE Operators 

**Authors**: Zhihao Li, Yu Feng, Zhilu Lai, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.21551)  

**Abstract**: Learning PDE dynamics for fluids increasingly relies on neural operators and Transformer-based models, yet these approaches often lack interpretability and struggle with localized, high-frequency structures while incurring quadratic cost in spatial samples. We propose representing fields with a Gaussian basis, where learned atoms carry explicit geometry (centers, anisotropic scales, weights) and form a compact, mesh-agnostic, directly visualizable state. Building on this representation, we introduce a Gaussian Particle Operator that acts in modal space: learned Gaussian modal windows perform a Petrov-Galerkin measurement, and PG Gaussian Attention enables global cross-scale coupling. This basis-to-basis design is resolution-agnostic and achieves near-linear complexity in N for a fixed modal budget, supporting irregular geometries and seamless 2D-to-3D extension. On standard PDE benchmarks and real datasets, our method attains state-of-the-art competitive accuracy while providing intrinsic interpretability. 

---
# Enhancing Multilingual Embeddings via Multi-Way Parallel Text Alignment 

**Authors**: Barah Fazili, Koustava Goswami  

**Link**: [PDF](https://arxiv.org/pdf/2602.21543)  

**Abstract**: Multilingual pretraining typically lacks explicit alignment signals, leading to suboptimal cross-lingual alignment in the representation space. In this work, we show that training standard pretrained models for cross-lingual alignment with a multi-way parallel corpus in a diverse pool of languages can substantially improve multilingual and cross-lingual representations for NLU tasks. We construct a multi-way parallel dataset using translations of English text from an off-the-shelf NMT model for a pool of six target languages and achieve strong cross-lingual alignment through contrastive learning. This leads to substantial performance gains across both seen and unseen languages for multiple tasks from the MTEB benchmark evaluated for XLM-Roberta and multilingual BERT base models. Using a multi-way parallel corpus for contrastive training yields substantial gains on bitext mining (21.3%), semantic similarity (5.3%), and classification (28.4%) compared to English-centric (En-X) bilingually parallel data, where X is sampled from a pool of multiple target languages. Furthermore, finetuning mE5 model on a small dataset with multi-way parallelism significantly improves bitext mining compared to one without, underscoring the importance of multi-way cross-lingual supervision even for models already pretrained for high-quality sentence embeddings. 

---
# LiLo-VLA: Compositional Long-Horizon Manipulation via Linked Object-Centric Policies 

**Authors**: Yue Yang, Shuo Cheng, Yu Fang, Homanga Bharadhwaj, Mingyu Ding, Gedas Bertasius, Daniel Szafir  

**Link**: [PDF](https://arxiv.org/pdf/2602.21531)  

**Abstract**: General-purpose robots must master long-horizon manipulation, defined as tasks involving multiple kinematic structure changes (e.g., attaching or detaching objects) in unstructured environments. While Vision-Language-Action (VLA) models offer the potential to master diverse atomic skills, they struggle with the combinatorial complexity of sequencing them and are prone to cascading failures due to environmental sensitivity. To address these challenges, we propose LiLo-VLA (Linked Local VLA), a modular framework capable of zero-shot generalization to novel long-horizon tasks without ever being trained on them. Our approach decouples transport from interaction: a Reaching Module handles global motion, while an Interaction Module employs an object-centric VLA to process isolated objects of interest, ensuring robustness against irrelevant visual features and invariance to spatial configurations. Crucially, this modularity facilitates robust failure recovery through dynamic replanning and skill reuse, effectively mitigating the cascading errors common in end-to-end approaches. We introduce a 21-task simulation benchmark consisting of two challenging suites: LIBERO-Long++ and Ultra-Long. In these simulations, LiLo-VLA achieves a 69% average success rate, outperforming Pi0.5 by 41% and OpenVLA-OFT by 67%. Furthermore, real-world evaluations across 8 long-horizon tasks demonstrate an average success rate of 85%. Project page: this https URL. 

---
# One Brain, Omni Modalities: Towards Unified Non-Invasive Brain Decoding with Large Language Models 

**Authors**: Changli Tang, Shurui Li, Junliang Wang, Qinfan Xiao, Zhonghao Zhai, Lei Bai, Yu Qiao, Bowen Zhou, Wen Wu, Yuanning Li, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.21522)  

**Abstract**: Deciphering brain function through non-invasive recordings requires synthesizing complementary high-frequency electromagnetic (EEG/MEG) and low-frequency metabolic (fMRI) signals. However, despite their shared neural origins, extreme discrepancies have traditionally confined these modalities to isolated analysis pipelines, hindering a holistic interpretation of brain activity. To bridge this fragmentation, we introduce \textbf{NOBEL}, a \textbf{n}euro-\textbf{o}mni-modal \textbf{b}rain-\textbf{e}ncoding \textbf{l}arge language model (LLM) that unifies these heterogeneous signals within the LLM's semantic embedding space. Our architecture integrates a unified encoder for EEG and MEG with a novel dual-path strategy for fMRI, aligning non-invasive brain signals and external sensory stimuli into a shared token space, then leverages an LLM as a universal backbone. Extensive evaluations demonstrate that NOBEL serves as a robust generalist across standard single-modal tasks. We also show that the synergistic fusion of electromagnetic and metabolic signals yields higher decoding accuracy than unimodal baselines, validating the complementary nature of multiple neural modalities. Furthermore, NOBEL exhibits strong capabilities in stimulus-aware decoding, effectively interpreting visual semantics from multi-subject fMRI data on the NSD and HAD datasets while uniquely leveraging direct stimulus inputs to verify causal links between sensory signals and neural responses. NOBEL thus takes a step towards unifying non-invasive brain decoding, demonstrating the promising potential of omni-modal brain understanding. 

---
# Training Generalizable Collaborative Agents via Strategic Risk Aversion 

**Authors**: Chengrui Qu, Yizhou Zhang, Nicholas Lanzetti, Eric Mazumdar  

**Link**: [PDF](https://arxiv.org/pdf/2602.21515)  

**Abstract**: Many emerging agentic paradigms require agents to collaborate with one another (or people) to achieve shared goals. Unfortunately, existing approaches to learning policies for such collaborative problems produce brittle solutions that fail when paired with new partners. We attribute these failures to a combination of free-riding during training and a lack of strategic robustness. To address these problems, we study the concept of strategic risk aversion and interpret it as a principled inductive bias for generalizable cooperation with unseen partners. While strategically risk-averse players are robust to deviations in their partner's behavior by design, we show that, in collaborative games, they also (1) can have better equilibrium outcomes than those at classical game-theoretic concepts like Nash, and (2) exhibit less or no free-riding. Inspired by these insights, we develop a multi-agent reinforcement learning (MARL) algorithm that integrates strategic risk aversion into standard policy optimization methods. Our empirical results across collaborative benchmarks (including an LLM collaboration task) validate our theory and demonstrate that our approach consistently achieves reliable collaboration with heterogeneous and previously unseen partners across collaborative tasks. 

---
# GradAlign: Gradient-Aligned Data Selection for LLM Reinforcement Learning 

**Authors**: Ningyuan Yang, Weihua Du, Weiwei Sun, Sean Welleck, Yiming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.21492)  

**Abstract**: Reinforcement learning (RL) has become a central post-training paradigm for large language models (LLMs), but its performance is highly sensitive to the quality of training problems. This sensitivity stems from the non-stationarity of RL: rollouts are generated by an evolving policy, and learning is shaped by exploration and reward feedback, unlike supervised fine-tuning (SFT) with fixed trajectories. As a result, prior work often relies on manual curation or simple heuristic filters (e.g., accuracy), which can admit incorrect or low-utility problems. We propose GradAlign, a gradient-aligned data selection method for LLM reinforcement learning that uses a small, trusted validation set to prioritize training problems whose policy gradients align with validation gradients, yielding an adaptive curriculum. We evaluate GradAlign across three challenging data regimes: unreliable reward signals, distribution imbalance, and low-utility training corpus, showing that GradAlign consistently outperforms existing baselines, underscoring the importance of directional gradient signals in navigating non-stationary policy optimization and yielding more stable training and improved final performance. We release our implementation at this https URL 

---
# A Knowledge-Driven Approach to Music Segmentation, Music Source Separation and Cinematic Audio Source Separation 

**Authors**: Chun-wei Ho, Sabato Marco Siniscalchi, Kai Li, Chin-Hui Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.21476)  

**Abstract**: We propose a knowledge-driven, model-based approach to segmenting audio into single-category and mixed-category chunks with applications to source separation. "Knowledge" here denotes information associated with the data, such as music scores. "Model" here refers to tool that can be used for audio segmentation and recognition, such as hidden Markov models. In contrast to conventional learning that often relies on annotated data with given segment categories and their corresponding boundaries to guide the learning process, the proposed framework does not depend on any pre-segmented training data and learns directly from the input audio and its related knowledge sources to build all necessary models autonomously. Evaluation on simulation data shows that score-guided learning achieves very good music segmentation and separation results. Tested on movie track data for cinematic audio source separation also shows that utilizing sound category knowledge achieves better separation results than those obtained with data-driven techniques without using such information. 

---
# Revisiting Text Ranking in Deep Research 

**Authors**: Chuan Meng, Litu Ou, Sean MacAvaney, Jeff Dalton  

**Link**: [PDF](https://arxiv.org/pdf/2602.21456)  

**Abstract**: Deep research has emerged as an important task that aims to address hard queries through extensive open-web exploration. To tackle it, most prior work equips large language model (LLM)-based agents with opaque web search APIs, enabling agents to iteratively issue search queries, retrieve external evidence, and reason over it. Despite search's essential role in deep research, black-box web search APIs hinder systematic analysis of search components, leaving the behaviour of established text ranking methods in deep research largely unclear. To fill this gap, we reproduce a selection of key findings and best practices for IR text ranking methods in the deep research setting. In particular, we examine their effectiveness from three perspectives: (i) retrieval units (documents vs. passages), (ii) pipeline configurations (different retrievers, re-rankers, and re-ranking depths), and (iii) query characteristics (the mismatch between agent-issued queries and the training queries of text rankers). We perform experiments on BrowseComp-Plus, a deep research dataset with a fixed corpus, evaluating 2 open-source agents, 5 retrievers, and 3 re-rankers across diverse setups. We find that agent-issued queries typically follow web-search-style syntax (e.g., quoted exact matches), favouring lexical, learned sparse, and multi-vector retrievers; passage-level units are more efficient under limited context windows, and avoid the difficulties of document length normalisation in lexical retrieval; re-ranking is highly effective; translating agent-issued queries into natural-language questions significantly bridges the query mismatch. 

---
# Adversarial Robustness of Deep Learning-Based Thyroid Nodule Segmentation in Ultrasound 

**Authors**: Nicholas Dietrich, David McShannon  

**Link**: [PDF](https://arxiv.org/pdf/2602.21452)  

**Abstract**: Introduction: Deep learning-based segmentation models are increasingly integrated into clinical imaging workflows, yet their robustness to adversarial perturbations remains incompletely characterized, particularly for ultrasound images. We evaluated adversarial attacks and inference-time defenses for thyroid nodule segmentation in B-mode ultrasound. Methods: Two black-box adversarial attacks were developed: (1) Structured Speckle Amplification Attack (SSAA), which injects boundary-targeted noise, and (2) Frequency-Domain Ultrasound Attack (FDUA), which applies bandpass-filtered phase perturbations in the Fourier domain. Three inference-time mitigations were evaluated on adversarial images: randomized preprocessing with test-time augmentation, deterministic input denoising, and stochastic ensemble inference with consistency-aware aggregation. Experiments were conducted on a U-Net segmentation model trained on cine-clips from a database of 192 thyroid nodules. Results: The baseline model achieved a mean Dice similarity coefficient (DSC) of 0.76 (SD 0.20) on unperturbed images. SSAA reduced DSC by 0.29 (SD 0.20) while maintaining high visual similarity (SSIM = 0.94). FDUA resulted in a smaller DSC reduction of 0.11 (SD 0.09) with lower visual fidelity (SSIM = 0.82). Against SSAA, all three defenses significantly improved DSC after correction, with deterministic denoising showing the largest recovery (+0.10, p < 0.001), followed by randomized preprocessing (+0.09, p < 0.001), and stochastic ensemble inference (+0.08, p = 0.002). No defense achieved statistically significant improvement against FDUA. Conclusion: Spatial-domain adversarial perturbations in ultrasound segmentation showed partial mitigation with input preprocessing, whereas frequency-domain perturbations were not mitigated by the defenses, highlighting modality-specific challenges in adversarial robustness evaluation. 

---
# Adversarial Intent is a Latent Variable: Stateful Trust Inference for Securing Multimodal Agentic RAG 

**Authors**: Inderjeet Singh, Vikas Pahuja, Aishvariya Priya Rathina Sabapathy, Chiara Picardi, Amit Giloni, Roman Vainshtein, Andrés Murillo, Hisashi Kojima, Motoyoshi Sekiya, Yuki Unno, Junichi Suga  

**Link**: [PDF](https://arxiv.org/pdf/2602.21447)  

**Abstract**: Current stateless defences for multimodal agentic RAG fail to detect adversarial strategies that distribute malicious semantics across retrieval, planning, and generation components. We formulate this security challenge as a Partially Observable Markov Decision Process (POMDP), where adversarial intent is a latent variable inferred from noisy multi-stage observations. We introduce MMA-RAG^T, an inference-time control framework governed by a Modular Trust Agent (MTA) that maintains an approximate belief state via structured LLM reasoning. Operating as a model-agnostic overlay, MMA-RAGT mediates a configurable set of internal checkpoints to enforce stateful defence-in-depth. Extensive evaluation on 43,774 instances demonstrates a 6.50x average reduction factor in Attack Success Rate relative to undefended baselines, with negligible utility cost. Crucially, a factorial ablation validates our theoretical bounds: while statefulness and spatial coverage are individually necessary (26.4 pp and 13.6 pp gains respectively), stateless multi-point intervention can yield zero marginal benefit under homogeneous stateless filtering when checkpoint detections are perfectly correlated. 

---
# MINAR: Mechanistic Interpretability for Neural Algorithmic Reasoning 

**Authors**: Jesse He, Helen Jenne, Max Vargas, Davis Brown, Gal Mishne, Yusu Wang, Henry Kvinge  

**Link**: [PDF](https://arxiv.org/pdf/2602.21442)  

**Abstract**: The recent field of neural algorithmic reasoning (NAR) studies the ability of graph neural networks (GNNs) to emulate classical algorithms like Bellman-Ford, a phenomenon known as algorithmic alignment. At the same time, recent advances in large language models (LLMs) have spawned the study of mechanistic interpretability, which aims to identify granular model components like circuits that perform specific computations. In this work, we introduce Mechanistic Interpretability for Neural Algorithmic Reasoning (MINAR), an efficient circuit discovery toolbox that adapts attribution patching methods from mechanistic interpretability to the GNN setting. We show through two case studies that MINAR recovers faithful neuron-level circuits from GNNs trained on algorithmic tasks. Our study sheds new light on the process of circuit formation and pruning during training, as well as giving new insight into how GNNs trained to perform multiple tasks in parallel reuse circuit components for related tasks. Our code is available at this https URL. 

---
# Causal Decoding for Hallucination-Resistant Multimodal Large Language Models 

**Authors**: Shiwei Tan, Hengyi Wang, Weiyi Qin, Qi Xu, Zhigang Hua, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.21441)  

**Abstract**: Multimodal Large Language Models (MLLMs) deliver detailed responses on vision-language tasks, yet remain susceptible to object hallucination (introducing objects not present in the image), undermining reliability in practice. Prior efforts often rely on heuristic penalties, post-hoc correction, or generic decoding tweaks, which do not directly intervene in the mechanisms that trigger object hallucination and thus yield limited gains. To address this challenge, we propose a causal decoding framework that applies targeted causal interventions during generation to curb spurious object mentions. By reshaping the decoding dynamics to attenuate spurious dependencies, our approach reduces false object tokens while maintaining descriptive quality. Across captioning and QA benchmarks, our framework substantially lowers object-hallucination rates and achieves state-of-the-art faithfulness without degrading overall output quality. 

---
# Provably Safe Generative Sampling with Constricting Barrier Functions 

**Authors**: Darshan Gadginmath, Ahmed Allibhoy, Fabio Pasqualetti  

**Link**: [PDF](https://arxiv.org/pdf/2602.21429)  

**Abstract**: Flow-based generative models, such as diffusion models and flow matching models, have achieved remarkable success in learning complex data distributions. However, a critical gap remains for their deployment in safety-critical domains: the lack of formal guarantees that generated samples will satisfy hard constraints. We address this by proposing a safety filtering framework that acts as an online shield for any pre-trained generative model. Our key insight is to cooperate with the generative process rather than override it. We define a constricting safety tube that is relaxed at the initial noise distribution and progressively tightens to the target safe set at the final data distribution, mirroring the coarse-to-fine structure of the generative process itself. By characterizing this tube via Control Barrier Functions (CBFs), we synthesize a feedback control input through a convex Quadratic Program (QP) at each sampling step. As the tube is loosest when noise is high and intervention is cheapest in terms of control energy, most constraint enforcement occurs when it least disrupts the model's learned structure. We prove that this mechanism guarantees safe sampling while minimizing the distributional shift from the original model at each sampling step, as quantified by the KL divergence. Our framework applies to any pre-trained flow-based generative scheme requiring no retraining or architectural modifications. We validate the approach across constrained image generation, physically-consistent trajectory sampling, and safe robotic manipulation policies, achieving 100% constraint satisfaction while preserving semantic fidelity. 

---
# On the Structural Non-Preservation of Epistemic Behaviour under Policy Transformation 

**Authors**: Alexander Galozy  

**Link**: [PDF](https://arxiv.org/pdf/2602.21424)  

**Abstract**: Reinforcement learning (RL) agents under partial observability often condition actions on internally accumulated information such as memory or inferred latent context. We formalise such information-conditioned interaction patterns as behavioural dependency: variation in action selection with respect to internal information under fixed observations. This induces a probe-relative notion of $\epsilon$-behavioural equivalence and a within-policy behavioural distance that quantifies probe sensitivity. We establish three structural results. First, the set of policies exhibiting non-trivial behavioural dependency is not closed under convex aggregation. Second, behavioural distance contracts under convex combination. Third, we prove a sufficient local condition under which gradient ascent on a skewed mixture objective decreases behavioural distance when a dominant-mode gradient aligns with the direction of steepest contraction. Minimal bandit and partially observable gridworld experiments provide controlled witnesses of these mechanisms. In the examined settings, behavioural distance decreases under convex aggregation and under continued optimisation with skewed latent priors, and in these experiments it precedes degradation under latent prior shift. These results identify structural conditions under which probe-conditioned behavioural separation is not preserved under common policy transformations. 

---
# ECHOSAT: Estimating Canopy Height Over Space And Time 

**Authors**: Jan Pauls, Karsten Schrödter, Sven Ligensa, Martin Schwartz, Berkant Turan, Max Zimmer, Sassan Saatchi, Sebastian Pokutta, Philippe Ciais, Fabian Gieseke  

**Link**: [PDF](https://arxiv.org/pdf/2602.21421)  

**Abstract**: Forest monitoring is critical for climate change mitigation. However, existing global tree height maps provide only static snapshots and do not capture temporal forest dynamics, which are essential for accurate carbon accounting. We introduce ECHOSAT, a global and temporally consistent tree height map at 10 m resolution spanning multiple years. To this end, we resort to multi-sensor satellite data to train a specialized vision transformer model, which performs pixel-level temporal regression. A self-supervised growth loss regularizes the predictions to follow growth curves that are in line with natural tree development, including gradual height increases over time, but also abrupt declines due to forest loss events such as fires. Our experimental evaluation shows that our model improves state-of-the-art accuracies in the context of single-year predictions. We also provide the first global-scale height map that accurately quantifies tree growth and disturbances over time. We expect ECHOSAT to advance global efforts in carbon monitoring and disturbance assessment. The maps can be accessed at this https URL. 

---
# Overconfident Errors Need Stronger Correction: Asymmetric Confidence Penalties for Reinforcement Learning 

**Authors**: Yuanda Xu, Hejian Sang, Zhengze Zhou, Ran He, Zhipeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.21420)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has become the leading paradigm for enhancing reasoning in Large Language Models (LLMs). However, standard RLVR algorithms suffer from a well-documented pathology: while they improve Pass@1 accuracy through sharpened sampling, they simultaneously narrow the model's reasoning boundary and reduce generation diversity. We identify a root cause that existing methods overlook: the uniform penalization of errors. Current approaches -- whether data-filtering methods that select prompts by difficulty, or advantage normalization schemes -- treat all incorrect rollouts within a group identically. We show that this uniformity allows overconfident errors (incorrect reasoning paths that the RL process has spuriously reinforced) to persist and monopolize probability mass, ultimately suppressing valid exploratory trajectories. To address this, we propose the Asymmetric Confidence-aware Error Penalty (ACE). ACE introduces a per-rollout confidence shift metric, c_i = log(pi_theta(y_i|x) / pi_ref(y_i|x)), to dynamically modulate negative advantages. Theoretically, we demonstrate that ACE's gradient can be decomposed into the gradient of a selective regularizer restricted to overconfident errors, plus a well-characterized residual that partially moderates the regularizer's strength. We conduct extensive experiments fine-tuning Qwen2.5-Math-7B, Qwen3-8B-Base, and Llama-3.1-8B-Instruct on the DAPO-Math-17K dataset using GRPO and DAPO within the VERL framework. Evaluated on MATH-500 and AIME 2025, ACE composes seamlessly with existing methods and consistently improves the full Pass@k spectrum across all three model families and benchmarks. 

---
# The Headless Firm: How AI Reshapes Enterprise Boundaries 

**Authors**: Tassilo Klein, Sebastian Wieczorek  

**Link**: [PDF](https://arxiv.org/pdf/2602.21401)  

**Abstract**: The boundary of the firm is determined by coordination cost. We argue that agentic AI induces a structural change in how coordination costs scale: in prior modular systems, integration cost grew with interaction topology (O(n^2) in the number of components); in protocol-mediated agentic systems, integration cost collapses to O(n) while verification scales with task throughput rather than interaction count. This shift selects for a specific organizational equilibrium -- the Headless Firm -- structured as an hourglass: a personalized generative interface at the top, a standardized protocol waist in the middle, and a competitive market of micro-specialized execution agents at the bottom. We formalize this claim as a coordination cost model with two falsifiable empirical predictions: (1) the marginal cost of adding an execution provider should be approximately constant in a mature hourglass ecosystem; (2) the ratio of total coordination cost to task throughput should remain stable as ecosystem size grows. We derive conditions for hourglass stability versus re-centralization and analyze implications for firm size distributions, labor markets, and software economics. The analysis predicts a domain-conditional Great Unbundling: in high knowledge-velocity domains, firm size distributions shift mass from large integrated incumbents toward micro-specialized agents and thin protocol orchestrators. 

---
# FedVG: Gradient-Guided Aggregation for Enhanced Federated Learning 

**Authors**: Alina Devkota, Jacob Thrasher, Donald Adjeroh, Binod Bhattarai, Prashnna K. Gyawali  

**Link**: [PDF](https://arxiv.org/pdf/2602.21399)  

**Abstract**: Federated Learning (FL) enables collaborative model training across multiple clients without sharing their private data. However, data heterogeneity across clients leads to client drift, which degrades the overall generalization performance of the model. This effect is further compounded by overemphasis on poorly performing clients. To address this problem, we propose FedVG, a novel gradient-based federated aggregation framework that leverages a global validation set to guide the optimization process. Such a global validation set can be established using readily available public datasets, ensuring accessibility and consistency across clients without compromising privacy. In contrast to conventional approaches that prioritize client dataset volume, FedVG assesses the generalization ability of client models by measuring the magnitude of validation gradients across layers. Specifically, we compute layerwise gradient norms to derive a client-specific score that reflects how much each client needs to adjust for improved generalization on the global validation set, thereby enabling more informed and adaptive federated aggregation. Extensive experiments on both natural and medical image benchmarking datasets, across diverse model architectures, demonstrate that FedVG consistently improves performance, particularly in highly heterogeneous settings. Moreover, FedVG is modular and can be seamlessly integrated with various state-of-the-art FL algorithms, often further improving their results. Our code is available at this https URL. 

---
# VCDF: A Validated Consensus-Driven Framework for Time Series Causal Discovery 

**Authors**: Gene Yu, Ce Guo, Wayne Luk  

**Link**: [PDF](https://arxiv.org/pdf/2602.21381)  

**Abstract**: Time series causal discovery is essential for understanding dynamic systems, yet many existing methods remain sensitive to noise, non-stationarity, and sampling variability. We propose the Validated Consensus-Driven Framework (VCDF), a simple and method-agnostic layer that improves robustness by evaluating the stability of causal relations across blocked temporal subsets. VCDF requires no modification to base algorithms and can be applied to methods such as VAR-LiNGAM and PCMCI. Experiments on synthetic datasets show that VCDF improves VAR-LiNGAM by approximately 0.08-0.12 in both window and summary F1 scores across diverse data characteristics, with gains most pronounced for moderate-to-long sequences. The framework also benefits from longer sequences, yielding up to 0.18 absolute improvement on time series of length 1000 and above. Evaluations on simulated fMRI data and IT-monitoring scenarios further demonstrate enhanced stability and structural accuracy under realistic noise conditions. VCDF provides an effective reliability layer for time series causal discovery without altering underlying modeling assumptions. 

---
# MrBERT: Modern Multilingual Encoders via Vocabulary, Domain, and Dimensional Adaptation 

**Authors**: Daniel Tamayo, Iñaki Lacunza, Paula Rivera-Hidalgo, Severino Da Dalt, Javier Aula-Blasco, Aitor Gonzalez-Agirre, Marta Villegas  

**Link**: [PDF](https://arxiv.org/pdf/2602.21379)  

**Abstract**: We introduce MrBERT, a family of 150M-300M parameter encoders built on the ModernBERT architecture and pre-trained on 35 languages and code. Through targeted adaptation, this model family achieves state-of-the-art results on Catalan- and Spanish-specific tasks, while establishing robust performance across specialized biomedical and legal domains. To bridge the gap between research and production, we incorporate Matryoshka Representation Learning (MRL), enabling flexible vector sizing that significantly reduces inference and storage costs. Ultimately, the MrBERT family demonstrates that modern encoder architectures can be optimized for both localized linguistic excellence and efficient, high-stakes domain specialization. We open source the complete model family on Huggingface. 

---
# Small Language Models for Privacy-Preserving Clinical Information Extraction in Low-Resource Languages 

**Authors**: Mohammadreza Ghaffarzadeh-Esfahani, Nahid Yousefian, Ebrahim Heidari-Farsani, Ali Akbar Omidvarian, Sepehr Ghahraei, Atena Farangi, AmirBahador Boroumand  

**Link**: [PDF](https://arxiv.org/pdf/2602.21374)  

**Abstract**: Extracting clinical information from medical transcripts in low-resource languages remains a significant challenge in healthcare natural language processing (NLP). This study evaluates a two-step pipeline combining Aya-expanse-8B as a Persian-to-English translation model with five open-source small language models (SLMs) -- Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Llama-3.2-3B-Instruct, Qwen2.5-1.5B-Instruct, and Gemma-3-1B-it -- for binary extraction of 13 clinical features from 1,221 anonymized Persian transcripts collected at a cancer palliative care call center. Using a few-shot prompting strategy without fine-tuning, models were assessed on macro-averaged F1-score, Matthews Correlation Coefficient (MCC), sensitivity, and specificity to account for class imbalance. Qwen2.5-7B-Instruct achieved the highest overall performance (median macro-F1: 0.899; MCC: 0.797), while Gemma-3-1B-it showed the weakest results. Larger models (7B--8B parameters) consistently outperformed smaller counterparts in sensitivity and MCC. A bilingual analysis of Aya-expanse-8B revealed that translating Persian transcripts to English improved sensitivity, reduced missing outputs, and boosted metrics robust to class imbalance, though at the cost of slightly lower specificity and precision. Feature-level results showed reliable extraction of physiological symptoms across most models, whereas psychological complaints, administrative requests, and complex somatic features remained challenging. These findings establish a practical, privacy-preserving blueprint for deploying open-source SLMs in multilingual clinical NLP settings with limited infrastructure and annotation resources, and highlight the importance of jointly optimizing model scale and input language strategy for sensitive healthcare applications. 

---
# The Mean is the Mirage: Entropy-Adaptive Model Merging under Heterogeneous Domain Shifts in Medical Imaging 

**Authors**: Sameer Ambekar, Reza Nasirigerdeh, Peter J. Schuffler, Lina Felsner, Daniel M. Lang, Julia A. Schnabel  

**Link**: [PDF](https://arxiv.org/pdf/2602.21372)  

**Abstract**: Model merging under unseen test-time distribution shifts often renders naive strategies, such as mean averaging unreliable. This challenge is especially acute in medical imaging, where models are fine-tuned locally at clinics on private data, producing domain-specific models that differ by scanner, protocol, and population. When deployed at an unseen clinical site, test cases arrive in unlabeled, non-i.i.d. batches, and the model must adapt immediately without labels. In this work, we introduce an entropy-adaptive, fully online model-merging method that yields a batch-specific merged model via only forward passes, effectively leveraging target information. We further demonstrate why mean merging is prone to failure and misaligned under heterogeneous domain shifts. Next, we mitigate encoder classifier mismatch by decoupling the encoder and classification head, merging with separate merging coefficients. We extensively evaluate our method with state-of-the-art baselines using two backbones across nine medical and natural-domain generalization image classification datasets, showing consistent gains across standard evaluation and challenging scenarios. These performance gains are achieved while retaining single-model inference at test-time, thereby demonstrating the effectiveness of our method. 

---
# Black-Box Reliability Certification for AI Agents via Self-Consistency Sampling and Conformal Calibration 

**Authors**: Charafeddine Mouzouni  

**Link**: [PDF](https://arxiv.org/pdf/2602.21368)  

**Abstract**: Given a black-box AI system and a task, at what confidence level can a practitioner trust the system's output? We answer with a reliability level -- a single number per system-task pair, derived from self-consistency sampling and conformal calibration, that serves as a black-box deployment gate with exact, finite-sample, distribution-free guarantees. Self-consistency sampling reduces uncertainty exponentially; conformal calibration guarantees correctness within 1/(n+1) of the target level, regardless of the system's errors -- made transparently visible through larger answer sets for harder questions. Weaker models earn lower reliability levels (not accuracy -- see Definition 2.4): GPT-4.1 earns 94.6% on GSM8K and 96.8% on TruthfulQA, while GPT-4.1-nano earns 89.8% on GSM8K and 66.5% on MMLU. We validate across five benchmarks, five models from three families, and both synthetic and real data. Conditional coverage on solvable items exceeds 0.93 across all configurations; sequential stopping reduces API costs by around 50%. 

---
# Towards Controllable Video Synthesis of Routine and Rare OR Events 

**Authors**: Dominik Schneider, Lalithkumar Seenivasan, Sampath Rapuri, Vishalroshan Anil, Aiza Maksutova, Yiqing Shen, Jan Emily Mangulabnan, Hao Ding, Jose L. Porras, Masaru Ishii, Mathias Unberath  

**Link**: [PDF](https://arxiv.org/pdf/2602.21365)  

**Abstract**: Purpose: Curating large-scale datasets of operating room (OR) workflow, encompassing rare, safety-critical, or atypical events, remains operationally and ethically challenging. This data bottleneck complicates the development of ambient intelligence for detecting, understanding, and mitigating rare or safety-critical events in the OR.
Methods: This work presents an OR video diffusion framework that enables controlled synthesis of rare and safety-critical events. The framework integrates a geometric abstraction module, a conditioning module, and a fine-tuned diffusion model to first transform OR scenes into abstract geometric representations, then condition the synthesis process, and finally generate realistic OR event videos. Using this framework, we also curate a synthetic dataset to train and validate AI models for detecting near-misses of sterile-field violations.
Results: In synthesizing routine OR events, our method outperforms off-the-shelf video diffusion baselines, achieving lower FVD/LPIPS and higher SSIM/PSNR in both in- and out-of-domain datasets. Through qualitative results, we illustrate its ability for controlled video synthesis of counterfactual events. An AI model trained and validated on the generated synthetic data achieved a RECALL of 70.13% in detecting near safety-critical events. Finally, we conduct an ablation study to quantify performance gains from key design choices.
Conclusion: Our solution enables controlled synthesis of routine and rare OR events from abstract geometric representations. Beyond demonstrating its capability to generate rare and safety-critical scenarios, we show its potential to support the development of ambient intelligence models. 

---
# Towards single-shot coherent imaging via overlap-free ptychography 

**Authors**: Oliver Hoidn, Aashwin Mishra, Steven Henke, Albert Vong, Matthew Seaberg  

**Link**: [PDF](https://arxiv.org/pdf/2602.21361)  

**Abstract**: Ptychographic imaging at synchrotron and XFEL sources requires dense overlapping scans, limiting throughput and increasing dose. Extending coherent diffractive imaging to overlap-free operation on extended samples remains an open problem. Here, we extend PtychoPINN (O. Hoidn \emph{et al.}, \emph{Scientific Reports} \textbf{13}, 22789, 2023) to deliver \emph{overlap-free, single-shot} reconstructions in a Fresnel coherent diffraction imaging (CDI) geometry while also accelerating conventional multi-shot ptychography. The framework couples a differentiable forward model of coherent scattering with a Poisson photon-counting likelihood; real-space overlap enters as a tunable parameter via coordinate-based grouping rather than a hard requirement. On synthetic benchmarks, reconstructions remain accurate at low counts ($\sim\!10^4$ photons/frame), and overlap-free single-shot reconstruction with an experimental probe reaches amplitude structural similarity (SSIM) 0.904, compared with 0.968 for overlap-constrained reconstruction. Against a data-saturated supervised model with the same backbone (16,384 training images), PtychoPINN achieves higher SSIM with only 1,024 images and generalizes to unseen illumination profiles. Per-graphics processing unit (GPU) throughput is approximately $40\times$ that of least-squares maximum-likelihood (LSQ-ML) reconstruction at matched $128\times128$ resolution. These results, validated on experimental data from the Advanced Photon Source and the Linac Coherent Light Source, unify single-exposure Fresnel CDI and overlapped ptychography within one framework, supporting dose-efficient, high-throughput imaging at modern light sources. 

---
# Representation Theorems for Cumulative Propositional Dependence Logics 

**Authors**: Juha Kontinen, Arne Meier, Kai Sauerwald  

**Link**: [PDF](https://arxiv.org/pdf/2602.21360)  

**Abstract**: This paper establishes and proves representation theorems for cumulative propositional dependence logic and for cumulative propositional logic with team semantics. Cumulative logics are famously given by System C. For propositional dependence logic, we show that System C entailments are exactly captured by cumulative models from Kraus, Lehmann and Magidor. On the other hand, we show that entailment in cumulative propositional logics with team semantics is exactly captured by cumulative and asymmetric models. For the latter, we also obtain equivalence with cumulative logics based on propositional logic with classical semantics. The proofs will be useful for proving representation theorems for other cumulative logics without negation and material implication. 

---
# Alignment-Weighted DPO: A principled reasoning approach to improve safety alignment 

**Authors**: Mengxuan Hu, Vivek V. Datla, Anoop Kumar, Zihan Guan, Sheng Li, Alfy Samuel, Daben Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.21346)  

**Abstract**: Recent advances in alignment techniques such as Supervised Fine-Tuning (SFT), Reinforcement Learning from Human Feedback (RLHF), and Direct Preference Optimization (DPO) have improved the safety of large language models (LLMs). However, these LLMs remain vulnerable to jailbreak attacks that disguise harmful intent through indirect or deceptive phrasing. Using causal intervention, we empirically demonstrate that this vulnerability stems from shallow alignment mechanisms that lack deep reasoning, often rejecting harmful prompts without truly understanding why they are harmful. To mitigate this vulnerability, we propose enhancing alignment through reasoning-aware post-training. We construct and release a novel Chain-of-Thought (CoT) fine-tuning dataset that includes both utility-oriented and safety-critical prompts with step-by-step rationales. Fine-tuning on this dataset encourages models to produce principled refusals grounded in reasoning, outperforming standard SFT baselines. Furthermore, inspired by failure patterns in CoT fine-tuning, we introduce Alignment-Weighted DPO, which targets the most problematic parts of an output by assigning different preference weights to the reasoning and final-answer segments. This produces finer-grained, targeted updates than vanilla DPO and improves robustness to diverse jailbreak strategies. Extensive experiments across multiple safety and utility benchmarks show that our method consistently improves alignment robustness while maintaining overall model utility. 

---
# Scaling View Synthesis Transformers 

**Authors**: Evan Kim, Hyunwoo Ryu, Thomas W. Mitchel, Vincent Sitzmann  

**Link**: [PDF](https://arxiv.org/pdf/2602.21341)  

**Abstract**: Geometry-free view synthesis transformers have recently achieved state-of-the-art performance in Novel View Synthesis (NVS), outperforming traditional approaches that rely on explicit geometry modeling. Yet the factors governing their scaling with compute remain unclear. We present a systematic study of scaling laws for view synthesis transformers and derive design principles for training compute-optimal NVS models. Contrary to prior findings, we show that encoder-decoder architectures can be compute-optimal; we trace earlier negative results to suboptimal architectural choices and comparisons across unequal training compute budgets. Across several compute levels, we demonstrate that our encoder-decoder architecture, which we call the Scalable View Synthesis Model (SVSM), scales as effectively as decoder-only models, achieves a superior performance-compute Pareto frontier, and surpasses the previous state-of-the-art on real-world NVS benchmarks with substantially reduced training compute. 

---
# Equitable Evaluation via Elicitation 

**Authors**: Elbert Du, Cynthia Dwork, Lunjia Hu, Reid McIlroy-Young, Han Shao, Linjun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.21327)  

**Abstract**: Individuals with similar qualifications and skills may vary in their demeanor, or outward manner: some tend toward self-promotion while others are modest to the point of omitting crucial information. Comparing the self-descriptions of equally qualified job-seekers with different self-presentation styles is therefore problematic.
We build an interactive AI for skill elicitation that provides accurate determination of skills while simultaneously allowing individuals to speak in their own voice. Such a system can be deployed, for example, when a new user joins a professional networking platform, or when matching employees to needs during a company reorganization. To obtain sufficient training data, we train an LLM to act as synthetic humans.
Elicitation mitigates endogenous bias arising from individuals' own self-reports. To address systematic model bias we enforce a mathematically rigorous notion of equitability ensuring that the covariance between self-presentation manner and skill evaluation error is small. 

---
# Group Orthogonalized Policy Optimization:Group Policy Optimization as Orthogonal Projection in Hilbert Space 

**Authors**: Wang Zixian  

**Link**: [PDF](https://arxiv.org/pdf/2602.21269)  

**Abstract**: We present Group Orthogonalized Policy Optimization (GOPO), a new alignment algorithm for large language models derived from the geometry of Hilbert function spaces. Instead of optimizing on the probability simplex and inheriting the exponential curvature of Kullback-Leibler divergence, GOPO lifts alignment into the Hilbert space L2(pi_k) of square-integrable functions with respect to the reference policy. Within this space, the simplex constraint reduces to a linear orthogonality condition <v, 1> = 0, defining a codimension-one subspace H0. Minimizing distance to an unconstrained target u_star yields the work-dissipation functional J(v) = <g, v> - (mu / 2) ||v||^2, whose maximizer follows directly from the Hilbert projection theorem. Enforcing the boundary v >= -1 produces a bounded Hilbert projection that induces exact sparsity, assigning zero probability to catastrophically poor actions through a closed-form threshold. To connect this functional theory with practice, GOPO projects from infinite-dimensional L2(pi_k) to a finite empirical subspace induced by group sampling. Because group-normalized advantages sum to zero, the Lagrange multiplier enforcing probability conservation vanishes exactly, reducing the constrained projection to an unconstrained empirical loss. The resulting objective has constant Hessian curvature mu I, non-saturating linear gradients, and an intrinsic dead-zone mechanism without heuristic clipping. Experiments on mathematical reasoning benchmarks show that GOPO achieves competitive generalization while maintaining stable gradient dynamics and entropy preservation in regimes where clipping-based methods plateau. 

---
# A Systematic Review of Algorithmic Red Teaming Methodologies for Assurance and Security of AI Applications 

**Authors**: Shruti Srivastava, Kiranmayee Janardhan, Shaurya Jauhari  

**Link**: [PDF](https://arxiv.org/pdf/2602.21267)  

**Abstract**: Cybersecurity threats are becoming increasingly sophisticated, making traditional defense mechanisms and manual red teaming approaches insufficient for modern organizations. While red teaming has long been recognized as an effective method to identify vulnerabilities by simulating real-world attacks, its manual execution is resource-intensive, time-consuming, and lacks scalability for frequent assessments. These limitations have driven the evolution toward auto-mated red teaming, which leverages artificial intelligence and automation to deliver efficient and adaptive security evaluations. This systematic review consolidates existing research on automated red teaming, examining its methodologies, tools, benefits, and limitations. The paper also highlights current trends, challenges, and research gaps, offering insights into future directions for improving automated red teaming as a critical component of proactive cybersecurity strategies. By synthesizing findings from diverse studies, this review aims to provide a comprehensive understanding of how automation enhances red teaming and strengthens organizational resilience against evolving cyber threats. 

---
# A General Equilibrium Theory of Orchestrated AI Agent Systems 

**Authors**: Jean-Philippe Garnier  

**Link**: [PDF](https://arxiv.org/pdf/2602.21255)  

**Abstract**: We establish a general equilibrium theory for systems of large language model (LLM) agents operating under centralized orchestration. The framework is a production economy in the sense of Arrow-Debreu (1954), extended to infinite-dimensional commodity spaces following Bewley (1972). Each LLM agent is modeled as a firm whose production set Y a $\subset$ H = L 2 ([0, T ], R R ) represents the feasible metric trajectories determined by its frozen model weights. The orchestrator is the consumer, choosing a routing policy over the agent DAG to maximize system welfare subject to a budget constraint evaluated at functional prices p $\in$ H A . These prices-elements of the Hilbert dual of the commodity space-assign a shadow value to each metric of each agent at each instant. We prove, via Brouwer's theorem applied to a finitedimensional approximation V K $\subset$ H, that every such economy admits at least one general equilibrium (p * , y * , $\pi$ * ). A functional Walras' law  holds as a theorem: the value of functional excess demand is zero for all prices, as a consequence of the consumer's budget constraint-not by construction. We further establish Pareto optimality (First Welfare Theorem), decentralizability of Pareto optima (Second Welfare Theorem), and uniqueness with geometric convergence under a contraction condition (Banach). The orchestration dynamics constitute a Walrasian t{â}tonnement that converges globally under the contraction condition, unlike classical t{â}tonnement (Scarf, 1960). The framework admits a DSGE interpretation with SLO parameters as policy rates. 

---
# AgenticTyper: Automated Typing of Legacy Software Projects Using Agentic AI 

**Authors**: Clemens Pohle  

**Link**: [PDF](https://arxiv.org/pdf/2602.21251)  

**Abstract**: Legacy JavaScript systems lack type safety, making maintenance risky. While TypeScript can help, manually adding types is expensive. Previous automated typing research focuses on type inference but rarely addresses type checking setup, definition generation, bug identification, or behavioral correctness at repository scale. We present AgenticTyper, a Large Language Model (LLM)-based agentic system that addresses these gaps through iterative error correction and behavior preservation via transpilation comparison. Evaluation on two proprietary repositories (81K LOC) shows that AgenticTyper resolves all 633 initial type errors in 20 minutes, reducing manual effort from one working day. 

---
# AngelSlim: A more accessible, comprehensive, and efficient toolkit for large model compression 

**Authors**: Rui Cen, QiangQiang Hu, Hong Huang, Hong Liu, Song Liu, Xin Luo, Lin Niu, Yifan Tan, Decheng Wu, Linchuan Xie, Rubing Yang, Guanghua Yu, Jianchen Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2602.21233)  

**Abstract**: This technical report introduces AngelSlim, a comprehensive and versatile toolkit for large model compression developed by the Tencent Hunyuan team. By consolidating cutting-edge algorithms, including quantization, speculative decoding, token pruning, and distillation. AngelSlim provides a unified pipeline that streamlines the transition from model compression to industrial-scale deployment. To facilitate efficient acceleration, we integrate state-of-the-art FP8 and INT8 Post-Training Quantization (PTQ) algorithms alongside pioneering research in ultra-low-bit regimes, featuring HY-1.8B-int2 as the first industrially viable 2-bit large model. Beyond quantization, we propose a training-aligned speculative decoding framework compatible with multimodal architectures and modern inference engines, achieving 1.8x to 2.0x throughput gains without compromising output correctness. Furthermore, we develop a training-free sparse attention framework that reduces Time-to-First-Token (TTFT) in long-context scenarios by decoupling sparse kernels from model architectures through a hybrid of static patterns and dynamic token selection. For multimodal models, AngelSlim incorporates specialized pruning strategies, namely IDPruner for optimizing vision tokens via Maximal Marginal Relevance and Samp for adaptive audio token merging and pruning. By integrating these compression strategies from low-level implementations, AngelSlim enables algorithm-focused research and tool-assisted deployment. 

---
# Urban Vibrancy Embedding and Application on Traffic Prediction 

**Authors**: Sumin Han, Jisun An, Dongman Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.21232)  

**Abstract**: Urban vibrancy reflects the dynamic human activity within urban spaces and is often measured using mobile data that captures floating population trends. This study proposes a novel approach to derive Urban Vibrancy embeddings from real-time floating population data to enhance traffic prediction models. Specifically, we utilize variational autoencoders (VAE) to compress this data into actionable embeddings, which are then integrated with long short-term memory (LSTM) networks to predict future embeddings. These are subsequently applied in a sequence-to-sequence framework for traffic forecasting. Our contributions are threefold: (1) We use principal component analysis (PCA) to interpret the embeddings, revealing temporal patterns such as weekday versus weekend distinctions and seasonal patterns; (2) We propose a method that combines VAE and LSTM, enabling forecasting dynamic urban knowledge embedding; and (3) Our approach improves accuracy and responsiveness in traffic prediction models, including RNN, DCRNN, GTS, and GMAN. This study demonstrates the potential of Urban Vibrancy embeddings to advance traffic prediction and offer a more nuanced analysis of urban mobility. 

---
# ACAR: Adaptive Complexity Routing for Multi-Model Ensembles with Auditable Decision Traces 

**Authors**: Ramchand Kumaresan  

**Link**: [PDF](https://arxiv.org/pdf/2602.21231)  

**Abstract**: We present ACAR (Adaptive Complexity and Attribution Routing), a measurement framework for studying multi-model orchestration under auditable conditions. ACAR uses self-consistency variance (sigma) computed from N=3 probe samples to route tasks across single-model, two-model, and three-model execution modes. The system is implemented on top of TEAMLLM, a deterministic execution substrate with immutable artifacts and complete decision traces. We evaluate ACAR on 1,510 tasks spanning four benchmarks: MathArena, Reasoning Gym, LiveCodeBench, and SuperGPQA, using Claude Sonnet 4, GPT-4o, and Gemini 2.0 Flash, producing more than 7,550 auditable runs. Results show that sigma-based routing achieves 55.6 percent accuracy, exceeding the two-model baseline of 54.4 percent while avoiding full ensembling on 54.2 percent of tasks. The routing mechanism is model-agnostic and requires no learned components. We also document negative results. First, retrieval augmentation reduced accuracy by 3.4 percentage points, as median retrieval similarity was only 0.167, demonstrating that experience injection without semantic alignment introduces noise rather than grounding. Second, when models agree on incorrect answers (sigma equals zero), no downstream ensemble can recover; this agreement-but-wrong failure mode is intrinsic to self-consistency and bounds achievable accuracy at approximately eight percentage points below full ensembling. Third, attribution estimates based on proxy signals such as response similarity and entropy showed weak correlation with ground-truth leave-one-out values, indicating that practical attribution requires explicit counterfactual computation. This work documents which assumptions fail in practice and provides falsifiable baselines for future research on routing, retrieval, and multi-model attribution. 

---
# ImpRIF: Stronger Implicit Reasoning Leads to Better Complex Instruction Following 

**Authors**: Yuancheng Yang, Lin Yang, Xu Wang, Chao Tong, Haihua Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.21228)  

**Abstract**: As applications of large language models (LLMs) become increasingly complex, the demand for robust complex instruction following capabilities is growing accordingly. We argue that a thorough understanding of the instruction itself, especially the latent reasoning structure embedded between the lines, is crucial for improving instruction following. Therefore we target complex instructions that involve implicit reasoning, intricate logical relations, and multi-constraint dependencies. We propose ImpRIF, a method to enhance LLMs' understanding of implicit reasoning instructions, thereby improving its ability to follow complex instructions. We formalize such instructions as verifiable reasoning graphs, enabling programmatic verification and graph-driven chain-of-thought reasoning. Based on this formulation, we synthesize large-scale single- and multi-turn data, propose fine-tuning with graph reasoning, and apply reinforcement learning to explicitly train models to reason along the graph. On five complex instruction following benchmarks, our models substantially outperform their base models. These results demonstrate that enhancing implicit reasoning capabilities can significantly improve complex instruction following. This project will be open-sourced in the near future. 

---
# Budget-Aware Agentic Routing via Boundary-Guided Training 

**Authors**: Caiqi Zhang, Menglin Xia, Xuchao Zhang, Daniel Madrigal, Ankur Mallick, Samuel Kessler, Victor Ruehle, Saravan Rajmohan  

**Link**: [PDF](https://arxiv.org/pdf/2602.21227)  

**Abstract**: As large language models (LLMs) evolve into autonomous agents that execute long-horizon workflows, invoking a high-capability model at every step becomes economically unsustainable. While model routing is effective for single-turn queries, agentic routing is a sequential, path-dependent problem: early mistakes compound, feedback is often at the end of the episode, and deployments often demand strict per-task spending limits. We propose Budget-Aware Agentic Routing, which selects between a cheap and an expensive model at each step to optimize the cost--success frontier and to operate under strict per-task budgets. We propose Boundary-Guided Training, which leverages two boundary policies (always-small vs.\ always-large) to build a difficulty taxonomy and to anchor learning under sparse rewards. Our approach warms start with boundary-guided SFT data synthesis via stratified sampling of cost-efficient trajectories, then applies Boundary-Guided Policy Optimization (BoPO), combining boundary-relative rewards with a reference-guided advantage to avoid degenerate cheap-failure solutions. Experiment results show that our method improves the efficiency frontier, matching strong routing baselines at substantially lower cost while demonstrating generalization to strict inference-time budget constraints. Overall, our work establishes a foundational framework for agentic routing, shifting the paradigm from static model selection to dynamic, budget-aware sequential decision-making. 

---
# IslamicLegalBench: Evaluating LLMs Knowledge and Reasoning of Islamic Law Across 1,200 Years of Islamic Pluralist Legal Traditions 

**Authors**: Ezieddin Elmahjub, Junaid Qadir, Abdullah Mushtaq, Rafay Naeem, Ibrahim Ghaznavi, Waleed Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2602.21226)  

**Abstract**: As millions of Muslims turn to LLMs like GPT, Claude, and DeepSeek for religious guidance, a critical question arises: Can these AI systems reliably reason about Islamic law? We introduce IslamicLegalBench, the first benchmark evaluating LLMs across seven schools of Islamic jurisprudence, with 718 instances covering 13 tasks of varying complexity. Evaluation of nine state-of-the-art models reveals major limitations: the best model achieves only 68% correctness with 21% hallucination, while several models fall below 35% correctness and exceed 55% hallucination. Few-shot prompting provides minimal gains, improving only 2 of 9 models by >1%. Moderate-complexity tasks requiring exact knowledge show the highest errors, whereas high-complexity tasks display apparent competence through semantic reasoning. False premise detection indicates risky sycophancy, with 6 of 9 models accepting misleading assumptions at rates above 40%. These results highlight that prompt-based methods cannot compensate for missing foundational knowledge. IslamicLegalBench offers the first systematic framework to evaluate Islamic legal reasoning in AI, revealing critical gaps in tools increasingly relied on for spiritual guidance. 

---
# Architecture-Agnostic Curriculum Learning for Document Understanding: Empirical Evidence from Text-Only and Multimodal 

**Authors**: Mohammed Hamdan, Vincenzo Dentamaro, Giuseppe Pirlo, Mohamed Cheriet  

**Link**: [PDF](https://arxiv.org/pdf/2602.21225)  

**Abstract**: We investigate whether progressive data scheduling -- a curriculum learning strategy that incrementally increases training data exposure (33\%$\rightarrow$67\%$\rightarrow$100\%) -- yields consistent efficiency gains across architecturally distinct document understanding models. By evaluating BERT (text-only, 110M parameters) and LayoutLMv3 (multimodal, 126M parameters) on the FUNSD and CORD benchmarks, we establish that this schedule reduces wall-clock training time by approximately 33\%, commensurate with the reduction from 6.67 to 10.0 effective epoch-equivalents of data. To isolate curriculum effects from compute reduction, we introduce matched-compute baselines (Standard-7) that control for total gradient updates. On the FUNSD dataset, the curriculum significantly outperforms the matched-compute baseline for BERT ($\Delta$F1 = +0.023, $p=0.022$, $d_z=3.83$), constituting evidence for a genuine scheduling benefit in capacity-constrained models. In contrast, no analogous benefit is observed for LayoutLMv3 ($p=0.621$), whose multimodal representations provide sufficient inductive bias. On the CORD dataset, all conditions converge to equivalent F1 scores ($\geq$0.947) irrespective of scheduling, indicating a performance ceiling. Schedule ablations comparing progressive, two-phase, reverse, and random pacing confirm that the efficiency gain derives from reduced data volume rather than ordering. Taken together, these findings demonstrate that progressive scheduling is a reliable compute-reduction strategy across model families, with curriculum-specific benefits contingent on the interaction between model capacity and task complexity. 

---
# Make Every Draft Count: Hidden State based Speculative Decoding 

**Authors**: Yuetao Chen, Xuliang Wang, Xinzhou Zheng, Ming Li, Peng Wang, Hong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.21224)  

**Abstract**: Speculative decoding has emerged as a pivotal technique to accelerate LLM inference by employing a lightweight draft model to generate candidate tokens that are subsequently verified by the target model in parallel. However, while this paradigm successfully increases the arithmetic intensity of memory-bound inference, it causes significant compute inefficiency: the majority of draft tokens fail verification and are discarded, resulting in waste of computation. Motivated by the goal of recollecting this wasted computation, we propose a novel system that transforms discarded drafts into reusable tokens. Our key insight is to perform auto-regressive prediction at the hidden states level and postpone the integrating token information after the hidden states generation, so the draft hidden states are not contaminated by incorrect tokens, enabling hidden state reuse. To implement such a system, first we introduce a draft model architecture based on auto-regressive hidden states, which preserves richer semantics than token-based drafters to facilitate draft repurposing. Second, we design an efficient token information injection mechanism that leverages our specialized draft model to construct high-quality draft token trees and enables resampling tokens from verification failures. Third, we eliminate the overhead hidden in our design to further maximize hardware utilization. We conducted extensive evaluations against various baselines, demonstrating up to a 3.3x speedup against standard speculative decoding. 

---
# Measuring Pragmatic Influence in Large Language Model Instructions 

**Authors**: Yilin Geng, Omri Abend, Eduard Hovy, Lea Frermann  

**Link**: [PDF](https://arxiv.org/pdf/2602.21223)  

**Abstract**: It is not only what we ask large language models (LLMs) to do that matters, but also how we prompt. Phrases like "This is urgent" or "As your supervisor" can shift model behavior without altering task content. We study this effect as pragmatic framing, contextual cues that shape directive interpretation rather than task specification. While prior work exploits such cues for prompt optimization or probes them as security vulnerabilities, pragmatic framing itself has not been treated as a measurable property of instruction following. Measuring this influence systematically remains challenging, requiring controlled isolation of framing cues. We introduce a framework with three novel components: directive-framing decomposition separating framing context from task specification; a taxonomy organizing 400 instantiations of framing into 13 strategies across 4 mechanism clusters; and priority-based measurement that quantifies influence through observable shifts in directive prioritization. Across five LLMs of different families and sizes, influence mechanisms cause consistent and structured shifts in directive prioritization, moving models from baseline impartiality toward favoring the framed directive. This work establishes pragmatic framing as a measurable and predictable factor in instruction-following systems. 

---
# Task-Aware LoRA Adapter Composition via Similarity Retrieval in Vector Databases 

**Authors**: Riya Adsul, Balachandra Devarangadi Sunil, Isha Nalawade, Sudharshan Govindan  

**Link**: [PDF](https://arxiv.org/pdf/2602.21222)  

**Abstract**: Parameter efficient fine tuning methods like LoRA have enabled task specific adaptation of large language models, but efficiently composing multiple specialized adapters for unseen tasks remains challenging. We present a novel framework for dynamic LoRA adapter composition that leverages similarity retrieval in vector databases to enable zero-shot generalization across diverse NLP tasks. Our approach constructs a task-aware vector database by embedding training examples from 22 datasets spanning commonsense reasoning, question answering, natural language inference, and sentiment analysis. At inference time, we retrieve the most similar training examples, compute task similarity distributions via nucleus sampling, and dynamically merge relevant LoRA adapters using retrieval weighted fusion strategies. We evaluated four merging methods Linear, Concatenation, TIES, and Magnitude Prune demonstrating that our dataset centric retrieval approach often matches or exceeds the performance of individually fine-tuned task-specific adapters. Notably, Linear merging achieves 70.95% on PIQA and 77.62% on RTE, substantially outperforming single-task baselines (46% and 52%, respectively). Our framework requires no additional retriever training, operates with frozen embeddings, and enables efficient, interpretable adapter composition. These results suggest that retrieval based dynamic merging offers a promising direction for scalable, parameter-efficient multitask learning without requiring full model retraining for each new task. 

---
# Latent Context Compilation: Distilling Long Context into Compact Portable Memory 

**Authors**: Zeju Li, Yizhou Zhou, Qiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.21221)  

**Abstract**: Efficient long-context LLM deployment is stalled by a dichotomy between amortized compression, which struggles with out-of-distribution generalization, and Test-Time Training, which incurs prohibitive synthetic data costs and requires modifying model weights, creating stateful parameters that complicate concurrent serving. We propose Latent Context Compilation, a framework that fundamentally shifts context processing from adaptation to compilation. By utilizing a disposable LoRA module as a compiler, we distill long contexts into compact buffer tokens -- stateless, portable memory artifacts that are plug-and-play compatible with frozen base models. Crucially, we introduce a self-aligned optimization strategy that eliminates the need for synthetic context-relevant QA pairs. By regularizing context reconstruction task with context-agnostic random queries, we force compressed tokens to reside within the model's existing instruction-following manifold. Experiments with Llama-3.1-8B demonstrate that Latent Context Compilation preserves fine-grained details and reasoning capabilities where prior methods falter, effectively decoupling memory density from model parameters even at a 16x compression ratio. 

---
# Field-Theoretic Memory for AI Agents: Continuous Dynamics for Context Preservation 

**Authors**: Subhadip Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2602.21220)  

**Abstract**: We present a memory system for AI agents that treats stored information as continuous fields governed by partial differential equations rather than discrete entries in a database. The approach draws from classical field theory: memories diffuse through semantic space, decay thermodynamically based on importance, and interact through field coupling in multi-agent scenarios. We evaluate the system on two established long-context benchmarks: LoCoMo (ACL 2024) with 300-turn conversations across 35 sessions, and LongMemEval (ICLR 2025) testing multi-session reasoning over 500+ turns. On LongMemEval, the field-theoretic approach achieves significant improvements: +116% F1 on multi-session reasoning (p<0.01, d= 3.06), +43.8% on temporal reasoning (p<0.001, d= 9.21), and +27.8% retrieval recall on knowledge updates (p<0.001, d= 5.00). Multi-agent experiments show near-perfect collective intelligence (>99.8%) through field coupling. Code is available at this http URL. 

---
# Reasoning-Based Personalized Generation for Users with Sparse Data 

**Authors**: Bo Ni, Branislav Kveton, Samyadeep Basu, Subhojyoti Mukherjee, Leyao Wang, Franck Dernoncourt, Sungchul Kim, Seunghyun Yoon, Zichao Wang, Ruiyi Zhang, Puneet Mathur, Jihyung Kil, Jiuxiang Gu, Nedim Lipka, Yu Wang, Ryan A. Rossi, Tyler Derr  

**Link**: [PDF](https://arxiv.org/pdf/2602.21219)  

**Abstract**: Large Language Model (LLM) personalization holds great promise for tailoring responses by leveraging personal context and history. However, real-world users usually possess sparse interaction histories with limited personal context, such as cold-start users in social platforms and newly registered customers in online E-commerce platforms, compromising the LLM-based personalized generation. To address this challenge, we introduce GraSPer (Graph-based Sparse Personalized Reasoning), a novel framework for enhancing personalized text generation under sparse context. GraSPer first augments user context by predicting items that the user would likely interact with in the future. With reasoning alignment, it then generates texts for these interactions to enrich the augmented context. In the end, it generates personalized outputs conditioned on both the real and synthetic histories, ensuring alignment with user style and preferences. Extensive experiments on three benchmark personalized generation datasets show that GraSPer achieves significant performance gain, substantially improving personalization in sparse user context settings. 

---
# EPSVec: Efficient and Private Synthetic Data Generation via Dataset Vectors 

**Authors**: Amin Banayeeanzade, Qingchuan Yang, Deqing Fu, Spencer Hong, Erin Babinsky, Alfy Samuel, Anoop Kumar, Robin Jia, Sai Praneeth Karimireddy  

**Link**: [PDF](https://arxiv.org/pdf/2602.21218)  

**Abstract**: High-quality data is essential for modern machine learning, yet many valuable corpora are sensitive and cannot be freely shared. Synthetic data offers a practical substitute for downstream development, and large language models (LLMs) have emerged as powerful engines for generating it. However, existing private text generation methods are severely inefficient: they are data-intensive, computationally slow, and often require large private corpora or batch sizes to achieve usable quality. We introduce EPSVec, a differentially-private lightweight alternative that steers LLM generation using *dataset vectors*--directions in activation space that capture the distributional gap between private data and public priors. EPSVec extracts and sanitizes steering vectors just once and then performs standard decoding. This decouples the privacy budget from generation, enabling arbitrarily many synthetic samples without additional privacy cost and yielding strong fidelity even in low-data regimes. Furthermore, we enhance our method by utilizing pretrained (base) models and introducing fixed-shot prompting to boost generation diversity and fidelity. Our experiments demonstrate that EPSVec outperforms existing baselines in distributional alignment and downstream utility, particularly in low-data regimes, while significantly reducing computational overhead. 

---
# Applied Sociolinguistic AI for Community Development (ASA-CD): A New Scientific Paradigm for Linguistically-Grounded Social Intervention 

**Authors**: S M Ruhul Alam, Rifa Ferzana  

**Link**: [PDF](https://arxiv.org/pdf/2602.21217)  

**Abstract**: This paper establishes Applied Sociolinguistic AI for Community Development (ASA-CD) as a novel scientific paradigm for addressing community challenges through linguistically grounded, AI-enabled intervention. ASA-CD introduces three key contributions: (1) linguistic biomarkers as computational indicators of discursive fragmentation; (2) development-aligned natural language processing (NLP), an AI optimisation paradigm prioritising collective outcomes; and (3) a standardised five-phase protocol for discursive intervention. A proof-of-concept study, incorporating real-world and synthetic corpora, demonstrates systematic associations between exclusionary language and negative sentiment and simulates intervention-based improvements. ASA-CD provides a unified methodological, ethical and empirical framework for scalable, value-aligned AI in the service of community empowerment. 

---
# EQ-5D Classification Using Biomedical Entity-Enriched Pre-trained Language Models and Multiple Instance Learning 

**Authors**: Zhyar Rzgar K Rostam, Gábor Kertész  

**Link**: [PDF](https://arxiv.org/pdf/2602.21216)  

**Abstract**: The EQ-5D (EuroQol 5-Dimensions) is a standardized instrument for the evaluation of health-related quality of life. In health economics, systematic literature reviews (SLRs) depend on the correct identification of publications that use the EQ-5D, but manual screening of large volumes of scientific literature is time-consuming, error-prone, and inconsistent. In this study, we investigate fine-tuning of general-purpose (BERT) and domain-specific (SciBERT, BioBERT) pre-trained language models (PLMs), enriched with biomedical entity information extracted through scispaCy models for each statement, to improve EQ-5D detection from abstracts. We conduct nine experimental setups, including combining three scispaCy models with three PLMs, and evaluate their performance at both the sentence and study levels. Furthermore, we explore a Multiple Instance Learning (MIL) approach with attention pooling to aggregate sentence-level information into study-level predictions, where each abstract is represented as a bag of enriched sentences (by scispaCy). The findings indicate consistent improvements in F1-scores (reaching 0.82) and nearly perfect recall at the study-level, significantly exceeding classical bag-of-words baselines and recently reported PLM baselines. These results show that entity enrichment significantly improves domain adaptation and model generalization, enabling more accurate automated screening in systematic reviews. 

---
# Inference-time Alignment via Sparse Junction Steering 

**Authors**: Runyi Hu, Jie Zhang, Shiqian Zhao, Jiale Meng, Jiwei Li, Jason Zeng, Ming Wu, Michael Heinrich, Yonggang Wen, Tianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.21215)  

**Abstract**: Token-level steering has emerged as a pivotal approach for inference-time alignment, enabling fine grained control over large language models by modulating their output distributions without parameter updates. While effective, existing methods rely on dense intervention at every decoding step. This persistent manipulation not only incurs substantial computational overhead but also risks compromising generation quality by excessively drifting from the model's intrinsic distribution. In this work, we show that dense intervention is unnecessary and propose Sparse Inference time Alignment (SIA), which performs sparse junction steering by intervening only at critical decision points along the generation trajectory. Our key insight is that high entropy junctions mark pivotal decision points in the generation trajectory and are particularly susceptible to misalignment, indicating the need to introduce alignment related reward signals at these points. Extensive experiments across different model families and alignment objectives show that steering only 20% to 80% of tokens achieves superior alignment-efficiency trade offs. For strong base models such as Qwen3, intervening on as few as 20% of tokens matches or even surpasses heavily post-trained instruct models. This sparsity enables stronger guidance while better preserving the model's native distribution, integrates seamlessly with search based methods such as Best-of-N, and reduces computational cost by up to 6x. 

---
