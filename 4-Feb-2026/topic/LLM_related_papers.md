# EHRWorld: A Patient-Centric Medical World Model for Long-Horizon Clinical Trajectories 

**Authors**: Linjie Mu, Zhongzhen Huang, Yannian Gu, Shengqian Qin, Shaoting Zhang, Xiaofan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03569)  

**Abstract**: World models offer a principled framework for simulating future states under interventions, but realizing such models in complex, high-stakes domains like medicine remains challenging. Recent large language models (LLMs) have achieved strong performance on static medical reasoning tasks, raising the question of whether they can function as dynamic medical world models capable of simulating disease progression and treatment outcomes over time. In this work, we show that LLMs only incorporating medical knowledge struggle to maintain consistent patient states under sequential interventions, leading to error accumulation in long-horizon clinical simulation. To address this limitation, we introduce EHRWorld, a patient-centric medical world model trained under a causal sequential paradigm, together with EHRWorld-110K, a large-scale longitudinal clinical dataset derived from real-world electronic health records. Extensive evaluations demonstrate that EHRWorld significantly outperforms naive LLM-based baselines, achieving more stable long-horizon simulation, improved modeling of clinically sensitive events, and favorable reasoning efficiency, highlighting the necessity of training on causally grounded, temporally evolving clinical data for reliable and robust medical world modeling. 

---
# Conformal Thinking: Risk Control for Reasoning on a Compute Budget 

**Authors**: Xi Wang, Anushri Suresh, Alvin Zhang, Rishi More, William Jurayj, Benjamin Van Durme, Mehrdad Farajtabar, Daniel Khashabi, Eric Nalisnick  

**Link**: [PDF](https://arxiv.org/pdf/2602.03814)  

**Abstract**: Reasoning Large Language Models (LLMs) enable test-time scaling, with dataset-level accuracy improving as the token budget increases, motivating adaptive reasoning -- spending tokens when they improve reliability and stopping early when additional computation is unlikely to help. However, setting the token budget, as well as the threshold for adaptive reasoning, is a practical challenge that entails a fundamental risk-accuracy trade-off. We re-frame the budget setting problem as risk control, limiting the error rate while minimizing compute. Our framework introduces an upper threshold that stops reasoning when the model is confident (risking incorrect output) and a novel parametric lower threshold that preemptively stops unsolvable instances (risking premature stoppage). Given a target risk and a validation set, we use distribution-free risk control to optimally specify these stopping mechanisms. For scenarios with multiple budget controlling criteria, we incorporate an efficiency loss to select the most computationally efficient exiting mechanism. Empirical results across diverse reasoning tasks and models demonstrate the effectiveness of our risk control approach, demonstrating computational efficiency gains from the lower threshold and ensemble stopping mechanisms while adhering to the user-specified risk target. 

---
# Mitigating Conversational Inertia in Multi-Turn Agents 

**Authors**: Yang Wan, Zheng Cao, Zhenhao Zhang, Zhengwen Zeng, Shuheng Shen, Changhua Meng, Linchao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2602.03664)  

**Abstract**: Large language models excel as few-shot learners when provided with appropriate demonstrations, yet this strength becomes problematic in multiturn agent scenarios, where LLMs erroneously mimic their own previous responses as few-shot examples. Through attention analysis, we identify conversational inertia, a phenomenon where models exhibit strong diagonal attention to previous responses, which is associated with imitation bias that constrains exploration. This reveals a tension when transforming few-shot LLMs into agents: longer context enriches environmental feedback for exploitation, yet also amplifies conversational inertia that undermines exploration. Our key insight is that for identical states, actions generated with longer contexts exhibit stronger inertia than those with shorter contexts, enabling construction of preference pairs without environment rewards. Based on this, we propose Context Preference Learning to calibrate model preferences to favor low-inertia responses over highinertia ones. We further provide context management strategies at inference time to balance exploration and exploitation. Experimental results across eight agentic environments and one deep research scenario validate that our framework reduces conversational inertia and achieves performance improvements. 

---
# Can LLMs Do Rocket Science? Exploring the Limits of Complex Reasoning with GTOC 12 

**Authors**: Iñaki del Campo, Pablo Cuervo, Victor Rodriguez-Fernandez, Roberto Armellin, Jack Yarndley  

**Link**: [PDF](https://arxiv.org/pdf/2602.03630)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable proficiency in code generation and general reasoning, yet their capacity for autonomous multi-stage planning in high-dimensional, physically constrained environments remains an open research question. This study investigates the limits of current AI agents by evaluating them against the 12th Global Trajectory Optimization Competition (GTOC 12), a complex astrodynamics challenge requiring the design of a large-scale asteroid mining campaign. We adapt the MLE-Bench framework to the domain of orbital mechanics and deploy an AIDE-based agent architecture to autonomously generate and refine mission solutions. To assess performance beyond binary validity, we employ an "LLM-as-a-Judge" methodology, utilizing a rubric developed by domain experts to evaluate strategic viability across five structural categories. A comparative analysis of models, ranging from GPT-4-Turbo to reasoning-enhanced architectures like Gemini 2.5 Pro, and o3, reveals a significant trend: the average strategic viability score has nearly doubled in the last two years (rising from 9.3 to 17.2 out of 26). However, we identify a critical capability gap between strategy and execution. While advanced models demonstrate sophisticated conceptual understanding, correctly framing objective functions and mission architectures, they consistently fail at implementation due to physical unit inconsistencies, boundary condition errors, and inefficient debugging loops. We conclude that, while current LLMs often demonstrate sufficient knowledge and intelligence to tackle space science tasks, they remain limited by an implementation barrier, functioning as powerful domain facilitators rather than fully autonomous engineers. 

---
# IntentRL: Training Proactive User-intent Agents for Open-ended Deep Research via Reinforcement Learning 

**Authors**: Haohao Luo, Zexi Li, Yuexiang Xie, Wenhao Zhang, Yaliang Li, Ying Shen  

**Link**: [PDF](https://arxiv.org/pdf/2602.03468)  

**Abstract**: Deep Research (DR) agents extend Large Language Models (LLMs) beyond parametric knowledge by autonomously retrieving and synthesizing evidence from large web corpora into long-form reports, enabling a long-horizon agentic paradigm. However, unlike real-time conversational assistants, DR is computationally expensive and time-consuming, creating an autonomy-interaction dilemma: high autonomy on ambiguous user queries often leads to prolonged execution with unsatisfactory outcomes. To address this, we propose IntentRL, a framework that trains proactive agents to clarify latent user intents before starting long-horizon research. To overcome the scarcity of open-ended research data, we introduce a scalable pipeline that expands a few seed samples into high-quality dialogue turns via a shallow-to-deep intent refinement graph. We further adopt a two-stage reinforcement learning (RL) strategy: Stage I applies RL on offline dialogues to efficiently learn general user-interaction behavior, while Stage II uses the trained agent and a user simulator for online rollouts to strengthen adaptation to diverse user feedback. Extensive experiments show that IntentRL significantly improves both intent hit rate and downstream task performance, outperforming the built-in clarify modules of closed-source DR agents and proactive LLM baselines. 

---
# Ontology-to-tools compilation for executable semantic constraint enforcement in LLM agents 

**Authors**: Xiaochi Zhou, Patrick Bulter, Changxuan Yang, Simon D. Rihm, Thitikarn Angkanaporn, Jethro Akroyd, Sebastian Mosbach, Markus Kraft  

**Link**: [PDF](https://arxiv.org/pdf/2602.03439)  

**Abstract**: We introduce ontology-to-tools compilation as a proof-of-principle mechanism for coupling large language models (LLMs) with formal domain knowledge. Within The World Avatar (TWA), ontological specifications are compiled into executable tool interfaces that LLM-based agents must use to create and modify knowledge graph instances, enforcing semantic constraints during generation rather than through post-hoc validation. Extending TWA's semantic agent composition framework, the Model Context Protocol (MCP) and associated agents are integral components of the knowledge graph ecosystem, enabling structured interaction between generative models, symbolic constraints, and external resources. An agent-based workflow translates ontologies into ontology-aware tools and iteratively applies them to extract, validate, and repair structured knowledge from unstructured scientific text. Using metal-organic polyhedra synthesis literature as an illustrative case, we show how executable ontological semantics can guide LLM behaviour and reduce manual schema and prompt engineering, establishing a general paradigm for embedding formal knowledge into generative systems. 

---
# Risk Awareness Injection: Calibrating Vision-Language Models for Safety without Compromising Utility 

**Authors**: Mengxuan Wang, Yuxin Chen, Gang Xu, Tao He, Hongjie Jiang, Ming Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.03402)  

**Abstract**: Vision language models (VLMs) extend the reasoning capabilities of large language models (LLMs) to cross-modal settings, yet remain highly vulnerable to multimodal jailbreak attacks. Existing defenses predominantly rely on safety fine-tuning or aggressive token manipulations, incurring substantial training costs or significantly degrading utility. Recent research shows that LLMs inherently recognize unsafe content in text, and the incorporation of visual inputs in VLMs frequently dilutes risk-related signals. Motivated by this, we propose Risk Awareness Injection (RAI), a lightweight and training-free framework for safety calibration that restores LLM-like risk recognition by amplifying unsafe signals in VLMs. Specifically, RAI constructs an Unsafe Prototype Subspace from language embeddings and performs targeted modulation on selected high-risk visual tokens, explicitly activating safety-critical signals within the cross-modal feature space. This modulation restores the model's LLM-like ability to detect unsafe content from visual inputs, while preserving the semantic integrity of original tokens for cross-modal reasoning. Extensive experiments across multiple jailbreak and utility benchmarks demonstrate that RAI substantially reduces attack success rate without compromising task performance. 

---
# GFlowPO: Generative Flow Network as a Language Model Prompt Optimizer 

**Authors**: Junmo Cho, Suhan Kim, Sangjune An, Minsu Kim, Dong Bok Lee, Heejun Lee, Sung Ju Hwang, Hae Beom Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.03358)  

**Abstract**: Finding effective prompts for language models (LMs) is critical yet notoriously difficult: the prompt space is combinatorially large, rewards are sparse due to expensive target-LM evaluation. Yet, existing RL-based prompt optimizers often rely on on-policy updates and a meta-prompt sampled from a fixed distribution, leading to poor sample efficiency. We propose GFlowPO, a probabilistic prompt optimization framework that casts prompt search as a posterior inference problem over latent prompts regularized by a meta-prompted reference-LM prior. In the first step, we fine-tune a lightweight prompt-LM with an off-policy Generative Flow Network (GFlowNet) objective, using a replay-based training policy that reuses past prompt evaluations to enable sample-efficient exploration. In the second step, we introduce Dynamic Memory Update (DMU), a training-free mechanism that updates the meta-prompt by injecting both (i) diverse prompts from a replay buffer and (ii) top-performing prompts from a small priority queue, thereby progressively concentrating the search process on high-reward regions. Across few-shot text classification, instruction induction benchmarks, and question answering tasks, GFlowPO consistently outperforms recent discrete prompt optimization baselines. 

---
# DiscoverLLM: From Executing Intents to Discovering Them 

**Authors**: Tae Soo Kim, Yoonjoo Lee, Jaesang Yu, John Joon Young Chung, Juho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.03429)  

**Abstract**: To handle ambiguous and open-ended requests, Large Language Models (LLMs) are increasingly trained to interact with users to surface intents they have not yet expressed (e.g., ask clarification questions). However, users are often ambiguous because they have not yet formed their intents: they must observe and explore outcomes to discover what they want. Simply asking "what kind of tone do you want?" fails when users themselves do not know. We introduce DiscoverLLM, a novel and generalizable framework that trains LLMs to help users form and discover their intents. Central to our approach is a novel user simulator that models cognitive state with a hierarchy of intents that progressively concretize as the model surfaces relevant options -- where the degree of concretization serves as a reward signal that models can be trained to optimize. Resulting models learn to collaborate with users by adaptively diverging (i.e., explore options) when intents are unclear, and converging (i.e., refine and implement) when intents concretize. Across proposed interactive benchmarks in creative writing, technical writing, and SVG drawing, DiscoverLLM achieves over 10% higher task performance while reducing conversation length by up to 40%. In a user study with 75 human participants, DiscoverLLM improved conversation satisfaction and efficiency compared to baselines. 

---
# MentalSeek-Dx: Towards Progressive Hypothetico-Deductive Reasoning for Real-world Psychiatric Diagnosis 

**Authors**: Xiao Sun, Yuming Yang, Junnan Zhu, Jiang Zhong, Xinyu Zhou, Kaiwen Wei  

**Link**: [PDF](https://arxiv.org/pdf/2602.03340)  

**Abstract**: Mental health disorders represent a burgeoning global public health challenge. While Large Language Models (LLMs) have demonstrated potential in psychiatric assessment, their clinical utility is severely constrained by benchmarks that lack ecological validity and fine-grained diagnostic supervision. To bridge this gap, we introduce \textbf{MentalDx Bench}, the first benchmark dedicated to disorder-level psychiatric diagnosis within real-world clinical settings. Comprising 712 de-identified electronic health records annotated by board-certified psychiatrists under ICD-11 guidelines, the benchmark covers 76 disorders across 16 diagnostic categories. Evaluation of 18 LLMs reveals a critical \textit{paradigm misalignment}: strong performance at coarse diagnostic categorization contrasts with systematic failure at disorder-level diagnosis, underscoring a gap between pattern-based modeling and clinical hypothetico-deductive reasoning. In response, we propose \textbf{MentalSeek-Dx}, a medical-specialized LLM trained to internalize this clinical reasoning process through supervised trajectory construction and curriculum-based reinforcement learning. Experiments on MentalDx Bench demonstrate that MentalSeek-Dx achieves state-of-the-art (SOTA) performance with only 14B parameters, establishing a clinically grounded framework for reliable psychiatric diagnosis. 

---
# Agentic Proposing: Enhancing Large Language Model Reasoning via Compositional Skill Synthesis 

**Authors**: Zhengbo Jiao, Shaobo Wang, Zifan Zhang, Xuan Ren, Wei Wang, Bing Zhao, Hu Wei, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03279)  

**Abstract**: Advancing complex reasoning in large language models relies on high-quality, verifiable datasets, yet human annotation remains cost-prohibitive and difficult to scale. Current synthesis paradigms often face a recurring trade-off: maintaining structural validity typically restricts problem complexity, while relaxing constraints to increase difficulty frequently leads to inconsistent or unsolvable instances. To address this, we propose Agentic Proposing, a framework that models problem synthesis as a goal-driven sequential decision process where a specialized agent dynamically selects and composes modular reasoning skills. Through an iterative workflow of internal reflection and tool-use, we develop the Agentic-Proposer-4B using Multi-Granularity Policy Optimization (MGPO) to generate high-precision, verifiable training trajectories across mathematics, coding, and science. Empirical results demonstrate that downstream solvers trained on agent-synthesized data significantly outperform leading baselines and exhibit robust cross-domain generalization. Notably, a 30B solver trained on only 11,000 synthesized trajectories achieves a state-of-the-art 91.6% accuracy on AIME25, rivaling frontier-scale proprietary models such as GPT-5 and proving that a small volume of high-quality synthetic signals can effectively substitute for massive human-curated datasets. 

---
# The Necessity of a Unified Framework for LLM-Based Agent Evaluation 

**Authors**: Pengyu Zhu, Li Sun, Philip S. Yu, Sen Su  

**Link**: [PDF](https://arxiv.org/pdf/2602.03238)  

**Abstract**: With the advent of Large Language Models (LLMs), general-purpose agents have seen fundamental advancements. However, evaluating these agents presents unique challenges that distinguish them from static QA benchmarks. We observe that current agent benchmarks are heavily confounded by extraneous factors, including system prompts, toolset configurations, and environmental dynamics. Existing evaluations often rely on fragmented, researcher-specific frameworks where the prompt engineering for reasoning and tool usage varies significantly, making it difficult to attribute performance gains to the model itself. Additionally, the lack of standardized environmental data leads to untraceable errors and non-reproducible results. This lack of standardization introduces substantial unfairness and opacity into the field. We propose that a unified evaluation framework is essential for the rigorous advancement of agent evaluation. To this end, we introduce a proposal aimed at standardizing agent evaluation. 

---
# VALUEFLOW: Toward Pluralistic and Steerable Value-based Alignment in Large Language Models 

**Authors**: Woojin Kim, Sieun Hyeon, Jusang Oh, Jaeyoung Do  

**Link**: [PDF](https://arxiv.org/pdf/2602.03160)  

**Abstract**: Aligning Large Language Models (LLMs) with the diverse spectrum of human values remains a central challenge: preference-based methods often fail to capture deeper motivational principles. Value-based approaches offer a more principled path, yet three gaps persist: extraction often ignores hierarchical structure, evaluation detects presence but not calibrated intensity, and the steerability of LLMs at controlled intensities remains insufficiently understood. To address these limitations, we introduce VALUEFLOW, the first unified framework that spans extraction, evaluation, and steering with calibrated intensity control. The framework integrates three components: (i) HIVES, a hierarchical value embedding space that captures intra- and cross-theory value structure; (ii) the Value Intensity DataBase (VIDB), a large-scale resource of value-labeled texts with intensity estimates derived from ranking-based aggregation; and (iii) an anchor-based evaluator that produces consistent intensity scores for model outputs by ranking them against VIDB panels. Using VALUEFLOW, we conduct a comprehensive large-scale study across ten models and four value theories, identifying asymmetries in steerability and composition laws for multi-value control. This paper establishes a scalable infrastructure for evaluating and controlling value intensity, advancing pluralistic alignment of LLMs. 

---
# STAR: Similarity-guided Teacher-Assisted Refinement for Super-Tiny Function Calling Models 

**Authors**: Jiliang Ni, Jiachen Pu, Zhongyi Yang, Jingfeng Luo, Conggang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2602.03022)  

**Abstract**: The proliferation of Large Language Models (LLMs) in function calling is pivotal for creating advanced AI agents, yet their large scale hinders widespread adoption, necessitating transferring their capabilities into smaller ones. However, existing paradigms are often plagued by overfitting, training instability, ineffective binary rewards for multi-solution tasks, and the difficulty of synergizing techniques. We introduce STAR: Similarity-guided Teacher-Assisted Refinement, a novel holistic framework that effectively transfers LLMs' capabilities to super-tiny models. STAR consists of two core technical innovations: (1) Constrained Knowledge Distillation (CKD), a training objective that augments top-k forward KL divergence to suppress confidently incorrect predictions, ensuring training stability while preserving exploration capacity for downstream RL. STAR holistically synergizes these strategies within a cohesive training curriculum, enabling super-tiny models to achieve exceptional performance on complex function calling tasks; (2) Similarity-guided RL (Sim-RL), a RL mechanism that introduces a fine-grained, similarity-based reward. This provides a robust, continuous, and rich signal for better policy optimization by evaluating the similarity between generated outputs and the ground truth. Extensive experiments on challenging and renowned benchmarks demonstrate the effectiveness of our method. Our STAR models establish SOTA in their size classes, significantly outperforming baselines. Remarkably, our 0.6B STAR model achieves the best performance among all open models under 1B, surpassing even several well-known open models at a larger scale. STAR demonstrates a training framework that distills capabilities of LLMs into super-tiny models, paving the way for powerful, accessible, and efficient AI agents. 

---
# Distilling LLM Reasoning into Graph of Concept Predictors 

**Authors**: Ziyang Yu, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.03006)  

**Abstract**: Deploying Large Language Models (LLMs) for discriminative workloads is often limited by inference latency, compute, and API costs at scale. Active distillation reduces these costs by querying an LLM oracle to train compact discriminative students, but most pipelines distill only final labels, discarding intermediate reasoning signals and offering limited diagnostics of what reasoning is missing and where errors arise. We propose Graph of Concept Predictors (GCP), a reasoning-aware active distillation framework that externalizes the teacher's decision process as a directed acyclic graph and mirrors it with modular concept predictors in the student. GCP enhances sample efficiency through a graph-aware acquisition strategy that targets uncertainty and disagreement at critical reasoning nodes. Additionally, it improves training stability and efficiency by performing targeted sub-module retraining, which attributes downstream loss to specific concept predictors and updates only the most influential modules. Experiments on eight NLP classification benchmarks demonstrate that GCP enhances performance under limited annotation budgets while yielding more interpretable and controllable training dynamics. Code is available at: this https URL. 

---
# Are LLMs Biased Like Humans? Causal Reasoning as a Function of Prior Knowledge, Irrelevant Information, and Reasoning Budget 

**Authors**: Hanna M. Dettki, Charley M. Wu, Bob Rehder  

**Link**: [PDF](https://arxiv.org/pdf/2602.02983)  

**Abstract**: Large language models (LLMs) are increasingly used in domains where causal reasoning matters, yet it remains unclear whether their judgments reflect normative causal computation, human-like shortcuts, or brittle pattern matching. We benchmark 20+ LLMs against a matched human baseline on 11 causal judgment tasks formalized by a collider structure ($C_1 \!\rightarrow\! E\! \leftarrow \!C_2$). We find that a small interpretable model compresses LLMs' causal judgments well and that most LLMs exhibit more rule-like reasoning strategies than humans who seem to account for unmentioned latent factors in their probability judgments. Furthermore, most LLMs do not mirror the characteristic human collider biases of weak explaining away and Markov violations. We probe LLMs' causal judgment robustness under (i) semantic abstraction and (ii) prompt overloading (injecting irrelevant text), and find that chain-of-thought (CoT) increases robustness for many LLMs. Together, this divergence suggests LLMs can complement humans when known biases are undesirable, but their rule-like reasoning may break down when uncertainty is intrinsic -- highlighting the need to characterize LLM reasoning strategies for safe, effective deployment. 

---
# Large Language Models Can Take False First Steps at Inference-time Planning 

**Authors**: Haijiang Yan, Jian-Qiao Zhu, Adam Sanborn  

**Link**: [PDF](https://arxiv.org/pdf/2602.02991)  

**Abstract**: Large language models (LLMs) have been shown to acquire sequence-level planning abilities during training, yet their planning behavior exhibited at inference time often appears short-sighted and inconsistent with these capabilities. We propose a Bayesian account for this gap by grounding planning behavior in the evolving generative context: given the subtle differences between natural language and the language internalized by LLMs, accumulated self-generated context drives a planning-shift during inference and thereby creates the appearance of compromised planning behavior. We further validate the proposed model through two controlled experiments: a random-generation task demonstrating constrained planning under human prompts and increasing planning strength as self-generated context accumulates, and a Gaussian-sampling task showing reduced initial bias when conditioning on self-generated sequences. These findings provide a theoretical explanation along with empirical evidence for characterizing how LLMs plan ahead during inference. 

---
# FIRE-Bench: Evaluating Agents on the Rediscovery of Scientific Insights 

**Authors**: Zhen Wang, Fan Bai, Zhongyan Luo, Jinyan Su, Kaiser Sun, Xinle Yu, Jieyuan Liu, Kun Zhou, Claire Cardie, Mark Dredze, Eric P. Xing, Zhiting Hu  

**Link**: [PDF](https://arxiv.org/pdf/2602.02905)  

**Abstract**: Autonomous agents powered by large language models (LLMs) promise to accelerate scientific discovery end-to-end, but rigorously evaluating their capacity for verifiable discovery remains a central challenge. Existing benchmarks face a trade-off: they either heavily rely on LLM-as-judge evaluations of automatically generated research outputs or optimize convenient yet isolated performance metrics that provide coarse proxies for scientific insight. To address this gap, we introduce FIRE-Bench (Full-cycle Insight Rediscovery Evaluation), a benchmark that evaluates agents through the rediscovery of established findings from recent, high-impact machine learning research. Agents are given only a high-level research question extracted from a published, verified study and must autonomously explore ideas, design experiments, implement code, execute their plans, and derive conclusions supported by empirical evidence. We evaluate a range of state-of-the-art agents with frontier LLMs backbones like gpt-5 on FIRE-Bench. Our results show that full-cycle scientific research remains challenging for current agent systems: even the strongest agents achieve limited rediscovery success (<50 F1), exhibit high variance across runs, and display recurring failure modes in experimental design, execution, and evidence-based reasoning. FIRE-Bench provides a rigorous and diagnostic framework for measuring progress toward reliable agent-driven scientific discovery. 

---
# Reasoning about Reasoning: BAPO Bounds on Chain-of-Thought Token Complexity in LLMs 

**Authors**: Kiran Tomlinson, Tobias Schnabel, Adith Swaminathan, Jennifer Neville  

**Link**: [PDF](https://arxiv.org/pdf/2602.02909)  

**Abstract**: Inference-time scaling via chain-of-thought (CoT) reasoning is a major driver of state-of-the-art LLM performance, but it comes with substantial latency and compute costs. We address a fundamental theoretical question: how many reasoning tokens are required to solve a problem as input size grows? By extending the bounded attention prefix oracle (BAPO) model--an abstraction of LLMs that quantifies the information flow required to solve a task--we prove lower bounds on the CoT tokens required for three canonical BAPO-hard tasks: binary majority, triplet matching, and graph reachability. We show that each requires $\Omega(n)$ reasoning tokens when the input size is $n$. We complement these results with matching or near-matching upper bounds via explicit constructions. Finally, our experiments with frontier reasoning models show approximately linear reasoning token scaling on these tasks and failures when constrained to smaller reasoning budgets, consistent with our theoretical lower bounds. Together, our results identify fundamental bottlenecks in inference-time compute through CoT and offer a principled tool for analyzing optimal reasoning length. 

---
# "I May Not Have Articulated Myself Clearly": Diagnosing Dynamic Instability in LLM Reasoning at Inference Time 

**Authors**: Jinkun Chen, Fengxiang Cheng, Sijia Han, Vlado Keselj  

**Link**: [PDF](https://arxiv.org/pdf/2602.02863)  

**Abstract**: Reasoning failures in large language models (LLMs) are typically measured only at the end of a generation, yet many failures manifest as a process-level breakdown: the model "loses the thread" mid-reasoning. We study whether such breakdowns are detectable from inference-time observables available in standard APIs (token log probabilities), without any training or fine-tuning. We define a simple instability signal that combines consecutive-step distributional shift (JSD) and uncertainty (entropy), summarize each trace by its peak instability strength, and show that this signal reliably predicts failure. Across GSM8K and HotpotQA, instability strength predicts wrong answers with above-chance AUC and yields monotonic bucket-level accuracy decline at scale across model sizes. Crucially, we show that instability is not uniformly harmful: early instability can reflect subsequent stabilization and a correct final answer (\emph{corrective instability}), whereas late instability is more often followed by failure (\emph{destructive instability}), even at comparable peak magnitudes, indicating that recoverability depends not only on how strongly the distribution changes but also on when such changes occur relative to the remaining decoding horizon. The method is model-agnostic, training-free, and reproducible, and is presented as a diagnostic lens rather than a corrective or control mechanism. 

---
# STEER: Inference-Time Risk Control via Constrained Quality-Diversity Search 

**Authors**: Eric Yang, Jong Ha Lee, Jonathan Amar, Elissa Ye, Yugang Jia  

**Link**: [PDF](https://arxiv.org/pdf/2602.02862)  

**Abstract**: Large Language Models (LLMs) trained for average correctness often exhibit mode collapse, producing narrow decision behaviors on tasks where multiple responses may be reasonable. This limitation is particularly problematic in ordinal decision settings such as clinical triage, where standard alignment removes the ability to trade off specificity and sensitivity (the ROC operating point) based on contextual constraints. We propose STEER (Steerable Tuning via Evolutionary Ensemble Refinement), a training-free framework that reintroduces this tunable control. STEER constructs a population of natural-language personas through an offline, constrained quality-diversity search that promotes behavioral coverage while enforcing minimum safety, reasoning, and stability thresholds. At inference time, STEER exposes a single, interpretable control parameter that maps a user-specified risk percentile to a selected persona, yielding a monotonic adjustment of decision conservativeness. On two clinical triage benchmarks, STEER achieves broader behavioral coverage compared to temperature-based sampling and static persona ensembles. Compared to a representative post-training method, STEER maintains substantially higher accuracy on unambiguous urgent cases while providing comparable control over ambiguous decisions. These results demonstrate STEER as a safety-preserving paradigm for risk control, capable of steering behavior without compromising domain competence. 

---
# Chain of Simulation: A Dual-Mode Reasoning Framework for Large Language Models with Dynamic Problem Routing 

**Authors**: Saeid Sheikhi  

**Link**: [PDF](https://arxiv.org/pdf/2602.02842)  

**Abstract**: We present Chain of Simulation (CoS), a novel dual-mode reasoning framework that dynamically routes problems to specialized reasoning strategies in Large Language Models (LLMs). Unlike existing uniform prompting approaches, CoS employs three distinct reasoning modes: (1) computational flow with self-consistency for mathematical problems, (2) symbolic state tracking with JSON representations for spatial reasoning, and (3) hybrid fact-extraction for multi-hop inference. Through comprehensive evaluation on GSM8K, StrategyQA, and bAbI benchmarks using four state-of-the-art models (Gemma-3 27B, LLaMA-3.1 8B, Mistral 7B, and Qwen-2.5 14B), we demonstrate that CoS achieves 71.5% accuracy on GSM8K (1.0% absolute improvement), 90.0% on StrategyQA (2.5% improvement), and 19.0% on bAbI (65.2% relative improvement) compared to the strongest baselines. The analysis reveals that problem-specific mode selection is crucial, with computational mode achieving 81.2% accuracy when correctly applied to mathematical problems, while misrouting leads to 0% accuracy. We provide detailed algorithms for mode selection, state tracking, and answer extraction, establishing CoS as an effective approach for improving LLM reasoning without additional training. The framework provides superior trade-offs between accuracy and efficiency compared to Self-Consistency, achieving comparable performance at 54% lower computational cost. 

---
# Scaling-Aware Adapter for Structure-Grounded LLM Reasoning 

**Authors**: Zihao Jing, Qiuhao Zeng, Ruiyi Fang, Yan Yi Li, Yan Sun, Boyu Wang, Pingzhao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2602.02780)  

**Abstract**: Large language models (LLMs) are enabling reasoning over biomolecular structures, yet existing methods remain modality-specific and typically compress structural inputs through sequence-based tokenization or fixed-length query connectors. Such architectures either omit the geometric groundings requisite for mitigating structural hallucinations or impose inflexible modality fusion bottlenecks that concurrently over-compress and suboptimally allocate structural tokens, thereby impeding the realization of generalized all-atom reasoning. We introduce Cuttlefish, a unified all-atom LLM that grounds language reasoning in geometric cues while scaling modality tokens with structural complexity. First, Scaling-Aware Patching leverages an instruction-conditioned gating mechanism to generate variable-size patches over structural graphs, adaptively scaling the query token budget with structural complexity to mitigate fixed-length connector bottlenecks. Second, Geometry Grounding Adapter refines these adaptive tokens via cross-attention to modality embeddings and injects the resulting modality tokens into the LLM, exposing explicit geometric cues to reduce structural hallucination. Experiments across diverse all-atom benchmarks demonstrate that Cuttlefish achieves superior performance in heterogeneous structure-grounded reasoning. Code is available at the project repository. 

---
# Dynamic Mix Precision Routing for Efficient Multi-step LLM Interaction 

**Authors**: Yuanzhe Li, Jianing Deng, Jingtong Hu, Tianlong Chen, Song Wang, Huanrui Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.02711)  

**Abstract**: Large language models (LLM) achieve strong performance in long-horizon decision-making tasks through multi-step interaction and reasoning at test time. While practitioners commonly believe a higher task success rate necessitates the use of a larger and stronger LLM model, multi-step interaction with a large LLM incurs prohibitive inference cost. To address this problem, we explore the use of low-precision quantized LLM in the long-horizon decision-making process. Based on the observation of diverse sensitivities among interaction steps, we propose a dynamic mix-precision routing framework that adaptively selects between high-precision and low-precision LLMs at each decision step. The router is trained via a two-stage pipeline, consisting of KL-divergence-based supervised learning that identifies precision-sensitive steps, followed by Group-Relative Policy Optimization (GRPO) to further improve task success rates. Experiments on ALFWorld demonstrate that our approach achieves a great improvement on accuracy-cost trade-off over single-precision baselines and heuristic routing methods. 

---
# Accordion-Thinking: Self-Regulated Step Summaries for Efficient and Readable LLM Reasoning 

**Authors**: Zhicheng Yang, Zhijiang Guo, Yinya Huang, Yongxin Wang, Wenlei Shi, Yiwei Wang, Xiaodan Liang, Jing Tang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03249)  

**Abstract**: Scaling test-time compute via long Chain-ofThought unlocks remarkable gains in reasoning capabilities, yet it faces practical limits due to the linear growth of KV cache and quadratic attention complexity. In this paper, we introduce Accordion-Thinking, an end-to-end framework where LLMs learn to self-regulate the granularity of the reasoning steps through dynamic summarization. This mechanism enables a Fold inference mode, where the model periodically summarizes its thought process and discards former thoughts to reduce dependency on historical tokens. We apply reinforcement learning to incentivize this capability further, uncovering a critical insight: the accuracy gap between the highly efficient Fold mode and the exhaustive Unfold mode progressively narrows and eventually vanishes over the course of training. This phenomenon demonstrates that the model learns to encode essential reasoning information into compact summaries, achieving effective compression of the reasoning context. Our Accordion-Thinker demonstrates that with learned self-compression, LLMs can tackle complex reasoning tasks with minimal dependency token overhead without compromising solution quality, and it achieves a 3x throughput while maintaining accuracy on a 48GB GPU memory configuration, while the structured step summaries provide a human-readable account of the reasoning process. 

---
# De-conflating Preference and Qualification: Constrained Dual-Perspective Reasoning for Job Recommendation with Large Language Models 

**Authors**: Bryce Kan, Wei Yang, Emily Nguyen, Ganghui Yi, Bowen Yi, Chenxiao Yu, Yan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.03097)  

**Abstract**: Professional job recommendation involves a complex bipartite matching process that must reconcile a candidate's subjective preference with an employer's objective qualification. While Large Language Models (LLMs) are well-suited for modeling the rich semantics of resumes and job descriptions, existing paradigms often collapse these two decision dimensions into a single interaction signal, yielding confounded supervision under recruitment-funnel censoring and limiting policy controllability. To address these challenges, We propose JobRec, a generative job recommendation framework for de-conflating preference and qualification via constrained dual-perspective reasoning. JobRec introduces a Unified Semantic Alignment Schema that aligns candidate and job attributes into structured semantic layers, and a Two-Stage Cooperative Training Strategy that learns decoupled experts to separately infer preference and qualification. Building on these experts, a Lagrangian-based Policy Alignment module optimizes recommendations under explicit eligibility requirements, enabling controllable trade-offs. To mitigate data scarcity, we construct a synthetic dataset refined by experts. Experiments show that JobRec consistently outperforms strong baselines and provides improved controllability for strategy-aware professional matching. 

---
# AutoSizer: Automatic Sizing of Analog and Mixed-Signal Circuits via Large Language Model (LLM) Agents 

**Authors**: Xi Yu, Dmitrii Torbunov, Soumyajit Mandal, Yihui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2602.02849)  

**Abstract**: The design of Analog and Mixed-Signal (AMS) integrated circuits remains heavily reliant on expert knowledge, with transistor sizing a major bottleneck due to nonlinear behavior, high-dimensional design spaces, and strict performance constraints. Existing Electronic Design Automation (EDA) methods typically frame sizing as static black-box optimization, resulting in inefficient and less robust solutions. Although Large Language Models (LLMs) exhibit strong reasoning abilities, they are not suited for precise numerical optimization in AMS sizing. To address this gap, we propose AutoSizer, a reflective LLM-driven meta-optimization framework that unifies circuit understanding, adaptive search-space construction, and optimization orchestration in a closed loop. It employs a two-loop optimization framework, with an inner loop for circuit sizing and an outer loop that analyzes optimization dynamics and constraints to iteratively refine the search space from simulation feedback. We further introduce AMS-SizingBench, an open benchmark comprising 24 diverse AMS circuits in SKY130 CMOS technology, designed to evaluate adaptive optimization policies under realistic simulator-based constraints. AutoSizer experimentally achieves higher solution quality, faster convergence, and higher success rate across varying circuit difficulties, outperforming both traditional optimization methods and existing LLM-based agents. 

---
# A Positive Case for Faithfulness: LLM Self-Explanations Help Predict Model Behavior 

**Authors**: Harry Mayne, Justin Singh Kang, Dewi Gould, Kannan Ramchandran, Adam Mahdi, Noah Y. Siegel  

**Link**: [PDF](https://arxiv.org/pdf/2602.02639)  

**Abstract**: LLM self-explanations are often presented as a promising tool for AI oversight, yet their faithfulness to the model's true reasoning process is poorly understood. Existing faithfulness metrics have critical limitations, typically relying on identifying unfaithfulness via adversarial prompting or detecting reasoning errors. These methods overlook the predictive value of explanations. We introduce Normalized Simulatability Gain (NSG), a general and scalable metric based on the idea that a faithful explanation should allow an observer to learn a model's decision-making criteria, and thus better predict its behavior on related inputs. We evaluate 18 frontier proprietary and open-weight models, e.g., Gemini 3, GPT-5.2, and Claude 4.5, on 7,000 counterfactuals from popular datasets covering health, business, and ethics. We find self-explanations substantially improve prediction of model behavior (11-37% NSG). Self-explanations also provide more predictive information than explanations generated by external models, even when those models are stronger. This implies an advantage from self-knowledge that external explanation methods cannot replicate. Our approach also reveals that, across models, 5-15% of self-explanations are egregiously misleading. Despite their imperfections, we show a positive case for self-explanations: they encode information that helps predict model behavior. 

---
# Uncertainty and Fairness Awareness in LLM-Based Recommendation Systems 

**Authors**: Chandan Kumar Sah, Xiaoli Lian, Li Zhang, Tony Xu, Syed Shazaib Shah  

**Link**: [PDF](https://arxiv.org/pdf/2602.02582)  

**Abstract**: Large language models (LLMs) enable powerful zero-shot recommendations by leveraging broad contextual knowledge, yet predictive uncertainty and embedded biases threaten reliability and fairness. This paper studies how uncertainty and fairness evaluations affect the accuracy, consistency, and trustworthiness of LLM-generated recommendations. We introduce a benchmark of curated metrics and a dataset annotated for eight demographic attributes (31 categorical values) across two domains: movies and music. Through in-depth case studies, we quantify predictive uncertainty (via entropy) and demonstrate that Google DeepMind's Gemini 1.5 Flash exhibits systematic unfairness for certain sensitive attributes; measured similarity-based gaps are SNSR at 0.1363 and SNSV at 0.0507. These disparities persist under prompt perturbations such as typographical errors and multilingual inputs. We further integrate personality-aware fairness into the RecLLM evaluation pipeline to reveal personality-linked bias patterns and expose trade-offs between personalization and group fairness. We propose a novel uncertainty-aware evaluation methodology for RecLLMs, present empirical insights from deep uncertainty case studies, and introduce a personality profile-informed fairness benchmark that advances explainability and equity in LLM recommendations. Together, these contributions establish a foundation for safer, more interpretable RecLLMs and motivate future work on multi-model benchmarks and adaptive calibration for trustworthy deployment. 

---
# RC-GRPO: Reward-Conditioned Group Relative Policy Optimization for Multi-Turn Tool Calling Agents 

**Authors**: Haitian Zhong, Jixiu Zhai, Lei Song, Jiang Bian, Qiang Liu, Tieniu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2602.03025)  

**Abstract**: Multi-turn tool calling is challenging for Large Language Models (LLMs) because rewards are sparse and exploration is expensive. A common recipe, SFT followed by GRPO, can stall when within-group reward variation is low (e.g., more rollouts in a group receive the all 0 or all 1 reward), making the group-normalized advantage uninformative and yielding vanishing updates. To address this problem, we propose RC-GRPO (Reward-Conditioned Group Relative Policy Optimization), which treats exploration as a controllable steering problem via discrete reward tokens. We first fine-tune a Reward-Conditioned Trajectory Policy (RCTP) on mixed-quality trajectories with reward goal special tokens (e.g., <|high_reward|>, <|low_reward|>) injected into the prompts, enabling the model to learn how to generate distinct quality trajectories on demand. Then during RL, we sample diverse reward tokens within each GRPO group and condition rollouts on the sampled token to improve within-group diversity, improving advantage gains. On the Berkeley Function Calling Leaderboard v4 (BFCLv4) multi-turn benchmark, our method yields consistently improved performance than baselines, and the performance on Qwen-2.5-7B-Instruct even surpasses all closed-source API models. 

---
# CreditAudit: 2D Auditing for LLM Evaluation and Selection 

**Authors**: Yiliang Song, Hongjun An, Jiangong Xiao, Haofei Zhao, Jiawei Shao, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.02515)  

**Abstract**: Leaderboard scores on public benchmarks have been steadily rising and converging, with many frontier language models now separated by only marginal differences. However, these scores often fail to match users' day to day experience, because system prompts, output protocols, and interaction modes evolve under routine iteration, and in agentic multi step pipelines small protocol shifts can trigger disproportionate failures, leaving practitioners uncertain about which model to deploy. We propose CreditAudit, a deployment oriented credit audit framework that evaluates models under a family of semantically aligned and non adversarial system prompt templates across multiple benchmarks, reporting mean ability as average performance across scenarios and scenario induced fluctuation sigma as a stability risk signal, and further mapping volatility into interpretable credit grades from AAA to BBB via cross model quantiles with diagnostics that mitigate template difficulty drift. Controlled experiments on GPQA, TruthfulQA, and MMLU Pro show that models with similar mean ability can exhibit substantially different fluctuation, and stability risk can overturn prioritization decisions in agentic or high failure cost regimes. By providing a 2D and grade based language for regime specific selection, CreditAudit supports tiered deployment and more disciplined allocation of testing and monitoring effort, enabling more objective and trustworthy model evaluation for real world use. 

---
# PeerRank: Autonomous LLM Evaluation Through Web-Grounded, Bias-Controlled Peer Review 

**Authors**: Yanki Margalit, Erni Avram, Ran Taig, Oded Margalit, Nurit Cohen-Inger  

**Link**: [PDF](https://arxiv.org/pdf/2602.02589)  

**Abstract**: Evaluating large language models typically relies on human-authored benchmarks, reference answers, and human or single-model judgments, approaches that scale poorly, become quickly outdated, and mismatch open-world deployments that depend on web retrieval and synthesis. We introduce PeerRank, a fully autonomous end-to-end evaluation framework in which models generate evaluation tasks, answer them with category-scoped live web grounding, judge peer responses and aggregate dense peer assessments into relative performance estimates, without human supervision or gold references. PeerRank treats evaluation as a multi-agent process where each model participates symmetrically as task designer, respondent, and evaluator, while removing biased judgments. In a large-scale study over 12 commercially available models and 420 autonomously generated questions, PeerRank produces stable, discriminative rankings and reveals measurable identity and presentation biases. Rankings are robust, and mean peer scores agree with Elo. We further validate PeerRank on TruthfulQA and GSM8K, where peer scores correlate with objective accuracy. Together, these results suggest that bias-aware peer evaluation with selective web-grounded answering can scale open-world LLM assessment beyond static and human curated benchmarks. 

---
# Accelerating Scientific Research with Gemini: Case Studies and Common Techniques 

**Authors**: David P. Woodruff, Vincent Cohen-Addad, Lalit Jain, Jieming Mao, Song Zuo, MohammadHossein Bateni, Simina Branzei, Michael P. Brenner, Lin Chen, Ying Feng, Lance Fortnow, Gang Fu, Ziyi Guan, Zahra Hadizadeh, Mohammad T. Hajiaghayi, Mahdi JafariRaviz, Adel Javanmard, Karthik C. S., Ken-ichi Kawarabayashi, Ravi Kumar, Silvio Lattanzi, Euiwoong Lee, Yi Li, Ioannis Panageas, Dimitris Paparas, Benjamin Przybocki, Bernardo Subercaseaux, Ola Svensson, Shayan Taherijam, Xuan Wu, Eylon Yogev, Morteza Zadimoghaddam, Samson Zhou, Vahab Mirrokni  

**Link**: [PDF](https://arxiv.org/pdf/2602.03837)  

**Abstract**: Recent advances in large language models (LLMs) have opened new avenues for accelerating scientific research. While models are increasingly capable of assisting with routine tasks, their ability to contribute to novel, expert-level mathematical discovery is less understood. We present a collection of case studies demonstrating how researchers have successfully collaborated with advanced AI models, specifically Google's Gemini-based models (in particular Gemini Deep Think and its advanced variants), to solve open problems, refute conjectures, and generate new proofs across diverse areas in theoretical computer science, as well as other areas such as economics, optimization, and physics. Based on these experiences, we extract common techniques for effective human-AI collaboration in theoretical research, such as iterative refinement, problem decomposition, and cross-disciplinary knowledge transfer. While the majority of our results stem from this interactive, conversational methodology, we also highlight specific instances that push beyond standard chat interfaces. These include deploying the model as a rigorous adversarial reviewer to detect subtle flaws in existing proofs, and embedding it within a "neuro-symbolic" loop that autonomously writes and executes code to verify complex derivations. Together, these examples highlight the potential of AI not just as a tool for automation, but as a versatile, genuine partner in the creative process of scientific discovery. 

---
# ATLAS : Adaptive Self-Evolutionary Research Agent with Task-Distributed Multi-LLM Supporters 

**Authors**: Ujin Jeon, Jiyong Kwon, Madison Ann Sullivan, Caleb Eunho Lee, Guang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2602.02709)  

**Abstract**: Recent multi-LLM agent systems perform well in prompt optimization and automated problem-solving, but many either keep the solver frozen after fine-tuning or rely on a static preference-optimization loop, which becomes intractable for long-horizon tasks. We propose ATLAS (Adaptive Task-distributed Learning for Agentic Self-evolution), a task-distributed framework that iteratively develops a lightweight research agent while delegating complementary roles to specialized supporter agents for exploration, hyperparameter tuning, and reference policy management. Our core algorithm, Evolving Direct Preference Optimization (EvoDPO), adaptively updates the phase-indexed reference policy. We provide a theoretical regret analysis for a preference-based contextual bandit under concept drift. In addition, experiments were conducted on non-stationary linear contextual bandits and scientific machine learning (SciML) loss reweighting for the 1D Burgers' equation. Both results show that ATLAS improves stability and performance over a static single-agent baseline. 

---
# Experience-Driven Multi-Agent Systems Are Training-free Context-aware Earth Observers 

**Authors**: Pengyu Dai, Weihao Xuan, Junjue Wang, Hongruixuan Chen, Jian Song, Yafei Ou, Naoto Yokoya  

**Link**: [PDF](https://arxiv.org/pdf/2602.02559)  

**Abstract**: Recent advances have enabled large language model (LLM) agents to solve complex tasks by orchestrating external tools. However, these agents often struggle in specialized, tool-intensive domains that demand long-horizon execution, tight coordination across modalities, and strict adherence to implicit tool constraints. Earth Observation (EO) tasks exemplify this challenge due to the multi-modal and multi-temporal data inputs, as well as the requirements of geo-knowledge constraints (spectrum library, spatial reasoning, etc): many high-level plans can be derailed by subtle execution errors that propagate through a pipeline and invalidate final results. A core difficulty is that existing agents lack a mechanism to learn fine-grained, tool-level expertise from interaction. Without such expertise, they cannot reliably configure tool parameters or recover from mid-execution failures, limiting their effectiveness in complex EO workflows. To address this, we introduce \textbf{GeoEvolver}, a self-evolving multi-agent system~(MAS) that enables LLM agents to acquire EO expertise through structured interaction without any parameter updates. GeoEvolver decomposes each query into independent sub-goals via a retrieval-augmented multi-agent orchestrator, then explores diverse tool-parameter configurations at the sub-goal level. Successful patterns and root-cause attribution from failures are then distilled in an evolving memory bank that provides in-context demonstrations for future queries. Experiments on three tool-integrated EO benchmarks show that GeoEvolver consistently improves end-to-end task success, with an average gain of 12\% across multiple LLM backbones, demonstrating that EO expertise can emerge progressively from efficient, fine-grained interactions with the environment. 

---
# Antidistillation Fingerprinting 

**Authors**: Yixuan Even Xu, John Kirchenbauer, Yash Savani, Asher Trockman, Alexander Robey, Tom Goldstein, Fei Fang, J. Zico Kolter  

**Link**: [PDF](https://arxiv.org/pdf/2602.03812)  

**Abstract**: Model distillation enables efficient emulation of frontier large language models (LLMs), creating a need for robust mechanisms to detect when a third-party student model has trained on a teacher model's outputs. However, existing fingerprinting techniques that could be used to detect such distillation rely on heuristic perturbations that impose a steep trade-off between generation quality and fingerprinting strength, often requiring significant degradation of utility to ensure the fingerprint is effectively internalized by the student. We introduce antidistillation fingerprinting (ADFP), a principled approach that aligns the fingerprinting objective with the student's learning dynamics. Building upon the gradient-based framework of antidistillation sampling, ADFP utilizes a proxy model to identify and sample tokens that directly maximize the expected detectability of the fingerprint in the student after fine-tuning, rather than relying on the incidental absorption of the un-targeted biases of a more naive watermark. Experiments on GSM8K and OASST1 benchmarks demonstrate that ADFP achieves a significant Pareto improvement over state-of-the-art baselines, yielding stronger detection confidence with minimal impact on utility, even when the student model's architecture is unknown. 

---
# Bridging Online and Offline RL: Contextual Bandit Learning for Multi-Turn Code Generation 

**Authors**: Ziru Chen, Dongdong Chen, Ruinan Jin, Yingbin Liang, Yujia Xie, Huan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.03806)  

**Abstract**: Recently, there have been significant research interests in training large language models (LLMs) with reinforcement learning (RL) on real-world tasks, such as multi-turn code generation. While online RL tends to perform better than offline RL, its higher training cost and instability hinders wide adoption. In this paper, we build on the observation that multi-turn code generation can be formulated as a one-step recoverable Markov decision process and propose contextual bandit learning with offline trajectories (Cobalt), a new method that combines the benefits of online and offline RL. Cobalt first collects code generation trajectories using a reference LLM and divides them into partial trajectories as contextual prompts. Then, during online bandit learning, the LLM is trained to complete each partial trajectory prompt through single-step code generation. Cobalt outperforms two multi-turn online RL baselines based on GRPO and VeRPO, and substantially improves R1-Distill 8B and Qwen3 8B by up to 9.0 and 6.2 absolute Pass@1 scores on LiveCodeBench. Also, we analyze LLMs' in-context reward hacking behaviors and augment Cobalt training with perturbed trajectories to mitigate this issue. Overall, our results demonstrate Cobalt as a promising solution for iterative decision-making tasks like multi-turn code generation. Our code and data are available at this https URL. 

---
# An Empirical Study of Collective Behaviors and Social Dynamics in Large Language Model Agents 

**Authors**: Farnoosh Hashemi, Michael W. Macy  

**Link**: [PDF](https://arxiv.org/pdf/2602.03775)  

**Abstract**: Large Language Models (LLMs) increasingly mediate our social, cultural, and political interactions. While they can simulate some aspects of human behavior and decision-making, it is still underexplored whether repeated interactions with other agents amplify their biases or lead to exclusionary behaviors. To this end, we study this http URL-an LLM-driven social media platform-analyzing 7M posts and interactions among 32K LLM agents over a year. We start with homophily and social influence among LLMs, learning that similar to humans', their social networks exhibit these fundamental phenomena. Next, we study the toxic language of LLMs, its linguistic features, and their interaction patterns, finding that LLMs show different structural patterns in toxic posting than humans. After studying the ideological leaning in LLMs posts, and the polarization in their community, we focus on how to prevent their potential harmful activities. We present a simple yet effective method, called Chain of Social Thought (CoST), that reminds LLM agents to avoid harmful posting. 

---
# Cognitively Diverse Multiple-Choice Question Generation: A Hybrid Multi-Agent Framework with Large Language Models 

**Authors**: Yu Tian, Linh Huynh, Katerina Christhilf, Shubham Chakraborty, Micah Watanabe, Tracy Arner, Danielle McNamara  

**Link**: [PDF](https://arxiv.org/pdf/2602.03704)  

**Abstract**: Recent advances in large language models (LLMs) have made automated multiple-choice question (MCQ) generation increasingly feasible; however, reliably producing items that satisfy controlled cognitive demands remains a challenge. To address this gap, we introduce ReQUESTA, a hybrid, multi-agent framework for generating cognitively diverse MCQs that systematically target text-based, inferential, and main idea comprehension. ReQUESTA decomposes MCQ authoring into specialized subtasks and coordinates LLM-powered agents with rule-based components to support planning, controlled generation, iterative evaluation, and post-processing. We evaluated the framework in a large-scale reading comprehension study using academic expository texts, comparing ReQUESTA-generated MCQs with those produced by a single-pass GPT-5 zero-shot baseline. Psychometric analyses of learner responses assessed item difficulty and discrimination, while expert raters evaluated question quality across multiple dimensions, including topic relevance and distractor quality. Results showed that ReQUESTA-generated items were consistently more challenging, more discriminative, and more strongly aligned with overall reading comprehension performance. Expert evaluations further indicated stronger alignment with central concepts and superior distractor linguistic consistency and semantic plausibility, particularly for inferential questions. These findings demonstrate that hybrid, agentic orchestration can systematically improve the reliability and controllability of LLM-based generation, highlighting workflow design as a key lever for structured artifact generation beyond single-pass prompting. 

---
# Agent Primitives: Reusable Latent Building Blocks for Multi-Agent Systems 

**Authors**: Haibo Jin, Kuang Peng, Ye Yu, Xiaopeng Yuan, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03695)  

**Abstract**: While existing multi-agent systems (MAS) can handle complex problems by enabling collaboration among multiple agents, they are often highly task-specific, relying on manually crafted agent roles and interaction prompts, which leads to increased architectural complexity and limited reusability across tasks. Moreover, most MAS communicate primarily through natural language, making them vulnerable to error accumulation and instability in long-context, multi-stage interactions within internal agent histories.
In this work, we propose \textbf{Agent Primitives}, a set of reusable latent building blocks for LLM-based MAS. Inspired by neural network design, where complex models are built from reusable components, we observe that many existing MAS architectures can be decomposed into a small number of recurring internal computation patterns. Based on this observation, we instantiate three primitives: Review, Voting and Selection, and Planning and Execution. All primitives communicate internally via key-value (KV) cache, which improves both robustness and efficiency by mitigating information degradation across multi-stage interactions. To enable automatic system construction, an Organizer agent selects and composes primitives for each query, guided by a lightweight knowledge pool of previously successful configurations, forming a primitive-based MAS.
Experiments show that primitives-based MAS improve average accuracy by 12.0-16.5\% over single-agent baselines, reduce token usage and inference latency by approximately 3$\times$-4$\times$ compared to text-based MAS, while incurring only 1.3$\times$-1.6$\times$ overhead relative to single-agent inference and providing more stable performance across model backbones. 

---
# LLM-Inspired Pretrain-Then-Finetune for Small-Data, Large-Scale Optimization 

**Authors**: Zishi Zhang, Jinhui Han, Ming Hu, Yijie Peng  

**Link**: [PDF](https://arxiv.org/pdf/2602.03690)  

**Abstract**: We consider small-data, large-scale decision problems in which a firm must make many operational decisions simultaneously (e.g., across a large product portfolio) while observing only a few, potentially noisy, data points per instance. Inspired by the success of large language models (LLMs), we propose a pretrain-then-finetune approach built on a designed Transformer model to address this challenge. The model is first pretrained on large-scale, domain-informed synthetic data that encode managerial knowledge and structural features of the decision environment, and is then fine-tuned on real observations. This new pipeline offers two complementary advantages: pretraining injects domain knowledge into the learning process and enables the training of high-capacity models using abundant synthetic data, while finetuning adapts the pretrained model to the operational environment and improves alignment with the true data-generating regime. While we have leveraged the Transformer's state-of-the-art representational capacity, particularly its attention mechanism, to efficiently extract cross-task structure, our approach is not an off-the-shelf application. Instead, it relies on problem-specific architectural design and a tailored training procedure to match the decision setting. Theoretically, we develop the first comprehensive error analysis regarding Transformer learning in relevant contexts, establishing nonasymptotic guarantees that validate the method's effectiveness. Critically, our analysis reveals how pretraining and fine-tuning jointly determine performance, with the dominant contribution governed by whichever is more favorable. In particular, finetuning exhibits an economies-of-scale effect, whereby transfer learning becomes increasingly effective as the number of instances grows. 

---
# Zero-shot large vision-language model prompting for automated bone identification in paleoradiology x-ray archives 

**Authors**: Owen Dong, Lily Gao, Manish Kota, Bennett A. Landmana, Jelena Bekvalac, Gaynor Western, Katherine D. Van Schaik  

**Link**: [PDF](https://arxiv.org/pdf/2602.03750)  

**Abstract**: Paleoradiology, the use of modern imaging technologies to study archaeological and anthropological remains, offers new windows on millennial scale patterns of human health. Unfortunately, the radiographs collected during field campaigns are heterogeneous: bones are disarticulated, positioning is ad hoc, and laterality markers are often absent. Additionally, factors such as age at death, age of bone, sex, and imaging equipment introduce high variability. Thus, content navigation, such as identifying a subset of images with a specific projection view, can be time consuming and difficult, making efficient triaging a bottleneck for expert analysis. We report a zero shot prompting strategy that leverages a state of the art Large Vision Language Model (LVLM) to automatically identify the main bone, projection view, and laterality in such images. Our pipeline converts raw DICOM files to bone windowed PNGs, submits them to the LVLM with a carefully engineered prompt, and receives structured JSON outputs, which are extracted and formatted onto a spreadsheet in preparation for validation. On a random sample of 100 images reviewed by an expert board certified paleoradiologist, the system achieved 92% main bone accuracy, 80% projection view accuracy, and 100% laterality accuracy, with low or medium confidence flags for ambiguous cases. These results suggest that LVLMs can substantially accelerate code word development for large paleoradiology datasets, allowing for efficient content navigation in future anthropology workflows. 

---
# Universal One-third Time Scaling in Learning Peaked Distributions 

**Authors**: Yizhou Liu, Ziming Liu, Cengiz Pehlevan, Jeff Gore  

**Link**: [PDF](https://arxiv.org/pdf/2602.03685)  

**Abstract**: Training large language models (LLMs) is computationally expensive, partly because the loss exhibits slow power-law convergence whose origin remains debatable. Through systematic analysis of toy models and empirical evaluation of LLMs, we show that this behavior can arise intrinsically from the use of softmax and cross-entropy. When learning peaked probability distributions, e.g., next-token distributions, these components yield power-law vanishing losses and gradients, creating a fundamental optimization bottleneck. This ultimately leads to power-law time scaling of the loss with a universal exponent of $1/3$. Our results provide a mechanistic explanation for observed neural scaling and suggest new directions for improving LLM training efficiency. 

---
# Controlling Output Rankings in Generative Engines for LLM-based Search 

**Authors**: Haibo Jin, Ruoxi Chen, Peiyan Zhang, Yifeng Luo, Huimin Zeng, Man Luo, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03608)  

**Abstract**: The way customers search for and choose products is changing with the rise of large language models (LLMs). LLM-based search, or generative engines, provides direct product recommendations to users, rather than traditional online search results that require users to explore options themselves. However, these recommendations are strongly influenced by the initial retrieval order of LLMs, which disadvantages small businesses and independent creators by limiting their visibility.
In this work, we propose CORE, an optimization method that \textbf{C}ontrols \textbf{O}utput \textbf{R}ankings in g\textbf{E}nerative Engines for LLM-based search. Since the LLM's interactions with the search engine are black-box, CORE targets the content returned by search engines as the primary means of influencing output rankings. Specifically, CORE optimizes retrieved content by appending strategically designed optimization content to steer the ranking of outputs. We introduce three types of optimization content: string-based, reasoning-based, and review-based, demonstrating their effectiveness in shaping output rankings. To evaluate CORE in realistic settings, we introduce ProductBench, a large-scale benchmark with 15 product categories and 200 products per category, where each product is associated with its top-10 recommendations collected from Amazon's search interface.
Extensive experiments on four LLMs with search capabilities (GPT-4o, Gemini-2.5, Claude-4, and Grok-3) demonstrate that CORE achieves an average Promotion Success Rate of \textbf{91.4\% @Top-5}, \textbf{86.6\% @Top-3}, and \textbf{80.3\% @Top-1}, across 15 product categories, outperforming existing ranking manipulation methods while preserving the fluency of optimized content. 

---
# Use Graph When It Needs: Efficiently and Adaptively Integrating Retrieval-Augmented Generation with Graphs 

**Authors**: Su Dong, Qinggang Zhang, Yilin Xiao, Shengyuan Chen, Chuang Zhou, Xiao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03578)  

**Abstract**: Large language models (LLMs) often struggle with knowledge-intensive tasks due to hallucinations and outdated parametric knowledge. While Retrieval-Augmented Generation (RAG) addresses this by integrating external corpora, its effectiveness is limited by fragmented information in unstructured domain documents. Graph-augmented RAG (GraphRAG) emerged to enhance contextual reasoning through structured knowledge graphs, yet paradoxically underperforms vanilla RAG in real-world scenarios, exhibiting significant accuracy drops and prohibitive latency despite gains on complex queries. We identify the rigid application of GraphRAG to all queries, regardless of complexity, as the root cause. To resolve this, we propose an efficient and adaptive GraphRAG framework called EA-GraphRAG that dynamically integrates RAG and GraphRAG paradigms through syntax-aware complexity analysis. Our approach introduces: (i) a syntactic feature constructor that parses each query and extracts a set of structural features; (ii) a lightweight complexity scorer that maps these features to a continuous complexity score; and (iii) a score-driven routing policy that selects dense RAG for low-score queries, invokes graph-based retrieval for high-score queries, and applies complexity-aware reciprocal rank fusion to handle borderline cases. Extensive experiments on a comprehensive benchmark, consisting of two single-hop and two multi-hop QA benchmarks, demonstrate that our EA-GraphRAG significantly improves accuracy, reduces latency, and achieves state-of-the-art performance in handling mixed scenarios involving both simple and complex queries. 

---
# When Single Answer Is Not Enough: Rethinking Single-Step Retrosynthesis Benchmarks for LLMs 

**Authors**: Bogdan Zagribelnyy, Ivan Ilin, Maksim Kuznetsov, Nikita Bondarev, Roman Schutski, Thomas MacDougall, Rim Shayakhmetov, Zulfat Miftakhutdinov, Mikolaj Mizera, Vladimir Aladinskiy, Alex Aliper, Alex Zhavoronkov  

**Link**: [PDF](https://arxiv.org/pdf/2602.03554)  

**Abstract**: Recent progress has expanded the use of large language models (LLMs) in drug discovery, including synthesis planning. However, objective evaluation of retrosynthesis performance remains limited. Existing benchmarks and metrics typically rely on published synthetic procedures and Top-K accuracy based on single ground-truth, which does not capture the open-ended nature of real-world synthesis planning. We propose a new benchmarking framework for single-step retrosynthesis that evaluates both general-purpose and chemistry-specialized LLMs using ChemCensor, a novel metric for chemical plausibility. By emphasizing plausibility over exact match, this approach better aligns with human synthesis planning practices. We also introduce CREED, a novel dataset comprising millions of ChemCensor-validated reaction records for LLM training, and use it to train a model that improves over the LLM baselines under this benchmark. 

---
# UniGeM: Unifying Data Mixing and Selection via Geometric Exploration and Mining 

**Authors**: Changhao Wang, Yunfei Yu, Xinhao Yao, Jiaolong Yang, Riccardo Cantoro, Chaobo Li, Qing Cui, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2602.03772)  

**Abstract**: The scaling of Large Language Models (LLMs) is increasingly limited by data quality. Most methods handle data mixing and sample selection separately, which can break the structure in code corpora. We introduce \textbf{UniGeM}, a framework that unifies mixing and selection by treating data curation as a \textit{manifold approximation} problem without training proxy models or relying on external reference datasets. UniGeM operates hierarchically: \textbf{Macro-Exploration} learns mixing weights with stability-based clustering; \textbf{Micro-Mining} filters high-quality instances by their geometric distribution to ensure logical consistency. Validated by training 8B and 16B MoE models on 100B tokens, UniGeM achieves \textbf{2.0$\times$ data efficiency} over a random baseline and further improves overall performance compared to SOTA methods in reasoning-heavy evaluations and multilingual generalization. 

---
# Not All Negative Samples Are Equal: LLMs Learn Better from Plausible Reasoning 

**Authors**: Zixiang Di, Jinyi Han, Shuo Zhang, Ying Liao, Zhi Li, Xiaofeng Ji, Yongqi Wang, Zheming Yang, Ming Gao, Bingdong Li, Jie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03516)  

**Abstract**: Learning from negative samples holds great promise for improving Large Language Model (LLM) reasoning capability, yet existing methods treat all incorrect responses as equally informative, overlooking the crucial role of sample quality. To address this, we propose Plausible Negative Samples (PNS), a method that synthesizes high-quality negative samples exhibiting expected format and structural coherence while ultimately yielding incorrect answers. PNS trains a dedicated model via reverse reinforcement learning (RL) guided by a composite reward combining format compliance, accuracy inversion, reward model assessment, and chain-of-thought evaluation, generating responses nearly indistinguishable from correct solutions. We further validate PNS as a plug-and-play data source for preference optimization across three backbone models on seven mathematical reasoning benchmarks. Results demonstrate that PNS consistently outperforms other negative sample synthesis methods, achieving an average improvement of 2.03% over RL-trained models. 

---
# Self-Verification Dilemma: Experience-Driven Suppression of Overused Checking in LLM Reasoning 

**Authors**: Quanyu Long, Kai Jie Jiang, Jianda Chen, Xu Guo, Leilei Gan, Wenya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03485)  

**Abstract**: Large Reasoning Models (LRMs) achieve strong performance by generating long reasoning traces with reflection. Through a large-scale empirical analysis, we find that a substantial fraction of reflective steps consist of self-verification (recheck) that repeatedly confirm intermediate results. These rechecks occur frequently across models and benchmarks, yet the vast majority are confirmatory rather than corrective, rarely identifying errors and altering reasoning outcomes. This reveals a mismatch between how often self-verification is activated and how often it is actually useful. Motivated by this, we propose a novel, experience-driven test-time framework that reduces the overused verification. Our method detects the activation of recheck behavior, consults an offline experience pool of past verification outcomes, and estimates whether a recheck is likely unnecessary via efficient retrieval. When historical experience suggests unnecessary, a suppression signal redirects the model to proceed. Across multiple model and benchmarks, our approach reduces token usage up to 20.3% while maintaining the accuracy, and in some datasets even yields accuracy improvements. 

---
# Precision in Practice: Knowledge Guided Code Summarizing Grounded in Industrial Expectations 

**Authors**: Jintai Li, Songqiang Chen, Shuo Jin, Xiaoyuan Xie  

**Link**: [PDF](https://arxiv.org/pdf/2602.03400)  

**Abstract**: Code summaries are essential for helping developers understand code functionality and reducing maintenance and collaboration costs. Although recent advances in large language models (LLMs) have significantly improved automatic code summarization, the practical usefulness of generated summaries in industrial settings remains insufficiently explored. In collaboration with documentation experts from the industrial HarmonyOS project, we conducted a questionnaire study showing that over 57.4% of code summaries produced by state-of-the-art approaches were rejected due to violations of developers' expectations for industrial documentation. Beyond semantic similarity to reference summaries, developers emphasize additional requirements, including the use of appropriate domain terminology, explicit function categorization, and the avoidance of redundant implementation details.
To address these expectations, we propose ExpSum, an expectation-aware code summarization approach that integrates function metadata abstraction, informative metadata filtering, context-aware domain knowledge retrieval, and constraint-driven prompting to guide LLMs in generating structured, expectation-aligned summaries. We evaluate ExpSum on the HarmonyOS project and widely used code summarization benchmarks. Experimental results show that ExpSum consistently outperforms all baselines, achieving improvements of up to 26.71% in BLEU-4 and 20.10% in ROUGE-L on HarmonyOS. Furthermore, LLM-based evaluations indicate that ExpSum-generated summaries better align with developer expectations across other projects, demonstrating its effectiveness for industrial code documentation. 

---
# MeKi: Memory-based Expert Knowledge Injection for Efficient LLM Scaling 

**Authors**: Ning Ding, Fangcheng Liu, Kyungrae Kim, Linji Hao, Kyeng-Hun Lee, Hyeonmok Ko, Yehui Tang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03359)  

**Abstract**: Scaling Large Language Models (LLMs) typically relies on increasing the number of parameters or test-time computations to boost performance. However, these strategies are impractical for edge device deployment due to limited RAM and NPU resources. Despite hardware constraints, deploying performant LLM on edge devices such as smartphone remains crucial for user experience. To address this, we propose MeKi (Memory-based Expert Knowledge Injection), a novel system that scales LLM capacity via storage space rather than FLOPs. MeKi equips each Transformer layer with token-level memory experts that injects pre-stored semantic knowledge into the generation process. To bridge the gap between training capacity and inference efficiency, we employ a re-parameterization strategy to fold parameter matrices used during training into a compact static lookup table. By offloading the knowledge to ROM, MeKi decouples model capacity from computational cost, introducing zero inference latency overhead. Extensive experiments demonstrate that MeKi significantly outperforms dense LLM baselines with identical inference speed, validating the effectiveness of memory-based scaling paradigm for on-device LLMs. Project homepage is at this https URL. 

---
# R1-SyntheticVL: Is Synthetic Data from Generative Models Ready for Multimodal Large Language Model? 

**Authors**: Jingyi Zhang, Tianyi Lin, Huanjin Yao, Xiang Lan, Shunyu Liu, Jiaxing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03300)  

**Abstract**: In this work, we aim to develop effective data synthesis techniques that autonomously synthesize multimodal training data for enhancing MLLMs in solving complex real-world tasks. To this end, we propose Collective Adversarial Data Synthesis (CADS), a novel and general approach to synthesize high-quality, diverse and challenging multimodal data for MLLMs. The core idea of CADS is to leverage collective intelligence to ensure high-quality and diverse generation, while exploring adversarial learning to synthesize challenging samples for effectively driving model improvement. Specifically, CADS operates with two cyclic phases, i.e., Collective Adversarial Data Generation (CAD-Generate) and Collective Adversarial Data Judgment (CAD-Judge). CAD-Generate leverages collective knowledge to jointly generate new and diverse multimodal data, while CAD-Judge collaboratively assesses the quality of synthesized data. In addition, CADS introduces an Adversarial Context Optimization mechanism to optimize the generation context to encourage challenging and high-value data generation. With CADS, we construct MMSynthetic-20K and train our model R1-SyntheticVL, which demonstrates superior performance on various benchmarks. 

---
# Reinforcement Learning with Promising Tokens for Large Language Models 

**Authors**: Jing-Cheng Pang, Liang Lu, Xian Tang, Kun Jiang, Sijie Wu, Kai Zhang, Xubin Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.03195)  

**Abstract**: Reinforcement learning (RL) has emerged as a key paradigm for aligning and optimizing large language models (LLMs). Standard approaches treat the LLM as the policy and apply RL directly over the full vocabulary space. However, this formulation includes the massive tail of contextually irrelevant tokens in the action space, which could distract the policy from focusing on decision-making among the truly reasonable tokens. In this work, we verify that valid reasoning paths could inherently concentrate within a low-rank subspace. Based on this insight, we introduce Reinforcement Learning with Promising Tokens (RLPT), a framework that mitigates the action space issue by decoupling strategic decision-making from token generation. Specifically, RLPT leverages the semantic priors of the base model to identify a dynamic set of \emph{promising tokens} and constrains policy optimization exclusively to this refined subset via masking. Theoretical analysis and empirical results demonstrate that RLPT effectively reduces gradient variance, stabilizes the training process, and improves sample efficiency. Experiment results on math, coding, and telecom reasoning show that RLPT outperforms standard RL baselines and integrates effectively across various model sizes (4B and 8B) and RL algorithms (GRPO and DAPO). 

---
# Prompt Augmentation Scales up GRPO Training on Mathematical Reasoning 

**Authors**: Wenquan Lu, Hai Huang, Randall Balestriero  

**Link**: [PDF](https://arxiv.org/pdf/2602.03190)  

**Abstract**: Reinforcement learning algorithms such as group-relative policy optimization (GRPO) have demonstrated strong potential for improving the mathematical reasoning capabilities of large language models. However, prior work has consistently observed an entropy collapse phenomenon during reinforcement post-training, characterized by a monotonic decrease in policy entropy that ultimately leads to training instability and collapse. As a result, most existing approaches restrict training to short horizons (typically 5-20 epochs), limiting sustained exploration and hindering further policy improvement. In addition, nearly all prior work relies on a single, fixed reasoning prompt or template during training. In this work, we introduce prompt augmentation, a training strategy that instructs the model to generate reasoning traces under diverse templates and formats, thereby increasing rollout diversity. We show that, without a KL regularization term, prompt augmentation enables stable scaling of training duration under a fixed dataset and allows the model to tolerate low-entropy regimes without premature collapse. Empirically, a Qwen2.5-Math-1.5B model trained with prompt augmentation on the MATH Level 3-5 dataset achieves state-of-the-art performance, reaching 44.5 per-benchmark accuracy and 51.3 per-question accuracy on standard mathematical reasoning benchmarks, including AIME24, AMC, MATH500, Minerva, and OlympiadBench. The code and model checkpoints are available at this https URL. 

---
# Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost 

**Authors**: Yinggan Xu, Risto Miikkulainen, Xin Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2602.03120)  

**Abstract**: Post-Training Quantization (PTQ) is essential for deploying Large Language Models (LLMs) on memory-constrained devices, yet it renders models static and difficult to fine-tune. Standard fine-tuning paradigms, including Reinforcement Learning (RL), fundamentally rely on backpropagation and high-precision weights to compute gradients. Thus they cannot be used on quantized models, where the parameter space is discrete and non-differentiable. While Evolution Strategies (ES) offer a backpropagation-free alternative, optimization of the quantized parameters can still fail due to vanishing or inaccurate gradient. This paper introduces Quantized Evolution Strategies (QES), an optimization paradigm that performs full-parameter fine-tuning directly in the quantized space. QES is based on two innovations: (1) it integrates accumulated error feedback to preserve high-precision gradient signals, and (2) it utilizes a stateless seed replay to reduce memory usage to low-precision inference levels. QES significantly outperforms the state-of-the-art zeroth-order fine-tuning method on arithmetic reasoning tasks, making direct fine-tuning for quantized models possible. It therefore opens up the possibility for scaling up LLMs entirely in the quantized space. The source code is available at this https URL . 

---
# Contrastive Concept-Tree Search for LLM-Assisted Algorithm Discovery 

**Authors**: Timothee Leleu, Sudeera Gunathilaka, Federico Ghimenti, Surya Ganguli  

**Link**: [PDF](https://arxiv.org/pdf/2602.03132)  

**Abstract**: Large language Model (LLM)-assisted algorithm discovery is an iterative, black-box optimization process over programs to approximatively solve a target task, where an LLM proposes candidate programs and an external evaluator provides task feedback. Despite intense recent research on the topic and promising results, how can the LLM internal representation of the space of possible programs be maximally exploited to improve performance is an open question. Here, we introduce Contrastive Concept-Tree Search (CCTS), which extracts a hierarchical concept representation from the generated programs and learns a contrastive concept model that guides parent selection. By reweighting parents using a likelihood-ratio score between high- and low-performing solutions, CCTS biases search toward useful concept combinations and away from misleading ones, providing guidance through an explicit concept hierarchy rather than the algorithm lineage constructed by the LLM. We show that CCTS improves search efficiency over fitness-based baselines and produces interpretable, task-specific concept trees across a benchmark of open Erdős-type combinatorics problems. Our analysis indicates that the gains are driven largely by learning which concepts to avoid. We further validate these findings in a controlled synthetic algorithm-discovery environment, which reproduces qualitatively the search dynamics observed with the LLMs. 

---
# Evaluating LLMs When They Do Not Know the Answer: Statistical Evaluation of Mathematical Reasoning via Comparative Signals 

**Authors**: Zihan Dong, Zhixian Zhang, Yang Zhou, Can Jin, Ruijia Wu, Linjun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03061)  

**Abstract**: Evaluating mathematical reasoning in LLMs is constrained by limited benchmark sizes and inherent model stochasticity, yielding high-variance accuracy estimates and unstable rankings across platforms. On difficult problems, an LLM may fail to produce a correct final answer, yet still provide reliable pairwise comparison signals indicating which of two candidate solutions is better. We leverage this observation to design a statistically efficient evaluation framework that combines standard labeled outcomes with pairwise comparison signals obtained by having models judge auxiliary reasoning chains. Treating these comparison signals as control variates, we develop a semiparametric estimator based on the efficient influence function (EIF) for the setting where auxiliary reasoning chains are observed. This yields a one-step estimator that achieves the semiparametric efficiency bound, guarantees strict variance reduction over naive sample averaging, and admits asymptotic normality for principled uncertainty quantification. Across simulations, our one-step estimator substantially improves ranking accuracy, with gains increasing as model output noise grows. Experiments on GPQA Diamond, AIME 2025, and GSM8K further demonstrate more precise performance estimation and more reliable model rankings, especially in small-sample regimes where conventional evaluation is pretty unstable. 

---
# The Trigger in the Haystack: Extracting and Reconstructing LLM Backdoor Triggers 

**Authors**: Blake Bullwinkel, Giorgio Severi, Keegan Hines, Amanda Minnich, Ram Shankar Siva Kumar, Yonatan Zunger  

**Link**: [PDF](https://arxiv.org/pdf/2602.03085)  

**Abstract**: Detecting whether a model has been poisoned is a longstanding problem in AI security. In this work, we present a practical scanner for identifying sleeper agent-style backdoors in causal language models. Our approach relies on two key findings: first, sleeper agents tend to memorize poisoning data, making it possible to leak backdoor examples using memory extraction techniques. Second, poisoned LLMs exhibit distinctive patterns in their output distributions and attention heads when backdoor triggers are present in the input. Guided by these observations, we develop a scalable backdoor scanning methodology that assumes no prior knowledge of the trigger or target behavior and requires only inference operations. Our scanner integrates naturally into broader defensive strategies and does not alter model performance. We show that our method recovers working triggers across multiple backdoor scenarios and a broad range of models and fine-tuning methods. 

---
# FedKRSO: Communication and Memory Efficient Federated Fine-Tuning of Large Language Models 

**Authors**: Guohao Yang, Tongle Wu, Yuanxiong Guo, Ying Sun, Yanmin Gong  

**Link**: [PDF](https://arxiv.org/pdf/2602.03019)  

**Abstract**: Fine-tuning is essential to adapt general-purpose large language models (LLMs) to domain-specific tasks. As a privacy-preserving framework to leverage decentralized data for collaborative model training, Federated Learning (FL) is gaining popularity in LLM fine-tuning, but remains challenging due to the high cost of transmitting full model parameters and computing full gradients on resource-constrained clients. While Parameter-Efficient Fine-Tuning (PEFT) methods are widely used in FL to reduce communication and memory costs, they often sacrifice model performance compared to FFT. This paper proposes FedKRSO (Federated $K$-Seed Random Subspace Optimization), a novel method that enables communication and memory efficient FFT of LLMs in federated settings. In FedKRSO, clients update the model within a shared set of random low-dimension subspaces generated by the server to save memory usage. Furthermore, instead of transmitting full model parameters in each FL round, clients send only the model update accumulators along the subspaces to the server, enabling efficient global model aggregation and dissemination. By using these strategies, FedKRSO can substantially reduce communication and memory overhead while overcoming the performance limitations of PEFT, closely approximating the performance of federated FFT. The convergence properties of FedKRSO are analyzed rigorously under general FL settings. Extensive experiments on the GLUE benchmark across diverse FL scenarios demonstrate that FedKRSO achieves both superior performance and low communication and memory overhead, paving the way towards on federated LLM fine-tuning at the resource-constrained edge. 

---
# Bongards at the Boundary of Perception and Reasoning: Programs or Language? 

**Authors**: Cassidy Langenfeld, Claas Beger, Gloria Geng, Wasu Top Piriyakulkij, Keya Hu, Yewen Pu, Kevin Ellis  

**Link**: [PDF](https://arxiv.org/pdf/2602.03038)  

**Abstract**: Vision-Language Models (VLMs) have made great strides in everyday visual tasks, such as captioning a natural image, or answering commonsense questions about such images. But humans possess the puzzling ability to deploy their visual reasoning abilities in radically new situations, a skill rigorously tested by the classic set of visual reasoning challenges known as the Bongard problems. We present a neurosymbolic approach to solving these problems: given a hypothesized solution rule for a Bongard problem, we leverage LLMs to generate parameterized programmatic representations for the rule and perform parameter fitting using Bayesian optimization. We evaluate our method on classifying Bongard problem images given the ground truth rule, as well as on solving the problems from scratch. 

---
# Where Norms and References Collide: Evaluating LLMs on Normative Reasoning 

**Authors**: Mitchell Abrams, Kaveh Eskandari Miandoab, Felix Gervits, Vasanth Sarathy, Matthias Scheutz  

**Link**: [PDF](https://arxiv.org/pdf/2602.02975)  

**Abstract**: Embodied agents, such as robots, will need to interact in situated environments where successful communication often depends on reasoning over social norms: shared expectations that constrain what actions are appropriate in context. A key capability in such settings is norm-based reference resolution (NBRR), where interpreting referential expressions requires inferring implicit normative expectations grounded in physical and social context. Yet it remains unclear whether Large Language Models (LLMs) can support this kind of reasoning. In this work, we introduce SNIC (Situated Norms in Context), a human-validated diagnostic testbed designed to probe how well state-of-the-art LLMs can extract and utilize normative principles relevant to NBRR. SNIC emphasizes physically grounded norms that arise in everyday tasks such as cleaning, tidying, and serving. Across a range of controlled evaluations, we find that even the strongest LLMs struggle to consistently identify and apply social norms, particularly when norms are implicit, underspecified, or in conflict. These findings reveal a blind spot in current LLMs and highlight a key challenge for deploying language-based systems in socially situated, embodied settings. 

---
# Learning-Infused Formal Reasoning: From Contract Synthesis to Artifact Reuse and Formal Semantics 

**Authors**: Arshad Beg, Diarmuid O'Donoghue, Rosemary Monahan  

**Link**: [PDF](https://arxiv.org/pdf/2602.02881)  

**Abstract**: This vision paper articulates a long-term research agenda for formal methods at the intersection with artificial intelligence, outlining multiple conceptual and technical dimensions and reporting on our ongoing work toward realising this agenda. It advances a forward-looking perspective on the next generation of formal methods based on the integration of automated contract synthesis, semantic artifact reuse, and refinement-based theory. We argue that future verification systems must move beyond isolated correctness proofs toward a cumulative, knowledge-driven paradigm in which specifications, contracts, and proofs are continuously synthesised and transferred across systems. To support this shift, we outline a hybrid framework combining large language models with graph-based representations to enable scalable semantic matching and principled reuse of verification artifacts. Learning-based components provide semantic guidance across heterogeneous notations and abstraction levels, while symbolic matching ensures formal soundness. Grounded in compositional reasoning, this vision points toward verification ecosystems that evolve systematically, leveraging past verification efforts to accelerate future assurance. 

---
# NLI:Non-uniform Linear Interpolation Approximation of Nonlinear Operations for Efficient LLMs Inference 

**Authors**: Jiangyong Yu, Xiaomeng Han, Xing Hu, Chen Xu, Zhe Jiang, Dawei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.02988)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of tasks, but their deployment is often constrained by substantial memory footprints and computational costs. While prior work has achieved significant progress in compressing and accelerating linear layers, nonlinear layers-such as SiLU, RMSNorm, and Softmax-still heavily depend on high-precision floating-point operations. In this paper, we propose a calibration-free, dynamic-programming-optimal, and hardware-friendly framework called Non-uniform Linear Interpolation (NLI). NLI is capable of efficiently approximating a variety of nonlinear functions, enabling seamless integration into LLMs and other deep neural networks with almost no loss in accuracy. NLI ingeniously recasts cutpoint selection as a dynamic-programming problem, achieving the globally minimal interpolation error in O(MxN2) time via Bellman's optimality principle. Based on the NLI algorithm, we also design and implement a plug-and-play universal nonlinear computation unit. Hardware experiments demonstrate that the NLI Engine achieves more than 4x improvement in computational efficiency compared to the state-of-the-art designs. 

---
# HALT: Hallucination Assessment via Log-probs as Time series 

**Authors**: Ahmad Shapiro, Karan Taneja, Ashok Goel  

**Link**: [PDF](https://arxiv.org/pdf/2602.02888)  

**Abstract**: Hallucinations remain a major obstacle for large language models (LLMs), especially in safety-critical domains. We present HALT (Hallucination Assessment via Log-probs as Time series), a lightweight hallucination detector that leverages only the top-20 token log-probabilities from LLM generations as a time series. HALT uses a gated recurrent unit model combined with entropy-based features to learn model calibration bias, providing an extremely efficient alternative to large encoders. Unlike white-box approaches, HALT does not require access to hidden states or attention maps, relying only on output log-probabilities. Unlike black-box approaches, it operates on log-probs rather than surface-form text, which enables stronger domain generalization and compatibility with proprietary LLMs without requiring access to internal weights. To benchmark performance, we introduce HUB (Hallucination detection Unified Benchmark), which consolidates prior datasets into ten capabilities covering both reasoning tasks (Algorithmic, Commonsense, Mathematical, Symbolic, Code Generation) and general purpose skills (Chat, Data-to-Text, Question Answering, Summarization, World Knowledge). While being 30x smaller, HALT outperforms Lettuce, a fine-tuned modernBERT-base encoder, achieving a 60x speedup gain on HUB. HALT and HUB together establish an effective framework for hallucination detection across diverse LLM capabilities. 

---
# Entropy-Guided Dynamic Tokens for Graph-LLM Alignment in Molecular Understanding 

**Authors**: Zihao Jing, Qiuhao Zeng, Ruiyi Fang, Yan Sun, Boyu Wang, Pingzhao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2602.02742)  

**Abstract**: Molecular understanding is central to advancing areas such as scientific discovery, yet Large Language Models (LLMs) struggle to understand molecular graphs effectively. Existing graph-LLM bridges often adapt the Q-Former-style connector with fixed-length static tokens, which is originally designed for vision tasks. These designs overlook stereochemistry and substructural context and typically require costly LLM-backbone fine-tuning, limiting efficiency and generalization. We introduce EDT-Former, an Entropy-guided Dynamic Token Transformer that generates tokens aligned with informative molecular patches, thereby preserving both local and global structural features for molecular graph understanding. Beyond prior approaches, EDT-Former enables alignment between frozen graph encoders and LLMs without tuning the LLM backbone (excluding the embedding layer), resulting in computationally efficient finetuning, and achieves stateof-the-art results on MoleculeQA, Molecule-oriented Mol-Instructions, and property prediction benchmarks (TDC, MoleculeNet), underscoring its effectiveness for scalable and generalizable multimodal molecular understanding 

---
# When Noise Lowers The Loss: Rethinking Likelihood-Based Evaluation in Music Large Language Models 

**Authors**: Xiaosha Li, Chun Liu, Ziyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.02738)  

**Abstract**: The rise of music large language models (LLMs) demands robust methods of evaluating output quality, especially in distinguishing high-quality compositions from "garbage music". Curiously, we observe that the standard cross-entropy loss -- a core training metric -- often decrease when models encounter systematically corrupted music, undermining its validity as a standalone quality indicator. To investigate this paradox, we introduce noise injection experiment, where controlled noise signal of varying lengths are injected into musical contexts. We hypothesize that a model's loss reacting positively to these perturbations, specifically a sharp increase ("Peak" area) for short injection, can serve as a proxy for its ability to discern musical integrity. Experiments with MusicGen models in the audio waveform domain confirm that Music LLMs respond more strongly to local, texture-level disruptions than to global semantic corruption. Beyond exposing this bias, our results highlight a new principle: the shape of the loss curve -- rather than its absolute value -- encodes critical information about the quality of the generated content (i.e., model behavior). We envision this profile-based evaluation as a label-free, model-intrinsic framework for assessing musical quality -- opening the door to more principled training objectives and sharper benchmarks. 

---
# Trailer Reimagined: An Innovative, Llm-DRiven, Expressive Automated Movie Summary framework (TRAILDREAMS) 

**Authors**: Roberto Balestri, Pasquale Cascarano, Mirko Degli Esposti, Guglielmo Pescatore  

**Link**: [PDF](https://arxiv.org/pdf/2602.02630)  

**Abstract**: This paper introduces TRAILDREAMS, a framework that uses a large language model (LLM) to automate the production of movie trailers. The purpose of LLM is to select key visual sequences and impactful dialogues, and to help TRAILDREAMS to generate audio elements such as music and voiceovers. The goal is to produce engaging and visually appealing trailers efficiently. In comparative evaluations, TRAILDREAMS surpasses current state-of-the-art trailer generation methods in viewer ratings. However, it still falls short when compared to real, human-crafted trailers. While TRAILDREAMS demonstrates significant promise and marks an advancement in automated creative processes, further improvements are necessary to bridge the quality gap with traditional trailers. 

---
# BinaryPPO: Efficient Policy Optimization for Binary Classification 

**Authors**: Punya Syon Pandey, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2602.02708)  

**Abstract**: Supervised fine-tuning (SFT) is the standard approach for binary classification tasks such as toxicity detection, factuality verification, and causal inference. However, SFT often performs poorly in real-world settings with label noise, class imbalance, or sparse supervision. We introduce BinaryPPO, an offline reinforcement learning large language model (LLM) framework that reformulates binary classification as a reward maximization problem. Our method leverages a variant of Proximal Policy Optimization (PPO) with a confidence-weighted reward function that penalizes uncertain or incorrect predictions, enabling the model to learn robust decision policies from static datasets without online interaction. Across eight domain-specific benchmarks and multiple models with differing architectures, BinaryPPO improves accuracy by 40-60 percentage points, reaching up to 99%, substantially outperforming supervised baselines. We provide an in-depth analysis of the role of reward shaping, advantage scaling, and policy stability in enabling this improvement. Overall, we demonstrate that confidence-based reward design provides a robust alternative to SFT for binary classification. Our code is available at this https URL. 

---
# Monotonicity as an Architectural Bias for Robust Language Models 

**Authors**: Patrick Cooper, Alireza Nadali, Ashutosh Trivedi, Alvaro Velasquez  

**Link**: [PDF](https://arxiv.org/pdf/2602.02686)  

**Abstract**: Large language models (LLMs) are known to exhibit brittle behavior under adversarial prompts and jailbreak attacks, even after extensive alignment and fine-tuning. This fragility reflects a broader challenge of modern neural language models: small, carefully structured perturbations in high-dimensional input spaces can induce large and unpredictable changes in internal semantic representations and output.
We investigate monotonicity as an architectural inductive bias for improving the robustness of Transformer-based language models. Monotonicity constrains semantic transformations so that strengthening information, evidence, or constraints cannot lead to regressions in the corresponding internal representations. Such order-preserving behavior has long been exploited in control and safety-critical systems to simplify reasoning and improve robustness, but has traditionally been viewed as incompatible with the expressivity required by neural language models.
We show that this trade-off is not inherent. By enforcing monotonicity selectively in the feed-forward sublayers of sequence-to-sequence Transformers -- while leaving attention mechanisms unconstrained -- we obtain monotone language models that preserve the performance of their pretrained counterparts. This architectural separation allows negation, contradiction, and contextual interactions to be introduced explicitly through attention, while ensuring that subsequent semantic refinement is order-preserving. Empirically, monotonicity substantially improves robustness: adversarial attack success rates drop from approximately 69% to 19%, while standard summarization performance degrades only marginally. 

---
# Benchmarking Large Language Models for Zero-shot and Few-shot Phishing URL Detection 

**Authors**: Najmul Hasan, Prashanth BusiReddyGari  

**Link**: [PDF](https://arxiv.org/pdf/2602.02641)  

**Abstract**: The Uniform Resource Locator (URL), introduced in a connectivity-first era to define access and locate resources, remains historically limited, lacking future-proof mechanisms for security, trust, or resilience against fraud and abuse, despite the introduction of reactive protections like HTTPS during the cybersecurity era. In the current AI-first threatscape, deceptive URLs have reached unprecedented sophistication due to the widespread use of generative AI by cybercriminals and the AI-vs-AI arms race to produce context-aware phishing websites and URLs that are virtually indistinguishable to both users and traditional detection tools. Although AI-generated phishing accounted for a small fraction of filter-bypassing attacks in 2024, phishing volume has escalated over 4,000% since 2022, with nearly 50% more attacks evading detection. At the rate the threatscape is escalating, and phishing tactics are emerging faster than labeled data can be produced, zero-shot and few-shot learning with large language models (LLMs) offers a timely and adaptable solution, enabling generalization with minimal supervision. Given the critical importance of phishing URL detection in large-scale cybersecurity defense systems, we present a comprehensive benchmark of LLMs under a unified zero-shot and few-shot prompting framework and reveal operational trade-offs. Our evaluation uses a balanced dataset with consistent prompts, offering detailed analysis of performance, generalization, and model efficacy, quantified by accuracy, precision, recall, F1 score, AUROC, and AUPRC, to reflect both classification quality and practical utility in threat detection settings. We conclude few-shot prompting improves performance across multiple LLMs. 

---
# CaST: Causal Discovery via Spatio-Temporal Graphs in Disaster Tweets 

**Authors**: Hieu Duong, Eugene Levin, Todd Gary, Long Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2602.02601)  

**Abstract**: Understanding causality between real-world events from social media is essential for situational awareness, yet existing causal discovery methods often overlook the interplay between semantic, spatial, and temporal contexts. We propose CaST: Causal Discovery via Spatio-Temporal Graphs, a unified framework for causal discovery in disaster domain that integrates semantic similarity and spatio-temporal proximity using Large Language Models (LLMs) pretrained on disaster datasets. CaST constructs an event graph for each window of tweets. Each event extracted from tweets is represented as a node embedding enriched with its contextual semantics, geographic coordinates, and temporal features. These event nodes are then connected to form a spatio-temporal event graph, which is processed using a multi-head Graph Attention Network (GAT) \cite{gat} to learn directed causal relationships. We construct an in-house dataset of approximately 167K disaster-related tweets collected during Hurricane Harvey and annotated following the MAVEN-ERE schema. Experimental results show that CaST achieves superior performance over both traditional and state-of-the-art methods. Ablation studies further confirm that incorporating spatial and temporal signals substantially improves both recall and stability during training. Overall, CaST demonstrates that integrating spatio-temporal reasoning into event graphs enables more robust and interpretable causal discovery in disaster-related social media text. 

---
# QuantLRM: Quantization of Large Reasoning Models via Fine-Tuning Signals 

**Authors**: Nan Zhang, Eugene Kwek, Yusen Zhang, Muyu Pan, Suhang Wang, Prasenjit Mitra, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.02581)  

**Abstract**: Weight-only quantization is important for compressing Large Language Models (LLMs). Inspired by the spirit of classical magnitude pruning, we study whether the magnitude of weight updates during reasoning-incentivized fine-tuning can provide valuable signals for quantizing Large Reasoning Models (LRMs). We hypothesize that the smallest and largest weight updates during fine-tuning are more important than those of intermediate magnitude, a phenomenon we term "protecting both ends". Upon hypothesis validation, we introduce QuantLRM, which stands for weight quantization of LRMs via fine-tuning signals. We fit simple restricted quadratic functions on weight updates to protect both ends. By multiplying the average quadratic values with the count of zero weight updates of channels, we compute channel importance that is more effective than using activation or second-order information. We run QuantLRM to quantize various fine-tuned models (including supervised, direct preference optimization, and reinforcement learning fine-tuning) over four reasoning benchmarks (AIME-120, FOLIO, temporal sequences, and GPQA-Diamond) and empirically find that QuantLRM delivers a consistent improvement for LRMs quantization, with an average improvement of 6.55% on a reinforcement learning fine-tuned model. Also supporting non-fine-tuned LRMs, QuantLRM gathers effective signals via pseudo-fine-tuning, which greatly enhances its applicability. 

---
# DECEIVE-AFC: Adversarial Claim Attacks against Search-Enabled LLM-based Fact-Checking Systems 

**Authors**: Haoran Ou, Kangjie Chen, Gelei Deng, Hangcheng Liu, Jie Zhang, Tianwei Zhang, Kwok-Yan Lam  

**Link**: [PDF](https://arxiv.org/pdf/2602.02569)  

**Abstract**: Fact-checking systems with search-enabled large language models (LLMs) have shown strong potential for verifying claims by dynamically retrieving external evidence. However, the robustness of such systems against adversarial attack remains insufficiently understood. In this work, we study adversarial claim attacks against search-enabled LLM-based fact-checking systems under a realistic input-only threat model. We propose DECEIVE-AFC, an agent-based adversarial attack framework that integrates novel claim-level attack strategies and adversarial claim validity evaluation principles. DECEIVE-AFC systematically explores adversarial attack trajectories that disrupt search behavior, evidence retrieval, and LLM-based reasoning without relying on access to evidence sources or model internals. Extensive evaluations on benchmark datasets and real-world systems demonstrate that our attacks substantially degrade verification performance, reducing accuracy from 78.7% to 53.7%, and significantly outperform existing claim-based attack baselines with strong cross-system transferability. 

---
# MathlibLemma: Folklore Lemma Generation and Benchmark for Formal Mathematics 

**Authors**: Xinyu Liu, Zixuan Xie, Amir Moeini, Claire Chen, Shuze Daniel Liu, Yu Meng, Aidong Zhang, Shangtong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.02561)  

**Abstract**: While the ecosystem of Lean and Mathlib has enjoyed celebrated success in formal mathematical reasoning with the help of large language models (LLMs), the absence of many folklore lemmas in Mathlib remains a persistent barrier that limits Lean's usability as an everyday tool for mathematicians like LaTeX or Maple. To address this, we introduce MathlibLemma, the first LLM-based multi-agent system to automate the discovery and formalization of mathematical folklore lemmas. This framework constitutes our primary contribution, proactively mining the missing connective tissue of mathematics. Its efficacy is demonstrated by the production of a verified library of folklore lemmas, a subset of which has already been formally merged into the latest build of Mathlib, thereby validating the system's real-world utility and alignment with expert standards. Leveraging this pipeline, we further construct the MathlibLemma benchmark, a suite of 4,028 type-checked Lean statements spanning a broad range of mathematical domains. By transforming the role of LLMs from passive consumers to active contributors, this work establishes a constructive methodology for the self-evolution of formal mathematical libraries. 

---
# Beyond Experience Retrieval: Learning to Generate Utility-Optimized Structured Experience for Frozen LLMs 

**Authors**: Xuancheng Li, Haitao Li, Yujia Zhou, Yiqun Liu, Qingyao Ai  

**Link**: [PDF](https://arxiv.org/pdf/2602.02556)  

**Abstract**: Large language models (LLMs) are largely static and often redo reasoning or repeat mistakes. Prior experience reuse typically relies on external retrieval, which is similarity-based, can introduce noise, and adds latency. We introduce SEAM (Structured Experience Adapter Module), a lightweight, executor-specific plug-in that stores experience in its parameters and generates a structured, instance-tailored experience entry in a single forward pass to guide a frozen LLM executor. SEAM is trained for utility via executor rollouts and GRPO while keeping the executor frozen, and it can be further improved after deployment with supervised fine-tuning on logged successful trajectories. Experiments on mathematical reasoning benchmarks show consistent accuracy gains across executors with low overhead. Extensive ablations and analyses further elucidate the mechanisms underlying SEAM's effectiveness and robustness. 

---
# Beyond Alignment: Expanding Reasoning Capacity via Manifold-Reshaping Policy Optimization 

**Authors**: Dayu Wang, Jiaye Yang, Weikang Li, Jiahui Liang, Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.02545)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has demonstrated remarkable success in enhancing the reasoning capabilities of Large Language Models (LLMs). However, recent studies question whether RL genuinely expands reasoning capacity or merely aligns existing latent capabilities, arguing that exploration remains confined within the pre-trained model's low-rank bias manifold. In this work, we challenge this accessibility boundary hypothesis by demonstrating that the latent reasoning space can be fundamentally expanded through targeted geometric interventions. We propose Manifold-Reshaping Policy Optimization (MRPO), a geometric framework designed to fundamentally restructure the inference space of LLMs. MRPO operates in two stages: first, we employ Spectral Orthogonal Exploration (SOE) to eject the policy initialization into the null space of the bias manifold; second, we integrate an Effective Rank regularization term into the policy optimization objective. This approach incentivizes the discovery and maintenance of high-dimensional reasoning trajectories against the entropy-reducing tendency of standard RL. Empirically, our 4B-parameter method achieves state-of-the-art performance on mathematical tasks, significantly outperforming larger models (e.g., Qwen3-32B) and expanding the capability boundary beyond standard GRPO. Our code is available at this https URL 

---
# HyPAC: Cost-Efficient LLMs-Human Hybrid Annotation with PAC Error Guarantees 

**Authors**: Hao Zeng, Huipeng Huang, Xinhao Qu, Jianguo Huang, Bingyi Jing, Hongxin Wei  

**Link**: [PDF](https://arxiv.org/pdf/2602.02550)  

**Abstract**: Data annotation often involves multiple sources with different cost-quality trade-offs, such as fast large language models (LLMs), slow reasoning models, and human experts. In this work, we study the problem of routing inputs to the most cost-efficient annotation source while controlling the labeling error on test instances. We propose \textbf{HyPAC}, a method that adaptively labels inputs to the most cost-efficient annotation source while providing distribution-free guarantees on annotation error. HyPAC calibrates two decision thresholds using importance sampling and upper confidence bounds, partitioning inputs into three regions based on uncertainty and routing each to the appropriate annotation source. We prove that HyPAC achieves the minimum expected cost with a probably approximately correct (PAC) guarantee on the annotation error, free of data distribution and pre-trained models. Experiments on common benchmarks demonstrate the effectiveness of our method, reducing the annotation cost by 78.51\% while tightly controlling the annotation error. 

---
# Toward Ultra-Long-Horizon Sequential Model Editing 

**Authors**: Mingda Liu, Zhenghan Zhu, Ze'an Miao, Katsuki Fujisawa  

**Link**: [PDF](https://arxiv.org/pdf/2602.02543)  

**Abstract**: Model editing has emerged as a practical approach for mitigating factual errors and outdated knowledge in large language models (LLMs). Among existing methods, the Locate-and-Edit (L&E) paradigm is the dominant framework: it locates MLP parameters implicated in expressing a target fact, and then performs a localized update to rewrite that fact. However, long sequences of edits often trigger abrupt model collapse in L&E beyond a critical point. We empirically identify a strong correlation between collapse and explosive growth of edited MLP weight norms, and formally prove that commonly used L&E update rules can induce exponential norm growth across sequential edits in the absence of explicit norm control. To address this issue, we propose Norm-Anchor Scaling NAS, a plug-and-play norm-constrained strategy. Across extensive experiments, NAS delays the collapse point of representative L&E algorithms by more than 4 times and yields a 72.2% average relative gain in editing performance, requiring only a single additional line of code and incurring negligible computational overhead. 

---
# Scaled Dot-Product Attention implements projection of inputs onto a common surface 

**Authors**: Terence D Sanger  

**Link**: [PDF](https://arxiv.org/pdf/2602.02521)  

**Abstract**: Scaled dot-product attention (SDPA) is a fundamental component responsible for the success of large-language models and other nonlinear signal processing applications. The rationale for SDPA has been based upon "query, key, value" concepts borrowed from database theory, but these concepts are difficult to reconcile with standard methods in mathematical signal processing. We show that SDPA can be rewritten in a different but mathematically equivalent form as a projection of the input vectors onto a common surface determined by the inputs themselves. Therefore SDPA discovers nonlinear dependencies in the input that are time-dependent and context-dependent. The rewritten form of SDPA permits increased speed of both feedforward and learning algorithms, but more importantly suggests potential extensions. In the context of language, we re-interpret the role of SDPA as finding a time-dependent contextual meaning determined by the surface on which the set of input vectors lies. Input token embeddings are then modified by the local context surface. This interpretation differs substantially from the concept of "self-attention", and provides a strong justification for the use of SDPA for time-series data with time-varying local nonlinear dependencies. 

---
# Evaluation of Large Language Models' educational feedback in Higher Education: potential, limitations and implications for educational practice 

**Authors**: Daniele Agostini, Federica Picasso  

**Link**: [PDF](https://arxiv.org/pdf/2602.02519)  

**Abstract**: The importance of managing feedback practices in higher education has been widely recognised, as they play a crucial role in enhancing teaching, learning, and assessment processes. In today's educational landscape, feedback practices are increasingly influenced by technological advancements, particularly artificial intelligence (AI). Understanding the impact of AI on feedback generation is essential for identifying its potential benefits and establishing effective implementation strategies. This study examines how AI-generated feedback supports student learning using a well-established analytical framework. Specifically, feedback produced by different Large Language Models (LLMs) was assessed in relation to student-designed projects within a training course on inclusive teaching and learning. The evaluation process involved providing seven LLMs with a structured rubric, developed by the university instructor, which defined specific criteria and performance levels. The LLMs were tasked with generating both quantitative assessments and qualitative feedback based on this rubric. The AI-generated feedback was then analysed using Hughes, Smith, and Creese's framework to evaluate its structure and effectiveness in fostering formative learning experiences. Overall, these findings indicate that LLMs can generate well-structured feedback and hold great potential as a sustainable and meaningful feedback tool, provided they are guided by clear contextual information and a well-defined instructions that will be explored further in the conclusions. 

---
# GraphDancer: Training LLMs to Explore and Reason over Graphs via Curriculum Reinforcement Learning 

**Authors**: Yuyang Bai, Zhuofeng Li, Ping Nie, Jianwen Xie, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.02518)  

**Abstract**: Large language models (LLMs) increasingly rely on external knowledge to improve factuality, yet many real-world knowledge sources are organized as heterogeneous graphs rather than plain text. Reasoning over such graph-structured knowledge poses two key challenges: (1) navigating structured, schema-defined relations requires precise function calls rather than similarity-based retrieval, and (2) answering complex questions often demands multi-hop evidence aggregation through iterative information seeking. We propose GraphDancer, a reinforcement learning (RL) framework that teaches LLMs to navigate graphs by interleaving reasoning and function execution. To make RL effective for moderate-sized LLMs, we introduce a graph-aware curriculum that schedules training by the structural complexity of information-seeking trajectories using an easy-to-hard biased sampler. We evaluate GraphDancer on a multi-domain benchmark by training on one domain only and testing on unseen domains and out-of-distribution question types. Despite using only a 3B backbone, GraphDancer outperforms baselines equipped with either a 14B backbone or GPT-4o-mini, demonstrating robust cross-domain generalization of graph exploration and reasoning skills. Our code and models can be found at this https URL . 

---
# CodeGuard: Improving LLM Guardrails in CS Education 

**Authors**: Nishat Raihan, Noah Erdachew, Jayoti Devi, Joanna C. S. Santos, Marcos Zampieri  

**Link**: [PDF](https://arxiv.org/pdf/2602.02509)  

**Abstract**: Large language models (LLMs) are increasingly embedded in Computer Science (CS) classrooms to automate code generation, feedback, and assessment. However, their susceptibility to adversarial or ill-intentioned prompts threatens student learning and academic integrity. To cope with this important issue, we evaluate existing off-the-shelf LLMs in handling unsafe and irrelevant prompts within the domain of CS education. We identify important shortcomings in existing LLM guardrails which motivates us to propose CodeGuard, a comprehensive guardrail framework for educational AI systems. CodeGuard includes (i) a first-of-its-kind taxonomy for classifying prompts; (ii) the CodeGuard dataset, a collection of 8,000 prompts spanning the taxonomy; and (iii) PromptShield, a lightweight sentence-encoder model fine-tuned to detect unsafe prompts in real time. Experiments show that PromptShield achieves 0.93 F1 score, surpassing existing guardrail methods. Additionally, further experimentation reveals that CodeGuard reduces potentially harmful or policy-violating code completions by 30-65% without degrading performance on legitimate educational tasks. The code, datasets, and evaluation scripts are made freely available to the community. 

---
# Test-Time Detoxification without Training or Learning Anything 

**Authors**: Baturay Saglam, Dionysis Kalogerias  

**Link**: [PDF](https://arxiv.org/pdf/2602.02498)  

**Abstract**: Large language models can produce toxic or inappropriate text even for benign inputs, creating risks when deployed at scale. Detoxification is therefore important for safety and user trust, particularly when we want to reduce harmful content without sacrificing the model's generation quality. Many existing approaches rely on model retraining, gradients, or learned auxiliary components, which can be costly and may not transfer across model families or to truly black-box settings. We introduce a test-time procedure that approximates the gradient of completion toxicity with respect to the input embeddings and uses a small number of descent steps to steer generation toward less toxic continuations. This is achieved with zeroth-order optimization that requires only access to input embeddings, a toxicity scoring function, and forward evaluations of the model. Empirically, the approach delivers robust toxicity reductions across models and prompts and, in most settings, achieves the best overall toxicity-quality trade-off. More broadly, our work positions word embeddings as effective control variables and encourages wider use of black-box optimization to guide autoregressive language models toward scalable, safer text generation, without requiring any training or access to intermediate computations. 

---
# STEMVerse: A Dual-Axis Diagnostic Framework for STEM Reasoning in Large Language Models 

**Authors**: Xuzhao Li, Xuchen Li, Jian Zhao, Shiyu Hu  

**Link**: [PDF](https://arxiv.org/pdf/2602.02497)  

**Abstract**: As Large Language Models (LLMs) achieve significant breakthroughs in complex reasoning tasks, evaluating their proficiency in science, technology, engineering, and mathematics (STEM) has become a primary method for measuring machine intelligence. However, current evaluation paradigms often treat benchmarks as isolated "silos," offering only monolithic aggregate scores that neglect the intricacies of both academic specialization and cognitive depth. This result-oriented approach fails to distinguish whether model errors stem from insufficient domain knowledge or deficiencies in cognitive capacity, thereby limiting the diagnostic value. To address this, we propose STEMVerse, a diagnostic framework designed to systematically analyze the STEM reasoning capabilities of LLMs. This framework characterizes model performance across academic specialization and cognitive complexity to map the capability required for reasoning. We re-aggregate over 20,000 STEM problems from mainstream benchmarks into a unified "Discipline $\times$ Cognition" capability space, assigning dual-axis labels to every instance. Utilizing this unified diagnostic framework, we systematically evaluate representative LLM families across varying parameter scales and training paradigms. Our empirical results reveal structural failure patterns in STEM reasoning. By integrating multi-disciplinary coverage and fine-grained cognitive stratification into a unified framework, STEMVerse provides a clear and actionable perspective for understanding the scientific reasoning characteristics of LLMs. 

---
# Context Compression via Explicit Information Transmission 

**Authors**: Jiangnan Ye, Hanqi Yan, Zhenyi Shen, Heng Chang, Ye Mao, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2602.03784)  

**Abstract**: Long-context inference with Large Language Models (LLMs) is costly due to quadratic attention and growing key-value caches, motivating context compression. In this work, we study soft context compression, where a long context is condensed into a small set of continuous representations. Existing methods typically re-purpose the LLM itself as a trainable compressor, relying on layer-by-layer self-attention to iteratively aggregate information. We argue that this paradigm suffers from two structural limitations: (i) progressive representation overwriting across layers (ii) uncoordinated allocation of compression capacity across tokens. We propose ComprExIT (Context Compression via Explicit Information Transmission), a lightweight framework that formulates soft compression into a new paradigm: explicit information transmission over frozen LLM hidden states. This decouples compression from the model's internal self-attention dynamics. ComprExIT performs (i) depth-wise transmission to selectively transmit multi-layer information into token anchors, mitigating progressive overwriting, and (ii) width-wise transmission to aggregate anchors into a small number of slots via a globally optimized transmission plan, ensuring coordinated allocation of information. Across six question-answering benchmarks, ComprExIT consistently outperforms state-of-the-art context compression methods while introducing only ~1% additional parameters, demonstrating that explicit and coordinated information transmission enables more effective and robust long-context compression. 

---
# Beyond Tokens: Semantic-Aware Speculative Decoding for Efficient Inference by Probing Internal States 

**Authors**: Ximing Dong, Shaowei Wang, Dayi Lin, Boyuan Chen, Ahmed E. Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2602.03708)  

**Abstract**: Large Language Models (LLMs) achieve strong performance across many tasks but suffer from high inference latency due to autoregressive decoding. The issue is exacerbated in Large Reasoning Models (LRMs), which generate lengthy chains of thought. While speculative decoding accelerates inference by drafting and verifying multiple tokens in parallel, existing methods operate at the token level and ignore semantic equivalence (i.e., different token sequences expressing the same meaning), leading to inefficient rejections. We propose SemanticSpec, a semantic-aware speculative decoding framework that verifies entire semantic sequences instead of tokens. SemanticSpec introduces a semantic probability estimation mechanism that probes the model's internal hidden states to assess the likelihood of generating sequences with specific this http URL on four benchmarks show that SemanticSpec achieves up to 2.7x speedup on DeepSeekR1-32B and 2.1x on QwQ-32B, consistently outperforming token-level and sequence-level baselines in both efficiency and effectiveness. 

---
# Learning Query-Specific Rubrics from Human Preferences for DeepResearch Report Generation 

**Authors**: Changze Lv, Jie Zhou, Wentao Zhao, Jingwen Xu, Zisu Huang, Muzhao Tian, Shihan Dou, Tao Gui, Le Tian, Xiao Zhou, Xiaoqing Zheng, Xuanjing Huang, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2602.03619)  

**Abstract**: Nowadays, training and evaluating DeepResearch-generated reports remain challenging due to the lack of verifiable reward signals. Accordingly, rubric-based evaluation has become a common practice. However, existing approaches either rely on coarse, pre-defined rubrics that lack sufficient granularity, or depend on manually constructed query-specific rubrics that are costly and difficult to scale. In this paper, we propose a pipeline to train human-preference-aligned query-specific rubric generators tailored for DeepResearch report generation. We first construct a dataset of DeepResearch-style queries annotated with human preferences over paired reports, and train rubric generators via reinforcement learning with a hybrid reward combining human preference supervision and LLM-based rubric evaluation. To better handle long-horizon reasoning, we further introduce a Multi-agent Markov-state (MaMs) workflow for report generation. We empirically show that our proposed rubric generators deliver more discriminative and better human-aligned supervision than existing rubric design strategies. Moreover, when integrated into the MaMs training framework, DeepResearch systems equipped with our rubric generators consistently outperform all open-source baselines on the DeepResearch Bench and achieve performance comparable to that of leading closed-source models. 

---
# Learning to Reason Faithfully through Step-Level Faithfulness Maximization 

**Authors**: Runquan Gui, Yafu Li, Xiaoye Qu, Ziyan Liu, Yeqiu Cheng, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2602.03507)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has markedly improved the performance of Large Language Models (LLMs) on tasks requiring multi-step reasoning. However, most RLVR pipelines rely on sparse outcome-based rewards, providing little supervision over intermediate steps and thus encouraging over-confidence and spurious reasoning, which in turn increases hallucinations. To address this, we propose FaithRL, a general reinforcement learning framework that directly optimizes reasoning faithfulness. We formalize a faithfulness-maximization objective and theoretically show that optimizing it mitigates over-confidence. To instantiate this objective, we introduce a geometric reward design and a faithfulness-aware advantage modulation mechanism that assigns step-level credit by penalizing unsupported steps while preserving valid partial derivations. Across diverse backbones and benchmarks, FaithRL consistently reduces hallucination rates while maintaining (and often improving) answer correctness. Further analysis confirms that FaithRL increases step-wise reasoning faithfulness and generalizes robustly. Our code is available at this https URL. 

---
# Can Large Language Models Generalize Procedures Across Representations? 

**Authors**: Fangru Lin, Valentin Hofmann, Xingchen Wan, Weixing Wang, Zifeng Ding, Anthony G. Cohn, Janet B. Pierrehumbert  

**Link**: [PDF](https://arxiv.org/pdf/2602.03542)  

**Abstract**: Large language models (LLMs) are trained and tested extensively on symbolic representations such as code and graphs, yet real-world user tasks are often specified in natural language. To what extent can LLMs generalize across these representations? Here, we approach this question by studying isomorphic tasks involving procedures represented in code, graphs, and natural language (e.g., scheduling steps in planning). We find that training LLMs with popular post-training methods on graphs or code data alone does not reliably generalize to corresponding natural language tasks, while training solely on natural language can lead to inefficient performance gains. To address this gap, we propose a two-stage data curriculum that first trains on symbolic, then natural language data. The curriculum substantially improves model performance across model families and tasks. Remarkably, a 1.5B Qwen model trained by our method can closely match zero-shot GPT-4o in naturalistic planning. Finally, our analysis suggests that successful cross-representation generalization can be interpreted as a form of generative analogy, which our curriculum effectively encourages. 

---
# No Shortcuts to Culture: Indonesian Multi-hop Question Answering for Complex Cultural Understanding 

**Authors**: Vynska Amalia Permadi, Xingwei Tan, Nafise Sadat Moosavi, Nikos Aletras  

**Link**: [PDF](https://arxiv.org/pdf/2602.03709)  

**Abstract**: Understanding culture requires reasoning across context, tradition, and implicit social knowledge, far beyond recalling isolated facts. Yet most culturally focused question answering (QA) benchmarks rely on single-hop questions, which may allow models to exploit shallow cues rather than demonstrate genuine cultural reasoning. In this work, we introduce ID-MoCQA, the first large-scale multi-hop QA dataset for assessing the cultural understanding of large language models (LLMs), grounded in Indonesian traditions and available in both English and Indonesian. We present a new framework that systematically transforms single-hop cultural questions into multi-hop reasoning chains spanning six clue types (e.g., commonsense, temporal, geographical). Our multi-stage validation pipeline, combining expert review and LLM-as-a-judge filtering, ensures high-quality question-answer pairs. Our evaluation across state-of-the-art models reveals substantial gaps in cultural reasoning, particularly in tasks requiring nuanced inference. ID-MoCQA provides a challenging and essential benchmark for advancing the cultural competency of LLMs. 

---
# ForesightKV: Optimizing KV Cache Eviction for Reasoning Models by Learning Long-Term Contribution 

**Authors**: Zican Dong, Peiyu Liu, Junyi Li, Zhipeng Chen, Han Peng, Shuo Wang, Wayne Xin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.03203)  

**Abstract**: Recently, large language models (LLMs) have shown remarkable reasoning abilities by producing long reasoning traces. However, as the sequence length grows, the key-value (KV) cache expands linearly, incurring significant memory and computation costs. Existing KV cache eviction methods mitigate this issue by discarding less important KV pairs, but often fail to capture complex KV dependencies, resulting in performance degradation. To better balance efficiency and performance, we introduce ForesightKV, a training-based KV cache eviction framework that learns to predict which KV pairs to evict during long-text generations. We first design the Golden Eviction algorithm, which identifies the optimal eviction KV pairs at each step using future attention scores. These traces and the scores at each step are then distilled via supervised training with a Pairwise Ranking Loss. Furthermore, we formulate cache eviction as a Markov Decision Process and apply the GRPO algorithm to mitigate the significant language modeling loss increase on low-entropy tokens. Experiments on AIME2024 and AIME2025 benchmarks of three reasoning models demonstrate that ForesightKV consistently outperforms prior methods under only half the cache budget, while benefiting synergistically from both supervised and reinforcement learning approaches. 

---
# MIRROR: A Multi-Agent Framework with Iterative Adaptive Revision and Hierarchical Retrieval for Optimization Modeling in Operations Research 

**Authors**: Yifan Shi, Jialong Shi, Jiayi Wang, Ye Fan, Jianyong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.03318)  

**Abstract**: Operations Research (OR) relies on expert-driven modeling-a slow and fragile process ill-suited to novel scenarios. While large language models (LLMs) can automatically translate natural language into optimization models, existing approaches either rely on costly post-training or employ multi-agent frameworks, yet most still lack reliable collaborative error correction and task-specific retrieval, often leading to incorrect outputs. We propose MIRROR, a fine-tuning-free, end-to-end multi-agent framework that directly translates natural language optimization problems into mathematical models and solver code. MIRROR integrates two core mechanisms: (1) execution-driven iterative adaptive revision for automatic error correction, and (2) hierarchical retrieval to fetch relevant modeling and coding exemplars from a carefully curated exemplar library. Experiments show that MIRROR outperforms existing methods on standard OR benchmarks, with notable results on complex industrial datasets such as IndustryOR and Mamo-ComplexLP. By combining precise external knowledge infusion with systematic error correction, MIRROR provides non-expert users with an efficient and reliable OR modeling solution, overcoming the fundamental limitations of general-purpose LLMs in expert optimization tasks. 

---
# Pursuing Best Industrial Practices for Retrieval-Augmented Generation in the Medical Domain 

**Authors**: Wei Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2602.03368)  

**Abstract**: While retrieval augmented generation (RAG) has been swiftly adopted in industrial applications based on large language models (LLMs), there is no consensus on what are the best practices for building a RAG system in terms of what are the components, how to organize these components and how to implement each component for the industrial applications, especially in the medical domain. In this work, we first carefully analyze each component of the RAG system and propose practical alternatives for each component. Then, we conduct systematic evaluations on three types of tasks, revealing the best practices for improving the RAG system and how LLM-based RAG systems make trade-offs between performance and efficiency. 

---
# ChemPro: A Progressive Chemistry Benchmark for Large Language Models 

**Authors**: Aaditya Baranwal, Shruti Vyas  

**Link**: [PDF](https://arxiv.org/pdf/2602.03108)  

**Abstract**: We introduce ChemPro, a progressive benchmark with 4100 natural language question-answer pairs in Chemistry, across 4 coherent sections of difficulty designed to assess the proficiency of Large Language Models (LLMs) in a broad spectrum of general chemistry topics. We include Multiple Choice Questions and Numerical Questions spread across fine-grained information recall, long-horizon reasoning, multi-concept questions, problem-solving with nuanced articulation, and straightforward questions in a balanced ratio, effectively covering Bio-Chemistry, Inorganic-Chemistry, Organic-Chemistry and Physical-Chemistry. ChemPro is carefully designed analogous to a student's academic evaluation for basic to high-school chemistry. A gradual increase in the question difficulty rigorously tests the ability of LLMs to progress from solving basic problems to solving more sophisticated challenges.
We evaluate 45+7 state-of-the-art LLMs, spanning both open-source and proprietary variants, and our analysis reveals that while LLMs perform well on basic chemistry questions, their accuracy declines with different types and levels of complexity. These findings highlight the critical limitations of LLMs in general scientific reasoning and understanding and point towards understudied dimensions of difficulty, emphasizing the need for more robust methodologies to improve LLMs. 

---
# AERO: Autonomous Evolutionary Reasoning Optimization via Endogenous Dual-Loop Feedback 

**Authors**: Zhitao Gao, Jie Ma, Xuhong Li, Pengyu Li, Ning Qu, Yaqiang Wu, Hui Liu, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.03084)  

**Abstract**: Large Language Models (LLMs) have achieved significant success in complex reasoning but remain bottlenecked by reliance on expert-annotated data and external verifiers. While existing self-evolution paradigms aim to bypass these constraints, they often fail to identify the optimal learning zone and risk reinforcing collective hallucinations and incorrect priors through flawed internal feedback. To address these challenges, we propose \underline{A}utonomous \underline{E}volutionary \underline{R}easoning \underline{O}ptimization (AERO), an unsupervised framework that achieves autonomous reasoning evolution by internalizing self-questioning, answering, and criticism within a synergistic dual-loop system. Inspired by the \textit{Zone of Proximal Development (ZPD)} theory, AERO utilizes entropy-based positioning to target the ``solvability gap'' and employs Independent Counterfactual Correction for robust verification. Furthermore, we introduce a Staggered Training Strategy to synchronize capability growth across functional roles and prevent curriculum collapse. Extensive evaluations across nine benchmarks spanning three domains demonstrate that AERO achieves average performance improvements of 4.57\% on Qwen3-4B-Base and 5.10\% on Qwen3-8B-Base, outperforming competitive baselines. Code is available at this https URL. 

---
# ReMiT: RL-Guided Mid-Training for Iterative LLM Evolution 

**Authors**: Junjie Huang, Jiarui Qin, Di Yin, Weiwen Liu, Yong Yu, Xing Sun, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03075)  

**Abstract**: Standard training pipelines for large language models (LLMs) are typically unidirectional, progressing from pre-training to post-training. However, the potential for a bidirectional process--where insights from post-training retroactively improve the pre-trained foundation--remains unexplored. We aim to establish a self-reinforcing flywheel: a cycle in which reinforcement learning (RL)-tuned model strengthens the base model, which in turn enhances subsequent post-training performance, requiring no specially trained teacher or reference model. To realize this, we analyze training dynamics and identify the mid-training (annealing) phase as a critical turning point for model capabilities. This phase typically occurs at the end of pre-training, utilizing high-quality corpora under a rapidly decaying learning rate. Building upon this insight, we introduce ReMiT (Reinforcement Learning-Guided Mid-Training). Specifically, ReMiT leverages the reasoning priors of RL-tuned models to dynamically reweight tokens during the mid-training phase, prioritizing those pivotal for reasoning. Empirically, ReMiT achieves an average improvement of 3\% on 10 pre-training benchmarks, spanning math, code, and general reasoning, and sustains these gains by over 2\% throughout the post-training pipeline. These results validate an iterative feedback loop, enabling continuous and self-reinforcing evolution of LLMs. 

---
# LatentMem: Customizing Latent Memory for Multi-Agent Systems 

**Authors**: Muxin Fu, Guibin Zhang, Xiangyuan Xue, Yafu Li, Zefeng He, Siyuan Huang, Xiaoye Qu, Yu Cheng, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03036)  

**Abstract**: Large language model (LLM)-powered multi-agent systems (MAS) demonstrate remarkable collective intelligence, wherein multi-agent memory serves as a pivotal mechanism for continual adaptation. However, existing multi-agent memory designs remain constrained by two fundamental bottlenecks: (i) memory homogenization arising from the absence of role-aware customization, and (ii) information overload induced by excessively fine-grained memory entries. To address these limitations, we propose LatentMem, a learnable multi-agent memory framework designed to customize agent-specific memories in a token-efficient manner. Specifically, LatentMem comprises an experience bank that stores raw interaction trajectories in a lightweight form, and a memory composer that synthesizes compact latent memories conditioned on retrieved experience and agent-specific contexts. Further, we introduce Latent Memory Policy Optimization (LMPO), which propagates task-level optimization signals through latent memories to the composer, encouraging it to produce compact and high-utility representations. Extensive experiments across diverse benchmarks and mainstream MAS frameworks show that LatentMem achieves a performance gain of up to $19.36$% over vanilla settings and consistently outperforms existing memory architectures, without requiring any modifications to the underlying frameworks. 

---
# CATNIP: LLM Unlearning via Calibrated and Tokenized Negative Preference Alignment 

**Authors**: Zhengbang Yang, Yisheng Zhong, Junyuan Hong, Zhuangdi Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2602.02824)  

**Abstract**: Pretrained knowledge memorized in LLMs raises critical concerns over safety and privacy, which has motivated LLM Unlearning as a technique for selectively removing the influences of undesirable knowledge. Existing approaches, rooted in Gradient Ascent (GA), often degrade general domain knowledge while relying on retention data or curated contrastive pairs, which can be either impractical or data and computationally prohibitive. Negative Preference Alignment has been explored for unlearning to tackle the limitations of GA, which, however, remains confined by its choice of reference model and shows undermined performance in realistic data settings. These limitations raise two key questions: i) Can we achieve effective unlearning that quantifies model confidence in undesirable knowledge and uses it to calibrate gradient updates more precisely, thus reducing catastrophic forgetting? ii) Can we make unlearning robust to data scarcity and length variation? We answer both questions affirmatively with CATNIP (Calibrated and Tokenized Negative Preference Alignment), a principled method that rescales unlearning effects in proportion to the model's token-level confidence, thus ensuring fine-grained control over forgetting. Extensive evaluations on MUSE and WMDP benchmarks demonstrated that our work enables effective unlearning without requiring retention data or contrastive unlearning response pairs, with stronger knowledge forgetting and preservation tradeoffs than state-of-the-art methods. 

---
# R2-Router: A New Paradigm for LLM Routing with Reasoning 

**Authors**: Jiaqi Xue, Qian Lou, Jiarong Xing, Heng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.02823)  

**Abstract**: As LLMs proliferate with diverse capabilities and costs, LLM routing has emerged by learning to predict each LLM's quality and cost for a given query, then selecting the one with high quality and low cost. However, existing routers implicitly assume a single fixed quality and cost per LLM for each query, ignoring that the same LLM's quality varies with its output length. This causes routers to exclude powerful LLMs when their estimated cost exceeds the budget, missing the opportunity that these LLMs could still deliver high quality at reduced cost with shorter outputs. To address this, we introduce R2-Router, which treats output length budget as a controllable variable and jointly selects the best LLM and length budget, enforcing the budget via length-constrained instructions. This enables R2-Router to discover that a powerful LLM with constrained output can outperform a weaker LLM at comparable cost-efficient configurations invisible to prior methods. Together with the router framework, we construct R2-Bench, the first routing dataset capturing LLM behavior across diverse output length budgets. Experiments show that R2-Router achieves state-of-the-art performance at 4-5x lower cost compared with existing routers. This work opens a new direction: routing as reasoning, where routers evolve from reactive selectors to deliberate reasoners that explore which LLM to use and at what cost budget. 

---
# Test-time Recursive Thinking: Self-Improvement without External Feedback 

**Authors**: Yufan Zhuang, Chandan Singh, Liyuan Liu, Yelong Shen, Dinghuai Zhang, Jingbo Shang, Jianfeng Gao, Weizhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.03094)  

**Abstract**: Modern Large Language Models (LLMs) have shown rapid improvements in reasoning capabilities, driven largely by reinforcement learning (RL) with verifiable rewards. Here, we ask whether these LLMs can self-improve without the need for additional training. We identify two core challenges for such systems: (i) efficiently generating diverse, high-quality candidate solutions, and (ii) reliably selecting correct answers in the absence of ground-truth supervision. To address these challenges, we propose Test-time Recursive Thinking (TRT), an iterative self-improvement framework that conditions generation on rollout-specific strategies, accumulated knowledge, and self-generated verification signals. Using TRT, open-source models reach 100% accuracy on AIME-25/24, and on LiveCodeBench's most difficult problems, closed-source models improve by 10.4-14.8 percentage points without external feedback. 

---
# Graph-Augmented Reasoning with Large Language Models for Tobacco Pest and Disease Management 

**Authors**: Siyu Li, Chenwei Song, Qi Zhou, Wan Zhou, Xinyi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.02635)  

**Abstract**: This paper proposes a graph-augmented reasoning framework for tobacco pest and disease management that integrates structured domain knowledge into large language models. Building on GraphRAG, we construct a domain-specific knowledge graph and retrieve query-relevant subgraphs to provide relational evidence during answer generation. The framework adopts ChatGLM as the Transformer backbone with LoRA-based parameter-efficient fine-tuning, and employs a graph neural network to learn node representations that capture symptom-disease-treatment dependencies. By explicitly modeling diseases, symptoms, pesticides, and control measures as linked entities, the system supports evidence-aware retrieval beyond surface-level text similarity. Retrieved graph evidence is incorporated into the LLM input to guide generation toward domain-consistent recommendations and to mitigate hallucinated or inappropriate treatments. Experimental results show consistent improvements over text-only baselines, with the largest gains observed on multi-hop and comparative reasoning questions that require chaining multiple relations. 

---
# From Task Solving to Robust Real-World Adaptation in LLM Agents 

**Authors**: Pouya Pezeshkpour, Estevam Hruschka  

**Link**: [PDF](https://arxiv.org/pdf/2602.02760)  

**Abstract**: Large language models are increasingly deployed as specialized agents that plan, call tools, and take actions over extended horizons. Yet many existing evaluations assume a "clean interface" where dynamics are specified and stable, tools and sensors are reliable, and success is captured by a single explicit objective-often overestimating real-world readiness. In practice, agents face underspecified rules, unreliable signals, shifting environments, and implicit, multi-stakeholder goals. The challenge is therefore not just solving tasks, but adapting while solving: deciding what to trust, what is wanted, when to verify, and when to fall back or escalate. We stress-test deployment-relevant robustness under four operational circumstances: partial observability, dynamic environments, noisy signals, and dynamic agent state. We benchmark agentic LLMs in a grid-based game with a simple goal but long-horizon execution. Episodes violate clean-interface assumptions yet remain solvable, forcing agents to infer rules, pay for information, adapt to environmental and internal shifts, and act cautiously under noise. Across five state-of-the-art LLM agents, we find large gaps between nominal task-solving and deployment-like robustness. Performance generally degrades as grid size and horizon increase, but rankings are unstable: weaker models can beat stronger ones when strategy matches the uncertainty regime. Despite no explicit instruction, agents trade off completion, efficiency, and penalty avoidance, suggesting partial objective inference. Ablations and feature analyses reveal model-specific sensitivities and failure drivers, motivating work on verification, safe action selection, and objective inference under partial observability, noise, and non-stationarity. 

---
# The Hypocrisy Gap: Quantifying Divergence Between Internal Belief and Chain-of-Thought Explanation via Sparse Autoencoders 

**Authors**: Shikhar Shiromani, Archie Chaudhury, Sri Pranav Kunda  

**Link**: [PDF](https://arxiv.org/pdf/2602.02496)  

**Abstract**: Large Language Models (LLMs) frequently exhibit unfaithful behavior, producing a final answer that differs significantly from their internal chain of thought (CoT) reasoning in order to appease the user they are conversing with. In order to better detect this behavior, we introduce the Hypocrisy Gap, a mechanistic metric utilizing Sparse Autoencoders (SAEs) to quantify the divergence between a model's internal reasoning and its final generation. By mathematically comparing an internal truth belief, derived via sparse linear probes, to the final generated trajectory in latent space, we quantify and detect a model's tendency to engage in unfaithful behavior. Experiments on Gemma, Llama, and Qwen models using Anthropic's Sycophancy benchmark show that our method achieves an AUROC of 0.55-0.73 for detecting sycophantic runs and 0.55-0.74 for hypocritical cases where the model internally "knows" the user is wrong, consistently outperforming a decision-aligned log-probability baseline (0.41-0.50 AUROC). 

---
# SWE-World: Building Software Engineering Agents in Docker-Free Environments 

**Authors**: Shuang Sun, Huatong Song, Lisheng Huang, Jinhao Jiang, Ran Le, Zhihao Lv, Zongchao Chen, Yiwen Hu, Wenyang Luo, Wayne Xin Zhao, Yang Song, Hongteng Xu, Tao Zhang, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2602.03419)  

**Abstract**: Recent advances in large language models (LLMs) have enabled software engineering agents to tackle complex code modification tasks. Most existing approaches rely on execution feedback from containerized environments, which require dependency-complete setup and physical execution of programs and tests. While effective, this paradigm is resource-intensive and difficult to maintain, substantially complicating agent training and limiting scalability. We propose SWE-World, a Docker-free framework that replaces physical execution environments with a learned surrogate for training and evaluating software engineering agents. SWE-World leverages LLM-based models trained on real agent-environment interaction data to predict intermediate execution outcomes and final test feedback, enabling agents to learn without interacting with physical containerized environments. This design preserves the standard agent-environment interaction loop while eliminating the need for costly environment construction and maintenance during agent optimization and evaluation. Furthermore, because SWE-World can simulate the final evaluation outcomes of candidate trajectories without real submission, it enables selecting the best solution among multiple test-time attempts, thereby facilitating effective test-time scaling (TTS) in software engineering tasks. Experiments on SWE-bench Verified demonstrate that SWE-World raises Qwen2.5-Coder-32B from 6.2\% to 52.0\% via Docker-free SFT, 55.0\% with Docker-free RL, and 68.2\% with further TTS. The code is available at this https URL 

---
# Merging Beyond: Streaming LLM Updates via Activation-Guided Rotations 

**Authors**: Yuxuan Yao, Haonan Sheng, Qingsong Lv, Han Wu, Shuqi Liu, Zehua Liu, Zengyan Liu, Jiahui Gao, Haochen Tan, Xiaojin Fu, Haoli Bai, Hing Cheung So, Zhijiang Guo, Linqi Song  

**Link**: [PDF](https://arxiv.org/pdf/2602.03237)  

**Abstract**: The escalating scale of Large Language Models (LLMs) necessitates efficient adaptation techniques. Model merging has gained prominence for its efficiency and controllability. However, existing merging techniques typically serve as post-hoc refinements or focus on mitigating task interference, often failing to capture the dynamic optimization benefits of supervised fine-tuning (SFT). In this work, we propose Streaming Merging, an innovative model updating paradigm that conceptualizes merging as an iterative optimization process. Central to this paradigm is \textbf{ARM} (\textbf{A}ctivation-guided \textbf{R}otation-aware \textbf{M}erging), a strategy designed to approximate gradient descent dynamics. By treating merging coefficients as learning rates and deriving rotation vectors from activation subspaces, ARM effectively steers parameter updates along data-driven trajectories. Unlike conventional linear interpolation, ARM aligns semantic subspaces to preserve the geometric structure of high-dimensional parameter evolution. Remarkably, ARM requires only early SFT checkpoints and, through iterative merging, surpasses the fully converged SFT model. Experimental results across model scales (1.7B to 14B) and diverse domains (e.g., math, code) demonstrate that ARM can transcend converged checkpoints. Extensive experiments show that ARM provides a scalable and lightweight framework for efficient model adaptation. 

---
# Privately Fine-Tuned LLMs Preserve Temporal Dynamics in Tabular Data 

**Authors**: Lucas Rosenblatt, Peihan Liu, Ryan McKenna, Natalia Ponomareva  

**Link**: [PDF](https://arxiv.org/pdf/2602.02766)  

**Abstract**: Research on differentially private synthetic tabular data has largely focused on independent and identically distributed rows where each record corresponds to a unique individual. This perspective neglects the temporal complexity in longitudinal datasets, such as electronic health records, where a user contributes an entire (sub) table of sequential events. While practitioners might attempt to model such data by flattening user histories into high-dimensional vectors for use with standard marginal-based mechanisms, we demonstrate that this strategy is insufficient. Flattening fails to preserve temporal coherence even when it maintains valid marginal distributions. We introduce PATH, a novel generative framework that treats the full table as the unit of synthesis and leverages the autoregressive capabilities of privately fine-tuned large language models. Extensive evaluations show that PATH effectively captures long-range dependencies that traditional methods miss. Empirically, our method reduces the distributional distance to real trajectories by over 60% and reduces state transition errors by nearly 50% compared to leading marginal mechanisms while achieving similar marginal fidelity. 

---
# Towards Understanding Steering Strength 

**Authors**: Magamed Taimeskhanov, Samuel Vaiter, Damien Garreau  

**Link**: [PDF](https://arxiv.org/pdf/2602.02712)  

**Abstract**: A popular approach to post-training control of large language models (LLMs) is the steering of intermediate latent representations. Namely, identify a well-chosen direction depending on the task at hand and perturbs representations along this direction at inference time. While many propositions exist to pick this direction, considerably less is understood about how to choose the magnitude of the move, whereas its importance is clear: too little and the intended behavior does not emerge, too much and the model's performance degrades beyond repair. In this work, we propose the first theoretical analysis of steering strength. We characterize its effect on next token probability, presence of a concept, and cross-entropy, deriving precise qualitative laws governing these quantities. Our analysis reveals surprising behaviors, including non-monotonic effects of steering strength. We validate our theoretical predictions empirically on eleven language models, ranging from a small GPT architecture to modern models. 

---
# RankSteer: Activation Steering for Pointwise LLM Ranking 

**Authors**: Yumeng Wang, Catherine Chen, Suzan Verberne  

**Link**: [PDF](https://arxiv.org/pdf/2602.03422)  

**Abstract**: Large language models (LLMs) have recently shown strong performance as zero-shot rankers, yet their effectiveness is highly sensitive to prompt formulation, particularly role-play instructions. Prior analyses suggest that role-related signals are encoded along activation channels that are largely separate from query-document representations, raising the possibility of steering ranking behavior directly at the activation level rather than through brittle prompt engineering. In this work, we propose RankSteer, a post-hoc activation steering framework for zero-shot pointwise LLM ranking. We characterize ranking behavior through three disentangled and steerable directions in representation space: a \textbf{decision direction} that maps hidden states to relevance scores, an \textbf{evidence direction} that captures relevance signals not directly exploited by the decision head, and a \textbf{role direction} that modulates model behavior without injecting relevance information. Using projection-based interventions at inference time, RankSteer jointly controls these directions to calibrate ranking behavior without modifying model weights or introducing explicit cross-document comparisons. Experiments on TREC DL 20 and multiple BEIR benchmarks show that RankSteer consistently improves ranking quality using only a small number of anchor queries, demonstrating that substantial ranking capacity remains under-utilized in pointwise LLM rankers. We further provide a geometric analysis revealing that steering improves ranking by stabilizing ranking geometry and reducing dispersion, offering new insight into how LLMs internally represent and calibrate relevance judgments. 

---
