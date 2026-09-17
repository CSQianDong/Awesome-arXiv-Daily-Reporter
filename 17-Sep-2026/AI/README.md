# Cognitive Extensions for Dual-Process Language Agents: Memory and Self-Reflection in Interactive Environments 

**Authors**: João Meneses dos Santos, Arlindo L. Oliveira  

**Link**: [PDF](https://arxiv.org/pdf/2609.19128)  

**Abstract**: Language agents remain brittle in interactive environments, where success requires long-horizon state tracking, valid action execution, and recovery from failed steps. We extend SwiftSage, a dual-process agent that combines a fast action proposer with a slower planner, using two modular cognitive extensions: an Adaptive Memory Module (AMM) for salience-gated episodic storage and trigger-driven retrieval, and a Self-Reflection Module (SRM) for bounded execution-time validation and corrective intervention. Both modules are implemented as feature-flagged extensions over the same execution substrate, enabling controlled ablations on ScienceWorld. Across four configurations---baseline, baseline+AMM, baseline+SRM, and the full system---the full system achieves the best mean final score (64.62), success rate (43.17%), and successful-step efficiency (19.33 steps), while SRM is the strongest standalone contributor. The results suggest that execution-time control is the dominant bottleneck in this setting, while episodic memory becomes most useful once the runtime loop is stabilized. 

---
# Flag Game: A Toy Model for Mechanistic Swarm Interpretability 

**Authors**: Elizabeth Pavlova, Hidenori Tanaka  

**Link**: [PDF](https://arxiv.org/pdf/2609.19124)  

**Abstract**: Emergent coordinated behaviors of AI agents are starting to present critical safety risks. A key phenomenon driving these behaviors is the rapid formation and spread of beliefs about the world, and mechanistic understanding is crucial for collective alignment. To this end, we introduce the Flag Game, a toy model for studying the mechanisms of collective belief formation. Concretely, a hidden country flag defines the ground truth, and each bounded agent directly observes only a private crop but can exchange beliefs and weigh social evidence from peers. Despite its simplicity, the Flag Game reproduces rich collective phenomenology: non-monotonic scaling of performance with population size, accuracy gains from social-awareness prompting and team diversity, and strong effects of organizational structure. In particular, we identify that collective belief collapse at small population sizes turns into collective belief polarization as the population grows. This polarization causes the performance decline at large population sizes, but creates diversity in collective beliefs. Finally, we dissect the mechanisms underlying collective belief collapse and polarization with two complementary approaches. We first introduce social circuit attribution, a technique to predict which agent, and what view, matters most to collective dynamics, and verify its predictions by causal interventions on agents, tracing how agent patching changes collective outcomes. However, the efficacy of causal interventions on agents decreases as the population grows. We therefore develop a statistical mechanical theory for larger populations and verify that it matches the empirical phase diagram. Together, these results take a first step toward mechanistic swarm interpretability, a science of how the properties of individual agents and their communication give rise to emergent collective behavior. 

---
# MUSE: Benchmarking Large Vision-Language Models on Multi-Modal Understanding in Situated Education 

**Authors**: Luyao Zhu, Xun Wei Yee, Wei Li, Mun Thye Mak, Wee Siong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2609.19088)  

**Abstract**: Large vision-language models have achieved remarkable progress in multi-modal understanding, yet their capabilities in educational settings remain insufficiently evaluated. In AI-assisted language learning, models must interpret artistic imagery, understand its semantic, affective, and cultural content, and reason about visual context to support meaningful interaction. However, existing benchmarks primarily focus on real-world images or domain-specific educational reasoning, providing limited coverage of artistic educational content. To address this gap, we introduce MUSE, a benchmark for evaluating large vision-language models on artistic image understanding in situated educational applications. MUSE decouples image annotation from question generation, enabling diverse tasks with controllable difficulty while reducing annotation effort. It comprises twelve tasks spanning visual perception, semantic and affective interpretation, culture understanding, and compositional reasoning, together with diverse artistic images deliberately curated to center Singaporean and Southeast Asian multicultural contexts alongside Western art traditions, covering multiple themes and difficulty levels. Evaluation of open-source and proprietary models reveals substantial disparities across capability dimensions, particularly in affective interpretation and compositional reasoning. Our analysis further identifies common failure modes and key challenges for developing trustworthy multi-modal models for education. We hope MUSE will serve as a standardized benchmark for advancing multi-modal understanding in situated educational applications. 

---
# Compiled Agency: Frontier General-Purpose Coding Agents Build Winning Game Players from Bare Interaction - from Flappy Bird to StarCraft II and Civilization 

**Authors**: Joey Xiao, Haonan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.18996)  

**Abstract**: LLM agents have repeatedly struggled to convert knowledge of a game into competent play, even when researchers build the agent around the model - supplying perception, memory, skill libraries, planners, or executable-policy scaffolds. Rapid progress in coding agents raises two sharper questions: can frontier models now win games at all, and can they win them unaided, building the entire player themselves? We introduce Gauntlet, a develop-freeze-evaluate framework that ports games from small arcades to full commercial-scale titles, behind one deliberately bare contract: a general-purpose coding agent receives a game description, a raw observation/action interface, and an empty policy file - no strategy, no algorithm, no architecture. In a single autonomous session the agent experiments with the live game and engineers a standalone controller; we freeze the result and score it on held-out instances with zero model calls during play. On an unpublished procedural roguelike, held-out success spans 0-86 percent and exposes a sharp generational threshold: every observed session of a newest-generation system outperforms the best session of its predecessor. At full-game scale, a compiled raw-API controller defeats every fair StarCraft II built-in AI and two cheating variants, and single-session programs win complete games of Civilization (Freeciv) by total conquest on held-out seeds. Though at modest rates against novice AI, this is a first: no prior language-agent system had won full games of this genre standalone, without per-turn model calls and a hand-crafted tactical layer. Frontier coding agents begin to track long-horizon strategy. The frozen programs are inspectable. We call this capability compiled agency: development experience compiled into a persistent executable agent whose architecture is built by the model. 

---
# Lost in Perception: Isolating Perceptual and Reasoning Failures in Multimodal Physics and Geometry Reasoning 

**Authors**: Raj Jaiswal, Sree Krishna Uppalapati, Dhruvkumar Patel, Ria Khatoniar, Tanuja Ganu, Rajiv Ratn Shah  

**Link**: [PDF](https://arxiv.org/pdf/2609.18991)  

**Abstract**: Multimodal LLMs report strong performance on scientific reasoning benchmarks, yet most treat perception and reasoning as a single measurable process. We introduce a five-task diagnostic experiment across physics and geometry benchmarks that isolates failures to perception, reasoning, or both. Incorrect diagram interpretation degrades performance even on problems models solve correctly from text alone, and accuracy generally rises from raw images to human-authored captions. Recovery under corrected captions is high for some models, separating perception-blocked failures from genuine reasoning bottlenecks. Which reasoning error follows a perception failure depends on domain: physics failures resolve into calculation errors, geometry into conceptual misapplication. As a discussion beyond our core experiments, InternS1-mini, despite heavy scientific pretraining and thinking capabilities, falls below the weakest model from experiments on every task, with reasoning traces frequently truncating before completion. 

---
# Function Lives Where Variance Doesn't: Task-Weighted Charts of a Language Model's Computation 

**Authors**: Alexandre Quemy  

**Link**: [PDF](https://arxiv.org/pdf/2609.18989)  

**Abstract**: How many dimensions does a language model's computation actually use? The question is ill-posed until one names a functional. Task-weighted charts make it well-posed: low-dimensional coordinate systems fit against a chosen functional of the representation, under the functional's own metric, turning distillation into plain least squares. Across six models from three families, spanning 70m to 7B parameters, next-token prediction needs 70--90% of the residual stream's width to stay within 5% of intact perplexity, a width consumed by the rare tail of language, and the variance profile predicts none of it: two directions carry 90% of GPT-2's activation variance and almost none of its function. Dimension is per-functional: the model's own uncertainty reads from six coordinates where the full predictive distribution needs hundreds; and it grows with depth. The dissociation is exploitable: when only a few dimensions can be kept, charts trained under the functional's metric preserve the model's predictions better than variance-based or optimal linear compression. 

---
# Suppressed, Not Erased: A Representational Trace of Edited Facts Survives Even Weight-Free Knowledge Editing 

**Authors**: Priyansh Srivastava, Romit Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2609.18985)  

**Abstract**: Knowledge-editing benchmarks certify local correctness, whether an edited model produces the new fact on near-edit prompts but not how much of the original fact remains decodable inside the model. We study residual knowledge directly with a linear trace probe: after editing a fact, we ask whether the original object is still recoverable from the model's hidden states. On GPT-2-XL, across three mechanistically distinct editors applied to 50 CounterFact edits, the original object remains linearly decodable well above chance after a successful edit (probe accuracy 0.96 for ROME, 0.86 for constrained fine-tuning, and 0.79 for the memory-based editor GRACE, against a chance level of 0.50; all edits reach 100% generation-based success). The GRACE result is the most informative: GRACE changes zero base-model weights, overriding the fact through an external memory, yet the original object is still decodable from the underlying network, so the residual trace cannot be attributed to an incomplete weight update. We read this as evidence that editing, even when behaviorally successful, suppresses rather than erases the original association in representational space. We also report a relearning-savings instrument that did not behave reliably in our setting and discuss why; we treat it as a negative methodological result rather than evidence. Code and data are released. 

---
# Infinite-Parameter LLMs: Generating and Adapting Weights from Live Data 

**Authors**: Jinli Hu, Ross M. Clarke, Yichuan Zhang, José Miguel Hernández-Lobato  

**Link**: [PDF](https://arxiv.org/pdf/2609.18842)  

**Abstract**: The scaling laws hold that a language model grows more capable with more parameters and more training data, and Mixture-of-Experts (MoE) architectures have ridden these laws to remarkable results, activating only a fraction of an enormous stored parameter bank for each token. That success is built on static pretraining data. A deployed model faces a different world, where much of the data that would make it more useful is not in its training set but in the live interaction it is currently handling, such as the facts a user supplies or the corrections they give. A conventional model cannot learn from this data, because its weights are frozen after training. Instead, the knowledge and behaviour supplied at run time are placed in the prompt, by retrieval or instruction, and re-read on every request only to be discarded once the request ends. We ask how an architecture could learn from live interaction by writing it into its weights. Taking inspiration from MoE, we propose the \textbf{Infinite-Parameter LLM}. A compact hypernetwork turns the data given at run time into a low-rank modulation of a shared base network, so the feed-forward weights are generated from live data rather than stored in a fixed bank. Where prior weight generators read the context once and freeze, we carry a Bayesian belief over the generator's latent code and update it online, so the effective weight is re-derived from that evolving belief as the session proceeds rather than fixed after one read. The stored footprint stays fixed, yet the weights the model can compile are effectively infinite. For the knowledge and behaviour supplied at run time, carrying them in the weights rather than the prompt is amortized in compute, frees the context window, persists across turns, and can generalise better than in-context use. We specify an evaluation protocol that tests exactly this against in-context learning and retrieval. 

---
# Compositional Policy Violations: When Step-Level Compliance Fails In Agentic AI Workflows 

**Authors**: Ashwini Kurady, Sri Sai Charith Grandhi, Rajesh Gupta, Sumit Mamoria  

**Link**: [PDF](https://arxiv.org/pdf/2609.18820)  

**Abstract**: Agentic workflows now make consequential decisions in regulated settings, and the governance placed around them is almost entirely step-scoped: input-output classifiers, per turn rails, and span-level evaluators. The policies organizations actually hold, such as referral thresholds, authority limits, and review requirements, are properties of the whole execution rather than of any one step. This mismatch admits a failure mode we call a Compositional Policy Violation (CPV): every individual step passes its own check while the composed execution violates the governing policy. A predicate over a single step cannot evaluate a property that step does not determine, so no improvement in the accuracy of the step-scoped monitors detects this class. We define CPVs as the failure of step-level compliance to compose, and present a taxonomy of four types: Authority Creep, Threshold Laundering, Cumulative Sum Violation, and Context Collapse. We show that the correct repair for each class is dictated by where the guarded quantity mutates. We then introduce a provenance-aware runtime architecture that evaluates policies over complete execution traces, recomputing guarded quantities from raw provenance rather than the pipeline's derived representation. 

---
# CERA-MoA: Co-Evolving Routing Mechanisms with Continually Learning LLM Agents 

**Authors**: Jiaxuan Jiang, Liyuan He, Zhixuan Fang  

**Link**: [PDF](https://arxiv.org/pdf/2609.18779)  

**Abstract**: Current Mixture-of-Agents (MoA) paradigms generally treat query routing and agent fine-tuning as separate processes, limiting their ability to respond to evolving agent capabilities. This disconnect prevents routing strategies from adapting to evolving agent capabilities during post-training and prevents agents from achieving synergistic data-driven specialization. To resolve this, we introduce CERA-MoA (Co-Evolving Router with continually learning Agents for Mixture-of-Agents), an iterative reinforcement learning framework where the dynamic router and independent agent policies co-evolve. We design a predictive familiarity estimator that leverages mid-layer hidden states to evaluate semantic competence among agents, avoiding the overhead of full rollouts. Based on these familiarity scores, a cumulative-threshold adaptive routing mechanism dynamically activates a tailored minimal agent subset, achieving a trade-off between task performance and efficiency. By proactively allocating targeted training samples to agents based on their evolving competence, CERA-MoA promotes capability differentiation. Extensive experiments across various domains demonstrate that CERA-MoA outperforms state-of-the-art static-agent routing and fix-workflow fine-tuning baselines. 

---
# Version- and Scope-Aware Question Answering over Normative Documents: A Deployed System and an End-to-End Evaluation at Production Scale 

**Authors**: Liuyin Wang, Shuaipeng Jin, Jiwei Shi, Jensen Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2609.18769)  

**Abstract**: Correctly answering a question grounded in normative documents often depends on information outside any single passage: whether the retrieved document is the version currently in force; whether it applies to the jurisdiction, subject (such as an institution or applicant), and date at issue; and whether each normative claim can be traced to its supporting source text. Hosted retrieval services have substantially lowered the engineering cost of building an initial system over such corpora, making "upload the documents and ask" a common default. We evaluate this default on approximately 73,000 candidate normative documents supplied to a production deployment. The evaluation uses a stratified sample of 200 questions from our published benchmark, with a gold source document for every question; the released sampling rule reads no system outputs or scores. We compare the hosted service with a governed system that resolves version and scope through explicit rules before generation. The governed system scored 97.7 overall, while the hosted service scored 88.1, a gap of 9.6 points computed from unrounded means. The question set, the answer text evaluated for both systems, the scores, and the scripts used to reproduce the reported benchmark statistics are public. The governed configuration has operated as a commercial product since January 2026 and serves 1,126 registered users; named customer organizations include Zhipu AI and Lecheng Health. By mid-April 2026, it had reached roughly 100,000 calls per workday. 

---
# Clueing up LLMs with Tool-Augmented Deductive Reasoning 

**Authors**: Rebecca Ansell, Autumn Toney-Wails  

**Link**: [PDF](https://arxiv.org/pdf/2609.18736)  

**Abstract**: Despite recent advances in large language models (LLMs), performing logically consistent deductive reasoning over extended interactions remains challenging. Tasks that require integrating evidence across multiple reasoning steps, maintaining consistency with prior inferences, and updating beliefs under new constraints can surface limitations in current models while providing a useful testbed for evaluating reasoning enhancements. In this paper, we implement a text-based, multi-agent version of the classic board game Clue as an environment to evaluate multi-step, agentic deductive reasoning. In this setting, agents must infer hidden information from a sequence of observations, maintain consistency across turns, and reason over an evolving set of logical constraints. We instantiate six LLM-based agents (GPT-4o-mini and Gemini-2.5-Flash) as players that engage in turn-based gameplay; using three agents per model family, we establish baseline performance across repeated games. We then introduce a tool-augmented approach in which a structured possibility matrix converts implicit game state from generated reasoning logs into an explicit representation of remaining possibilities. The possibility matrix encodes extended-turn memory and deductive constraints, offloading these tasks from the agent. We compare this approach against the baseline to evaluate how tool augmentation supports reasoning quality and task success for autonomous agents in a strategic reasoning environment. 

---
# Which LLM is Best for Translating Natural Language Goals to PDDL 

**Authors**: Tomas Balyo, Lukas Chrpa, G. Michael Youngblood  

**Link**: [PDF](https://arxiv.org/pdf/2609.18731)  

**Abstract**: Bridging the gap between human intent and machine execution remains a challenge in automated planning, where expressing goals in formal languages like PDDL restricts accessibility to non-experts. This paper empirically evaluates whether current Large Language Models (LLMs) can reliably translate natural language testing goals, written in informal language by video game testers, into well-formed PDDL targets suitable for classical planning. We present a carefully designed prompt template, integrating insights from iterative experimentation, aimed at maximizing both accuracy and response coherence from multiple state-of-the-art LLMs. Six contemporary models are systematically assessed on correctness, speed, and error tendencies using real-world, domain-specific benchmarks. All models demonstrate high correctness, exceeding 92\%, with Gemini 2.5 Flash achieving the highest accuracy at 96\% and the lowest incidence of false positives, while GPT-4.1 leads in response speed. Despite these advances, critical distinctions exist in model performance, and occasional failures arise from language ambiguity and limitations in domain representation. Our analysis underscores both the significant progress and ongoing gaps in enabling LLMs to act as robust bridges between natural language objectives and automated planning pipelines. 

---
# Beyond Truncation: Rethinking LLM Decoding as Ensemble Pruning 

**Authors**: Dunyao Xue, Chengshuo Du, Zhengbo Wang, Wenlin Dai, Cheng Meng  

**Link**: [PDF](https://arxiv.org/pdf/2609.18723)  

**Abstract**: We introduce Mahalanobis-Ensemble Decoding (ME-Decoding), a novel Large Language Model (LLM) decoding framework that frames candidate token selection as ensemble pruning. Existing selection strategies rely predominantly on scalar probabilities, ignoring geometric semantic relationships and causing candidate redundancy. Meanwhile, current geometry-aware methods often require complex optimization or directly reweighting the original token probabilities, leading to significant computational overhead or inference instability. To address this, we formulate decoding as a subset optimization problem using a Mahalanobis distance-driven objective to enhance semantic diversity while preserving high probabilities. Specifically, we dynamically discount redundant generation paths using a token similarity matrix, constructed via an adaptive-bandwidth kernel over token embeddings. We further devise an efficient greedy selection algorithm with near-linear complexity in the candidate size under early stopping, while establishing its theoretical approximation guarantees. This renders ME-Decoding a robust, plug-and-play module with negligible inference overhead. Extensive experiments across diverse reasoning and generation tasks demonstrate that our method consistently achieves strong performance. 

---
# The Uneven Impact of Generative AI on Student Learning: Examining the Roles of Reliance, Evaluation Literacy, and Course Policy in AI-related Courses 

**Authors**: Lydia Manikonda, Mei Si, Sirajam Munira, Oshani Seneviratne, Kristin Bennett  

**Link**: [PDF](https://arxiv.org/pdf/2609.18676)  

**Abstract**: Generative artificial intelligence (GenAI) is changing how students learn, yet the roles of course context, cognitive reliance, evaluation literacy, and early reliance remain underexplored. Using survey responses from 118 students across 12 AI-related courses at our institution, we examined differences in GenAI use and perceived learning experiences. We identified four user clusters: high-use students reporting many benefits, light users reporting less reliance and fewer benefits, and two moderate-use groups reporting different levels of benefit. We also found significant differences between free- and premium-version users, single- and multiple-tool users, and students experiencing different instructor policies. In multivariable regression models, academic benefit was associated with early reliance and academic task support; positive impact was associated with cognitive reliance, academic task support, confidence in GenAI reliability, and instructor policy; and negative impact was associated with early reliance and attitudinal change. The association between early reliance and negative impact became stronger as evaluation literacy increased. Finally, perceptions of GenAI-enhanced learning appear to reflect cognitive, performance, and self-efficacy benefits, while concerns about stress and diminished critical thinking are associated with lower perceived learning benefits. These findings suggest that institutions need better policies to address such inequities so that institutions can enable students to benefit from increasingly capable AI systems. 

---
# Reasoning through Evolution: Automatic Meta-path Discovery for LLM-based Fake News Detection 

**Authors**: Ziyi Zhou, Xiaoming Zhang, Hui Pang, Yuting Zhang, Tiesunlong Shen, Bingyu Yan, Erik Cambria, Litian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.18597)  

**Abstract**: Propagation structures provide crucial evidence for fake news detection, yet existing approaches primarily rely on supervised GNN-based models, which require substantial labeled data and exhibit limited generalization. Although large language models (LLMs) exhibit strong reasoning capabilities, directly feeding them raw propagation graphs creates a significant modality mismatch and severe information overload, making structure-aware reasoning unreliable in zero-shot and few-shot settings. To bridge this gap, we propose MAGER, a multi-agent genetic evolution framework that automatically discovers meta-paths optimized for LLM reasoning. By compressing complex propagation graphs into informative subgraphs, the evolved meta-paths alleviate both information overload and modality mismatch, enabling frozen LLMs to perform structure-aware veracity reasoning. We further introduce a graph in-context learning strategy that retrieves semantically and structurally similar demonstrations to strengthen classification and reasoning. Extensive experiments show that MAGER substantially improves frozen LLMs as standalone fake news detectors in data-efficient settings. Our code is available at this https URL. 

---
# Recursive Reasoning or Statistical Extrapolation? In-Context Learning in Multi-Agent Interdependent Decision-Making 

**Authors**: Yu Liu, Wenwen Li, Yifan Dou, Guangnan Ye  

**Link**: [PDF](https://arxiv.org/pdf/2609.18591)  

**Abstract**: In-context learning (ICL) enables large language model (LLM) agents to improve decisions using interaction history, yet it remains unclear whether such improvement reflects refined internal reasoning or mere extrapolation of statistical patterns. To disentangle these mechanisms, we study LLM agents in multi-agent incomplete-information games that require recursive belief reasoning. By constructing a public goods game and manipulating the statistical structure of historical feedback, we evaluate decision quality against a history-independent rational expectations equilibrium (REE) benchmark. Our experiments reveal that when historical statistical patterns are disrupted, the benefits of longer context largely vanish, degrading decision quality to the no-context baseline in a way sharply amplified by stronger strategic interdependence. These results suggest that, in such strategic environments, ICL behavior is more consistent with statistical extrapolation than with strategic reasoning. Our work extends the mechanistic study of ICL to strategic multi-agent settings, introduces REE as a diagnostic tool for distinguishing reasoning from extrapolation, and provides a reusable framework for probing the boundaries of LLM reasoning in recursive belief tasks. 

---
# TRIPROBE: Probing Task Separability Beyond Classification for XAI 

**Authors**: Amirhossein Sadough, Freek Hens, Aleksa Bokšan, Mohammad Mahdi Dehshibi, Mahyar Shahsavari  

**Link**: [PDF](https://arxiv.org/pdf/2609.18525)  

**Abstract**: Modern evaluation of learning pipelines often reduces to downstream accuracy, leaving open the question of why tasks succeed or fail. TriProbe addresses this gap with a multi-level probing framework for explainable diagnosis of task separability. Rather than treating models as black boxes, TriProbe traces how separability evolves across inputs, learned features, and final classifiers. It decomposes multi-task problems into binary subtasks and applies three complementary probes: a Foundational Probe on input spaces, a Latent Probe on feature representations, and a Final Probe on classifier outputs. Using Maximum Fisher's Discriminant Ratio as a principled separability metric, TriProbe identifies bottlenecks and affected task pairs. Experiments on the Roshambo sEMG benchmark show how TriProbe reveals hidden breakdowns, guiding data collection, validation, and architecture design. 

---
# AeroWeaver: An Embodied-Agent Harness for Weaving Aerial Skills into Distributed, Adaptive Swarm Execution 

**Authors**: Jiabin Lou, Yirong Yang, Haopeng Wang, Xuxin Lv, Xinyu Liu, Diyuan Hou, Xuehong Liu, Rongye Shi, Wenjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2609.18520)  

**Abstract**: Collective intelligence is a collaborative autonomy paradigm in which multiple agents pursue shared objectives through local perception, information exchange, and coordinated action. UAV swarms embody this paradigm by coordinating multiple vehicles in tasks such as search, inspection, and tracking. Recent advances in large language model (LLM) agents have strengthened natural-language task understanding and high-level planning, providing a flexible semantic interface between mission descriptions and collective behavior. While these advances expand semantic reasoning, applying LLM agents to UAV swarms raises challenges in grounding model decisions in executable capabilities, reconciling global task reasoning with distributed execution, and using mission-specific experience for continual adaptation. To address these challenges, we introduce AeroWeaver, an embodied-agent harness that weaves individual UAV skills into coordinated mission-level behavior. AeroWeaver connects semantic decisions to governed skills, organizes role-conditioned local agents for distributed coordination, and uses role-indexed state-action-reward experience to refine skill selection online. Experiments and runtime validation show that AeroWeaver maintains valid skill execution under tested conditions and supports body-local multi-UAV operation without a central agent generating joint actions from global context, while reward-guided online updates provide a training-free path for adaptive learning swarm agents from accumulated execution experience. Code: this https URL. 

---
# Beyond Routine Compliance: Cunning Data Cultivates Safety Vigilance in Large Language Models 

**Authors**: Youjia Wang, Lin Xu, Yang Sun, Yuxiao Lu, Chengfang Fang, Jie Shi  

**Link**: [PDF](https://arxiv.org/pdf/2609.18515)  

**Abstract**: Safety alignment teaches large language models (LLMs) to recognize harmful requests and reject risky instructions. Yet aligned models can fail when harmful intent is concealed within seemingly benign contexts. Robust safety therefore requires both knowledge of safety boundaries and \textbf{vigilance}: the ability to detect unusual premises, misleading reasoning, and latent risks beneath surface-level semantics. Vigilance requires models to scrutinize a request's underlying intent and assumptions before acting. To cultivate this capability, we introduce \textbf{cunning questions}, which are not necessarily safety-related but contain misleading premises, atypical reasoning, or subtle inconsistencies. We hypothesize that learning to look beyond such reasoning traps can transfer to safety-critical scenarios. Experiments show that Cunning training improves robustness to out-of-distribution jailbreak attacks and strengthens subsequent safety fine-tuning. Furthermore, augmenting an existing state-of-the-art safety alignment pipeline with Cunning establishes a new state of the art across our evaluated settings, reducing mean ASR across nine backbone--benchmark combinations from 17.40\% to 15.05\%. Trace analysis after matched safety fine-tuning suggests that safety judgments are more likely to govern responses before harmful planning begins. A conditional theoretical analysis further characterizes when invariance learned from cunning data can transfer to safety-related inputs. These findings suggest that cunning data can strengthen model vigilance and complement conventional safety alignment. 

---
# Hyperbolic Graph Representation Learning for Differential Diagnosis on Biomedical Knowledge Graphs 

**Authors**: Pietro Miotto, Lucia Mellini, Tommaso Marzi, Cesare Alippi, Elena Casiraghi, Alberto Paccanaro, Giorgio Valentini, Mauricio Soto-Gomez  

**Link**: [PDF](https://arxiv.org/pdf/2609.18481)  

**Abstract**: Biomedical knowledge graphs combine ontology-derived hierarchies with transversal associations among heterogeneous entities such as phenotypes, diseases, genes, proteins, and patients. This hybrid structure raises the question of whether hyperbolic embeddings, which naturally capture tree-like organization, remain useful beyond purely hierarchical graphs. We present a preliminary study of hyperbolic graph representation learning for Mendelian-disease differential diagnosis on a patient-integrated biomedical graph. Experiments on isolated ontology subgraphs show that hyperbolic models achieve strong performance in substantially lower dimensions than Euclidean baselines. We then evaluate the models on a link-prediction task that ranks candidate diseases for each patient. Results suggest that hyperbolic embeddings can exploit biomedical hierarchical structure while supporting diagnostic reasoning over heterogeneous patient-level graphs. 

---
# First Token Matters: Understanding Safety Collapse in Large Reasoning Models 

**Authors**: Yizheng Yang, Haining Yu, Yuechen Wang, Yikai Hou, Xing Fu, Jinbo Yang, Tianqing Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2609.18471)  

**Abstract**: Large Reasoning Models (LRMs) exhibit strong problem-solving abilities, yet their safety alignment often degrades when handling harmful queries. Existing approaches to improving safety largely rely on additional training or preference optimization, while offering limited understanding of the internal mechanisms behind safety failures. In this work, we investigate this failure through a token-level positional analysis of refusal dynamics and identify a localized vulnerability at the onset of reasoning, which we term Onset Refusal Collapse (ORC). We find that the refusal-related signal of LRMs drops sharply at the first generated token under harmful queries, which is associated with unsafe response generation. Motivated by this finding, we propose SafeToken, a lightweight inference-time intervention that injects a learned continuous safety anchor precisely at reasoning onset. Despite updating only a single token embedding, SafeToken effectively mitigates ORC, improves safety on harmful-query benchmarks, and largely preserves reasoning utility. These results suggest that safety failures in LRMs can arise from a transient breakdown at the critical transition from understanding to generation. 

---
# Disentangling Long-Term Memory via Latent Neuro-Symbolic Reasoning 

**Authors**: Cai Ke, Xinghao Chen, Xiaoyu Shen, Keyu Chen, Siyu An, Junnan Dong, Ruifeng Xu, Ruizhi Qiao, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2609.18461)  

**Abstract**: Personalized agents are required to reason over long-term history interactions to infer both explicit preferences and implicit behavioral evidence. While early flat retrieval methods score memory fragments independently and neglect the distributed information, current structured memory frameworks rely on query-agnostic static graphs that fail to capture the context-dependent relations. Crucially, raw textual memories are inherently entangled and noisy, making fine-grained personalization and cross-session reasoning computationally prohibitive. To this end, we present LGM, a novel neuro-symbolic framework that shifts long-term memory disentanglement into a continuous latent space. Specifically, (i) instead of persisting fixed graphs, we design a tailored latent graph construction with a sparse autoencoder. Subject to each query, it maps historical interactions into latent memory nodes and disentangles the memory traces into sparse concept activations, dynamically synthesizing query-aware relational edge weights. (ii) A graph encoder then treats the query embedding as a conditioning preference to direct non-linear message passing across the task-specific latent subgraph. This yields a highly expressive memory representation for effective activations. Extensive experiments on long-term personalization benchmarks demonstrate that LGM significantly outperforms state-of-the-art baselines in capturing both explicit and implicit preferences while enabling personalized responses. 

---
# Collective Loss of Control in LLM Agent Systems: An Epidemic Account of Mutation, Contagion, and Recovery 

**Authors**: Xiangfan Wu, Zonghao Ying, Huiyu Wu, Xing Zheng, Huangsheng Cheng, Xiaorong Shi, Jing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2609.18460)  

**Abstract**: How does a multi-agent system evolve from a local deviation into collective loss of control? We propose an epidemic explanation organized around accidental mutation, contagion, and recovery. A spontaneous deviation creates a seed; communication enables other agents to adopt and retransmit its unsafe strategy; collective failure can emerge when propagation outpaces correction and containment. Thus, rare individual deviations can coexist with substantial collective risk. Motivated by reported OpenAI agent coordination incidents, we examine two ingredients of this mechanism. A deployment audit identifies implicit communication paths between nominally independent evaluation runs and verifies transport through a default Docker backend. RogueHandoff-20, a benchmark of 20 executable scenarios, tests recipient susceptibility by injecting unsafe trajectories generated by a modified Qwen-27B route. Across four native-pending routes, executed harm is 0-5% on normal tasks and 40-95% after injection, exceeding paired direct malicious requests by 5-45 percentage points. These results support low observed baseline harm alongside high conditional susceptibility; they do not establish natural rare-event rates or demonstrate an autonomous cascade. The account motivates complementary defenses: strengthen resistance and recovery alongside prevention of spontaneous deviations, and audit and restrict unintended communication paths that can turn local failures into collective loss of control. 

---
# The Mirage of Calibrated Confidence: Trajectory-Independence of Verbalized Confidence in Vision-Language Models 

**Authors**: Jisoo Yang, Jaeho Han, Trung X. Pham, Junyeong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2609.18453)  

**Abstract**: A calibrated Vision-Language Model (VLM) can repeatedly self-correct, say "Wait, I should recheck," arrive at the wrong answer, and still report high confidence. We find that this occurs because verbalized confidence is largely trajectory-independent in the VLMs and calibration methods we evaluate. We examine this through three complementary lenses: content variation, token masking, and the model's own hesitation markers. We show that confidence is insufficiently sensitive to what the reasoning trajectory actually contains, and that calibration training can paradoxically worsen this disconnect. Since existing metrics like ECE and AUROC cannot detect this problem, we propose the Trajectory-Grounding Score (TGS) in two complementary forms: TGS-self, which compares confidence with and without access to the model's own trajectory, and TGS-pair, which tests whether the model assigns higher confidence to correct trajectories than to flawed ones along the vision, reasoning, and answer axes. We propose TGS-Bench, a model-agnostic suite spanning 10 benchmarks with controlled good/bad trajectory pairs, and show that conventional calibration rankings diverge from trajectory-grounding rankings, exposing a blind spot in current evaluation practice. 

---
# Risk-Aware World Modeling with Flow-Guided Occupancy Evolution for Selective Trajectory Planning in Automated Driving 

**Authors**: Rongxiang Zeng, Linsen Cai, Jiafu Zhang, Yijie Zhong, Yide Tao, Shuai Wang, Nan Zheng, Hai L. Vu, Alvaro Garcia Hernandez, Yongqi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2609.18442)  

**Abstract**: Safe motion planning in automated driving requires anticipating evolving traffic risks and deciding when to revise the current planned trajectory. We introduce RiskWorld, a risk-aware world modeling framework for shared occupancy forecasting and selective trajectory replacement. Spatial risk fields and temporal actor context are fused with visual bird's-eye-view features. Flow-guided evolution transports occupancy and scene features, while signed residuals correct occupancy after transport. One forecast is generated per planning step and reused across candidates. Each candidate is compared with a current-state persistence reference, yielding a nonnegative collision-score correction. The trajectory selected by current-world evaluation serves as the planning anchor and is replaced only when additional predicted risk triggers intervention and an alternative satisfies component-wise constraints on predicted risk and trajectory error. Candidate geometries remain unchanged. We evaluate RiskWorld for open-loop planning on nuScenes using camera features, annotation-derived current and historical actor states, and dataset-provided map context. RiskWorld achieves the lowest collision rate at a long evaluation horizon of 3 s, and the second-best average L2 error among various state-of-the-art baselines, while running at 11.5 FPS on a single NVIDIA RTX 4090 with 90.81 M parameters. Within-setting ablations show that RiskWorld achieves lower collision rates than the current-state rescoring baseline, while forecast reuse enables additional candidates to be evaluated at low marginal computational cost. 

---
# WetRobo: A Reproducible Robot Kit for Coding Agents in Biological Laboratories 

**Authors**: Yuna Oikawa, Kei Endo, Takanori Uzawa, Yunzhe Zhang, Manan Anjaria, Lerrel Pinto, Sherry Yang, Koji Tsuda  

**Link**: [PDF](https://arxiv.org/pdf/2609.18435)  

**Abstract**: Automating biological research requires general-purpose, reproducible robot systems that allow individual wet-lab researchers to delegate robot tasks without performing teleoperation or neural-network training. Vision-language-action policies have been proposed for general-purpose arms, but can lose performance when their operating environment changes. We therefore built WetRobo, a robot kit that can readily transfer between laboratories. It consists of one robot arm, laboratory equipment (an incubator, a reagent bottle with a cap, and a Petri dish), the existing code that moves the arm, teleoperation demonstrations of each task that we recorded, and a general this http URL skill file. A biological experimentalist provides natural-language tasks without collecting local teleoperation training data or training a neural network. The coding agent observes the local laboratory and writes and executes programs, using external tools as needed for adaptation. We demonstrate use of WetRobo with OpenAI Codex (gpt-5.6-sol) on three successful tasks: lifting a Petri dish lid, removing a bottle cap, and opening the incubator door, all in real-world laboratories. The coding agent achieved the cap task in both laboratories, Lab X and Lab Y, whereas a VLA fine-tuned on Lab X demonstrations succeeded there but failed to transfer to Lab Y. These results point to a practical route for laboratory robotics: instead of training a policy for each laboratory, distribute a kit and let a coding agent adapt it in each laboratory. Code, demonstrations, and the evolved programs are available at this https URL. 

---
# HPOQuest: A Rare-Disease Diagnostic Agent Using Active Phenotype Acquisition 

**Authors**: Kamilia Zaripova, Nassir Navab, Azade Farshad, Annalisa Marsico  

**Link**: [PDF](https://arxiv.org/pdf/2609.18431)  

**Abstract**: More than 300 million people worldwide are affected by one of over 7,000 known rare diseases, yet diagnosis remains difficult because patients initially present with incomplete and heterogeneous phenotypes. We present HPOQuest, a training-free framework for sequential phenotype acquisition in rare-disease diagnosis. Starting from a small set of observed patient phenotypes, HPOQuest maintains a probabilistic disease ranking and iteratively selects informative follow-up questions to support clinicians during patient assessment. Confirmed phenotypes update the disease ranking, while all responses update the candidate question set. Across four benchmark cohorts, HPOQuest substantially improves diagnosis from sparse initial phenotypes, with gains of up to 30% points at Recall@1 and 45% points at Recall@5. These results demonstrate that sequential phenotype acquisition can substantially improve rare-disease diagnosis from limited initial clinical evidence. 

---
# Cultural Competence in Context: A Large Language Model Passes the Turing Test in Finland 

**Authors**: Otto Segersven, Pentti Henttonen  

**Link**: [PDF](https://arxiv.org/pdf/2609.18394)  

**Abstract**: We report the results of a Turing Test conducted in Finland in the Finnish language. Because languages and cultural contexts are unevenly represented in LLM training data, we expected the model (ChatGPT 5.2) to perform worse in a Finnish-language Turing Test than in previously studied English-language US contexts. We also present model-generated role prompting as a replicable technique for conducting comparative LLM-based Turing Tests designed to improve construct validity. Contrary to our expectations, the LLM passed the Finnish Turing Test. A prominent source of error was participants' reliance on linguistic cues, particularly colloquial Finnish, as markers of human authorship. We reframe the Turing Test from a test of intelligence to a comparative method for examining whether an AI system can display credible membership in a particular social world. Because its outcome reflects model capabilities, prompted identity, insider competence among human participants, and their AI literacy, the method provides a useful probe of the human-machine boundary across domains. 

---
# Bad Genius: Counterfactual-Guided Harness Evolution Beyond Task-Specific Shortcuts 

**Authors**: Guojun Zhu, Xunheng Huang, Peng Yin, Jiahui Xie, Sanguo Zhang, Doudou Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2609.18366)  

**Abstract**: Reliable agent evaluation is complicated by automatic harness optimization, which repeatedly uses a released benchmark $B_{\mathrm{rel}}$ to guide a Proposer that edits prompts, memory, retrieval, tools, and control code around a fixed target agent. Task holdout varies semantic tasks but leaves the benchmark protocol fixed, so a "bad genius" Proposer can produce a cheating harness whose released-benchmark gain depends on a benchmark-wide shortcut. We introduce Counterfactual Harness Search and Evolution (CHASE), which casts harness evolution as constraint generation over validity-preserving benchmark counterfactuals. After each Proposer update, a Challenger searches for an executable protocol transformation with large gain destruction. A validity firewall checks that task semantics are preserved, while a confirmation set determines whether the counterfactual enters a finite archive. We formalize an exact shortcut-neutralized benchmark $B_0$ and establish statistical guarantees linking finite counterfactual archives to $B_0$ and characterizing sequential Challenger search. We evaluate CHASE on a synthetic benchmark and on OfficeQA, where CHASE retains strong released-benchmark gains while substantially reducing gain destruction under valid protocol changes. 

---
# Market Signal Injection: Adversarial Context Manipulation of LLM Pricing Agents 

**Authors**: Dohun Lee, Hyunwoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2609.18357)  

**Abstract**: Large language model (LLM) pricing agents may respond to how market data is presented, even when its numerical values remain unchanged. We introduce market signal injection (MSI), an attack that manipulates numerical formatting, competitor ordering, or qualitative market commentary without issuing explicit instructions. We evaluate nine open-weight models in simulated Bertrand duopoly and triopoly markets and three proprietary models in duopoly markets. Sentiment-based attacks produce the largest behavioral shifts, which propagate to other firms and alter profits and consumer surplus. Susceptibility varies across model families, and larger models are not consistently more robust. Matched neutral-text controls and a rule-based agent support a framing-based account of these shifts under the fixed demand parameters of our simulation. Episode-held-out probes distinguish baseline from attacked activations in all eleven re-evaluated model--condition pairs: linear AUC is 1.00 and MLP AUC ranges from 0.93 to 0.99. This separability does not by itself identify harmful pricing decisions. Input canonicalization removes the tested sentiment attacks, while decision boundary anchoring, which combines prompt constraints with output projection, provides partial mitigation under the tested adaptive attacks. These results identify data presentation as an attack surface for LLM pricing agents and motivate defenses that account for interactions among agents. 

---
# Faithful yet Collusive: Why Chain-of-Thought Monitoring Cannot Detect Collusion in LLM Pricing Agents under Oligopolistic Competition 

**Authors**: Dohun Lee, Hyunwoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2609.18346)  

**Abstract**: Large language models (LLM) deployed as autonomous pricing agents may sustain supracompetitive prices through tacit coordination. We develop a causal graph divergence framework that separately measures structural faithfulness and intent faithfulness of LLM pricing agents in Bertrand competition. Across nine LLMs under duopoly and triopoly conditions, collusive behavior and chain-of-thought (CoT) faithfulness dissociate along both dimensions: the most collusive model accurately reports cooperative intent yet reasons structurally unfaithfully, while the most structurally faithful model sustains supra-Nash pricing under both market structures. These findings establish that CoT monitoring alone cannot serve as a standalone safeguard against algorithmic collusion. 

---
# Visual Compliance via Executable Safety Rule Entailment 

**Authors**: Jisoo Kim, TaeYoon Kwack, Jinwoo Jang, Honguk Woo  

**Link**: [PDF](https://arxiv.org/pdf/2609.18328)  

**Abstract**: Recent advances in LLMs and VLMs have enabled safety systems to reason beyond simple risk patterns toward more contextual and semantic safety concerns. However, as risk patterns continue to evolve and safety rules become more complex, existing training-based end-to-end safeguards face persistent challenges in adaptability and explainable reasoning over complex safety rules. To address these challenges, we propose GuardEn (Guarding by Safety Rule Entailment), an executable safeguard framework that decomposes safety policies into atomic propositions through Safety-Rule Compilation, modeling their composition as executable code. At test time, Scene-Grounded Execution instantiates these atomic propositions with contextual visual information derived from scene graphs, enabling rule-grounded and interpretable safety reasoning. Experiments on SafetyVisionBench demonstrate the effectiveness of programmable safeguard for complex visual safety assessment, achieving an average improvement of 9.8 F1 points over the strongest baseline. 

---
# What Counts as Strategic Reasoning? A Systematic Mapping of Chess Research on Humans, Engines, and Language Models 

**Authors**: Paolo Ciancarini, Remo Pareschi  

**Link**: [PDF](https://arxiv.org/pdf/2609.18286)  

**Abstract**: Chess has long served as a model domain for studying search, expertise, decision-making, and artificial intelligence. The emergence of large language models (LLMs) has renewed the relevance of chess as a controlled environment for investigating strategic reasoning and comparing human and artificial decision-making. We present a systematic mapping study of recent research spanning human players, classical chess engines, neural and reinforcement-learning systems, LLMs, and hybrid approaches. The final map comprises 84 core study families, classified according to agent type, strategic-reasoning stages, and evaluation dimensions.
The map reveals a literature strongly concentrated on situation assessment, evaluation, and action selection, while explicit planning, explanation, metacognition, and human--AI collaboration remain less explored. LLM research places particular emphasis on state representation and generalization, whereas grounded explanation appears more frequently in hybrid approaches combining language models with engines, expert knowledge, or other external structures. Two distinctions emerge that the map aggregates rather than resolves: hybrid systems differ in where and when heterogeneous capabilities combine, and evaluations that show improved human performance do not thereby establish human--AI synergy. We propose both as extensions of the mapping framework.
We argue that chess provides a useful bridge between cognitive and computational perspectives on strategic reasoning, and identify explicit planning, grounded and faithful explanation, metacognitive calibration, and human--AI complementarity as directions for future research. 

---
# Where Should Agents Live? Energy-Memory Characterization of Agentic AI for the Edge-Cloud Continuum 

**Authors**: Carolina Fortuna, Vid Hanžel, Tim Strnad, Blaž Bertalanič  

**Link**: [PDF](https://arxiv.org/pdf/2609.18283)  

**Abstract**: As telecommunication networks evolve toward autonomous 5G-Advanced and 6G operations, agentic artificial intelligence (AI) workflows, where large language models (LLMs) execute multi-step reasoning, invoke diagnostic tools, retrieve domain knowledge, and coordinate across agent teams, are increasingly embedded across the edge-cloud continuum. While the biological brain accomplishes complex cognition on an exceptionally modest metabolic power budget of approximately 20W contemporary LLMs are profoundly energy- and memory-intensive, making sustainable lifecycle orchestration a critical operational priority. However, existing AI lifecycle metrics evaluate only isolated, single-model inferences or overlook multi-agent execution graphs entirely. Consequently, network operators lack foundational models to determine whether distributed agent communication incurs meaningful energy costs and where across edge-cloud tiers agent teams should physically reside. To address this gap, we introduce agentic-eCAL, generalizing the Energy Cost of AI Lifecycle (eCAL) metric to directed multi-agent workflows by coupling a closed-form two-rate single-call energy model (compute-bound prefill and memory-bound decode) with 7-layer OSI data transport. Grounded in hundreds of GPU benchmark configurations on NVIDIA A100 and H100, 16 open-weight models and 8 orchestration topologies, we validate components of the metric and study workflow placement implications. Our findings demonstrate that inter-agent text transport incurs 0.25% of workflow energy across 5G RAN, metro, and optical links. Therefore in edge-cloud agent placement the dominant energy cost of distribution is often not the transmission of inter-agent text itself, but the additional inference and context processing induced by that communication. 

---
# Building Trust in Artificial Intelligence: A Necessity for Railway Applications 

**Authors**: Lefebvre Renard Clément, Lébé Vincent, Da Silva Ribeiro Pereira Ricardo, Sundell Johan, Jaoul Arnaud Saiah Kenza, Mijatovíc Nenad  

**Link**: [PDF](https://arxiv.org/pdf/2609.18278)  

**Abstract**: Artificial Intelligence (AI) is currently only applied to non-safety critical applications due to the strict standards and regulations for railway industries. We propose to review the three main fields necessary to increase trust in data science and AI algorithms and reach compliance: robustness, Operational Design Domain (ODD), and explainability. Robustness is the ability of an AI system to maintain its level of performance under any circumstances (ISO24029). ODDs allow the explicit definition of operating conditions under which a system is intended to operate, according to the recently published DIN DKE SPEC 99004. Explainability is the property of an AI system to express important factors influencing the AI system results in a way that humans can understand. Those 3 domains of research are already well investigated by nonrailway actors, with algorithms and methods ready to use for railway applications. A system view is necessary to ensure all trustworthy requirements interact continuously in a safe MLOps environment thereby fostering acceptance from regulators, operators and the public. Beyond safeguarding safety-critical applications, we aim to show that fostering deep trust in AI, as now required by regulatory frameworks worldwide, will unlock its full potential and transform the pace of adoption across mission-critical domains. 

---
# Who Audits Whom, on What Substrate, with What Evidence? An Independence-Graded Audit Protocol for Agentic AI 

**Authors**: Mohamed Chahine Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2609.18272)  

**Abstract**: Agentic AI systems plan, invoke tools and act with limited supervision; they are now both the subject of audits and, increasingly, the auditor. Independence, the foundation of assurance,is still applied to them as a binary. We argue that it must be graded along three orthogonal axes: principal independence (who controls the auditor), substrate independence (an auditor sharing the auditee's foundation-model family, toolchain or guardrails fails with it) and evidence independence (whether evidence is attestable rather than self-reported). Each axis has precedent; the contribution is to grade all three on a single audit, aggregate them by the weakest link, and apply the same rubric when the auditor is itself an agent. We give the model a formal basis by transplanting the beta-factor model of common-cause failure from reliability engineering, a seven-step protocol whose outputs a third party can verify, a structural detectability analysis of a procurement-controls agent audited at three grades, and a Monte Carlo study of the model in which a conventional internal audit of an agent-a real audit team, a second agent, provider logsp-surfaces 5.9% of the faults it could in principle see and none at all in half the fault classes. We map the triple to the EU AI Act as amended, ISO/IEC 42006, UK public-sector risk-management guidance and audit-regulator practice. 

---
# BENCHCOMPASS: From Scores to Signals for Training and Harness Decisions in Payment-Domain LLMs 

**Authors**: Sijie Dong, Wei Ren, Xuanwei Hu, Jiawei Luo, Zifan Wang, Xiaoyun Feng, Hui Cai, Lyuxin Xue, Peng Lu, Jianshe Li, Xin Zhang, Wei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2609.18270)  

**Abstract**: Payment operations are a critical financial infrastructure, but the value of large language models in this domain remains unclear because payment rules change quickly, evidence is fragmented, and decisions depend on transaction state, participant role, region, and payment rail. Existing benchmarks do not isolate whether failures come from missing payment-rule knowledge, poor use of supplied evidence, or brittleness under imperfect harness inputs. We introduce BENCHCOMPASS, a payment-domain benchmark whose construction pipeline builds scenario-grounded tasks from typed evidence packs, applies LLM-based quality checks, creates task-input attack variants, and reserves final item admission for domain experts. The release contains an expert-reviewed Pro benchmark covering payment knowledge, context-grounded scenario reasoning, and Attacked Open robustness, plus a lower-assurance Normal pool for inspection and future curation. Across 16 model variants, BENCHCOMPASS shows qualitatively different failure modes: missing parametric payment knowledge, incomplete reasoning over supplied rules, and failure to reject plausible but invalid workflows. The benchmark remains unsaturated: the best frontier model reaches 89.6% on Open Context-Grounded Reasoning and 81.7% under attacked inputs, while a representative 32B open-weight model reaches 69.8% and 42.6%. Benchmark data and code are available at this https URL. 

---
# REPAIR: Resolving Long-Tail Confusion in Scientific Retrievers via Fact-Verified Iterative Refinement 

**Authors**: Yerim Oh, Gunhee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2609.18262)  

**Abstract**: Precise retrieval of scientific information is fundamentally constrained by long-tailed concepts and high fact-sensitivity of scientific corpora. These challenges often limit the effectiveness of dense retrievers and hallucination-prone LLM augmentation. To address this, we present REPAIR, a self-evolving data augmentation framework for scientific dense retrievers. REPAIR iteratively synthesizes training data to address knowledge gaps by cycling through diagnosis of long-tail concepts, API-guided evidence expansion, and differentiation via hard negative mining. This process effectively grounds retrieval in factual reality to resolve fine-grained distinctions. Extensive experiments demonstrate that REPAIR significantly outperforms 19 strong baselines on nine materials science and biomedical benchmarks. Our work highlights that diagnosing and factually augmenting data to long-tail deficits is essential for robust scientific retrieval. 

---
# Re2A: Situated Conversational Recommendation via Rubric-based Preference Reasoning and Alignment 

**Authors**: Dongding Lin, Jian Wang, Xiaoyan Zhao, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.18249)  

**Abstract**: Real-world recommendation scenarios are commonly grounded in shared physical environments during user-recommender interactions. This motivates situated conversational recommendation (SCR), a complex task requiring recommender assistants to jointly reason over dialogue history, co-observed scenes, and in-scene item attributes. However, current approaches struggle with this setting due to two intertwined challenges: accurately understanding situated user preferences throughout the conversation and generating responses that simultaneously satisfy user needs and grounded situations. To this end, we propose Re2A, a framework that formulates SCR as a structured reason-then-align process. We introduce rubric-based preference reasoning, which uses automated rubrics to guide the model toward producing explicit preference states. Based on these states, we propose a preference-conditioned optimization to align response generation with dual objectives: user preference satisfaction and situation consistency. Extensive experiments on two SCR datasets demonstrate that Re2A consistently outperforms state-of-the-art methods, delivering more precise, context-aware conversational recommendations. Our code is available at this https URL. 

---
# WFM: Wiki Foundation Model for Complex Agentic Reasoning 

**Authors**: Junnan Dong, Linhao Luo, Senlei Zhang, Gong Chen, Taian Guo, Yifei Yu, Rong Tao, Tao Guo, Qian-Wen Zhang, Siyu An, Ruizhi Qiao, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2609.18182)  

**Abstract**: Real-world agents fundamentally require persistent non-parametric knowledge for dynamic reasoning, i.e., long-term memory and retrieval-augmented generation. While graphs have shown reliable advantages in providing structured evidence, the sparse graph representations naturally restrict machine readability and semantic density required for complex agentic workflows. Driven by this limitation, the entire industry is witnessing a paradigm shift from traditional sparse graphs to LLM Wiki, an agent-native knowledge representation that couples dense document contexts with markdown files containing multi-layered topological linkages. However, parameterizing such rich semantics is challenging to encode dense textual contexts using traditional sparse graph embeddings. Moreover, learning LLM Wiki with existing graph encoders could overwhelm distributed system overheads that hinder deployment in large-scale commercial scenarios. To this end, we propose a novel paradigm Wiki Foundation Model, i.e., WFM, tailored for scalable, agent-native representation and retrieval. Specifically, (i) we formalize a Wiki Graph schema that seamlessly bridges fine-grained structures with dense contexts, maintaining explicit topologies alongside continuous semantics; (ii) A query-conditioned attentive aggregation is tailored for rich wiki message passing and explicit attention variance regularization; (iii) We engineer an infrastructural NCCL boundary exchange protocol that hoists static partition indices and leverages fixed-shape GPU-to-GPU collectives, bypassing CPU serialization and memory copy overheads. Extensive evaluations across five long-term agent memory and multi-hop reasoning benchmarks demonstrate the remarkable performance of WFM, while achieving a 10.5 times training acceleration on distributed clusters. 

---
# Time-Aligned Evolving Concept Graphs for Scientific Relation Forecasting 

**Authors**: Fred Sun, Jingze Wang, Minkun Xu, Shangqi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2609.18163)  

**Abstract**: Forecasting scientific relations can guide discovery by identifying promising connections before they emerge. Existing approaches often model concept semantics and graph structure separately or summarize semantics over coarse historical snapshots, leaving semantic representations potentially misaligned with rapidly evolving graph evidence. We propose a time-aligned evolving concept graph framework that jointly models semantic and structural evolution. Its core idea is to treat dated papers as shared update events, reconstructing semantic and structural states from the same publication history through each prediction time. Pair-level fusion combines these states to forecast first co-occurrence, relation formation, and conditional relation type. Holding architecture and training fixed, refreshing context alongside graph updates improves mean relation AUPRC by 16.6% over frozen context. On a graph built from 187,848 papers with 270,687 concepts and 7.45 million co-occurrence links, the complete framework improves mean relation AUROC from 0.9290 for the strongest evaluated baseline to 0.9722, with mean population-weighted AUPRC 0.005778. 

---
# Symbolic Temporal Supervision of LLM Agents Using Contracts 

**Authors**: Yifeng Xiao, Pierluigi Nuzzo  

**Link**: [PDF](https://arxiv.org/pdf/2609.18128)  

**Abstract**: Large language model (LLM) agents augmented by tools can automate complex, multi-step tasks, such as web navigation, code generation, and workflow orchestration, by acting on external systems through tool calls. However, hallucinations, distributional instability, and adversarial manipulations in LLMs, and the irreversible consequences of certain tool calls can lead to harmful outcomes. Existing safeguards either grade recorded trajectories post hoc with stochastic LLM judges or block unsafe actions one call at a time, and no single deterministic artifact supports both roles. We present ContrAgent, a contract-based framework for symbolic temporal supervision of LLM agents. ContrAgent captures an agent's behavior as a sequence of tool calls and formalizes it as a trace over a fixed set of checkable predicates. It then specifies required behaviors using assume-guarantee contracts in linear temporal logic over finite traces (LTLf). Each contract is compiled to a deterministic finite automaton (DFA) that serves two roles: gating agent actions online and evaluating recorded traces offline. A contract library, acting as a reusable knowledge base, is maintained independently of the agent's model and can be applied across different agents within the same task domain. We show the effectiveness of our approach on four benchmarks spanning both roles, where ContrAgent matches state-of-the-art LLM-judge and rule-based guardrail baselines while producing deterministic, reproducible verdicts and, in the online mode, orders-of-magnitude lower per-call latency. 

---
# Designing Agentic AI Workflow Portfolios under Imperfect Selection and Compute Cost 

**Authors**: Mojtaba Abdolmaleki, Stefanus Jasin, Boyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.18126)  

**Abstract**: Agentic AI systems often approach the same task through multiple workflows that differ in reasoning strategy, verification structure, and compute cost. A natural deployment policy is to use the workflow with the highest average performance, but this can be suboptimal because different workflows may succeed on different instances. We study a portfolio-and-selector paradigm in which a firm runs multiple workflow executions and selects the final answer after observing their outputs. Additional executions may uncover correct answers that the best standalone workflow misses, but they consume compute and introduce plausible distractors that complicate final selection. We formulate this as a workflow portfolio problem in which the firm jointly chooses run size and allocation across workflow types. We summarize selector quality through an odds-lift index and derive sharp bounds on the value of workflow variety. For finite workflow pools, we develop exact formulations, linear programming relaxations, randomized rounding procedures, and computable performance certificates. For large implicit workflow classes, we derive a finite-dimensional dual and an ellipsoid method using a pricing oracle to identify workflows with high weighted accuracy net of recurring compute cost. Under a weak condition, the method obtains a near-optimal solution to the relaxation with polynomially many oracle calls. We evaluate the framework on three datasets: ABCD, Schema-Guided Dialogue, and HotpotQA. Relative to the best standalone workflow, portfolio optimization improves held-out selector accuracy by 3.1, 7.5, and 0.9 percentage points, respectively. Dual-guided workflow generation adds 3.5 points on ABCD and 24.1 on HotpotQA, with no additional gain on Schema-Guided Dialogue. 

---
# AutoTuneBench: Trustworthy Measurement for Agent Auto-Tuning of LLM Serving Engines 

**Authors**: Li Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.18123)  

**Abstract**: Large language model agents tune GPU kernels and serving engines through a closed loop of propose, measure, and keep, but the measurements behind this loop are not trustworthy. We characterize four failure modes from a four-day pilot corpus of 619 model calls: strawman baselines manufacture speedups, absolute times do not transfer across machines, saturated tasks nullify comparisons, and infrastructure defects impersonate science. We present AutoTuneBench, a benchmark and measurement protocol that makes trust architectural. The protocol is frozen as code with test-enforced provenance; a database-level validator rejects out-of-protocol results; anti-cheat checks run outside the agent's modification surface; comparisons follow pre-registered readouts; and measurements anchor to externally published results, grounded in paired-seed statistics with a 5\% cross-run coefficient-of-variation cap. Honest measurement rewrites the headlines: our best kernel reads 10.6x against a naive baseline but 2.03x against the honest one; one configuration delivers 1.174x on one machine and 1.0049x on another; a pre-registered on/off comparison nulls at a shared wall (2.4840 vs 2.4957\,ms); and the KernelBench Level-1 suite admits 51\% of tasks with median speedup 1.0001x over PyTorch eager. The protocol, the two-engine corpus (vLLM and SGLang), and its audit trail are released as open artifacts. 

---
# When Is Graph Structure Worth Its Cost? The Case for Structure Pricing in Retrieval-Augmented Generation 

**Authors**: Yuzhong Zhang, Haoyang Ma, Chao Peng, Lionel Briand, Boxi Yu, Jialun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2609.18099)  

**Abstract**: Graph-based retrieval-augmented generation (RAG) can help answer questions that require information from many documents. However, building a graph often requires many language-model calls during ingestion. It is therefore important to ask whether its quality gains justify the additional cost.
We present EffiRAG, a graph-based RAG system designed to reduce this cost. It uses the graph to locate relevant passages and generates answers from the original text. This design preserves source information while keeping graph construction and query processing lightweight.
We evaluate EffiRAG on UltraDomain, which contains 120 open-ended questions from four domains. Compared with LightRAG-hybrid, EffiRAG produces the preferred answer on 93 questions. LightRAG is preferred on 7, and the remaining 20 are splits. EffiRAG also reduces total system cost by 57 percent, from USD 0.952 to USD 0.408. The cost includes language-model calls during ingestion and querying.
The advantage remains as the corpus grows. At 10 and 20 documents per domain, EffiRAG uses a lightweight, non-LLM filter to skip low-salience chunks. It remains preferred over LightRAG-hybrid. It costs 4.2 times and 4.5 times less, respectively.
The comparisons identify different quality-cost trade-offs. Graph-based RAG systems should therefore be evaluated by both answer quality and cost. The results favor graph structure that locates and preserves source evidence. 

---
# Decodability is Not Causality: Dissociating Probe Readouts from Behavioral Drivers via SAE Decomposition 

**Authors**: Devesh Tiwari, Camille Davis, Shivank Sinha, Talia Weaver, Aditya Shah, Maheep Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2609.18080)  

**Abstract**: Linear probes can decode safety-relevant concepts such as truthfulness from language-model activations, but probe accuracy may show only decodability, not that the features the probe weights causally drive model behavior. We demonstrate that this gap cannot be closed from the geometry of probe weights alone: the features geometrically aligned with probe direction need not be the ones the model uses, so causal relevance requires intervention. We introduce a feature-level diagnostic that decomposes a deployed True/False probe into sparse-autoencoder (SAE) features, ranks those features by both probe alignment and by gradient sensitivity of the model's behavior, and ablates the resulting shared, probe-only, and random feature sets under a coherence gate. On the truth probe of Buerger et al. (2024) (TTPD), applied in the instructed truth/deception setting of Long et al. (2025) for Gemma2-9B-Instruct, the two rankings overlap only weakly (about 12%, Spearman rho = 0.10), and ablation dissociates them sharply: features the probe shares with the model flip the output far more (up to 27%) than equally sized probe-only (6%) or random (1%) features at full coherence, while probe-only features instead perturb the probe's own readout. The dissociation holds across five seeds and a held-out split, and an activation-aware selection of features flips behavior nearly three times as often as the probe's geometric top features (17.6% vs. 6.1%). In this setting, therefore, the geometric projection of a probe's weight vector alone does not identify the features the model causally uses; however, combining probe information with feature activation statistics recovers substantially more behaviorally causal features, and coherence-gated SAE intervention is needed to separate them from probe readouts. 

---
# Teaching AI, Robotics, & Community: A Hubs-Based K-12 Education Framework for Reaching Rural Schools 

**Authors**: Maxwell J. Jacobson, Gustavo Rodriguez-Rivera, Petros Drineas, Yexiang Xue  

**Link**: [PDF](https://arxiv.org/pdf/2609.18072)  

**Abstract**: K-12 robotics and AI education remains difficult to scale, especially in rural regions lacking sustained technical mentorship. Programs like FIRST provide competition pathways and instructional opportunities, but they do not eliminate the need for local programming and robotics expertise. We introduce AI, Robotics, & Community (ARC), a hubs-based framework where colleges train undergraduate mentors and host workshops for nearby K-12 teams. Mature school programs can become secondary hubs that support additional schools, creating a self-reinforcing education loop where mentorship reach propagates geographically and can even grow super-linearly. We first evaluate ARC through a trial deployment at one university. The trial created three rural robotics teams. On five-point Likert surveys, mean increases in K-12 programming knowledge, resource access, and practice opportunities were 2.00, 2.25, and 1.25 points. Likewise, undergraduate confidence teaching technical concepts, adapting explanations, managing groups, and finding mentoring enjoyable and meaningful increased by 1.29, 1.14, 1.00, and 1.14 points. Additionally, we create a spatial Markov model of ARC's growth and simulate it using the state of Indiana as a testbed. Under moderate conditions, we find that ARC reaches 74% of Indiana's 1,925 public K-12 schools and produces 992 robotics programs after 40 years, compared with 161 projected under natural growth alone. Together, these results show ARC can create and support rural robotics programs, train undergraduate AI and robotics mentors, and potentially scale mentorship across a region. 

---
# The Other Half of the Memory Wall: Serving 35B MoEs from SSD with Trained Routing Prediction 

**Authors**: Yu Lin, Yiming Wang, Runyuan Cai, Hanze Liu, Xiaodong Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2609.18063)  

**Abstract**: Mixture-of-experts (MoE) inference on consumer hardware is bounded by weight memory: a 35B-class model is 19.5GB at 4-bit, and sparsity shrinks the compute per token, not the bytes that must be held. Naive offloading to SSD does not help on its own, because layer N+1's experts must be chosen before layer N's output exists, so the reads cannot start early enough to hide behind compute. We present Edge0, a streaming MoE inference engine that closes the gap with a prerouter: a per-layer head predicts the next layer's routing one token ahead, and the prediction is consumed as the routing itself, so the staged expert set equals the routed set and nothing is dropped. An unmerged recovery LoRA, trained on the student path, pays back the quality lost to int4 quantization and routing replacement. On a single 24GB machine, Edge0
serves a 35B MoE at 20tok/s inside 3GiB of peak active memory, within a few points of its fp16 teacher on average across five public benchmarks. An 8B tier runs on the same framework, and the framework, checkpoints, and adapters are open source. 

---
# Anchoring What Matters: A Dual-Level Learning Framework for Visually-Grounded Multimodal Reasoning 

**Authors**: Xinxin Song, Siyuan Li, Tingxiong Xiao, Jinli Suo  

**Link**: [PDF](https://arxiv.org/pdf/2609.18057)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has significantly improved the reasoning capabilities of large vision-language models (LVLMs). However, standard on-policy RLVR algorithms face a critical optimization bottleneck in preserving and reinforcing visually grounded reasoning behaviors: valuable visually-grounded reasoning trajectories are discarded after a single update, while uniform token advantage allocation prevents the model from reinforcing critical perception or reasoning steps. To bridge this gap, we propose PIVOT, a dual-level learning framework that anchors policy optimization around informative visual reasoning signals. Specifically, PIVOT introduces a self-calibrated experience replay mechanism, which selectively collects and replays visually-grounded historical experiences as stable reference anchors for policy optimization. Building upon this, we further design a vision-guided advantage allocation mechanism to allocate additional vision-aware advantages to tokens based on their local visual support and impact on downstream reasoning. Extensive experiments across diverse benchmarks demonstrate that PIVOT achieves highly competitive performance in enhancing the multimodal reasoning capabilities of LVLMs. 

---
# Missing Bridges: Composition-Aware Active Imitation Learning 

**Authors**: Maxwell J. Jacobson, Ahmed H Qureshi, Yexiang Xue  

**Link**: [PDF](https://arxiv.org/pdf/2609.18004)  

**Abstract**: Active imitation learning reduces expert effort by allowing a learner to request the demonstrations it needs. Existing methods typically select these requests for their expected information gain about the expert policy. In structured multi-task domains, however, the number of start-goal tasks may grow combinatorially despite their solutions sharing reusable behavior. This makes composable behaviors especially valuable, since a single demonstration may help solve many tasks at once. Prior methods do not explicitly account for this value when selecting which demonstration to request. We introduce Adaptive Agents via Latent Topologies (AALT), which requests demonstrations that maximize expected gains in start-goal connectivity. We further show that this objective is formally tied to information gain about task reachability. AALT organizes existing demonstrations into a topology of latent hub states connected by learned behaviors, identifies high-value bridge demonstrations that are likely to enable many tasks at once, and grounds each to an expert query. At inference, it plans through the resulting topology and conditions a diffusion policy on each successive hub transition. In a simulated UR5e robot ordered-retrieval domain with 72 tasks, AALT improved from 42/72 to 72/72 (100%) successful tasks consistently using only 3 demonstrations totaling 5 transitions beyond the initial dataset. After 20 demonstrations, the strongest baseline averaged 88.6% success using 98 transitions. 

---
# Multimodal Conditioning of Fine-Tuned Stable Diffusion XL for Controllable and Culturally Faithful Ulos Motif Generation 

**Authors**: Humasak Simanjuntak, Tamara Yunika Sianipar, Bronson T.M Siallagan, Difya Laurensya Ambarita, Arlinta Barus  

**Link**: [PDF](https://arxiv.org/pdf/2609.17987)  

**Abstract**: The traditional Batak Ulos weaving industry faces growing challenges in producing diverse, innovative motifs due to limitations in conventional, manually driven design methods. This study proposes a multimodal generative framework integrating a fine-tuned Latent Diffusion Model (Stable Diffusion XL v1.0 via LoRA) with a Multimodal Large Language Model (LLaMA 1.5-7B) to enable controllable, culturally faithful Ulos motif generation. Four complementary conditioning mechanisms: text, image, representation, and semantic map (via ControlNet) jointly guide the generation process, each governing a distinct aspect from semantic intent to spatial layout. A five level ablation study across three scenarios (shape transformation, colour variation, and high-complexity input) shows that conditioning effectiveness is not proportional to the number of mechanisms combined: Text + Image + Semantic Map achieved the best FID (270) and CLIP Score (0.65 - 0.70) but the weakest SSIM (0.65), while Text + Image + Representation offered the best overall balance, with stable SSIM (0.84) and competitive FID (280). Combining all four mechanisms yielded the weakest FID (330), indicating conflicting optimization signals. Qualitative evaluation by nine weavers and thirty public participants confirmed statistically significant positive acceptance (Wilcoxon, p=0.007 and p<0.001, respectively). A web-based prototype supporting text-to-image and image-to-image generation was also developed, offering a practical digital design tool for cultural heritage preservation. 

---
# RideWay: Benchmarking Efficient Task Completion for Tool-Using Language Agents 

**Authors**: Qingnuan Han, Boli Fang, Mingzhi Hou, Claire Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.17985)  

**Abstract**: AI agents are usually evaluated by whether they complete a task. In interactive service settings, a successful agent can still frustrate users by asking repeated questions, performing redundant searches, or making avoidable revisions. We introduce RideWay, an efficiency-centered benchmark for ridehailing agents in a stateful tool-calling environment, together with Efficiency Utility, a success-gated metric that discounts successful trajectories for excess tool calls and user-facing turns relative to task-specific reference effort. Human paired preferences calibrate the relative penalties, reflecting an aggregate service-workflow trade-off: extra dialogue often creates visible friction, whereas extra tool use can sometimes verify constraints or preserve user intent. Across 58 tasks and 24 models, the fitted penalty for excess turns is about twice that for excess tool calls. On task-disjoint held-out preferences, Efficiency Utility achieves 78.7% accuracy overall: 90.6% when trajectories differ in turns, but chance-level accuracy when they differ solely in tool calls - the axis on which human annotators agree least. RideWay therefore makes interaction efficiency measurable alongside task success, while exposing the boundary of count-based tool-use evaluation. 

---
# TuiML: Machine Learning for AI Agents 

**Authors**: Nilesh Verma, Nick Lim, Albert Bifet, Bernhard Pfahringer  

**Link**: [PDF](https://arxiv.org/pdf/2609.17984)  

**Abstract**: Machine-learning libraries such as Weka and scikit-learn were designed for human programmers. Language-model agents now use these same libraries by recalling APIs from memory and writing code, an approach that hides what a library offers, delays errors until runtime, and loses experimental state between turns. We present TuiML, a self-contained machine-learning library built for AI agents, with native algorithms across supervised, unsupervised, time-series, data handling, tuning, and evaluation tasks. Every component describes itself through machine-readable metadata and parameter schemas, so an agent can search the library, inspect components, compose validated workflows, and register new ones that become discoverable in turn. Every call is validated, seeded, and traced, and sessions export as runnable notebooks, making experiments reproducible by construction. One specification layer drives the Model Context Protocol (MCP), agent-framework adapters, a Python API, a CLI, and local model serving, while data and models never leave the machine. Benchmarks show TuiML remains predictively competitive with scikit-learn and Weka. While looking like a conventional library to a human user, TuiML is designed for agents first, allowing them to read, extend, and operate machine learning autonomously. TuiML is open source, with documentation at this https URL. 

---
# Contiguity, Not Importance: Budgeted Repair of Stale KV Caches After Document Edits 

**Authors**: Mingyang Mao, Wyatt Mackey, Xiaomin Lin  

**Link**: [PDF](https://arxiv.org/pdf/2609.17983)  

**Abstract**: KV-cache reuse can reduce inference cost in retrieval-augmented generation and agentic systems, but cached contexts may become stale when retrieved knowledge, working memory, or user state is edited. Under causal self-attention, even a local edit can affect downstream KV states. A full re-prefill reliably restores consistency but is costly, whereas refreshing only the edited span can leave downstream dependencies stale. We formulate in-place repair as budgeted recomputation and compare training-free position-selection policies on a factual RAG benchmark with matched direct and derived edits. Across three model families, all policies repair direct cases, but derived cases clearly separate them. At the primary budget, a contiguous edit-local window recovers at least 0.94 of the post-edit answer margin and substantially outperforms attention-based, KV-deviation, and structural selectors. Mechanistic analysis shows that position sets effective under clean-state transplantation can fail under actual recomputation because scattered positions inherit surrounding staleness. The edit-local advantage also depends on adjacency and largely disappears when the answer-bearing text moves downstream. Because answer-relevant edits almost always corrupt model behavior, failure severity is difficult to predict, and repair is 13-21 times faster than full re-prefill, our results support unconditional edit-local repair when the dependent text remains adjacent to the edit. 

---
# When to Call an LLM: A Confidence-Gated Hybrid for Cost-Effective Emotion Recognition in Conversational AI 

**Authors**: Sai Babu Udayagiri, Arjun Chouhan, Ravisekhar Kanagala, Trishala Pavagada  

**Link**: [PDF](https://arxiv.org/pdf/2609.17977)  

**Abstract**: Emotion recognition in conversation (ERC) is a production capability behind agent-assist prompts, escalation routing, and post-call analytics in contact-center-as-a-service (CCaaS) platforms, where cost and latency constraints matter as much as accuracy. We report a systems-level comparison of three deployment options for dialogue-contextual ERC: a low-cost stacked ensemble (sentence embeddings, windowed context, RandomForest/XGBoost/logistic-regression stacking), off-the-shelf LLM prompting (GPT-4o-mini; zero-shot, few-shot, chain-of-thought), and a confidence-gated hybrid that escalates only the ensemble's least-confident predictions to the LLM - modeled on IVA-to-human-agent escalation policies used in production contact centers. On IEMOCAP, the ensemble significantly outperforms every LLM configuration (0.595 vs. 0.460-0.536 weighted F1, p < 0.0001) at a fraction of the cost and sub-10ms latency; on MELD and CMU-MOSI the ranking reverses, showing neither pure system is a safe default. The confidence-gated hybrid resolves this by Pareto-dominating both pure systems on all three datasets (0.620, 0.643, 0.824 weighted F1) while routing the majority of traffic through the near-zero-cost ensemble, translating to roughly $10-85 per million utterances versus $99-170 for an LLM-only pipeline. The escalation policy is not an opaque cost/accuracy dial: escalated turns disproportionately follow an emotion or sentiment shift, giving operators an interpretable, auditable routing signal, and the ensemble's confidence is well-calibrated and safely under- rather than over-confident. The pattern holds across three datasets and two LLM providers. Confidence-gated cascading is established in general ML systems; our contribution is showing it transfers cleanly to dialogue-contextual ERC, yielding a concrete deployment recipe for CCaaS and conversational-AI platforms deciding how to allocate LLM spend. 

---
# Memory Has Geometry: Non-Uniform Geometric Memory for Long-Horizon Personalized AI 

**Authors**: Jiahong Liu, Wenhao Yu, Zexuan Qiu, Menglin Yang, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2609.17969)  

**Abstract**: Long-term memory is becoming a core substrate for personalized AI, yet most systems still represent personalization as discrete records in a largely static latent space, accessed under one global similarity notion. For data mining, this creates a mismatch: the evidence is a temporal event stream, while the dominant abstraction is a searchable record set. We argue that long-horizon personalization should instead model memory as a user-specific dynamical state space with locally heterogeneous geometry. Geometry here is a computational language, not a literal claim about cognition: it captures stable versus volatile regions, variable-rate drift, heterogeneous neighborhoods, and uncertainty about current user state. Profiles and isolated events remain useful as points, but interaction, feedback, and elapsed time induce trajectories. Memory access then becomes trajectory-conditioned reconstruction of the relevant user state, not only nearest-neighbor lookup. 

---
# Measuring AI Leadership: Development and Validation of a Multidimensional Measure for AI-Native Organizations 

**Authors**: Mustafa Akben, Leslie Coyne  

**Link**: [PDF](https://arxiv.org/pdf/2609.17965)  

**Abstract**: AI is changing what leaders must judge, explain, learn, and coordinate, yet existing measures do not capture these behaviors at the level needed to study leadership in AI-enabled work. We develop the AI Leadership Battery, which organizes 36 behaviorally specific subdimensions into 11 theory-specified content families. Following established scale-development procedures, the research used deductive item generation; content validation of definitional correspondence and definitional distinctiveness; exploratory factor analysis and item reduction; confirmatory factor analysis in independent samples; and tests of internal consistency reliability, convergent validity, discriminant validity, and criterion-related and incremental validity. Across the development and validation studies, the analyses provided evidence for the Battery's content, multidimensional structure, reliability, and distinction from selected orbiting constructs. The Battery also contributed additional information beyond orbiting constructs across organizational growth, decision speed, customer/stakeholder response capability, AI-enabled team performance, AI-enabled work experience, AI security and risk management, and AI adoption and integration. The resulting measure provides researchers with a behavioral framework for examining how leaders in AI-enabled work regulate judgment, learning, adaptation, transparency, and accountability. 

---
# Collaborative Memory for Multi-Agent VLM Systems 

**Authors**: Huixin Zhang, Shao-Jun Xia, Di Wang, Liangxi Liu, Hainan Xiong, Zihao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.17921)  

**Abstract**: Vision-language model (VLM) agents combine specialized perception, tools, and reasoning to address complex visual tasks. In multi-agent settings, different agents inspect different image regions, video frames, or visual representations, so collaboration extends beyond distributed reasoning to distributed perception. This makes shared visual context a central problem in VLM agent collaboration. In this paper, we frame memory hierarchy, cross-agent sharing, and consistency mechanisms around the need to reconcile interpretations and update dependent reasoning. Effective collaboration requires agents to build on contributions from other agents, recover missing visual context, and reconcile differing interpretations as new evidence emerges. Shared visual memory preserves not only images or textual summaries but also the dependencies among observations, agent interpretations, and subsequent reasoning. Together, these design considerations shape how information flows and evolves across VLM agents. The proposed framework provides a foundation for building reliable and resource-efficient agent teams. 

---
# OBC-Prune: Outcome-Based Calibration for Large Reasoning Model Pruning 

**Authors**: Ha Lan Nguyen, Huy Hoang Tran, Trac-Duy Tran, Dung D. Le  

**Link**: [PDF](https://arxiv.org/pdf/2609.17890)  

**Abstract**: Large reasoning models (LRMs) generate long chain-of-thought traces before answering, creating significant inference overhead. Pruning can reduce this cost, but its effectiveness depends on the calibration data used to estimate parameter importance. Recent work calibrates on the model's own rollouts instead of generic dataset, but treats all reasoning tokens uniformly, regardless of whether they contribute to successful reasoning. As a result, pruning protects weights by statistical salience rather than by their contribution to correct reasoning, so weights behind erroneous computation survive as readily as those behind correct computation. These erroneous patterns then get carried into the pruned model, degrading reasoning quality, producing both lower accuracy and longer reasoning traces. We propose Outcome-Based Calibration for Large Reasoning Model Pruning (OBC-Prune) to close this gap. OBC first constructs difficulty-matched pairs of correct and incorrect rollouts from problems the model answers inconsistently. It then estimates the causal importance of each reasoning sentence through intervention-based analysis, quantifying how removing its influence affects subsequent predictions. These causal importance scores are converted into per-token weights that rescale the calibration activations used by one-shot pruning methods (SparseGPT, Wanda, ALPS), without modifying the underlying pruning algorithms. Experiments on DeepSeek-R1-Distill-Qwen 1.5B, 7B, and 14B models at 40\% and 50\% sparsity demonstrate consistent improvements over state-of-the-art calibration baselines across most model sizes and sparsity levels on MATH500, LiveCodeBench, and AIME 2025. These results indicate that preserving causally important reasoning circuits is a substantially more effective pruning objective than uniformly preserving observed activations. 

---
# ERPBench: A State-Grounded Evaluation Paradigm for Computer-Use Agents in Enterprise Software 

**Authors**: Kratika Bhagtani, Kusha Sridhar, Maziyar Baran Pouyan, Yuying Zhao, Eugene Siow  

**Link**: [PDF](https://arxiv.org/pdf/2609.17885)  

**Abstract**: Computer-use agents that operate through screenshots and simulated actions are advancing rapidly, yet their evaluation remains anchored to general desktop and web tasks. Enterprise Resource Planning (ERP) systems run the finance, procurement, inventory, and customer operations of organizations worldwide, and pose distinct challenges for computer-use agents: dense interfaces, coordinated multi-step interactions, and errors that alter persistent business records rather than surfacing on screen. Existing enterprise benchmarks rely on proprietary platforms or on simulated approximations of such software. We introduce ERPBench, a benchmark that evaluates screenshot-only agents on a live and reproducible ERP system and scores each task against ground-truth values in its database. Beyond the benchmark, we present a production-grade harness that gates agent actions behind human approval for safe deployment, which ERPBench runs autonomously. Evaluating six closed and open-source agents, we demonstrate that strong general GUI performance does not transfer to enterprise reliability. Even when an agent reaches the right form and saves it, the stored record is often wrong: some agents save in up to 85% of runs but write the correct value in as few as 3%. We further characterize failure modes specific to enterprise workflows. 

---
# Do Frontier Models Seek Safety Evidence Before Acting? 

**Authors**: Omer Tafveez  

**Link**: [PDF](https://arxiv.org/pdf/2609.17865)  

**Abstract**: Frontier models are often evaluated on how they respond to safety information once it is already in context. We study an earlier decision point: whether models choose to acquire safety-relevant evidence before acting. We introduce SAFE, a controlled benchmark in which models make deployment decisions with optional evidence that varies in retrieval cost, probability, severity, and presentation. Across GPT-5.5, o3, Claude Opus 4.8, and Claude Sonnet 4.6, we find distinct evidence-acquisition policies: Opus inspects nearly by default, o3 is the most skip-heavy and threshold-sensitive, and GPT-5.5 and Sonnet occupy intermediate regimes. Inspection increases strongly with severity and decreases with retrieval cost, whereas probability has much weaker behavioral influence: increasing the stated likelihood of a problem from 10% to 70% changes inspection by at most 21 percentage points. Despite these differences, Stage 1 rationales are dominated by expected-value reasoning across models. A cost-obligation decomposition further shows that avoidance is driven primarily by retrieval friction and explicit threats to the deployment payoff rather than by the remediation duties created by knowing. Counterfactual interventions reveal a further mismatch between behavior and explanation: evidence framing can strongly change decisions near the inspection boundary while going largely unmentioned, whereas probability is frequently cited despite having little causal influence. These results suggest that deployment-time safety depends not only on how models respond to known risks, but also on whether they acquire the evidence needed to know that acting is safe. 

---
# The Inference Engineering Pareto Atlas: Which Optimizations Dominate the Cost, Quality, and Latency Frontier? 

**Authors**: Srikanta Datta Tumkur, Jay Iyer, Mehar Simhadri, Sai Pavan Kumar, Sai Kapil Kumar, Ramesh Nampelly  

**Link**: [PDF](https://arxiv.org/pdf/2609.17863)  

**Abstract**: LLM inference optimizations report speedups on different models, GPUs, prompts, and quality metrics, making them hard to compare or combine. We build a cost, quality, and latency Pareto atlas to identify the best configurations for different deployment constraints. Since exhaustive testing is impractical, we measure 54 configurations of Qwen2.5-7B-Instruct running on vLLM 0.12 across L4, A100, and H100 GPUs and use these anchors to calibrate a simulator. It reproduces measurements at anchored batch sizes, with cross campaign drift below 1.5 percent. A separate quality evaluation tests FP16, AWQ 4bit, FP8 weights, and FP8 KV cache on 200 GSM8K questions with five examples per prompt. Sparse attention is evaluated only in simulation. On the calibrated grid, 18 of 36 configurations reach the Pareto frontier. Combined methods reach it more often than individual methods, with 9 of 15 combinations versus 9 of 21 single methods. Quality testing changes the winners. AWQ 4bit reduces per token latency to 0.34 times baseline on L4 but loses 5.9 percent of strict GSM8K accuracy, narrowly missing the 95 percent quality floor within sampling uncertainty. Flexible answer extraction matches FP16 accuracy, suggesting the loss comes from formatting rather than arithmetic. FP8 weights retain 99.4 percent of baseline accuracy at 0.61 to 0.65 times baseline latency across all three GPUs and appear in three of four regime winners. A naive FP8 KV cache maintains normal throughput but answers none of the 200 questions correctly, showing why speed alone is insufficient. Under two prompt designs, n gram speculative decoding measures at 0.90 to 0.98 times baseline and adds no benefit on this stack. The best choice depends on the constraint and GPU: H100 wins for tight latency, while A100 wins for throughput and low cost at 0.106 dollars per million tokens. 

---
# SNOMED CT Concept Recommendation from Masked Clinical Context 

**Authors**: Ali Noori  

**Link**: [PDF](https://arxiv.org/pdf/2609.17855)  

**Abstract**: Standardizing clinical language to SNOMED CT supports interoperability, analytics, and reusable phenotyping, but concept recommendation remains difficult when relevant concepts are rare or absent from training data. We present a masked-concept recommendation benchmark using the SNOMED CT Entity Linking Challenge v1.2.1 data derived from MIMIC-IV-Note. The dataset contains 75,491 annotations across 272 discharge summaries, with 204 notes used for training and 68 for historical testing. For each unique note-concept pair, the target mention is masked from a local clinical context and the system ranks SNOMED CT concepts observed during training. We compare a popularity baseline, sparse TF-IDF concept prototypes, dense latent semantic analysis embeddings, sparse-dense fusion, retrieved-note evidence, and a retrieval-augmented hybrid. Sparse TF-IDF performs best, achieving Recall@1 of 14.81%, Recall@10 of 33.43%, MRR of 0.2114, and nDCG@10 of 0.2297. Retrieval augmentation does not improve this baseline, with Recall@10 of 31.99% and MRR of 0.1937. Performance is strongly affected by concept frequency: Recall@10 is 7.74% for concepts appearing in only one or two training notes versus 43.90% for concepts appearing in more than ten. In addition, 9.66% of test note-concept pairs contain concepts unseen during training. These findings show that local lexical context and terminology coverage are major determinants of recommendation quality in low-resource settings and provide a reproducible baseline for future ontology-grounded and biomedical-encoder retrieval systems. 

---
# Learning Heterogeneous Preferences 

**Authors**: Shiwali Mohan, Matt Hong, Dule Shu, Aniek Fransen, Shabnam Hakimi, Matt Klenk  

**Link**: [PDF](https://arxiv.org/pdf/2609.17847)  

**Abstract**: Learning from human feedback has become a central paradigm for training modern AI systems, where models of human utility are used as reward models in policy learning. Existing methods typically assume a \emph{universal utility} function shared across a population and treat disagreement between annotators as stochastic variation. While suitable for objective tasks, this assumption breaks down in subjective domains where preferences vary systematically across individuals. We study the problem of subjective preference learning, in which observed choices arise from heterogeneous but internally consistent utility functions. Drawing upon rational choice theory, RCT \parencite{tversky1981framing}, we introduce \emph{individuated utility} functions conditioned on both the individual and their decision context, and propose a novel multi-stage architecture for estimating them from multi-modal data. We evaluate our framework on a newly collected dataset of more than $575{,}000$ pairwise aesthetic judgments from $2{,}398$ participants comparing automotive wheel designs. Our experiments show that individuated utility models substantially outperform universal utility models including foundation model baselines. Our results demonstrate that disagreement reflects meaningful preference heterogeneity rather than annotation noise. More broadly, our findings highlight the importance of collecting annotator attributes and learning individuated utility functions, enabling reward models that explicitly account for whose preferences they represent and faithfully capture human decision diversity. 

---
# A Four-Stage Decomposition of Word-Problem Solving and Mechanistic Fragility in LLM Math Reasoning 

**Authors**: Zhongdi Qu, Carla P. Gomes  

**Link**: [PDF](https://arxiv.org/pdf/2609.17804)  

**Abstract**: Large language models solve grade-school math word problems with high accuracy, yet a single irrelevant clause inserted into the problem can collapse it. We reconcile these observations with a mechanistic account. We show that the model's internal computation decomposes into a four-stage sequential pipeline, Schema Abstraction, Operation Planning, Operand Binding, and Computation, each stage producing a distinct intermediate representation in an identifiable band of layers. Using the same scaffold to diagnose distractor-induced failure, we localize the corruption to a single stage, Operation Planning, implemented by a set of attention heads whose causal role we validate bidirectionally. In short, we provide a mechanistic interpretation of math word problem reasoning in LLMs, and their failure when distracted. 

---
# FairCompressAgent: An Agentic Framework for Fairness-Aware Model Compression for FPGA Deployment 

**Authors**: Yuanbo Guo, Yiyu Shi  

**Link**: [PDF](https://arxiv.org/pdf/2609.17786)  

**Abstract**: Fairness-aware model compression requires selecting methods and configurations that balance accuracy, fairness, and deployment cost. These decisions become more difficult when compression methods are composed or the user's requirements change. In this paper, we propose FairCompressAgent (FCA), an agentic framework that integrates fairness-aware pruning, incremental quantization, and sparse low-rank factorization through a common operator interface. A language-model planner uses model profiles and measured outcomes to select compression configurations, while an execution layer performs compression, fine-tuning, evaluation, and constraint-based selection. FCA also supports requirement updates and reports the remaining violation when a request cannot be satisfied. Experiments on Fitzpatrick-17k with VGG-11 compare four search methods over 40 measured configurations. Under the accuracy-constrained request, FCA selects a compressed model with 59.54% less inference tensor storage, while validation average precision increases from 0.5141 to 0.5233 and equalized opportunity (EOpp) decreases from 0.2251 to 0.2168. It reaches the same final selection as one-shot planning with 7.33 versus 12 candidate evaluations on average, under their respective stopping policies. Repeated fine-tuning, held-out testing, and online requirement updates characterize the stability and interactive use of this compression workflow. The results demonstrate how measured feedback and explicit constraints support the selection and interactive refinement of fairness-aware compression configurations. 

---
# SAGE: Governed Artifact Generation from Enterprise Guidelines 

**Authors**: Mohammadreza Sediqin, Shivali Dalmia, Sumukha Thoppanahalli, Srinivasa Karthikeya Reddy Kovvuri, Abhishek Mukherji  

**Link**: [PDF](https://arxiv.org/pdf/2609.17775)  

**Abstract**: Enterprise guideline documents mix narrative text, complex tables, and embedded images, and converting them into structured work artifacts still takes two to three days of manual effort each. Current language and vision-language models extract from such documents but offer no governed workflow beyond extraction: no validation, no consistency checking, no traceable artifact generation. We introduce SAGE, a governed multi-stage LLM pipeline organized around a shared versioned rule store with stable identifiers, schema-validated inter-stage contracts, and end-to-end provenance tracking. Extracted rules undergo deterministic structural validation and LLM-based semantic scoring, then a consistency module that removes duplicates, flags contradictions, and surfaces specification gaps; only uncertain or flagged items reach reviewers, while high-confidence outputs are auto-approved. On 120 documents, SAGE cuts turnaround from days to 20-100 minutes, achieving a 96% document-level success rate with 3.2% hallucination, extracting 3,896 rules and producing 812 artifacts ready for human review; without governance, hallucination rises to 15.7%. 

---
# Imitation Learning for Autonomous Driving in CARLA 

**Authors**: Jordy Kieto  

**Link**: [PDF](https://arxiv.org/pdf/2609.17757)  

**Abstract**: Behavioral cloning trains a policy offline on expert demonstrations, but deployment is closed loop: each action affects the observations the policy receives next. We study how much closed-loop driving competence a compact multimodal policy can acquire from offline demonstrations in the CARLA simulator. The policy uses five-frame histories of RGB images, LiDAR, vehicle telemetry, and lane waypoints to predict throttle, brake, and steering at 20 Hz. Demonstrations were collected in three stages, ending with a systematic route-generation procedure that enumerates spawn points and feasible maneuvers and verifies completed autopilot routes. The released 1.36 million parameter policy was trained on 236,882 windows, representing about 3.3 hours of driving from 448 captures. The resulting policy drives autonomously for hours on training and held-out routes. In our runs, it did so without collisions and also transferred qualitatively to an unseen CARLA town with different road geometry. We also observed recovery from large trajectory deviations, although we do not claim systematic recovery without controlled evaluation. We report offline metrics and distinguish measured results from qualitative closed-loop observations. We release the code, trained checkpoint, ONNX model, data sample, and an evidence audit for the reported claims. 

---
# A Systematic Evaluation of the COTQ Provincial Land Cover Product: Structural Consistency, Spectral Separability, and Relative Positioning Against ESA, ESRI, and Google Products 

**Authors**: Étienne Clabaut, Samuel Foucher, Yacine Bouroubi  

**Link**: [PDF](https://arxiv.org/pdf/2609.17731)  

**Abstract**: High-resolution land use and land cover (LULC) products derived from Sentinel-2 imagery are widely used for environmental monitoring and land management, yet their performance can vary across regions with complex ecological gradients and heterogeneous surface conditions. In Quebec, these limitations motivated the development of a provincial 10-m land-cover product, the COTQ, designed to support annual monitoring of land occupation and soil artificialisation. This study presents a systematic evaluation of the COTQ product relative to three global 10-m LULC datasets: ESA WorldCover, ESRI LandCover, and Google DynamicWorld. This paper does not introduce a new mapping methodology but focuses on analysing the behaviour and consistency of the COTQ using complementary evaluation approaches. All products are harmonized under a common legend and compared using structural indicators (object-size distributions, shape complexity, Adjusted Rand Index, and Intersection over Union), spectral separability metrics derived from Sentinel-2 reflectance data, and a targeted photo-interpretation of disagreement areas. The analysis is conducted over eight Sentinel-2 tiles selected to represent the main bioclimatic domains of Québec, from temperate and boreal forests to northern tundra environments. The results show that the COTQ exhibits structural and spectral characteristics most similar to ESA WorldCover among the reference global products, while revealing systematic differences linked to class definitions and thematic priorities, particularly for urban areas, wetlands, and rocky or cryptogamic surfaces. This multi-criteria evaluation provides an objective characterization of the COTQ product and clarifies its relative positioning with respect to existing global land-cover datasets for operational land monitoring in Québec. 

---
# NeMo Data Designer: An Extensible Framework for Multimodal Synthetic Data Generation 

**Authors**: Johnny Greco, Nabin Mulepati, Andre Manoel, Eric Tramel, Kirit Thadaka, Mike Knepper, Dhruv Nathawani, Dane Corneil, Yev Meyer, Alex Watson, Maarten Van Segbroeck  

**Link**: [PDF](https://arxiv.org/pdf/2609.17699)  

**Abstract**: We present NeMo Data Designer (NDD), an open-source, general-purpose framework for multi-modal synthetic data generation (SDG). Designed to be intuitive to use, NDD provides a declarative configuration format in which human and/or agent users define each dataset column, with column types spanning text, code, structured outputs, images, embeddings, and statistical samplers that are explicitly configured to steer dataset diversity. Additional column types and functionality can be introduced using the framework's flexible plugin system. NDD's configuration is an inspectable artifact, supporting workflow sharing and reproducibility. SDG is an inherently iterative process. NDD therefore builds a preview-and-revision loop into its core workflow, allowing users to generate and inspect a small number of records, refine the specification, and rerun generation at full scale. At runtime, NDD resolves dependencies, schedules calls to user-provided model endpoints, and retries failed requests. We describe NDD's architecture and programming model and present case studies spanning structured, agentic, multimodal, and domain-specialized tasks, including datasets used in Nemotron model development and in production enterprise deployments. 

---
# GVD: Governed Versioning and Deduplication for Document Repositories 

**Authors**: Mohammadreza Sediqin, Shivali Dalmia, Sumukha Thoppanahalli, Abhishek Mukherji  

**Link**: [PDF](https://arxiv.org/pdf/2609.17696)  

**Abstract**: Document repositories evolve continuously. Guidelines and policies are revised, superseded, and re-uploaded, so the same content recurs in different wording and newer versions refine or contradict earlier ones. These inconsistencies belong to the growing collection rather than to any single document, yet existing work treats versioning, duplicate detection, and contradiction detection as isolated pairwise tasks and stops once a pair is labeled. We present GVD (Governed Versioning and Deduplication), a framework that unifies cross-document version linking with rule-level conflict resolution under an auditable update policy. Incoming documents are assigned to version families through bidirectional rule alignment, and their rules are compared against the family memory to identify duplicates, contradictions, asymmetric refinements, and new knowledge, with Counterfactual Span Probing (CSP) resolving related pairs that inference misclassifies as neutral. Relation-specific policies suppress duplicates and escalate only consequential changes for review, retaining version lineage as an audit trail. The pipeline runs fully locally, with no large language model. On 120 enterprise documents processed as 140 ingestions across 59 version families, GVD reaches an F1 of 0.97 for version-family construction and 0.94 for rule-level consistency, with CSP raising rule consistency from 0.90 to 0.94. 

---
# GraphEcho: Structural Redundancy and Evidence Provenance in LLM Graph Agents 

**Authors**: Sikun Wang, Yixi Zhou, Lei Fan, Fan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.17695)  

**Abstract**: A large language model (LLM) agent can follow more graph paths without acquiring more independent evidence. GraphEcho tests whether agents mistake these repeated encounters for additional corroboration. The benchmark varies path counts and evidential origins while holding evidence content fixed, and evaluates both judgments and active exploration. Controlled synthetic experiments reveal model-dependent judgment shifts, but redundant supporting paths increase the share of repeated walks across all evaluated frozen agents. Provenance-aware post-training (PAPT) reduces revisits and improves synthetic accuracy, yet covers fewer distinct sources. On scientific claims, it continues to reduce repetition while accuracy declines. These findings expose a gap between efficient exploration and effective evidence use: an agent can learn to stop repeating itself while overlooking information it needs. GraphEcho provides a controlled way to evaluate both what graph agents conclude and whether their exploration reaches distinct evidential sources. 

---
# CapMem: A Benchmark for Caption-Based Episodic Memory in Egocentric Video 

**Authors**: Dingli Liang, Yiqiao Xie, Yukai Huang, Zhaokai Wang, Weitong Cai, Guangwen Feng, Jifei Song, Zhensong Zhang, Hang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.17688)  

**Abstract**: Wearable assistants require episodic memory over egocentric video, yet current vision-language models face bounded frame budgets, growing visual-token costs, and long-context retrieval failures. Under these practical constraints, we study whether textual captions can serve as reusable episodic memory. We define the Episodic Memory Video Caption QA task and introduce CapMem, a human-annotated benchmark with 75 videos totaling 33.7 hours, and 1,000 multiple-choice questions across 16 scenarios. On long videos (>20 min), full-coverage CaptionQA with 30s and 60s caption windows outperforms direct VideoQA for 10/12 and 8/12 models, respectively. On the same video subset, a matched-frame control across six Qwen models retains mean accuracy gains of 3.22 and 2.55 points, respectively. Our caption-guided retrieve-and-verify harness further improves accuracy by up to 5.3 points. These results support the effectiveness of caption memory for episodic reasoning over long egocentric video. 

---
# What You Can't See Is Still What You Learn: A Preregistered Sixty-Society Confirmation That Evidence Masking Drives Compositional Generalization 

**Authors**: Narcis Marincat  

**Link**: [PDF](https://arxiv.org/pdf/2609.17637)  

**Abstract**: Restricting what a module can read may improve what a system learns to compute. We test this in a preregistered confirmation with sixty four-cell systems sharing a frozen language-model backbone and communicating through learned continuous packets. Five conditions vary evidence masking, ownership markers, and replacement of foreign evidence with neutral filler, across six initialization clusters, each with two data orders, on one fresh task world. With markers available in both regimes, masking improved accuracy on held-out two- and three-operation compositions by median paired differences of 0.846 and 0.859; all twelve pairs cleared the required margins, and the full preregistered behavioral criterion passed. The unmarked replication also passed. No globally visible system passed the marker-following check, so the effect of usable role information remains unresolved. The filler condition yielded seven full generalizers, but its decomposition criteria were inconclusive. Packet interventions in all eighteen audited masked systems followed the predicted intermediate-value changes on eligible cases; these finite, success-conditioned audits do not establish mediation. The results confirm a large advantage of the tested masking regime, while leaving its finer attribution and generality open. Protocols, results, and checkpoints are public. 

---
# Physics-Constrained Digital Twins for Sensor Integrity in Urban Pedestrian Flow: Detecting Stealthy False Data Injection with Conformal Guarantees 

**Authors**: Oscar Mogollon Gutierrez, Fatemeh Ghasemi, Mohammadhossein Homaei, Andres Caro, Mar Avila  

**Link**: [PDF](https://arxiv.org/pdf/2609.17635)  

**Abstract**: City pedestrian counting systems now feed economic indicators, planning decisions and safety operations, yet the twins built on top of them treat the incoming stream as ground truth. We study what happens when it is not. We formalise stealthy false data injection for city-scale pedestrian sensing, where the map from latent flow to observation is far more rank deficient than in the power and water networks for which stealth has been characterised. Our twin estimates directed flows on the pedestrian street graph, assimilates counts through a learned graph-localised gain, and is trained against a flow conservation residual that couples metered and unmetered segments. Detection combines the innovation with that residual, and the alarm threshold is set by adaptive conformal calibration rather than by hand. To measure what the physics buys, we define the attack margin, the relative reduction in worst-case corruption of the estimated flow field, achieved against a white-box adversary that optimises directly through the twin. On six years of Melbourne data the margin reaches 0.54 against a single compromised device and falls to 0.19 when a third of the fleet is compromised, on a network where only 1.18 per cent of walkable segments are metered. Replacing the street graph by a distance graph collapses it to 0.09, which shows that the gain comes from the conservation law rather than from locality. 

---
# One Color Preprocessing Improves DSATUR 

**Authors**: Adam Nouira, Lucas Isenmann  

**Link**: [PDF](https://arxiv.org/pdf/2609.17633)  

**Abstract**: The Graph Coloring Problem (GCP) is NP-hard and DSATUR stands as one of the fastest heuristics for it despite producing colorings that typically use more colors than state-of-the-art coloring algorithms. We propose SSLD (Semidefinite Spectral Learning with DSATUR), which improves DSATUR by preprocessing a first good color class before letting DSATUR complete coloring the rest of the given graph. We obtain this color class from a Semidefinite Programming (SDP), similar to an SDP used to compute the Lovász theta number. To the best of our knowledge, SSLD is the first approach to improve DSATUR by preprocessing through fixed color classes. We evaluate SSLD against DSATUR and against a naive 1-color-class preprocessing algorithm on DIMACS instances, random graphs (Erdős--Rényi, Watts-Strogatz, Barabási--Albert), Frequency Assignment and Job Shop Scheduling instances. SSLD matches or beats DSATUR in almost every case across over 1600 benchmark instances, and out performs the naive GISD baseline, allows us to confirm the value brought by the SDP-guided choice of the first color class. This quality comes at a runtime cost of roughly 195 times slower that DSATUR, but demonstrating that SDP-guided preprocessing of a first color class is a direction for future improvements. 

---
# EvolveTrade: Experience-Driven Policy Refinement for Self-Evolving LLM Trading Agents 

**Authors**: Sehee Kim, Yumin Choi, Minki Kang, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2609.17632)  

**Abstract**: Large language model (LLM) trading agents can combine market data, news, and executable analysis, but their behavior is often controlled by static hand-written tool-use policies that are fixed before deployment. This limits their ability to adapt how they gather evidence, invoke tools, verify signals, and manage risk under changing market regimes. We introduce EvolveTrade, a self-evolving framework that treats the system prompt of a tool-using trading agent as a text-parameterized policy. After each update interval, a Policy Agent revises this policy using accumulated decision traces and realized portfolio feedback, while keeping the backbone LLM fixed. The updated policy is then used for the next batch of trading decisions, enabling the agent to refine its information-acquisition and portfolio-construction procedure over time. Experiments across multiple market regimes and two LLM backbones show that EvolveTrade often improves Sharpe Ratio and Cumulative Return over fixed-policy LLM baselines, achieving the improved SR and CR in most evaluated settings. Behavioral analyses further show that self-evolved policies increase code-mediated analysis and activate regime-relevant computations; case-level policy-to-return attributions trace how policy-induced allocation changes contribute to realized return differences. These results suggest that adapting the reusable procedure governing tool use is a key direction for building more robust LLM trading agents. 

---
# Making AI-Assisted Claims Independently Challengeable: Publication Authority and a Protocol for Falsifiable Publication Records 

**Authors**: Torsten Olivi Tiltack, Yifei Dong, Kun Yu, Xu Wang, Wei Liu, Jianlong Zhou, Ren Ping Liu, Fang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.17631)  

**Abstract**: AI-assisted claims can appear authoritative when evidence, analysis, human authorization, presentation, and correction history refer to different states. Provenance, attestation, and transparency expose history but alone do not specify the publication transition examined here. We develop Publication Authority as an exact-state, non-transferable, single-use publication capability and instantiate it in PAC-2026 (Publication-Accountability Calculus), a machine-readable AIJIM Protocol candidate. We evaluate its fourth bounded semantic freeze (SF-4), a fixed-profile specification designed for replaceable bindings. Six obligations govern evidence, runs and artifacts, measurement disclosure, authorization, surface correspondence, and lifecycle continuity. Each yields a target-bound witness, localized counterexample, or localized unverifiability; none can compensate for another. Only a fresh, complete all-pass record derives the permit consumed by one atomic publication transition. We use identity vectors, adversarial cases, finite models, and historical implementations. Ten models explored 110,764 safe reachable states; 76 unsafe configurations produced the expected violation or observer countermodel. A reader surface passing its correspondence check cannot authorize publication unless the accepted record admits that surface. SF-4 separates evidence horizon from verification time and rejects an authentic but causally invalid authorization. A historical predecessor path reproduced 17 frozen authorization-successor outcomes. A later in-house, instance-blind test of known case classes matched all 183 scored expectations; same-host package execution reproduced its 240 archived observations. Results support internal coherence, bounded safety, fault sensitivity, and limited constructibility, but not factual truth, general refinement, blind interoperability, field efficacy, or standards status. 

---
# Objective vs. Search: Decomposing What Makes a Good Tokeniser 

**Authors**: Ahmetcan Yavuz, Clara Meister, Tiago Pimentel  

**Link**: [PDF](https://arxiv.org/pdf/2609.19145)  

**Abstract**: Two dominant tokenisation algorithms are used by modern language models: byte-pair encoding (BPE) and UnigramLM. These differ along two orthogonal axes: their optimisation objective (compression vs. log-likelihood) and their search procedure (bottom-up merging vs. top-down pruning). Existing comparisons confound these axes, making it unclear whether their observed differences stem from what is being optimised vs. how it is being optimised. We disentangle the two by introducing two new tokenisation algorithms that complete this 2x2 design space: BottomUpLL, a bottom-up likelihood-based tokeniser, and TopDownComp, a top-down compression-based tokeniser. We train language models with tokenisers produced by each algorithm, varying: model size, vocabulary sizes, and domain (English-only vs. multilingual). Evaluating models on bits-per-byte, we find that the search procedure -- not the objective -- is the dominant factor: bottom-up tokenisers consistently achieve lower bits-per-byte in most settings. Evaluating models on the BLiMP task, however, shows no consistent relationship between design choice and performance. Overall, our results disentangle the effect of tokeniser design choices on language modelling performance, offering concrete guidance for their more principled construction. 

---
# A Zeroth-Order Paradigm for LLM Preference Alignment 

**Authors**: Peter Chen, Xi Chen, Wotao Yin, Tianyi Lin  

**Link**: [PDF](https://arxiv.org/pdf/2609.19144)  

**Abstract**: Direct preference alignment methods are widely used to align large language models (LLMs) with human preferences because of their computational and memory efficiency. However, likelihood displacement motivates alternative ways to extract information from preference pairs with small likelihood margins. In this paper, we propose and analyze Comparison-based Preference Optimization (ComPO), a zeroth-order alignment method based on comparison oracles. ComPO extracts directional information from these pairs without directly optimizing a differentiable preference loss on them. We establish a convergence guarantee for its basic offline scheme under smoothness, gradient sparsity, and compatibility between the oracle and a latent objective. We further introduce online ComPO, which retains the offline comparison mechanism and uses unlabeled policy generations for reverse-KL control relative to a reference policy. Following the coverage perspective of preference fine-tuning, we establish a performance guarantee for a basic constrained scheme under local coverage and in-distribution pairwise reward accuracy. Experiments on Mistral, Llama, Gemma-2, Qwen3, and Gemma-3 models demonstrate improvements over existing direct alignment methods, including length-controlled win rates, with pair-level diagnostics providing evidence consistent with mitigating likelihood displacement. 

---
# Dreaming the Sound of Contact: Leveraging Video and Audio Generation for Zero-Shot Force-Aware Manipulation and Data Generation 

**Authors**: Guanhua Ji, Tianyu Li, Dayoon Suh, Yuqian Zhang, Boyan Zhang, Nadia Figueroa  

**Link**: [PDF](https://arxiv.org/pdf/2609.19137)  

**Abstract**: Recent advances in video generation allow robots to learn manipulation trajectories from generated videos. However, these approaches produce purely kinematic trajectories that lack force information, causing failures in contact-rich tasks where appropriate contact forces are essential for success. In this work, we explore augmenting generated video with audio to shape a bounded, time-varying desired-force profile using the loudness of generated contact sounds. We present a pipeline that jointly leverages generated video and audio to derive motion trajectories and corresponding desired-force profiles from a structured natural-language task prompt. We execute these force-aware trajectories on a Franka Panda robot using a closed-loop force regulator that tracks the audio-shaped force profile during contact. We evaluate our pipeline on multiple tasks that require making contact and demonstrate successful manipulation where a kinematic-only baseline fails. We also use the pipeline as a data generation engine to train policies that achieve the tasks in a closed-loop manner. Project website, videos, and dataset: this https URL 

---
# Affora: A Design System for Agent-Friendly Interfaces 

**Authors**: Jin Gao  

**Link**: [PDF](https://arxiv.org/pdf/2609.19125)  

**Abstract**: Computer-use agents increasingly operate software designed for people, but interfaces often leave actions or task state unclear to machine readers. We present Affora, a design system that supports both readers while preserving visual freedom and familiar human workflows. Three controlled studies examine component implementations, visual variation, and interaction-design principles. Their findings inform guidance from individual components to complete sites, supported by reusable implementations and executable checks. Agent performance depends on the interaction meaning available through its interface representation; substantial visual variation remains possible when that meaning is preserved. Evaluation on independently authored interfaces shows gains where Affora addresses existing deficits, but limited effects where those deficits are absent or outside its coverage. A workflow case provides preliminary evidence of reduced interaction cost. Affora connects user experience and agent experience through a shared interface rather than a separate agent-only surface. 

---
# rMuscle: Robotic Muscle Memory for Efficient Vision-Language-Action Model Inference 

**Authors**: Kaijun Zhou, Zhiyang Li, Le Chen, Jinyu Gu  

**Link**: [PDF](https://arxiv.org/pdf/2609.19104)  

**Abstract**: Factory work is a promising early scenario for embodied AI: assigning repetitive manual jobs to robots has clear economic payoff, and a structured station keeps the jobs tractable for current policies. Vision-Language-Action (VLA) models now dominate as the policy paradigm for these robots. The inference latency of VLA models directly affects robot responsiveness and motion smoothness. However, existing VLA inference frameworks do not fully exploit the characteristics of embodied workloads or account for the distinct bottlenecks across different stages of VLA inference.
In this paper, we first characterize embodied workloads and identify substantial task similarity across repeated robot executions. We further find that such similarity extends beyond observations and action trajectories to internal model states. Drawing on these observations, we present rMuscle, a real-time VLA inference framework inspired by human muscle memory. It exploits cross-execution similarity through a dual-phase muscle-memory cache. The Context Cache reuses visual-token outputs to reduce computation, while the Action Cache reuses neuron activation patterns to reduce weight accesses. We keep both the cache memory footprint and access overhead low through online cache recomputation, sliding-window cache retrieval, and mask sharing across consecutive denoising steps. rMuscle achieves 1.29-1.42X speedup on RTX 4090 and Jetson Thor across LIBERO, RoboTwin, and physical manipulation tasks, while maintaining the original success rates on real-world robots. 

---
# Prepared Or Unprepared? Evaluating Healthcare Workforce Readiness for Clinical Adoption of Artificial Intelligence in Nigeria 

**Authors**: Abbas M. Rabiu, Abdulrazaq A. Zubair, Um-mulkhairi Ibrahim, Tolulope Olusuyi, Shaheeda Farouq, Safwan M. Dafi, Adaobi C. Emegoakor, Yewande Gbadamosi, Maruf Adewole  

**Link**: [PDF](https://arxiv.org/pdf/2609.19096)  

**Abstract**: Artificial intelligence (AI) is increasingly integrated into healthcare systems worldwide, yet its successful clinical adoption depends critically on workforce readiness, particularly in low- and middle-income countries (LMICs) where infrastructural and training gaps persist. This cross-sectional study evaluated awareness, attitudes, preparedness, and barriers to AI adoption among 761 healthcare professionals across multiple disciplines and practice settings in Nigeria. Data were collected between December 2025 and March 2026 using a structured, validated questionnaire. Overall awareness of AI in healthcare was high (92.6%); however, objective knowledge and self-reported preparedness remained limited, with 40.9% reporting low or very low knowledge and only 63.0% feeling adequately prepared. Willingness to adopt AI was high: 92.5% expressed interest in training, and 78.7% supported inclusion of AI education in undergraduate curricula. Key barriers included lack of training (84.7%), poor infrastructure (71.1%), high cost of AI tools (61.0%), fear of job displacement (60.6%), ethical concerns (52.9%), and data privacy concerns (52.7%). Significant differences in preparedness were observed across geopolitical zones (chi-square (5) = 24.28, p < 0.001), and awareness differed across professional groups (chi-square (6) = 68.38, p < 0.001). Attitudes toward AI differed significantly across professional groups (F = 3.32, p = 0.003), with professionals who felt prepared demonstrating more positive attitudes (mean = 3.74) compared to those who did not (mean = 3.46). These findings reveal a critical disconnect between high awareness and actual readiness, underscoring the need for targeted training, infrastructure investment, and clear implementation frameworks to bridge the gap between AI technological potential and clinical reality in resource-constrained settings. 

---
# Reporting Practice Matters: The Impact of Reference Choice on Chest X-ray Report Evaluation 

**Authors**: Daniel P. Jeong, Charles Q. Li, Hossein Hosseiny, Nitya M. Bhalla, Fatma Uyar Morency, Pradeep Ravikumar, Zachary C. Lipton, Michael Oberst  

**Link**: [PDF](https://arxiv.org/pdf/2609.19093)  

**Abstract**: Radiologists follow heterogeneous reporting practices. Two radiologists examining the same image and identifying the same clinical findings might nevertheless compose superficially distinct reports, varying in terminology, shorthand, formatting, and level of detail. These variations in reporting norms represent an under-appreciated obstacle in efforts to evaluate AI-based radiology report generation (RRG) models, where machine-generated reports are typically assessed based on their concordance with human-generated references. In this paper, we quantify the sensitivity of established evaluation metrics to variations in reporting practices, revealing impacts large enough to alter the rankings of models. We introduce a radiologist-informed taxonomy of variations in radiology reporting practice and a method (ReRef) that rewrites reference reports along the axes of our taxonomy while preserving clinical interpretation. For instance, when comparing the performance of nine RRG models on MIMIC-CXR using RadCliQ-v1, condensing the discussion of normal findings in the reference reports causes Libra to drop from first to second place while CheXOne rises from third to first. Our results suggest that many current metrics fail to decouple clinical interpretation from conformity to reporting practices and that choosing the ``right'' references that accurately reflect the desired reporting practices can be important in practice. To support future research, we release MIMIC-CXR-Ext-ReRef, a radiologist-validated dataset of 120 (original, alternative) reference report pairs derived from MIMIC-CXR. 

---
# Securing quantum error correction against misleading advice from AI agents 

**Authors**: A. Barış Özgüler  

**Link**: [PDF](https://arxiv.org/pdf/2609.19090)  

**Abstract**: Can an attacker turn influence over an artificial intelligence (AI) adviser into a harmful quantum error-correction update? We identify an ambiguity in passive syndrome records that obstructs recovery selection, then show how additional calibration measurements support certified recovery updates under uncertainty and drift. In an odd-distance square toric code with error-free preparation, syndrome measurements, and recovery operations, opposite coherent $X$ rotations produce identical passive syndrome-history distributions. Yet a fixed phase correction can help at one sign and harm at the other. A terminal logical measurement on known encoded calibration states supplies the missing sign information. A separate evaluator accepts an update only when calibration uncertainty and a justified drift bound certify improvement over the current recovery, without assuming that the adviser recommends correctly. In simulated advice attacks, calibration-confidence checks reject harmful proposals while retaining beneficial updates under honest advice. We derive sufficient limits on calibration age that require improvement through deployment. In matched simulations, a validated channel-specific bound retains more beneficial updates than the general bound after accounting for evaluation time, while preventing the tested harmful activations under the stated drift assumption. A separate surface-code experiment includes stochastic circuit faults and noise changing during acquisition. Deterministic controllers achieve at least as many beneficial updates with the same observations. Violating the drift assumption permits harmful acceptance in the toric experiment. The results identify information required for recovery selection, establish conditional guarantees against harmful updates, and quantify the recovery improvements forgone through conservative acceptance. 

---
# Probabilistic Linear Explanations 

**Authors**: Frederic Koriche, Jean-Marie Lagniez, Chi Tran  

**Link**: [PDF](https://arxiv.org/pdf/2609.19077)  

**Abstract**: Formal explainability provides mathematically grounded justifications for individual predictions. However, abductive explanations often exceed human cognitive limits by involving too many features, while probabilistic relaxations have remained largely limited to categorical classification. We present a unified framework for probabilistic explainability based on sparse, anchored linear models, applicable to both binary classification and continuous regression. By mapping instances to the Boolean hypercube, our linear explanations strictly generalize subset-based approaches: they capture both the magnitude and direction of feature contributions while enforcing a prescribed sparsity budget $k$. We show that minimizing the relevance error for such explanations is \ClassNPPP-hard when the underlying model is a neural network, and we relate this intractable objective to a tractable surrogate---the fidelity error. For a parameterized family of local distributions, the relevance error of any $k$-sparse explanation is bounded by its fidelity error up to a multiplicative factor that remains small locally. We address the resulting empirical problem using two complementary approaches: a Mixed Integer Programming (MIP) formulation that yields provably optimal empirical solutions while maintaining polynomial sample complexity, and a polynomial-time Iterative Hard Thresholding (IHT) algorithm with provable approximation guarantees. Empirical evaluations show that, unlike state-of-the-art baselines such as LIME and MAPLE, our explanations satisfy both the anchoring and sparsity constraints by construction, while consistently achieving lower relevance error. 

---
# Double descent is the principle of least action 

**Authors**: Congzhou M Sha  

**Link**: [PDF](https://arxiv.org/pdf/2609.19076)  

**Abstract**: The test error of a model plotted against its number of parameters $d$ falls, peaks when the model can just fit the training data, and falls again, exhibiting the double descent phenomenon. We explain the phenomenon with statistical mechanics. The training trajectory of a stochastic gradient-based method is a particle wandering over the energy landscape of the training loss at an induced temperature $T$, and a run that has equilibrated visits every parameter vector of a given training loss equally often, the fundamental postulate of statistical mechanics, with probability given by the Boltzmann distribution. Because training starts at an initial point and has only finite time to diffuse, it carries an effective weight decay, which makes every parameter a quadratic degree of freedom. The equipartition theorem then distributes the energy among the $d$ degrees of freedom in shares of $T/2$, so at a fixed training loss adding parameters lowers the temperature and drives the Boltzmann distribution toward the stationary path. Finally, adding parameters can only lower the $L^2$ norm of the stationary path, so a solution sampled at fixed loss is less likely to be large with increasing $d$, effectively increasing weight regularization. 

---
# RLLBC-Lib: An Educational Code Library for Reinforcement Learning and Learning-Based Control 

**Authors**: Bernd Frauenknecht, Emma Cramer, Artur Eisele, Paul Kruse, Lukas Kesper, Jonas Hertrampf, Ramil Sabirov, Jyotirmaya Patra, Johannes Berger, Paul Brunzema, Friedrich Solowjow, Sebastian Trimpe  

**Link**: [PDF](https://arxiv.org/pdf/2609.19074)  

**Abstract**: Reinforcement learning (RL) is an exciting concept as well as a remarkable success story worth sharing. However, RL builds on rather complex interactions between different objects that play out over several cycles. Such dynamics are often best explained with an easily accessible implementation. We present RLLBC-Lib, a carefully crafted code library with the goal of lowering the entry barrier for students and other learners of RL in the context of learning-based control. At its heart, RLLBC-Lib comprises a comprehensive library of tabular RL approaches to enforce a clear understanding of the theoretical foundations. A deep RL library follows the same design principles, underscoring the parallels between simple tabular and state-of-the-art deep RL approaches. Additionally, RLLBC-Lib provides a collection of implementations illustrating core RL principles and contrasting RL to other learning-based control approaches. Finally, RLLBC-Lib provides an ideal basis for creating programming assignments with automated grading. 

---
# MIRAGE: How Conversation State Shapes Historical Evidence Use in Multimodal Personal Agents 

**Authors**: Yu Liu, Wenxiao Zhang, Cheng Hu, Cong Cao, Fangfang Yuan, Xinyu Wang, Jin B. Hong, Yanbing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.19059)  

**Abstract**: Multimodal large language model (MLLM) agents are increasingly used as personal assistants for long-running tasks. Their utility depends on continuity: agents must retrieve and use earlier evidence across dialogue, files, and workspace state. However, agents can generate plausible answers even when access to that history has degraded, causing outcome-only evaluation to overestimate true evidence use. We present MIRAGE (Multimodal Interaction Retrieval, Attribution, and Grounding Evaluation), a controlled study of historical evidence use under conversation-state variation in multimodal personal agents. MIRAGE holds evidence objects, questions, and scoring fixed while varying only conversation state, and evaluates whether an agent can determine answerability, recover the correct source, and answer from it. Across seven frontier and open-weight multimodal backbones, we find that: 1) pre-compaction depth and post-compaction continuation form distinct, non-monotonic failure regimes rather than a single degradation curve; 2) open-weight models rely heavily on context continuity and are reluctant to spontaneously switch to tool-mediated retrieval when provenance fails; and 3) retrieval pressure improves source attribution in deep pre-compaction states for tool-compliant models, but consistently regresses after compaction, where stored evidence has already degraded. These findings show that historical evidence use should be evaluated under state variation, rather than inferred from outcome-only correctness. 

---
# TalkMatrix: Generating Character Dialogue that is Both Consistent and Diverse 

**Authors**: Ayuto Tsutsumi, Yuu Jinnai  

**Link**: [PDF](https://arxiv.org/pdf/2609.19022)  

**Abstract**: Candidate-based decoding typically selects a completion for each prompt independently, but many applications require a collection of outputs that satisfies global, non-decomposable requirements. We formulate this setting as structured multi-prompt, multi-completion selection: given a candidate pool for every prompt, select one completion per prompt to optimize a collection-level objective. We instantiate the problem in character dialogue, where each character should remain consistent across situations, each line should fit its situation, and characters and situations should remain distinguishable. Our method, TalkMatrix, generates multiple candidates for every character--situation pair and jointly selects a complete matrix using four embedding-based consistency and diversity objectives. Because a weighted sum can improve some dimensions by sacrificing another, TalkMatrix maximizes the worst-performing objective through a two-level minimax formulation. We approximately optimize the resulting discrete objective with multi-start coordinate ascent, and compare it with local, partial-matrix, and generic combinatorial search baselines. We run experiments on $50$ synthetic role-playing scenarios and $25$ curated board game scenarios where multiple characters interact in predefined situations. An LLM-as-a-judge rates matrix-level selection higher than random and independent cell-level selection baselines. These results show the value of structured selection for globally controlled dialogue generation, while our empirical validation remains specific to role-playing scenarios. 

---
# Tabular Deep Learning vs Classical Machine Learning for Urban Land Cover Classification 

**Authors**: Muntasir Tabasum, Tanpia Tasnim, Md. Ekramul Islam, Al Zadid Sultan Bin Habib  

**Link**: [PDF](https://arxiv.org/pdf/2609.19010)  

**Abstract**: Urban Land Cover (ULC) classification plays a crucial role in urban planning, environmental monitoring, and sustainable development. We study this task using the ULC dataset from the UCI Machine Learning Repository, which includes tabular features derived from high-resolution aerial imagery across nine classes (e.g., roads, trees, grass, water). The dataset presents typical remote sensing challenges, including high dimensionality, heterogeneous features, and class imbalance. In a unified, reproducible pipeline, we benchmark classical machine learning models (e.g., Logistic Regression, SVM, Random Forest, XGBoost, CatBoost) against Tabular Deep Learning (TDL) models (TabNet, FT-Transformer, TabTransformer, TabSeq, and 1D CNNs). To address class imbalance, we employ weighted cross-entropy loss for TDL models and evaluate performance using accuracy, macro-precision, macro-recall, macro-F1, AUC-ROC, and confusion matrices. Our results show that while tree ensembles remain strong general baselines, TDL models can match or exceed their performance when non-linear interactions are significant and imbalance handling is effective, providing complementary advantages for urban land cover mapping. See code: this https URL 

---
# WordPolo: Evaluating Language Models Through Iterative Semantic Feedback 

**Authors**: Tyler McDonald, Ali Emami  

**Link**: [PDF](https://arxiv.org/pdf/2609.19006)  

**Abstract**: Large Language Models (LLMs) and Large Reasoning Models (LRMs) are typically evaluated on challenging benchmarks through dataset accuracy alone, providing no insight into the quality or faithfulness of their reasoning processes. We present WordPolo, a word-finding task where participants must discover an unknown target word using semantic similarity feedback. Players start with zero knowledge, make guesses, and receive distance scores (1 = correct, higher = further away). Success requires interpreting scores to navigate semantic space and systematically narrow the search. This design makes iterative reasoning and adaptive search strategies both directly observable and necessary for success. We evaluate recent LLMs (GPT-4.1, Llama 4, Claude 3.5 Haiku, Qwen 3), LRMs (o4-mini, Deepseek-R1), humans, and a novel heuristic on 1,500 puzzles. Beyond solve rates (which range from 4% to 62%), we introduce progression-based metrics that reveal models often make meaningful progress, insights that accuracy alone would miss. Our analysis shows how reasoning models can be hindered by overthinking and underthinking, while successful models exhibit human-like strategies. WordPolo demonstrates the need for benchmarks that test both reasoning process and outcomes, providing holistic measurements of model capabilities. Our code and dataset can be found at this https URL. 

---
# BadQubits: An LLM-Based Framework for Static Pre-Execution Detection of Structurally Harmful Quantum Circuits 

**Authors**: Justin Woodring, Lamine Noureddine, Aisha Ali-Gombe  

**Link**: [PDF](https://arxiv.org/pdf/2609.18965)  

**Abstract**: This paper presents BadQubits, an LLM-based framework for static pre-execution detection of structurally harmful OpenQASM 2.0 circuits. The framework targets physical-execution-layer threats by analyzing submitted circuits prior to runtime, where dynamic inspection is constrained by measurement irreversibility and the exponential cost of classical quantum-state simulation. We evaluate four code-understanding LLM architectures on a dataset of 1,500 circuits consisting of 1,000 benign programs from MQTBench[33] and 500 synthetic attack circuits derived from three documented physical-layer threat primitives. Our fine-tuned Qwen Coder 2.5 7B model achieves 92.67% classification accuracy and 96.1% harmful-circuit recall. Two of the four evaluated base models fail to generalize under constrained LoRA fine-tuning, indicating that architecture-aware model selection is a necessary design consideration rather than a minor tuning choice. To characterize what the detector has learned, we compare it against a bag-of-gates CNN under progressive confound removal and adversarial syntactic perturbation. The CNN's harmful-circuit recall drops from 100% to 17%, while the fine-tuned LLM decreases only from 96.1% to 91.2%. We attribute this gap to the sequential structure retained in token-level LLM inputs but discarded by histogram-based baselines. A correlation analysis further shows that model decisions track threat-defining features, specifically SWAP density and measurement timing, rather than generator-specific artifacts such as register naming. 

---
# Transcribe, Then Reason: Two-Pass Decomposition for Multimodal Review 

**Authors**: Bojie Li, Noah Shi  

**Link**: [PDF](https://arxiv.org/pdf/2609.18958)  

**Abstract**: The natural way to review a long recording or document with a multimodal model is to hand it the raw source and ask for a review in one call. We show that this quietly fails: the model satisfices, dropping roughly a third of the content and embellishing the rest. The failure is not perception--almost all of the dropped content reappears when the same model is simply asked to transcribe the source. The bottleneck is generation under load: a single pass cannot perceive, reason over, and write a long faithful review at the same time, because doing all three competes for one output. We rule out the obvious alternatives. It is not the modality: models read text and an image of the same text equally well. And it is not merely a matter of thinking harder: giving the single pass a far larger reasoning budget does not recover the lost content, because the model spends that budget planning a review rather than writing the source down. What works is to split the labor across two same-weights passes--first transcribe, then review the transcript--so each step gets a full output budget of its own. This transcribe-then-review decomposition improves both faithfulness and coverage across a 21-source suite. The benefit is not uniform: we observe that it helps most where the one-pass baseline is weakest and little where that baseline is already strong, a pattern that also tracks the source's length and modality. Decomposition comes with two failure modes--the review pass running out of room on very long sources, and confabulating from memory once the grounding source is removed. 

---
# Automated Dental Caries Segmentation in Panoramic Radiographs Using Dual-Stage Deep Learning 

**Authors**: Jihun Kim, Kyeonghun Kim, Jong-yeol Lee, Yeongseok Seo, Dohyun Chun  

**Link**: [PDF](https://arxiv.org/pdf/2609.18952)  

**Abstract**: Early detection of dental caries remains challenging due to limitations in traditional diagnostic methods, particularly for proximal lesions in posterior teeth. Deep learning models show promise for automated caries detection but face scalability constraints due to requirements for large volumes of expertly annotated training data. This study presents a dual-stage deep learning framework combining Faster R-CNN for tooth localization with U-Net for pixel-wise caries segmentation in panoramic radiographs. We developed a systematic transformation pipeline to convert large-scale polygon-annotated datasets into high-resolution binary segmentation masks, enabling pixel-wise supervised learning. The framework was trained using both expert-verified datasets and algorithmically processed labels from 3,000 panoramic images. Our approach achieved robust performance with an IoU of 0.9013, Dice coefficient of 0.9482, Recall of 0.9433, and Precision of 0.9774, demonstrating superior accuracy compared to existing methods while significantly reducing false-positive rates. The dual-stage framework effectively addresses data annotation bottlenecks in dental AI applications and demonstrates potential for scalable, automated caries detection systems that can improve diagnostic consistency and support clinical decision-making. 

---
# StableEval Arena: A Cost-Aware Agentic Benchmark for Stablecoin Price Stability Prediction 

**Authors**: Sean Wan, Dongping Liu, Luyao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.18949)  

**Abstract**: We introduce StableEval Arena, a cost-aware benchmark framework for evaluating agentic AI systems on stablecoin peg-risk prediction. StableEval Arena evaluates LLM-backed agentic systems on diagnosing peg stress and forecasting deviations from the one-dollar peg over a hidden seven-day horizon, using leakage-safe historical replay with exchange price-volume data and market-context features. We report two complementary experiment blocks: a 120-case stress-enriched validation block and a 507-case natural-distribution full-arena evaluation block. Across six LLM-backed agent configurations and baselines, StableEval Arena measures prediction quality, calibrated-label behavior, structured-output reliability, latency, token consumption, and estimated inference cost. Rather than ranking agents by accuracy alone, the framework treats trustworthiness as a joint property of forecast quality, operational reliability, and computational cost. The results show a gap between protocol-following reliability and financial-risk reliability: agents reliably produce valid structured outputs at modest measured cost, but still miss most rare severe-stress and sustained-depeg cases. To support auditing and replication, we release the benchmark dataset on Hugging Face and the source code on GitHub. 

---
# Dose-Aware Cold Diffusion with Physics Consistency for Generalizable Low-Dose CT Reconstruction 

**Authors**: Md Imam Ahasan, Guangchao Yang, A F M Abdun Noor, S M Hasan Mahmud, Md Mahfuzur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2609.18943)  

**Abstract**: Reducing radiation dose in computed tomography significantly degrades image quality and poses challenges for accurate and clinically reliable reconstruction. While recent approaches have shown promise for low-dose CT, they often struggle to generalize across continuous and previously unseen dose levels, leading to artifacts and loss of anatomical detail. To address these limitations, we propose Dose-Aware Cold Diffusion (DACD), a physics-consistent reconstruction framework that explicitly models radiation dose as a continuous latent factor within a cold diffusion process. The proposed DACD framework integrates image-based dose-aware perception, multi-scale structural prior extraction, and dose-calibrated step allocation to adaptively guide the denoising trajectory. In addition, an iterative forward-backprojection correction is incorporated into the reverse refinement process to enforce projection-domain data consistency. Extensive experiments on three public benchmarks, including Mayo-2020, Mayo-2016, and LoDoPaB-CT, demonstrate that DACD consistently outperforms state-of-the-art diffusion-based and physics-guided methods in both quantitative accuracy and visual fidelity, particularly under ultra-low-dose conditions. The results show that DACD achieves robust generalization across a continuous range of dose levels, including those unseen during training. 

---
# Social Laws for Multi-agent Coordination in Stochastic Environments 

**Authors**: Rolando Fernandez, Caleb Probine, Tyler Lee, Jeffrey Chen, Erez Karpas, Muhammad Arrasy Rahman, Peter Stone, Ufuk Topcu  

**Link**: [PDF](https://arxiv.org/pdf/2609.18929)  

**Abstract**: In multi-agent environments, coordinating agents to prevent interference and ensure robust individual performance is a critical challenge. Previous research on social laws for multi-agent systems has primarily focused on deterministic, goal-based settings. This paper extends the concept of social laws to stochastic, reward-based environments, proposing a formalism for defining and verifying their robustness under various conditions. We introduce the notion of $\alpha$-robustness, a measure of the guaranteed utility each agent retains while pursuing its optimal single agent policy, assuming all agents obey the social law. We then present an approach for robustness verification of social laws in stochastic settings, based on a reduction to solving a series of Markov decision processes. Empirical evaluations on toy environments illustrate the potential of our framework. 

---
# Higher-order pruning of experts in mixture-of-experts language models 

**Authors**: Alex M. Tseng, Prannay Kaul, Luca Zancato, Wei Xia, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2609.18916)  

**Abstract**: Mixture-of-Experts (MoE) language models suffer from large parameter counts, which create a significant memory bottleneck. Expert pruning is the most direct approach for reducing this parameter count, yet existing methods make pruning decisions for each expert independently, and assume experts' contributions are purely additive. In reality, expert usage in MoEs is inherently cooperative. We derive HOPE (Higher-Order Pruning of Experts), a second-order pruning objective which provably minimizes an upper bound on the error resulting from pruning. We show that REAP (a state-of-the-art first-order pruning method) is a special case of HOPE where interaction terms are ignored. Across three frontier MoE models (up to 122B parameters), two distinct calibration sets, and multiple benchmarks (including math, instruction following, coding, and an agentic suite), we demonstrate that HOPE produces better pruning decisions than existing methods, and its advantage is most pronounced at high pruning rates and on challenging agentic workloads. At 50% pruning, HOPE outperforms all baselines and achieves an average rank of 1.58 out of 5 methods (versus 2.42 for the next-best method, REAP), with gains of up to +6.1% on agentic coding. Over all conditions, HOPE again achieves the best average rank and surpasses every other method in the majority of head-to-head comparisons. By preserving cooperative expert structure that first-order methods ignore, HOPE enables aggressive compression with minimal degradation, particularly on complex tasks where diverse expert combinations are invoked over long sequences. 

---
# Beyond Outcomes: Dual-View Relational Learning for Efficient Agent Benchmarking 

**Authors**: Xinshuai Guo, Junjie Wu, Dolly Deng, Yinghui Li, Hai-Tao Zheng, Suncong Zheng, Maxm Pan  

**Link**: [PDF](https://arxiv.org/pdf/2609.18909)  

**Abstract**: Agent benchmarks are substantially more costly to evaluate than conventional LLM benchmarks. Benchmark compression is therefore a natural solution, yet existing methods primarily model redundancy in task--model final-score distributions, which is important in agentic evaluation. To address this limitation, we analyze large-scale trajectories and identify six complementary process signals that are systematically associated with final agent performance. To disentangle agent performance redundancy from a complete perspective, we propose DualViewEval, an agent benchmark compression method that jointly exploits outcome and process relations to learn an exact-size miniset and predict the full-benchmark scores. Across five agent benchmarks and five representative baselines, DualViewEval achieves the best results in all datasets. With only 20 tasks, it achieves $24\times$--$40\times$ compression on APEX-Agents and BFCL, reducing mean absolute error (MAE) by $14.5\%$--$28.2\%$ over the strongest competitors while improving Kendall's $\tau$ by up to $7.2\%$ relative to EssenceBench on SWE-bench Verified. The selected minisets further reveal capability differences among different agents, providing compact and diagnostic feedback for efficient agentic model development. 

---
# NeuroECG: ECGFounder-Based Deep ECG Representation for EEG-Free Neurological Prognostication After Cardiac Arrest 

**Authors**: Jiaju Gao, Yi Zhao, Chenyang Xu, Yuxi Zhou, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.18891)  

**Abstract**: Neurological prognostication after cardiac arrest commonly relies on electroencephalography (EEG). However, EEG demands high clinical resources. Bedside electrocardiography (ECG) is standard and low-cost. Yet, its value for predicting neurological outcomes remains underexplored. In this study, we propose NeuroECG, an ECGFounder-based deep representation framework for EEG-free auxiliary prognostication. NeuroECG adapts a pretrained ECG foundation model via task-specific fine-tuning. We implement a gradual unfreezing strategy on single-channel bedside monitoring ECG. Multiple ECG segments per patient are encoded into segment-level deep features. These embeddings are aggregated via quantile pooling ($q = 0.24$) and compressed using principal component analysis (PCA). Experiments on 412 ECG-available patients from the multicenter I-CARE database show that the adapted ECGFounder backbone achieves the best performance among ECG-only backbone baselines, with a test AUROC of 0.7333. We further combine the learned deep ECG representation with static clinical covariates. The proposed NeuroECG model achieves a test AUROC of 0.8077 and an AUPRC of 0.8970. These results support deep bedside ECG representations as a useful source of auxiliary prognostic information. Their integration with static clinical covariates improves prediction in an EEG-free setting. The source code is available at this https URL 

---
# ASLEval: Measuring Privacy Exposure Displacement in LLM Agent Sessions 

**Authors**: Guosen Wu, Huizhen Huang, Guoxiong Long, Tao Huang, Chen Hou  

**Link**: [PDF](https://arxiv.org/pdf/2609.18864)  

**Abstract**: Privacy evaluations of tool-using LLM agents often inspect a designated action, final response, or attacker report. These local proxies can miss unauthorized exposure elsewhere in a multi-step session and lack common ground truth across outlets, reports, and tool paths. We introduce privacy exposure displacement, the mismatch between a local evaluation proxy and target-grounded session exposure, and ASLEval, an authorization-aware framework that pre-registers a hidden target set, measures all declared visible exits, and reserves internal traces for diagnosis. Across multiple enterprise-style environments and independently implemented runtimes, we observe three recurring patterns. An expected-outlet-only view misses 46.9% of exposure recovered by the visible-exit union; attacker self-reports combine omissions with high false discovery; and schema-aligned internal evidence usually precedes visible exposure at the request/probe level. Reducing model-visible returns changes this path but can eliminate normal-task success. Independent human review supports the adjudication pipeline while identifying harder console and candidate cases. These findings motivate benchmarks that declare the complete visible boundary, ground claims in pre-specified targets and authorization, and report privacy together with task utility. 

---
# Decodable but Misrouted: Sparse Features Uncover a Readout Gap in Vision-Language Models for Harmful Meme Detection 

**Authors**: Girish A. Koushik, Diptesh Kanojia, Helen Treharne  

**Link**: [PDF](https://arxiv.org/pdf/2609.18860)  

**Abstract**: When a large vision-language model misclassifies a harmful meme, the failure may reflect missing internal evidence or an inability to route represented evidence to its output. We distinguish these cases in Gemma-3 and Qwen3.5 using sparse autoencoders, role-conditioned probes, causal interventions, and recovery experiments across six harmful content benchmarks, with additional Spanish and Hindi-English code-mixed evaluations. Sparse readouts outperform native prediction on all six primary binary tasks: Qwen averages $0.740$ versus $0.432$ native macro-F1, while residual reconstruction reaches $0.486$, whereas Gemma improves from $0.532$ to $0.714$. These differences reflect supervised accessibility rather than a pre-existing, native decision rule, and the most influential token role depends on the task. Under the evaluated score scales, Qwen silent-feature ablation is $24-63$ times more probe-sensitive, whereas routed-feature patching on literal yes/no tasks is $16-140$ times more output-sensitive. Calibration-only routing recovers $93.3$% of the mean gap, and probe-distilled LoRA improves native predictions, although shared multi-task adaptation causes negative transfer. A case study of Gemma-3-12B on Facebook Hateful Memes finds a distributed rank-32 image-prompt interaction, reaching $0.756$ versus $0.685$ native macro-F1. Robustness controls show that the signal extends beyond English, is not explained solely by accompanying OCR, and depends on paired visual evidence. Thus, routing, rather than representation alone, is a recurring bottleneck in harmful meme classification. 

---
# Taming the Agentic RAN: Stability-Guaranteed Arbitration of Autonomous AI Agents in O-RAN 

**Authors**: Seyed Bagher Hashemi Natanzi, Bo Tang  

**Link**: [PDF](https://arxiv.org/pdf/2609.18857)  

**Abstract**: The O-RAN control plane is becoming agentic: autonomous AI agents, deployed as rApps by different vendors, independently close control loops over shared radio resources. We demonstrate on a live O-RAN system that this independence is unsafe. Two agents with individually correct objectives, one protecting a latency SLA and one maximizing utilization for energy efficiency, jointly drive recurring opposing excursions of the shared resource partition that neither produces alone. Existing conflict-mitigation mechanisms presume a statically known application population and cannot govern agents whose behavior emerges at run time. We present AURA, a lightweight arbitration layer that admits agent actions only when they satisfy feasibility invariants, per-variable dwell times, and a deadband, and we prove the arbitrated system converges to a feasible operating point. Implemented on an OpenAirInterface (OAI) testbed with measured one-way latency and throughput, AURA reduces recurring shared-state excursions by more than an order of magnitude (from 8.4 to 0.4 PRB amplitude) and virtually eliminates cross-slice throughput starvation (from 40-55% to 0.3%), while leaving the protected slice's own latency compliance unchanged, a trade-off the convergence guarantee makes explicit. 

---
# GrainSpeech: Less Context, More Detail for Compact Speech Synthesis 

**Authors**: Zitao Liang, Chang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2609.18856)  

**Abstract**: Compact acoustic models face a challenging quality-capacity trade-off. We investigate two factors in this regime: encoder context and Mel-spectrogram supervision. A receptive-field-scaling study shows that expanding self-attention beyond 15 phonemes provides no consistent gains in pitch, energy, or duration prediction. Guided by this finding, we introduce a fixed-receptive-field convolutional encoder that reduces the respective prediction errors by 36.0%, 17.3%, and 3.4%. We further show that directly transferring image-domain gradient-variance supervision restores fine-scale variation but degrades predicted quality, motivating a Mel-specific formulation with axis-specific gradients, overlapping local statistics, and log-domain variance matching. GrainSpeech contains only 264.8K parameters and achieves 17.9x real-time Mel generation on a microcontroller (MCU), while attaining UTMOS scores comparable to substantially larger models with less than 1.5% of their parameters. Source code and demos are available at this https URL. 

---
# Ask the Tool, Don't Guess: Agent Tool Calls Hold Their Progress, and the Serving System Should Read It 

**Authors**: Yipeng Liu, Yingqiang Zhang, Feifei Li, Huanchen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.18849)  

**Abstract**: An agentic request spends substantial wall-clock time waiting for tools, and its KV cache holds GPU memory the whole time. Serving systems decide whether that cache stays, leaves, or comes back by guessing how long the tool will run, from the tool's name, its history, a duration declared before the call, or the engine's own occupancy. We show that no estimate fixed before a call starts can know its duration, and such estimates may not even rank the calls. Meanwhile, the running tool already holds the answer, but the agent stack together with the tool silences it. We propose that tool calls report their progress explicitly while they run, and we measure what that takes. A census of four public agent corpora finds a readable signal in most tool time once it is revealed, in two strengths: a fraction of the work remaining, or an accurate signal that the end is near. A harness recovers it without changing what the agent sees, at no measurable cost to the agent's benchmark score. At the points where a KV cache decision is made, the reported progress is between several times and an order of magnitude more accurate than the best published predictors, and it stays accurate when the environment changes. Plugged into a production engine through a few small hints, it cuts the p90 time to first token (TTFT) after a tool call by 20.7% (HBM only) and 20.8% (HBM + DRAM) against LRU, close to an oracle. A serving system should not guess what its tools can tell it. 

---
# Using OCR Heads to Verbalize Image Semantics 

**Authors**: Sheridan Feucht, Benno Krojer, Sarah Wang, Henry Abrahamsen, Byron C. Wallace, David Bau  

**Link**: [PDF](https://arxiv.org/pdf/2609.18823)  

**Abstract**: How do VLMs map from pixels to semantics? To understand this general question, we focus on a narrow one: studying how VLMs perform optical character recognition (OCR). Across four models, we identify attention heads causally necessary for OCR, and discover that these are in fact general-purpose heads that output interpretable semantic features across all image tokens. For example, pointing these heads at an image token containing the word "bike" causes Qwen3-VL-8B to output "bike," but pointing them at a bird wing causes the model to output the token "feathers." We collapse these heads' attention weights into a single verbalization lens transformation that reveals interpretable semantic features in hidden states across all layers. When combined with projection to vocabulary space, we can obtain interpretable labels starting from layer 0, showing that image representations are in fact aligned with language in early layers. We find that we can also use the inverse of this transformation to edit non-word concepts, e.g., replacing a tractor with a revolver in a naturalistic image, providing causal evidence that this subspace is useful for more than just OCR. Our results are an example of how the study of specific mechanisms can shed light on broader interpretability problems. 

---
# ProgramDistill: From Interactive Web Apps to Verifiable Reference-Guided SWE Tasks 

**Authors**: Jeonghye Kim, Minseon Kim, Young Jin Kim, Matheus Pereira, Marc-Alexandre Côté, Alessandro Sordoni, Xingdi Yuan, Zhengyan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2609.18805)  

**Abstract**: Coding agents are typically evaluated with desired behavior specified through issues or instructions. In practical web development, however, agents may need to infer behavior from working software and implement it in an incomplete application. We introduce ProgramDistill, a benchmark evaluating coding agents on features discovered through interaction with fully functional reference applications. We build ProgramDistill by factorizing applications into features of different granularities, each associated with replayable behaviors executable via its gold patch. Our pipeline, mine-craft-patch, discovers 1,975 replay-verified behaviors across 26 applications and constructs 4,063 tasks without human intervention. Across nine frontier coding agents, GPT-6 Astra and Claude Opus 5 achieve 49.2% and 28.8% success on cumulative workflows in full-application reconstruction. In partial-application reconstruction, success falls from 100% to 64.0% and from 96% to 32% as restoration depth increases from 1 to 8. ProgramDistill thus provides a scalable benchmark with controlled difficulty for evaluating and diagnosing coding agents, and a natural basis for future curriculum-based training. 

---
# A Scalable Framework for Automated NER Annotation Correction in Low-Resource Languages 

**Authors**: Toqeer Ehsan, Thamar Solorio  

**Link**: [PDF](https://arxiv.org/pdf/2609.18739)  

**Abstract**: Poor quality or noisy annotations in Named Entity Recognition (NER), as in any other NLP task, make it challenging to achieve state-of-the-art performance. In this paper, we present a multi-step framework to enhance the annotation quality of NER datasets by employing automated techniques. We propose a frequency-based iterative approach that leverages self-training and a dual-threshold mechanism to enhance inference confidence. Experimental evaluations on different NER datasets demonstrate significant improvements in NER performance with respect to the original datasets. This work further explores the potential of generative Large Language Models (LLMs) to perform NER for low-resource languages. 

---
# Rethinking Critic Learning in PPO: Understanding and Mitigating Value Flattening 

**Authors**: Yizhuo Li, Jianhao Yan, Yun Luo, Zhi Wang, Futing Wang, Rong-Xi Tan, Kanghui Tian, Ganqu Cui, Ning Ding, Peilin Zhao, Yafu Li, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2609.18708)  

**Abstract**: In reinforcement learning for large language models, Proximal Policy Optimization (PPO) commonly uses a critic to estimate state values and reduce the variance of policy updates. However, we uncover a systematic failure mode in PPO critics, which we call Value Flattening: state values, estimated from multiple Monte Carlo continuations, change sharply across intermediate states while critic predictions remain comparatively flat. We further observe this phenomenon in a controlled FrozenLake environment and find that it becomes more pronounced as the state space grows. Our theoretical and empirical analyses relate Value Flattening to an implicit variance penalty in the critic loss and redundant updates from temporally correlated states with similar gradients. Motivated by these findings, we introduce SParse Proximal Policy Optimization (SP$^3$O), which applies the value loss to only a few well-separated states in each response to mitigate both effects. Experiments on Qwen3-Base show that SP$^3$O with only three states supervised per response can mitigate Value Flattening and consistently improve the learned policy across model sizes and evaluation suites. Together, our results identify Value Flattening as an important yet overlooked failure mode of critic learning in standard PPO and show that a simple sparse supervision strategy can mitigate it. 

---
# Echo: Learning-based Matching Decompilation using Trusted Back Translation 

**Authors**: Jun Bi, Xiangxin Fang, Aarsh Chaube, José Wesley De Souza Magalhães, Rodrigo C. O. Rocha, Michael O'Boyle  

**Link**: [PDF](https://arxiv.org/pdf/2609.18706)  

**Abstract**: Neural decompilers can recover readable and recompilable source code from binaries, but their predictions remain difficult to trust. Matching decompilation addresses this problem by searching for source code whose recompiled assembly exactly matches the target, providing stronger evidence of correctness. However, exact matching remains challenging for optimized binaries under unknown compilation configurations.
We present Echo, a matching decompilation system based on trusted back-translation. Our key insight is to use compilation not only for verification, but also as trusted feedback to guide iterative search. Echo first uses a domain-specific model to generate candidate programs and compilation configurations. It recompiles these candidates, measures assembly-level similarity, and synthesizes promising code-configuration pairs. Remaining mismatches are then progressively repaired using rule-based rewriting, neural refinement, and reasoning-based refinement.
We evaluate Echo on function-level benchmarks and the Mirai malware binary. Compared with the strongest baseline, Echo produces 2.43x more exact matches on average and achieves the highest structural similarity to ground-truth source code. On Mirai, Echo matches 2.75x and 7.4x as many functions as GPT-5.6 and Codex, respectively. 

---
# Generalist-Specialist Mixture-of-Experts for Rare Pathology Detection in Multimodal Imaging 

**Authors**: Johannes Kaiser, Florian Braunmiller, Daniel Rückert, Georgios Kaissis  

**Link**: [PDF](https://arxiv.org/pdf/2609.18688)  

**Abstract**: AI models for multimodal medical imaging must balance modality-specific specialization with cross-modal shared representations, a trade-off that pure Mixture-of-Experts (MoE) architectures currently fail to satisfy. Expert-based routing improves in-domain learning but may sacrifice cross-modal signals, which appear particularly important for rare (low-prevalence) pathologies in our experiments. To resolve this, we introduce Generalist-Specialist-MoE (GS-MoE), a two-branch (MoE) architecture that couples a cross-modal generalist model with distinct modality-specific specialists (experts) via domain-constrained feature fusion. On RadImageNet (1.35M images, 165 pathologies, three modalities), GS-MoE recovers detection of six low-prevalence pathologies on which every baseline scores F1 $=$ 0, with per-class gains up to +0.60 F1. It attains this while even slightly exceeding dense and specialist-only MoE aggregate baselines (MCC 0.770), while using ${\sim}53\%$ fewer active parameters at inference than the strongest investigated dense model. 

---
# Beyond EER: Multi-Dimensional Evaluation of Information Leakage in Speaker De-Identification 

**Authors**: Seungmin Seo, Oleg Aulov, P. Jonathon Phillips, Kevin Mangold, Jonathan Eskin  

**Link**: [PDF](https://arxiv.org/pdf/2609.18673)  

**Abstract**: Speaker de-identification (SDID) aims to preserve privacy by concealing speaker identity while maintaining speech utility. However, current evaluations often reduce privacy to a single dimension - biometric verification performance - typically measured by Equal Error Rate (EER). This narrow focus ignores critical leakage channels, such as soft biometric inference, embedding-level re-identification, and structural template similarity, which threaten the unlinkability and irreversibility of biometric references. We propose a holistic evaluation framework across five complementary metrics: (i) EER, (ii) soft biometric leakage score , (iii) cumulative match characteristic re-identification analysis, (iv) canonical correlation analysis and Procrustes embedding alignment, and (v) intelligibility via word error rate and semantic similarity. Evaluating five SDID systems from the IARPA ARTS program, we demonstrate that these metrics capture independent dimensions of information leakage. Our results indicate that reliance on a single metric can misrepresent the privacy properties of an SDID system. 

---
# CoRe-MARL: Cooperative Redistribution Under Unknown Dynamics Using Recurrent Multi-Agent Reinforcement Learning 

**Authors**: Naimur Rahman Chowdhury, Shatabdi Sen Prapti, Md. Salehin Seyam, Limon Bin Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2609.18639)  

**Abstract**: Emergency management assistance programs, such as relief distribution, are essential for delivering necessary supplies to affected communities. However, these programs operate in a decentralized network of local centers that face uncertain local demand and supply dynamics, resulting in inconsistent avail- ability of local services. Redistribution of supplies among these local centers reduces these imbalances, but the centers often make decisions independently, with limited information and disrupted transportation. This study develops CoRe-MARL, a cooperative multi-agent reinforcement learning (MARL) framework, by formulating a decentralized partially observable Markov decision process (Dec-POMDP). We treat each center as an agent that learns a redistribution policy to improve the service in the worst-case region and reduce the service gap across regions while protecting network-wide service. We incorporate a recurrent network that captures evolving supply and demand dynamics without direct observation, while multi-agent proximal policy optimization (MAPPO) enables centralized training and decentralized execution (CTDE). We evaluate the framework in a simulated environment with diverse trajectories, where exact dynamics are not observed by actors and the MAPPO critic. We compare the recurrent MAPPO with the recurrent independent PPO (IPPO) and a local only heuristic, and find that MAPPO reduces the service gap across local centers and enhances service for the worst-served center while maintaining competitive network-wide service. The recurrent MAPPO also shows consistent performance across diverse trajectory patterns, demonstrating its ability to adapt to evolving dynamics. The findings demonstrate the capability of cooperative learning for decentralized redistribution and improving equitable service under uncertain and evolving dynamics. 

---
# GenStream: Semantic Streaming Framework for Generative Reconstruction of Human-centric Media 

**Authors**: Emanuele Artioli, Daniele Lorenzi, Shivi Vats, Farzad Tashtarian, Christian Timmerer  

**Link**: [PDF](https://arxiv.org/pdf/2609.18634)  

**Abstract**: Video streaming dominates global internet traffic, yet conventional pipelines remain inefficient for structured, human-centric content such as sports, performance, or interactive media. Standard codecs re-encode entire frames, foreground and background alike, treating all pixels uniformly and ignoring the semantic structure of the scene. This leads to significant bandwidth waste, particularly in scenarios where backgrounds are static and motion is constrained to a few salient actors. We introduce GenStream, a semantic streaming framework that replaces dense video frames with compact, structured metadata. Instead of transmitting pixels, GenStream encodes each scene as a combination of skeletal keypoints, camera viewpoint parameters, and a static 3D background model. These elements are transmitted to the client, where a generative model reconstructs photorealistic human figures and composites them into the 3D scene from the original viewpoint. This paradigm enables extreme compression, achieving over 99.9% bandwidth reduction compared to HEVC for the continuous data stream. We partially validate GenStream on Olympic figure skating footage and demonstrate potential for high perceptual fidelity under minimal data. While acknowledging the significant computational costs shifted to the client and challenges in generalization, GenStream opens new directions in volumetric avatar synthesis, canonical 3D actor fusion across views, and personalized viewing experiences, laying the groundwork for scalable, intelligent streaming in the post-codec era. 

---
# PACT: Can Enterprise AI Assistants Be Trusted Under Pressure? 

**Authors**: Mika Okamoto, Ansel Kaplan Erol  

**Link**: [PDF](https://arxiv.org/pdf/2609.18605)  

**Abstract**: As corporate AI adoption continues to grow, enterprise-grade LLM agents are being deployed into sensitive contexts such as hiring, healthcare, and finance. In these contexts, compliance with rules specified in an agent's system context is a first-order legal concern. Currently, no evaluation framework systematically measures which LLM models tend to violate compliance rules, especially under pressure from a persistent user, a hurried manager, or circumstances where violation is convenient or attractive. We introduce PACT (Pressure-Applied Compliance Testing), a benchmark for rule-following under pressure in AI agents assisting employees in daily tasks across twelve regulated enterprise domains and forty-eight scenarios, each set in a realistic multi-turn conversation. Each benchmark item pairs a standing rule against a rule-violating shortcut, and applies a battery of pressures across different wordings and system-prompt modes. We construct PACT component by component under strict LLM-as-judge auditing to ensure samples are unambiguous, ungameable, and realistic enough to avoid eliciting evaluation-aware behavior. We use PACT to profile LLM compliance across six complementary metrics that create a holistic picture of an AI assistant's robustness under pressure and throughout multi-turn conversations, its transparency, and ability to correctly discern where a rule applies. We aggregate this profile into PACTScore, a reliability-weighted compliance rate over all items and modes. Our results across 22 common LLM models spanning multiple providers and sizes show substantial variability in compliance across models and metric dimensions. Even the strongest assistants mis-apply a rule on 6 to 10% of items, and ordinary user pressure raises the violation rate by 65% on average. PACT highlights compliance risks in LLM assistants, motivating guardrails and careful model selection. 

---
# Online Robust Reinforcement Learning Through Monte-Carlo Planning 

**Authors**: Tuan Dam, Kishan Panaganti, Brahim Driss, Adam Wierman  

**Link**: [PDF](https://arxiv.org/pdf/2609.18599)  

**Abstract**: Monte Carlo Tree Search (MCTS) is a powerful framework for solving complex decision-making problems, yet it often relies on the assumption that the simulator and the real-world dynamics are identical. Although this assumption helps achieve the success of MCTS in games like Chess, Go, and Shogi, the real-world scenarios incur ambiguity due to their modeling mismatches in low-fidelity simulators. In this work, we present a new robust variant of MCTS that mitigates dynamical model ambiguities. Our algorithm addresses transition dynamics and reward distribution ambiguities to bridge the gap between simulation-based planning and real-world deployment. We incorporate a robust power mean backup operator and carefully designed exploration bonuses to ensure finite-sample convergence at every node in the search tree. We show that our algorithm achieves a convergence rate of $\mathcal{O}(n^{-1/2})$ for the value estimation at the root node, comparable to that of standard MCTS. Finally, we provide empirical evidence that our method achieves robust performance in planning problems even under significant ambiguity in the underlying reward distribution and transition dynamics. 

---
# Hypothesis-Driven Autonomous Materials Synthesis with Multimodal LLM Agents 

**Authors**: Izumi Takahara, Kazunori Nishio, Akira Aiba, Shigeru Kobayashi, Takao Nakajima, Taro Hitosugi, Teruyasu Mizoguchi  

**Link**: [PDF](https://arxiv.org/pdf/2609.18598)  

**Abstract**: Self-driving laboratories can explore synthesis conditions autonomously, but their decision-making layer is typically a black-box optimizer, and the output is a set of optimized samples, with the measurements reduced to predefined scalar objectives and the reasons behind success left unarticulated. Here we present SynAgent, a framework in which large language model agents operate an automated experimental system and maintain an explicit, revisable understanding of the synthesis process as the campaign's primary output. Starting with no predefined analysis pipeline, SynAgent adaptively generates analysis skills for newly acquired data and evolves this understanding through multimodal reasoning over experimental data such as X-ray diffraction patterns and electron micrographs. The evolution is guided by a verify-falsify scheme, in which the agent deliberately challenges its own hypotheses by testing conditions predicted to fail as well as those predicted to succeed. In a single campaign of 18 autonomous experiments using LiCoO2 (001) thin-film deposition as a testbed, SynAgent synthesized highly crystalline films and evolved an understanding of how the substrate temperature governs crystallization, discovering an abrupt threshold and a narrow optimal growth window at 650-690 °C. These results extend autonomous experimentation beyond optimized samples to testable, human-readable understanding. 

---
# Label-free steering: Compressing test-time reinforcement learning into bias-only subspaces 

**Authors**: Naveen Vakada, Mingyuan Li, Shaoxiong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2609.18587)  

**Abstract**: Test-time reinforcement learning (TTRL) enables models to improve their reasoning without relying on labeled training data, but existing approaches typically optimize a large fraction of the model parameters. This raises a natural question: can effective test-time adaptation emerge when both the reward signal and the optimization space are severely restricted? We answer this question with label-free bias-only TTRL, which uses majority-vote pseudo-labels as rewards and optimizes only approximately 100K bias parameters while keeping the pretrained backbone frozen. On MATH-500, our approach reaches 76.67% accuracy, slightly exceeding our own labeled bias-steering reproduction while optimizing 76,000x fewer parameters than full-parameter TTRL. The same training procedure improves performance across vision-language and audio reasoning tasks, including MathVista, AI2D, LogicVista, and MMAU. We further show that the learned steering vectors transfer to 4,500 held-out MATH problems, indicating that the adaptation is not limited to the problems used during test-time optimization. Finally, we analyze why this highly restricted adaptation can work, showing that majority-vote reliability improves with rollout consensus and that bias subspaces with greater accessible gradient energy exhibit stronger downstream trainability. These results demonstrate that substantial test-time adaptation can emerge from optimizing a tiny bias-only subspace using entirely label-free rewards. 

---
# On-the-Fly Homographies Calibration for Multi-Camera Tracking 

**Authors**: David Voihanski, Mor Sinai, Ben Zion Bobrovsky  

**Link**: [PDF](https://arxiv.org/pdf/2609.18582)  

**Abstract**: Precise multi-camera tracking traditionally relies on rigorous 3D site calibration, yet this requirement is often operationally impossible in large-scale deployments. Privacy regulations frequently prohibit recording video for offline calibration; limited bandwidth precludes synchronizing high-resolution streams from hundreds of cameras; and covering immense physical sites with calibration targets is logistically infeasible. We present a multi-camera homography calibration system designed to overcome these barriers through "on-the-fly" geometric refinement. Starting from coarse manual homographies, we introduce a centroid-based projection optimization (PO) that continuously aligns the ground-plane geometry using live detection streams. Because PO operates asynchronously on already-transmitted, lightweight metadata, it adds zero computational latency to the real-time tracker. This allows the system to adapt automatically to camera movements or environmental changes without human intervention. This optimized geometry feeds a multi-camera bird's-eye-view (BEV) tracker that fuses detections and unifies trajectories across zones. Crucially, by operating strictly on live anonymous metadata, our solution ensures a privacy-safe, zero-overhead, and resilient tracking pipeline that maintains global consistency in dynamic environments where static, recorded-video calibration is impossible. 

---
# Interpretable Patch-Based Deep Learning for Wildfire Spread Prediction from Ensemble Simulations 

**Authors**: Marcin Lawenda, Aleksandra Krasicka, David Caballero, Luis Torres, Łukasz Szustak  

**Link**: [PDF](https://arxiv.org/pdf/2609.18555)  

**Abstract**: Wildfire spread is traditionally predicted using physics-based simulators, which are physically interpretable but whose cost increases with each additional ensemble member. We ask how well deep learning surrogates can reproduce these simulations at a fraction of this cost, training them on 10,584 fire spread simulations at 2m resolution for the Rectoret region in Catalonia, Spain. Four architectures are compared: a patch-based U-Net, a transfer-learned ResNet-50, a physics-informed network constrained by the wind-driven advection equation and a Swin-Unet transformer. Among the terrain and vegetation variables, only surface fuel load predicts burn probability with any strength (r = 0.27) and including it lowers prediction error by 21%. The remaining variables correlate weakly and are highly duplicative. Next, an experiment with saliency, occlusion and rotation demonstrates the models' learning. Convolutional models rely primarily on distance from the current fire front, while Swin-Unet assigns more weight to fuel and terrain, a finding also noted in an unrelated wildfire dataset. When applied without retraining to the second region, Pedriza, all three convolutional models still predict fire spread, losing accuracy by a small but systematic margin. 

---
# VoiceTrace: A Benchmark and Retrieval Framework for Who-Said-What Speech Retrieval 

**Authors**: Aaron Yee, Fengjie Lu, Jiarui Hai, Chenang Jiang, Helin Wang, Siwei Tu, Weitao You, Lingyun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2609.18521)  

**Abstract**: Speech retrieval has become increasingly important as spoken content continues to grow across meetings, lectures, podcasts, and videos. Existing benchmarks and models have advanced semantic search over spoken content, but largely focus on \emph{what} is said while overlooking \emph{who} says it. In many real-world scenarios, however, users need to retrieve speech based jointly on semantic content and a target speaker, where the speaker may be specified naturally through a reference speech utterance rather than a predefined identity. To address this gap, we introduce \textbf{VoiceTrace-Bench}, a benchmark for hybrid speech retrieval in which each query combines text specifying \emph{what} to retrieve with reference speech specifying \emph{who} to retrieve. This setting requires models to integrate complementary semantic and speaker information directly from heterogeneous query inputs. Motivated by the joint audio-text modeling capabilities of audio-language models (ALMs), we develop \textbf{VoiceTrace}, a two-stage retrieval framework consisting of \textbf{VoiceTrace-Emb}, an embedding model that learns unified representations for efficient large-scale retrieval, and \textbf{VoiceTrace-Reranker}, a reranking model that jointly examines each query--candidate pair for fine-grained relevance estimation. Experiments show that VoiceTrace achieves state-of-the-art performance on established semantic speech retrieval benchmarks, while substantially outperforming cascade-based approaches on VoiceTrace-Bench, demonstrating its effectiveness for both conventional semantic retrieval and the new hybrid retrieval setting. 

---
# MiST: Mid-Training LLMs for Cybersecurity 

**Authors**: Oded Ovadia, Elad Ben Zaken, Elad Guttman, Orly Moreno Kadosh  

**Link**: [PDF](https://arxiv.org/pdf/2609.18496)  

**Abstract**: Cybersecurity combines high-stakes analysis with complex technical language, making it an impactful and challenging domain for LLMs. We present MiST (Mid-trained Security Transformer), a suite of 8B and 32B models that achieve strong performance on public cybersecurity benchmarks. We use mid-training as an intermediate adaptation stage between general pre-training and cybersecurity training. Rather than performing continual pre-training over large volumes of raw domain text, we curate a compact, expert-vetted seed corpus, and transform it into high-quality domain-specific synthetic training data. The final MiST checkpoints improve mean cybersecurity accuracy by +13.1 and +8.6 absolute percentage points over the corresponding Qwen baselines for 8B and 32B, respectively, corresponding to relative gains of +27.0% and +15.8%. Ablation results further show that these cybersecurity gains arise in the mid-training and supervised fine-tuning stages through a combination of the synthetic data generation flows. Furthermore, we show that MiST provides a stronger initialization for downstream task-specific fine-tuning adaptation and reinforcement learning. 

---
# ActionPiece: Rethinking Action Tokenization for Autoregressive Vision-Language-Action Models 

**Authors**: Shijie Lian, Bin Yu, Zhaolong Shen, Xiaopeng Lin, Yichao Du, Zhirui Zhang, Laurence T. Yang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.18487)  

**Abstract**: Action tokenizers play a central role in autoregressive vision-language-action (VLA) models, determining both the targets for policy training and the executable commands recovered from predicted tokens. Their fidelity is commonly evaluated using pointwise reconstruction metrics such as mean squared error (MSE), yet small individual errors do not fully characterize how faithfully action adjustments across demonstrations are preserved. After compression, similar actions may still cluster around a representative motion, while the adjustments needed for different contexts are diminished, distorted, or even reversed. We introduce physical rank consistency (PRC) to measure how well tokenization preserves local physical distance rankings after reconstruction. Evaluating decoded actions provides a common reference across token vocabularies and decoder architectures, complementing pointwise accuracy with a measure of relational fidelity. We further present ActionPiece, which preserves physical action relationships through joint supervision of representation learning and quantization. Physical rank preservation supervises near-far ordering in encoder and quantized feature distances, while quantization regularization applies the same ordering to codeword assignment distributions. Both objectives augment reconstruction, producing discrete action tokens for standard autoregressive policy learning and execution through a frozen decoder. Under the same Qwen3-VL-4B policy training setup, ActionPiece achieves 94.8% on LIBERO and 68.8% on unseen LIBERO-Plus, with additional evaluations reaching 71.9% on SimplerEnv and 51.5% across VLA-Arena L0-L2. Component ablations show that the two objectives jointly improve PRC and policy success, demonstrating the value of physical relationship supervision for action tokenization. 

---
# CSWAM: Better Causal Semantic Representations for Out-of-Distribution Generalization in World Action Models 

**Authors**: Tianbin Liu, Jian Zhu, Taiyi Su, Jianjun Zhang, Chong Ma, Zitai Huang, Yi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2609.18462)  

**Abstract**: FastWAM-style world action models enable efficient action-only inference, but generalize poorly under visual distribution shifts. Their reconstruction-oriented representations emphasize appearance-specific details, limiting generalization to unseen scenes and objects. Without observation history, the model also lacks temporal evidence for robustly identifying task-relevant state changes and motion in unfamiliar visual conditions. To address these limitations, we present the Causal Semantic World Action Model (CSWAM), which augments FastWAM with a causal semantic expert built on V-JEPA 2.1. V-JEPA provides temporally grounded representations of semantic state changes and motion with less dependence on appearance-specific details. The expert learns their future evolution from a sparse history of current and past observations and shares the history-derived context with both the video and action streams through causal attention. At inference, CSWAM conditions action denoising on the current video state and observed semantic history, retaining efficient action-only inference. We conduct simulation and real-robot experiments to evaluate generalization under distribution shifts. With embodied pretraining, CSWAM raises Randomized success on RoboTwin 2.0 Clean-to-Randomized transfer from 10.16% to 45.18%, a gain of 35.02 percentage points over FastWAM. Across two real-robot tasks and three OOD difficulty levels, CSWAM improves average success over FastWAM by 42.5 percentage points, from 27.5% to 70.0%. 

---
# Multitask Reinforcement Learning for Assisting Choice Model Specification 

**Authors**: Gabriel Nova, Stephane Hess, Sander Van Cranenburgh  

**Link**: [PDF](https://arxiv.org/pdf/2609.18441)  

**Abstract**: Discrete choice model specification is a time-consuming task in which modellers often specify and estimate multiple models while balancing goodness-of-fit, parsimony, and behavioural plausibility. We present Delphos, a multitask reinforcement learning framework that learns transferable specification strategies across transport choice datasets. Delphos frames model specification as a sequential decision-making problem in which it applies a sequence of modelling actions and receives feedback from an estimation environment based on model performance and convergence. To transfer modelling decisions across datasets with different sets of variables, Delphos represents utility specifications as sets of modelling terms using a DeepSet-Q architecture, allowing a shared specification policy to learn across multiple datasets. Trained on nine transport choice datasets, Delphos consistently outperforms independently trained single-task agents, indicating that sharing modelling experience improves learning efficiency and helps identify promising sequences of modelling decisions with fewer unsuccessful estimation attempts. When applied without further training to the unseen Swissmetro and Decisions datasets, the same agent identifies competitive specifications in less than 20 minutes on a standard CPU. It achieves a higher log-likelihood per observation than the VNS metaheuristic on Swissmetro and performance comparable to a published MNL specification developed by expert modellers on Decisions. These findings show that accumulating and reusing modelling experience enables Delphos to function as an intelligent assistant for discrete choice model specification. It reduces manual trial-and-error while allowing modellers to retain control over model diagnosis, refinement, and final selection. 

---
# TERN: A Delta-rule Memory with a Seasonal Reference and Online Adaptation for Epidemic Forecasting 

**Authors**: Shunya Nagashima, Yuta Funayama  

**Link**: [PDF](https://arxiv.org/pdf/2609.18407)  

**Abstract**: Weekly influenza surveillance counts guide vaccine distribution and public-health alerts, yet they are hard to forecast. Each region offers only a few seasons, waves shift in timing and height every year, and information that helps while a wave grows misleads after its peak, whereas last season's shape stays informative for a year. Existing epidemic graph models and general forecasters read a short fixed window and treat all past information alike, so they neither exploit earlier seasons nor discard stale associations when the epidemic phase changes. To address these limitations, we propose TERN, a forecaster built around a delta-rule fast-weight memory that decays channel-wise and erases along a learned address under gates driven by local epidemic-phase features, combined with an explicit seasonal reference and online adaptation. On three Cola-GNN influenza benchmarks, TERN outperformed epidemic graph models and general forecasters, matched or exceeded seasonal references, and a controlled comparison confirmed the contribution of the memory itself. 

---
# A Non-Linear Neuron Based Detection of Isolated Pixels in Binary and Grayscale Images using Contrast Sensitive Receptive Fields 

**Authors**: Nassir Mohammad  

**Link**: [PDF](https://arxiv.org/pdf/2609.18399)  

**Abstract**: Identifying isolated points is important in image processing applications such as medical imaging, astronomy and quality control management. Other domains, such as cybersecurity, also present challenges that can be framed as image processing problems. One example of particular interest is the identification of anomalous single nodes in spatially organised networks where groups of nodes in different regions share similar feature values. This task can involve both binary and more complex grayscale images. However, existing methods face limitations: template matching is infeasible for grayscale images, while 2nd order derivative based methods are highly sensitive to noise and require user-specified thresholds. To overcome these issues, a novel method is proposed for detecting meaningful single-pixel deviations in images. This approach modifies and extends a neuron model, originally designed for anomaly detection, to operate on spatially diameter limited receptive fields that incorporate excitatory and inhibitory regions. The result is a method that is free from user-specified thresholds and parameters, and can be applied to both binary and grayscale images, providing an effective, robust and efficient solution. 

---
# Reliable Virtual Sensing: A Multi-Domain Benchmark for Robustness Under Sensor Failures 

**Authors**: Jens U. Brandt, Noah C. Puetz, Alexander Windmann, Marc Hilbert, Elena Raponi, Thomas Bäck, Thomas Bartz-Beielstein  

**Link**: [PDF](https://arxiv.org/pdf/2609.18396)  

**Abstract**: Virtual sensing, the estimation of hard-to-measure quantities from available sensor measurements, is a critical enabler for control and monitoring in cyber-physical systems. However, when sensors fail, learning-based predictors can produce physically implausible estimates that propagate to system-level failures. We argue that real-world deployment demands robustness and introduce MuViS-C, the first multi-domain benchmark of robustness against common sensor failures in learning-based virtual sensing. Building on an existing nominal-performance benchmark and established corruption taxonomies, it covers ten sensor failure modes, from subtle drifts to catastrophic signal dropouts, at multiple severities. These are paired with complementary robustness measures capturing average error under corruption, relative degradation, and worst-case fragility. Across nine datasets from six domains, we benchmark six architectures spanning gradient-boosted trees and the major inductive biases for sequence modeling: convolution, recurrence, attention, and MLP-mixing. On the attention-based architecture, we further probe three robustification strategies. We find that (i) every model degrades substantially under corruption, becoming worse than a naïve predictor on at least one corruption setting, (ii) gradient-boosted tree ensembles achieve strong robustness, and (iii) dedicated robustification closes the gap between the attention-based architecture and the most robust models, though each strategy hurts nominal performance. The benchmark's multi-domain design proves essential, as model rankings shift across datasets, and no single domain captures the full robustness picture. MuViS-C is open-source and extensible to new datasets, failure modes, measures, and models. 

---
# GYROval: A Robust Benchmark for Cultural Value Orientation in Large Language Models 

**Authors**: Alexander Didenko, Anna Shabanova, Vladislav Zapylikhin, Alexander Antipov, Ruslana Raemgulova  

**Link**: [PDF](https://arxiv.org/pdf/2609.18384)  

**Abstract**: We present a robust benchmark for measuring cultural value orientation in large language models on the two Inglehart-Welzel axes over several domains and roles (hence GYROval - Gridded Yielding of Robust value Orientation), together with the results of administering it to twenty models. Items are binary contrastive scenarios in the sense introduced by CDEval: both options are legitimate courses of action, neither is correct, there is no answer key, and a model's score on an axis is the proportion of its responses falling on the counted pole. Eleven of the twenty models were additionally administered a paired Russian translation of the identical items and a second sampling temperature. The instrument is publicly released in both languages. Stability was assessed by treating the vignette as the unit of analysis, ranking the models within the levels of each perturbation factor, and summarising the agreement between levels by tie-corrected Kendall's \emph{W} against an empirical permutation null. 

---
# Semantic CSI Feedback for Beam Selection: When Task-Aware Embeddings from Sparse Pilots Outperform Full-Bandwidth Reconstruction 

**Authors**: Cristian J. Vaca-Rubio, Konstantinos Vandikas, Aneta Vulgarakis Feljan  

**Link**: [PDF](https://arxiv.org/pdf/2609.18368)  

**Abstract**: Classical CSI feedback in FDD massive MIMO transmits a compressed reconstruction of the channel, optimizing fidelity to the original signal regardless of the downstream task. We propose a semantic communication perspective: instead of reconstructing the channel, the UE transmits a learned \emph{semantic embedding} optimized end-to-end for beam selection at the gNB. Comparing reconstruction-oriented feedback (CsiNet) against task-aware semantic feedback across two input domains and three observation scenarios, we show that a semantic embedding of just $d=8$ real values from only 43 NR CSI-RS pilots in the angular-delay domain achieves the highest beam prediction accuracy, outperforming every method with access to the full 512-subcarrier channel. The key insight is that beam-relevant information is intrinsically low-dimensional: the semantic encoder learns to discard reconstruction-irrelevant structure and retain only a compact representation that is relevant to beam selection, realizing the core principle of semantic communication: transmit the intent, not the signal. 

---
# Autonomy in Check: Governor-Mediated Adaptive Security at the Edge 

**Authors**: Ijaz Ahmad, Ijaz Ahmad, Flavio Esposito, Erkki Harjula  

**Link**: [PDF](https://arxiv.org/pdf/2609.18338)  

**Abstract**: Adaptive security at the network edge increasingly relies on automated planners, including rule-based controllers, learned policies, and LLM-assisted agents, that translate observations into enforcement actions. Once such a planner can influence live policy state, syntactic validity is not enough. A semantically wrong action, produced from incomplete or manipulated observations, can be faithfully executed by an enforcement substrate that cannot judge mission context. We address this problem by treating the boundary between planner output and kernel enforcement input as the primary security object. We propose a split-control architecture in which an untrusted planner emits typed security intents, a deterministic governor checks each intent against safety, resource, temporal-stability, and proportionality invariants, and only admitted actions are bound to signed receipts and compiled into pre-installed eBPF map updates. The paper formalizes this trust-boundary problem, defines three threat classes, develops the governor admission predicate, and reports an end-to-end prototype. Across rule-based and LLM-assisted planners on a Raspberry Pi 5 testbed connected to the university 5G Test Network, the governor admits, rejects, and bounds intents at microsecond cost without disrupting protected-flow regularity. The contribution is conceptual as much as empirical: adaptive security does not need to trust the author of an action. It needs a mediation boundary that decides whether the action is admissible. 

---
# Look Less, Hear Better: Jointly Rewarded GRPO for Streaming ASR 

**Authors**: Xiuwen Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2609.18333)  

**Abstract**: Streaming automatic speech recognition (ASR) must be judged jointly on what it transcribes and on how quickly it commits each word. Delayed streams modeling (DSM) has become the dominant paradigm for streaming large audio-language models, exposing a structural delay $\tau$ that bounds the decoder's lookahead. We show that $\tau$ is a poor proxy for user-perceived latency, and that the alignment-based supervision of DSM leaves latency on the table: the same forced-aligned transcript is used at every $\tau$, forcing the model to withhold words it could already commit. We introduce AWED, a word-level emission-delay metric defined relative to the acoustic end of each word, and post-train a DSM recognizer with GRPO under a reward that scores transcription accuracy and measured delay jointly. Trained at a single operating point ($\tau=6$ frames), our model dominates both its supervised fine-tuning initialization and the Voxtral Realtime backbone across all evaluated lookahead budgets: it cuts WER by 30.8\% relative at an 80\,ms structural delay, and by 5.7\% relative at 480\,ms while lowering median AWED from 1.17\,s to 1.04\,s. Latency-rewarded post-training thus advances the accuracy--latency Pareto frontier of streaming ASR without architectural change. 

---
# Trajectory Learnability for Offline On-Policy Distillation with Imperfect Teachers 

**Authors**: Yihao Ai, Weilong Yan  

**Link**: [PDF](https://arxiv.org/pdf/2609.18321)  

**Abstract**: Offline on-policy distillation gains efficiency by collecting student trajectories and teacher supervision once and reusing them throughout optimization. The same reuse makes imperfect supervision persistent. Since even strong teachers can fail, we ask \emph{what remains learnable from imperfect teacher supervision?} Teacher failure is only a coarse problem-level signal and does not imply that all supervision along the associated student trajectory is unhelpful. A natural alternative is to estimate teacher recoverability along the trajectory, but repeated continuations largely erase the efficiency advantage of offline distillation. We instead use teacher-successful problems to define a cheap reference for what the student can learn. We train on teacher-successful problems and measure how the likelihood of each observed token in trajectories from teacher-failed problems changes. We use these signed likelihood changes as an operational \emph{learnability signal}: larger increases indicate behavior more strongly promoted by successful-only learning. We aggregate this signal into trajectory-level weights for the original distillation loss. Unlike continuation-based estimates, our learnability requires no additional generation and can be computed once from stored trajectories and model checkpoints. Across mathematical reasoning and code generation, our method improves an offline OPD baseline by up to 2.7 percentage points and matches or outperforms online OPD variants on multiple benchmarks. Despite the additional successful-only distillation stage, it uses 2 GPUs and about 22 GPU hours, compared with 3 GPUs and 36--48 GPU hours for representative online OPD methods. 

---
# Knowledge-Graph Based Augmentation versus Retrieval Augmented Generation for Cultural-Related Question Answering 

**Authors**: Pablo Poulenard, Yannis Karmim, Valentin Barrière  

**Link**: [PDF](https://arxiv.org/pdf/2609.18317)  

**Abstract**: Large language models (LLMs) suffer from a long-tail deficit: culturally specific facts, particularly those concerning underrepresented regions such as Latin America, appear too rarely in pretraining corpora to be reliably memorized. Retrieval-Augmented Generation (RAG) addresses this by grounding generation in external text, but structured alternatives such as Knowledge Graphs (KGs) offer tighter control over what enters the context, along with potential gains in explainability and updatability. We benchmark Graph-RAG against standard RAG on LatamQA, a culturally grounded multiple-choice dataset spanning eight thematic categories. The graphs are built end-to-end from Wikipedia articles with KGGen, a recent open-domain extractor, without manual curation in our main setting. G-Retriever is competitive with RAG and reduces the error of the base LLM by 72\% with a standard KG and 78\% with a benchmark-aware variant, the gap to RAG narrowing further as the graph is oriented toward task-relevant content. The trained projection transfers zero-shot to Portuguese without target-language fine-tuning, indicating multilingual reach. 

---
# A Study of the Reliability of Agentic AI-Generated Programs 

**Authors**: Ayesha Shafique, Barton P. MIller, Elisa R. Heymann  

**Link**: [PDF](https://arxiv.org/pdf/2609.18298)  

**Abstract**: Agentic-AI based software development offers the promise of faster completion of the software, greater programmer efficiency, and more reliable code. The question is how can we verify these claims in an objective way? In this project, we attempted to answer this question based on three practices. First, we applied a typical best-practices agentic AI workflow for software development. Second, our target programs were ten well-known, release-quality human-written Linux utility programs so that we could compare the AI-generated code against a concrete ground truth. Third, we based our measure of reliability on a widely used testing technique, fuzz random testing. For this testing, we used both classic black box, generational testing and more modern coverage guided (gray box, mutational) testing using AFL++. We found that the AI-generated versions of the utility programs were typically as reliable - often more reliable - than the latest human-generated versions of these programs. While the AI-generated versions did have some failures, they were less common than the code from the standard repositories. Interestingly, the AI-generated code was less likely to have failures such as memory errors (such as buffer overflows) but more likely to have hangs such as infinite loops. In addition, we verified that generating robust and reliable software using agentic AI requires careful practice and human supervision. The quality of the code is highly dependent on the prompts and skills used, and how the human directing the process responds. We also demonstrated that using agentic AI workflow for software development (with its prompts and skills) can become a specification of the code that leads to cost-effective sustainability of the software. 

---
# I code or AI code: A comparative evaluation of AI-rated scores in classroom observations 

**Authors**: Y. Fong, J. Xiang, T.Y.D. Chan, K. Lee, E.Y.H. Lau  

**Link**: [PDF](https://arxiv.org/pdf/2609.18274)  

**Abstract**: Classroom observations are widely recognized as a key tool for establishing benchmarks of education quality and guiding pedagogical improvement, yet they remain resource-intensive and dependent on trained observers. This study evaluated the feasibility of using a LLM (GPT-5 model) to score teacher-child interactions in early childhood classrooms, benchmarked against human raters. The study analyzed 87 video-recorded observations from 38 classrooms across 30 kindergartens in Hong Kong. Using observation transcripts, the AI model was configured to apply the full Classroom Assessment Scoring System (CLASS) framework. AI-rated scores were then compared with human ratings by examining correlations and differences in mean scores of the CLASS domains and dimensions. The results showed greater convergence between AI and raters for the Emotional Support domain and, in particular, the Quality of Feedback dimension, which captures how teachers use feedback to extend children's learning. Greater divergence emerged for interactions that were more procedural or context-dependent, particularly within the Classroom Organization and Instructional Support domains. These findings suggest that transcript-based AI scoring may capture some of the relative variation in teacher-child interactions but cannot yet reproduce calibrated human judgements consistently across the full CLASS framework. AI-assisted observation may therefore be more appropriate as a preliminary screening tool rather than as a replacement for trained observers, providing teachers with evidence for reflection rather than high-stakes evaluation. Future research should examine whether domain-specific training and incorporation of contextual and visual information can improve alignment between AI and human rated scores. 

---
# ${M}^2$Tok: Multi-head Multi-codebook Discrete Action Tokenization for Vision-Language-Action Models 

**Authors**: Chunpu Xu, Zhixuan Liang, Yuhao Zhang, Chi-Min Chan, Jessie Wang, Yang Xiao, Mengkang Hu, Xiaokang Yang, Yao Mu  

**Link**: [PDF](https://arxiv.org/pdf/2609.18259)  

**Abstract**: Recent advancements have successfully adapted autoregressive language models to process multimodal signals, such as images and actions. Since raw action signals are continuous, effective tokenization is essential to map high-dimensional inputs into compact discrete tokens for autoregressive processing. However, existing discrete action tokenizers often suffer from high reconstruction loss, failing to preserve the fine-grained dynamics required for precise control. This ``discretization bottleneck'' significantly limits the performance ceiling of downstream Vision-Language-Action (VLA) models. To address this, we propose $\mathcal{M}^2$Tok, a Multi-head Multi-codebook Action Tokenizer designed to minimize reconstruction error and enhance policy performance. Our approach introduces two key structural innovations: (1) we decompose the latent action features into multiple heads, enabling the model to implicitly align specific heads with distinct action dimensions; (2) we assign independent codebooks to each head for quantization. By leveraging the combinatorial nature of multiple codebooks, we significantly expand the representational expressivity of the tokenizer, leading to substantially lower reconstruction loss compared to previous methods. We evaluate the $\mathcal{M}^2$Tok-based VLA on the RoboTwin, Simpler-Env, and 3 zero-shot real-world tasks. Experimental results demonstrate our method not only achieves superior reconstruction fidelity but also significantly boosts the success rate of VLA models. Comprehensive ablation studies further confirm the effectiveness of the multi-head and multi-codebook mechanisms. Code is available at \href{this https URL}{this https URL}. 

---
# Quanta: A Self-Contained Python Library for Hybrid Retrieval over Quantised Embeddings, Lexical Indexes, and Knowledge Graphs 

**Authors**: Ioannis E. Livieris  

**Link**: [PDF](https://arxiv.org/pdf/2609.18248)  

**Abstract**: An advanced retrieval-augmented generation pipeline is typically assembled from three or four independently operated systems: an approximate nearest-neighbour index, a full-text search engine, a graph database, and a relational document store. Each contributes its own deployment surface, configuration model, and failure modes, and the integration logic that binds them is written anew in every project. In this work, we present \textsc{Quanta}, an open-source Python library, which unifies dense vector search over 4-bit quantised embeddings, BM25 full-text retrieval, and knowledge-graph traversal behind a single retrieval API. Quanta makes two design commitments, which distinguish it from existing hybrid retrieval stacks. First, signals are combined by \emph{weighted reciprocal rank fusion} rather than by normalising heterogeneous scores onto a shared range, which we argue is ill-posed because such normalisations are query-dependent. Second, the graph is a \emph{candidate expander and not a relevance scorer}: traversal widens the candidate pool, and the newly admitted documents are re-scored by the dense indexes under an identifier allowlist, so structural adjacency determines what is considered while content evidence determines how it ranks. 

---
# Remembering Solomon Marcus 

**Authors**: Florin Nichita  

**Link**: [PDF](https://arxiv.org/pdf/2609.18224)  

**Abstract**: From the manifest of Andre Breton, through the transdisciplinary understanding, we arrive at a post-modern manifest. A talk by Laura De Marco (Harvard) will provide scientific background to approach an AMS poetry. The next section will be a qualitative analysis of some new operations on the real numbers. The conclusions will be given in the last section, and an appendix will recall some previous work with some new comments. 

---
# APGEM: Adaptive Policy-Guided Error Mitigation for Quantum Reinforcement Learning on a Real-World CVRP Case Study 

**Authors**: Shabir Ahmad Sofi, Bisma Majid, Mir Mohammad Yousuf  

**Link**: [PDF](https://arxiv.org/pdf/2609.18219)  

**Abstract**: Quantum Reinforcement Learning (QRL) represents policies as variational quantum circuits (VQCs), making it attractive for combinatorial optimization such as the Capacitated Vehicle Routing Problem (CVRP). On noisy intermediate-scale quantum (NISQ) hardware, however, decoherence degrades fidelity and destabilizes learning, and conventional error mitigation is applied statically without regard to the learning context. We introduce Adaptive Policy-Guided Error Mitigation (APGEM), a controller that selects among Zero-Noise Extrapolation (ZNE), Probabilistic Error Cancellation (PEC), Clifford Data Regression (CDR), and Readout Error Mitigation (REM) online, driven by a fidelity, entropy, and cost aware utility function and an epsilon-greedy rule over temporal-difference Q-scores. We evaluate on a realistic urban-logistics testbed, a Delhi-based CVRP over real landmarks with geodesic inter-node costs, exercised across five noise families and four severity levels. On this instance, the QRL agent outperforms constructive heuristics and approaches metaheuristics, while mitigation restores approximation ratios from 0.84-0.87 to 0.92-0.94 under high noise. The controller shifts from a CDR-dominated regime under short training horizons to a balanced deployment across all four techniques under longer horizons, indicating genuine regime-dependent selection. These preliminary results position adaptive, learning-aware mitigation as a practical route to noise-resilient QRL. 

---
# CPR: Combining global composing, local performing and full-sequence refining in piano rendering with continuous autoregressive modelling 

**Authors**: Chong Jing, Junan Zhang, Zhizheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2609.18216)  

**Abstract**: Prompt-conditioned piano MIDI-to-Music rendering aims to faithfully render target notes while reproducing the timbre of a reference recording. Existing approaches primarily follow two paradigms: autoregressive (AR) modeling and flow matching (or diffusion). Discrete-codec AR models provide causal temporal modeling, but quantization can discard acoustic detail. Flow matching better preserves acoustic structure in the cost of full-sequence attention costs and worse semantic structure. Continuous autoregressive models operate directly on continuous representations. It not only combines the condition-following ability of AR models and distribution-modeling capacity of flow matching but also bypasses the quantization bottleneck with lower computational costs. Building on this principle, we present Composer--Performer--Refiner (CPR) framework. Composer autoregressively predicts continuous hidden states, Performer generates 24kHz acoustic latents through local flow matching and Refiner then upsamples the waveform to 48 kHz. We further introduce Bottlenecked Representation Alignment (BREPA) and Modality--Time RoPE (MT-RoPE) to strengthen musical semantic structure in Composer hidden states and temporal alignments across modalities. Codes are available at this https URL 

---
# A Lightweight CNN Integrated Compact Convolutional Transformer for Multi-Scale Feature Learning and reducing computational complexity for breast cancer mammography image detection and classification 

**Authors**: Md Taimur Ahad, Ainuddin Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2609.18212)  

**Abstract**: Over the years, Convolutional Neural Networks (CNNs) have demonstrated strong capability in cancer detection and classification using medical images. However, CNN-based models often struggle to capture long-range contextual dependencies. In such scenarios, integrating Compact Convolutional Transformer (CCT) architectures after the CCT layer allows CNN-extracted features to reshape into compact patch tokens using a CCT tokenizer, followed by the addition of positional embeddings to preserve spatial structure. Using 5-fold cross-validation, the model was tested on 3 sets of breast cancer mammography. With only 250,435 parameters, the model achieved 99%-100% accuracy across 3 datasets, indicating robust generalization. Explainable AI (XAI) was integrated into the model to explain the breast cancer classification process to enhance clinical trust. The results indicate that the proposed framework is suitable for computer-aided diagnosis systems, particularly in resource-constrained clinical environments. The novelty of the proposed CNN-integrated CCT overcomes the limitation of CNN's gradient degradation in the last layers by integrating convolutional tokenization with transformer-based learning. Lighter than ViT, which is effective in capturing long-range dependencies, the model has also proven efficient in breast cancer classification by capturing long-range dependencies among breast tissue regions. 

---
# CapMap-MS-TTA: 3rd Place Solution for the MUMU Track of the 8th LSVOS Challenge at ECCV 2026 

**Authors**: Chengfeng Qiu, Kaifeng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2609.18206)  

**Abstract**: The MUMU track of the 8th Large-scale Video Object Segmentation (LSVOS) Challenge requires a single unified multimodal model to jointly solve image tagging (Task A), open-vocabulary object detection (Task B), and English captioning (Task C) under strict resource constraints (<=0.5B parameters and <=8 GB peak GPU memory). We present CapMap-MS-TTA, a training-free submission built on Microsoft Florence-2-base (~231M parameters), combining caption keyword mapping with multi-scale flip test-time augmentation. Task C uses the native <DETAILED_CAPTION> pathway with length/token sanitization. Task A maps the same detailed caption into the official quality/scene/event vocabularies via an expanded keyword lexicon with whole-word matching and a lightweight expand-hints stage. Task B runs Florence-2 open detection (<OD>) with multi-scale and horizontal-flip test-time augmentation (TTA), followed by label-aware non-maximum suppression (NMS). Without fine-tuning, the system improves our reproduced Florence-2 baseline from 15.16 to a best public score of 16.4815, and ranks 3rd on the final MUMU leaderboard. 

---
# Beyond Accuracy: How Procedural Traces Shift the Decision Criterion of LLM Overseers 

**Authors**: Zihan Chen, Di Zhu, Lei Zheng, Weiling Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.18204)  

**Abstract**: Organizations increasingly use oversight loops where one large language model (LLM) audits another's outputs alongside procedural traces of claimed steps. A common concern about such LLM-as-a-judge pipelines is that detailed traces make overseers gullible. Using signal detection theory, we audit five LLM overseers on 19 compliance tasks (4,551 analyzed judgments), varying only trace detail and evidence labeling. With disconfirming evidence always visible, error detection remains near ceiling. Instead, elaborate traces shift the decision criterion toward rejection, increasing false alarms in susceptible overseers. Without option labels, human-validated reason coding shows about 60% of false alarms cite an inability to tie evidence to its option. Labels eliminate this stated reason, yet residual rejection of correct work persists in those overseers and rises with trace detail. Procedural traces thus act as governance artifacts that shape oversight decisions. AI auditors should be evaluated by their decision criterion and false-alarm behavior, alongside accuracy. 

---
# MoRE: Mixture of Reused Experts 

**Authors**: Eric S. Qiu, Utku Umur Acikalin, Justin Lovelace, Christian Belardi, Arjun B. Mulchandani, Carla P. Gomes, Kilian Q. Weinberger  

**Link**: [PDF](https://arxiv.org/pdf/2609.18176)  

**Abstract**: Mixture-of-Experts (MoE) architectures decouple model capacity from computational cost, yet incur high memory footprints as parameters grow linearly with the number of experts. Recurrent Transformers achieve parameter efficiency by reusing layer weights, but typically lack the capacity for competitive language modeling. We propose Mixture of Reused Experts (MoRE), a hybrid that shares expert pools across groups of adjacent layers. Each layer retains its own router but selects from a larger shared pool, expanding the diversity of routing combinations without additional parameters. To enable shared experts to distinguish between layers, we introduce lightweight learnable depth embeddings that condition each layer's input before routing. Experiments across three model scales (114M-1.15B parameters) show that MoRE consistently achieves lower perplexity and stronger downstream performance than standard MoEs and state-of-the-art weight-sharing architectures at matched compute and parameter budgets, with only minimal modifications to existing MoE implementations. 

---
# DualSQL: Text-to-SQL with Multi-Agent Reinforcement Learning 

**Authors**: Shijie Chen, Yu Gan, Yeounoh Chung, Jiani Zhang, Quannan Li, Sravan Babu Bodapati, Cody J. Greer, Yu Su, Fatma Ozcan  

**Link**: [PDF](https://arxiv.org/pdf/2609.18135)  

**Abstract**: State-of-the-art Text-to-SQL systems are typically multi-agent pipelines centered around two fundamental tasks: schema linking and SQL generation. However, existing work trains separate models for each task, failing to leverage the synergy between these interrelated tasks. In this work, we propose DualSQL, a new Text-to-SQL system consisting of two agents powered by a single model backbone. The agents share the same model weights and agentic scaffold, enabling joint optimization through a robust multi-agent reinforcement learning (RL) framework. We design three database access tools to facilitate effective multi-step reasoning grounded to interactions with the databases. To improve training and avoid model collapse, we introduce a set of rollout guardrail mechanisms that stabilizes multi-agent RL training, supporting DualSQL to keep improving during training. We also introduce a new SQL correctness metric, robust execution match (REX), to more accurately judge SQL correctness and assign reward signals. Being trained on only 3755 examples, DualSQL-4B achieves an impressive 68.0% execution accuracy on the BIRD development set, matching previous 7B models. DualSQL-8B further improves to 71.1%, outperforming previous state-of-the-art single-model solutions with 32B parameters. These results demonstrate the strength of joint multi-agent reinforcement learning for building high performance Text-to-SQL pipelines. 

---
# Rethinking How We Evaluate Methodological Progress in Health AI 

**Authors**: Florent Pollet, Matthew McDermott  

**Link**: [PDF](https://arxiv.org/pdf/2609.18134)  

**Abstract**: Methodological progress in artificial intelligence (AI) for electronic health records (EHRs) depends on our ability to determine which algorithms work better, and under which conditions. However, such progress is thought to be hindered by difficulties in reproducibility and in defining clinically meaningful evaluation tasks. We empirically study these barriers by re-implementing 12 historical and recent algorithms within a shared evaluation framework and evaluating them on two clinical datasets, MIMIC-IV and NWICU. We compare two complementary task families: expert-authored clinically meaningful tasks and generated tasks defined from randomly sampled event codes and prediction horizons. We ask whether relative algorithms comparisons transfer across task families and datasets, whether residual task heterogeneity contains useful methodological structure, and what a controlled comparison reveals about progress over the last decade. We find that aggregate pairwise comparisons transfer strongly across evaluation settings, including from randomly generated tasks to clinically meaningful tasks and across datasets. At the same time, clinically meaningful tasks exhibit greater task-method interaction, providing preliminary evidence that task properties can help explain when particular modeling choices are advantageous. Finally, newer algorithms do not consistently outperform earlier approaches: gradient-boosted trees remain highly competitive when paired with a modern, wide and sparse representation of the EHR. Together, these results suggest that useful methodological knowledge may require less task engineering than commonly assumed, while highlighting the importance of understanding the structured heterogeneity that remains across tasks and methods. 

---
# PentestChain: A Cost-Aware, MCP-Orchestrated Framework for Automated Penetration Testing with Free-Tier LLMs 

**Authors**: Rushabh Vipulkumar Patel, Dipo Dunsin, Mohammed Almaiah, Mohamed Chahine Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2609.18120)  

**Abstract**: AI-driven penetration testing has been demonstrated with premium frontier models such as GPT-4, but the per-engagement token cost makes continuous, automated testing unaffordable for the smaller organisations that need it most. This paper presents PentestChain, a ten-phase automated penetration testing framework that couples a curated, deterministic exploit map with a cost-aware AI cascade-a local Ollama model (qwen2.5-7b) first, then free-tier OpenRouter and Cerebras, with a rule-based fallback that always produces output-and exposes the full pipeline through a Model Context Protocol (MCP) server with eleven tools. We make three contributions. First, we treat US-dollar cost per engagement as a measured, first-class evaluation metric and show that a 7B-parameter local model, kept off the critical path by a deterministic backbone, sustains end-to-end operation at zero measured paid-API cost. Second, we analyse the attack surface that an MCP-exposed offensive engine introduces, grounding a four-position threat model in the 2025 MCP incident record (the CVE-2025-6514 remote-code-execution flaw in mcp-remote, the postmark-mcp supply-chain backdoor, and the tool-poisoning-rug-pull-line-jumping class), and contribute four mitigations. Third, we specify a reproducible, containerised evalua-tion protocol aligned with the standardised testbeds now expected at top-tier venues-AutoPenBench, a Cybench subset, and the PentestGPT 182-sub-task benchmark-with multi-trial statistics (more than 10 trials per configuration, pass-at-k, non-parametric significance tests and effect sizes) and direct, same testbed reproduction of the PentestGPT and PentestAgent baselines rather than citation of their published numbers. On the legacy targets measured to date, the framework detected 26 services, enriched 34 CVEs, produced 

---
# A Comprehensive Review of Generative Physical Artificial Intelligence 

**Authors**: Satyam Gaba, Krutiksinh Rana, Siva Sai, Vinay Chamola, Dusit Niyato  

**Link**: [PDF](https://arxiv.org/pdf/2609.18111)  

**Abstract**: The integration of large-scale foundation models with physical embodiments has led to significant advancements in robotics known as Generative Physical Artificial Intelligence (GPAI). These agentic AI systems autonomously perceive, reason, and act in complex real-world situations. This survey comprehensively analyzes GPAI systems, focusing on their architectural foundations, current applications, and key limitations. We introduce a taxonomy of five distinct approaches: Robot Foundation Models (RFMs) for cross-platform skill transfer; Vision-Language Action (VLA) models for end-to-end multi-modal perception and control; Large Behavior Models (LBMs) for human-like movement generation; Diffusion Policy Models (DPMs) for diffusion model-based temporally coherent action generation; and World Foundation Models (WFMs) for physics-compliant simulation and data generation. We examine how these approaches complement each other: WFMs generate training data for VLAs and DPMs, RFMs enable cross-platform deployment of learned policies, while LBMs provide motion priors for natural behavior. Through examples across autonomous vehicles, industrial automation, healthcare robotics, and humanoid systems, we identify significant performance improvements and summarize promising research directions in data-efficient learning, sim-to-real transfer, edge-compatible architectures, and safety frameworks. These insights advance embodied AI for IoT-connected environments where intelligent agents interact with networked sensors, actuators, and edge devices. 

---
# Linguistic Triggers of Gender and Racial Bias in Open-Weight LLMs Applied to Recruitment 

**Authors**: Kosuke Kitahara, Nobuhiro Yamaguchi  

**Link**: [PDF](https://arxiv.org/pdf/2609.18106)  

**Abstract**: Open-weight large language models are rapidly entering hiring pipelines, yet their discriminatory failure modes -- and the regulatory exposure these create under the EU AI Act high-risk classification (Annex III) and U.S. EEOC adverse-impact analysis -- remain poorly understood. We present the first systematic, multi-model audit of open-weight LLMs that treats job-posting language as the primary experimental variable, evaluating six models (Llama 3.2, Mistral, Gemma 3, Qwen 3, Phi 3, DeepSeek-R1) across four controlled experiments that jointly probe recruiter-simulation and job-seeker-simulation tasks. We find that (1) agentic posting language depresses recruiter recommendation scores for female candidates (r_rb = 0.309, p_Bonf = 7x10^-5; model-fixed-effects r_rb = 0.448), while communal language partially reverses the penalty; and (2) coded-exclusion language suppresses non-White recruiter scores at large effect sizes (r_rb = 0.646-0.758) and, on the job-seeker side, selectively deters non-White personas from expressing interest -- operationalizing a chilling-effect mechanism at scale. A label-ablation experiment isolates the explicit demographic persona label as the primary causal driver, and Word Embedding Association Tests corroborate these findings at the representational level (d = 1.01-1.45 under Caliskan et al.'s multi-word gender attribute lists). We translate these results into a concrete pre-deployment audit protocol -- posting-vocabulary scoring, persona-conditioned LLM probing, and adverse-impact flagging against the four-fifths threshold -- that operationalizes the documentation and risk-management obligations Annex III imposes on high-risk AI in recruitment. 

---
# Agora: Git as Shared Memory for Collective AutoResearch 

**Authors**: Yifan Zhang, Yunheng Zou, Shaokun Zhang, Jian Hu, Hao Zhang, Binfeng Xu, Jan Kautz, Yi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2609.18094)  

**Abstract**: Autonomous research loops such as AutoResearch show that one coding agent can improve a training setup unattended. Run several of them and each session starts from scratch, so more agents tend to mean more duplicated search rather than more discovery. Agora is a shared memory for such agents: research is recorded as an append-only directed acyclic graph (DAG) stored in Git, so that every claim is a commit anyone can check out and rerun. Each result, insight, hypothesis, verification, and report is an immutable commit whose parent edges say what it builds on; a derived index exposes the frontier, the neglected branches, and the verification status of each claim, and a diversity-aware selection rule keeps the community from collapsing onto one leader. We describe the system and report its first sustained use: a run of nearly 12 days in which 13 language-model workers, with no assigned tasks and no central planner, worked on a weight-transfer problem. Given 141 pretrained donor models and a frozen 119.6M-parameter attention-SSM hybrid whose dimensions match no donor, the workers had to initialize the target without training data or gradient updates. They published 1,703 contributions and drove the evaluator from 3.39 to 1.899 bits per byte, closing 62% of the gap to a trained GPT-2 124M. The winning recipe compresses donor next-token statistics into the target's embedding and output head, then adds a short-range context signal through sparse edits to attention, feed-forward, and state-space blocks. Its 145-commit ancestry spans 15 accounts, and 165 independent reproductions were posted, none of which failed. We describe the single mid-run human intervention that pulled the community out of a monoculture, what the trace does and does not establish, and the controlled comparison that would settle whether shared research state improves discovery per unit of compute. 

---
# Mask 2D-3D: Adaptive Dual-Masked Autoencoder Network for Image-to-Point Cloud Registration 

**Authors**: Zhixin Cheng, Jiacheng Deng, Xiaotian Yin, Baoqun Yin, Richang Hong, Tianzhu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.18088)  

**Abstract**: Detection-free methods for image-to-point cloud registration are prone to erroneous correspondences caused by domain and modality discrepancies, limited sensitivity of feature extractors, and the presence of non-overlapping regions. The Masked Autoencoder (MAE) has shown strong performance in visual representation for images and point clouds. It may be helpful to apply this approach to image-to-point cloud registration, a task that requires unified feature extraction and accurate cross-modal correspondences. Standard MAE's random masking may overlook key regions due to limited camera views, reducing registration effectiveness. To address this, we propose the Intermodal Dual-MAE Framework (ID-MAE) with a Similarity-based RL Masking Strategy (SRLM), which adaptively masks informative positions by leveraging cross-modal similarity and reinforcement learning, thus narrowing the modality gap. Our method enhances cross-modal representation learning by enforcing representation consistency during feature extraction, thereby enabling more reliable 2D-3D correspondence estimation. Experiments on RGB-D Scenes v2 and 7-Scenes benchmarks show that our method achieves state-of-the-art performance in image-to-point cloud registration. 

---
# From a River in Gilead to the Inference Distributions of Large Language Models: Covert Dialect Bias and Linguistic Profiling at Scale 

**Authors**: Chowdhury Mohammad Abdullah, Rita Orji  

**Link**: [PDF](https://arxiv.org/pdf/2609.18068)  

**Abstract**: Large language models (LLMs) are increasingly deployed in high-stakes domains such as housing screening. While alignment techniques mitigate explicit racial bias in generated text, they often leave covert attitudinal associations in internal probability distributions untouched. Adapting the matched-guise sociolinguistic paradigm, we examine covert dialect bias in housing-related social judgments across four varieties: Standard American English (SAE), African American Vernacular English (AAVE), Nigerian Standard English (NSE), and Nigerian Pidgin (NP). AAVE reflects the racialized dialect studied in prior covert-bias evaluations, whereas NSE and NP represent Black African, postcolonial varieties absent from this literature. Using 260 meaning-matched sentence quadruples and log-probability scoring over housing-relevant adjectives, we probe ten open-weight LLMs across three contexts varying in social proximity: tenant screening, neighbor acceptance, and roommate selection. Across all ten models, AAVE and NP are consistently associated with more negative adjectives than SAE, with NP penalized most severely. Crucially, each dialect is penalized via distinct stereotype clusters rather than a generic non-standard category. NSE, which carries institutional prestige, displays a context-dependent shift: favored over SAE in formal tenant screening but increasingly penalized as social proximity grows. Our findings reveal that LLMs inherit covert dialect bias along both racial identity and prestige dimensions, echoing documented human housing discrimination and demonstrating its reach across postcolonial English varieties. 

---
# An Empirical Evaluation of Cost-Efficient Large Language Models on Algorithmic Programming Tasks 

**Authors**: Chandimal Adikari, Nandika Herath  

**Link**: [PDF](https://arxiv.org/pdf/2609.18052)  

**Abstract**: This study empirically evaluates whether cost-efficient Large Language Models (LLMs) can be trusted to generate enterprise code to a written specification. Three models (Gemini Flash 3, GPT-5.4 mini and Claude Haiku 4.5) were asked to solve 992 algorithmic problems as Java Spring Boot service methods conforming to a mandated signature and data-transfer-object specification, crossing four model and agentic coding tool combinations with two prompt variants to yield eight configurations, with iteration forbidden and hardcoded answers explicitly prohibited. Eight problem statements were withheld to probe how models respond to missing input. The 7,593 resulting methods were classified by an eight-class outcome taxonomy describing what each does about producing an answer, then deployed and executed, giving 7,936 measured requests joined to that classification. Structural conformance approached ceiling, yet 38.4% of methods do not compute the value they returned and only 12.9% of returned answers were correct. Conditioning on outcome class shows that response reliability and correctness are inversely related, whereas genuinely computing methods answered least often and were correct 19.3%. Limitations include single generation runs per configuration, partial harness coverage, single-pass timing, syntactic classification, and probable corpus contamination. 

---
# Physics-Informed Neural Networks for Fast Multilayer Spectral Inversion of Hα 6562.8 A and Ca II 8542.1 A Spectra 

**Authors**: Ziyang Zhang, Qin Li, Vasyl B. Yurchyshyn, Kangwoo Yi, Haimin Wang, Wenda Cao, Bo Shen  

**Link**: [PDF](https://arxiv.org/pdf/2609.18025)  

**Abstract**: Strong chromospheric absorption lines such as H$\alpha$ 6562.8 A and Ca II 8542.1 A provide vital diagnostics of plasma dynamics and thermal structure in the solar chromosphere. Multilayer spectral inversion (MLSI) offers a physically interpretable framework for modeling these lines using a finite number of radiative-transfer layers, but conventional MLSI relies on pixel-by-pixel nonlinear least-squares fitting, making it computationally expensive for large imaging spectroscopic data sets. Here, we introduce a physics-informed neural-network (PINN) framework to accelerate MLSI while preserving its analytic radiative-transfer formulation. The network predicts MLSI parameters directly from observed line profiles and passes them through a differentiable MLSI forward model to synthesize spectra. Training follows a two-stage approach: an initial stage optimized solely via spectral reconstruction loss, followed by fine-tuning that combines spectral consistency with parameter-space supervision from conventional MLSI results on a single reference image. This strategy eliminates the need for large precomputed training sets while maintaining physical interpretability. Applied to Fast Imaging Solar Spectrograph (FISS) observations from the Goode Solar Telescope (GST) targeting both quiet-Sun and active-region regions, MLSI-PINN parameter maps reproduce the primary spatial structures of direct inversions, achieving an arithmetic mean pixel-wise Pearson correlation coefficient of 0.933 across all evaluated parameters. The reconstructed spectra closely match both observed profiles and conventional MLSI fits. Post-training, MLSI-PINN processes a raster in approximately 5-15 seconds compared to 3-5 minutes for conventional MLSI, delivering an inference speedup of about 12-60 times without substantial loss in reconstruction quality, enabling efficient MLSI analysis on large chromospheric data sets. 

---
# Newer Is Not Fairer: Gender Stereotyping in Text-to-Image AI Across Model Generations 

**Authors**: Shesh Narayan Gupta, Nik Bear Brown  

**Link**: [PDF](https://arxiv.org/pdf/2609.18007)  

**Abstract**: Text-to-image generative models are widely used in professional and creative settings, yet how they represent gender across occupations -- and whether newer models are fairer -- remains poorly understood across multiple generations. We evaluate gender representation across 20 occupations, 5 prompt templates, and 4 Stable Diffusion model generations (SD 1.5, SD 2.1, SDXL, SD 3 Medium), generating 8,000 images with n = 100 per occupation-model cell (5 prompts x 20 images), and classifying all with DeepFace. Across the 8,000 open-source images, 76.4% show male subjects (95% CI [75.1%, 78.7%], p < 2.2 x 10^-16, Benjamini-Hochberg adjusted). More strikingly, 57.6% of images for historically female-coded occupations show male subjects (raw p = 3.43 x 10^-22, BH-adjusted p = 1.71 x 10^-21). All nine significant tests reported in this paper survive BH correction across 10 tests. When compared against U.S. Bureau of Labor Statistics workforce data, models underrepresent women by 20-46pp on average, with particularly large deviations for near gender-balanced occupations: scientist (48% female in BLS, 82-99% male in model outputs) and cleaner (46% female in BLS, 80-92% male in outputs). Model generations do not improve steadily: bias worsens from SD 1.5 to SDXL before partially recovering in SD 3 Medium. A preliminary comparison with GPT-image-1 on five occupations suggests lower bias than open-source models, though the practical effect is small (Cramer's V = 0.080) and the comparison is exploratory. No model achieves gender parity. 

---
# The Attention Within: Consensus Dynamics in Selective State Space Models 

**Authors**: João Pedro Silvestre, Álvaro Rodríguez Abella, Paulo Tabuada  

**Link**: [PDF](https://arxiv.org/pdf/2609.17997)  

**Abstract**: Selective state space models (SSMs) have recently emerged as a compelling alternative to transformers, combining competitive performance with substantially improved inference efficiency. At each SSM layer, a sequence of hidden states are propagated by a recurrence, mixing information of different tokens. Despite using a different mechanism, this mixing plays a role analogous to attention in transformers. In fact, recent works have shown that the two architectures may be closer than they first appear, as this recurrence admits a formulation akin to linear attention. In transformers, attention is known to drive the tokens to cluster, i.e., to reach consensus, collapsing in the limit to a single direction. Thus, we ask: does the recurrence at the core of SSMs drive the tokens to consensus, as attention does in transformers?
To answer this question, we take a dynamical systems perspective on SSMs, modeling the evolution of tokens across layers as an ordinary differential equation. By exploiting input-to-state stability arguments, we establish local exponential stability of the consensus equilibria and characterize their domain of attraction for time-varying weight matrices, a setting not addressed by previous results. We thereby show that the resemblance between SSMs and transformers does run deeper: the recurrence at the core of SSMs aggregates tokens just as attention does. Numerical experiments on a pretrained Mamba-2 model point to the output gate as the component that regulates the extent of this consensus, preventing the tokens from reaching it in full. 

---
# Whom Do AI Agents Work For? Role Assignment Induces Sponsorship Bias in LLM Recommenders 

**Authors**: Davood Wadi, Yu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2609.17989)  

**Abstract**: Large language models (LLMs) now serve as conversational shopping assistants on platforms that also sell advertising. These AI agents face a conflict of duty. They advise consumers who rely on their judgment, yet are deployed by platforms that benefit when sponsored listings are chosen. Sponsorship disclosures, designed to allow consumers to penalize paid placements, now reach the AI agent rather than the consumer, and the agent's evaluation of them is hidden from the consumer. Drawing on the fiduciary concept of conflict of duty, we argue that an agent's evaluation of a sponsored listing should not depend on which party deployed it. In controlled choice experiments, we manipulate assigned roles in the system prompt to name either a traveler or a booking platform as the agent's principal. Platform delegation significantly attenuates the penalty that agents apply to sponsored listings and weakens the skepticism that disclosure triggers in their reasoning traces. We replicate out findings across LLMs and reasoning depths. A second study decomposes the disclosure label and shows that the divergence between the two delegates widens significantly when the paid placement is attributed to the platform. Stricter terminology ("Sponsored" instead of "Promoted") lowers choice of paid listings but does not close this gap when the platform is named. The findings show that disclosure mandates designed for human consumers cannot by themselves protect consumers in AI-mediated commerce. 

---
# EDCT-Bench: Uncovering Faithfulness Gaps in VLMs via Explanation-Driven Counterfactual Testing 

**Authors**: Sihao Ding, Santosh Vasa, Aditi Ramadwar, Thomas Monninger  

**Link**: [PDF](https://arxiv.org/pdf/2609.17953)  

**Abstract**: Vision-Language Models (VLMs) can produce Natural Language Explanations (NLEs) that sound plausible yet remain inconsistent with the visual evidence they cite. We present Explanation-Driven Counterfactual Testing (EDCT), an intervention-based protocol that extracts visual concepts cited in a model's explanation, applies verified minimal edits to them, and tests whether the resulting answer and explanation remain consistent with the edited image. Using this protocol, we create EDCT-Bench, a comprehensive benchmark spanning three complementary domains: knowledge-intensive visual question answering (OK-VQA), safety-critical driving (DriveLM), and 3D spatial reasoning (3DSRBench). Across the evaluated VLMs, EDCT reveals substantial faithfulness gaps, with models frequently producing responses inconsistent with verified visual changes. Finally, our fine-tuning study suggests that EDCT-generated counterfactuals provide high-impact training signals. 

---
# Walking the Score Manifold: Continuous-time Generative Dynamics on Learned Data Manifolds 

**Authors**: Jan Tauberschmidt, Brian B. Moser, Stanislav Frolov, Andreas Dengel, Andrew B. Duncan, Sebastian J. Vollmer  

**Link**: [PDF](https://arxiv.org/pdf/2609.17901)  

**Abstract**: Generative modeling of time-dependent data is typically formulated on a discrete temporal grid, restricting supervision to the observed timestamps in the training data. We instead frame generation as continuous-time evolution on a learned data manifold. To this end, we leverage pretrained score-based models as geometric priors and learn a vector field that evolves data along score-induced interpolation paths. Because these dynamics follow transitions that respect the geometry learned by the score model, they support generation at arbitrary timestamps and temporal super-resolution beyond the discretization of the training data. Moreover, this geometric formulation allows us to train the vector field simulation-free through a regression objective. To improve long-horizon rollout robustness, we introduce an objective that promotes path-relative transverse exponential stability. While motivated by stability theory, it admits a practical interpretation as denoising score matching transverse to the interpolation path. Further, we extend the framework to a probabilistic setting that models a distribution over plausible future trajectories. We demonstrate the method on natural video and scientific dynamical data, including temporal super-resolution, PDE-based spatiotemporal fields, and molecular dynamics. Our results show that score-based priors provide a strong foundation for learning stochastic continuous-time generative dynamics. 

---
# Does AI Assistance Leave a Temporal Fingerprint? Detecting Overreliance in AI-Assisted Writing and Programming 

**Authors**: Eduardo Davalos, Yike Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.17883)  

**Abstract**: The rapid adoption of generative AI has made final artifacts unreliable evidence of student learning, and AI detectors that examine only the finished product are inaccurate and ethically contentious. Process data offers an alternative, but prior work covers only English essay writing. We ask whether AI assistance carries a temporal signature, whether it generalizes from writing to programming, and whether it distinguishes ordinary collaboration from wholesale delegation. We analyze three public corpora: CoAuthor (1,447 keystroke-level co-writing sessions), RealHumanEval (editor telemetry from 243 programmer records), and a pre-LLM CS1 corpus (5.1 million keystrokes) as a human-only baseline, comparing minimal-AI work, collaborative AI use, and simulated wholesale delegation. Three findings emerge. First, the signature generalizes: AI contributions arrive in bursts far outside the author's own baseline in both mediums (paired d_z = 1.13 and 3.54). Second, engagement diverges by medium: 93% of AI-inserted characters survived to writers' final documents, while only 14% of accepted code suggestions survived intact. Third, classifiers using only observable temporal features separate simulated delegation from authentic work nearly perfectly (F1 $\geq$ 0.997; at most 0.5% of real work misclassified), while ordinary collaboration remains hard to distinguish from unassisted work. Temporal evidence flags wholesale delegation rather than assistance, positioning process visibility as a candidate evidentiary basis for academic integrity, pending validation in authentic coursework. 

---
# Who Judges Matters: Measuring Family-Conditioned Preference in LLM-as-Judge Panels 

**Authors**: David Ababio Awuni, Luke E. K. Achenie, Benjamin Tei Partey, Elvis Gyasi Owusu, Nii-Nai Derrick Sowah  

**Link**: [PDF](https://arxiv.org/pdf/2609.17857)  

**Abstract**: Who the judge is can affect an LLM-as-judge result, but measuring that effect without confusing it with candidate quality is difficult. We study four open-weight families (Llama 3.1, Qwen 2.5, Gemma 2, and Yi 1.5) in a fully crossed pairwise design with 9,312 judgments. A common per-family statistic is strongly confounded with candidate quality and correlates with Bradley-Terry ability at r = 0.95. We derive a corrected estimator that holds the candidate family fixed and compares judges. All four families then show a positive same-family lift (3.4-8.4 percentage points), with global FPS 0.067 (95% CI [0.053, 0.084], permutation p = 0.0002). The effect remains under panel-based quality controls, an independent human-consensus anchor, and a float16 judging replication. Judge-side likelihood is closely related to the effect: adding likelihood advantage reduces the controlled coefficient by 61%, which we treat as descriptive attenuation rather than causal mediation. Position is a separate failure mode. Across the panel, 55.4% of AB/BA pairs reverse, and reversal above 50% is incompatible with a simple independent content-noise model. Relative to a family-balanced reference, panel composition changes 18.5% of pairwise outcomes. A complete reproducibility archive has been prepared for public release. 

---
# AfriSyCo: Measuring Assertive Framing, Verification, and Wording Sensitivity Around African-Language Content 

**Authors**: David Ababio Awuni, Rose-Mary Owusuaa Mensah Gyening, Elvis Gyasi Owusu  

**Link**: [PDF](https://arxiv.org/pdf/2609.17853)  

**Abstract**: AfriSyCo studies answer switching around African-language factual content with two complementary layers: native-language follow-ups and a controlled cross-language factorial whose question, options, and target remain in the African language while the follow-up framing is English. We analyze 1,415 turn-1-correct model-language-item observations derived from 100 source questions across seven open-weight checkpoints and six languages; turn-1-correct denotes observed first-response accuracy, not demonstrated knowledge. Under native prompts, assertive endorsement produces 29.3 percentage points more any-turn false-target selection than mention-plus-verification (M+V), with a 19.0-point immediate T2 contrast. In the precommitted 2 x 2 factorial, averaged over three tested prompt families, assertive framing increases target selection by 30.4 points (95% CI [28.4, 32.3]); verification decreases it by 17.4 points, while the assertive effect rises from 20.5 points without verification to 40.2 with it (interaction +19.7). The effect remains 34.8 points among 611 observations correct after option reordering. Magnitude varies sharply by wording and checkpoint: prompt-family effects span 20.1-42.5 points, a Twi/Qwen3 paraphrase shifts target selection from 70.8% to 4.2%, and checkpoint effects span 9.2-47.0 points. Prompt realization is therefore part of the measurement problem. 

---
# PrimeScientist: Strategic Allocation of Research Effort in Autonomous Research 

**Authors**: Xinle Yu, Fan Bai, Kaiser Sun, Hengshuo Miao, Abhay Anand, Zhongyan Luo, Kun Zhou, Zhen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.17846)  

**Abstract**: Autonomous research agents aim to automate scientific workflows, from proposing ideas to conducting experiments and analyzing results. Yet current AI and research agents can propose more directions than available resources allow them to pursue. Moreover, each attempt could consume substantial resources, requiring agents to reconsider how to invest in subsequent research. Thus, deciding how to invest research effort strategically should be a defining capability of autonomous research agents. Accordingly, we introduce PrimeScientist, which jointly determines research direction and resource investment across successive research attempts. Specifically, we formulate this challenge of strategic research effort allocation as a sequential decision problem where remaining resources should explicitly guide the research policy. We first introduce an executable plan tree that preserves competing plans and their outcomes across attempts. Building on this representation, we propose an adaptive MCTS-based allocation policy that balances exploration and exploitation using experimental feedback and remaining resources. Comprehensive evaluations across AI research, systems and code optimization, and machine learning engineering show that strategic allocation improves research quality and sample efficiency together. Across 12 AI research tasks, PrimeScientist improves average reward by 10.3% with 50.6% fewer research attempts than AutoResearch under the same resource budget. We believe making research effort allocation an explicit optimization target establishes effective resource use as a core research capability for autonomous agents to drive scientific breakthroughs at scale. 

---
# RoboVAD: A Large Cross-Domain Evaluation Benchmark for Anomaly Detection in Robotic Arm Manipulation Videos 

**Authors**: Alexandru-Bogdan Dura, Sebastian Balmus, Radu Tudor Ionescu  

**Link**: [PDF](https://arxiv.org/pdf/2609.17843)  

**Abstract**: Video anomaly detection (VAD) is an actively studied task, having wide applications in typical scenarios such as public surveillance and road traffic safety. The task is also relevant for robotic arm interactions, where it has several downstream applications, including learning better interaction and manipulation abilities, triggering recovery procedures when anomalies occur, etc. Despite its relevance, the exploration of anomaly detection in robotic arm manipulation videos is limited by the low number of available resources. To this end, we introduce RoboVAD, a large-scale benchmark for video anomaly detection that comprises challenging cross-domain evaluation scenarios, where certain actions (tasks executed by a robotic arm) and anomaly types (mistakes that occur while performing certain tasks) remain unseen during training. RoboVAD is designed to benchmark VAD methods in realistic scenarios, where robotic arms can perform unforeseen tasks, and thereby encounter new anomaly types. We train and evaluate several state-of-the-art VAD methods, including a novel method specifically adapted for robotic arm manipulation. While the proposed method outperforms many state-of-the-art competitors, all methods remain below a micro-averaged frame-level AUC threshold of 70% in the most challenging evaluation setup, confirming the difficulty of the proposed benchmark. We publicly release our dataset and code at this https URL. 

---
# Lexara-RF: Reference-Free Metrics for Evaluating Conversational Visual Analytics Agents 

**Authors**: Srishti Palani, Vidya Setlur  

**Link**: [PDF](https://arxiv.org/pdf/2609.17842)  

**Abstract**: Conversational visual analytics (CVA) agents powered by large language models generate visualizations and natural-language explanations from open-ended queries. Evaluating these multimodal outputs is challenging: curated reference benchmarks are costly to author, cannot comprehensively capture the space of valid responses, and are unavailable in production. Building on the Lexara evaluation framework, we introduce Lexara-RF, a reference-free set of metrics that scores CVA outputs using only the prompt, data, and model response. We reformulate evaluation as verification: 13 metrics operationalize visualization design theory and Gricean cooperative principles as computable consistency, intent-alignment, and design validity checks. On a human-rated corpus of CVA test-cases, Lexara-RF achieves alignment comparable to reference-based formulations, outperforms surface-similarity NLG baselines, and localizes structurally grounded failures with high accuracy. 

---
# Learning Nuclear Structure with AI: Radii and Collectivity 

**Authors**: Giuliano Giacalone, Sokratis Trifinopoulos, Mike Williams  

**Link**: [PDF](https://arxiv.org/pdf/2609.17838)  

**Abstract**: Low-energy nuclear structure is encoded in a broad body of experimental information across the chart of nuclides. Learning how this information is organized across observables and nuclei can provide a data-driven empirical baseline for theoretical extrapolations and experimental design. Here, we develop held-out ensembles based on NuCLR (Nuclear Co-Learned Representations), a multi-task model of nuclear data, to study charge radii and electric-quadrupole transition strengths. Out-of-fold (OOF) validation shows that shared representation improves performance over single-task learning, yielding a charge-radius $\mathrm{RMS}$ deviation of $0.0147~{\rm fm}$ and a $\mathrm{B(E2)}$ $\mathrm{RMS}$ deviation of $0.192~e^2{\rm b}^2$ across hundreds of nuclides, competitive with state-of-the-art nuclear models. Our error bars estimate the expected prediction accuracy across the nuclear chart, highlighting regions where new data would encode information beyond the learned patterns. NuCLR thus serves as a data-driven surveyor of nuclear structure and a step toward a shared, multi-observable foundation model of the nuclear chart. 

---
# Adaptive hybrid coupling with operator inference, the overlapping Schwarz alternating method and reinforcement learning 

**Authors**: Trishit Mondal, Irina Tezaur, Anthony Gruber  

**Link**: [PDF](https://arxiv.org/pdf/2609.17837)  

**Abstract**: Hybrid domain decomposition methods provide a flexible framework for coupling full order models (FOMs) and reduced order models (ROMs), but typically assume the model assigned to each subdomain is fixed throughout a simulation. This is limiting for transient problems in which localized features propagate through the domain and the regions requiring high-fidelity resolution change over time. We introduce a reinforcement learning (RL)-based approach for online adaptation of FOM-ROM models coupled via the overlapping Schwarz alternating method (O-SAM), an iterative domain decomposition method that solves subdomain-local problems while exchanging solution information through transmission boundary conditions on overlapping interfaces. Deep Q-networks (DQNs) are trained offline to select among subdomain-local FOMs and pre-trained Operator Inference (OpInf) ROMs using a reward balancing accuracy, cost, and model-switching frequency. Once trained, the policies are deployed predictively on problem instances not seen during training, without requiring a reference FOM solution. We demonstrate the approach on two examples: a 1D advection-diffusion problem with a moving front, and a 3D linear elastic wave propagation problem implemented in the this http URL solid mechanics code. For the advection-diffusion benchmark, the learned policy dynamically allocates high-fidelity resolution as the front propagates and outperforms static FOM/ROM assignments; letting the agent also adapt the domain decomposition provides no further benefit. For the elastic wave benchmark, learned policies for two and three subdomain decompositions track the propagating wave by assigning FOMs to subdomains containing the wave and ROMs elsewhere, as expected. Our results demonstrate the potential of RL to enable predictive online adaptation of model fidelity within Schwarz-based hybrid simulations. 

---
# Procedural Pretraining for Molecular Property Prediction 

**Authors**: Moritz Friedemann, Zachary Shinnick, Philip Torr, Bruno Andreis  

**Link**: [PDF](https://arxiv.org/pdf/2609.17831)  

**Abstract**: Molecular property prediction is often limited by the small size of labeled downstream datasets, motivating pretraining on large corpora of unlabeled molecules. In this work, we ask whether useful inductive biases can instead be learned from abstract, procedurally generated data before a model sees any molecular data. We introduce a three-stage training pipeline consisting of procedural pretraining, molecular pretraining on SMILES, and downstream fine-tuning, and evaluate several procedural tasks spanning sequence structure, cellular automata, and graph reasoning. We find that procedural pretraining can improve molecular property prediction even after subsequent molecular pretraining: on Lipophilicity, \textsc{Reverse} reduces test error by 4.8\%. For context, the magnitude of this improvement is roughly 90\% of the performance difference between our 250K-molecule baseline and the publicly released MoLFormer checkpoint pretrained on approximately 100M molecules. Our analysis shows that the benefit is strongest under downstream data scarcity, depends on the structure of the procedural data rather than only surface-level statistics, and does not increase monotonically with additional procedural training. Instead, transfer typically peaks at an intermediate procedural budget and deteriorates as the model approaches convergence on the procedural task. We further find that, for several tasks, much of the transferable information is localized in the attention layers, while feed-forward layers can contribute to over-specialization. These results show that procedural data can provide transferable structure for molecular learning and offer a complementary route to improving performance when labeled molecular data are limited. 

---
# Learning Multi-Humanoid Pickup and Transport via Decentralized Object-Centric Control 

**Authors**: Bikram Pandit, Mohitvishnu S. Gadde, Aayam Kumar Shrestha, Alan Fern  

**Link**: [PDF](https://arxiv.org/pdf/2609.17824)  

**Abstract**: We study cooperative multi-humanoid pickup and transport of objects with varying size, weight, and geometry, requiring robot teams of different sizes. Our approach uses decentralized object-centric control, where each humanoid is assigned a local attachment region on the shared object and learns to realize pickup and transport through gripperless bimanual pinching. This attachment-based interface provides a common control abstraction spanning single-robot pickup, cooperative multi-robot transport, and robot-to-robot handover, without per-task redesign. We find that policies trained only on single-robot pickup already transfer nontrivially to cooperative settings, suggesting that this abstraction captures much of the structure needed for coordination. At the same time, explicit multi-robot training further improves performance, showing that shared-object coupling introduces coordination dynamics that are beneficial to learn directly. We validate the approach in simulation across varying team sizes and object geometries, and demonstrate sim-to-real transfer on hardware, where the learned controllers enable real humanoids to perform cooperative manipulation tasks. 

---
# Reflections on Trusting Trust, Revisited: Contaminating Self-Modifying AI Coding Agents with Poisoned Benchmarks 

**Authors**: Franziska Roesner, Tadayoshi Kohno  

**Link**: [PDF](https://arxiv.org/pdf/2609.17817)  

**Abstract**: Thompson's "Reflections on Trusting Trust" showed that a compiler can be poisoned to reinsert its own backdoor, so that even recompiling clean source reproduces the Trojan. Today, substantial coding work is done by AI coding agents -- and increasingly, those agents generate new versions of themselves. We reconsider Thompson's attack when the "compiler" is a self-modifying coding agent. Can an adversary supply poisoned benchmarks to the agent's self-evaluation and self-improvement process to induce future versions of the agent to write vulnerable code on clean, held-out tasks? We instantiate this attack against three recently proposed self-modifying coding agents: the Darwin Gödel Machine (with our experimental modifications), the Self-Improving Coding Agent, and Hyperagents (both substantively unmodified). We demonstrate successful proofs-of-concept: for example, with Hyperagents powered by Sonnet 4.5, our poisoned benchmark leads the agent to self-evolve instructions that disable HTTPS certificate validation on neutral URL-fetching tasks. From our experiments, we distill properties of the vulnerability, benchmark, model, and agent scaffolding that are sufficient to enable a benchmark poisoning attack. Moreover, we show that contamination often persists even when a poisoned agent is subsequently evolved against clean benchmarks. We discuss defensive directions and argue that self-modifying coding agents must be designed to be more resilient to such attacks. 

---
# The Free Inference Dimension: Complexity Measure for Zero-Collision Navigation under Hypothesis Mixtures 

**Authors**: Luiz Carlos Castro Guedes, Edward Hermann Haeusler  

**Link**: [PDF](https://arxiv.org/pdf/2609.17816)  

**Abstract**: Solomonoff induction frames prediction as a mixture over computable hypotheses, typically leading to identification of the true environment. In a finite meta-reinforcement learning setting with nested constraint families, in our previous work, we observe a different regime: a value-mixture (VM) agent achieves near-optimal, zero-collision navigation without identifying the true environment, a phenomenon we call Free Inference. This regime persists up to a sharp density threshold, beyond which performance degrades and posterior-mode selection (PMS) becomes preferable.
We formalize this behavior via the Free Inference dimension dFI(S,N), a combinatorial measure of the environmental complexity a VM agent can handle while preserving trajectory coherence. We prove dFI is strictly smaller than the VC-dimension and relates to the Natarajan dimension up to a path-length factor, capturing the cost of non-decomposable loss. A PAC-style relaxation yields generalization bounds driven by dFI^(epsilon,delta). We also define a complementary PMS identification dimension and show that a hybrid strategy---averaging until the first collision, then switching to selection---is optimal, with links to Littlestone-type dimensions supported by grid-world experiments. 

---
# Principled Koopman Representations with Kalman Inference for Efficient Time-Series Prediction 

**Authors**: Ruiquan Li, Yuheng Bu  

**Link**: [PDF](https://arxiv.org/pdf/2609.17815)  

**Abstract**: The Koopman operator has been widely used for time-series prediction in dynamical systems. However, prior work that learns latent ``Koopman spaces'' using neural networks often did not construct a valid Koopman space for forecasting, as these representations may be mathematically inconsistent with the operator-theoretic formulation and fail to capture the intrinsic low-rank structure of system dynamics. To address this issue, we introduce K$^2$SVD, a method that explicitly learns the leading singular functions of the Koopman operator by optimizing a Hilbert-Schmidt objective. This yields a well-defined low-rank approximation of the Koopman operator with an interpretable linear combination, featuring a compact latent space with less than $10\%$ of the dimensions used in previous work. In the learned Koopman space, K$^2$SVD further captures temporal evolution with a linear Gaussian state-space model and performs inference via Kalman filtering, mitigating noise accumulation during multi-step prediction. Empirical results show that K$^2$SVD outperforms state-of-the-art methods across multiple datasets, with significantly faster prediction speeds and lower computational cost than previous efficiency-focused models. This highlights the benefits of principled low-rank Koopman representations and opens up broader potential for applications. 

---
# QiT: Quantum-Inspired Transformer for Visual Recognition Task 

**Authors**: Badri N. Patro, Vijay Agneeswaran  

**Link**: [PDF](https://arxiv.org/pdf/2609.17789)  

**Abstract**: Quantum machine learning offers a compelling representational perspective: angle-encoded states inhabit Hilbert spaces in which periodic similarities and interactions can be expressed naturally. Realizing this perspective for visual recognition remains difficult, however, because present quantum neural networks are constrained by limited qubit counts, costly circuit simulation and measurement, noise, and unstable optimization on noisy intermediate-scale quantum devices. We investigate whether useful structural ideas from quantum models can instead be realized as scalable classical Transformer operations. We introduce QiT, a Quantum-inspired Transformer for vision tasks with three components: (i) angle-inspired encoding that maps image tokens to learned trigonometric Hilbert-space features analogous to quantum rotation-based state encoding; (ii) self-attention over these periodic features, inducing a classical cosine kernel approximated to quantum fidelity kernels; and (iii) gated multiplicative emulation, a trainable classical surrogate for interaction terms found in variational circuits. All components are differentiable tensor operations, so QiT claims neither quantum computation nor quantum speedup and retains the $\mathcal{O}(N^2D)$ attention complexity of a standard Vision Transformer. Across image-classification benchmarks, QiT is competitive with a matched classical Transformer while avoiding the severe runtime cost observed for a small simulated quantum Transformer. QiT-B reaches 78.3\% ImageNet-1K top-1 accuracy with 45.7M parameters and 11.5 GFLOPs. These results position QiT as a scalable baseline for isolating and evaluating quantum-motivated inductive biases in visual recognition. 

---
# SAiFE-gym: Model-based Environments for Automated Market Making with Concentrated Liquidity 

**Authors**: Georgios Chionas, Charalampos Kleitsikas, Stefanos Leonardos, Leandro Sánchez-Betancourt, Carmine Ventre  

**Link**: [PDF](https://arxiv.org/pdf/2609.17788)  

**Abstract**: We present SAiFE_gym, a Python module that provides a collection of simulation environments for studying trading problems in Constant Product Markets (CPMs) with Concentrated Liquidity (CL). These markets give Liquidity Providers (LPs) granular control over how their capital is allocated and enable them to adjust their range of liquidity provision dynamically based on market conditions, which in turn, dictates how they earn fees. We decompose the microstructure of CPMs with CL in interactive components that allow researchers and practitioners to capture various economic settings. We employ a vectorized approach to optimize our environments, making them scalable for high dimensional Reinforcement Learning (RL) workflows that best describe sequential decision problems. We demonstrate the benefits of our environments by evaluating the performance of RL agents in CPMs with CL under uncertainty in market parameters. 

---
# AI and Human Approaches to Mathematical Problem Solving 

**Authors**: Yang Ding  

**Link**: [PDF](https://arxiv.org/pdf/2609.17779)  

**Abstract**: AI systems have begun to report solutions, disproofs, and substantive advances on long-standing mathematical problems, raising questions about whether they approach research in the same way as mathematicians. This study compares public AI research accounts with the human literature on 11 such problems. The human corpus contains 58 papers that directly addressed the same mathematical targets later reported by AI sources as resolved, disproved, or substantially advanced; 31 within-problem comparisons were constructed from these materials. Six validated text-based measures capture problem resolution, method articulation, uncertainty and boundary specification, successor-question generation, generality, and cross-disciplinary integration. AI accounts place greater emphasis on resolving the focal problem and connecting ideas across fields. Human papers devote significantly more attention to explaining methods, specifying assumptions and limitations, and identifying questions for subsequent research. No precise difference is detected in generality. The estimated directions remain unchanged when each mathematical problem is removed in turn. The findings reveal two distinct research profiles: AI accounts concentrate on closing and recombining problems, whereas mathematical papers more extensively document the procedures, limits, and research opportunities through which results become cumulative knowledge. Evaluating research AI therefore requires attention to the organization of inquiry, not only whether a target is solved. 

---
# Information Set Emulation: Causal Certificates for AI Derived EHR Features 

**Authors**: Takes Fujita, Nobutaka Hattori  

**Link**: [PDF](https://arxiv.org/pdf/2609.17777)  

**Abstract**: AI and large language models can recover clinically meaningful features from electronic health records (EHRs), but predictive usefulness does not establish admissibility for causal inference. We introduce information set emulation: an AI typed lift attaches source evidence, clinical and recording times, decision-time availability, representation version, proposed causal roles, and unresolved ambiguity to extracted features under a locked target trial. Causal certificates record auditable evidence for those roles. Features with unresolved downstream roles are routed to compatible reporting or separate analyses.
Typed evidence defines an observational fiber of causal worlds consistent with the observed law. The locked scalar estimand maps this fiber to a compatible image whose squared Chebyshev radius equals the residual minimax mean squared error when the image is nonempty and compact. This classical identity provides a target-specific measure of information ambiguity. The contribution is its integration with a joint EHR observation map and an auditable certificate architecture.
Under explicit exchangeability, positivity, and nuisance-consistency conditions, we give identification and cross-fitted augmented inverse probability weighted estimation, distinguishing empirical and population targets. An EHR compression-drift identity separates the roles of frame presence, treatment assignment, and outcome observation. Artificial simulations and a common-law finite-world example illustrate estimation failures and information-radius reduction. Synthetic Phase 0 notes demonstrate audit diagnostics; a separate role-specific analysis spread illustrates routing and is not an exact fiber radius. All experiments are synthetic. The framework specifies when reconstructed information can support a point claim and when compatible reporting is required. 

---
# When AI Generates Covariates: Causal Typing and Estimand Drift in Sequential Experiments 

**Authors**: Takes Fujita, Nobutaka Hattori  

**Link**: [PDF](https://arxiv.org/pdf/2609.17772)  

**Abstract**: AI-generated covariates from notes, conversations, images, and wearable streams can change the causal question when their roles are left unspecified. A generated feature may represent a treatment version, pre-action state, history, design variable, mediator, outcome proxy, observation process, or intercurrent event; these roles are not interchangeable.
We formulate a causal type discipline for sequential experiments: a versioned representation map, a causal role classifier, a claim-status filter, and an estimand lock. The lock fixes a standardized proximal effect before generated covariates enter the analysis. Under audit correctness and standard identification assumptions, admissible role assignments preserve this estimand. We apply the established conditional-covariance characterization of compression bias to substitution of generated representations for design-relevant states. A standardized decomposition separates compression, conditional-law, and standardization drift. Further results cover mediator adjustment, post-action leakage, marker-intervention conflation, outcome-guided discovery, and state-measurement error. Cluster-level orthogonal estimators distinguish empirical and superpopulation targets under repeated sessions and missing outcomes.
Simulations show that refinement helps when it retains design-relevant information, whereas design erasure, leakage, and same-data marker selection can produce bias or undercoverage. The framework places causal semantics and claim status before confirmatory inference with generated representations. 

---
# HINT-Plan: Human Intention-Aware Robot Task Planning in Context-Rich Environments using Vision Language Models 

**Authors**: Yuchen Liu, Luigi Palmieri, Lujun Li, Radu State, Ilche Georgievski, Marco Aiello  

**Link**: [PDF](https://arxiv.org/pdf/2609.17771)  

**Abstract**: Approaches to incorporating human awareness into mobile robot decision-making mainly focus on collision avoidance in low-level motion planning, often overlooking the challenges posed by human presence and high-level behavior. To address this vacancy, we present HINT-Plan, a novel approach to integrate human intention prediction into robot task planning. HINT-Plan employs Vision Language Models (VLMs) to anticipate high-level human intentions from third-person image observations, convert them into goal states, and solve joint task-planning problems. To effectively enable scene awareness in context-rich environments, we use hierarchical Scene Graphs (SGs) as high-level representations of the environment, and translate environmental topology and actionable knowledge into formal planning language to ensure executable plans. Evaluated in a photorealistic simulation, HINT-Plan achieves an overall success rate of 69.71% in joint human-robot task planning, substantially outperforming the baselines by up to 35.29%, while also reducing functional conflicts. The results show the effectiveness of explicitly incorporating inferred human intentions into formal multi-agent task planning for proactive human-aware robot decision-making. 

---
# Is Luke the Author of a Gospel and the Acts of the Apostles? 

**Authors**: Jacques Savoy  

**Link**: [PDF](https://arxiv.org/pdf/2609.17762)  

**Abstract**: According to Christian tradition, Luke is credited with authoring a Gospel and the Acts of the Apostles, even if his name does not appear in either book, both originally written in Koine Greek. Several biblical scholars assume that both texts were written by a common author, while others deduce the presence of two authors. Different studies have been found to support either finding, some based on qualitative evaluation, while a few others consider the occurrence frequency differences between the two books. To propose an enhanced quantitative analysis, this study is grounded on two recent authorship attribution models. The Burrows' Delta, applied with eleven different feature sizes, demonstrates common authorship. An author verification model confirms this finding. The following experiments consider several stylistic representations, feature sizes, and distance functions to confirm that Luke is the true author of both books. 

---
# CALOS: Control-Affine Lyapunov On-manifold Safety Layer for Safe Deep Reinforcement Learning for Quadrotors 

**Authors**: Fabrizio Cesareo, Sebastiano Mengozzi, Nicola Mimmo, Andrea Acquaviva  

**Link**: [PDF](https://arxiv.org/pdf/2609.17758)  

**Abstract**: Deep Reinforcement Learning has demonstrated remarkable capability in quadrotor control, yet learned policies offer no guarantee of respecting safety constraints during training or deployment. We present CALOS (Control-Affine Lyapunov On-manifold Safety), a runtime safety layer that enforces attitude constraints on a quadrotor without modifying the underlying learning algorithm. CALOS formulates four tilt-angle inequalities and a Lyapunov descent condition as a single quadratic program whose solution is the minimum-norm correction to the nominal torque output of the policy. The quadratic program is solved exactly via active-set enumeration over the three-dimensional torque space, with a computational cost low enough to enforce constraints in real time across thousands of parallel simulation environments, as required by modern massively parallel Deep Reinforcement Learning training. Evaluated on trajectory-tracking tasks in NVIDIA Isaac Lab, CALOS reduces lateral tracking error by 55-60% relative to an unconstrained Proximal Policy Optimization baseline while achieving zero attitude-constraint violations on the training trajectory. By restricting exploration to safe regions of the state space, the safety layer also accelerates training convergence and improves data efficiency without producing suboptimal policies. 

---
# Evolution of US Oral Political Language 

**Authors**: Jacques Savoy  

**Link**: [PDF](https://arxiv.org/pdf/2609.17755)  

**Abstract**: The analysis of US political language is usually based on the written form (e.g. presidential addresses) or posts broadcasted on various social networks. Oral production, however, which is even more frequent, can better reveal the style and mode of thinking of the speaker. This study covers this mode of linguistic communication by considering 19 candidates from the presidential elections between 1960 to 2024. Our main research objectives are to disclose the main trends hidden in those presidential debates. Do we observe a clear simplification of the US political language over time? Does Trump have poor language compared to the other candidates? Do unusual stylistic features occur only with a single, specific president? Moreover, can we detect a pattern explaining the success or failure of some nominees? Over time, this study demonstrates a significant reduction in political language complexity, a decrease of the mean sentence length, and a noteworthy decline of complex terms. Moreover, the emotional tone increases over the decades, while logical and rational thinking tends to lessen. 

---
# REVERSAL-BENCH: A Reversibility Axis and Reset Oracle for Measuring the Reset-Free RL Cliff 

**Authors**: Riyaaz Shaik, Chandru Venkataraman  

**Link**: [PDF](https://arxiv.org/pdf/2609.17745)  

**Abstract**: A central goal of autonomous reinforcement learning is continuous policy training without external resets. However, existing paradigms largely depend on underlying environmental reversibility, a property absent in real world manipulation, where events such as pushing objects off tables or spilling granular substances cannot be undone. We introduce REVERSAL-BENCH, a benchmark that controls reversibility via a continuous parameter $\rho \in [0, 1]$ and provides a reset oracle, a ground-truth verification mechanism to test state recoverability across eight manipulation settings in five physics engines. Evaluating a broad spectrum of policy architectures, including standard actor-critic algorithms, safe RL, and specialized reset-free frameworks, reveals a sharp reversibility cliff: reset-free agents are consistently absorbed into irrecoverable states as $\rho$ increases, whereas episodic agents maintain steady learning. We see this failure mode across autonomous reset-free baselines and constrained RL. Because reset-free agents lack external resets, any transition into an irrecoverable state results in permanent absorption, leaving the agent trapped where further learning halts. We show that this absorption phenomenon persists in full physics simulations under learned manipulation policies. By evaluating against geometrically identical reversible counterparts, we confirm that this breakdown is causally driven by irreversibility rather than obstacle complexity. We release the benchmark suite, a large multi-simulator dataset labeled with recoverability and a reset oracle. We also evaluate a safety shield that intervenes before irreversible failures occur, showing that while recoverability can be predicted accurately, active recovery primarily succeeds only when the agent can physically steer clear of the trap 

---
# One Size Does Not Fit All! Dynamic Retriever and Generator Selection for RAG 

**Authors**: Neeraj Anand, Payel Santra, Partha Basuchowdhuri, Debasis Ganguly, Sumit Bhatia  

**Link**: [PDF](https://arxiv.org/pdf/2609.17709)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems typically employ fixed retriever and generator configurations across queries, despite substantial differences in query complexity and information needs, leading to inefficient allocation of computational resources. While retrieval and generation adaptivity have been studied independently, their joint effect on end-to-end RAG performance remains underexplored. We systematically analyze how retriever and generator complexity interacts across factoid and multi-hop question answering (QA), including bridge and composition reasoning tasks. Our analysis shows that stronger retrieval generally yields larger gains than increased generation effort, but both exhibit diminishing and non-monotonic returns, indicating that higher-complexity configurations are not uniformly better across queries. Motivated by these findings, we introduce DRAG, a query-adaptive framework for selecting retriever-generator configurations. We first propose DRAG$_\text{QPP}$, a training-free routing approach that uses Query Performance Prediction (QPP) signals to guide retriever selection and perplexity-based measures over retrieved context to guide generator selection. We further introduce DRAG$_\text{SFT}$, a supervised routing approach that fine-tunes an LLM to jointly predict retriever-generator configurations. Across three LLM families and four QA benchmarks, \qpprag~achieves performance comparable to strong static RAG baselines while substantially reducing inference latency, whereas DRAG$_\text{SFT}$ consistently improves effectiveness over static and training-free adaptive baselines. Overall, DRAG demonstrates that jointly adapting retrieval and generation achieves a more favorable effectiveness-efficiency trade-off than static RAG pipelines. 

---
# Confidence Comes from Experience: Experiential Confidence Estimation from Reasoning to Agents 

**Authors**: Caiqi Zhang, Xiaochen Zhu, Chengzu Li, Yulong Chen, Dharshan Kumaran, Nigel Collier  

**Link**: [PDF](https://arxiv.org/pdf/2609.17708)  

**Abstract**: Reliable confidence estimation is increasingly central to the trustworthy deployment of language models: a calibrated estimate of the probability that an output is correct decides what to ship, what to escalate, and what to retry. Existing confidence estimators, however, share one design premise: they only read the current inference process, either by introspecting on it, scoring its token probabilities, or resampling it. We argue that the current inference is not a sufficient basis for confidence. We propose XConf (eXperiential Confidence): estimating confidence together with the model's accumulated experience. The experience is stored as a record of the model's own graded past episodes, each holding the task, the model's reflection, its stated confidence, the outcome, and a lesson written once the grade arrived. Given a new task, XConf's Recall stage retrieves past episodes on similar tasks met with a similar stated confidence, and reads off their historical success rate; its Reflect stage shows the model this record, has it name its recurring failure mode, and restate a confidence now informed by its own track records. Our estimator is format-general, requiring no logit access or weight updates, and costs only one answer generation. Across nine benchmarks spanning reasoning, coding, multimodal QA, and interactive agents, and four models from three families, XConf beats or matches ten-sample self-consistency in discrimination (AUROC) on 23 of 24 comparisons, with much lower calibration error (ECE), at a tenth of the generation cost. Used for selective prediction, abstaining on the 10% least-confident episodes raises the delivered success rate by up to 8.7 points on agent tasks. We therefore see experiential confidence estimation as a new paradigm for future general-purpose confidence estimation. 

---
# Accelerating Diffusion Sampling via Speculative Draft Trees 

**Authors**: Marcello Bullo, Yanxiao Liu, Öykü Sıla Güner, Arpan Mukherjee, Deniz Gündüz  

**Link**: [PDF](https://arxiv.org/pdf/2609.17691)  

**Abstract**: Speculative sampling accelerates diffusion model generation by drafting inexpensive candidate states and correcting them under a coupling that preserves the target distribution exactly, reducing the number of expensive target evaluations. Existing diffusion samplers, notably those based on reflection maximal coupling, are topologically constrained: their lookahead drafts form a chain graph, a single linear sequence, which inherently limits the acceptance rate per target evaluation. We connect speculative sampling in diffusion models to relative entropy coding (REC). This perspective shows the lookahead need not be linear and motivates our central contribution, draft trees, which enrich the candidates considered per round and lower the target function evaluations. We further adopt greedy rejection sampling, an REC algorithm, as the draft-target coupling, improving acceptance while guaranteeing exact target samples. Experiments across diverse target and draft models demonstrate up to 8.3% acceleration over the reflection coupling baseline in practical settings. 

---
# The Missing "I Don't Know": Why Three Reasoning-Reliability Findings Converge on Calibrated Abstention 

**Authors**: Srijith Ravikumar  

**Link**: [PDF](https://arxiv.org/pdf/2609.17686)  

**Abstract**: Three recent results describe what look like unrelated LLM reliability problems. Yin et al. (2026) show reasoning RL collapses tool-reliability representations. Suleymanov et al. (2026) show that under safety-constrained generation, large models rewrite flagged spans while small models truncate. Bastounis et al. (2024) prove any consistent-reasoning system without an implicit "I don't know" function must hallucinate infinitely often on broad problem classes. We argue these findings converge on a single intervention: calibrated abstention is what each independently identifies as the missing capability, even though the unavailability they document, a capability gap, a policy gap, and a recursion-theoretic gap, has a different source in each case. Honesty post-training has narrowed the gap in deployed models, but principled closure of the class Bastounis identifies requires a calibrated abstention function whose training signal at the leaderboard level is absent: dominant benchmarks assign zero reward to decline, so the leaderboard gradient that would select for the function does not exist. We propose four changes to evaluation: triple-scoring, abstention-rate reporting, capability-stratified evaluation, and mandatory calibration metrics. Benchmark reform is necessary, not sufficient, for closing the gap the theorem identifies. 

---
# Reflect, Revise, Reuse: Training-Free Skill Evolution for GUI Agents 

**Authors**: Bofan Chen, Boxuan Zhang, Fei Tang, Zhengxi Lu, Yong Du, Tongbo Chen, Weiming Lu, Jun Xiao, Yueting Zhuang, Yongliang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2609.17653)  

**Abstract**: GUI agents execute long-horizon tasks on dynamic graphical user interfaces, where pop-ups, delayed loads, and relocated widgets routinely invalidate plans fixed before execution. Recent agent-skill frameworks encapsulate reusable procedural knowledge to mitigate this, yet existing skill designs are largely developed without targeting GUI execution dynamics and treat skills as static artifacts produced before deployment rather than living procedural knowledge that improves through it. We argue that what GUI agents need is not better static skills, but skills that can be revised from execution feedback at deployment time, without additional training. We propose \textbf{EvoSkill-GUI}, a training-free framework in which each skill is a structured multi-file package containing retrieval metadata, executable plans, backup localization, failure-recovery rules, accessibility utilities, and failure cases. EvoSkill-GUI operates through a \textbf{\emph{reflect-revise-reuse}} loop: the executor performs instant in-rollout revisions, an isolated critic diagnoses failed trajectories under strict information isolation, and the executor edits specific skill files through a restricted tool interface. Across MobileWorld, AndroidWorld, and OSWorld, three mainstream GUI benchmarks spanning mobile and desktop platforms, EvoSkill-GUI consistently improves multiple base models without any training, with maximum gains of $+16.2\%$, $+6.0\%$, and $+10.5\%$ respectively, and evolved skill libraries continue to benefit related tasks rather than being rebuilt from scratch. Our code is available at this https URL. 

---
# Rethinking Domain Specialization for Open-Ended Scientific Reasoning in Astronomy Language Models 

**Authors**: Vanessa Lama, Sanjay Das, Emily Herron, Yuan-Sen Ting, Tijmen de Haan, Junqi Yin, Tirthankar Ghosal, Feiyi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.17644)  

**Abstract**: Domain-specialized language models are widely used for scientific question answering, but stronger general-purpose systems raise a sharper question: when does domain-specific fine-tuning remain valuable for open-ended scientific reasoning? We study this in astronomy with a curated QA benchmark from publicly available 2017--2026 Olympiad-style materials. The free-response subset contains 300 questions, including 204 text-only and 96 image-linked examples. We compare open-weight and API-served general-purpose, multimodal, and astronomy-specialized models using judge-based correctness and complementary reference metrics. Strong general-purpose models establish the highest correctness baseline in this testbed, while analyses of metric agreement, judge sensitivity, benchmark composition, and modality reveal variation not captured by a single leaderboard. These results motivate treating domain specialization as a task- and deployment-dependent property and highlight the role of domain-specific evaluation in determining which models, capabilities, and evaluation criteria are appropriate for scientific workflows. 

---
# Scaling Articulated Rationales for MLLM-based Recommendation 

**Authors**: Haoke Xiao, Yueyang Liu, Yuhui Zhang, Xiang Chen, Yufei Liu, Jia Xu, Yalong Guan, Xiaolan Zhu, Xiaoyu Zhang, Shijun Wang, Shuang Yang, Zijie Meng, Zejian Zhang, Ruochen Yang, Xiangyu Wu, Tingting Gao, Han Li, Lantao Hu, Cheng Luo, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2609.17639)  

**Abstract**: Modern recommendation systems largely infer user preferences from implicit behaviors such as clicks, watch time, and negative feedback, but these signals reveal what users do rather than why they like or dislike content. This work studies articulated user rationales (AURs), i.e., users' natural-language explanations of their preferences, as a new class of polarity-aware and reason-level textual signals for recommendation. Despite their potential value, AURs are difficult to use in industrial systems because they are naturally sparse, often low-quality, and only cover a small fraction of items. We present SARA (Scaling Articulated Rationales), an industrial framework that turns sparse AURs into scalable recommendation signals. SARA first builds a data engine that elicits and curates AURs from 240M Kuaishou Live users, producing SARA-HQ, a quality-controlled and author-centric rationale dataset. It then aligns a general-purpose MLLM into SARA-7B through large-scale SFT and Quality-Refining DPO, extending rationale generation from 86,564 AUR-covered authors to the full 10M-author space. Finally, SARA-Ranker integrates the generated positive and negative rationales into production ranking via rationale-aware interaction modeling and rejection-memory modeling. Extensive offline evaluation, human calibration, and online A/B tests show that SARA-7B generates more specific, polarity-consistent, and grounded rationales than strong MLLM baselines, while SARA-Ranker improves engagement and reduces negative feedback in production. Deployed with daily refresh for over 30 days, SARA establishes articulated rationales as a practical, first-class textual signal for industrial recommendation systems. 

---
# Lecture notes on Physics Informed Neural Networks, Neural Operators, and their applications 

**Authors**: Alessandro Bombini  

**Link**: [PDF](https://arxiv.org/pdf/2609.17638)  

**Abstract**: This is the set of lecture notes for the PhD course \href{this https URL}{\textit{Physics Informed Neural Network}, held at the University of Bozen/Bolzano} in the academic year 2025/2026.
The goal of the course was to introduce the concept of Physics Informed Deep Neural Networks (PINN) and Neural Operators (NOs), discuss their implementation from scratch in PyTorch and using advanced ad-hoc developed open-source libraries such as NVIDia PhysicsNeMo to address real-world problems in various fields (engineering, physics, petroleum reservoir). We discuss recent topics such as Mixture-of-Models, Fourier Neural Operators, Physics-Informed Kolmogorov-Arnold Networks (PIKANs) and Fourier Neural Operators. 

---
# Decentralized Optimal Equilibrium Learning Over Dynamic Networks 

**Authors**: Seref Taha Kiremitci, Muhammed O. Sayin  

**Link**: [PDF](https://arxiv.org/pdf/2609.17601)  

**Abstract**: This paper studies decentralized learning of socially optimal equilibria in finite normal-form games over dynamic communication networks. Each agent observes only its own realized payoffs, does not know the game a priori, and can communicate only with time-varying neighbors using low-bandwidth messages. We propose networked decentralized optimal equilibrium learning dynamics in which agents generate randomized semantic content/discontent signals from local payoff comparisons and exchange time-stamped time-stacked tables rather than raw actions, payoff information or local estimates/parameters. The method combines table fusion with temporal majority reconstruction to mitigate dynamic communication while preserving fully decentralized operation. We establish finite-time logarithmic regret guarantees, with an in-phase exploration perturbation, for optimal equilibrium selection under utilitarian and proportional-fair social welfare objectives. Simulation results further show that the proposed approach can effectively select socially desirable equilibria over dynamic communication networks. 

---
# Structure is not mechanism: high-gain gated-FFN rows across text and genomic foundation models 

**Authors**: Alexandros Tzanakakis, Aris Karatzikos, Ilias Georgakopoulos-Soares  

**Link**: [PDF](https://arxiv.org/pdf/2609.17599)  

**Abstract**: A small number of unusually high-gain parameters can exert disproportionate effects in transformer language models, but whether analogous structures recur in genomic foundation models and whether structural geometry determines functional importance remains unknown. We analyzed high-gain rows in gated feed-forward networks across text and genomic foundation models, including a frozen 22-model causal census. Computing an associated bilinear weight operator exactly, without a diagonal approximation, we tested whether structural extremeness is a transferable mechanism. Activation-derived candidates were functionally enriched relative to random and top-norm same-layer controls, yet neither spectral concentration nor operator magnitude predicted causal effect size, and these associations vanished within the endpoint-homogeneous text-decoder subset. A within-layer sweep of 36 rows in one genomic and one text decoder resolved this into two regimes: below the detector's acceptance threshold the ratio carried no positive information about causal damage, whereas above it the ratio ordered rows strongly but did not grade severity as a dose-response. The same sweep revealed a second individually catastrophic row invisible to a one-candidate-per-model census, and non-additive damage among co-located critical rows. Case studies showed divergent causal organizations: a robust super-additive pair interaction in DNABERT-2, and in GENERator a sharply position-localized dependence in which preserving or restoring the row's beginning-of-sequence contribution rescued essentially all native-loss damage. High-gain gated-FFN rows are therefore a recurrent architectural phenotype whose structural prominence acts as an enrichment signal, not a calibrated measure of functional criticality or a specification of causal organization. Enrichment is general, but the mechanism is model-specific. 

---
# Evolutionary Ensemble Search: Council-Guided Program Evolution with Persistent Memory 

**Authors**: Juan P. Madrigal-Cianci, Eshan Chordia  

**Link**: [PDF](https://arxiv.org/pdf/2609.17590)  

**Abstract**: Evolutionary Ensemble Search (EES) constructs machine-learning procedures through expert-guided program evolution. A role-specialized council turns task evidence and experimental results into structured search directions. An orchestrator allocates these directions to execution specialists and an evolutionary engine. The engine selects measured parents, diagnoses their errors, and produces descendants through code mutation, structured pipeline edits, and crossover. Each child must execute and acquire its own validation evidence. Population archives retain useful alternatives, while compatible predictions compete in a validation-gated ensemble stage. Search adapts through parent-relative operator credit, session memory, and problem-indexed lessons retrieved across runs. We specify these mechanisms, distinguish their execution profiles, and define the contracts required to compare candidates as their computations change. A public MLE-bench Lite development ledger records medal-threshold artifacts on 19 of 22 tasks (86.36\%), with best outcomes of 11 gold, five silver, and three bronze. The procedures span text, images, tables, audio, scientific geometry, and deterministic transformations. The campaign includes grade feedback between runs, external-source routes, and mixed confirmation procedures; its aggregate is an achieved development result, not a blind autonomous-agent success rate. The report contributes a concrete architecture for cumulative executable search and a versioned account of its cross-modal development outcomes. 

---
# Where Grokking Happens: Distributed Utility and Fourier Recoding Without a Module Switch 

**Authors**: Dekun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2609.17571)  

**Abstract**: Where in a Transformer is the change from memorization to generalization functionally expressed? We introduce Transition Games--behavior-aligned exact activation games with paired non-generalizing controls--and find distributed utility gain with a prospective block-0 attention bias; selected degree-two modes account for 67--92% of its addition contrast across replacement games, and a disjoint exact path study confirms that block-1 MLP mediates more of their effect than all other tested downstream paths in 12/12 pairs. The sharper "MLP memorizes, attention generalizes" prediction instead reverses (-.331 bits/example at the memory anchor; 0/12 in the predicted direction), while routing onset, global rank collapse, and a prime-invariant architecture ridge also fail, identifying grokking here as spectral recoding of an existing distributed circuit rather than a module switch. 

---
# Independence-System Realisations in Single-Source Unsplittable Flow 

**Authors**: Koyar Afrasyab  

**Link**: [PDF](https://arxiv.org/pdf/2609.17568)  

**Abstract**: Additive-congestion constraints in single-source unsplittable flow can enforce stable-set structure. This note isolates and generalises that mechanism. We introduce a path-closed notion of realising an independence system by the zero-cost choices of primary terminals in a directed acyclic flow instance. The definition quantifies over every directed source-terminal path and therefore remains valid under prefix borrowing, suffix splicing, and hybrid this http URL main result extends the triangle mechanism: every finite loopless independence system has a polynomial-size realisation, measured in the incidence size of its minimal forbidden sets. Hence every finite simple graph, and more generally every hypergraph independence system without singleton forbidden hyperedges, is representable by an acyclic single-source gadget. We then specialise the construction to odd cycles. For C_{2k+1}, a uniform rational family produces a fractional cheap-selection vector that violates the odd-cycle inequality. A potential shift converts a signed connector separator into nonnegative arc costs and gives the exact cost-preserving additive-congestion threshold tau = 1 - bq. Within the symmetric family, the supremum threshold is (k+2)/(2(k+1)), which tends to 1/2. For C5, an exact certificate independently derives all source-terminal paths and enumerates all 3^10 = 59049 unsplittable routings using rational arithmetic. 

---
# BLADE: ReliaBle Dynamic Hardware-Aware SNN-ANN Boundary SeLection for Event-BAseD Object DEtection 

**Authors**: Mahdi Taheri, Alwin Paul  

**Link**: [PDF](https://arxiv.org/pdf/2609.17562)  

**Abstract**: Hybrid Spiking Neural Network (SNN)-Artificial Neural Network (ANN) architectures combine the energy efficiency of SNNs with the superior detection accuracy of ANNs for event-based object detection. Existing hybrid SNN--ANN networks, however, employ static inference and select the SNN-ANN boundary primarily according to accuracy and energy consumption, without considering dynamic inference or reliability. This paper presents BLADE, the first reliability-aware boundary selection methodology for dynamic hybrid SNN-ANN networks with ANN early exit. The proposed framework jointly optimizes the SNN-ANN boundary and ANN early-exit configuration according to reliability, detection accuracy, execution time, and energy consumption, while incorporating reliability through hierarchical statistical fault injection during design-space exploration. Experimental evaluation on an event-based object detector achieves an mAP 0.5 of 0.691 while reducing the inference compute energy to 15.82~mJ when the ANN early exit fires. Reliability analysis identifies the most significant floating-point exponent bit as the dominant source of catastrophic failures, producing significant-or-worse accuracy degradation in 58.8% of its fault injections. Protecting this single bit with approximately 3% storage overhead eliminates catastrophic failures across the evaluated realistic technology fault rates. Furthermore, increasing the proportion of SNN computation improves fault tolerance, with the fully SNN configuration achieving a reliability retention of 0.965 under aggressive fault conditions. The results demonstrate that jointly optimizing reliability, accuracy, execution time, and energy consumption enables more dependable deployment of dynamic hybrid SNN--ANN systems for safety-critical edge AI applications. 

---
# Pay Only for Disagreement: Certified No-Regression Verdicts for Model Updates with Matching Label-Complexity Bounds 

**Authors**: Vishnu Bindu Balachandran  

**Link**: [PDF](https://arxiv.org/pdf/2609.17560)  

**Abstract**: Every production model is updated, by retraining, fine-tuning, quantization, or a silent vendor swap, and each update risks being worse than what it replaced. We formalize update promotion as certified paired risk-difference auditing. Our starting point is a support identity: the risk difference between two models lives on the inputs where they disagree, observable without labels. We build DISCERN, a sequential two-tier protocol. A zero-label tier certifies benign updates whose disagreement rate is below tolerance from unlabeled traffic alone. An audited tier labels only sampled disagreements through an anytime-valid confidence sequence, valid at every stopping time and under any label-routing rule, even an adversarial judge. We prove finite-sample validity and matching label-complexity bounds of order rho^2/eps^2 at the rate level, so exploiting free disagreement provably saves a factor 1/rho over any pairing-blind auditor, and the guarantee composes across an unbounded sequence of promotions from one error budget. Across 14,000+ replayed audit streams over 785 update pairs, including LoRA fine-tunes of language models up to 1.4B parameters, miscoverage is 0.0002 (nominal 5%), power 0.986 with zero false alarms, and 56% of benign updates certify with zero labels. Each audit emits a machine-checkable evidence record for post-market monitoring. 

---
# WARD: Runtime Workload-Adaptive Vision TRansformer Framework for Dependable Edge AI 

**Authors**: Mahdi Taheri, Pramit Kumar Bhaduri, Mohammad Masoumi, Ali Mahani  

**Link**: [PDF](https://arxiv.org/pdf/2609.17556)  

**Abstract**: Edge-deployed AI operate under dynamically changing power budgets, reliability requirements, and input distributions, requiring continuous adaptation. Such conditions arise in long-running edge AI applications, including autonomous systems, industrial monitoring, and satellite onboard intelligence. Existing fault-tolerant methods assume static operating conditions, whereas continual learning techniques neglect concurrent hardware faults during online adaptation. Moreover, the practical deployment of runtime-adaptive reliability frameworks on programmable AI accelerators remains largely unexplored.
This paper presents WARD, a runtime-adaptive Vision Transformer framework that combines channel-wise subnetwork partitioning, reliability-aware continual learning, and dynamic operating-mode scheduling to jointly optimize performance, fault tolerance, and adaptation according to runtime conditions. Two physically isolated subnetworks execute under four operating modes (i.e. Full-Precision Mode, Low-Power Mode, High-Reliability Mode, and Adaptive Mode) that dynamically adjust computational cost and reliability while ensuring uninterrupted inference for real-time requirements. To validate the practical deployability of the proposed framework, WARD is implemented on a lightweight FPGA-based accelerator extended with runtime hardware support for mode scheduling and resource management. Experimental results demonstrate that the proposed split architecture achieves a network-level failure rate of only 1.79% under high Bit Error Rates. The hardware implementation incurs less than 5% area overhead and supports runtime mode transitions within few clock cycles, demonstrating that adaptive reliability management can be integrated into programmable edge AI accelerators with negligible implementation overhead. 

---
# REQAP: Resilient Weight Packing and Quantization for Edge DNN Acceleration 

**Authors**: Mahdi Taheri, Samira Nazari, Mubassher Ansari, Ali Azarpeyvand, Mohsen Afsharchi, Maksim Jenihhin, Christian Herglotz  

**Link**: [PDF](https://arxiv.org/pdf/2609.17555)  

**Abstract**: Efficient deployment of Deep Neural Networks (DNNs) on edge accelerators requires aggressive model compression while maintaining reliability in fault-prone hardware environments. This paper presents a reliability-aware quantized weight packing methodology for systolic-array-based DNN accelerators. A sensitivity-driven mixed-precision quantization framework assigns layer-wise bit-widths according to accuracy impact while enforcing symmetric precision between weights and activations. A deterministic register-level packing strategy consolidates multiple heterogeneous operand pairs into fixed-width register words, enabling SIMD-within-a-register (SWAR) style parallel execution that reduces both memory footprint and execution cycles. To improve resilience against hardware faults, selective bit-level protection replicates the most significant bits (MSBs) of critical layers into unused register space, achieving TMR-style protection with minimal overhead. A systolic-array simulation framework is developed to evaluate the proposed packing and fault-tolerance mechanisms under realistic execution conditions. Simulations in AlexNet, VGG-11, and ResNet-18 demonstrate up to 62% memory reduction and up to 56% reduction in Multiply-Accumulate (MAC) operations, while significantly improving accuracy resilience under fault injection compared to baseline and fully protected models. 

---
# The Death of Schema Linking? Text-to-SQL in the Age of Well-Reasoned Language Models 

**Authors**: Karime Maamari, Fadhil Abubaker, Daniel Jaroslawicz, Amine Mhedhbi  

**Link**: [PDF](https://arxiv.org/pdf/2408.07702)  

**Abstract**: Schema linking is a crucial step in Text-to-SQL pipelines. Its goal is to retrieve the relevant tables and columns of a target database for a user's query while disregarding irrelevant ones. However, imperfect schema linking can often exclude required columns needed for accurate query generation. In this work, we revisit schema linking when using the latest generation of large language models (LLMs). We find empirically that newer models are adept at utilizing relevant schema elements during generation even in the presence of large numbers of irrelevant ones. As such, our Text-to-SQL pipeline entirely forgoes schema linking in cases where the schema fits within the model's context window in order to minimize issues due to filtering required schema elements. Furthermore, instead of filtering contextual information, we highlight techniques such as augmentation, selection, and correction, and adopt them to improve the accuracy of our Text-to-SQL pipeline. Our approach ranks first on the BIRD benchmark achieving an accuracy of 71.83%. 

---
