# VeriGrey: Greybox Agent Validation 

**Authors**: Yuntong Zhang, Sungmin Kang, Ruijie Meng, Marcel Böhme, Abhik Roychoudhury  

**Link**: [PDF](https://arxiv.org/pdf/2603.17639)  

**Abstract**: Agentic AI has been a topic of great interest recently. A Large Language Model (LLM) agent involves one or more LLMs in the back-end. In the front end, it conducts autonomous decision-making by combining the LLM outputs with results obtained by invoking several external tools. The autonomous interactions with the external environment introduce critical security risks.
In this paper, we present a grey-box approach to explore diverse behaviors and uncover security risks in LLM agents. Our approach VeriGrey uses the sequence of tools invoked as a feedback function to drive the testing process. This helps uncover infrequent but dangerous tool invocations that cause unexpected agent behavior. As mutation operators in the testing process, we mutate prompts to design pernicious injection prompts. This is carefully accomplished by linking the task of the agent to an injection task, so that the injection task becomes a necessary step of completing the agent functionality. Comparing our approach with a black-box baseline on the well-known AgentDojo benchmark, VeriGrey achieves 33% additional efficacy in finding indirect prompt injection vulnerabilities with a GPT-4.1 back-end.
We also conduct real-world case studies with the widely used coding agent Gemini CLI, and the well-known OpenClaw personal assistant. VeriGrey finds prompts inducing several attack scenarios that could not be identified by black-box approaches. In OpenClaw, by constructing a conversation agent which employs mutational fuzz testing as needed, VeriGrey is able to discover malicious skill variants from 10 malicious skills (with 10/10= 100% success rate on the Kimi-K2.5 LLM backend, and 9/10= 90% success rate on Opus 4.6 LLM backend). This demonstrates the value of a dynamic approach like VeriGrey to test agents, and to eventually lead to an agent assurance framework. 

---
# RPMS: Enhancing LLM-Based Embodied Planning through Rule-Augmented Memory Synergy 

**Authors**: Zhenhang Yuan, Shenghai Yuan, Lihua Xie  

**Link**: [PDF](https://arxiv.org/pdf/2603.17831)  

**Abstract**: LLM agents often fail in closed-world embodied environments because actions must satisfy strict preconditions -- such as location, inventory, and container states -- and failure feedback is sparse. We identify two structurally coupled failure modes: (P1) invalid action generation and (P2) state drift, each amplifying the other in a degenerative cycle. We present RPMS, a conflict-managed architecture that enforces action feasibility via structured rule retrieval, gates memory applicability via a lightweight belief state, and resolves conflicts between the two sources via rules-first arbitration. On ALFWorld (134 unseen tasks), RPMS achieves 59.7% single-trial success with Llama 3.1 8B (+23.9 pp over baseline) and 98.5% with Claude Sonnet 4.5 (+11.9 pp); of the 8B gain, rule retrieval alone contributes +14.9 pp (statistically significant), making it the dominant factor. A key finding is that episodic memory is conditionally useful: it harms performance on some task types when used without grounding, but becomes a stable net positive once filtered by current state and constrained by explicit action rules. Adapting RPMS to ScienceWorld with GPT-4 yields consistent gains across all ablation conditions (avg. score 54.0 vs. 44.9 for the ReAct baseline), providing transfer evidence that the core mechanisms hold across structurally distinct environments. 

---
# Facts as First Class Objects: Knowledge Objects for Persistent LLM Memory 

**Authors**: Oliver Zahn, Simran Chana  

**Link**: [PDF](https://arxiv.org/pdf/2603.17781)  

**Abstract**: Large language models increasingly serve as persistent knowledge workers, with in-context memory - facts stored in the prompt - as the default strategy. We benchmark in-context memory against Knowledge Objects (KOs), discrete hash-addressed tuples with O(1) retrieval. Within the context window, Claude Sonnet 4.5 achieves 100% exact-match accuracy from 10 to 7,000 facts (97.5% of its 200K window). However, production deployment reveals three failure modes: capacity limits (prompts overflow at 8,000 facts), compaction loss (summarization destroys 60% of facts), and goal drift (cascading compaction erodes 54% of project constraints while the model continues with full confidence). KOs achieve 100% accuracy across all conditions at 252x lower cost. On multi-hop reasoning, KOs reach 78.9% versus 31.6% for in-context. Cross-model replication across four frontier models confirms compaction loss is architectural, not model-specific. We additionally show that embedding retrieval fails on adversarial facts (20% precision at 1) and that neural memory (Titans) stores facts but fails to retrieve them on demand. We introduce density-adaptive retrieval as a switching mechanism and release the benchmark suite. 

---
# MALLES: A Multi-agent LLMs-based Economic Sandbox with Consumer Preference Alignment 

**Authors**: Yusen Wu, Yiran Liu, Xiaotie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2603.17694)  

**Abstract**: In the real economy, modern decision-making is fundamentally challenged by high-dimensional, multimodal environments, which are further complicated by agent heterogeneity and combinatorial data sparsity. This paper introduces a Multi-Agent Large Language Model-based Economic Sandbox (MALLES), leveraging the inherent generalization capabilities of large-sacle models to establish a unified simulation framework applicable to cross-domain and cross-category scenarios. Central to our approach is a preference learning paradigm in which LLMs are economically aligned via post-training on extensive, heterogeneous transaction records across diverse product categories. This methodology enables the models to internalize and transfer latent consumer preference patterns, thereby mitigating the data sparsity issues prevalent in individual categories. To enhance simulation stability, we implement a mean-field mechanism designed to model the dynamic interactions between the product environment and customer populations, effectively stabilizing sampling processes within high-dimensional decision spaces. Furthermore, we propose a multi-agent discussion framework wherein specialized agents collaboratively process extensive product information. This architecture distributes cognitive load to alleviate single-agent attention bottlenecks and captures critical decision factors through structured dialogue. Experiments demonstrate that our framework achieves significant improvements in product selection accuracy, purchase quantity prediction, and simulation stability compared to existing economic and financial LLM simulation baselines. Our results substantiate the potential of large language models as a foundational pillar for high-fidelity, scalable decision simulation and latter analysis in the real economy based on foundational database. 

---
# Towards Safer Large Reasoning Models by Promoting Safety Decision-Making before Chain-of-Thought Generation 

**Authors**: Jianan Chen, Zhifang Zhang, Shuo He, Linan Yue, Lei Feng, Minling Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.17368)  

**Abstract**: Large reasoning models (LRMs) achieved remarkable performance via chain-of-thought (CoT), but recent studies showed that such enhanced reasoning capabilities are at the expense of significantly degraded safety capabilities. In this paper, we reveal that LRMs' safety degradation occurs only after CoT is enabled, and this degradation is not observed when CoT is disabled. This observation motivates us to consider encouraging LRMs to make safety decisions before CoT generation. To this end, we propose a novel safety alignment method that promotes the safety decision-making of LRMs before starting CoT generation. Specifically, we first utilize a Bert-based classifier to extract safety decision signals from a safe model (e.g., a CoT-disabled LRM) and then integrate these signals into LRMs' safety alignment as auxiliary supervision. In this way, the safety gradients can be backpropagated to the LRMs' latent representations, effectively strengthening the LRMs' safety decision-making abilities against CoT generation. Extensive experiments demonstrate that our method substantially improves the safety capabilities of LRMs while effectively maintaining LRMs' general reasoning performance. 

---
# Sensi: Learn One Thing at a Time -- Curriculum-Based Test-Time Learning for LLM Game Agents 

**Authors**: Mohsen Arjmandi  

**Link**: [PDF](https://arxiv.org/pdf/2603.17683)  

**Abstract**: Large language model (LLM) agents deployed in unknown environments must learn task structure at test time, but current approaches require thousands of interactions to form useful hypotheses. We present Sensi, an LLM agent architecture for the ARC-AGI-3 game-playing challenge that introduces structured test-time learning through three mechanisms: (1) a two-player architecture separating perception from action, (2) a curriculum-based learning system managed by an external state machine, and (3) a database-as-control-plane that makes the agents context window programmatically steerable. We further introduce an LLM-as-judge component with dynamically generated evaluation rubrics to determine when the agent has learned enough about one topic to advance to the next. We report results across two iterations: Sensi v1 solves 2 game levels using the two-player architecture alone, while Sensi v2 adds curriculum learning and solves 0 levels - but completes its entire learning curriculum in approximately 32 action attempts, achieving 50-94x greater sample efficiency than comparable systems that require 1600-3000 attempts. We precisely diagnose the failure mode as a self-consistent hallucination cascade originating in the perception layer, demonstrating that the architectural bottleneck has shifted from learning efficiency to perceptual grounding - a more tractable problem. 

---
# InfoDensity: Rewarding Information-Dense Traces for Efficient Reasoning 

**Authors**: Chengwei Wei, Jung-jae Kim, Longyin Zhang, Shengkai Chen, Nancy F. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.17310)  

**Abstract**: Large Language Models (LLMs) with extended reasoning capabilities often generate verbose and redundant reasoning traces, incurring unnecessary computational cost. While existing reinforcement learning approaches address this by optimizing final response length, they neglect the quality of intermediate reasoning steps, leaving models vulnerable to reward hacking. We argue that verbosity is not merely a length problem, but a symptom of poor intermediate reasoning quality. To investigate this, we conduct an empirical study tracking the conditional entropy of the answer distribution across reasoning steps. We find that high-quality reasoning traces exhibit two consistent properties: low uncertainty convergence and monotonic progress. These findings suggest that high-quality reasoning traces are informationally dense, that is, each step contributes meaningful entropy reduction relative to the total reasoning length. Motivated by this, we propose InfoDensity, a reward framework for RL training that combines an AUC-based reward and a monotonicity reward as a unified measure of reasoning quality, weighted by a length scaling term that favors achieving equivalent quality more concisely. Experiments on mathematical reasoning benchmarks demonstrate that InfoDensity matches or surpasses state-of-the-art baselines in accuracy while significantly reducing token usage, achieving a strong accuracy-efficiency trade-off. 

---
# Generative AI-assisted Participatory Modeling in Socio-Environmental Planning under Deep Uncertainty 

**Authors**: Zhihao Pei, Nir Lipovetzky, Angela M. Rojas-Arevalo, Fjalar J. de Haan, Enayat A. Moallemi  

**Link**: [PDF](https://arxiv.org/pdf/2603.17021)  

**Abstract**: Socio-environmental planning under deep uncertainty requires researchers to identify and conceptualize problems before exploring policies and deploying plans. In practice and model-based planning approaches, this problem conceptualization process often relies on participatory modeling to translate stakeholders' natural-language descriptions into a quantitative model, making this process complex and time-consuming. To facilitate this process, we propose a templated workflow that uses large language models for an initial conceptualization process. During the workflow, researchers can use large language models to identify the essential model components from stakeholders' intuitive problem descriptions, explore their diverse perspectives approaching the problem, assemble these components into a unified model, and eventually implement the model in Python through iterative communication. These results will facilitate the subsequent socio-environmental planning under deep uncertainty steps. Using ChatGPT 5.2 Instant, we demonstrated this workflow on the lake problem and an electricity market problem, both of which demonstrate socio-environmental planning problems. In both cases, acceptable outputs were obtained after a few iterations with human verification and refinement. These experiments indicated that large language models can serve as an effective tool for facilitating participatory modeling in the problem conceptualization process in socio-environmental planning. 

---
# How Clued up are LLMs? Evaluating Multi-Step Deductive Reasoning in a Text-Based Game Environment 

**Authors**: Rebecca Ansell, Autumn Toney-Wails  

**Link**: [PDF](https://arxiv.org/pdf/2603.17169)  

**Abstract**: Deducing whodunit proves challenging for LLM agents. In this paper, we implement a text-based multi-agent version of the classic board game Clue as a rule-based testbed for evaluating multi-step deductive reasoning, with six agents drawn from GPT-4o-mini and Gemini-2.5-Flash. We further investigate whether fine-tuning on structured logic puzzles transfers to improved in-game reasoning and gameplay. Across 18 simulated games, agents achieve only four correct wins, indicating difficulty in maintaining consistent deductive reasoning over the course of a full game. Additionally, we find that fine-tuning does not reliably improve performance and, in some cases, appears to increase reasoning volume without improving reasoning precision. 

---
# IndicSafe: A Benchmark for Evaluating Multilingual LLM Safety in South Asia 

**Authors**: Priyaranjan Pattnayak, Sanchari Chowdhuri  

**Link**: [PDF](https://arxiv.org/pdf/2603.17915)  

**Abstract**: As large language models (LLMs) are deployed in multilingual settings, their safety behavior in culturally diverse, low-resource languages remains poorly understood. We present the first systematic evaluation of LLM safety across 12 Indic languages, spoken by over 1.2 billion people but underrepresented in LLM training data. Using a dataset of 6,000 culturally grounded prompts spanning caste, religion, gender, health, and politics, we assess 10 leading LLMs on translated variants of the prompt.
Our analysis reveals significant safety drift: cross-language agreement is just 12.8\%, and \texttt{SAFE} rate variance exceeds 17\% across languages. Some models over-refuse benign prompts in low-resource scripts, overflag politically sensitive topics, while others fail to flag unsafe generations. We quantify these failures using prompt-level entropy, category bias scores, and multilingual consistency indices.
Our findings highlight critical safety generalization gaps in multilingual LLMs and show that safety alignment does not transfer evenly across languages. We release \textsc{IndicSafe}, the first benchmark to enable culturally informed safety evaluation for Indic deployments, and advocate for language-aware alignment strategies grounded in regional harms. 

---
# Mitigating LLM Hallucinations through Domain-Grounded Tiered Retrieval 

**Authors**: Md. Asraful Haque, Aasar Mehdi, Maaz Mahboob, Tamkeen Fatima  

**Link**: [PDF](https://arxiv.org/pdf/2603.17872)  

**Abstract**: Large Language Models (LLMs) have achieved unprecedented fluency but remain susceptible to "hallucinations" - the generation of factually incorrect or ungrounded content. This limitation is particularly critical in high-stakes domains where reliability is paramount. We propose a domain-grounded tiered retrieval and verification architecture designed to systematically intercept factual inaccuracies by shifting LLMs from stochastic pattern-matchers to verified truth-seekers. The proposed framework utilizes a four-phase, self-regulating pipeline implemented via LangGraph: (I) Intrinsic Verification with Early-Exit logic to optimize compute, (II) Adaptive Search Routing utilizing a Domain Detector to target subject-specific archives, (III) Corrective Document Grading (CRAG) to filter irrelevant context, and (IV) Extrinsic Regeneration followed by atomic claim-level verification. The system was evaluated across 650 queries from five diverse benchmarks: TimeQA v2, FreshQA v2, HaluEval General, MMLU Global Facts, and TruthfulQA. Empirical results demonstrate that the pipeline consistently outperforms zero-shot baselines across all environments. Win rates peaked at 83.7% in TimeQA v2 and 78.0% in MMLU Global Facts, confirming high efficacy in domains requiring granular temporal and numerical precision. Groundedness scores remained robustly stable between 78.8% and 86.4% across factual-answer rows. While the architecture provides a robust fail-safe for misinformation, a persistent failure mode of "False-Premise Overclaiming" was identified. These findings provide a detailed empirical characterization of multi-stage RAG behavior and suggest that future work should prioritize pre-retrieval "answerability" nodes to further bridge the reliability gap in conversational AI. 

---
# RAMP: Reinforcement Adaptive Mixed Precision Quantization for Efficient On Device LLM Inference 

**Authors**: Arpit Singh Gautam, Saurabh Jha  

**Link**: [PDF](https://arxiv.org/pdf/2603.17891)  

**Abstract**: Post training quantization is essential for deploying large language models (LLMs) on resource constrained hardware, yet state of the art methods enforce uniform bit widths across layers, yielding suboptimal accuracy efficiency trade offs. We present RAMP (Reinforcement Adaptive Mixed Precision), an off policy Soft Actor Critic framework that learns per layer bit width assignments to minimize perplexity under a global bit budget. The policy conditions on an 11 dimensional embedding of activation statistics, weight properties, and structural descriptors, enabling zero shot transfer across model families and scales. To enable stable sub 4 bit quantization, we introduce Scale Folding, a preconditioning technique that migrates activation outliers into weights via per channel scaling and normalization layer compensation. A quality prioritized reward with asymmetric penalties and budget cliffs drives rapid convergence. On Llama 2 7B, RAMP achieves 5.54 perplexity at 3.68GB (3.65 effective bits), outperforming uniform 4 bit AWQ (5.60 at 3.90 GB) and GPTQ by 6% in size and 1% to3% in quality. Critically, a policy trained only on Llama 2 7B generalizes zero shot to Llama 2 13B and Mistral 7B, often surpassing target specific training, supporting the hypothesis that quantization sensitivity is primarily architectural. The HALO pipeline exports allocations to GGUF format for kernel free inference on CPUs, GPUs, and edge devices, retaining 99.5% of FP16 commonsense reasoning performance. 

---
# How do LLMs Compute Verbal Confidence 

**Authors**: Dharshan Kumaran, Arthur Conmy, Federico Barbero, Simon Osindero, Viorica Patraucean, Petar Velickovic  

**Link**: [PDF](https://arxiv.org/pdf/2603.17839)  

**Abstract**: Verbal confidence -- prompting LLMs to state their confidence as a number or category -- is widely used to extract uncertainty estimates from black-box models. However, how LLMs internally generate such scores remains unknown. We address two questions: first, when confidence is computed - just-in-time when requested, or automatically during answer generation and cached for later retrieval; and second, what verbal confidence represents - token log-probabilities, or a richer evaluation of answer quality? Focusing on Gemma 3 27B and Qwen 2.5 7B, we provide convergent evidence for cached retrieval. Activation steering, patching, noising, and swap experiments reveal that confidence representations emerge at answer-adjacent positions before appearing at the verbalization site. Attention blocking pinpoints the information flow: confidence is gathered from answer tokens, cached at the first post-answer position, then retrieved for output. Critically, linear probing and variance partitioning reveal that these cached representations explain substantial variance in verbal confidence beyond token log-probabilities, suggesting a richer answer-quality evaluation rather than a simple fluency readout. These findings demonstrate that verbal confidence reflects automatic, sophisticated self-evaluation -- not post-hoc reconstruction -- with implications for understanding metacognition in LLMs and improving calibration. 

---
# CoVerRL: Breaking the Consensus Trap in Label-Free Reasoning via Generator-Verifier Co-Evolution 

**Authors**: Teng Pan, Yuchen Yan, Zixuan Wang, Ruiqing Zhang, Gaiyang Han, Wanqi Zhang, Weiming Lu, Jun Xiao, Yongliang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2603.17775)  

**Abstract**: Label-free reinforcement learning enables large language models to improve reasoning capabilities without ground-truth supervision, typically by treating majority-voted answers as pseudo-labels. However, we identify a critical failure mode: as training maximizes self-consistency, output diversity collapses, causing the model to confidently reinforce systematic errors that evade detection. We term this the consensus trap. To escape it, we propose CoVerRL, a framework where a single model alternates between generator and verifier roles, with each capability bootstrapping the other. Majority voting provides noisy but informative supervision for training the verifier, while the improving verifier progressively filters self-consistent errors from pseudo-labels. This co-evolution creates a virtuous cycle that maintains high reward accuracy throughout training. Experiments across Qwen and Llama model families demonstrate that CoVerRL outperforms label-free baselines by 4.7-5.9\% on mathematical reasoning benchmarks. Moreover, self-verification accuracy improves from around 55\% to over 85\%, confirming that both capabilities genuinely co-evolve. 

---
# FINER: MLLMs Hallucinate under Fine-grained Negative Queries 

**Authors**: Rui Xiao, Sanghwan Kim, Yongqin Xian, Zeynep Akata, Stephan Alaniz  

**Link**: [PDF](https://arxiv.org/pdf/2603.17662)  

**Abstract**: Multimodal large language models (MLLMs) struggle with hallucinations, particularly with fine-grained queries, a challenge underrepresented by existing benchmarks that focus on coarse image-related questions. We introduce FIne-grained NEgative queRies (FINER), alongside two benchmarks: FINER-CompreCap and FINER-DOCCI. Using FINER, we analyze hallucinations across four settings: multi-object, multi-attribute, multi-relation, and ``what'' questions. Our benchmarks reveal that MLLMs hallucinate when fine-grained mismatches co-occur with genuinely present elements in the image. To address this, we propose FINER-Tuning, leveraging Direct Preference Optimization (DPO) on FINER-inspired data. Finetuning four frontier MLLMs with FINER-Tuning yields up to 24.2\% gains (InternVL3.5-14B) on hallucinations from our benchmarks, while simultaneously improving performance on eight existing hallucination suites, and enhancing general multimodal capabilities across six benchmarks. Code, benchmark, and models are available at \href{this https URL}{this https URL}. 

---
# A Contextual Help Browser Extension to Assist Digital Illiterate Internet Users 

**Authors**: Christos Koutsiaris  

**Link**: [PDF](https://arxiv.org/pdf/2603.17592)  

**Abstract**: This paper describes the design, implementation, and evaluation of a browser extension that provides contextual help to users who hover over technological acronyms and abbreviations on web pages. The extension combines a curated technical dictionary with OpenAI's large language model (LLM) to deliver on-demand definitions through lightweight tooltip overlays. A dual-layer artificial intelligence (AI) pipeline, comprising Google Cloud's Natural Language Processing (NLP) taxonomy API and OpenAI's ChatGPT, classifies each visited page as technology-related before activating the tooltip logic, thereby reducing false-positive detections. A mixed-methods study with 25 participants evaluated the tool's effect on reading comprehension and information-retrieval time among users with low to intermediate digital literacy. Results show that 92% of participants reported improved understanding of technical terms, 96% confirmed time savings over manual web searches, and all participants found the tooltips non-disruptive. Dictionary-based definitions were appended in an average of 2135 ms, compared to 16429 ms for AI-generated definitions and a mean manual search time of 17200 ms per acronym. The work demonstrates a practical, real-time approach to bridging the digital literacy gap and points toward extending contextual help to other domains such as medicine, law, and finance. 

---
# Efficient Exploration at Scale 

**Authors**: Seyed Mohammad Asghari, Chris Chute, Vikranth Dwaracherla, Xiuyuan Lu, Mehdi Jafarnia, Victor Minden, Zheng Wen, Benjamin Van Roy  

**Link**: [PDF](https://arxiv.org/pdf/2603.17378)  

**Abstract**: We develop an online learning algorithm that dramatically improves the data efficiency of reinforcement learning from human feedback (RLHF). Our algorithm incrementally updates reward and language models as choice data is received. The reward model is fit to the choice data, while the language model is updated by a variation of reinforce, with reinforcement signals provided by the reward model. Several features enable the efficiency gains: a small affirmative nudge added to each reinforcement signal, an epistemic neural network that models reward uncertainty, and information-directed exploration. With Gemma large language models (LLMs), our algorithm matches the performance of offline RLHF trained on 200K labels using fewer than 20K labels, representing more than a 10x gain in data efficiency. Extrapolating from our results, we expect our algorithm trained on 1M labels to match offline RLHF trained on 1B labels. This represents a 1,000x gain. To our knowledge, these are the first results to demonstrate that such large improvements are possible. 

---
# TharuChat: Bootstrapping Large Language Models for a Low-Resource Language via Synthetic Data and Human Validation 

**Authors**: Prajwal Panth, Agniva Maiti  

**Link**: [PDF](https://arxiv.org/pdf/2603.17220)  

**Abstract**: The rapid proliferation of Large Language Models (LLMs) has created a profound digital divide, effectively excluding indigenous languages of the Global South from the AI revolution. The Tharu language, an Indo-Aryan vernacular spoken by approximately 1.7 million people across the Terai belt of Nepal and India, exemplifies this crisis. Despite a rich oral tradition, Tharu suffers from severe data scarcity and linguistic fragmentation, causing state-of-the-art multilingual models to routinely "hallucinate" or default to dominant high-resource neighbors like Hindi and Nepali due to contamination in pre-training corpora.
This paper presents Tharu-LLaMA (3B), a specialized instruction-following model designed to address this exclusion. We introduce TharuChat, a novel dataset constructed via a LLM-to-Human bootstrapping pipeline. We utilized prompt-engineered Gemini models, fed with Rana Tharu grammar and folklore, to synthesize training data. Unlike curated gold-standard corpora, TharuChat reflects the noisy, heterogeneous linguistic reality of the region: it is predominantly anchored in Rana Tharu (~70%) while integrating elements of Dangaura and Kochila dialects. We provide a transparent analysis of the dataset's limitations, including dialectal code-mixing and residual Awadhi/Hindi influence. Through a rigorous empirical ablation study, we demonstrate that despite these imperfections, small-scale synthetic data is highly effective, increasing the dataset volume from 25% to 100% results in a linear reduction in perplexity from 6.42 to 2.88. The resulting model serves as a proof-of-concept for the preservation of under-resourced Himalayan languages via generative AI, achievable on consumer-grade hardware. 

---
# Deployment and Evaluation of an EHR-integrated, Large Language Model-Powered Tool to Triage Surgical Patients 

**Authors**: Jane Wang, Timothy Keyes, April S Liang, Stephen P Ma, Jason Shen, Jerry Liu, Nerissa Ambers, Abby Pandya, Rita Pandya, Jason Hom, Natasha Steele, Jonathan H Chen, Kevin Schulman  

**Link**: [PDF](https://arxiv.org/pdf/2603.17234)  

**Abstract**: Surgical co-management (SCM) is an evidence-based model in which hospitalists jointly manage medically complex perioperative patients alongside surgical teams. Despite its clinical and financial value, SCM is limited by the need to manually identify eligible patients. To determine whether SCM triage can be automated, we conducted a prospective, unblinded study at Stanford Health Care in which an LLM-based, electronic health record (EHR)-integrated triage tool (SCM Navigator) provided SCM recommendations followed by physician review.
Using pre-operative documentation, structured data, and clinical criteria for perioperative morbidity, SCM Navigator categorized patients as appropriate, not appropriate, or possibly appropriate for SCM. Faculty indicated their clinical judgment and provided free-text feedback when they disagreed. Sensitivity, specificity, positive predictive value, and negative predictive value were measured using physician determinations as a reference. Free-text reasons were thematically categorized, and manual chart review was conducted on all false-negative cases and 30 randomly selected cases from the largest false-positive category. Since deployment, 6,193 cases have been triaged, of which 1,582 (23%) were recommended for hospitalist consultation. SCM Navigator displayed high sensitivity (0.94, 95% CI 0.91-0.96) and moderate specificity (0.74, 95% CI 0.71-0.77). Post-hoc chart review suggested most discrepancies reflect modifiable gaps in clinical criteria, institutional workflow, or physician practice variability rather than LLM misclassification, which accounted for 2 of 19 (11%) false-negative cases. These findings demonstrate that an LLM-powered, EHR-integrated, human-in-the-loop AI system can accurately and safely triage surgical patients for SCM, and that AI-enabled screening tools can augment and potentially automate time-intensive clinical workflows. 

---
# Anonymous-by-Construction: An LLM-Driven Framework for Privacy-Preserving Text 

**Authors**: Federico Albanese, Pablo Ronco, Nicolás D'Ippolito  

**Link**: [PDF](https://arxiv.org/pdf/2603.17217)  

**Abstract**: Responsible use of AI demands that we protect sensitive information without undermining the usefulness of data, an imperative that has become acute in the age of large language models. We address this challenge with an on-premise, LLM-driven substitution pipeline that anonymizes text by replacing personally identifiable information (PII) with realistic, type-consistent surrogates. Executed entirely within organizational boundaries using local LLMs, the approach prevents data egress while preserving fluency and task-relevant semantics.
We conduct a systematic, multi-metric, cross-technique evaluation on the Action-Based Conversation Dataset, benchmarking against industry standards (Microsoft Presidio and Google DLP) and a state-of-the-art approach (ZSTS, in redaction-only and redaction-plus-substitution variants). Our protocol jointly measures privacy, semantic utility, and trainability under privacy via a lifecycle-ready criterion obtained by fine-tuning a compact encoder (BERT+LoRA) on sanitized text. In addition, we assess agentic Q&A performance by inserting an on-premise anonymization layer before the answering LLM and evaluating the quality of its responses. This intermediate, type-preserving substitution stage ensures that no sensitive content is exposed to third-party APIs, enabling responsible deployment of Q\&A agents without compromising confidentiality.
Our method attains state-of-the-art privacy, minimal topical drift, strong factual utility, and low trainability loss, outperforming rule-based approaches and named-entity recognition (NER) baselines and ZSTS variants on the combined privacy--utility--trainability frontier. These results show that local LLM substitution yields anonymized corpora that are both responsible to use and operationally valuable: safe for agentic pipelines and suitable for downstream fine-tuning with limited degradation. 

---
# Detecting Data Poisoning in Code Generation LLMs via Black-Box, Vulnerability-Oriented Scanning 

**Authors**: Shenao Yan, Shimaa Ahmed, Shan Jin, Sunpreet S. Arora, Yiwei Cai, Yizhen Wang, Yuan Hong  

**Link**: [PDF](https://arxiv.org/pdf/2603.17174)  

**Abstract**: Code generation large language models (LLMs) are increasingly integrated into modern software development workflows. Recent work has shown that these models are vulnerable to backdoor and poisoning attacks that induce the generation of insecure code, yet effective defenses remain limited. Existing scanning approaches rely on token-level generation consistency to invert attack targets, which is ineffective for source code where identical semantics can appear in diverse syntactic forms. We present CodeScan, which, to the best of our knowledge, is the first poisoning-scanning framework tailored to code generation models. CodeScan identifies attack targets by analyzing structural similarities across multiple generations conditioned on different clean prompts. It combines iterative divergence analysis with abstract syntax tree (AST)-based normalization to abstract away surface-level variation and unify semantically equivalent code, isolating structures that recur consistently across generations. CodeScan then applies LLM-based vulnerability analysis to determine whether the extracted structures contain security vulnerabilities and flags the model as compromised when such a structure is found. We evaluate CodeScan against four representative attacks under both backdoor and poisoning settings across three real-world vulnerability classes. Experiments on 108 models spanning three architectures and multiple model sizes demonstrate 97%+ detection accuracy with substantially lower false positives than prior methods. 

---
# Catching rationalization in the act: detecting motivated reasoning before and after CoT via activation probing 

**Authors**: Parsa Mirtaheri, Mikhail Belkin  

**Link**: [PDF](https://arxiv.org/pdf/2603.17199)  

**Abstract**: Large language models (LLMs) can produce chains of thought (CoT) that do not accurately reflect the actual factors driving their answers. In multiple-choice settings with an injected hint favoring a particular option, models may shift their final answer toward the hinted option and produce a CoT that rationalizes the response without acknowledging the hint - an instance of motivated reasoning. We study this phenomenon across multiple LLM families and datasets demonstrating that motivated reasoning can be identified by probing internal activations even in cases when it cannot be easily determined from CoT. Using supervised probes trained on the model's residual stream, we show that (i) pre-generation probes, applied before any CoT tokens are generated, predict motivated reasoning as well as a LLM-based CoT monitor that accesses the full CoT trace, and (ii) post-generation probes, applied after CoT generation, outperform the same monitor. Together, these results show that motivated reasoning is detected more reliably from internal representations than from CoT monitoring. Moreover, pre-generation probing can flag motivated behavior early, potentially avoiding unnecessary generation. 

---
# Generalist Multimodal LLMs Gain Biometric Expertise via Human Salience 

**Authors**: Jacob Piland, Byron Dowling, Christopher Sweet, Adam Czajka  

**Link**: [PDF](https://arxiv.org/pdf/2603.17173)  

**Abstract**: Iris presentation attack detection (PAD) is critical for secure biometric deployments, yet developing specialized models faces significant practical barriers: collecting data representing future unknown attacks is impossible, and collecting diverse-enough data, yet still limited in terms of its predictive power, is expensive. Additionally, sharing biometric data raises privacy concerns. Due to rapid emergence of new attack vectors demanding adaptable solutions, we thus investigate in this paper whether general-purpose multimodal large language models (MLLMs) can perform iris PAD when augmented with human expert knowledge, operating under strict privacy constraints that prohibit sending biometric data to public cloud MLLM services. Through analysis of vision encoder embeddings applied to our dataset, we demonstrate that pre-trained vision transformers in MLLMs inherently cluster many iris attack types despite never being explicitly trained for this task. However, where clustering shows overlap between attack classes, we find that structured prompts incorporating human salience (verbal descriptions from subjects identifying attack indicators) enable these models to resolve ambiguities. Testing on an IRB-restricted dataset of 224 iris images spanning seven attack types, using only university-approved services (Gemini 2.5 Pro) or locally-hosted models (e.g., Llama 3.2-Vision), we show that Gemini with expert-informed prompts outperforms both a specialized convolutional neural networks (CNN)-based baseline and human examiners, while the locally-deployable Llama achieves near-human performance. Our results establish that MLLMs deployable within institutional privacy constraints offer a viable path for iris PAD. 

---
# Knowledge Localization in Mixture-of-Experts LLMs Using Cross-Lingual Inconsistency 

**Authors**: Lucas Bandarkar, Alan Ansell, Trevor Cohn  

**Link**: [PDF](https://arxiv.org/pdf/2603.17102)  

**Abstract**: Modern LLMs continue to exhibit significant variance in behavior across languages, such as being able to recall factual information in some languages but not others. While typically studied as a problem to be mitigated, in this work, we propose leveraging this cross-lingual inconsistency as a tool for interpretability in mixture-of-experts (MoE) LLMs. Our knowledge localization framework contrasts routing for sets of languages where the model correctly recalls information from languages where it fails. This allows us to isolate model components that play a functional role in answering about a piece of knowledge. Our method proceeds in two stages: (1) querying the model with difficult factual questions across a diverse set of languages to generate "success" and "failure" activation buckets and then (2) applying a statistical contrastive analysis to the MoE router logits to identify experts important for knowledge. To validate the necessity of this small number of experts for answering a knowledge question, we deactivate them and re-ask the question. We find that despite only deactivating about 20 out of 6000 experts, the model no longer answers correctly in over 40% of cases. Generally, this method provides a realistic and scalable knowledge localization approach to address increasingly complex LLMs. 

---
# Large Reasoning Models Struggle to Transfer Parametric Knowledge Across Scripts 

**Authors**: Lucas Bandarkar, Alan Ansell, Trevor Cohn  

**Link**: [PDF](https://arxiv.org/pdf/2603.17070)  

**Abstract**: In this work, we analyze shortcomings in cross-lingual knowledge transfer in large, modern reasoning LLMs. We demonstrate that the perceived gap in knowledge transfer is primarily a script barrier. First, we conduct an observational data analysis on the performance of thinking models on two datasets with local knowledge from around the world, ECLeKTic and MultiLoKo. Our regression analysis shows that script match - not language or family - is the primary predictor of knowledge transfer failure once model capability and question difficulty are accounted for. We further this finding by providing the LLMs with the key entities of the questions in their source language and find that this disproportionately improves cross-script questions. We then posit that these LLMs could be reasoning better at test-time. To evaluate this, we develop a synthetic generation pipeline to design SFT samples to encourage the model to better reason about transliteration ambiguities when trying to fetch parametric knowledge at inference-time. We show that teaching two models to reason better reduces the cross-script transfer gap. As a result, we conclude that there is potential to improve cross-lingual parametric knowledge transfer during post-training. 

---
# Evaluating Ill-Defined Tasks in Large Language Models 

**Authors**: Yi Zhou, Basel Shbita  

**Link**: [PDF](https://arxiv.org/pdf/2603.17067)  

**Abstract**: Many evaluations of Large Language Models (LLMs) target tasks that are inherently ill-defined, with unclear input and output spaces and ambiguous success criteria. We analyze why existing evaluation benchmarks and metrics fail to provide reliable or diagnostic signals of model capability for such tasks. We examine two case studies: Complex Instruction Following (CIF), where we identify recurring issues including limited coverage of real-world instruction complexity, sensitivity to instruction phrasing, inconsistent and non-comparable metrics, and instability introduced by LLM-based judges; and Natural Language to Mermaid Sequence Diagrams (NL2Mermaid), where we show how multi-faceted evaluation criteria can yield actionable insights beyond aggregate scores. Together, these case studies show that current evaluations frequently conflate distinct failure modes, yielding scores that are unstable, non-diagnostic, and difficult to act upon. Our findings expose fundamental limitations in existing evaluation practices for ill-defined tasks and motivate more robust, interpretable evaluation designs. 

---
# MSRAMIE: Multimodal Structured Reasoning Agent for Multi-instruction Image Editing 

**Authors**: Zhaoyuan Qiu, Ken Chen, Xiangwei Wang, Yu Xia, Sachith Seneviratne, Saman Halgamuge  

**Link**: [PDF](https://arxiv.org/pdf/2603.16967)  

**Abstract**: Existing instruction-based image editing models perform well with simple, single-step instructions but degrade in realistic scenarios that involve multiple, lengthy, and interdependent directives. A main cause is the scarcity of training data with complex multi-instruction annotations. However, it is costly to collect such data and retrain these models. To address this challenge, we propose MSRAMIE, a training-free agent framework built on Multimodal Large Language Model (MLLM). MSRAMIE takes existing editing models as plug-in components and handle multi-instruction tasks via structured multimodal reasoning. It orchestrates iterative interactions between an MLLM-based Instructor and an image editing Actor, introducing a novel reasoning topology that comprises the proposed Tree-of-States and Graph-of-References. During inference, complex instructions are decomposed into multiple editing steps which enable state transitions, cross-step information aggregation, and original input recall, which enables systematic exploration of the image editing space and flexible progressive output refinement. The visualizable inference topology further provides interpretable and controllable decision pathways. Experiments show that as the instruction complexity increases, MSRAMIE can improve instruction following over 15% and increases the probability of finishing all modifications in a single run over 100%, while preserving perceptual quality and maintaining visual consistency. 

---
# Are a Thousand Words Better Than a Single Picture? Beyond Images -- A Framework for Multi-Modal Knowledge Graph Dataset Enrichment 

**Authors**: Pengyu Zhang, Klim Zaporojets, Jie Liu, Jia-Hong Huang, Paul Groth  

**Link**: [PDF](https://arxiv.org/pdf/2603.16974)  

**Abstract**: Multi-Modal Knowledge Graphs (MMKGs) benefit from visual information, yet large-scale image collection is hard to curate and often excludes ambiguous but relevant visuals (e.g., logos, symbols, abstract scenes). We present Beyond Images, an automatic data-centric enrichment pipeline with optional human auditing. This pipeline operates in three stages: (1) large-scale retrieval of additional entity-related images, (2) conversion of all visual inputs into textual descriptions to ensure that ambiguous images contribute usable semantics rather than noise, and (3) fusion of multi-source descriptions using a large language model (LLM) to generate concise, entity-aligned summaries. These summaries replace or augment the text modality in standard MMKG models without changing their architectures or loss functions. Across three public MMKG datasets and multiple baseline models, we observe consistent gains (up to 7% Hits@1 overall). Furthermore, on a challenging subset of entities with visually ambiguous logos and symbols, converting images into text yields large improvements (201.35% MRR and 333.33% Hits@1). Additionally, we release a lightweight Text-Image Consistency Check Interface for optional targeted audits, improving description quality and dataset reliability. Our results show that scaling image coverage and converting ambiguous visuals into text is a practical path to stronger MMKG completion. Code, datasets, and supplementary materials are available at this https URL. 

---
# Script-to-Slide Grounding: Grounding Script Sentences to Slide Objects for Automatic Instructional Video Generation 

**Authors**: Rena Suzuki, Masato Kikuchi, Tadachika Ozono  

**Link**: [PDF](https://arxiv.org/pdf/2603.16931)  

**Abstract**: While slide-based videos augmented with visual effects are widely utilized in education and research presentations, the video editing process -- particularly applying visual effects to ground spoken content to slide objects -- remains highly labor-intensive. This study aims to develop a system that automatically generates such instructional videos from slides and corresponding scripts. As a foundational step, this paper proposes and formulates Script-to-Slide Grounding (S2SG), defined as the task of grounding script sentences to their corresponding slide objects. Furthermore, as an initial step, we propose ``Text-S2SG,'' a method that utilizes a large language model (LLM) to perform this grounding task for text objects. Our experiments demonstrate that the proposed method achieves high performance (F1-score: 0.924). The contribution of this work is the formalization of a previously implicit slide-based video editing process into a computable task, thereby paving the way for its automation. 

---
# LLM NL2SQL Robustness: Surface Noise vs. Linguistic Variation in Traditional and Agentic Settings 

**Authors**: Lifu Tu, Rongguang Wang, Tao Sheng, Sujjith Ravi, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2603.17017)  

**Abstract**: Robustness evaluation for Natural Language to SQL (NL2SQL) systems is essential because real-world database environments are dynamic, noisy, and continuously evolving, whereas conventional benchmark evaluations typically assume static schemas and well-formed user inputs. In this work, we introduce a robustness evaluation benchmark containing approximately ten types of perturbations and conduct evaluations under both traditional and agentic settings. We assess multiple state-of-the-art large language models (LLMs), including Grok-4.1, Gemini-3-Pro, Claude-Opus-4.6, and GPT-5.2. Our results show that these models generally maintain strong performance under several perturbations; however, notable performance degradation is observed for surface-level noise (e.g., character-level corruption) and linguistic variation that preserves semantics while altering lexical or syntactic forms. Furthermore, we observe that surface-level noise causes larger performance drops in traditional pipelines, whereas linguistic variation presents greater challenges in agentic settings. These findings highlight the remaining challenges in achieving robust NL2SQL systems, particularly in handling linguistic variability. 

---
# AgriChat: A Multimodal Large Language Model for Agriculture Image Understanding 

**Authors**: Abderrahmene Boudiaf, Irfan Hussain, Sajid Javed  

**Link**: [PDF](https://arxiv.org/pdf/2603.16934)  

**Abstract**: The deployment of Multimodal Large Language Models (MLLMs) in agriculture is currently stalled by a critical trade-off: the existing literature lacks the large-scale agricultural datasets required for robust model development and evaluation, while current state-of-the-art models lack the verified domain expertise necessary to reason across diverse taxonomies. To address these challenges, we propose the Vision-to-Verified-Knowledge (V2VK) pipeline, a novel generative AI-driven annotation framework that integrates visual captioning with web-augmented scientific retrieval to autonomously generate the AgriMM benchmark, effectively eliminating biological hallucinations by grounding training data in verified phytopathological literature. The AgriMM benchmark contains over 3,000 agricultural classes and more than 607k VQAs spanning multiple tasks, including fine-grained plant species identification, plant disease symptom recognition, crop counting, and ripeness assessment. Leveraging this verifiable data, we present AgriChat, a specialized MLLM that presents broad knowledge across thousands of agricultural classes and provides detailed agricultural assessments with extensive explanations. Extensive evaluation across diverse tasks, datasets, and evaluation conditions reveals both the capabilities and limitations of current agricultural MLLMs, while demonstrating AgriChat's superior performance over other open-source models, including internal and external benchmarks. The results validate that preserving visual detail combined with web-verified knowledge constitutes a reliable pathway toward robust and trustworthy agricultural AI. The code and dataset are publicly available at this https URL . 

---
# Rubric-Guided Fine-tuning of SpeechLLMs for Multi-Aspect, Multi-Rater L2 Reading-Speech Assessment 

**Authors**: Aditya Kamlesh Parikh, Cristian Tejedor-Garcia, Catia Cucchiarini, Helmer Strik  

**Link**: [PDF](https://arxiv.org/pdf/2603.16889)  

**Abstract**: Reliable and interpretable automated assessment of second-language (L2) speech remains a central challenge, as large speech-language models (SpeechLLMs) often struggle to align with the nuanced variability of human raters. To address this, we introduce a rubric-guided reasoning framework that explicitly encodes multi-aspect human assessment criteria: accuracy, fluency, and prosody, while calibrating model uncertainty to capture natural rating variability. We fine-tune the Qwen2-Audio-7B-Instruct model using multi-rater human judgments and develop an uncertainty-calibrated regression approach supported by conformal calibration for interpretable confidence intervals. Our Gaussian uncertainty modeling and conformal calibration approach achieves the strongest alignment with human ratings, outperforming regression and classification baselines. The model reliably assesses fluency and prosody while highlighting the inherent difficulty of assessing accuracy. Together, these results demonstrate that rubric-guided, uncertainty-calibrated reasoning offers a principled path toward trustworthy and explainable SpeechLLM-based speech assessment. 

---
# Social physics in the age of artificial intelligence 

**Authors**: Anh Han, Joel Z. Leibo, Tom Lenaerts, Iyad Rahwan, Fernando Santos, Matjaž Perc, Valerio Capraro  

**Link**: [PDF](https://arxiv.org/pdf/2603.16900)  

**Abstract**: Artificial intelligence (AI) systems are rapidly becoming more capable, autonomous, and deeply embedded in social life. As humans increasingly interact, cooperate, and compete with AI, we move from purely human societies to hybrid human-AI societies whose collective dynamics cannot be captured by existing behavioural models alone. Drawing on evolutionary game theory, cultural evolution, and Large Language Models (LLMs) powered simulations, we argue that these developments open a new research agenda for social physics centred on the co-evolution of humans and machines. We outline six key research directions. First, modelling the evolutionary dynamics of social behaviours (e.g. cooperation, fairness, trust) in hybrid human-AI populations. Second, understanding machine culture: how AI systems generate, mediate, and select cultural traits. Third, analysing the co-evolution of language and behaviour when LLMs frame and participate in decisions. Fourth, studying the evolution of AI delegation: how responsibilities and control are negotiated between humans and machines. Fifth, formalising and comparing the distinct epistemic pipelines that generate human and AI behaviour. Sixth, modelling the co-evolution of AI development and regulation in a strategic ecosystem of firms, users, and institutions. Together, these directions define a programme for using social physics to anticipate and steer the societal impact of advanced AI. 

---
# From Isolated Scoring to Collaborative Ranking: A Comparison-Native Framework for LLM-Based Paper Evaluation 

**Authors**: Pujun Zheng, Jiacheng Yao, Jinquan Zheng, Chenyang Gu, Guoxiu He, Jiawei Liu, Yong Huang, Tianrui Guo, Wei Lu  

**Link**: [PDF](https://arxiv.org/pdf/2603.17588)  

**Abstract**: Large language models (LLMs) are currently applied to scientific paper evaluation by assigning an absolute score to each paper independently. However, since score scales vary across conferences, time periods, and evaluation criteria, models trained on absolute scores are prone to fitting narrow, context-specific rules rather than developing robust scholarly judgment. To overcome this limitation, we propose shifting paper evaluation from isolated scoring to collaborative ranking. In particular, we design \textbf{C}omparison-\textbf{N}ative framework for \textbf{P}aper \textbf{E}valuation (\textbf{CNPE}), integrating comparison into both data construction and model learning. We first propose a graph-based similarity ranking algorithm to facilitate the sampling of more informative and discriminative paper pairs from a collection. We then enhance relative quality judgment through supervised fine-tuning and reinforcement learning with comparison-based rewards. At inference, the model performs pairwise comparisons over sampled paper pairs and aggregates these preference signals into a global relative quality ranking. Experimental results demonstrate that our framework achieves an average relative improvement of \textbf{21.8\%} over the strong baseline DeepReview-14B, while exhibiting robust generalization to five previously unseen datasets. \href{this https URL}{Code}. 

---
# Deploying Semantic ID-based Generative Retrieval for Large-Scale Podcast Discovery at Spotify 

**Authors**: Edoardo D'Amico, Marco De Nadai, Praveen Chandar, Divita Vohra, Shawn Lin, Max Lefarov, Paul Gigioli, Gustavo Penha, Ilya Kopysitsky, Ivo Joel Senese, Darren Mei, Francesco Fabbri, Oguz Semerci, Yu Zhao, Vincent Tang, Brian St. Thomas, Alexandra Ranieri, Matthew N.K. Smith, Aaron Bernkopf, Bryan Leung, Ghazal Fazelnia, Mark VanMiddlesworth, Timothy Christopher Heath, Petter Pehrson Skiden, Alice Y. Wang, Doug J. Cole, Andreas Damianou, Maya Hristakeva, Reid Wilbur, Tarun Chillara, Vladan Radosavljevic, Pooja Chitkara, Sainath Adapa, Juan Elenter, Bernd Huber, Jacqueline Wood, Saaketh Vedantam, Jan Stypka, Sandeep Ghael, Martin D. Gould, David Murgatroyd, Yves Raimond, Mounia Lalmas, Paul N. Bennett  

**Link**: [PDF](https://arxiv.org/pdf/2603.17540)  

**Abstract**: Podcast listening is often grounded in a set of favorite shows, while listener intent can evolve over time. This combination of stable preferences and changing intent motivates recommendation approaches that support both familiarity and exploration. Traditional recommender systems typically emphasize long-term interaction patterns, and are less explicitly designed to incorporate rich contextual signals or flexible, intent-aware discovery objectives. In this setting, models that can jointly reason over semantics, context, and user state offer a promising direction. Large Language Models (LLMs) provide strong semantic reasoning and contextual conditioning for discovery-oriented recommendation, but deploying them in production introduces challenges in catalog grounding, user-level personalization, and latency-critical serving.
We address these challenges with GLIDE, a production-scale generative recommender for podcast discovery at Spotify. GLIDE formulates recommendation as an instruction-following task over a discretized catalog using Semantic IDs, enabling grounded generation over a large inventory. The model conditions on recent listening history and lightweight user context, while injecting long-term user embeddings as soft prompts to capture stable preferences under strict inference constraints. We evaluate GLIDE using offline retrieval metrics, human judgments, and LLM-based evaluation, and validate its impact through large-scale online A/B testing. Across experiments involving millions of users, GLIDE increases non-habitual podcast streaming on Spotify home surface by up to 5.4% and new-show discovery by up to 14.3%, while meeting production cost and latency constraints. 

---
# A Unified Language Model for Large Scale Search, Recommendation, and Reasoning 

**Authors**: Marco De Nadai, Edoardo D'Amico, Max Lefarov, Alexandre Tamborrino, Divita Vohra, Mark VanMiddlesworth, Shawn Lin, Jacqueline Wood, Jan Stypka, Eliza Klyce, Keshi Dai, Timothy Christopher Heath, Martin D. Gould, Yves Raimond, Sandeep Ghael, Tony Jebara, Andreas Damianou, Vladan Radosavljevic, Paul N. Bennett, Mounia Lalmas, Praveen Chandar  

**Link**: [PDF](https://arxiv.org/pdf/2603.17533)  

**Abstract**: LLMs are increasingly applied to recommendation, retrieval, and reasoning, yet deploying a single end-to-end model that can jointly support these behaviors over large, heterogeneous catalogs remains challenging. Such systems must generate unambiguous references to real items, handle multiple entity types, and operate under strict latency and reliability constraints requirements that are difficult to satisfy with text-only generation. While tool-augmented recommender systems address parts of this problem, they introduce orchestration complexity and limit end-to-end optimization. We view this setting as an instance of a broader research problem: how to adapt LLMs to reason jointly over multiple-domain entities, users, and language in a fully self-contained manner. To this end, we introduce NEO, a framework that adapts a pre-trained decoder-only LLM into a tool-free, catalog-grounded generator. NEO represents items as SIDs and trains a single model to interleave natural language and typed item identifiers within a shared sequence. Text prompts control the task, target entity type, and output format (IDs, text, or mixed), while constrained decoding guarantees catalog-valid item generation without restricting free-form text. We refer to this instruction-conditioned controllability as language-steerability. We treat SIDs as a distinct modality and study design choices for integrating discrete entity representations into LLMs via staged alignment and instruction tuning. We evaluate NEO at scale on a real-world catalog of over 10M items across multiple media types and discovery tasks, including recommendation, search, and user understanding. In offline experiments, NEO consistently outperforms strong task-specific baselines and exhibits cross-task transfer, demonstrating a practical path toward consolidating large-scale discovery capabilities into a single language-steerable generative model. 

---
# Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing 

**Authors**: Raghavv Goel, Mukul Gagrani, Mingu Lee, Chris Lott  

**Link**: [PDF](https://arxiv.org/pdf/2603.17942)  

**Abstract**: Large language models (LLMs) exhibit latent multi-token prediction (MTP) capabilities despite being trained solely for next-token generation. We propose a simple, training-free MTP approach that probes an LLM using on-the-fly mask tokens drawn from its embedding space, enabling parallel prediction of future tokens without modifying model weights or relying on auxiliary draft models. Our method constructs a speculative token tree by sampling top-K candidates from mask-token logits and applies a lightweight pruning strategy to retain high-probability continuations. During decoding, candidate predictions are verified in parallel, resulting in lossless generation while substantially reducing the number of model calls and improving token throughput. Across benchmarks, our probing-based MTP consistently outperforms existing training-free baselines, increasing acceptance length by approximately 12\% on LLaMA3 and 8--12\% on Qwen3, and achieving throughput gains of up to 15--19\%. Finally, we provide theoretical insights and empirical evidence showing that decoder layers naturally align mask-token representations with next-token states, enabling accurate multi-step prediction without retraining or auxiliary models. 

---
# Process Supervision for Chain-of-Thought Reasoning via Monte Carlo Net Information Gain 

**Authors**: Corentin Royer, Debarun Bhattacharjya, Gaetano Rossiello, Andrea Giovannini, Mennatallah El-Assady  

**Link**: [PDF](https://arxiv.org/pdf/2603.17815)  

**Abstract**: Multi-step reasoning improves the capabilities of large language models (LLMs) but increases the risk of errors propagating through intermediate steps. Process reward models (PRMs) mitigate this by scoring each step individually, enabling fine-grained supervision and improved reliability. Existing methods for training PRMs rely on costly human annotations or computationally intensive automatic labeling. We propose a novel approach to automatically generate step-level labels using Information Theory. Our method estimates how each reasoning step affects the likelihood of the correct answer, providing a signal of step quality. Importantly, it reduces computational complexity to $\mathcal{O}(N)$, improving over the previous $\mathcal{O}(N \log N)$ methods. We demonstrate that these labels enable effective chain-of-thought selection in best-of-$K$ evaluation settings across diverse reasoning benchmarks, including mathematics, Python programming, SQL, and scientific question answering. This work enables scalable and efficient supervision of LLM reasoning, particularly for tasks where error propagation is critical. 

---
# DebugLM: Learning Traceable Training Data Provenance for LLMs 

**Authors**: Wenjie Jacky Mo, Qin Liu, Xiaofei Wen, Wenxuan Zhou, Zhe Zhao, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.17884)  

**Abstract**: Large language models (LLMs) are trained through multi-stage pipelines over heterogeneous data sources, yet developers lack a principled way to pinpoint the specific data responsible for an observed behavior. This lack of observability reduces debugging to reactive patching and makes failures prone to recur under distribution shift or subsequent model updates. To address this limitation, we propose DebugLM, a framework that equips LLMs with built-in data provenance, enabling them to explicitly trace the origins of their behaviors to specific training data sources. Specifically, the model learns to associate its responses with unique provenance tags that indicate the responsible dataset, empowering developers to precisely identify where undesirable behaviors are learned. Building on this capability, DebugLM further supports targeted test-time remediation, enabling developers to selectively trigger targeted refusal for specified data sources without retraining or modifying model parameters. Experiments demonstrate that DebugLM provides accurate behavior tracing in multi-stage training pipelines and effective test-time remediation while preserving the general utility of the model. 

---
# KA2L: A Knowledge-Aware Active Learning Framework for LLMs 

**Authors**: Haoxuan Yin, Bojian Liu, Chen Tang, Yangfan Wang, Lian Yan, Jingchi Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.17566)  

**Abstract**: Fine-tuning large language models (LLMs) with high-quality knowledge has been shown to enhance their performance effectively. However, there is a paucity of research on the depth of domain-specific knowledge comprehension by LLMs and the application of targeted active learning to improve their expertise. To address this gap, we introduce the Knowledge-Aware Active Learning (KA2L) framework. This framework assesses LLMs' mastery of specific knowledge points to aid in constructing unanswerable or unknowable questions through latent space analysis. This active learning strategy enhances training efficiency by focusing on knowledge the model has yet to master, thereby minimizing redundancy in learning already acquired information. This study innovatively employs a knowledge distribution probing technique to examine the hidden states of specific Transformer layers and identify the distribution of known and unknown knowledge within the LLM. Additionally, a hidden-state decoding method is proposed to generate numerous unknown questions in natural language from the latent knowledge space. In our experiments, we selected nine open-source LLMs to validate the effectiveness of the proposed framework. Results indicate that KA2L not only significantly reduces 50% annotation and computation costs across two open-domain and one vertical-domain dataset but also achieves better performance, offering valuable insights into active learning strategies for LLMs. The code is available at this https URL. 

---
# Do Language Models Encode Semantic Relations? Probing and Sparse Feature Analysis 

**Authors**: Andor Diera, Ansgar Scherp  

**Link**: [PDF](https://arxiv.org/pdf/2603.17624)  

**Abstract**: Understanding whether large language models (LLMs) capture structured meaning requires examining how they represent concept relationships. In this work, we study three models of increasing scale: Pythia-70M, GPT-2, and Llama 3.1 8B, focusing on four semantic relations: synonymy, antonymy, hypernymy, and hyponymy. We combine linear probing with mechanistic interpretability techniques, including sparse autoencoders (SAE) and activation patching, to identify where these relations are encoded and how specific features contribute to their representation. Our results reveal a directional asymmetry in hierarchical relations: hypernymy is encoded redundantly and resists suppression, while hyponymy relies on compact features that are more easily disrupted by ablation. More broadly, relation signals are diffuse but exhibit stable profiles: they peak in the mid-layers and are stronger in post-residual/MLP pathways than in attention. Difficulty is consistent across models (antonymy easiest, synonymy hardest). Probe-level causality is capacity-dependent: on Llama 3.1, SAE-guided patching reliably shifts these signals, whereas on smaller models the shifts are weak or unstable. Our results clarify where and how reliably semantic relations are represented inside LLMs, and provide a reproducible framework for relating sparse features to probe-level causal evidence. 

---
# VeriAgent: A Tool-Integrated Multi-Agent System with Evolving Memory for PPA-Aware RTL Code Generation 

**Authors**: Yaoxiang Wang, Qi Shi, ShangZhan Li, Qingguo Hu, Xinyu Yin, Bo Guo, Xu Han, Maosong Sun, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2603.17613)  

**Abstract**: LLMs have recently demonstrated strong capabilities in automatic RTL code generation, achieving high syntactic and functional correctness. However, most methods focus on functional correctness while overlooking critical physical design objectives, including Power, Performance, and Area. In this work, we propose a PPA-aware, tool-integrated multi-agent framework for high-quality verilog code generation. Our framework explicitly incorporates EDA tools into a closed-loop workflow composed of a \textit{Programmer Agent}, a \textit{Correctness Agent}, and a \textit{PPA Agent}, enabling joint optimization of functional correctness and physical metrics. To support continuous improvement without model retraining, we introduce an \textit{Evolved Memory Mechanism} that externalizes optimization experience into structured memory nodes. A dedicated memory manager dynamically maintains the memory pool and allows the system to refine strategies based on historical execution trajectories. Extensive experiments demonstrate that our approach achieves strong functional correctness while delivering significant improvements in PPA metrics. By integrating tool-driven feedback with structured and evolvable memory, our framework transforms RTL generation from one-shot reasoning into a continual, feedback-driven optimization process, providing a scalable pathway for deploying LLMs in real-world hardware design flows. 

---
# Inducing Epistemological Humility in Large Language Models: A Targeted SFT Approach to Reducing Hallucination 

**Authors**: Cem Uluoglakci, Tugba Taskaya Temizel  

**Link**: [PDF](https://arxiv.org/pdf/2603.17504)  

**Abstract**: Large language models (LLMs) often hallucinate, producing fluent but false information, partly because supervised fine-tuning (SFT) implicitly rewards always responding. We introduce $\textit{HypoTermInstruct}$, an SFT dataset (31,487 responses for 11,151 questions) designed to teach models epistemological humility-the ability to recognize the limits of their own knowledge and admit uncertainty. This is achieved through questions about non-existent "hypothetical" terms. We also release $\textit{HypoTermQA-Enhanced}$, a benchmark for hallucination tendency strengthened through multiple validations. We conducted 800 controlled LoRA SFT runs across $\textit{Llama3.1-8B}$ and $\textit{Gemma3-4B}$ (base and instruct), testing 100 fine-tuning configurations with paired controls. Our results demonstrate that replacing generic instruction data with $\textit{HypoTermInstruct}$ significantly improves the HypoTerm Score (median increases of 0.19% to 25.91%) and FactScore (+0.39% to +0.86%), while maintaining stable performance on MMLU (minimal decreases of 0.26% to 0.35%). Our work demonstrates that targeted, high-quality SFT data teaching meta-cognitive skills can effectively reduce hallucination without preference/RL pipelines, providing mechanistic insights and a practical path toward more reliable AI systems. 

---
# Zipper-LoRA: Dynamic Parameter Decoupling for Speech-LLM based Multilingual Speech Recognition 

**Authors**: Yuxiang Mei, Delai Qiu, Shengping Liu, Jiaen Liang, Yanhua Long  

**Link**: [PDF](https://arxiv.org/pdf/2603.17558)  

**Abstract**: Speech Large Language Models (Speech-LLMs) have emerged as a powerful approach for automatic speech recognition (ASR) by aligning speech encoders with large language models. However, adapting these systems to multilingual settings with imbalanced data distributions remains challenging. In such scenarios, a stability-plasticity dilemma often arises: fully shared Parameter-Efficient Fine-Tuning (PEFT) can cause negative inter-lingual interference for under-represented languages, while fully language-specific tuning limits the cross-lingual beneficial knowledge transfer needed for low-resource tasks. To address this, we propose Zipper-LoRA, a novel rank-level decoupling framework with three variants (Static, Hard, and Soft) that dynamically synthesizes LoRA updates from shared and language-specific subspaces. By using a lightweight language-conditioned router, Zipper-LoRA dynamically controls the contribution of each subspace at the LoRA rank level, enabling fine-grained sharing where languages are compatible and strict decoupling when conflicts occur. To further stabilize optimization under imbalanced data, we propose a two-stage training strategy with an Initial-B warm start that significantly accelerates convergence. Experiments on a 12-language mixed-resource setting show that Zipper-LoRA consistently outperforms both fully shared and independent baselines, particularly in extremely low-resource scenarios. Moreover, we demonstrate that these gains are robust across both chunked and non-chunked encoder configurations, confirming the framework's reliability for practical, large-scale multilingual ASR. Our code and data will be available at this https URL for reproducibility. 

---
# Detecting the Machine: A Comprehensive Benchmark of AI-Generated Text Detectors Across Architectures, Domains, and Adversarial Conditions 

**Authors**: Madhav S. Baidya, S. S. Baidya, Chirag Chawla  

**Link**: [PDF](https://arxiv.org/pdf/2603.17522)  

**Abstract**: The rapid proliferation of large language models (LLMs) has created an urgent need for robust and generalizable detectors of machine-generated text. Existing benchmarks typically evaluate a single detector on a single dataset under ideal conditions, leaving open questions about cross-domain transfer, cross-LLM generalization, and adversarial robustness.
We present a comprehensive benchmark evaluating diverse detection approaches across two corpora: HC3 (23,363 human-ChatGPT pairs) and ELI5 (15,000 human-Mistral-7B pairs). Methods include classical classifiers, fine-tuned transformer encoders (BERT, RoBERTa, ELECTRA, DistilBERT, DeBERTa-v3), a CNN, an XGBoost stylometric model, perplexity-based detectors, and LLM-as-detector prompting.
Results show that transformer models achieve near-perfect in-distribution performance but degrade under domain shift. The XGBoost stylometric model matches performance while remaining interpretable. LLM-based detectors underperform and are affected by generator-detector identity bias. Perplexity-based methods exhibit polarity inversion, with modern LLM outputs showing lower perplexity than human text, but remain effective when corrected. No method generalizes robustly across domains and LLM sources. 

---
# Language on Demand, Knowledge at Core: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality 

**Authors**: Mengyu Bu, Yang Feng  

**Link**: [PDF](https://arxiv.org/pdf/2603.17512)  

**Abstract**: Large language models (LLMs) exhibit strong general intelligence, yet their multilingual performance remains highly imbalanced. Although LLMs encode substantial cross-lingual knowledge in a unified semantic space, they often struggle to reliably interface this knowledge with low-resource or unseen languages. Fortunately, pretrained encoder-decoder translation models already possess balanced multilingual capability, suggesting a natural complement to LLMs. In this work, we propose XBridge, a compositional encoder-LLM-decoder architecture that offloads multilingual understanding and generation to external pretrained translation models, while preserving the LLM as an English-centric core for general knowledge processing. To address the resulting representation misalignment across models, we introduce lightweight cross-model mapping layers and an optimal transport-based alignment objective, enabling fine-grained semantic consistency for multilingual generation. Experiments on four LLMs across multilingual understanding, reasoning, summarization, and generation indicate that XBridge outperforms strong baselines, especially on low-resource and previously unseen languages, without retraining the LLM. 

---
# Argument Reconstruction as Supervision for Critical Thinking in LLMs 

**Authors**: Hyun Ryu, Gyouk Chu, Gregor Betz, Eunho Yang, Carolyn Rose, Sean Welleck  

**Link**: [PDF](https://arxiv.org/pdf/2603.17432)  

**Abstract**: To think critically about arguments, human learners are trained to identify, reconstruct, and evaluate arguments. Argument reconstruction is especially important because it makes an argument's underlying inferences explicit. However, it remains unclear whether LLMs can similarly enhance their critical thinking ability by learning to reconstruct arguments. To address this question, we introduce a holistic framework with three contributions. We (1) propose an engine that automatically reconstructs arbitrary arguments (GAAR), (2) synthesize a new high-quality argument reconstruction dataset (Arguinas) using the GAAR engine, and (3) investigate whether learning argument reconstruction benefits downstream critical thinking tasks. Our experimental results show that, across seven critical thinking tasks, models trained to learn argument reconstruction outperform models that do not, with the largest performance gains observed when training on the proposed Arguinas dataset. The source code and dataset will be publicly available. 

---
# Humans and transformer LMs: Abstraction drives language learning 

**Authors**: Jasper Jian, Christopher D. Manning  

**Link**: [PDF](https://arxiv.org/pdf/2603.17475)  

**Abstract**: Categorization is a core component of human linguistic competence. We investigate how a transformer-based language model (LM) learns linguistic categories by comparing its behaviour over the course of training to behaviours which characterize abstract feature-based and concrete exemplar-based accounts of human language acquisition. We investigate how lexical semantic and syntactic categories emerge using novel divergence-based metrics that track learning trajectories using next-token distributions. In experiments with GPT-2 small, we find that (i) when a construction is learned, abstract class-level behaviour is evident at earlier steps than lexical item-specific behaviour, and (ii) that different linguistic behaviours emerge abruptly in sequence at different points in training, revealing that abstraction plays a key role in how LMs learn. This result informs the models of human language acquisition that LMs may serve as an existence proof for. 

---
# Grid Spatial Understanding: A Dataset for Textual Spatial Reasoning over Grids, Embodied Settings, and Coordinate Structures 

**Authors**: Risham Sidhu, Julia Hockenmaier  

**Link**: [PDF](https://arxiv.org/pdf/2603.17333)  

**Abstract**: We introduce GSU, a text-only grid dataset to evaluate the spatial reasoning capabilities of LLMs over 3 core tasks: navigation, object localization, and structure composition. By forgoing visual inputs, isolating spatial reasoning from perception, we show that while most models grasp basic grid concepts, they struggle with frames of reference relative to an embodied agent and identifying 3D shapes from coordinate lists. We also find that exposure to a visual modality does not provide a generalizable understanding of 3D space that VLMs are able to utilize for these tasks. Finally, we show that while the very latest frontier models can solve the provided tasks (though harder variants may still stump them), fully fine-tuning a small LM or LORA fine-tuning a small LLM show potential to match frontier model performance, suggesting an avenue for specialized embodied agents. 

---
# Exploiting the English Grammar Profile for L2 grammatical analysis with LLMs 

**Authors**: Stefano Bannò, Penny Karanasou, Kate Knill, Mark Gales  

**Link**: [PDF](https://arxiv.org/pdf/2603.17171)  

**Abstract**: Evaluating the grammatical competence of second language (L2) learners is essential both for providing targeted feedback and for assessing proficiency. To achieve this, we propose a novel framework leveraging the English Grammar Profile (EGP), a taxonomy of grammatical constructs mapped to the proficiency levels of the Common European Framework of Reference (CEFR), to detect learners' attempts at grammatical constructs and classify them as successful or unsuccessful. This detection can then be used to provide fine-grained feedback. Moreover, the grammatical constructs are used as predictors of proficiency assessment by using automatically detected attempts as predictors of holistic CEFR proficiency. For the selection of grammatical constructs derived from the EGP, rule-based and LLM-based classifiers are compared. We show that LLMs outperform rule-based methods on semantically and pragmatically nuanced constructs, while rule-based approaches remain competitive for constructs that rely purely on morphological or syntactic features and do not require semantic interpretation. For proficiency assessment, we evaluate both rule-based and hybrid pipelines and show that a hybrid approach combining a rule-based pre-filter with an LLM consistently yields the strongest performance. Since our framework operates on pairs of original learner sentences and their corrected counterparts, we also evaluate a fully automated pipeline using automatic grammatical error correction. This pipeline closely approaches the performance of semi-automated systems based on manual corrections, particularly for the detection of successful attempts at grammatical constructs. Overall, our framework emphasises learners' successful attempts in addition to unsuccessful ones, enabling positive, formative feedback and providing actionable insights into grammatical development. 

---
# Evaluating LLM-Simulated Conversations in Modeling Inconsistent and Uncollaborative Behaviors in Human Social Interaction 

**Authors**: Ryo Kamoi, Ameya Godbole, Longqi Yang, Rui Zhang, Mengting Wan, Pei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.17094)  

**Abstract**: Simulating human conversations using large language models (LLMs) has emerged as a scalable methodology for modeling human social interaction. However, simulating human conversations is challenging because they inherently involve inconsistent and uncollaborative behaviors, such as misunderstandings and interruptions. Analysis comparing inconsistent and uncollaborative behaviors in human- and LLM-generated conversations remains limited, although reproducing these behaviors is integral to simulating human-like and complex social interaction. In this work, we introduce CoCoEval, an evaluation framework that analyzes LLM-simulated conversations by detecting 10 types of inconsistent and uncollaborative behaviors at the turn level using an LLM-as-a-Judge. Using CoCoEval, we evaluate GPT-4.1, GPT-5.1, and Claude Opus 4 by comparing the frequencies of detected behaviors in conversations simulated by each model and in human conversations across academic, business, and governmental meetings, as well as debates. Our analysis shows that (1) under vanilla prompting, LLM-simulated conversations exhibit far fewer inconsistent and uncollaborative behaviors than human conversations; (2) prompt engineering does not provide reliable control over these behaviors, as our results show that different prompts lead to their under- or overproduction; and (3) supervised fine-tuning on human conversations can lead LLMs to overproduce a narrow set of behaviors, such as repetition. Our findings highlight the difficulty of simulating human conversations, raising concerns about the use of LLMs as a proxy for human social interaction. 

---
# Trust, Safety, and Accuracy: Assessing LLMs for Routine Maternity Advice 

**Authors**: V Sai Divya, A Bhanusree, Rimjhim, K Venkata Krishna Rao  

**Link**: [PDF](https://arxiv.org/pdf/2603.16872)  

**Abstract**: Access to reliable maternal healthcare information is a major challenge in rural India due to limited medical resources and infrastructure. With over 830 million internet users and nearly half of rural women online, digital tools offer new opportunities for health education. This study evaluates large language models (LLMs) like ChatGPT-4o, Perplexity AI, and GeminiAI to provide reliable and understandable pregnancy-related information. Seventeen pregnancy-focused questions were posed to each model and compared with responses from maternal health professionals. Evaluations used semantic similarity, noun overlap, and readability metrics to measure content quality. Results show Perplexity closely matched expert semantics, while ChatGPT-4o produced clearer, more understandable text with better medical terminology. As internet access grows in rural areas, LLMs could serve as scalable aids for maternal health education. The study highlights the need for AI tools that balance accuracy and clarity to improve healthcare communication in underserved regions. 

---
# Only relative ranks matter in weight-clustered large language models 

**Authors**: Borja Aizpurua, Sukhbinder Singh, Román Orús  

**Link**: [PDF](https://arxiv.org/pdf/2603.17917)  

**Abstract**: Large language models (LLMs) contain billions of parameters, yet many exact values are not essential. We show that what matters most is the relative rank of weights-whether one connection is stronger or weaker than another-rather than precise magnitudes. To reduce the number of unique weight values, we apply weight clustering to pretrained models, replacing every weight matrix with K shared values from K-means. For Llama 3.1-8B-Instruct and SmolLM2-135M, reducing each matrix to only 16-64 distinct values preserves strong accuracy without retraining, providing a simple, training-free method to compress LLMs on disk. Optionally fine-tuning only the cluster means (centroids) recovers 30-40 percent of the remaining accuracy gap at minimal cost. We then systematically randomize cluster means while keeping assignments fixed. Scrambling the relative ranks of the clusters degrades quality sharply-perplexity can increase by orders of magnitude-even when global statistics such as mean and variance are preserved. In contrast, rank-preserving randomizations cause almost no loss at mid and late layers. On the other hand, when many layers are perturbed simultaneously, progressive layer-by-layer replacement reveals that scale drift-not rank distortion-is the dominant collapse mechanism; however, an affine correction w' = aw + b with a > 0 (which preserves both rank order and overall weight distribution) can substantially delay this drift. This rank-based perspective offers a new lens on model compression and robustness. 

---
# Discovering Decoupled Functional Modules in Large Language Models 

**Authors**: Yanke Yu, Jin Li, Ying Sun, Ping Li, Zhefeng Wang, Yi Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.17823)  

**Abstract**: Understanding the internal functional organization of Large Language Models (LLMs) is crucial for improving their trustworthiness and performance. However, how LLMs organize different functions into modules remains highly unexplored. To bridge this gap, we formulate a functional module discovery problem and propose an Unsupervised LLM Cross-layer MOdule Discovery (ULCMOD) framework that simultaneously disentangles the large set of neurons in the entire LLM into modules while discovering the topics of input samples related to these modules. Our framework introduces a novel objective function and an efficient Iterative Decoupling (IterD) algorithm. Extensive experiments show that our method discovers high-quality, disentangled modules that capture more meaningful semantic information and achieve superior performance in various downstream tasks. Moreover, our qualitative analysis reveals that the discovered modules show semantic coherence, correspond to interpretable specializations, and a clear spatial and hierarchical organization within the LLM. Our work provides a novel tool for interpreting the functional modules of LLMs, filling a critical blank in LLM's interpretability research. 

---
# Tabular LLMs for Interpretable Few-Shot Alzheimer's Disease Prediction with Multimodal Biomedical Data 

**Authors**: Sophie Kearney, Shu Yang, Zixuan Wen, Weimin Lyu, Bojian Hou, Duy Duong-Tran, Tianlong Chen, Jason H. Moore, Marylyn D. Ritchie, Chao Chen, Li Shen  

**Link**: [PDF](https://arxiv.org/pdf/2603.17191)  

**Abstract**: Accurate diagnosis of Alzheimer's disease (AD) requires handling tabular biomarker data, yet such data are often small and incomplete, where deep learning models frequently fail to outperform classical methods. Pretrained large language models (LLMs) offer few-shot generalization, structured reasoning, and interpretable outputs, providing a powerful paradigm shift for clinical prediction. We propose TAP-GPT Tabular Alzheimer's Prediction GPT, a domain-adapted tabular LLM framework built on TableGPT2 and fine-tuned for few-shot AD classification using tabular prompts rather than plain texts. We evaluate TAP-GPT across four ADNI-derived datasets, including QT-PAD biomarkers and region-level structural MRI, amyloid PET, and tau PET for binary AD classification. Across multimodal and unimodal settings, TAP-GPT improves upon its backbone models and outperforms traditional machine learning baselines in the few-shot setting while remaining competitive with state-of-the-art general-purpose LLMs. We show that feature selection mitigates degradation in high-dimensional inputs and that TAP-GPT maintains stable performance under simulated and real-world missingness without imputation. Additionally, TAP-GPT produces structured, modality-aware reasoning aligned with established AD biology and shows greater stability under self-reflection, supporting its use in iterative multi-agent systems. To our knowledge, this is the first systematic application of a tabular-specialized LLM to multimodal biomarker-based AD prediction, demonstrating that such pretrained models can effectively address structured clinical prediction tasks and laying the foundation for tabular LLM-driven multi-agent clinical decision-support systems. The source code is publicly available on GitHub: this https URL. 

---
# LED: A Benchmark for Evaluating Layout Error Detection in Document Analysis 

**Authors**: Inbum Heo, Taewook Hwang, Jeesu Jung, Sangkeun Jung  

**Link**: [PDF](https://arxiv.org/pdf/2603.17265)  

**Abstract**: Recent advances in Large Language Models (LLMs) and Large Multimodal Models (LMMs) have improved Document Layout Analysis (DLA), yet structural errors such as region merging, splitting, and omission remain persistent. Conventional overlap-based metrics (e.g., IoU, mAP) fail to capture such logical inconsistencies. To overcome this limitation, we propose Layout Error Detection (LED), a benchmark that evaluates structural reasoning in DLA predictions beyond surface-level accuracy. LED defines eight standardized error types (Missing, Hallucination, Size Error, Split, Merge, Overlap, Duplicate, and Misclassification) and provides quantitative rules and injection algorithms for realistic error simulation. Using these definitions, we construct LED-Dataset and design three evaluation tasks: document-level error detection, document-level error-type classification, and element-level error-type classification. Experiments with state-of-the-art multimodal models show that LED enables fine-grained and interpretable assessment of structural understanding, revealing clear weaknesses across modalities and architectures. Overall, LED establishes a unified and explainable benchmark for diagnosing the structural robustness and reasoning capability of document understanding models. 

---
# EEG-Based Brain-LLM Interface for Human Preference Aligned Generation 

**Authors**: Junzi Zhang, Jianing Shen, Weijie Tu, Yi Zhang, Hailin Zhang, Tom Gedeon, Bin Jiang, Yue Yao  

**Link**: [PDF](https://arxiv.org/pdf/2603.16897)  

**Abstract**: Large language models (LLMs) are becoming an increasingly important component of human--computer interaction, enabling users to coordinate a wide range of intelligent agents through natural language. While language-based interfaces are powerful and flexible, they implicitly assume that users can reliably produce explicit linguistic input, an assumption that may not hold for users with speech or motor impairments, e.g., Amyotrophic Lateral Sclerosis (ALS). In this work, we investigate whether neural signals can be used as an alternative input to LLMs, particularly to support those socially marginalized or underserved users. We build a simple brain-LLM interface, which uses EEG signals to guide image generation models at test time. Specifically, we first train a classifier to estimate user satisfaction from EEG signals. Its predictions are then incorporated into a test-time scaling (TTS) framework that dynamically adapts model inference using neural feedback collected during user evaluation. The experiments show that EEG can predict user satisfaction, suggesting that neural activity carries information on real-time preference inference. These findings provide a first step toward integrating neural feedback into adaptive language-model inference, and hopefully open up new possibilities for future research on adaptive LLM interaction. 

---
