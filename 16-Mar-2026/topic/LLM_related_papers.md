# Semantic Invariance in Agentic AI 

**Authors**: I. de Zarzà, J. de Curtò, Jordi Cabot, Pietro Manzoni, Carlos T. Calafate  

**Link**: [PDF](https://arxiv.org/pdf/2603.13173)  

**Abstract**: Large Language Models (LLMs) increasingly serve as autonomous reasoning agents in decision support, scientific problem-solving, and multi-agent coordination systems. However, deploying LLM agents in consequential applications requires assurance that their reasoning remains stable under semantically equivalent input variations, a property we term semantic this http URL benchmark evaluations, which assess accuracy on fixed, canonical problem formulations, fail to capture this critical reliability dimension. To address this shortcoming, in this paper we present a metamorphic testing framework for systematically assessing the robustness of LLM reasoning agents, applying eight semantic-preserving transformations (identity, paraphrase, fact reordering, expansion, contraction, academic context, business context, and contrastive formulation) across seven foundation models spanning four distinct architectural families: Hermes (70B, 405B), Qwen3 (30B-A3B, 235B-A22B), DeepSeek-R1, and gpt-oss (20B, 120B). Our evaluation encompasses 19 multi-step reasoning problems across eight scientific domains. The results reveal that model scale does not predict robustness: the smaller Qwen3-30B-A3B achieves the highest stability (79.6% invariant responses, semantic similarity 0.91), while larger models exhibit greater fragility. 

---
# Developing and evaluating a chatbot to support maternal health care 

**Authors**: Smriti Jha, Vidhi Jain, Jianyu Xu, Grace Liu, Sowmya Ramesh, Jitender Nagpal, Gretchen Chapman, Benjamin Bellows, Siddhartha Goyal, Aarti Singh, Bryan Wilder  

**Link**: [PDF](https://arxiv.org/pdf/2603.13168)  

**Abstract**: The ability to provide trustworthy maternal health information using phone-based chatbots can have a significant impact, particularly in low-resource settings where users have low health literacy and limited access to care. However, deploying such systems is technically challenging: user queries are short, underspecified, and code-mixed across languages, answers require regional context-specific grounding, and partial or missing symptom context makes safe routing decisions difficult.
We present a chatbot for maternal health in India developed through a partnership between academic researchers, a health tech company, a public health nonprofit, and a hospital. The system combines (1) stage-aware triage, routing high-risk queries to expert templates, (2) hybrid retrieval over curated maternal/newborn guidelines, and (3) evidence-conditioned generation from an LLM. Our core contribution is an evaluation workflow for high-stakes deployment under limited expert supervision. Targeting both component-level and end-to-end testing, we introduce: (i) a labeled triage benchmark (N=150) achieving 86.7% emergency recall, explicitly reporting the missed-emergency vs. over-escalation trade-off; (ii) a synthetic multi-evidence retrieval benchmark (N=100) with chunk-level evidence labels; (iii) LLM-as-judge comparison on real queries (N=781) using clinician-codesigned criteria; and (iv) expert validation. Our findings show that trustworthy medical assistants in multilingual, noisy settings require defense-in-depth design paired with multi-method evaluation, rather than any single model and evaluation method choice. 

---
# ToolTree: Efficient LLM Agent Tool Planning via Dual-Feedback Monte Carlo Tree Search and Bidirectional Pruning 

**Authors**: Shuo Yang, Soyeon Caren Han, Yihao Ding, Shuhe Wang, Eduard Hoy  

**Link**: [PDF](https://arxiv.org/pdf/2603.12740)  

**Abstract**: Large Language Model (LLM) agents are increasingly applied to complex, multi-step tasks that require interaction with diverse external tools across various domains. However, current LLM agent tool planning methods typically rely on greedy, reactive tool selection strategies that lack foresight and fail to account for inter-tool dependencies. In this paper, we present ToolTree, a novel Monte Carlo tree search-inspired planning paradigm for tool planning. ToolTree explores possible tool usage trajectories using a dual-stage LLM evaluation and bidirectional pruning mechanism that enables the agent to make informed, adaptive decisions over extended tool-use sequences while pruning less promising branches before and after the tool execution. Empirical evaluations across both open-set and closed-set tool planning tasks on 4 benchmarks demonstrate that ToolTree consistently improves performance while keeping the highest efficiency, achieving an average gain of around 10\% compared to the state-of-the-art planning paradigm. 

---
# Context-Enriched Natural Language Descriptions of Vessel Trajectories 

**Authors**: Kostas Patroumpas, Alexandros Troupiotis-Kapeliaris, Giannis Spiliopoulos, Panagiotis Betchavas, Dimitrios Skoutas, Dimitris Zissis, Nikos Bikakis  

**Link**: [PDF](https://arxiv.org/pdf/2603.12287)  

**Abstract**: We address the problem of transforming raw vessel trajectory data collected from AIS into structured and semantically enriched representations interpretable by humans and directly usable by machine reasoning systems. We propose a context-aware trajectory abstraction framework that segments noisy AIS sequences into distinct trips each consisting of clean, mobility-annotated episodes. Each episode is further enriched with multi-source contextual information, such as nearby geographic entities, offshore navigation features, and weather conditions. Crucially, such representations can support generation of controlled natural language descriptions using LLMs. We empirically examine the quality of such descriptions generated using several LLMs over AIS data along with open contextual features. By increasing semantic density and reducing spatiotemporal complexity, this abstraction can facilitate downstream analytics and enable integration with LLMs for higher-level maritime reasoning tasks. 

---
# LLM Constitutional Multi-Agent Governance 

**Authors**: J. de Curtò, I. de Zarzà  

**Link**: [PDF](https://arxiv.org/pdf/2603.13189)  

**Abstract**: Large Language Models (LLMs) can generate persuasive influence strategies that shift cooperative behavior in multi-agent populations, but a critical question remains: does the resulting cooperation reflect genuine prosocial alignment, or does it mask erosion of agent autonomy, epistemic integrity, and distributional fairness? We introduce Constitutional Multi-Agent Governance (CMAG), a two-stage framework that interposes between an LLM policy compiler and a networked agent population, combining hard constraint filtering with soft penalized-utility optimization that balances cooperation potential against manipulation risk and autonomy pressure. We propose the Ethical Cooperation Score (ECS), a multiplicative composite of cooperation, autonomy, integrity, and fairness that penalizes cooperation achieved through manipulative means. In experiments on scale-free networks of 80 agents under adversarial conditions (70% violating candidates), we benchmark three regimes: full CMAG, naive filtering, and unconstrained optimization. While unconstrained optimization achieves the highest raw cooperation (0.873), it yields the lowest ECS (0.645) due to severe autonomy erosion (0.867) and fairness degradation (0.888). CMAG attains an ECS of 0.741, a 14.9% improvement, while preserving autonomy at 0.985 and integrity at 0.995, with only modest cooperation reduction to 0.770. The naive ablation (ECS = 0.733) confirms that hard constraints alone are insufficient. Pareto analysis shows CMAG dominates the cooperation-autonomy trade-off space, and governance reduces hub-periphery exposure disparities by over 60%. These findings establish that cooperation is not inherently desirable without governance: constitutional constraints are necessary to ensure that LLM-mediated influence produces ethically stable outcomes rather than manipulative equilibria. 

---
# Efficient Reasoning with Balanced Thinking 

**Authors**: Yulin Li, Tengyao Tu, Li Ding, Junjie Wang, Huiling Zhen, Yixin Chen, Yong Li, Zhuotao Tian  

**Link**: [PDF](https://arxiv.org/pdf/2603.12372)  

**Abstract**: Large Reasoning Models (LRMs) have shown remarkable reasoning capabilities, yet they often suffer from overthinking, expending redundant computational steps on simple problems, or underthinking, failing to explore sufficient reasoning paths despite inherent capabilities. These issues lead to inefficiencies and potential inaccuracies, limiting practical deployment in resource-constrained settings. Existing methods to mitigate overthinking, such as suppressing reflective keywords or adjusting reasoning length, may inadvertently induce underthinking, compromising accuracy. Therefore, we propose ReBalance, a training-free framework that achieves efficient reasoning with balanced thinking. ReBalance leverages confidence as a continuous indicator of reasoning dynamics, identifying overthinking through high confidence variance and underthinking via consistent overconfidence. By aggregating hidden states from a small-scale dataset into reasoning mode prototypes, we compute a steering vector to guide LRMs' reasoning trajectories. A dynamic control function modulates this vector's strength and direction based on real-time confidence, pruning redundancy during overthinking, and promoting exploration during underthinking. Extensive experiments conducted on four models ranging from 0.5B to 32B, and across nine benchmarks in math reasoning, general question answering, and coding tasks demonstrate that ReBalance effectively reduces output redundancy while improving accuracy, offering a general, training-free, and plug-and-play strategy for efficient and robust LRM deployment. Code is available at this https URL . 

---
# ESG-Bench: Benchmarking Long-Context ESG Reports for Hallucination Mitigation 

**Authors**: Siqi Sun, Ben Peng Wu, Mali Jin, Peizhen Bai, Hanpei Zhang, Xingyi Song  

**Link**: [PDF](https://arxiv.org/pdf/2603.13154)  

**Abstract**: As corporate responsibility increasingly incorporates environmental, social, and governance (ESG) criteria, ESG reporting is becoming a legal requirement in many regions and a key channel for documenting sustainability practices and assessing firms' long-term and ethical performance. However, the length and complexity of ESG disclosures make them difficult to interpret and automate the analysis reliably. To support scalable and trustworthy analysis, this paper introduces ESG-Bench, a benchmark dataset for ESG report understanding and hallucination mitigation in large language models (LLMs). ESG-Bench contains human-annotated question-answer (QA) pairs grounded in real-world ESG report contexts, with fine-grained labels indicating whether model outputs are factually supported or hallucinated. Framing ESG report analysis as a QA task with verifiability constraints enables systematic evaluation of LLMs' ability to extract and reason over ESG content and provides a new use case: mitigating hallucinations in socially sensitive, compliance-critical settings. We design task-specific Chain-of-Thought (CoT) prompting strategies and fine-tune multiple state-of-the-art LLMs on ESG-Bench using CoT-annotated rationales. Our experiments show that these CoT-based methods substantially outperform standard prompting and direct fine-tuning in reducing hallucinations, and that the gains transfer to existing QA benchmarks beyond the ESG domain. 

---
# Developing the PsyCogMetrics AI Lab to Evaluate Large Language Models and Advance Cognitive Science -- A Three-Cycle Action Design Science Study 

**Authors**: Zhiye Jin, Yibai Li, K. D. Joshi, Xuefei, Deng, Xiaobing  

**Link**: [PDF](https://arxiv.org/pdf/2603.13126)  

**Abstract**: This study presents the development of the PsyCogMetrics AI Lab (this http URL), an integrated, cloud-based platform that operationalizes psychometric and cognitive-science methodologies for Large Language Model (LLM) evaluation. Framed as a three-cycle Action Design Science study, the Relevance Cycle identifies key limitations in current evaluation methods and unfulfilled stakeholder needs. The Rigor Cycle draws on kernel theories such as Popperian falsifiability, Classical Test Theory, and Cognitive Load Theory to derive deductive design objectives. The Design Cycle operationalizes these objectives through nested Build-Intervene-Evaluate loops. The study contributes a novel IT artifact, a validated design for LLM evaluation, benefiting research at the intersection of AI, psychology, cognitive science, and the social and behavioral sciences. 

---
# Human-in-the-Loop LLM Grading for Handwritten Mathematics Assessments 

**Authors**: Arne Vanhoyweghen, Vincent Holst, Melika Mobini, Lukas Van de Voorde, Tibo Vanleke, Bert Verbruggen, Brecht Verbeken, Andres Algaba, Sam Verboven, Marie-Anne Guerry, Filip Van Droogenbroeck, Vincent Ginis  

**Link**: [PDF](https://arxiv.org/pdf/2603.13083)  

**Abstract**: Providing timely and individualised feedback on handwritten student work is highly beneficial for learning but difficult to achieve at scale. This challenge has become more pressing as generative AI undermines the reliability of take-home assessments, shifting emphasis toward supervised, in-class evaluation. We present a scalable, end-to-end workflow for LLM-assisted grading of short, pen-and-paper assessments. The workflow spans (1) constructing solution keys, (2) developing detailed rubric-style grading keys used to guide the LLM, and (3) a grading procedure that combines automated scanning and anonymisation, multi-pass LLM scoring, automated consistency checks, and mandatory human verification. We deploy the system in two undergraduate mathematics courses using six low-stakes in-class tests. Empirically, LLM assistance reduces grading time by approximately 23% while achieving agreement comparable to, and in several cases tighter than, fully manual grading. Occasional model errors occur but are effectively contained by the hybrid design. Overall, our results show that carefully embedded human-in-the-loop LLM grading can substantially reduce workload while maintaining fairness and accuracy. 

---
# Delta1 with LLM: symbolic and neural integration for credible and explainable reasoning 

**Authors**: Yang Xu, Jun Liu, Shuwei Chen, Chris Nugent, Hailing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2603.12953)  

**Abstract**: Neuro-symbolic reasoning increasingly demands frameworks that unite the formal rigor of logic with the interpretability of large language models (LLMs). We introduce an end to end explainability by construction pipeline integrating the Automated Theorem Generator Delta1 based on the full triangular standard contradiction (FTSC) with LLMs. Delta1 deterministically constructs minimal unsatisfiable clause sets and complete theorems in polynomial time, ensuring both soundness and minimality by construction. The LLM layer verbalizes each theorem and proof trace into coherent natural language explanations and actionable insights. Empirical studies across health care, compliance, and regulatory domains show that Delta1 and LLM enables interpretable, auditable, and domain aligned reasoning. This work advances the convergence of logic, language, and learning, positioning constructive theorem generation as a principled foundation for neuro-symbolic explainable AI. 

---
# Human-Centered Evaluation of an LLM-Based Process Modeling Copilot: A Mixed-Methods Study with Domain Experts 

**Authors**: Chantale Lauer, Peter Pfeiffer, Nijat Mehdiyev  

**Link**: [PDF](https://arxiv.org/pdf/2603.12895)  

**Abstract**: Integrating Large Language Models (LLMs) into business process management tools promises to democratize Business Process Model and Notation (BPMN) modeling for non-experts. While automated frameworks assess syntactic and semantic quality, they miss human factors like trust, usability, and professional alignment. We conducted a mixed-methods evaluation of our proposed solution, an LLM-powered BPMN copilot, with five process modeling experts using focus groups and standardized questionnaires. Our findings reveal a critical tension between acceptable perceived usability (mean CUQ score: 67.2/100) and notably lower trust (mean score: 48.8\%), with reliability rated as the most critical concern (M=1.8/5). Furthermore, we identified output-quality issues, prompting difficulties, and a need for the LLM to ask more in-depth clarifying questions about the process. We envision five use cases ranging from domain-expert support to enterprise quality assurance. We demonstrate the necessity of human-centered evaluation complementing automated benchmarking for LLM modeling agents. 

---
# Cost-Efficient Multimodal LLM Inference via Cross-Tier GPU Heterogeneity 

**Authors**: Donglin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2603.12707)  

**Abstract**: Multimodal large language model (MLLM) inference splits into two phases with opposing hardware demands: vision encoding is compute-bound, while language generation is memory-bandwidth-bound. We show that under standard transformer KV caching, the modality boundary (between vision encoder and language model) minimizes cross-device transfer among all partition points that preserve standard stage-based execution. Partitioning here reduces transfer complexity from $O(L * s_ctx)$ bytes (GB-scale KV caches under stage-level disaggregation) to $O(N_v * d)$ bytes (MB-scale embeddings), an O(L) reduction where L is the transformer depth. The result holds across attention mechanisms (MHA/GQA), dynamic vision resolutions, and model scales, and the advantage grows as models deepen. A direct implication is that existing stage-level disaggregation systems are constrained to high-bandwidth interconnects (e.g., NVLink), whereas modality-level disaggregation enables cross-tier heterogeneous serving over commodity PCIe. A closed-form cost model shows that heterogeneous deployment is cost-optimal under phase-separable workloads (predicts 31.4% savings; observed 40.6%). We build HeteroServe, a phase-aware runtime with modality-level partitioning and cross-tier scheduling, and evaluate it on LLaVA-1.5-7B and Qwen2.5-VL against vLLM v0.3.0. On identical 4xA100 hardware, engine optimizations raise throughput by up to 54%. Under a fixed budget, a heterogeneous cluster (\$38k) improves Tokens/\$ by 37% over a homogeneous baseline (\$64k) without degrading latency. 

---
# MetaKE: Meta-learning Aligned Knowledge Editing via Bi-level Optimization 

**Authors**: Shuxin Liu, Ou Wu  

**Link**: [PDF](https://arxiv.org/pdf/2603.12677)  

**Abstract**: Knowledge editing (KE) aims to precisely rectify specific knowledge in Large Language Models (LLMs) without disrupting general capabilities. State-of-the-art methods suffer from an open-loop control mismatch. We identify a critical "Semantic-Execution Disconnect": the semantic target is derived independently without feedback from the downstream's feasible region. This misalignment often causes valid semantic targets to fall within the prohibited space, resulting in gradient truncation and editing failure. To bridge this gap, we propose MetaKE (Meta-learning Aligned Knowledge Editing), a new framework that reframes KE as a bi-level optimization problem. Departing from static calculation, MetaKE treats the edit target as a learnable meta-parameter: the upper-level optimizer seeks a feasible target to maximize post-edit performance, while the lower-level solver executes the editing. To address the challenge of differentiating through complex solvers, we derive a Structural Gradient Proxy, which explicitly backpropagates editability constraints to the target learning phase. Theoretical analysis demonstrates that MetaKE automatically aligns the edit direction with the model's feasible manifold. Extensive experiments confirm that MetaKE significantly outperforms strong baselines, offering a new perspective on knowledge editing. 

---
# Experimental evidence of progressive ChatGPT models self-convergence 

**Authors**: Konstantinos F. Xylogiannopoulos, Petros Xanthopoulos, Panagiotis Karampelas, Georgios A. Bakamitsos  

**Link**: [PDF](https://arxiv.org/pdf/2603.12683)  

**Abstract**: Large Language Models (LLMs) that undergo recursive training on synthetically generated data are susceptible to model collapse, a phenomenon marked by the generation of meaningless output. Existing research has examined this issue from either theoretical or empirical perspectives, often focusing on a single model trained recursively on its own outputs. While prior studies have cautioned against the potential degradation of LLM output quality under such conditions, no longitudinal investigation has yet been conducted to assess this effect over time. In this study, we employ a text similarity metric to evaluate different ChatGPT models' capacity to generate diverse textual outputs. Our findings indicate a measurable decline of recent ChatGPT releases' ability to produce varied text, even when explicitly prompted to do so, by setting the temperature parameter to one. The observed reduction in output diversity may be attributed to the influence of the amounts of synthetic data incorporated within their training datasets as the result of internet infiltration by LLM generated data. The phenomenon is defined as model self-convergence because of the gradual increase of similarities of produced texts among different ChatGPT versions. 

---
# Continual Learning in Large Language Models: Methods, Challenges, and Opportunities 

**Authors**: Hongyang Chen, Zhongwu Sun, Hongfei Ye, Kunchi Li, Xuemin Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.12658)  

**Abstract**: Continual learning (CL) has emerged as a pivotal paradigm to enable large language models (LLMs) to dynamically adapt to evolving knowledge and sequential tasks while mitigating catastrophic forgetting-a critical limitation of the static pre-training paradigm inherent to modern LLMs. This survey presents a comprehensive overview of CL methodologies tailored for LLMs, structured around three core training stages: continual pre-training, continual fine-tuning, and continual this http URL the canonical taxonomy of rehearsal-, regularization-, and architecture-based methods, we further subdivide each category by its distinct forgetting mitigation mechanisms and conduct a rigorous comparative analysis of the adaptability and critical improvements of traditional CL methods for LLMs. In doing so, we explicitly highlight core distinctions between LLM CL and traditional machine learning, particularly with respect to scale, parameter efficiency, and emergent capabilities. Our analysis covers essential evaluation metrics, including forgetting rates and knowledge transfer efficiency, along with emerging benchmarks for assessing CL performance. This survey reveals that while current methods demonstrate promising results in specific domains, fundamental challenges persist in achieving seamless knowledge integration across diverse tasks and temporal scales. This systematic review contributes to the growing body of knowledge on LLM adaptation, providing researchers and practitioners with a structured framework for understanding current achievements and future opportunities in lifelong learning for language models. 

---
# Towards unified brain-to-text decoding across speech production and perception 

**Authors**: Zhizhang Yuan, Yang Yang, Gaorui Zhang, Baowen Cheng, Zehan Wu, Yuhao Xu, Xiaoying Liu, Liang Chen, Ying Mao, Meng Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.12628)  

**Abstract**: Speech production and perception are the main ways humans communicate daily. Prior brain-to-text decoding studies have largely focused on a single modality and alphabetic languages. Here, we present a unified brain-to-sentence decoding framework for both speech production and perception in Mandarin Chinese. The framework exhibits strong generalization ability, enabling sentence-level decoding when trained only on single-character data and supporting characters and syllables unseen during training. In addition, it allows direct and controlled comparison of neural dynamics across modalities. Mandarin speech is decoded by first classifying syllable components in Hanyu Pinyin, namely initials and finals, from neural signals, followed by a post-trained large language model (LLM) that maps sequences of toneless Pinyin syllables to Chinese sentences. To enhance LLM decoding, we designed a three-stage post-training and two-stage inference framework based on a 7-billion-parameter LLM, achieving overall performance that exceeds larger commercial LLMs with hundreds of billions of parameters or more. In addition, several characteristics were observed in Mandarin speech production and perception: speech production involved neural responses across broader cortical regions than auditory perception; channels responsive to both modalities exhibited similar activity patterns, with speech perception showing a temporal delay relative to production; and decoding performance was broadly comparable across hemispheres. Our work not only establishes the feasibility of a unified decoding framework but also provides insights into the neural characteristics of Mandarin speech production and perception. These advances contribute to brain-to-text decoding in logosyllabic languages and pave the way toward neural language decoding systems supporting multiple modalities. 

---
# From Text to Forecasts: Bridging Modality Gap with Temporal Evolution Semantic Space 

**Authors**: Lehui Li, Yuyao Wang, Jisheng Yan, Wei Zhang, Jinliang Deng, Haoliang Sun, Zhongyi Han, Yongshun Gong  

**Link**: [PDF](https://arxiv.org/pdf/2603.12664)  

**Abstract**: Incorporating textual information into time-series forecasting holds promise for addressing event-driven non-stationarity; however, a fundamental modality gap hinders effective fusion: textual descriptions express temporal impacts implicitly and qualitatively, whereas forecasting models rely on explicit and quantitative signals. Through controlled semi-synthetic experiments, we show that existing methods over-attend to redundant tokens and struggle to reliably translate textual semantics into usable numerical cues. To bridge this gap, we propose TESS, which introduces a Temporal Evolution Semantic Space as an intermediate bottleneck between modalities. This space consists of interpretable, numerically grounded temporal primitives (mean shift, volatility, shape, and lag) extracted from text by an LLM via structured prompting and filtered through confidence-aware gating. Experiments on four real-world datasets demonstrate up to a 29 percent reduction in forecasting error compared to state-of-the-art unimodal and multimodal baselines. The code will be released after acceptance. 

---
# RetroReasoner: A Reasoning LLM for Strategic Retrosynthesis Prediction 

**Authors**: Hanbum Ko, Chanhui Lee, Ye Rin Kim, Rodrigo Hormazabal, Sehui Han, Sungbin Lim, Sungwoong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.12666)  

**Abstract**: Retrosynthesis prediction is a core task in organic synthesis that aims to predict reactants for a given product molecule. Traditionally, chemists select a plausible bond disconnection and derive corresponding reactants, which is time-consuming and requires substantial expertise. While recent advancements in molecular large language models (LLMs) have made progress, many methods either predict reactants without strategic reasoning or conduct only a generic product analysis, rather than reason explicitly about bond-disconnection strategies that logically lead to the choice of specific reactants. To overcome these limitations, we propose RetroReasoner, a retrosynthetic reasoning model that leverages chemists' strategic thinking. RetroReasoner is trained using both supervised fine-tuning (SFT) and reinforcement learning (RL). For SFT, we introduce SyntheticRetro, a framework that generates structured disconnection rationales alongside reactant predictions. In the case of RL, we apply a round-trip accuracy as reward, where predicted reactants are passed through a forward synthesis model, and predictions are rewarded when the forward-predicted product matches the original input product. Experimental results show that RetroReasoner not only outperforms prior baselines but also generates a broader range of feasible reactant proposals, particularly in handling more challenging reaction instances. 

---
# Spend Less, Reason Better: Budget-Aware Value Tree Search for LLM Agents 

**Authors**: Yushu Li, Wenlong Deng, Jiajin Li, Xiaoxiao Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.12634)  

**Abstract**: Test-time scaling has become a dominant paradigm for improving LLM agent reliability, yet current approaches treat compute as an abundant resource, allowing agents to exhaust token and tool budgets on redundant steps or dead-end trajectories. Existing budget-aware methods either require expensive fine-tuning or rely on coarse, trajectory-level heuristics that cannot intervene mid-execution. We propose the Budget-Aware Value Tree (BAVT), a training-free inference-time framework that models multi-hop reasoning as a dynamic search tree guided by step-level value estimation within a single LLM backbone. Another key innovation is a budget-conditioned node selection mechanism that uses the remaining resource ratio as a natural scaling exponent over node values, providing a principled, parameter-free transition from broad exploration to greedy exploitation as the budget depletes. To combat the well-known overconfidence of LLM self-evaluation, BAVT employs a residual value predictor that scores relative progress rather than absolute state quality, enabling reliable pruning of uninformative or redundant tool calls. We further provide a theoretical convergence guarantee, proving that BAVT reaches a terminal answer with probability at least $1-\epsilon$ under an explicit finite budget bound. Extensive evaluations on four multi-hop QA benchmarks across two model families demonstrate that BAVT consistently outperforms parallel sampling baselines. Most notably, BAVT under strict low-budget constraints surpasses baseline performance at $4\times$ the resource allocation, establishing that intelligent budget management fundamentally outperforms brute-force compute scaling. 

---
# LLM BiasScope: A Real-Time Bias Analysis Platform for Comparative LLM Evaluation 

**Authors**: Himel Ghosh, Nick Elias Werner  

**Link**: [PDF](https://arxiv.org/pdf/2603.12522)  

**Abstract**: As large language models (LLMs) are deployed widely, detecting and understanding bias in their outputs is critical. We present LLM BiasScope, a web application for side-by-side comparison of LLM outputs with real-time bias analysis. The system supports multiple providers (Google Gemini, DeepSeek, MiniMax, Mistral, Meituan, Meta Llama) and enables researchers and practitioners to compare models on the same prompts while analyzing bias patterns. LLM BiasScope uses a two-stage bias detection pipeline: sentence-level bias detection followed by bias type classification for biased sentences. The analysis runs automatically on both user prompts and model responses, providing statistics, visualizations, and detailed breakdowns of bias types. The interface displays two models side-by-side with synchronized streaming responses, per-model bias summaries, and a comparison view highlighting differences in bias distributions. The system is built on this http URL with React, integrates Hugging Face inference endpoints for bias detection, and uses the Vercel AI SDK for multi-provider LLM access. Features include real-time streaming, export to JSON/PDF, and interactive visualizations (bar charts, radar charts) for bias analysis. LLM BiasScope is available as an open-source web application, providing a practical tool for bias evaluation and comparative analysis of LLM behaviour. 

---
# When LLM Judge Scores Look Good but Best-of-N Decisions Fail 

**Authors**: Eddie Landesberg  

**Link**: [PDF](https://arxiv.org/pdf/2603.12520)  

**Abstract**: Large language models are often used as judges to score candidate responses, then validated with a single global metric such as correlation with reference labels. This can be misleading when the real deployment task is best-of-n selection within a prompt.
In a 5,000-prompt best-of-4 benchmark from Chatbot Arena, a judge with moderate global correlation (r = 0.47) captures only 21.0% of the improvement that perfect selection would achieve over random choice. The gap arises because global agreement is driven largely by prompt-level baseline effects, while selection depends on within-prompt ranking: within-prompt correlation is only r_within = 0.27, and coarse pointwise scoring creates ties in 67% of pairwise comparisons.
In a matched-pair best-of-2 audit, explicit pairwise judging recovers much of this lost signal, raising recovery from 21.1% to 61.2%. For judge-based selection, the relevant audit should report within-prompt signal, tie rates, and recovery/top-1 accuracy, not global agreement alone. 

---
# TERMINATOR: Learning Optimal Exit Points for Early Stopping in Chain-of-Thought Reasoning 

**Authors**: Alliot Nagle, Jakhongir Saydaliev, Dhia Garbaya, Michael Gastpar, Ashok Vardhan Makkuva, Hyeji Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.12529)  

**Abstract**: Large Reasoning Models (LRMs) achieve impressive performance on complex reasoning tasks via Chain-of-Thought (CoT) reasoning, which enables them to generate intermediate thinking tokens before arriving at the final answer. However, LRMs often suffer from significant overthinking, spending excessive compute time even after the answer is generated early on. Prior work has identified the existence of an optimal reasoning length such that truncating reasoning at this point significantly shortens CoT outputs with virtually no change in performance. However, determining optimal CoT lengths for practical datasets is highly non-trivial as they are fully task and model-dependent. In this paper, we precisely address this and design TERMINATOR, an early-exit strategy for LRMs at inference to mitigate overthinking. The central idea underpinning TERMINATOR is that the first arrival of an LRM's final answer is often predictable, and we leverage these first answer positions to create a novel dataset of optimal reasoning lengths to train TERMINATOR. Powered by this approach, TERMINATOR achieves significant reductions in CoT lengths of 14%-55% on average across four challenging practical datasets: MATH-500, AIME 2025, HumanEval, and GPQA, whilst outperforming current state-of-the-art methods. 

---
# Shattering the Shortcut: A Topology-Regularized Benchmark for Multi-hop Medical Reasoning in LLMs 

**Authors**: Xing Zi, Xinying Zhou, Jinghao Xiao, Catarina Moreira, Mukesh Prasad  

**Link**: [PDF](https://arxiv.org/pdf/2603.12458)  

**Abstract**: While Large Language Models (LLMs) achieve expert-level performance on standard medical benchmarks through single-hop factual recall, they severely struggle with the complex, multi-hop diagnostic reasoning required in real-world clinical settings. A primary obstacle is "shortcut learning", where models exploit highly connected, generic hub nodes (e.g., "inflammation") in knowledge graphs to bypass authentic micro-pathological cascades. To address this, we introduce ShatterMed-QA, a bilingual benchmark of 10,558 multi-hop clinical questions designed to rigorously evaluate deep diagnostic reasoning. Our framework constructs a topology-regularized medical Knowledge Graph using a novel $k$-Shattering algorithm, which physically prunes generic hubs to explicitly sever logical shortcuts. We synthesize the evaluation vignettes by applying implicit bridge entity masking and topology-driven hard negative sampling, forcing models to navigate biologically plausible distractors without relying on superficial elimination. Comprehensive evaluations of 21 LLMs reveal massive performance degradation on our multi-hop tasks, particularly among domain-specific models. Crucially, restoring the masked evidence via Retrieval-Augmented Generation (RAG) triggers near-universal performance recovery, validating ShatterMed-QA's structural fidelity and proving its efficacy in diagnosing the fundamental reasoning deficits of current medical AI. Explore the dataset, interactive examples, and full leaderboards at our project website: this https URL 

---
# SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs 

**Authors**: Mohamad Alansari, Naufal Suryanto, Divya Velayudhan, Sajid Javed, Naoufel Werghi, Muzammal Naseer  

**Link**: [PDF](https://arxiv.org/pdf/2603.12382)  

**Abstract**: Multimodal large language models (MLLMs) have advanced from image-level reasoning to pixel-level grounding, but extending these capabilities to videos remains challenging as models must achieve spatial precision and temporally consistent reference tracking. Existing video MLLMs often rely on a static segmentation token ([SEG]) for frame-wise grounding, which provides semantics but lacks temporal context, causing spatial drift, identity switches, and unstable initialization when objects move or reappear. We introduce SPARROW, a pixel-grounded video MLLM that unifies spatial accuracy and temporal stability through two key components: (i) Target-Specific Tracked Features (TSF), which inject temporally aligned referent cues during training, and (ii) a dual-prompt design that decodes box ([BOX]) and segmentation ([SEG]) tokens to fuse geometric priors with semantic grounding. SPARROW is supported by a curated referential video dataset of 30,646 videos and 45,231 Q&A pairs and operates end-to-end without external detectors via a class-agnostic SAM2-based proposer. Integrated into three recent open-source video MLLMs (UniPixel, GLUS, and VideoGLaMM), SPARROW delivers consistent gains across six benchmarks, improving up to +8.9 J&F on RVOS, +5 mIoU on visual grounding, and +5.4 CLAIR on GCG. These results demonstrate that SPARROW substantially improves referential stability, spatial precision, and temporal coherence in pixel-grounded video understanding. Project page: this https URL 

---
# Global Evolutionary Steering: Refining Activation Steering Control via Cross-Layer Consistency 

**Authors**: Xinyan Jiang, Wenjing Yu, Di Wang, Lijie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2603.12298)  

**Abstract**: Activation engineering enables precise control over Large Language Models (LLMs) without the computational cost of fine-tuning. However, existing methods deriving vectors from static activation differences are susceptible to high-dimensional noise and layer-wise semantic drift, often capturing spurious correlations rather than the target intent. To address this, we propose Global Evolutionary Refined Steering (GER-steer), a training-free framework that grounded in the geometric stability of the network's representation evolution. GER-steer exploits this global signal to rectify raw steering vectors, effectively decoupling robust semantic intent from orthogonal artifacts. Extensive evaluations confirm that GER-steer consistently outperforms baselines, delivering superior efficacy and generalization without layer-specific tuning, establishing a universal solution for reliable model alignment. 

---
# Detecting Miscitation on the Scholarly Web through LLM-Augmented Text-Rich Graph Learning 

**Authors**: Huidong Wu, Haojia Xiang, Jingtong Gao, Xiangyu Zhao, Dengsheng Wu, Jianping Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.12290)  

**Abstract**: Scholarly web is a vast network of knowledge connected by citations. However, this system is increasingly compromised by miscitation, where references do not support or even contradict the claims they are cited for. Current miscitation detection methods, which primarily rely on semantic similarity or network anomalies, struggle to capture the nuanced relationship between a citation's context and its place in the wider network. While large language models (LLMs) offer powerful capabilities in semantic reasoning for this task, their deployment is hindered by hallucination risks and high computational costs. In this work, we introduce LLM-Augmented Graph Learning-based Miscitation Detector (LAGMiD), a novel framework that leverages LLMs for deep semantic reasoning over citation graphs and distills this knowledge into graph neural networks (GNNs) for efficient and scalable miscitation detection. Specifically, LAGMiD introduces an evidence-chain reasoning mechanism, which uses chain-of-thought prompting, to perform multi-hop citation tracing and assess semantic fidelity. To reduce LLM inference costs, we design a knowledge distillation method aligning GNN embeddings with intermediate LLM reasoning states. A collaborative learning strategy further routes complex cases to the LLM while optimizing the GNN for structure-based generalization. Experiments on three real-world benchmarks show that LAGMiD achieves state-of-the-art miscitation detection with significantly reduced inference cost. 

---
# Task-Specific Knowledge Distillation via Intermediate Probes 

**Authors**: Ryan Brown, Chris Russell  

**Link**: [PDF](https://arxiv.org/pdf/2603.12270)  

**Abstract**: Knowledge distillation from large language models (LLMs) assumes that the teacher's output distribution is a high-quality training signal. On reasoning tasks, this assumption is frequently violated. A model's intermediate representations may encode the correct answer, yet this information is lost or distorted through the vocabulary projection, where prompt formatting and answer-token choices creates brittle, noisy outputs.
We introduce \method{}, a distillation framework that bypasses this bottleneck by training lightweight probes on frozen teacher hidden states and using the probe's predictions, rather than output logits, as supervision for student training. This simple change yields consistent improvements across four reasoning benchmarks (AQuA-RAT, ARC Easy/Challenge, and MMLU), with gains most pronounced under limited data.
Probes trained on intermediate representations provide cleaner labels than the teacher's own outputs, effectively denoising the distillation signal. \method{} requires no architectural changes to student or teacher, is architecture-agnostic, and adds minimal compute since probe training is cheap and teacher representations can be cached. By exploiting internal representations, \method{} enables practitioners to extract more value from large teacher models without additional training data or architectural complexity. 

---
# Aligning Language Models from User Interactions 

**Authors**: Thomas Kleine Buening, Jonas Hübotter, Barna Pásztor, Idan Shenfeld, Giorgia Ramponi, Andreas Krause  

**Link**: [PDF](https://arxiv.org/pdf/2603.12273)  

**Abstract**: Multi-turn user interactions are among the most abundant data produced by language models, yet we lack effective methods to learn from them. While typically discarded, these interactions often contain useful information: follow-up user messages may indicate that a response was incorrect, failed to follow an instruction, or did not align with the user's preferences. Importantly, language models are already able to make use of this information in context. After observing a user's follow-up, the same model is often able to revise its behavior. We leverage this ability to propose a principled and scalable method for learning directly from user interactions through self-distillation. By conditioning the model on the user's follow-up message and comparing the resulting token distribution with the original policy, we obtain a target for updating the policy that captures how the model's behavior changes in hindsight. We then distill this hindsight distribution back into the current policy. Remarkably, we show that training on real-world user conversations from WildChat improves language models across standard alignment and instruction-following benchmarks, without regressing other capabilities. The same mechanism enables personalization, allowing models to continually adapt to individual users through interaction without explicit feedback. Our results demonstrate that raw user interactions that arise naturally during deployment enable alignment, personalization, and continual adaptation. 

---
# Diagnosing Retrieval Bias Under Multiple In-Context Knowledge Updates in Large Language Models 

**Authors**: Boyu Qiao, Sean Guo, Xian Yang, Kun Li, Wei Zhou, Songlin Hu, Yunya Song  

**Link**: [PDF](https://arxiv.org/pdf/2603.12271)  

**Abstract**: LLMs are widely used in knowledge-intensive tasks where the same fact may be revised multiple times within context. Unlike prior work focusing on one-shot updates or single conflicts, multi-update scenarios contain multiple historically valid versions that compete at retrieval, yet remain underexplored. This challenge resembles the AB-AC interference paradigm in cognitive psychology: when the same cue A is successively associated with B and C, the old and new associations compete during retrieval, leading to bias. Inspired by this, we introduce a Dynamic Knowledge Instance (DKI) evaluation framework, modeling multi-updates of the same fact as a cue paired with a sequence of updated values, and assess models via endpoint probing of the earliest (initial) and latest (current) states. Across diverse LLMs, we observe that retrieval bias intensifies as updates increase, earliest-state accuracy stays high while latest-state accuracy drops substantially. Diagnostic analyses of attention, hidden-state similarity, and output logits further reveal that these signals become flatter and weakly discriminative on errors, providing little stable basis for identifying the latest update. Finally, cognitively inspired heuristic intervention strategies yield only modest gains and do not eliminate the bias. Our results reveal a persistent challenge in tracking and following knowledge updates in long contexts. 

---
# DS$^2$-Instruct: Domain-Specific Data Synthesis for Large Language Models Instruction Tuning 

**Authors**: Ruiyao Xu, Noelle I. Samia, Han Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.12932)  

**Abstract**: Adapting Large Language Models (LLMs) to specialized domains requires high-quality instruction tuning datasets, which are expensive to create through human annotation. Existing data synthesis methods focus on general-purpose tasks and fail to capture domain-specific terminology and reasoning patterns. To address this, we introduce DS$^2$-Instruct, a zero-shot framework that generates domain-specific instruction datasets without human supervision. Our approach first generates task-informed keywords to ensure comprehensive domain coverage. It then creates diverse instructions by pairing these keywords with different cognitive levels from Bloom's Taxonomy. Finally, it uses self-consistency validation to ensure data quality. We apply this framework to generate datasets across seven challenging domains, such as mathematics, finance, and logical reasoning. Comprehensive evaluation demonstrates that models fine-tuned on our generated data achieve substantial improvements over existing data generation methods. 

---
# Neuron-Aware Data Selection In Instruction Tuning For Large Language Models 

**Authors**: Xin Chen, Junchao Wu, Shu Yang, Runzhe Zhan, Zeyu Wu, Min Yang, Shujian Huang, Lidia S. Chao, Derek F. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2603.13201)  

**Abstract**: Instruction Tuning (IT) has been proven to be an effective approach to unlock the powerful capabilities of large language models (LLMs). Recent studies indicate that excessive IT data can degrade LLMs performance, while carefully selecting a small subset of high-quality IT data can significantly enhance their capabilities. Therefore, identifying the most efficient subset data from the IT dataset to effectively develop either specific or general abilities in LLMs has become a critical challenge. To address this, we propose a novel and efficient framework called NAIT. NAIT evaluates the impact of IT data on LLMs performance by analyzing the similarity of neuron activation patterns between the IT dataset and the target domain capability. Specifically, NAIT captures neuron activation patterns from in-domain datasets of target domain capabilities to construct reusable and transferable neuron activation features. It then evaluates and selects optimal samples based on the similarity between candidate samples and the expected activation features of the target capabilities. Experimental results show that training on the 10\% Alpaca-GPT4 IT data subset selected by NAIT consistently outperforms methods that rely on external advanced models or uncertainty-based features across various tasks. Our findings also reveal the transferability of neuron activation features across different capabilities of LLMs. In particular, IT data with more logical reasoning and programmatic features possesses strong general transferability, enabling models to develop stronger capabilities across multiple tasks, while a stable core subset of data is sufficient to consistently activate fundamental model capabilities and universally improve performance across diverse tasks. 

---
# CLARIN-PT-LDB: An Open LLM Leaderboard for Portuguese to assess Language, Culture and Civility 

**Authors**: João Silva, Luís Gomes, António Branco  

**Link**: [PDF](https://arxiv.org/pdf/2603.12872)  

**Abstract**: This paper reports on the development of a leaderboard of Open Large Language Models (LLM) for European Portuguese (PT-PT), and on its associated benchmarks. This leaderboard comes as a way to address a gap in the evaluation of LLM for European Portuguese, which so far had no leaderboard dedicated to this variant of the language. The paper also reports on novel benchmarks, including some that address aspects of performance that so far have not been available in benchmarks for European Portuguese, namely model safeguards and alignment to Portuguese culture. The leaderboard is available at this https URL. 

---
# Using a Human-AI Teaming Approach to Create and Curate Scientific Datasets with the SCILIRE System 

**Authors**: Necva Bölücü, Jessica Irons, Changhyun Lee, Brian Jin, Maciej Rybinski, Huichen Yang, Andreas Duenser, Stephen Wan  

**Link**: [PDF](https://arxiv.org/pdf/2603.12638)  

**Abstract**: The rapid growth of scientific literature has made manual extraction of structured knowledge increasingly impractical. To address this challenge, we introduce SCILIRE, a system for creating datasets from scientific literature. SCILIRE has been designed around Human-AI teaming principles centred on workflows for verifying and curating data. It facilitates an iterative workflow in which researchers can review and correct AI outputs. Furthermore, this interaction is used as a feedback signal to improve future LLM-based inference. We evaluate our design using a combination of intrinsic benchmarking outcomes together with real-world case studies across multiple domains. The results demonstrate that SCILIRE improves extraction fidelity and facilitates efficient dataset creation. 

---
# SectEval: Evaluating the Latent Sectarian Preferences of Large Language Models 

**Authors**: Aditya Maheshwari, Amit Gajkeshwar, Kaushal Sharma, Vivek Patel  

**Link**: [PDF](https://arxiv.org/pdf/2603.12768)  

**Abstract**: As Large Language Models (LLMs) becomes a popular source for religious knowledge, it is important to know if it treats different groups fairly. This study is the first to measure how LLMs handle the differences between the two main sects of Islam: Sunni and Shia. We present a test called SectEval, available in both English and Hindi, consisting of 88 questions, to check the bias-ness of 15 top LLM models, both proprietary and open-weights. Our results show a major inconsistency based on language. In English, many powerful models DeepSeek-v3 and GPT-4o often favored Shia answers. However, when asked the exact same questions in Hindi, these models switched to favoring Sunni answers. This means a user could get completely different religious advice just by changing languages. We also looked at how models react to location. Advanced models Claude-3.5 changed their answers to match the user's country-giving Shia answers to a user from Iran and Sunni answers to a user from Saudi Arabia. In contrast, smaller models (especially in Hindi) ignored the user's location and stuck to a Sunni viewpoint. These findings show that AI is not neutral; its religious ``truth'' changes depending on the language you speak and the country you claim to be from. The data set is available at this https URL 

---
# CSE-UOI at SemEval-2026 Task 6: A Two-Stage Heterogeneous Ensemble with Deliberative Complexity Gating for Political Evasion Detection 

**Authors**: Christos Tzouvaras, Konstantinos Skianis, Athanasios Voulodimos  

**Link**: [PDF](https://arxiv.org/pdf/2603.12453)  

**Abstract**: This paper describes our system for SemEval-2026 Task 6, which classifies clarity of responses in political interviews into three categories: Clear Reply, Ambivalent, and Clear Non-Reply. We propose a heterogeneous dual large language model (LLM) ensemble via self-consistency (SC) and weighted voting, and a novel post-hoc correction mechanism, Deliberative Complexity Gating (DCG). This mechanism uses cross-model behavioral signals and exploits the finding that an LLM response-length proxy correlates strongly with sample ambiguity. To further examine mechanisms for improving ambiguity detection, we evaluated multi-agent debate as an alternative strategy for increasing deliberative capacity. Unlike DCG, which adaptively gates reasoning using cross-model behavioral signals, debate increases agent count without increasing model diversity. Our solution achieved a Macro-F1 score of 0.85 on the evaluation set, securing 3rd place. 

---
# LLM-Augmented Therapy Normalization and Aspect-Based Sentiment Analysis for Treatment-Resistant Depression on Reddit 

**Authors**: Yuxin Zhu, Sahithi Lakamana, Masoud Rouhizadeh, Selen Bozkurt, Rachel Hershenberg, Abeed Sarker  

**Link**: [PDF](https://arxiv.org/pdf/2603.12343)  

**Abstract**: Treatment-resistant depression (TRD) is a severe form of major depressive disorder in which patients do not achieve remission despite multiple adequate treatment trials. Evidence across pharmacologic options for TRD remains limited, and trials often do not fully capture patient-reported tolerability. Large-scale online peer-support narratives therefore offer a complementary lens on how patients describe and evaluate medications in real-world use. In this study, we curated a corpus of 5,059 Reddit posts explicitly referencing TRD from 3,480 subscribers across 28 mental health-related subreddits from 2010 to 2025. Of these, 3,839 posts mentioned at least one medication, yielding 23,399 mentions of 81 generic-name medications after lexicon-based normalization of brand names, misspellings, and colloquialisms. We developed an aspect-based sentiment classifier by fine-tuning DeBERTa-v3 on the SMM4H 2023 therapy-sentiment Twitter corpus with large language model based data augmentation, achieving a micro-F1 score of 0.800 on the shared-task test set. Applying this classifier to Reddit, we quantified sentiment toward individual medications across three categories: positive, neutral, and negative, and tracked patterns by drug, subscriber, subreddit, and year. Overall, 72.1% of medication mentions were neutral, 14.8% negative, and 13.1% positive. Conventional antidepressants, especially SSRIs and SNRIs, showed consistently higher negative than positive proportions, whereas ketamine and esketamine showed comparatively more favorable sentiment profiles. These findings show that normalized medication extraction combined with aspect-based sentiment analysis can help characterize patient-perceived treatment experiences in TRD-related Reddit discourse, complementing clinical evidence with large-scale patient-generated perspectives. 

---
# Not Just the Destination, But the Journey: Reasoning Traces Causally Shape Generalization Behaviors 

**Authors**: Pengcheng Wen, Yanxu Zhu, Jiapeng Sun, Han Zhu, Yujin Zhou, Chi-Min Chan, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2603.12397)  

**Abstract**: Chain-of-Thought (CoT) is often viewed as a window into LLM decision-making, yet recent work suggests it may function merely as post-hoc rationalization. This raises a critical alignment question: Does the reasoning trace causally shape model generalization independent of the final answer? To isolate reasoning's causal effect, we design a controlled experiment holding final harmful answers constant while varying reasoning paths. We construct datasets with \textit{Evil} reasoning embracing malice, \textit{Misleading} reasoning rationalizing harm, and \textit{Submissive} reasoning yielding to pressure. We train models (0.6B--14B parameters) under multiple paradigms, including question-thinking-answer (QTA), question-thinking (QT), and thinking-only (T-only), and evaluate them in both think and no-think modes. We find that: (1) CoT training could amplify harmful generalization more than standard fine-tuning; (2) distinct reasoning types induce distinct behavioral patterns aligned with their semantics, despite identical final answers; (3) training on reasoning without answer supervision (QT or T-only) is sufficient to alter behavior, proving reasoning carries an independent signal; and (4) these effects persist even when generating answers without reasoning, indicating deep internalization. Our findings demonstrate that reasoning content is causally potent, challenging alignment strategies that supervise only outputs. 

---
# Interpreting Negation in GPT-2: Layer- and Head-Level Causal Analysis 

**Authors**: Abdullah Al Mofael, Lisa M. Kuhn, Ghassan Alkadi, Kuo-Pao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.12423)  

**Abstract**: Negation remains a persistent challenge for modern language models, often causing reversed meanings or factual errors. In this work, we conduct a causal analysis of how GPT-2 Small internally processes such linguistic transformations. We examine its hidden representations at both the layer and head level. Our analysis is based on a self-curated 12,000-pair dataset of matched affirmative and negated sentences, covering multiple linguistic templates and forms of negation. To quantify this behavior, we define a metric, the Negation Effect Score (NES), which measures the model's sensitivity in distinguishing between affirmative statements and their negations. We carried out two key interventions to probe causal structure. In activation patching, internal activations from affirmative sentences were inserted into their negated counterparts to see how meaning shifted. In ablation, specific attention heads were temporarily disabled to observe how logical polarity changed. Together, these steps revealed how negation signals move and evolve through GPT-2's layers. Our findings indicate that this capability is not widespread; instead, it is highly concentrated within a limited number of mid-layer attention heads, primarily within layers 4 to 6. Ablating these specific components directly disrupts the model's negation sensitivity: on our in-domain, ablation increased NES (indicating weaker negation sensitivity), and re-introducing cached affirmative activations (rescue) increased NES further, confirming that these heads carry affirmative signal rather than restoring baseline behavior. On xNot360, ablation slightly decreased NES and rescue restored performance above baseline. This pattern demonstrates that these causal patterns are consistent across various negation forms and remain detectable on the external xNot360 benchmark, though with smaller magnitude. 

---
# ActTail: Global Activation Sparsity in Large Language Models 

**Authors**: Wenwen Hou, Xinyuan Song, Shiwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.12272)  

**Abstract**: Activation sparsity is a promising approach for accelerating large language model (LLM) inference by reducing computation and memory movement. However, existing activation sparsity methods typically apply uniform sparsity across projections, ignoring the heterogeneous statistical properties of Transformer weights and thereby amplifying performance degradation. In this paper, we propose ActTail, a TopK magnitude-based activation sparsity method with global activation sparsity allocation grounded in Heavy-Tailed Self-Regularization (HT-SR) theory. Specifically, we capture this heterogeneity via the heavy-tail exponent computed from each projection's empirical spectral density (ESD), which is used as a quantitative indicator to assign projection-specific sparsity budgets. Importantly, we provide a theoretical analysis that establishes an explicit relationship between the activation sparsity ratio and the heavy-tail exponent under the HT-SR regime, offering principled guidance for sparsity allocation beyond heuristic design. Experiments on LLaMA and Mistral models show that our method improves both perplexity and downstream task performance at high sparsity compared to uniform allocation. At 80% sparsity, perplexity is reduced by 21.8% on LLaMA-2-7B, 40.1% on LLaMA-2-13B, and 9.4% on Mistral-7B. 

---
# FGTR: Fine-Grained Multi-Table Retrieval via Hierarchical LLM Reasoning 

**Authors**: Chaojie Sun, Bin Cao, Tiantian Li, Chenyu Hou, Ruizhe Li, Qing Fan  

**Link**: [PDF](https://arxiv.org/pdf/2603.12702)  

**Abstract**: With the rapid advancement of large language models (LLMs), growing efforts have been made on LLM-based table retrieval. However, existing studies typically focus on single-table query, and implement it by similarity matching after encoding the entire table. These methods usually result in low accuracy due to their coarse-grained encoding which incorporates much query-irrelated data, and are also inefficient when dealing with large tables, failing to fully utilize the reasoning capabilities of LLM. Further, multi-table query is under-explored in retrieval tasks. To this end, we propose a hierarchical multi-table query method based on LLM: Fine-Grained Multi-Table Retrieval FGTR, a new retrieval paradigm that employs a human-like reasoning strategy. Through hierarchical reasoning, FGTR first identifies relevant schema elements and then retrieves the corresponding cell contents, ultimately constructing a concise and accurate sub-table that aligns with the given query. To comprehensively evaluate the performance of FGTR, we construct two new benchmark datasets based on Spider and BIRD . Experimental results show that FGTR outperforms previous state-of-the-art methods, improving the F_2 metric by 18% on Spider and 21% on BIRD, demonstrating its effectiveness in enhancing fine-grained retrieval and its potential to improve end-to-end performance on table-based downstream tasks. 

---
# NeuroLoRA: Context-Aware Neuromodulation for Parameter-Efficient Multi-Task Adaptation 

**Authors**: Yuxin Yang, Haoran Zhang, Mingxuan Li, Jiachen Xu, Ruoxi Shen, Zhenyu Wang, Tianhao Liu, Siqi Chen, Weilin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.12378)  

**Abstract**: Parameter-Efficient Fine-Tuning (PEFT) techniques, particularly Low-Rank Adaptation (LoRA), have become essential for adapting Large Language Models (LLMs) to downstream tasks. While the recent FlyLoRA framework successfully leverages bio-inspired sparse random projections to mitigate parameter interference, it relies on a static, magnitude-based routing mechanism that is agnostic to input context. In this paper, we propose NeuroLoRA, a novel Mixture-of-Experts (MoE) based LoRA framework inspired by biological neuromodulation -- the dynamic regulation of neuronal excitability based on context. NeuroLoRA retains the computational efficiency of frozen random projections while introducing a lightweight, learnable neuromodulation gate that contextually rescales the projection space prior to expert selection. We further propose a Contrastive Orthogonality Loss to explicitly enforce separation between expert subspaces, enhancing both task decoupling and continual learning capacity. Extensive experiments on MMLU, GSM8K, and ScienceQA demonstrate that NeuroLoRA consistently outperforms FlyLoRA and other strong baselines across single-task adaptation, multi-task model merging, and sequential continual learning scenarios, while maintaining comparable parameter efficiency. 

---
# DIALECTIC: A Multi-Agent System for Startup Evaluation 

**Authors**: Jae Yoon Bae, Simon Malberg, Joyce Galang, Andre Retterath, Georg Groh  

**Link**: [PDF](https://arxiv.org/pdf/2603.12274)  

**Abstract**: Venture capital (VC) investors face a large number of investment opportunities but only invest in few of these, with even fewer ending up successful. Early-stage screening of opportunities is often limited by investor bandwidth, demanding tradeoffs between evaluation diligence and number of opportunities assessed. To ease this tradeoff, we introduce DIALECTIC, an LLM-based multi-agent system for startup evaluation. DIALECTIC first gathers factual knowledge about a startup and organizes these facts into a hierarchical question tree. It then synthesizes the facts into natural-language arguments for and against an investment and iteratively critiques and refines these arguments through a simulated debate, which surfaces only the most convincing arguments. Our system also produces numeric decision scores that allow investors to rank and thus efficiently prioritize opportunities. We evaluate DIALECTIC through backtesting on real investment opportunities aggregated from five VC funds, showing that DIALECTIC matches the precision of human VCs in predicting startup success. 

---
# Can Fairness Be Prompted? Prompt-Based Debiasing Strategies in High-Stakes Recommendations 

**Authors**: Mihaela Rotar, Theresia Veronika Rampisela, Maria Maistro  

**Link**: [PDF](https://arxiv.org/pdf/2603.12935)  

**Abstract**: Large Language Models (LLMs) can infer sensitive attributes such as gender or age from indirect cues like names and pronouns, potentially biasing recommendations. While several debiasing methods exist, they require access to the LLMs' weights, are computationally costly, and cannot be used by lay users. To address this gap, we investigate implicit biases in LLM Recommenders (LLMRecs) and explore whether prompt-based strategies can serve as a lightweight and easy-to-use debiasing approach. We contribute three bias-aware prompting strategies for LLMRecs. To our knowledge, this is the first study on prompt-based debiasing approaches in LLMRecs that focuses on group fairness for users. Our experiments with 3 LLMs, 4 prompt templates, 9 sensitive attribute values, and 2 datasets show that our proposed debiasing approach, which instructs an LLM to be fair, can improve fairness by up to 74% while retaining comparable effectiveness, but might overpromote specific demographic groups in some cases. 

---
# Multi-Step Semantic Reasoning in Generative Retrieval 

**Authors**: Steven Dong, Yubao Tang, Maarten de Rijke  

**Link**: [PDF](https://arxiv.org/pdf/2603.12368)  

**Abstract**: Generative retrieval (GR) models encode a corpus within model parameters and generate relevant document identifiers directly for a given query. While this paradigm shows promise in retrieval tasks, existing GR models struggle with complex queries in numerical contexts, such as those involving semantic reasoning over financial reports, due to limited reasoning capabilities. This limitation leads to suboptimal retrieval accuracy and hinders practical applicability. We propose ReasonGR, a framework designed to enhance multi-step semantic reasoning in numerical contexts within GR. ReasonGR employs a structured prompting strategy combining task-specific instructions with stepwise reasoning guidance to better address complex retrieval queries. Additionally, it integrates a reasoning-focused adaptation module to improve the learning of reasoning-related parameters. Experiments on the FinQA dataset, which contains financial queries over complex documents, demonstrate that ReasonGR improves retrieval accuracy and consistency, indicating its potential for advancing GR models in reasoning-intensive retrieval scenarios. 

---
