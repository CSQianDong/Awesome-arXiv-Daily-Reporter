# Very Efficient Listwise Multimodal Reranking for Long Documents 

**Authors**: Yiqun Sun, Pengfei Wei, Lawrence B. Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2605.11864)  

**Abstract**: Listwise reranking is a key yet computationally expensive component in vision-centric retrieval and multimodal retrieval-augmented generation (M-RAG) over long documents. While recent VLM-based rerankers achieve strong accuracy, their practicality is often limited by long visual-token sequences and multi-step autoregressive decoding. We propose ZipRerank, a highly efficient listwise multimodal reranker that directly addresses both bottlenecks. It reduces input length via a lightweight query-image early interaction mechanism and eliminates autoregressive decoding by scoring all candidates in a single forward pass. To enable effective learning, ZipRerank adopts a two-stage training strategy: (i) listwise pretraining on large-scale text data rendered as images, and (ii) multimodal finetuning with VLM-teacher-distilled soft-ranking supervision. Extensive experiments on the MMDocIR benchmark show that ZipRerank matches or surpasses state-of-the-art multimodal rerankers while reducing LLM inference latency by up to an order of magnitude, making it well-suited for latency-sensitive real-world systems. The code is available at this https URL. 

---
# Overview of the MedHopQA track at BioCreative IX: track description, participation and evaluation of systems for multi-hop medical question answering 

**Authors**: Rezarta Islamaj, Joey Chan, Robert Leaman, Jongmyung Jung, Hyeongsoon Hwang, Quoc-An Nguyen, Hoang-Quynh Le, Harikrishnan Gurushankar Saisudha, Ganesh Chandrasekar, Rustam R. Taktashov, Nadezhda Yu. Bizyukova, Sofia I. R. Conceição, Paulo R. C. Lopes, Reem Abdel Salam, Mary Adewunmi, Zhiyong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.12313)  

**Abstract**: Multi-hop question answering (QA) remains a significant challenge in the biomedical domain, requiring systems to integrate information across multiple sources to answer complex questions. To address this problem, the BioCreative IX MedHopQA shared task was designed to benchmark in multi-hop reasoning for large language models (LLMs). We developed a novel dataset of 1,000 challenging QA pairs spanning diseases, genes, and chemicals, with particular emphasis on rare diseases. Each question was constructed to require two-hop reasoning through the integration of information from two distinct Wikipedia pages. The challenge attracted 48 submissions from 13 teams. Systems were evaluated using both surface string comparison and conceptual accuracy (MedCPT score). The results showed a substantial performance gap between baseline LLMs and enhanced systems. The top-ranked submission achieved an 89.30% F1 score on the MedCPT metric and an 87.30% exact match (EM) score, compared with 67.40% and 60.20%, respectively, for the zero-shot baseline. A central finding of the challenge was that retrieval-augmented generation (RAG) and related retrieval-based strategies were critical for strong performance. In addition, concept-level evaluation improved answer assessment when correct responses differed in surface form. The MedHopQA dataset is publicly available to support continued progress in this important area. Challenge materials: this https URL and benchmark this https URL 

---
# AgentDisCo: Towards Disentanglement and Collaboration in Open-ended Deep Research Agents 

**Authors**: Jiarui Jin, Zexuan Yan, Shijian Wang, Wenxiang Jiao, Yuan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.11732)  

**Abstract**: In this paper, we present AgentDisCo, a novel Disentangled and Collaborative agentic architecture that formulates deep research as an adversarial optimization problem between information exploration and exploitation. Unlike existing approaches that conflate these two processes into a single module, AgentDisCo employs a critic agent to evaluate generated outlines and refine search queries, and a generator agent to retrieve updated results and revise outlines accordingly. The iteratively refined outline is then passed to a downstream report writer that synthesizes a comprehensive research report. The overall workflow supports both handcrafted and automatically discovered design strategies via a meta-optimization harness, in which the generator agent is repurposed as a scoring agent to evaluate critic outputs and generate quality signals. Powerful code-generation agents (e.g., Claude-Code, Codex) systematically explore agent configurations and construct a policy bank, a structured repository of reusable design strategies, enabling the framework to self-refine without extensive human intervention. We evaluate AgentDisCo on three established deep research benchmarks (DeepResearchBench, DeepConsult, DeepResearchGym) using Gemini-2.5-Pro, achieving performance comparable to or surpassing leading closed-source systems. Observing that existing benchmarks inadequately reflect real-world user needs, we introduce GALA (General AI Life Assistants), a benchmark that mines latent research interests from users' historical browsing behavior. We further develop a rendering agent that converts research reports into visually rich poster presentations, and demonstrate an end-to-end product, AutoResearch Your Interest, which delivers personalized deep research recommendations derived from individual browsing histories. 

---
# Latent Causal Void: Explicit Missing-Context Reconstruction for Misinformation Detection 

**Authors**: Hui Li, Zhongquan Jian, Jinsong Su, Junfeng Yao  

**Link**: [PDF](https://arxiv.org/pdf/2605.12156)  

**Abstract**: Automatic misinformation detection performs well when deception is visible in what an article explicitly states. However, some misinformation articles remain locally coherent and only become misleading once compared with contemporaneous reports that supply background facts the article omits. We study this omission-relevant setting and observe that current omission-aware approaches typically either attach retrieved context as auxiliary evidence or infer a categorical omission signal, leaving the specific missing fact implicit. We propose \emph{Latent Causal Void} (LCV), a retrieval-guided detector that explicitly reconstructs the missing fact for each target sentence and uses it as a textual cross-source relation in graph reasoning. Concretely, LCV retrieves temporally aligned context articles, asks a frozen instruction-tuned large language model to generate a short missing-context description for each sentence--article pair, and feeds the resulting relation text into a heterograph over target sentences and context articles. On the bilingual benchmark of Sheng et al., LCV improves over the strongest omission-aware baseline by $2.56$ and $2.84$ macro-F1 points on the English and Chinese splits, respectively. The results indicate that modeling the missing cross-source fact itself, rather than only attaching retrieved evidence or predicting an omission signal, is a useful representation for omission-aware misinformation detection. 

---
# Caraman at SemEval-2026 Task 8: Three-Stage Multi-Turn Retrieval with Query Rewriting, Hybrid Search, and Cross-Encoder Reranking 

**Authors**: David-Maximilian Caraman, Gheorghe Cosmin Silaghi  

**Link**: [PDF](https://arxiv.org/pdf/2605.12028)  

**Abstract**: We describe our system for SemEval-2026 Task 8 (MTRAGEval), participating in Task A (Retrieval) across four English-language domains. Our approach employs a three-stage pipeline: (1) query rewriting via a LoRA-fine-tuned Qwen 2.5 7B model that transforms context-dependent follow-up questions into standalone queries, (2) hybrid BM25 and dense retrieval combined through Reciprocal Rank Fusion, and (3) cross-encoder reranking with BGE-reranker-v2-m3. On the official test set, the system achieves nDCG@5 of 0.531, ranking 8th out of 38 participating systems and 10.7% above the organizer baseline. Development comparisons reveal that domain-specific temperature tuning for query generation, where technical domains benefit from deterministic decoding and general domains from controlled randomness, provides consistent gains, while more complex strategies such as domain-aware prompting and multi-query expansion degrade performance. 

---
# Mitigating Context-Memory Conflicts in LLMs through Dynamic Cognitive Reconciliation Decoding 

**Authors**: Yigeng Zhou, Wu Li, Yifan Lu, Yequan Wang, Xuebo Liu, Wenya Wang, Jun Yu, Min Zhang, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.12185)  

**Abstract**: Large language models accumulate extensive parametric knowledge through pre-training. However, knowledge conflicts occur when outdated or incorrect parametric knowledge conflicts with external knowledge in the context. Existing methods address knowledge conflicts through contrastive decoding, but in conflict-free scenarios, static approaches disrupt output distribution. Other dynamic decoding methods attempt to measure the degree of conflict but still struggle with complex real-world situations. In this paper, we propose a two-stage decoding method called Dynamic Cognitive Reconciliation Decoding (DCRD), to predict and mitigate context-memory conflicts. DCRD first analyzes the attention map to assess context fidelity and predict potential conflicts. Based on this prediction, the input is directed to one of two decoding paths: (1) greedy decoding, or (2) context fidelity-based dynamic decoding. This design enables DCRD to handle conflicts efficiently while maintaining high accuracy and decoding efficiency in conflict-free cases. Additionally, to simulate scenarios with frequent knowledge updates, we constructed ConflictKG, a knowledge conflict QA benchmark. Experiments on four LLMs across six QA datasets show that DCRD outperforms all baselines, achieving state-of-the-art performance. 

---
# ClinicalBench: Stress-Testing Assertion-Aware Retrieval for Cross-Admission Clinical QA on MIMIC-IV 

**Authors**: Alex Stinard  

**Link**: [PDF](https://arxiv.org/pdf/2605.11143)  

**Abstract**: Reasoning benchmarks measure clinical performance on clean inputs. We evaluate the step before reasoning: retrieval over real EHR notes, where negation, temporality, and family-versus-patient attribution can flip a correct answer to a wrong one. EpiKG carries an assertion label and a temporality tag with every fact in a patient knowledge graph, then routes retrieval by question intent. ClinicalBench is a 400-question test over 43 MIMIC-IV patients across 9 assertion-sensitive categories. A 7-condition ablation tests each piece of EpiKG across six LLMs (Claude Opus 4.6, GPT-OSS 20B, MedGemma 27B, Gemma 4 31B, MedGemma 1.5 4B, Qwen 3.5 35B). Three physicians blindly adjudicated 100 paired items. The author-blind primary endpoint, leave-author-out paired exact McNemar on 50 unanimous-strict items rated by two external physicians, yields +22.0 percentage points (95 percent Newcombe CI [+5.1, +31.5], p=0.0192). The architectural novelty, intent-aware KG-RAG over a Contriever dense-RAG baseline (C2b to C4g_kw on the change-excluded n=362 endpoint), is +8.84 percentage points (paired McNemar p=1.79e-3); +12.43 percentage points under oracle intent. Sensitivities agree directionally: three-rater physician majority +24.0 percentage points (subject to single-author circularity); deterministic keyword reproducibility proxy +39.5 percentage points. Across the six models, the gain shrinks as the LLM-alone baseline rises (beta=-1.123, r=-0.921, p=0.009). With n=6 this looks more like regression to the mean than encoding substituting for model size. Physician adjudication identified 56 percent of auto-generated reference answers as defective, a methodological finding indicating that NLP-pipeline clinical-QA benchmarks require physician adjudication to be usable. ClinicalBench, the frozen evaluator, three-rater adjudication data, and the EpiKG output stack are publicly released. 

---
# Test-Time Compute for Dense Retrieval: Agentic Program Generation with Frozen Embedding Models 

**Authors**: Han Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2605.11374)  

**Abstract**: Test-time compute is widely believed to benefit only large reasoning models. We show it also helps small embedding models. Most modern embedding checkpoints are distilled from large LLM backbones and inherit their representation space; a frozen embedding model should therefore benefit from extra inference compute without retraining. Using an agentic program-search loop, we explore 259 candidate inference programs over a frozen embedding API across ninety generations. The entire Pareto frontier collapses onto a single algebra: a softmax-weighted centroid of the local top-K documents interpolated with the query. This parameter-free default lifts nDCG@10 statistically significantly across seven embedding-model families spanning a tenfold parameter range, with held-out full-BEIR validation confirming the lift on every model tested. 

---
# Goal-Oriented Reasoning for RAG-based Memory in Conversational Agentic LLM Systems 

**Authors**: Jiazhou Liang, Armin Toroghi, Yifan Simon Liu, Faeze Moradi Kalarde, Liam Gallagher, Scott Sanner  

**Link**: [PDF](https://arxiv.org/pdf/2605.12213)  

**Abstract**: LLM-based conversational AI agents struggle to maintain coherent behavior over long horizons due to limited context. While RAG-based approaches are increasingly adopted to overcome this limitation by storing interactions in external memory modules and performing retrieval from them, their effectiveness in answering challenging questions (e.g., multi-hop, commonsense) ultimately depends on the agent's ability to reason over the retrieved information. However, existing methods typically retrieve memory based on semantic similarity to the raw user utterance, which lacks explicit reasoning about missing intermediate facts and often returns evidence that is irrelevant or insufficient for grounded reasoning. In this work, we introduce Goal-Mem, a goal-oriented reasoning framework for RAG-based agentic memory that performs explicit backward chaining from the user's utterance as a goal. Rather than progressively expanding from retrieved context, Goal-Mem decomposes each goal into atomic subgoals, performs targeted memory retrieval to satisfy each subgoal, and iteratively identifies what information from memory should be retrieved when intermediate goals cannot be resolved. We formalize this process in Natural Language Logic, a logical system that combines the verifiability of reasoning provided by FOL with the expressivity of natural language. Through extensive experiments on two datasets and comparing to nine strong memory baselines, we show that Goal-Mem consistently improves performance, particularly on tasks requiring multi-hop reasoning and implicit inference. 

---
# SAGE: A Self-Evolving Agentic Graph-Memory Engine for Structure-Aware Associative Memory 

**Authors**: Juntong Wang, Haoyue Zhao, guanghui Pan, Xiyuan Wang, Yanbo Wang, Qiyan Deng, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.12061)  

**Abstract**: Long-term memory is becoming a central bottleneck for language agents. Exsting RAG and GraphRAG systems largely treat memory graphs as static retrieval middleware, which limits their ability to recover complete evidence chains from partial cues, exploit reusable graph-structrual roles, and improve the memory itself through downstream feedback. We introduce SAGE, a Self-evolving Agentic Graph-memory Engine that models graph memory as a dynamic long-term memory substrate. SAGE couples two roles: a memory writer that incrementally constucts structured graph memory from interaction histories, and a Graph Foundation Model-based memory reader to perform retrieval and provide feedback to the memory writer. We provide rigorooous theoretical annalyses supporting the framework. Across multi-hop QA, open-domain retireval, domain-specific review QA, and long-term agent-memory benchmarks, SAGE improves evidence recovery, answer grounding, and retrieval efficiency: after two self-evolution rounds, it achieves the best average rank on multi-hop QA; in zero-shot open-domain transfer, it reaches 82.5/91.6 Recall@2/5 on NQ. Further results on LongMemEval and HaluMem show that traning and reader-writer feedback improve multiple long-term memory and hallucination-diagnostic metrics, suggesting that self-evolving, structure-aware graph memory is a promising foundation for robust long-horizon language agents. 

---
# LegalCheck: Retrieval- and Context-Augmented Generation for Drafting Municipal Legal Advice Letters 

**Authors**: Virgill van der Meer, Julien Rossi  

**Link**: [PDF](https://arxiv.org/pdf/2605.12012)  

**Abstract**: Public-sector legal departments in the Netherlands face acute staff shortages, increased case volumes, and increased pressure to meet regulatory compliance. This paper presents LegalCheck, a novel system that addresses these challenges by automating the drafting of objection response letters through a combination of Retrieval-Augmented Generation (RAG) and Context-Augmented Generation (CAG). Using a large language model (LLM) alongside curated legal knowledge bases, LegalCheck performs retrieval of relevant laws and precedents, and uses controlled prompting to incorporate both external knowledge and case-specific details into a coherent draft. An expert-in-the-loop review ensures that each generated letter is legally sound and contextually appropriate. In a real-world deployment within the Municipality of Amsterdam, LegalCheck produced near-final advice letters in minutes rather than hours, while maintaining high legal consistency and factual accuracy. The output is based on actual regulations and prior cases, providing explainable outputs that captured the vast majority of required legal reasoning (often 80\% to 100\% of essential content). Legal professionals found that the system reduced their workload and ensured a consistent application of legal standards, without replacing human judgment. These results demonstrate substantial efficiency gains, improved legal consistency, and positive user acceptance. More broadly, this work illustrates how responsible AI can be deployed in the legal domain by augmenting LLMs with domain knowledge and governance mechanisms. 

---
# CuSearch: Curriculum Rollout Sampling via Search Depth for Agentic RAG 

**Authors**: Jianghan Shen, Siqi Luo, Xinyu Cheng, Jing Xiong, Yue Li, Jiyao Liu, Jiashi Lin, Yirong Chen, Junjun He  

**Link**: [PDF](https://arxiv.org/pdf/2605.11611)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a promising paradigm for training agentic retrieval-augmented generation (RAG) systems from outcome-only supervision. Most existing methods optimize policies from uniformly sampled rollouts, implicitly treating all trajectories as equally informative. However, trajectories differ substantially in search depth and are therefore not equally informative: deeper-search trajectories contain more retrieval decision points and provide denser direct supervision for the retrieval sub-policy. Moreover, this heterogeneity grows over training as the within-batch depth distribution shifts toward higher values, yet uniform rollout sampling remains blind to this shift. To address this, we propose CuSearch, a curriculum rollout sampling framework built on Search-Depth Greedy Allocation (SDGA), a batch-level operator that reallocates a fixed update budget toward deeper-search trajectories. SDGA-Auto always targets the deepest available trajectories in the current batch, yielding an implicit training-aligned curriculum as the depth distribution shifts upward. SDGA-Phase explicitly advances the curriculum threshold as deeper trajectories become sufficiently abundant. Experiments across model types and retrieval frameworks show that CuSearch consistently improves performance, achieving up to 11.8 exact-match points over standard GRPO on ZeroSearch. These results establish per-trajectory search depth as a reliable, annotation-free proxy for retrieval supervision density in RLVR-based agentic RAG training. The code is available at this https URL. 

---
# Read, Grep, and Synthesize: Diagnosing Cross-Domain Seed Exposure for LLM Research Ideation 

**Authors**: Yunju Choi, Min Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.11532)  

**Abstract**: The discovery of novel methodologies for emerging problems is a continuing cycle in ML, often driven by the migration of techniques across domains. Building on this observation, we ask whether current LLM ideation systems benefit from targeted cross-domain retrieval or simply from exposure to diverse mechanisms. We study this question through PaperGym, a three-stage pipeline: (1) tool-augmented seed extraction via read, grep, and bash over an isolated paper environment, (2) cross-domain seed retrieval via paraphrasing across seven ML domains, and (3) method synthesis from retrieved seeds, each scored by rubric-based judges. Tool-augmented extraction improves specificity, and paraphrase-based retrieval broadens domain coverage. In synthesis, cross-domain retrieval receives more pairwise novelty wins than no-retrieval and same-domain baselines, but shows no significant difference from a random diverse-seed control. These findings suggest LLM ideation systems benefit from diverse seed exposure, but do not yet reliably exploit the semantic reason particular seeds were retrieved. We release the seed library, rubric prompts, and run scripts at this https URL 

---
# Persistent and Conversational Multi-Method Explainability for Trustworthy Financial AI 

**Authors**: Georgios Makridis, Georgios Fatouros, John Soldatos, George Katsis, Dimosthenis Kyriazis  

**Link**: [PDF](https://arxiv.org/pdf/2605.11687)  

**Abstract**: Financial institutions increasingly require AI explanations that are persistent, cross-validated across methods, and conversationally accessible to human decision-makers. We present an architecture for human-centered explainable AI in financial sentiment analysis that combines three contributions. First, we treat XAI artifacts -- LIME feature attributions, occlusion-based word importance scores, and saliency heatmaps -- as persistent, searchable objects in distributed S3-compatible storage with structured metadata and natural-language summaries, enabling semantic retrieval over explanation history and automatic index reconstruction after system failures. Second, we enable multi-method explanation triangulation, where a retrieval-augmented generation (RAG) assistant compares and synthesizes results from multiple XAI methods applied to the same prediction, allowing users to assess explanation robustness through natural-language dialogue. Third, we evaluate the faithfulness of generated explanations using automated checks over grounding completeness, hallucinated claims, and method-attribution behavior. We demonstrate the architecture on an EXTRA-BRAIN financial sentiment analysis pipeline using FinBERT predictions and present evaluation results showing that constrained prompting reduces hallucination rate by 36\% and increases method-attribution citations by 73\% compared to naive prompting. We discuss implications for trustworthy, human-centered AI services in regulated financial environments. 

---
# Rethinking Evaluation for LLM Hallucination Detection: A Desiderata, A New RAG-based Benchmark, New Insights 

**Authors**: Wenbo Chen, Veena Padmanabhan, Tootiya Giyahchi, Elaine Wong, Leman Akoglu  

**Link**: [PDF](https://arxiv.org/pdf/2605.11330)  

**Abstract**: Hallucination, broadly referring to unfaithful, fabricated, or inconsistent content generated by LLMs, has wide-ranging implications. Therefore, a large body of effort has been devoted to detecting LLM hallucinations, as well as designing benchmark datasets for evaluating these detectors. In this work, we first establish a desiderata of properties for hallucination detection benchmarks (HDBs) to exhibit for effective evaluation. A critical look at existing HDBs through the lens of our desiderata reveals that none of them exhibits all the properties. We identify two largest gaps: (1) RAG-based grounded benchmarks with long context are severely lacking (partly because length impedes human annotation); and (2) Existing benchmarks do not make available realistic label noise for stress-testing detectors although real-world use-cases often grapple with label noise due to human or automated/weak annotation. To close these gaps, we build and open-source a new RAG-based HDB called T RIVIA+ that underwent a rigorous human annotation process. Notably, our benchmark exhibits all desirable properties including (1) T RIVIA+ contains samples with the longest context in the literature; and (2) we design and share four sets of noisy labels with different, both sample-dependent and sampleindependent, noise schemes. Finally, we perform experiments on RAG-based HDBs, including our T RIVIA+, using popular SOTA detectors that reveal new insights: (i) ample room remains for current detectors to reach the performance ceiling on RAG-based HDBs, (ii) the basic LLM-as-a-Judge baseline performs competitively, and (iii) label noise hinders detection performance. We expect that our findings, along with our proposed benchmark 1 , will motivate and foster needed research on hallucination detection for RAG-based tasks. 

---
# EHR-RAGp: Retrieval-Augmented Prototype-Guided Foundation Model for Electronic Health Records 

**Authors**: Saeed Shurrab, Mariam Al-Omari, Dana El Samad, Farah E. Shamout  

**Link**: [PDF](https://arxiv.org/pdf/2605.12335)  

**Abstract**: Electronic Health Records (EHR) contain rich longitudinal patient information and are widely used in predictive modeling applications. However, effectively leveraging historical data remains challenging due to long trajectories, heterogeneous events, temporal irregularity, and the varying relevance of past clinical context. Existing approaches often rely on fixed windows or uniform aggregation, which can obscure clinically important signals. In this work, we introduce EHR-RAGp, a retrieval-augmented foundation model that dynamically integrates the most relevant patient history across diverse clinical event types. We propose a prototype-guided retrieval module that acts as an alignment mechanism and estimates the relevance of retrieved historical chunks with respect to a given prediction task, guiding the model towards the most informative context. Across multiple clinical prediction tasks, EHR-RAGp consistently outperforms state-of-the-art EHR foundation models and transformer-based baselines. Furthermore, integrating EHR-RAGp with existing clinical foundation models yields substantial performance gains. Overall, EHR-RAGp provides a scalable and efficient framework for leveraging long-range clinical context to improve downstream performance. 

---
# Beyond Similarity Search: Tenure and the Case for Structured Belief State in LLM Memory 

**Authors**: Jeffrey Flynt  

**Link**: [PDF](https://arxiv.org/pdf/2605.11325)  

**Abstract**: Why do we need another AI to help the AI? We argue you don't. Stateless LLM sessions impose re-orientation costs on iterative, session-heavy workflows. Prior work addresses cross-session memory through retrieval-augmented approaches: store history, embed it, retrieve by semantic similarity. Cross-session memory is a state management problem, not a search problem. Similarity search fails for named entity resolution within bounded vocabulary contexts because beliefs about a shared technical domain are semantically proximate by construction. A single user is the simplest bounded vocabulary context; engineering teams converge on the same property through shared codebases and terminology.
We present Tenure, a local-first proxy that maintains a typed belief store with epistemic status, versioned supersession, and scope isolation, injecting curated context into every LLM session through precision-first retrieval. Hard scope isolation provides a structural guarantee: the right beliefs surface, and only within the boundaries the user has authorized. Tenure's typed schema converts extracted facts into imperative instructions via a why it matters field, making injected beliefs directly actionable rather than raw material for the model to re-derive.
A controlled evaluation on 72 retrieval cases demonstrates the gap. Cosine similarity over dense embeddings achieves mean precision of 0.12. Alias-weighted BM25 maintains mean precision of 1.0, passing 72/72 cases versus 8/72 for cosine similarity on the same corpus. Hybrid retrieval typically solves vocabulary mismatch between disparate authors; Tenure eliminates this structurally: query and belief authors are the same person, and an alias enrichment flywheel continuously indexes their specific vocabulary. Under multi-turn topic drift this worsens: the vector backend produces drift scores of 0.43--0.50 on noise-critical turns where BM25 maintains 0. 

---
# Adversarial SQL Injection Generation with LLM-Based Architectures 

**Authors**: Ali Karakoc, H. Birkan Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2605.11188)  

**Abstract**: SQL injection (SQLi) attacks are still one of the serious attacks ranked in the Open Worldwide Application Security Project (OWASP) Top 10 threats. Today, with advances in Artificial Intelligence (AI), especially in Large Language Models (LLMs), an opportunity has been created for automating adversarial attack tests to measure the defense mechanisms. In this paper, we aim to create a comprehensive evaluation of use cases that utilize LLMs for adversarial SQL injection generation. We introduce two novel LLM-based systems, Retrieval Augmented Generation for Adversarial SQLi (RADAGAS) and Reflective Chain-of-Thought SQLi (RefleXQLi), and compare them with existing baselines against 10 Web Application Firewalls (WAFs) and one execution-based MySQL validator. To perform a comprehensive test, we used six rule-based open-source WAFs (ModSecurity PL1--3, Coraza PL1--3), 2 AI/ML-based WAFs (WAF Brain, CNN-WAF), and 2 commercial WAFs (AWS WAF and Cloudflare WAF). For the LLM models, we used GPT-4o, Claude 3.7 Sonnet, and DeepSeek R1. Our tests consist of 240 experiments that generate 240,000 payloads and perform 2.2 million tests against WAFs. Our comprehensive evaluation reveals that RADAGAS-GPT4o outperforms other baseline models with a 22.73\% bypass rate. The proposed RADAGAS variants are highly successful on AI/ML-based WAFs (92.49\% on WAF-Brain by RADAGAS-DeepSeek, 80.48\% on CNN-WAF by RADAGAS-Claude), but struggle to bypass rule-based WAFs (0--5.70\% on ModSecurity and Coraza). In addition to these findings, another observation is that creating less diverse payloads achieves more bypasses, however they show poor results if the initially chosen payload is not successful. We observe that our findings provide a comprehensive view on using LLM-based approaches in security testing. 

---
# Context-Gated Associative Retrieval: From Theory to Transformers 

**Authors**: Moulik Choraria, Argyrios Gerogiannis, Vidhata Jayaraman, Ankur Mani, Lav R. Varshney  

**Link**: [PDF](https://arxiv.org/pdf/2605.10970)  

**Abstract**: Hopfield networks and their generalizations have established deep connections among biological associative memories, statistical physics, and transformers. Yet most models treat retrieval as a fixed query-to-memory mapping, ignoring the role of external context in recall. In this work, we propose a two-stage associative memory architecture, wherein a context-gate subcircuit reshapes the retrieval energy landscape before and during recall. We show theoretically that context gating increases inter-memory separation while inducing sparsity, translating into exponential improvements in retrieval. Crucially, we prove that the system admits a unique self-consistent fixed point, revealing that the resulting retrieval state is driven by both a direct contextual bias and a second-order retrieval-gate feedback loop. We then bridge this theory to transformers; specifically, we evaluate a first-order approximation on Llama-3, confirming that in-context learning acts as context-gated retrieval. Native dynamics mirror our theory: context localizes a memory subspace, enabling the zero-shot query to cleanly discriminate. Ultimately, this framework provides a mechanistic link between associative memory theory and LLM phenomenology. 

---
