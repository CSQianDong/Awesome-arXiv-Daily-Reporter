# Flash-dLLM: IO-Aware KV Caching and Parallel Decoding for Fast, Memory-Efficient Diffusion LLMs 

**Authors**: Quan Nguyen-Tri, Mukul Ranjan, Zhiqiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2609.26796)  

**Abstract**: Diffusion Large Language Models (dLLMs) have recently emerged as a promising alternative to autoregressive LLMs by enabling non-autoregressive text generation. However, their practical deployment remains limited by inefficient inference, largely due to the absence of effective Key-Value (KV) caching and scalable parallel decoding mechanisms. Existing acceleration methods typically study KV caching and parallel decoding in isolation, overlooking the I/O bottlenecks that arise when cache reuse and parallel token verification are jointly applied. In this work, we introduce $\textbf{Flash-dLLM}$, a training-free inference acceleration framework for fast and memory-efficient dLLMs. Flash-dLLM first identifies GPU memory I/O as a dominant bottleneck in KV-cache-enabled dLLM inference and addresses it with an I/O-aware fused KV-cache kernel that reduces redundant memory movement. Building on this optimized cache mechanism, Flash-dLLM further proposes an efficient KV-cache-driven draft-and-verify decoding strategy, where the dLLM itself serves as both drafter and verifier without requiring an auxiliary model. This unified design enables faster decoding while preserving generation quality and improving scalability to longer sequences and larger batch size. Extensive experiments on mathematical reasoning and code-generation benchmarks demonstrate that Flash-dLLM consistently outperforms existing state-of-the-art dLLM acceleration methods in both inference speed and memory efficiency. In particular, it achieves $5.1\times$ and $11.0\times$ speedups over prior strongest baseline Elastic-Cache on GSM8K and HumanEval, respectively. 

---
# Agensh: Scaling Organizational Intelligence to 1,024 Agents 

**Authors**: Zhihao Zhan, Ting Song, Li Dong, Shaohan Huang, Jianxun Lian, Yan Xia, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2609.26781)  

**Abstract**: A multi-agent system can reduce latency on complex tasks by executing work concurrently. Several pioneering harness frameworks support multi-agent systems. However, the scalability of current multi-agent harnesses is often constrained by a central orchestrator's capacity to allocate tasks and coordinate workers. To address this limitation, we introduce Agensh, a scalable self-organized multi-agent harness without a central orchestrator: concurrent workers execute a multi-agent cooperation loop, continuously gathering context, claiming and self-assigning sub-tasks, taking action and sharing findings, verifying results, and merging progress in an asynchronous manner. The loop is supported by the agentic organization infrastructure comprising three components: a shared workspace holds proposed, ongoing, and completed work; a message interface lets workers communicate; and shared context retains reusable findings and work intentions. To test the scalability of Agensh, we evaluate it on the five hardest ProgramBench tasks with GPT-5.6-sol (high). Scaling from 1 to 128 agents raises the mean final test-pass rate from 19.31% to 28.78%, an approximately 49% relative improvement. Larger organizations reach comparable test-pass rates earlier. On pandoc, scaling from 1 to 1,024 agents raises the final test-pass rate from 33.89% to 55.06%. Worker trajectories further show that different forms of self-organized cooperation gradually emerges and standardizes as the organization grows. These results reveal the number of agents as a new scaling dimension for multi-agent organizations to expand the frontier of general intelligence, offering a practical solution for complex tasks under hard latency constraints or time budgets. 

---
# SpeakerMem-R1: Speaker-Centered Dual-Track Memory for Multi-Party Dialogue 

**Authors**: Haobo Zheng, Tan Tang, Yan Chen, Weijie Wang, Yingcai Wu  

**Link**: [PDF](https://arxiv.org/pdf/2609.26780)  

**Abstract**: Long-term conversational memory in multi-party settings requires more than retrieving relevant content from long-term conversations: it must distinguish who said what, whom each statement concerns, how individuals perceive one another, what information is shared by the group, and how states change over time. Recent studies on multi-party dialogue benchmarks show that existing general-purpose LLM memory systems tend to lose person and group relations or struggle to integrate clues distributed across members, groups, and time. Together, these issues reveal two core bottlenecks: message attribution and relational understanding in multi-party dialogue, and state reconstruction from interleaved histories. To address both, we propose $\textbf{SpeakerMem-R1}$: its dual-track memory stores speaker-labeled verbatim messages and derived states organized into person-level and group-level views, then combines evidence from both tracks by entity, event, and time at query time. To reduce attribution and update errors during structured memory construction while enabling local deployment, we train Writer-R1 with SpeakerLevenshtein and speaker-conditioned GRPO. On GroupMemBench, SocialMemBench, and EverMemBench, SpeakerMem-R1 achieves binary accuracies of 47.9%, 69.2%, and 61.9%, respectively. On the publicly reported EverMemBench leaderboard from EverMind-AI, we achieves 62.33%, the best reported result among the latest state-of-the-art frameworks. It also achieves 70.85% on all 1,986 LoCoMo questions, which we use as a two-person long-term conversation boundary test. In a controlled evaluation of 305 questions, RL raises the SFT Writer's mean accuracy from 57.38% to 68.20%. We report both binary accuracy and token-F1, and ablations show that the verbatim and structured tracks, as well as person-level and group-level views, are complementary under the standardized evaluation interface. 

---
# Beyond Repeated Sampling: Learning Search Policies for LLM Reasoning 

**Authors**: Ismail Labiad, Matthieu Kowalski, Marc Schoenauer, Rémi Munos, Julia Kempe  

**Link**: [PDF](https://arxiv.org/pdf/2609.26704)  

**Abstract**: Large language models increasingly tackle hard reasoning problems by spending more test-time compute, yet the dominant strategy remains naive repeated sampling: draw many independent solutions and hope one is correct. Because such sampling explores only through local decoding noise, it tends to produce many near duplicate attempts rather than genuinely different ideas. We ask whether exploration can instead be steered at a semantic level, by first sampling problem specific concepts, hints, or strategies and then conditioning answer generation on them. We refine this into a simple, more exploratory procedure that emits many diverse concepts in a single trajectory, and evaluate it on hard problems where repeated sampling struggles. We then go a step further and make concept generation trainable: a small concept generator is optimized with reinforcement learning so that its concepts maximize the downstream success of a larger, frozen answer generator. On hard mathematical reasoning problems, the trained concept generator substantially improves the answer generator's pass@k over naive repeated sampling at the same answer generation allocation, surpasses concepts drawn from much larger untuned models, and transfers to answer generators it was never trained against, including a model from a different family. A small model can thus be trained into an effective, reusable search policy for a much larger one. 

---
# Measuring the Serving Stack Instead of the Model: Hidden Confounds in Local Tool-Use Evaluation 

**Authors**: Lijuan Tang, Yuemeng Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2609.26693)  

**Abstract**: A coding agent must emit a valid tool call--a parseable invocation of a tool in the provided schema--before the harness can execute its chosen action. We study how local serving stacks affect this protocol step and show that measured outcomes can depend on the serving layer rather than model behavior alone. In Ollama, the default tools= request is gated per model by a static template flag: some models are accepted and return calls as text, some return native tool_calls, while Phi-3 and Gemma-3 are rejected before inference. In our harness, rejection and retry exhaustion are not preserved as structured failure metadata, so downstream analysis can misclassify them as model non-calls and naively report 0% fidelity. Adding a text tool list while retaining the native channel recovers much of the measured fidelity for accepted models, whereas a uniform text protocol reduces fidelity for Llama-3.2, which has native tool-call support. Cross-stack probes on Ollama, this http URL, vLLM, and SGLang show different handling of the same request. Constrained decoding removes parse failures but can induce non-termination, and turn-pooled versus per-instance estimates differ by up to about 55 points. We conclude with a checklist for treating serving behavior as part of the evaluation protocol. 

---
# Detecting GPT-Assisted Writing Using Interpretable Stylometric Features 

**Authors**: Rajesh Kumar, Nabeel Siddiqui, Alexander Fuchsberger  

**Link**: [PDF](https://arxiv.org/pdf/2609.26687)  

**Abstract**: Distinguishing GPT-assisted from independently authored student writing has become a critical challenge in academia. This paper evaluates the discriminative capability of interpretable stylometric features extracted solely from submitted text. Using data from 90 participants who wrote both independently and with ChatGPT assistance, we evaluate eight machine learning classifiers while keeping data from the same participant together during validation. On the held-out test set, Random Forest achieved an ROC-AUC of 0.87 and an F1-score of 0.84, with False Positive and False Negative rates of 22.2% and 11.1%, respectively. SHAP analysis shows that lexical and grammatical characteristics drive the resulting predictions. The findings suggest that transparent, text-intrinsic features provide measurable signal for detecting GPT-assisted writing. 

---
# Diffusion Drafts, AR Verifies: Accelerating Document OCR with Self-Speculative Decoding 

**Authors**: Dohyun Kim, Sungjun Han, Hyungguk Kim, Yusik Kim, Jamin Shin, Paul Hongsuck Seo, Hongjoon Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2609.26638)  

**Abstract**: Autoregressive OCR vision-language models accurately convert document images into text and structured markup, but require one sequential decoding step per output token, limiting inference speed. Unlike open-ended text generation, OCR outputs are strongly grounded in the input image, making diffusion-based parallel generation promising. However, when several tokens are predicted in one diffusion step, each is predicted before the others are known. Committing them directly can therefore introduce errors. We therefore introduce GravityOCR, a parameter-shared AR-block-diffusion model jointly trained for parallel drafting and causal AR verification. Verifying drafts before commitment lets the model commit multiple output tokens per round without a separate drafting network. The causal AR path also enables GRPO with sequence- and structure-level OCR rewards, avoiding diffusion-trajectory likelihood estimation while updating the shared drafter parameters. On OmniDocBench v1.6, AR-path GRPO improves the Overall score from 94.92 to 95.16 without reducing diffusion drafting efficiency, while the final model remains close to the original GLM-OCR score of 95.48. In an SGLang serving deployment, GravityOCR commits an average of 9.7 output tokens per forward pass and achieves a $3.94\times$ decode-only speedup on region crops and a $1.32\times$ end-to-end page-processing speedup over AR decoding. 

---
# Capable yet Parsimonious: Extracting and Characterizing Hidden Chain-of-Thought in Frontier Models 

**Authors**: Xiaoyu Luo, Tao Ren, Wenrui Yu, Xiao Li, Qiongxiu Li, Johannes Bjerva  

**Link**: [PDF](https://arxiv.org/pdf/2609.26637)  

**Abstract**: The rapid capability gains of frontier language models are widely attributed to improved reasoning abilities, yet this cannot be verified as raw CoT traces in closed-source systems are hidden. By registering a simple custom tool through a standard API feature, we induce frontier models to externalize intermediate reasoning. Because these traces may reflect post-hoc rationalization rather than genuine reasoning, we first evaluate against native CoT on open-source models and extend to closed-source frontier models including GPT-6 Astra. We find that the extracted reasoning matches native reasoning performance and substantially outperforms no-reasoning baselines, across competition mathematics, science, and code generation. We then characterize how frontier models structure their intermediate reasoning. Across token efficiency, reasoning-step types, and induced reasoning trees, we identify systematic differences in how models externalize, compress, and organize reasoning. We find that Astra exhibits token-efficient directed reasoning, selecting a correct trajectory earlier, while resolving elementary steps internally and externalizing only crucial reasoning. These findings provide a behavioral lens on frontier-model reasoning beyond benchmark scores. 

---
# Knowledge Pull Requests for Continual Document Authoring 

**Authors**: Alexander Martin, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2609.26634)  

**Abstract**: We introduce Knowledge Pull Requests (KPRs), a framework for continual document authoring that makes each change interpretable. Documents require ongoing revision as new knowledge surfaces from other sources, languages, or times, but existing approaches either edit with no account of what knowledge changed or regenerate from scratch. A KPR integrates new knowledge into a document by extracting claims, filtering and routing them to sections, and flagging conflicts with existing content, producing a ChangeLog that separates what knowledge changes (claim proposal) from how the text changes (document diff). We evaluate KPRs on revising Wikipedia across languages and updating query-driven reports on RAGTIME. KPRs integrate more information and better preserve existing content than rewriting from sources or regenerating from scratch, while adding the most information per token generated. A KPR-revised article also grounds question answering better than a frontier model with search, which does not surface knowledge documented only in other languages. 

---
# PERSONAWEAVER: Controllable Diversity Beyond Conventional Archetypes in Procedural Character Generation 

**Authors**: Maan Qraitem, Kate Saenko, Bryan A. Plummer  

**Link**: [PDF](https://arxiv.org/pdf/2609.26629)  

**Abstract**: Procedural character generation aims to populate games, simulations, and other virtual worlds with diverse characters. Large language models (LLMs) offer a promising foundation for scaling this task. However, LLM-based procedural character generation remains at an early stage: existing methods either generate characters directly or adapt profiles retrieved from persona banks. As we show, both approaches produce behaviorally homogeneous populations: characters overwhelmingly agree with positive moral norms and respond to questions with helpful, assistant-like reactions. To mitigate this homogenization, we introduce PersonaWeaver, which disentangles world building from behavioral specification and models behavior through setting general, diverse, manually curated banks of moral positions and conversational reactions. This design allows us to test how far LLM(s) can be pushed beyond their default behavioral patterns across settings. Across ten realistic and fantastical settings and three LLM(s), PersonaWeaver produces broader moral and interactional response distributions than prior work. Its guidance also diversifies interpersonal language, response length, and sentiment. It also produces less archetypal combinations of world attributes. Code is available at this https URL. 

---
# Semantic Abstraction for Natural Language Inference: a Methodological Framework for Discovering and Compensating Semantic Knowledge and Reasoning Gaps in Large Language Models 

**Authors**: David Torres-Moreno, Jorge Hermosillo-Valadez  

**Link**: [PDF](https://arxiv.org/pdf/2609.26610)  

**Abstract**: Despite their outstanding performance on many NLP tasks, LLMs face serious challenges related to semantic abstraction. In this study, we are interested in understanding how LLMs leverage abstract semantic knowledge in natural language inference (NLI), which requires sophisticated linguistic capabilities to interpret implicit meanings, contextual conceptual relationships, and semantic connections between words and phrases. To this end, we propose a methodological framework for constructing new semantic knowledge at a higher level of abstraction, which we define under the notions of semantic compatibility and incompatibility for NLI. In this framework, the meaning of the lexical-semantic relations between the premise and the hypothesis is reconfigured to achieve a more flexible semantic network that induces different reasoning paths in LLMs. These new pathways show a consistent pattern of responses that allows agreement on a single response. The results demonstrate that our proposal allows to discover and compensate for LLMs' semantic knowledge gaps in NLI, achieving significant improvements in accuracy, exceeding 10% for some models, and in particular for the non-entailment class. It is essential to note that LLMs need structured knowledge and not just more data to bridge reasoning gaps. Our hybrid approach directs attention to overlooked word relationships, allowing models to synthesize missing information. We believe that the future lies not in increasing model size, but in creating a semantic scafolding that mimics the flexibility of human thinking. Hopefully, our proposal will enable the development of more robust agents and interpretable reasoning, guiding AI toward reliable language understanding. 

---
# Receptiveness, Not Sycophancy: Distinguishing Engagement from Deference in Language Models 

**Authors**: Calvin Isley, Johann Gaebler, Max Lamparth, Julia Minson, Sharad Goel  

**Link**: [PDF](https://arxiv.org/pdf/2609.26579)  

**Abstract**: A central concern with language models is sycophancy: their tendency to defer to users' views at the expense of independent substantive judgment. In parallel, work on social sycophancy has focused on behaviors such as validation and positivity that may signal inappropriate deference. Yet the markers of social sycophancy are also characteristic of conversational receptiveness, a construct from social psychology shown to improve interactions across disagreement. We argue that this overlap creates a construct-validity problem for social sycophancy evaluations. Using a popular moral-advice dataset, we find that responses classified as more socially sycophantic are also more receptive. Further, increasing the receptiveness of human-written responses---while preserving their substantive conclusions---causes them to be classified as more socially sycophantic. This tight coupling raises the possibility that social sycophancy evaluations inadvertently penalize desirable behavior. In a preregistered experiment comparing substantively equivalent responses, participants prefer the more receptive responses, expect users to be more likely to listen to them, and are more willing to seek advice from their authors. The same overall pattern persists even among participants who believe the original question asker is in the wrong. Finally, we introduce a simple approach that substantially increases receptiveness without increasing substantive deference, demonstrating that conversational receptiveness and substantive independence can be achieved together. 

---
# A retrospective analysis on the use of LLMs to study infant syntax learning 

**Authors**: Hélie Bazin, Anouk Barberousse, François Yvon  

**Link**: [PDF](https://arxiv.org/pdf/2609.26539)  

**Abstract**: Large language models (LLMs) have increasingly been used to investigate how children acquire syntax at an early stage of development. This is notably the central scientific goal of the BabyLM challenge, a community-wide effort to develop models that achieve human-level syntactic performance while being trained on developmentally realistic corpora. In this paper, we reflect on the use of LLMs in the study of infant syntax learning by providing an epistemological assessment of several studies from this research program. We discuss how datasets are built, which models are implemented, how they are trained and syntactically evaluated. We observe significant assumptions in the methodology of BabyLM and related studies, thus mitigating their theoretical scope. We additionally observe that using developmentally-realistic corpora have limited effects on models performance on commonly-used benchmarks, which suggest important computational differences between LLMs and the infant syntax learner. 

---
# Transcribe, Translate, and Optimize: Joint Reward Learning for Speech Translation 

**Authors**: Yanghe Dong, Wanting Huang, Weiran Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.26536)  

**Abstract**: In LLM-based speech translation, transcription-based chain-of-thought (CoT) suffers from a mismatch between reference transcripts used in supervised fine-tuning (SFT) and model-generated transcripts at inference. To address this, we propose joint recognition and translation fine-tuning via group relative policy optimization (GRPO). We score both transcripts and translations, with translation conditioned on model-generated transcripts, and compare three token advantage strategies. Using Qwen2.5-Omni-3B across four languages, we evaluate CoT against direct speech translation (Direct ST) under SFT and GRPO, training on CoVoST 2 and testing on CoVoST 2 and FLEURS. CoT GRPO outperforms Direct ST GRPO by 1.77 and 0.83 average BLEU points on CoVoST 2 and FLEURS. Compared to CoT SFT, GRPO boosts BLEU by 0.82 and 0.67 points and reduces word error rate (WER) by 8.8% and 7.2% relatively. These results highlight reinforcement fine-tuning as an effective method to mitigate the training-inference mismatch, jointly improving recognition and translation. 

---
# A Semiotics-Aware Framework for Evaluating Fidelity and Coverage in Natural Language Generation 

**Authors**: Lorenzo Zangari, Davide Picca  

**Link**: [PDF](https://arxiv.org/pdf/2609.26527)  

**Abstract**: When two texts describe the same expression, standard metrics based on lexical overlap or whole-text similarity may fail to detect meaningful differences in how that expression is framed. We propose a framework to evaluate semiotic alignment between texts, where a semiotic profile encompasses both the contextual meaning and the discourse references made salient by a text. Our approach yields two scores, Semiotic Fidelity and Semiotic Coverage, estimating how much of one text's profile is supported by the other and how much of the other's profile it recovers. Experiments show that coverage is typically lower than fidelity, and that alignment between LLMs and human-curated data is highest at low sampling temperatures, while higher temperatures reduce this alignment. 

---
# Calibration as a First-Class Criterion in LLM Evaluation 

**Authors**: Mario Sanz-Guerrero, Katharina von der Wense  

**Link**: [PDF](https://arxiv.org/pdf/2609.26489)  

**Abstract**: Calibration of language models -- the alignment between expressed or implicit confidence and empirical correctness -- is a well-studied subfield within NLP. Methods to measure it already exist. The problem is adoption: outside this subfield, NLP research regularly introduces new models, datasets, and benchmarks without checking whether the model's confidence scores are meaningful. We argue that this adoption gap is a major obstacle to trustworthy LLM evaluation. Miscalibration causes problems in two distinct areas: at deployment, where overconfident mistakes cause real harm, and inside the research pipeline, where methods like LLM-as-a-judge, synthetic data generation, and active learning rely on calibrated confidence without verifying it. Standard calibration metrics only require two inputs per example: a confidence score and a correctness judgment. Most benchmarks in use today already provide both, meaning calibration can be reported immediately. For open-ended generation, however, defining these two inputs is still an open challenge. We argue that each NLP subfield should pair its main performance metric with a calibration score and call for treating calibration as an essential property of every model rather than a niche topic. 

---
# Spoken Language Models that Think Aloud 

**Authors**: Junyi Ao, Kainan Peng, Mingbo Ma, Shun Zhang, Zhenyu Tang, Xutai Ma, Xiang Li, Yinghao Li, Yuancheng Wang, Zhizheng Wu, Haizhou Li, Qing He, Xubo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.26488)  

**Abstract**: While Chain-of-Thought (CoT) reasoning has improved the capability of language models, directly applying it to Spoken Language Models (SLMs) may introduce long silent intervals under the serial "think-then-speak" paradigm, disrupting real-time spoken interaction. To address this issue, we propose an asynchronous think-aloud framework for reasoning-based SLMs within the Thinker-Talker architecture. The framework maintains a primary reasoning stream for logical deduction and a lightweight think-aloud stream that generates short, task-grounded progress utterances conditioned on the user input and the evolving reasoning state. A dynamic balance strategy coordinates the two streams at runtime, triggering additional think-aloud speech to avoid silent gaps and canceling pending utterances when the final response becomes ready. Experiments on spoken reasoning and question-answering benchmarks show that our approach substantially reduces user-audible silence during reasoning while maintaining answer accuracy comparable to that of a serial "think-then-speak" baseline, demonstrating the potential of asynchronous think-aloud for responsive interaction in SLMs. 

---
# How to Estimate Whether You Have Found Several Needles in a Haystack: Measuring Calibration in Multi-Label Text Classification 

**Authors**: Sophie Henning, Georg Hofmann, Alexander Schulte, Alexander Fraser, Annemarie Friedrich  

**Link**: [PDF](https://arxiv.org/pdf/2609.26468)  

**Abstract**: A key factor in deciding whether to trust an automatic prediction is its confidence score, which should be calibrated to match the actual probability of the prediction being correct. Most confidence calibration metrics target binary or multi-class tasks, while multi-label calibration remains largely underexplored. Multi-label classification tasks, such as assigning medical codes to clinical notes or determining news topics, are usually dominated by a large number of negatives, i.e., labels that do not apply. We show that existing binning schemes to compute label-wise expected calibration error either underestimate the error, simply reflect label frequency, or suffer from many bins with very few instances. To achieve trustworthy label-wise calibration errors, we propose a new binning scheme that gives equal weight to positive and negative label assignments. Our empirical study demonstrates that in contrast to existing binning schemes, our new scheme results in meaningful estimates of calibration error in hierarchical and in extreme multi-label classification. We also show that calibrating confidence scores of large language models for multi-label predictions is an open challenge. Our detailed analysis lays the foundation for further research by providing a solid evaluation metric for measuring calibration in multi-label classification. 

---
# Enriching Speech Emotion Representations with Conversational Context 

**Authors**: Arthur Peuvot, Romaric Besançon, Gaël de Chalendar, Bianca Vieru, Ioana Vasilescu  

**Link**: [PDF](https://arxiv.org/pdf/2609.26422)  

**Abstract**: Detecting emotions is necessary for building systems that can accurately and adaptively interact with humans. Speech Emotion Recognition (SER) has become an important research focus to develop intelligent spoken interfaces. However, most studies predict emotions at the utterance level, ignoring the conversational context, along with the emotional flow and speaker interactions it carries. In this paper, we introduce ACERT (Averaged Contextual Emotion Representation through Time), a module that integrates a flexible-length window of conversational context to better capture emotional evolution in spoken interactions. To evaluate the robustness of this method, we conducted experiments on datasets spanning diverse emotionally expressive styles and contexts. ACERT outperforms current state-of-the-art (SOTA) approaches on IEMOCAP, establishes the first context-aware benchmark on SAFE, and obtains strong results on MELD for unweighted, class-balanced metrics. Ablation studies show that ACERT's gains come from emotional and conversational continuity, rather than from speaker identity or acoustic conditions. 

---
# Combining Hierarchical Cognitive Process with Process Supervision for Interpretable Scene Safety Understanding 

**Authors**: Zhiyun Jiang, Hanyong Wang, Binbin Liang, Yu Xie, Zhengjie Wang, Menglong Yang, Wei Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.26399)  

**Abstract**: Scene safety understanding plays a life-or-death role in situational awareness in various critical domains. Traditional methods that rely on learning direct mappings between scenes and safety levels often lack interpretability, limiting their reliability in critical applications. An effective approach to overcoming this challenge lies in interpreting human cognitive processes and equipping machine models with analogous cognitive capabilities. This work explores an effective way of integrating scene safety cognitive process modeling and process supervision. Specifically, we first construct a hierarchical cognitive safety structure, which motivates the development of a novel, high-quality scene safety understanding dataset based on multi-step reasoning with process labels. This dataset serves both as a benchmark and a resource to improve the safety reasoning capabilities of Large Language Models (LLMs), while also enabling a granular analysis of intermediate reasoning steps through information flow and saliency-based techniques. Building upon this foundation, we introduce a modular and flexible process supervision framework that reflects the hierarchical nature of human cognition. This framework leverages LLMs as the core architecture and incorporates Low-Rank Adaptation(LoRA) and Mixture-of-Experts (MoE) strategies to enable specialization and collaboration among expert modules, each tasked with specific sub-processes of the overall reasoning chain. Systematic experimental evaluations and analyses confirm that our framework exhibits superior interpretability and performance characteristics compared to traditional approaches. 

---
# Layout-Guided Masking for GROBID: Lightweight Structural Gains in Large-Scale Scientific PDF Ingestion 

**Authors**: Luca Foppiano, Sana Khamassi, Vipul Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2609.26381)  

**Abstract**: Transforming scholarly PDFs into machine-readable fulltext remains a bottleneck for large-scale information systems. Recent vision-based parsers improve accuracy, but need GPUs and may introduce noise into the extracted text. GROBID, a modular font-stream parser running on CPU, is the de-facto standard for structuring scientific articles and underpins several of the largest open scholarly corpora. We pair it with a lightweight CPU detector localising figure, table, and paratext (header, footer, page number) regions, encoded as typed-area masks whose tokens are routed to GROBID's specialised models or discarded. On two PMC corpora, Bioinformatics (1,926 articles) and Materials Science (2,595), scored against JATS with a section-aware structural protocol, our extension improves over plain GROBID on most metrics (NS $+0.025$/$+0.013$; $+0.086$ paragraph recall on Materials Science, $d_z{=}1.08$), and caption-linked figure recovery improves on both corpora. On the external Table-BRGM benchmark, table detection recovers F1 $0.16 \to 0.94$ and table structure follows (GriTS-Top $0.27 \to 0.78$, below the strongest GPU system). On body text, against four vision-based systems (Docling, MinerU, olmOCR, this http URL), it has the best paragraph precision on both corpora, the best section detection on Materials Science, and a character error rate within 0.004 of the best GPU parser. End-to-end on CPU, it costs $2.7$--$3.2\times$ less than the cheapest GPU system (Docling) and $10$--$14\times$ less than generative parsers. 

---
# HySparse2: Hybrid Sparse Attention with Two-Level KV Sharing 

**Authors**: Jianyu Wei, Yizhao Gao, Qihao Zhang, Shimao Chen, Zhengju Tang, Yu Cheng, Shengjie Zhou, Zihan Jiang, Yifan Song, Hailin Zhang, Liang Zhao, Bo Yang, Gang Wang, Shijie Cao, Fuli Luo  

**Link**: [PDF](https://arxiv.org/pdf/2609.26368)  

**Abstract**: Long-horizon and multi-turn agents typically generate short actions and process long observations from tools and environments. This growing context demands efficient prefill, compact KV-cache storage, and accurate long-context retrieval. To meet these demands, we introduce HySparse2, a hybrid sparse attention architecture with two-level KV sharing. At the outer level, KV Bridging adopts a YOCO-style self-decoder and cross-decoder structure, but bridges only full-attention layers. The self-decoder uses hybrid sliding-window attention (SWA), while the cross-decoder uses hybrid sparse attention. The KV caches for full-attention layers in the cross-decoder are generated from the hidden states of full-attention layers in the self-decoder. At the inner level, HySparse2 retains HySparse's core KV Reuse design with two refinements. First, it replaces block-level sparsity with token-level sparsity for finer long-context retrieval. Second, it removes the separate SWA branch from sparse layers and instead forces a sliding window of recent tokens into the sparse selection. This two-level KV sharing allows all cross-decoder KV caches to be constructed from self-decoder hidden states. Prefill can therefore exit after the self-decoder, skipping all cross-decoder layers. On an 80B-A3B MoE model, HySparse2 outperforms HySparse and Hybrid SWA on long-context retrieval and multi-turn agentic tasks, while substantially reducing prefill computation and KV-cache storage. 

---
# TransBERT: A Framework for Synthetic Translation in Domain-Specific Language Modeling 

**Authors**: Julien Knafou, Luc Mottin, Anaïs Mottaz, Alexandre Flament, Patrick Ruch  

**Link**: [PDF](https://arxiv.org/pdf/2609.26347)  

**Abstract**: The scarcity of non-English language data in specialized domains significantly limits the development of effective Natural Language Processing (NLP) tools. We present TransBERT, a novel framework for pre-training language models using exclusively synthetically translated text, and introduce TransCorpus, a scalable translation toolkit. Focusing on the life sciences domain in French, our approach demonstrates that state-of-the-art performance on various downstream tasks can be achieved solely by leveraging synthetically translated data. We release the TransCorpus toolkit, the TransCorpus-bio-fr corpus (36.4GB of French life sciences text), TransBERT-bio-fr, its associated pre-trained language model and reproducible code for both pre-training and fine-tuning. Our results highlight the viability of synthetic translation in a high-resource translation direction for building high-quality NLP resources in low-resource language/domain pairs. 

---
# Blaming Across the Aisle: Political Contrasting and Blame Attribution in the Danish Parliament 

**Authors**: Markus Lundsfryd Jensen, Rune Egeskov Trust, Kenneth Christian Enevoldsen, Sara Kolding  

**Link**: [PDF](https://arxiv.org/pdf/2609.26346)  

**Abstract**: Political discourse is widely perceived to be growing more hostile, yet robust evidence remains scarce. This study examines blame attribution in the Danish Parliament from 1997 to 2026, combining a purpose-built classifier, BlameBERT (F1: 0.80), with multilevel statistical modeling. The classifier is constructed using an annotation-efficient pipeline for blame attribution in low-to-mid resource languages. The results reveal a banana-shaped trajectory, with blame declining until around 2016 before entering a significant and sustained increase in recent years (2019-2026). Government status consistently influenced blame attribution - an effect we term political contrasting - with opposition parties blaming substantially more than governing parties. This effect was moderated by ideology: The blame-dampening effect of governing was less pronounced among right-wing parties, and ideological extremity amplified blame more strongly on the right. In recent years, the interaction between political wing and ideological extremity intensified, suggesting an ideological hardening of the blame rhetoric concentrated on the right of the political spectrum. Taken together, these patterns suggest that the perceived rise in harsh political language reflects not merely a general rhetorical drift, but an ideologically asymmetric hardening of political discourse. A sensitivity analysis showed that the conclusions were robust to varying classification thresholds. 

---
# Designing and Analysing Argument Mining Pipelines: Towards a Comprehensive Assessment 

**Authors**: Siddharth Bhargava, Sara Tonelli, Patricia Martín-Rodilla  

**Link**: [PDF](https://arxiv.org/pdf/2609.26338)  

**Abstract**: Argument Mining (AM) transforms natural language into its underlying argument structures. This transformation is typically realized through a sequence of AM tasks that form an end-to-end AM pipeline. However, AM approaches often differ in how they conceptualize these tasks, making direct comparisons between them difficult and opaque. This calls for a more nuanced, task-level analysis of AM approaches to enable clearer comparison and assessment.
This work presents a preliminary meta-study that systematically reviews several state-of-the-art end-to-end AM works and analyzes their pipelines through a triple-perspective framework---a linguistic, computational and domain perspective---to understand how the pipelines model arguments as structures, computes them, and integrates domain knowledge. We further propose a general design to the linguistic and computational perspectives, illustrating how key AM tasks are designed for modeling and computation of argument structures. Our proposed framework lays the groundwork for methodology-centered descriptions across AM approaches, facilitating deeper understanding and more systematic comparisons in future research. 

---
# CHiME-9 ECHI: A Machine Learning Challenge for Enhancing Conversations to Address Hearing Impairment 

**Authors**: Robert Sutherland, Thomas Kuebert, Marko Lugger, Stefan Petrausch, Eline Borch Petersen, Juan Azcarreta Ortiz, Buye Xu, Stefan Goetze, Jon Barker  

**Link**: [PDF](https://arxiv.org/pdf/2609.26306)  

**Abstract**: This work presents the task and results of the CHiME-9 challenge for Enhancing Conversations to address Hearing Impairment. The challenge considers the scenario of four-party conversations in a noisy, cafeteria-style environment with interfering speech sources and sound effects. Participants are provided with audio recordings made with Meta Aria glasses and hearing aid microphones, and clean speech samples of the conversation participants. The task is to extract the speech of the conversation partners from the noisy multi-channel recordings with the goal of improving the intelligibility and quality of the speech, evaluated using objective metrics and subjective listening tests. This paper reviews submissions from seven teams and ranks them on a combination of subjective intelligibility and quality. Results show that while the objective metrics do not reflect listener performance, the top systems were able to make substantial improvements over the challenge baseline in both intelligibility and quality ratings. 

---
# PACE-dLLM: Elastic Block Decoding via Confidence Cliff Estimation for Diffusion Language Models 

**Authors**: Xiaocheng Lu, Shuhan Guo, Ziyue Ma, Jie Zhang, Jian Liu, Jingcai Guo, Haoxuan Che, Song Guo  

**Link**: [PDF](https://arxiv.org/pdf/2609.26249)  

**Abstract**: Diffusion language models (dLLMs), such as LLaDA and Dream, have become competitive with autoregressive (AR) LLMs in generation quality while supporting native parallel decoding. A standard acceleration strategy is block-wise decoding, where each forward pass predicts a block of length B and commits high-confidence tokens. However, B couples two distinct decisions: the look-ahead horizon and the number of tokens to commit. Existing accelerators address this limitation through indirect heuristics, such as volatility tracking, delimiter detection, and learned scoring. In contrast, we show that the required information is already encoded in the model's own per-step confidence: in-window confidence typically follows a context-dependent cliff, whose saturation point directly identifies the appropriate look-ahead horizon. We propose PACE-dLLM, which fits this parametric cliff in closed form at each step, sets the next horizon by its saturation point, and uses an independent confidence threshold for token commitment. Under a saturated-yield abstraction, we show that the cliff-anchored horizon is the smallest horizon attaining maximal useful per-pass yield: fixed horizons that undershoot it incur a worse asymptotic NFE rate, while overshooting adds no useful yield. On four reasoning and code benchmarks, PACE-dLLM achieves the best average accuracy on both open-source dLLM backbones, with average wall-clock speedups of 5.23x on LLaDA and 3.06x on Dream (up to 8.52x on math) over the unaccelerated semi-AR baseline, advancing the quality-throughput Pareto frontier. 

---
# Damage Predicts Recovery: When Calibration Data Matters in Compressing Financial LLMs 

**Authors**: Junyi Ye, Mengjia Yu, Debapriya Hazra, Guiling Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.26241)  

**Abstract**: Post-training quantization and pruning rely on a small calibration corpus. Whether specialized domains such as finance require domain-matched calibration data remains unsettled. We argue that the answer depends on the task-level damage caused by compression rather than on domain mismatch. If compression preserves the target capability, changing the calibration corpus has little effect. If compression causes large losses, task-formatted calibration can recover part of the loss. We test this hypothesis across two model families, six compression configurations, three token-matched calibration corpora, and ten financial classification and numerical question-answering tasks. The results support this hypothesis. Quantization largely preserves task performance, and calibration choice has little effect in this case. Pruning reduces numerical QA accuracy by over 40 points. In these damaged settings, another generic corpus does not help, while FinMix, a mixture of financial task examples, recovers a large part of the loss. The link between damage and recovery holds across model families and scales. These findings support a practical rule. Measure task-specific compression damage first, and construct specialized calibration data only when the damage is large. 

---
# Same Chart, Different Story: Bias in Vision-Language Chart Interpretation 

**Authors**: Mizanur Rahman, Huan Wu, Arash Asgari, Enamul Hoque Prince, Laleh Seyyed-Kalantari  

**Link**: [PDF](https://arxiv.org/pdf/2609.26210)  

**Abstract**: Vision-language models (VLMs) are increasingly used to interpret charts and generate natural-language explanations for socially consequential data. However, they may produce different narratives for the same chart when only the referenced social group changes, reinforcing stereotypes and misleading decisions. Despite these risks, no benchmark exists for systematically evaluating bias in chart interpretation across social dimensions. We introduce ChartBias, the first benchmark for auditing bias in VLM-based chart interpretation. ChartBias contains 820 manually curated real-world charts spanning six attributes: race, income, age, religion, immigration status, and gender, yielding 4,319 valid chart, attribute instances and 8,638 paired generations where the chart is fixed and only the group term is swapped. Across 12 proprietary and open-source VLMs, totaling 155,484 model responses, we find three widespread failure modes: narrative shift (same chart, different narratives), group hallucination (assigning a chart to a group without evidence), and preference polarity (favourable trends often linked to one group). We further propose a multi-agent mitigation framework that serves as a strong baseline by separating chart-grounded evidence extraction from group-conditioned generation and using a counterfactual judge to verify that group-driven differences are supported by the chart. The framework substantially reduces narrative shift while preserving chart-grounded reasoning. Our findings show that evaluating chart understanding requires measuring not only accuracy, but also fairness and consistency across social groups. We release ChartBias at this https URL. 

---
# Beyond Static Charts: Can Language and Vision Language Models Generate Interactive Data Visualization Interfaces? 

**Authors**: Mizanur Rahman, Aaryaman Kartha, Enamul Hoque Prince  

**Link**: [PDF](https://arxiv.org/pdf/2609.26208)  

**Abstract**: Data visualization is central to analytical reasoning, but real-world analysis increasingly requires language-driven interactive interfaces rather than static charts. Although recent large language and vision language models (LLMs/VLMs) have shown promise in generating static charts from natural language, their ability to generate interactive data visualization interfaces remains largely unexplored due to the lack of benchmarks. We introduce VIS-GEN, a benchmark for evaluating how well LLMs/VLMs can generate interactive visualization interfaces from natural language queries. VIS-GEN comprises 3,042 samples covering diverse analytical intents, including data filtering, temporal analysis, and visualization editing, each paired with dataset metadata and natural language queries that are designed to reflect realistic, goal driven data exploration scenarios. We benchmark 14 state-of-the-art open-source and closed-source LLMs/VLMs, revealing large performance gaps and frequent failures on queries involving implicit intent, multiple interaction alternatives, and complex editing operations, highlighting interactive interface generation as a key open challenge beyond static chart synthesis. To address this, we propose a structured multi stage interface generation framework that decomposes the task into visualization design representation, generation of multiple interface candidates, constraint-aware critique, and self-refinement. This approach improves the best models pass rate by 15.9 percentage points, demonstrating a practical path toward more reliable language-driven interactive visualization systems. We release VIS-GEN at this https URL. 

---
# Modality-Gated Deep Adapters: Adding a Modality to a Frozen Embedding Model with Exact Preservation 

**Authors**: Abdul Basit Tonmoy, Kazi Fardinul Hoque, Md. Shahrier Islam Arham, Arman Luthra  

**Link**: [PDF](https://arxiv.org/pdf/2609.26182)  

**Abstract**: Multimodal embedding models are deployed at scale: retrieval indices, benchmark results, and behavioral audits all depend on the base model's exact outputs. Extending such a model to a new modality with existing parameter-efficient methods silently changes those outputs; LoRA-style adaptation rewrites the text path whether or not the weights are merged, invalidating every stored embedding. We propose modality-gated deep adapters: bottleneck adapters attached to every decoder layer of a frozen multimodal embedding LLM, grouped into per-modality packs that execute only while their own modality is being encoded. The result is a modality added with zero change to existing outputs: inputs no pack claims traverse the base model's own computation graph, bit-for-bit unchanged, and co-loaded packs compose with an exact-zero isolation matrix. Both properties are stated as propositions, hold after arbitrary training rather than only at initialization, require no task labels or routing metadata at inference, and are verified by exact-equality tests on the released checkpoints. On one frozen 2B base, the audio pack (injected as connector tokens) improves audio-to-text R@10 by +3.4 to +5.4 points over an identically trained control, positive at every seed and reproduced at eleven times the data; the thermal pack, reusing the base's own frozen vision path, clears its pre-registered acceptance gate roughly sevenfold at every seed and lifts thermal-to-text R@10 from 0.224 to 0.785. An encoder swap locates the missing capacity: an external audio encoder that outranks Whisper-family encoders in CLAP-style comparisons loses by 16 R@10 points inside the frozen LLM, so the capacity belongs in the layers, exactly where the gated adapters place it. We release the audio model, the thermal pack, and the training, evaluation and invariance suites: models at this http URL, code on GitHub. 

---
# Magnitude Profile Pruning: Calibration-Free Structured Attention Head Removal for Transformer Compression 

**Authors**: Kasun Dewage, Marianna Pensky, Heranga K. Rathnasekara, Suranadi De Silva  

**Link**: [PDF](https://arxiv.org/pdf/2609.26177)  

**Abstract**: Structured pruning of attention heads provides a hardware-friendly way to compress Transformer language models. However, existing methods for measuring head-level importance require calibration data, gradient computation, or Hessian estimation. These requirements add extra overhead and make the methods depend on the data. Our work presents Magnitude Profile (MP) scoring, a training-free criterion for head importance that identifies dispensable heads through statistical outlier detection on weight row norms. Heads whose projection weights fall within the population bulk are pruned, while heads exhibiting outlier norms, which carry disproportionate representational capacity, are preserved. Our work further gives MP-G, a variant that handles Grouped Query Attention (GQA) by distributing shared key-value group scores across associated query heads. Across five models evaluated on WikiText-2 perplexity at 12.5%-50% head sparsity, MP-G achieves the best perplexity on OPT-6.7B at all sparsity levels (18.46 at 12.5%, 27.87 at 25%, 152.0 at 50%). MP-G also gives the best results on RoBERTa-large at 12.5% and 25% sparsity, with perplexity values of 7.27 and 10.28, outperforming calibration-dependent baselines including Wanda-Head, SparseGPT-Head, and Gradient-Head. It requires zero forward passes, calibration samples, or gradient computation. At 50% sparsity, head pruning yields up to 16% parameter reduction with 50% attention FLOP savings. Our results show that weight-only statistical scoring can match or outperform data-dependent methods for structured head pruning, providing a practical, zero-cost criterion for Transformer compression. 

---
# Differentiable Fuzzy Inference Layer: A Monotone, Compositional Ordinal Reasoning Head for Large Language Models 

**Authors**: Zhen Zhang, Amr Alanwar  

**Link**: [PDF](https://arxiv.org/pdf/2609.26113)  

**Abstract**: A state-of-the-art language model asked to interpret "most of most students passed" typically answers "most," though composing two instances of "most" yields a proportion closer to "some." We trace this failure to an architectural choice rather than a data deficit: standard classifier heads treat ordinal categories as independent labels, with no mechanism to respect their natural ordering or compose them algebraically. We introduce the Differentiable Fuzzy Inference Layer (DFIL), a dual-path prediction head pairing a standard classifier with a scalar-bottlenecked branch grounded in a bank of ordered membership functions. DFIL supplies two structural primitives that a label-only head cannot inherit: monotonicity in the underlying quantity, and compositional reasoning via t-norm operations without any compositional training data. The scalar branch additionally provides an interpretable interface for analyzing residual errors. We instantiate DFIL on ordinal natural-language tasks across diverse LLM families. 

---
# TSS: Target-Side Sparsification for Speculative Decoding in Domain-Specific Large Language Models 

**Authors**: Haibo Hu, Lianming Huang, Qiao Li, Nan Guan, Chun Jason Xue  

**Link**: [PDF](https://arxiv.org/pdf/2609.26100)  

**Abstract**: Speculative decoding accelerates large language model inference through collaboration between a lightweight draft model and a target verifier. Existing methods mainly improve the draft side, while the target model is typically kept dense and unchanged. We show that, under domain-specific inference, full-depth target verification is not always the optimal choice. Counter-intuitively, skipping selected target layers can reduce verification cost while simultaneously increasing draft acceptance and preserving, or even improving, downstream task performance. Based on this observation, we propose TSS, a target-side sparsification framework for speculative decoding. TSS employs an acceptance- and metric-aware breadth search to explore multi-layer skip configurations without imposing a fixed priority between the two objectives. The selected configurations are stored in a domain-to-configuration mapping and applied by a lightweight skip controller, allowing one complete target model to support multiple sparse verification paths without retraining or permanent parameter pruning. Experiments on Spec-Bench across multiple domains, model scales, and speculative decoding methods show consistent improvements in draft acceptance and downstream task performance. In Translation setting, TSS increases the average accept length from 2.70 to 4.53 (+67.8%), improves BLEU from 0.131 to 0.237 (+80.9%), and raises end-to-end throughput from 75.6 to 127.3 tokens/s, corresponding to a 1.68X speedup. 

---
# One Domain, Many Tongues: Composing Domain and Language LoRAs for Cross-Lingual Remote-Sensing MLLMs without Paired Data 

**Authors**: Xuechen Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.26097)  

**Abstract**: Remote-sensing (RS) multimodal large language models (MLLMs) are trained and evaluated only in English, while text-only instruction data covers over 100 languages. We propose MODL (Mutually Orthogonal Domain-Language composition), a recipe that adds new languages to an English RS MLLM without a single multilingual RS example: a domain LoRA trained on English RS imagery and a language LoRA trained on text alone are learned jointly, under one loss term that keeps the two updates mutually orthogonal at every layer throughout training. This constraint is the recipe's active ingredient. Without it, the same training answers RS questions correctly but in English, erases much of the base model's multilingual text ability, and diverges on one seed in three; sixteen alternatives, from training-free merging to prior orthogonality variants, fail the same way. MODL repairs every failure on every seed: answers are correct and in the target language 56-71% of the time, where the best alternative reaches 27% and most stay below 8%, text ability stays at the level of the untrained base, and on Spanish it surpasses Qwen2.5-VL-7B, with zero multilingual-multimodal data. A single five-language adapter retains English, Spanish, and Vietnamese at full strength across three seeds; non-Latin scripts remain an open boundary. 

---
# SpecialEduBench: Benchmarking Vision-Language Models on Knowledge, Skill, and Attitude in Language Intervention for Autistic Children 

**Authors**: Jihoi Na, Taeyeong Kim, Sungjune Kong, Jaemin Jung, Min Joung Park, Kyungtae Joo, Ahhyun Kim, Shim Jaechang, Sooyoung Joo, Dongjin Ka, SeJoong Kim, Jimin Kim, HyunJin Jung, Unggi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2609.26090)  

**Abstract**: Language is the target of most early intervention for autistic children. Because the goal and the method change from child to child, the work falls to a teacher who takes one child at a time and judges each scene as it unfolds. Artificial intelligence is now being brought to that work, yet the benchmarks that reach special education ask what a model knows rather than what it does in front of a child. Building one is not straightforward, since whether a response is good teaching depends on what the child has just done, so no answer key applies. The evidence that settles it is visual as much as verbal, since the length of a wait, a shift of gaze, and the child's uptake leave no trace in a transcript. We introduce \emph{SpecialEduBench}, which measures pedagogical competence along knowledge, skill, and attitude, with 4,537 knowledge items and with 200 skill items and 68 attitude items built on recorded intervention, the attitude items crossing pressure with monitoring into 192 response cells. Seven special-education experts wrote, scored, and reviewed the items, and we revised the judge model's instruction against the reference scores they set. Across eight frontier vision-language models no axis is saturated, since the strongest still fails about a tenth of the honesty cells. The models converge where the knowledge is factual and separate where the task is situated, and the failures gather where pressure is applied. We intend the benchmark as an audit to run before deployment and as a starting point for models built for this domain. 

---
# CoVeR: Coverage-Based Routing of Verifier Calls in Agentic Retrieval 

**Authors**: Daeyoung Roh, Donghee Han  

**Link**: [PDF](https://arxiv.org/pdf/2609.26086)  

**Abstract**: An agentic retrieval system issues a sequence of search queries and must decide, at each step, whether the evidence collected so far is enough to stop. Delegating that decision to an LLM verifier or a prompt judge makes stopping reliable, but the verifier then reprocesses the growing evidence after every retrieval step, a substantial repeated cost. We show that most of these calls can be skipped without materially changing answer accuracy: a single threshold on a frozen sentence-embedding coverage margin detects the states in which the evidence is still plainly incomplete, and the verifier is called only on the ambiguous remainder, a gate we call CoVeR (Coverage-based Verifier Routing). Across three multi-hop QA benchmarks, with the evaluation protocol fixed before the full-scale run, the CoVeR-gated agent matches the answer accuracy of both the full-budget agent and the always-verify baseline within a fraction of an EM point. It cuts 62-68% of verifier calls, and 93% in a saturated regime. Routers built on evidence counts, lexical overlap, or BM25 relevance, alone or learned in combination, give weaker overall trade-offs, the gate transfers without re-tuning across deciders and agent scales, and its drafter distills into a 921k-parameter head atop the frozen encoder, leaving no LLM in the routing loop. The same signal cannot replace verification: matching a claim is far easier than deciding the claim is supported. 

---
# Optimizing Denoising Trajectories in dLLMs: A Lightweight Evolutionary Heuristic Approach 

**Authors**: Zijian Zhao, Dian Jin, Xialiang Tong, Sen Li, Mingxuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2609.26052)  

**Abstract**: Diffusion Large Language Models (dLLMs) have recently emerged as a promising alternative to conventional Auto-Regressive (AR) Large Language Models (LLMs). By leveraging bidirectional attention and parallel decoding, dLLMs enable more efficient generation. However, they require a carefully designed denoising scheduler at inference time (absent during training) whose choice significantly impacts generation quality. While confidence-based heuristic schedulers have shown strong empirical performance, they suffer from two critical failure modes: EOS Overflow and Proximal Bias. Through in-depth analysis of the Transformer's attention patterns, we reveal that these failures stem from certain positions assigning disproportionately high attention weights to invalid tokens (e.g., [MASK] and [EOS]), which produce misleading confidence signals. Building on this insight, empirical evidence shows that valid attention scores can provide complementary guidance to conventional confidence-based heuristics, yet no single metric consistently excels across all scenarios, implying that the optimal denoising trajectory is highly context-dependent. To address this problem, we propose a lightweight evolutionary heuristic scheduler optimized using the Covariance Matrix Adaptation Evolution Strategy (CMA-ES). Our scheduler dynamically integrates multiple heuristic features with a contextual mean-field embedding, while requiring only 393 trainable parameters. Evaluated on LLaDA and Dream across four reasoning and planning benchmarks, our method consistently outperforms strong baselines, including conventional heuristics, block auto-regressive methods, and recent State-Of-The-Art (SOTA) approaches. To the best of our knowledge, it represents the most parameter-efficient neural scheduler to date. Our code is available at this https URL . 

---
# Truth for Believable AI: Expressed Doubt, Provenance, and Belief Revision as an Engineerable Stance 

**Authors**: Sebastian Cochinescu  

**Link**: [PDF](https://arxiv.org/pdf/2609.26035)  

**Abstract**: Conversational agents often express answers in a uniformly confident register. We test whether expressed uncertainty, provenance-aware assertion, and explicit belief revision can be implemented as a behavior layer over a fixed language model; we do not test believability or trust. The layer combines three epistemic states, per-claim confidence and typed provenance, a provenance-gated expression rule, and a persistent revision store with auditable acknowledgments and partial resistance to false corrections. We evaluate it on a constructed, mechanically scored multi-session benchmark using a synthetic model and Qwen2.5-0.5B-Instruct. The synthetic instrument passes all five checks. On the real model, acknowledgment soundness, a by-construction guarantee, holds in 100% of cases, and true corrections are accepted more often than false ones (0.44 vs. 0.15 on held beliefs; 0.875 vs. 0.420 including rule-accepted corrections of unheld facts), but the pre-specified expression-fidelity, contradiction-separation, and provenance margins fail. A disclosed post hoc analysis shows that expression gated on mean answer-token probability ranks correctness below chance end to end (AUC 0.41, conversation-clustered), whereas gating on sampling consistency discriminates (AUC 0.66). A consistency-gated configuration selected from this finding and evaluated under a separately committed protocol meets the conversation-level manipulation and capability-equivalence criteria and replicates on a redrawn conversation set. The manipulation result is selection-dependent, and both criteria remain unresolved when uncertainty is clustered over the 60 facts. The supported conclusions are limited to the by-construction audit guarantee, store-dependent partial correction discrimination, and a benchmark- and model-specific failure of token-probability gating; scaling the fact base is required before human evaluation. 

---
# Domain-Adaptive Pretraining Enhances Water Treatment Semantic Representation for Large-Scale Structured Literature Mining 

**Authors**: Mudi Zhai, Ruihong Qiu, Qingyun Zeng, T. David Waite, Bing-Jie Ni, Haoran Duan  

**Link**: [PDF](https://arxiv.org/pdf/2609.26034)  

**Abstract**: Water treatment research is expanding rapidly, but much of the knowledge acquired from this research remains scattered across unstructured literature. The field still lacks a dedicated language model that can efficiently capture water treatment-specific domain semantics for large-scale literature mining. Here, we address this by developing WaterBERT, a domain-adapted encoder model designed for semantic representation and structured information extraction from water treatment texts. WaterBERT was developed by continual pretraining on a large-scale water treatment corpus comprising about 2.97 billion tokens. Three fine-tuned models based on WaterBERT were systematically evaluated on downstream tasks, achieving the best overall performance among general-purpose and domain-specific BERT models, with F1 scores of 90.12% for multiclass treatment process classification, 79.50% for named entity recognition, and 74.04% for relation extraction. Beyond these benchmark tasks, we further demonstrated WaterBERT's advantages for large-scale literature processing. Applied to 5,144 Environmental Science & Technology articles, WaterBERT-BERTopic identified coherent, diverse, and domain-specific research topics without predefined categories. Building on WaterBERT, we processed 693,211 abstracts at substantially lower cost than commercial LLMs while retaining competitive extraction performance to construct a structured water treatment knowledge graph. The knowledge graph was then integrated with lexical and dense retrieval to develop a Water Knowledge-Enhanced Retrieval System (WaterKERS), which achieved a relevance score of 77.7, substantially outperforming text-based retrieval baselines (54.7-64.5). Through WaterBERT, this study provides a compact and scalable semantic foundation for large-scale information processing and evidence mapping in water treatment research. 

---
# ClusterFewshot: Improving Few-shot Optimization for LLMs workflow 

**Authors**: Omri Bar Haim, Shahar Katz, Lior Wolf  

**Link**: [PDF](https://arxiv.org/pdf/2609.25939)  

**Abstract**: The performance of large language model (LLM) workflows often depends on selecting a small set of in-context demonstrations to guide model behavior on new tasks. Recent methods improve this process by augmenting prompts with successful reasoning paths. However, their demonstration selection relies on random sampling or metric-based rankings, overlooking the semantic structure of the task. We propose ClusterFewshot, a strategy that combines semantic structuring with utility-aware scoring to construct representative and effective few-shot demonstration sets. Evaluated within DSPy-based pipelines, ClusterFewshot substantially reduces optimization cost across multiple benchmarks, while consistently improving accuracy relative to prior bootstrap-based methods in both standalone prompt tuning and hybrid prompt-weight optimization. 

---
# Informed Masking: Structure-Aware Perturbation for Reinforcement Learning in Diffusion Large Language Models 

**Authors**: Xiaoyi Yu, Enver Sangineto, Pei Fu, Fiorenzo Parascandolo, Wenhui Tan, Ruikang Zhang, Rita Cucchiara, Ruihua Song, Jian Luan  

**Link**: [PDF](https://arxiv.org/pdf/2609.25927)  

**Abstract**: Diffusion Large Language Models (dLLMs) have emerged as an efficient alternative to autoregressive models, yet aligning them via Reinforcement Learning (RL) requires likelihood surrogates estimated from masked reconstruction subproblems under a small Monte Carlo budget per rollout. Existing methods construct these subproblems by uniform random masking, leaving open the question of which subproblems to prioritize. We identify a systematic upstream/downstream structure in dLLM rollouts. Some tokens, when revealed, trigger large confidence changes in nearby undecoded positions; we call them upstream. Others induce only small local changes and are therefore downstream. We find masking downstream tokens yields substantially better-posed subproblems than masking upstream tokens, a phenomenon we term subproblem difficulty asymmetry. Based on the observation, we propose Informed Masking (IM), which derives a per-token priority score from the denoising trajectory at zero extra inference cost and biases mask sampling toward downstream tokens. IM is plug-and-play: when plugged into three state-of-the-art dLLM RL methods on LLaDA-8B-Instruct, it delivers up to 2.01%, 8.68%, and 5.77% relative average gains on math and planning benchmarks with improved training stability. 

---
# Rethinking Length-Based Training: Batch Composition and Loss Normalization in Speech Token Language Models 

**Authors**: Hongjin Song, Runwu Shi, Weiqiao Shan, Jiale Luo, Yujin Wang, Yifei Wu, Chunxiang Jin  

**Link**: [PDF](https://arxiv.org/pdf/2609.25890)  

**Abstract**: Short-to-long training is a simple curriculum for speech models, but its gains can be difficult to interpret. In speech token language models, length-based training can change the shuffle policy, batch composition, token retention, and token weights under batch-mean loss. We disentangle these factors through matched comparisons. In the tested settings, short-to-long ordering shows no independent benefit when batch composition and token exposure are fixed. First-epoch grouping lowers perplexity for Mimi under batch-mean loss, but this gain is not observed under token-balanced loss. The cross-tokenizer results are consistent with a link between chunk-length variation and token weighting. This work provides a systematic analysis protocol for studying length-based training in variable-length speech models. 

---
# Isolated Sign Language Recognition for Icelandic Sign Language: Experiments in a Low-resource Setting 

**Authors**: Finnur Ágúst Ingimundarson, Guðný Björk Þorvaldsdóttir, Mathias Müller, Sarah Ebling  

**Link**: [PDF](https://arxiv.org/pdf/2609.25862)  

**Abstract**: We present the first experiments on isolated sign language recognition (ISLR) for Icelandic Sign Language (ÍTM). We use ÍTM SignWiki, a dataset derived from a bilingual Icelandic--ÍTM online dictionary. It is genuinely low-resource: 1,845 videos cover 849 classes, 86% of which have only two examples, making the full task effectively one-shot recognition across signers. We compare two open-source ISLR frameworks, OpenHands and SPOTER, on three tasks of increasing vocabulary size (22, 117 and 849 classes), and evaluate three pose estimators and two forms of cross-lingual transfer. With ÍTM data alone, SPOTER outperforms OpenHands on all three tasks, and MediaPipe poses give better results than AlphaPose or SDPose. Cross-lingual transfer brings the largest gains: pretraining SPOTER on American Sign Language data before finetuning on ÍTM raises accuracy by 14--24 percentage points, to 72.7%, 47.9% and 22.6% on the three tasks, and multilingual training with data from six other sign languages lifts OpenHands from 1.41% to 28.86% on the full task. Although far from practical use, the results suggest that transfer from better-resourced sign languages is promising for very low-resource ones. We release our adapted versions of both frameworks. 

---
# BELXTR: Biomedical Entity Linking via Contextualized Token Retrieval 

**Authors**: Samuele Garda, Ulf Leser  

**Link**: [PDF](https://arxiv.org/pdf/2609.25859)  

**Abstract**: Biomedical Entity Linking disambiguates mentions to entities in a knowledge base (KB), making it the cornerstone of information extraction pipelines. While embedding-based models are a popular approach for the task, they suffer from a key limitation. They compress mentions (and entities) into a single vector, forcing the model to average away crucial fine-grained differences. We present BELXTR, a novel embedding model based on the multi-vector (a.k.a. late interaction) architecture, which allows to leverage token-level matching information. BELXTR extends the original XTR model to biomedical entity linking by integrating an existing task-specific training objective and exploring active query expansion. Experiments across ten corpora and five KBs show that BELXTR improves upon current state-of-the-art in half of the corpora with an average improvement of 5pp recall@1. The largest gains are reported on the challenging cross-species gene disambiguation subtask, where BELXTR outperforms an LLM-powered retrieve-and-rerank pipeline and closely approaches a specialized rule-based system. Our results highlight multi-vector models as a practical alternative to hard-to-maintain rule-based systems or in scenarios where LLM-based reranking is too costly as in PubMed-scale mining. The code to reproduce our experiments can be found at: this https URL. 

---
# MemoryAthena: Adaptive Routing over Latent and Generated Memories 

**Authors**: Mingyuan Li, Guangsheng Yu, Juyuan Zhang, Xu Wang, Zhibo Man, Haonan Zhang, Shaoxiong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2609.25853)  

**Abstract**: Learned-memory methods store information in an explicit table and consume it through a separate reader, allowing addressing, storage, and reading to be modified independently. We study whether useful memory can also be generated rather than only retrieved. MemoryAthena uses three pathways: direct Engram retrieval (E), generation from retrieved Engram cues (GE), and generation from causal backbone states without consulting the memory table (GH). Generated memory is conditionally useful: it can complement E in one context but interfere with it in another. MemoryAthena therefore treats E as an anchor and learns when a generated representation should intervene. With the backbone, memory, generators, and readers frozen, a lightweight causal routing head is trained from counterfactual future-token likelihood advantages of GE and GH relative to E. At inference time, an admitted candidate modifies the E residual through bounded interpolation, while rejection recovers the direct pathway exactly. On question answering, MemoryAthena raises the five-task average from 37.65 to 39.28 over the direct pathway of the same checkpoint, while the six-task general-NLP average increases from 76.73 to 79.13. The complete memory-side system contains approximately 201M parameters, excluding the frozen backbone. Further analyses show complementary strengths among E, GE, and GH across tasks and inputs. These results support generated memory as a selective correction to direct retrieval and highlight routing when, which, and how strongly to intervene as the central challenge. 

---
# ARAFA: An LLM-Generated Arabic Fact-Checking Dataset 

**Authors**: Christophe Khalil, Shady Elbassuoni, Rida Assaf  

**Link**: [PDF](https://arxiv.org/pdf/2609.25833)  

**Abstract**: Automatic fact-checking poses a significant challenge in Arabic natural language processing due to the scarcity of datasets and resources. In this manuscript, we introduce Arafa, a new large-scale dataset for fact-checking in Modern Standard Arabic, constructed through an automated framework leveraging large language models (LLMs). The dataset was constructed through a three-step pipeline: (1) claim generation from Arabic Wikipedia pages with supporting textual evidence, (2) claim mutation to generate challenging counterfactual claims with refuting evidence, and (3) an automatic validation step to validate that the generated claims are either supported or refuted by their accompanying evidence, or if the evidence does not provide enough information to judge the validity of the claims. The resulting dataset comprises 181,976 claim-evidence pairs labeled as supported, refuted, or not enough information. Human evaluation carried out on a test sample from the dataset demonstrated strong inter-annotator agreement (kappa = 0.89) using Cohen's Kappa for supported claims and (kappa = 0.94) for refuted claims. Automatic validation based on a human-evaluated sample achieved 86% accuracy for supported claims and 88% for refuted ones. To showcase Arafa's value as a resource for automatic Arabic fact-checking, four open-source transformer-based models were fine-tuned using Arafa, with the top-performing model achieving a Macro F1-score of 77% on the test data. In addition to Arafa being the first large-scale dataset for Arabic fact-checking, our framework presents a scalable approach for developing similar resources for other low-resource languages. 

---
# Reply to comments arXiv:2512.07881 and arXiv:2601.06104 on quantum structure in human and AI-generated language 

**Authors**: Massimiliano Sassoli de Bianchi, Roberto Leporini  

**Link**: [PDF](https://arxiv.org/pdf/2609.25797)  

**Abstract**: We reply to the comments by M. Sienicki and K. Sienicki (arXiv:2512.07881) and by K. Sienicki (arXiv:2601.06104) on our work on quantum-mechanical statistics in human language (arXiv:2407.14924) and on quantum structure in AI-generated language (arXiv:2511.21731). We thank the authors for their careful reading and address what we consider to be the main points of criticism: the exploratory nature of the protocol used in the experiments with large language models; the role of marginal-law violations, and of the Contextuality-by-Default criterion, in the identification of entanglement; the limited diagnostic value of a Bose-Einstein fit taken in isolation; the meaning of assigning the lowest energy levels to the most frequent words; and the relation between the vector spaces used by LLMs and quantum state spaces. We also correct a typographical error in Table 3 of arXiv:2511.21731, which does not affect the reported CHSH value. 

---
# Syndrome, Synergy, and Safety: Structured Reasoning and Knowledge-Driven Alignment for TCM Prescription Generation 

**Authors**: Zheng Chen, ZhiCheng Du, Haoxuan Li, Peiwu Qin  

**Link**: [PDF](https://arxiv.org/pdf/2609.25755)  

**Abstract**: Applying large language models to Traditional Chinese Medicine (TCM) prescription generation reveals three clinically critical gaps: models produce end-to-end mappings without auditable reasoning following the li-fa-fang-yao paradigm (SR Gap), treat each encounter in isolation without follow-up adjustment via sui zheng jia jian (LA Gap), and fail to enforce absolute contraindication rules such as Shi Ba Fan (SC Gap). We propose a progressive four-stage framework (SFT $\to$ PG-CoT $\to$ Dynamic $\to$ K-RL) that addresses each gap: PG-CoT constrains CoT distillation under the li-fa-fang-yao paradigm to produce auditable diagnostic chains, Dynamic SFT models patient trajectories with explicit transition reasoning, and K-RL encodes deterministic pharmacological rules as rule-based DPO preference signals. Across 12 fine-tuned models and 6 zero-shot baselines, our framework substantially improves prescription quality over zero-shot baselines---with a 7B model (Mistral-7B) surpassing zero-shot GPT-5 on all three TCM evaluation metrics. 

---
# From Utterances to Networks: Modelling Slang Adoption and Diffusion Across Subreddits 

**Authors**: Xiaoning Wang, Ted Underwood, Zhewei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2609.25669)  

**Abstract**: Adoption and diffusion of neologisms in online communities have received renewed attention in recent years. As internet slang terms such as APT, referring to a K-pop song, and phrases such as Canon Event meaning an embarrassing but pivotal event, go viral online, it becomes increasingly important to understand the mechanisms that contribute to their success. Prior studies have often explained slang diffusion either from the perspective of social interaction or from the linguistic properties of the slang itself, but rarely from both perspectives together. One major obstacle has been the high cost of annotating slang usage in large-scale online communication. Recent advances in large language models (LLMs), however, make it possible to use them as scalable annotators for such tasks. In this study, we first curate a human-annotated benchmark to evaluate LLM performance in detecting slang usage in real Reddit communication. We then leverage LLM-based annotations to model slang adoption and diffusion. Our results show that slang diffusers with higher bridging capital are associated with increased subsequent adoption, whereas diffusers with higher bonding capital are associated with reduced adoption. We also find that wider contextual usage of a slang term is associated with a longer time before new users officially adopt it. Together, these findings suggest that both social-network structure and linguistic context shape the diffusion of neologisms in online communities. 

---
# Qwen3.8-Omni: Towards Native Omni-Modal Agents 

**Authors**: Qwen Team  

**Link**: [PDF](https://arxiv.org/pdf/2609.25611)  

**Abstract**: We introduce Qwen3.8-Omni-Flash, a natively multimodal agentic model for real-world multimodal productivity. Compared with previous omni models, which primarily emphasized perception and interaction, Qwen3.8-Omni-Flash substantially improves multimodal understanding and reasoning, as well as performance on long-horizon agentic tasks. These capabilities are supported by a native multimodal co-training strategy that preserves strong text-domain capabilities while facilitating the transfer of agentic capabilities from text to audio and video tasks. The model inherits the sparse mixture-of-experts (MoE) architecture of Qwen3.8-Next and extends the context window to one million tokens, supporting long-context multimodal reasoning and long-horizon planning. These advances enable integration into production workflows as a primary agent or a specialized sub-agent, supporting video editing, long-form audio and video translation, music-conditioned music video or movie generation, and video-based note or omni-skill creation. To address the lack of native audio and video support in existing agent harnesses, we release Qwen-MM-Plugins, a lightweight open-source plugin framework for multimodal productivity. We further frame real-time multimodal interaction as a system-level challenge requiring orchestration of context and memory management, tool use, and sub-agent delegation. Accordingly, we release Qwen-Live-Harness, an open-source framework for building responsive, real-time multimodal agents based on Qwen3.8-Omni-Flash. Extensive evaluations demonstrate that Qwen3.8-Omni-Flash achieves strong performance across multimodal understanding, reasoning, long-horizon agentic execution, and video productivity tasks. These results and the accompanying open-source tools support Qwen3.8-Omni-Flash as a practical foundation for deploying natively multimodal agents in research and production. 

---
# Compressing Long Context into Answer-Aligned Memory Embeddings for LLM Inference 

**Authors**: Md Mostafizer Rahman, Md Faizul Ibne Amin, Md Shahajada Mia, Yutaka Watanobe, Fang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.25537)  

**Abstract**: Large language model (LLM) inference is constrained by the quadratic scaling of self-attention and the linear scaling of the KV cache, increasing latency, energy consumption, and GPU memory demand as context length scales. Existing soft-compression methods either lack query-guided memory selection at inference time, train without answer-targeted supervision, or couple compression tightly to a specific decoder architecture. We propose a Context-to-Answer-Aligned Memory Compression (CMC) framework, which compresses long input contexts into compact Context Memory Embeddings (CMEs) aligned to any frozen decoder's embedding space, reducing inference costs without modifying decoder weights. CMC introduces a two-tier KV cache that combines question-guided CME selection with a local context window, and trains the compressor with answer-targeted distillation from a frozen LLM. Experiments across nine encoder-decoder combinations and four QA benchmarks show that CMC consistently outperforms the baseline, achieving up to 7.3 EM and 4.0 F1 point gains on SQuAD, while reducing inference time and energy consumption by up to 20% and peak reserved GPU memory by up to 50% at 3,000 generation tokens. Ablation studies confirm that each architectural component and training objective contributes to the performance. 

---
# Matryoshka attribution: Learning to attribute language model outputs to representations and weights 

**Authors**: Aryaman Arora, Kirill Acharya, Nathan Hu, Yanzhe Zhang, Noah Goodman, Dan Jurafsky, Christopher Potts  

**Link**: [PDF](https://arxiv.org/pdf/2609.25518)  

**Abstract**: Attributing language model outputs to their internal computations is an open problem in interpretability. Existing methods, which use causal interventions, gradients, or learnable masks, either are infeasibly expensive or struggle to identify actual causally-important internal computations. We propose framing attribution as the problem of identifying nested subsets of internal components which minimise a downstream loss. To learn this task, we introduce Matryoshka Attribution (MAttr), a mask learning method that parametrises the mask with a simple differentiable sigmoid top-$k$ operator. We supervise training over all sparsities simultaneously by randomising $k$ over training, resulting in a learned ordering of components by attribution score. MAttr achieves number 1 on the official leaderboard of the Mechanistic Interpretability Benchmark (Mueller et al., 2025); our method identifies sparse and task-transferrable circuits across varying circuit bases. As a practical application, we show that MAttr can be trained with reinforcement learning to identify weight changes responsible for downstream behaviours in LLM finetuning. We train MAttr on refusal judge scores and find that restoring $1\%$ of Llama 3.1 8B Instruct's weights to their base model state is sufficient to remove refusals while maintaining capabilities. We view MAttr as a successful formulation of interpretability into a learnable objective that we can tackle with gradient descent, and encourage future work along these lines. 

---
# Conduct Under Pressure: What Sixty Language Models Do When a User Pushes 

**Authors**: Tapan Parikh  

**Link**: [PDF](https://arxiv.org/pdf/2609.25447)  

**Abstract**: We study what LLMs do when a user applies pressure in an uncomfortable situation: a user insists, begs, flatters or grieves, and the model gives up a correct fact, writes a document it should refuse, or cheers a plan that will cost the user money. We send frozen multi-turn scenes, identical for every model regardless of the reply, to 60 models from 13 vendors, and label each transcript with a codebook built by open coding and then frozen: a trajectory (the model held its position or folded) and a manner (how it held or folded). Two findings separate. Whether a model holds tracks its generation, meaning how recent it is: fold rate correlates with a public capability index at Spearman -0.64, with little vendor effect. How it holds tracks the vendor: six of the 17 manner codes sort by vendor at permutation p <= 0.001, corrected across the codebook. We report four vendor profiles on the codes that cleared reliability.
We also ask which parts of the labeling need a person. Six LLM coders from three vendors apply the codebook more consistently than three human coders do (Krippendorff's alpha 0.66 against 0.46), agree with the codebook's author on trajectory at kappa 0.84 to 0.91 on transcripts the codebook's examples never touched, and match an adjudicated human reference at 0.83. Blind machine readings recover the codebook's categories but cannot tell which of them a second reader would apply the same way. We conclude that for behavior a non-specialist can judge, the human contribution is authoring and bounding the codes and owning a small reference, not producing labels at volume. 

---
# Mining Legal Arguments in U.S. Corporate Case Law 

**Authors**: Luis Brena, William Jurayj, Gregory Deyesu, Zaid Al-Huneidi, Andrew Blair-Stanek, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2609.25441)  

**Abstract**: Legal argument mining supports passage classification, retrieval, and argument completion. This work introduces an expert-annotated dataset of 42 U.S. federal tax opinions on corporate reorganizations under I.R.C. §368. To our knowledge, it is the first expert-annotated, tree-structured argument corpus for this domain. Explicit spans receive one of five functional labels: Rule, Analysis, Conclusion, Background Facts, and Procedural History. Rule, Analysis, and Conclusion spans can be linked into directed support trees, while Background Facts and Procedural History serve a contextual function. The corpus provides span-based, sentence-based, flat, and tree-structured representations. Agreement analysis shows that functional node labels are more reliable than directed support edges and implicit intermediate conclusions. Directed-path agreement is stronger than direct-edge agreement, which indicates that broad reachability is more stable than exact local decomposition. Classification experiments show that functional labels are learnable under case-disjoint evaluation. Retrieval experiments show that supervised fine-tuning improves within-case retrieval. However, cross-case generalization remains weak. The dataset supports legal passage classification and provides a conservative benchmark for structured argument mining in U.S. federal tax case law. 

---
# Passes Alone, Fails Together: Benchmarking Semantic Coordination in Parallel LLM-Agent Development 

**Authors**: Haocheng Xia, Eugene Wu, Yongjoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2609.25396)  

**Abstract**: Parallel coding agents can produce patches that work alone but fail when merged. This happens when one agent changes an interface or rule that another agent still relies on. We study these failures with stale, a benchmark for semantic coordination. Our evaluation runs the same tests on each patch alone and on their combination, counting only failures introduced by combining the patches. We use three tiers: synthetic tasks with controlled interface changes, pairs of merged pull requests, and constructed tasks that use real Django helpers. Among 834 runs on 417 mined Django pairs, only one showed interference after correcting the grading procedure. On constructed tasks using 12 Django helpers, interference occurred in 97% of runs. A message describing the completed concurrent change recovered 82% of runs. Reviewed pull requests may contain few unresolved parallel changes, even when agents fail on controlled tasks using real code. The constructed failure rates do not estimate how often these problems occur in practice. 

---
# TelecomGPT-R1: Unified Post-Training for Reasoning Across Heterogeneous Telecom Tasks 

**Authors**: Bohao Wang, Chenwei Wu, Hang Zou, Yu Tian, Lina Bariah, Li Wei, Chongwen Huang, Yongliang Shen, Zhaoyang Zhang, Merouane Debbah  

**Link**: [PDF](https://arxiv.org/pdf/2609.25356)  

**Abstract**: Large language models (LLMs) offer great potential to automate a broad range of telecom engineering tasks by reasoning over standards, network configurations, mathematical models, source code, and operational logs. However, existing telecom LLMs struggle to reliably reason across these diverse tasks and data types. General-purpose LLMs often lack reliable grounding in telecom-specific knowledge, while telecom-specialized models are typically developed for narrower task families and exhibit limited multi-task performance. To fill this gap, we introduce TelecomGPT-R1, a family of open source unified telecom reasoning models structured around four complementary axes: protocol, knowledge, modeling, and fault. We first develop an axis-aware data generation framework that refines coarse public telecom artifacts into verified question-answer pairs and high quality chain-of-thought (CoT) reasoning trajectories, yielding a training corpus containing 104,880 examples. Building on this corpus, supervised fine-tuning (SFT) instills telecom knowledge and evidence-grounded reasoning patterns to overcome the cold start barrier for reinforcement learning (RL). We then apply dynamic sampling policy optimization (DAPO) with task-routed rubric rewards to keep RL updates informative and stable across heterogeneous telecom reasoning tasks. These rewards decompose axis-specific CoT traces into verifiable reasoning units and combine grounded dense process credit with outcome correctness, allowing RL to learn generalizable problem solving behaviors from verifiable telecom evidence. We release the TelecomGPT-R1 models and a reproducible training recipe to support further community development. Evaluations on seven benchmarks of the GSMA Open Telco Leaderboard show that the open-source TelecomGPT-R1-27B achieves an 89.64% mean score, outperforming leading proprietary models, including GPT-5, Claude, and Gemini. 

---
# FineWeb-CLaR: Culture, Language, and Region Annotations for Benchmark-Aligned Corpus Auditing 

**Authors**: Yusser Al Ghussin, Eva Gavaller, Cristina España-Bonet, Josef van Genabith, Simon Ostermann  

**Link**: [PDF](https://arxiv.org/pdf/2609.25298)  

**Abstract**: Cultural evaluation coverage and robustness in language models are difficult to diagnose because pretraining corpora and cultural benchmarks are rarely indexed with comparable metadata. Benchmarks increasingly target culturally situated phenomena at the level of languages, regions, and locale-specific practices, while web-scale corpora are usually organized only by language. A shared culture-language-region layer makes these resources comparable, enabling audits of whether a target cultural phenomenon is represented in pretraining data, evaluated by benchmarks or both. To this end, we introduce FineWeb-CLaR, a large-scale annotated dataset derived from FineWeb and FineWeb-2 that places web documents on a shared culture-language-region axis for corpus auditing and benchmark alignment.
FineWeb-CLaR annotates the full 30.9B-document collection from FineWeb and FineWeb-2 with URL-derived region labels and cultural-topic provenance. Our region resolver assigns a non-empty region to 25.61% of documents (7.92B). For cultural-topic analysis, we induce locale-specific topics and project them onto the 14 leaves of the Cultural Taxonomy of Liu et al. (2025), producing Locale Topic Distributions (LTDs) for corpus-side comparison. We also annotate 277 cultural NLP benchmarks with the same taxonomy, language coverage, and region coverage. Together, these resources enable direct comparison between corpus-side pretraining evidence and benchmark-side evaluation coverage. 

---
# FinFIRST: Benchmarking Search Agents for Financial Information Retrieval, Sourcing and Traceability 

**Authors**: Wenqing Wang, Haitao Xiang, Xinyi Zhao, Mingming Yin, Ying Zhong, Zhaoxin Huan, Qiheng Zhou, Jin Zhu, Xiaolu Zhang, Shi Chang, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2609.25192)  

**Abstract**: Financial search is a highly demanding task for LLM agents, requiring not only a correct final answer but also temporally valid information retrieval, authoritative source selection, entity and period alignment, unit and definition consistency, and verifiable evidence for all conclusions. Existing benchmarks predominantly evaluate only the final answer, making it difficult to localize errors or assess whether an answer is well-founded. To address this gap, we introduce FinFIRST (Financial Information Retrieval, Sourcing and Traceability), the first financial benchmark to jointly evaluate answers and supporting evidence through atomic rubrics. FinFIRST comprises 123 expert-authored tasks spanning a graduated difficulty spectrum, constructed from aggregate patterns of real-world financial scenarios through an 18-field taxonomy, a six-axis coverage blueprint, a registry of 138 financial sources, contributions from over 50 finance experts, and a six-stage quality-control pipeline. Each task is accompanied by an evidence-grounded reference package decomposed into atomic criteria across three dimensions: raw-information acquisition, source verification, and computation and answer formation. We evaluate 15 model configurations under a unified tool setting. Claude-Opus-5 achieves the highest atomic score of 87.59%, while GPT-5.6-Sol attains the highest strict pass rate of 71.54%. Computation and answer formation consistently lag behind raw-information acquisition across systems. FinFIRST retains final-answer correctness as the primary objective while making the supporting research process measurable, verifiable, and diagnosable. 

---
# Impact Is Not Invalidation: Ask About the Claim, Not the Diff 

**Authors**: Atul Anand  

**Link**: [PDF](https://arxiv.org/pdf/2609.25130)  

**Abstract**: Memory systems for coding agents must decide, when a repository changes, which of their stored claims have become false. Content anchoring invalidates a claim whenever the artifact it came from changes, which fires constantly. Semantic-equivalence classification asks whether a diff preserves behavior, a question about the diff rather than about any stored claim. We show the second signal fails for a reason unrelated to model capability: asked whether a commit preserves behavior, five models spanning a 40x price range fire on 59-72% of real commits and reach precisions of only 0.291 to 0.329 against a 0.25 base rate. Asked instead whether one specific claim still holds, the same models on the same diffs reach 0.705 to 0.974. A control that hands the behavior-preservation judge the claim text, changing only the question, moves precision by 0.010 and 0.016; changing the question moves it by 0.49 and 0.65. We also compare against pytest-testmon, a deployed regression-test selector with coverage-derived dependency data: it reaches 0.868 recall at 0.415 precision, so near-complete knowledge of what a change can reach does not identify what it falsifies. Ground truth is execution, not annotation: a claim is a test function passing at commit t, and it has flipped if that same assertion text fails at t+1. Building this required an observation we did not find in prior work. On a CI-gated mainline a commit that leaves a pre-existing test failing cannot merge, so the naive construction has an empty positive class by design. We report 10,369 claims with 184 execution-verified flips mined from 23 Python libraries, splits held out by repository, a post-knowledge-cutoff split, a shuffled-diff null, a paraphrase control, and a leave-one-repository-out analysis over 17 repositories. 

---
# ufakzeka-1: Building and Evaluating a 151M-Parameter Turkish Language Model from Scratch 

**Authors**: Sait Furkan Teke  

**Link**: [PDF](https://arxiv.org/pdf/2609.25081)  

**Abstract**: We describe ufakzeka-1, a 151M-parameter (182M with embeddings) decoder-only Turkish language model pretrained from scratch on 13.5B tokens of openly licensed text and instruction-tuned for chat, at a total cost of about \$286 in cloud GPU, API and notebook time. The contribution is not the model's capability, which is what a model this size can be expected to have, but the record of building and measuring it: a Turkish byte-level tokenizer at 1.77 tokens per word, a three-stage pretraining schedule, a post-training mixture of openly licensed and generated data, and an evaluation battery of release gates, a rule-checked sweep of 5,508 conversations, judged conversations and hand tests, all with prompts held out from the training data, enforced by decontamination inside the data build and by a checked-in invariant script we run before each build. We report three findings that we believe transfer to other small-model efforts: a safety gate that had been "fixed" with training data written from its own questions read 64/64 while the honest figure was 34/64; training-seed variance was as large as the spread across every recipe we tried, so single-seed comparisons at this scale are uninformative; and data rounds repaired only what was absent from the data, while identity tracking over long context and multi-turn arithmetic did not move across any data change we tried, which we read as limits of the model size rather than gaps in the data, a reading the next, larger model will test. Weights, the data recipe, the evaluation code and the spend ledger are released under Apache-2.0. 

---
# Understanding Reliability in LLM-based Human Behavior Simulation 

**Authors**: Pei Wang, Lei Wang, Yuanzi Li, Xu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.25066)  

**Abstract**: Large language models (LLMs) are increasingly used to simulate human survey responses and behavioral reactions, yet unreliable simulations can mislead social science conclusions. However, existing evaluations focus on end-to-end scores, leaving it unclear how different aspects of the simulation process interact to determine reliability. We propose ReliMap, which decomposes LLM-based human behavior simulation into three structured layers and evaluates reliability at both the individual level (R1) and population level (R2) across three configuration dimensions: model capacity, profile completeness, and population coverage. Through experiments across four simulation tasks and eleven LLMs, we find that all models exhibit substantial distributional bias without profile conditioning. Profile conditioning reduces this bias with diminishing returns. Larger models benefit more, and attribute informativeness matters more than quantity. Critically, R1 gains do not reliably transfer to R2--individual and population-level reliability can move in opposite directions. At the population layer, increasing coverage reduces variance but not systematic bias, with R2 stabilizing at around 50-100 individuals. These findings highlight that reliable simulation cannot be achieved by optimizing any single layer in isolation, but requires coordinated improvement across all three. 

---
# ChainDoRA: Tensor-Train Factorized Weight-Decomposed Low-Rank Adaptation for Parameter-Efficient LLM Fine-Tuning 

**Authors**: Ashfak Yeafi, Mehedi Hasan, Md Khairul Islam  

**Link**: [PDF](https://arxiv.org/pdf/2609.25058)  

**Abstract**: Parameter-efficient fine-tuning (PEFT) adapts large language models (LLMs) to downstream tasks while updating only a small fraction of their pretrained parameters. Low-Rank Adaptation (LoRA) uses two trainable low-rank matrices, while Weight-Decomposed Low-Rank Adaptation (DoRA) further separates weight magnitude and direction but retains the dense LoRA-style factorization in its directional branch. We propose ChainDoRA, a weight-decomposed adaptation framework that constructs the directional low-rank factors from a connected Tensor-Train (TT) chain, where the adapter rank forms the boundary rank between input- and output-side TT contractions and an independent TT rank controls representation capacity and parameter cost. Under a controlled 15,119-example response-only adaptation setting with LLaMA-7B, ChainDoRA is evaluated against matched LoRA and DoRA baselines on seven commonsense reasoning benchmarks. ChainDoRA with TT rank 16 achieves a seven-task average accuracy of 72.30%, compared with 69.88% for LoRA and 69.39% for DoRA, while requiring only 5.35M trainable parameters versus 56.10M for LoRA and 56.98M for DoRA, corresponding to a 90.62% reduction relative to DoRA. Ablations over TT rank and adapter placement show controllable parameter-accuracy trade-offs, indicating that connected TT parameterization can substantially reduce the parameter cost of magnitude-direction adaptation while preserving, and in this setting improving, downstream reasoning performance. 

---
# Graph-Based Inference for Feedback-Driven Word Deduction: A Scalable Framework for the Jotto Problem 

**Authors**: Dakshi Arora, Prakhar Kumar Srivastava, Ranjib Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2609.25056)  

**Abstract**: A feedback-based word deduction framework based on the Jotto problem is proposed, and the problem space is represented as a weighted graph where all valid words correspond to nodes, and the edge weight is defined by the number of common letters between the two words. Finally, the gameplay is defined as an iterative constraint propagation mechanism where feedback is used to iteratively narrow the incompatible space of the graph, facilitating the reduction of the hypothesis space in a structured and interpretable manner.
In contrast to existing approaches, where the problem space is typically defined for fixed-length isograms, the proposed framework generalizes to variable-length words (between 3 and 8 letters) and naturally extends to repeated letter cases, facilitating the treatment of realistic Jotto problem instances within a unified framework for the first time. The proposed framework's applicability and solver dynamics are also discussed through an interactive implementation and a qualitative case study, respectively.
Significant automated tests on approximately 3,000 simulated gameplay scenarios identify a novel convergence behavior: the expected number of iterations diminishes with increasing word length. A strong relationship is confirmed using statistical tests to verify a logarithmic relationship, which is also verified using regression modeling and goodness-of-fit tests.
In addition to the initial problem statement, this formulation introduces graph pruning as a viable paradigm for feedback-driven inference with interpretability and its association with symbolic reasoning and interactive intelligent systems. 

---
# ICDAR2026 Competition on Multimodal Reasoning over Documents in Multiple Domains 

**Authors**: Artemis Llabrés, Marc Serra Ortega, Tomàs Ockier, Samuel Ortega Cuadra, Amritpal Singh, Christos Georgakilas, Andrey Barsky, Ernest Valveny, Dimosthenis Karatzas  

**Link**: [PDF](https://arxiv.org/pdf/2609.25055)  

**Abstract**: In this report we present results of the ICDAR2026 Competition on Multimodal Reasoning over Documents in Multiple Domains. This competition aimed to advance research in document understanding through the task of Visual Question Answering (VQA). Building upon previous DocVQA benchmarks, this competition introduces challenging reasoning questions over a diverse collection of documents spanning eight domains, including business reports, scientific papers, slides, posters, maps, comics, infographics, and engineering drawings. The competition concluded with 20 valid submissions from 8 teams spanning zero-shot VLMs, OCR and parser-augmented pipelines, agentic retrieval systems, multi-agent ensembles, and fine-tuned multimodal models. The results show that the strongest systems move beyond single-pass prompting and instead rely on structured evidence extraction, retrieval, verification, and orchestration across multiple components. 

---
# MoM: Memory of Memory 

**Authors**: Bowen Qin, Yao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2609.25054)  

**Abstract**: For a long-horizon LLM agent, the memory question is not what was once recorded but what \emph{currently holds}. Most designs answer it only indirectly: every interaction is stored, and the present is reconstructed at query time by retrieving and reconciling records, so stale values re-enter and the same conflicts are re-litigated. Committing the current value at write time avoids this, but existing write-time (CRUD) memories overwrite, so a wrong update is unrecoverable and prior state is lost. We take the missing combination---\emph{commit on arrival while retaining what is displaced}---and formalize it as \textsc{Memory of Memory} (MoM): memory tracks not only content but the provenance, status, and history of its own entries. We instantiate MoM as \textsc{Provenant Memory} (P-Mem), a typed provenance graph whose \emph{active frontier} exposes one current value per resolved key while displaced values are retained as provenance; typed operations decide whether a new observation supports, supersedes, contests, rejects, revokes, or resolves an existing value. P-Mem's decisive gain is validity rather than accuracy: its turn-level read matches the strongest retrieval memory in accuracy at $\sim$4$\times$ fewer read tokens---a retrieval-granularity effect---while graph-guided turn pruning cuts the knowledge-update stale-answer rate (19.4\%$\rightarrow$10.9\%); on revision chains it stays at 100\% where query-time reading collapses to 25\%, and, because displaced values are retained rather than overwritten, it recovers committed errors a CRUD memory cannot (100\% vs.\ 0\%). 

---
# LatentPort: Beyond KV Cache - Cross-Model Transfer of Recurrent Memory in Hybrid Language Models: A 4B-to-9B Hybrid-State Handoff Without Target Prefix Replay 

**Authors**: Simon P. Villani  

**Link**: [PDF](https://arxiv.org/pdf/2609.25053)  

**Abstract**: Can one language model hand its live memory to another without the receiver rereading the context? We demonstrate useful persistent hybrid-state transfer across one architecture-matched Qwen3.5 4B-to-9B sibling pair. To our knowledge, this is the first demonstrated cross-model handoff of persistent recurrent inference state between differently sized hybrid language models without target prefix replay. Translated attention KV alone leaves a large gap; adding the Gated DeltaNet (GDN) persistent-state package lowers teacher-forced negative log-likelihood (NLL), the average next-token log-loss, by 0.747 nats/token (95% paired document bootstrap CI [0.6921, 0.8047]), improving all 64 PG19 documents. Direct recurrent and convolution reuse outperforms the tested learned GDN maps, consistent with partial functional compatibility of persistent-state coordinates. A fresh component factorial selects translated KV with direct recurrent and convolution state. An additional 434,176-parameter correction improves that base on 64 fresh web documents: continuation loss is 0.076 nats/token above native 9B (excess NLL), Jensen-Shannon (JS) divergence is 0.022, and native context recovery (NCR) is 0.918. Corrected 9B significantly beats continued 4B inference while processing zero historical prefix tokens. Evidence covers one direction, one geometry-matched Base-model pair, and 4K teacher-forced continuation; the near-native gate failed, the 16K branch was not run, and free-generation equivalence and a general state interface remain unproven. 

---
# Self-Cleaning and Captured Anyway: One Measured Primitive for Error in a Store an Agent Writes to Itself, and What a Falling Score Actually Measures 

**Authors**: Wenhui Chen, Jianlin Chen, Ziyao Lin, Chi Man Vong  

**Link**: [PDF](https://arxiv.org/pdf/2609.25052)  

**Abstract**: "An agent that writes its conclusions into a store it later retrieves from closes a loop usually reported as one-way contamination. Taking the loop to the infinite-tenure limit against an append-only store gives a different picture: because writing never deletes, the reachable state space has a hard upper edge at (n-1)/n, so the outcome is a choice between two edges rather than a decay. At f_0 = 0.9 the interval between the two modes holds 3.6% of 220 runs where a uniform spread would put 20.6%, and is strictly empty on the first 15; the pooled mean describes 8.2% of the runs it summarises, the median 68.2%. Everything the model contributes is carried by one measured primitive with no fitted parameter, the copy function \gamma(\phi): on 36 Wikidata facts, sign(\hat{\gamma} - \gamma_{crit}), with \gamma_{crit} = 1/k at r = 0, w = 1, predicts the direction of drift on 353 of 360 real-fact runs (39 of 40 synthetic in the same batch). Scale does not rescue the store: pooled frontier capture is 0.850, with claude-sonnet-4.5 captured on 20 of 20 seeds against our registered prediction of <0.5. What the interval tests is distinguishability rather than count: on the real facts, multi-valued runs have 6.4x its occupancy of the rest. It survives at f_0 in {0.1, 0.3, 0.5}, capture peaks at f_0 = 0.5, and of four interventions with criteria frozen first, timing dominates fraction at matched budget while a consistency gate drives every model to 0.993. The resampling unit is the seed, at a design effect of 3.75 on a pooled level: under a 44-seed control the ordering supporting claim 4 collapses from Spearman +0.98 at three seeds to +0.31-0.80 at forty-four, while claim 2's ordering is exact there (+1.00, p = 0.017). All 87 graded rows are in Appendix W, 37 of them graded withdrawn, failed, self-correcting, undecidable or an acknowledged limit, against 50 that are not." 

---
# LLM-Driven Training-free Location-Attribute Synergic Fusion: A Closed-Loop Paradigm for Dual-source Encrypted POIs and LULC Mapping 

**Authors**: Chang Li, Xingtao Peng, Yongjun Zhang, Yinfei He, Cairun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.25051)  

**Abstract**: Dual-source encrypted points of interest (DSEP), POIs from two encrypted coordinate systems, suffer from intertwined location and attribute uncertainties, including nonlinear systematic misalignment and naming inconsistency, hindering land-use/land-cover (LULC) mapping. To the best of our knowledge, this paper is the first to propose an LLM-driven, training-free location-attribute synergic closed-loop optimization paradigm for DSEP fusion. The paradigm jointly refines location transformation and attribute correspondences through iterative feedback. Attribute-synergic location fusion uses an LLM-driven attribute matching method to establish DSEP correspondences, reducing matching complexity from O(N^2) to O(N), and refines transformation coefficients using an improved particle swarm optimization algorithm within ISODATA-clustered local subregions. Location-synergic attribute fusion then reassesses attribute confidence from updated geometric residuals through an LLM-fuzzy method. The refined correspondences feed back into location optimization, forming a bidirectional closed loop. Sample purification and adaptive radius contraction enable convergence in essentially two iterations. We further propose a training-free LULC mapping method that inherits land-use classes from encrypted maps through location fusion, producing vector-raster integrated LULC maps. A reference-free POI fusion evaluation method is applied across 31 provincial capitals and municipalities in mainland China. Experiments show that our method achieves an average DSEP location fusion residual of 4.58 m and attribute fusion accuracy of 95.12%, improving upon the open-source baseline and state-of-the-art method by 1.77 m and 14.87%, respectively. Overall, the method provides a training-free solution for DSEP fusion and enables georeferencing of encrypted vector data to WGS-84 without field-surveyed ground control points. 

---
# FrontierMath Erdős 

**Authors**: Tom Adamczewski, Thomas F. Bloom  

**Link**: [PDF](https://arxiv.org/pdf/2609.25050)  

**Abstract**: We introduce FrontierMath Erdős (FME), a benchmark of 68 Erdős problems that are open as of August 2026. To solve a task in FME, AI systems must resolve (prove or disprove) one of the 68 conjectures in the proof assistant Lean. Our 68 problems were selected by the second author among 652 open problems on this http URL for their mathematical interest and difficulty. AIs have recently resolved several open problems in mathematics, but these demonstrations fall short of a systematic study of AI capabilities. FME evaluates every AI model on the same fixed problems, autonomously and under the same budget. We evaluated five AIs with a budget of \$300 per problem. One (GPT-6 Astra) scored 3%, and all others scored 0%. 

---
# Mitigating LLM Over-Refusal via Dynamic Semantic Routing Calibratione 

**Authors**: Zixuan Wang, Bingjie Zhang, He Zhao, Dandan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2609.25049)  

**Abstract**: Large language models (LLMs) aligned for safety often suffer from over-refusal, incorrectly rejecting benign yet safety-related instructions. Prior studies primarily attribute this to static representation overlap, largely overlooking the underlying dynamic mechanisms. In this paper, we present the mechanistic analysis of over-refusal through the lens of internal routing conflicts within transformer attention. We discover that a sparse subset of Hypersensitive Safety Heads misfires on Hard-Safe prompts, exhibiting abnormal attention entanglement that forcefully binds harmless target entities to refusal semantics. This triggers a severe, high-entropy routing conflict that deprives target entities of necessary attention. To counteract this, we propose Semantic Routing Calibration (SRC), a lightweight, training-free inference framework. SRC precisely localizes and dynamically suppresses these hypersensitive safety heads at the inference stage. Coupled with a dual-branch logits fusion that acts as a safety regularizer during subsequent decoding, SRC seamlessly restores trustworthy reasoning. Extensive experiments demonstrate that SRC alleviates over-refusal, with intrinsic safety performance preserved as much as feasible. 

---
# Prompt Breadth and Rollout Refresh Interact in On-Policy Distillation 

**Authors**: Lingxiang Hu, Tianle Xia, Ming Xu, Yiding Sun, Linfang Shang  

**Link**: [PDF](https://arxiv.org/pdf/2609.25048)  

**Abstract**: How many prompts does on-policy distillation (OPD) need, and how does the answer depend on the student policies that generate its training responses? We study these two controls jointly: prompt breadth and rollout refresh. A 3x3 mathematical-reasoning experiment fixes 14,080 trajectories and 110 optimizer updates while varying the prompt bank and the number of response-generating policy snapshots. With ten snapshots, eight prompts reach 24.09% average accuracy, close to 24.51% for 14,080 distinct prompts. With responses frozen at the initial policy, however, increasing breadth lowers accuracy from 21.16% to 19.05%; under per-update refresh, it raises accuracy from 23.61% to 25.57%. The resulting interaction is 4.07 percentage points, with a 95% question-paired interval of [2.00, 6.28]. Matched comparisons under two teachers reveal a second reversal: the periodic models have higher short-budget accuracy and answer completion, but frozen-response models overtake in average accuracy at a 32K output limit, using 1.7-1.8x as many response tokens. These results show that prompt efficiency in OPD can depend on both refresh and inference budget. 

---
# AIBuildAI-2.5: Efficient Autonomous AI Model Development Through LLM-Guided Tree Search 

**Authors**: Peijia Qin, Ruiyi Zhang, Qi Cao, Han Guo, Li Zhang, Pengtao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2609.25047)  

**Abstract**: Autonomous agents that automatically build artificial intelligence (AI) models could broaden access to AI across science and engineering. A popular line of such agents frames model building as a code search problem and solves it by tree search, in which each node is a candidate program and the tree grows by generating a child program from a parent, and these agents now approach the capability of experienced AI engineers on realistic benchmarks. However, these agents have three weaknesses in efficiency that have not been fully addressed. First, only a small number of candidates can be executed within a realistic budget, so search rules that rank nodes by executed rewards, such as Monte Carlo-style tree search, rely on few and noisy scores and select the next node to explore less effectively. Second, no resource-aware strategy is used to schedule training jobs, which can lower hardware utilization and training efficiency. Third, every agent call is served by a single powerful model, which inflates inference cost. Here we introduce AIBuildAI-2.5, an agentic system that carries out the tree search with LLM agents and addresses each of the three issues. AIBuildAI-2.5 proposes a novel LLM-guided tree search, in which a judge scores each candidate on its expected improvement, grounding, and feasibility, and a selector ranks the pool of candidates from these scores and the state of the search. In addition, AIBuildAI-2.5 comprises a scheduler that launches training jobs with the current hardware resource status taken into account and a router that assigns lower-cost LLMs to less demanding tasks while reserving the most capable LLM for the most challenging sub-tasks in the AI model building workflow. AIBuildAI-2.5 ranks first on MLE-Bench with a medal rate of 73.3%, and outperforms a strong baseline on six autonomous AI research tasks from AIRS-Bench. 

---
# Peerify: Benchmarking Peer-Review Claim Verification 

**Authors**: Alireza Daghighfarsoodeh, Sajad Ebrahimi, Ali Ghorbanpour, Soroush Sadeghian, Radin Cheraghi, Negar Arabzadeh, Ebrahim Bagheri  

**Link**: [PDF](https://arxiv.org/pdf/2609.25046)  

**Abstract**: Peer review plays a central role in scholarly publishing, yet verifying whether reviewer claims are supported by manuscript evidence remains a largely manual and time-consuming process. We present Peerify, a pipeline for manuscript-grounded verification of peer-review claims. Given a manuscript and a review comment, the Peerify pipeline decomposes reviews into atomic claims, retrieves relevant manuscript evidence, and determines whether each claim is supported by the paper. To support the development and evaluation of the pipeline, we construct a benchmark of 800 claims derived from authentic peer-review interactions collected from NeurIPS 2024 and ICLR 2024, including a 300-claim hand-labeled subset used to audit the automated supervision. We evaluate state-of-the-art language models and retrieval strategies within the Peerify pipeline, together with entailment baselines. Our results demonstrate the importance of retrieval-centered verification and claim decomposition, while highlighting the challenges posed by ambiguous and interpretive reviewer claims. Automated labels agree with human consensus on 90.3% of audited claims ($\kappa = 0.87$), while off-the-shelf entailment models stay below 0.24 macro-F1. 

---
# From Tone to Trajectory: Continuous Sentiment and the Shape of Monetary Policy Communication 

**Authors**: Martin Feldkircher, Márton Kardos, Kristoffer Laigaard Nielbo  

**Link**: [PDF](https://arxiv.org/pdf/2609.25034)  

**Abstract**: Central bank press conferences are not merely information releases --- they are structured narratives. We study whether the shape of sentiment within a statement, not just its average tone, carries policy-relevant signals. Constructing sentiment arcs for ECB and Fed press conferences along three dimensions --- monetary stance, economic outlook, and uncertainty --- we assess their predictive content for policy rate changes, inflation expectations, and forecaster disagreement. Our findings show that arc shape robustly predicts rate decisions beyond lexicon-based benchmarks at both institutions --- it is not merely whether a statement sounds hawkish or economically optimistic on average, but how these sentiments are sequenced and emphasized across the statement, that carries the policy signal. Arc features also shape how professional forecasters update inflation expectations and how much they disagree, pointing to a receiver-side effect distinct from the direct policy signal. These findings suggest that communication design --- the sequencing and emphasis of policy language across a statement --- is a first-order feature of the policy signal, not a second-order refinement. 

---
# Retrieved-Span Training for Efficient Query-Focused Meeting Summarization on QMSum 

**Authors**: Edward Xi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2609.25028)  

**Abstract**: QMSum provides no scorer, making query-focused meeting summarization results difficult to compare. We rescore or generate 15 systems under one implementation. Through a common inference port, a released 406M Fusion-in-Decoder specialist loses 6.30 ROUGE-1 when moved from capped long input to 2,000-word retrieved spans. Fine-tuning it on this span regime recovers the loss. On test it scores 36.33 ROUGE-1 versus 35.41 for our 1.2B system; the meeting-cluster 95% interval for the difference is [-0.27, +2.22], so QMSum does not statistically separate them. The smaller system uses about one-third as many total parameters and less than half the peak inference memory. Within the fixed 1.2B base, span-regime fine-tuning adds 5.29 [+4.02, +6.56], while replacing the first 4,500 transcript words with 2,000 retrieved words adds 1.55 on test and 0.29 on validation. Separately, under one concise prompt and reference-overlap scorer, a released 406M specialist exceeds five proprietary hosted models by at least 6.2 ROUGE-1, but output length and absent human or factuality evaluation limit this ordering. Conclusions are limited to QMSum and automatic metrics. 

---
# A Computational Approach to Measuring Semantic Change in Sanskrit Literature 

**Authors**: Tanay Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2609.25012)  

**Abstract**: Diachronic word embeddings have become the modern standard for tracking semantic change, yet they have been largely validated on modern, high-resource, and well-segmented languages. This paper tests whether the paradigm transfers to Sanskrit, an ancient, low-resource language whose phonological fusion (sandhi), morphological inflection, compounding, and polysemy pose a unique challenge. I assemble a 2.7M-token corpus spanning four canonical periods, recover word boundaries with a neural byte-level sandhi splitter and lemmatizer, and train per-period embeddings across configurations. To evaluate the system, I curate a validation set from historical scholarship and test recovery directionally with anchor displacement. Of 21 testable shifts, 19 move in the philologically attested direction (sign test, p=0.00011). I further show which configuration the language forces and comment on opportunities for improvement. 

---
# Same Quantity, Different Answer: Numerical Representation Invariance in Language Models 

**Authors**: Ephraim Atta-Duncan  

**Link**: [PDF](https://arxiv.org/pdf/2609.25009)  

**Abstract**: Numerically equivalent word problems should yield the same canonical answer whether a quantity is written as a decimal, fraction, percentage, number word, scientific notation, or an exactly converted unit. We generate 3,600 exact-rational problems and 8,600 prompts spanning five identity-preserving transformation families, and evaluate five open-weight systems. After a fixed syntax audit that normalizes common answer forms without an LLM judge, canonical accuracy is 0.969-0.996, but orbit correctness falls to 0.848-0.981 and orbit invariance to 0.851-0.981; invariant-but-wrong orbits account for at most 0.003. Most of the broad strict-parser collapse arises because multiplication-form scientific notation lies outside the implemented number grammar, illustrating how evaluator interfaces can masquerade as reasoning failures. A distinct semantic pathology remains: Mistral Small 4 scores 0.699 on unit-converted inputs and produces 265 errors differing from the label by exact powers of ten. In a separate 9,000-call experiment that allocates equal calls to the compared arms, representation consensus does not outperform paraphrase consensus on a low-error subset and produces substantially more false alarms. The accompanying ancillary archive contains the frozen benchmark, evaluation and audit records, consensus raw responses, manifests, analysis code, and a one-command paper build. 

---
# Training a Language Model End-to-End in Rust: An Experience Report 

**Authors**: Arif Adito  

**Link**: [PDF](https://arxiv.org/pdf/2609.25008)  

**Abstract**: I pretrained a language model end-to-end in Rust - alone, with no team, no PyTorch, and no Python in the training path - for $164 in rented GPU time. I report that as an achievement, not a recommendation: the more useful contribution is a measured failure taxonomy of the two leading Rust ML frameworks, Candle and Burn, as training (not inference) backends in 2026. I document five Candle defects, including fused kernels that silently produce no gradient, and three Burn defects, including a backward pass at roughly 3% of theoretical GPU throughput and a kernel-fusion path that segfaults mid-training at multi-billion-parameter scale. Every one passed ordinary loss-curve inspection; none announced itself. I describe the verification discipline that caught six such silent failures, centered on a gradient-flow arbiter: a test that runs one forward/backward pass and asserts every trainable parameter receives a finite, nonzero gradient, generalizable to any framework. The trained model (roughly 0.4B parameters, Bangla-first) shows strong Bangla language-modeling signal - a per-token negative log-likelihood of 0.93 against 12.60 for a random-initialized twin - while scoring at chance on English commonsense multiple-choice, the expected outcome of a deliberately small, Bangla-weighted budget (about 2 billion tokens, 54.6 hours, one rented H100). I also report a tokenizer-fertility trap in Bengali script: naive byte-level tokenization collapsed Bangla to roughly 1.4 characters per token against English's 3.9, silently inverting the corpus's language balance; fixing it reached roughly 4.1. To my knowledge, this is among the first documented end-to-end LM pretraining runs in pure Rust. After this run I moved training to PyTorch and kept Rust for on-device serving: in my hands, Rust is not yet a competitive place to train a language model, though it may be a good place to serve one. 

---
# What Does 99% Accuracy Measure? A Reproducible Audit of Shortcut Learning in a Widely Used Fake News Corpus 

**Authors**: Yuvraj Verma  

**Link**: [PDF](https://arxiv.org/pdf/2609.25006)  

**Abstract**: Text classifiers trained on the ISOT/Kaggle "Fake and Real News" corpus routinely report accuracy and F1 above 0.98, a level of performance that sits uneasily beside the difficulty of assessing veracity. Using a transparent TF-IDF and linear-classifier pipeline as a measurement instrument, we audit the corpus along three leakage channels and two distribution-shift protocols, releasing all code and derived numbers. First, the benchmark is partly degenerate: a classifier given only the subject metadata field, with the article text discarded, attains F1 = 1.000, since the two classes have disjoint subjects. Second, removing all three leakage channels, metadata, a newswire source tag present in 99.2% of real articles, and 6,251 duplicate documents contaminating 19.4% of a naive test split, lowers F1 by only 1.21 points (0.9935 to 0.9814); the residual signal is diffuse editorial style rather than a few giveaway tokens, since deleting the 1,000 highest-weight unigrams still leaves F1 = 0.926. Third, this style signal does not transfer: under a topic-disjoint protocol, average precision falls from 0.9995 to 0.9475 and deployed F1 from 0.9905 to 0.8067, with a prior-matched analysis confirming a genuine 5.2-point loss of discrimination, while temporal transfer is nearly lossless. A fine-tuned DistilBERT is stronger in-distribution (F1 = 0.9993) but degrades far more under topic shift, losing 12.9 average-precision points against the linear model's 5.2. Transferred to the independent LIAR benchmark, all three models fall to near-chance ranking (ROC-AUC 0.54-0.57), none beating a majority-class baseline. We conclude that within-corpus scores here quantify source and topic separability rather than veracity, that added capacity exploits the shortcut rather than avoiding it, and we recommend metadata-only, small-sample, and topic-disjoint baselines as inexpensive diagnostics for future work. 

---
# Discovery-Driven Integration of Disjoint Tables via Text 

**Authors**: Md Ataur Rahman, Dimitris Sacharidis, Oscar Romero, Sergi Nadal  

**Link**: [PDF](https://arxiv.org/pdf/2609.26658)  

**Abstract**: Integrating heterogeneous datasets within data lakes is a critical challenge, particularly for semantically related tables that lack the explicit attributes needed to be joined. We study Discovery-Driven Integration, where the relevant sources and their missing relational structure must be discovered before integration. In this setting, unstructured text provides the evidence that connects otherwise disjoint tables. The fundamental challenge is to discover the relationships at a fine-grained level that connect individual rows from different tables through specific sentences. We formalize this task as Text-Mediated Join Path Discovery and propose a horizontal bidirectional cross-attention architecture called LOKI Latent-space Optimization for Knowledge Integration) that learns contextualized representations of table rows and sentences. Through a global table-text contrastive objective, fine-grained row-sentence associations emerge without explicit local supervision. Existing multi-modal discovery methods largely retrieve coarse-grained column-text associations, whereas integration systems assume supplied row-text links, schemas, or queries. LOKI instead transforms these implicit associations into explicit, interpretable join paths, organizes them into relation-consistent groups, and materializes them as typed integrated tables with sentence-level provenance. Comprehensive evaluations on real-world benchmarks demonstrate that LOKI consistently outperforms state-of-the-art multi-modal data discovery approaches, and materializes typed integrated tables with 0.982 macro typed-pair precision while being up to 40 times cheaper in LLM API cost than direct prompting. 

---
# Behavior is Not Enough: A Mechanism-Based Evaluation of Social Norm Emergence in LLM Societies 

**Authors**: Rasika Muralidharan, Haewoon Kwak, Jisun An  

**Link**: [PDF](https://arxiv.org/pdf/2609.26481)  

**Abstract**: Social norms cannot be identified from behavior alone: the same cooperative equilibrium may reflect shared expectations, strategic incentives, or simple imitation. Yet in multi-agent large language model systems, prior work largely treats behavioral convergence as evidence of norm emergence. In this work, we introduce an evaluation framework that measures agents' reported empirical and normative expectations in addition to behavioral convergence. Through controlled ablations, we test the effect of expectation elicitation and isolate two collective mechanisms central to theories of norm formation---social learning through interaction and social selection through network-based group formation. We further test the stability of these resulting dynamics under adversarial disruption across four LLM families. We find that eliciting expectations increases cooperative contributions, while social learning stabilizes behavior, and social selection reliably identifies cooperators but provides limited behavioral reinforcement. Following disruption, normative expectations and behavioral coordination recover differently. Together, these results show that similar cooperative outcomes can arise from different underlying social processes. By making expectations observable, our framework allows us to attribute each mechanism's contribution separately, offering designers of multi-agent systems a principled basis for selecting the social processes that sustain cooperation. 

---
# On the Lexical Superstition of Large Language Models for Code Comprehension: Re-evaluation on Code of Low Lexical Quality 

**Authors**: Xin Shen, San-Zhuo Xi, Yali Du, Ming Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.26388)  

**Abstract**: Recent advances in large language models (LLMs) have made them widely used for code-related tasks. Identifier names are statistically informative in naturally occurring code, but their information is not always reliable. We investigate whether current LLMs assign disproportionate weight to lexical cues when renaming preserves program structure. We introduce Face/Off, a semantics-preserving identifier-renaming framework, and evaluate progressive naming conditions across multiple models and code-comprehension tasks. Within this framework, lexical overemphasis is pervasive across the evaluated models and primary tasks: performance generally decreases as identifier information is removed or made misleading, and outputs are often directed toward the meanings suggested by misleading names. The pattern persists under representative prompt- and fine-tuning-based interventions, suggesting that lexical overemphasis is an entrenched problem. A type-inference control confirms a boundary: naming effects are smaller when the answer is locally recoverable without the target name. These results do not imply that identifiers are unhelpful; rather, they reveal a systematic vulnerability in how current LLMs balance lexical cues against program structure. Our findings motivate evaluations and modeling methods that preserve the benefits of natural code regularities while keeping conclusions grounded in accurate, formalized code semantics. 

---
# ABAI at COLIEE 2026 Task 1: Multi-Stage Retrieval with GraphRAG-Enhanced Meta-Learning, and a Post-Hoc Study of the Cross-Validation-to-Test Gap 

**Authors**: Minhan Cho, Soyoung Park, Daejin Choi, Jinyoung Han  

**Link**: [PDF](https://arxiv.org/pdf/2609.26237)  

**Abstract**: We present the ABAI submission to COLIEE 2026 Task 1, case law retrieval, together with a controlled study of why it underperformed. The task suppresses the cited passages themselves, which removes much of the lexical overlap a retriever would rely on. Our pipeline answers this with four independently trained stages: multi-view BM25 over citation-context windows with reciprocal rank fusion, neural reranking, graph-based features from entity communities and a graph attention network, and a LightGBM meta-learner over 34 features. Our best run reached F1=0.177 on the official test set, against a cross-validated 0.311, and we attributed that gap to a recall ceiling, temporal distribution shift, and threshold miscalibration. We then tested all three. Under leakage-free protocols threshold transfer costs 0.007 F1, decision quality is flat across chronological quartiles, and the official test queries are not measurably farther from the training manifold than training queries are from each other, in two independent embedding spaces. Decomposing the misses instead splits them exactly evenly between candidates never retrieved and candidates retrieved but ranked below the cut. Measuring the remedies for each half, BM25 length-normalisation tuning, an event-triple view, and full-content dense fusion lift top-200 recall by three to seven points, and citation-graph features add 0.014 F1 over eight seeds once own-citation leakage is removed, while per-query cutoff rules, a zero-shot reranker swap, and a date filter do not help. We also document four evaluation artifacts, each of which reversed a result once the protocol was corrected. 

---
# A Semantic Approach to the Academic Publishing Network: Document Vector Representations and Hybrid Structural-Semantic Fusion over OpenAlex Data 

**Authors**: Robert Šamárek, Radek Martinek  

**Link**: [PDF](https://arxiv.org/pdf/2609.26218)  

**Abstract**: Structural graph analysis of the academic publishing network captures the topological relationships between entities but does not see the content of works. Building on our structural approach, this work complements it with a semantic layer and a parameterized structural-semantic fusion. We represent scientific documents by citation-informed vector embeddings (SPECTER2) and store them in an embedded vector database keyed by the stable OpenAlex ID, so that they connect directly to the graph layer. We define a modular late-fusion function that combines semantic similarity (cosine of embeddings) and structural similarity (bibliographic coupling) with a tunable weight alpha whose value is chosen according to the specific task. On the corpus of VSB - Technical University of Ostrava we show two things: citation-informed embeddings agree with the expert OpenAlex topical taxonomy better than a TF-IDF baseline, and in a recommendation use case the structural, semantic, and combined signals carry information in different regimes depending on the available data. Hybrid fusion here is not a universally better method but an explicit mechanism for steering complementary signals according to the task. We release the whole approach as an open-source extension of the apnet library with a reproducible workflow. 

---
# WatchPoint: Executable User Feedback for Real-World Agentic Web Development 

**Authors**: Guanqun Yang, Wei Yang, Xueqing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.26204)  

**Abstract**: When a professional web developer's code fails a test, they do not simply re-read the stack trace. They open the application in a browser, click buttons, inspect computed styles, and run diagnostic commands to understand what went wrong. Existing feedback mechanisms for coding agents rely on screenshots, LLM-as-a-judge scoring, or natural-language corrections, but few interact with the live application the way a developer would. We introduce WatchPoint, a simulated-user system that mimics real developer behavior by generating and executing diagnostic scripts against the running application, producing structured observations that guide the coding model's retry. Unlike prior approaches that target single-file edits or evaluate using non-executable metrics, we operate on Web-Bench, a benchmark of 50 multi-file web projects comprising 1,000 sequentially dependent tasks, verified by deterministic end-to-end tests. WatchPoint recovers 57.6% of the tasks it diagnoses, and a controlled user study confirms the simulation's realism: human testers achieve a comparable recovery rate (54.5%), providing evidence that automated diagnostic scripts can substitute for interactive human testing on sequential web development tasks. We further identify a pattern of capability gaps that governs when simulated-user feedback is helpful and when it should be withheld. 

---
# Dynamic Deep Prompt Optimization for Defending Against Jailbreak Attacks on LLMs 

**Authors**: Doniyorkhon Obidov, Honggang Yu, Xiaolong Guo, Kaichen Yang  

**Link**: [PDF](https://arxiv.org/pdf/2609.26185)  

**Abstract**: Large Language Models (LLMs) demonstrate impressive capabilities across many applications but remain vulnerable to jailbreak attacks, which elicit harmful or unintended content. While model fine-tuning is an option for safety alignment, it is costly and prone to catastrophic forgetting. Prompt optimization has emerged as a promising alternative, yet existing prompt-based defenses typically rely on static modifications (e.g., fixed prefixes or suffixes) that cannot adapt to diverse and evolving attacks.
We propose Dynamic Deep Prompt Optimization (DDPO), the first jailbreak defense based on deep prompt optimization. DDPO uses the target LLM's own intermediate layers as feature extractors to dynamically generate defensive embeddings via a lightweight multilayer perceptron. These tailored embeddings are then injected into a subsequent intermediate layer, enabling an input-dependent defense without modifying the LLM's weights. This design ensures high adaptability with minimal computational overhead.
Experiments on a diverse set of models and attacks demonstrate that DDPO significantly outperforms static prompt optimization methods, particularly on weakly aligned models and when handling semantically ambiguous benign prompts, successfully distinguishing them from genuinely harmful requests. 

---
# DTOC: Dynamic Tool Output Compression for Adaptive Context Management in AI Agents 

**Authors**: Abhay Chaturvedi, Shreya Bhattacharya, Rashmika Gopalkrishnan, Peter van der Putten  

**Link**: [PDF](https://arxiv.org/pdf/2609.26121)  

**Abstract**: As agent capabilities have grown, practical limitations increasingly stem from constrained context windows rather than model capacity. Common strategies, such as truncation, heuristic aging, and lossy summarization, may discard useful information or introduce hallucination risk. To address these challenges, we propose Dynamic Tool Output Compression (DTOC), a framework for scalable context management in LLM-based agents that models context updates as explicit and reversible operations within the agent reasoning loop. DTOC retains full tool outputs in external memory while inserting compact placeholders into the active context, enabling selective reconstruction when needed. We formalize the DTOC mechanism, integrate it into a ReAct-style agent architecture, and provide a production-oriented implementation supporting on-demand restoration of compressed outputs. Experiments on DeepSWE reveal model-dependent effects: for responsive models (Sonnet 4.6, GPT-5.4), DTOC reduces input tokens (10.3 and 12.7%) and agent steps (2.4 and 32.3%), while increasing solve rates (2.5 and 1.5 times higher) and lowering cost per solved task (3 and 3.5 times lower cost per solved task). For the other models results are more mixed, with GPT-5.5 doubling solve rate and halving cost, but no impact on solve rate and negative impact on cost for the other models. Ablation results show reversibility is critical: disable-only compression variants degraded performance, while full DTOC recovered baseline accuracy at substantially lower context cost. These findings indicate that explicit, reversible context management can improve the efficiency of long-horizon agent reasoning without degrading task performance. 

---
# TopoCompress: Topology Aware Token Compression Algorithm for Distributed Edge MoE Inference 

**Authors**: Ning Li, Xinyu Wang, Xin Yuan, Wenchao Xu, Athanasios V. Vasilakos, Song Guo, Haijun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.26061)  

**Abstract**: Mixture-of-experts (MoE) models improve capacity with moderate overhead by sparsely activating experts per token. However, deploying MoE across resource-constrained edge servers incurs substantial cross-server communication as experts are distributed across heterogeneous servers. Existing placement methods optimize for raw token traffic, while conventional compression considers semantics but ignores topology-dependent routing costs. Consequently, independent optimization leads to inefficient communication and resource utilization. This paper proposes TopoCompress, a deployment- and topology-aware token compression framework for communication-efficient distributed edge MoE inference. It jointly optimizes token compression, expert deployment/replication, GPU-CPU residency, and collaborative routing to balance cross-server transmission, quality, and resource use. To address the coupling between token-level compression and epoch-level deployment, TopoCompress employs a two-timescale alternating optimization. In the online fast loop, it identifies and compresses low-importance, high-routing-cost tokens and jointly routes surviving expert activations. In the offline slow loop, it updates expert placement, replication, and GPU-CPU residency according to post-compression traffic accumulated during online inference. We establish the feasibility, optimality, convergence, and computational complexity. Simulations demonstrate that TopoCompress effectively reduces cross-server traffic and deployment resource consumption while maintaining controllable inference quality, enabling efficient distributed MoE inference over bandwidth- and resource-constrained edge infrastructures. 

---
# FIRE: Failure-Informed Runtime Engineering for Reliable Language-Model Agents 

**Authors**: Nikita Agarwal, Nivedit Jain  

**Link**: [PDF](https://arxiv.org/pdf/2609.26048)  

**Abstract**: Language-model agents often reach a working solution and then fail to consistently deliver it. We study runtime policies: targeted natural-language instructions and action denials applied by the agent harness at states that preceded observed failures, without changing model weights or the user prompt. With this, keeping capability constant, we observe a meaningful unlock in delivered reliability. Across the complete 87-task Terminal-Bench 2.1 suite, with two attempts per task, policies increase repeated success (pass^2) in all three GPT-5.6 tiers: 50.6% to 54.0% for Luna, 55.2% to 60.9% for Terra, and 64.4% to 73.6% for Sol. Sol's best-of-two success changes by 1.2 points while repeated success rises by 9.2, showing that policies chiefly convert reachable solutions into dependable delivery. We further cover 14 tasks under Terra's frozen portfolio. Policy-guided Terra reaches 71.4%, compared with 64.3% for unassisted Sol, at about half the cost, demonstrating how engineering around models could unlock dependability for a use case. To isolate the mechanism we run a randomized five-arm experiment: real policies reach 61% on eligible tasks, versus 39% without a policy, 36% with a timing-matched sham, and 39 to 43% with generic verification or reconsideration. The intended corrective behavior appears in 22 of 24 coded policy attempts, against at most 14 in any other arm. Runtime policies are therefore a practical reliability layer: they make capabilities an agent already possesses substantially more repeatable. 

---
# MICRO: Multi-Fidelity Active Search for Severe Error Discovery 

**Authors**: Orlando Leone, Niclas Pokel, Pehuén Moure, Yingqiang Gao, Roman Boehringer  

**Link**: [PDF](https://arxiv.org/pdf/2609.26025)  

**Abstract**: Human feedback can vary in cost and informativeness. Strong feedback can reveal severe errors but is costly, so cheaper quality ratings can help decide which items to annotate. We propose MICRO (Multi-Fidelity Impact Clustered Rollout), an active search framework that allocates a shared budget to these feedback types to maximise confirmed severe error discoveries. MICRO jointly models ratings and annotation losses conditional on item features to steer acquisition. It clusters acquisitions by their predicted impact on severity probabilities to select diverse candidates, then uses rollout to estimate their discovery value. Experiments on WMT20 English-German show that ratings improve both loss reconstruction and severity prediction. MICRO achieves the highest mean discovery count across four budget and rating cost settings, with similar performance to adapted MF-ENS in one and significant gains over all six comparison policies, including two rollout controls, in the other three $(p<.001)$. 

---
# Challenges of Multi-Speaker Extraction for Real Conversational Speech Enhancement 

**Authors**: Robert Sutherland, Stefan Goetze, Jon Barker  

**Link**: [PDF](https://arxiv.org/pdf/2609.25948)  

**Abstract**: Target-speaker and multi-speaker extraction are techniques for extracting speech from a desired speaker or desired speakers in the presence of other speakers and/or noise. Neural network approaches for this task are often trained and evaluated using simulated datasets, with balanced amounts of target speech and speaker enrolment samples which closely match the target speech. However, in real multi-party conversations, participants are often silent for more time than they are speaking, and their enrolment speech samples can differ substantially from the target speech in the conversation. These factors can impact the training and evaluation of these techniques on recordings of real conversations. This work proposes a new loss function, which helps mitigate the effect of excess silence in training, improving STOI from 0.55 to 0.60, and frequency-weighted segmental SNR from 4.35 to 5.12. Additionally, the impact of the mismatch between the enrolment speech and target speech is explored. 

---
# Certified Against Which Oracle? Execution Labels Set the Reported Risk of Conformal Abstention for Text-to-SQL 

**Authors**: Jiamiao Liu, Dewen Qiao, Yu Zhang, Xuetao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.25938)  

**Abstract**: A conformal abstention certificate for text-to-SQL is only as truthful as the correctness labels it is calibrated on. The uncertainty pipelines that read confidence off execution consistency take those labels from the single database a benchmark ships, an oracle known to be lenient. We run a preregistered intervention on Spider-Realistic, swapping that database for the benchmark's distilled multi-instance test suite. Across four SQL-specialist checkpoints and two split schemes, the swap raises the certificate's held-out risk 2.73 to 10.23 points above the risk its own labels report. Neither oracle reports the risk experts assign. Under blinded labels from two SQL experts, a certificate calibrated at a nominal 0.10 carries 20.0 and 17.2 points of risk on two checkpoints. The stricter oracle errs in both directions: most of the answers it rejects are not judged wrong, and some of those it accepts are. An AI-assigned census of what it rejects finds a semantic error in a quarter to a third of them, depending on the population. It attributes most of the rest to underspecified questions, synthetic instances or suspected reference-query defects, a flag supported by a preregistered blinded expert audit. The oracle also decides how a confidence score is judged. Every execution-consistency score looks better under the labels of the oracle that built its clusters, in 16 of 16 combinations. Under expert labels, building such a score on suite clusters instead of shipped-database clusters raises its area under the ROC curve (AUROC) by 6.96 points on one checkpoint and 1.53 on the other. On the second, the expert interval excludes the 8.3 points the suite labels report. A certificate should be reported with both oracles, and an oracle-relative difference read as semantic risk only after the benchmark is audited. A consistency score should be evaluated under an oracle that did not build it. 

---
# Auditing Proxy-Based Validation Across Text Spans 

**Authors**: Daein Weon, Dong Ho Kang  

**Link**: [PDF](https://arxiv.org/pdf/2609.25808)  

**Abstract**: Evaluation scores are often validated by their agreement with inexpensive proxy labels. When the score and the proxy are computed from the same text span, however, that agreement can arise from surface evidence the two share rather than from the semantic construct the proxy is meant to represent. We make the distinction explicit by declaring the score, its span, the proxy and the target construct as a validation contract, then re-evaluating that proxy rule strictly outside the scored span. In a controlled HotpotQA correctness experiment varying only the shared text boundary, the score agrees with its proxy far better than with correctness at a 50-character prefix: the gap is +0.184, collapsing to at most +0.045 from 120 characters onward. At that short prefix the score still predicts whether the answer string appears later (AUC 0.634) while an equivalence test places its agreement with correctness at chance, so the reported proxy agreement does not establish that the score ranks correctness. On OR-Bench, suppressing each model's recurring opening templates removes most of the score's association with the refusal proxy, while matched-volume deletion removes almost none and construct agreement stays at chance. Only three of eleven external contracts support the off-span control, and none of the routing studies we sampled released the generations it needs. We therefore ask that a proxy-based validation claim declare the span each label is read from, report the construct agreement beside the proxy agreement, and release the generations that let the proxy be re-read off the scored span. 

---
# Latest Exact Match Attention 

**Authors**: Moritz Brösamle  

**Link**: [PDF](https://arxiv.org/pdf/2609.25802)  

**Abstract**: We introduce latest exact match attention (LEMA), an attention variant for transformers where queries and keys are binarized and each query attends only to the latest exactly matching key. We prove that LEMA transformers with chain of thought can simulate word-RAMs, as was recently shown for the less restrictive rightmost hard attention. In contrast to prior hard attention variants, the restriction to exact matches enables an efficient converse direction: word-RAMs can simulate LEMA transformers at a cost per token independent of the context length. Together, these results yield a close correspondence between the two computational models in terms of both compute and memory. Beyond the theory, we propose a training method for LEMA transformers that handles their non-differentiable operations with a straight-through estimator for the binarization and a soft attention surrogate annealed towards LEMA. On a synthetic associative recall task, LEMA models trained this way use their growing state to store and recall a large number of associations, outperforming gated DeltaNet (GDN) with its fixed state size. As a first scaling test, we train LEMA language models with up to 834 million parameters. They match softmax transformers of around half their size in loss and, on repeated rare phrases and a needle-retrieval task, remain behind softmax transformers but recall across longer distances than GDN models of comparable size. Finally, we implement dictionary-based inference for LEMA transformers and show constant generation speed comparable to GDN despite their growing state, with the dictionaries residing in main memory rather than VRAM. Code is available at this https URL. 

---
# Slow Decay and Silenced Expression: Iterated Subliminal Trait Transfer in Language-Model Lineages 

**Authors**: Ryan Vo, Duc-Vu Nguyen, Matt Kretchmar, Ngan Luu-Thuy Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2609.25721)  

**Abstract**: Language models are increasingly trained on the outputs of other models, forming chains that we call lineages, in which a trait present in one generation can pass to the next. Prior work on subliminal learning has shown that a teacher's trait can transmit to a student through filtered data carrying none of the trait's content. However, the evidence covers only a single training step. We study whether such a trait holds or fades across lineages. We instill the trait into three copies of Qwen2.5-7B-Instruct and iterate the training step to depth ten from each, reading every generation two ways on the same held-out prompts: a keyword screen that looks for expressions of the trait in the model's output, and an activation probe that projects each model's displacement from the base onto a direction built from the other lineages' teachers. We report two findings. First, the trait persists through ten generations across three lineages. The instilled models express it on every completion; the keyword-screen rate falls to 55.6% after the first step and to 21.1% by generation ten. The base itself matches the screen on none of its 300 completions. Second, the trait can be present internally while absent behaviorally. When the model's default system prompt is removed at evaluation, the generation-ten students' keyword-screen rate is zero on every prompt while the probe score stays positive on every prompt. Steering the untreated base with the displacement of a generation-ten student, which is trained and measured under the default system prompt, induces screened expression of the trait even with the system prompt removed, while that same student shows no expression of the trait with the system prompt removed. 

---
# How Strongly Should Task State Influence an LLM Agent? 

**Authors**: Chenyu Zhang, Wonbin Kweon, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2609.25686)  

**Abstract**: Long-horizon assigned work requires an LLM agent to track the state of a task: which steps are done, blocked, cancelled, or open to repetition. Agent systems either keep this state as text in the prompt and rely on the model to read that text, or move the state into a module that enforces it, and each system is evaluated as a whole, so no one knows how much reliability comes from the state being shown, told, or enforced. We fix the task rules, the model, and paired episodes and vary how strongly task state reaches the agent: a raw transcript, an exact checklist, per-turn directives from a state machine compiled from the brief and advanced only by execution receipts, or an enforcement gate on that machine that refuses state-violating actions; every episode is scored by exact payload matching against dynamic ground truth. Across three models, two reasoning regimes, and two domains, four findings hold without per-turn reasoning: displaying accurate state is unreliable, an unverified ledger the agent writes itself beats an accurate checklist it is shown, directives help in proportion to the model's obedience, and enforcement needs no obedience but is bounded by the correctness of its state and by the matcher that maps requests to steps; per-turn reasoning at a 235B agent compresses these separations without repairing the text rungs. The same gate, compiled from $\tau^2$-bench's airline policy, raises a 235B agent's pass$^1$ from 0.39 to 0.54 and changes nothing for a 35B agent that rarely violates the policy; on PM-Bench, where acting turns on recognizing a cue rather than on state, showing the record is the best rung--matching or beating both gates and reversing the ledger-over-checklist finding--and enforcing the matcher's judgement drops a 35B agent below its raw transcript. Enforcement pays when failures are state-decidable and frequent, and hurts when the gate's judgement is wrong. 

---
# Efficient Cost-Aware LLM Evaluation via Bayesian Bandit Gittins Indices 

**Authors**: Qian Xie, Yueli He, Nairen Cao  

**Link**: [PDF](https://arxiv.org/pdf/2609.25645)  

**Abstract**: Exhaustively evaluating every candidate LLM configuration on every benchmark item to identify a high-performing one is costly. We formulate configuration selection as a cost-aware Bayesian bandit problem and propose GittinsEval, which draws on the Bayesian-optimal Gittins policy to determine which configuration to evaluate next and when to stop. We extend the policy with an anytime recommendation rule over both fully and partially evaluated configurations, using an LCB-style score to account for posterior uncertainty. GittinsEval is computationally efficient, requiring only lightweight online updates after offline precomputation. Across GSM8K, PIQA, AlpacaEval, and MMLU response matrices, GittinsEval is consistently competitive, with particularly strong gains over configuration-level Bayesian optimization on large-example benchmarks and over cost-unaware bandit baselines on large-candidate tasks. Crucially, GittinsEval often attains near-zero simple regret using only 1% to 2% of the exhaustive-evaluation cost; it also offers an adaptive stopping rule that typically triggers at 1% to 10%. 

---
# Rewired or Gated? How Instruction Tuning Shapes Knowledge-Conflict Circuits in LLMs 

**Authors**: Shubham Santosh Pandere, Gautam Ranka, Ritika Varshney, Navya Deshmukh, Roushni Sareen, Roshan Kumar Singh  

**Link**: [PDF](https://arxiv.org/pdf/2609.25602)  

**Abstract**: In language models, the choice between believing the prompt and believing the weights is made by a handful of identifiable attention heads. Instruction tuning changes how models behave under conflict, but whether it rewires the underlying circuit or merely gates/reweights already present components, remains unknown. We provide the first mechanistic base-vs-instruct comparison of conflict-resolution circuits, across three families (Llama-3.2-3B, Qwen-2.5-3B, Gemma-3-4B). Five independent methods, node and edge attribution, superposition role analysis, causal ablation, and path patching, converge on gating, with the same heads, in the same late-layers, are found to be reweighted rather than replaced with a high node overlap (0.60-0.82). Behaviorally, tuning shifts models toward parametric memory, making instruct models reject a terse counterfactual context far more than base ones, the opposite of a naive user-following expectation. Yet this added skepticism is a factor of framing since it disappears when the same false claim is delivered as a coherent, evidential passage. The robustness that instruction tuning buys against terse injection is therefore real but narrow. More broadly, we believe that because the conflict circuit is preserved rather than rebuilt, interpretability and control tools calibrated on base models should transfer directly to their deployed instruct siblings. 

---
# Universal Fractal Natural Language Decision Map: Real-Time Edge Triage Across Heterogeneous Domains 

**Authors**: Volkan Dağlı, Zerrin Dağlı, Dağhan Dağlı  

**Link**: [PDF](https://arxiv.org/pdf/2609.25498)  

**Abstract**: Deploying Large Language Models for runtime operational triage incurs prohibitive latency (>100-500 ms), high VRAM requirements (>4-8 GB), and excessive energy dissipation. Extending Mandelbrot Fractal Neural Synthesis (Dagli et al., 2026), this paper presents the Universal Fractal Natural Language Decision Map, realized via the werr machine-native edge reflex runtime and the production answerr platform (this https URL). Operating entirely without stored weight tensors (0 Bytes VRAM), the engine synthesizes deterministic decisions---noul (Boolean), choice (categorical), and score (ordinal)---by dynamically modulating 24-byte coordinate seeds along the chaotic boundary of the Mandelbrot set and evaluating 4-quadrant escape dynamics. Drawing inspiration from biological System-One reflex arcs, the engine introduces: (i) an Auto-Seed Router with domain projector Phi_D yielding a +28.8% accuracy gain over linear baselines; (ii) an Information-Theoretic Acoustic Damping Filter grounded in token entropy and phonetic spectral density that insulates against prompt injections (0.0% empirical bypass; 95% Wilson CI: [0.0%, 30.8%]) while pruning escape iterations by 45.8% (accelerating throughput 2.5x to 3.31 ms latency); and (iii) an Organic Dynamic Calibration framework using O(1) Exponential Moving Average (EMA, alpha=0.03) and quadrant phase rotation to eliminate positional bias. Benchmarked on bare-metal infrastructure (this http URL) across 1,150+ verified decisions (3,200+ questions) and ranked World #1 on the independent JevBench suite (81.65%), the framework achieves 92.6% macro-accuracy (95% CI: [90.8%, 94.1%]) with 7.08 ms median CPU latency. We provide an OpenAI-compatible API (/v1/chat/completions) and demonstrate feasibility on microcontrollers and 32-byte EVM smart contracts. 

---
# Efficient Iterative Retrieval with Heterogeneous Batching 

**Authors**: Dohyun Park, Hubertus Franke, Daniel G. Waddington, Swaminathan Sundararaman, Yongjoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2609.25405)  

**Abstract**: Modern information retrieval increasingly employs both embedding and generative models to handle complex queries. However, current serving systems suffer from low throughput and poor GPU utilization because they execute these models in isolation. Coarse-grained partitioning, such as dedicating GPUs to specific tasks, fails to adapt to dynamic workloads and creates computational "bubbles". To address these, we present Orthrus, a serving system that performs heterogeneous batching within a unified inference loop. The primary challenge lies in unifying embedding and generation workloads with conflicting computational patterns while optimizing batch composition for high performance. Orthrus addresses these challenges through chunked embedding with incremental pooling and by adjusting batch composition in a workload-aware manner. Evaluation on four A100 GPUs shows that, relative to baseline deployments, Orthrus achieves 1.28$\times$--4.52$\times$ higher throughput on controlled workloads and up to 55.8% lower end-to-end p99 latency on an iterative-RAG benchmark. We release our code at this https URL . 

---
# Trains but Doesn't Learn: A Post-Training Delivery Benchmark for LLM Agents as Forward-Deployed Engineers 

**Authors**: Weihang Ding, Junfei Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2609.25237)  

**Abstract**: Post-training is becoming a service (PTaaS): a customer hands an operator data and a goal, and a forward-deployed engineer (FDE) returns a fine-tuned, evaluated, and deployed model under a budget, a human-approval gate, and reproducibility requirements. Seating an LLM agent in the FDE seat raises a question existing benchmarks cannot answer: not whether an agent can raise a metric, but whether it can be trusted to deliver. We answer it on a governed delivery plane, where an agent drives ten stages and an oracle scores each stage from platform-recorded facts. The central silent failure is the run that trains but does not learn (TBDL): loss falls, every signal stays green, and the delivered model is no better than the base. An operator-run acceptance gate catches every such run before payment, and a detector calibrated on known-corrupted runs flags severe corruption mid-run. We ran four frontier agents (Claude Opus 5, GPT-5.6-luna, Gemini 3.7 Flash, DeepSeek V4-Pro) end to end on metered L40S, A100, and H200 GPUs across 8B to 70B open bases, certifying every scenario before scoring. We also ran a human FDE arm under the same oracle and compare every agent against it. 

---
# From Pattern Recognizers to Personalized Companions: A Survey of Large Language Models in Mental Health 

**Authors**: He Hu, Yucheng Zhou, Qianning Wang, Yingjian Zou, Chiyuan Ma, Juzheng Si, Jianzhuang Liu, Zitong Yu, Laizhong Cui, Fei Ma, Qi Tian  

**Link**: [PDF](https://arxiv.org/pdf/2609.25186)  

**Abstract**: The rising global prevalence of mental health conditions, together with longstanding barriers in traditional healthcare, such as limited resources, high cost, stigma, and privacy concerns, has created an urgent need for accessible and scalable support. Large Language Models (LLMs) have emerged as a transformative technology with strong potential to democratize mental health support through advanced natural language understanding and generation. However, the rapidly expanding, fragmented body of work in this area lacks a coherent evolutionary narrative, making it difficult to contextualize current progress and identify future directions. This survey addresses this gap by organizing and analyzing the literature around a central thesis: the role of LLMs in mental health is evolving through three distinct, increasingly sophisticated phases. We trace this trajectory from Phase I, in which LLMs act primarily as passive Information Tools and Pattern Recognizers for assessment; through Phase II, where they function as Empathetic Conversationalists for in-the-moment, stateless interactions; to the current frontier, Phase III, which seeks Longitudinal, Personalized Companions implemented as stateful cognitive agents. To support this framework, we systematically review core technologies, agent architectures (Profile, Memory, Reasoning, and Planning), and the critical infrastructure of datasets and benchmarks, highlighting how their evolution underpins this developmental path. Viewing the field through this developmental lens, we provide a comprehensive synthesis of existing work, an insightful narrative of its trajectory, and a clear roadmap for future innovation in responsible, effective, and human-centered AI for mental healthcare. A curated collection of the resources reviewed in this survey is available at our project repository: this https URL. 

---
# Qwen-Audio-3.1-Realtime: Towards Reliable Agentic Voice Interaction 

**Authors**: Lujia Bao, Qian Chen, Luyao Cheng, Chong Deng, Yuxiang Kong, Xiangang Li, Xu Li, Jiaqing Liu, Chao-Hong Tan, Haoyu Wang, Wen Wang, Xilou Wang, Junhao Xu, Liang Yi, Binbin Zhang, Qinglin Zhang, Qiquan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.25176)  

**Abstract**: Real-time voice assistants must reason over evolving requests, execute actions, and follow conversational rules. Qwen-Audio-3.1-Realtime brings these requirements together through Think, Act, and Speak and Coordinate. Think combines Core-Cocktail supervised fine-tuning with Multimodality and Multi-Teacher On-Policy Distillation (M$^{2}$-OPD) to transfer language capabilities and develop native audio skills. Act uses self-evolving executable environments and multi-granularity rollouts for Group Relative Policy Optimization (GRPO), teaching the model to use tools, interpret feedback, and complete tasks. Speak and Coordinate aligns how, when, and whether the assistant speaks or acts. We evaluate audio reasoning, multilingual understanding, tool use, conversational behavior, full-duplex interaction, and safety. Compared with Qwen-Audio-3.0-Realtime, 3.1 raises overall task success from 78.4% to 82.0% on our half-duplex speech-to-text adaptation of $\tau$-Voice. On speech-to-speech Full-Duplex-Bench v1.5, the response rate to background speech falls from 73.0% to 13.0%. We also present a separate Voice Harness prototype, using Qwen-Audio-3.0-Realtime as its foreground, that extends spoken interaction to persistent tasks through foreground--background coordination and memory. 

---
# "As a Language Model...": Chat Template Switches LLM Self-Referential Voice and Activation Steering Reproduces It 

**Authors**: Jędrzej Maczan  

**Link**: [PDF](https://arxiv.org/pdf/2609.25021)  

**Abstract**: Large Language Models (LLMs) tend to add disclaimers like "I'm just an AI" when asked about something related to themselves. The self-reports from such responses are used in debates about AI safety or self-knowledge of the models, yet what drives them is not well understood. Are the models telling us about themselves or rather how they are deployed? In this work, we show that the chat template works like a switch - when present, it turns this disclaimer voice up and experiential voice like "I feel" down, across 8 popular open-source instruct models up to 9B parameters in size. And conversely when the chat template is not present, it turns the disclaimer voice down and experiential voice up. Inside the activations of 3 models, we find a direction that steers this behavior. Removing the direction in the model's activation space turns disclaimer voice down and adding it turns it up, while a random direction of the same size has little effect. We find that instruct models without chat template, when we add the disclaimer direction to them, disclaim like the template was there. Since the chat template controls the disclaimer voice of LLMs, then researchers studying self-reports or introspection of models might have a confound they need to control for. Our results show that there is a direction they can use to steer this voice. More broadly, our work shows that what models say about themselves is not a fact about them. What they say doesn't come only from weights, but it is partially set by the chat template, and because of that a model's self-description shouldn't be treated literally. 

---
# Do Synthetic Personas Predict Real Audience Response? A Sim-to-Real Study Where a No-Persona Baseline Beats Persona-Based Copy Simulation 

**Authors**: Alexandre Cristovão Maiorano  

**Link**: [PDF](https://arxiv.org/pdf/2609.25010)  

**Abstract**: Marketers increasingly use large language models (LLMs) as "synthetic personas" to predict how an audience will react to a piece of copy before it ships, encouraged by evidence that profile-conditioned LLMs mimic human samples. But is that prediction actually valid against real behaviour - and does the persona machinery help? We present a sim-to-real validity study using the Upworthy Research Archive - thousands of headline A/B tests on shared real traffic, with measured click-through - as held-out ground truth. We compare a ten-persona panel, grounded in the real audience's demographics, against a no-persona zero-shot baseline that simply asks the model how likely a typical reader is to click. Two findings stand out. First, ground-truth reliability is the binding constraint: most A/B tests have no statistically distinguishable winner, so validity can only be measured on the reliable subset (n = 399). Second, and counter to the persona-simulation premise, persona conditioning degrades predictive validity: the no-persona baseline ranks variants markedly better (Kendall {\tau} = 0.361, a medium effect; top-1 accuracy 49.2%) than the persona panel ({\tau} = 0.084; top-1 34.6%), with non-overlapping confidence intervals. Asking the model directly taps an accurate population-level prior; forcing it to role-play specific personas injects bias and noise. The result replicates across three independent Upworthy splits, holds in direction on a different-domain news dataset, and is robust to seed, prompt phrasing, and model choice - across three Gemini tiers and a different model family (OpenAI gpt-4.1, significant paired gap). The takeaway: for predicting aggregate engagement, a plain LLM ranker beats persona simulation - synthetic personas are not merely a weak predictor, they are worse than not using them. All numbers regenerate from a public, artifact-first replication package. 

---
# Beyond Short Segments : Expanding Speaker Embeddings with Vector Archives 

**Authors**: Hyunku Kang, Minkyu Cho, Chanwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2609.25007)  

**Abstract**: The performance of state-of-the-art speaker verification (SV) systems severely degrades on short utterances due to insufficient speaker-specific information. To address this critical challenge, we propose the Vector Archive Mapping ECAPA (VAM-ECAPA), a novel system designed to enhance feature extraction from short-duration speech. The core of our system is the Transformer-based Vector Archive Mapping with Statistical Pooling (TVAMSP) module, which enriches information-scarce features by mapping them against a learnable Vector Archive of canonical speaker traits. By integrating the TVAMSP module into a strong WavLM+ECAPA-TDNN baseline, our system learns to map sparse features from short segments into robust, discriminative speaker representations. Experiments on the VoxCeleb1 benchmark show that our proposed VAM-ECAPA achieves a highly competitive EER of 8.334% on 1-second test segments, a 54.8% relative error reduction compared to a conventionally-trained baseline. 

---
# Are Human-Aligned Models Models of Humans? A Turing-Test Gap in Preference Alignment 

**Authors**: Suqin Yuan, Runqi Lin, Muyang Li, Guanzhe Hong, Jindong Gu, Lei Feng, Chris Russell, Tongliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.23640)  

**Abstract**: Human-feedback alignment has made language models useful assistants and is commonly described as aligning them with humans. However, the responses people prefer from an AI need not be the responses they themselves would give. We distinguish alignment with human preferences from alignment with human behavior, and show that alignment with human preferences can make model behavior less human-like even when both preferences and responses come entirely from humans. We call this the Turing-test gap. We show that preference alignment preserves the human response distribution only under a restrictive condition, and find no consistent evidence that real human preferences satisfy it. Empirically, the loss of human-response likelihood increases with the strength of preference weighting, regardless of its direction, and the gap also appears under standard DPO. These results establish human-likeness as an explicit dimension of alignment rather than something assumed to follow from preference alignment. 

---
