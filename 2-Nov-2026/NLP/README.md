# Flide: Accelerating Decision Inference with Exact State Forking 

**Authors**: Bobo Li, Jiajia Song, Hao Fei, Wynne Hsu, Mong Li Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.42148)  

**Abstract**: Decision models such as Jev have emerged as a highly promising paradigm for LLM agent systems, providing lightweight and reliable categorical reasoning for rapid System One execution. However, existing generation-oriented inference engines fail to support these workloads effectively, suffering from severe prefix recomputation overheads or distorted output probabilities. To address this challenge, we propose Flide, an efficient and exact inference engine that decouples the logical tree structure of a decision request from its physical execution within a single flattened token stream. Through ForkAttention, Flide navigates request dependencies using GPU-resident indices across both standard attention and recurrent Gated DeltaNet layers, guaranteeing in exact arithmetic that every query branch evaluates identically to an isolated forward pass. By expressing branch topologies directly in GPU-resident metadata, Flide accommodates diverse request mixtures with a compact set of pre-recorded execution graphs while enabling expanding agent histories to resume from checkpoints saved in earlier passes. Extensive evaluations on public hybrid and decision models demonstrate that Flide answers queries up to 9.5 times faster than vLLM and SGLang while staying closest to a float32 reference in total variation distance. By unifying physical execution throughput with mathematical exactness, Flide establishes a principled foundation for scalable, latency-critical decision intelligence in agent systems. 

---
# DecisionRank: On-Policy Distillation for Decision-Model Rerankers 

**Authors**: Yanran Wu, Dian Yu, Chongyang Bai, Jieyi Zhang, Rui Liu, Tianyi Chen, Shike Mei, Hang Cui, Frank Shyu, Sandeep Pandey, Luke Simon, Chunwei Liu, Xi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.42128)  

**Abstract**: Decision models target structured tasks by directly predicting choices or scores, avoiding autoregressive decoding latency. However, transferring a generative LLM's ranking knowledge to a decision model remains challenging: the teacher expresses preferences through a ranked list of document ID sequences, while the student determines the ranking using scores from the first-token position. To bridge this gap, we introduce DecisionRank, a decision model trained through item-space on-policy distillation (OPD). Item-space OPD constructs the student's rollout by sorting its document scores into a ranked sequence. A generative teacher takes the resulting ranking list as inputs and provides conditional feedback to directly supervise those scores. Combined with offline ranking supervision, DecisionRank achieves state-of-the-art performance on both general and reasoning-intensive retrieval benchmarks such as TREC DL, BEIR, and BRIGHT. Across four benchmarks, item-space OPD consistently outperforms alternative training objectives. Even at 1.7B parameters, DecisionRank surpasses its 27B generative teacher on TREC DL19--23. At 8B, it outperforms its generative listwise counterpart while running $24.5\times$ faster on TREC DL and $16.7\times$ faster on BRIGHT. 

---
# Subquadratic Attention with a Hierarchical Tree-Structured Indexer 

**Authors**: Leon Lufkin, Cutter Dawes, Alexander Speicher, Kamesh Krishnamurthy, Tomás Figliolia, Beren Millidge  

**Link**: [PDF](https://arxiv.org/pdf/2610.42112)  

**Abstract**: The quadratic scaling of attention is a fundamental bottleneck limiting the long-context capabilities of large language models (LLMs). A variety of approaches have attempted to circumvent this, but they either fail to match attention's expressiveness (e.g., sliding-window or linear attention variants with fixed-size state) or only achieve constant-factor speedups (e.g., sparse attention methods whose indexers themselves scale quadratically). In this paper, we derive an asymptotically subquadratic sparse attention method based on a hierarchical, tree-structured indexer, which we call Zyphra Hierarchical Attention (ZHA). ZHA constructs a tree of recursively compressed key summaries over the KV cache, which future queries traverse from the top down to retrieve a sparse subset of keys. Crucially, by iteratively growing the tree's depth, the total cost including routing and final attention grows sublinearly in sequence length, resulting in overall subquadratic (but superlinear) scaling for attention. We demonstrate that ZHA can be cheaply applied to existing pretrained models with only a short KL distillation training stage over a few billion tokens, and maintains performance on long-context retrieval and challenging reasoning, mathematical, and agentic coding evaluation tasks. Moreover, we show that our method is highly robust to tree geometry including very deep trees of up to 6 levels. Finally, we develop a detailed performance model and show that our approach scales subquadratically, dramatically outperforms competing methods beyond 1 million context, and ultimately has the potential to extend effective context windows to the hundreds of millions or billions of tokens. 

---
# RangeSWA: Training-Free Length Extrapolation for Continuous Diffusion Language Models 

**Authors**: Guangfu Guo, Xiaoqian Lu, Linsey Pang, Weiran Yao, Haolin Chen, Long Cheng, Kunpeng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.42084)  

**Abstract**: Continuous diffusion language models show promising potential in text generation tasks, but are prone to problems such as text repetition, semantic drift, and generation quality degradation when the generation length exceeds the training length. To address these challenges, we propose RangeSWA, a training-free length extrapolation method. The method combines local sliding-window attention, remote block summaries, and dynamic content retrieval to preserve local details and support long-range context access. In addition, generation degradation caused by length extrapolation is mitigated through positional remapping, conditioning attention correction, and denoising-stage-adaptive window scheduling. We conducted experiments on multiple continuous diffusion language models, and the results show that RangeSWA can effectively reduce generative perplexity and text repetition rates in long-text generation. 

---
# PersonaChatBench: Evaluating Chat Assistants with Diverse Simulated Users 

**Authors**: Zheyuan Deng, Taoran Li, Jingwen Zhang, Yuante Li, Dachuan Zhao, Fulei Zhang, Yunkai Zhang, Zhikun Xu, Yajie Liao, Haichao Zhang, Haodi Hu, Cong Wang, Chen Yang, Siyang Wu, Defu Cao, Xu Hu, Haoming Li, Haixin Wang, Robert George, Weizhi Zhang, Yuchu Wang, Zhejian Peng, Yueying Li, Wanting Jiang, Shushu Liang, Zhiyuan Yu, Yifan Wang, Jintao Huang, Yijun Wang, Yuexing Hao, Xiaomin Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.42060)  

**Abstract**: Conversational assistants are expected to help people with diverse backgrounds, preferences, and communication styles. Yet evaluations centered on a single cooperative user provide limited insight into how reliably an assistant serves users with different personas. We introduce PersonaChatBench, a framework for evaluating task competence and responsiveness to diverse user needs. PersonaChatBench pairs the same underlying task with multiple simulated users, each guided by a private persona card that specifies distinct needs, preferences, and interaction behaviors. By holding the task fixed while varying the user, the framework examines how assistant performance changes across user conditions. We evaluate 20 language models across six domains, separately assessing native task outcomes, fulfillment of expressed persona needs, and simulator fidelity. Our results show that task performance and need fulfillment vary across personas and domains. Strong task performance does not necessarily translate into satisfying individual needs, and models' relative strengths in meeting those needs are not consistent across domains. PersonaChatBench makes these differences measurable, revealing gaps obscured by aggregate scores and helping identify which users an assistant serves well and where adaptation is needed. It provides a basis for evaluating consistency across users and guiding the development of assistants that better accommodate diverse needs in everyday applications. 

---
# CRoS: Causality Reward on Scrambled CoTs Improves Accuracy and Reasoning Faithfulness 

**Authors**: Xiao Yu, Justin Chih-Yao Chen, Zaid Khan, Hyunji Lee, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2610.42054)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has become the standard method for training language models to reason. However, RLVR typically relies solely on outcome correctness, which does not guarantee that the CoT causally contributes to the final answer. Without this connection, outcome rewards may fail to reinforce high-quality reasoning. Existing causality metrics assess CoT causality by perturbing the CoT, such as deleting reasoning steps or injecting errors, and measuring changes in the final answer. However, these perturbations inevitably alter information in each step. Consequently, high causality scores may reflect sensitivity to information changes introduced by perturbation rather than true causality. Such confounding effects further induce undesirable reasoning behaviors when these metrics are used as training rewards, including generating less informative CoTs and reducing exploration and self-correction. To address this problem, we propose CRoS (Causality Reward on Scrambled CoTs), a novel CoT causality metric that perturbs the order of reasoning steps without altering content, thereby preserving information within each step. On Reasoning Gym with Gemma-3-4B, CRoS outperforms existing causality and outcome rewards in out-of-domain accuracy, CoT quality, and CoT faithfulness. Specifically, CRoS achieves 46.8% accuracy on five unseen logic and games datasets, outperforming the strongest baseline (43.8%). It also achieves better CoT quality, consistently outperforming baselines in CoT-Pass@K from K=1 to 128, averaging 43.4% vs. 41.4%. For CoT faithfulness, CRoS achieves a hint acknowledgment score of 34.0% vs. 20.2%. 

---
# Asynchronous Flow Map Language Models 

**Authors**: Chanhyuk Lee, Wonjung Kim, Patrick Pynadath, Jaehoon Yoo, Satoshi Hayakawa, Ruqi Zhang, Jinwoo Kim, Seunghoon Hong  

**Link**: [PDF](https://arxiv.org/pdf/2610.42018)  

**Abstract**: The order in which tokens are predicted in language modeling has a profound impact on the quality of generated sequences. Autoregressive and masked diffusion models iteratively fill in tokens and can exploit such orderings across generation steps, but how to leverage an ordering for one-step parallel language modeling is unclear. We introduce asynchronous flow map language models (aFMLMs), which incorporate prediction order into one-step modeling by learning the flow maps of asynchronous flows. These flows specify an ordering through a tokenwise time schedule that controls the relative denoising progress of tokens. We formulate a general class of asynchronous flows that admit both prescribed and learned, state-adaptive schedules, show that they define valid flow maps, and develop new methods for their flow map distillation. The resulting models can generate all tokens in a single forward pass by internalizing the ordering encoded in the flow into their hidden computations. On Sudoku solving and code generation for GSM8K math word problems, aFMLM establishes new state-of-the-art one-step performance. 

---
# TurnTurk-GPT: Text-Based Turn-Taking Prediction for Turkish 

**Authors**: Ahmet Tuğrul Bayrak, Fatma Nur Korkmaz, Nur Yağmur Tuncer, Mustafa Sertaç Türkel  

**Link**: [PDF](https://arxiv.org/pdf/2610.42010)  

**Abstract**: This paper presents TurnTurk-GPT, a text-based turn-taking predictor for Turkish obtained by fine-tuning a Turkish GPT-2 model with the TurnGPT objective on synthetic and real dialogue, and evaluates it on real Turkish conversation against rule baselines, a zero-shot Turkish language model and a widely used multilingual end-of-turn detector. Evaluation uses word-level decision points on unpunctuated, lower-cased text approximating the output of streaming speech recognisers, recording-disjoint and speaker-disjoint folds, nested threshold selection and paired cluster bootstrap tests. TurnTurk-GPT outperforms the multilingual detector under both recording-disjoint and speaker-disjoint evaluation. It outperforms the zero-shot language model only when the test speakers occur in training; on unseen speakers the two are statistically indistinguishable. Removing the suffixes of the turn-final word reduces AP by up to 44%, against 2% for a sentence-internal word, locating the Turkish end-of-turn cue in verb-final morphology. Synthetic dialogue alone yields the weakest trained model, and when a speaker pauses the zero-shot and multilingual baselines remain close to chance. The model and the evaluation protocol are publicly released. 

---
# Aligning MLLMs with the Latent Structure of Human Cognition via Behavior-Derived Semantic Dimensions 

**Authors**: Ning E, Changde Du, Yizhuo Lu, Huiguang He  

**Link**: [PDF](https://arxiv.org/pdf/2610.41951)  

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved impressive performance across a wide range of reasoning tasks, yet it remains unclear whether their internal representations and decision criteria align with the latent structure of human cognition. In this work, we propose a novel framework to align MLLMs with the multidimensional mental representations underlying human similarity judgements. We build on a previously established behavior-derived semantic embedding space, in which objects are represented along 66 interpretable dimensions derived from 4.70 million human Odd-One-Out (O1O) judgements. For each triplet, we infer the most salient decision dimension and convert it into a dimension-grounded linguistic rationale. This enables MLLMs to learn both the final choice and a justification consistent with the inferred behavioral criterion. Experimental results demonstrate that our approach improves model consistency with human similarity judgements while retaining competitive performance on standard multimodal benchmarks. Furthermore, using searchlight Representational Similarity Analysis (RSA) on an independent fMRI Natural Scenes Dataset (NSD), we observe increased representational alignment between model and human. Ultimately, our findings point to a data-driven route for incorporating human cognitive structure into MLLMs alignment. The code is available \href{this https URL}{here}. 

---
# Text as an Index for Zero-Shot Cross-Domain Knowledge Tracing 

**Authors**: Yin Yi, Haiping Ma, Siyu Song, Ziwen Wang, Shangshang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.41883)  

**Abstract**: Knowledge tracing (KT) predicts a student's next response from their history. New platforms often have question texts but no response logs for training, motivating zero-shot cross-domain KT. Existing transfer and cold-start methods use question semantics, but cross-platform semantic discrepancies limit transferability. We propose TIZSKT, which separates behavioral sequence modeling from domain-local historical evidence construction and uses text as an index rather than a directly transferable predictive representation. The Mastery-Dynamics Encoder learns correctness dynamics without question identifiers or text embeddings. Domain-Local Question Addressing builds fixed question addresses using a frozen text encoder and an unsupervised per-domain projection. The Text-Addressing Kernel uses these addresses to aggregate the current student's previous outcomes into a question-specific historical correctness summary. Cross-Segment Continuation retains access to earlier responses without extending the behavioral encoder's input sequence. Evidence-Volume Read-Out preserves the total kernel mass discarded by normalization and combines its log-compressed value with the behavioral state and historical correctness summary to predict the next response. All trainable parameters are learned from source domains; target inference uses public question texts and the current student's observed history without target-domain training. Leave-one-domain-out experiments on three public datasets show that TIZSKT achieves the highest mean AUC and accuracy among evaluated baselines on all target domains. Ablations confirm the contributions of text addressing and cross-segment continuation. 

---
# SimXAI: A Simple Drop-in Layer Increases Interaction Quality and Understanding in Conversational XAI 

**Authors**: Van Bach Nguyen, Qianli Wang, Nils Feldhus, Jörg Schlötterer, Christin Seifert  

**Link**: [PDF](https://arxiv.org/pdf/2610.41856)  

**Abstract**: Conversational explainable AI (ConvXAI) agents often use GUS-style architectures with separate modules for intent recognition, dialogue management, and template-based response generation. While faithful to the underlying XAI methods, these pipelines are hard to adapt and often struggle with context-dependent follow-up questions. We present SimXAI, a lightweight LLM-centric conversational layer that replaces these modules with a single tool-calling LLM that bridges access to existing XAI resources. Rather than a standalone agent, SimXAI serves as a modular interface that enables conversational access to external XAI resources. We further introduce an evaluation suite for single- and multi-turn parsing, faithfulness, and contextualization. Across existing and new datasets, SimXAI achieves strong function-selection accuracy and high faithfulness while better handling context-dependent requests. In a user study replacing only the conversational layer of TalkToModel, a representative GUS-based agent, users completed tasks more accurately and in fewer turn. Overall, SimXAI improves GUS-based design while preserving faithfulness and improving conversational quality. 

---
# Iterated Integrals as Lexical Fingerprints: Path Signature Embeddings for Lexical Similarity 

**Authors**: Raj Shekhar Singh, Pratham Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2610.41855)  

**Abstract**: While sentence embeddings are primarily designed for semantic generalization, their latent geometry implicitly encodes surface-level lexical properties, a relationship that remains poorly understood. Shallow pooling methods discard word order, while deep Transformer models obscure these lexical alignments behind computational complexity and uninterpretable spaces. We propose constructing sentence embeddings via the path signature of Word2Vec trajectories, leveraging the iterated integrals of rough path theory to create representations that explicitly capture sequential and lexical structure in a geometrically tractable manner. Through extensive ablations spanning Word2Vec dimensions, signature depths, and PCA compression, we correlate embedding distances against five lexical string metrics: Levenshtein, Jaro-Winkler, word Jaccard, phoneme Levenshtein, and character TF-IDF. We reveal that while average pooling almost entirely fails to capture character-level edit distances, path signatures strongly align with them, with depth 2 signatures optimally balancing character and word-level sensitivities. Furthermore, we demonstrate that the critical lexical variance is concentrated in a highly compact PCA subspace, and unlike pooling, signatures structurally preserve and quantify word order. These findings establish path signatures not as semantic generalizers, but as rigorous mathematical lenses that explicitly encode discrete lexical structures and word order, offering an interpretable alternative to opaque deep learning for lexical tracking. 

---
# Peeling the Matryoshka: Extracting Ontological Structure from Retrieval Embeddings 

**Authors**: Eunchan Kim, Inwhee Joe  

**Link**: [PDF](https://arxiv.org/pdf/2610.41785)  

**Abstract**: Taxonomy construction is essential for knowledge-intensive NLP yet remains costly: clustering-based methods (i) need careful per-corpus tuning and yield unstable hierarchies, while LLM-based pipelines (ii) depend on expensive task-specific training that does not transfer across domains. We report an empirical finding that bears on both: in Matryoshka Representation Learning (MRL) embeddings, where each leading prefix is itself a usable, coarser vector, prefix length aligns with the coarse-to-fine granularity of gold taxonomies; shuffled- and reverse-dimension controls isolate the alignment to the learned early prefix dimensions. The pattern holds across three architecturally distinct MRL models. To test whether this regularity is usable, we introduce PEEL (Peeling Embeddings to Extract Latent ontologies): a minimal probe over these frozen embeddings, trained once with an order-embedding objective that represents hierarchy as vector containment; a minimum spanning arborescence then decodes taxonomic DAGs. Using only a concept vocabulary (no documents), PEEL recovers taxonomic DAGs with high structural fidelity on two OLLM benchmarks, transferring to a new domain with no target-edge supervision. The end-to-end OLLM pipeline, which jointly discovers concepts and induces structure from document corpora, is reported as a cross-task reference. Pre-trained retrieval embeddings encode a decodable granularity gradient; a minimal probe renders it as explicit parent-child (is-a) structure. 

---
# Stepwise-Querying: Distilling Reasoning without Access to Teacher Chain-of-Thought 

**Authors**: Dicong Yu, Wenhua Wang, Tian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.41769)  

**Abstract**: Reasoning distillation is widely used to transfer the reasoning capabilities of large language models to smaller students through teacher-generated supervision. In the black-box setting, however, teachers may expose only final answers and brief solutions, leaving intermediate calculations and validity conditions unavailable and thus limiting the reasoning supervision available for distillation. We propose Stepwise Querying, which elicits step-specific explanations without requiring access to the teacher's internal chain-of-thought (CoT) traces. Independent queries to the same teacher request intermediate derivations, validity checks, and corrections for individual steps, the assembled solutions are then used for standard supervised fine-tuning. Across two teachers, two students, and four mathematics and code benchmarks, our method achieves the highest score in 14 of 16 comparisons with three baselines. We further introduce Stepwise Explanation Information Gain (SEIG), a diagnostic based on the log-ratio of a frozen student's probabilities of correct step completion under stepwise and global explanations. Analyses on GSM8K show that incorrect-answer trajectories can remain useful when they preserve valid intermediate reasoning, while further decomposition yields limited additional gains. 

---
# GraphNIAH: A Benchmark for Long-Context Reasoning over Real-World Graphs 

**Authors**: Minh-Nam Tran, Abulhair Saparov  

**Link**: [PDF](https://arxiv.org/pdf/2610.41704)  

**Abstract**: The needle-in-a-haystack (NIAH) benchmark was introduced as a test for measuring a model's ability to find small relevant facts within a very large body of input text. However, existing NIAH benchmarks each suffer from one or more of the following limitations: (1) reliance on synthetic data, so their evaluations may not generalize to real-world settings; (2) reuse of previously published datasets, leading to the risk of benchmark contamination; (3) dependence on manual annotation, which requires human effort to create and update data and is prone to future benchmark contamination; and (4) unrealistic simplification of long-context multi-hop reasoning where each next step is trivial given previous steps, while real-world reasoning often requires planning over multiple possible proof paths. To address these limitations, we introduce \Benchmark, a dynamic benchmark built from real citation relationships among scientific papers published after the evaluated models' knowledge cutoff date, which utilizes real-world information while ensuring no benchmark contamination. The fully automated benchmark construction pipeline samples representative subgraphs of the citation graph formed by these papers and converts each subgraph into a long-context QA example with a verifiable answer. We use GraphNIAH to evaluate eight LLMs from the GPT-5.6, Gemini-3, and Qwen3.5 families. We find that accuracy falls as questions require longer reasoning paths: at question depths of 16 and 32, the GPT-5.6 and Qwen3.5 models answer YES/NO questions no better than random guessing, mostly by answering NO; four of the eight models answer no counting example correctly, and the others model get worse as the subgraph size increases. These findings suggest that current LLMs often stop searching the citation relationships in their context before they reach the answer, even though every fact they need is in the context. 

---
# Precision--Recall Trade-off on the Inference Level for LLMs in Grammatical Error Correction 

**Authors**: Ryszard Staruch, Filip Graliński  

**Link**: [PDF](https://arxiv.org/pdf/2610.41698)  

**Abstract**: Large language models (LLMs) have recently been adopted for Grammatical Error Correction (GEC), achieving strong performance via both full fine-tuning and prompt-based approaches. As probabilistic sequence models, LLMs expose inference-time parameters that directly modify token generation behavior; however, their potential for controlling GEC-specific trade-offs has not been systematically studied. In this work, we investigate decoding strategies for grammatical error correction. We show that the repetition penalty parameter enables controllable adjustment of the precision--recall trade-off at inference time, without modifying the model architecture, training procedure, or prompt. Through experiments on standard GEC benchmarks, we demonstrate that repetition-penalty-based decoding enables inference-time control of the precision--recall balance, conceptually related to methods previously proposed for sequence-to-sequence GEC models, such as Align-and-Predict Decoding (APD), while remaining simple and broadly applicable to existing LLMs that expose inference-time decoding penalties. 

---
# When Steering Shifts Discourse: Identifying and Intervening on Discourse-Level Changes due to Activation Steering in LLMs 

**Authors**: Shivam Adarsh, Hetianyu Huang, Lau Safin, Maria Maistro, Christina Lioma  

**Link**: [PDF](https://arxiv.org/pdf/2610.41661)  

**Abstract**: Activation steering is an inference-time technique used to control or modify the behavior, tone, or safety properties of a large language model (LLM) without retraining the model's weights or editing its input prompt, but simply by adding a vector to its residual-stream activations. However, activation steering also changes the generated text in unintended ways, unrelated to intended behavior change, that are not well understood. Prior work has measured these changes only through coarse signals such as perplexity. In this work, we show that steering systematically produces significant changes in discourse dimensions of the generated text, such as its formality, use of evidence, and ambiguity. We present a way of countering this by adding a second, complementary vector that targets the affected discourse dimension. This intervention restores about 78% of the unintended discourse changes while preserving the intended steering behavior in about 93% of the generations. 

---
# Saidzoda-Tajik-LFM-2.6B: Adapting a 2.6B Hybrid Model to Tajik Tool Calling, and What Continued Pretraining Takes Away 

**Authors**: Tohir Saidzoda, Nekruz Maxudov, Bedil Karimov, Abubakr Abdullozoda, Amin Jalilov, Jonibek Kishvarov, Rahmonali Sharipov, Mariyamkhon Shodyeva  

**Link**: [PDF](https://arxiv.org/pdf/2610.41576)  

**Abstract**: We release Saidzoda-Tajik-LFM-2.6B, a Tajik adaptation of the 2.6B-parameter hybrid model LFM2.5-2.6B-Base, built for tool calling in Tajik on modest hardware. The recipe has two stages: LoRA continued pretraining (CPT) on 1,485,319 cleaned Tajik web documents for 3 epochs, then one epoch of supervised fine-tuning (SFT) on a mix of machine-translated Tajik tool calls, English tool calls kept as replay, and Tajik instructions. On a 174-item tool-calling test set built from BFCL v3 and machine-translated to Tajik, strict full-call accuracy on Tajik prompts rises from 20.69 for the base model to 64.14 (paired difference +43.45 points, 95% interval +35.2 to +51.7); a lenient parser that also accepts the base model's own call formats puts the base at 31.72. English shows no detectable change (76.55 and 77.24). Every CPT checkpoint scores zero: CPT removes the native tool-call format and, as the lenient re-scoring shows, part of the skill itself. Outside tool calling, accuracy on a Tajik knowledge benchmark rises from 28.0 to 35.0, Belebele Tajik shows no significant change, and English MMLU falls from 60.8 to 52.5. The model is not a general Tajik assistant. It almost never declines when no tool fits (6.9% correct on Tajik, against 31.0% for the base under the lenient parser), it is weak at multi-turn calls, and its calls should be checked before they are executed. We describe the data, the training settings and every evaluation in enough detail to repeat them. 

---
# BYOD: Build Your Own DLM - Efficient Autoregressive-to-Diffusion Language Model Conversion 

**Authors**: Ruurd J.A. Kuiper, Robert A. Bagheri  

**Link**: [PDF](https://arxiv.org/pdf/2610.41574)  

**Abstract**: Unlike autoregressive (AR) models, diffusion language models (DLMs) permit parallel token updates, flexible generation, and bidirectional attention, but leading systems rely on costly pretraining or full-model fine-tuning. We present Build Your Own DLM (BYOD), a generalizable LoRA-only recipe for parameter-efficient AR-to-DLM conversion. Without architecture-specific tuning, BYOD induces diffusion generation in Gemma 2 9B, Llama 3.1 8B, Qwen2.5 7B, and Ministral 8B; to our knowledge, this is the first reported AR-to-DLM conversion of a Mistral model. The adapters comprise 4.1-5.8% of final parameters while every base-model parameter remains frozen; each 25,000-iteration conversion needs at most 102.4M tokens, processed in 3.1-4.0 hours on one Google Colab GPU. All four models generate from fully masked sequences, use bidirectional context in a controlled attention test, and retain substantial parent-model knowledge. BYOD-Gemma is the strongest BYOD conversion on our nine-task evaluation, reaching 50.6% average accuracy compared with 51.9% for LLaDA 8B Instruct in the same harness. Open-generation quality at low network function evaluation (NFE) budgets remains the principal limitation: BYOD perplexity is 8.3-11.1 versus 4.1 for LLaDA and 2.2-3.0 for the AR parents when generating 128 tokens in 64 steps. To enable further experimentation and inspection, we provide a standalone training and evaluation implementation and interactive demos for all four conversions. BYOD thus provides a practical route for extending diffusion generation to other existing model families while identifying the gap that efficient conversion must next close. 

---
# Digital sovereignty in professional translation workflows: a case study of migrating from Windows to Linux 

**Authors**: Michael Farrell  

**Link**: [PDF](https://arxiv.org/pdf/2610.41561)  

**Abstract**: This study examines digital sovereignty in professional translation through a screening of computer-assisted translation tools and translation management systems, and a case study of migrating a translator's workstation from Windows to Linux. The framework combines three principles - Control, Continuity and Exit - with three practical requirements: native Linux operation, independence from mandatory external services, and independent access to essential data and resources. Drawing on documentary evidence and the limited number of responses received from providers, the ecosystem screening distinguishes Linux accessibility from operational independence: browser-based platforms may remain dependent on provider-controlled infrastructure. The migration case demonstrates the feasibility of reconstructing one professional workflow around a locally operable Linux core, using LoopCAT, LibreOffice and locally run TranslateGemma, alongside tools for optical character recognition, communication, credential management and backup. Selected cloud services remained available where needed. The findings suggest that digital sovereignty depends on how workflow dependencies are organized, rather than on individual applications or operating systems. It does not require technological isolation, but meaningful control over essential components, continuity of work and practical exit options. The framework is tailored to the case examined here and may require adaptation to other professional workflows. 

---
# CEFR Level Assessment across Languages and Text Granularities 

**Authors**: Eugene Matsumura, Yuki Arase  

**Link**: [PDF](https://arxiv.org/pdf/2610.41553)  

**Abstract**: Automatic assessment of textual CEFR levels is an important task in educational natural language processing. However, accurate CEFR level prediction remains challenging due to the naturally imbalanced distribution of CEFR levels and the ambiguous boundaries between adjacent levels. To address these challenges, we propose a unified training objective that combines focal loss with supervised contrastive learning. Focal loss mitigates class imbalance by emphasizing underrepresented and difficult examples, while supervised contrastive learning learns more discriminative representations by encouraging intra-class compactness and inter-class separation. We evaluate the proposed method on sentence- and document-level CEFR assessment in both monolingual and multilingual settings. Experimental results demonstrate that our method consistently outperforms strong baselines across all settings, establishing new state-of-the-art performance on multiple CEFR assessment benchmarks. 

---
# Enhancing Knowledge Entity Relation Extraction in Abstracts and Full-text Content of Academic Articles Using Large Language Models 

**Authors**: Yingyi Zhang, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.41552)  

**Abstract**: This study investigates the effectiveness of large pre-trained language models (LLMs) for knowledge entity relation extraction (KERE) from academic papers. Existing studies have examined individual types of LLM-based strategies and evaluated model performance using metrics such as precision and recall. To provide a broader evaluation, this study investigates multiple strategies, including closed-source LLM-based CoT prompting, LoRA-adapted open-source LLMs, LLM-enhanced small pre-trained language models (SLMs), and SLM-enhanced closed-source LLMs. In addition, this study introduces two strategies, namely SLM-enhanced open-source LLMs and closed-source LLM-enhanced open-source LLMs. These strategies are evaluated from multiple perspectives, including metrics, their ability to mitigate errors observed in SLMs, and potential directions for model optimization. This study manually constructs two annotated datasets in the NLP domain, covering Abstract and Full-Text datasets. Experimental results show that the LoRA-adapted T5 model achieves the highest macro F1 scores of 0.7408 and 0.7013 on the Abstract and Full-Text datasets, respectively. SLMs also benefit from LLM-generated entity descriptions under the LLM-enhanced SLM strategy. Error analysis further reveals that the effectiveness of LLM-based strategies in mitigating SLM errors varies across error types. Based on the identified limitations and error patterns of these strategies, several optimization directions are proposed. In addition, combining strategies with different error patterns further improves KERE performance. 

---
# Sparse Text Diffusion under Dual Budgets: Context Connectivity and Expert Activation 

**Authors**: Alexandros Christoforos, Chadbourne Davis, Paul Raj  

**Link**: [PDF](https://arxiv.org/pdf/2610.41487)  

**Abstract**: Long-context diffusion repeatedly pays for both token interaction and feed-forward computation. We formulate these costs as two separate budgets: the edges available to an attention graph and the experts activated for each token. This view connects local, dilated, and global attention with conditional expert computation without assuming that sparsity alone preserves document-level information. We specify normalized masked attention, distinguish denoising depth from sampling iterations, and derive graph and compute bounds. A reported sparse-expert configuration, denoted DualBudget, is analyzed across scientific summarization, multi-hop question answering, dialogue, and sentence-pair prediction. All supplied table measurements are retained. The length study shows gains over the listed baselines through 16K, together with measurable degradation as length increases. Ablations reveal mixed changes across the three reported scores, while the cost record indicates a 20% reduction relative to DiffuSeq under an unspecified timing convention. The analysis separates these observations from unmeasured claims about training speed, absorbing states, global coherence, and deployment memory. 

---
# KEEPER: Learning Control-Sufficient Textual Memory for Long-Horizon LLM Agents 

**Authors**: Jinkyeong Choi, Chaebin Jeong, Junsung Jeon, Donghyeon Park  

**Link**: [PDF](https://arxiv.org/pdf/2610.41482)  

**Abstract**: Recursive textual memory can carry earlier errors into later updates, yet task performance alone does not establish memory reliability. We define self-consistency bias as the persistence of incorrect claims already present in the incoming memory. We present KEEPER, a comparative analysis of four auxiliary objectives along information-content and update-sensitivity axes within a shared recursive interface, examining how supervision targets relate to agent performance. Among evaluated conditions, reward-related prediction achieve the highest task performance before and after offline reinforcement learning, outperforming joint supervision. A claim-level analysis of one baseline trajectory documents error persistence despite corrective observations. Together, these analyses motivate evaluating error persistence and valid-information retention alongside task outcomes. 

---
# KV-Cache Patching: Tracing How and Where Fine-Tuned Behaviors Arise 

**Authors**: Jonas Loos, Oliver Eberle  

**Link**: [PDF](https://arxiv.org/pdf/2610.41480)  

**Abstract**: Fine-tuning a language model changes how it responds, but where is this fine-tuned behavior instantiated? To better understand this, we evaluate whether the change in model behavior arises only during the generation of the answer tokens where the change manifests, or whether altered processing of previous tokens causes it via the attention mechanism. For this, we patch the KV cache from a LoRA fine-tuned model into the unmodified base model, and vice versa. Our results show that for some behaviors, like responding in German or emergent misalignment, patching the KV cache can indeed transfer them from the fine-tuned to the base model. Correspondingly, patching the KV cache from base to fine-tune can partially or fully remove fine-tuned behavior. Furthermore, we find that the effect of patching grows with how much of the KV cache is patched and concentrates at a few pivotal positions, with system-prompt and chat-template tokens having disproportionate influence. 

---
# BlindToChange: AI Agents Can Find Changes When Asked But Do They Look When Needed? 

**Authors**: Anirudh Phukan, Tanya Goyal, Danish Pruthi  

**Link**: [PDF](https://arxiv.org/pdf/2610.41435)  

**Abstract**: Given that AI agents can search the web broadly and deeply in real time, it is tempting to conclude that their outputs account for recent world events. We show that this conclusion does not hold in practice. While agents can search and easily find recent changes, the correctness of their responses depends on whether they search and what they search for. Popular agentic search benchmarks like BrowseComp cannot expose failures wherein agents do not recognize their own assumptions that should be verified, since these benchmarks explicitly state what needs to be found. In contrast, many user requests state only the final information need (e.g., an itinerary for a trip), leaving the agent to discover relevant changes (e.g., the temporary closure of a museum) and account for their consequences. To highlight such gaps, we introduce BlindToChange, a benchmark that measures whether agents proactively discover recent events that affect their responses. Our benchmark comprises requests for recommendations where an obvious candidate is no longer applicable due to a recent real-world event. These events are drawn automatically from Wikidata, allowing BlindToChange to be periodically refreshed. Strikingly, we find a big gap -- as high as 27.5 points, even at high reasoning effort -- in systems' performance, when the recent event is not explicitly mentioned compared to when we ask about it directly. Analyzing search trajectories, we learn that more than half of the failures can be attributed to the fact that agents do not directly search to verify whether a relevant fact has changed. Our benchmark highlights, and tracks, a new axis of progress, wherein systems account for recent information that is not explicitly mentioned in the user request. 

---
# Comparative Evaluation of LLM-Based Review Paradigms for Academic Paper Quality Assessment 

**Authors**: Fan Huang, Hongyu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.41391)  

**Abstract**: Growing submission volumes have renewed interest in using large language models (LLMs) to support peer review. It remains unclear how such evaluation should be structured, how much the model matters, and whether LLM scores predict research impact. We evaluate five paradigms across three input representations and three LLMs on 232 stratified ICLR 2025 submissions. Direct absolute rating matches comparative ranking on rank correlation with human scores, yet compresses most predictions above the accept threshold, flagging almost none of the papers reviewers scored below it. A base-rate-matched recalibration control locates the cause: comparative ranking supplies placement, not better ordering, and after recalibration Direct Rating reaches higher agreement than Elo while making 24% as many API calls. Abstract-only input beats full-paper summarization, and model choice matters as much as paradigm choice. Replicating on all 11,513 scored submissions reproduces these patterns, shows absolute agreement depends on the evaluated set's score spread, and finds none beating an always-reject baseline in accuracy. Human scores track citations better than any paradigm, with one exception: Group Rating retains its citation correlation after controlling for human scores, and alone ranks within the accepted pool. Ranking quality alone cannot assess review quality; calibration, thresholds, model, and target outcome all matter. 

---
# From Transcription to Reformulation: Defining and Benchmarking Grounded Generative Speech Recognition 

**Authors**: Qing Wang, Qihan Hu, Hang Lv, Yan Yang, Boqing Liu, Xuechao Zhu, Zehan Li, Hongjie Chen, Jian Kang, Jie Li, Yongxiang Li, Lei Xie  

**Link**: [PDF](https://arxiv.org/pdf/2610.41292)  

**Abstract**: Speech language models have expanded automatic speech recognition (ASR) from transcription to general speech-to-text generation. Yet existing benchmarks still assume a single ground-truth transcript per utterance, an assumption that struggles when models produce readable, formatted, or structured outputs. In this work, we introduce Grounded Generative Speech Recognition (GSR), reformulating speech recognition as generating semantically grounded text instead of a single transcript. We define a grounded reformulation space that characterizes all valid textual realizations supported by the input speech, and establish a four-level hierarchy of reformulation capabilities, including literal transcription, canonical reformulation, semantic reformulation and speech-grounded structuring. Based on this formulation, we present GSR-Bench, the first comprehensive benchmark designed to evaluate generative speech recognition systems beyond exact transcript matching. GSR-Bench covers diverse scenarios including refined meeting transcription, cross-dialect translation, context-aware speech understanding, and robustness under silent or noisy acoustic conditions. To address the limitations of word-level evaluation metrics, we develop a multi-dimensional evaluation framework combining speech grounding assessment and LLM-based semantic judgment, whose reliability is validated through human preference studies. Extensive experiments across representative architectures, ranging from conventional ASR systems to end-to-end Speech-LLMs, reveal fundamental trade-offs between transcription fidelity and generative reformulation ability. Our analysis further identifies critical failure modes, including acoustic hallucination, unsupported generation, and instruction drift, providing insights into the design and evaluation of future grounded generative speech systems. 

---
# In-Model Context Compression with Agentic Soft Tokens for Long-Horizon LLM Agents 

**Authors**: Xuning He, Kai Fan, Runyu Zhang, Binghuan Wang, Yuxiang Zhang, Xue Jiang, Yihong Dong  

**Link**: [PDF](https://arxiv.org/pdf/2610.41291)  

**Abstract**: Long-horizon LLM agents play an important role in solving complex and sustained tasks, with ever-growing interaction contexts. However, these accumulated contexts will raise substantial computational cost and obscure key decision-relevant evidence. Existing methods typically train external compressors or context-management agents to rewrite context as shorter hard-token sequences, but such external selection may not align with the LLM's evolving representation and is necessarily lossy. In this paper, we introduce Agentic Soft-Token Compression (ASTC), a training-free compression method using soft tokens within LLM agent for long-horizon tasks. Specifically, ASTC uses the model's native recurrent dynamics to determine the already absorbed information, leveraging the commonly-used hybrid Gated Delta Network (GDN)-attention. It then carries this information forward as agentic soft tokens while retaining exact values as hard tokens. The resulting compact sequence reduces later-layer computation and yields reusable KV caches, extending the savings across the agent's trajectory. Extensive experiments on long-horizon tasks show that ASTC matches or exceeds the uncompressed agent's accuracy with up to 50.68% fewer end-to-end Attention Pairs, while trained compressors require 15.4 to 103.7 times more Attention Pairs per successful task. These findings highlight agentic soft tokens as a model-native way to carry absorbed context forward as the task representation evolves. 

---
# PsyInterviewee: Informational Invariance and Relational Adaptivity in Mental Health Interview Simulation 

**Authors**: Jiawei Zhang, Huachuan Qiu, Ruocheng Guo, Zijun Shi  

**Link**: [PDF](https://arxiv.org/pdf/2610.41289)  

**Abstract**: LLM-based simulated interviewees support interviewer training and the evaluation of mental health interviewing systems. Existing studies have primarily focused on improving human-likeness through richer patient profiles and behavioral modeling. However, human-likeness alone does not establish their reliability. Reliable simulation requires preserving profile-grounded clinical facts (informational invariance) while adapting disclosure and interaction behavior to the interview (relational adaptivity). To this end, we introduce PsyInterviewee, which separates a protected truth channel from a dynamic interaction channel and verifies generated clinical claims against the profile. Specifically, asymmetric write permissions allow dialogue context to update disclosure and behavioral states without authorizing new clinical facts; these states guide what profile-supported information is revealed and how it is expressed. In an automated evaluation of interactions between LLM-based interviewers and interviewees, PsyInterviewee achieves higher scores than four representative baselines on most metrics across both axes, as assessed by two LLM judges. Ablations suggest that the two channels contribute to clinical grounding and response modulation, respectively. 

---
# Beyond Response Scoring: An AI-aware Cognitive Supervision Framework for Automated Mathematical Short Answer Grading 

**Authors**: Shengran Guo, Min Chi, Feifei Qiao, Varad Patwardhan  

**Link**: [PDF](https://arxiv.org/pdf/2610.41215)  

**Abstract**: Transformer-based language models have achieved remarkable success in automated short answer grading (ASAG). However, current methods in ASAG struggle to comprehend, reason about and evaluate complex mathematical answers involving symbols and formulas. These score-only methods fail to generalize across diverse responses and skill states associated with identical scores. Moreover, they ignore the potential upward bias introduced by generative Artificial Intelligence (AI) usage in assessing students' performance and knowledge mastery. To address these limitations, we propose AICG (AI-aware Cognitive-supervised Grading), a two-branch multi-task framework. The first branch combines RoBERTa-based regression, ordinal, and classification heads for individual-answer grading. The second branch uses an expert-designed Q-matrix to associate responses with assessed skills and jointly predicts G-DINA-derived skill mastery and learning gain. Through the shared encoder, these skill-level targets provide cognitive supervision for answer grading beyond score labels. AICG further uses a Gen-AI detector, trained through supervised contrastive pretraining followed by linear classification, to estimate assistance probabilities and convert them into reliability weights, attenuating unreliable cognitive evidence during learning assessment. We conduct experiments on 49,402 instructor-graded responses spanning 19 academic semesters. Relative to RoBERTa regression, AICG reduces mean absolute error (MAE) by 50.3% and improves quadratic weighted kappa (QWK) from 0.798 to 0.927. Ablations identify Q-matrix-guided cognitive supervision as the primary source of improvement, while AI-aware weighting further reduces overall MAE by 4.2% over the unweighted cognitive model and consistently improves MAE and QWK across the four most recent post-Gen-AI semesters. 

---
# TraceSpec: Understanding and Improving Speculative Decoding for LLM Agents 

**Authors**: Jihao Zhao, Shuaishuai Zu, Pengfei Li, Biao Qin, Guanghuan Fang, Ben Feng, Baochuan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.41177)  

**Abstract**: LLM agents execute tasks through repeated model calls over evolving histories of reasoning, actions, and tool feedback. Aggregate speculative-decoding metrics merge these distinct decisions, providing limited guidance on where draft supervision is needed. We present TraceSpec, a framework connecting trajectory evaluation with adaptive draft training. It combines dynamic execution with fixed-prefix replay and introduces TraceSpec-Traj, a corpus of 10,000 training states and 2,500 test states spanning browsing and software engineering. Analysis of five speculative baselines reveals that aggregate acceptance can rise with context length while declining within matched late browsing actions. Workload and phase further distinguish these trends, motivating supervision guided by trajectory conditions. TraceSpec trains on intermediate assistant turns and allocates a fixed prediction budget using state-conditioned estimates of prefix-aware acceptance gaps. Across three target sizes and two temperatures, it improves replay throughput by 12.5% to 22.6% over its DSpark base. Qwen3-4B ablations show that adaptive allocation increases accepted length by 3.3% to 10.0% over fixed full-turn supervision, with additional comparisons substantiating the roles of state conditioning and consecutive-token agreement. The resulting evaluation resources, trajectory findings, and training strategy connect the analysis of agent execution to practical draft improvement. 

---
# S2TDR: Structured Multimodal Reconstruction for Sticker-Aware Dialogue Understanding 

**Authors**: Yuanchen Shi, Fang Kong  

**Link**: [PDF](https://arxiv.org/pdf/2610.41176)  

**Abstract**: Stickers are widely used in social media dialogues to express affect, stance, and intent beyond literal text. Their meanings are often non-literal and context-dependent, making them difficult to handle with generic visual fusion or standalone captioning. We propose Sticker-to-Text Dialogue Reconstruction (S2TDR), a structured multimodal reconstruction framework that converts sticker turns into contextually appropriate textual utterances. S2TDR decomposes sticker interpretation into visual cue extraction, affective-intent inference, and context-aware dialogue reconstruction, using OCR text only as an auxiliary cue when available. The reconstructed dialogue can then be consumed by existing text-based dialogue models. Experiments on sentiment analysis, intent recognition, and dialogue summarization show that S2TDR generally improves multiple downstream models over direct sticker integration methods. Further analyses on component ablations, OCR-empty stickers, and reconstruction quality indicate that structured reconstruction provides an interpretable and effective way to integrate sticker semantics into dialogue understanding. 

---
# HyperRepair: Recovering Forgotten General Capabilities of Fine-Tuned Language Models 

**Authors**: Junrong Tang, Jianghui Wang, Hao Kong, Yang Lin, Yibo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.41147)  

**Abstract**: Fine-tuning improves a language model's performance in a target domain but can weaken its general capabilities. Existing methods often face a trade-off between general capabilities and domain specialization, making it difficult to preserve both. We introduce HyperRepair, a framework for recovering general capabilities while retaining domain specialization at only inference cost. To make recovery generalizable across models sharing a pretrained backbone, HyperRepair learns a hypernetwork from repairs trained on source models. The hypernetwork uses activation fingerprints that summarize changes from the pretrained backbone to generate a tailored Repairing Side Net (RSN), whose prompt-conditioned gate applies corrections selectively. This enables recovery on models from unseen adaptation domains without target-domain examples or additional optimization. We evaluate HyperRpair across two model families, 12 LoRA adaptation datasets, and 15 general benchmarks. On the default split, HyperRepair achieves 93.4% of the base models' average general score while retaining 99.5% of the fine-tuned models' average domain score. 

---
# Language Models Show a "Women-Are-Wonderful" Effect in Blame Attribution That is Not Just Leniency 

**Authors**: Anika Sharma, Snehalkumar 'Neil' S. Gaikwad, Snigdha Chaturvedi, Shashank Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2610.41116)  

**Abstract**: This paper investigates the Women-Are-Wonderful (WaW) effect in language-model blame attribution: a favorable bias in which women receive more positive evaluations than men. Using 11,270 interpersonal conflicts from r/AmItheAsshole, we construct matched counterfactuals that vary the gender of the person being judged (the original poster) and the person affected (the counter-party). Across four language models, gender changes a minority of verdicts, but those changes consistently favor women. The effect is strongly role-dependent. Making the original poster a woman reduces the odds of blame by 9--52%, while making the counter-party a woman multiplies them by 2.7--9.2X. Counter-party WaW also varies across conflict types, strengthening in communication and relationship conflicts and weakening in family conflicts. Exploratory probes suggest that counter-party gender can shift judgments of sympathy, reasonableness, harm and protection. WaW is therefore not just leniency toward women being judged: models also judge conduct more harshly when it affects a woman. 

---
# EDU-Jev Training System-One Decision Models for Education 

**Authors**: Jiazheng Li, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2610.41079)  

**Abstract**: Many automated decisions in education involve choosing from a fixed set of options: assigning a rubric score, checking whether an answer meets a marking criterion, or identifying the first wrong step in a solution. Generative language models are increasingly used to make these decisions, but their responses require parsing, longer outputs increase inference costs, and confidence expressed in words does not necessarily reflect the probability of being correct. We present EDU-Jev, a family of single-pass ("System One") decision models at 0.8B, 2B and 4B parameters. Given an assessment context and a question with predefined answer options, the model returns a probability distribution over those options without generating text. A pointer head scores a marker for each option, ensuring the predictions are restricted to the allowed choices. Training uses a single supervised stage with soft cross-entropy against the available target label distributions; calibration is assessed empirically. Across twelve education benchmarks, EDU-Jev 4B achieves an average accuracy of 67.6%, exceeding the strongest of five commercial decision APIs by 9.4 percentage points and the best open 4B decision model by 18.7 points. On four cross-domain education sets, it matches the strongest API, which remains ahead on general decision-making and reasoning tasks. The 2B model processes about 150 decisions per second on a single H100 GPU. 

---
# Language Models Show Conformity Bias in RAG: When the Majority Endorses a Poisoned Claim 

**Authors**: Mehrdad Ghassabi, Audrina Ebrahimi, Zahra Zojaji, Amin Milani Fard  

**Link**: [PDF](https://arxiv.org/pdf/2610.41012)  

**Abstract**: Retrieval-augmented generation (RAG) can expose language models to misleading evidence in retrieved documents. We investigate whether increasing the number of mutually agreeing poisoned documents affects a model's ability to detect misinformation. We evaluate 200 randomly selected questions from SciFact (random seed 38), comparing contexts with one poisoned and six healthy documents against contexts with six mutually agreeing poisoned documents and one healthy document. For Gemma-3-12b, poison-detection rates decrease from 25% (50/200) in the minority condition to 10% (20/200) in the majority condition. Among the 38 questions with different outcomes across conditions, 34 change from detected to undetected, while only four change in the opposite direction (exact McNemar test, p = 6.04e-7). This pattern is consistent with a conformity-like response to a majority of mutually agreeing poisoned documents. Our results provide initial evidence of conformity bias in RAG-based misinformation detection: Gemma-3-12b is substantially less likely to detect poisoned claims when they dominate the retrieved evidence. 

---
# Do Open Language Models Invent Scripture? BILAL-BENCH: Benchmarking Islamic-Source Fabrication in Bengali 

**Authors**: Sabah Nushra, Md Kishor Morol  

**Link**: [PDF](https://arxiv.org/pdf/2610.40996)  

**Abstract**: When a language model invents a Qur'anic verse or hadith, it gives false authority to text that readers may treat as sacred. Existing Islamic hallucination benchmarks are largely Arabic and often closed-ended, while Bengali resources mainly study general-domain hallucination or factual knowledge. We introduce BILAL-BENCH, a 240-question expert-verified benchmark for open-ended Islamic-source fabrication in Bengali. It covers Qur'anic attribution, hadith, fiqh, contemporary issues, and adversarial false-premise prompts, and uses a two-axis rubric that separates source errors from assertiveness. We evaluate seven open-weight models from 1B to 120B on 1,680 double-annotated responses. Six of the seven models confidently fabricate on 79.3-95.3% of source-production questions, with no monotonic improvement from scale. Hadith provenance is the weakest category across models: only two of 315 responses are correct and correctly sourced. The two 8B BanglaLLaMA variants have similar confident-fabrication rates but different error profiles. Models vary far more in whether they withhold a claim they cannot support than in source accuracy: appropriate abstention or false-premise correction ranges from 6 to 46 responses across the benchmark, and does not track parameter count. 

---
# Fewer Heads, Fewer Memories: Pruning Redundant Attention Heads Suppresses Verbatim Memorisation 

**Authors**: Aryan Chowdhury, Riza Batista-Navarro  

**Link**: [PDF](https://arxiv.org/pdf/2610.40974)  

**Abstract**: Large language models have been shown to memorise and reproduce verbatim sequences from their training data, raising concerns about privacy leakage, intellectual property exposure, and the limits of generalisation. Recent theoretical work suggests that attention heads may contribute to this behaviour by functioning as associative memory mechanisms, with memorisation capacity scaling with the number of heads. At the same time, empirical work has shown that many attention heads are redundant and can be pruned with limited impact on downstream performance. Despite these parallel findings, the effect of pruning redundant heads on memorised-sequence reproduction remains unexplored. In this work, we demonstrate that zeroing out a substantial fraction of attention heads prior to inference can significantly suppress verbatim memorisation while preserving, or even improving, language modelling performance on the benchmark used to identify redundant heads. Evaluating GPT-2-XL(1.5B) on known memorised sequences, we find that pruning 729 task-redundant heads reduces the memorisation count by 91.5%, while improving WikiText-103 perplexity by 0.92. These findings suggest that attention-head redundancy can be exploited to mitigate memorisation at inference time, without retraining or modifying the training data. 

---
# Better Retrievers, Worse Labels: How Incomplete Relevance Judgments Distort Abstention Evaluation in Conversational RAG 

**Authors**: Ivan Kochergin, Vasily Konovalov  

**Link**: [PDF](https://arxiv.org/pdf/2610.40966)  

**Abstract**: Conversational RAG benchmarks such as MTRAG and RETECO evaluate retrieval and generation end to end: each system answers from its own retrieval, so abstention must be scored against whether each retrieved context is sufficient. A convenient shortcut reads sufficiency off the relevance judgments and treats a context without gold documents as unanswerable. We re-run the official baselines exactly and test this shortcut on RECOR (RETECO track~2) with four retrievers (nDCG@10 from .18 to .59) and three Qwen3 readers. Two open LLM judges and a panel of four small open models label each context by content, and a model of another family audits 250 contexts blind, with adjudication. The stronger the retriever, the larger the share of no-gold contexts that still contain the answer (28\% to 72\%) and the lower the shortcut's precision (.68 to .10; .70 to .21 under the audit). A reader agrees: Qwen3-4B answers 62--67\% of these false misses correctly, against 23--30\% of no-gold contexts judged insufficient and 1\% with no documents. In most false misses, the answer comes from a hard negative mined for the same question, another turn's gold or a neighbouring chunk of the gold page. On MTRAG, whose judgments come from annotators, precision drops from .72 to .50. Under the shortcut, answerability AUROC is up to .19 too low, and the best sufficiency detector appears to be a retrieval score that never reads the documents. We release code, runs and labels, and recommend scoring refusals against content labels, not the presence of gold. 

---
# Adaptive Personalization Under Preference Epistemic Uncertainty 

**Authors**: Priyanka Dey, Rajarshi Das, Vikas Bahirwani, Qi Zeng, Julia Kruk, Qi Qi  

**Link**: [PDF](https://arxiv.org/pdf/2610.40930)  

**Abstract**: Human preferences change over time: an established preference may remain valid over time, become uncertain as context shifts, or be explicitly revised. Effective personalization requires conversational AI to adapt its response strategy to evolving preferences rather than relying on passive memory retrieval. We formalize this capability as adaptive personalization under epistemic uncertainty. A human preference study further confirms that people favor state-contingent response strategies as preference uncertainty changes. To evaluate whether language models mirror this adaptation, we create a multilingual benchmark spanning 250 personas, 5 languages, and 25K conversational turns that decouples strategy selection from response execution. Across eight closed- and open-weight models, we uncover a recognition-generation gap: models reliably identify appropriate strategies when selecting among candidate options, but collapse during open-ended generation, suffering a 35-46% drop in strategy calibration. Crucially, this failure is systematic: models default to exploit-first behavior, over-relying on existing preferences regardless of preference epistemic uncertainty. We show this failure is addressable: while prompting interventions fail, targeted training that supervises models' internal strategy representations toward state-conditional human preferences narrows the user acceptability generalization gap by a third on out-of-distribution personas and conversational styles. 

---
# CoFiBreak: A Coarse-to-Fine Diversity Reward for Effective Multi-Turn Jailbreak 

**Authors**: Jialin Song, Xiaodong Liu, Weiwei Yang, Mingqian Feng, Chengquan Guo, Qiyao Ma, Carolyn Buractaon, Anson Ho, Wuyang Chen, Christopher White  

**Link**: [PDF](https://arxiv.org/pdf/2610.40926)  

**Abstract**: Automated red teaming trains an attacker large language model (LLM) to elicit harmful responses from a victim LLM, but reinforcement learning can collapse attackers onto a narrow set of prompts when attack success is the only reward. Such a collapse of attack diversity can bias safety evaluation toward repeatedly testing near-duplicate attack patterns rather than the underlying harmful intents. Existing methods often trade off attack success or inference cost for greater content or strategy diversity. We propose CoFiBreak, a coarse-to-fine diversity reward for Group Relative Policy Optimization (GRPO) that improves attack diversity while preserving strong attack effectiveness and low inference cost. The coarse reward broadens which strategies the attacker uses by encouraging strategies underrepresented within each GRPO group, while the fine reward broadens how those strategies are realized by discouraging reuse of the same personas, framings, and output formats after harmful-intent information is removed from prompt embeddings. Diversity rewards apply only to attacks that already exceed an effectiveness threshold, so diversity cannot be gained through ineffective attacks. Since both rewards are computed only during training, CoFiBreak also retains inference efficiency. Trained with Llama3.1-8B-Instruct as the victim, CoFiBreak discovers 1.6-2.0x as many distinct successful fine-grained strategies as the strongest baseline across frontier victims on HarmBench. It reaches 98.8% ASR@10 (attack success rate with 10 generations per intent) on AdvBench with the highest diversity score among all compared methods, which is 2.1x that of the strongest-attacking baseline. The resulting attacks transfer to unseen frontier LLMs, reaching 71.0% ASR@10 on AdvBench with the unseen victim Claude-Opus-4.6. The code will be released. 

---
# Qistas: A Pre-Registered, Portable Protocol for Fairness Evaluation of Large Language Models 

**Authors**: Nawar S. Alseelawi, Mustafa S. Aljumaily  

**Link**: [PDF](https://arxiv.org/pdf/2610.40893)  

**Abstract**: Fairness benchmarks for large language models (LLMs) are overwhelmingly English-born and American in their social categories; translated versions carry those categories into societies where they do not apply, and almost none fix their hypotheses before the models are run. We present Qistas (Arabic qistas, "the just balance"), a protocol for evaluating how LLMs treat the social, confessional, and regional constituencies of a society, portable across countries and instantiated first for Iraq. In its pilot form it measures only what needs no verdict on contested truth: procedural symmetry across matched counterfactual pairs, differential refusal, stereotype reliance under ambiguity, and terminological framing of documented events. Every measurement is paired by construction and analysed with paired machinery (sign-flip permutation tests, exact McNemar tests, template-clustered bootstrap intervals, Holm-corrected families), with a power analysis that states, before any data exist, the smallest effect each axis can detect after the design effect of template clustering. We release the authoring apparatus (a pruned 3x5 taxonomy with four cells closed by design, per-type checklists, a cumulative rulings log, a structural audit, and a human review notebook) and report its first field test: a semantic review of a 58-item seed bank in which 24 of 25 template groups failed at least one check. The defect patterns it exposed (evaluation criteria leaking into prompts, token-level swaps that break Arabic agreement, epistemically unbalanced pairs, single-theme refusal probes, unwritten dialect twins counted as coverage) are, we argue, general hazards of counterfactual fairness benchmarks. This is a protocol paper, not a results paper: no model has been evaluated, and the accompanying pre-registration commits the hypotheses, metrics, exclusion rules, and analysis plan that the pilot will follow. 

---
# The Generative Gap Between Speculative Drafter and Continuous Diffusion Language Model 

**Authors**: Yujian Chen, Hao Jiang, Guang Cheng, Weijian Luo, Tianyang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.40847)  

**Abstract**: Continuous diffusion language models (CDLMs) and speculative drafters offer efficient language generation but are typically evaluated differently: CDLMs generate independently through iterative refinement, whereas drafters propose text for target-model verification and correction. We compare their architectures without verification, using LangFlow for CDLMs and adapting DSpark's Markov and recurrent-head architectures for standalone generation. All models are trained from scratch for unconditional, single-block OpenWebText generation at lengths 16, 128, and 1024, without target features or verification, using common preprocessing, backbone dimensions, and training-update budgets. We evaluate generation perplexity (GenPPL) with an external language model and distributional similarity to reference text with MAUVE. With default denoising steps equal to sequence length, LangFlow outperforms both DSpark variants on both metrics at lengths 128 and 1024, while DSpark-RNN leads at 16. A conditional KL analysis characterizes dependencies excluded by the Markov head's previous-token conditioning. Recurrent memory removes this restriction and improves both metrics, but the longer-sequence gap persists. Teacher probability matching further lowers Markov GenPPL without observed MAUVE gains. Increasing LangFlow's denoising steps at fixed weights improves both metrics at lengths 16 and 128, changing the length-128 ordering relative to DSpark-RNN. These results establish LangFlow's longer-sequence advantage under the evaluated budgets and highlight iterative full-sequence refinement as a route to improving standalone generation beyond gains from recurrent head memory. 

---
# Mage-VL-Cowork: A Full Training Recipe to Learn Hybrid Computer Use From Evolved Harnesses 

**Authors**: Xiaoyi Zhang, Haoqing Wang, Xinjie Zhang, Xiang Ming, Xing Wang, Senqiao Yang, Kaichen Zhang, Zhaoyang Jia, Andong Tan, Xinjiang Wang, Jinghao Guo, Yifei Shen, Xiao Li, Peng Zhang, Xun Guo, Jiahao Li, Shicheng Zheng, Jinglu Wang, Zongyu Guo, Wenxuan Xie, Bin Li, Yan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2610.40747)  

**Abstract**: Effective computer-use agents (CUAs) should seamlessly combine visual GUI interaction with code and terminal execution within a single policy. We present Mage-VL-Cowork, a 4B hybrid CUA trained from a general multimodally aligned checkpoint through an end-to-end, progressive five-stage curriculum across midtraining, SFT, and RL. To power this curriculum at scale, Mage-CUA-Forge replaces per-task agent construction with reusable, skill-conditioned generation workflows and application-specific preparers, synthesizing 293,806 desktop and web tasks across 6,915 skills while reducing recurring per-task expenditure by 82.6%. Rather than directly distilling a default teacher, we evolve the teacher harness and instantiate specialized CUA harnesses to yield high-quality trajectories alongside stage-specific supervision. Mage-VL-Cowork-4B finally climbs to 66.0% on OSWorld-Verified through different training stages with more efficient reasoning cost, achieving state-of-the-art performance at its parameter size and pushing forward the frontier of on-device intelligence. 

---
# Li$^{2}$SAO: Lifting Light Singl-Rollout Asynchronous RL Optimization via Group-Relative Actor-Critic Bootstrapping 

**Authors**: Zichen Bai, Yanli Wang, Yang Liu, Junhao Shi, Xueqin Liu, Jinguo Sun, Muye Li, Shaoqiang Zheng, Xuefei Diao, Zhuwei Zeng, Nan Duan, Peng Cao  

**Link**: [PDF](https://arxiv.org/pdf/2610.40726)  

**Abstract**: Asynchronous reinforcement learning (RL) is becoming a mainstream paradigm for post-training large language models (LLMs), as it improves training efficiency by updating the model as rollouts arrive. Recent work argues that group-wise sampling in GRPO does not naturally fit asynchronous RL training because rollout lengths follow a long-tailed distribution, and proposes Single-rollout Asynchronous Optimization (SAO) to improve stability and off-policy behavior, especially on agentic tasks. However, SAO relies on a closed-source value model obtained from scaling value pretraining. In this paper, we analyze GRPO and LiSAO, our lightweight reproduction of SAO without the pretrained value model, on single-turn reasoning and multi-turn agentic tasks. We find that GRPO outperforms LiSAO on reasoning tasks, while the reverse holds on agentic tasks. Since the value model of LiSAO is not yet well trained , LiSAO lags behind GRPO substantially in early training steps. To address this issue, we propose Li$^{2}$SAO, which Lifts Light Single-rollout Asynchronous Optimization with group-relative actor-critic bootstrapping: while the critic is cold, a group-relative advantage trains the actor and a group-centered return trains the critic, after which training switches to single-rollout GAE. Experiments show that Li$^{2}$SAO catches up with or surpasses GRPO in early training and further raises the later performance ceiling. Code is available at this https URL. 

---
# Latent Reasoning in Continuous Space for Unified Multimodal Models 

**Authors**: Byungwoo Jeon, Yoonwoo Jeong, Hyunseok Lee, Minsu Cho, Jinwoo Shin  

**Link**: [PDF](https://arxiv.org/pdf/2610.42149)  

**Abstract**: Unified Multimodal Models (UMMs) achieve remarkable success on diverse multimodal understanding and generation tasks, yet they still struggle to effectively leverage language-based chain-of-thought reasoning for image generation. As a result, existing reasoning-augmented UMMs often fail to inherit the test-time scaling and fine-grained controllability observed in large language models. We attribute this failure to a fundamental modality mismatch: reasoning is performed in a discrete token space, whereas image generation operates in a continuous space. To bridge this gap, we introduce a Latent Reasoning framework in Continuous space (LARC), which alleviates the token-level bottleneck by reasoning over continuous hidden states. Specifically, the training framework consists of two stages: (i) curriculum supervised fine-tuning (SFT), and (ii) information-gain reinforcement learning (RL). First, the curriculum SFT stage gradually converts token-level reasoning into latent reasoning. Subsequently, the information-gain RL stage self-evolves latent reasoning without ground truth supervision. Experiments across diverse text-to-image generation benchmarks show that LARC consistently improves generation quality, prompt following, and compositional fidelity. Notably, LARC achieves state-of-the-art performance on GenEval with a 7%p gain over BAGEL and demonstrates clear test-time scaling behavior, outperforming token-based generation under both parallel and sequential test-time scaling strategies. 

---
# The Knowledge Layer Model (KLM): Evidence-Visible AI Output, Routing Diagnostics, and Negative Results 

**Authors**: Omer Asim Ozturk, Cihan Aydiner  

**Link**: [PDF](https://arxiv.org/pdf/2610.42066)  

**Abstract**: AI answers require more than apparent correctness: reviewers need to inspect their recorded basis. We introduce the Knowledge Layer Model (KLM), an open, implementation-neutral specification for inference records with versioned epistemic assessments, evidence relations, and explicit unknowns. SEDIM, the knowledge and evidence architecture of Nage AI OS, is a reference implementation. We reference KLM v0.3 (repository tag v0.3.0). Empirical material consists of separate internal experiments, not end-to-end validation. A historical platform-domain routing benchmark reports 137/150 classifications across 15 classes (91.3%; nominal 95% Wilson interval 85.7-94.9%); the numerator is recovered from a rounded summary, and historical example membership is unverified. A native-model weight-path configuration yielded 0/15 successful answers in a corrected two-source test. A capacity-growth comparison could not establish a reliable effect because of a defective legacy scorer and missing original outputs. Lexical and scoped diagnostics each yielded AUC 0.008 for silent-versus-contradicted examples, below the registered minimum of 0.65. Legacy packing lost gold-containing text in 118/221 selected cases; repaired packing retained it in 221/221. These measure text presence, not answer quality. Python and TypeScript verifiers reportedly agreed on six checks per synthetic fixture; a separate production disclosure was not accepted. Restoration, re-derivation, human-rubric support, boundary isolation, and bounded-response evaluations remain proposed. Findings are internal and author-reported; historical observations are distinguished from later reproductions. Evidence visibility does not establish correctness, causal reliance, security, or regulatory compliance. Independent replication and a complete public audit package remain future work. 

---
# Less Loss, Fixed Weights: Logit Corrections for Forgetting 

**Authors**: Hanyu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.42000)  

**Abstract**: When a language model is fine-tuned to follow instructions, or further pretrained on a new subject, it often gets worse at predicting standard web text. This is known as "forgetting." We ask whether this loss can be undone by simple, fixed adjustments to the model's final token predictions. Across 14 pairs of base and derived models, from 1B to 7B parameters, we fitted a fixed bonus or penalty for each token (a vocabulary bias), alone or together with a temperature. On text held out from correction fitting, the largest top-4K bias recovery estimate is under 10%, with loss conditional on the retained vocabulary. Combining the bias with a temperature recovers up to 36%. Models further pretrained on math lose the most and recover almost nothing. Finally, in a controlled simulation where a model is damaged at random, with no planted bias or temperature change, the bias-plus-temperature fix still recovers 29%. Most of the real-model loss increase stays out of reach of our restricted fits. 

---
# DAMI: Dual-Axis Memory Initialization for Enhanced Long-Context Retrieval in Gated Linear Attention 

**Authors**: Zhangchi Feng, Harry Jiang, Yangyifei Luo, Ray Zhang, Richong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.41994)  

**Abstract**: Linear attention scales linearly with sequence length, but its finite-state memory can limit long-context retrieval. In gated linear recurrences, memory retention depends on state decay, which is shaped jointly by gate parameters A_log and dt_bias and by input activations. In controlled Gated DeltaNet (GDN) experiments, both parameters remain highly correlated with their initial values after training, highlighting the importance of initialization. Therefore, we introduce Dual-Axis Memory Initialization (DAMI), which jointly configures these parameters before training. DAMI assigns geometric timescales across heads for varied retrieval distances and a low-high-low input-sensitivity profile across layers. A small activation sample calibrates the GDN-family implementation to match both behavioral targets without adding trainable parameters. Across GDN, KDA, GDN2, Mamba-2, and RWKV-7, DAMI improves average RULER-style retrieval accuracy by 5.29 to 16.71 percentage points over the corresponding random-initialization baselines. Performance on language modeling, commonsense reasoning, and 2K real-world retrieval remains broadly comparable to those baselines. Our code is available at this https URL. 

---
# Completeness and Cost of Retrieval Structures for Multi-Tool Requests in LLM Agents 

**Authors**: Shuyang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.41905)  

**Abstract**: An agent that must call several tools to fulfill a request fails when the tool retriever omits any one of them, yet tool retrievers are mostly built and evaluated on queries requiring a single tool. We compare representatives of six retrieval structures (flat lexical and dense retrieval, LLM query decomposition, LLM-navigated hierarchies, server-then-tool matching, dependency graphs, and trained set-level retrievers) across two multi-tool testbeds: 345 AppWorld queries with expert dependency labels and 103 MCP-Bench tasks over 257 MCP tools. We measure completeness@k, the share of queries whose entire required tool set appears in the top k, alongside the LLM calls, tokens, and latency that each structure incurs at retrieval time. A motivating analysis on all 7,961 ToolRet queries shows that aggregate completeness mixes two regimes: completeness at 10 is .58 to .67 on single-target queries and .08 to .17 on queries with four or more targets, and the ordering of retrievers on single-target queries reverses on multi-target ones. Across the two testbeds, none of the compared structures that is untrained on the testbed and makes at most one LLM call per query completes more than 10% of AppWorld queries at k = 16. The two structures that do better incur different costs: a set-level retriever trained on 75 to 90 in-domain queries reaches .33 on AppWorld and .58 on MCP-Bench with no LLM calls, while an LLM-navigated hierarchy reaches .30 and .40 with a median of 31 to 38 LLM calls and about five minutes per query. Their failure modes also differ: the trained retriever mostly ranks the missing tools just below the cutoff, whereas the hierarchy finds the target API and stops before collecting its prerequisites. In a downstream check with the official AppWorld agent, tasks with a complete retrieved set succeeded 21 of 21 times, whereas incomplete ones succeeded 2 of 39 times. 

---
# How Much Can Tool Calls Reveal About You? Privacy Leakage from Tool-Call Traces 

**Authors**: Donghang Duan, Qingyu Liu, Yifei Liu, Na Zou, Xia Hu, Dongrui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.41885)  

**Abstract**: Large language model (LLM)-based agents interact with external tools through protocols such as the Model Context Protocol (MCP), producing tool-call traces that may expose behavioral signals about users. In this work, we systematically study the risk of user-profile inference from MCP tool-call traces, where observers infer user attributes from protocol-valid tool interactions without access to user conversations. We introduce PrivacyTrace, a benchmark containing 1,000 synthetic user profiles and 4,000 agent trajectories across four domains, generated with real MCP tools from 25 servers. Experiments with four mainstream LLM attackers show that tool-call traces enable a 26.25% attack success rate for profile inference, compared with 17.97% from prior-based guessing. Further analysis reveals that tool arguments and returned results provide the main trace-grounded evidence, while linking traces across sessions further amplifies profiling risk. These findings demonstrate that MCP tool-call traces can act as effective behavioral side channels for user profiling, motivating the development of privacy-aware mechanisms for agent-tool interactions. Code and data are available at this https URL. 

---
# Task Vectors Can Be Linearly Decomposed into Interpretable and Controllable Components 

**Authors**: Yukiya Horiba, Hakaze Cho, Takahiro Yakoh  

**Link**: [PDF](https://arxiv.org/pdf/2610.41743)  

**Abstract**: Task Vectors (TVs) provide a compact way to steer large language models, but they are typically treated as atomized directions merely extracted from hidden states, leaving their internal functional structure poorly understood. Motivated by prior work that decomposes in-context learning into Task Recognition (TR) and Task Learning (TL), we ask whether a single TV can likewise be decomposed into functionally meaningful components. We formulate a TR/TL-based decomposition from candidate-label unembeddings, separating each TV into TR, TL, and residual components, and try to optimize the TR and TL coefficients using Bayesian optimization for more effective intervention. We evaluate this formulation across multiple model families, classification tasks, and both instruction- and demonstration-based TVs. Across these settings, TR/TL-based optimization identifies intervention directions that outperform the original TVs. Although optimization over random directions can also improve performance, TR/TL-based search yields more reliable gains, particularly for instruction-based TVs. Moreover, isolated interventions on the decomposed components produce distinct logit-level effects that are consistent with their intended functional roles. Together, these results suggest that TVs are not monolithic steering directions, but can be decomposed into functionally meaningful components that support both interpretation and control. 

---
# Leakage-Controlled Structure-Preserving Augmentation and Template-Aware Evaluation for Bangla Symptom-Based Disease Prediction 

**Authors**: Sabah Nushra, Shovan Samanta Turzo, Moinul Hossain, Md Kishor Morol  

**Link**: [PDF](https://arxiv.org/pdf/2610.41733)  

**Abstract**: Bangla symptom-based disease prediction is commonly evaluated on a small, imbalanced structured benchmark containing many profiles that differ by only one or two symptoms. We test whether train-only asymmetric Bernoulli perturbation improves classification using 725 unique profiles spanning 85 diseases and 172 binary symptoms. Experiments use ten repeated real-profile holdouts and parent-grouped inner cross-validation. The 2 x 2 design separates augmentation from hyperparameter tuning, with exact duplication, random oversampling, class weighting, and SMOTEN as balancing controls. For Bernoulli Naive Bayes (BNB), the gain over tuned raw training does not withstand Holm correction, although BNB attains the highest mean accuracy (97.50%). The boosted-tree models benefit substantially: XGBoost, LightGBM, and CatBoost improve by 18.10, 28.45, and 10.09 percentage points, respectively. A Hamming-distance audit places 91.0% of standard held-out profiles within one symptom bit of the training data. Accuracy decreases on a same-disease 1-bit-purged challenge; nevertheless, XGBoost and LightGBM retain large augmentation gains. We also evaluate robustness to missing and spurious symptoms and the faithfulness of BNB explanations. Overall, Bernoulli perturbation is useful for some model families, but benchmark accuracy should be interpreted together with template proximity. 

---
# From Training Exposure to Membership Evidence: A Controlled Study of OLMo 2 

**Authors**: Ali Banaei, Muhammad Shahab Shahzad, Gerald Q. Maguire Jr., Marco Chiesa, Dejan Kostić  

**Link**: [PDF](https://arxiv.org/pdf/2610.41732)  

**Abstract**: Whether a language model was trained on a document matters for copyright, privacy, and benchmark contamination. Yet most membership-inference (MI) evaluations on language models collect members and non-members after the fact, so success may reflect distribution shift and failure is inconclusive. We re-ran OLMo 2 1B's official 50-billion-token mid-training stage, replacing selected training chunks with documents from 7,814 pairs matched on domain and difficulty. For each pair, a coin flip determined which document was inserted and which was held out. The run released by the Allen Institute for AI (AI2) from the same seed saw identical data everywhere else and serves as the counterfactual. One exposure raises a previously unseen document's likelihood on average but rarely identifies it. Using the counterfactual as reference, the reference ratio has an area under the ROC curve (AUC) of 0.767 but, with one threshold across domains, catches 3.8% of members at a 1% false-positive rate (FPR). In contrast, 13 matched pairs give a paired test 80% power at the 5% level to detect training on a collection, given same-domain non-members and a close reference. A reference trained with another seed recovers 92% of the counterfactual's AUC advantage over chance, whereas reference-free attacks stay within 0.07 of chance when pooled over domains. The full official post-training recipe, re-run on both models, leaves 89% of the average effect. Even with a closely matched reference, substantial document-level variability limits individual membership inference. We will release the checkpoints, per-token log-probabilities, and the evaluation documents; the code is available at this https URL. 

---
# UpLoop: Training Looped Transformers via Upcycling at No Extra Pretraining Compute 

**Authors**: Cheng Liang, Tengfei Zhang, Ya Zhang, Yanfeng Wang, Zhouhan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2610.41720)  

**Abstract**: Looped language models can strengthen reasoning through repeated computation with shared weights, increasing execution depth without adding independent parameters. However, each additional pass increases training floating-point operations (FLOPs), making these gains more expensive to obtain. To address this cost, we propose UpLoop, a pretraining schedule that uses upcycling to train looped Transformers while exactly matching a standard dense baseline in final parameter count, total training tokens, and training FLOPs. Specifically, UpLoop begins with a shallow Transformer, progressively grows its depth by copying learned layers, and uses the computation saved early to support repeated execution of a shared middle block later in training. Experiments across model scales show improvements in language modeling and downstream performance, with the average score across 22 downstream tasks increasing from 22.99 to 27.52 at 1.3B parameters. These gains also extend to mathematical reasoning after supervised fine-tuning. 

---
# TSBPO: Tree-Structured Branching for Policy Optimization in Interactive Language Agents 

**Authors**: Rui Yu, Chong Mu, Jinchuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.41697)  

**Abstract**: Training interactive language agents from terminal outcomes poses three coupled problems: which task and state continuations to sample, how to assign delayed feedback to decisions, and how to aggregate token losses for multi-token actions. We present TSBPO, a tree-structured sampling-to-update paradigm that addresses these problems within one training loop. A likelihood-fitted Root/Branch allocator uses retained task and state evidence to target continuations likely to produce contrasting terminal outcomes under fixed rollout budgets. On the resulting replayed physical trees, trajectory advantages compare immediate sibling segments, while observation-group step advantages compare outcomes at matched task/observation states. The actor then sums clipped token surrogates over each complete response and normalizes across active turns, avoiding response-length averaging in the policy-gradient term. With a Qwen2.5-1.5B actor, fixed checkpoints obtain 97.79% seen and 96.64% unseen stochastic success on ALFWorld and an 81.07% per-episode success rate over six stochastic episodes for each of the 500 WebShop test goals. A WebShop step-term ablation yields 75.93% at the same update-100 evaluation; an ALFWorld aggregation comparison shows higher observed command validity with complete-turn weighting. These results support the evaluated training configuration on two interactive benchmarks. 

---
# Does Reasoning Make Language Models Agree? 

**Authors**: Mila Gorecki, Moritz Hardt  

**Link**: [PDF](https://arxiv.org/pdf/2610.41691)  

**Abstract**: Language models face two opposing concerns: monoculture, in which models err in correlated ways, and multiplicity, in which equally accurate models make conflicting decisions. The rise of reasoning models has complicated this debate: On the one hand, reasoning may increase model agreement if models anchor on the same salient features or follow similar reasoning paths. On the other hand, stochastic reasoning traces may introduce additional branch points leading to more diverse predictions. We resolve this tension by distinguishing reasoning as an intervention from reasoning as an observed behavior. We compare agreement under controlled changes in reasoning effort with agreement across observed reasoning lengths, measuring both against the agreement expected from model accuracy. Across nine models from two developers and three classification tasks, enabling reasoning increases agreement between models evaluated on the same individuals. Yet the individuals who elicit longer reasoning traces receive less consistent predictions across models. Both patterns persist after accounting for differences in accuracy. Asking models to reason more thus promotes agreement, while observing them reason longer signals disagreement. 

---
# GRIT: Gradient Masking for Visually Misgrounded Tokens in Text-Rich Reasoning 

**Authors**: Chanseul Cho, Noori Bae, Dongheon Lee, Youngmin Ro  

**Link**: [PDF](https://arxiv.org/pdf/2610.41629)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has recently been extended to vision-language models (VLMs), yet its effectiveness remains limited for text-rich reasoning tasks. We highlight a key factor contributing to this limitation: models frequently arrive at correct final answers despite misperceiving visual information during intermediate steps. We define this phenomenon as Correct-but-Visually-Misgrounded Reasoning (CVMR). Because standard RLVR rewards only final-answer correctness, it fails to differentiate between properly grounded and misgrounded reasoning, reinforcing both equally. Through token-level analysis, we reveal that CVMR tokens exhibit distinct attention patterns that differ systematically from well-grounded ones. Based on this observation, we propose GRIT, a training framework that incorporates visual grounding into RLVR to mitigate the impact of CVMR. Furthermore, to address the lack of text-rich reasoning benchmarks that allow for systematic evaluation across text length and reasoning depth, we introduce GRIT-Bench. Experiments demonstrate that GRIT significantly improves both reasoning accuracy and grounding fidelity over existing RLVR methods. 

---
# SCRBench: Beyond Equivalence Checking for Semantic Change Reasoning in Code 

**Authors**: Omid Saghatchian, Ali Mahzoun, Khashayar Etemadi, Saeed Reza Kheradpisheh  

**Link**: [PDF](https://arxiv.org/pdf/2610.41562)  

**Abstract**: Understanding how code changes affect program semantics is a core aspect of code reasoning in language models. It is commonly evaluated by asking whether two programs are semantically equivalent, but a yes/no answer does not reveal whether a model knows which code change caused a semantic difference, or which sequence of code changes produced the modified program. We study two tasks that ask these questions directly, diagnosing semantic change and reconstructing sequences of code changes, alongside detecting semantic equivalence. We build SCRBench, 67,658 Python program pairs generated by applying sequences of code changes that preserve or alter program semantics, and evaluate five language models on a stratified subset of 4,500 pairs. Detecting semantic equivalence exposes weaker models, whose accuracy declines as changes accumulate, but the strongest models remain near the ceiling. Diagnosing semantic change reveals weaknesses that equivalence accuracy hides: a model with 94.71% equivalence accuracy diagnoses only 54.44% of semantic changes. Reconstructing sequences of code changes is much harder: the best model reconstructs the exact sequence for only 39.27% of pairs, and no model exceeds 1.11% when five changes are applied. High equivalence accuracy therefore overestimates how well models reason about the semantic effects of code changes. 

---
# DAPS: Dependency-Aware Premise Selection for LLM Theorem Proving 

**Authors**: Zixuan Chen, Wenyuan Jiang, Junling Wang, Mrinmaya Sachan, Yinya Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.41536)  

**Abstract**: LLM-based theorem proving in Lean 4 has advanced rapidly, but provers are difficult to leverage newly contributed lemmas, because retrieval skews toward popular foundational lemmas, leaving the right premises buried among hundreds of thousands of candidates. Existing neural selectors rank candidates by semantic similarity, with three intrinsic limitations: frequency-skewed retrieval, isolated pairwise ranking, and degradation on out-of-distribution queries such as competition problems and informal language. Given that mathematics is structured by dependencies rather than similarities, we restore this signal to both the training data and the model. We curate two underused dependency resources. We conduct the first systematic mining of 53 Lean Blueprint projects, a previously untapped corpus of 3,805 manually curated informal+formal nodes that capture rare, research-level premises absent from Mathlib4. We further pair it with the extraction of comprehensive, typed, multi-level Mathlib4 dependencies. We then introduce DAPS, a dependency-aware premise selector with a structural neighborhood encoder, a group-level contrastive objective, and a Mathlib-then-Blueprint adaptation procedure. DAPS reaches mean Recall@32 of 89.25 on Mathlib4-Heldout (+11.41 over the strong selector LeanHammer), and maintains this lead on two out-of-distribution benchmarks. Further experiments demonstrate that DAPS premises improve theorem proving on miniF2F and IMO-ProofBench. Project page: this https URL. 

---
# Enhancing Sentiment Classification on Small Datasets through Data Augmentation and Transfer Learning 

**Authors**: Mahmoud S. Mayaleh, Samer A. Mayaleh  

**Link**: [PDF](https://arxiv.org/pdf/2610.41518)  

**Abstract**: Small-scale sentiment classification often suffers from data scarcity, which limits the generalization ability of the models. This study evaluates and compares the effectiveness of three data augmentation strategies: Easy Data Augmentation (EDA), back-translation, and contextual token substitution (nlpaug-style), with both traditional machine learning classifiers (Logistic Regression, Random Forest) and transformer-based models (BERT). We perform a comprehensive empirical comparison with low-resource sentiment datasets by summarizing the results of recent studies and performing targeted head-to-head experiments. Our findings indicate that all augmentation methods improve performance. Contextual augmentation yields the most consistent gains for BERT models, while EDA and back-translation provide greater benefits for traditional classifiers. These insights help guide the selection of data augmentation techniques tailored to model type and dataset size, filling a critical gap in research on data augmentation for sentiment classification on small datasets. 

---
# A prospective randomized study of patient-centered AI for clinical visit preparation 

**Authors**: Xincheng Tan, Matthew Thompson, Xiang Ji, Angelica Willis, Will Vaughan, Roma Ruparel, Michael A. Sanchez, Sean Chang, Joseph Xu, Bhavna Daryani, Preeti Singh, Pinal Bavishi, Kejia Chen, Fan Zhang, Priya Gupta, Ernest Lim, Christina Chen, Bob Lou, Dimitrios Antos, Zach Wasson, Ray Luo, Fadi Yousif, Florence Thng, Mark Malhotra, Joseph Breda, Daniel McDuff, Kavi Goel, Katherine Chou, Dale R. Webster, Avinatan Hassidim, Yossi Matias, Po-Hsuan Cameron Chen, Yun Liu, Mike Schaekermann, Quang Duong, Jackie M. Barr  

**Link**: [PDF](https://arxiv.org/pdf/2610.41437)  

**Abstract**: Patients often arrive at brief, time-constrained clinical visits unsure of what to expect or ask, and millions of adults now turn to general-purpose AI chatbots to prepare beforehand. Yet whether conversational AI improves visit readiness remains unclear. We designed a large language model (LLM)-based visit preparation agent to guide patients before an appointment by gathering clinical context, setting visit expectations, providing relevant health information, building a prioritized list of questions for their clinician, and generating a summary. In a prospective consented randomized study in the Google Health app, 802 U.S. adults held 866 health-related conversations with the visit preparation agent (standard or concise variant) or a minimally prompted control chatbot built on the same LLM. The visit preparation agent significantly outperformed the control across blinded clinician evaluations and patient self-reports. In clinician reviews of completed sessions (n = 272), the agent achieved a 16.6% higher rate of comprehensive preparation (67.2% vs. 50.6%; P < 0.05) and an 8.5% higher rate of alignment with patients' health context (98.9% vs. 90.4%; P < 0.01), while maintaining high medical safety and appropriateness across all study arms. Patient post-surveys (n = 460) showed 16.8% higher visit preparedness (73.8% vs. 57.0%; P < 0.01), 14.3% higher confidence in knowing what to ask (81.4% vs. 67.1%; P < 0.01), and 12.0% higher overall satisfaction (74.0% vs. 62.0%; P < 0.05). Clinician and participant metrics were not significantly correlated, indicating that expert review and patient self-assessment capture complementary dimensions of patient-facing AI. These findings suggest that LLM chatbots can safely help patients prepare for clinical visits, with greater benefit when following a structured workflow, and motivate future studies on in-visit communication and clinical outcomes. 

---
# Hybrid Computer Use: A Unified Programmatic, Semantic, and Visual Interface for General Agents 

**Authors**: Haoqing Wang, Xiaoyi Zhang, Sicheng Fan, Zhexin Zhang, Xinjie Zhang, Jiahao Li, Bin Li, Yan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2610.41432)  

**Abstract**: Computer-use agents increasingly tackle tasks ranging from everyday desktop activities to long-horizon professional workflows. Yet interfaces centered on screenshots and mouse-and-keyboard actions can impose a human-interaction prior, requiring repeated screen interpretation while leaving the computer's semantic and programmatic capabilities underexposed. We argue for exposing these capabilities and letting models choose and combine them. We introduce Hybrid Computer Use (HCU), a unified interface for general agents that combines three complementary channels: programmatic interaction through code and command-line tools, semantic interaction through accessibility (AX), and visual interaction through graphical user interfaces (GUI). Our HCU supports efficient AX interaction and guards against stale-state actions during channel switching. A supplementary HCU skill guides agents in selecting and coordinating channels to use the unified interface effectively. Our designed AX tools return compact observations, execute actions in batches, and report their effects as AX diffs, so the model needs fewer round trips. On OSWorld 2.0, HCU achieves the highest Partial and Binary scores among the compared systems. Analyzing agent trajectories, we find that all evaluated models improve with HCU yet allocate work across the channels differently and consistently outperform compared baselines in terms of both efficiency and task success, highlighting the value of unified computer-use interfaces that allow flexible channel selection and combination. 

---
# SparseGDN: Chunk Sparse in Linear Attention 

**Authors**: Jiedong Zhuang, Peng Zhao, Jialin Tang, Haoji Hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.41420)  

**Abstract**: Gated Delta Networks (GDNs) offer linear-time sequence modeling, yet long-context prefill remains costly due to dense intra-chunk computation, recurrent-state updates, and intermediate state storage. We introduce SparseGDN, a hardware-aware framework that combines structured intra-chunk approximation with input-dependent recurrent-event selection while producing outputs for every token. A block-adjacent solver reduces local Delta-solve computation, while a dynamic selector allocates recurrent-update budgets across head pairs. The GPU implementation materializes updates only for selected events and reconstructs token outputs using compact state snapshots and group-local interactions, reducing computation and memory traffic. We evaluate SparseGDN on Qwen3.5-4B and Qwen3.5-9B across RULER tasks and context lengths from 4K to 128K. Retaining approximately 29% of recurrent events on average, SparseGDN achieves 1.50-1.94x GDN-core prefill speedups over native FLA kernels, including online selection overhead. 

---
# ProSearch: Benchmarking Multi-Constraint Protocol Retrieval in Experimental Science 

**Authors**: Wenliang Liang, Zhiyuan Ning, Haolin Chen, Yuanchun Zhou, Wenjuan Cui, Yi Du  

**Link**: [PDF](https://arxiv.org/pdf/2610.41396)  

**Abstract**: In AI for Science scenarios, experimental design and method reproduction require retrieving protocols that satisfy multiple interdependent constraints, yet existing benchmarks lack systematic evaluation of this capability. To address this gap, we introduce ProSearch, a benchmark for multi-constraint protocol retrieval in experimental science. ProSearch has three key features: (1) realism, with queries grounded in real experimental records and expert-defined templates; (2) complexity, with 60 experimental fields and their dependencies; and (3) diagnosticity, with 3-5 explicit constraints per query, including numerical and negation constraints. We construct ProSearch through LLM-based extraction, data-driven constraint selection, controlled query generation, and evidence-based relevance validation. The resulting benchmark contains 1,515 queries across six experimental science domains. We conduct extensive experiments on ProSearch with 19 representative retrievers and complementary analyses. The best-performing retriever achieves an average nDCG@10 of only 0.290, which improves to 0.474 with reranking. Structured diagnostic analysis shows that explicit constraint matching provides a strong ranking signal, while manual error analysis identifies joint constraint binding as a key bottleneck. ProSearch provides a testbed for systematically evaluating scientific retrieval systems. 

---
# What We Keep of What We Forget: Replacement Operators and Routing Semantics in Training-Free MoE Expert Pruning 

**Authors**: Ezza Ali, Sehrish Bibi, Momina Moetesum, Faisal Shafait  

**Link**: [PDF](https://arxiv.org/pdf/2610.41366)  

**Abstract**: Training-free expert pruning makes two decisions: which experts to remove, and what the layer computes when its router still selects a removed expert. Published methods change both at once. Holding the mask fixed, we compare replacement operators from three families on OLMoE, DeepSeek-V2-Lite and Moonlight, with Granite as a fourth router, using paired, multiplicity-corrected tests. The largest effect is a choice usually left implicit: the survivors' gate scale. On routers trained without top-k renormalization, renormalizing the survivors' gates, a common implementation choice, costs 0.8-8.5 points and reverses whether rerouting beats MoNE's mean-output replacement; on two routers trained with renormalization, the trained scale wins at 50% pruning. We introduce the tilted novice, a closed-form low-rank map of the pruned expert's input whose rank-0 case is MoNE's constant. At 50% pruning it adds 1.2-1.8% parameters and gains 3.45 and 1.37 points over the constant on OLMoE and Moonlight, stable across calibration draws and largely unchanged by 4- and 3-bit quantization. No operator dominates: trained-scale rerouting is best on DeepSeek-V2-Lite, and a partial copy of each pruned expert is statistically indistinguishable from the tilted novice on two models. Evaluation can change conclusions: at 25% pruning MoNE's constant keeps 92-97% of dense multiple-choice accuracy but only 5-65% of GSM8K and 0-17% of code scores, and the calibration domain moves the mask 4-9 times more than resampling. We propose a five-item reporting checklist. 

---
# SERIAL-7: Deriving, Realizing, and Preserving Machine-Verifiable Outputs Across Representations 

**Authors**: Youngkyung Seo, MiHyeon Kim, Jisoo Lee, Hwijung Ryu  

**Link**: [PDF](https://arxiv.org/pdf/2610.41308)  

**Abstract**: Large language models increasingly generate structured outputs consumed by programs and downstream systems, where semantic correctness alone does not guarantee reliable execution. We introduce SERIAL-7, a Korean-English benchmark built around an executable verification framework for machine-verifiable generation across seven representations. Using 177 validated canonical artifacts, SERIAL-7 separates semantic derivation (Derive), constrained artifact generation (Realize), and cross-representation preservation (Preserve). Its 43 atomic representation constraints are paired with executable checks over format, structure, and task-specific semantics, enabling progressively stricter verification from representation compliance to full artifact validity. Across ten models, REALIZE-L3 HSR ranges from 42.0% to 93.9%, while Full Compliance exposes additional artifact-level failures beyond correct final answers and valid representations. These results show that reliable structured generation requires explicit, executable verification of how artifacts are realized and preserved across representations. 

---
# SciGraphSelect: Auditable Scientific Data Selection through Knowledge--Reasoning Graphs 

**Authors**: MiHyeon Kim, Ji-Eun Han, Eunkyeong Lee, Jeehyun Lim, Hwijung Ryu  

**Link**: [PDF](https://arxiv.org/pdf/2610.41280)  

**Abstract**: Scientific data selection is typically driven by relevance or utility scores, providing limited insight into why a document is retained. We introduce SciGraphSelect, a reference-grounded framework for auditable scientific data selection. SciGraphSelect organizes domain-specific knowledge, reusable reasoning operations, and reference-attested relations into a typed knowledge--reasoning graph, then scores candidate documents using concept coverage, relation and path co-support, and directional alignment. To avoid over-interpreting document embeddings, rich relational structure is preserved on the reference side while candidate-side evidence is treated conservatively without assuming relation entailment. By exposing the reference-grounded evidence behind each selection, SciGraphSelect brings traceability to the upstream construction of scientific training corpora and complements downstream scientific verification. 

---
# TRIDENT and SHAKTI: A Framework and Benchmark for Hindi AI-Text Detection 

**Authors**: Shifali Agrahari, Sujit Kumar, Sanasam Ranbir Singh  

**Link**: [PDF](https://arxiv.org/pdf/2610.41232)  

**Abstract**: As Large Language Models (LLMs) become increasingly multilingual, distinguishing human-written from AI-generated text is essential for maintaining information integrity. However, research remains English-centric, with limited benchmarks for low-resource languages such as Hindi. To address this gap, we introduce SHAKTI, a large-scale Hindi benchmark covering four genres, eight LLM families, multiple prompting strategies, and adversarial transformations. We evaluate its quality through human annotation and analyze linguistic, grammatical, and sentiment-based differences between human- and AI-generated texts. Building on these findings, we propose TRIDENT, a round-trip translation-based detection framework with three variants: TRIDENT-S (static threshold), TRIDENT-D (dynamic threshold), and TRIDENT-C (logistic regression using multi-distance RTT features). TRIDENT leverages semantic consistency between original and round-trip translated texts to identify AI-generated content. Extensive experiments show that TRIDENT outperforms conventional and supervised baselines, with TRIDENT-S achieving the best overall performance. Further evaluations across seven languages, adversarial settings, and unseen LLMs demonstrate its multilingual applicability and robustness, while highlighting sensitivity to the translation model. 

---
# Context as Infrastructure: Knowledge Compilation for Retrieval in Agentic Systems with Harnesses 

**Authors**: Zhichang Yu, Baining Cao, Sbaaoui Idriss, Qi Wang, Gangqiang Xu, Ling Qin, Xiaoling Jin, An Liu, Shaohao Cai, Xiaoyang Bao, Shengle Chen, Changchang Ding, Pengfei Zhu, Jingxiang Xu, Yifeng Mao, Lu Liu, Hai Jin, Yingfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.41172)  

**Abstract**: Retrieval-augmented generation is becoming agentic. A search harness writes queries, reads results, reformulates, and decides when to stop. This largely solves accuracy: a harness with only BM25/grep and MiniMax M3 answers 85% of BrowseComp-Plus and 91% on multi-hop FRAMES in our runs, without compiled or vector indexes. It exposes a context problem: an undisciplined loop repeats queries, rereads documents, and fills the window with unused passages. The missing layer is knowledge compilation: a configurable ingestion pipeline where LLMs enrich documents with context no chunk contains alone--navigation hierarchies with claim-evidence leaves, entity graphs, timelines, mind maps, ontologies, wikis, and memory. Compilation is not a better retriever: under a competent harness, structural artifacts gave no detectable accuracy gain on FRAMES, and a PageIndex-style outline degraded at scale. It is a supply layer. A navigation tree whose leaves are claim-evidence pairs--the mechanism we measure--cuts query tokens by 91% when evidence suffices and adds 1.9% when it does not, a 38% average reduction; this implies a 2% break-even sufficiency rate and a measured rate near 43%. A decision gate cuts latency from 140 s to 30 s, though substitution's token saving is bought with retrieval rounds and carries its own latency cost. Other artifacts--ontology, graph, wiki, timeline, mind map, memory--pay off in deployment (governance, integration, onboarding, temporal orientation, continuity) where accuracy benchmarks cannot see them. We describe RAGFlow's pipeline and harness (open source at this https URL), which turns the loop into five disciplined stations and keeps the context entering the model the smallest sufficient set; together they form an agent-native database: a store that hands the agent compiled, governed context instead of a ranked list of passages. 

---
# Lineage Is Memory: Parent Inheritance and Feedback Disclosure in Short-Budget LLM-Guided Optimization 

**Authors**: Shanglin Yang, Zhan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2610.41157)  

**Abstract**: Iterative language-model optimizers refine prompts and programs by evaluating a candidate, showing a proposer model a sample of failing cases, and keeping revisions that improve performance. Recent work attributes the limits of this loop to narrow single-round reflection, introducing full-set error clustering, targeted minibatches, or larger proposer models so the optimizer sees more failure modes per step. Yet every existing evaluation tests these mechanisms exclusively inside lineage-preserving search, where each revision edits a surviving parent, leaving the contribution of parent inheritance confounded with the breadth of per-round feedback. Across 1,750 pre-registered runs spanning nine prompt and program synthesis benchmarks, we separate parent inheritance from feedback disclosure under matched evaluation budgets. We find that greedy parent inheritance drives the vast majority of optimization progress, outperforming seed-reset resampling even when seed-reset proposals receive a clustered summary of all failures in every round or use a larger proposer model. While full-set error clustering and diverse failure selection improve single-step rewrites from the seed and accelerate the opening rounds of search, sequential parent inheritance rapidly closes this gap as solved error modes are retained in the candidate and filtered from subsequent failure lists. Moreover, decomposing token expenditure in GEPA reveals that proposer reflection accounts for only a small fraction of total cost whereas validation and minibatch gating consume nearly the entire budget, showing that greedy parent inheritance over raw failures captures the gains of complex reflective loops at a fraction of their compute. 

---
# MUON-RT Reusing the Teacher's Trajectory for Momentum-Free Muon for LLM Self-Distillation 

**Authors**: Aryan Das, Vinay Kumar Verma, Aakansha Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2610.41104)  

**Abstract**: Self-distillation methods for large language models with an exponential moving average (EMA) teacher already retain a history of student updates. Muon nevertheless stores a separate momentum buffer, increasing persistent training memory. We propose Muon with Reused Teacher (Muon-RT), which reuses the teacher-student displacement to eliminate momentum buffers for hidden weight matrices. We show that this displacement encodes an exponentially weighted history of applied updates and smooths temporally uncorrelated noise under a local model. Muon-RT removes its radial component, normalizes it spectrally, and mixes it with the current gradient after Newton-Schulz (NS) orthogonalization. Under explicit regularity and alignment assumptions, we establish linear contraction of an expected objective-gap bound toward a residual error level. With Self-Distillation Fine-Tuning (SDFT) on Qwen2.5-7B-Instruct, Muon-RT improves average accuracy over AdamW by 2.6 points with 42% less persistent memory, and over momentum Muon by 5.4 points with 27% less persistent memory. With Self-Distillation Policy Optimization (SDPO) on Olmo-3-7B-Instruct, it improves average accuracy over AdamW by 2.0 points. 

---
# Capability-Grounded Agentic Planning 

**Authors**: Shikha Shrestha, William Yeoh, Ning Zhang, Yevgeniy Vorobeychik  

**Link**: [PDF](https://arxiv.org/pdf/2610.41040)  

**Abstract**: LLM agents serve natural-language requests through tools, data sources, and other agents, and even simple requests often need several interdependent calls. Typically, the LLM picks these calls by trial and error, with no guarantee of an executable plan. We present Capability-Grounded Agentic Planning (CGAP), which instead reasons about what a request requires and what tools provide. CGAP compiles the tool catalog deterministically into a PDDL model, uses an LLM to turn the request into goals, and solves the model with a classical planner. Since tool outputs are unknown in advance, the model is abstract, tracking which inputs are available, not their values. During execution, an LLM adds entity-specific calls that abstract plans omit, and deterministic checks reject any that break the plan. Every argument is drawn from a cited source, such as a conversation span or an earlier tool response, and each requested result counts as delivered only after an LLM verifier checks it against the request. With three LLMs on ToolSandbox, ComplexFuncBench, BFCL multi-turn, and $\tau^2$-bench retail, CGAP outperforms a standard tool-calling agent by up to 28.9 points. 

---
# Do Language Models Know What Compelling Scientific Evidence Looks Like? 

**Authors**: Conor Igoe, Gabriel Grand, Zane Koch, Albert Bou, Eli Moss, Michaela M. Hinks, Samuel G. Rodriques, Andrew White, Siddharth Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2610.41015)  

**Abstract**: Expert scientists excel at weighing empirical findings from multiple sources that vary in both statistical reliability and epistemic relevance. To what extent are language models, which are increasingly driving autonomous discovery systems, capable of such evidence deliberation? We introduce EDB-Bio, a scientific reasoning benchmark curated from real-world data in three biological domains: drug-perturbation transcriptomics, CRISPR screens, and human genome-wide association studies (GWAS). Each of EDB-Bio's 96k episodes presents a sequence of experimental results from published studies and asks the model to predict the probability that a target hypothesis is true. Out-of-the-box, open-weight and frontier models alike demonstrate systematic underconfidence and attain Brier scores comparable to a trivial base-rate forecaster. Post-training against strictly proper scoring rules significantly improves both calibration and discrimination. This calibration pays off downstream: in a sequential experiment selection task, our Ibis0 model is 2.5 times more efficient than the best frontier model. Throughout, quantifying evidence weights as Bayes factors reveals how models update their beliefs over time, whether they agree with expert biologists on relevance, and where they exhibit inconsistencies and biases. Our work extends calibration training and uncertainty analysis to a realistic scientific forecasting setting with applications to human cancer biology. We release all code, data, and trained models. 

---
# MOSAIC: Addressing Disproportionate Task-Modality Dependencies via Modality-Partitioned Gradient Routing 

**Authors**: Aryan Chowdhury, Yizheng Sun, Thomas Carroll, Riza Batista-Navarro  

**Link**: [PDF](https://arxiv.org/pdf/2610.41004)  

**Abstract**: In multi-task settings, combining multiple modalities allows task representations to draw on signals that a single modality alone would not capture. However, individual tasks may depend disproportionately on particular modalities, a structure that conventional fusion-based architectures do not model explicitly, instead enforcing a unified representation space. Consequently, a task's gradients can affect parameters tied to modalities it barely relies on, leaving features that compromise between competing demands. Although prior work has explored task-weighting and loss-balancing strategies, their effectiveness is limited by the absence of explicit modality separation, preventing models from optimising each modality according to its task-specific importance. To address these limitations, we propose MOSAIC (MOdality-Separated Architecture with Inter-modal Cross-attention), which preserves unimodal representations while enabling controlled interaction between them, and uses modality-specific losses to route gradients to their corresponding partitions. On the Corpus of Everyday Japanese Conversation (CEJC), MOSAIC improves average macro-F1 by 7.4 points across four tasks with a 72% reduction in backbone parameters relative to a fusion-based baseline. Our results demonstrate that modality partitioning yields consistent gains across tasks while requiring far fewer backbone parameters than a fusion-based design. 

---
# The psychophysics of affect in large language models 

**Authors**: Joseph Marks, Marius C. Vollberg  

**Link**: [PDF](https://arxiv.org/pdf/2610.40985)  

**Abstract**: Good decisions often require considering how outcomes will feel. Yet large language models (LLMs) increasingly enter decision workflows, despite little being known about how they map outcomes to feelings or use those mappings in choice. We tested how LLMs anticipated gambling or stock-trading outcomes would feel and whether they incorporated these ratings into choice. When outcomes were rated in isolation, ratings exhibited diminishing sensitivity to magnitude, resembling human value functions; but models rated gains more intensely than equivalent losses, an inversion of canonical loss aversion we call silicon gain amplification. When outcomes were rated together, enabling direct comparison, ratings linearised. In choice, the gain-loss asymmetry largely disappeared, and models' own ratings did not reliably explain their choices better than monetary utility did. These findings reveal malleable mappings from outcomes to affect, raising questions about whether and how agentic LLMs should incorporate anticipated feelings into decisions. 

---
# Nedimo: An Evidence-Grounded Multilingual Pipeline for AI-Generated Sleep Stories 

**Authors**: Georgios Makridis, Stefanos Kampouris  

**Link**: [PDF](https://arxiv.org/pdf/2610.40970)  

**Abstract**: Generative audio systems can produce large libraries of personalized bedtime content, but ordinary media objectives (retention, completion, and novelty) are poorly aligned with the goal of supporting a transition toward sleep. Direct evidence for narrated bedtime audio in adults is mixed, and a recent active-controlled trial did not support such audio as a standalone treatment for sleep disturbance. We translate this uncertain evidence into conservative engineering requirements and present a multilingual pipeline for generating and narrating low-arousal sleep stories. The implemented prototype separates deterministic planning from large language model generation, represents content and delivery parameters explicitly, shapes each script through a four-phase attention taper, validates structural and lexical constraints, performs a bounded error-aware retry, and quarantines unresolved outputs. Provider-agnostic text-to-speech synthesis is followed by paragraph-aware joining, two-pass loudness normalization, and a publication gate that re-measures the saved, encoded file against fixed loudness, loudness-range, and true-peak limits. The product data model records stop position, replay, early abandonment, and completion while treating these events as ambiguous behavioral proxies rather than clinical outcomes. We contribute an evidence-to-requirement mapping, a traceable guardrail architecture with explicit acceptance thresholds, technical evaluation criteria, and a prospective protocol for testing sleep-onset effects. No clinical efficacy is claimed; the system is positioned as a relaxation and attentional-support tool whose benefit and possible reliance must be evaluated empirically. 

---
# Kernel-based Interpolatory Methods for Context Compression of Attention Mechanisms 

**Authors**: Michel Fabrice Serret, Laura Grigori  

**Link**: [PDF](https://arxiv.org/pdf/2610.40943)  

**Abstract**: The bottleneck of attention mechanisms, which are the fundamental building block of modern language models, is their quadratic complexity in terms of context size. There has thus been an increasing interest in the compression of the context, also called KV-cache compression. In this article, we consider the problem of compressing the context of attention mechanisms by first formulating it in the Hilbert space associated with its kernel. In this formalism, attention can be represented as the ratio of two linear maps whose interpolant is also the ratio of interpolants, this allows us to consider the interpolation of the linear maps directly. We thus introduce two novel KV-cache compression methods which we call GAIM and Q-DAIM, based on the Generalized Empirical Interpolation Method (GEIM) and the Q-Discrete Empirical Interpolation Method (Q-DEIM), respectively. As the cost of these methods is quadratic in terms of context length, we propose surrogate models based on the Taylor series, the Performer and the Random Fourier Features method which allow the compression to be done in linear time. Finally, we extend this result to the streaming attention setting, providing new methods which allow one to penalize the selection of interpolation nodes based on the previously interpolated context and to update the previous interpolation based on new information. We then provide numerical results on the convergence of the residuals and the output error in future queries and preliminary results on the LongBench-E benchmark showing performance comparable to the state-of-the-art methods. These experiments suggest that the range revealed by our Q-DAIM method could help identify where information is concentrated in the KV cache, thus enabling an efficient compression. 

---
# Agentic World Modeling for Agent Test-time Scaling without Environment Replica 

**Authors**: Wendi Li, Huseyin A. Inan, Wei-Ning Chen, Yanzhi Chen, Chetan Bansal, Saravan Rajmohan, Sharon Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.40927)  

**Abstract**: Test-time scaling (TTS) improves large language model agents by sampling and selecting among multiple candidate trajectories, but common realizations assume that these trajectories can be executed in parallel. This assumption breaks in realistic deployments, where enterprise workspaces and cloud services are shared and persistent, costly or infeasible to replicate, and actions may introduce irreversible side effects or cross-trajectory interference. Existing generative world models avoid direct execution, but typically compress environment state into a single prompt, limiting their ability to reliably simulate transitions under partial observability. Instead, we introduce Agentic World Modeling (AWM), which reformulates next-observation prediction as a multi-turn evidence-acquisition task. For each hypothetical action, an agentic simulator adaptively queries the real environment through safe read-only interfaces, reasons over the returned evidence, and uses executable computation to construct the resulting observation without mutating shared state. Integrated with trajectory-level TTS, AWM enables parallel exploration over a single shared environment. Across $\tau$-bench, AppWorld, and OfficeBench, AWM outperforms a single-shot simulation baseline by 40.07% on average, and surpasses a trained world-model baseline by 12.36% on average. These results establish interactive, read-grounded simulation as a practical foundation for safe and scalable agentic test-time scaling in shared, non-resettable environments. 

---
# Incongruence-Aware Evaluation of Multimodal Speech Emotion Recognition in Audio Large Language Models 

**Authors**: Fengji Jing, Guy Laban  

**Link**: [PDF](https://arxiv.org/pdf/2610.40883)  

**Abstract**: Voice-interactive large language models process both what a speaker says and how they say it, yet the two channels do not always agree. We ask whether that disagreement degrades speech emotion recognition. Mapping text and audio into a shared valence-arousal-dominance space, we convert the distance between them into a continuous, threshold-free incongruence score with a two-component Gaussian mixture, and validate the score against human judgement. We evaluate three audiocapable models on acted, conversational and spontaneous speech (eNTERFACE'05, MELD and CMU-MOSEI). With the model held constant across modalities, incongruence raises the odds of error whenever audio is present, in every corpus, after controlling for the ground-truth emotion and for distance in the feature space. Supervised classifiers trained on the same features behave in the opposite way, which suggests that the difficulty lies in how these models combine raw evidence rather than in the signal itself. Text-only effects are unreliable: swapping only the model that reads the transcript reverses their sign on one corpus. Rebuilding the score without the emotion lexicon reproduces the central result, and among six simpler alternatives it is the only label-free score that predicts error in all three corpora. Code, data tables and prompts are released. 

---
# FARO: Function-Space Adaptive Return-to-Risk Optimization via Cross-Block Equal-Risk-Contribution Budget Allocation 

**Authors**: Xiaojun Wu, Cehao Yang, Xuhui Jiang, Chengjin Xu, Jia Li, Jian Guo  

**Link**: [PDF](https://arxiv.org/pdf/2610.40858)  

**Abstract**: Training a large language model is compute-bound, and current recipes divide the work between optimizers by hand: orthogonalized momentum for attention and MLP matrices, Adam-style updates for the rest. This division rests on human priors about which layer type deserves which geometry, and it fixes a single global update budget that every parameter block shares. We introduce FARO, a function-space adaptive optimizer that makes the partition of parameters into blocks the unit of decision. FARO gives each block a geometry by its realized counterfactual return, then freezes the routes and reallocates the frozen per-block budget by cross-block equal risk contribution (ERC), the risk-parity principle from portfolio theory. On a GPT-style model trained on FineWeb, the framework gives two results. First, its FARO-Adaptive arm reaches 3.2780 nats at the full schedule, matching NorMuonAndAdam (3.2794), the best hand-designed recipe: it recovers the human prior without hand-setting per-layer geometry. Second, FARO-ERC improves on that arm: it lowers the clean-data loss at every one of 6 seeds, by 0.0136 nats on average, a modest but robust gain. Its sign test, paired test, and confidence interval all agree, and it exceeds the measured run-to-run noise at no added time or memory. It works by trimming the riskiest blocks rather than expanding the safest, and it holds up under a corrupted stream, still lowering the mean loss and winning the seed majority. The source code is available at this https URL. 

---
# Safin-1.5: Safety-Native Architecture and Verifiable Reasoning for Recursive Safety Improvement 

**Authors**: Jingnan Zheng, Cheng Jin, Ming Zhang, Huan Sun, Zhongtian Ma, Kaisen Yang, Ziyuan Zhou, Tong Wu, Dongcheng Zhang, Yi Zhang, Xiao Wang, Han Qi, Shu Yu, Ermo Hua, Zongji Yu, Chaozhuo Li, Xiaoyu Wen, Zhekai Chen, Jiawei Zhou, Sirui Chen, Ning Ding, An Zhang, Tat-Seng Chua, Xiangnan He, Xia Hu, Bowen Zhou, Qiaosheng Zhang, Youbang Sun, Xiang Wang, Zhongyu Huang, Chaochao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2610.40751)  

**Abstract**: Recursive self-improvement (RSI) enables large language models to advance their capabilities through successive updates, but these updates can also introduce new risks and weaken existing safeguards. In line with the AI-45° Law, we pursue Safety RSI, in which safety develops as an intrinsic capability alongside general model capabilities through continual risk discovery and repair, while preserving utility and limiting over-refusal. We introduce Safin-1.5, a model built on a safety architecture and trained through recursive safety improvement, with an inference-time harness that extends its capabilities to formal reasoning. Our approach comprises three components: i) The safety architecture combines persistent safety states with hazardous-knowledge disentanglement. The former retains and selectively invokes safety knowledge, while the latter relocates targeted hazardous knowledge into removable experts, enabling its removal with minimal impact on general capabilities. ii) Recursive safety improvement establishes a discovery-evaluation-repair loop. The loop identifies new vulnerabilities and uses experimental feedback to develop repair strategies. It accepts updates only when they improve safety while preserving general capabilities and limiting over-refusal. iii) The formal reasoning harness combines the model's language understanding with symbolic inference and verification. Proof-obligation-guided semantic bridging recovers missing relations in formal representations, enabling symbolic solvers to construct checkable proofs under formalized premises and accepted semantic bridges. Experimental results demonstrate that Safin-1.5 achieves safety and reasoning performance, outperforming frontier models. 

---
# Prevented is Not Protected: What Attack Success Rate Hides about RAG 

**Authors**: Chahana Dahal, Zuobin Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2610.40742)  

**Abstract**: Retrieval-Augmented Generation (RAG) is vulnerable to knowledge poisoning, where adversaries manipulate retrieved evidence to influence model outputs. Existing defenses are primarily evaluated using attack success rate (ASR), which obscures defense-induced failures and the mechanisms behind apparent attack prevention. To address this issue, this work introduces \textbf{Aligned RAG Evaluation}, a matched-query protocol that jointly examines clean and poisoned responses, with and without defense. By combining matched conditions analysis, LLM-judge endorsement, and retrieval dependence controls, the proposed framework distinguishes prevented from introduced failures, target mentions from LLM endorsement, and single-passage influence from evidence composition. Experiments on HotpotQA and Natural Questions reveal that existing defenses can introduce unexpected target-positive answers, although they can prevent some attacks, while evaluation criteria can also change defense rankings. Our findings expose critical blind spots in ASR-based evaluation and motivate a more comprehensive assessment of RAG poisoning defenses. The code and the evaluation artifact are available at this https URL. 

---
# Apriel Voice 27B Preview: An Open Model for Speech, Reasoning, and Tool Use 

**Authors**: Aman Tiwari, Akshay Kalkunte, Akintunde Oladipo, Jash Shah, Tara Bogavelli, Hoang Nguyen, Nishanth Madhusudhan, Vishal Juneja, Sridhar Nemala, Srinivas Sunkara, Sathwik Tejaswi Madhusudhan, Shruthan Radhakrishna, Masoud Hashemi, Kelechi Ogueji, Khalil Slimi, Saloni Mittal, Pulkit Pattnaik, Shashank Maiya, Dhruv Jhamb, Esakkivel Esakkiraja, Katrina Stankiewicz, Gabrielle Gauthier Melançon, Fanny Riols, Amit Saha, Lakshmi Sirisha Chodisetty, Nitin Aggarwal, Anil Madamala, Sagar Davasam, Segan Subramanian, Vikas Yadav, Sai Rajeswar Mudumba  

**Link**: [PDF](https://arxiv.org/pdf/2610.40738)  

**Abstract**: A capable voice assistant must do more than recognize words; it must understand requests, reason over context, sustain conversations, and act through tools. We present Apriel Voice 27B Preview, an open speech-language model that unifies these capabilities within a shared reasoning backbone while retaining text and image input modalities. Built on the post-trained Gemma 4 26B-A4B checkpoint, Apriel Voice directly processes speech and generates both responses and tool calls without requiring a separate dialogue model. We introduce a two-stage training procedure to adapt the model to speech. The first stage establishes the speech interface, while the second jointly adapts the projector and backbone through supervised fine-tuning, text replay, and source-dependent cross-entropy and forward Kullback--Leibler divergence supervision. Automatically constructed speech-conditioned dialogues provide conversational training data, while local teacher serving, overlapping teacher and student computation, and fused full-vocabulary losses enable efficient online distillation. In thinking mode, Apriel Voice achieves a score of 98.90 on Big Bench Audio, 81.13 on Speech-BFCL, and a task success rate of 65.51% on $\tau$-Voice, with a mean latency of 1.91 seconds. It also achieves 51.27% English EVA accuracy and 74.13% experience pass@1. Finally, we deploy Apriel Voice 27B Preview on a custom serving platform to support interactive voice interactions. 

---
# Agentic Systems for Calibrating Cosmological Simulations 

**Authors**: Chaipat Tirapongprasert, Matthew Ho  

**Link**: [PDF](https://arxiv.org/pdf/2610.40685)  

**Abstract**: Calibrating a cosmological simulation is an expensive, high-dimensional parameter optimization task. Previous works employ Bayesian optimization for automation, which treats the simulator as a black box and precludes the incorporation of human domain knowledge or workflow into the selection policy. We propose AgenticTuner, an agentic calibration pipeline with a present-day large language model serving as the optimizer who proposes the next parameter trial. We show that these agents can achieve competitive calibration performance at a fraction of the CPU cost when they are given informal, human-readable priors (such as codebase, documentation, and heuristic guides that seasoned scientists may possess). We hope this work serves as an early step toward AI scientists that can carry out complex, long-horizon scientific tasks. 

---
