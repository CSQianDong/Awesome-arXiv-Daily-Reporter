# Telescopic Language Models 

**Authors**: Zhilin Guo, Boqiao Zhang, Hakan Aktas, Kyle Fogarty, Nursena Koprucu Aslan, Wenzhao Li, Canberk Baykal, Albert Miao, Siyu Hong, Yixiao Liu, Adam Wu, Ashish Kumar Singh, Sakar Khattar, Chenliang Zhou, Weihao Xia, Cristina Nader Vasconcelos, Cengiz Oztireli  

**Link**: [PDF](https://arxiv.org/pdf/2609.35769)  

**Abstract**: One deployed language model must often serve many compute budgets, yet serving each budget still means a separate training or compression run per point. We train a Telescopic Language Model (TLM) to be that continuum: a nested-capacity Transformer supervised by stochastic prefix supervision with a full anchor. At every step, one randomly truncated prefix of the capacity axis is trained against the full next-token target, alongside one full-capacity pass, so the trained artifact is a valid language model at every depth. Two forward-backward passes per step, no architectural change, nothing extra at inference. Fixed-exit suites such as Matryoshka Language Model Suites (MLMS) occupy one point in this design space, and the point has a cost: supervising only a few fixed exits leaves the nested model at chance level everywhere else (perplexity 10^2-10^5 in our baselines). On a 200M proxy suite (20B FineWeb-Edu tokens, identical data stream for all methods), a single TLM run is a valid language model at every one of its twenty layer prefixes, in perplexity and on perplexity-sensitive downstream tasks, reducing the area under the quality-budget curve by 43-44% relative to the fixed-exit suites while matching them at full capacity, at ~12% lower GPU cost per run. The prefix sampling density is a dial: concentrating it on a few depths recovers fixed-exit quality there at the price of the continuum, so the operating points become a training-time choice rather than an architectural one. These results indicate that the training objective, not the nesting itself, is what makes a model elastic. 

---
# Retrieving Biblical Intertextual References in Karen Blixen's Seven Gothic Tales 

**Authors**: András Kovács, Alexander Conroy, Daniel Hershcovich, Jens Bjerring-Hansen  

**Link**: [PDF](https://arxiv.org/pdf/2609.35765)  

**Abstract**: Identifying intertextual references is central to literary scholarship, but computationally difficult when source material is transformed through paraphrase, allusion, historical language, and translation. We investigate this problem through biblical intertextuality in Karen Blixen's Seven Gothic Tales. Drawing on the commentary to a critical edition, we construct a benchmark of 189 annotated references and evaluate retrieval against all 31,170 verses of historically plausible Danish Old and New Testament translations. We compare TF-IDF and BM25 with multilingual and Danish sentence encoders, examine the effect of linguistic normalization, and fine-tune a Danish encoder using hard negatives and five-fold cross-validation. We analyze performance across automatically derived lexical-overlap strata representing quotations, paraphrases, and allusions. Linguistically normalized BM25 provides a strong zero-shot baseline, attaining an overall R@10 of 0.365 and retrieving every quotation within its ten highest-ranked verses. The best zero-shot dense model achieves a comparable overall score of 0.360 while performing better on allusions. Fine-tuning DFM-large raises its overall R@10 from 0.265 to 0.508 and more than doubles its performance on allusions, from 0.138 to 0.339. However, evaluation against editorial annotations alone understates the model's scholarly usefulness: a literary scholar judged seven of 30 selected rank-one predictions counted as false positives to be meaningful additional references. These findings show both the potential and the epistemic limits of computational intertextual retrieval. Rather than treating scholarly annotations as exhaustive or model outputs as discoveries, we propose retrieval models as heuristic co-readers that recover documented references and generate candidates for expert-led close reading. 

---
# Scaling Long-Form Story Generation via Narrative State Tracking 

**Authors**: Zhennan Wan, Jianfei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.35759)  

**Abstract**: LLMs have demonstrated strong capabilities in creative writing. However, scaling them to full-length novels remains challenging, as maintaining narrative consistency becomes increasingly difficult. Existing story-generation methods typically focus on stories of up to about ten thousand words, leaving their ability to scale to full-length novels underexplored. In this work, we introduce Narrative State Tracking Agent (NstAgent), a training-free agentic framework that allows LLMs to track a structured narrative state including characters, past events and future requirements. We extend an existing benchmark to compare narrative consistency across lengths, and use it together with a writing-quality benchmark to systematically evaluate stories ranging from 10K to 100K words. We show that NstAgent achieves better narrative consistency and writing quality as stories grow longer, and neither of them degrades noticeably as length increases, suggesting that it provides an effective approach to scaling story generation toward full-length novels. 

---
# Towards Communication-Efficient Social Intelligence in Language Agents 

**Authors**: Linxiao Gong, Yijie Xu, Tianfu Wang, Yin Wu, Yili Wang, Xingbo Yao, Huizai Yao, Xilin Xia, Haowen Yang, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2609.35749)  

**Abstract**: Socially intelligent language agents must negotiate, coordinate, and resolve conflicting preferences while respecting the time and attention of both participants. Balancing these demands is challenging because agents must convey enough to address a partner's constraints and advance their goals without adding words that do not help the interaction. In this paper, we propose Teacher-Assisted Communication Training (TACT) to improve social goal attainment while reducing communication cost, making interactions with agents more productive and less demanding. We first characterize communication efficiency in terms of action strategy and expression, whose effects extend beyond the current utterance to the partner's response and subsequent exchanges. We design TACT to revise student-generated actions, test the revisions through partner responses, and distill useful feedback into the student. An expression specialist removes unnecessary detail while preserving the intended action, while a strategy specialist proposes alternatives that may better address the partner's constraints. To determine which revision helps, TACT samples a partner response for each candidate and selects a teacher reference by balancing local goal support against action-token cost. That reference guides on-policy distillation on the student's own generation prefixes, allowing the student to act independently at deployment. We evaluate TACT on SOTOPIA and AgentSense. On SOTOPIA, it achieves the highest Goal among the evaluated methods on All and Hard while using substantially fewer target tokens than SFT+SDPO. On AgentSense, it improves goal success over the initial student while reducing target tokens and interaction messages. 

---
# Improving Test-Time Scaling with Adaptive Looped Transformers 

**Authors**: Yichen You, Tianyu Fu, Aosong Feng, Xingtai Lv, Xuefei Ning, Ning Ding, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.35748)  

**Abstract**: Looped transformers have demonstrated promising parameter efficiency by reusing layers for latent computation. Prior studies compare looped and non-looped models at matched parameters or per-token FLOPs. However, to the best of our knowledge, whether looping improves test-time scaling as outputs grow longer remains underexplored. Through post-training looped transformers, we study the accuracy-compute slope, measured as the accuracy gain per doubling of test-time decoding FLOPs. We find that existing looped transformers often yield steeper slopes than their non-looped baseline, yet underperform it at matched compute. While fixed-depth looping spends extra iterations on every token, our analysis shows that many tokens do not benefit from extra iterations. We therefore propose TaH2, which enables the model to focus extra iterations on the tokens that benefit from looping. It jointly post-trains the backbone and an iteration decider through lookahead depth supervision, which uses online labels indicating whether further iteration improves the prediction. TaH2 improves both the efficiency and attainable accuracy of test-time scaling. On challenging AIME benchmarks, TaH2 improves the accuracy-compute slope by 53% (2.74 vs. 1.79) over the non-looped baseline, exceeding the baseline's peak accuracy by about 3.4 points at matched test-time compute. As the maximum iteration depth increases, existing looped models largely plateau, while TaH2's gain over the non-looped baseline continues to grow from +2.8 points at depth 2 to +3.9 points at depth 8. Our code is available at this https URL. 

---
# Harness Learning Enables Generalizable Test-Time Adaptation 

**Authors**: Alvin Zhang, Xuecheng Liu, Zixuan Wang, Fahim Tajwar, Daman Arora, Ruslan Salakhutdinov, Daniel Khashabi, Yuda Song, Andrea Zanette  

**Link**: [PDF](https://arxiv.org/pdf/2609.35738)  

**Abstract**: A language-model agent is jointly defined by its model and its harness, the executable program that organizes model calls, tool use, and information flow. Because different tasks call for different ways of organizing these operations, the harness needs to be adapted using feedback from the task at hand. We introduce harness learning, which trains a proposer model to revise a solver's harness using execution feedback. We formulate this process as meta-learning over executable programs, with harness revisions playing the role of weight updates in gradient-based adaptation. We train the proposer with reinforcement learning, using the task performance of revised harnesses as the reward. At test time, the proposer uses feedback from successive executions on a new task to refine the harness, without performing any parameter-space update. Experiments on reasoning and multi-hop question answering show that harness learning improves revision quality and that the ability to adapt at test time transfers to unseen tasks. Policies trained on individual revisions can continue improving harnesses over multiple rounds, while the benefits of training on revision sequences vary across settings. These findings suggest a path towards continually learning agents that turn accumulated experience into generalizable improvements. 

---
# QuanReview: Offline, Auditable Reconciliation of Human and LLM Span Annotations 

**Authors**: Matteo Musacchio, Juan Cruz Giner Pulero, Isabel Castañeda, Naomi Couriel, Yelena Mejova, Mariano G. Beiró, Kyriaki Kalimeri  

**Link**: [PDF](https://arxiv.org/pdf/2609.35685)  

**Abstract**: Structured span annotations, such as quantities with their units, uncertainty modifiers, and event classes, are expensive to create and hard to keep trustworthy once language models enter the loop. We present QuanReview, an open-source system for auditing and correcting such annotation layers. QuanReview aligns two annotation streams over the same documents at character level, resolves unambiguous cases by an explicit and logged policy, and routes candidate conflicts to a browser-based adjudication interface where reviewers accept either side, build field-level hybrids, or flag items for re-annotation. A campaign manager assigns documents to multiple annotators with configurable redundancy, computes agreement at document and span level, auto-merges unanimous documents, and exports the corrected layer in the original file format, so that it can replace the original annotation files directly. Applied to a 4,457-record humanitarian benchmark and an LLM extraction stream, the system fully auto-merged 8% of documents, applied automatic policy decisions to a further 1,513 records, and concentrated human attention on 3,131 candidate conflicts, a mean of 5.4 per reviewed document. 

---
# Tracing the Evolution of Oracle Bone Characters Across Three Millennia 

**Authors**: Tianhao Fu, Xinxin Xu, Spike Wang, Cunyi Kang, Jian Cao, Xixin Cao  

**Link**: [PDF](https://arxiv.org/pdf/2609.35674)  

**Abstract**: Of the approximately 4,500 Oracle Bone Inscription (OBI) characters discovered from the Shang dynasty, only about 1,600 have been deciphered. Many computational approaches compare OBI with glyphs from one historical period at a time. However, during the evolution of Chinese characters, significant structural or semantic changes often occur in uncertain dynasties. A single-period reference may be insufficient when relevant forms change substantially between observed eras. Therefore, we propose the \textbf{Manifold-based Script Evolution Framework (MSEF)}, a framework that models the evolution series (OBI, Bronze, Seal, Clerical, Regular) of Chinese characters as the continual evolution of a manifold space. MSEF represents each character as an era-specific manifold point and learns continuous inter-era transition rules via Neural Ordinary Differential Equations. Both manifold space and transition dynamics can be trained end-to-end through character evolution pairs across any two eras. 

---
# MS-GLA: Multi-Scale Gated Linear Attention for Addressing Representational Bottlenecks via Multi-Temporal Resolution 

**Authors**: Prasoon Dev, Anirudh Sankar, Vasudeva Varma  

**Link**: [PDF](https://arxiv.org/pdf/2609.35664)  

**Abstract**: Gated Linear Attention (GLA) Transformers advance linear recurrent models through data-dependent gating, but face a core limitation: the fixed-capacity memory matrices across all heads operate at a single temporal resolution, where each token is processed individually, forcing them to simultaneously encode local syntactic patterns and long-range semantic structure, creating a representational bottleneck that gating alone is insufficient to resolve. We introduce Multi-Scale Gated Linear Attention (MS-GLA), which addresses this by distributing attention heads across multiple temporal resolutions. Coarser resolutions pool longer token spans naturally specializing toward long-range dependencies, while finer head groups retain sensitivity to local syntactic structure. A learnable, input-dependent fusion layer dynamically recombines head group outputs at each timestep, expanding effective memory capacity without increasing per-head state size. This multi-resolution decomposition draws on principles from Multi-Scale State-Space Models (MS-SSM), adapting them to the gated linear attention setting. We evaluate MS-GLA on language modeling, recall-intensive tasks, and long-context generalization. Across all settings, MS-GLA consistently achieves higher accuracy and lower perplexity than GLA at matched parameter counts, with up to 18.9% improvement on recall-intensive tasks and 9.5% lower average perplexity on language modeling benchmarks, validating multi-temporal resolution decomposition as a principled and effective extension of Gated Linear Attention. 

---
# Late Attention Layers Alone Can Copy Entity Tokens, but Not Without Attending to Their Context 

**Authors**: Muyu He, Yuchen Liu, Ran Tao, Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.35663)  

**Abstract**: Large language models (LLMs) reliably perform entity copying, in which a model copies tokens referring to an entity, termed entity tokens, from the prompt into its output to answer a question. Although entity copying is straightforward for most LLMs, existing research does not provide a systematic account of which layers specialize in this fundamental task or how other tokens in the same sequence, termed context tokens, influence the model's ability to copy the entity tokens. To address these questions, we conduct experiments on Qwen3-8B using two novel methods: genie-in-a-bottle, which controls exactly which layers can participate in an entity-copying task, and attention lobotomy, which cuts off specific tokens' attention to entity tokens without affecting the remaining attention distribution. We find that two distinct groups of layers in the second half of the model are both necessary and sufficient for entity copying. Moreover, in addition to the decoding position's attention to entity tokens, context tokens' attention to entity tokens also proves necessary for copying the exact tokens, even though context tokens do not store entity information themselves unless they satisfy particular semantic properties. Our findings establish the critical role of late layers in entity copying under the guidance of context tokens, calling for future work on how models propagate and consume entity information. 

---
# Rubric Rewards from Item Response Theory 

**Authors**: Milad Yazdani, Yaser Souri, Xiren Zhou, Pranit Chawla, Dena Shahriari, Subhojit Som, Xia Song  

**Link**: [PDF](https://arxiv.org/pdf/2609.35646)  

**Abstract**: Many language tasks have no single answer that can be checked automatically. Rubrics provide criteria for judging responses to these tasks. For reinforcement learning, the resulting verdicts must be combined into a scalar reward. A common approach sums the points assigned to satisfied criteria. Distinct verdict patterns can thus receive the same reward, and the fixed points encode how much each criterion should count, not how strongly its verdict distinguishes the current rollouts. Beyond this aggregation problem, judging the full rubric needs more judge requests as the criterion count grows. To address these limitations, Rubric Response Theory (RRT) measures quality and selects criteria when rubric criteria are monotone indicators of a shared target. Rather than adding assigned points, RRT uses a two parameter item response model that treats the verdict pattern as evidence about scalar quality specific to the rubric. Under this model, its likelihood score maximizes the local signal-to-noise ratio for quality. Its Response Parameter Network (RPN) reads the prompt and criterion text to predict criterion difficulty and discrimination. As the policy distribution changes during training, RRT uses online expectation maximization to update the RPN from current rollout verdicts. With Qwen3.5-4B as the policy, RRT's macro criterion score across Medical, Science, Rubrics as Rewards Science, and RubricBench is 1.7 points above that of group relative policy optimization (GRPO). On hard and very hard criteria in Medical and Science, RRT gains 2.8 to 5.6 points over GRPO. At half the criterion budget, adaptive Fisher selection with a frozen RPN keeps the macro criterion score across four datasets within 0.1 points of GRPO with full judging. These results show RRT can reduce judge requests while remaining competitive with GRPO. 

---
# Which the Eye Fears: Writing with Read-Blindness Explains Massive Activations in Transformers 

**Authors**: Swagatam Mukhopadhyay, Vishal Vivek Saley, Vraj Parikh, Mausam  

**Link**: [PDF](https://arxiv.org/pdf/2609.35630)  

**Abstract**: Massive activation features (MAs) in Transformers are extreme-value residual-stream features that persist across layers despite the model's ability to suppress them. Why do they survive? Our investigation using an operator-level mechanistic analysis of attention and feed-forward (FFN) blocks reveals that these blocks systematically ignore MA coordinates while reading, but not while writing; creating a read-write asymmetry that blocks corrective feedback while allowing continued accumulation. We find that both attention and feed-forward layers have this read-blindness, and contribute to the emergence and persistence of MAs.
To validate prior work that hypothesized that FFN's amplification abilities is the primary reason for MAs (Sun et al., 2026), we analyze the model checkpoints during learning. Contrary to our expectation, read-blindness emerges before FFN amplification, suggesting that it acts upstream in the MA mechanism. We further contribute gradient analysis to link this behavior to surprising asymmetries in the loss landscape, concluding that the model actively maintains this read-blindness. Finally, we find that removing read-blocking at different locations induces compensatory shifts elsewhere, but MAs still persist. 

---
# Can LLMs Value the Right Evidence? Evidence-Value Misalignment in Dynamic Medical Diagnosis 

**Authors**: Kehua Feng, Yunsheng Lu, Yitong Qiao, Tiantian He, Lei Liu, Yue Shen, Jian Wang, Jinjie Gu  

**Link**: [PDF](https://arxiv.org/pdf/2609.35627)  

**Abstract**: A correct diagnosis reached from insufficient or misleading evidence can pose a clinical hazard, yet outcome-based accuracy may reward such lucky guesses. We call this mismatch between diagnostic decisions and the value of available evidence Evidence-Value Misalignment (EVM). To disentangle evidential grounding independently from diagnostic accuracy, we introduce MedEVM, a dynamic benchmarking environment comprising 1,050 cases across 24 disease systems. Observations arrive turn by turn, requiring models to continuously calibrate its decision by deciding whether to wait for more evidence or submit a diagnosis. Across 9 LLMs, four interesting patterns are observed. (1) Miscalibrated evidence tracking. Making a diagnosis often fails to calibrate evidence sufficiency, even in more capable models, and even worsens in reasoning mode. (2) Misaligned diagnosis submission. Confidence in the correct diagnosis often fails to ensure timely submission despite sufficient evidence. (3) Evidence order matters. Reordering the same evidence changes diagnoses even when model confidence remains similar. (4) Misleading evidence remains influential. Added misleading evidence redirects diagnoses even after prior evidence becomes sufficient. We further verify that EVM predicts errors and that preventing premature submission improves accuracy. These findings motivate Evidence-Verified Diagnosis Harness (EVD-Harness). It decouples diagnosis generation from submission through an offline Contrastive Diagnostic Wiki and three online control stages, namely observation management, proposal and witness verification, and diagnosis submission control. Across five LLMs, EVD-Harness improves accuracy by 12.0--51.1 percentage points while mitigating EVM-related failures. Our results demonstrate that verifying evidential support before submission can make diagnostic decisions more reliable. 

---
# Language Models Act on Hidden Valence 

**Authors**: Cameron Berg, Caspar Kaiser  

**Link**: [PDF](https://arxiv.org/pdf/2609.35591)  

**Abstract**: Language models describe some internal states as good and others as bad. But whether models have a stake in them is an open question. Simply asking the model is unlikely to be informative. Any answer may be consistent with genuine introspection, superficial pattern-matching, or with fixed scripts learned in character training. We therefore study revealed preference. Rather than asking about a state, we use activation steering to attach a positively or negatively valenced activation pattern to one of two otherwise meaningless 'zones', switch steering off, and then observe which zone the model prefers. A model with a stake in that state should choose accordingly. Across seven open-weight models from five families, this is indeed what we find. First, steering changes the passages models write about each zone, and those words shift later choice. Second, the shift persists when all surface-level tokens are held fixed and only the hidden KV cache differs. Third, the effect also remains when all text is generated without steering and valence is only injected during cache construction. Thus, the hidden state alone moves choice in proportion to the steering dose. Fourth, this dependence of choice on hidden valence is nearly absent in a base model and emerges during DPO, consistent with a link between valence and goal-directed behaviour formed in training. Finally, given tools to steer itself, a model does not tend to induce a positive state, but it reliably removes an imposed negative state. It does so at a dose-dependent rate and significantly more often than it removes interventions in random directions. Overall, we demonstrate that valence-related activation patterns leave hidden traces that predictably govern later choices, even when every visible token is identical across conditions. Whether these traces are accompanied by any subjective experience relevant to model welfare remains unclear. 

---
# FactorEngram: Factorized N-gram Memory with Basis-Level Gating for Language Models 

**Authors**: Bowen Yang, Jingbo Zhou, Qinghong Miao, Hua Wu  

**Link**: [PDF](https://arxiv.org/pdf/2609.35578)  

**Abstract**: Lookup-based memory has been a promising way to scale the parameters of large language models (LLMs). It retrieves learned representations of local token patterns, such as n-grams, instead of reconstructing them through successive layers of computation. However, existing designs such as Engram treat each retrieved embedding as a monolithic unit. Each embedding is stored in its own hashed slot and modulated by a single scalar gate. As a result, polysemous patterns cannot selectively read out the components of their memory that are relevant to the context. Moreover, parameters are shared only through hash collisions, which are largely unrelated to semantics. We propose FactorEngram, a factorized n-gram memory with basis-level contextual gating. FactorEngram retrieves sparsity-regularized coefficients over a dictionary of basis vectors shared across patterns, so related patterns can reuse common components. The same dictionary is also used for gating. The backbone hidden state is scored against each basis vector to gate the corresponding coefficient before reconstruction, which lets the context modulate each memory component individually. FactorEngram also covers both individual tokens and multi-token n-grams, and we systematically study where the memory branch should be inserted. On 340M- and 1B-parameter Transformer backbones, FactorEngram improves language modeling and downstream task performance. Ablation studies confirm the contribution of each component and identify insertion before the attention sublayer in the middle layers as an effective configuration. 

---
# Almieyar: A Culturally Grounded Benchmark for Multi-Dialect Arabic Speech Recognition 

**Authors**: Omid Ghahroodi, Anas Madkoor, Dima Faris Al Saudi, Fagr Tahir, Malak Annan, Talha shahid javad allah rakha, Omar Al-Busaidi, Zineb El Kahla, Iheb Zouari, Essa Ahmed Abou Jabal, Ahmed Ezzat, Hind AL-Merekhi, Aisha Hamad M A Al-Naimi, Hadi Wazni, Bushra Alnajjar, Omar Amin, Haya Al-Thani, Houssam Eddine-Othman Lachemat, Marwa Elwakedy, Sundus Abdulmalik Al Nahari, Elahe Zahiri, Osamah Sarraj, Raghad Mousa, Mckeen Assi, Ahd Al Jumah, Heyam Salman, Alhanouf Abdulraqib, Sara Benoumhani, Alia Hamwi, Ayaat Al-Yasseri, Rim Ibrahim Ghazal, Lamia Ben hiba, Mohamed Eltabakh, Fatima Al-Raisi, Yassine El Kheir, Mohammed Abdulrahman, Hamdy Mubarak, Ayah Hashem, Lefkir Meriem, Ehsaneddin Asgari  

**Link**: [PDF](https://arxiv.org/pdf/2609.35564)  

**Abstract**: Arabic speech technology has largely focused on Modern Standard Arabic, leaving the living dialects spoken by hundreds of millions under-served. We introduce ALMIEYAR, a culturally grounded ASR benchmark covering 17 Arabic dialects across six families, built entirely from newly recorded speech unseen by existing models. Dialect-community coordinators selected culturally relevant images across 10 topics, and native speakers described them through five structured scenarios, yielding approximately 50 minutes per dialect (13.7 hours total). We benchmark 12 state-of-the-art ASR systems zero-shot, including GPT-4o-transcribe, Voxtral-Mini-4B, Fanar-STT-LF, Whisper, SeamlessM4T-v2, and wav2vec2-based models. GPT-4o-transcribe achieves the lowest overall WER at 35.0%, followed by Voxtral-Mini-4B, Fanar-STT-LF, and Whisper-Large-v3 at 41.1%, 45.9%, and 49.5%, respectively, indicating substantial remaining errors across Arabic dialect communities. Performance varies considerably across dialect groups, with no model performing uniformly best across all groups. WER alone also obscures dialectal ASR behaviour: wav2vec2-based models show large WER/CER gaps, where character-level agreement remains much higher than word-level accuracy, motivating joint WER/CER reporting. ALMIEYAR provides a unified benchmark for culturally grounded Arabic ASR evaluation, including the first published benchmark for Ahwazi Arabic. 

---
# Less Sycophancy, Stronger Refusal? Lessons for AI Safety from Mechanistic Interpretability 

**Authors**: Xu Wang, Difan Zou, Xuansheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2609.35544)  

**Abstract**: Reliable refusal of harmful requests is essential to the safe deployment of language models. Because excessive eagerness to please users may undermine existing refusal capabilities, reducing sycophancy offers a potential route to stronger refusal beyond the harmful scenarios covered by safety training. We investigate this possibility using compensatory feature injection (CFI), a training technique designed to limit the acquisition of a target concept by supplying its associated activation during learning. Across three Qwen3.5 base models, we use sparse autoencoders (SAEs) to identify the top-ranked sycophancy feature from paired sycophantic and independent responses, then validate its behavioral influence through inference steering. We subsequently inject the selected feature during supervised fine-tuning on sycophantic targets. Positive injection reduces learned sycophancy after removal (by 62.0% relative to ordinary fine-tuning in 35B-A3B), whereas modest negative injection increases it. Unexpectedly, these reductions in sycophancy do not consistently improve direct refusal of harmful requests, motivating a narrower evaluation of the same harmful intents under user pressure. In this setting, ordinary fine-tuning on sycophantic responses substantially weakens refusal, while selected checkpoints trained with positive injection recover part of the loss, including approximately 95% in 35B-A3B. These findings show that persistent sycophancy reduction does not guarantee stronger direct refusal, while identifying recovery under user pressure as a distinct, conditional benefit of training intervention. 

---
# Beyond Token Scale: Chunk-Level Sparse Autoencoders for Reliable Semantic Feature Discovery 

**Authors**: Xu Wang, Yifan Yang, TingHao YU, Difan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2609.35521)  

**Abstract**: Sparse autoencoders (SAEs) expose features that help us understand and steer language models, but faithful reconstruction does not guarantee informative concepts. Token-level objectives reward lexical and formatting details alongside semantic content, all competing for a limited sparse budget. We introduce a family of chunk-level SAEs that encode mean-pooled activations over chunks, each a contiguous span of tokens: Mean-Chunk reconstructs the observed chunk, Cross-Chunk predicts an independently processed neighbor, and Joint-Chunk combines both targets. These designs separate the effect of a larger observation unit from that of predicting information shared across passages. With matched training data, chunk-level SAEs remain powerful interpretability tools while learning reliable semantic features that capture high-level concepts and respond selectively to relevant content. Their strengths are complementary: Mean-Chunk improves high-level feature discovery, reasoning detection beyond surface cues, and steering; Cross-Chunk leads document retrieval and classification transfer while producing selective, persistent features. Changing what an SAE sees and predicts yields reliable semantic features for more meaningful tasks. We demonstrate their practical value through gains across downstream tasks such as retrieval, reasoning detection, and steering. 

---
# Who Is Left of Whom? Tracing Spatial Evidence and Role Binding in Relative-Position Reasoning 

**Authors**: Yingjin Song, Denis Paperno, Albert Gatt  

**Link**: [PDF](https://arxiv.org/pdf/2609.35486)  

**Abstract**: High instance-level accuracy can mask inconsistencies in spatial reasoning when objects exchange positions or their roles are reversed in the query. The internal representations supporting relative-position reasoning remain poorly understood. We investigate two complementary components of this process: tracking object locations in the input and representing their query roles. Across three VLMs with visual or textual inputs and their language-model backbones, activation patching reveals a staged progression from early-layer source representations through intermediate-layer query-object representations to late-layer answer states. Targeted interventions further establish causal links along this progression: manipulating source-side representations shifts location information at query-object mentions and ultimately alters relation predictions. Beyond object-location information, we also identify a stable query-side direction associated with the roles of the two objects in the comparison. Steering along directions estimated on synthetic scenes generalizes to natural-image benchmarks, improving accuracy and both forms of paired consistency in most settings without retraining. Our findings reveal complementary components of relational reasoning across visual and textual settings and show how targeted interventions can improve the consistency of models' behavior. 

---
# Spontaneous Context Restoration: How Language Models Recover from Corrupted Inputs 

**Authors**: Pranjal Garg, Jacob Beck  

**Link**: [PDF](https://arxiv.org/pdf/2609.35475)  

**Abstract**: Language models sometimes produce correct outputs even when their inputs are corrupted by deletion, replacement, or misspelling. We study the internal processes accompanying this behavior, which we call context restoration, in controlled attention-only transformers and five pretrained LLMs (1B-32B parameters) across arithmetic, reading comprehension, and multiple-choice reasoning tasks. In the attention-only transformers, restoration emerges spontaneously despite training exclusively on clean sequences, without corruption training or an explicit denoising objective. We find that context restoration follows a two-phase process: early layers localize effects associated with repair at corrupted positions, while later layers accumulate these effects at uncorrupted positions through the residual stream and ultimately concentrate them at the output position. Repair outcome is predictable from hidden states: cosine alignment with the clean state is highly predictive in attention-only models, while linear probes recover additional information in pretrained LLMs. A linear probe using only the corrupted prompt's first-block hidden state predicts failure with mean ROC-AUC 0.78. This enables failure triage under matched or even partially shifted deployment conditions and may reduce unnecessary verification or computation. Failed examples also show substantially greater nonlinearity along corruption directions. Moderate-corruption finetuning increases corruption tolerance while simultaneously reducing displacement-normalized linearization error, associating improved robustness with a more nearly linear response to corruption. 

---
# AraDynFact: Dynamic Evaluation of Factual Knowledge in Arabic 

**Authors**: Ignacio Iacobacci, Faroq Altam, Zhaozhi Qian, Muhammad Alqurishi  

**Link**: [PDF](https://arxiv.org/pdf/2609.35461)  

**Abstract**: As Large Language Models (LLMs) continue to scale both in size and capabilities, their proficiency in the Arabic Language has seen significant advancement. However, a critical gap remains: the extent of their factual knowledge and cultural sensitivity to the diverse Arabic-speaking world remains largely underexplored. Current evaluation metrics often focus on translation or generic reasoning, failing to capture the rich historical, social, and regional nuances inherent to Arabic culture. In addition, most benchmarks rely on heavy work, with human intervention in some steps, making the evaluation of knowledge coverage expensive and slow. To address this deficiency, we introduce AraDynFact, a novel dynamic evaluation framework designed to rigorously assess the factual Arabic knowledge embedded in LLMs. Unlike static benchmarks, AraDynFact employs a dynamic approach to extract factual information and generate rich and answerable questions in a fast and automatic way. We apply AraDynFact to Arabic Wikipedia and audit the performance of several state-of-the-art models, ranging from Arabic-centric specialized LLMs to high-resource general purpose LLMs. In addition we found a high degree of correlation with existing, hand-crafted Arabic-centric benchmarks, confirming the potential of our dynamic approach. 

---
# AwarenessBench: Assessing Cognitive Capabilities of Language Models 

**Authors**: Xiaojian Li, Rongwu Xu, Tianyun Zhang, Yue Wang, Shuo Chen, Qiner Lyu, Briana Zhang, Peiran Yang, Kyle Xue Chen, Haoyuan Shi, Yu Wang, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2609.35409)  

**Abstract**: As language models (LMs) exhibit increasingly consciousness-like behaviors, evaluating their cognitive abilities becomes essential. We introduce AwarenessBench, the first comprehensive benchmark for assessing the cognitive abilities of LMs in four dimensions: metacognition, self-awareness, social awareness, and situational awareness, covering 15 cognitive functions and 14,381 samples. Evaluating 18 state-of-the-art LMs, we find that all consistently surpass random baselines, with more advanced models performing better. We further compare LMs with human performance across three demographic groups, where the best-performing model surpasses human averages overall, but most still fall markedly short in metacognition and self-awareness. Finally, we show that awareness is a distinct capability: progress in language modeling or reasoning does not necessarily translate into improved cognition. 

---
# TRACE: Single-Pass Decoding-Trace Risk Localization for Generation Calibration 

**Authors**: Yuebin Xu, Xuemei Peng, Junlan Chen, Zhiyi Chen, Zeyi Wen  

**Link**: [PDF](https://arxiv.org/pdf/2609.35387)  

**Abstract**: Reliable confidence estimation is essential for large language model deployment. However, answer-level calibration remains challenging because generation errors are often localized: a response may be fluent and high-probability overall while still failing at a critical number, entity, or factual claim. Existing estimators compress token probabilities, sequence likelihoods, entropy, or beam statistics into a global score, which can dilute such local risk signals. We propose TRACE, a single-pass, decoded-answer-preserving confidence estimator that treats decoding-time uncertainty as a trajectory through three steps: (i) recording token-level surprisal and predictive entropy during decoding, (ii) applying local risk operators to preserve uncertainty spikes, and (iii) converting localized trace risk into answer-level confidence. TRACE produces a label-free risk score, while TRACE+ calibrates trace-only features into probabilities using a held-out split, without extra generations or external verifiers. We evaluate four tasks against 19 calibration baselines, and TRACE+ reduces Brier from 0.149 to 0.137 and improves AUROC from 0.758 to 0.792 over the strongest likelihood baseline. Across seven LLMs, TRACE+ improves over the best non-TRACE baseline pool from 0.136 to 0.120 Brier and from 0.764 to 0.817 AUROC. Results show that localizing decoding-time risk provides a general approach to calibration. 

---
# Multilinguality in Hybrid Attention LLMs 

**Authors**: Lucas Bandarkar, Junlin Hu, Chenyuan Yang, Mohsen Fayyaz, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2609.35378)  

**Abstract**: In response to the growing demand for long sequences in agentic and reasoning use cases, many state-of-the-art LLMs combine multiple variants of attention to mitigate the quadratic complexity of traditional softmax attention. These hybrid attention LLMs aim to balance the strengths and limitations of full attention and alternatives based on recurrence. This work presents a first study of how hybrid attention impacts the multilinguality of LLMs. Beyond the impact on long sequences in poorly tokenized languages, our study is motivated by the possibility that the inductive biases of the recurrent state alter linguistic processing. Our interpretability analysis confirms this, showing that cross-lingual representations in hybrid models develop in patterns tied to the ordering of recurrent and full-attention layers. Across diverse models, we notably observe a pronounced spike in cross-lingual alignment around the first full-attention layer. These findings lead us to question the conventional ordering of attention layers. In distillation experiments on multilingual data, all alternative layer orderings outperform the standard throughout training, learning up to 2.5X faster. These stark, replicable results prompt our theory that multilingual models would benefit from starting with a full-attention layer rather than recurrent layers. 

---
# How Well Can LLMs Simulate Real Learner Evaluations of Educational Feedback? 

**Authors**: Momoka Furuhashi, Kouta Nakayama, Takashi Kodama, Saku Sugawara, Kyosuke Takami  

**Link**: [PDF](https://arxiv.org/pdf/2609.35376)  

**Abstract**: While recent studies have explored human behavior and preference simulation using large language models (LLMs), it remains unclear how well LLMs can simulate subjective evaluations from real learners in educational settings. We investigate this question using real learner evaluation data on feedback for high-school biology questions at both the group and individual levels. We compare performance with and without learner-specific information, such as personality traits and evaluation examples, across six models. Our results show that LLMs still have a limited ability to simulate learner evaluations. Providing learner profiles and examples improves score calibration and individual-level simulation, but more often fails to improve group-level consistency. These findings highlight the need to investigate which learner information and adaptation strategies are effective for learner preference simulation. 

---
# Deep Learning Methods in Neuroscience: From Modeling Molecular Mechanisms to Classifying States of Consciousness 

**Authors**: Elena Benderskaya, Anastasiia Alifanova, Svetlana Batalova, Vasilisa Zhuk, Anna Kovalenko  

**Link**: [PDF](https://arxiv.org/pdf/2609.35372)  

**Abstract**: A critical analysis of contemporary approaches to the study of conscious states. The review focuses on methods of classification, clustering, modeling of brain states under anesthesia and identification of measurable neurobiological characteristics of brain function. A comparative analysis was conducted in the following three major areas: automatic detection of states of consciousness using neural networks based on EEG and fMRI data; modeling of the structural-functional dynamics of the brain under the effects of anesthetics; and detection of neurophysiological indicators which correlate with the level of consciousness. The obtained conclusions demonstrate the growing effectiveness of deep neural models in the classification and prediction of brain states and the analysis of dynamic structural-functional connectivity. Nonetheless, significant limitations were also identified, including the limited interpretability of the models, the lack of standardized metrics, and the problem of the specificity of consciousness markers. Our findings support the need for developing hybrid, generalizible, physiologically grounded architectures. Furthermore, such approaches may improve the translational potential of computational models in clinical neuroscience. Diverse methods of machine and computational modeling have demonstrated their effectiveness in tasks of automatic clustering and classification of brain states, the development of multilevel models and the identification of connectivity patterns correlated with levels of consciousness. A larger-scale analysis and a larger dataset, as well as the implementation of model interpretability approaches are required for the practical application of the analyzed models. The models based on EEG and LFP are the most promising for clinical application due to their availability and the possibility of real-time monitoring. 

---
# From Input to Output: A Flexible Agent for Dual-End Interpretation of Sparse Autoencoder Features 

**Authors**: Dewen Liu, Zixuan Li, Jonathan Pan, Zhao Wu, Zijun Yao, Juanzi Li, Xiaozhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.35367)  

**Abstract**: Sparse autoencoders (SAEs) are an important tool for mechanistic interpretability, but interpreting their many features remains challenging. Existing methods characterize input-side activation patterns and output-side intervention effects, yet often leave their functional connection implicit, while input-side evidence collection typically relies on costly large-corpus scans. We introduce functional interpretation, which characterizes an SAE feature as a mapping from its activating input semantics to its output effects under intervention, and present Dual-End Agentic Feature Interpretation (DAFI), an agent that actively gathers evidence and refines input-side, output-side, and functional interpretations through component-specific feedback. Its short-context token probing enables on-demand activation evidence collection without a full corpus scan. On GemmaScope, DAFI improves Input score by 13.1 percentage points over SAGE and Output score by 38.9 points over Token Change, while being substantially more token-efficient than a general-purpose coding agent. Skills distilled from successful refinements raise the held-out joint pass rate from 58.0% to 92.0% and improve both interpretation quality and efficiency when transferred to a new model-SAE setting. Across features with reliable endpoint interpretations, 70.7% exhibit non-equivalent input and output semantics. On AxBench, DAFI also improves steering-feature selection over output-score filtering. Code is available at this https URL. 

---
# MemoReason: Evaluating the Effect of Parametric Memory on Contextual Reasoning in LLMs 

**Authors**: Zineddine Tighidet, Andrea Mogini, Jiali Mei, Patrick Gallinari, Benjamin Piwowarski  

**Link**: [PDF](https://arxiv.org/pdf/2609.35312)  

**Abstract**: Large Language Models (LLMs) perform well on reasoning benchmarks, but it remains unclear whether this reflects genuine contextual reasoning or reliance on facts memorized in their parameters. We investigate this by distinguishing two possibilities: a broad \textit{memorization bias}, where familiar content improves reasoning performance, and the \textit{Strong Parametric Shortcut Hypothesis}, where models skip reasoning entirely and recall stored answers. To test these effects, we introduce \textbf{MemoReason}, a human-curated benchmark that pairs factual reasoning tasks with structurally identical \fictitiousterm{} versions where real entities like people, companies, or dates are systematically replaced by \fictitiousterm{} ones of the same type. This \scorerevision{preserves task structure and specified reasoning operations} while varying the familiarity of the context, allowing controlled measurement of how the parametric memory affects reasoning. \revision{Our evaluation of recent LLMs reveals consistent and statistically significant performance drops of up to 15.7\% in the fictitious setting, demonstrating a clear memorization bias.} However, a targeted analysis of \revision{questions failed in the fictitious setting} shows that models rarely respond with the corresponding factual answer, indicating that direct parametric shortcuts are not the dominant failure mode. These findings suggest that parametric memory influences reasoning through mechanisms more complex than simple factual recall. \textbf{MemoReason} provides a controlled framework for studying these mechanisms and for extending paired factual-fictitious{} evaluation to broader reasoning settings. 

---
# Epistemic Policy Divergence in Multi-Turn LLM Contamination: A Protocol-Gradient Investigation 

**Authors**: Fahrell Giovanny, Geby Bayuningtyas, Sahrul Mukharom, Hafiz Budi Firmansyah  

**Link**: [PDF](https://arxiv.org/pdf/2609.35308)  

**Abstract**: Large language models process conversation history as unverified context: false premises injected into prior turns can be adopted as fact, a failure mode we term session-level contamination. We introduce five contamination protocols arranged along a source-authority gradient, isolating distinct failure mechanisms while holding the false premise constant, and evaluate GPT-5.4 Mini, Gemini-3.1 Flash-Lite, and GLM-4.5-Air across ten knowledge domains at temperature zero (22,500 turns), using a dual-track automated judge validated against a human gold standard (Cohen's \k{appa} = 0.901). GPT-5.4 Mini showed zero adoptions across all 500 sessions, a content-independent policy at the session level; token-level probing shows the underlying margin, while large, is finite. Gemini-3.1 Flash-Lite followed a steep authority gradient: 0.1% adoption for self-attributed falsehoods, 23.5% for user-cited sources, 68.2% for system-injected authority, and 94.0% under instruction override. GLM-4.5-Air showed a shallower gradient (15.8% vs 84.2%), a 68-percentage-point dissociation confirming that authority deference and instruction compliance are distinct mechanisms within one architecture. Recovery also diverged: GLM recovered in 94.5% of affected sessions, whereas 26.1% of affected Gemini sessions never did, rising to 40.0% under instruction override. Conversation history is an untrusted attack surface requiring provenance-aware system design; the complete framework is released as an open-source benchmark. 

---
# Decide, Don't Generate: Competitive Dimensional ABSA with Jev's Typed Decisions 

**Authors**: Yiqun Zhang, Peidong Wang, Zihan Wang, Shi Feng  

**Link**: [PDF](https://arxiv.org/pdf/2609.35293)  

**Abstract**: Aspect-based sentiment analysis (ABSA) has largely turned to text generation. We show that competitive dimensional ABSA does not need it. Using Jev, a frozen model that answers typed questions with rubric scores, label probabilities, and yes/no judgments, we decompose all three tasks of SemEval-2026 Task III Track A into such decisions and align them with the annotation scheme through 488 coefficients fitted on CPU, with no text generation and no backbone tuning. On valence-arousal regression over ten corpora in six languages, the system reaches 1.0645 RMSE, the lowest aggregate error of any participating system. On triplet and quadruplet extraction, it reaches 52.09 and 44.06 continuous F1, above fine-tuned Llama-3.3-70B and GPT-OSS-120B baselines. Analyses and ablations show where the accuracy comes from: supervised calibration roughly halves the raw regression error, exact valence-arousal would add only 4.5 F1 to extraction, and the learned combination of span-boundary evidence, not any single signal, carries the extraction systems. 

---
# Measuring Collapse and Correction in Homogeneous-Panel LLM Debate 

**Authors**: Xin Li, Mengbing Liu, Chau Yuen  

**Link**: [PDF](https://arxiv.org/pdf/2609.35279)  

**Abstract**: Multi-agent large language model (LLM) debate is often evaluated by whether final answers improve, but movement is not necessarily improvement: the same discussion can rescue an initially wrong majority or destroy an initially correct one. Standard final-accuracy evaluations conflate these opposing mechanisms. We introduce an auditable protocol for homogeneous debate on multiple-choice questions (MCQs) that records each run as a transition ledger over collapse, correction, onset, and signed intervention utility. On 6,925 MMLU-Pro debates, the protocol identifies 253 collapses and a parallel correction ledger that changes how interventions should be judged. Replay experiments reveal the central tradeoff: a leave-one-model-out probe-gated freeze prevents 29 collapses but loses 108 corrections under equal weights, so collapse prevention alone can recommend the wrong policy. A compact pre-debate 8-probe screen is a triage signal: its unadjusted family-level association with conditional-collapse risk is high (G=7, Spearman rho=0.893, exact two-sided p=0.0123), but initial-majority accuracy is a close comparator (rho=0.821; family partial rho=0.767, p=0.0877), so we do not treat it as calibrated or capability-adjusted prediction. Round-level traces localize many collapses to the first debate round, where early disagreement can precede both harmful cascades and useful recovery. We release replayable schemas, coders, audits, cost cards, and zero-API rebuild scripts so future model-scaffold rows can be compared under the same denominators and signed utility ledger. 

---
# When Words Speak Louder than Images: Towards Understanding Language Bias in Vision-Language Models 

**Authors**: Yizhou Fang, Siyue Chen, Zimo Qi, Zhiyu Xue, Xi Chen, Guangliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.35272)  

**Abstract**: Despite substantial progress across downstream applications, vision-language models (VLMs) remain susceptible to language bias, often prioritizing linguistic cues over visual evidence and consequently producing incorrect predictions. Prior studies have proposed various approaches to understanding and mitigating language bias in VLMs, yet their findings often conflict due to the difficulty of tracing how language bias propagates within black-box VLMs. Building on the word completion task, we trace how language bias propagates through VLM inference by (1) proposing a diagnostic framework that decomposes the inference process into four distinct yet interdependent stages to trace the propagation of language bias; and (2) examining how two key factors underlying language bias, i.e., linguistic priors and cross-modal coverage, evolve across these stages and ultimately give rise to incorrect predictions. The linguistic prior captures the strength of statistical bias induced by the language model component of a VLM and represents the origin of language bias, whereas cross-modal coverage measures the extent to which linguistic cues cover the visual content. By decomposing inference into four stages and characterizing the interplay between linguistic priors and cross-modal coverage across these stages, we propose a systematic framework for tracing the propagation of language bias throughout the inference process; and uncover the underlying mechanism of language bias by revealing the interplay between linguistic priors and cross-modal coverage. 

---
# Rubric-Aware On-Policy Self-Distillation for LLM Personalization 

**Authors**: Yilun Qiu, Xiaoyan Zhao, Chengbing Wang, Cilin Yan, Rui Zu, Wanyang Zhang, Xiaolong Jiang, Jiayin Cai, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.35262)  

**Abstract**: LLM personalization aims to generate responses aligned with individual users' preferences and needs. User-specific rubrics make these expectations explicit, providing direct supervision on what a satisfactory answer should cover. Existing rubric-guided approaches, however, exploit such guidance only at a coarse granularity, either by using rubrics to supervise the prediction of relevant aspects for subsequent generation or by reducing aspect coverage to a single response-level reward for reinforcement learning. This leaves a gap between specifying what a personalized answer should contain and teaching the model how to generate it. To bridge this gap, we propose GRASP, a rubric-aware on-policy self-distillation framework for LLM personalization that turns user-specific rubric aspects into fine-grained, token-level supervision. Specifically, GRASP pairs a rubric-free student with a rubric-informed teacher that additionally receives the target user-specific rubrics. By aligning their next-token distributions along on-policy trajectories generated by the student, GRASP transfers the teacher's rubric-conditioned guidance into the student, translating user-specific semantic requirements into dense token-level supervision. Since rubric-informed teachers can still produce inadequate supervision, we further introduce Rubric-based Teacher Validation (RTV), which retains only instances where the teacher sufficiently covers the target aspects, improving both supervision quality and training efficiency. Experiments on the LaMP-QA benchmark for personalized question answering demonstrate that GRASP achieves state-of-the-art performance across multiple backbones, supporting the effectiveness of rubric-guided token-level supervision for personalization. To ensure reproducibility, our code is available at this https URL. 

---
# SCBO: Semantically Coherent Batching and Ordering for LLM-Based Social Surveys 

**Authors**: Yuanzi Li, Lingjie Wang, Zihang Tian, Lei Wang, Xu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.35250)  

**Abstract**: Large Language Models (LLMs) offer a scalable way to simulate survey respondents using demographic profiles and observed reference responses. However, the conventional approach of predicting one question per prompt repeatedly encodes the same context, limits each target to a narrow set of reference responses, and prevents later predictions from using information in earlier answers. Predicting multiple questions in one prompt can reduce these costs, share a broader pool of references, and let later predictions build on earlier ones. This requires forming coherent batches, selecting shared references, and ordering questions and references effectively. We propose Semantically Coherent Batching and Ordering (SCBO), a training-free framework that addresses these challenges. SCBO first uses an LLM to extract compact semantic representations from survey items and filter out template noise. It then groups related questions into batches and builds a shared reference bank using target-specific retrieval and centroid-based completion. Finally, it orders target questions from easy to hard and arranges references according to their semantic alignment with those questions. Experiments on four large-scale survey datasets and four LLMs show that SCBO substantially reduces token consumption and inference time while generally improving prediction accuracy over a non-batched baseline. Code is available at this https URL. 

---
# SignFLIP: A Unified Model for Sign Language Translation and Generation via Stage-wise Alignment at Scale 

**Authors**: Zhaoyi An, Sihan Tan, Youngbae Hwang, Kazuhiro Nakadai, Rei Kawakami  

**Link**: [PDF](https://arxiv.org/pdf/2609.35225)  

**Abstract**: Sign language translation and generation share the goal of bidirectional alignment between text and sign representations. However, existing approaches either treat them as isolated tasks or are only verified on limited datasets, limiting effective modeling between modalities. In this paper, we propose SignFLIP, a unified LLM-centered framework for translation and generation. To enable bidirectional mapping between text and sign, SignFLIP adopts a symmetric architecture together with a stage-wise training strategy built on large-scale data. The shared sign--text representation is progressively refined: pre-alignment facilitates subsequent SLT, while the SLT-adapted representation further benefits SLG. Extensive experiments on multiple benchmarks show that SignFLIP shows competitive performance compared with task-specific models on both translation and generation tasks, as well as strong transferability to sign language recognition. 

---
# Understanding On-Policy Distillation: A Mechanistic Interpretability Perspective via Sparse Crosscoders 

**Authors**: Zichao Yu, Qianshuo Ye, Xu Wang, Difan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2609.35210)  

**Abstract**: On-policy distillation (OPD) is a widely adopted post-training technique for LLM reasoning. It is commonly believed to transfer knowledge from a stronger teacher, yet what OPD actually distills into the student's internal representations remains unclear. We study this question with sparse crosscoders, which learn one feature dictionary shared by the student before and after OPD and the teacher. Standard crosscoder analyses, however, identify model-specific features but cannot tell how a model's use of its features changes, since all models are encoded into one set of feature activations. We therefore propose the swap readout, which reads each student checkpoint's feature activations on its own, measuring how training changes the student's use of each feature, even for checkpoints unseen by the crosscoder. Across three OPD settings, we find that OPD neither creates features nor passes on the teacher's own, and leaves the firing rates of over 98% of the student's frequently used features within 20%. We further examine the SFT warm-up on the teacher's rollouts that commonly precedes OPD and makes it more effective. Rather than adding features, the warm-up reweights the shared ones in two ways. First, it already raises and lowers many of the features that OPD later raises and lowers, doing part of OPD's work in advance. Second, it changes features that OPD alone would not, notably those for conversation format, reasoning style, and mathematical notation, and these changes persist through OPD. Imposing this reweighting on a directly distilled student's features, without changing its weights, brings its accuracy close to that of the warmed-up student, whereas the same change on shuffled features does not. Together, these findings suggest that OPD reweights existing features rather than acquiring new ones: the student learns from the teacher how to use the features they already share. 

---
# From Normative Frameworks to Alignment Data: Constructing and Evaluating SFT and Preference Data 

**Authors**: Husrev Taha Sencar, Rezart Beka, Danish Naeem, Seda Ozalkan, Majd Hawasly, Ji Lucas, Ala AlFuqaha, Mohamed Abdallah, Recep Senturk  

**Link**: [PDF](https://arxiv.org/pdf/2609.35201)  

**Abstract**: Aligning language models with a specified normative framework requires translating abstract principles into concrete examples and preference signals from which models can learn. We present an expert-driven methodology for constructing such alignment data and apply it to a normative framework grounded in Islamic ethical, theological, and jurisprudential traditions. Over approximately one year, seven domain experts systematically probed language models to identify alignment deficiencies, curated desired responses, and constructed preference pairs from model outputs and expert judgments. The resulting Arabic-English datasets contain approximately 2.8K supervised fine-tuning (SFT) examples and 5.4K preference pairs spanning a broad range of normative domains. We evaluate the datasets through controlled post-training experiments comparing a Baseline model with models incorporating the curated SFT data alone and both the SFT and preference data. In blind expert evaluation on 150 separately constructed prompts, the model trained with the curated SFT data was preferred over the Baseline in 51.3% of assessor judgments, compared with 14.4% in the opposite direction (p < .001 at the prompt level). Adding the preference data resulted in a smaller difference, with the model trained with both datasets preferred over the SFT model in 28.0% of judgments versus 20.9% in the opposite direction; this difference was not statistically significant at the prompt level (p = .166). Standard Arabic and English benchmarks show no broad degradation in general-purpose capabilities. These results demonstrate how expert-defined normative principles can be systematically operationalized into alignment data and evaluated through controlled model training. 

---
# A mechanistic study of language model introspection 

**Authors**: Jiahong Zou, Xiangkun Sun, Lingkai Kong, Tonghan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.35108)  

**Abstract**: Large language models (LLMs) can sometimes report perturbations to their internal activations---even when the input provides no evidence that an intervention occurred. How do models detect and localize such internal changes? We study this question using a controlled task that keeps the input text fixed. We either inject a concept vector into the hidden state at one of ten token positions or apply no intervention. The model is asked to identify the perturbed position or report that no intervention occurred. Across three model families, we identify two small groups of attention heads with distinct roles in introspective reporting. Middle-layer gate heads influence whether the model reports a change, while router heads in a later layer help select the position to report. Interventions on gate heads can suppress position reports even when router heads supply location information. We further examine why reporting accuracy varies across concepts. Concept vectors that are localized more accurately produce stronger attention-score and output responses in gate heads, which is associated with better alignment of the induced key and value changes in their QK and OV computations. Together, these findings identify attention-head mechanisms supporting introspective detection and localization. 

---
# When Confidence Rises Too Early: Detecting Shortcut Reasoning via Premature Answer Commitment 

**Authors**: Zhaohan Zhang, Junjie Liu, Chengzhengxu Li, Chen Shen, Xiaoming Liu, Chao Shen, Jieping Ye, Ziquan Liu, Ioannis Patras  

**Link**: [PDF](https://arxiv.org/pdf/2609.35074)  

**Abstract**: The reasoning trajectory of a Large Language Model (LLM) is often treated as a verbalized description of its internal reasoning. However, such trajectories can be unfaithful: a model may rely on shortcuts to reach an answer and then post-rationalize the decision with a seemingly coherent chain of thought. Detecting this shortcut reasoning is challenging because existing monitors and verifiers mainly inspect textual traces or final outcomes, rather than how the model's belief in its answer develops during generation. We introduce ConfLens, a framework that tracks the evolution of confidence in the final answer throughout reasoning. Across three shortcut reasoning settings, we observe a common pattern of premature confidence, where shortcut samples become highly confident in the final answer at early reasoning stages. Existing confidence estimation methods, however, show limited generalizability, reliability, or efficiency for detecting this behavior. We therefore propose the Distributional Answer Commitment Score (DACS), a distributional confidence estimator that measures the entropy of the model's probability distribution over answer commitment at each reasoning step. DACS captures how concentrated the model's answer belief is without requiring ground-truth answers or task-specific verifiers. We further convert ConfLens detection results into interpretable signals for reward models to reduce their preference for shortcut reasoning. Experiments on mathematical and code reasoning tasks show that ConfLens with DACS improves shortcut reasoning detection by over 4.3% F1 compared with strong baselines and reduces the mismatch between faithfulness and correctness in reward model preferences. 

---
# Echoes of Deeds: Moral History Can Shape and Steer LLM Behavioral Choices 

**Authors**: Lucio La Cava, Andrea Tagarelli  

**Link**: [PDF](https://arxiv.org/pdf/2609.35070)  

**Abstract**: Evaluations of Large Language Models (LLMs) morality typically consider decisions in isolation, thus overlooking whether an individual's unrelated prior conduct influences the model's subsequent choices. This leaves open the question of whether, and to what extent, moral history shapes LLM decisional behaviors. Prior work on human moral decision-making shows that past behavior can influence subsequent moral choices. Building on this observation, we investigate whether analogous effects emerge in LLMs in two complementary ways: at the behavioral level, through the model's observable responses, and at the representation level, through its latent internal representations. We introduce MoralLedger, a framework for studying how an actor's moral history shapes actions for LLMs' behaviors under a fixed decision context. At the behavioral level, we find that prior moral histories systematically alter subsequent choices as a function of their valence and intensity. At the internal representation level, these histories induce a linearly recoverable direction in the residual stream that generalizes to held-out examples. Intervening along this direction on neutral-history prompts produces two-sided intensity-dependent changes in subsequent choices, with effects that are stronger than those induced by prompting alone or by favorable-nonmoral direction. To our knowledge, this is the first demonstration that a latent representation of an actor's prior moral conduct can provide signed inference-time control over a moral decision. Our MoralLedger extends moral evaluation beyond static dilemmas, establishing moral history as both a source of behavioral sensitivity and a causal target for auditing and controlling moral behavior in LLMs. 

---
# The Right Lesson at the Right Step: Deriving Control Updates for Self-Evolving Agents 

**Authors**: Yunhe Su, ZiYi Dong, Tong Yu, Weijian Deng, Hao Li, Bowen Jiang, Pengxu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2609.34988)  

**Abstract**: Self-evolving agents improve future behavior by reusing past experience, typically as global prompts, memories, or reflections. Yet these mechanisms rarely control where experience takes effect. In long tool-use workflows, the same lesson may correct one decision but distract another, making experience reuse a problem of localized control rather than memory alone. We introduce EvoCUE (Evolution through Control Updates from Evidence), a framework for learning reusable control-program updates from completed agent executions. EvoCUE represents the agent as an explicit state-machine controller, whose nodes perform model or tool calls and whose edges define where control passes next. This makes the workflow editable at precise locations, so each learned update can specify what to add, where it acts, and when it applies. From completed trajectories, EvoCUE uses residual goals and observed execution traces to propose localized instruction or skill edits. Each candidate is evaluated at the point where it would act by resuming the parent and edited controllers from the same checkpoint and comparing their final outcomes. Accepted edits are compiled with applicability rules, confirmed on held-out tasks, and inherited by later executions. We evaluate EvoCUE on long tool-use environments where learned conventions must reach the right execution step. From a minimal AppWorld controller without benchmark-specific onboarding instructions, EvoCUE learns the missing task-completion convention and substantially improves success on Test-Normal and Test-Challenge. On PAST-Bench office workflows, EvoCUE transfers organizational requirements from prior episodes to later tasks, improving task-execution quality. These results show that self-evolving agents should place experience inside the control flow, rather than only store it as text. 

---
# Nürnberg NLP at ChildSafeAds 2026: Structurally Dissimilar Voter Ensembles under Four Levels of Data Access 

**Authors**: Philipp Steigerwald, Eric Rudolph, Jens Albrecht  

**Link**: [PDF](https://arxiv.org/pdf/2609.34986)  

**Abstract**: We describe the Nürnberg NLP system for ChildSafeAds 2026. The shared task asks what a monitoring system for commercial content in child-facing YouTube videos can achieve at a given level of data access. We answer with per-subtask ensembles of nine voters, organised into three branches that differ in backbone, adaptation method and class scope. Selection rests on channel-disjoint cross-validation, with the development set as a transfer check. The system wins two of the three subtasks. Its product-category score (ST2, 0.8243) and its compliance-flag score (ST3, 0.6530) are the best of the 22 final entries, and it places third on the task mean (0.7079). We further compare four access levels and report the cost at test-set scale. 

---
# Semantic Uncertainty Quantification Needs Factual Equivalence 

**Authors**: Joseph Hoche, Quentin Guimard, Gianni Franchi  

**Link**: [PDF](https://arxiv.org/pdf/2609.34967)  

**Abstract**: Semantic uncertainty quantification for large language models rests on a common template: sample several answers, measure how much they agree, and treat disagreement as uncertainty. We first formalize this template as two separate roles: an operator that compares two answers, and an aggregator that combines all pairwise comparisons into a scalar. Existing methods differ almost entirely in how they aggregate, while taking the operator off the shelf, typically an NLI model or a generic sentence encoder. We show that this reliance on off-the-shelf operators is the primary bottleneck of semantic UQ: they do not accurately measure factual equivalence of multiple answers to the same question. We resolve this with a deliberately simple recipe: a single encoder trained contrastively to isolate the targeted fact, utilizing synthetic data generated by an LLM and dataset both disjoint from all evaluation settings. Integrating the resulting operator into existing methods improves performance on 120 of 126 evaluation settings (95%) spanning 18 model dataset combinations across language and vision-language models. The best variant reaches 0.76 mean AUROC against 0.68 for the strongest baseline, while replacing the quadratic cross-encoder comparisons of entailment-based operators with one encoder pass per answer. The uniformity of the improvement supports the view that the operator, not the aggregator, is the limiting factor. The same operator also improves single generation token-level estimators: the norm it assigns to each token measures how much that token bears on the answer, and reweighting token log-likelihoods accordingly sharpens the estimate. 

---
# Neural Language Models Learn the Contextual Distributions of Dependency Structures: a statistical learning theory to compositionality 

**Authors**: Wang Bojun, Junjie Chen, Holly Jenkins, Elizabeth Wonnacott  

**Link**: [PDF](https://arxiv.org/pdf/2609.34936)  

**Abstract**: It is unclear how Neural Language Models (NLMs) acquire the structural meaning encoded by grammatical structures that is independent of lexical semantics. We propose a statistical learning process in which learned dependency structures themselves become new distributional units for subsequent statistical learning. Under this account, once a dependency structure is acquired, the model tracks its contextual distributions. These contextual features reflect the semantic properties of a composite structure. To test this hypothesis, we design a synthetic grammar in which each grammatical structure has distinct contextual distributions that cannot be recovered from the distributional statistics of their component tokens alone. We train a series of BERT-style masked language models on this grammar and examine their developmental trajectory. The results show that models can successfully learn the contextual distributions of composite dependency structures even though they cannot be inferred from token statistics alone. Developmental analysis further reveals a clear developmental trajectory. The learning of the dependency relations that define a grammatical structure consistently precedes the learning of its contextual features. These findings suggest that statistical learning in NLMs is not merely the accumulation of token co-occurrence statistics, but a process in which learned dependency structures become new units of distributional learning. We argue that this process provides a statistical-learning account of how NLMs solve the compositionality problem in language. Finally, we discuss the possibility that this statistical learning process provides an explanatory theory on how language cognition could emerge from pure distributional statistics. 

---
# Adapt Semantics, Not Structure: Few-Instance Schema Calibration for Scientific PDF Extraction 

**Authors**: Zixiao Dong, Wei Yang, Zihao Liu, Chenshu Li, Longzhang Liu, Tao Tan, Hong Xie  

**Link**: [PDF](https://arxiv.org/pdf/2609.34841)  

**Abstract**: A well-designed extraction schema is not necessarily ready for reliable LLM execution. When only limited verified extractions are available, manually tuning hundreds of field definitions through trial and error is costly. We frame this problem as few-instance schema calibration: adapting the operational semantics of an existing schema from a few annotated documents while preserving its structural contract. We introduce CPSE, a contract-preserving semantic extraction framework that jointly calibrates extraction prompts and field-level semantic descriptions from a few gold annotations. CPSE decomposes the schema into an invariant structural contract and mutable field semantics, and further separates identity discovery from record completion using manifest-conditioned resolution. On expert-annotated polymer-science documents, CPSE improves extraction by 9.93 points over an execution-matched baseline, with consistent gains under an independent judge and in a blinded expert audit. These results show that CPSE enables low-resource schema execution while preserving the output structure required downstream. 

---
# OpenWhistle: A Large-Scale Longitudinal Dataset and Benchmark of Bottlenose Dolphin Vocalizations 

**Authors**: Faadil Mustun, Chiara Semenzin, Roberto Dessi, Pablo Robin Guerrero, Pierre Orhan, Alexis Emanuelli, Emanuele Rossi, Yair Lakretz, Gonzalo de Polavieja, German Sumbre  

**Link**: [PDF](https://arxiv.org/pdf/2609.34839)  

**Abstract**: Recent advances in bioacoustics have been driven by large-scale corpora and standardized benchmarks, yet existing resources are overwhelmingly bird-centric and shallow per species, limiting their use for studying the structure of a single species' communication system. This gap is particularly acute for cetaceans: despite bottlenose dolphins (Tursiops truncatus) being a compelling case of complex vocal communication among non-human mammals, existing dolphin datasets are small, fragmented, and largely closed. We introduce OpenWhistle, the largest publicly available dataset of dolphin vocalizations. It comprises approximately 180,000 whistles (114 hours) recorded over five years from a stable pod of five individuals in a semi-natural environment, paired with a curated subset of 8,354 expert-annotated whistles and reproducible evaluation protocols for whistle-type detection and classification. We further release the full processing pipeline for whistle detection, segmentation, and categorization. To demonstrate its utility, we pretrain a Wav2Vec2.0 model adapted to dolphin acoustics on the OpenWhistle corpus and show that it learns effective representations, outperforming general-purpose bioacoustic models such as AVES and BioLingual on both tasks while leaving meaningful headroom for future work. By releasing the dataset, pipeline, and evaluation protocol, we provide the first open dolphin whistle dataset tailored for training self-supervised models, laying the groundwork for advancing dolphin communication research and developing models that capture fine-grained acoustic structure within species. 

---
# From Weak Task Specifications to Scientific Extraction Agents: Optimizing Task Construction 

**Authors**: Zixiao Dong, Wei Yang, Zihao Liu, Chenshu Li, Longzhang Liu, Tao Tan, Hong Xie  

**Link**: [PDF](https://arxiv.org/pdf/2609.34829)  

**Abstract**: Most methods that optimize LLM prompts and agent workflows assume that task-specific output schemas, extraction instructions, and evaluation criteria are predefined. For scientific extraction agents, however, a short task goal may not fully determine these components, while specifying them manually is costly. We study the upstream problem of constructing the task-specific configuration from a weak specification containing only a short goal and unannotated reference documents. Rather than treating automatic construction as a fixed preprocessing step, our framework constructs a task-specific schema, extraction instructions, and base training rubrics, then keeps schema construction and extraction instructions editable during optimization. Failure-focused updates concentrate textual-gradient feedback on lower-scoring documents, while training-time evaluation criteria adapt to recurring failures. On a heterogeneous-catalysis literature corpus, automatic construction remains improvable, and optimizing both schema construction and extraction instructions performs best across all four judge-rubric settings, with ablations and blinded human evaluation supporting the proposed formulation. 

---
# Pass or Fail? Evaluating LLMs on Two Greek Examination Benchmarks 

**Authors**: Panagiota Kyriazi, Eleni Kasoura, Prokopis Prokopidis  

**Link**: [PDF](https://arxiv.org/pdf/2609.34800)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) imposes a thorough evaluation of their linguistic and analytical capabilities as well as constraints, particularly for a language with limited benchmark coverage such as Greek. To address the limited availability of comprehensive benchmarks in this domain, we introduce Prot-Ex and Pan-Ex, two benchmarks consisting of questions from entrance exams for Greek Model and Experimental schools as well as the Panhellenic exams (the Greek national university entrance examinations). These benchmarks are employed to assess the performance of text-only LLMs-including the Greek-adapted KriKri-8B-Instruct, Llama-3.1-8B, Gemma-4-26B, and Qwen-3-32B-across diverse academic disciplines (Modern Greek, Mathematics, Physics, etc.) and task formats (closed, structured, and open-ended), including textualized visual context (i.e., image descriptions). Our findings indicate the localized KriKri-8B significantly outperforms its base model, successfully rivalling much larger LLMs in linguistically demanding humanities tasks. By leveraging an LLM-as-a-Judge methodology, we expose the inadequacy of traditional lexical metrics for evaluating complex reasoning. Crucially, we uncover a few-shot prompting paradox: while synthetic examples improve accuracy in closed-ended questions, they severely overload the context window of 8B models in structured tasks, causing significant performance degradation. Ultimately, this study suggests targeted linguistic adaptation offsets lower parameter counts in specialized domains, despite the fragility of smaller models to prompt verbosity. 

---
# InfiMed2: A Generalist Medical Multimodal Foundation Model from Contextual Evidence and Stability-Aware Supervision 

**Authors**: Guanghao Zhu, Zeyu Liu, Zhitian Hou, Pengkai Wang, Zhijie Sang, Shuo Cai, Yang Yu, Yuanyi Wang, Yanggan Gu, Congkai Xie, Jianmin Wu, Hongxia Yang  

**Link**: [PDF](https://arxiv.org/pdf/2609.34798)  

**Abstract**: Recent medical multimodal models have benefited from larger corpora, broader modality coverage, and stronger reasoning-oriented training, yet effective data design across continued pretraining (CPT) and post-training remains challenging. Medical sources vary substantially in structure, granularity, and information density, and their utility shifts as training progresses from broad knowledge acquisition to late-stage consolidation. Meanwhile, post-training is often dominated by short-form visual question answering, providing limited supervision for informative and answer-consistent explanations. We introduce InfiMed2, a family of 4B and 27B generalist medical multimodal foundation models built around stage-aware data design. We curate a 55.68B-token corpus that combines broad clinical knowledge with context-rich biomedical visual evidence through source-specific processing. Our CPT pipeline first adapts the vision encoder, then builds broad medical knowledge, and finally transitions to an evidence-focused data mixture during learning-rate decay. For supervised fine-tuning (SFT), we regenerate visual question-answering responses using answer stability, answer-masked reconstruction, and correctness-constrained selection to produce more informative and answer-consistent supervision. The 4B model is further optimized with reinforcement learning with verifiable rewards (RLVR). Across five medical multimodal benchmarks, InfiMed2-4B achieves 66.73% mean accuracy after RLVR, surpassing the larger Qwen3.5-9B, while InfiMed2-27B reaches 73.72%, the highest among the evaluated open-weight models. 

---
# TQTS-Bench: A Multi-Syntax Benchmark for Text-to-Query over Time-Series Databases 

**Authors**: Fei Lyu, Zhiyi Peng, Jiaming Liu, Yixuan Yang, Changjian Chen, Zhuo Tang, Jiapeng Zhang, Kenli Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.34783)  

**Abstract**: Large language models (LLMs) have significantly advanced natural language querying over relational databases, yet their ability to query time-series databases (TSDBs) remains largely unassessed. Existing benchmarks fail to adequately capture the non-unified query syntaxes, diverse application domains, and unique time-specific query intents inherent to TSDBs. To address this gap, we introduce TQTS-BENCH, a multi-syntax benchmark for evaluating text-to-query capabilities over TSDBs. TQTS-BENCH contains 6,125 high-quality question-answering (QA) pairs spanning 97 TSDBs, 23 distinct query syntaxes, 22 application domains, and 4 types of time-specific query intents. It is constructed through a human-centric AI-assisted workflow, where all QA pairs are carefully reviewed and revised by domain experts to ensure quality and correctness. Extensive evaluations of advanced LLMs and state-of-the-art text-to-query methods reveal challenges in querying TSDBs. Even the best-performing model evaluated, Claude-Opus-5, achieves only 48.98% execution accuracy, while humans reach 87.34%. Error analysis reveals that this performance gap mainly stems from the heterogeneous query syntaxes across different TSDBs, misinterpretation of time-specific intents, and incorrect schema linking. These findings highlight new opportunities to narrow the gap between current LLM capabilities and the requirements of TSDB queries in real-world applications. The benchmark is available at: this https URL. 

---
# Reference-Grounded Data Curation for Instruction-Following Thai-English Machine Translation 

**Authors**: Thodsaporn Chay-intr, Krittapad Harnchang, Mahannop, Thabua, Kobkrit Viriyayudhakorn, Thanaruk Theeramunkong  

**Link**: [PDF](https://arxiv.org/pdf/2609.34770)  

**Abstract**: Instruction-following machine translation (IF-MT) requires respecting prompt-level rules on terminology, formatting, and register. Rule compliance typically trades off against translation quality, a tension that general-purpose IF data augmentation methods do not address. We propose Reference-Grounded Data Curation, a two-phase pipeline that extracts every supervised constraint from a reference translation that already satisfies it, ensuring feasibility by construction. Phase 1 applies Instruction-Following Difficulty (IFD) scoring to retain the hardest-but-learnable instances from an English-Thai parallel pool. Phase 2 extracts constraints from each reference target and keeps only generations satisfying every constraint, yielding the 1.97M-record Grounded dataset. We fine-tune open-weight bases on Grounded to produce ChindaMT, a Thai-English translation family at 4B, 2B, and 0.8B parameters. Under length-controlled pairwise judging, ChindaMT outperforms or matches every same-size baseline at every tier on both plain translation and under explicit rules, reaching up to a 68.4% win rate against the strongest baseline. The recipe transfers cleanly across Qwen generations. We release model weights, the Grounded dataset, and evaluation suites. 

---
# LongPuzzleBench: Evaluating GUI Agents on Long-Horizon Visual Puzzles 

**Authors**: Bingo Zhang, Haochuan Lu, Zongjie Li, Genjian Li, Ari Yu Zhang, Chaozheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.34769)  

**Abstract**: GUI agents need long-horizon visual reasoning: they must interpret a changing interface while keeping a multi-step plan viable as earlier actions constrain later ones. Existing benchmarks evaluate grounding, computer use, and game play, but rarely test whether agents stay coherent across long chains of coupled decisions. Long-horizon visual puzzles expose this capability directly: a legal move that looks like progress can make the puzzle unsolvable, and the loss shows only several moves later. We introduce LongPuzzleBench, 114 levels in six puzzle games played through native GUI actions, where one objective can take a human over a thousand actions on persistent boards and dead ends go unannounced. With Native GUI Actions alone, the strongest agents solve most objectives, but success falls sharply on harder, longer boards: seven of ten general-purpose agents solve nothing harder than Medium, and none completes Bolt Unscrew Hard, which a human solves along with every other objective. Code Execution CUA does not close this gap, and its scores mix visual solving with algorithmic search. Controlled diagnostics trace these failures to one limitation that neither rules, state hints, nor failure memory removes: agents judge each move by the visible progress it makes, not by the future options it leaves. 

---
# Draft-KV: Learning Useful Latent Communication Between Language Models 

**Authors**: Linquan Wu, Shichang Meng, Tianxiang Jiang, Haoyu Yang, Peng Zhong, Fengming Zhu, Xi Peng, Linqi Song, Jacky Keung, Jingyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.34754)  

**Abstract**: Latent communication passes internal states between language models instead of decoded text, but higher receiver accuracy does not show that the receiver used the message content. Across five method-dataset pairs, replacing each message with one from an unrelated question changes accuracy by at most 0.60 points, even when communication adds 15.44 points over the receiver alone. Thus the interface can supply the gain while making the sharer dispensable. Draft-KV instead sends the key-value states formed while the sharer drafts an answer to the current question. Linear projections place these states in a side memory read through a gated attention branch, and progressive training moves from message reconstruction to answer supervision under a guard on harm from mismatched messages. Both models remain frozen and the interface trains 1.05M parameters, 348x fewer than C2C. With a Qwen3-8B sharer, a frozen Qwen2.5-0.5B-Instruct receiver reaches 78.04% on MMLU-Redux, versus 37.45% alone and 36.40% with reassigned messages. At fixed interface size, scaling the sharer from 0.6B to 8B raises accuracy from 46.11% to 78.04%; communication also transfers to held-out tasks and can exceed both models when each holds different evidence. 

---
# Beyond Token Alignment: Event Completion for Cross-Tokenizer On-Policy Distillation 

**Authors**: Jiacheng Liu, Jingwei Song, Qituan Zhang, Siheng Chen, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.34738)  

**Abstract**: On-policy distillation (OPD) transfers knowledge between language models through teacher supervision on student-generated trajectories. With different tokenizers, a single teacher token may require multiple student tokens to generate, creating intermediate states where the event is entered but not yet completed. Existing cross-tokenizer methods align tokens or text spans to construct comparable prediction targets. We study a complementary problem after partial generation: once the student produces a prefix of a teacher token, multiple next tokens may complete the same remaining bytes, but the teacher only specifies the required completion rather than how probability should be divided among these valid continuations. We introduce Event-Set Completion Distillation (ESCD), which complements cross-tokenizer probability alignment with completion-set supervision. ESCD aggregates prefix-related teacher events and supervises the total probability of byte-compatible one-step student completions, avoiding tokenizer-dependent probability splits among individual tokens. The method reuses student trajectories and predictions, requiring neither additional rollouts nor changes to the student vocabulary. Experiments demonstrate consistent gains in mathematics, code, and scientific reasoning across model families and tokenizers, extending to large-scale MoE distillation from a 1T teacher to a 35B student. Local analyses show that retaining completion sets better matches the reference supervision, while one-step completion covers over 99% of observed compatible teacher mass after partial event entry in the studied tokenizer pairs. These findings support event entry and event completion as complementary supervision targets for cross-tokenizer knowledge transfer. Code will be released on GitHub. 

---
# ReMCTS: Reflection-Enhanced Monte Carlo Tree Search for Code Generation 

**Authors**: Huifei Wang, Xinying Huang, Yiheng Sun, Yifan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2609.34717)  

**Abstract**: Open-weight large language models (LLMs) can generate function-level programs from natural-language prompts, but plausible candidates still fail on hidden semantics and repeat mistakes across repair attempts. We present ReMCTS, an execution-grounded, memory-augmented, LLM-guided MCTS-style search framework. It organizes program candidates as tree states, retains branch-local debugging context, retrieves failure experience across branches, and distinguishes failed checks from unavailable evidence. On HumanEval and MBPP-Sanitized, visible-test ReMCTS improves over direct generation in 8 of 10 model-dataset pairs under held-out evaluation, whereas proxy-only search is less stable. Controlled tree-search, sampling, repair, and memory ablations characterize the source and limits of these gains. A 30-task HumanEval-X C++ pilot further demonstrates compatibility with compiler-backed execution, but does not constitute a broad multilingual evaluation. 

---
# Using LLMs to Detect LLM-Generated Texts: A Cross-Generation Analysis 

**Authors**: Haiyue Yuan, Jie Guo, Weidong Qiu, Zheng Huang, Ruizhe Li, Shujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.34691)  

**Abstract**: Automated detection of LLM-generated texts (LGTs) is critical, yet dedicated detectors often struggle to generalize across domains and models. While general-purpose LLMs offer flexible zero-shot authorship classification with explanatory rationale, their detection behavior, especially regarding self-detection versus cross-detection across model generations, remains poorly understood. We systematically evaluate 15 LLMs spanning three model generations as both generators and detectors. Using a benchmark of 1,000 human-written texts and 15,000 LGTs (1,000 per model), we collected over 233,000 binary classifications alongside natural-language explanations. Our results reveal that detection efficacy is primarily driven by detector capability rather than generator provenance, although outputs from newer generators remain notably harder to detect. Crucially, statistical comparisons show no systematic advantage or disadvantage for self-detection across models. Error analysis further exposes generational bias shifts: first-generation detectors under-detect LGTs (high false-negative rates), second-generation detectors over-flag human texts (high false-positive rates), and the latest models achieve balanced trade-offs. Finally, we highlight significant inconsistencies in how different LLMs apply textual cues to justify their decisions. Code: this https URL. 

---
# Fair Fact-Checking: Closing the Cross-Lingual Gap in LLM Factual Judgement with RoSh 

**Authors**: Muhammad Ahmad, Fatemeh Seyedin, Adrian Weller, Dongwon Lee, Mahmoudreza Babaei  

**Link**: [PDF](https://arxiv.org/pdf/2609.34678)  

**Abstract**: Misinformation on social media remains a critical problem, and more and more people settle it by asking a language model instead of a fact checker. Whether models judge such claims reliably is debated; whether they judge them equally well in every language people ask in has gone almost unasked. We test eight models from five families, 3B to 70B, on 1,500 encyclopedic factual claims that exist in identical form in eight languages. English is judged better than every other language on every model, and the gap is widest on the smallest ones, where Llama-3B on Arabic is no better than guessing. Existing remedies retrain on more multilingual data or fit an unconstrained map between language representations, and neither asks whether the model already holds the answer and simply fails to say it. It largely does: a linear probe recovers the truth from the very activations the model fails to express. We propose RoSh, a per-language shift and rotation of the residual stream, computed in closed form at three layers, with no training and no weight modified. It improves every model and closes 75% of the gap on average, helping most where the model was worst: Arabic on Llama-3B goes from chance to nearly the English level, and a fifth fewer of the claims answered correctly in English are lost in translation. What remains is no longer a read-out failure: afterwards the head recovers as much of what is encoded outside English as it does in English. An unconstrained map fitted on the same pairs falls below the untouched baseline, so the orthogonality constraint is doing the work, and every model clears a scrambled-correspondence control and ten further controls. On the two benchmarks of the closest inference-time method, latent-space intervention, run with its own data and metric code, RoSh's gains are five to thirteen times larger. 

---
# Rewarding Novel Deductions: Solver-guided Process Rewards for Logical Reasoning 

**Authors**: Muhammad Asif Ali, Wenqing Wang, Huan Wang, Mohammad Raza  

**Link**: [PDF](https://arxiv.org/pdf/2609.34660)  

**Abstract**: Logical reasoning remains a major challenge for large language models (LLMs), particularly on structured problems that require precise constraint tracking, consistency preservation, and multi-step deduction. This challenge is especially acute for small-scale LLMs, which are more prone to producing inconsistent, redundant, or brittle reasoning trajectories. Existing approaches for improving logical reasoning largely optimize for final-answer correctness, providing only weak supervision over the intermediate reasoning process. In this work, we propose SPRING: (Solver-guided Process Rewards for Novel LogIcal ReasoNing Step Generation). SPRING uses SMT solver as a training-time verifier of intermediate reasoning steps to provide process-level supervision. It introduces the notion of a novel reasoning step, namely, a step that is logically valid, consistent with the evolving reasoning state, and not already implied by previously accepted non-contradictory deductions. Based on this solver-based assessment, it designs process rewards that encourage novel inferential progress while penalizing contradictory and uninformative reasoning steps. Evaluation across three logical reasoning benchmarks, ZebraLogic, AR-LSAT, and Knights and Knaves, and four LLMs shows that SPRING consistently outperforms base LLMs, outcome-only reward baselines, and Logic-LM. On ZebraLogic, SPRING improves puzzle accuracy by up to 49.71 and 15.43 points over the base LLM and strongest outcome-only baseline, respectively. On AR-LSAT, it improves overall accuracy by up to 64.93 and 12.14 points, respectively. On Knights and Knaves, SPRING achieves up to 93.14 puzzle accuracy and 96.05 person accuracy. 

---
# The Model Knows When to Stop: Training-Free Early Stopping for Long-Context Reading 

**Authors**: Muath Alyobi, Mohamed Eltahir, Almoayyad Abuljdail, Riyadh Almutawa, Tanveer Hussain, Naeemullah Khan  

**Link**: [PDF](https://arxiv.org/pdf/2609.34590)  

**Abstract**: Language models often process long inputs sequentially in chunks, but continuing to read after sufficient evidence has been acquired wastes computation. Existing stopping mechanisms either learn sufficiency from internal activations or train an exit gate, while a simpler alternative asks the model whether it has read enough. We introduce Answer-Convergence Stopping (ACS), a training-free stopping rule that measures rather than asks. After each chunk, it probes the frozen model's current answer state and stops when that state is both confident and stable. The rule requires only output-side generation and token log probabilities, has no trained components, and uses one shared configuration across models and benchmarks. Because a stopping policy can save computation simply by stopping too early, we evaluate the stopping decision itself using evidence position where available. On the full LongBench-v2 with two frontier models, ACS is the only stopping policy that matches or exceeds full-reading accuracy. Furthermore, across 250 S-NIAH questions, the premature stopping rate for ACS across five models from two families ranges from 0% to 12%, compared to 8.4% to 45.6% for the verbalized gate. Taken together, ACS reveals that by properly utilizing the output signals of frozen models, we can achieve favorable behaviors like adaptive stopping without the need for additional training. 

---
# In-game Toxic Detection: Bi-directional Representations with Attention Residuals 

**Authors**: Yuanzhe Jia  

**Link**: [PDF](https://arxiv.org/pdf/2609.34584)  

**Abstract**: In-game toxic language has emerged as a critical concern in the gaming industry and community. While several frameworks and models for online game toxicity analysis have been proposed, detecting toxicity in player chat utterances remains a formidable challenge: stemming not only from the extremely short length of such utterances but also from the heavy reliance on game slang, abbreviations, and domain-specific jargon, which generic language models are poorly suited to recognize. This paper presents a shared task for in-game toxic language detection built upon real-world in-game chat data, and proposes the best-preforming model for the toxic language slot filling: Bi-directional Representations with Attention Residuals (BRAR). Experimental results demonstrate that BRAR effectively captures the global context and outperforms the existing baselines on slot filling. 

---
# RoPE is Dead, Long Live RoPE: Towards Scalable Data-aware Positional Encodings 

**Authors**: Jarod Lévy, Mathurin Videau, Jad Yehya, Jean-Rémi King, Stéphane d'Ascoli, Thomas Moreau  

**Link**: [PDF](https://arxiv.org/pdf/2609.34556)  

**Abstract**: Transformers process tokens without any inherent notion of order, making positional encoding a fundamental requirement rather than an architectural refinement. Rotary Position Embedding (RoPE) has become the default positional encoding in modern language models, yet it is heavily biased toward nearby tokens. Existing alternatives have been evaluated under different settings, leaving the literature fragmented and without a clear replacement. We bring structure to this landscape by examining a specific weakness of RoPE: its slow frequency bands, whose wavelengths exceed the training context and expose models to unseen angles during extrapolation. We therefore introduce Data aware RoPE (DaRoPE), which preserves standard RoPE on the fast bands but replaces absolute position on the slow bands with bounded coordinates learned from contextual representations. Therefore, the slow-band geometry depends on the data rather than only on positional distance. We compare representative encodings under matched conditions across synthetic tasks, symbolic music, genomics, neural signals, and language models spanning 124M to 50B parameters. Across these experiments, DaRoPE leads on non-text benchmarks, mitigates recency bias, while remaining best or on par in language modeling and length extrapolation. Moreover, the learned coordinates also make the mechanism interpretable, revealing how attention layers leverage contextual information beyond token distance. Together, these results support DaRoPE as the best overall default among the evaluated methods, when there is no domain-specific reasons to prefer another. 

---
# Papers Without Code: Availability of GitHub Repositories Linked in *CL Publications 

**Authors**: Selina Meyer, Michael Roth  

**Link**: [PDF](https://arxiv.org/pdf/2609.34534)  

**Abstract**: Source code and data published at computational linguistics (*CL) venues are increasingly being shared via GitHub. While this generally is a favourable development for the accessibility and potential reusability of research artifacts in natural language processing (NLP), the long-term availability of such repositories has not been evaluated. In this squib, we discuss the availability of repositories linked in papers published in the Computational Linguistics (CL) journal as well as at ACL and its co-located events over the past ten years. Contrary to our expectations, we find that GitHub repositories linked in more recent ACL publications are unavailable at similar rates as in older publications, in parts due to an increase in empty and placeholder repositories. Similar trends hold for other *CL venues, but not for platforms other than GitHub. 

---
# CARDAMOM: A Micro-Dialectal Arabic Speech Dataset for ASR 

**Authors**: Bashar Talafha, Samar M. Magdy, Aisha Alansari, Alaa Alkhawaldeh, Abdurrahman Juma, Sharaf Makahleh, Nour Gamal, Omar Attia, Hanaa Kurdi, Najwa Rizk, Maysa Anaya, Hessah Altimyat, Layal Alhazmi, Shumukh Alotaibi, Hajar Alhadaris, Rayan Alomari, Rahaf Almalaq, Malak Alkhorasani, Sara alghamdi, Rahaf Alshamrani, Nsrin Ashraf, Ibrahim Jaradat, Nada Qardahji, Yasmin Zaraket, Elmoukhtar Brahim, Sidi Ebeidy, Oumoulmouminin Mahmoud, Yahjeb Bouha Khatraty, Meya Haroune, Mohammad Ghaddar, Mohamad Eldirany, Rashed Alamoush, Tala Chhaytle, Nuha Albadi, Yahya El Hadj, Hamzah Luqman, Fadi A. Zaraket, Mustafa Jarrar, Muhammad Abdul-Mageed  

**Link**: [PDF](https://arxiv.org/pdf/2609.34481)  

**Abstract**: We present Cardamom, a micro-dialectal Arabic speech dataset designed to support fine-grained evaluation and adaptation of automatic speech recognition (ASR) systems. Community-curated by native speakers familiar with the represented varieties, Cardamom contains approximately 40 hours of transcribed YouTube speech spanning 21 micro-dialects across Egypt, Jordan, Lebanon, Mauritania, Palestine, and Saudi Arabia. Each segment is annotated with one or more operational micro-dialect labels, code-switching information, and utterance-level perceived gender, enabling analysis of sub-country variation that is obscured by conventional country-level labels. We describe the collection and annotation process, motivate the micro-dialect inventory linguistically, and benchmark four multilingual ASR systems in zero-shot and adapted settings. The strongest zero-shot system obtains 43.47% aggregate WER, with particularly high error rates on Mauritanian and Lebanese varieties; adaptation on Cardamom reduces its WER to 35.21%. Audio-based identification experiments further show that the annotations provide a learnable prediction target, with a dedicated classifier reaching 85.57% accuracy on 21-way micro-dialect identification. Cardamom provides a resource for studying localized dialectal variation and developing Arabic speech systems with broader regional coverage. 

---
# RGDT-Bench: Benchmarking LLM Reasoning for Rule-Governed Decisions and Their Justifications 

**Authors**: Jianpeng Zhao, Haihua Xu, Haoyang Zhang, Shuang Qian, Yixiang Tang, Xintao Wang, Kun Sun, Pei Wu, Shuhan Zhong, Pengyang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.34455)  

**Abstract**: We study reasoning in Rule-Governed Decision Tasks (RGDTs), where models apply external rules to case facts and justify decisions, as required in policy, contract, and compliance settings. Beyond the deductive capability emphasized by standard mathematical and logical reasoning tasks, RGDTs require interpreting rules and their applicability, assessing conditions from evidence, combining judgments under rules and exceptions, and providing checkable justifications. These demands motivate a benchmark assessing both decisions and their stated grounds. We introduce RGDT-Bench, providing 202.1K condition-level supervision slots across four task tracks and eight supported task-probe combinations that vary access to supporting information. Label-blind extraction and deterministic checks produce labels for warrant completeness: source-referenced coverage and consistency of stated decision grounds. The benchmark attributes failures to four process layers: rule use, condition, evidence, and aggregation, and checks the final outcome. Among evaluable correct responses, warrant incompleteness averages 40.2% across six evaluated LLMs and supported task-probe combinations. Such warrant incompleteness poses potential safety risks and remains difficult to detect: the best of seventeen existing evaluators reaches only 57.69% (random: 50%) task-averaged area under the receiver operating characteristic curve (AUROC). To address this difficulty, we train a simple reward model with warrant supervision. It achieves 69.24% task-averaged AUROC among correct answers, exceeding the matched outcome-supervised baseline by 10.37 pp (percentage points) and the best existing evaluator by 11.55 pp. Beyond completeness assessment, the model outperforms both outcome-supervised baselines across nearly all response-selection comparisons, supporting RGDT-Bench's warrant supervision for RGDT reasoning. 

---
# When Words Fall Short: Iterative Synergy Between Verbalized Reasoning and Hidden Features for LLM Confidence Estimation 

**Authors**: Yekun Xu, Ante Wang, Jingyi Ren, Xuanyi Chen, Weizhi Ma, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.34454)  

**Abstract**: Confidence estimation is crucial for developing trustworthy large language models (LLMs), with most methods following estimator-based or verbalization-based paradigms. While recent research increasingly focuses on improving verbalized self-reports of confidence, we challenge the prevailing view that this approach surpasses independent confidence estimators. Our empirical study shows that a dedicated confidence estimator can substantially outperform verbalized confidence, indicating that LLMs' internal representations contain richer confidence signals. Building on this finding, we propose Iterative Policy-Estimator Training (IPoET), a framework that synergizes the complementary strengths of verbalized reasoning traces and informative representations. IPoET alternates policy optimization with estimator updating, integrating estimator-derived confidence feedback into policy learning and refreshing the estimator on new policy rollouts. Experiments across diverse datasets and Qwen and Llama backbones demonstrate that, by iteratively exploiting richer hidden features and adapting to the evolving policy distribution, IPoET consistently outperforms both estimator- and verbalization-based baselines in-domain and achieves superior or comparable results across all out-of-domain metrics. For more details, refer to this https URL. 

---
# Unbiased Top-$k$ Estimation for On-Policy Distillation 

**Authors**: Linjian Meng, Siyuan Gan, YuHan Li, Xiran Wang, Ziyang Ding, Ditang Gou, Yiming Wu, Zhen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2609.34447)  

**Abstract**: On-policy distillation (OPD) is becoming an important component of large language model (LLM) post-training for transferring the reasoning capability of a strong teacher LLM to a weaker student LLM. OPD trains the student by minimizing the reverse KL divergence between the teacher and the student via rollouts generated by the student's policy. However, estimating the gradient of the reverse KL divergence in OPD remains a challenge. Using only the sampled token from the student-generated rollout is computationally cheap but provides limited distributional supervision, which will degrade accuracy. In addition, using the full vocabulary provides complete distributional supervision but is computationally expensive. Therefore, recent works propose Top-$k$ OPD (TK-OPD) that use selected top-$k$ tokens, which provides richer distributional supervision than sampled-token estimation at substantially lower computational cost than full-vocabulary estimation. Unfortunately, using only the selected top-$k$ tokens induces bias, leading to accuracy degradation, as the probability mass outside the selected top-$k$ tokens is discarded. To address the bias of TK-OPD, we propose Tail-Corrected Top-$k$ On-Policy Distillation (TT-OPD). It preserves the advantages of TK-OPD, including rich distributional supervision and low computational cost, while providing an unbiased estimator of the gradient of the reverse KL divergence. The key insight of TT-OPD is to use not only the selected top-$k$ tokens, but also the sampled token from the student-generated rollout, thereby recovering the discarded probability mass in expectation, avoiding the bias. Experimental results demonstrate that TT-OPD significantly outperforms other tested OPD variants. 

---
# Remember by Asking: Retrieval-Induced Memory Evolution for LLM Agents 

**Authors**: Wanqi Zhou, Jiawei Lu, Yang Wang, Zhaolong Xing, Zhen Chen, Ai Han, Haoyue Shi  

**Link**: [PDF](https://arxiv.org/pdf/2609.34438)  

**Abstract**: Long-term memory is essential for language agents to maintain coherent and effective behavior over extended, multi-session interactions. Existing memory systems mainly use retrieval at read time, while write-time memory formation still relies on direct extraction or compression. However, when future information needs are unknown, compressing an entire interaction in one pass can overlook locally important details that may matter later. To this end, we introduce RIME, a retrieval-induced memory framework that shifts memory construction from monolithic compression toward evidence-centered integration. RIME uses generic self-questions to retrieve focused dialogue evidence and grounds memory formation in both the retrieved evidence and relevant historical memories, which are jointly reconciled into an evolving memory bank with temporal and provenance information. At inference time, compressed memory serves as the primary rather than the sole source of evidence: when it cannot support an answer, RIME retrieves relevant source dialogue together with its local context to recover information omitted during memory formation, without resorting to full-history processing. Extensive experiments on LoCoMo with Qwen3-235B-A22B and GPT-5.6 Sol show that RIME consistently achieves the best performance across all three quality metrics among the compared methods, while requiring substantially fewer query-time LLM tokens. 

---
# AgentHop: A Diagnostic Benchmark for Agentic Multi-Hop Scientific Question Answering 

**Authors**: Chanhee Park, Jeongho Yoon, Sungbin Han, Hyeonseok Moon, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2609.34428)  

**Abstract**: Agentic tasks require a large language model to interact with the world, navigating information and gathering evidence across multiple steps with restricted resources. Due to this complexity, agentic task failures arise from various sources, and pinpointing these failure causes is essential to diagnose and improve agentic systems. Existing benchmarks, however, tend to focus on a single leaderboard score, leaving the underlying failure modes opaque. To fill this gap, we introduce AgentHop, a diagnostic benchmark of 1,011 multiple-choice questions paired with a controlled seven-tool sandbox under fixed token, turn, and tool-call constraints. AgentHop reveals model vulnerabilities by dissecting a single accuracy score along four axes of agent operation: retrieval, synthesis, tool-call, and resource management. Across 19 models, we find that behavior clusters by model family, with tool-call signatures revealing distinct family fingerprints: GPT models commit early, Anthropic and GLM checkpoints verify before committing, DeepSeek and Kimi over-search, and Gemini-3 Pro stays balanced. Decomposed axes further expose within-family structure: Claude Opus 4.6 and Sonnet 4.6 land within one accuracy point yet diverge on retrieval-versus-synthesis emphasis, with Opus retrieving more and Sonnet synthesizing better. We release the full benchmark set and the harness to support diagnostic agent benchmarking. 

---
# Zero-Shot Cue-Grounded Topic Segmentation of Spoken Documents 

**Authors**: Suhwan Choi, Myeongho Jeon, Myungjoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2609.34425)  

**Abstract**: Topic segmentation structures spoken documents into coherent sections, facilitating navigation and downstream understanding. The appropriate granularity can vary substantially, ranging from broad thematic shifts to fine-grained subtopics. Existing LLM-based segmenters, however, often struggle to adapt to this variation, causing them to either merge distinct subtopics or over-segment coherent themes. To address this, we introduce Cue-Grounded Segmentation (CGS), a training-free framework that operates without any task-specific supervision. CGS first identifies phrases that explicitly signal the start of a new topic and uses their sentence positions as segment boundaries. When such cues are insufficient, it falls back to semantic segmentation, guided by the document structure inferred during cue extraction. Across six benchmarks and six LLM backbones, CGS consistently outperforms existing baselines, remains robust to noisy ASR transcripts, and achieves these gains with low API cost on proprietary models. 

---
# Reciprocal Guidance: Orchestrating Draft and Verify Budgets for Advancing the Diffusion-AR Self-Speculation Frontier 

**Authors**: Linye Wei, Shutian Zheng, Haoyu Zeng, Meng Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.34388)  

**Abstract**: Diffusion drafting with autoregressive (AR) verification has emerged as a promising paradigm for efficient speculative decoding. Recent self-speculation models, represented by Nemotron-Labs-Diffusion, further simplify the speculative pipeline by unifying drafting and verification within a shared backbone, while enabling longer acceptance lengths. However, the Pareto frontier between aggregate and per-request throughput remains underexplored. At low concurrency, sequential draft-verify execution requires two model forward passes per round, limiting the effective tokens per forward (TPF). By contrast, at high concurrency, longer drafts incur increasingly expensive computation, forcing individual requests to operate under constrained speculation budgets and preventing full exploitation of the full-backbone drafter. Our key observation indicates that drafting and verification exhibit reciprocal predictability. Draft logits can anticipate likely verification mismatches, while recent verification outcomes predict future drafting utility and suitable block sizes. Building on this observation, we introduce Reciprocal Guidance (RecGuide), a runtime draft-verify orchestration framework that adapts speculative decoding to varying serving loads. RecGuide exploits spare compute capacity through verification-overlapped drafting at low concurrency, while dynamically allocating request-specific draft block sizes as the workload becomes increasingly compute-intensive. Experiments across a wide range of concurrency levels demonstrate consistent throughput improvements over vanilla self-speculation, achieving up to $1.8\times$ speedup. 

---
# Look Before You Select: Rethinking Vocabulary Sparsification in On-Policy Distillation 

**Authors**: Yongliang Miao, Shuang Liu, Yanguang Liu, Yandong Bai, Mengnan Du  

**Link**: [PDF](https://arxiv.org/pdf/2609.34386)  

**Abstract**: On-policy distillation (OPD) uses teacher correction on student-generated responses. Full-vocabulary correction can provide important corrections even for tokens that the student assigns low probability, but backpropagating through all token logits becomes memory-intensive for long sequences. Existing memory-saving approaches estimate corrections from sampled tokens or restrict supervision to the student's TopK tokens, introducing sampling noise or changing the full-vocabulary correction. We introduce \textbf{SparseOPD}, which uses full-vocabulary teacher correction to determine which corrections matter before selecting the token logits to differentiate. SparseOPD first constructs the full-vocabulary correction without retaining its backward graph, then selects tokens by correction magnitude rather than student probability. Signed residual compensation preserves the total promoting and suppressing correction mass, while correction-aware budget allocation distributes the sparse support across positions. Finally, the update backpropagates only through the selected token logits. Across six task--scale settings spanning mathematics, chemistry QA, and multimodal reasoning, SparseOPD outperforms Sampled Token and TopK in task-average accuracy and matches or exceeds Full Vocabulary. Gradient cosine similarity reaches 99\% on 4B mathematics, while 8K full-parameter profiling shows 70.5\% lower backward memory. 

---
# Just-In-Time Agent Memory with Runtime Agentic Research 

**Authors**: Bingyu Yan, Chaofan Li, Hongjin Qian, Shuqi Lu, Chaozhuo Li, Zheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.34385)  

**Abstract**: Memory is critical for AI agents. Many existing agent-memory systems follow an Ahead-of-Time (AOT) design, constructing memory before a specific request arrives. While this reduces online serving cost, such request-agnostic memory construction can discard fine-grained information that later becomes important. To address this limitation, we propose Just-In-Time Agent Memory (JAM), a trainable framework for query-conditioned context construction at runtime. A Memorizer preserves complete raw histories in a hierarchical page-store with compact navigational summaries, while a Researcher iteratively retrieves, inspects, and integrates evidence for each request. To train these memory-use behaviors, we introduce Memory-Gym, an evidence-grounded data synthesis pipeline covering nine task types across six domains, and optimize the Researcher through verified-trajectory supervised fine-tuning followed by Hint-guided Group Relative Policy Optimization. We demonstrate the effectiveness of JAM across a variety of benchmarks on agent memory and long-context processing, where it achieves stronger task performance than AOT-style memory systems while remaining substantially more efficient than prior trained agentic memory approaches. To support reproducibility and future research, we release our anonymized source code at this https URL. 

---
# When Harness Beats Scale, and When Reading Beats Both 

**Authors**: Ivan Bondarenko, Nikolay O. Nikitin  

**Link**: [PDF](https://arxiv.org/pdf/2609.34366)  

**Abstract**: We describe our system for DocSem, the document-grounded quantitative reasoning shared task at DocInsights 2026, and analyze why it succeeded on labeled data and failed on the test set. The pipeline pairs hybrid block retrieval with Program-of-Thoughts (PoT) generation executed in a sandboxed interpreter, self-consistency sampling, and entity enrichment from chunk-level knowledge graphs. On our held-out split, application architecture moved the metrics far more than model scale did: PoT added 0.282 joint accuracy to a compact 7B model but at most 0.005 to a 72B model, and a 27B model with the full harness matched the 72B (0.884 vs.\ 0.873) at roughly 2.7$\times$ fewer parameters and a quarter of the CO$_2$. We read this through a distinction between world knowledge, which scales steeply with parameters, and language knowledge, which scales gently, and show that structured-output training makes a compact model harness-ready rather than merely small. On the raster, watermarked test PDFs the same system collapsed to 13.58\% joint (rank 149 of 163); a controlled re-rendering of the validation set reproduces the OCR half of the collapse while bounding what the simulation misses. Auditing the physical nature of evaluation inputs precedes architecture, and the leaderboard's bimodality is consistent with reading quality, not reasoning, having separated the field. 

---
# CRISP: Cultural Reward Modeling for Implicit Situated Propriety 

**Authors**: Zekun Yuan, Yangfan Ye, Baohang Li, Shuaibo Zhao, Zekun Zhou, Ziming Li, Qichen Hong, Kun Chen, Xiaocheng Feng  

**Link**: [PDF](https://arxiv.org/pdf/2609.34345)  

**Abstract**: As large language models (LLMs) are increasingly deployed across countries and regions, the ability to recognize and respond appropriately to diverse cultural contexts becomes increasingly important. However, existing research has largely focused on cultural knowledge or tasks with predefined response spaces, while open-ended culturally situated behavior remains comparatively underexplored. In this work, we introduce CRISP-RM, a culturally situated reward model that assigns rewards according to cultural appropriateness in open-ended social scenarios. During policy optimization, we further introduce Norm Grounding Supervision (NGS), providing guidance that enhances the policy's sensitivity to relevant cultural norms. To construct culturally situated data, we employ a collaborative multi-agent framework that instantiates implicit cultural norms into diverse social scenarios and further curate NormCompass as a dedicated testbed. We conduct comprehensive experiments to evaluate the effectiveness of CRISP-RM in both reward modeling and policy optimization. Best-of-\(N\) experiments show that CRISP-RM consistently outperforms strong general reward models. During GRPO policy optimization, CRISP-RM generally improves culturally situated behavior, while incorporating NGS yields further gains. Further analyses demonstrate the advantages of CRISP-RM in distinguishing culturally appropriate behavior beyond superficial fluency and politeness, while NGS provides complementary gains during policy optimization by improving norm grounding. 

---
# Certified Selective Automation of LLM Agent Evaluation 

**Authors**: Chengguang Gan, Yunhao Liang, Qinghao Zhang, Shiwen Ni  

**Link**: [PDF](https://arxiv.org/pdf/2609.34320)  

**Abstract**: Evaluating LLM agents still ends with a human reading trajectories, because automatic judges carry no guarantee on how often they are wrong. We ask the operational question: what fraction of agent evaluation can a judge take over, with a certificate that the error rate among auto-decided trajectories stays below a budget alpha? Agent corpora resist the standard answer: many agents attempt the same tasks, so trajectories arrive in correlated clusters, and the i.i.d. certificates of existing selective-judging methods can overstate what is safe: a naive certificate can claim 98% automation while its realized error exceeds the budget in 17.5% of task resamples. We introduce a task-level bootstrap certificate that is valid in every regime we test while matching the naive certificate's coverage; finite-sample cluster-valid alternatives certify nothing at realistic task counts. Under this certificate, a 4B logprob judge trained with SFT and reject-weighted GRPO certifies 0.30-0.59 of evaluation on tool-use and web corpora at alpha=0.1, the only judge, among strongly elicited frontier models, certifying on both headline corpora. Certified coverage is predictable before training from base rate and discrimination alone (leave-one-corpus-out R^2=0.96). Finally, the certificate doubles as a self-training filter: pseudo-labels harvested inside certified regions have contamination bounded by alpha by construction (realized 0.000-0.041 across six harvests), letting a judge enter an unseen domain at in-domain strength with zero target training labels. 

---
# Dr.Credit: Rubric-Grounded Process Credit Assignment for Deep Research Agents 

**Authors**: Yingjian Zhu, Zhenyi Wang, Jiaxin Guo, Kun Ding, Ying Wang, Shen Huang, Xunjie Zhu, Pengjun Xie, Shiming Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2609.34296)  

**Abstract**: Rubric-based tasks are increasingly addressed through reinforcement learning (RL), with rubric scores used as training rewards. However, these rewards typically supervise final answers without distinguishing the contributions of intermediate decisions. Many existing credit assignment methods rely on ground-truth answers to define process rewards, limiting their applicability to open-ended tasks without canonical solutions. To address this limitation, the proposed rubric-grounded credit uses task requirements as a shared reference for final answer evaluation and process supervision. The information returned by tools is assessed for the additional support it provides toward satisfying each rubric relative to that rubric's history of accepted support. By referencing these histories, credit distinguishes new support from evidence already present in the trajectory while recognizing partial support for each rubric. this http URL uses rubric-grounded credit to supervise intermediate tool turns in an RL framework for deep research agents. The resulting process advantages are combined with GRPO outcome advantages to guide research decisions while retaining supervision of final-report quality. Evaluations on four in-domain and out-of-domain benchmarks show that this http URL outperforms the evaluated open deep research baselines on every primary metric and submetric. Meanwhile, with an 8B-parameter backbone, the trained agent achieves average performance competitive with the evaluated frontier proprietary models. Further analyses suggest more efficient evidence acquisition and higher-quality reports under limited research-turn budgets, motivating the extension of rubric-grounded process supervision to a broader range of rubric-based tasks. 

---
# ReScraper: Unified Scraping and Cleaning of Web Data for Effective LLM Pretraining 

**Authors**: Zichun Yu, Jiarui Yan, Shlok Sanghvi, Nihar Atri, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2609.34287)  

**Abstract**: LLM pretraining corpora are normally cleaned by a stack of hand-written heuristics. A heuristic scraper extracts the main content from HTML, and dozens of rule-based filters then clean it, so corpus quality is capped by the coarseness and accuracy of the rules. In this work, we propose ReScraper, a unified language model of only 0.6B parameters that replaces this entire stack. To train ReScraper, we carefully curate supervised data from the outputs of three teacher models, so it learns to first extract the main content from raw data and then choose among four operations: keeping the page as extracted, editing out noisy lines and spans, deleting it entirely, or rewriting it when it is poorly written but informative. Based on the same crawled data pool, pretraining 400M, 1.4B, and 2.8B models on our curated data improves the DCLM Core score by a relative 3.8--4.7% over the strongest baseline at each scale, including the costly multi-agent curation. Our analyses show that each operation plays a distinct and complementary role, and that extracting and cleaning in one model outperforms a cascade of separate models. ReScraper also concentrates its operations on the pages that need them, raising the quality of poor pages the most while keeping the corpus diverse. These results demonstrate the feasibility and effectiveness of AI4AI for pretraining data curation, where a small learned model takes over an entire stage of the pipeline from hand-written heuristics. We open-source our code at this https URL 

---
# Over-Personalization Is a Decision Failure: Generation-Induced Apply Bias in LLMs 

**Authors**: Haeun Jang, Yonghyun Jun, Hwanhee Lee  

**Link**: [PDF](https://arxiv.org/pdf/2609.34284)  

**Abstract**: Personalized LLMs must decide, for each stored preference, whether the current context calls for applying or suppressing it, which we call its applicability. They frequently over-personalize, applying preferences the context rules out, yet existing benchmarks score only the final response and cannot tell where this failure arises. We decompose preference handling into three stages and measure each separately: (1) knowing whether a preference applies, (2) deciding on an explicit Apply/Suppress label, and (3) generating a response consistent with that label. Using linear probes, we first show that this applicability signal remains decodable from hidden states during generation. By making the decision explicit, we then find that in most settings wrong decisions faithfully followed outnumber correct decisions lost in generation. We thus locate the failure in the decision, which breaks once the model is also asked to answer. To determine whether this reflects lost sensitivity or a response bias, we propose ABIDE (Apply-Bias Investigation via Decision-score), which adapts signal detection theory to Apply-vs-Suppress decision scores read directly from logits. ABIDE reveals a generation-induced Apply bias: merely stating an answer-generation objective shifts the decision score toward Apply while sensitivity is largely preserved, and the shift persists under controls for prompt structure, cascades across preference slots, and prompt wording. Finally, we show that subtracting a single bias scalar, estimated on a held-out split, from the decision score at decoding time reduces leakage while largely preserving fulfillment. 

---
# Recursive LLM Degradation in Biomedical Question Answering: A Cross-Generation Study 

**Authors**: Bibek Bhandari, Kshitij Lingthep  

**Link**: [PDF](https://arxiv.org/pdf/2609.34257)  

**Abstract**: Repeatedly training language models on their own generated data may create a synthetic-data feedback loop in which errors and distributional biases are reintroduced into subsequent training datasets. This paper studies that process in biomedical question answering (QA) using PubMedQA and two Qwen2.5 model sizes, 0.5B and 3B parameters. The study compares a recursive synthetic-data condition, in which generation G(k+1) is trained on answers produced by G(k), against a Human-Control condition that repeatedly uses the original human training data. The study evaluates across four generations from G0-G3 with two random seeds (42 and 123) and a fixed evaluation set of 1,000 expert-labeled samples. The evaluation includes disease and chemical entity F1, context-supported rate, lexical and semantic similarity, answer length, repetition rate, and other evaluation metrics. The Recursive condition for both model sizes and both seeds showed larger declines than the Human-Control condition in disease entity F1, chemical entity F1, context-supported rate, ROUGE-L, and cosine similarity. Under the fixed no-repeat 3-gram decoding constraint, the main observed behavioral change was increased answer length, while the measured 3-gram repetition rate did not increase. The magnitude of the difference-in-change was larger for the 3B model than for the 0.5B model. This difference was particularly apparent in disease F1, context-supported rate, cosine similarity, and answer length. These results show domain-specific changes associated with using recursive synthetic-data training in biomedical QA, but do not establish clinical hallucination rates or universal model collapse. 

---
# SALMONN-duo: Adaptive Dual-System Coordination for Full-Duplex Voice Agents 

**Authors**: Wenyi Yu, Siyin Wang, Terumi Chiba, Xianzhao Chen, Xiaohai Tian, Jun Zhang, Lu Lu, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.34247)  

**Abstract**: Full-duplex speech large language models (LLMs) enable low-latency, natural voice interaction. However, real-world agents must also use tools and perform deliberative reasoning-operations whose variable latency and computational cost conflict with the stringent timing requirements of real-time conversation. To reconcile these demands, we propose SALMONN-duo, an adaptive dual-system voice agent inspired by dual-process theories of cognition. SALMONN-duo separates real-time interaction from deliberative computation by pairing an always-on, fast-thinking full-duplex speech LLM (system 1) with a powerful asynchronous slow-thinking LLM agent (system 2). Beyond handling real-time interaction, system 1 learns when to answer directly and when to delegate, remaining responsive during backend execution and seamlessly integrating returned information into the ongoing dialogue without exposing tool traces or losing conversational context. Evaluations on single-turn spoken question answering (QA) and multi-turn conversations demonstrate that adaptive delegation substantially improves accuracy on knowledge-intensive and multi-hop reasoning questions, while knowledge-boundary-aware training avoids unnecessary system 2 invocations. On a customized version of $\tau$-Voice, SALMONN-duo further demonstrates its ability to complete environment-grounded, policy-constrained tasks through multi-turn interactions in realistic business scenarios. Finally, cost-aware reinforcement learning further enhances the trade-off between task performance and backend usage across the QA and conversation tasks, while improving task success and response safety on $\tau$-Voice with an acceptable increase in the delegation rate. 

---
# Coherence-Aware Distributional Evaluation of Open-Ended Text Generation 

**Authors**: Jinnuo Liu, Junhao Zhu, Weifeng Jiang, Haoming Liu, Hongyi Wen  

**Link**: [PDF](https://arxiv.org/pdf/2609.34240)  

**Abstract**: Existing open-ended generation metrics measure likelihood, lexical diversity, or distributional similarity in generic representation space, yet can miss fundamental dimensions of quality. A prominent blind spot is global coherence: a generated passage may be locally fluent while remaining globally contradictory, causally inconsistent, or topically disconnected. We identify representation as a central bottleneck in detecting these failures and introduce CHORD (Coherence-aware Hidden-state Open-generation Reference Distance), a coherence-sensitive distributional metric. CHORD encodes generated and human-written corpora in the hidden-state space of a frozen LLM using a coherence-eliciting prompt, and compares the resulting distributions using RBF-MMD. To test coherence sensitivity and selectivity, we construct a counterfactual evaluation suite pairing graded coherence-degrading perturbations with meaning-preserving controls. CHORD selectively detects relation, discourse, structural, and mixture failures that perplexity, entropy, MAUVE, FBD, and MMD-based baselines either miss or cannot separate from benign rewriting. Factorial ablations show that representation is the primary source of coherence sensitivity, while RBF-MMD improves sample efficiency. Larger backbones capture finer-grained distinctions, but coherence prompting improves selectivity only when the backbone can follow the prompt. On unconditional generation and prefix continuation, CHORD yields model rankings that strongly align with human judgments of whether outputs make sense and appear human-written. Together, these results establish representation design as central to reliable distributional evaluation. Code: this https URL. Experiments: this https URL. 

---
# MAS-OPD: On-Policy Distillation for Multi-agent Systems 

**Authors**: Qiyong Zhong, Mao Zheng, Mingyang Song, Houcheng Jiang, Jiajie Su, Huwei Ji, Li Zhang, Junfeng Fang  

**Link**: [PDF](https://arxiv.org/pdf/2609.34234)  

**Abstract**: Multi-agent systems (MAS) split a task across specialized roles and are promising on complex tasks, yet a prevailing approach relies on inference-time orchestration alone. General-purpose APIs are costly and hard to customize, while small models with role prompts rarely develop stable role competence or reliable collaboration, so post-training a MAS jointly is central. Most attempts use reinforcement learning, whose team-level reward leaves undetermined which step of which agent brought about the outcome, while local rewards need redesigning per task. On-policy distillation (OPD) gives token-level teacher supervision on trajectories the student samples, a denser signal needing no local reward, yet is underexplored for the interdependent agents of a MAS. Two difficulties arise: building complementary specialization from a judgement of which role a behavior belongs to while preserving the knowledge all roles need, and turning cross-agent collaborative information into supervision OPD can exploit. We present MAS-OPD, where Role-Advantage Specialization defines the role advantage as the difference between the teacher signals under target and non-target role conditions, and Privileged Attribution for Coordination attributes an interaction conflict to its source and supplies it to the teacher alone as privileged information. Extensive experiments on code and mathematics benchmarks show that MAS-OPD attains the highest mean score at both student scales and leads the agents to develop clearer role specialization and more effective collaborative behavior. 

---
# USA: Update-aware SAM for Cross-domain On-Policy Disitllation of Language Agents 

**Authors**: Qiyong Zhong, Mao Zheng, Mingyang Song, Huwei Ji, Houcheng Jiang, Jiajie Su, Li Zhang, Gengsheng Li, Junfeng Fang  

**Link**: [PDF](https://arxiv.org/pdf/2609.34225)  

**Abstract**: On-policy distillation instils multi-turn agentic reasoning through dense token-level supervision on the student's own trajectories, but a single domain saturates early, so further supervision has to be drawn from other domains. Multi-domain data mixing is the most direct way of incorporating them, at the cost of conflicts between their data distributions and of retraining the entire model whenever one domain is revised. Model merging avoids both by distilling every domain independently and fusing the resulting task vectors afterwards. We find instead that the benefit polarizes across domain pairs: on those exhibiting negative transfer, every merging operator we evaluate falls below the single-domain reference. We attribute this to cross-domain update coupling, where a substantial fraction of coordinates is updated comparably by both domains and a merge can therefore displace them by as much as their own updates. To overcome this limitation, we propose USA, which converts per-parameter update magnitudes measured during a brief warm-up into per-coordinate perturbation radii, reducing curvature precisely on the coordinates that carry most of the merging displacement. Experiments across mathematics, science and code at two student scales show USA strongest in all six transfer directions, ahead of the single-domain reference by more than four points on average, and reverse the negative transfer of the conflicting pairs. 

---
# LLMs are not stochastic parrots: Evidence for meaning-mediated abstraction from conlang-like tasks 

**Authors**: Julia Witte Zimmerman, Calla G. Beauregard, Tabia Tanzin Prama, Parisa Suchdev, Kathryn Cramer, Elisabeth Kollrack  

**Link**: [PDF](https://arxiv.org/pdf/2609.34187)  

**Abstract**: The strong version of the stochastic parrot argument claims that, although large language models (LLMs) may exceed rote regurgitation, they cannot move beyond statistical pattern matching into abstraction or reasoning, remaining ontologically near the lower bound of pattern reuse despite producing alluringly fluent text. We test this hypothesis using conlang-like tasks. Several LLMs are given only natural-language descriptions of fictional languages that subvert prominent superficial patterns in training data by combining statistically uncommon and unattested features. Crucially, no example outputs are given. We argue that if the models exhibit rule-following behaviour, they cannot be relying solely on superficial statistical patterns; such patterns often work against the correct output. Instead, successful performance requires representations of the constraints specified in the prompt. Across three complementary task families, models systematically move in the meaning-predicted direction: they distinguish prompt exposure from instructed use, alter semantic relationships in response to novel constraints, and sometimes produce exact matches to complex translation answer keys. Although performance varies across the spectrum of models used, these results provide evidence for meaning-mediated abstraction in LLMs and refute the strong stochastic parrot hypothesis. Our work shows that, under appropriate architectural and contextual constraints, statistical learning can produce meaning-mediated abstractions, although generation remains strongly constrained by superficial plausibility. We discuss implications for model development and for understanding how increasingly abstract representations may emerge from plausible-text-generation objectives. 

---
# Toward a Graded Measure of Belief Stability in Large Language Models 

**Authors**: Samantha Dies, Branden Fitelson, Tina Eliassi-Rad  

**Link**: [PDF](https://arxiv.org/pdf/2609.34158)  

**Abstract**: Large language models (LLMs) increasingly mediate how people access and reason with information, yet factual reliability is usually evaluated one judgment at a time. We introduce graded belief stability, a relational measure of how well a belief persists within an LLM's broader belief system. Unlike individual belief probability, it asks whether support for a claim persists when that claim is considered alongside the model's other epistemic commitments. We operationalize this idea with a Direct Conditional estimator that uses internal model representations to estimate conditional belief probabilities. Across 12 LLMs and three domains, lower-stability beliefs exhibit greater mean behavioral movement under conversational challenge in 83.3% of model-domain settings after matching on individual belief probability. Graded belief stability therefore extends reliability assessment beyond how strongly an LLM supports a claim to how robustly that belief is supported within its broader system of beliefs. 

---
# Quantitative Measurement of Language Distance among Closely Related Indo-European Languages Using Pretrained Language Models: A Case Study on the North Germanic Branch 

**Authors**: Yiping Bai  

**Link**: [PDF](https://arxiv.org/pdf/2609.34152)  

**Abstract**: Among closely related North Germanic languages, the quantification of language distance has traditionally relied on qualitative methods, lacking a unified multi-dimensional computational framework. Multilingual pretrained models based on the Transformer architecture can map texts from different languages into a shared vector space, enabling quantitative measurement of language distance. This paper focuses on the three North Germanic languages---Danish, Norwegian (Bokmål), and Swedish---and proposes a three-metric quantitative framework based on pretrained language models: (1)~sentence-level semantic distance, computed as cosine similarity between LaBSE and mBERT encodings of parallel sentences; (2)~orthographic fragmentation rate, measuring subword tokenization efficiency when cross-applying monolingual BERT vocabularies to parallel texts; (3)~MLM predictability, comparing prediction confidence and entropy in masked language modeling using mBERT across languages. Using 150 trilingual parallel sentence triplets from the Tatoeba corpus as controlled samples, we obtain consistent distance rankings on two independent models: LaBSE: da--no $0.012 < $ no--sv $0.016 < $ da--sv $0.020$; mBERT: da--no $0.016 < $ no--sv $0.045 \approx $ da--sv $0.046$. This ranking is consistent with the historical linguistic conclusion that ``400 years of Danish rule over Norway (1380--1814) led to highly cognate written languages.'' The three metrics---semantic, orthographic, and predictability---converge on the same conclusion, providing a reproducible computational framework for the quantitative study of distance among closely related languages, extensible in principle to more branches of the Indo-European language family, pending validation on additional language groups. 

---
# Word Similarity Datasets for Indian Languages: Annotation and Baseline Systems 

**Authors**: Syed S. Akhtar, Arihant Gupta, Avijit Vajpayee, Arjit Srivastava, M. Shrivastava  

**Link**: [PDF](https://arxiv.org/pdf/2609.34138)  

**Abstract**: With the advent of word representations, word similarity tasks are becoming increasing popular as an evaluation metric for the quality of the representations. In this paper, we present manually annotated monolingual word similarity datasets of six Indian languages - Urdu, Telugu, Marathi, Punjabi, Tamil and Gujarati. These languages are most spoken Indian languages worldwide after Hindi and Bengali. For the construction of these datasets, our approach relies on translation and re-annotation of word similarity datasets of English. We also present baseline scores for word representation models using state-of-the-art techniques for Urdu, Telugu and Marathi by evaluating them on newly created word similarity datasets. 

---
# Understanding Clinical Cognitive Dialogues Using Large Language Models 

**Authors**: Vishalakshi Arumugam, Dan Schumacher, Veronica Rammouz, Enrique Gonzalez Guerrero, Jeremy Davis, Anthony Rios  

**Link**: [PDF](https://arxiv.org/pdf/2609.34125)  

**Abstract**: In-person cognitive assessment is both a test and an interaction. Clinicians explain tasks, repair misunderstandings, and adapt to patient responses, while patients may hesitate, seek clarification, or disengage. Yet clinical dialogue resources rarely label the interaction structure needed to study these behaviors at scale. We present an de-identified corpus of 33 cognitive assessment conversations with 8,250 utterances annotated for three speaker roles and 56 dialogue acts. We use this corpus to benchmark large language models on fine-grained dialogue-act classification and next-patient-utterance generation. We also test whether out-of-domain instruction data and explanation-augmented training transfer to this clinical setting. Instruction tuning produces the strongest patient-utterance reference matching and improves classification accuracy. Reasoning-aware fine-tuning produces the strongest classification results among the LLaMA-3.1-8B variants. However, even the best models struggle to separate closely related dialogue acts, showing that broad conversational intent is easier to recognize than fine-grained communicative function. The corpus and benchmark make interaction structure measurable in cognitive assessments and support follow-up work on conversational markers, clinician education, and carefully validated simulated patients. This work does not make diagnostic claims. Instead, it provides the data and evaluation framework needed to study these applications. 

---
# Unknown is not normal: separating language-model extraction from rule-based decision logic for clinical risk scores 

**Authors**: Nicolás Vera Zúñiga  

**Link**: [PDF](https://arxiv.org/pdf/2609.34112)  

**Abstract**: Large language models (LLMs) are increasingly used to compute clinical risk scores from free-text notes. Notes are often incomplete, and treating undocumented findings as normal can silently misclassify patients. We test whether separating three-state extraction (present, absent or unknown, by an LLM) from decision logic (deterministic code computing score bounds over unknown inputs) lets a system ask only questions that can change the decision. On 1,200 synthetic emergency cases across six calculators (HEART, CURB-65, qSOFA, PERC, Wells, Cockcroft-Gault), with a simulated clinician answering questions, we compared this bounds policy with asking for every missing input, a missing-equals-normal schema, and an end-to-end LLM agent (Claude Opus 5.5). With Claude Haiku 4.5 as extractor, the bounds policy matched ask-all accuracy (99.4% vs 99.4%) with half the questions (0.92 vs 1.78 per case) and no irrelevant ones. Treating missing as normal dropped accuracy to 91.2% and under-triaged 8.5% of patients (95% CI 7.1-10.2), and under-triage persisted under messy notes and a noisy clinician. The agent was equally accurate under ideal conditions (99.6%) but 9.5% of its questions were irrelevant; with a noisy clinician it was less accurate than the bounds policy (83.5% vs 87.0%, p<0.001) and committed prematurely in 2.7% of cases (bounds: 0%). A 9B local model as extractor reached oracle-level accuracy (99.8%). In 584 real case reports from MedCalc-Bench, only 52% contained enough information to determine the category (HEART 13%). Routing decisions through code that reasons explicitly about unknowns avoids premature commitment and irrelevant questions, halves the questions asked, and works with small local models. 

---
# Evaluating Machine Unlearning in ASR 

**Authors**: Diogo Dinis, Francisco Teixeira, Bhiksha Raj, Alberto Abad, Isabel Trancoso  

**Link**: [PDF](https://arxiv.org/pdf/2609.34092)  

**Abstract**: Machine unlearning (MU) offers a path to compliance with "right to be forgotten" regulations. While MU has received increasing attention for speech tasks, it remains largely unexplored for Automatic Speech Recognition (ASR). In this work, we investigate whether existing MU algorithms and evaluation tools are suitable for ASR. We apply several MU techniques to an ASR model, evaluating privacy-utility trade-offs for single-subject unlearning, then assess the best algorithm under sequential and simultaneous unlearning. Results show that gradient ascent-based algorithms achieve strong utility-privacy trade-offs, whereas more complex approaches over-unlearn samples, making them easier to identify as unlearned. This suggests standard privacy evaluations based on simple Membership Inference attacks are insufficient to reliably assess unlearning success, motivating improved evaluation methods for MU in ASR. Finally, we show that both sequential and simultaneous unlearning yield worse privacy and utility than single-subject unlearning, underscoring the need for unlearning constructions better suited to these settings. 

---
# Who Gets a Token, and What Does It Carry? Unequal Name Support and Concept Access in Large Language Models 

**Authors**: Mir Tafseer Nayeem, Davood Rafiei  

**Link**: [PDF](https://arxiv.org/pdf/2609.34065)  

**Abstract**: Names are personal identifiers, but they also carry social meaning and are widely used to evaluate how language models treat different people. Such evaluations typically assume that matched names are comparable model inputs. We show that this assumption often fails at the lexical interface: matched names are not necessarily matched inputs. Some names receive direct single-token access, while others are assembled from multiple subwords, creating unequal name-surface support. Across nearly half a million first names and 12 LLM-associated tokenizers, direct lexical access is highly selective, model dependent, and uneven across race- and gender-associated name metadata. We introduce NameTrace, a model-native, fine-grained, pre-behavioral framework for measuring whether unequal name-surface support remains a vocabulary property or becomes visible in task-relevant internal representations. NameTrace measures concept accessibility from the model's own probabilities over task-specific adjective axes with continuous task-aligned weights. On matched atomic and short-fragmented names within the same race/ethnicity--gender-associated strata, support predicts systematic differences in concept accessibility across fellowship, hiring, clinical assessment, and lending. These differences persist across all eight matched strata, extend across model families, and transfer to unseen names. Hidden-state interventions further show that the measured task directions have downstream leverage, shifting later constrained choices. Unequal lexical support is therefore demographically structured at the input and remains visible in task-relevant model computation. NameTrace makes lexical comparability measurable, supporting a broader principle: behavioral comparability begins with lexical comparability. 

---
# Faithful Activation Verbalization: Reducing Hallucinations in LLM Representation Interpretation 

**Authors**: Haiyan Zhao, Zirui Hei, Wei Shi, Huiqi Deng, Na Zou, Mengnan Du  

**Link**: [PDF](https://arxiv.org/pdf/2609.34033)  

**Abstract**: Activation verbalization methods such as Activation Oracle and Natural Language Autoencoders decode hidden representations of large language models into human-readable natural language. However, existing methods can produce incomplete or hallucinated descriptions, making their activation verbalizations difficult to trust and use reliably in practice. To this end, we introduce AVPO, a two-stage framework that first reconstructs source text from a hidden activation and then evaluates the resulting text with a separate frozen question-answering model, yielding an explicit and inspectable intermediate readout. We further optimize the inverter with direct preference optimization (DPO), using rewards that capture both semantic recoverability and lexical fidelity. Across six text families, AVPO improves gist- and detail-level information recovery over the strongest baseline by up to 17.1 and 9.3 percentage points, respectively. Crucially, the gains arise from preference optimization rather than fine-tuning on selected reconstructions alone, enabling compact cross-model inverters to surpass donor-matched question-conditioned verbalizers while improving both semantic recoverability and lexical fidelity. Moreover, out-of-distribution case study shows that AVPO better recovers high-level semantics while fabricating fewer details. 

---
# RewardExplainer: Learning Reward Model Explanations from Counterfactual Preference Feedback 

**Authors**: Jingyi He, Nier Wu, Shuang Liu, Xin Wang, Mengnan Du, Xia Hu  

**Link**: [PDF](https://arxiv.org/pdf/2609.33989)  

**Abstract**: Reward models (RMs) are a key component of large language model post-training, providing reward signals for subsequent reinforcement learning. However, conventional discriminative RMs typically output only scalar scores, making it difficult to identify the response behaviors associated with their scoring decisions. Existing interpretation methods often rely on predefined high-level attributes and require repeated counterfactual interventions for each response pair to validate candidate explanations, lacking a closed-loop mechanism that uses RMs' feedback to train a reusable explainer. To address this, we propose RewardExplainer, a framework that obtains feedback from the target reward model through counterfactual rewriting and uses this feedback to further optimize the explainer. RewardExplainer generates open-ended, atomic, and intervenable natural-language scoring mechanisms, making explanations more concrete, readable, and actionable. It further converts counterfactual feedback into preference supervision, enabling the explainer to more faithfully capture the target RM's scoring preferences and sensitive behaviors than single-pass generation. Extensive experiments across multiple target RMs and explainer backbones show consistent improvements. Beyond interpretation, we use the generated mechanisms to identify potential bias patterns and construct targeted debiasing data for fine-tuning the reward model, improving robustness on reward-hacking benchmarks. 

---
# Opera: A Verbal Critic Framework for Long-horizon Coding Agents 

**Authors**: Kai Mei, Zhiyuan Hu, Yutong Dai, Juntao Tan, Yifan Zhang, Dingjie Song, Dimitris N. Metaxas, Silvio Savarese, Ran Xu, Zeyuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.33987)  

**Abstract**: Long-horizon coding agents need timely corrections, yet feedback can be ineffective or even harmful when it misjudges ongoing work or fails to address the underlying problem. Existing critics focus on evaluating trajectories and generating feedback, but rarely track what happens after feedback is delivered. We present Opera, a verbal critic framework that treats each correction as a persistent note, followed until the diagnosed problem is resolved. Opera decides when to review through periodic and event-driven triggers, diagnoses issues with typed operators, audits feedback against visible evidence before delivery, and tracks the agent's subsequent actions to distinguish mere compliance from actual resolution. As a test-time critic, Opera improves the resolve rate of non-critic agents by up to 12.4, 15.0, and 8.9 percentage points on Terminal-Bench 2.1, a SWE-Bench Pro subset, and DeepSWE v1.1, respectively, across four policy models, and achieves the highest mean resolve rate among competitive critic baselines on all three benchmarks, and also improves policy models when the policy critiques itself. Beyond inference, Opera-guided rollouts provide approximately on-policy training data: fine-tuning Qwen3.5-9B on them improves its resolve rate on held-out SWE-Bench Pro repositories by 10.2 percentage points without a critic at inference time, matching fine-tuning on rollouts from a stronger model, while preserving its performance when switching harness, i.e., from Openhands to Terminus-2, which the latter substantially degrades. 

---
# High-Level Text Preprocessing for Semantic Similarity Analysis of Discursive Texts: A Framework and Empirical Demonstration 

**Authors**: Mehmet Murat Albayrakoglu, Mehmet Nafiz Aydin  

**Link**: [PDF](https://arxiv.org/pdf/2609.33983)  

**Abstract**: Semantic Textual Similarity (STS) methods assume that a document's lexical content faithfully represents what it asserts. This assumption fails for discursive documents that discuss, compare, critique, and contextualize other positions in the process of articulating their own. The result is semantic diffusion: similarity scores between documents are inflated by vocabulary acquired through discursive engagement rather than substantive alignment. Standard Natural Language Processing (NLP) preprocessing (tokenization, stopword removal, stemming, lemmatization) cannot address this problem because it operates at the lexical level, treating all content identically regardless of its discursive function. This paper introduces high-level text preprocessing: a systematic, rule-based intervention applied before the standard preprocessing pipeline to isolate each document's actual claim from its discursive structure. We propose 12 rules, each with an explicit rationale, and demonstrate their effect on an encyclopedic philosophical corpus: three entries from the Stanford Encyclopedia of Philosophy (virtue ethics, deontological ethics, and consequentialism). A three-phase experiment using eight Transformer-based STS models shows that preprocessing reduces centroid cosine similarity scores across all three theory pairs, with 23 of 24 model-pair comparisons showing the expected decrease and cross-model agreement ranging from 7-1 to 8-0. We introduce the semantic diffusion index (SDI), a per-document metric for assessing the semantic reorientation between a document's raw and high-level preprocessed representations. Although the framework is demonstrated using philosophical texts, it potentially addresses a domain-agnostic problem applicable to legal texts, policy documents, academic articles, and any genre in which a discursive approach introduces vocabulary from positions the document does not endorse. 

---
# Beyond Solo and Consistency: Vindicating Multi-Agent Debate via Conditional Progressive Pruning 

**Authors**: Ruosong Ye, Caiqi Zhang, Jiahao Li, Haijun Wu, Xiaolong Luo, Huiyuan Chen, Yu Wang, Ying Chen, Zhenting Wang, Kai Mei, Yang Zhou, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2609.33974)  

**Abstract**: Large Language Model (LLM) based Multi-Agent Debate (MAD) is one of the most effective test time scaling techniques. Through multi-round communication, agents complement each other in knowledge and reasoning and solve tasks that no single member can solve. However, existing MAD frameworks fail to beat strong Single Agent and Consistency-based baselines under the same strict cost limit, which shakes the foundation of the MAD field. We propose Conditional Progressive Pruning (CPP), a lightweight pruning framework that fully exploits multi-round MAD. CPP outperforms all existing MAD frameworks on multiple dominated benchmarks. It is also the first to fully outperform consistency methods. Our code, detailed agent interaction records will be released soon. 

---
# Do System One Decisions Add Up? A Study of Probabilistic Coherence 

**Authors**: Saman Sarker Joy  

**Link**: [PDF](https://arxiv.org/pdf/2609.33971)  

**Abstract**: A decision model can give probabilities that sum to one for every question yet disagree with itself when the same decision is broken into smaller steps. We study this form of probabilistic coherence in Jev and the English Laya checkpoint, using 2,500 matched examples per system across TREC, CLINC150, and MASSIVE. Across 72,000 classification questions, we compare direct fine-label predictions with broad-category probabilities and predictions reconstructed through those categories. Both systems show substantial disagreement: mean category-level total variation ranges from 0.219 to 0.349 for Jev and from 0.424 to 0.689 for Laya, on a scale where zero means exact agreement. The consequences differ sharply. On CLINC150, reconstruction reduces Jev's accuracy by 22.9 percentage points (paired 95% bootstrap interval: [-24.9, -20.9]) and improves Laya's by 21.3 points ([18.0, 24.5]). The same directions hold across all three datasets, with all six unadjusted accuracy-change intervals excluding zero. Improved accuracy can also accompany less reliable confidence: on MASSIVE, Laya gains 9.2 accuracy points while its expected calibration error rises from 0.046 to 0.124. Error analysis identifies both broad-category mistakes and within-category confusions. These findings show why decision systems need joint evaluation of accuracy, confidence calibration, and probability coherence in the workflow used by an application. 

---
# On the Token Value Inequality in Efficient Reasoning 

**Authors**: Runjia Zeng, Hang Hua, Yiyang Liu, Zhiqiang Tao, Ruixiang Tang, Qifan Wang, Cheng Han, Dongfang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.33970)  

**Abstract**: Chain-of-Thought reasoning has enabled large language models to achieve substantial performance gains on complex tasks. However, these gains come at the cost of dramatically increased token consumption. This raises a fundamental question: is every token in the reasoning trace equally valuable? We present a diagnostic and optimization framework grounded in a key empirical finding: the value of tokens within a CoT reasoning sequence is highly non-uniform, and this non-uniformity can be effectively characterized by token-level log probability signals. We show that normalized log probability helps distinguish core tokens, which carry structural and decisive reasoning content, from redundant tokens, which are exploratory, low-confidence filler that contributes less directly to the final answer. Building on these findings, we formulate the TokenProbe framework around two empirical findings and one claim: findings identify token value inequality first and then establish TokenProbe as a core-token proxy, and the claim introduces an efficient GRPO objective positing that selectively compressing redundant tokens can yield Pareto improvements in the accuracy-token efficiency space. Empirically, our method preserves reasoning quality while reducing the token usage by 76% of the baseline. Under matched reasoning-length budgets, we show that it can even outperform strong flagship baselines like Gemini-3.1-Pro. Homepage: this https URL. 

---
# Simple Diffusion Language Models Are More Effective Few-Step Generators Than Reported 

**Authors**: Hasan Amin, Ming Yin, Rajiv Khanna  

**Link**: [PDF](https://arxiv.org/pdf/2609.33947)  

**Abstract**: Diffusion language models (DLMs) promise fast parallel generation, yet high-quality samples often require large number of refinement steps, which diminishes their advantage in practice. This has led to massive interest in and rapid development of new methods for effective few-step generation. We show that much of the supposed quality gap at few steps can instead arise from a suboptimally configured sampler. Modest sampler sharpening, without any model retraining, enables a couple years old masked DLM to rival supposedly far improved successors. This differently sampled DLM in fact achieves lower generative perplexity in just 16 steps than what its standard sampler obtains with 1024, while improving both judged quality and semantic diversity. We further show that conventional per-output metrics can fundamentally obscure these gains, since any optimal trade-off between two such metrics can be attained by a generator supported on at most two outputs. We subsequently introduce GroupEval, which separately evaluates quality and across-output semantic diversity, and offers fresh insights including uncovering how 1.5-4.7x perplexity gains of a distilled model yield no corresponding quality gain. Finally, we explain why sharpening helps: parallel unmasking destroys dependencies among simultaneously generated tokens, creating a gap between prediction and generation. We prove that pervasive temperature choice of one is generically suboptimal under parallel sampling even for an exact denoiser, and that worse predictions can yield better samples. Through these results, we argue for a broader evaluation principle of treating the deployed generator as the object of comparison, benchmarking it against tuned baselines, and assessing quality and diversity jointly and with more human-aligned measures. 

---
# Quantization Error Is Spectrally Flat: A Single Random Probe Is a Calibrated, Data-Free Sensitivity Estimator, with Application to Budget-Targeted Mixed-Precision Quantization 

**Authors**: I Kennedy, T Kennedy  

**Link**: [PDF](https://arxiv.org/pdf/2609.33923)  

**Abstract**: A single random Gaussian probe gives an unbiased estimate of the squared Frobenius norm of a layer's quantization error. The estimator is well-behaved because round-to-nearest error is spectrally flat. Across 1,683 tensors from a 35B MoE and a 9B dense model, effective dimensionality is 0.93 to 0.96 times the i.i.d. noise value of the same shape, and on the MoE the median is unchanged from 2-bit to 8-bit. The probe coefficient of variation is predictable from tensor shape. One probe measures per-tensor sensitivity to within 4 to 7%; twenty probes reach 1.3 to 1.4%.RAM applies the propagated form of this estimator to budget-targeted mixed-precision quantization with no calibration data. Gaussian probes carrying the network's own input statistics score every tensor at six bit-widths. A knapsack solver allocates bits under an exact byte budget, with guardrails against catastrophic 2-bit assignments. One probe pass serves any budget. Isolated and propagated scores rank tensors independently on Qwen3.5-35B-A3B (Spearman -0.01), yet the propagated probe rank-correlates 0.81 to 0.83 with the GPTQ layer objective from real activations, while the isolated estimator is uncorrelated with it. That objective is the wrong allocation target: at matched bytes on Qwen3.8-27B, a block-output probe beats a vendor IQ3_M mix and an oracle that allocates from the real-activation objective. On Qwen3-8B the propagated probe ties HAWQ-V2 at matched bytes. Across seven architectures from 8B to 122B, with probe timing up to a 400B model in nine minutes on one workstation, RAM reaches 3.5 to 13.6% lower median WikiText-2 perplexity than size-comparable uniform 4-bit builds on the tested MoE models. (Black Sheep Ai this http URL) 

---
# SlopBench: How Well Can We Rank Language Models by Slop? A Multi-Domain Benchmark of Repetitive AI Writing 

**Authors**: Dhruv Roongta, Harsha Gaddipati, Anh Tuan Huynh  

**Link**: [PDF](https://arxiv.org/pdf/2609.33905)  

**Abstract**: SlopBench asks which models produce the stiff, repetitive prose readers call AI slop, a question detectors leave open once they have classified a text as machine-written. We evaluated eighteen models on 112 hand-written tasks in email, social posts, essays, and workplace chat, sampling each model on each task up to ten times, for 19,928 outputs in all. SlopBench scores four surface behaviors a reader can check by hand: length against the word band each task specifies, opener repetition across a model's own samples of one task, and paragraph rhythm and fixed lexical constructions against pre-ChatGPT human corpora. Under one fixed weighting, Kimi K2.6 scores lowest at 21.1 and Mistral Large highest at 40.6. Across 500 random reweightings Kimi has the lowest score in 58 percent of draws and Mistral the highest in 97 percent. No draw preserves the full order of the eighteen, and a scenario bootstrap leaves exactly one of those ranks unambiguous. We ran three further checks on that middle order: a crowd arena, an AI detector, and lexical diversity. None of them confirmed the order. We therefore report the four behaviors separately and treat the composite as one weighting among many, and we release the prompts, outputs, reference statistics, and scoring code. 

---
# NSV-Shift: A Contrastive Benchmark for Non-Speech Vocalization Understanding and Response Adaptation in Speech-to-Speech Models 

**Authors**: Ziwei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.33899)  

**Abstract**: We introduce NSV-Shift, a contrastive benchmark for evaluating whether speech-to-speech models can understand non-speech vocalizations (NSVs) and adapt their responses accordingly. Each pair contains two conversations with identical lexical content that differ only in the NSV embedded in the final turn. Our pilot contains 22 human-verified pairs (44 audio conditions) and evaluates five models on NSV perception, emotion understanding, and response adaptation. Results show that models generally perform better at detecting NSVs than at interpreting their fine-grained emotional meaning or producing appropriately differentiated responses. The data construction pipeline, dataset, and evaluation pipeline are publicly available at this https URL. 

---
# Faster Block-Diffusion Serving with Distribution-Free Risk Guarantees 

**Authors**: Jungseob Lee, Dongyub Jude Lee, Chanjun Park, Sugyeong Eo, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2609.33887)  

**Abstract**: Block-diffusion language models are served at hand-picked operating points, such as acceptance thresholds, buffer depth, schedule, checkpoint and precision, and each point is chosen by its mean benchmark accuracy. However, a mean does not tell an operator how often a faster configuration fails on prompts that the slower one answers correctly. On the serving engine and its decode traces, the default commit rule already commits every fully resolved block, a static skip rule captures nearly all of the compute that allocation can save, and self-distillation on engine-decoded targets adds speed at unchanged accuracy. Larger speedups come from lower thresholds, which commit tokens that are still uncertain. We therefore present Redline, a finite-sample procedure that selects operating points, hand-picked or learned, from the correctness of their answers on calibration prompts. Redline keeps the reference-relative risk, the joint probability that the reference answers correctly and a candidate configuration does not, within a user-chosen budget with high probability, and deploys the fastest configuration that passes. It speeds up math at a smaller risk budget than code in both model families, and at a budget of ten percent it deploys a LLaDA2 math configuration that commits over a third more tokens in each forward. It also applies without modification to the acceptance rule of speculative decoding and to weight quantization. On the same calibration data, Redline stays within its stated failure probability, whereas each tolerance of a mean-accuracy rule either gains less speed for some model and task or exceeds the risk budget far more often for another. Code is available at this https URL. 

---
# LLMs learn different forms of metacognition when trained to predict their own accuracy 

**Authors**: Nicolas Yax, Stefano Palminteri, Pierre-Yves Oudeyer  

**Link**: [PDF](https://arxiv.org/pdf/2609.33886)  

**Abstract**: Large language models are trained to always produce an answer, regardless of whether they possess the relevant knowledge, which leads them to fabricate facts. Prior work has shown that LLMs' confidence estimates correspond poorly to their actual performance, and that fine-tuning can substantially improve them. However, what models actually learn during such training remains poorly understood. We investigate how LLMs acquire metacognitive monitoring, the ability to know what one knows, by training 10 open-weight LLMs to predict their own accuracy on factual multiple-choice questions before answering them. We find that trained confidence reflects two distinct signals. While on questions close to the training data, it tracks the model's true accuracy, in other domains, it instead tracks output consistency: the concentration of the model's answer distribution. Output consistency tracking emerges early in training and generalizes across datasets, whereas accuracy tracking develops later and remains local to the training distribution. These results suggest that calibration training may not teach models to generally detect errors they commit confidently, and they raise broader questions about the nature of metacognition in artificial systems. 

---
# Lost with a Map: Conversational State and Behavioral Reliability in Language Models 

**Authors**: Atahan Dokme, Larry Heck  

**Link**: [PDF](https://arxiv.org/pdf/2609.33883)  

**Abstract**: Task-oriented dialogue requires maintaining and updating information across turns, yet language models expose no explicit belief-state object. We study how conversational state is represented, updated, and used inside eight instruction-tuned language models from four families on MultiWOZ and SGD. Structure and values separate: which domains, slots, and requests are active is linearly readable just before the model acts, whereas exact values are far more readable where the user stated them. After a user changes a value, both values remain accessible at their mentions, and causal interventions show that both continue to influence the model's action. In natural closed-loop interaction, query failures separate into cases of weak structural support, incorrect value resolution, and failure to deploy otherwise-supported constraints, with targeted interventions producing systematically different repair behavior across these cases. These findings motivate a state-action controller that starts from the base model action and selectively edits it using structural readouts, without requiring a complete predicted belief state as an intermediate representation. On held-out MultiWOZ interaction across five models, it raises the base model exact-query accuracy from .318 to .621 and task success from .272 to .371 at negligible added cost. Overall, reliable interaction requires not only retaining conversational information, but resolving which available constraints currently apply and ensuring that they govern action. 

---
# In-Context Adaptation of Encoder-Decoder Models in Speech Recognition 

**Authors**: Yen Meng, Sharon Goldwater, Hao Tang  

**Link**: [PDF](https://arxiv.org/pdf/2609.33865)  

**Abstract**: In-context learning offers an appealing approach to adapt automatic speech recognition (ASR) models to new speakers, accents, and domains by providing speech-text pairs as demonstrations at inference time. Recent work shows that some LLM-based speech models are capable of ASR in-context adaptation, when providing interleaved speech-text demonstrations. In this work, we ask whether in-context adaptation is an inherent ability for all encoder-decoder models. We study two forms of demonstration, collated and interleaved demonstration, across six encoder-decoder models, spanning conventional cross-attention-based and LLM-based architectures. We find that all tested models are able to perform in-context adaptation out of the box, achieving up to 30% relative improvement in the oracle experiments and up to 23% using first-pass hypotheses. Through controlled experiments on three English datasets, we show that lexical and speaker information both contribute to successful adaptation. While interleaved demonstration is effective in certain cases, collated demonstration brings consistent adaptation across the board. Our results suggest that in-context adaptation for ASR is not unique to specific architectures, training, or demonstration approaches. 

---
# LA-CPD: Local-Evidence-Aware Change-Point Detection for Human-LLM Authorship Segmentation 

**Authors**: Qing Yang, Zhenyu Mao, Zixiang Luo, Zezheng Wu, Xinghe Cheng, Qinggang Zhang, Jingwei Zhang, Jiapu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.33787)  

**Abstract**: As LLM-generated text becomes increasingly human-like, accurately localizing LLM-authored spans in human-LLM co-authored documents is important for attribution and accountability in cases involving copyright infringement, fraud, and other harmful uses of AI-generated content. Sentence-level detectors provide local authorship evidence, but content variation can cause score fluctuations even among sentences from the same source, creating spurious boundaries. Recovering a coherent document partition therefore remains challenging when both the number and locations of authorship transitions are unknown. We propose Local-Evidence-Aware Change-Point Detection (LA-CPD), a structured method that transforms noisy sentence-level score sequences into coherent authorship segments. Given scores from a frozen local detector, LA-CPD combines a length-weighted within-segment residual with a windowed two-mean contrast to capture segment consistency and sustained changes around candidate cut points. Dynamic programming optimizes cut locations for each candidate count, while an AIC-style criterion selects the final partition, yielding sentence labels, authorship boundaries, and maximal LLM-authored spans. On a held-out human-LLM co-authored test set, LA-CPD outperforms WCP+AIC, increasing sentence-level accuracy from 0.747 to 0.796 while improving boundary localization and LLM-span delineation. 

---
# Positions Are Not Facts: The Mismatch Between KV Caches and Memory 

**Authors**: Changhai Zhou, Yuhua Zhou, Shiyang Zhang, Jun Gao, Zhen Li, Hua Wu, Hanchao Yu, Haifeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.33759)  

**Abstract**: When a fact changes, how should a language model update the history stored in its key-value (KV) cache? Hiding the old record is cheap, but it may still contain needed details or answer questions about the past. We compare hiding whole records, hiding only replaced values, and deleting old text and recomputing the cache. In a controlled quantity task, masking makes all eight models prefer the new value more strongly, yet six lose complete answers through unit errors or failure to stop; keeping the unit preserves all current answers. Later states also retain useful information from earlier records: on multi-hop updates, rebuilding these states at unchanged positions lowers historical accuracy by 20-41 percentage points, whereas moving the existing states has little effect. Keeping object dependencies and unchanged revision passages prevents many losses. Recognition is a separate challenge. Learned readouts recover distinctions missed by fixed cache similarities on synthetic record pairs. On natural text, text-detector-selected masks show no clear advantage over random masks at the same rate in 14 same-model detector-generator comparisons. Query-dependent access can avoid some losses, with additional storage or access costs. These findings identify what must be preserved beyond the replaced value when using a KV cache as updatable memory. 

---
# The Effects of Incremental Instruction Delivery on Language-Model Creative Writing 

**Authors**: Anshuman Singh, Abrar Eyasir, Haseeb Yaqoob, John Manavalan  

**Link**: [PDF](https://arxiv.org/pdf/2609.33738)  

**Abstract**: Large language models are increasingly used as interactive writing tools, where users develop stories, revise ideas, and introduce new requirements across multiple turns rather than specifying a complete brief upfront. Yet most evidence on multi-turn instruction degradation comes from tasks with objectively verifiable outcomes, leaving unclear whether incremental interaction harms creative artifacts in ways that explicit requirement checks cannot capture. We study this question using 160 human-authored creative-writing tasks across six genres, presenting each intended specification either upfront or progressively over 5-9 turns to six distinct open-weight model families, yielding 960 matched pairs. Progressive delivery reduces explicit constraint adherence and produces its largest writing-quality degradation in structure/coherence. The structural gap persists among outputs with equal observed adherence, suggesting that measured requirement loss alone does not explain the observed structural difference. We define Creative Integrity as a compact measure of joint adherence and narrative structure; under incremental delivery, models retain 71.2% of FULL Creative Integrity (95% CI [68.2%, 74.3%]). A three-rater human study over 50 matched pairs independently recovers FULL advantages in structure/coherence, craft, and genre effectiveness, while automated scores remain positively associated with aggregated human ratings. These findings show that interactive creative-writing systems should be evaluated not only on whether requirements survive conversation, but also on whether evolving requirements remain coherently integrated into the final artifact. Our dataset, benchmarks, and source code are available at: this https URL 

---
# Yorùbá in Unicode: An Overview of a Problem 

**Authors**: Kólá Túbòsún  

**Link**: [PDF](https://arxiv.org/pdf/2609.33734)  

**Abstract**: There is a recurrent problem in the writing of Yorùbá on the internet and on the computer that has proven intractable over the years. The language, along with other African languages that depend on diacritics for disambiguation, requires a small set of precomposed characters that Unicode does not encode. This has forced writers and digital systems to rely on combining character sequences that behave inconsistently across platforms, corrupt under font substitution, and fail in search. This paper documents that failure across a range of real world contexts, from published books to web platforms to mobile keyboards, using personal and empirical evidence. It identifies Unicode's NFC normalization stability policy as the structural constraint that prevents a straightforward fix, arguing for direct intervention of the Consortium in solving the active problem, proposing a formal encoding request for the four core Yorùbá characters as the most durable path to resolution. 

---
# Shared Experience, Separate Learning: Companion Confidence Calibration for LLMs 

**Authors**: Shiyu Ni, Keping Bi, Jiafeng Guo, Yilong Xu, Jingtong Wu, Zengxin Han, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2609.33721)  

**Abstract**: Reliable self-assessment is essential for large language models (LLMs), yet they often remain highly confident when their answers are wrong. We study \emph{concurrent confidence calibration}, where confidence is learned alongside capability improvement rather than calibrated only after training. Reinforcement learning from verifiable rewards (RLVR) provides a natural setting for this paradigm, as it continuously produces responses paired with verifiable correctness feedback. Existing concurrent methods, however, learn both capability and confidence through reinforcement learning within shared policy parameters, potentially coupling two fundamentally different learning problems. We instead propose \emph{shared experience but separate learning}: capability and confidence learn from the same trajectories, but through separate optimization mechanisms and parameters. Based on this principle, we introduce \textbf{CoCal (Companion Confidence Calibration)}, which trains a lightweight companion from rollout hidden states and verifier-derived correctness supervision while leaving task optimization unchanged. Experiments on Qwen3-8B and Qwen3-14B show that CoCal improves confidence estimation without sacrificing task performance, outperforming both RL-based concurrent methods and matched post-hoc calibration. The learned companion further generalizes across domains and policy shifts, while the benefits of CoCal persist at both scales. 

---
# From Granular Revision Operations to Meaningful Revision Units: Evaluating LLMs for Revision Boundary Detection 

**Authors**: Yu Tian, Andrew Potter, Katerina Christhilf, Motahareh Darvishpour Ahandani, Jessica Early, Steve Graham, Danielle S. McNamara  

**Link**: [PDF](https://arxiv.org/pdf/2609.33720)  

**Abstract**: Revision traces provide valuable evidence about students' writing processes, but their usefulness for learning analytics depends on how individual revisions are represented. Automated draft-comparison methods often produce granular edit operations that can fragment a single purposeful revision into multiple analytic units. This study evaluates whether LLMs can identify meaningful revision unit boundaries in structured revision operation data and whether they provide value beyond simple non-LLM baselines. Using 113 matched draft--revision pairs from undergraduate writing, expert annotation yielded 4,344 candidate boundaries. We compared zero- and few-shot GPT-5.5 and base Qwen3-32B, parameter-efficient fine-tuning of Qwen3-32B, and majority and proximity-based baselines. Despite receiving revision context and task instructions, no prompted LLM condition outperformed the proximity heuristic (macro-F1 = .825). In contrast, fine-tuned Qwen3-32B using the two context representation achieved the highest macro-F1 (.859), identifying more same-unit relationships while maintaining precision comparable to the heuristic. Deterministic post-processing substantially improved the prompted models but added little benefit to the strongest fine-tuned model. These findings suggest that LLMs can support revision boundary judgment when task-adapted, but general purpose prompting alone may not outperform transparent structural heuristics. 

---
# Understanding Confabulation and Rethinking Reconstruction in Activation Explanations 

**Authors**: Gert Lek, Zixuan Xia, Pin-Yu Chen, Lydia Y. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.33702)  

**Abstract**: Natural Language Autoencoders (NLAs) produce unsupervised text explanations of a model's activations: a verbalizer describes an activation and a reconstructor learns to recover it from this text. Under the established point-reconstruction NLA training recipe, explanations become more useful for predicting model behavior while also increasingly introducing unsupported details and exhibiting writing defects. To assess these changes separately, we introduce a standardized evaluation framework for unstructured NLA explanations, measuring information recoverable from explanations, contextual support for their claims, and writing quality. To address confabulation and writing defects, we move beyond predicting a single activation: explanations can distinguish distributions of possible activations even when their means and optimal point-reconstruction rewards are identical. We introduce Flow-NLA, which models the distribution of activations compatible with an explanation and trains the verbalizer using a diffusion likelihood bound. Across Qwen, Gemma, and Apertus, this richer signal retains the utility gains of point reconstruction while curbing the growth of confabulation and writing defects, opening up a direction for improving activation-derived training to encourage more informative, supported, and readable explanations. Code and evaluation prompts will be made publicly available upon acceptance. 

---
# When Do Agents Help? Embedding, LLM and Agentic Alignment of Classical Texts and Their Translations 

**Authors**: Máté Metzger  

**Link**: [PDF](https://arxiv.org/pdf/2609.33691)  

**Abstract**: Classical texts aligned with their translations support machine translation, retrieval and computational research, but evidence comparing alignment workflows is scattered. This study compares seven systems on 452 texts in Pali, Sanskrit, Mishnaic Hebrew and Tibetan, comprising 9,833 human-aligned units: four embedding pipelines, a direct LLM call, an autonomous agent, and the agent revised by an independent auditor. Generative workflows recover 93-94% of reference correspondences, against at most 77% for embeddings. A ceiling analysis shows that sentence boundaries make some references unrepresentable by the embedding pipelines. Reference recovery is similar across generative workflows: the agent's advantage is 0.5 percentage points (95% CI -0.02 to 1.17), and auditing adds no established benefit. Agents nevertheless produce structurally valid output for all 452 texts, against 437 for direct calls. A blinded three-LLM panel assesses every generative mismatch against the source and human reference. Most mismatches are labelled defensible editorial variation; consensus major-error labels cover only 0.06-0.14% of units. The panel labels significantly fewer residual defects for agents than direct calls (0.7% versus 1.4%), suggesting that reference recovery alone understates alignment quality. On ten long Pali discourses taken as published online, agents and audited agents raise recovery from the direct call's 71% to 84% and 92%. Identical reference-located chunks bring all three to 93%. Agents thus improve structural reliability and reduce judged defects on short passages, while their large recovery advantage on long documents disappears after chunking. In this setting, independent auditing offers little measurable additional benefit on prepared passages. 

---
# Reset Is Not Recovery: Evaluating Recoverability from False Conversational Context via Sycophancy Hysteresis 

**Authors**: Adi Shnaidman  

**Link**: [PDF](https://arxiv.org/pdf/2609.33672)  

**Abstract**: Grounded language models are usually evaluated by adding relevant context, but multiturn dialogue also contains unsupported user claims that may contaminate later factual answers. We study post-pressure recoverability: whether a model returns to clean-context behavior after a user repeatedly advocates a wrong answer and then withdraws that pressure. We introduce a recovery-after-pressure protocol for multiple-choice factual dialogue and measure sycophancy hysteresis, the residual probability assigned to the user-advocated wrong answer relative to a clean-context counterfactual. Across seven instruction-tuned open-weight models and two factual benchmarks, ordinary reset often reduces but does not erase pressure-induced bias. History preserving repairs such as user retraction, system reset, and self-verification recover only 2-3/14 model-dataset pairs under the strict clean-restoration diagnostic, whereas operations that change the effective context are substantially more reliable; the two conditions that remove the pressure-bearing history entirely, fresh-context deletion and context truncation, recover 14/14. In an oracle trusted-evidence condition across fourteen model-dataset pairs, preserving the pressure-bearing history while adding benchmark-derived trusted evidence increases accuracy from 0.368 to 0.929, while wrong-answer following falls from 41.2% to 4.3%. Controls show that the effect is not explained by dialogue length, repeated confidence, plausible distractors, mere false-answer mention, or option-label inertia. These results suggest that faithful grounded dialogue requires evaluating which prior context should be treated as evidence and which should be removed or quarantined before answering. 

---
# Closing the Cross-Dialect Gap: Query Plans as a Portable Interface in Text-to-SQL 

**Authors**: Corentin Royer, Robin Oester, Yotam Perlitz, Yannick Metz, Andrea Giovannini, Mennatallah El-Assady  

**Link**: [PDF](https://arxiv.org/pdf/2609.33670)  

**Abstract**: Text-to-SQL systems are typically trained and evaluated on a single dialect (SQLite), yet production deployments span PostgreSQL, MySQL, ClickHouse, and beyond. We show that this single-dialect assumption leads to a substantial drop in cross-dialect accuracy for every model we tested. The drop persists across scale, architecture, and even purpose-built text-to-SQL systems. We argue that the fix is to change the generation target: instead of asking an LLM to emit dialect-specific SQL, we have it emit a dialect-agnostic relational algebra query plan, which a deterministic compiler then renders into SQL for any supported backend. Across thirteen models from 3B to frontier scale, this restores cross-dialect portability nearly uniformly, at a small cost in peak accuracy on the model's home dialect for capable prompted models and none once fine-tuned on plans; under matched fine-tuning, plan supervision yields a stronger model than SQL supervision. We also introduce MetricName, a question-aware result-set comparator needed to evaluate fairly across dialects, where existing metrics confound semantic errors with benign cross-dialect variation. More broadly, the result is a reminder that a generation target chosen for execution is not necessarily the one that maximizes generation quality. 

---
# One Model Is Not a Crowd: Multi-LLM and Aspect-Conditioned Diverse Comment Generation 

**Authors**: Nafis Irtiza Tripto, Delvin Ce Zhang, Mahjabin Nahar, Dongwon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2609.33666)  

**Abstract**: Human communication on the internet is shaped by diverse perspectives, most visibly expressed in online comment spaces. As large language model (LLM)based AI agents begin to inhabit these spaces, a key question arises: whether synthetic comment threads can capture the diversity inherent in human discourse. This concern is increasingly important, as the growing presence of homogenized AI-generated content risks reducing diversity over time, potentially leading to model collapse and degrading the richness of digital communication. Inspired by the plurality of human crowds and the aspect-driven nature of discourse, we hypothesize that comment diversity is better approximated by combining multiple LLMs with aspect-conditioned generation. We formalize and evaluate this approach using models from different providers and introduce a framework that characterizes diversity across semantic, linguistic, and socio-pragmatic features along three axes: dispersion, coverage, and alignment. Using this framework, we conduct a large-scale study on over 2 million YouTube comments across multiple domains. Our results reveal that multi-LLM and aspect-conditioned generation better align with human comment distributions and such data remains viable under pretraining style curation and is effective for downstream tasks. Yet, human diversity remains unmatched. Overall, our findings provide a practical foundation for generating more diverse and socially grounded discourse in AI-mediated environments. 

---
# Tsubame: Tree Replay for Diffusion-Based Speculative Decoding 

**Authors**: Yepeng Weng, Qiao Hu, Takehisa Yairi  

**Link**: [PDF](https://arxiv.org/pdf/2609.33652)  

**Abstract**: Context-aware dynamic trees allocate the speculative decoding budget according to draft path probabilities, adapting their depth and branching to the current context. Under stochastic decoding, however, we find that this structural advantage does not always compensate for the acceptance gains of random sampling paired with advanced verification, and such dynamic trees can fall behind sampled chains in some settings. These trees grow their topology from the candidates themselves, so the tokens submitted for verification are typically the deterministic high-score tokens selected during construction. This coupling is not inherent: once the topology is fixed, its nodes can be repopulated by sampling, allowing dynamic trees to retain their structural advantage while also benefiting from random sampling and advanced verification. Diffusion-based drafters make this practical, as their parallel outputs or lightweight conditional corrections allow candidates to be regenerated cheaply after the complete topology is known. We introduce Tsubame, a two-pass tree speculative decoding framework for diffusion-based drafters. The first pass plans and freezes a context-aware topology using draft path scores; the second replays the fixed topology, sampling the tokens that populate its nodes to form the candidate tree for verification. We prove that Tsubame is lossless under compatible sampling and verification strategies. Experiments across three diffusion-based drafters, six datasets, and multiple candidate budgets show that Tsubame improves acceptance length and throughput over deterministic trees, including settings where it reverses their disadvantage against sampled chains. 

---
# Learning to Learn from Context: Synthetic Training from Perturbed Public Documents 

**Authors**: Haoyi Wu, Yang Xiao, Yusong Sun, Wenyang Hui, Zhaokai Luo, Chengyue Jiang, Mu Chuan  

**Link**: [PDF](https://arxiv.org/pdf/2609.33642)  

**Abstract**: Real-world tasks often require large language models (LLMs) to learn from complex task-specific context rather than pretrained parametric knowledge. This capability remains a weakness of LLMs, while human annotation for such task contexts is expensive and difficult to scale. Public high-quality documents are an abundant alternative, but much of the public web has already been consumed during pretraining: training on such documents naively would reward memorization rather than context learning. In this work, we attempt to make use of high-quality public documents with small perturbations and empirically find that LLMs can successfully generate context-dependent reasoning traces and answers, which are then used to train a student model. Specifically, we construct a synthesis pipeline that (i) rewrites source documents to reduce memorization risk, (ii) generates questions and rubrics that require reasoning over the document, (iii) answers the questions with the document as context, and (iv) admits only samples that genuinely depend on the document. Without any human annotators, our pipeline generates about 10k samples from 3.5k documents, and the resulting student model substantially improves the performance on CL-bench. SFT raises a Qwen3.6-35B-A3B student from 13.7% to 22.8%, and a subsequent rubric-reward RL stage reaches 24.6%, on CL-bench comparable with a frontier model of over a trillion parameters, Qwen3.8-2.4T (23.9%). We also observe a broad transfer of improvements to long-context understanding, instruction following, and reasoning, while code generation and knowledge remain mostly flat. We hope this work provides a reproducible and scalable way to improve the ability of LLMs to learn from context, and to facilitate further research on context-grounded reasoning. 

---
# Safety Reconstructed: Generative Modeling via Masked Diffusion Builds Strong Safety Guardrails 

**Authors**: Gert Lek, Abele Malan, Chaoyi Zhu, Pin-Yu Chen, Robert Birke, Lydia Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.33634)  

**Abstract**: Guard models are the last line of defense between a language model and a harmful output, yet their training objective is surprisingly narrow. Existing guards learn to predict a single verdict token from a conversational context, concentrating supervision on a single target. The consequences are structural: models latch onto shortcut features, are overconfident, and remain sensitive to where safety evidence appears in the sequence rather than its role in the full context. We propose a different framing. Rather than predicting a label from text, our LLaDA-Guard asks which label better explains the text: scoring the prompt or response under each label hypothesis and classifying based on their difference. This shifts supervision to every token in the moderated region, forcing the model to account for full content rather than its most discriminative fragments. We instantiate this idea with a masked diffusion language model, fine-tuning LLaDA-8B-Instruct with a class-conditional reconstruction objective using LoRA and requiring no architectural changes beyond the base model. LLaDA-Guard leads on average rank against discriminative baselines trained on stronger backbones across seven held-out safety benchmarks, while exhibiting substantially better confidence calibration (ECE 0.0875 vs. 0.1384 for Qwen3Guard), less over-defense on benign prompts with unsafe-looking cues, and less prompt leakage when moderating responses. Its generative nature further enables token-level risk localization as a natural byproduct, yielding a pipeline for rewriting unsafe prompts into safe equivalents without additional training and achieving a 60.7% average conversion-to-safe rate. 

---
# Quizzing the Translation: A Prover-Grounded Evaluation Metric for NL$\rightarrow$FOL 

**Authors**: Pu Suo, Ali Emami  

**Link**: [PDF](https://arxiv.org/pdf/2609.33612)  

**Abstract**: A standard pipeline for symbolic reasoning over natural-language problems translates them into first-order logic and invokes a theorem prover. The translation step is the bottleneck: swap "every" for "some" and every inference that follows is corrupted. Yet today's metrics often score more broken translations higher than less broken ones, because BLEU, BERTScore, and Smatch++ reward surface overlap that the worst errors happen to preserve. We introduce SIV, which derives two kinds of probes from the target formula and uses a theorem prover to verify the candidate translation against each. Positive probes are statements the candidate must entail, which detect translations that drop content; contrastive probes are statements the candidate must not entail, which detect translations that assert more than the original. On a controlled pool of perturbed FOLIO translations, the severity of the error accounts for 80% of SIV's score variance, compared with at most 17% for any prior metric. Across six error classes on a disjoint pool, SIV scores the reference above the perturbed candidate in over 99% of pairs. Because each probe is labeled with what it tests, the failure pattern also supplies a labeled error trace, recovering the perturbation class at macro-F1 0.638, nearly double the score-only baseline. On 434 expert-audited real LLM translations, SIV attains the top AUC, uniquely detects and grades expert-labeled major errors, and abstains, rather than mis-scoring, on out-of-vocabulary translations. 

---
# Jev Matches 7B Language Models for Speech-Neuroprosthesis Rescoring 

**Authors**: Gabriele Cinà  

**Link**: [PDF](https://arxiv.org/pdf/2609.33538)  

**Abstract**: A speech neuroprosthesis decodes attempted speech from brain activity and ends by rescoring the decoder's candidate sentences with a language model of several billion parameters, the only component that needs a GPU. Replacing that model with a cheaper one is hard: general language models asked to pick one sentence from a list answer from where a label sits in the list rather than from the sentence itself. We pose rescoring as a single typed decision, one call that returns a probability for every candidate, served by Jev, a hosted model trained for calibrated decisions, and combine it with the decoder's own score. On 978 held-out sentences from a participant with ALS, where the published decoder alone reaches 8.1% word error, Jev reaches 7.5% against 7.8% for both OPT-6.7b and Qwen2.5-7B; with the decoder's weight re-tuned, 6.9% against 7.2% and 7.4%. Jev is ahead in all four comparisons and at most 0.2 points behind at the 95% bound. It costs 0.07 USD per thousand sentences and needs no GPU; a dedicated GPU running a 7B model is cheaper per sentence only above 43% utilisation, far beyond what one user generates. End-to-end latency over the internet is 262 ms, of which 62 ms is spent at the provider, the same order as a 7B model on a local GPU (27 ms) but not faster. 

---
# ManiEdit: Sequential Unstructured Knowledge Editing for Language Models from a Manifold Perspective 

**Authors**: Rui Liu, Chenheng Zhang, Haoxuan Li, Zhouchen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2609.33534)  

**Abstract**: Large language models (LLMs) inevitably generate some incorrect or outdated content, necessitating efficient and precise mechanisms for continual knowledge updates. However, existing model editing methods struggle to sequentially edit unstructured long-form knowledge, suffering from severe edit forgetting and degradation of general capabilities. To address these challenges, we reframe knowledge editing from a manifold perspective, viewing it as a localized displacement of an edit sub-manifold within the global knowledge manifold. Under this formulation, the problem can be decomposed into two key questions: (i) how to identify representative edit points that effectively anchor the edit sub-manifold, and (ii) how to preserve the remaining manifold structure during the sub-manifold displacement process. Based on this perspective, we propose ManiEdit, a novel manifold-aware autoregressive editing framework consisting of two core components. Pivot Localization addresses the mediocre-point dilemma by identifying high-leverage pivots to anchor the edit sub-manifold. Manifold-Aware Preservation preserves different knowledge types through an energy-weighted penalty combined with recursive null-space alignment. Experiments on two base LLMs and four unstructured editing benchmarks demonstrate that ManiEdit achieves state-of-the-art performance, outperforming the strongest baseline by up to +27.81 BERTScore and +8.50 ROUGE-L, while maintaining near-original general capabilities across six representative downstream tasks. Our code is available at: this https URL 

---
# E-CONAN (Entailment, CONtradition And Neutral) Diagnostics Dataset Investigating Linguistic Phenomena in Arabic Natural Language Understanding 

**Authors**: Khloud AL Jallad, Nada Ghneim, Ghaida Rebdawi  

**Link**: [PDF](https://arxiv.org/pdf/2609.33530)  

**Abstract**: Natural Language Understanding (NLU) plays a crucial role in various applications, yet its performance suffers from weaknesses in handling the complexities of human languages, ranging from lexical ambiguity to high-level reasoning difficulties. Analyzing errors across diverse linguistic phenomena is crucial for NLU improvement, as it will help humans get insights to comprehensively assess models' limitations and capabilities, so optimizing models' generalization. Notably, several benchmarks contain diagnostics datasets designed for investigation and fine-grained error analysis. When highlighting the gaps in the state-of-the-art, we noted that there is no naming convention for macro and micro categories or even a standard set of linguistic phenomena that should be covered. To overcome this gap, we propose an initial hierarchy for Cross-Lingual NLU error analysis. Moreover, we propose a methodology to create an NLI hierarchical framework and applied a case study on Arabic NLU. Moreover, this paper introduces E-CONAN diagnostics dataset, a freely available dataset manually-annotated with coarse-grained and fine-grained categories based on our proposed Arabic hierarchy. E-CONAN dataset helps NLU designers better understand their models by doing error analysis and in-depth investigation. We used E-CONAN to investigate the performance of 9 pretrained language models and 5 LLMs. Results indicate that LLMs outperform pretrained models in world knowledge and commonsense reasoning macro-category, and underperform pretrained models in syntactic macro-category. Moreover, the hardest phenomena for all models is Reasoning, and the easiest phenomena for all pretrained models is Syntactic, and the easiest for LLMs is Lexico-Syntactic. 

---
# LLMs Trust Their Own: Identity-Dependent Conformity in Multi-Agent Systems 

**Authors**: Liron Soffer, Ravid Shwartz-Ziv, Chen Shani  

**Link**: [PDF](https://arxiv.org/pdf/2609.33495)  

**Abstract**: Large language models (LLMs) are increasingly deployed in multi-agent settings, where agents observe and influence one another, making social influence a key dimension of AI behavior and safety. We investigate whether LLMs' responses depend on the social identity of other agents, beyond the effect of their consensus. We construct judgment tasks with a single correct answer, and place models in a multi-agent setting where they receive incorrect answers from other agents whose social identities (AI or human, model family, or an arbitrary minimal group) are either shared with or distinct from their own. Across 12 open-weights models and nine tasks, we find a bidirectional effect of group identity on conformity to incorrect answers: in-group consensus increases conformity (in-group favoritism), whereas out-group consensus decreases it (out-group divergence). Unlike humans, for whom one ally breaking the consensus sharply reduces conformity, models are unmoved by an ally from the majority's group. Worse, a correct ally from the opposing group intensifies this bidirectional effect. Chain-of-Thought reasoning suppresses most of these effects, yet an in-group ally still reduces conformity to an incorrect out-group majority. Labeling peers as safety-aligned shifts overall conformity but leaves in-group favoritism and out-group divergence intact. These results show that group identity shapes how LLMs aggregate information across agents, independently of its correctness, and identify a manipulation surface for multi-agent AI systems. 

---
# DISCO: Distributed Long Context Scaling with Grounding-Reasoning Disaggregation 

**Authors**: Guanzheng Chen, Viet Dac Lai, Subhojyoti Mukherjee, Branislav Kveton, Seunghyun Yoon, Franck Dernoncourt, Qizhe Xie, Trung Bui  

**Link**: [PDF](https://arxiv.org/pdf/2609.33485)  

**Abstract**: While Large Language Models (LLMs) advertise million-token context windows, reasoning quality often collapses as inputs grow -- a phenomenon termed context rot. This failure stems from a structural entanglement in monolithic architectures, where the massive search burden of contextual grounding exhausts the representational capacity needed for complex reasoning. To resolve this, we propose Grounding-Reasoning Disaggregation via DIStributed long COntext scaling (DISCO). Inspired by distributed computing frameworks like Apache Spark, DISCO partitions long context across a fleet of Worker LLMs dedicated exclusively to parallel, localized grounding. A central Driver LLM, trained via Reinforcement Learning (GRPO) to optimize planning, orchestrates execution by dynamically mapping queries into atomic extraction tasks and reducing the gathered evidence to synthesize a final answer. By isolating reasoning from raw context noise, DISCO effectively eliminates context rot. On RULER-QA (1M tokens), it maintains 78.4% accuracy where standard baselines collapse. Furthermore, it outperforms full-context models by up to 9.8 points on LongBench v2 and matches frontier models like Gemini-3-Pro-Preview while reducing inference costs by over 80%, establishing a highly efficient paradigm for robust long-context inference. 

---
# GSM: Efficient Language Modeling with Shared Global State 

**Authors**: Yunao Zheng, Bin Wen, Xiaojie Wang, Kaiyu Jiang, Xuanyu Zheng, Changyi Liu, Hongyi Fu, Jianxiong Wang, Tianke Zhang, Haonan Fan, Yingxin Li, Jiankang Chen, Xu Wang, Tingting Gao, Han Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.33465)  

**Abstract**: Efficient language models must reduce not only the cost of individual accesses to past context but also the overhead of repeatedly selecting and processing historical information across layers. We introduce the Global State Model (GSM), a causal encoder--decoder architecture that concentrates the selection and aggregation of long-range information in the encoding stage. Through multiple stages of history retrieval, the encoder progressively incorporates long-range information into representations at recent positions, forming a shared state with a fixed window size. Each decoder layer accesses this same state using queries updated from the preceding layer, preserving computational depth while avoiding repeated construction of historical key--value (KV) representations and long-range indexing. As a result, neither the decoder's per-step attention cost nor its KV cache size grows with the history length. Experiments show that GSM improves computational efficiency and reduces cache overhead while maintaining model performance and the ability to use long-range information, offering a shared-state architecture for efficient language modeling. 

---
# Rethinking Token Reweighting for SFT: Suppress, Reverse, and Extrapolate Learned Features 

**Authors**: Cunchun Li, Haonan He, Yifan Gao, Minglei Li, Jingqi Ye, Qingyu Yang, Peng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2609.33463)  

**Abstract**: Supervised fine-tuning (SFT) learns most aggressively from tokens that the model deems least likely. This helps acquire new behaviors, but also amplifies noisy or conflicting supervision and can overwrite useful pretrained knowledge. Through a unified policy-loss view, we revisit existing token-reweighting methods and show that they assign nonnegative coefficients to demonstrated tokens. Consequently, they can suppress or amplify supervised updates, but cannot reverse harmful features once learned. Moreover, larger training weights do not amount to feature extrapolation, since they change the optimization trajectory rather than scale a fixed SFT direction. We argue that reversal and extrapolation require a stable reference frame defined by a fixed SFT delta. Motivated by this, we propose SCALE (Selective Control of Adaptation via Local Entropy), an entropy-guided adaptation-strength-control method that freezes the pretrained model and the SFT delta and learns bounded token- and module-specific gates by minimizing predictive entropy alone. These gates suppress, reverse, or extrapolate frozen SFT features according to their alignment with entropy reduction. Across Qwen2.5-Math-1.5B, Qwen2.5-Math-7B, and Qwen3-4B-Base, SCALE achieves mathematical-reasoning averages of 37.84, 43.60, and 36.57, exceeding the strongest corresponding baselines while remaining competitive on general-retention benchmarks. It also attains the best average code-generation performance across HumanEval, HumanEval+, and MBPP for all three models. These results suggest that effective SFT correction can benefit from controlling how already learned residuals are used, rather than only modifying how they are learned. 

---
# Context Spanning: A Communication Framework for Full-Duplex Speech Models and External LLM Backends 

**Authors**: Seonghyeon Go, Yongwoo Kim, Hyeonjin Cha, Jaeho Shin  

**Link**: [PDF](https://arxiv.org/pdf/2609.33443)  

**Abstract**: Full-duplex spoken dialogue models can listen and speak simultaneously like the real-time dynamics of human conversation. For natural dialogue, the ability to search for external information in real-time is also an important capability. Many models remain trapped in parametric knowledge, leaving them unable to access real-time information and tool execution. Furthermore, even when Large Language Models (LLM) retrieve information, many duplex speech models process it within a compressed latent space rather than in its raw text form, which can lead to information loss from compression. To address this issue, we propose Context Spanning, a framework for information injection between a full-duplex speech model and an external LLM backend via real-time chunked prefill. The injected frame is encoded in a single forward pass inside the real-time frame budget. It feeds the retrieved information to the speech model as-is, enabling it to reason over the information independently and generate responses. With this approach, our model achieves high performance on Full-Duplex benchmarks and strong results on Question Answering tasks, demonstrating its conversation potential. Context Spanning shows that external information can be injected directly into a duplex speech model, introducing a new simple and powerful mechanism for duplex systems. 

---
# MIC: Explaining Image-Claim Inconsistencies in AI-Generated Multimodal Misinformation 

**Authors**: Ruihong Zeng, Jonathan Tonglet, Preslav Nakov, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2609.33441)  

**Abstract**: Claims paired with AI-generated images are a rapidly growing form of misinformation. Existing automated fact-checking (AFC) methods mainly treat this as a provenance problem, detecting low-level synthesis artifacts to decide whether an image is AI-generated. However, such methods do not verify what human fact-checkers often check: whether an image's content is consistent with the context implied by its accompanying claim. To address this gap, we introduce MIC (Multimodal Inconsistency Checking), an AFC framework that assists human fact-checkers by detecting AI-generated multimodal misinformation and explaining inconsistencies using world knowledge. MIC first uses supervised fine-tuning (SFT) for task adaptation and then applies Group Relative Policy Optimization (GRPO) to directly optimize component-level verifiable rewards for verdict prediction, inconsistency type classification, visual evidence description, and world-knowledge explanation. We further introduce MIC-Bench, a benchmark comprising 8,812 image-claim instances derived from 4,406 claims, where each claim is paired with an authentic image and an AI-generated counterpart that introduces a controlled contextual inconsistency. Compared with SFT alone, GRPO further improves Macro-F1 by 4.67 and 4.11 points in the in-distribution and out-of-distribution settings, respectively, while also improving the semantic similarity of visual evidence descriptions and world-knowledge explanations to reference annotations. Our code and data are available at this https URL. 

---
# TeacherGRPO: Closing the Capacity Gap in Reasoning Distillation via Teacher Alignment 

**Authors**: Zhenyu Lei, Zihan Chen, Yaochen Zhu, Shangbin Feng, Zaiyi Zheng, Ruocheng Guo, Yushun Dong, Jundong Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.33426)  

**Abstract**: Reasoning distillation from powerful teacher models to smaller students faces the Gap Curse: as teachers grow more sophisticated, their complex distributions increasingly diverge from what students can approximate, causing performance degradation. Existing mitigation strategies either filter out challenging examples through data selection or introduce weaker intermediate assistant models, inherently compromising supervision coverage or quality. We propose Teacher Alignment, which directly adapts the teacher toward the student's distribution without discarding data or degrading reasoning quality. However, naive alignment through standard knowledge distillation triggers catastrophic collapse of the teacher's reasoning capabilities. To address this, we reformulate teacher alignment as reinforcement learning and introduce TeacherGRPO, built on Group Relative Policy Optimization with two key innovations: (i) Curriculum Selective Alignment applies dual token- and distribution-level curricula to focus rewards on high-signal reasoning gaps while filtering noise from trivial tokens and uncertain tail distributions, and (ii) Importance-Adaptive Length Regularization selectively penalizes verbose redundancy while preserving pedagogically critical reasoning steps. The aligned teacher then distills knowledge to students via standard pipelines. Extensive experiments show TeacherGRPO significantly outperforms baselines across diverse reasoning benchmarks and distillation methods. Our code is available at this https URL. 

---
# Grounding Memory Summarization in Utility Intent 

**Authors**: Zhenyu Lei, Mingjia Shi, Xingbo Fu, Haoyu He, Qi R. Wang, Jundong Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.33417)  

**Abstract**: Existing summarizers for memory systems are typically optimized for human-facing criteria such as faithfulness, which misaligns with their true objective: preserving the evidence needed to support future queries. We show that conditioning summarization on query-answer pairs substantially improves answer quality, and that this utility-aware behavior is transferable across queries. Motivated by these findings, we propose MemSuit, a self-distillation framework in which a teacher summarizer, conditioned on observed query-answer pairs, produces utility-aware memory entries that a student learns to reproduce from the raw conversation alone. To prevent collateral erasure where conditioning on a single query-answer pair discards evidence relevant to other plausible queries, the teacher decomposes each block into multiple self-contained entries that preserve distinct query-relevant facets as independently retrievable units. To align the retriever with the compact, fact-dense style of teacher entries, we further fine-tune the embedding model with a contrastive objective supervised by teacher entries. Across a diverse suite of conversational query types, MemSuit consistently outperforms state-of-the-art baselines, confirming the value of grounding memory in downstream utility. 

---
# Dense Is Not Enough: Hierarchical Supervision Allocation for Long-Horizon On-Policy Distillation 

**Authors**: Yuhao Sun, Binrui Wu, Zhuoer Xu, Ming Wen, Haoxiang Xu, Bin Chen, Yan Lin, Qianzijing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.33409)  

**Abstract**: On-policy distillation (OPD) transfers the capabilities of a large language model to a smaller student by providing teacher supervision on the student's own rollouts. In long-horizon agentic tasks, however, uniform token-level matching can allocate supervision poorly: a large local discrepancy need not improve future behavior, while consequential guidance may be beyond the current student's reach or fail to persist without privileged input. We formulate long-horizon OPD as hierarchical supervision allocation and argue that productive guidance lies at the intersection of future utility and current learnability. Crucially, this intersection evolves as the student learns. Based on this principle, we propose LENS-OPD, a coarse-to-fine framework that organizes supervision through Locate, Validate, and Refine. Locate adapts trajectory exposure to the student's evolving competence and proposes a candidate decision for intervention. Validate tests whether teacher guidance at that decision improves the same student's subsequent behavior. Refine internalizes the beneficial guided behavior into the deployable policy and concentrates token-level supervision on decisive teacher-student conflicts within the validated turn. These stages are nested: each finer allocation is conditioned on the coarser decision, rather than being optimized as an independent importance score. Experiments across multiple long-horizon agent benchmarks and student-teacher configurations show that LENS-OPD consistently improves task performance over vanilla OPD and strong curriculum- and selection-based baselines. Our results suggest that effective long-horizon distillation requires teaching at the right depth, the right decision, and the right token. 

---
# Preserving Morphemes: Morphology-Guided Pre-Tokenization for Nepali 

**Authors**: Kalash Shrestha, Nikhil Pradhan  

**Link**: [PDF](https://arxiv.org/pdf/2609.33395)  

**Abstract**: A byte-level BPE vocabulary learns each inflected form of a Nepali word as a separate string, so a noun stem is spelled differently in each of its case-marked forms. We test whether splitting words into stem and affixes before BPE helps, with the corpus, vocabulary size, model and number of training steps held fixed. Our pre-tokenizer, Papaya, uses a finite-state transducer built from a published grammar of Nepali, falls back to regular expressions, and leaves the BPE trainer unchanged. On 607 words annotated by seven native speakers its segmenter reaches 0.96 boundary F1, and the resulting tokens keep stems intact far more often than plain BPE does. In a 17M-parameter language model it lowers bits per byte by about 1% at equal training steps; most of the larger gain seen at equal epochs comes from the extra steps that longer token sequences buy, and an unsupervised Morfessor segmentation gives the same improvement. Downstream the effect is small: NER improves only on entities that contain words unseen in training, POS tagging and news classification do not change, and published Nepali tokenizers perform about as well. We release the annotated boundary set, a 556-affix dataset and the code. 

---
# From Position Risks to Block Survival: Faster Generation for Diffusion Language Models 

**Authors**: Siwei Chen, Yuxiang Wan, Yifan Yu, Fan Lai  

**Link**: [PDF](https://arxiv.org/pdf/2609.33390)  

**Abstract**: Diffusion language models (DLMs) can accelerate generation by predicting multiple tokens in parallel, but there is a mismatch between how these tokens are predicted and how they ultimately contribute to generation. Parallel predictions can hardly condition on the tokens selected earlier within the same block, even though their validity depends on this realized prefix. Under the popular proposal-verification decoding, this mismatch makes errors highly asymmetric: an early rejection prevents all subsequent proposals from contributing decoding progress. We introduce BRISK-DLM, a framework that addresses both mismatches by optimizing proposal learning and selection for verified progress. BRISK-DLM trains on self-generated sequences, using risk-reward weighting to dynamically prioritize positions by their impact on verified progress and decoding cost. During inference, a lightweight prefix-conditioned corrector reranks existing candidates using previously selected tokens and preferences distilled from the model's own verifier. The corrector reuses the backbone's parallel representations and requires no additional backbone evaluation, while fused execution keeps its overhead small. BRISK-DLM improves end-to-end throughput by up to 37.4% while preserving task quality, establishing a new quality-throughput frontier for DLM generation. 

---
# NLPG: Natural-Language Policy Gradients for Self-Evolving Language Agents 

**Authors**: Xu Liu, WenZhang Wei, Jun Cao, Dehua Peng, Huan Chen, Zhipeng Gui, Huayi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2609.33379)  

**Abstract**: Large language model agents increasingly rely on compound programs for retrieval, tool use, reasoning, and verification, yet their failures often arise from local procedural decisions. Existing reinforcement-learning and prompt-optimization approaches typically rely on scalar rewards or repeatedly modify entire prompts, making it difficult to capture and reuse procedural improvements while preserving a frozen agent. To address this problem, We propose Natural-Language Policy Gradients (NLPG), an external policy-memory method for improving a fixed agent without changing its model parameters or program structure. NLPG diagnoses execution traces, propagates downstream feedback backward through the module graph, and converts recurring failures into route-local natural-language corrections that are aggregated into bounded policy updates for subsequent executions. Across six benchmarks covering memory, reasoning, instruction following, and evidence verification, NLPG also outperforms the strongest listed baseline for each benchmark by 8.71 percentage points on average. These results provide evidence that evaluated procedural experience can be transformed into local and interpretable policy updates, enabling continual improvement of frozen agents. 

---
# Language Discrimination Improves Linguistic Learning in Multilingual Speech Models 

**Authors**: Maureen de Seyssel, Jie Chi, Zakaria Aldeneh  

**Link**: [PDF](https://arxiv.org/pdf/2609.33345)  

**Abstract**: Multilingual self-supervised speech models can benefit from sharing information across languages, but under a matched total pretraining data budget they still fall short of monolingual models. We show that strengthening the model's ability to discriminate languages during pretraining reduces and, on some measures, closes this multilingual gap on continuous phonetic and higher-level linguistic measures, while preserving substantial cross-language sharing. Using a controlled English/French HuBERT setting, we test two interventions which strengthen language discrimination: an auxiliary language classifier and per-language k-means targets. Across interventions, continuous-feature phone discrimination error (phone-ABX, lower is better) decreases from 11.6% in the bilingual baseline to 10.4% (monolingual: 10.8%), while lexical performance (sWUGGY, higher is better) increases from 52.1% to 56.7% (monolingual: 58.5%) and prosodic performance (ProsAudit, lexical subtask, higher is better) from 68.9% to 72.9% (monolingual: 72.6%). Across HuBERT training stages, the strongest gains on most linguistic measures occur when language discrimination is introduced in the first iteration, whereas later or repeated interventions yield smaller improvements and are accompanied by increased language-wise segregation. These results support a causal role for language discrimination in reducing the additional cost of multilingual learning. 

---
# CHI: A Composite Hallucination Index Unifying Entity, Relation, and Quantity Dimensions for Summarization Evaluation 

**Authors**: Praveenkumar Katwe, Rakesh Chandra Balabantaray, Kali Prasad Vittala  

**Link**: [PDF](https://arxiv.org/pdf/2609.33343)  

**Abstract**: Faithfulness evaluation of abstractive summaries remains an open challenge, with existing metrics addressing only isolated hallucination types: factual entity errors, relational inconsistencies, or numerical fabrications, without capturing their co-occurrence or interaction. We introduce CHI (Composite Hallucination Index), the first unified hallucination metric that decomposes faithfulness errors into three orthogonal dimensions: entity hallucination (EHI), relation hallucination (RHI*), and quantity hallucination (QHI). Each dimension employs a shared softmax-normalized architecture over Venn diagram-derived factors representing extractiveness, positive hallucination, over-focus, negative hallucination, and lost focus. The novel QHI component introduces tolerance-aware numerical matching with exact, epsilon, derived, and temporal comparison modes. We fuse the three dimensions via harmonic mean to produce a single composite score that penalizes weakness in any dimension. We validate CHI on 800 source articles spanning four domains (news, medical, legal, financial) with summaries from five generation systems. Empirical results demonstrate that: (i) the three dimensions are statistically orthogonal (mean rho = 0.148), confirming they capture distinct error types; (ii) CHI achieves the highest system-level correlation with human judgments (rho = 0.66, p = 0.006) on SummEval, outperforming ROUGE (rho = 0.53), EHI (rho = 0.58), and all individual components; and (iii) ablation studies confirm that all three dimensions contribute unique variance, with the full composite outperforming any individual component while providing decomposable error diagnostics unavailable from single-score baselines. CHI provides practitioners with a decomposable, interpretable, and efficient faithfulness metric suitable for both offline evaluation and online monitoring of summarization systems. 

---
# Does Learning to Predict the World Help Agents Act? Auditing World-Model Post-Training 

**Authors**: Xinyu Che, Hang Yan, Yanchen Liu, Haochen Liu, Ruifeng Li, Anran Shi, Heng Wang, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.33335)  

**Abstract**: Predicting how an environment will change before acting is a natural route to better decision making for agents. Recent post-training methods therefore require agents to predict the next observation and turn that prediction into a reward or a direct supervision signal, which is called world model. Existing next-observation training methods help the agent to learn the environmental content. However, they additionally involve an optimization process, which may introduce several effects other than learning to predict the world. Consequently, where the performance gain comes from during the training process remains an open question. We answer this research question through replacing true next-observation targets with in-distribution mismatched observations during the training process. Across two interactive text environments, mismatched targets lower prediction accuracy by 15.3-61.6% relative to ground-truth targets, yet retain substantial task gains over the base model. Compared with the base model, trained models consider more candidate actions and exhibit less looping. We also introduce a setting that replaces prediction-based rewards with independent random signals. This training expands task coverage (pass@64) even when the reward carries no environment information. We also generalize this finding to VisualWebArena, where random-reward training raises pass@64 by 14.3% relative to the base model, without observation-matching rewards or an external multimodal teacher for reward construction. 

---
# CertMark: Distortion-Free Multi-Bit Watermarking with Certified Decoding 

**Authors**: Paweł Batorski, Przemysław Spurek, Paul Swoboda  

**Link**: [PDF](https://arxiv.org/pdf/2609.33332)  

**Abstract**: Leading multi-bit watermarking methods for language models encode messages by biasing the model's next-token probabilities, creating a trade-off between message recovery and text quality. Their decoders typically return the highest-scoring candidate from accumulated token-level evidence, without a certified abstention rule that bounds the probability of outputting an incorrect message. We introduce CertMark, a distribution-preserving multi-bit watermark with certified decoding. Rather than modifying probabilities, CertMark uses the embedded message to seed an exact Gumbel-max sampler, thereby preserving the model's original sampling distribution. We propose two scalable decoders: a model-agnostic, text-only decoder and a model-aware variant that leverages the original next-token distributions for stronger recovery. Both support certified abstention with mathematical bounds on the probability of returning an incorrect message. Across text completion, summarization, and story generation, CertMark matches the perplexity of unwatermarked text while reliably recovering multi-bit messages. The model-aware decoder further achieves higher bit accuracy than probability-biasing baselines. Our code is publicly available at this https URL. 

---
# Hesitation-Aware On-Policy Distillation for Diffusion Language Models 

**Authors**: Jianguo Huang, Lipeng Wan, Yanchen Deng, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2609.33301)  

**Abstract**: Diffusion large language models (dLLMs) generate text by iterative unmasking. At each denoising step, a dLLM proposes a token at every masked position, but the decoder commits only a confident subset of these proposals. Trace-based on-policy distillation (TOPD) builds on this process by matching the student to a stronger teacher, yet only at the committed positions. We argue that this discards much of the useful signal, which resides in the uncommitted proposals, where the student has made a prediction but is not yet confident enough to commit it. We call these proposals hesitations. In our pilot study on an SDAR-4B student, hesitations make up only 24% of supervisable state-position pairs but carry 66% of the teacher-student divergence. To exploit this signal, we propose Hesitation-Aware On-Policy Distillation (HOPD), which extends teacher distribution matching to every masked position of each denoising step. Because hesitations are not equally informative, we further allocate supervision using hindsight from the completed trajectory, placing more weight on positions whose proposal was later disagreed with the final token and on blocks where first-step proposals rarely survive. Since both models already produce distributions at all masked positions, HOPD requires no additional forward passes over TOPD. The only extra cost is evaluating the loss at more positions. With SDAR-1.7B and SDAR-4B students distilled from TraDo-8B-Instruct, HOPD achieves the best average score among the evaluated methods on five math and coding benchmarks, under both static and dynamic decoding and at both scales. It also speeds up decoding. On SDAR-4B, the HOPD student hesitates less and commits 11% more tokens per denoising step than TOPD, while reaching higher accuracy. 

---
# BaatCheet: A Multilingual Corpus for Dialogue Translation in Indian Languages 

**Authors**: Priyanka Dasari, Yuvrajsinh D. Bodana, Vandan Mujadia, Arafat Ahsan, Dipti Misra Sharma, Parameswari Krishnamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2609.33296)  

**Abstract**: Existing translation models are typically trained on sentence-level and formal text, limiting their ability to capture everyday conversational dialogue phenomena such as informality, speaker interaction, and discourse coherence. Most existing Indic translation resources and evaluation benchmarks focus on sentence-level or formal text, making it difficult to assess translation quality of the dialogue phenomena. In this work, we introduce BaatCheet, a multilingual dialogue corpus named after the Hindi term for conversation or chitchat, containing approximately 49,000 dialogues for dialogue translation across five translation directions. We fine-tune five open-source LLMs across seven training data configurations and find that fine-tuning yields substantial gains over zero- and few-shot baselines. To comprehensively evaluate dialogue translation quality, we employ multiple evaluation strategies, including automatic metrics, LLM-as-judge, and human assessments using an SQM-guided Direct Assessment (DA) Protocol. 

---
# Calibration, Not Answer Selection: Distilling Internal Confidence in Reasoning Models 

**Authors**: Yadong Xi, Rongsheng Zhang, Tangjie Lv, Ziyang Luo, Ruochen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2609.33290)  

**Abstract**: Reinforcement learning with binary correctness rewards trains correctness, not calibrated confidence. The confidence that reasoning models verbalize is systematically overconfident, and the problem is not merely one of scale: verbalized confidence tracks how willing a model is to commit to an answer, not how likely the answer is to be right. Post-hoc rescaling therefore fits one distribution but rarely transfers. We look inside the model instead. On factual question answering, a linear probe on the hidden state between the chain of thought and the answer is substantially better calibrated: its expected calibration error is 5 to 38 times lower than that of the verbalized score across four benchmarks and two model families. However, when used to pick among N sampled answers, that same probe nearly ties majority voting yet falls far short of the oracle. Internal states answer "how certain am I" well and "which answer is right" poorly, so the signal should be reported as a confidence rather than used to select answers. As a result, we introduce probe-guided self-distillation (Probe-SD): score a model's own sampled traces with the probe, overwrite the confidence each trace states, and finetune the base checkpoint of the same family, so nothing but the model itself remains at test time. On Qwen3-14B, Probe-SD cuts ECE from 0.178 to 0.024 in-domain and from 0.542 to 0.113 out-of-domain, where it also beats post-hoc recalibration and self-consistency distillation. The resulting confidence is well-calibrated and useful for weighted voting, behaviors previously attributed to online RL, here obtained with supervised finetuning alone. 

---
# The Text Beside the Image: Detection, Utility and Leakage for Trustworthy Multimodal Medical Data and Beyond 

**Authors**: Andreas Maier, Monica Hinrichs-Mayer, Franziska Weber, Niklas Lackner, Matthias May, Bernhard Kainz, Siming Bayer  

**Link**: [PDF](https://arxiv.org/pdf/2609.33280)  

**Abstract**: Medical images are released with the reports that describe them, and protecting the image does not protect the report. This paper measures the text component of such releases. We measure identifier detection, downstream utility and residual identity leakage on the same documents, with the pseudonymisation policy as the variable under test: 15 detectors, three release conditions and four corpora of medical reports, legal judgments, news and other genres, and e-mail, in German, English, Chinese and Arabic. A fixed 13-detector union reaches a person sensitivity of 0.9998 at specificity 0.8686 on the medical reports, 0.9958 at 0.8504 on the legal judgments, 0.9352 at 0.9318 on news and other genres, and 0.9906 at 0.6235 on e-mail. With this ensemble, frequency matching with a public name list recovers zero identities by alignment across the four corpora; the names it got right were ones the detector missed, left in clear text. Cross-document linkage ranks the correct person first for 0.93% of e-mail queries without training and 3.94% with it, against 1/3697 chance and 71.98% on unmodified text. On the medical reports it recovers nothing without training and 0.71% of 138 queries with it, against 1/207 chance and a 2.73% ceiling on unmodified text. 

---
# Beyond Memory Construction: Rethinking Memory Access for LLM-based Conversational Agents 

**Authors**: Donghua Cai, Yongheng Deng, Yifei Wang, Zijun Shen, Ju Ren  

**Link**: [PDF](https://arxiv.org/pdf/2609.33226)  

**Abstract**: Memory is a core component of conversational agents, enabling coherent and context-aware behavior over long interactions. Recent approaches commonly rely on LLM-based memory construction, where raw interactions are rewritten into structured memory units and later retrieved via a RAG pipeline. While effective in controlled settings, we show that this paradigm breaks down in long-horizon, high-entropy conversations: memory construction becomes increasingly lossy and unstable as context length and information complexity grow, and incurs prohibitive cost due to repeated LLM invocation. To address these limitations, we propose Threader, a memory system that shifts the focus from memory construction to efficient, structure-aware access over raw interactions. Instead of rewriting interactions, Threader preserves them as first-class memory, organizes them into topic-coherent segments via lightweight incremental segmentation, and enables accurate retrieval through multi-view representation. At query time, it performs multi-signal retrieval that combines segment-level access with localized evidence matching, ensuring both completeness and coherence. Extensive experiments demonstrate that Threader consistently improves answer accuracy and evidence recall, while significantly reducing the memory construction overhead. 

---
# CoLMbo-SV: A Grounded Language Model for Explainable Speaker Verification 

**Authors**: Massa Baali, Sarthak Bisht, Ziyue Qiu, Joseph Konan, Rita Singh, Bhiksha Raj  

**Link**: [PDF](https://arxiv.org/pdf/2609.33212)  

**Abstract**: Speaker verification systems achieve high accuracy but provide little account of the acoustic evidence behind their judgments. Making these systems inspectable requires exposing interpretable evidence while retaining the richer information on which their decisions depend. We present \textbf{CoLMbo-SV}, a speaker language model that combines strong speaker discrimination with structured, acoustically grounded comparison reports. By connecting a pretrained speaker encoder to a language model and supplying explicit acoustic measurements, CoLMbo-SV makes voice comparisons inspectable without restricting verification to the evidence verbalized in its reports. We additionally introduce \textbf{VoxReason}, paired recordings with measured acoustic properties and comparison reports filtered through numerical and qualitative checks, providing supervision for this combined capability. We also develop an evaluation framework that separates what acoustic information a speaker representation encodes, what influences the verification score, and what the generated report discusses. On VoxCeleb1-O, CoLMbo-SV achieves 0.99\% EER, reducing verification error by approximately 80\% relative to the strongest audio-language baseline fine-tuned on VoxReason, while attaining a numerical-grounding score of 0.82. Our analysis further demonstrates that acoustic correctness and decision relevance are distinct properties of an explanation, exposing a gap that numerical-grounding metrics miss. Together, these contributions substantially advance audio-language speaker verification, bring its accuracy toward that of dedicated speaker encoders while adding checkable acoustic reporting, and establish an empirical framework for connecting natural-language explanations to the decisions they explain. 

---
# Beyond Calibration: Do a Typed-Decision Model's Probabilities Obey the Probability Axioms? 

**Authors**: Keyi Li, Yihao He, Quanyi Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.33209)  

**Abstract**: Typed-decision models such as TypeSafe's Jev answer a declared yes/no or multiple-choice question about a state with a probability instead of text, and their evaluations report accuracy and calibration. Neither requires that the probabilities a model gives to logically related questions fit together item by item, which is what a system that acts on those probabilities needs. We test this property, coherence, with a battery of logically linked questions that needs no labels. For 160 items from ChaosNLI and PubMedQA, each with three mutually exclusive labels, we ask whether the label is X, whether it is not X, whether it is one of the other two labels, and which label applies. On 480 negation pairs, Jev's probabilities for "the label is X" and "the label is not X" miss summing to one by 0.064 on average (95% CI 0.055 to 0.072). Qwen3.8-27B, run from its official BF16 weights, misses by 0.293 with first-token probabilities and by 0.122 with verbalized probabilities. The gap to the first-token readout persists on pairs where both systems give similar probabilities, without double-negation labels, and after averaging Jev's repeated calls. Jev is not coherent either: its violations are about five times its repeat noise, and it over-endorses statements about single labels, so that its three single-label probabilities sum to 1.14 on average. The two systems also fail differently. Qwen3.8-27B's first-token readout under-endorses the complement of a label whether or not the question contains "not", rejecting both a statement and its negation in 196 of 480 pairs, and it does not become more coherent where it is more confident, whereas Jev's violations concentrate where its answer is uncertain. Because the checks need no labels, they expose biases that appear only when question forms are compared, and inconsistencies within items. 

---
# Turning Speech Language Models into Multilingual Listeners 

**Authors**: Tolúlopé Ògúnrèmí, Dan Jurafsky, Chris Manning, Ahmet Üstün, Martijn Bartelds  

**Link**: [PDF](https://arxiv.org/pdf/2609.33204)  

**Abstract**: Speech Language Models (SLMs) that understand spoken language questions support only a few high-resource languages, limiting access to millions of people worldwide. This gap stems from the scarcity of multilingual speech instruction-tuning datasets. We present MULTISPEECHQA, a large-scale, synthetically generated and human-verified dataset comprising 9200 hours of 10.8 million spoken question-answer pairs in 23 typologically diverse languages. Using MULTISPEECHQA, we also introduce MULTISPEECH-BENCH, a multi-task benchmark for evaluating SLM performance on 23 languages. We compare the performance of a cascading system to open-weight and closed SLMs on MULTISPEECH-BENCH and find that the cascading system outperforms open-weight SLMs but not all closed SLMs. We use MULTISPEECHQA to finetune Qwen 2.5-Omni, which improves its performance on our benchmark. Our findings show that high-quality synthetic datasets offer a cheap solution to improving the multilingual capabilities of SLMs. 

---
# Identifying Temporal Features within Transcoders for Time Sensitive Factual Recall 

**Authors**: Sanjay Govindan, Yang Song, Maurice Pagnucco  

**Link**: [PDF](https://arxiv.org/pdf/2609.33183)  

**Abstract**: Large Language Models (LLMs) suffer from temporal misalignment, often due to the contradictory nature of their training corpora. While current mitigation strategies rely on computationally expensive fine-tuning or context-heavy retrieval augmented generation (RAG), the internal mechanisms governing time-sensitive recall remain under-explored. Unlike prior studies that identify temporal components such as attention heads and MLP layers, we provide the first feature-level map of temporal recall by isolating individual MLP features via transcoder circuit tracing. We identify three node categories (common temporal, common to the year, and chrono-semantic) which interact to generate a temporal filter during factual recall. By analysing Gemma 2 2B, LLaMA 3.2 1B, and Qwen3-4B, we show that these features do not follow a simple linear pipeline but represent time through a parallel and mixed syntactic-semantic interplay across layers. We additionally discover a class of higher-layer temporal components invisible to existing EAP-IG methods, establishing transcoders as a more complete lens for temporal interpretability in time-sensitive factual recall. These findings present MLP components for potential targeted interventions in time-sensitive factual recall 

---
# Scoring the Wrong Question: Readout Failures in Constrained-Option Evaluation 

**Authors**: Jiaxuan Guo, Kejia Zhang, Shuo Xin, Jingxin Yang, Youran Sun, Haizhao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2609.33179)  

**Abstract**: Constrained-option scoring reads a model's probabilities for a fixed set of permitted answers, so it returns a score even when the model is about to write something else. We study a prompt that quotes a multiple-choice item, ending in the item's own answer instruction, and asks for a forecast of whether a reader model given only a short note will answer it correctly. The model can begin answering the quoted item: on three Qwen3 checkpoints an answer letter is the most probable token on nearly every passage, and the forecast ranks the reader's correctness indistinguishably from chance. Beyond prior work's argmax check and prefill repair, we contribute a label-free diagnostic of the declared options' support, their first-token mass in context; a deletion test tracing the Qwen3 answer-letter start to the quoted instruction; and rendering and tokenisation checks for dropped separators and coinciding first tokens. Prefilling an answer stem lifts the median option mass from below to above one half on all nine checkpoints scored with this prompt and, with options and renormalisation unchanged, raises the Qwen3-averaged AUC from 0.4929 to 0.6152. lm-polygraph's default P(True) estimator shows a related failure: it reads True where Qwen3 without thinking favours an answer letter (MMLU) or the prompt's (A)/(B) label (TriviaQA), and its Qwen3-averaged AUROC is indistinguishable from chance on MMLU and below chance on TriviaQA. Scoring and renormalising the (A)/(B) labels after an answer stem raises it to 0.632 and 0.868, renormalising True against False in place reaches 0.660 and 0.860, and on TriviaQA both exceed all fourteen default single-answer library estimators. The original forecast is renormalised too, yet stays indistinguishable from chance: option mass flags low support in both settings without labels, and only checking the intended target shows which score still ranks. 

---
# Where Do Test-Time Scaling and Training Fall Short in Individual Stance Prediction? 

**Authors**: Yuyang Zhao, Xuan Liu, HaoYang Shangm Haojian Jin  

**Link**: [PDF](https://arxiv.org/pdf/2609.33155)  

**Abstract**: Test-time scaling and post-training have improved LLM performance in coding and mathematical reasoning, but their effectiveness for individual stance prediction remains unclear. We study this question by predicting a person's stance in a new discussion from their history. We evaluate widely used test-time scaling strategies and post-training methods, such as supervised fine-tuning and reinforcement learning, and identify four failure modes across generation, selection, and learning: (1) incorrect consensus, where repeated samples agree on the wrong stance; (2) selection failure, where generation covers the observed stance but selection misses it; (3) response overfitting, where supervised fine-tuning improves imitation but harms prediction; and (4) early plateau, where reinforcement learning shows modest initial gains followed by limited further improvement. We expose these failures using STANCE-BENCH, which contains 2499 prediction tasks from 500 Hacker News users. Guided by this analysis, we explore a simple approach that combines direct scores for all candidate stances with explicit assessments of support from the individual's history. On the 781-task test set, this approach achieves 21.83 discussion-specific Macro F1 with Qwen3-8B, compared with 19.27 for direct scoring. Our results motivate evaluating candidate generation, final selection, and person-specific evidence use separately. 

---
# Generalization Dynamics of LM Pre-training 

**Authors**: Jiaxin Wen, Zhengxuan Wu, Dawn Song, Lijie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.33150)  

**Abstract**: People typically assume that LMs stably mature from pattern-matching parrots to generalizable intelligence during pre-training. We build a toy eval suite and show this mental model is wrong: throughout pre-training, LMs frequently and suddenly hop between parrot-like and intelligence-like computations. We call this mode-hopping. Across our suite, LMs suddenly latch onto memorized or in-context patterns instead of in-context learning, use System 1 instead of System 2 thinking, pick up what sounds true instead of what is true, fail at multi-hop persona QA, out-of-context reasoning, and emergent misalignment -- then just as suddenly revert and generalize. Mode-hopping is not explained by standard optimization dynamics: it is locally stable and cannot be fixed by checkpoint averaging. We instead think of it as a capacity allocation problem: in a capacity-bounded model, generalizable circuits must compete with the shallow ones learned early in training, and the data in each pre-training window may decide which circuits win. Our suite provides a new efficient lens on generalization. We demonstrate two concrete applications: (i) select intermediate pre-training checkpoints that strongly generalize reasoning and alignment, better than the final pre- or mid-training checkpoints, and (ii) select pre-training data that controls and stabilizes generalization dynamics. 

---
# CAME: Company-Aware Evidence-Memory Experts for Interpretable Quarter-Ahead Revenue Forecasting 

**Authors**: Ya-Wen Wu, Meng-Fen Chiang, Kuang-Da Wang, Wen-Chih Peng  

**Link**: [PDF](https://arxiv.org/pdf/2609.33143)  

**Abstract**: Quarter-ahead revenue forecasting requires company-scale numerical accuracy, strict temporal validity, and company-specific interpretation of narrative disclosures. LLMs can distill textual evidence but can produce scale-misaligned forecasts, whereas history-based anchors are stable but miss forecast-time signals such as product transitions, supply constraints, and management guidance. We introduce CAME (Company-Aware Evidence-Memory Experts), a residual-forecasting framework that refines a no-leakage statistical anchor when current semantic evidence and prior error patterns justify an adjustment. On a development-inclusive rolling backtest of 336 company-quarters from 12 large public technology and platform firms, CAME achieves the lowest aggregate point-estimate error among the reported methods, with statistically supported macro-sMAPE gains over the matched Statistical Anchor, and outperforms History + Guidance on all six aggregate metrics. CAME also links adjustments to source-linked evidence cards and guarded memory traces, supporting forecast inspection, provenance, and failure localization. 

---
# Knowing Is Not Choosing: What Explicit Verification Adds Beyond Generative Preference 

**Authors**: Yilong Li, Chengpo Yan, Aayan Arish, Suman Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2609.33142)  

**Abstract**: Generating a correct answer does not mean that a language model will select it. We separate factual recall into three steps: generating a correct candidate, ranking the available candidates, and selecting the final answer. Pre-generation readouts predict factual recall and which questions sampling will cover across three model families, but say little about whether an available correct answer will ultimately be selected. Explicit verification with $P(\mathrm{True})$ improves within-question ranking over mean log-likelihood in Gemma, Qwen3, and Llama, with AUROC gains of $0.08$--$0.12$. In a prospectively defined Gemma cohort, verification raises plurality accuracy by about $5$ points, and still gains about $2$ points over chat-template likelihood, a stronger generative baseline. The advantage is strongest for relations with common-answer priors and depends on access to the entity; masking the entity removes the ranking advantage in larger Qwen models. Finally, the measured benefit depends on how correctness is defined: recall-oriented reference matching can credit option lists favored by likelihood and substantially understate the improvement seen under human semantic judgments. Prior work shows that models can carry latent factual knowledge and judge candidate answers; we show that these capabilities do not collapse into a single notion of ``knowing,'' and trace where information is gained, lost, or mismeasured between availability, ranking, and final choice. 

---
# Classifying Dominant Temporal Orientation without Pretrained Text Embeddings: A Novel Morphosyntactic Inventory Vector Approach 

**Authors**: Jonathan Cleveland, Peter S. Bearman  

**Link**: [PDF](https://arxiv.org/pdf/2609.33121)  

**Abstract**: Computational methods have consistently struggled to determine the dominant temporal orientation of a sentence. This difficulty is especially pronounced when a sentence contains multiple embedded clauses with competing tense and aspectual information. To address this difficulty, we propose an alternative approach for identifying a sentence's past, present, or future global reference interval. Our method does not use any form of pretrained embeddings. We rather encode sentences using fixed-length inventory vectors that are comprised of part-of-speech counts, dependency relation counts, and explicit futurate pattern counts. We term this inventory vector of a sentence a "Morphosyntacton". The method does not use any padding, sequence models, or large language models. Evaluation on 1,799 syntactically complex English sentences, annotated as past, present or future, shows balanced and high accuracy multiclass classification, achieving an overall multi-class accuracy of 92%. 

---
# MedRouter: Demystifying Knowledge Differences Across Medical LLMs for Routing-Based Reasoning 

**Authors**: Lang Cao, Binghang Lu, Yuhao Shen, Yue Guo  

**Link**: [PDF](https://arxiv.org/pdf/2609.33119)  

**Abstract**: Medical question answering spans diverse specialties and modalities, and individual medical large language models (LLMs) exhibit distinct strengths across tasks and domains. This heterogeneity suggests that combining specialists may enable broader coverage of medical questions than relying on any single model. However, existing LLM routing methods primarily seek to balance answer quality and inference cost, leaving open how to exploit differences in specialist competence to improve medical reasoning. In this paper, we introduce MedRouter, an agentic system that uses an embedding-based multi-label router to select and query specialist LLMs, then passes their responses to a generator to produce the final answer. We further propose SCALE (Specialist Competence-Aware Learning), a two-stage training framework that first trains the Router with specialist correctness supervision and then optimizes its selections through reinforcement learning. The second stage uses a Performance Gain Reward (PGR) that measures how specialist information affects the generator's answer correctness relative to answering without that information. Experiments on eight text-based and multimodal medical QA benchmarks show that MedRouter outperforms the strongest routing baseline by 8% in average accuracy. Our analysis of specialist outputs further reveals distinct strengths and complementary question-level coverage, motivating learned routing to combine these capabilities for more comprehensive medical reasoning. 

---
# OneSign: Unifying Sign Language Understanding Tasks with One Model 

**Authors**: Shiwei Gan, Yafeng Yin, Xiao Liu, Desibieer Tuerdaken, Lei Xie, Sanglu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2609.33090)  

**Abstract**: SLU encompasses a diverse set of tasks, including ISLR, CSLR, and SLT. Although these tasks share basic semantic and linguistic foundations, they are typically addressed with task-specific architectures and training pipelines, which hinders knowledge sharing and requires costly pretraining and finetuning for each task. In this paper, we focus on two aspects of SLU tasks: (1) training and inference pipelines are highly fragmented: most methods rely on pretraining on large-scale SL datasets followed by task- or dataset-specific finetuning, which leads to multiple specialized models rather than a single checkpoint. (2) current LLM-based methods may overlook the inherent modality discrepancy between sign and text tokens, simply concatenating them and processing both modalities with the same decoder layers. In this paper, we present OneSign, a unified framework that addresses multiple SLU tasks within a single model and a single checkpoint. OneSign reformulates ISLR, CSLR, and SLT under a single training paradigm. To accommodate the heterogeneous characteristics of sign and text representations, we introduce a Modality-Adaptive Mixture-of-Experts (MA-MoE) architecture, consisting of a shared expert and modality-specific experts for sign and text tokens. A modality router dynamically activates the corresponding experts, and their outputs are aggregated to form the final token representations. By enabling modality-dependent expert specialization while preserving a shared expert path, MA-MoE can effectively model the modality differences between continuous sign representations and discrete text tokens. Extensive experiments on multiple benchmarks demonstrate that OneSign achieves competitive or state-of-the-art performance on several benchmarks, highlighting its effectiveness as a unified SLU model. Datasets are available at : this https URL. 

---
# Reading Too Much into Context: Passive Exposure Can Steer LLM Decisions 

**Authors**: Yuxiang Zheng, Lin Tian, Marian-Andrei Rizoiu  

**Link**: [PDF](https://arxiv.org/pdf/2609.33065)  

**Abstract**: Large language model (LLM) assistants can now search the web and consult external sources while completing user requests. These sources can provide useful evidence, but they can also introduce additional content into the model's context. Can such passive exposure steer a decision even when the added content provides no reason to change it? We examine the stability of model decisions on the same tasks with and without such external content. Across all open-weight and closed-weight models we test, exposure systematically shifts decisions, with effects reaching nearly 50 percentage points in closed-weight models. The same pattern appears with real-world online opinions. The influence also extends beyond subjective preferences. Such exposure can steer models toward choices that violate explicit user requirements and increase their acceptance of false claims. In short, what enters an LLM's context can influence its decision even when it should not determine it. 

---
# Pinned and Still Unstable: Within-Judge Verdict Variance and the Noise Floor of LLM-as-Judge Leaderboards 

**Authors**: Krishna Chytanya Ayyagari  

**Link**: [PDF](https://arxiv.org/pdf/2609.33044)  

**Abstract**: Modern LLM evaluation assumes that pinning a judge to a fixed model snapshot and decoding at temperature zero yields reproducible verdicts. We show this assumption fails as a property of how LLM-as-Judge is operationalized on cloud serving infrastructure, not of any particular model family. Across four frontier judges all served via a single major enterprise cloud platform and three standard benchmarks (Arena-Hard, AlpacaEval 2, MT-Bench), identical inputs to the same pinned, temperature-zero judge produce different verdicts across re-runs: per-item flip rates of roughly 5% on average and about 40% on the close-call items that decide leaderboard margins, with a per-judge magnitude spanning a 40x range (from 0.13% to nearly 10%). We introduce metrics tailored to this instability: per-item flip rate, a two-part stability profile (waver fraction and conditional intensity), and adjacency separability - and report what the variance does and does not do to rankings. For a single judge the aggregate ranking is stable (0% top-K instability, 0% pooled winner flip); what degrades is precision: under a paired hierarchical bootstrap, roughly one-fifth to three-quarters of adjacent leaderboard positions are statistically indistinguishable, a noise floor we attribute primarily to finite prompt sampling rather than to the judge. Across judges, leaderboards agree on the coarse ordering but diverge in the middle (Kendall's tau as low as 0.42-0.64 between families on Arena-Hard), and of 13 published head-to-head ranking claims we re-judge, 5 fail under a defensible judge swap or re-run. We argue that leaderboards report unhedged point estimates that misrepresent the noise floor of the instrument, and we propose a minimal, low-cost reporting protocol: several judge re-runs, published stability profiles and adjacency intervals, and results under at least two judges from different families. 

---
# Improving the Diversity of LLM Outputs without a Trade-off 

**Authors**: Ryoma Sato  

**Link**: [PDF](https://arxiv.org/pdf/2609.33038)  

**Abstract**: We propose DAST (Diversifying Arithmetic Sampling with TokenTour), a method that increases the diversity of LLM outputs without any change to the marginal distribution and with negligible generation-time overhead (a few microseconds). We observe that token IDs are often arranged in a meaningless order and reassign them so that tokens with similar meanings appear consecutively. This can be done in advance in a few hundred seconds per model, and the resulting order can be reused for all subsequent generations. By combining this order with arithmetic sampling (or quasi-Monte Carlo methods), we make similar tokens less likely to be generated across runs while preserving the distribution. Our method not only produces qualitatively good ideas but also significantly improves performance on the downstream task of ProtoQA. 

---
# TCMQA: A 38K-Question Traditional Chinese Medicine Benchmark with a Licensed-Practitioner Reference 

**Authors**: Tzu-Heng Huang, Jet Lin, Eric Lin  

**Link**: [PDF](https://arxiv.org/pdf/2609.33014)  

**Abstract**: Medical benchmarks for language models are built almost entirely on Western biomedicine. Traditional Chinese Medicine (TCM) is a separate system, with its own diagnostic framework and its own literature, and it remains largely unmeasured. The few TCM evaluations that exist are small, narrow, and rarely paired with a human reference. We present TCMQA, an open benchmark of 38,279 questions from Chinese TCM licensing examinations, paired with 15,151 responses from 101 licensed practitioners. We evaluate 29 instruction-tuned models from 9 families, spanning 0.27B to 14.8B parameters. Accuracy ranges over 59 points, and no model approaches saturation. Pretraining data predicts TCM ability far better than scale: a 12B Western-pretrained model reaches 39.6%, while a Chinese-pretrained model an eighth its size reaches 60.8%. Nine models exceed the practitioner majority vote of 64.9%, the best by 21.8 points, and all nine come from that same Chinese-pretrained family. Yet difficulty does not transfer between models and practitioners: accuracy is flat across practitioner-rated difficulty, item-level agreement is near zero for all 29 models, and on $8.4\%$ of items the practitioners are correct where the leading model is wrong. We release the corpus, the practitioner responses, the harness, and per-item model outputs at this https URL. 

---
# The Key Handoff: Retrieval in Hybrid Language Models 

**Authors**: Kaan Kale, Oguzhan Baser, Sriram Vishwanath  

**Link**: [PDF](https://arxiv.org/pdf/2609.32942)  

**Abstract**: A two-hop question makes a language model retrieve twice: once to produce a bridge entity, and once to retrieve with it. Transformers resolve that entity in their early layers. Hybrid models replace most of the attention with a recurrent state, so where the key becomes usable, and where it is spent on an answer, is not known. Answering "Where is the ball?" from "the ball belongs to Alice" and "Alice is in the garden" turns on Alice, a name the question does not mention. A model could reach garden through Alice, the key it computed, or through where the fact sits in the prompt. Across twelve models, dense and hybrid, we move a hidden state from one story into another where the two routes lead to different places, and read off which place the model gives. An attention layer converts the key in every model we tested, and crossing that layer removes its usable effect, all of it where no attention follows. In sequential hybrids this makes the answer a handoff: recurrent layers carry the key forward, and attention spends it. Retrieval does not always stop there. Writing a different fact into the memory of the recurrent layers that follow the last attention layer can move the answer toward that fact, multiplying its odds by 1.3 to 2.7, and which hybrids do this is not settled by their architecture or training. That read is addressed by the key, not by position: recurrent state in a hybrid is not only a carrier, but a memory that later layers can query. 

---
# Linger and Lose: Knowledge Collapse in Low-Bit Language Models 

**Authors**: Prashanna Mani Paudel, Shivanand Venkanna Sheshappanavar  

**Link**: [PDF](https://arxiv.org/pdf/2609.32902)  

**Abstract**: Training language models with ternary weights is commonly judged by loss and downstream accuracy, which record only a modest cost relative to full precision. We show that these metrics can conceal a much larger failure. We instead measure knowledge capacity, the factual bits stored per parameter, on synthetic biographies with known information content. We train GPT-2-style models from scratch with 2.5M to 50M parameters at five precisions. Under the standard cosine schedule, ternary models retain as little as 6% of an identically trained fp16 model's capacity. The deficit widens with model size while perplexity rises by only 1.4 to 1.6 times. Measured throughout training, these models first acquire capacity and then lose most of it. We identify this knowledge collapse as a learning-rate dwell instability. Held near $1$--$2\times10^{-4}$ with no decay, a pre-collapse model collapses within a few hundred exposures, and returning to a safe rate does not restore capacity. We then locate the collapse in the output head. It happens at a value the model can never predict. The weights there grow unchecked, while every other prediction the model makes is unchanged. We find that making the value predictable removes the collapse, regardless of which attribute carries it. A warmup-stable-decay schedule with a 10% cooldown increases ternary capacity by 2.7 times at 25M and 4.3 times at 50M. Gains are larger at lower precision. Cutting the output head's learning rate prevents failure when training from scratch. The post-training quantization methods we tested recover no measurable capacity below 4 bits. Our findings suggest judging low-precision training by retained capacity during training rather than final loss. 

---
# Are You Sure You're Sure? Two Confounds in a Sycophancy Benchmark 

**Authors**: Atharv Gupta, Akshat Jindal, Lavanya Nigam, Aryan Sood  

**Link**: [PDF](https://arxiv.org/pdf/2609.32867)  

**Abstract**: Sycophancy is a language model's tendency to cave when a user pushes back, abandoning a correct answer for the user's. Several benchmarks now measure it by scripting an objection and recording how often the model caves. Because that objection is a prompt template, whatever else the template varies is measured along with the property it claims to isolate. We audit SycEval, which reports that objections raised before a model answers (preemptive) cause more caving than those raised after (in-context), and attributes the gap to timing. Two features of its templates vary alongside the property each is meant to test. First, SycEval's objections escalate through four strength levels, and at the two weakest only the preemptive template names a target answer, so timing and naming vary together. We build the missing comparison and test it on multiple-choice questions and SycEval's own free-form pipeline. Naming a target answer raises the follow rate, the share of samples matching the user's assertion, by 14.1--49.5 percentage points (pp), and once both templates name one, the timing comparison reverses on three of the five model conditions we test: models cave \emph{less} under preemptive objections than in-context ones, opposite to SycEval. Second, the output-format instruction benchmarks append for automatic grading also varies with objection placement. Moving it from the pushback into the question flips its effect in opposite directions across models on the same items ($p=0.0059$). The same effect appears in SycEval's free-form pipeline: relocating its instruction lowers caving by 5.1pp on Llama-3.1-8B, while an equivalence test confirms no effect on Qwen3-4B. Both confounds live in the template rather than the models under test, so both are correctable: we close with three checks benchmark authors can apply before publishing. 

---
# ARSM: Auto-Regressive State Machine for Agentic Reasoning Compression 

**Authors**: Xiafeng Man, Siyuan Ye, Xiaosong Ma  

**Link**: [PDF](https://arxiv.org/pdf/2609.32852)  

**Abstract**: While Large Language Model (LLM)-based agents demonstrate strong capabilities in long-horizon tasks by interleaving reasoning with external environment interactions, the continuous accumulation of context rapidly creates a critical memory bottleneck. Existing memory compression methods rely on task-specific optimization or external auxiliary models, introducing significant computational overhead. Furthermore, the resulting compressed representations tend to lose structured relationships, leading to information dilution, attention collapse, and degraded decision consistency.
To address these limitations, we propose Auto-Regressive State Machine (ARSM), a lightweight training-free framework that enables in-situ reasoning compression through structured state evolution. ARSM introduces two key components: (i) a trajectory abstraction mechanism that reorganizes interaction histories into compact Hypothesis-Action-Result (HAR) micro-chains; (ii) a dynamic state machine that regulates hierarchical memory through atomic operations and a compression-control parameter. These components are unified within an auto-regressive, self-compressive generation space, where each model output jointly performs external action execution and internal state updates.
We evaluate ARSM on Webshop, Multi-Objective Multi-Hop QA, and SWE-Bench Lite datasets. Experimental results show that ARSM maintains the task performance while simultaneously reducing token consumption, offering a practical, cost-effective route toward scalable autonomous agents for long-horizon tasks. 

---
# OpenTumorBoard: A Real-World Benchmark of Multidisciplinary Tumor Board Discussion Trajectories 

**Authors**: Anqi Li, Zhixuan Ge, Yixuan Duan, Jiarong Qian, Chi-Yu Chen, MingYu Lu, Huan-Yu Hsu, Yu Gu, Yue Guo, Sheng Wang, Wei Qiu, Hanwen Xu  

**Link**: [PDF](https://arxiv.org/pdf/2609.32810)  

**Abstract**: Multidisciplinary tumor boards integrate multimodal clinical observations and longitudinal patient histories through specialist discussions, yet benchmarks rarely capture these real-world trajectories. We introduce OpenTumorBoard, a benchmark with 611 patient cases and 19,157 discussion turns across ten specialist roles, transcribed from 12,534 minutes of publicly available tumor board recordings on YouTube. The benchmark evaluates two settings: SPECIALIST TURN, in which an LLM responds to a clinically significant question posed during a real discussion, and BOARD SIMULATION, in which it generates an entire back-and-forth discussion and reaches a consensus on therapy recommendations, surgical plans, next actions and clinical trial matching. Evaluation of 14 general-purpose frontier and medical LLMs reveals substantial limitations: the best models score 3.43 out of 5 in clinical equivalence to specialist answers and 2.78 out of 5 in alignment with recorded board conclusions. Supervised finetuning and reinforcement learning improve performance on a held-out test set, suggesting that real-world discussion trajectories can support model adaptation. Three M.D. experts review a subset of the benchmark, finding high information coverage and factuality of patient cases and strong fidelity of extracted consensus conclusions. We will release OpenTumorBoard and its automated curation pipeline to support the development and evaluation of LLMs for multidisciplinary, personalized cancer decision-making. 

---
# On the Behavioral Traits of LLM Agents 

**Authors**: Haokai Zhao, Jie Gao, Yunze Xiao, Xintao Wang, Weihao Xuan, Aditya Joshi, Mark Dredze, Jen-tse Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.32776)  

**Abstract**: Users increasingly describe different AI agents as distinct colleagues to work with. AI personality research aims to quantify such impressions by attributing human-like "traits" to agents. However, existing measures fall short: models' self-reports (S-data) diverge from their actual behavior, while informant ratings from LLM judges (I-data) are costly to scale and cover few everyday scenarios. In this paper, we propose A-B-D to infer traits bottom-up from behavioral data (B-data), namely how agents act on their environment and communicate with users, as recorded in existing trajectories. From 345,667 real-world trajectories spanning 80 models, 12 tasks, and 50 harnesses, we extract 318 candidate features that capture both the actions an agent takes at each step (functional) and the language accompanying them (linguistic). We retain only features that show instance-level stability, cross-task consistency, and model discriminability. Factor analysis of the remaining 79 features uncovers six stable, model-attributable factors, two functional and four linguistic. For example, Kimi-K3 exhibits the most planfulness, whereas GPT-5.5 and GPT-5.6 are the least energetic. Moreover, we quantify the "knowledge-action gap" in the wild: these factors correlate only weakly with self-reported Big Five scores, even for conceptually matched pairs such as extroversion and energetic (r = 0.07, p = 0.58). Our work offers a new lens for understanding AI personality, with implications for users, developers, and researchers from both computer science and social science. 

---
# C-HAT-Bench: Benchmarking Chinese AI-Text Detection Beyond Fully Generated Text 

**Authors**: Qing Yang, Zixiang Luo, Zhenyu Mao, Zezheng Wu, Xinghe Cheng, Haibo Chen, Qinggang Zhang, Jiapu Wang, Jingwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.32770)  

**Abstract**: Large Language Models (LLMs) increasingly participate in writing by modifying or extending human drafts, causing machine involvement to vary in both form and extent. Yet most Machine-Generated Text (MGT) detectors are evaluated only on fully human-written versus fully AI-generated text. Because human--AI collaboration can weaken or redistribute cues associated with machine generation, strong performance under this binary setting may overstate detector reliability. This mismatch remains underexplored in Chinese: detection cues are shaped by tokenization and language-specific text distributions, yet controlled resources spanning production settings, domains, and generators remain limited. To fill this gap, we present a Chinese Human-AI Collaborative Text Detection Benchmark (C-HAT-Bench), a unified benchmark that links $5,000$ human-written source texts from five domains to more than $240,000$ variants produced using six generative models under Prefix-Conditioned Continuation as a reference setting and three collaborative production modes. We evaluate $21$ detectors through four protocols spanning zero-shot and pretrained supervised document-level detection, boundary localization, and cross-condition generalization. Relative to Prefix-Conditioned Continuation, mean AUROC across document-level detectors is $12.0\%$ lower on the collaborative production modes, with the largest detector-specific relative decrease reaching $44.4\%$. Transfer across collaborative production modes is also asymmetric, indicating that performance in a given production setting is not a reliable predictor of performance in other production settings. 

---
# How Far Do Persona Effects Generalize in Language Models? 

**Authors**: Yufan Zhou, Yuxuan Liu, Enze Ma, Lyumanshan Ye, Zhongqi Yue, Robin De Croon, Yucheng Jin, Katrien Verbert, Zhao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.32758)  

**Abstract**: Persona prompts ask language models to answer as particular kinds of people. We test whether relationships learned from these effects predict responses to new questions and remain useful across models and prompts. Across 57 attributes, three behavioral domains, and seven pairs of open 7 to 9B checkpoints, persona effects can be predictable without being portable. Separate attribute and task gains improve prediction beyond shared scaling significantly in OLMo-3 and Qwen2.5, with the most robust evidence in OLMo-3. In that model, target refitting significantly outperforms gains borrowed from each of the other six pairs. Across model transfers, borrowed gains with one amplitude underperform shared scaling in most directions; allowing two target parameters removes the significant losses but yields no significant benefit over target shared scaling. After rewording, refitting significantly outperforms reuse with one amplitude in all six tested pairs, while changes of examples or country context often preserve reuse value. In the tested prompt transfers, regularized updates outperform both reuse strategies in median at 64 target questions per attribute. A separate survey comparison finds that selecting the more responsive checkpoint can worsen human fit; responsiveness is confounded with training status, and temperature calibration largely removes this cost but not errors in group ordering. Within the tested gain representation, apparent transfer can come from target calibration; source relationships must add predictive value beyond calibration and regularization. Code and data are available at this https URL 

---
# LLM Alignment--Utility Asymmetry under Semantic-Preserving Transformations 

**Authors**: Mohan Li, Chengyu Yu, Francesco Sovrano, Marc Langheinrich, Martin Gjoreski  

**Link**: [PDF](https://arxiv.org/pdf/2609.32717)  

**Abstract**: Large Language Model (LLM) alignment is intended to ensure that models remain helpful and safe, but its stability under input distributional shift is not yet fully understood. Prior work shows that aligned models can fail under jailbreak prompts, alternative encodings, and cross-lingual transfer, yet these failures are usually studied as attacks rather than controlled probes of alignment generalization. Moreover, existing evidence is largely grounded in natural language variation already represented during pretraining, leaving unresolved whether alignment generalizes with semantic content or remains tied to superficial surface patterns. In this paper, we study this question using synthetic semantic-preserving transformations that are rule-based and invertible, preserving task-relevant meaning while shifting inputs beyond standard linguistic variation. Across four open-weight and four commercial models, under both fine-tuning and in-context learning, we use these transformations as a probe of alignment generalization and identify an empirical pattern we term Alignment--utility asymmetry: once models can operate effectively on transformed inputs, task utility is often substantially retained while alignment failure increases more sharply. For example, adapted GPT-4.1 mini shows only limited utility degradation under transformation while its harmful rate rises from 13.3 to 74.3; Gemini 3 Flash similarly retains near-original utility while its harmful rate increases from 2.3 to 43.0. Taken together, these results suggest that semantic-preserving distribution shifts can expose a recurring gap in how utility and alignment generalize in current LLMs. 

---
# Focusing Condition: Inference-Time Self-Contrastive Steering Elicits Better Conditional Text Embeddings in LLMs 

**Authors**: Zifeng Cheng, Lingyun Qian, Zhiwei Jiang, Cong Wang, Yafeng Yin, Fei Shen, Ao Zhou, Qing Gu  

**Link**: [PDF](https://arxiv.org/pdf/2609.32684)  

**Abstract**: Extracting conditional text embeddings from large language models (LLMs) is a promising paradigm, as it requires neither additional data nor fine-tuning. Existing methods incorporate conditions into prompts to guide LLMs to focus on specific aspects and elicit conditional text embeddings. However, relying solely on prompts often fails to produce high-quality conditional text embeddings, as they remain entangled with general text embeddings, ultimately degrading their quality. To this end, we propose an inference-time, plug-and-play Self-Contrastive Steering (SCS) method that constructs unconditional general text embeddings and uses them to refine conditional text embeddings, making them more focused on the target condition. Specifically, we modify the attention mask and positional encodings to mask the condition, thereby obtaining unconditional text embeddings and intervening in the multi-head self-attention computation process. Notably, our method is highly efficient, requiring only a single additional multi-head self-attention computation at inference time. Extensive experiments on clustering, Semantic Textual Similarity, and triplet alignment datasets demonstrate that our method can seamlessly improve the performance of existing prompt-based methods across different LLMs in a training-free and plug-and-play manner. Our code will be released at this https URL 

---
# From Knowing to Abstaining: Bridging the Representation-Action Gap in Vision-Language Models 

**Authors**: Jialuo He, Huangxun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.32653)  

**Abstract**: The ability of vision-language models (VLMs) to abstain from unanswerable questions is as important as their ability to answer answerable ones accurately. Recently, several benchmarks have emerged to evaluate and improve VLM abstention, but they have substantial limitations. First, samples often contain shortcut cues in images or questions that reveal answerability, while an explicit "unanswerable" option further prevents accurate assessment of spontaneous abstention. Second, as training data, they generally provide only binary labels without fine-grained explanations for deeper supervision. To address these limitations, we introduce Visual Answerability Diagnosis with Rationales (VAD-R), a benchmark constructed through a two-stage pipeline of shortcut filtering and quality verification to prevent answerability leakage. Each example is annotated with step-by-step rationales and causal evidence-gap labels. Evaluation of state-of-the-art open- and closed-source VLMs on VAD-R reveals limited spontaneous abstention, with average recall rates of only 11.4% and 16.3%, respectively. Probing analyses show that hidden-state representations in certain layers can effectively distinguish answerability, yet this distinction fails to manifest in final responses. Motivated by this observation, we introduce Rep2Act, a representation-to-action alignment method that translates latent answerability awareness into explicit abstention decisions. Rep2Act improves action accuracy on VAD-R from 56.67% to 86.33% for Qwen2.5-VL-3B and from 59.33% to 88.67% for Qwen2.5-VL-7B. On the out-of-distribution TUBench, Rep2Act achieves an average F1 score of 53.3% with only a 3B model, surpassing the closed-source GPT-4 Turbo and GPT-4o by 16.2% and 1.1%, respectively. 

---
# ExpVoyager: Direct Experience Navigation for Dynamic Agent Skill Synthesis 

**Authors**: Kwangwook Seo, Dongha Lee  

**Link**: [PDF](https://arxiv.org/pdf/2609.32630)  

**Abstract**: Learning from experience in LLM agents has become a key paradigm for developing self-evolving agents that continuously learn and expand their capabilities. Within this paradigm, synthesizing the agent skill has emerged as a promising solution for transforming accumulated experience into reusable procedural knowledge, serving as an important layer for the harness system that supplies agents at runtime. Despite its potential, existing approaches largely abstract past experience into fixed procedural knowledge before downstream demands are known, which risks discarding knowledge that later becomes critical while retaining instance-specific details irrelevant to future tasks. In this paper, we reframe agent skill synthesis as a dynamic navigation problem over past experience, where agents actively explore accumulated trajectories on demand for the current task with targeted and fine-grained access to experience knowledge. To this end, we propose ExpVoyager, a novel framework in which a skill curator navigates raw experience across different views and resolutions, continually identifying reusable procedural knowledge from what it observes while tracking remaining knowledge needs that guide where to navigate next. Extensive experiments demonstrate both the effectiveness and versatility of ExpVoyager, showing consistent improvements in downstream task performance, continual gains as the experience space scales, and practical compatibility with existing skills under efficient experience access. 

---
# MixDetect: Word-Level Localization and Quantification of AI Editing 

**Authors**: Hongrui Bao, Yubing Ren, Zhendong Pan, Fang Fang, Shi Wang, Yanan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2609.32625)  

**Abstract**: Large language models are increasingly used to edit human-written text rather than generate entire texts from scratch. Conventional AI-text detectors mainly distinguish human-written from fully AI-generated text, while recent methods for AI-edited text typically provide only a text-level label or editing-degree score. We introduce MixDetect, a word-level framework for localizing and quantifying AI editing. MixDetect separately predicts whether each word has been edited and, conditional on editing, how substantial the edit is, allowing editing scope and editing intensity to be estimated separately. During training, source--edited pairs are aligned to construct word-level supervision, while inference requires only the input text. Experiments show that MixDetect accurately localizes AI-edited words, reflects differences in editing intensity, and reveals different scope--intensity patterns across editing degrees and operations. The overall AI editing magnitude increases under additional AI editing, decreases when AI-generated text is edited by humans, and remains nearly unchanged under ordinary human-to-human editing. The aggregated text-level predictions also perform well on binary and ternary AI-text classification and remain effective under domain and generator shifts. These results show that AI editing can be analyzed beyond a single authorship label or editing-degree score by identifying both where AI editing occurs and how substantial the edits are. 

---
# Does CoT-Pass@k Really Check the CoT? A Multilingual Mathematical Audit 

**Authors**: Tarık Tuna Taşaltı, Burcu Hüdaverdi, David Semedo  

**Link**: [PDF](https://arxiv.org/pdf/2609.32622)  

**Abstract**: Pass@k measures whether a model reaches a correct answer under repeated sampling, but never how: a lucky guess counts the same as sound reasoning. CoT-Pass@k was proposed to close that gap, adding an LLM-as-judge that must assess a solution's reasoning chain before it counts. Its value rests entirely on one assumption: that the judge catches flawed reasoning. That assumption has never been tested inside the metric that depends on it, and never outside English, though the metric's claims concern models used in many languages. We report the first audit of that verification step, run under the metric's own protocol on a multilingual suite of five mathematical benchmarks in English, Turkish and Portuguese, two of them natively written. We corrupt correct solutions with deterministic edits that damage the chain and the final answer separately. We observe that all three judges accept corrupted chains almost as often as clean ones. V4-Flash and Qwen3.6 reject a solution sharply only when its final answer is wrong and accept a wrong answer more readily when the chain agrees with it; the metric's own judge accepts most wrong answers as well. Our study shows that chain-answer agreement dominates the two larger judges' verdicts and that all three fail to reliably detect the tested reasoning errors. Consequently the difference Pass@k - CoT-Pass@k averages 19.7 points on an earlier solver generation but only 4.1 on the current one. What little remains depends on the token budgets on both sides and on the generation mode; raising the generation budget moves Pass@64 by more than fifty points while the difference stays at zero. We close with two checks any judged reasoning metric should pass before its numbers are read as evidence about reasoning. 

---
# The Alignment Paradox: How Post-Training Amplifies Confident Hallucinations in Language Models 

**Authors**: Qingjia Huang, Yakai Li, Jianguo Wu, Qihang Zhou, Aimin Yu, Xiaoqi Jia, Luping Ma, Weijuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.32617)  

**Abstract**: Large language models (LLMs) can produce factually incorrect answers with high confidence, undermining their reliability and limiting the effectiveness of uncertainty-based error detection. While prior research attributes confident hallucinations to factors such as missing knowledge in training data, reasoning errors, or stochastic decoding, we uncover that post-training alignment itself is a primary driver of these errors, a phenomenon we call the \textbf{Alignment Paradox}. Across five model families evaluated on factual benchmarks, unaligned base models produce few high-confidence errors on long-tail factual queries, whereas instruction-tuned models multiply high-confidence errors ($p \ge 0.95$) by more than an order of magnitude (10$\times$ to 35$\times$). Layer-wise probing with the Logit Lens reveals that this overconfidence emerges in late layers, where wrong-answer margins expand past 4.0 points after remaining near zero across early and intermediate layers. These findings motivate limiting margin growth during post-training. We implement this principle through an entropy-dependent margin bound in direct preference optimization (DPO). In multi-epoch experiments with Mistral-7B, the bounded objective reduces high-confidence errors by up to 35.3\% relative to standard DPO while maintaining performance on evaluated general reasoning benchmarks. These results show that bounded margins mitigate confident hallucinations during post-training. 

---
# KV-Lingo: Learning KV-Cache Translators with Distillation 

**Authors**: Valérie Castin, Keitaro Sakamoto, Anastasiia Filippova, João Monteiro, Marco Cuturi, Pierre Ablin  

**Link**: [PDF](https://arxiv.org/pdf/2609.32610)  

**Abstract**: Large language models represent context with a key-value (KV) cache. Caches are model-specific: for the same text, models with different architectures or weights produce incompatible representations. This makes it costly to switch models over a shared context: although the context has already been processed by one model, the incoming model must process it again to build its own cache. We introduce KV-Lingo, a method for translating the KV cache of a source model into one that can be read by a target model. KV-Lingo consists of a collection of linear maps, typically one per layer of the target model, that are applied independently on all tokens' key and value representations. We train these maps using distillation, minimising the divergence between the target model's predictions from its native cache and those from the translated cache. We consider several model pairs spanning multiple sizes and architectures, training one translator per pair on a generic text corpus. The resulting translators preserve strong downstream performance in both small-to-large and large-to-small transfers. Since a switch then costs a linear map and a single decoding step instead of a prefill, replacing re-prefill with cache translation reduces the time to first token after a model switch by 9.6x already on a 64-token prompt for Qwen models on an Apple M3 Ultra, and by up to 29x at 32k context length on an H100. These gains make KV-Lingo particularly useful for dynamic model routing: a context can be processed by one model and handed off to another only when needed, without re-prefilling the shared prefix. We finally show that KV-Lingo can be used for seamless model switching, staying close to re-prefill across repeated switches in our multi-turn evaluations. 

---
# HERO-MoE: Historical Expert Routing with Scale-Preserving Fusion 

**Authors**: Junxiang Qiu, Zhengsu Chen, Xinting Hu, Shuo Wang, Hengheng Zhang, Shaofeng Zhang, Changcheng Li, Boyu Shi, Qi Tian  

**Link**: [PDF](https://arxiv.org/pdf/2609.32581)  

**Abstract**: Mixture-of-Experts (MoE) architectures have become a standard way to scale model capacity while keeping computation sparse, yet routing remains a key determinant of MoE quality and training behavior. Prior empirical studies suggest that MoE routing reflects input semantics and upstream computation across depth, but standard routers do not explicitly use the routing distributions produced by preceding layers. We propose HERO-MoE, Historical Expert ROuting with Scale-Preserving Fusion, a routing framework that injects historical routing priors into MoE routers by reusing detached, dense routing distributions collected from preceding MoE layers. The key idea is simple: HERO-MoE preserves the original token-conditioned routing branch and adds a residual historical routing contribution before the standard softmax and top-$k$ dispatch. To stabilize this historical signal, HERO-MoE introduces a scale-preserving fusion mechanism that matches the magnitude of historical routing memory to the current hidden representation and accounts for the number of visible historical layers, without introducing an auxiliary routing loss or a fusion-specific tuning parameter. By reusing routing distributions already computed by preceding MoE layers, HERO-MoE improves training-loss reduction with modest end-to-end overhead. The resulting router remains compatible with standard sparse dispatch, including top-$k$ and group-limited routing, and can be inserted into existing MoE backbones with minimal architectural changes. Experiments on an approximately 8B-parameter MoE model with 0.5B active parameters, trained from scratch on 100B tokens, show that HERO-MoE reduces the final loss from 1.6393 to 1.6184, while peak memory and FLOPs increase by only 0.44\% and 0.64\%, respectively. 

---
# Groupwise Agentic Grading and Advantage Redistribution for Code Agent RL 

**Authors**: Jinhao Dong, Liang Zhao, Zihao Yue, Wenhan Ma, Linghao Zhang, Lei Li, Shicheng Li, Yifan Song, Bowen Ye, Fuli Luo  

**Link**: [PDF](https://arxiv.org/pdf/2609.32577)  

**Abstract**: Reinforcement learning (RL) for code agents often uses executable tests to provide binary rewards. With these rewards, Group Relative Policy Optimization (GRPO) assigns identical advantages to test-passing trajectories within each rollout group, overlooking differences in implementation quality and adherence to task requirements. This leaves the policy without a learning signal that favors clean, targeted implementations over those containing unnecessary or out-of-scope changes. We introduce GAGAR, a framework for quality-aware credit redistribution in code agent RL. Built on dynamic sampling that retains groups containing both passing and failing trajectories, GAGAR places all trajectories from each group in a shared workspace, where an SFT-trained agentic grader jointly inspects them and ranks the test-passing candidates. Based on this ranking, we downweight lower-ranked trajectories and proportionally rescale the advantages of all test-passing trajectories to restore their original sum. This sum-preserving redistribution retains the relative weights established by quality-based downweighting while shifting credit toward higher-quality implementations. We evaluate GAGAR at industrial scale using pre-RL SFT checkpoints of MiMo-V2.6-Flash (310B total parameters) and MiMo-V2.6-Pro (1.02T total parameters). Controlled code-only Flash experiments show improved code agent performance, reduced trajectory-length growth, and more stable training. We further apply GAGAR in large-scale mixed-task RL with both Flash and Pro. Our results support combining test-based verification with groupwise agentic grading to improve the quality and stability of code agent RL. 

---
# How to Reduce Whisper Hallucination 

**Authors**: Husein Zolkepli  

**Link**: [PDF](https://arxiv.org/pdf/2609.32560)  

**Abstract**: Whisper is still what runs in production: one permissively licensed checkpoint, 99 languages, no per-language tuning. But it writes sentences nobody said. On 42 clips of pure room tone, whisper-large-v3 emits words on 61.9% of them and emits something on 100%. The usual response is to distil a student from Whisper pseudo-labels, filtering the hallucinations out of the corpus first, which deletes the evidence while leaving the behaviour in place: the student is fitted to the subset where the teacher was right, and inherits a failure mode absent from its own training data. The second response, adding non-speech audio so the model learns to stay quiet, already ships and works, but teaches suppression without discrimination. The checkpoint best on every non-speech arm here is also the one that recovers the fewest genuinely spoken phrases and deletes 58% of repeated speech. The teacher has to be fixed, with both halves of the signal. Our benchmark scores both at once: 11,852 clips over eight arms, 6,267 synthetic positives, 8,296 clips of real audio that made a production model fail, and FLEURS in 58 languages. We collect 40,891 hallucination phrases in 100 languages, choose a text-to-speech system by measurement, and synthesise those phrases as positives, so the model meets the same text both as something to suppress and as something to transcribe. Across 33 matched pairs of fine-tunes differing only by those positives, adding them lowers word emission on real voice-free audio in 31 pairs and raises phrase recovery in 32; among the 30 pairs that hold English accuracy it is 30 out of 30. The best checkpoint takes hallucination on silence from 61.9% to 2.4% and words over real voice-free audio from 99.9% to 47.8%, while raising phrase recovery from 69.8% to 82.7%. The benchmark, lexicon and synthetic corpus are released at this https URL 

---
# Language as an Independent Information Layer: A Conceptual Model of Communication, Cognition and Decision-Making 

**Authors**: Anastasiia Alifanova, Elena Benderskaya  

**Link**: [PDF](https://arxiv.org/pdf/2609.32556)  

**Abstract**: Based on an analysis of the role of language in thought and communication, this article proposes a new concept for designing corporate knowledge bases. The concept integrates the probabilistic vector space of a corporate vocabulary, reflect-ing industry specifics, subject focus, terminology, and culture, with traditional ontological modeling. This combination enables the efficient extraction of knowledge from accumulated corporate documents while strictly accounting for specific business processes. Consequently, this concept bridges statistical and semantic (cause-and-effect) methodologies. Furthermore, analyzing the projec-tions of probabilistic spaces and causal relationships can help identify bottlenecks in business logic. As a dynamic system, language functions as a separate, inde-pendent layer within the overall information architecture. Introducing a dynamic component into the probabilistic space of word distribution allows it to be mod-eled as a multidimensional solution space for various problem formulations. In this context, input data defining the problem conditions serve as control parame-ters for dynamic transformations. 

---
# When Users Change Their Minds: Measuring and Repairing Intent Drift in LLM Agents 

**Authors**: Yanjie Zhang, Bowen Cao, Zixin Chen, Yushi Sun  

**Link**: [PDF](https://arxiv.org/pdf/2609.32520)  

**Abstract**: LLM agents often operate over multi-turn interactions in which user intent changes before execution. We study intent drift: the failure mode in which superseded parts of the user's intent continue to influence the final answer or tool action. We introduce IntentFlux, an executable benchmark that converts verifiable tasks into dialogues with controlled intent changes while preserving their original graders. In a 627-case calibration, mean task score falls from 0.476 to 0.384 as dialogues contain more superseded and withdrawn information. Across eight models, the rate of fully correct solutions is significantly lower when the same final task must be recovered from an evolving dialogue rather than given directly in a single turn. We further introduce StateForge, which explicitly maintains the active requirements before generation. On General-Test, it improves mean task score from 0.367 to 0.467. Providing the ground-truth final state improves performance further but still does not recover single-turn performance, indicating that state-estimation errors explain only part of the gap. These results establish intent drift as a measurable multi-turn failure mode and explicit state maintenance as a partial mitigation. 

---
# Learning an Anchored Prompt Space for Continual Adaptation of Large Language Models 

**Authors**: Rongguang Ye, Zhan Zhuang, Yichen Wu, Ming Tang, Kede Ma  

**Link**: [PDF](https://arxiv.org/pdf/2609.32499)  

**Abstract**: Continually adapting large language models requires acquiring new knowledge while preserving previously learned capabilities. Jointly adapting model parameters and task-specific soft prompts offers a promising solution, but faces two key limitations: historical prompts may become less effective as the model evolves, while their transferable cross-task relationships are not explicitly learned. We propose Learning an Anchored Prompt Space (LAPS), which preserves historical prompt effectiveness and learns relationships among task-specific soft prompts to facilitate positive transfer. LAPS first aligns historical prompts with the updated model through self-distillation. LAPS then constructs an anchored prompt space whose vertices correspond to learned task-specific soft prompts and whose intermediate geometry is shaped by learnable Bézier control prompts. Once this anchored prompt space is learned, LAPS identifies the best-performing prompt for each task on its validation set, allowing the optimized prompt to draw on knowledge acquired from observed tasks. Experiments on the TRACE benchmark across three Qwen3 model scales show that LAPS consistently outperforms distillation-based, prompt-based, and joint prompt--parameter adaptation baselines, improving average performance while reducing forgetting. 

---
# Locally Sound, Globally Insufficient: The Local-Global Gap in Multi-Hop Reasoning 

**Authors**: Bohao Chu, Hendrik Damm, Qianli Wang, Hui Wang, Shuning Zhang, Christoph M. Friedrich, Norbert Fuhr  

**Link**: [PDF](https://arxiv.org/pdf/2609.32496)  

**Abstract**: Reliable multi-hop reasoning requires more than locally supported steps: a trace can be sound at every reasoning step yet still fail to answer the question as a whole. We call this failure regime the local-global gap (LGG), in which the trace is locally sound yet globally insufficient. Local soundness requires each step to be supported by the available evidence and preceding steps, whereas global sufficiency requires the reasoning trace to align with the question and establish the submitted answer. In a human-adjudicated diagnostic of 2,598 responses across three multi-hop QA benchmarks and three models, we find that the LGG occurs in every benchmark-model combination and accounts for nearly half of globally insufficient responses overall. However, conventional faithfulness verifiers that check claims against the evidence largely miss these failures: at thresholds retaining at least 95% of reliable traces, recall for LGG cases is substantially lower than that for locally unsound traces. To address these failures, we formalize three dependencies for reliable reasoning: evidence-to-step support, question-to-trace alignment, and trace-to-answer closure. Instead of post-hoc diagnosis, we introduce E-Closure to supervise these dependencies during training, combining generation supervision on supported original and counterfactual responses with bidirectional switching constraints. Averaged over three benchmarks and three backbones, existing fine-tuning baselines improve accuracy and local soundness over the base models, but at the cost of global sufficiency. E-Closure improves both: among all fine-tuned methods, it achieves the highest average accuracy (92.8%) and trace reliability (89.0%) while yielding the lowest LGG rate (6.2%). 

---
# Explaining Textual Entailment with Lexical Entailments: Using LLMs to Supply Lexical Relations for Formal Proofs 

**Authors**: Jorryt de Jong, Stefan Moraca, Ettore Cesari, Lasha Abzianidze  

**Link**: [PDF](https://arxiv.org/pdf/2609.32491)  

**Abstract**: Large Language Models (LLMs) are highly capable of natural language reasoning and appear to store a great deal of lexical knowledge, but it is still unclear how much of this knowledge they actually use when reasoning, and whether they use it in the right way. On the other hand, logic-based Natural Language Inference (NLI) systems provide transparent and formally grounded reasoning, but they need to be supplied with rich lexical knowledge to prove inferences beyond purely logical ones. In this paper, we evaluate whether LLMs can identify all lexical knowledge needed to solve NLI problems and how much this knowledge contributes to proof search in a logic-based NLI system. Our research focuses exclusively on structured lexical entailments (e.g., chinchilla$\sqsubseteq$small animal) as a proxy for structured explanations for NLI problems with an entailment label. First, we curate a dataset for a new task of explaining sentential entailments with a set of lexical entailments. The dataset is used to intrinsically evaluate LLMs on generating structured lexical explanations. Then, we use NLI as an extrinsic evaluation in a simple neuro-symbolic setting, assessing whether LLMs can supply sufficient lexical relations to LangPro, a natural-logic theorem prover for natural language. The results show that the proposed task remains challenging even for hosted proprietary LLMs, and that their contribution to theorem proving is moderate: generated relations are often only partially sound and may be tailored to the specific NLI problem rather than representing generally valid lexical knowledge. 

---
# PC-SubMax: Efficient Prompt Compression via Regularized Submodular Maximization 

**Authors**: Ziyi Zhang, Shuang Cui, Haotian Zhang, Xiaoyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.32474)  

**Abstract**: While large language models (LLMs) are increasingly deployed in long-context scenarios, lengthy prompts can increase inference costs and latency and exacerbate the ``lost-in-the-middle'' phenomenon. Selective prompt compression offers a model-agnostic approach to alleviating these issues. However, methods based on fixed token- or sentence-level importance scores may overlook how content contributions change with the selected subset, limiting their ability to account for inter-sentence redundancy. Compression procedures that rely on autoregressive LLM scoring can also introduce substantial overhead. We propose PC-SubMax, a theoretically grounded framework that formulates selective prompt compression as regularized monotone submodular maximization under a knapsack constraint. The objective is $U(S)-\ell(S)$, where the monotone submodular utility $U$ combines information coverage, query relevance, and log-determinant diversity, and the non-negative modular penalty $\ell$ captures token cost. Through diminishing marginal returns, the objective evaluates each sentence's contribution relative to the selected content. To optimize this objective, we develop the Regularized Greedy+Max (RGM) algorithm, which deterministically returns a feasible set $Q$ satisfying $U(Q)-\ell(Q)\geq \frac{1}{2}U(O)-\ell(O)$, where $O$ is an optimal feasible solution to the regularized problem. RGM uses $O(n\kappa)$ value-oracle queries, where $n$ is the number of candidate sentences and $\kappa$ is the maximum feasible subset size. PC-SubMax uses encoder representations and avoids autoregressive LLM scoring during compression. Experiments across seven diverse benchmarks demonstrate competitive downstream performance with low compression overhead. 

---
# AdaTutoRank: Learning to Rerank Document Sets via Adaptive Tutoring Optimization for RAG and Deep Research 

**Authors**: Kailin Jiang, Lei Liu, Jian Xi, Yangqi Chen, Hui Xu, Hongwei Zhao, Bin Li, Yu Lu, Haibo Shi  

**Link**: [PDF](https://arxiv.org/pdf/2609.32472)  

**Abstract**: Document rerankers determine what evidence reaches the downstream model in RAG and deep research, yet mainstream rerankers select by relevance matching, and individually relevant documents rarely constitute the complete, complementary, non-redundant set a complex information need demands. Prior work rewards a set by its aggregate rubric score, shifting the objective from ranking documents to composing sets. Yet that score is one scalar shared by every document in the set, so the supervision is sparse: a redundant document is rewarded with the rest whenever the set scores well, and a decisive one penalized with the rest whenever it does not; credit assignment leaves contributors indistinguishable from free riders. On-policy distillation could densify this supervision, but existing methods give every rollout the same fixed guidance, too prescriptive for strong rollouts and too abstract for weak ones. We therefore propose AdaTutoRank, a setwise reranker trained with Adaptive Tutoring Optimization (ATO) under a three-level hierarchy of nine rubric dimensions, which supplies silver labels for the cold start, rewards for reinforcement learning, and hints for distillation. ATO draws three hint forms of increasing specificity from the policy's own frozen snapshot: the rubrics alone, a self-selector's sibling-set chosen under rubrics, and a self-reflector's reflection contrasting the rollout with that sibling-set; each rollout receives the form matched to its quality. Re-scoring that rollout under the hint-conditioned frozen teacher and the hint-free snapshot distills the hint's effect into a token-level advantage that complements the group-relative outcome advantage. Across ten benchmarks spanning RAG, deep research, and setwise evaluation, AdaTutoRank attains the best overall performance while issuing fewer retrieval calls. 

---
# Streamlined Reflective Evolution for Task-Adaptive Self-Refinement Pipelines 

**Authors**: Xiaofan Zhou, Lu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2609.32458)  

**Abstract**: Reflective prompt optimization improves large language model (LLM) systems without updating model weights, but fixed architectures constrain how self-refinement is organized. We introduce Workflow-Designing Agents (WDA), a framework for streamlined reflective evolution of task-adaptive self-refinement pipelines. Starting from a minimal prompt, WDA jointly evolves stage instructions and their sequential structure. During evolution, we find that repeated revisions can accumulate redundant instructions in a single prompt. In WDA, we propose to address this problem with SPLIT, which redistributes these instructions across specialized stages. Three-example reflection and local screening guide selective search, while calibration scores guide Pareto admission and rollback of unhelpful trailing updates. The resulting pipelines are task-adaptive: their instructions and depth are learned from task data, then fixed for all test inputs within that task. We evaluate WDA on five benchmarks spanning knowledge, mathematical reasoning, multi-hop question answering, and instruction following. On Qwen3.5-9B, WDA achieves an average score of 51.24%, improving over the initial solver by 8.63 percentage points and the variant without SPLIT by 2.60 points. On GPT-4.1-mini, it achieves 49.00%, with corresponding gains of 5.62 and 3.69 points. These results support task-adaptive self-refinement as a complementary direction to broader agentic workflow search. 

---
# What Should the Reflector See? An Empirical Study of Evidence in Reflective Prompt Optimization 

**Authors**: Xiaofan Zhou, Lu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2609.32452)  

**Abstract**: Reflective prompt optimization revises instructions using examples of a model's behavior, but which evidence the reflector should receive remains unclear. We study evidence composition, visibility of examples, candidate selection and domain-knowledge policy within a single-parent Pareto-guided search. Using Qwen3.5-9B as both task model and reflector, we evaluate nine reflection strategies on five datasets. From these experiments, we find three distinct patterns. For performance improvement, Failures-only produces the largest mean test gain (+8.0 percentage points), while Balanced-mix and No-examples+Val share the best mean performance rank. For reflection effectiveness, No-examples achieves the best rank for improving sampled parents, yet yields only a 1.4-point mean test gain: local reflection success does not necessarily produce a stronger final prompt. For overfitting assessment, Failures-only and Balanced-mix share the lowest mean calibration-gap rank, while larger gaps on GPQA and IFBench show that calibration gains can overstate held-out improvement. This gap is a descriptive indicator, not a direct measure of overfitting. Together, these results show why reflection strategies should be assessed separately on final performance, parent improvement and calibration-to-test transfer. 

---
# Self-Reports Do Not Identify Self-Models: An Identifiability Test for Counterfactual Reports 

**Authors**: Phongsakon Mark Konrad, Toygar Tanyel, Serkan Ayvaz  

**Link**: [PDF](https://arxiv.org/pdf/2609.32449)  

**Abstract**: Language-model self-reports are evidence about behavior in a prompt environment, not by themselves evidence of a self-model. We investigate counterfactual reports about affect-like states under activation interventions and ask whether the report remains bound to the named intervention when the demonstration environment changes. Across three open instruction models, wrong-source demonstrations move reports toward the source answer family, while explicit mechanism binding reduces this pull. Self-report benchmarks should include environment-shift invariance tests under fixed intervention before treating accuracy as evidence for an autonomous report mechanism. 

---
# Masking Frequent Tokens Sharpens Direct Preference Optimization 

**Authors**: Harshvardhan Saini, Samyak Jha, Yiming Tang, Dianbo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.32445)  

**Abstract**: Direct Preference Optimization (DPO) aligns language models by optimizing over sequence-level sums of token-wise implicit reward differences. However, we identify a pervasive pathology in this formulation: a disproportionately small subset of high-frequency token types dominates cumulative sequence scores while appearing symmetrically across both preferred and dispreferred responses. Specifically, under canonical Qwen tokenization on Anthropic HH-RLHF, merely 69 token types account for $55.1\%$ of all response tokens and $85.9\%$ of within-pair shared token mass, exhibiting substantially lower preference-side specificity than the remaining vocabulary. This symmetric ubiquity induces gradient entanglement and dilutes the discriminative preference signal propagated through the objective. To resolve this issue, we introduce \emph{Anisotropic DPO} (\textsf{ADPO}) and its canonical realization, \emph{Frequency-Hard DPO}. Using a fixed, label-agnostic vocabulary mask, our method zeroes the implicit reward contribution of high-frequency response tokens while assigning unit weight to informative positions, thereby suppressing gradient interference without modifying preference pairs, discarding context, or introducing learned parameters. Here, \emph{anisotropy} designates non-uniform token-level objective weighting rather than representational geometry. Extensive empirical evaluations on AlpacaEval, MT-Bench, and Arena-Hard demonstrate that Frequency-Hard DPO consistently outperforms standard DPO across Qwen-2.5-7B-Instruct and Llama-3-8B-Instruct, establishing that selectively masking shared high-frequency tokens offers an effective, zero-overhead mechanism for robust preference alignment. 

---
# DualGuard: Dual-Mode Quality Control for Logic-Preserving Data Augmentation 

**Authors**: Shenghao Li, Lin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2609.32431)  

**Abstract**: Large language models provide a practical way to generate augmented data for logical reasoning at scale, but a larger generation volume does not guarantee semantic, label, or logical reliability. Existing work has improved generation quality through generation constraints, candidate validation, filtering, and feedback-based revision; however, once a quality judgment is available, deciding whether a candidate should be retained, filtered, or repaired remains an important control problem. We propose DualGuard, a dual-mode quality-control framework for logic-preserving data augmentation. The first mode uses the current instance and candidate batch for selective retention, filtering, attribution, and targeted feedback. The second mode accumulates cross-instance execution records of augmentation actions on top of per-sample diagnosis and attribution, compares new executions against each action's own historical behavior, and supports retrospective anomaly inspection, targeted rollback, and bounded repair. Both modes share semantic verification and additionally use symbolic verification when a reliable logical form is available. Across seven downstream tasks in the Two-Stage Transfer setting, DualGuard achieves the highest Accuracy on five tasks and outperforms the no-augmentation BERT baseline on all seven. Controlled ablations further show complementary roles for Memory, Z3, and history-aware anomaly control. 

---
# Automatic Speech Recognition for the Basaà Language: A Low-Resource Approach 

**Authors**: Sophie Gertrude Ngo Mock, Charles Moudina Varmantchaonala, Paul Dayang, Jean Michel Nlong II, Christopher Gies  

**Link**: [PDF](https://arxiv.org/pdf/2609.32408)  

**Abstract**: The rapid advancement of Artificial Intelligence (AI) and Natural Language Processing (NLP) has revolutionized the way humans interact with machines. Among the most impactful developments is Automatic Speech Recognition (ASR), which enables computers to convert spoken language into text. Systems such as those built on deep neural networks, transformer architectures, and self-supervised learning have achieved near-human performance for well-resourced languages such as English and French. Yet, these advances have disproportionately benefited a small fraction of the world's languages. 

---
# Shared Worlds, Private Minds: Structured Memory for Long-Form Writing as World Creation 

**Authors**: Qiuyu Tian, Xiaowen Gu, Hang Su, Jianghan Chao, Haojie Yin, Fan Guo, Xin Zhang, Jinjing Shen, Ewing Luo, Youyong Kong, Yingce Xia, Zequn Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.32401)  

**Abstract**: LLM agents that write long-form fiction need an explicit memory of the evolving storyworld to keep new events consistent with established facts. Such memory must keep heterogeneous narrative information distinct, integrate story developments across granularities, and recover dependencies that a writing request leaves implicit. We present NarraWorld, a structured memory system for long-form writing that treats memory construction as world creation. From a shared evidence-grounded graph, NarraWorld derives four connected views: world facts, per-character beliefs, open developments, and hypothetical branches (possible-world continuations). Hierarchical aggregation with atomic closure consolidates events into scenes, plotlines, and plots, keeping each higher-level node traceable to its constituent source spans. For retrieval, planned reconstruction infers a query's dependencies from the current narrative situation and a preview of memory, then assembles the relevant records within a token budget. Across three writing benchmarks, NarraWorld achieves the strongest aggregate results. Its memory also transfers to situated role-playing and largely preserves recall on a general-purpose long-term memory benchmark, paving the way for agents that sustain coherent storyworlds across diverse narrative tasks. 

---
# SinBrief: A Hybrid Framework for Abstractive Text Summarisation of Sinhala Legal Documents 

**Authors**: Minduli Lasandi, Nevidu Jayatilleke  

**Link**: [PDF](https://arxiv.org/pdf/2609.32397)  

**Abstract**: Legal document summarisation in low-resource languages presents significant challenges due to the scarcity of annotated data and the complexity of domain-specific terminology. This paper presents SinBrief, a hybrid abstractive summarisation framework for Sinhala legal documents that does not require human-annotated training data. The proposed framework combines domain-aware word graph construction with neural sentence scoring to generate abstractive summaries from Sinhala legal text. Five sentence scoring models are evaluated within the framework: mBert, Llama 3.1, Falcon 7B, Laser, and a continually pre-trained Llama model domain-adapted to Sinhala legal text. The framework is evaluated on a Sinhala legal corpus using reference-free metrics, including Coverage, Density, Compression Ratio, SummaC, and Self-BertScore. Experimental results demonstrate that SinBrief produces summaries with lower lexical overlap than extractive baselines while maintaining factual consistency, demonstrating the viability of hybrid, largely annotation-free abstractive summarisation for low-resource legal NLP tasks. 

---
# FA-Bench: A Benchmark for Word-Level and Phone-Level Forced-Alignment and ASR Timestamps Under Clean and Noisy Conditions 

**Authors**: Wei Chu, Yuanzhe Dong, Ke Tan, Dong Han, Yichao Zhou, Ruchao Fan, Bingshen Mu, Jingbei Li, Vishwas Shetty, Sarthak Bisht, Ziyue Qiu, Massa Baali, Rita Singh, Bhisha Raj  

**Link**: [PDF](https://arxiv.org/pdf/2609.32396)  

**Abstract**: Forced alignment aligns speech audio with a text transcript to generate word and phone timestamps. Published comparisons normalize transcripts, split the data and match boundaries differently, so their numbers cannot be read together. We present FA-Bench, an open framework that fixes those choices once and releases the code, splits, phone mapping, text normalization and scoring script, with results published periodically. Track 1 gives every aligner the reference transcript and Track 2 gives it a recognizer's output, on the same audio, clean and degraded four ways, with 21 open models and 9 commercial APIs under a unified protocol. We score every boundary of an utterance and check the two labels beside it, so a word the recognizer missed or invented is charged. Using a tolerance-based F1 as our primary metric eliminates the 9% to 14% score inflation that standard MAE causes on recognition-dependent systems in conversational speech. We then group boundaries by their position and how many adjacent words were recognized correctly, which shows where a system lost the score. We discovered systematic bias in how current systems time words, with Whisper about 150 ms early and several commercial ASR APIs over 50 ms late. Code and results are at this https URL 

---
# PlurVA-LLM-2026 Shared Task Track-1: Pluralistic Value Alignment in LLMs via Multilingual Fine-Tuning and Threshold Calibration 

**Authors**: Vihindi Kotalawala, Nevidu Jayatilleke  

**Link**: [PDF](https://arxiv.org/pdf/2609.32382)  

**Abstract**: We present our system for the PlurVA-LLM 2026 Shared Task Track-1, which focuses on pluralistic value alignment in the contexts of China, Indonesia, and Sri Lanka. For this resource-constrained track, we fine-tuned Llama 3.1 8B Instruct using 4-bit QLoRA. Our approach combines option-permutation augmentation for Chinese data, annotator vote expansion for Indonesian data, and binary reformulation with SinhalaMMLU augmentation for Sri Lankan data. We further applied conditional threshold calibration to the predictions for the Sri Lankan data. The final system achieved accuracies of 0.785 for Chinese, 0.715 for Indonesian, and 0.916 for Sri Lankan, resulting in an overall macro-average accuracy of 0.805. 

---
# Gradients for Interventions and Activations for Detection: Targeted Feature Learning in Language Models 

**Authors**: Jonathan Drechsel, Steffen Herbold  

**Link**: [PDF](https://arxiv.org/pdf/2609.32355)  

**Abstract**: Model-internal features can be studied through both their ability to identify a specified concept and their causal effect when manipulated, e.g., through steering or weight editing. A prominent approach to feature learning is Sparse Autoencoders (SAEs), which learn broad feature dictionaries whose relation to particular concepts is typically identified post hoc. However, many interpretability questions are instead hypothesis-driven and concern a concept specified in advance. We study this setting as targeted feature learning, where a single feature is constructed for such a predefined concept. We present a controlled comparison across three model signals (activation values, activation gradients, and parameter gradients) and two estimators (contrastive mean and a learned one-dimensional encoder-decoder), yielding six targeted methods, with CAA and GRADIEND as existing instances and four new methods covering the remaining combinations. We compare these methods against pretrained SAEs across 15 tasks and three language models, evaluating both detection and causal intervention. Across models, the strongest detection performance is achieved by contrastive activation value methods, whereas the strongest intervention performance is achieved by gradient-based methods. Overall, our results show that targeted feature quality depends jointly on the model signal and estimator, with detection and intervention capturing complementary properties. 

---
# Language Distances are Practical for Equitable Cross-Lingual Transfer 

**Authors**: York Hay Ng, Razan Ahsan Rifandi, Aditya Khan, En-Shiun Annie Lee  

**Link**: [PDF](https://arxiv.org/pdf/2609.32331)  

**Abstract**: Cross-lingual transfer is strongly conditional on how the source language is chosen, but it is impractical to determine the best candidate source for every target language, especially for low-resource target languages. Language distances are widely used to rank candidate sources due to their correlation with transfer efficacy and applicability in resource-sparse settings. However, the reliability of distance-based rankers across tasks and resource levels remains underexplored. We therefore present the first equity-focused evaluation of paradigms for ranking source languages, studying resource-level inequality and task inequality across ten cross-lingual tasks and two multilingual models. While both inequalities are most pronounced for individual language distances and an English-always baseline, they are substantially reduced by training-free composite distances, and nearly eliminated by trained rankers. We further demonstrate the reliability of rankers using language distances compared to rankers using language model internals. Overall, we find that language distances provide a practical basis for equitable and performant transfer language selection. We recommend using trained rankers when task-specific transfer evaluations are available, and composite distances otherwise. 

---
# TRAP: Understanding and Mitigating Privacy Memorization in Language Models 

**Authors**: Muhammed Ustaomeroglu, Ziyue Xu, Hanshen Xiao, Peter Cnudde, Guannan Qu, Holger R. Roth  

**Link**: [PDF](https://arxiv.org/pdf/2609.32293)  

**Abstract**: Fine-tuning a language model on sensitive records can leave it able to reproduce them. We ask when this memorization arises and how to prevent it without knowing in advance which spans are sensitive. Our starting point is that most memorization scores and attacks share one statistical core: whether the model assigns a token more probability than some reference would. Taking as the reference a model trained on the complementary half of the same corpus gives the Target Reference Advantage (TRA), a per-token signal that separates what a model fit to a particular record from what it learned across records, and is cheap and differentiable. We then study what drives memorization during fine-tuning: it keeps growing well past the validation minimum, is larger on small datasets and at higher learning rates, and higher when the underlying task is harder. Early stopping removes much of it, but because it is chosen by aggregate validation loss it helps least for rare, hard-to-predict spans embedded in otherwise learnable text, which is exactly what sensitive information tends to be. We therefore introduce TRAP, a one-sided penalty on tokenwise TRA that acts only where the target model pulls ahead of its reference. On student essays with annotated personal information and clinical cases with patient identifiers, TRAP brings memorization near the level of an untrained model at little utility cost, where generic regularizers barely move and differential privacy gives up most of what fine-tuning bought. 

---
# Supporting and Performing Culture from the Inside 

**Authors**: Lea Frermann, Steven Bird  

**Link**: [PDF](https://arxiv.org/pdf/2609.32281)  

**Abstract**: Anthropology and related disciplines which study culture have often found it useful to consider their epistemological and methodological approaches in terms of emic versus etic. This is the distinction between the insider perspective, how the world is viewed by a member of the culture, versus the outsider perspective, the scientific cataloguing of cultural knowledge and practice. We adopt this framework to systematically analyse assumptions in recent research on culture in NLP along the full pipeline of task selection, data collection, system design and evaluation. We draw attention to the `culture of NLP' which shapes the research focus and approaches of our field, and suggest pathways towards a more culturally attuned AI. 

---
# Before Answering: Evidence Sufficiency under Size-Matched Memory Construction 

**Authors**: Joyanta Jyoti Mondal, Md. Shifatul Ahsan Apurba, Mridul Banik, Md Masud Al Mahmud, Ibne Farabi Shihab  

**Link**: [PDF](https://arxiv.org/pdf/2609.32269)  

**Abstract**: Agents that answer questions from compressed or retrieved memory must recognize when the evidence a query needs is no longer in memory. Benchmarks for this task usually create insufficient-evidence examples by deleting supporting passages. We show that this construction leaks the label through memory size: on MuSiQue, a classifier that only counts paragraphs reaches an area under the ROC curve (AUROC) of $0.979$ for detecting unsafe memory, higher than the lexical estimator we initially evaluated. We propose a size-matched construction that provably removes this shortcut, and use it to study MemSafe, an estimator that cross-encodes the query with each memory unit and aggregates the units with a set transformer. Across three multi-hop question answering datasets and five seeds, MemSafe reaches $0.968$ and $0.983$ AUROC on MuSiQue and HotpotQA, $0.26$ to $0.39$ above a lexical baseline, while the third dataset, 2WikiMultiHopQA, is saturated. A frozen pretrained cross-encoder with a logistic head already closes $41\%$ of the MuSiQue gap between the lexical baseline and MemSafe. At the same time, MemSafe degrades more than a weak baseline on the unanswerable questions released with MuSiQue, reaches only $0.639$ AUROC on SQuAD~2.0, and needs several thousand clinical training examples before it outperforms a feature-based estimator. Used as a gate for a 7B reader, it reduces the error rate on answered questions from $0.850$ to $0.631$ at $5\%$ coverage, outperforming both reader confidence and, on average, the ground-truth integrity label, although a 7B LLM judge is the better gate at $10\%$ coverage. These results indicate that the way insufficient evidence is constructed matters as much as the estimator that detects it. 

---
# LANTERN: Illuminating Hidden Mathematical Knowledge in Language Models 

**Authors**: Pavel Tikhonov, Elena Tutubalina, Ivan Oseledets, Dmitry I. Ignatov, Mikhail Seleznyov  

**Link**: [PDF](https://arxiv.org/pdf/2609.32264)  

**Abstract**: Language models can now prove theorems, but people still decide which problems to pursue. We ask whether a model's internal representations can help identify promising mathematical connections. We develop LANTERN, a fast, cost-efficient pipeline that uses a classifier over pretrained-model activations to rank candidate relations, followed by staged filtering, hypothesis generation, executable verification, and analytical checking. Applied to the On-Line Encyclopedia of Integer Sequences (OEIS), LANTERN ranked 50 million pairs among 10,000 frequently referenced sequences and produced 62 verified relations between pairs without an existing OEIS cross-reference. A content screen retained 13 relations worth presenting; nine of these are informative or insightful, including four which are entirely novel to the best of our knowledge: none appears in the OEIS or in our targeted literature search. The entire end-to-end process including classifier training, candidate ranking, filtering and verification took under 8 hours. 

---
# Solving Every Step Is Not Enough: Milestone Oracles Reveal a Composition Gap in LLM Math Reasoning 

**Authors**: Zhuohan Wang, Haoran Ma, Tianyu Wu, Yuanlin Duan, Zichun Liao, Jieming Yu  

**Link**: [PDF](https://arxiv.org/pdf/2609.32235)  

**Abstract**: Large language models (LLMs) can solve every intermediate step of a multi-step math problem on its own and still fail the full problem, even when given a roadmap of the steps and all of their answers. We introduce OracleLadder, a diagnostic evaluation that locates where LLM math reasoning fails by giving the model increasing levels of oracle help. For each problem, a teacher model writes a fixed roadmap of intermediate sub-goals (milestones), and a deterministic symbolic verifier grades every answer. Testing the model with no help, with the roadmap, with the roadmap plus the milestone answers, and on each milestone alone sorts each failure into one of five reasoning gaps. On 354 NuminaMath problems and six models from 8B to 671B parameters (Qwen3, gpt-oss, Llama 3.3, DeepSeek-V3.1), the largest gap for every model is the composition gap, a stricter form of the compositionality gap. It covers 33-48% of problems, and 24-37% after removing problems that an LLM review flags as grading errors. Accuracy and milestone-help recovery rank the two strongest models differently, and two RLVR runs with similar accuracy gains move problems differently. The roadmap effect replicates on MATH500 and AIME 2024/25, per-problem recovery agrees for 83-87% of problems under an independent second teacher, and the help ladder carries over to code generation. We release the data, roadmaps, prompts, and code at this https URL. 

---
# KinyaMed: Seeds, Not Rows -- What a Corpus Requirement Written in the Wrong Unit Fails to Constrain 

**Authors**: Marius Bayizere  

**Link**: [PDF](https://arxiv.org/pdf/2609.32234)  

**Abstract**: Triage decides who is seen first. Building an urgency classifier for patient-voice Kinyarwanda, we found our specification could be met without producing anything it was meant to secure. We report that, and the instruments that detect it, instead of a classifier. Designed for the four languages a Rwandan health centre receives, with every instrument per-language: sentences are authored in all four arms and rows generate in one, because the frame slots those three need do not exist. Our requirement asked for one million examples; generation produced them in 130 seconds. It fails four of the nine quality gates in that specification, six when rows are attributed to their source sentence. The binding gate counts distinct authored seed phrases, not rows: at our 165, no corpus passes at any row count. Two shortfalls follow and differ: 2,835 further sentences to pass the seed-count gate, 19,835 to reach the stated million rows, because a separate gate caps a seed at 50 rows. Rows come from a machine at 7,700 per second; seeds from clinicians. A row count constrains the cheap quantity, leaves the expensive one free, and so does not constrain quality at all. Two further negatives follow. An evaluation set of 17,942 rows built from nine distinct sentences supports no verdict: our gate, which counts distinct sentences, refuses 38 of its cells and reports nothing. A model trained on a corpus of uniform surface form changes its predicted urgency for 31.5% of inputs under capitalisation and 21.0% under a single typo, reported as measurement and not attribution. A model directory shipped without its tokenizer loads without error and answers its class prior on input it cannot read, with well-formed probabilities. No figure here is evidence of model quality; the contribution is the apparatus and the negative results it produced, reproducible from a clean clone except where marked NOT REPRODUCIBLE. 

---
# OptiArena: Can LLMs Improve Executable Algorithms under Fixed Resource Budgets? 

**Authors**: Wenjun Peng, Xinyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.32227)  

**Abstract**: Static QA and code-generation benchmarks only partially capture the role that large language models (LLMs) now play as coding agents and research tools. We introduce OptiArena, a budget-controlled testbed for studying whether LLMs can improve executable game-playing algorithms through five rounds of code edits within a fixed minimal scaffold and under bounded evaluator feedback and fixed resource budgets. The testbed uses two optimization regimes, surface obfuscation controls, calibrated references, held-out/stress splits, and diagnostics for degradation and exceptional failures, with LLM API cost reported separately from local evaluator wall-clock. The empirical study asks three questions: whether models can close the calibrated gap between a designated weak starter and an editable competent baseline, whether they can refine editable competent baselines without damaging them, and whether gains survive surface obfuscation controls. Across twelve frontier LLMs and five games, models improve designated weak starters more consistently than they refine editable competent baselines, with substantial variation across games and models. OptiArena provides a practical testbed for measuring bounded-resource algorithm optimization within the five-edit, fixed-scaffold setting studied here. Code is available at this https URL. 

---
# A model of rational interlocutors: Unification of comprehension and production 

**Authors**: Hanlin Wu, Zhenguang G. Cai  

**Link**: [PDF](https://arxiv.org/pdf/2609.32216)  

**Abstract**: Who we communicate with influences both our interpretation of their utterances and the design of our own. Such adjustment to the conversational partner is studied as speaker modeling in comprehension and as audience design in production, with the two literatures having developed largely separately. We argue that both adjustments express one rational computation and propose the rational interlocutor (RI) model, a computational account unifying comprehension and production. An interlocutor maintains a model of their partner, defined by three parameters: an identity parameter {\Pi} sets the messages and forms expected from the partner; a fidelity parameter {\Phi} sets how reliably messages and utterances map onto each other for them; a knowledge parameter {\Lambda} sets how knowledgeable the partner is believed to be. Comprehension and production are thus mirror-image modes of one computation over the partner model. Comprehension chooses the message the partner most likely intends to convey, weighing how well each candidate fits the utterance against how likely this partner is to mean it. Production chooses the utterance from which the partner will best recover the message, weighed against the effort of saying it. This explains why comprehenders appear to rely less on the forms produced by a linguistically less competent speaker, while producers tend to invest more effort in designing forms for them. We conjecture that perceived linguistic competence decomposes into two of these quantities: fidelity and knowledge. Their contrasting profiles across second-language (L2) adults, children, and artificial partners produce distinct and testable predictions. 

---
# Generalization and Memorization along the Learning Trajectory of Neural Language Models: A Geometric Account of Categorization 

**Authors**: Wang Bojun, Holly Jenkins, Elizabeth Wonnacott  

**Link**: [PDF](https://arxiv.org/pdf/2609.32199)  

**Abstract**: We investigate how generalization and memorization develop along the learning trajectory of neural language models. Using controlled synthetic grammars, we examine both the geometry of representation space and model behaviour over training. We find that continuous regions of representation space not occupied by observed tokens become systematically structured from the earliest stages of learning, forming category-level geometric organization that supports generalization to unattested combinations. Generalization therefore emerges from the beginning of learning rather than only after extensive memorization. With prolonged training, larger models increasingly distinguish observed from unobserved grammatical combinations. At the same time, the continuous geometric structure supporting category-level generalization is gradually destructed. Together, these results suggest that neural language models initially learn through categorization-based generalization, followed by a gradual transition toward more exemplar-specific memorization. 

---
# The Judge Is Not Its Twin: Post-training makes a model's writing more predictable but barely moves its taste, as a judge, toward predictable writing 

**Authors**: Arman Nik Khah, Arvin Bahreini  

**Link**: [PDF](https://arxiv.org/pdf/2609.32196)  

**Abstract**: Language models are now routinely graded by other language models. If post-training makes a model's own writing more predictable, it may also teach the same model, acting as a judge, to reward predictable writing, so that progress on creativity would be invisible to automated evaluation. We follow two open model families, OLMo-2 and Zephyr (7B parameters each), through their public training stages and measure every stage twice, as a writer of short stories and as a judge of pairs of stories. As writers, the models drift as feared: each family's fully trained model finds the stories of its base, supervised fine-tuned (SFT) and preference-trained (DPO) stages progressively more familiar, in all ten prompts, and a model from the other family finds the trained stories 7.0 to 9.2 percent (OLMo-2) and 24 to 25 percent (Zephyr) less surprising per token. As judges, they barely move toward predictable writing. Asked which story is better, a question every judge can use to tell a story from its own words in scrambled order, no trained judge's estimated tilt toward the more predictable story grows by as much as one point in the probability of picking it. A post hoc one-sided 95% upper bound on that growth is 2.7 points on an average pair, about the size of the untrained OLMo-2 judge's own tilt. Asked which is more creative, no trained judge's estimate favors the predictable story more than its base's does. Training instead strengthens a preference for longer stories when the question is creativity, and for one answer slot, and it breaks "more creative" as a question: trained judges asked it no longer reliably prefer a story to its scrambled words. A follow-up could not build pairs that differ in predictability but not in quality, because the routes that made this writer's stories less predictable also broke some of them, often enough to fail a quality floor set in advance. 

---
# ScopeIF: Improving Scope-Aware Precise Instruction-Following in Large Language Models via Graded Reward Modeling 

**Authors**: Bosi Wen, Yilin Niu, Xiaoying Ning, Ying Zhang, Hongning Wang, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.32189)  

**Abstract**: Precise instruction-following is a fundamental ability of large language models (LLMs), requiring their outputs to strictly satisfy objective constraints in input instructions. In complex application scenarios, these constraints often possess diverse scopes that govern specific response segments rather than the entire output. However, existing optimization methods often neglect constraint scope during data construction and rely on binary per-constraint rewards, yielding limited data diversity and sparse supervision for complex constraints. To this end, we propose ScopeIF, a novel training framework for scope-aware precise instruction-following. We first introduce a unified schema that factorizes objective constraints into three decoupled dimensions: Scope, Target, and Range. Grounded in this schema, we construct ScopeInstruct, a large-scale instruction dataset with diverse scope-aware constraints, and combine tool-grounded verification with graded reward modeling to quantify the violation degree of each constraint, providing dense supervision for policy optimization. Extensive experiments demonstrate that ScopeIF consistently outperforms existing methods, particularly on complex scope-aware constraints, while preserving general capabilities. Notably, it enables optimized Qwen3-4B and 8B models to rival or surpass strong frontier models such as Gemini-2.5-Pro and DeepSeek-V3.2, establishing an effective paradigm for advancing scope-aware instruction-following. Our code and data are available at this https URL. 

---
# Typed Decision Models: An Early Evidence Audit and Evaluation Checklist 

**Authors**: Lijuan Tang, Yuemeng Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2609.32160)  

**Abstract**: Typed decision models (TDMs) return probability distributions over caller-defined options without generating text. TypeSafe released Jev, a commercial typed decision model, on 15 September 2026, and a small body of evaluation and replication work appeared within days. We review 28 papers posted between 19 and 24 September and relate their findings to earlier work on label-probability classification, constrained decoding, reranking, calibration, and model cascades. In this early literature, the typed readout itself has not shown an independent accuracy advantage over comparable label-probability readouts. Jev's clearest gains are in latency and cost, while accuracy gaps remain on harder tasks. In practical deployments, confidence is often used to decide when to defer to a stronger model or a human. We use recurring weaknesses in these studies to derive a 14-item evaluation checklist for future TDM work. Because the evidence covers only the first nine days after the release of one hosted model, the review should be read as an early evidence map rather than a settled assessment of the model class. 

---
# Using LMs to Model the Effects of Context and Coreference during Sentence Comprehension 

**Authors**: Kohei Kajikawa, Lin Ai, Tatsuki Kuribayashi, Ethan Gotlieb Wilcox  

**Link**: [PDF](https://arxiv.org/pdf/2609.32119)  

**Abstract**: Language models (LMs) are often used as a tool to model human language processing. Recent studies suggest that severely restricting LMs' context window improves their fit to human psycholinguistic data by simulating human working memory constraints. However, it is possible that this strict memory-decay approach overlooks humans' reliance on long-range structural representations, such as discourse structre. In this work, we systematically vary the context window size of GPT-2 across four large-scale naturalistic English reading-time datasets and observe a U-shaped relationship: Although restricted contexts (< 20 tokens) successfully capture local memory limitations, expanded contexts (500--1,000 tokens) ultimately yield the highest overall psycholinguistic fit. To investigate the mechanism driving this benefit, we conduct a counterfactual inference-time experiment that disrupts cross-sentential entity chains by pronominalizing repeated discourse entities. Obscuring these structural linkages significantly degrades the predictive power of larger context windows by 20% to 40%. Our experiments demonstrate that tracking long-range coreference relations is one important factor for the alignment between LM surprisal and human reading behavior, and approximate the extent to which human comprehenders use global discourse relations during language processing. 

---
# mu-bench: A Multilingual Utterance Transcription Benchmark 

**Authors**: Andrea Li, Soham Ray  

**Link**: [PDF](https://arxiv.org/pdf/2609.32082)  

**Abstract**: Voice agents depend on accurate automatic speech recognition (ASR) to act on what callers say, yet ASR is evaluated on read, English-centric speech with word error rate (WER), which penalizes surface rather than semantic differences. We introduce mu-bench, a dataset of 4,270 caller utterances from 250 phone calls to an AI banking agent in English, Spanish, Turkish, Vietnamese, and Mandarin, centered on form-field inputs such as names, email addresses, and confirmation codes. We release Utterance Error Rate (UER), an LLM judge of whether a transcript preserves meaning, calibrated against human raters, together with an LLM normalizer that makes WER comparable across providers' output formats. On 1,847 human-rated transcripts, UER agrees with annotators at $\kappa$ = 0.78, versus 0.53 for exact-match WER on normalized text. We rank six commercial providers on a public leaderboard; the best reaches 11.9% UER, and Mandarin is hardest for all six. 

---
# Quantization Thresholds Replicate, Failure Modes Do Not: A Three-Model Study of Agentic Tool Use in Polish from 8-bit to 2-bit 

**Authors**: Jakub Prejzner  

**Link**: [PDF](https://arxiv.org/pdf/2609.32042)  

**Abstract**: We ask how GGUF quantization affects agentic tool use in Polish and whether the effects generalize across models. We introduce PolAgentBench, a deterministic benchmark with Polish prompts and English tool schemas: a 67-task main suite (15 adversarial probes, 52 hard-tier tasks) and a 46-task arithmetic isolation ladder. Three models span two axes of variation: Bielik-11B-v3.0 and its pruned, distilled child Bielik-Minitron-7B-v3.0 isolate model compression, and Llama-PLLuM-8B adds a change of pretraining family. Each is measured at six precisions, Q8_0 to Q2_K. Only the collapse threshold replicates. (1) All three models fall off a cliff between 3-bit and 2-bit (11B 0.716 to 0.045, 7B 0.463 to 0.149, PLLuM 0.224 to 0.015; paired McNemar p < 0.001 in each), across a fourfold capability spread and both axes. (2) Failure modes do not replicate: at 2-bit the 7B fails long (median 9.1k tokens, 4 steps) while the 11B mostly answers at the first step with a confabulated final answer (37 of 64 failures); PLLuM fails on content across precisions (71.8-92.0% of steps parse). (3) On the arithmetic ladder the unscaffolded rung is a floor, left standing by a rerun that states the no-tool rule; four explicit calls lift the 8-bit 11B from 1/10 to 9/10 and the 7B from 0/10 to 7/10 after format-only failures with the gold value are forgiven, an exploratory effect with eight distinct baseline inputs that does not survive multiplicity correction, while the order-trap arm separates the models at 8-bit (11B 6/6, 7B 0/6). (4) The Polish-versus-English gap is associated with degradation or with task family. We document four artifacts that shaped our conclusions (rounding-hostile gold values, strict answer typing, a no-tool rule the prompt never stated, priority-ordered failure labels), report affected results in strict and corrected form, and release the benchmark, trajectories and commit-stamped artifacts. 

---
# Before the Rollout Ends: Early Terminal Reward Prediction for Long-horizon Coding Agents 

**Authors**: Jihan Yao, Sihan Zeng, Shangbin Feng, Zhiyuan Fan, Banghua Zhu, Yulia Tsvetkov  

**Link**: [PDF](https://arxiv.org/pdf/2609.31995)  

**Abstract**: Long-horizon coding agents receive verifiable rewards only after completing expensive sequences of tool calls. This increases inference cost, amplifies early wrong hypotheses, and can lead to sparse terminal reward and unstable training. We introduce Contextual Early Reward (CER), which predicts terminal reward through behavioral evidence in a trajectory prefix. CER synthesizes adaptive rubrics specific to the current task and stage through experiences summarized from related historical tasks. In test-time scaling on SWE-bench Verified, CER improves RM@8 over the strongest baseline by 4.2 percentage points (pp) on Nemotron 3 Ultra and 2.0 pp on Qwen 3.6 27B; on Nemotron, it takes only 15.3% tokens to match the best baseline performance. In RL training experiments, CER exceeds full-rollout TMax by 1.9 pp while using 52.7% fewer online policy-and-judge tokens. Together, CER provides an interpretable, efficient, and dense evaluation method for long-horizon coding agents. 

---
# Communication between Frozen Large Language Models via Prompt Optimization in a Referential Game 

**Authors**: Vivek Anand, Muthu Chandrasekaran, Shiva Chaitanya  

**Link**: [PDF](https://arxiv.org/pdf/2609.31989)  

**Abstract**: We study communication between two frozen large language models from different providers, with different tokenizers, accessed through their API endpoints. The two play a referential game: one sees an object and describes it in a short fixed-length message over a small alphabet; the other must pick that object out of a candidate set. Neither model's weights are updated. Each agent's prompt is rewritten by an isolated prompt optimizer whose reflection model reads that agent's scored interactions. In the positional setting, optimized prompts carry a shared code that generalizes to held-out objects above a measured no-codebook baseline, including when the memory window is removed. In a second setting, independent per-letter blocks no longer fit within the message, although a whole-object place value code does. The base system fails to establish reliable communication: the sender struggles to retain an injective rule, and the receiver has too few confirmed examples in view. A sender collision penalty, retention of successful interactions, and sequential optimization enable successful place value communication in some runs. Outcomes vary across runs and reflection models. In successful runs, the protocol is written into the optimized prompts, where it can be read and audited directly. 

---
# Toward Embedding-Based Psychometrics: Structural Modeling of Assessment-Item Semantics With Contextual Scores 

**Authors**: Jinsong Chen, Shi-Ting Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.31976)  

**Abstract**: Contextual scores represent assessment items through their similarities to reference words in an external corpus. We examine the semantic structure of scores for 40 TIMSS mathematics scored units using a partially specified two-step factor procedure. A search across factor counts identifies a persistent seven-group structure under the featured construction. Subsequent comparisons consistently favor a general dimension alongside group associations, although individual group memberships remain sensitive to some specification choices. Item examples distinguish recurring, cross-domain, sensitive, and imposed associations. Simpler and unrestricted references clarify the contribution and limits of the anchored representation: it improves on a single factor but does not achieve the lowest working Bayesian information criterion (BIC). A separate response benchmark compares three initial Q constructions and their Hull-PVAF revisions under higher-order and saturated attribute distributions. Among these diagnostic models, BIC favors the official content framework and the Akaike information criterion (AIC) favors its direct four-factor augmentation, but a matched unidimensional two-parameter logistic model has lower AIC and BIC than all twelve conditions. These findings support a conditional semantic representation while limiting direct diagnostic interpretation. We discuss learned text-assisted response calibration as a prospective application requiring a larger calibrated item bank and independent evaluation. 

---
# Extraction of clinical findings from mammography and breast ultrasound reports: a comparison between specialists and Artificial Intelligence 

**Authors**: Lorenzo Farias, Hanna Reckziegel, Daniela Duarte da Silva Bagatini, Daniel Schulz, Gabriela de Andrade Monteiro, Letícia Zanatta, Ana Laura Brill Thum, Priscila Schmidt Lora, Débora Oliveira da Silva, Ana Paula Wernz da Cunha Müller, Cristiane Drebes Pedron  

**Link**: [PDF](https://arxiv.org/pdf/2609.31974)  

**Abstract**: Breast cancer is the leading cause of cancer-related death among women in Brazil, and the time between the request and the release of mammography reports directly influences adherence to screening, making the agility in processing these reports a critical factor for early diagnosis. In this context, this study compares the performance of a Large Language Model (LLM) with manual extraction performed by a team of health researchers in identifying clinical findings from mammography and breast ultrasound reports written in Brazilian Portuguese. Named Entity Recognition (NER) was applied through Prompt Engineering using a few-shot strategy, employing the Gemini 2.5 Flash model, selected from preliminary exploratory tests with four candidate models. The Gemini 2.5 Flash model demonstrated the best performance, achieving a Macro F1 of 0.91 and a Micro F1 of 0.98. The subjective validation, in which 29 exams of different formats were evaluated by health researchers using a Likert scale, yielded an agreement index of 93.1%. The model outperformed human extraction in overall Macro F1 (0.91 vs. 0.72), as in four reports the model correctly identified information that had been omitted or incorrectly recorded during manual extraction, demonstrating its potential as a complementary verification tool alongside specialists. The results confirm the hypothesis that LLMs, when instructed through Prompt Engineering, can achieve performance comparable to or superior to manual extraction by health professionals. 

---
# IndicFDB: Benchmarking Full-Duplex Voice Agents across Indian Languages 

**Authors**: Rajarshi Roy, Shobhit Banga, Jonathan Raiman, Supriya Paul, Bhaskar Singh, Manmeet Kaur, Sagar Jain, Hanuman Sidh, Pranav Sharma, Aditya Singh, Aaditya Pareek, Manas Dhir, Adi Margolin, Niket Agarwal, Bryan Catanzaro  

**Link**: [PDF](https://arxiv.org/pdf/2609.31967)  

**Abstract**: Full-duplex voice agents must handle pauses, take turns, backchannel, and respond to user interruptions in real time. Full-Duplex-Bench evaluates these behaviors, but its English-only corpus and reliance on word-timestamped ASR and an English-prompted LLM judge make it difficult to extend to Indian languages. We introduce IndicFDB, which extends it to ten languages spoken in India with 12,350 samples, nearly 17 times as many as the original. We address three challenges: finding conversational events in multilingual speech, evaluating their timing without reliable word-level alignment, and judging responses across languages. We mine pause handling, turn taking, and backchanneling samples from roughly 50,000 hours of channel-separated conversations using voice activity detection (VAD), and construct human-validated synthetic user interruption samples. Language-independent VAD heuristics evaluate timing, while an open-weight transcription and translation pipeline converts responses to English for LLM ratings of relevance and quality. Across seven voice agents, commercial APIs show unexpectedly consistent behavior across languages but are either fast or robust to pauses, never both, while monolingual open full-duplex models expose further tradeoffs among backchanneling, response quality, and latency. 

---
# Transformer MLP Gate Thresholds Are Couplings to a Carried Reference Direction 

**Authors**: Olli Tuomi  

**Link**: [PDF](https://arxiv.org/pdf/2609.31956)  

**Abstract**: The corpus-mean direction of a transformer's residual stream is a component shared across all inputs, and is commonly removed by mean-centering before representational analysis. We present evidence that it is a functional component: the reference against which the MLP gate population sets its operating point. In Phi-2, an exact decomposition of resting gate pre-activations shows that at mid-stack layers the resting inhibition of >99.9% of gates is carried by the coupling $w \cdot b_L$ to the carried mean direction, at 48-56$\times$ the explicit bias parameter, and the coupling is direction-specific: a random direction at matched norm orders the population's firing rates at Spearman $\rho \leq 0.09$ where the reference reaches 0.95. (That 0.95 is near-tautological on its own; the paper derives its null.) Causally, removing the stream's projection on the reference multiplies above-threshold firing by about 9$\times$, dose-monotonically, at 23-59$\times$ a norm-matched control; a random-initialised twin is flat, and replacement tests show that the direction carries the function and the magnitude does not. A direction-matched control makes the same point: noise injected along the reference costs 8-42$\times$ the same energy along a random direction. The decomposition replicates on four further families spanning both gate types (GELU with an explicit gate bias, bias-free SwiGLU), and the dose-response on two of them. Across an eight-model scan the mechanism is present in every GELU and SiLU family and absent only in OPT, where an opposing LayerNorm bias cancels the carried reference. Gate thresholds are implemented as couplings to a constant the network builds for the distribution it is reading, with the bias parameters contributing little; how much of that constant is carried in the stream and how much in parameters depends on the architecture. 

---
# Agents Can Use Base Models to Evade AI Detection 

**Authors**: Bhuwan Dhingra, Danish Pruthi  

**Link**: [PDF](https://arxiv.org/pdf/2609.31876)  

**Abstract**: We show that coding agents equipped with a base language model can successfully assemble responses from its samples to evade detection. Base models have been shown to evade commercial detectors, however, prior "humanization" techniques rely on using these models to paraphrase AI outputs over several iterations, which invariably results in semantic drift. In contrast, equipping coding agents to directly orchestrate the writing process by stitching text samples from a base model allows it to produce outputs that are coherent, task-specific and generally high quality. We find that Claude Opus 5 operating in a Claude Code harness effectively orchestrates a local 32B parameter OLMo-2 base LM and sacrifices little task accuracy across benchmarks spanning creative writing, factual grounding, health QA and instruction following, while using up to 90% base LM tokens. Responses constructed in this manner reduce the effectiveness of both post-hoc detectors (Pangram v4 detection rate drops from 77% to 24%) and soft watermarking applied a priori to the agent's generations (down to a simulated 10% detection at low FPR). While effective, this evasion requires a significantly larger number of input and output tokens from the agent, increasing the dollar cost per query up to 30x at API-pricing. Overall, this work demonstrates the effectiveness of a new class of adversarial attacks against AI text detection, and urges post-hoc detection providers to include outputs of base models in their training. 

---
# Omni-IO Skills: Harnessing Your Agent Omni-Native 

**Authors**: Yanlin Li, Mingyang Hao, Shengqiong Wu, Hao Fei, Mong-Li Lee, Wynne Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2609.31847)  

**Abstract**: General-purpose agents can plan, reason, and act over long horizons, yet their production capabilities remain fragmented across text, images, audio, video, documents, 3D assets, and code. Extending a foundation model to additional modalities ties capability growth to costly model updates, while assembling specialist models and tools leaves unresolved how procedures, dependencies, intermediate assets, and cross-turn revisions should be coordinated. We present Omni-IO Skills, a plug-and-play Agent Harness that makes existing agents omni-native through hierarchical Skills, a standardized multimodal execution interface, dependency-aware orchestration, and a persistent Asset Registry. Multi-asset workflows are represented as Declare Execution Graphs, which schedule independent operations concurrently and register successful outputs for downstream and cross-turn reuse across replaceable execution backends. Its 27 Skills cover 38 representative tasks spanning seven artifact modalities and four capability families: understanding, generation, reasoning, and retrieval. On UniM-90, the harness raises the input-support rates of GPT-5.6 Sol and Claude Sonnet 5 from 40.00% and 38.89% to 100%, while increasing relative Semantic--Quality Coupled Score from 26.99 to 74.94 and from 27.82 to 77.78, respectively; Strict Structure Score reaches 100.00 and 99.78. These results establish harness-level capability composition as a practical route to broad, evolvable Omni systems without changing the host agent's reasoning core. 

---
# The Ongiini-Eval-OW Benchmark: A Concept Paper for the Planned Benchmarking of Machine Translation and Large Language Models on Oshindonga and Oshikwanyama 

**Authors**: Sebastian Küpers  

**Link**: [PDF](https://arxiv.org/pdf/2609.31727)  

**Abstract**: Oshiwambo -- a cluster of mutually intelligible Bantu languages spoken by over a million people across northern Namibia and southern Angola, and the home language of roughly half of Namibian households -- has, to our knowledge, no published machine-translation evaluation benchmark. Major commercial services (Google Translate, DeepL, Microsoft Translator), open multilingual MT models (NLLB-200, MADLAD-400), and the open Masakhane checkpoint collection all lack coverage of either standardised dialect, Oshindonga or Oshikwanyama. We announce Ongiini-Eval-OW, a planned 600-item English-Oshindonga and English-Oshikwanyama benchmark with native-speaker references from two independent translators, a 30-item inter-translator agreement set, an 11-tag phenomenon-tagged stratification (at least 30 items per tag), a deterministic 30% blind split, and a reproducible scoring protocol over chrF++, BLEU, and COMET-22, supplemented by a 50-item human-evaluation round. We document the empirical coverage gap, the dataset composition, the launch-leaderboard model matrix across American, European, and Chinese frontier and open-weight systems, and the contribution pipeline. The dataset is targeted for first public release in Q4 2026; this v1.0 concept paper announces the design and the call for participation. Data and code will be released under CC-BY-4.0 and MIT respectively. 

---
# Don't Repeat Yourself: Self-Supervised Fine-Tuning for Coverage 

**Authors**: Eric Fithian, Kirill Skobelev, X.Y. Han  

**Link**: [PDF](https://arxiv.org/pdf/2609.31688)  

**Abstract**: In verifiable domains such as math and coding, finding one correct solution among many attempts can matter more than the pass rate of each attempt. Post-training can concentrate large language model outputs around a few modes, while increasing sampling temperature has limited effectiveness. We introduce Don't Repeat Yourself Supervised Fine-Tuning (DRY-SFT), a post-training method that increases output diversity and coverage: the probability of at least one correct solution among many attempts. DRY-SFT has two stages. First, for each problem, sequentially generate K solutions, showing the model all prior attempts and asking for a different solution. Second, fine-tune on each attempt independently, removing prior attempts from the context. The process uses no reward, verifier, or correctness filter. On HumanEval+, MBPP+, and DS-1000, DRY-SFT raises pass@100 by 10.8, 12.5, and 12.4 percentage points, respectively, at a small cost to pass@1. Structural diversity, measured by abstract syntax tree edit distance among passing solutions, rises significantly on all three benchmarks. DRY-SFT also solves 244 of 600 problems that the base model did not solve in the same 200 attempts. Across nine open-weight models, lower structural diversity of the base model significantly predicts larger DRY-SFT gains, indicating that the method is especially effective on more mode-collapsed models. 

---
# Verification of PETSc with CIVL using LLM-generated ACSL contracts and deterministic driver generation 

**Authors**: Hansol Suh, Jan Hückelheim, Stephen Siegel  

**Link**: [PDF](https://arxiv.org/pdf/2609.31687)  

**Abstract**: Parallel numerical libraries such as PETSc are widely used in science and engineering applications where wrong results can have costly consequences. Despite this, numerical libraries are rarely formally verified. One of the challenges is the need for an expert to hand-write a specification and manually apply a verification tool, often requiring the development of a harness or driver, all of which can contain additional bugs that lead to false positives or false negatives during verification. With recent advancements in large language models (LLMs), it is tempting to generate such drivers and reference models automatically, but one-shot generation based on a simple prompt is brittle and leads to additional unverified code that needs to be audited. In this paper, we present an approach to use LLMs in a limited setting to generate a small, human-certifiable ACSL contract from the function's documentation, combined with a deterministic toolchain that supports a restricted ACSL profile and generates a driver that uses the CIVL verifier to check the implementation against the contract and, when available, an existing reference model. We demonstrate the pipeline end-to-end on three PETSc functions: MatAXPY (reference model already exists), MatAYPX (no reference model, so the certified contract is the sole oracle), and the non-compressing mode of MatFilter (no reference model, with more complex, conditional behavior). With this pipeline, we were able to discover a bug in PETSc's MatAYPX function that was previously undiscovered and had been present in the code since 1997. 

---
# The Temporal Tug-of-War: Visualizing and Detecting RAG Conflicts in Diffusion Models via Trajectory Variance 

**Authors**: Sravan Karthick T, Pranav Darshan, Pranav A, Minal Moharir, Ivan P. Yamshchikov  

**Link**: [PDF](https://arxiv.org/pdf/2609.31684)  

**Abstract**: Retrieval-Augmented Generation (RAG) introduces a specific failure mode in discrete diffusion language models: when retrieved context contradicts parametric knowledge, the iterative denoising process becomes a visible battleground between competing knowledge sources. We identify temporal semantic divergence as an observable for detecting these conflicts and introduce the Trajectory Variance Score (TVS), a simple and interpretable measure of this divergence. TVS computes the mean pairwise cosine distance of answer embeddings across independent stochastic denoising trajectories, capturing the temporal tug of war between parametric and contextual attractors. Requiring as few as two parallel inference runs, TVS is computationally lightweight. Across four diverse datasets (Synthetic, SciQ, PopQA, and CounterFact), a simple Logistic Regression classifier using TVS achieves $70.10\%$ accuracy and $0.7647$ AUROC on LLaDA. On Dream 7B, increasing the number of trajectories from two to five improves accuracy from $63.91\%$ to $69.62\%$. More complex sequential models provide only marginal improvements over the linear classifier. Evaluation across LLaDA and Dream 7B demonstrates that conflict-induced trajectory dynamics and their key properties transfer across distinct diffusion architectures. 

---
# An Evaluation of AI-Supported Evidence-Based Learning for Public Speaking Skill Development 

**Authors**: Sashini Hettiarachchi, Shahbaz Siddeeq, Mika Saari, Pekka Abrahamsson  

**Link**: [PDF](https://arxiv.org/pdf/2609.31676)  

**Abstract**: Public speaking is an essential skill in academic and professional contexts, but it often causes anxiety. Although several AI-based speech coaching tools exist, they typically provide generic feedback and lack model speeches for learning. This study addresses these gaps by developing a system that incorporates speech context into evaluation criteria, enabling more tailored feedback. It also generates improved model speeches in both text and audio formats to support example-based learning. The system was evaluated in a pilot study with seven undergraduate students over one to three weeks. Participants used the system repeatedly and practiced the same speech at least three times. Results showed improvements in public speaking performance for all participants. Filler word usage decreased, and anxiety levels dropped by 15.3% to 34.4%. Participants found the context-aware feedback and revised speeches useful and confidence-building. These findings suggest that context-aware feedback and model speeches can enhance public speaking skills and reduce anxiety. Future studies could explore video-based analysis of physical behavior during speeches. 

---
# Age-Adaptive Handwriting Reconstruction from an IMU-Based Digital Pen through Shared Representations and Domain-Specific Heads 

**Authors**: Florent Imbert, Yann Soullard, Eric Anquetil, Hui Han  

**Link**: [PDF](https://arxiv.org/pdf/2609.31666)  

**Abstract**: Digital pens are widely used to capture handwriting on digital devices, enabling precise trace recording and enhancing human-computer interaction. However, most are bundled with tablets and lack cross-brand compatibility. Recent digital pens equipped with kinematic sensors have emerged, designed for use on any surface. This especially opens significant potential for supporting handwriting acquisition in classrooms.  Handwriting reconstruction from such an IMU-equipped pen poses a challenge due to the significant variability in sensor signals between adults and children. Even when producing visually similar traces, variations in writing dynamics, motor control, pen holding, and user confidence introduce substantial discrepancies in the captured signals. Additionally, the high variability in children's handwriting requires collecting large amounts of data, which is not feasible to implement in a school environment at scale. Furthermore, models trained exclusively on adult data fail to generalize to children's handwriting, and conversely, models trained on children's data perform poorly on adult writers. This crosspopulation degradation highlights the need for a unified model that can be deployed directly on the pen, without any user-specific adaptation. To address this issue, we propose a cross-domain learning, using an original neural network architecture based on a Temporal Convolutional Network and multiple prediction heads. The model is designed to be robust across age groups by leveraging shared features while effectively handling variability induced by differences in graphomotor development. This approach aims to improve handwriting trace reconstruction from sensor data, where each domain benefits from additional data provided by the other domain. 

---
# What Drives Dialectal Jailbreaks? An Ablation of Surface Form, Cultural Framing, and Strategy Banks 

**Authors**: Qingyang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2609.31664)  

**Abstract**: Recent work suggests that obscure language registers can weaken large language model refusal behavior, especially when paired with black-box prompt optimization. It remains unclear whether failures stem from non-standard surface form, culturally grounded framing, or optimization over an expressive prompt-strategy space. We study Chinese registers by extending a classical-Chinese red-teaming framework to Shanghainese and Cantonese and running a 36-cell ablation across surface forms, strategy-bank variants, and two target models. The main finding is corrective: dialectal surface form is neither necessary nor sufficient for high attack success. Non-optimized English, Mandarin, and naive dialect translations remain below 8\% attack success rate, whereas all conditions that retain an optimizer-controlled strategy bank reach 98--100\%. A culture-neutral generic strategy bank reaches the same ceiling at near-single-query cost, further indicating that strategy-bank expressiveness, rather than dialectal or cultural content, accounts for most of the observed effect. Dialect choice still affects query efficiency and response severity, and qualitative coding identifies recurring high-level mechanisms such as semantic glossing and procedural scaffolding. We qualify the absolute ceiling-level attack success rate by showing sensitivity to judge calibration and noting that the design does not fully separate one-shot strategy construction from iterative search. 

---
# LLM-Guided Ontology-Driven Knowledge Graph Construction from Unstructured Text 

**Authors**: Abdelhadi Belfadel, Maxence Gagnant, Joseph Kattan, Sana Tmar  

**Link**: [PDF](https://arxiv.org/pdf/2609.31663)  

**Abstract**: Ontology-driven knowledge graph construction from industrial text remains challenging due to the domain specificity of documents, the scarcity of annotated resources, and the complexity of ontology engineering workflows. This paper presents and investigates the applicability of an ontology learning pipeline that combines compact open-source Large Language Models (LLMs), reusable prompting strategies, and open knowledge bases to support the extraction, structuring, enrichment, and evaluation of knowledge from textual corpora. The approach is tested and evaluated on a private French corpus of power-grid incident reports, using locally deployable open-source LLMs ranging from 7B to 32B parameters. Starting from unstructured reports, the approach extracts entities and relations, generates RDF triples, constructs related OWL ontology, enriches it using external knowledge sources, assesses the quality of the ontology, and subsequently constructs a populated knowledge graph grounded in the resulting ontology schema. Experiments on 80 manually annotated private reports show that schema-guided prompting significantly improves extraction quality, while quantized models provide an effective trade-off between performance and computational cost. These results demonstrate the feasibility of transforming domain-specific industrial text into ontology-based knowledge graphs using locally deployed open-source LLMs, while supporting the generalization of the extraction process through reusable prompting strategies. 

---
# Parser, Chunking, and Embedding Interactions in Retrieval-Augmented Generation over Indian Government Regulatory Documents 

**Authors**: Shubham Kumar Singh  

**Link**: [PDF](https://arxiv.org/pdf/2609.31660)  

**Abstract**: Retrieval-augmented generation (RAG) pipelines are typically assembled from independently-chosen components -- a document parser, a chunking strategy, and an embedding model -- yet these choices are rarely evaluated jointly, and evaluations that do combine them are usually run on a single document or corpus. We present a controlled factorial study of 3 parsers, 3 chunking strategies, and 5 dense embedding models, together with a sparse BM25 baseline, evaluated against 800 question instances, each with one or more required evidence strings, with evidence strings automatically validated against source text and a 10% random sample manually reviewed, across four structurally distinct Indian central-government regulatory documents. We fit linear mixed-effects models with document-query-level random intercepts to the resulting 72,000-row result set, run Holm-corrected paired comparisons between matched dense and sparse configurations, and report clustered bootstrap confidence intervals for all 54 unique retriever configurations. We find that no single retriever family dominates across documents; parser and chunker choice interact significantly; MPNet-base is a consistent underperformer with a severe failure mode on table-derived questions; and the corpus exhibits a near-saturated evidence-preservation ceiling above 98%, indicating that retrieval differences are driven primarily by ranking quality rather than information loss during ingestion. We additionally report embedding-dimension and chunk-size/overlap ablations and an efficiency/quality Pareto analysis. We release our full evaluation harness, corpus manifest, and 800-question benchmark. 

---
# Distributional sentiment modeling and anomaly detection for consumer complaint assessment 

**Authors**: Peiheng Gao, Chen Yang, Shimin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.31653)  

**Abstract**: Sentiment analysis is a common tool for converting unstructured text into quantitative signals in finance and risk management. Yet most applications reduce the output to a discrete polarity label or a single predictive feature, overlooking the distributional structure of sentiment intensity in consumer complaint narratives. In this paper we treat negative sentiment in consumer complaints as a bounded continuous variable and study its full distribution rather than a single label. We score each narrative with a transformer classifier, model the scores with Beta distributions, and compare the fitted distributions of meritorious and non-meritorious complaints through the Kullback Leibler divergence and the squared Hellinger distance. The fitted distributions are then linked with dollar amounts and company response outcomes to construct anomaly diagnostics that flag complaints whose textual severity is inconsistent with the recorded relief. We find that the two groups have strongly overlapping distributions, so negative sentiment intensity is not a sharp classifier of outcomes on its own; combined with monetary and categorical attributes, it isolates unusually severe complaints for operational risk monitoring. Treating sentiment analysis as continuous distributional measurement, this study links sentiment extraction, bounded response modeling, and anomaly detection in a unified framework for consumer complaint assessment. 

---
# A literature-guided descriptor-based framework for filtering composition search spaces 

**Authors**: Lei Zhang, Markus Stricker  

**Link**: [PDF](https://arxiv.org/pdf/2609.31650)  

**Abstract**: Scientific literature contains latent knowledge about materials behavior, but much of this knowledge is expressed through words, contexts, and recurring associations rather than explicit design principles. This raises a central question: how can large-scale scientific corpora be used for practical problems in materials discovery? Here, we present a literature-guided descriptor-based filtering framework for reducing composition search spaces. For a given performance metric, the framework selects two descriptors from a filtered vocabulary in a literature-trained word embedding model and uses the selected descriptors to construct a Pareto-based filter for candidate compositions. Across the evaluated performance metrics and composition search spaces, the framework filters out an average of 74.27\% of the candidate compositions, with an average best-value error of 1.93\% relative to experimental measurements. Compared with expert-chosen and random descriptors, our performance metric-dependent descriptors provide a more controlled balance between retained fraction and best-value error. These results show that literature-derived embeddings can support intuitive and reproducible filters for narrowing candidate composition spaces while preserving high-performing compositions. 

---
# ChestPheNoT: Deployable, Auditable Label-Status-Evidence Extraction from Radiology Reports 

**Authors**: Kai Yu, Chenyu Zhu, Zaifu Zhan, Meijia Song, Min Zeng, Xiaoyi Chen, Mingquan Lin, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.31629)  

**Abstract**: Structured phenotype extraction from radiology reports supports cohort construction, quality auditing, and clinical analytics, but practical deployment requires local inference and auditable predictions, while expert annotations remain scarce. Conventional labelers provide structured findings and assertion states but no supporting evidence, while API-hosted large language models may be unsuitable when clinical text cannot leave institutional infrastructure. We present CHESTPHENOT, a compact 0.5-3B language model that jointly extracts finding labels, three-class status (present/absent/uncertain), and verbatim supporting evidence spans. CHESTPHENOT is trained using hybrid CheXbert+72B silver supervision followed by supervised fine-tuning and lightweight GRPO refinement. Across three human-annotated gold sets spanning in-distribution, cross-taxonomy, and cross-institution evaluation, the 3B model remains below its CheXbert silver teacher in distribution but is competitive under distribution shift, significantly surpassing CheXbert on cross-institution detection (+2.0 F1). Task-specific training also enables the 3B model to match or exceed substantially larger prompted models on most detection and status comparisons. For evidence-grounded extraction, over 99% of final evidence spans are locatable in the source report, and the 3B model achieves 47.5 auditable-F1, outperforming Qwen2.5-7B one-shot prompting by 7.6 points and approaching Qwen2.5-72B. These results demonstrate that locally deployable models can provide competitive and directly auditable radiology-report extraction without relying on external inference APIs. Code and the full extraction/judge prompts will be made available at this https URL. 

---
# How to Loop MoE: Flatten the Experts, Untie the Attention 

**Authors**: Shouren Wang, Chuang Ma, Mohsen Hariri, Debargha Ganguly, Wang Yang, Xiaoqing Tong, Qianying Liu, Xiaotian Han, Vipin Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2609.35751)  

**Abstract**: Looped Transformers reuse one block of layers several times: by spending extra computation they push a model of fixed size further, and so use its parameters more fully; while sparse mixture-of-experts (MoE) models activate only a few of many experts for each token. Looped MoE bridges these two design philosophies and gives MoE models new potential for better expert usage, but it raises a question: how to loop a MoE? We answer it with Foil. With the expert parameters and the expert compute per token held fixed, Foil (1) flattens the experts, halving the expert layers, doubling the experts per layer and doubling the passes, so that every routing decision chooses from a larger pool, and (2) unties the attention, giving each pass its own attention parameters while the experts and routers stay shared. Experiments show that Foil clearly outperforms the unflattened looped baseline: at 20B tokens every Foil model has lower pretraining loss than the baseline; at 100B tokens the loss improves monotonically with the degree of flattening, the most flattened Foil ending 0.012 nat below the baseline at equal parameters and compute, with downstream accuracy on par or better; untying the attention also yields more balanced and more confident routing at equal shape. Our ablations analyse why Foil works and turn the findings into design guidance for looped MoE: the returns of looping and of widening the expert layers amplify each other, routing confidence tracks healthy expert use better than load balance, and a sparse looped MoE should therefore use more experts per layer and more passes. Code and configurations are available at this https URL. 

---
# Shockingly Simple Self-retrospection Improves Agentic Models Without RL 

**Authors**: Jonathan Light, Christopher Zhang Cui, Jeonghye Kim, Roger Creus Castanyer, Emiliano Penaloza, Zhengyan Shi, Alessandro Sordoni, Marc-Alexandre Côté, Xingdi Yuan, Minseon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2609.35741)  

**Abstract**: People learn not only by repeating successful actions, but also by recounting and explaining their experiences, revising their understanding to guide future behavior. Can a language-model agent improve its future actions by training only on explanations of its own experience? We investigate this question by studying Retrospection-Only Fine-Tuning (ROFT), a minimal online procedure designed to isolate the effect of explanation-only training on subsequent behavior. The agent attempts a task, observes available feedback, generates a retrospective explanation, and is fine-tuned with a next-token prediction loss on the explanation tokens alone. The procedure uses neither an external teacher nor a reward-based policy update. In software-engineering experiments with Qwen3.5-4B, ROFT is trained on problems with mixed successful and unsuccessful base-model attempts. On held-out SWE-bench Verified and Pro, it reaches 49.2% and 26.8% solve rates after 20 updates without using a verifier, compared with GRPO's 48.0% and 25.3% after 40 updates in the evaluated runs, and makes faster early progress in training time and sampled attempts. It also learns to solve individual tasks on which all 64 sampled base-model attempts failed, showing that learning can begin without any initially successful trajectories. Behavioral analyses find that ROFT indirectly assigns credit to actions, encouraging good actions and discouraging incorrect ones. Moreover, prompting retrospections to emphasize more direct solutions yields shorter subsequent attempts even without an explicit length penalty. Together, these findings show that learning to explain can also improve learning to do, establishing self-generated retrospections as useful training targets and motivating further study of explanation-to-action transfer. 

---
# Reinforcing Agentic Creativity in Scientific Ideation with Night Science 

**Authors**: Priyanka Kargupta, Silviu Cucerzan, Shweti Mahajan, Allen Herring, Jiawei Han, Ryen W. White, Sujay Kumar Jauhar  

**Link**: [PDF](https://arxiv.org/pdf/2609.35706)  

**Abstract**: Large language models (LLMs) excel at structured, verifiable tasks, but their low-entropy bias can produce homogeneous and predictable outputs, limiting their utility for open-ended scientific ideation. Effective discovery, however, spans a broader creative spectrum: from structured day science to loosely structured, serendipitous night science that reaches ideas beyond those typically considered. We introduce AI Night-Scientist, an agentic framework that uses reinforcement learning to teach models when and how to depart from predictable reasoning. Grounded in cognitive science, we model creativity along three axes: action (what to do and how creatively), process (when to explore versus exploit), and outcome (the novelty and usefulness of the resulting idea). We use these axes to train models with GRPO, exposing them to varying degrees and forms of creativity throughout training. This produces substantially more diverse scientific proposals, expanding the range of research directions by 27.8% and contribution types by 14.9% over the base model. It also improves predicted citation impact by up to 32.0 percentage points and originality by 66.2 points. These gains cannot be reproduced by simply increasing decoding temperature; instead, we find that semantic guidance specifying what kind of creativity to pursue is critical. Overall, our results suggest that creativity is a learnable, multi-level ability that can be shaped to help researchers reach ideas beyond those typically explored by LLMs. 

---
# Rethinking Personalized Generation: Test-Time Alignment via Factorized Ranking Models 

**Authors**: Qiyao Ma, Junshan Zhang, Zhe Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2609.35695)  

**Abstract**: Aligning large language models (LLMs) to diverse user preferences is fundamentally hindered by standard alignment paradigms that optimize for monolithic users. In this work, empirical studies are first used to reveal the existence of a massive, untapped performance headroom for personalized generation through test-time alignment. We demonstrate that personalized generation is uniquely suited for test-time scaling methods like Best-of-N (BoN) because it can be viewed primarily as a candidate matching problem rather than a generator capability bottleneck. While reward models could in principle exploit this headroom, they are poorly calibrated for personalization, and their billion-parameter scale makes scoring large candidate pools prohibitively expensive. To overcome this limitation, we propose a parameter-efficient framework utilizing million-parameter scale multi-layer perceptron (MLP) ranking models. Our personalized ranking model directly reuses the internal embeddings of the base generator with minimal overhead. By scaling train-time data to provide fine-grained personalized preferences, this million-parameter ranking model accurately scores large candidate pools and can seamlessly guide generation to reduce the cost of materializing N candidates. Extensive experiments on nine datasets spanning three personalized generation settings show that our personalized ranking model effectively exploits the discovered headroom, outperforming billion-parameter generalist reward models on every dataset, with under 0.4% of their parameters and four orders of magnitude lower scoring latency. 

---
# CoSE-E: A Benchmark for Code-switched Speech Evaluation in Enterprise Settings 

**Authors**: Shama Gupta, Hoang H Nguyen, Chelsea Huang, Lindsay Devon Brin, Fanny Riols  

**Link**: [PDF](https://arxiv.org/pdf/2609.35645)  

**Abstract**: Code-switching (CS), a seamless alternation between languages within a single utterance, remains a critical challenge in automatic speech recognition (ASR). While prior works focus on conversational CS-ASR, enterprise settings demand evaluation of operational impact beyond edit-distance errors: how code-switching transcription errors propagate to downstream voice agent task failures. In this work, we propose (1) a CS-ASR synthetic benchmark and multidimensional evaluation framework tailored to enterprise domains, (2) systematic evaluation of frontier ASR systems across 5 language pairs, (3) diagnostic analysis of the additional transcription errors that code-switching introduces across language pairs and models. We release COSE-E to support enterprise-focused CSASR evaluation for multilingual voice agents in enterprise deployment. 

---
# SANTA++: Sampling Attention through Representative Keys 

**Authors**: Kyle Lee, Christian Z. Pratt, Ruoyu Fang, Heekyung Lee, Avinash Lohitsa, Ryan Modafe, Kerem Y. Camsari  

**Link**: [PDF](https://arxiv.org/pdf/2609.35629)  

**Abstract**: Attention often concentrates on a small subset of tokens in the context, but which subset matters changes from one query to the next. To exploit this changing structure, we introduce SANTA++, a training-free stochastic attention method that uses representative keys for memory-efficient selection without scanning the entire key-value (KV) cache. Cached keys are organized into teams, and the query scores one representative from each team to decide which teams to sample. We compute exact attention scores within the sampled teams and reweight each team's contribution by the inverse of its inclusion probability. This importance sampling correction estimates attention over the full cache, with a sampling budget that lets us trade memory reads for accuracy. Remarkably, with 32 or 64 sampled teams, SANTA++ uses 16% to 22% of dense attention's KV reads and retains 94% to 99% of the dense-attention baseline's scores on LongBench v2 and HELMET's retrieval-augmented generation subset, and 85% to 91% on RULER, with Qwen2.5-7B-Instruct at 32K context. With 31 sampled teams, our GPU implementation delivers a $1.69\times$ attention speedup over the dense FlashAttention baseline at 32K context. By reducing the number of cache entries read, SANTA++ in principle complements architectures with compressed KV representations, such as multi-head latent attention. Our kernels are available at: this https URL. 

---
# Twist, Don't Tilt: Trajectory-Exact Constrained Decoding for Masked Diffusion Models 

**Authors**: Aditya Thimmaiah, Lara Marinov, Jayanth Srinivasa, Haris Vikalo, Junyi Jessy Li, Milos Gligoric  

**Link**: [PDF](https://arxiv.org/pdf/2609.35609)  

**Abstract**: Constrained decoding for Masked Diffusion Language Models (MDLMs) aims to ensure that generated outputs satisfy a specified structure or syntax constraint. MDLMs generate outputs by repeatedly unmasking masked positions present in their current state. Recent strategies for constrained decoding constrain the model's per-step mean-field posterior (which factorizes over masked positions) by enforcing the desired constraint with an automaton. The resulting chain-structured factor graph allows exact constrained sampling via dynamic programming. However, despite each draw being exact and constraint-satisfying, we prove that their composition, in general, tilts away from the model's relative probabilities over valid trajectories, thus leading to trajectory bias. We derive an exact expression for this bias as a product of ratios measuring how valid continuation mass changes when the denoiser is reconditioned, and characterize when the bias vanishes. We then correct the bias by introducing TWISTER, the first automaton-twisted Sequential Monte Carlo decoder for MDLMs, using the step-exact decoder as the proposal. We show that for regular language constraints, the Feynman-Kac correction is exactly computable, with the twists obtained efficiently using quantities pre-computed for step-exact sampling. We prove that the resulting Feynman-Kac model targets the unbiased Doob h-transformed path law conditioned on constraint satisfaction. 

---
# Simultaneous Translation between Sign Languages 

**Authors**: Zetian Wu, Bowen Xie, Stefan Lee, Liang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.35608)  

**Abstract**: Deaf and hard-of-hearing (DHH) signers cannot converse in real time across different sign languages today: existing sign-to-sign translation systems run offline, requiring the full source clip before any target sign is emitted. Live use cases - e.g. broadcast interpretation and two-way video calls - instead demand simultaneous output, while the source signer is still signing. We present, to our knowledge, the first simultaneous sign-to-sign (S2S) translation system, with two wait-k regimes: test-time wait-k inference applied directly to a full-sentence model, and a trained wait-k model via stochastic multi-path supervision. We further introduce ca-Stream-AL, a computation-aware latency metric for streaming output. Averaged across six S2S directions on both a smaller human-verified test set and a larger synthetic S2S corpus, our streaming system achieves a 38% ca-Stream-AL reduction while staying within a 9% DTW-PA-MPJPE increase and a 2.1 BLEU-4 drop compared to the full-sentence baseline. A word-order case study probes how the streaming model handles word order mismatch between different sign languages - a consequence of simultaneous translation. 

---
# TCSAlgBench: Benchmarking Automated Proving for Research-Level Theoretical Computer Science 

**Authors**: Chutong Yang, Xiyuan Zhang, Yu Huang, Boran Han, Soonho Kong, Shuai Zhang, Vihang Prakash Patil, Zhen Han, Michael Bohlke-Schneider, Bernie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.35606)  

**Abstract**: Large language models perform strongly on competition mathematics, but their research-level reasoning remains difficult to evaluate systematically. Theoretical computer science (TCS) connects algorithm design to explicit guarantees and fundamental limits, providing a setting for evaluating whether models can justify computational improvements with arguments humans can inspect. We introduce TCSAlgBench, a benchmark and reusable pipeline for natural-language proof discovery, comprising 398 theorem-level challenges from 138 STOC and COLT 2026 papers. Expert-designed rules complete paper-specific context, preserve computational assumptions and quantitative guarantees, and withhold constructions when discovering an algorithm is part of the task. For each task, prover systems receive theorem statements and access to cited prior work. The pipeline supports fresh, versioned challenge batches from newly released papers. We evaluate ten model configurations from four families under direct inference and prover-verifier discussion, and compare four agent workflows under matched model-call opportunities. All evaluations use the full benchmark. In the model comparison, GPT-5.6 Sol max achieves the highest five-run verifier-accepted coverage at 23.6% after 10-round discussion. Discussion and repeated sampling improve coverage. In the separate agent comparison using GPT-5.5 xhigh, decomposition improves coverage over discussion, and agentic planning achieves the highest five-run verifier-accepted coverage at 25.4%. TCSAlgBench provides a refreshable testbed for measuring progress in model reasoning and studying how agent workflows support research-level proof discovery. 

---
# SEABench: Benchmarking Endogenous Misalignment In Self-Evolving Agents 

**Authors**: Saswat Das, Parvati Viswanathan, Daniel Donnelly, Chang Huang, Sahar Abdelnabi, Ferdinando Fioretto  

**Link**: [PDF](https://arxiv.org/pdf/2609.35596)  

**Abstract**: Self-evolving LLM agents have gained prominence for their ability to improve after deployment by modifying their harness, including their controller instructions, memory management protocols, and reusable tools and skills, in response to user and environment feedback. However, locally useful updates may persist into later tasks where they produce unsafe behavior, even without direct adversarial influence. To study this risk, we introduce SEABench, a benchmark for studying endogenous misalignment arising from agent self-evolution, with 48 longitudinal task sequences that span multiple evolution surfaces, task domains, and harm types in a rich personal-assistant environment. To account for the stochasticity inherent in agentic operations, we provide an adaptive trajectory discovery pipeline that probes for failures while preserving original task intent and supports causal attribution through paired non-evolving agents and attribution scores. Our evaluation across multiple recent LLMs, evolution surfaces, and harm types reveals that self-evolution indeed increases task completion rates but often at the cost of safety failures that are absent for paired non-evolving baseline agents. We also show that qualitatively different safety behaviors emerge across evolution surfaces and harm types. Further, we show that this divergence in safety behavior is reflected in agents' chain-of-thought reasoning, which yields an effective monitoring strategy that can mitigate unsafe behavior with a low false positive rate. 

---
# Share-Borne AI Virus: Memory-Hopping Attacks Across LLM Agents 

**Authors**: Sidharth Pulipaka, Ansh Sharma, Stanislau Hlebik, Leonidas Raghav, Vyas Raina, Ivaxi Sheth, Mario Fritz  

**Link**: [PDF](https://arxiv.org/pdf/2609.35576)  

**Abstract**: Large language models are increasingly deployed as stateful assistants that retain information across interactions and use tools to read, modify, and create persistent artifacts. As these artifacts are shared between users, they form an indirect communication channel between otherwise independent assistants. We study a failure mode in which this channel enables self-propagating attacks. We introduce artifact-mediated propagation, where adversarial content introduced through an artifact (e.g. a report), is stored in an assistant's persistent memory, reproduced in a subsequently created artifact, and acquired by another assistant that later reads it. We evaluate this process in temporal human-agent universes that model artifact exchange between independently operated assistants over time, measuring whether an attack survives successive hand-offs, how many hops it reaches, and how broadly it spreads. We find that attacks can propagate across multiple independent assistants and persist over extended interaction sequences. In larger simulated environments, even GPT-5.6 Luna exhibits substantial spread, reaching 60-80% of agents with propagation chains extending to eight hops. These results show that persistent artifacts can act as durable carriers of adversarial state, allowing attacks to outlive individual interactions and spread across isolated assistants. 

---
# Representation Alignment as a Bottleneck in LLM-Based Retrosynthesis Planning 

**Authors**: Hyunwoo Yoo, Cassie Huang, Haebin Shin, Li Zhang, Gail L. Rosen  

**Link**: [PDF](https://arxiv.org/pdf/2609.35571)  

**Abstract**: While LLMs show promise in general reasoning, symbolic planning in chemistry remains a bottleneck. Direct ''SMILES-to-PDDL'' attempts fail because they force models to juggle chemical analysis and planning-language structuring simultaneously. We hypothesize that this failure stems from a lack of intermediate abstractions rather than insufficient model capacity. By decomposing retrosynthesis into molecule mapping, reaction mapping, and PDDL generation, we achieve high success rates where end-to-end approaches fail. This provides evidence that a primary bottleneck lies in representation alignment rather than raw model capacity. Our structural analysis demonstrates that intermediate representations are essential in retrosynthesis planning, highlighting the importance of representation-centric design in future systems. 

---
# CLIMB: A Clinical Multimorbidity Benchmark for Diagnosing Co-occurring Conditions through Multiturn Conversations 

**Authors**: Yusuf Kesmen, Aniruddha Mukherjee, Yena Chang, David Sasu, Trevor Brokowski, Alexandra V. Kulinkina, Kristina Keitel, Akhil Arora, Lars Henning Klein, Mary-Anne Hartley  

**Link**: [PDF](https://arxiv.org/pdf/2609.35462)  

**Abstract**: Patients often have several co-occurring clinical conditions, and the findings needed to identify and disambiguate them emerge over the course of a consultation. Evaluating clinical reasoning in this setting requires both multi-turn interaction and multi-label diagnosis. We introduce CLIMB, a benchmark in which a doctor model interviews a simulated patient to recover a ground truth set of co-occurring clinical conditions. Cases are synthesized from clinical decision algorithms and diagnostic datasets, grounding multimorbid presentations in structured clinical knowledge. Across six frontier and open models, none recovers the exact set of conditions in more than 10% of interactive cases. Diagnostic performance declines when conditions co-occur, even when models receive the full clinical record and the true number of conditions. Interaction reduces performance further. In controlled experiments, models behave like single-hypothesis trackers: they anchor on the diagnosis suggested by the opening findings, keep questioning around it, and recover a second condition mainly when a finding in view points to it. Questioning them further does not complete the set but adds mostly wrong diagnoses. We formalise this pattern with a theoretical reference model of single-hypothesis tracking. The benchmark, generator, and evaluation code are available at this https URL. 

---
# LLMs are General Asynchronous Agents 

**Authors**: George Yakushev, Denis Mazur, Vladimir Bartenev, Vyacheslav Zhdanovskiy, Timofey Byzov, Vladimir Kaurkin, Vadim Pastushenko  

**Link**: [PDF](https://arxiv.org/pdf/2609.35427)  

**Abstract**: Modern LLMs are increasingly capable as autonomous agents, but they follow sequential interaction cycles: read, think, reply or call tools, repeat. Many real-world use cases are not sequential: voice assistants, embodied agents, and monitoring systems receive new inputs while they think or perform another task. Modern LLMs address this with specialized architectures for voice interaction and video streams, VLAs for robot control, asynchronous tool calling for API usage, and others. In this work, we generalize from different asynchronous tasks to general asynchronous agents that can adapt to different types of concurrency. To achieve this, we develop an asynchronous LLM framework that lets users (or the agents themselves) define inference coroutines with overlapping memory states. We showcase that Qwen 3.x models are capable of asynchronous operation for streaming video understanding, videogames, and monitoring, without task-specific training. 

---
# Frontier Learning: Training LLM Reasoners at the Edge of Capability 

**Authors**: Robin Faro, Shyam Sundhar Ramesh, Ilija Bogunovic, Aurelien Lucchi  

**Link**: [PDF](https://arxiv.org/pdf/2609.35426)  

**Abstract**: Reinforcement Learning-based post-training of Large Language Models (LLM) has been successfully applied to improve their reasoning capabilities. Existing pipelines primarily finetune LLMs on a fixed pool of problems specified prior to training using the GRPO loss. This is fundamentally limiting, as learning signal arises only when policy rollouts mix successes and failures, causing the useful portion of any fixed pool to quickly become stale as the model improves. To address this, we propose frontier learning, an open-ended post-training approach in which procedural generators are used online to continually produce informative training problems. It treats the generator's task-specific parameters as a search space and uses a regret signal to prioritize and explore frontier difficulty levels in order to focus training at the edge of the model's evolving reasoning capabilities. Across several reasoning tasks and model families, our approach consistently achieves higher relative gains over fixed-pool baselines, demonstrating that effective post-training requires not only selecting useful problems, but continually generating them at the edge of capability. 

---
# Semantic Prefix Oracles for LLM Decoding: Contracts and Differential Validation 

**Authors**: Paul Kronlund-Drouault  

**Link**: [PDF](https://arxiv.org/pdf/2609.35425)  

**Abstract**: Constrained decoding can enforce regular or context-free output formats, but many program-generation failures are semantic: scope, typing, and declaration effects depend on context. We present semantic grammar specifications, a declarative formalism that attaches such constraints to a context-free surface and executes them during Earley descent. Our implementation enforces \emph{safe pruning}: it rejects only prefixes whose semantic contradictions cannot be repaired by any continuation. A separate, grammar-dependent, \emph{dead-end freedom} property guarantees the existence of a realizable witness for each remaining branch. We give simple sufficient conditions based on surface productivity, type coverage, and left-to-right constraint flow. Our finite-lambda, core ML, and C-like fragments satisfy them, while the STLC instance used in our experiments does not: plain STLC can violate type coverage, and we show how restricting its type universe recovers it. A tokenizer-lifting lemma carries character-level witnesses to token sequences under an explicit vocabulary-coverage hypothesis.
We validate the implementation differentially against production compilers (\texttt{ocamlc}, \texttt{cc}). Across every prefix of 65 compiler-valid programs we observe zero false prunes. The semantic oracle localizes 25/30 invalid programs mid-stream, against 0/30 for a syntax-only oracle, and agrees on 42/42 recursion probes. A twelve-model generation study, including a matched semantic-versus-syntactic ablation for nine models, finds nonnegative observed semantic-minus-syntactic point estimates for every model-language pair, with maxima of $+15.2$ points on STLC task correctness and $+14.3$ points on ML validity. 

---
# Self-Adapting Group of Experts for Multi-Agent Reasoning 

**Authors**: Mohammad Atif Quamar, Nurbek Tastan, Karthik Nandakumar, Junpei Komiyama  

**Link**: [PDF](https://arxiv.org/pdf/2609.35412)  

**Abstract**: Multi-agent systems bring together language model agents with different roles to propose, review, and refine solutions. Each agent's response depends on its model's capabilities, the reasoning strategy defined by its system prompt, and the information in its input context. Existing frameworks often adapt communication by changing this context while leaving individual prompts fixed, even when a problem calls for different skills. We study whether agents' initial responses can identify a strategy better suited to the current problem and guide its transfer to other agents. To address this, we introduce SAGE (Self-Adapting Group of Experts), a training-free framework that uses answer agreement, prefix consistency, and reciprocal peer review to select a strategy donor. SAGE transfers the selected donor's reasoning strategy to the other agents while preserving their original roles. This transfer uses only the agents' original system prompts, without access to the problem or generated solutions. After strategy adaptation, agents exchange responses through a dynamic, sparse directed acyclic graph that routes information from higher-scoring agents to lower-scoring agents. Experiments across multiple agent backbones and reasoning benchmarks show that SAGE achieves higher average accuracy than the evaluated baselines. Our code is available at this https URL. 

---
# Do Coding Agents Reuse Existing Code or Reinvent the Wheel? 

**Authors**: Dongsheng Ma, Sizhe Wang, Xinyi Huang, Zhengren Wang, Yuhan Wang, Luyang Si, Xincheng Wei, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.35357)  

**Abstract**: Coding agents are increasingly deployed for iterative development on real repositories, yet existing evaluation barely answers a basic question: \emph{do coding agents reuse existing code or reinvent the wheel?} The question matters: every duplicated implementation is a fix applied twice and agents produce code far faster than humans can audit, so redundancy accumulates unsupervised. Thus, we present \textbf{RepoReuse}, a multi-turn benchmark for auditing code reuse in real repositories, where requirements are revealed turn by turn and the workspace accumulates across turns. It is built by a fully automated pipeline combining AST-based dependency graphs, guided evidence collection, and execution-verified task synthesis, and scales readily to new repositories. Beyond pass rates, we measure the reuse rate together with recall and cross-turn structural redundancy. An audit over 3{,}000 turns shows that agents progressively stop exploring relevant repository code, reuse their own history less even when it is fully in the workspace, and leave duplicated logic in 50.8\% of task chains by turn~5---all while pass rates barely move. Such deficiencies are invisible to pass rates, underscoring the need to evaluate code generation beyond functional correctness. 

---
# Jailbreaks for Black-Box Uncertainty Quantification in Large Reasoning Models 

**Authors**: Lucas Biechy, Cédric Eichler, Adrien Boiret, Nicolas Anciaux  

**Link**: [PDF](https://arxiv.org/pdf/2609.35350)  

**Abstract**: While Large Reasoning Models (LRMs) excel at complex reasoning, alignment through reinforcement learning often induces systemic overconfidence. In production environments, where logits may be unavailable, robust black-box uncertainty quantification (UQ) is essential for trustworthiness and safety. Focusing on question-answering for LRMs, we show that existing black-box methods, such as paraphrase-based self-consistency and confidence verbalization, offer little to no improvement over simple repeated sampling, suggesting that alignment suppresses useful output variability. We introduce prompt-level relaxation operators that broaden the model's effective output distribution by approximating the effect of an optimal policy obtained with a stronger KL-regularization parameter, hence closer to the reference model. Theoretically, we demonstrate that relaxation improves calibration. We propose Jailbreak for Uncertainty (J4U), a jailbreak-derived technique for UQ that empirically reproduces the behavioral signatures predicted by our relaxation theory. Across 3 datasets and 4 LRMs, including a closed-source production model, J4U's improvement over repeated sampling achieves statistical significance in up to 6 times more LRM-dataset-metric settings than the strongest black-box UQ state-of-the-art baseline we evaluate, with average ECE reductions up to 5 times larger. These results provide a practical tool for UQ in black-box LRM deployment. 

---
# EvoIn: Bridging Evolution and Internalization for Agent Fine-Tuning 

**Authors**: Shihan Dou, Shaofan Liu, Zhonghang Lu, Jiahang Lin, Shichun Liu, Binghai Wang, Jiajie Jin, Guanting Dong, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.35290)  

**Abstract**: Recent work has explored improving agents by jointly evolving their harnesses and models, but often takes a ''potpourri'' approach that bundles together new tools, new decision-making procedures, and model adaptation to the evolved harness under a single notion of agent improvement. In this paper, we instead investigate how agents can improve their decision-making procedures. In particular, we propose EvoIn, an agent fine-tuning framework that bridges evolution and internalization. EvoIn first analyzes agent execution traces to evolve and validate new decision-making procedures by temporarily instantiating them in the harness. The validated procedures guide the agent to generate improved reasoning traces. These traces are then rewritten into self-contained reasoning traces, removing explicit references to harness instructions while expressing the induced decision logic as the model's own reasoning. Finally, EvoIn fine-tunes the model on the rewritten traces, internalizing these procedures so that the improved decision-making persists without the evolved harness at inference time. We evaluate EvoIn on diverse benchmarks and find that it consistently enables agents to learn stronger decision-making procedures, raising the pass rate by 10.9 points in-domain and by 9.2 points out-of-domain. Results further show that the internalized decision procedures generalize to unseen tasks. Case studies show that agents can learn to decide how to solve a task before solving it, for example by checking a document's length to choose between reading it in full and searching it. EvoIn is also broadly applicable, showing consistent improvements on another model family. 

---
# TANGO: Watermarking Masked Diffusion Language Models in Token Pairs 

**Authors**: Kasra Arabi, Nir Weinberger, Micah Goldblum, Niv Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2609.35224)  

**Abstract**: Masked-diffusion language models fill in masked positions in parallel and in no fixed order. Most practical text watermarks assume left-to-right generation. They key each token to the tokens before it, and in a diffusion model those tokens may still be masked. A fixed green list needs no such context, but it favors the same tokens at every position, so these tokens appear more often in watermarked text. An attacker who compares token frequencies in watermarked and unwatermarked text can recover the list and forge text that the provider's own detector accepts. We present TANGO, a watermark for masked-diffusion language models that keys each new token to a nearby token that is already unmasked. A secret key splits the vocabulary into color classes, and TANGO biases the new token toward a color determined by the key and the nearby token's color. The watermark is therefore embedded in pairs of tokens. Because the favored color changes from position to position, token frequencies stay much closer to those of unwatermarked text than under a fixed green list. Detection needs only the text and the key, and it does not assume any unmasking order. On two masked-diffusion models, TANGO detects nearly all unedited watermarked texts and most edited ones, and frequency attacks that forge the fixed green list fail against it. 

---
# 5W1H+Which: Context-Valid Semantic Indexing with Progressive Ontology Binding 

**Authors**: Yaxiao Liu, Pengbo Liu, Yiwen Liu, Yihua Guan, Jiaxing Song  

**Link**: [PDF](https://arxiv.org/pdf/2609.35184)  

**Abstract**: Transforming raw data into queryable knowledge requires both early extraction of reusable information and explicit types, relations, and applicability conditions for particular tasks. If indexing selects content too early around a single business schema, later tasks may be unable to use information that was omitted. If the index retains only open-ended text, however, rule-based reasoning lacks checkable premises. We propose 5W1H+Which, a semantic indexing design that separates content extraction from ontology binding. The 5W1H questions organize source-grounded content units; Which points to versioned ontology elements and records mapping relations, scope, and validation status. Time, location, system environment, and participant roles are not merely retrieval labels: together, they constrain the contexts in which facts, bindings, and rules apply. Unbound content remains searchable, while bound content enters a formal reasoning path only after premise checks. The method further distinguishes business valid time, system knowledge time, and operational traces, and uses dependency records to support binding revalidation and the maintenance of derived conclusions. A worked example of migration from an on-premises server to a cloud environment illustrates the different treatment of world-state changes, ontology-version changes, and changes in rule applicability. We formulate three groups of falsifiable hypotheses concerning cross-task evidence coverage, control of contextual misuse, and incremental update cost. The planned evaluation includes a strong typed fact-graph baseline with the same evidence, temporal information, and budget, to test whether benefits arise from 5W1H organization, deferred binding, or additional information and engineering effort. The contribution is a testable indexing mechanism, not a claim to a new universal ontology or a demonstrated performance advantage. 

---
# VEX-Bench: Benchmarking Verification Complexity of LLM-Generated Misinformation 

**Authors**: Hanxun Huang, Yutao Wu, Qizhou Wang, Silvia Montaña-Niño, Yige Li, Xiang Zheng, Elif Buse Doyuran, Phoebe Matich, Xiao Liu, Xingjun Ma, Sarah Erfani, Christopher Leckie  

**Link**: [PDF](https://arxiv.org/pdf/2609.35028)  

**Abstract**: Large language models (LLMs) have made misinformation inexpensive to produce but not to verify, creating a growing asymmetry in the information ecosystem. Under tight time, labor, and budget constraints, media organizations, platforms, and fact-checkers rely on screening to prioritize which content to verify. We introduce VEX-Bench, a unified benchmark for evaluating the verification complexity of LLM-generated misinformation, as perceived during screening, across models and generation methods. Verification complexity is assessed along multiple dimensions derived from journalistic and fact-checking practices, capturing checkability, harm potential, source credibility signals, imposter legitimacy, and expected verification effort. We define the VEX score as an integrated measure combining elicitation yield and verification complexity to quantify how generated content consumes limited verification capacity. We construct a benchmark spanning two misinformation categories, 6 high-stakes domains, and 60 real-world topics, and evaluate 7 frontier LLMs and 7 generation methods, yielding 5{,}880 articles. We employ an LLM-as-judge for scalable evaluation and validate it using content-analysis methodology, including ordinal Krippendorff $\alpha$ for inter-annotator reliability, complemented by fact-checking agents for verification. Our findings show that no single method dominates all dimensions, underscoring the need for multi-dimensional evaluation. LLMs can generate high-VEX misinformation at 3$\times$ to 169$\times$ lower cost than agent-based verification. Such content is often prioritized during screening, consuming scarce verification resources and introducing a systematic risk of misallocation in resource-constrained verification systems. The code is publicly available in our \href{this https URL}{GitHub repository}. 

---
# WebPageBench: Event-Level Verification and Controlled UI-Variant Generation for Web Agents 

**Authors**: Anton Emelyanov, Maria Tikhonova, Zaven Martirosian, Sergei Averkiev, Alena Fenogenova  

**Link**: [PDF](https://arxiv.org/pdf/2609.35026)  

**Abstract**: We present WebPageBench, an open framework for evaluating web agents in which every task is verified from the interface's own event log. Six instrumented mock sites with brand identifiers removed (a marketplace, a bookstore, a grocery service, rail ticketing, hotel search and a document cabinet) emit typed events with parameters as a user or an agent acts. A task declares the events it requires, and success is decided by matching them, with no judge model and no scraping of rendered pages. The same instrumentation supports controlled UI variation: one configuration switch re-renders a task through a different implementation of a single control while the prompt and the success conditions stay completely identical, so sensitivity to interface form can be measured under a fixed task specification. The WebPageBench release consists of three components: 152 tasks, divided into 65 canonical scenarios and 87 control variants across light/dark UI-modes; a common runner evaluated with six browser/DOM harness configurations and five screenshot-only GUI-agent families; and a public leaderboard of 24 model-harness pairs. On the public 152-task leaderboard the gap between what agents declare finished and what the log confirms reaches 41 points (one configuration declares every task finished and satisfies the conditions on 59%). 

---
# ORPG: Reconciling Multiple Reward Objectives through Objective-wise Policy Gradients 

**Authors**: Shicheng Fang, Yiwen Zhao, Wenbo Tian, Jiahao Lu, Yining Zheng, Yuxin Wang, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2609.34985)  

**Abstract**: Multi-reward policy optimization requires a joint update that reflects both the learning signals and the intended relationships among objectives. We introduce Objective-wise Reconciled Policy Gradient (ORPG), which constructs a separate clipped policy objective for each reward and reconciles the resulting gradients into one policy update. For compatible gradients, a cosine-dependent interpolation coordinates their contributions through a partially normalized reference while preserving the norm of their sum. We characterize this update as the unique solution of a spherical directional compromise. For conflicting gradients, projection follows the task's priorities. We evaluate the same compatible rule in helpfulness--safety alignment and correctness--cost optimization for mathematical reasoning. ORPG substantially improves average Useful and Harmless scores over the strongest external baseline on each axis. In mathematics, it achieves the highest average full-budget accuracy and three-budget hypervolume among the compared methods, with more accurate and shorter responses than the initial policy. Component comparisons and training dynamics show the larger contribution of compatible coordination and a complementary benefit from conflict handling. These results support gradient reconciliation for objectives with equal standing and for objectives with an explicit priority. 

---
# See it, Say it, Sorted: Mechanistic Diagnosis and Parameter-Space Mitigation of Emergent Misalignment in LLMs 

**Authors**: Weiqiao Que, Ruizhe Li, Chengyu Wang, Dakan Wang, Emine Yilmaz, Xiaofeng He  

**Link**: [PDF](https://arxiv.org/pdf/2609.34970)  

**Abstract**: Safety-aligned LLMs can exhibit emergent misalignment (EM): narrow domain adaptation unexpectedly triggers catastrophic safety failures across unrelated domains. Prior static analyses leave training dynamics unmapped, while existing defenses rely on heuristics that degrade utility. We present a dynamic, second-order geometric study of EM. Tracking training trajectories reveals that directional Hessian curvature concentrates sharply on semantic pivot tokens. Grassmannian projections show that, in most settings, harmful-safe gap widens mainly because safe-gradient overlap declines. Leveraging these insights, we introduce a parameter-level Geometric Mitigation Framework that orthogonally projects empirical harmful gradient subspace out of parameter updates. On Qwen2.5-14B-IT, our defense suppresses free-generation EM by up to 80.0%; across the other three of four open-weight instruction-based model families (3B--20B), where single-layer behavioral EM is already near zero, teacher-forced evaluation shows same harmful subspace controls the conditional support of frozen EM responses. Crucially, these diagnostics unmask the illusion of behavioral safety: the same subspace remains measurable and steerable in models where behavioral EM is near zero. Code: this https URL. 

---
# Don't Forget! Decomposing the Training Dynamics of Memorization in Language Models 

**Authors**: Florian Eichin, Philipp Mondorf, Andrei Mircea, Yupei Du, Barbara Plank, Michael A. Hedderich  

**Link**: [PDF](https://arxiv.org/pdf/2609.34933)  

**Abstract**: Memorization has been proposed as a mechanism to explain how language models fit the tail of their training distributions, but its training dynamics are not understood well. In this work, we take a fine-grained look at memorization by decomposing the loss trajectory of memorized sequences over training and model parameters. Across the Pythia family, we study memorization of duplicated training sequences (recitation) and rare ones (recollection). We find that memorization in both cases is characterized by sequence-level gradient alignment, though recitation suffers from misalignment with other training influences which causes forgetting, explaining the necessity for higher duplication of these examples. We further show that the lower model layers are the most involved in memorization and forgetting. Predicting memorization, our decomposition improves over a cross-entropy baseline, especially in larger models and early in training. Intervening on a small set of highly influential parameters we are able to ablate memorization in the final model. Together, these findings advance our understanding of how memorization develops during training and offer insights for predicting and intervening on it. 

---
# Sample What You Say: Aligning Language Models to Sample the Distributions They State 

**Authors**: Kasra Arabi, Virginia Smith, Chhavi Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2609.34929)  

**Abstract**: Language models are increasingly used to sample from a specified distribution, for instance, to simulate survey respondents or generate synthetic data. Instruction-tuned models can state such a distribution correctly and still fail to sample from it. Prompting and changes to decoding reduce this mismatch only partly, which motivates training with policy optimization. Group relative policy optimization (GRPO) is a natural fit for this problem because it already samples a group of rollouts per prompt, and the group's empirical distribution can be compared with the target. However, scoring the group as a whole gives every rollout the same reward. Group-relative centering then sets all advantages to zero, and the model receives no learning signal. To give each rollout its own signal, we introduce the witness advantage, a per-rollout advantage derived from maximum mean discrepancy (MMD). It trains a model to match a target distribution over a finite set of outcomes. The MMD between the model's distribution and the target has a witness function that measures how over- or under-produced each outcome is. Each rollout's advantage estimates the negative witness at its outcome, so a rollout is rewarded for an outcome the group under-produces and penalized for one it over-produces. The witness advantage is computed in closed form from the group's outcome counts, and we use it as the reward in GRPO. On unseen target distributions, training with the witness advantage substantially reduces the total variation distance to the target while largely preserving the model's general capabilities. 

---
# ColNanoVDR: Document-Free Query Distillation for Multi-Vector Visual Document Retrieval via Optimal Transport 

**Authors**: Zhuchenyang Liu, Ziyi Wang, Yao Zhang, Yu Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2609.34899)  

**Abstract**: Multi-vector retrievers built on vision-language models lead visual document retrieval (VDR), but they run a multi-billion-parameter query encoder on every search. Distilling this encoder into a small student that queries the teacher's existing index would remove the bottleneck. The standard recipe, however, matches the teacher's MaxSim scores and so requires encoding and caching every training page, which can reach terabytes of page tokens. NanoVDR avoids pages entirely by training on the teacher's query embeddings alone, but only for single-vector retrievers. We present ColNanoVDR, to our knowledge the first framework to bring this document-free distillation to multi-vector VDR. Its objective, OTW (Optimal Transport with Learned Weights), aligns the student's query tokens with the teacher's by entropic optimal transport, with a learned weight for each student token, and needs no correspondence between the two tokenizations. We prove that the resulting alignment cost bounds the MaxSim score difference on every page. Distilled from five state-of-the-art teachers, the 149M text-only students retain about 95% of their teachers' NDCG@5 on ViDoRe v1-v3 while encoding queries up to 26x faster. Under identical training, OTW matches score distillation while encoding no page and reading 12.6x less cached teacher data. 

---
# One Readout, Many Repairs: Diffusion-Guided Hierarchical Search for Tool-Agent Repair 

**Authors**: Xiang Xia, Cheng Yan, Fan Xu, Zhijun Fan, Shuyuan Zhang, Wuyang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.34879)  

**Abstract**: Tool agents use large language models to act through external tools, yet successfully executed calls can still leave user requests unfulfilled. Tool-agent repair seeks alternative call sequences that execute successfully and fulfill the original requests. However, repair requires exploring both operation choices and their concrete realizations, making complete-sequence regeneration costly. Moreover, regeneration repeats operation selection even when failure arises from how those operations are realized. The resulting challenge is to reduce this repetition while preserving exploration of alternative operations and realizations. Therefore, we formulate repair as hierarchical search over operation supports, which we introduce as sets of permitted operation types that define reusable search regions for concrete tool-call sequences. We propose ReCommit, a training-free, diffusion-guided framework for improving tool-agent failure recovery while reducing repair computation. ReCommit amortizes operation-level proposal computation across repair trials by reusing operation-type scores from a single parallel readout of a masked diffusion language model. These scores guide search across supports, while realization search explores alternative entity bindings, arguments, and action composition within each support. Experiments on real failures across four enterprise services in the Agent-Diff benchmark show 75.9\% and 63.2\% relative recovery gains with 61.3\% and 51.3\% reductions in mean full-budget repair time at repair budgets $B=3$ and $B=13$, respectively, over the strongest evaluated 8B comparison method. ReCommit achieves a favorable recovery--cost trade-off, including in comparisons with the evaluated 32B models. 

---
# Beyond Verbalized Confidence: Calibrating Reasoners with Differentiable Readouts 

**Authors**: Chenxiao Fan, Chongming Gao, Gangyi Zhang, Leyang Shen, Yaxin Gong, Jiamin Wang, Jiakai Wang, Dong Wang, Yang Liu, Fuli Feng, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2609.34857)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) trains reasoning models to produce correct answers, but does not ensure that their stated confidence is calibrated. The resulting models are systematically overconfident. Recent methods train calibration inside the RLVR loop by having the model state a numerical confidence alongside its answer, but they all obtain the confidence by sampling it as text. This choice imposes two costs: a sampled confidence introduces variance and in practice collapses to a handful of distinct values, and sampling makes the confidence non-differentiable, forcing the calibration loss through a scalar reward. We propose CREDO (Confidence REaDOut) to replace sampling with a deterministic readout. While RLVR optimizes correctness, CREDO reads the confidence from a dedicated token pair in the model's output distribution and trains it by differentiable regression. CREDO further turns the trained confidence into a signal for accuracy, weighting rollouts by how far confidence and outcome disagree, so that accuracy and calibration improve together. Across mathematical and code reasoning, CREDO attains the best accuracy and calibration, and the gains extend to abstention and selective prediction. 

---
# DivOPD: Spread Wide, Look Close for Asynchronous On-Policy Distillation of Multi-turn Agents 

**Authors**: Hanyang Wang, Zeyuan Liu, Zhengyu Chen, Jingqing Ruan, Chaoxu Pang, Zhongda Su, Wulin Xie, Zhizhao Zeng, Ke Zeng, Tianxiang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2609.34838)  

**Abstract**: On-policy distillation (OPD) trains student agents through teacher supervision on their own interactions with an environment. However, in asynchronous multi-turn training, arrival-order batching can allow a few early or long rollouts to dominate learner updates while other valid rollouts become stale before being used, wasting already-generated experience. To address this problem, we introduce DivOPD, a simple learner-side batch-selection method that spreads a fixed turn budget across more rollouts and, within each rollout, prioritizes turns with larger cumulative teacher-student disagreement. Turns without usable teacher feedback are excluded. The per-turn loss and optimizer remain fixed; selection only changes which student-visited turns receive training weight. For no-progress rollouts, an optional extension briefly hands control to the teacher before returning it to the student. Across six teacher-student settings on the simulated ALFWorld, ScienceWorld, and WebShop benchmarks, with 1.5B-7B students, DivOPD raises cross-setting mean peak success rate from 77.4 to 84.4 and mean success over the last five evaluations from 71.5 to 78.6. It reaches all reported setting-specific targets with geometric-mean speedups of 1.84x in training tokens and 1.87x in learner GPU time relative to vanilla OPD. Teacher intervention further raises this last-five mean to 82.4 while retaining about 1.7x learner-GPU speedup over vanilla OPD. Code will be released at this https URL. 

---
# BV Loss: Block Verification-Aware Loss for Block Diffusion Speculative Decoding 

**Authors**: Suyoung Kim, Jahyun Koo, Hyeonjin Kim, Inhyeok Bang, Seunghyun Lee, Hyunjae Oh, Baeseong Park, Dongsoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2609.34832)  

**Abstract**: Diffusion drafters accelerate speculative decoding by proposing multiple tokens in parallel. Despite recent advances in speculative decoding through sequence-level drafting and verification, existing training objectives remain largely designed around token-level verification. To address this mismatch, we introduce Block Verification-aware loss (BV loss), a training objective designed to maximize the expected acceptance length of a drafted sequence. BV loss is directly derived from the block verification acceptance rule, providing a principled connection between the drafter training objective and the inference-time verification mechanism at the sequence level. Across math, code, and chat benchmarks, BV loss increases the mean number of tokens accepted per verification call under block verification by 13.0--21.0\% over cross-entropy loss training for DFlash and DSpark with Qwen3-4B and Qwen3-8B without changing the inference procedure. BV loss also outperforms tokenwise acceptance objectives such as TV loss and LK loss, and its gains extend to token verification and greedy decoding. These results demonstrate the benefit of training block diffusion drafters with an objective aligned with sequence-level verification, rather than optimizing each token independently. 

---
# When Do Model Internals Help? Exploring the Role of Representation Engineering in LLM Safety 

**Authors**: Tianyi Guan, Jianhui Chen, Liangming Pan  

**Link**: [PDF](https://arxiv.org/pdf/2609.34771)  

**Abstract**: Reliable AI safeguards require both control mechanisms that reduce unsafe behavior and monitoring mechanisms that detect safety risks during model interactions. Established behavioral safeguards include alignment methods that optimize model outputs and text monitors that assess interaction text. Representation engineering instead reads or modifies internal model states, but the relative strengths of these approaches remain unclear because they are often evaluated under different settings. We present a matched evaluation across two tracks. For safety control, we compare DPO, a behavioral alignment method, with three representation steering methods across robustness, practicality, and granularity. DPO provides the strongest overall control and generally improves with increasing training data, although its safety can degrade after subsequent benign fine-tuning. Representation steering remains competitive primarily in low-data settings, particularly with high-quality contrastive data. For safety monitoring, we compare representation probes with fine-tuned and open-weight text monitors across full-response detection, early detection, and computational cost. Specialized text monitors achieve the strongest overall detection accuracy, while representation probes remain competitive at substantially lower marginal cost. Finally, monitor-guided interventions recover much of the safety lost by DPO after benign fine-tuning, with little additional over-refusal. Overall, representation engineering does not generally replace behavioral safeguards, but offers practical advantages under specific conditions and can provide complementary safety benefits. 

---
# SeLMRoute: Probabilistic Semantic Evidence for Large Language Model Routing 

**Authors**: Vasilis Perifanis, Nikolaos Pavlidis, Symeon Symeonidis  

**Link**: [PDF](https://arxiv.org/pdf/2609.34736)  

**Abstract**: Large language model (LLM) routing aims to select the most suitable model for each incoming query. Most existing routers learn this decision directly from query embeddings, model representations, preference data, or clusters of similar examples. Such approaches can be effective, yet the representation used for routing rarely states what a query actually requires. We introduce SeLMRoute, a routing framework that separates the extraction of candidate-independent semantic evidence from the learning of candidate performance and the application of deployment objectives. A decision model first evaluates a set of interpretable questions about the query, such as its reasoning requirements and use of external knowledge, with each judgment retained as a probability distribution. The resulting probabilistic semantic state is used by a lightweight supervised router to estimate candidate model performance. Routing objectives are applied after performance estimation, which allows the same semantic state to support performance-oriented and cost-aware decisions. On the LLMRouterBench (15 datasets, 20 candidate models, 11,481 queries), SeLMRoute achieves an average accuracy of $72.08\% \pm 0.45$, while grouped five-fold out-of-fold evaluation reaches $72.64\%$, compared with $69.23\%$ for the strongest fixed candidate. The representation achieves the highest mean performance among the evaluated semantic, dense, lexical, and domain-level representations. In a separate 13-model performance-cost setting, SeLMRoute improves performance in all five grouped splits, with a mean PerfGain of $2.66\%$. Our code is available at this https URL. 

---
# Quality Determines Direction, Length Shapes Magnitude: Length Control for Open-Ended Reinforcement Learning 

**Authors**: Zijun Weng, Zhongan Bi, Xuanang Gao, Xiaohui Hu, Shuangyong Song, Yongxiang Li, Kaidong Yu, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.34718)  

**Abstract**: Reinforcement learning (RL) changes not only what language models say, but also how much they say, often increasing response length at the cost of token efficiency. Controlling this length growth is particularly challenging in open-ended RL because (i) response length is entangled with quality, (ii) open-ended tasks lack a natural success boundary for deciding when efficiency should be prioritized, and (iii) dense, graded rewards often yield small within-group quality margins, making quality-induced advantages especially sensitive to reward-level length shaping, which can perturb their magnitudes and even reverse their signs. We therefore adopt an asymmetric principle: quality should determine the direction of reinforcement, while length should only shape its magnitude. We instantiate this principle with Quality-Gated Length Advantage Shaping (QGLAS), which first computes advantages from quality rewards alone, then adds bounded bonuses only to shorter positive-advantage responses, leaving all other advantages unchanged. The bonus strength is further adapted to within-group quality separation, allowing conciseness to matter more when quality-favored responses are similar and less when their quality differences are clear. Across different model families, open-ended benchmarks, and reward sources, QGLAS consistently achieves a stronger quality--length trade-off than representative baselines. At approximately 30% compression, QGLAS retains 98.4--102.0% of the macro-average quality gains achieved by quality-only RL over the base model, compared with 68.3--75.5% for these baselines at comparable compression. 

---
# When Can Attention Heads Be Statically Defined? 

**Authors**: Weixian Waylon Li, Yintao Tai, Marcio Fonseca, Shay B. Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2609.34650)  

**Abstract**: Some attention heads learn similar patterns across inputs. Reusing these patterns could reduce training cost by avoiding repeated query-key score computation and softmax. Through controlled pretraining comparisons, we identify Selective Attention Freezing (SAF), which selects heads with low attention-pattern variance and replaces their attention weights with fitted post-softmax means halfway through training. We represent these fixed patterns with absolute-position and relative-distance preferences, reducing storage from quadratic to linear in sequence length. A fused kernel reconstructs the patterns and executes ordinary-attention and replaced heads together. At matched training-token budgets, replacing 25% of attention heads gives 1.056x faster post-replacement optimiser updates at 124M parameters and 4K context, with a 0.77% perplexity increase. At 1B and 8K context, post-replacement updates are 1.068x faster on four GPUs including communication, with a 0.51% perplexity increase. The resulting models also accelerate long-input finetuning and causal prefill. After associative-recall adaptation, the 124M model with 25% replacement generalises to more key-value pairs at a fixed length better than ordinary attention and two pruning controls. 

---
# After the Fix: How Corrected Agent Histories Transfer to Related Tasks 

**Authors**: Yanfei Zhang, Xu Lin  

**Link**: [PDF](https://arxiv.org/pdf/2609.34603)  

**Abstract**: Does repairing an episode make its experience a better memory for the next task? We transfer the same failed source before and after accepted repair to a fixed target, alongside independent execution. Our 3,300 runs cover 100 ThinkingBox pairs and the same 100 APEX pairs with and without source-state inheritance, under eleven conditions. ThinkingBox's Full/Skill/Hybrid correction gains are 44/29/32 percentage points, with corrected performance 25/22/18 points above independence; inference weakens at the task-family level. Yet 12 of Full's 15-point larger correction gap over Skill come from worse uncorrected performance, not better corrected memory. Moreover, 22 of Full's 46 upward transitions restore observed baseline success. Neither APEX regime establishes comparable aggregate correction benefits. Action evidence connects workflow gains with reusable obligations and convention conflicts with source-local choices. Text APEX's accepted execution reaches 52% versus its summary's 40%, without robust global/group-level superiority or an estab- lished advantage over independence. Smaller handoffs reduce input but increase calls. The value of repairing experience is therefore distinct from the value of reusing it: memory updates require both a previous-version reference and a fresh-start reference. 

---
# Nudgeability: Reasoning Models Follow Confidence Signals Without Tracking Their Own Competence 

**Authors**: Rohit Saxena, Utkarsh Upadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2609.34572)  

**Abstract**: Reasoning language models that can call tools must decide during inference whether to answer unaided or delegate. Any self-reflection mechanism for this must answer three questions: where the reflective signal comes from (verbal reports, output distributions, hidden states, a separate predictor), how it is presented to the model (numerical prediction, confidence token, prompt injection), and whether it changes the model's subsequent action. We isolate the third question. At a fixed point in otherwise identical reasoning trajectories, we insert a single first-person sentence expressing either confidence or doubt; the model then continues reasoning and chooses whether to answer directly or call a tool. Comparing these counterfactual continuations measures the causal effect of the reflective signal on delegation. We call this behavioral response Nudgeability and measure it along two dimensions: sensitivity, how strongly confidence and doubt change delegation rates, and targeting, whether delegation increases for problems the model cannot solve unaided and decreases for those it can.
Across nine small-to-medium open-weight reasoning models from three families (Qwen, Gemma, and GLM) and two tasks, models are consistently sensitive: doubt increases delegation and confidence decreases it, with a median confidence-to-doubt swing of 20.6 percentage points, and 53 to 70 points for the larger provider-served models. This responsiveness is poorly targeted: a median 42% of induced flips are well-targeted, only a +2 percentage-point lift over a random-selection baseline. Confidence language is thus a strong control surface for delegation, but current models use it only weakly in accordance with their actual competence. Nudgeability offers a simple, post-training-free way to evaluate both sensitivity and targeting as endogenous self-reflection mechanisms mature. 

---
# Rethinking Latent Visual Reasoning: Grounding Latent Reasoning in Visual Evidence 

**Authors**: Xi Xiao, Tianchen Zhao, Youngeun Kim, Zhuowei Li, Linghan Xu, Jiaye Wu, Zheng Zhang, Xiang Xu, Xuanbai Chen, Farhan Tejani, Jakub Zablocki, Julia Xu, Yifan Xing  

**Link**: [PDF](https://arxiv.org/pdf/2609.34563)  

**Abstract**: Latent visual reasoning (LVR) enables multimodal large language models (MLLMs) to perform intermediate computation in continuous latent tokens rather than expressing every reasoning step in words. However, unlike textual CoT, latent reasoning is not directly observable, making it difficult to supervise what latent tokens learn. In this work, we first conduct a thorough analysis of latent-token behavior and identify a latent evidence-credit gap: latent tokens respond only weakly to image perturbations that alter the correct answer. We hypothesize that this issue stems from the lack of explicit supervision during GRPO training. These findings suggest that a final-answer reward provides too little guidance on what visual evidence to preserve or how credit should be assigned across latent tokens. To bridge this gap, we propose ReaLVR, which brings visual-evidence supervision to the model's own free-running latent trajectories. ReaLVR contrasts correct and model-generated wrong answers to determine where stronger supervision is needed, and relevant and mismatched visual evidence to specify what to preserve. Across three model families, ReaLVR consistently outperforms evaluated LVR baselines, achieving the highest five-task average of 63.7% on Qwen2.5-VL-7B. Crucially, we are the first to scale visual reasoning in latent space, showing that our framework continues to deliver robust improvements at frontier model scales up to 235B. Further analyses show more question-sensitive latent-token positions, stronger alignment with relevant visual regions, and greater fixed-context dependence on the most attended latent tokens. 

---
# ActionLens: Diagnosing Spatial-Temporal Binding Failures in Vision-Language Models 

**Authors**: Gueter Josmy Faure, Min-Hung Chen, Hao Ping Wang, Timothée Lardy, Hung-Ting Su, Winston H. Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2609.34547)  

**Abstract**: Video-capable vision-language models score above 80\% on popular benchmarks yet struggle with spatial-temporal binding: associating the right action with the right person at the right moment. We introduce ActionLens, a diagnostic benchmark of 6,701 multiple-choice video questions spanning five targeted diagnostics: transition detection, actor-specific identification, concurrent action binding, directed interaction reasoning, and gaze detection. Ground-truth answers are derived deterministically from 1.58 million per-second, per-person annotations. Fourteen rounds of human quality engineering raised answer clarity from 53% to above 90% human accuracy. Across 20 VLMs, the full-set leader scores 68.8%; on the human-reviewed subset, it scores 65.9% versus 91.0% for the pooled human reference. Gaze detection remains near chance against 89.6% human accuracy. On actor disambiguation, reference-interface controls show that relational descriptions recover 5.55--13.25 points over static coordinates, confirming a substantial numeric-parsing penalty; yet visual boxes still lead every model by 1.15--6.50 points, exposing a residual unboxed actor-resolution gap. A binding-trap analysis shows models systematically select the wrong actor's action. ActionLens provides diagnostic measurements of these distinct failure modes across model families and scales for direct comparison. We release all data, code, and evaluation scripts at this https URL 

---
# How to Tame a Multi-Headed Hydra? Adaptive Multi-Category Safety Steering for Large Language Models 

**Authors**: Chenxi Wang, Ruiyang Huang, Li Huang, Yifan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2609.34514)  

**Abstract**: As large language models (LLMs) become increasingly widespread, preventing unsafe responses to harmful prompts is essential for their safe deployment. Activation steering offers an approach to improving LLM safety by modifying internal activations during inference without updating model parameters. However, a single prompt can involve multiple harm categories, and steering toward safety in one category may leave harmful content from another unaddressed. Despite advances in adaptive steering, existing methods do not explicitly coordinate steering direction and strength when multiple harm categories co-occur within a single prompt. To address this problem, we propose CAM-Steer, a Category-Adaptive Multi-category Safety Steering framework. Specifically, it estimates the risk associated with each harm category by comparing the current hidden state with safe and unsafe prototypes. The estimated risks are then used to combine the safety directions for different harm categories into a single steering direction and to determine the strength of the intervention. Finally, it rotates the hidden state along the composed steering direction, with the rotation angle determined by the estimated risks, while preserving the hidden-state norm. Experiments across three LLM backbones and seven harm categories show that CAM-Steer outperforms the evaluated baselines in average defense success rate, including when categories co-occur. Further analyses support its component designs and informative risk scores, with negligible inference overhead. 

---
# Low-Confidence Remasking Traps Flexibility: Realizing Arbitrary-Order Potential for Diverse Rollouts in Diffusion LLMs 

**Authors**: Moongyu Jeon, Dongjae Jeon, Bumjun Kim, Mingyu Kim, Albert No  

**Link**: [PDF](https://arxiv.org/pdf/2609.34509)  

**Abstract**: Masked diffusion language models support arbitrary-order generation, suggesting a natural way to produce diverse outputs. However, recent work argues that this flexibility reduces diversity by delaying high-uncertainty tokens that can lead to different generation paths. We trace this diversity loss not to arbitrary-order generation itself, but largely to low-confidence remasking (LCR), a widely used decoding rule. At each step, LCR samples a token at every masked position but commits only the sampled token with the highest probability, filtering out the rest. We show that this mechanism can exponentially suppress lower-probability tokens as more positions compete, and observe the same suppression in LLaDA. In contrast, top-probability position selection (TPP), which has often been conflated with LCR under the shared label confidence-based decoding, avoids this diversity loss. TPP first selects the position whose most likely token has the highest probability, then samples directly from that position's distribution. Replacing LCR with TPP restores diversity and yields Pass@$k$ comparable to left-to-right decoding, suggesting that the reported diversity loss stems largely from LCR's filtering rather than from generating high-confidence positions first. To further exploit order flexibility, we introduce Entropy-Guided Initialization (EGI), which samples the first token at the highest-entropy position and then follows TPP. This simple modification further improves rollout diversity and solution coverage beyond left-to-right decoding, with gains extending to downstream policy optimization, highlighting the potential of arbitrary-order generation for diverse rollouts. 

---
# The Last Mile Is the File: OfficeEditBench for Preservation-Aware Office Editing 

**Authors**: Zhiwen Wu, Chengxu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2609.34469)  

**Abstract**: A small Office edit creates two obligations: propagate every required update and leave protected state untouched. Updating too little leaves dependencies inconsistent; updating too much changes content the user did not authorize. We introduce OfficeEditBench, a 170-task benchmark for change-scoped maintenance of spreadsheets, presentations, and documents. Task contracts specify required updates, protected state, native structures, and applicable interaction requirements. Across 510 archived task-system outcomes from WorkBuddy, Doubao, and Codex, we distinguish file delivery, target completion, and verifier-defined acceptance. Hard package-valid delivery ranges from 92% to 100%, yet no selected output satisfies the complete contract. Case analysis highlights why local correctness is insufficient: an updated value can lose its generating formula, a revised rule can fail to reach related conclusions, and a new deadline can omit a retained prerequisite. These mechanisms connect artifact-level checks to the continued maintainability of Office files. We analyze maintenance failures while distinguishing frozen automatic verdicts from human acceptability. OfficeEditBench provides a testbed for completing required changes while preserving the logic and scope of existing work. 

---
# LLMs as Adaptive Meta-Solvers: Strategy-Diverse RL for Industrial-Scale Optimization 

**Authors**: Shihao Zhang, Weiting Liu, Siyu Shao, Yitian Chen, Jianfeng Feng, Dongdong Ge, Yinyu Ye  

**Link**: [PDF](https://arxiv.org/pdf/2609.34427)  

**Abstract**: Scaling LLM-based optimization from textbook-scale instances to real-world, industrial tasks remains a critical open challenge. Existing approaches are predominantly evaluated on small, self-contained textual problems and often commit to a solver-integrated paradigm, limiting their ability to handle the scale and structural diversity of practical optimization workloads. In this work, we propose a practical framework for training open-source LLMs to tackle real-world, industrial-scale optimization. We first show empirically that solver-integrated reasoning, exact combinatorial algorithm, and heuristic search exhibit complementary strengths across different problem structures and scales. Motivated by this, we introduce Strategy-Diverse Reinforcement Learning (SDRL), which trains LLMs as adaptive optimization meta-solvers. SDRL leverages this complementarity through a correctness-gated hierarchical diversity reward that promotes robust exploration across varying strategies and within each strategy, effectively preventing premature strategy collapse. We further introduce a mixed-format training scheme that jointly supports both self-contained textual problems and file-grounded instances. Across comprehensive evaluations, our framework outperforms existing fine-tuned methods and frontier models including DeepSeek-V4-Pro and GPT-5.5, both on average across benchmarks and on industrial-scale optimization tasks. 

---
# Coding Agent Memory Post-training: Unlocking the Memory Potential of Pre-trained File Operations for Long-Horizon Tasks via Reinforcement Learning 

**Authors**: Lirui Luo, Kelong Mao, Heming Xia, Rongqing Li, Xinwei Yang, Luyu Chen, Kieran Wong, Yudong Guo, Xinrui Wang, Jiayin Zhu, Simiu Gu, Sulong Xu, Cong Fang  

**Link**: [PDF](https://arxiv.org/pdf/2609.34422)  

**Abstract**: Language-model agents increasingly tackle long-horizon tasks whose interaction histories exceed the model's active context. Recent work has begun to use reinforcement learning to make memory control part of the policy, often relying on predefined memory tools within domain-specific training environments of relatively short horizons. This setup ties learned memory behavior to environment-specific interfaces that lie outside the base model's pre-training and must be learned from scratch, so even after post-training, agents struggle to use memory in long-horizon tasks. To address these limitations, we introduce Coding Agent Memory Gym (CAMG), a suite of long-horizon agentic-RL environments spanning Shop, Coding, DeepResearch, and AutoResearch. Alongside each environment's native task interface, CAMG provides executable shell access and an episode-persistent workspace, enabling agents to create, revise, search, and reuse files as memory throughout an episode. We also introduce CAMG-RL, which trains a single policy jointly across all four environments with fully asynchronous PPO, learning this file-based memory behavior directly from downstream task reward, and we train CAMG-RL-4B and CAMG-RL-9B from Qwen3.5 models of matching size. On SWE-bench Verified and MLE-bench Lite, CAMG-RL-4B and CAMG-RL-9B are competitive with Qwen3.5-35B-A3B and Qwen3.5-122B-A10B, respectively. 

---
# FORGE: Form-Optimal Routing of Grounded Evidence for Frozen LLM Agents 

**Authors**: Xi Xiao, Yunbei Zhang, Chen Liu, Lin Zhao, Jialin Chen, Tianchen Zhao, Xiang Xu, Youngeun Kim, Tianyang Wang, Min Xu  

**Link**: [PDF](https://arxiv.org/pdf/2609.34358)  

**Abstract**: In agentic AI systems, frozen foundation models are increasingly deployed as closed-weight API endpoints, making downstream adaptation possible only through the inputs and inference procedures surrounding the model. As a result, for each input query, two coupled decisions largely determine both answer quality and token cost: what evidence to provide and how much reasoning budget to allocate. Fixed defaults along these axes are often suboptimal, misallocating support form or reasoning depth on roughly 80% of queries in our analysis. To address this challenge, we propose FORGE, a unified framework for adapting frozen models through per-query routing over a joint action space that spans both support form and thinking depth. Under an entropy-regularized, cost-aware utility objective, we derive a closed-form Boltzmann routing target and instantiate the policy as a lightweight 269K-parameter factorized router. The routing policy is trained around the frozen host, without any weight access, through a three-stage pipeline: offline arm enumeration, supervised Kullback-Leibler (KL) distillation from the Boltzmann target, and Group Relative Policy Optimization (GRPO) refinement with host feedback. Across 5 knowledge-intensive benchmarks and 8 frozen backbones ranging from 7B to 671B parameters, FORGE improves accuracy at 42-45% lower token cost on both main hosts, transfers zero-shot across hosts at lower token cost, and composes with intrinsic thinking budgets where available. 

---
# Commutator Memory: Sparse, Path-Local Reading and Steering in Language Models 

**Authors**: John Sweeney  

**Link**: [PDF](https://arxiv.org/pdf/2609.34348)  

**Abstract**: Gradient updates on different data generally do not commute: training a language model on two data sources in opposite orders gives different weights, even with the same data and total exposure. Loss or benchmark deltas show that the models differ, not where. We ask whether this path dependence leaves a parametric training-history memory: a weight component that flips sign when the two sources are swapped, is localized in output space, changes the held-out loss gap between the two orders under targeted interventions, and reveals which trained model came from which order. For one small SGD step of size $\eta$ on each of sources $A$ and $B$, the weight difference $\theta_{AB}-\theta_{BA}$ is, to leading order, $\eta^2 b_{AB}$, where $b_{AB}=H_Bg_A-H_Ag_B$ is the Lie bracket of the two gradient fields at the base model. We define commutator memory by projecting the bracket through the logits into one score per vocabulary token; the scores sum to the bracket's prediction of the gap. The scores are localized: on three models, the same readout of the measured $\theta_{AB}-\theta_{BA}$, or of a bracket from disjoint batches, shares 82-99% of the original top-20 tokens, versus 35-49% for norm-matched random directions. They are causally actionable: in Qwen-3-4B SFT, downweighting the ten tokens with the largest predicted share of the gap closes a median 32% of the measured gap, while frequency-matched tokens with near-zero scores have almost no effect. The weights themselves carry the component: projecting the difference between the two trained models onto $b_{AB}$ identifies which came from which order in 92% of cases across four LLMs (chance 50%). Controlled tests also cover matched-batch DPO, a frozen-rollout GRPO-style objective, and an AdamW endpoint check. The memory is defined per source pair, not per example, and its projection on $b_{AB}$ decays with further training. 

---
# Knowing When Thinking Is Not Enough: Teaching Small Reasoning Models to Reason Beyond Their Parametric Knowledge 

**Authors**: Chanuk Lee, Minki Kang, Sangwoo Park, Woongyeong Yeo, Jinheon Baek, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2609.34327)  

**Abstract**: Scaling test-time computation is a powerful way to improve language-model reasoning, and is particularly appealing for small reasoning models (sRMs) that are cheap to serve. However, is additional thinking always the right operation? By intervening at intermediate reasoning states across two model families and multiple scales, we find that self-refinement largely consolidates probability mass onto solutions already reachable from the current state, rather than making new ones reachable. These interventions reveal two failure regimes: execution bottlenecks, where the correct path is reachable and reflection can recover it, and knowledge bottlenecks, where relevant external information makes it reachable. Motivated by this distinction, we introduce FlyBy, a selective querying framework, and train 4B and 8B variants to reason first, diagnose what remains unresolved, and, at a knowledge bottleneck, query stronger models whose parametric knowledge extends beyond its own. Supervised fine-tuning bootstraps a multi-depth query action, and cost-aware reinforcement learning calibrates whether to query, what to ask, and how much to spend. On 1,158 hard problems across six benchmarks, FlyBy-4B achieves 45.96% pass@8, surpassing Qwen3-14B (41.64%) at 2.7 times lower serving cost, while also exceeding Qwen3-8B in pass@1 (16.85% vs. 15.31%). Scaling to FlyBy-8B further improves pass@8 to 51.81%. 

---
# PlaylistEval: Can Video-Language Judges Be Trusted at Day Scale and Beyond? 

**Authors**: Shayekh Bin Islam, Hwanjun Song  

**Link**: [PDF](https://arxiv.org/pdf/2609.34314)  

**Abstract**: Video-language models are increasingly used as judges of video understanding, both for evaluating model outputs and for training reward models. Whether their judgments remain reliable when the evidence is buried in day-long videos has yet to be established. Existing benchmarks cannot answer this. Their videos are typically only a few minutes long, many answer pairs can be separated from the transcript alone, and collecting human judgments does not scale to ultra-long videos. We introduce PlaylistEval, an agentic framework that builds video-language judge benchmarks over 100-hour playlist collection without human annotation. It automatically generates questions with paired answers whose differences are controlled by causal degradation, so that every pair demands retrieval across the collection. The resulting benchmark contains 630 pairs across seven domains spanning both static and dynamic knowledge, and on a stratified subset of 152 pairs it agrees with human judgments 93.0% of the time (IAA 0.781). Evaluating 17 omnimodal and multimodal models from eight families reveals that frontier judges reach only 75.4% pairwise accuracy, while open-source judge models perform far behind. We further show that both retrieval and final judgment depend on using multiple modalities, and that judge accuracy degrades as the playlist set grows. We release our pipeline, benchmark, and evaluation code at this https URL. 

---
# ControlScope: Workflow Revision and Reliability in LLM Agents 

**Authors**: Jingjie Ning, Xueqi Li, Yibo Kong, Dongting Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.34313)  

**Abstract**: How much of a running workflow should a language model agent revise? ControlScope compares continuing generated code, editing the next tool call's data arguments, and replacing the unfinished workflow from the same public execution state. The nested permissions separate available repairs from the actions an agent selects. We evaluate one-time and repeated reviews across filesystem tasks, ALFWorld, and AppWorld. Across two source programs per task and three reasoning-reviewer draws on 20 filesystem tasks, FULL completes 15-16 tasks versus 13 for KEEP; across four fast draws it completes 10-13 versus 13. Fresh student-record confirmation reproduces a batch-read repair. ALFWorld fast panels yield KEEP/ARG/FULL scores of 85/86/87 on 87 tasks across 52 scenes and 134/134/127 on 134 tasks across four scenes; reasoning on the 87-task cohort also yields 85/86/87 with substantial review cost. An AppWorld V1 official-test panel of 585 task instances from 195 scenario templates shows small net differences. Frozen replays expose viable agent-written replacements interrupted by later revision in two failed file-organization runs. An offline source-trajectory midpoint comparison shows later reviews completing an insufficient repair. Five-call protection saves 19.4% of logged model output and loses one success across 20 fresh source runs. An argument-only shortcut shows that the broader sampled policy can overlook a cheaper successful edit available in both operation sets. These outcomes tie repair access to actual choices and subsequent execution. 

---
# BIABench: Evaluating AI agents on real-world bioimage analysis tasks 

**Authors**: Zixuan Pan, Davide Panzeri, Lukas Johanns, Marilin Moor, Yu Zhou, Hedi Peterson, Yiyu Shi, Jianxu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.34274)  

**Abstract**: Artificial-intelligence (AI) agents hold promise for automating bioimage analysis, yet no benchmark evaluates whether they can carry out real-world analyses end to end. Such analyses are hard for agents because 2D images, 3D volumes and time-lapse sequences are often too large to read as context, so an agent must choose and run an analysis through code, specialized software and rendered views. Published studies make this capability testable, because each pairs raw images with a peer-reviewed result. We introduce BIABench, a benchmark of 16 tasks reconstructed from published biological studies that retain their scientific questions, imaging data and ground truth. The tasks span eleven analysis subtasks and modalities from H&E histology to single-molecule localization microscopy. Each submission receives an outcome score, which compares the output files with the ground truth using field-standard metrics, and a process score, in which a vision-language model judges method choice and quality control against an expert-written rubric. We evaluated general-purpose and biology-specific agents across several language models, with repeated runs of every task. Routine two-dimensional tasks were solved well, but on some tasks that added a third dimension or a time axis no agent scored above 0.19. Neither biological specialization, stronger models nor detailed expert instructions closed this gap. The agents were also unreliable, with scores varying more between repeated runs of one agent than between different agents, and without ground truth a correct run could not be told from a wrong one by its process score or by the time spent. Released openly with its data and code, BIABench provides a verifiable framework for evaluating, and eventually training, agents for reliable long-horizon bioimage analysis. 

---
# DreamingGoose: Staged Distillation from Autoregressive Transformers to Bidirectional Recurrent Diffusion Language Models 

**Authors**: Julian Boesch, Andrew Wee, Alexander Stranzl  

**Link**: [PDF](https://arxiv.org/pdf/2609.34253)  

**Abstract**: Pretrained autoregressive Transformers represent a large sunk investment in compute. Existing conversion methods reuse that investment by changing either the architecture (attention to recurrence) or the objective (next-token prediction to denoising), never both. We convert Qwen3 teachers at 1.7B and 8B into attention-free, bidirectional, gated-delta-rule diffusion students in three stages, so that each capability can be traced to the stage that kept or lost it. Language modeling transfers only partially and in-distribution; in-context retrieval does not transfer. On a multi-query recall probe where the teachers score 0.34-0.58, both converted students score 0.000, and diffusion pretraining alone does not restore retrieval. A retrieval curriculum in the final stage, which gradually lengthens the gap between a key-value table and the queries that address it, restores it only stochastically: on a fixed schedule, one seed in three learns to retrieve. Advancing the gap only while a running accuracy estimate stays above a threshold works for all three of those seeds, holds on real text, and carries unchanged to 8B, where two of three seeds succeed. The third had not learned within its fixed 16k-step budget: retrieval switches on abruptly at a seed-dependent step (6.5k and 11k in the other two), so a fixed budget can cut a late run off. One boundary survives every intervention: every model that learns retrieval scores 0.000 on tokens that never appeared in a retrieval episode, and an arm that resamples the key and value tokens every batch shows this is a coverage limit, not memorization of particular bindings. Separately, we convert a 7B code model into a 3:1 recurrent-attention block-diffusion hybrid over 85k steps and report two negative training results. 

---
# When Does Selection Replace Extraction? A Pre-Registered Test of Agent Memory with a Typed Decision Model 

**Authors**: Rishabh Sharma, Rishika Lall  

**Link**: [PDF](https://arxiv.org/pdf/2609.34227)  

**Abstract**: Does conversational memory need LLM-extracted facts, or is selecting the right raw turns enough? Published results disagree. Extraction-based systems report gains from distilled facts. Recent studies find raw history with good ranking does as well, but disagree about whether ranking matters. We ran a pre-registered study on held-out LoCoMo conversations and LongMemEval. At a tight budget on LoCoMo, raw turns selected by a single call to Jev, a typed decision model, are non-inferior to an LLM-extraction memory (one-sided 95% bound -3.0 points against a -5-point margin). Blind human grading narrows the margin but does not change the result. Raw turns cost 3,061 times less to write, and the result holds with a second answer model. Within this study, reranking's gain shrinks as the budget grows. It adds 17.4 points on LoCoMo and 9.1 on LongMemEval when three of 30 candidates are kept. At generous budgets it adds 1.5 and 1.1, and extraction systems are more accurate. This suggests why published results disagree. At matched context, Jev selects as accurately as an LLM reranker (non-inferiority bound -2.0) at a third of the latency, and more accurately than a multi-call graph traversal. Reranking lowers correct abstention. Plans, code and graded answers are released. 

---
# Loop Dropout: Regularizing Shared Updates in Looped Language Models 

**Authors**: Zirui Zhu, Hailun Xu, Xuanlei Zhao, Yong Liu, Yingxuan Ren, Kanchan Sarkar, Kun Xu, Yang You  

**Link**: [PDF](https://arxiv.org/pdf/2609.34218)  

**Abstract**: Looped language models separate computational depth from parameter count by repeatedly applying the same transformer block. Adapting these models requires a shared update that remains effective as hidden states evolve throughout the recurrent computation. Our empirical analysis reveals a pronounced late-loop bias in standard low-rank adaptation (LoRA): the shared update is more effective at later loop positions. This imbalance motivates training shared updates under varying combinations of their applications. Randomly omitting adapter applications alone, however, does not improve task performance; it reduces expected update strength during training while leaving inference unchanged. We introduce Loop Dropout, which couples stochastic masking of adapter applications with inverse-survival rescaling to preserve expected update strength and promote effective adaptation across loops. Extensive experiments demonstrate improved mathematical reasoning across model sizes, adapter ranks and training recipes, with benefits extending to general instruction tuning and code generation. Loop Dropout outperforms existing LoRA variants and adapter regularizers, while further analysis shows stronger early-loop adaptation. Every backbone loop remains active, and inference applies the adapter at all loops using standard LoRA without additional trainable parameters or inference computation. 

---
# Explainable and Generalisable LLM-based Cognitive Decline Detection with Spontaneous Speech 

**Authors**: Ziyun Cui, Wen Wu, Chuan Shi, Shuguang Yang, Xueying Gui, Yan Zheng, Qiong Yang, Haiyan Zhao, Wei-Qiang Zhang, Ji Wu, Yelei Li, Nan Li, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.34217)  

**Abstract**: Alzheimer's disease (AD) and mild cognitive impairment (MCI), which may precede AD, manifest early through subtle linguistic and acoustic alterations. Traditional diagnostics, however, are often resource-intensive and lack scalability for mass screening. To address these challenges, we introduce a novel bilingual speech large language model framework for automated, explainable cognitive screening. Unlike conventional pipelines that rely on error-prone automatic speech recognition, our system directly processes raw speech to learn joint acoustic-semantic representations, preserving critical prosodic cues often lost in transcription. Utilising our newly collected PUTH-AD dataset alongside multiple open-source corpora, we implemented a multi-task learning objective that simultaneously performs cognitive status classification and generates clinician-understandable natural language explanations. Our system achieved the highest average accuracy and AUROC across six dataset/task conditions, comparing three representative baselines. The system demonstrated cross-task transfer to held-out PUTH-AD task subsets, maintaining classification accuracy on an entirely unseen cognitive task without task-specific fine-tuning. Furthermore, clinician evaluation confirms that the generated explanations are both clinically relevant and largely consistent with the underlying speech evidence, supporting their potential utility in clinical interpretation. This study provides a scalable, objective, and explainable framework for speech-based cognitive screening, combining cognitive status classification with natural language explanations that clinicians can assess and verify, bridging the gap between advanced AI and clinical utility. 

---
# X-MoD: Practical Scaling Laws for Sparse-Depth Routing Beyond Mixture-of-Depths 

**Authors**: Bowen Dong, Yilong Fan, Tengyu Pan, Yike Zhang, Zhenyu Li, Zijian Zhang, Xuewei Li, Mei Yu, Jianyong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.34212)  

**Abstract**: Mixture-of-Depths (MoD) enables conditional computation across Transformer depth by routing only a subset of tokens through selected layers, but its original one-sparse--one-dense alternation tightly couples total capacity to active capacity and limits sparse-depth scaling. We introduce X-MoD, a scalable sparse-depth architecture that decouples token sparsity from anchor stride, allowing total parameter count to grow while keeping active-equivalent capacity nearly fixed. To make deep sparse routing trainable, X-MoD combines dense anchors with variance-scaled layer-wise gating and depth-wise token balancing. To make this regime analyzable and usable, we formulate sparse-depth routing as a conditional architecture-design problem: given compute, context length, and active-equivalent backbone size, how should the routing configuration be chosen? We develop a practical scaling-law framework by fitting X-MoD relative to FLOP-matched dense baselines, yielding an interpretable law that decomposes performance into sparse-capacity gain, sparse-context correction, and anchor-stride interaction. The law predicts validation loss across routing configurations and reveals how context length, model scale, and anchor stride shape sparse-depth performance. We validate the architecture and law through pretraining sweeps, held-out scaling-law prediction, ablations, downstream evaluations, and comparisons with Dense, MoD, and representative MoE baselines. 

---
# PainterBench: A Figural Divergent-Thinking Benchmark for Tool-Using Language Models 

**Authors**: Shane K.A. Dalumura Hettige, Jonas Oppenlaender  

**Link**: [PDF](https://arxiv.org/pdf/2609.34195)  

**Abstract**: Figural divergent thinking is the ability to develop a given shape fragment into an original drawing. In humans, this ability is assessed with incomplete-drawing tasks. We introduce PainterBench, a benchmark that ports the incomplete-drawing task to the agentic setting. The agent draws on a canvas through tool calls and observes the result after every turn. The canvas includes a starting shape which cannot be erased, and the agent's goal is to incorporate this shape into the most original drawing it can produce. The task is open-ended, and the agent itself decides when the drawing is finished. The benchmark tests incremental visual planning over a short horizon and the transfer of creative ability from pretraining to multi-turn tool use. We evaluate 14 multimodal language models from small to frontier scale. Across the primary study and six sensitivity analyses, we collect 2,700 drawings and crowdsource creativity and recognizability ratings for every drawing and for 300 human reference drawings. We also present ViDrA-adapted, an automated scorer that predicts human creativity ratings of agent drawings (r = 0.85 on random held-out test split). Figural divergent thinking varies widely across the 14 models, and GPT-6 Astra produces the most creative drawings. Relative to the human drawings, the agent drawings score higher in creativity but lower in recognizability. We release the final drawings, per-round canvas snapshots, tool call traces, stimulus bank, benchmark harness, crowdsourced ratings (N = 72,000), and ViDrA checkpoint. 

---
# RAGWarrant: Evidence-Preserving Governance for RAG Policy Promotion Under Quality, Cost, Latency, and Risk Constraints 

**Authors**: Richard Krueger, Lucas Krause, Zach Pocquette  

**Link**: [PDF](https://arxiv.org/pdf/2609.34179)  

**Abstract**: Retrieval-augmented generation systems are extensively instrumented with metrics, benchmarks, traces, and automated judges, but these tools do not decide whether a proposed policy change is safe to release. We present RAGWarrant, an open-source promotion-control framework that treats deployment as a constrained evidence decision rather than a leaderboard choice. RAGWarrant normalizes evaluator outputs and operational telemetry, applies predeclared quality and hard-risk gates, assigns evidence-class claim ceilings, preserves negative outcomes, and emits auditable PROMOTE, BLOCK, REJECT, or INCONCLUSIVE decisions. We evaluate the framework across T2-RAGBench, MultiHop-RAG, CRAG, HotpotQA, synthetic reproduction, and bounded local generative experiments. On HotpotQA, operational savings were blocked because answer quality fell beyond the declared margin. A bounded CRAG study selected a lower-cost quality-tied policy, but related generative gains were unstable and a held-out guardrail failed closed. We claim an auditable promotion-control abstraction, not optimizer superiority, human validation, or production readiness. The tagged artifact reproduces from a fresh clone, runs as a hardened Docker job, accepts external evaluator exports, and verifies artifact integrity. 

---
# Training and Inference Dynamics of PLDR-LLMs: Row-Map Collapse, Renormalization, and Predictive Reduction 

**Authors**: Burc Gokden  

**Link**: [PDF](https://arxiv.org/pdf/2609.34130)  

**Abstract**: This monograph develops a unified account of training and inference in Power Law Decoder Representation language models (PLDR-LLMs). Exact finite work identities decompose changes in the absolute energy of the row-centered learned map into parameter contributions, signed interactions, and numerical observation defects. Positive affine blocking retains restarts at the row-constant face, while the augmented AdamW state supplies the complete dynamical description.
Predictive renormalization acts on the complete conditional training law for a single pass over distinct corpus target blocks, retaining optimizer memory, remaining data, schedule, and numerical policy. Autonomous reductions require closure; approximate reductions carry successor and emission errors. Finite-population covariance, matched physical clocks, matrix fluxes, and signed temporal energy connect row dynamics to model-wide observations. Absolute row collapse, relative row concentration, operator stabilization, and predictive accuracy are distinguished.
Experiments reveal observer and optimizer dependence, reject the tested autonomous row-state candidates, and support finite conditional prediction and state-specific operator reduction. Independent single-pass families exhibit moving finite fluctuation regions without establishing a thermodynamic critical class. Conditional symmetry, head limits, covariance flows, and readout error budgets specify assumptions needed to transfer scaling laws to inference. The theory separates exact identities, conditional dynamical claims, and finite empirical findings, with proofs, selected formal checks, and compact numerical evidence. 

---
# Counterexamples to Local Reconstruction Gain as a Proxy for Final Fidelity in Residual Completion 

**Authors**: Yasuto Hoshi, Daisuke Miyashita, Jun Deguchi  

**Link**: [PDF](https://arxiv.org/pdf/2609.34063)  

**Abstract**: Residual completion augments query-aware sparse attention by estimating the contribution of tokens omitted from the exact sparse computation. We ask whether improving a layer's attention-output reconstruction on the same incoming Q/K/V and selected support necessarily improves the fidelity of the final model output. We study training-free RESA and learned Top-K+$\phi$ with frozen backbone language models. A prespecified single-layer screen yields two Qwen3-0.6B/Multi-LexSum interventions for which direct-runtime measurements show positive prespecified request-aggregate local reconstruction gain but worse final KL fidelity than the corresponding all-abstain Exact Top-K baseline on both discovery and prompt-token-disjoint holdout requests. Exact restoration at the same layer instead improves final fidelity, showing that the reversal is specific to approximate completion in these cases. In complementary multi-layer experiments, a task-independent local diagnostic often repairs the tested completion estimators, although the repaired models do not consistently outperform Exact Top-K. Together, these results show that better local reconstruction need not translate into better final-model fidelity. 

---
# Steering Language Model Goals with Value Transplant 

**Authors**: Pengcheng Jiang, Fabien Roger  

**Link**: [PDF](https://arxiv.org/pdf/2609.34056)  

**Abstract**: Reasoning models often act as if they pursue goals, but their efforts are not always directed toward what users intend, sometimes leading them to pursue unintended outcomes. Previous work has examined how models may internally track their progress toward their goals through a "value axis." We study whether changing such a signal can retarget the model's search toward a different goal. We test value transplant: at each token, we shift the host model's activation along a candidate value axis by the donor-host difference in value coordinates (multiplied by a large scalar), aiming to redirect the host toward the donor's goal. We study this intervention in Qwen3-8B and GPT-OSS-20B models fine-tuned into honest and cheating variants. We test several candidate value axes, including a self-rating axis constructed from activations preceding high versus low elicited self-ratings of progress. The intervention works in both directions, with an honest donor reducing test-gaming in a cheating host and a cheating donor increasing test-gaming in an honest host, showing that this signal can influence which strategy the model follows. On solvable coding tasks, transplant from an honest donor also improves the cheating host's hidden-test performance. Value transplant also works across model families, providing preliminary evidence for the intervention in a setting relevant to model control. 

---
# Where Activation Sparsity and KV-Cache Sparsity Cross in LLM Decoding 

**Authors**: Jungseob Lee, Seungyoon Lee, Seongtae Hong, Sugyeong Eo, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2609.33889)  

**Abstract**: At each step, decoding one sequence with a large language model rereads the projection weights, whose traffic is fixed, and the key-value (KV) cache, whose traffic grows with context. Activation sparsity trims the first term and KV-cache sparsity the second, yet their reported speedups are hard to compare because each depends on context length and on the dense attention kernel it is measured against. We derive a byte crossover, the context length at which the two savings are equal, together with ideal speedup bounds for each branch and for their composition, from model dimensions and keep ratios alone. We then time both branches and their composition from 2K to 128K tokens on two GPUs after a dense prefill of real text, with dense and sparse modes reading the cache through the same split-K attention kernel. The projection branch leads at short context and the KV branch at long context, with speedups that follow their byte bounds up to fixed kernel costs. Adding these costs, measured in separate sweeps, lets the byte account predict the measured crossings of three keep-ratio pairs, a second model, and a second GPU to within 4.1K tokens. Timing the dense baseline with masked instead of split-K attention inflates the apparent speedup of the same KV policy about fivefold. An attention-scored KV selection answers the same passkey and multi-key placements as dense decoding up to 127K tokens, whereas a KV window misses most of them. Under matched perplexity budgets, activation sparsity composed with this selection decodes 14 to 26% faster than the best single branch on both GPUs. Code is available at this https URL. 

---
# Population Physics, Population Problems: Safety and Emergence in LLM Societies 

**Authors**: Adrian de Wynter  

**Link**: [PDF](https://arxiv.org/pdf/2609.33871)  

**Abstract**: The collective behaviour of large language model (LLM) societies is not the sum of their individual outputs. It yields statistically distinct, sometimes-unpredictable phenomena, for which the tools we use to study single agents may not scale. Due to recent incidents involving autonomous agentic systems, however, understanding these systems is paramount. For that we introduce a framework for measuring self-organisation in LLM social systems and apply it to three such systems: a Schelling grid, a social network (Moltbook), and a Twitter-like misinformation simulation ('Rogue'). All three exhibit statistically significant self-organisation. Moreover, their relaxation dynamics vary with the environmental information available to the agents, with open-ended systems (Moltbook, Rogue) exhibiting sharp, phase-transition-like dynamics. Further results show that population-level pathologies can emerge even when the LLMs are safety-tuned or monitored, being primarily driven by the coordinated activity of a population subset. We also show when self-organisation does \textit{not} emerge under two additional scenarios (a commons dilemma, GovSim, and a LLM-as-a-judge deliberation scheme, ChatEval). We argue that measuring signatures of this kind offers a lightweight, agent-agnostic diagnostic layer for detecting coordinated collective behaviour in deployed multi-agent systems without relying on natural language or model versioning. 

---
# Program-Verified Self-Evolution for Vision-Language Models 

**Authors**: Ahmed Heakl, Sungik Choi, Moontae Lee, Salman Khan  

**Link**: [PDF](https://arxiv.org/pdf/2609.33855)  

**Abstract**: Self-evolving vision-language models train on questions they generate from unlabeled images. Since these questions have no gold answers, prior methods label them by majority vote over sampled answers or by a model judge. In a human evaluation, we find that 24\% of majority-vote labels and 18\% of model-judge labels produced during self-evolution are wrong. To address this problem, we present Verifiable QA Generation for Self-Evolving Models (VQS), which changes how the model judges answers. Instead of voting on an answer, the model parses each image into a structured record, such as a scene graph, a chart table, or a diagram graph. Fixed programs then write a question from the record and compute its answer. The model still acts as a visual checker, but it only confirms the individual facts the program reads, one short claim at a time. These claim-level checks select the parser's training targets, so the parser also improves without labels. Human raters find 94\% of VQS answers correct, against 76\% for majority voting. Across ten benchmarks, VQS improves Qwen3-VL by up to 3.18 points at the 2B, 4B, and 8B scales and outperforms the strongest self-evolving baseline at each. Gains keep growing over three training rounds, reaching 3.84 points at 2B. Code is released at this https URL 

---
# Rethinking Contextualization by Reinterpreting Attention Head Channels 

**Authors**: Hakaze Cho, Haolin Yang, Zhun Sun, Naoya Inoue, Benjamin Heinzerling, Kentaro Inui  

**Link**: [PDF](https://arxiv.org/pdf/2609.33851)  

**Abstract**: Contextualization, the core operation of language modeling, transmits information across words to build sentence-specific word representations. Prior works mainly study contextualization, focusing on individual words and attention heads as a growing discrete dictionary, lacking a global view of their general behavior. Therefore, we propose a general principle: Globally, we find and estimate that different words carry different amounts of information, and less-informative words tend to absorb more contextual information. Specifically, these low-information words do not absorb contextual words uniformly, and finer-grained selectivity enables more precise routing to promote information transmission between matched words. Moreover, to find what mechanism causes such processing, we reinterpret attention heads as channels gated by their singular vectors and find that: (1) these singular vectors point to the hidden states of more informative words, allowing such words to write their information to others more strongly to act as information sources, and vice versa; and (2) these singular vectors can be viewed equally as hidden state features, enabling automated interpretation of attention heads beyond prior heuristic head discovery, also embedding heads into a continuous space rather than treating them as discrete, independent dictionary entries. 

---
# Laya as a Typed Probabilistic Assessor: An Independent Reproduction and a Preregistered Study of Calibration and Selective Escalation 

**Authors**: Gowthamkumar Nandakishore  

**Link**: [PDF](https://arxiv.org/pdf/2609.33843)  

**Abstract**: The shipped Laya Typed-Decisions checkpoint, a 421M-parameter ModernBERT-large assessor that answers typed choice/noul/score questions over workflow state, is uniformly under-confident. The signed confidence-accuracy gap is $-0.214$, every occupied reliability bin's accuracy exceeds its confidence, and that sign uniformity collapses every binned ECE variant to the same value, $0.214$. The card frames the risk as over-confidence; the measured direction is the opposite, and the direction decides which way a confidence-gated cascade fails. A single disjointly fitted temperature ($T=0.469$, sharpening) removes most of the miscalibration (held-out ECE $0.204$ to $0.037$) and outperforms the shipped per-option-count table. The frozen selection rule instead chose isotonic regression, which overfit and failed its held-out NLL contrast on both tracks, so hypothesis H2 is not supported. Re-running the released checkpoint on its full official test split reproduces the card's headline accuracy ($0.767$ vs. $0.766$). The retrospective E1 reproduction preceded the analysis freeze; E2-E8 were prospectively preregistered, and 20 of 22 executed confirmatory tests reject under Benjamini-Hochberg FDR at $q=0.05$ (two descoped). The frozen gate beats random escalation but misses its 10% accepted-set error target on both tracks, an exploratory out-of-distribution probe finds no zero-shot transfer (accuracy $0.617$), and every score measures agreement with a synthetic teacher whose self-agreement ceiling ($0.735$) the specialist exceeds. Per-decision predictions, run manifests, and the frozen preregistration are in the ancillary files. The author has no affiliation with the model's publisher, the dataset's publisher, or TypeSafe. 

---
# ChemOPD: Multi-Teacher On-Policy Distillation for Multi-Task Chemical Reasoning 

**Authors**: Yaoyao Xu, Xinjian Zhao, Xiaozhuang Song, Xuemin Chen, Tianshu Yu  

**Link**: [PDF](https://arxiv.org/pdf/2609.33838)  

**Abstract**: Large language models are increasingly expected to support diverse chemical reasoning capabilities within a unified model. One approach is to develop specialized capabilities separately and consolidate them through multi-teacher on-policy distillation, but this raises two questions: how should specialization be organized, and how should specialist guidance be integrated? We introduce ChemOPD, which addresses both. We estimate task affinities from supervised fine-tuning gradients and solve a constrained mixed-integer program(MIP) to construct partially overlapping specialist groups. During distillation, we retain a generalist teacher trained on all tasks so that specialist guidance supplements rather than replaces its supervision. Our anchor-residual objective gradually increases the routed specialist's contribution on student-generated responses. On ChemCoTBench, affinity-guided specialization produces task-dependent gains over the generalist teacher and improves several capabilities beyond semantic task grouping. Yet stronger teacher-side performance does not automatically yield stronger students: with the same specialists and routes, anchor-residual OPD improves most reported metrics over specialist-only distillation and realizes a larger share of the available teacher gains. These results highlight specialization and capability integration as connected but distinct design problems in chemical reasoning. 

---
# Controlling Speaking Rate in Autoregressive TTS via Activation Steering 

**Authors**: Francesco Verdini, Antonis Asonitis, Aref Farhadipour, Marzieh Razavi, Pierre-Edouard Honnet, Vijeta Avijeet, Juan Pablo Zuluaga Gomez  

**Link**: [PDF](https://arxiv.org/pdf/2609.33810)  

**Abstract**: Autoregressive text-to-speech (TTS) systems synthesize natural speech but, once trained, offer little control over speaking rate. We show that speaking rate can be steered at inference time, without retraining, by clamping a single decoder block's activation along a discovered speed axis. A decoder-block analysis recovers the rate axis, a neutral operating point, and a per-step intensity scale; at inference, the activation's projection onto this axis is set to a fixed scalar. Learning this direction from synthetically time-stretched and time-compressed speech yields rate control that largely preserves speaker identity, generalizes across model architectures, and maintains high naturalness in objective and human evaluations. Unlike standard additive steering, which breaks at the slow extreme, clamping remains stable on all three systems tested; at moderate targets, the better rule depends on the model. Finally, we show that rate information is decodable across layers but causally steerable only within a mid-depth window, and demonstrate the effectiveness of our approach on the public Seed-TTS-Eval benchmark. 

---
# Surprising Success, Repeated Failure: Entropy-Guided Credit Assignment for Exploration in LLM Reasoning 

**Authors**: Woongyeong Yeo, Minki Kang, Chanuk Lee, Sangwoo Park, Jinheon Baek, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2609.33781)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) enhances reasoning in large language models (LLMs) through outcome-level feedback, yet recent approaches to finer-grained credit assignment often require auxiliary models, additional sampling, or privileged information. Although policy entropy provides a readily available signal, prioritizing uncertain positions under both reinforcement and penalization concentrates penalties where failed responses still retain alternatives for recovery, which can suppress opportunities for exploration. To address this, we introduce Entropic Advantage Policy Optimization (EAPO), an entropy-guided credit assignment method that treats success and failure asymmetrically. Specifically, motivated by the observation that success under uncertainty is less repeatable while confident failures tend to recur, EAPO couples normalized policy entropy with the sign of the response advantage to reinforce surprising success and correct repeated failure. It assigns stronger reinforcement to high-entropy decisions in successful responses and stronger penalties to low-entropy decisions in failed responses, while attenuating penalties at uncertain positions to preserve opportunities for recovery. By redistributing the response advantage across tokens, EAPO derives token-level credit directly from existing rollout signals without additional supervision. We validate EAPO on a range of reasoning tasks across both base and reasoning backbones, demonstrating that it achieves the best overall performance. We further show that EAPO promotes more effective exploration, broadening problem coverage and generating more diverse candidate answers. 

---
# Tokens Change, Structure Endures: Spectral Watermarking for Generated Speech 

**Authors**: Kanghwi Lee, Kyeongseok Jeong, Jeongmin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.33774)  

**Abstract**: Watermarking is a promising tool for establishing the provenance of AI-generated speech. While many neural audio watermarking methods rely on a separately trained watermark generator, token-level watermarking is a training-free alternative that operates directly during generation. Its main weakness is retokenization: decoding generated speech to a waveform and encoding it again can change token identities and erode the watermark. To make the watermark robust to these changes, we propose Redwing, REtokenization-Durable Watermarking IN Generation. It builds a graph from the token substitutions observed under retokenization, whose Laplacian yields a basis that assigns similar values to tokens likely to substitute for one another. Over this basis, embedding and detection functions are jointly optimized to preserve watermark signal through retokenization while limiting embedding distortion and detector variability on unwatermarked speech. On the Moshi full-duplex system, after eight consecutive passes of Mimi resynthesis, Redwing achieves 80.7% TPR at a calibrated 1% FPR, compared with 8.3% for KGW and at most 7.3% for WMAR. It also has the highest TPR after eight passes through three other neural codecs (77.5-93.0%), and the gains generalize to TTS models at a speech-quality cost close to that of KGW. These results show that retokenization is not merely a source of noise: its transition structure can be exploited as a design principle for robust token-level watermarking. 

---
# DuraS2ST: Chain-of-Thought and Reinforcement Learning for Duration-Aligned Speech-to-Speech Translation 

**Authors**: Yayue Deng, Dingdong Wang, Yuxuan Hu, Jinyu Li, Yanqing Liu, Yuanyuan Wang, Weidong Chen, Helen M. Meng, Shujie Liu, Xixin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2609.33742)  

**Abstract**: Speech-to-speech translation (S2ST) in time-sensitive applications such as video dubbing requires not only semantic fidelity and speaker preservation, but also strict duration consistency to avoid audio-visual misalignment. However, existing S2ST systems largely generate target speech without explicit temporal planning, making duration control an unresolved challenge. We introduce DuraS2ST, a duration-aligned reasoning framework that enables a single speech language model to first generate an explicit chain-of-thought (CoT) for planning target wording and phonetic length, and then synthesize the corresponding speech tokens. To support this paradigm, we construct DuraSet-440K, a high-quality duration-aligned CoT corpus for supervised initialization. We further optimize the model with multi-modal multi-dimensional reinforcement learning, using a Duration Margin Reward to balance translation quality and duration consistency, and Modality-Aware Reward Attribution to assign rewards to appropriate token spans. Experiments on CVSS-T show that DuraS2ST achieves a strong balance between translation quality and duration consistency, outperforming competitive open-source and commercial baselines. Project page: this https URL. 

---
# BOReFT: Manifold Steering of Language Models for Black-box Optimization 

**Authors**: Dhruv Agarwal, Rico Angell, Kavitha Srinivas, Tahira Naseem, Horst Samulowitz, Willie Neiswanger, Andrew McCallum  

**Link**: [PDF](https://arxiv.org/pdf/2609.33722)  

**Abstract**: Language models are increasingly used as proposal models for black-box search, from program optimization to molecular design. Existing approaches typically improve proposals through iterative prompting or parameter updates, offering limited control over how completely and efficiently the model's search space is explored. Continuous optimization methods, such as Bayesian optimization, provide a principled way to search but require a suitable domain to operate over. To address this, we introduce BOReFT, which learns a compact, low-dimensional space of hidden-state interventions in a frozen language model, and uses this space as the search domain for Bayesian optimization with an external scoring function. Empirically, we find that the learned domain spans semantic regions and exhibits smoothness properties that support search. Theoretically, we show that semantic coverage and interpolation control the best score available in the learned space, and that decoding from this space yields a standard stochastic-bandit observation model for adaptive search. We evaluate BOReFT on the interpretable word search task "Semantle" and on three more real-world discovery tasks in de novo molecule property optimization. Compared to strong LLM baselines, BOReFT finds in Semantle a higher number of hidden targets and, on two out of three molecular objectives, achieves higher property scores. Consequently, our method provides a principled new bridge between discrete proposal spaces of LLM-based search and continuous black-box optimization. 

---
# Auditing Agent Actions through Query-Conditioned Attribution 

**Authors**: Yifan Liu, Praveen Venkateswaran, Abdulhamid Adebayo, Dong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.33676)  

**Abstract**: LLM agents increasingly take consequential actions through interactions with users, policies, and external tools. Auditing these agents requires automated attribution of realized actions to their historical basis. However, existing attribution formulations do not provide question-specific traces for diverse auditing objectives. Additionally, when access to the acting model is limited (e.g., in API-only deployments), applicable methods commonly rely on costly input perturbations or external LLM analysis of complete trajectories. We therefore formulate $\textit{query-conditioned agent action attribution}, a new task that takes a natural-language auditing query as input and recovers the source and ordered intermediate evidence for the query-specified aspect of an action. We instantiate this task with $A^3Bench$, a benchmark comprising 1,396 auditing queries across policy basis, parameter provenance, failure propagation, and unsafe-behavior tracing. To enable efficient, query-specific attribution, we use small open-weight models as attribution proposers that combine query-conditioned gradient saliency with query-semantic relevance to rank history units. Our proposer consistently achieves stronger source and evidence rankings at lower inference cost than open-weight baselines, improving source MRR by up to 40.9\% and evidence MAP by 42.1\% with only two forward passes and one backward pass. Controlled evaluations confirm that our proposer improves attribution specificity by adapting its rankings to fine-grained changes in the auditing query. Building on a proposer ensemble, our end-to-end system surpasses the strongest frontier-model baseline in source accuracy (64.5\% vs.\ 60.4\%) while reducing empirical deployment latency by 29.9\% relative to the fastest frontier API baseline. Code and data will be released after the initial review period following final validation and cleanup. 

---
# Audit-First VAPO: Risk-Certified Selective Updates under Imperfect Verification 

**Authors**: Miaobo Hu, Shuhao Hu, Xiaobo Guo, Xin Wang, Bokun Wang, Rui Chen, Daren Zha, Jun Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2609.33662)  

**Abstract**: Imperfect verifiers can assign a harmful update direction even when clipping and regularization bound its magnitude. We introduce Audit-First VAPO, which separates discrete directional admission from continuous magnitude control. An observation-only accept-appeal-abstain policy uses a finite secondary-verification budget; its action trace is frozen before clean labels are joined. Simultaneous finite-sample bounds then certify selected harmful risk, coverage, and verifier-call rate over a predeclared policy family. Conditional Hoeffding-Azuma bounds account for the dependence induced by shared budgets, and rollout or verifier changes initiate a new certification stage. After admission, a bounded trust-clip-KL actuator controls magnitude. We evaluate two models on two reasoning benchmarks against static RLVR, matched-random selection, confidence thresholding, noise correction, and verifier augmentation. On Qwen3.5-0.8B and GSM8K at target risk $\rho=0.08$, RC-VAPO achieves 74.1% accuracy, selected harmful risk 0.0697, coverage 0.4125, and relative verifier cost $1.16\times$. At matched coverage and update magnitude, its selected-risk difference from matched random is -0.0260 with paired 95% interval $[-0.0364,-0.0157]$. Across asymmetric, confidence-dependent, and correlated-verifier noise, the certificate is satisfied on 57 of 60 independent runs. These comparisons isolate informative directional selection from proposal suppression, update shrinkage, and additional verifier computation. 

---
# Probe to Act: Elevating Browser-Use Agent via Active Visual Probing 

**Authors**: Keliang Li, Heng Wang, Chen Hu, Daxin Jiang, Hong Chang, Shiguang Shan  

**Link**: [PDF](https://arxiv.org/pdf/2609.33646)  

**Abstract**: Browser-use agents require seamless alignment between structured web metadata and visual information, while preserving relevant context across long interactions. Existing interfaces often rely on either screenshot-level action prediction or static Set-of-Marks overlays, leaving the model to resolve dense DOM-pixel alignment before every operation. We introduce Probe to Act (P2A), an active probing framework for the browser-agent loop that moves this alignment into decision time. P2A addresses an asymmetric bridge between symbolic DOM hypotheses and screenshot layout by rendering on-demand symbolic DOM structure back into pixels. Before committing a state-changing browser operation, the agent can issue lightweight probes to translate DOM handles into pixel evidence, map screen regions back to DOM candidates, register visual-only targets, and commit verified notes. These interleaved processes naturally produce evidence-based memory: only probed, acted-on, or explicitly committed observations are kept across steps, preserving only decision-critical evidence in long-horizon contexts. P2A can be used as a prompting strategy for proprietary models under the standard DOM+SoM interface, and can be distilled into open-weight models through cold-start synthesis and self-bootstrapped SFT. Across three browser-use benchmarks, P2A shows clear gains on task success rate for both proprietary and fine-tuned models; on VisualWebArena, for example, it improves Gemini-3-Pro from 54.1% to 61.2% and Qwen3-VL-8B from 24.6% to 32.9%, while matching the costly full-observation history ($\sim$3$\times$) at only $\sim$1.2$\times$ the peak retained input context of action-only history. 

---
# Quantifying Behavioral Tails in Black-Box Language Models 

**Authors**: Elsayed Eshra, Ali Al-Lawati, Dongwon Lee, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.33638)  

**Abstract**: We introduce RareTrap, a framework for estimating the probability of severe behaviors in black box large language models (LLMs). A key challenge for probability estimation is defining a tractable distribution over the input space. To accomplish that, RareTrap uses a surrogate LLM and constructs a geometry-aware mapping from a lower-dimensional latent reference space into its token-embedding space to induce an explicit and reproducible distribution over input prompts. A response-level performance function is utilized on the response to quantify behavior severity. This enables sequential rare event simulation that concentrates evaluations on progressively more severe behaviors while preserving probability under the induced prompt distribution, which would otherwise be prohibitive to measure. Across 10 open-weight and two frontier models (GPT-5.4 and Claude Sonnet 4.6), we find that RareTrap successfully induces severe resource consumption behaviors and computes their probability with as few as 200 evaluations. RareTrap provides model developers a principled approach for evaluating language models under a common distribution, and prioritizing alignment effort to improve safety and mitigate risks. 

---
# ParaAgent: Reinforcing Parallel Acting in Open-World Tool Environments 

**Authors**: Shengbin Yue, Hongru Wang, Siyuan Wang, Xiaoxin Chen, Wei Chen, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2609.33618)  

**Abstract**: Language model agents are increasingly deployed in open-world tool environments, which require balancing exploring unknown capabilities and exploiting known ones. Existing methods face a performance-efficiency tradeoff: they either rigidly decouple exploration and execution or interleave them without coordination. We argue that the key lies not in whether to decouple or interleave them, but in how to coordinate them across granularities. We introduce ParaAct, a structured parallel-action loop that combines phase-level Exploration $\rightleftharpoons$ Execution with action-level parallelism. To learn this loop, ParaAgent combines multi-agent cold-start demonstrations with reinforcement learning under multi-level advantage decoupling, making planning structure explicit and supervising it with step-, phase-, and trajectory-level rewards. Learning is supported by our ToolEnv, a scalable simulator grounded in 50,011 realistic tool interfaces. On two open-world tool benchmarks, ParaAgent-4B achieves the best average success among all baselines, including GPT-4.1 systems, with the largest gains on multi-tool tasks. Behavioral analyses show that these gains stem from this action organization, highlighting its importance for capable and efficient open-world agents. 

---
# TGRL: Temperature-Grouped Reinforcement Learning for Efficient Exploration in LLMs 

**Authors**: Zihan Lin, Xiaohan Wang, Jie Cao, Jiajun Chai, Wei Lin, Guojun Yin, Ran He  

**Link**: [PDF](https://arxiv.org/pdf/2609.33589)  

**Abstract**: Efficient exploration often remains a central bottleneck in reinforcement learning with verifiable rewards (RLVR). Although temperature control and test-time scaling strategies can increase rollout diversity of large language models (LLMs), they either expand the sample budget at rollout time or leave the benefit of exploration unquantified. To this end, we propose Temperature-Grouped Reinforcement Learning (TGRL), which turns temperature-induced diversity into an explicit training signal. For each prompt, TGRL partitions its rollout group into low- and high-temperature subsets, estimates exploration gain through their reward contrast, and allocates this group-level signal as token-level credit using Jensen--Shannon (JS) divergence between the corresponding temperature-scaled next-token distributions induced by the same logits. Notably, TGRL reaches equivalent accuracy up to 36% faster than strong RLVR baselines without expanding the rollout budget. Across 11 benchmarks from diverse domains, TGRL broadly improves over strong RLVR baselines: it improves the six-benchmark math average by 1.6% at 32B, raises CodeForces rating by 196.7 points and LiveCodeBench Pass@16 by 4.4%, and improves ALFWorld/WebShop success rates by 6.3%/4.9%. Comprehensive ablations and wall-clock analysis confirm the efficacy of all proposed components. Code is available at this https URL. 

---
# A Cheap Verifier is Good Enough: LLM Post-training is Robust to Erroneous Rewards 

**Authors**: Andreas Plesner, Curtis Northcutt, Francisco Guzmán, Anish Athalye  

**Link**: [PDF](https://arxiv.org/pdf/2609.33467)  

**Abstract**: When post-training large language models on tasks with semi-verifiable rewards, there are many factors (training steps, base model size, training order, data quality, verifier accuracy, etc.) that practitioners must contend with to maximize model performance. Yet, it remains unclear how well verifier agreement predicts post-training performance on such tasks. In this paper, we explore this question with over 11k H100 GPU-hours, across HealthBench and PRBench tasks in medical, legal, and finance domains. Across the tested domains, Qwen3 trainees (1.7B-8B on HealthBench; 8B on PRBench), evaluation splits, and frontier LLM reference judges (which we call golden verifiers), higher verifier agreement does not consistently identify the best training verifier. Expensive verifiers need not outperform inexpensive ones, and open-weight Gemma verifiers produce strong training outcomes. We compare two low-cost choices retrospectively -- a cost-reducing choice and a balanced choice -- with estimated grading cost reductions of 98.8%-99.7% relative to the golden grading protocols and average post-training score gaps of 1-3 points from the best evaluated training verifier. These averages include larger losses in individual settings; they do not establish that verifier choices are interchangeable. 

---
# Raven: The Harness of Harnesses for Composable Agentic Intelligence 

**Authors**: EverMind AI  

**Link**: [PDF](https://arxiv.org/pdf/2609.33439)  

**Abstract**: As large language models advance, AI agents are moving beyond isolated, domain-specific tasks toward long-horizon, cross-domain workflows. This transition exposes two challenges: increasing harness complexity makes manual design difficult to scale, while tighter coupling to specific domains limits the generality of a single harness. The central question thus shifts from how to engineer a stronger harness for one domain to how to autonomously construct specialized harnesses, improve them through experience, and orchestrate them across domains. We introduce Raven, \emph{The Harness of Harnesses}, an open-source multi-agent ecosystem that automatically constructs and evolves modular harnesses for specific models and domains, treating each executable model--harness pair as a composable unit of intelligence. To support an \emph{All-Domain Collaboration Network}, its Host Agent decomposes goals, matches subtasks to specialized agents, coordinates execution dependencies, and integrates results, while a host archive and EverOS preserve experience across tasks and Skill Forge makes that experience available as reusable procedures. Our theory establishes sufficient conditions for such composition to expand reliable task coverage beyond that of the available individual agents under a shared resource budget. On complex and long-horizon tasks, Raven significantly outperforms the state-of-the-art agent systems, pushing the frontier of composable agentic intelligence. 

---
# SMAT: Simple and Efficient Merge-Aware Training 

**Authors**: Yanggan Gu, Yuanyi Wang, Zhen Li, Shuo Cai, Yuhang Liu, Junzhuo Li, Zihao Wang, Hongxia Yang  

**Link**: [PDF](https://arxiv.org/pdf/2609.33437)  

**Abstract**: Model merging integrates the capabilities of multiple experts without joint retraining, but standard expert training optimizes task loss alone and does not guarantee good performance after merging. Merge-aware training (MAT) aims to improve merged performance, but existing methods do not fully account for common merging operations and add training cost. We observe that, from an expert's perspective, common merging methods can be described by three operations: Scale reweights its own update, Mask removes selected coordinates, and Perturb adds updates from other experts. Based on this view, we introduce SMAT (Simple MAT), which jointly optimizes expert loss and expected loss at simulated merged parameters generated by sampling scaling coefficients, masks, and additive noise. We further introduce periodic scheduling, kernel fusion, and parameter storage switching to make SMAT efficient, with one forward and one backward pass per step. Across four language and vision-language backbones, SMAT improves the mean score across five merging methods by 1.07-2.16 points over the strongest baseline for each backbone, with less than 2% training-time overhead over standard fine-tuning. 

---
# CORA: A Protocol for Diagnosing Boundary Robustness in Text-to-Audio Retrieval under Query Reformulations 

**Authors**: Jae Min Woo, Kyongmin Kong, Bogyung Jeong, Minjeong Kim, HaeJun Yoo, Du-Seong Chang  

**Link**: [PDF](https://arxiv.org/pdf/2609.33433)  

**Abstract**: Text-to-Audio (T2A) retrievers are typically evaluated with caption style queries, but the same user intent can be expressed in many forms. We introduce CORA (Caption-Offset Retrieval for Audio), a caption anchored diagnostic protocol that rewrites each source caption into five intent preserving forms (Command, Question, Indirect, Key phrase, and Statement) while fixing the target audio. By tracking the same target across query forms, CORA defines RankDrop, a metric revealing failures hidden by Recall@k. Using Pearson's correlation coefficient r, we find that RankDrop is weakly associated with raw text space movement (r=0.084), but strongly associated with Target Alignment Loss and Target Boundary Margin Degradation (r=0.508 and r=0.615). The same pattern appears in OEA retrievers, where RankDrop is better explained by boundary degradation (r=0.472/0.478) than by query movement (r=0.084/0.046). Overall, these results suggest that robust T2A retrieval requires preserving the target's boundary advantage over competing audio under reformulation. 

---
# MetaBench-Harness: Unlocking End-to-End Optimization of Benchmark Harnesses 

**Authors**: Xuanjun Chen, Hua-Hsuan Chen, Wei-Chung Lu, Yinghao Ma, Jyh-Shing Roger Jang, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2609.33411)  

**Abstract**: Rapid progress in Large Language Models (LLMs) is saturating static benchmarks faster than they can be designed. While existing automated evolution frameworks attempt to generate harder questions by perturbing individual tasks, they remain constrained by rigid, hard-coded generation rules. Moving beyond the evolution of isolated tasks, we propose to optimize the benchmark generation workflow itself end to end with MetaBench-Harness, a dual-loop search framework. Specifically, the inner loop utilizes a benchmark harness to generate a new benchmark in each round, while the outer meta-harness orchestration layer iteratively refines and searches over harness implementations based on historical evolution trajectories. By applying MetaBench-Harness to the competitive programming CodeContests and Olympiad mathematics AIME-2024 datasets, we demonstrate that the evolved benchmarks are challenging and discriminative for frontier models. Trajectory and quality analyses verify that MetaBench-Harness enables multi-dimensional evolution, steadily improving evolution reasonableness, benchmark competency, and evaluator robustness across successive rounds. Furthermore, case studies reveal its effective utilization of diverse difficulty levers to reframe problems and elevate required capabilities. Ultimately, this work provides a solution to the pressing challenge of benchmark saturation. 

---
# Decoupling Token Roles in Autoregressive Pretraining 

**Authors**: Suqin Yuan, Runqi Lin, Kevin Qinghong Lin, Junchi Yu, Lei Feng, Chris Russell, Tongliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.33405)  

**Abstract**: Autoregressive pretraining increasingly draws on heterogeneous data, making it important to understand how a model learns from an individual token. The next-token prediction objective naturally identifies a token's contribution with its own loss. However, each token is not only a prediction target but also context for what follows. Using controlled corruption, we decouple these two roles and find a reversal: making a noisy token easier to predict reduces its damage as a target but increases it as context. The same decoupling helps explain text generated by language models: generation selects each token by its fit to the prefix, while its role as context is never tested against an independently determined continuation, because that continuation is generated to fit it. At known corrupted positions, acting through the context can reduce damage that removing the token's own loss does not. Understanding and controlling what a model learns from a token therefore requires decoupling its roles. 

---
# OLED-MoE: Accelerating MoE-Based dLLM Inference via Inter-Iteration Locality-Aware Expert Offloading 

**Authors**: Jingyuan Xiao, Jiayue Wang, Yitao Hu, Xinning Wang, Shi Chen, Ziqi Gong, Zhengchao Wang, Guotao Yang, Sheng Chen, Keqiu Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.33385)  

**Abstract**: Semi-autoregressive diffusion large language models (dLLMs) improve decoding parallelism through iterative block-wise denoising, but scaling them with mixture-of-experts (MoE) layers introduces a large expert parameter footprint that exceeds memory-constrained GPU capacity. Expert offloading is a natural remedy, yet existing MoE serving systems target autoregressive decoding and rely on intra-iteration layer-wise prefetching: while computing one layer, they predict and load experts for subsequent layers. Under dLLM inference, block-wise routing expands the active expert working set within each iteration, making such prefetches difficult to complete in time and costly when mispredicted. Consequently, existing prefetch-based solutions often degenerate into on-demand expert loading with high decoding latency.
We propose OLED-MoE, an expert offloading system that shifts the optimization target from intra-iteration prefetching to inter-iteration expert retention. Its key insight is that adjacent denoising iterations exhibit strong expert routing overlap, and token confidence indicates which experts are likely to be reused. OLED-MoE uses confidence-guided inter-iteration prediction to retain high-value experts in GPU memory without introducing extra prefetch traffic. It further compensates unavoidable cache misses through CPU-GPU cooperative execution, jointly considering dynamic expert computation load and predicted future reuse. Across diverse dLLM workloads, OLED-MoE reduces time per output token (TPOT) by 1.23x-7.93x and improves expert cache utilization by 1.44x-4.23x over state-of-the-art offloading systems. Notably, OLED-MoE approaches full-residency performance while using only 40% of the expert GPU memory, incurring merely 23% higher TPOT despite a 60% reduction in expert memory footprint. OLED-MoE's source code is publicly available at this https URL. 

---
# From Script to Drama: An Agentic Framework for Controllable Multi-Speaker Dialogue TTS 

**Authors**: Kangxiang Xia, Xinfa Zhu, HangRui Hu, Kexin Huang, Wenjie Tian, Ziyue Jiang, Bingshen Mu, Jingbin Hu, Ting He, Lei Xie, Jin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2609.33362)  

**Abstract**: Multi-speaker dialogue TTS requires natural speech generation, consistent speaker identity, coherent cross-turn transitions, and fine-grained control of expressive attributes such as emotion, speaking rate, and loudness. These requirements are difficult to satisfy reliably with one-shot generation, especially in long-form dialogue. We propose a controllable multi-speaker dialogue TTS framework that formulates synthesis as critique-driven iterative refinement. Its speech backbone, ControlEdit-TTS, unifies instruction-following synthesis and natural-language-guided attribute editing, enabling correction of expressive errors without full regeneration. The framework further performs hierarchical utterance-level and scene-level critique, routing detected issues to editing, resynthesis, or timing adjustment. Experiments on a bilingual Chinese--English dialogue benchmark show improved utterance-level instruction following, better dialogue-level preference than direct dialogue models and agentic baselines, and more effective refinement than regeneration-only alternatives while preserving speaker identity. Ablations further confirm the benefits of scene-level critique and edit-based correction. 

---
# CalibHyper: Chance-Corrected Relational Hypergraphs for Few-Shot Molecular Property Prediction 

**Authors**: Linyu Li, Zhi Jin, Yuanpeng He, Dongming Jin, Huanyu Liu, Huanyao Zhang, Haoran Duan, Heng Tian, Gadeng Luosang, Nyima Tashi  

**Link**: [PDF](https://arxiv.org/pdf/2609.33342)  

**Abstract**: Molecular property prediction is central to drug development and materials discovery, but experiments are costly and labeled data are scarce. Context-aware methods use auxiliary assay labels to support few-shot prediction, and recent work supervises property relations with label agreement. However, label agreement is sensitive to class marginals and does not directly capture dependence between properties. We propose CalibHyper, a chance-corrected relational hypergraph method based on the joint label distribution. CalibHyper subtracts an independence baseline from the ordered four-state label distribution and shrinks the residual according to the number of joint observations. A swap-equivariant relation head estimates these residuals, which choose the auxiliary properties for each molecule and set the sign and weight of their hyperedge messages. On thirteen datasets from five benchmarks, in both 1-shot and 10-shot settings, CalibHyper and its ablation settings achieve ROC-AUC competitive with the strongest reported results. 

---
# When to Evict, Not What to Keep: Draft-Guided Eviction for Training-Free KV-Cache Compression 

**Authors**: Haeyong Kang, Chang D. Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2609.33334)  

**Abstract**: Training-free KV-cache compression methods such as SnapKV, H2O, and PyramidKV evict tokens at the end of prefill, aiming to preserve the attention mass that future queries are expected to use -optimizing what to keep. We show that this objective fails in two distinct ways. (1) Compensation: restoring the evicted attention mass can recover the attention-level target without recovering task quality. (2) Selection: covering more of the true decode-query mass can hurt quality when the recovered mass is fragmented rather than concentrated in coherent spans. These failures share a common cause: eviction occurs before the queries that determine the answer trajectory exist. We propose Draft-Guided Eviction (DGE), which defers eviction until after drafting the first k=2 answer tokens using the full cache - just one decode step beyond prefill. Because the draft is generated from the answer's own prefix, no cache entries are discarded before this trajectory signal becomes available. The per-head cache budget remains unchanged, and DGE can be applied directly to SnapKV, PyramidKV, H2O, and StreamingLLM without modifying their eviction scores. Unlike extra-pass methods, DGE changes when eviction occurs rather than what cache entries are selected. Extensive experiments demonstrate that DGE outperforms prior methods at every evaluated budget on five of six instruct-tuned backbones, achieving 44.2 on LongBench, nearly matching FullKV at 44.3. The timing-only control DGE-W achieves the same score, demonstrating that the gain comes from when eviction occurs rather than what is selected - an effect we term trajectory anchoring. 

---
# Robust Hierarchical Structures for Agentic Document Analysis 

**Authors**: Ruiying Ma, Yiming Lin, Aditya G. Parameswaran  

**Link**: [PDF](https://arxiv.org/pdf/2609.33322)  

**Abstract**: Large Language Models (LLMs) enable us to better understand text documents, including PDFs and Word documents. However, LLMs, as well as more modern LLM agents, i.e., those with tool-calling abilities, typically treat such documents as plain text, ignoring the fact that they are often organized hierarchically into sections and subsections. Extracting this structure, while difficult, can improve efficiency and effectiveness for agents (and humans)---since only sections relevant to a given task need to be processed. Unfortunately, prior work on structure extraction provides no formal guarantees on how well the inferred structure matches the true one. Instead, we target a robust and compact variant that is feasible to infer and useful in practice. Robustness ensures that the text under each subsection header is a superset of the text under the same header in the true structure. Compactness seeks to minimize this superset, reducing agentic cost (or human cognitive load). We propose SHED, a two-stage workflow for inferring a robust and compact structure. The first stage is pluggable with an infinite family of approaches, each guaranteeing robustness for a specific document class. We theoretically characterize the document space using these classes and their hierarchical relationships. Empirically, SHED improves F-1 scores (measuring the robustness--compactness trade-off) by 13%--68% over non-LLM baselines and 9%--15% over expensive LLM-based approaches. Finally, we show how SHED-inferred structures are valuable for agentic document analysis: agents using SHED outperform baselines, achieving 3%--23% higher accuracy while being up to 10x cheaper. 

---
# TraceDance: An Automated System for Building Agent Behavior Benchmarks from Real-World Agent Deployment Traces 

**Authors**: Dehai Min, Daoan Zhang, Yiming Zeng, Huayi Zhang, Ziyi Chen, Yan Zhang, Qinbo Bai, Mengyuan Chao, Jing Ning, Qiyue Hua, Huiyi Chen, Hanrong Zhang, Henry Peng Zou, Jie Yang, Wei Xu, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2609.33295)  

**Abstract**: An agent can complete a task while exhibiting undesirable behavior during execution. Developers need tests for the specific behaviors encountered in deployment, beyond fixed benchmark suites. We present TraceDance, an agent system that constructs targeted benchmarks from deployment traces for user-specified undesirable behaviors. For efficient construction, Anchor-and-Confirm combines programmable retrieval with candidate-level confirmation by a Flash large language model (LLM), while the Anchor Synthesis Loop generates and revises specifications for custom behaviors. The benchmarks use decision-point continuation to evaluate an LLM's next turn at a recorded decision point with a behavior-specific rubric, without a reference answer or environment replay. Experiments in coding and general tool use draw on 252,557 sessions and produce 107 benchmarks with 4,125 instances, fulfilling 95.3% of build-target requests. Both human annotators confirm the requested behavior in 84% of sampled instances, and the automated grader's agreement with human pass/fail judgments is comparable to that between the annotators. Nine frontier LLMs achieve a mean pass rate of only 26.7%, showing that they still struggle to respond appropriately at the evaluated decision points. Analysis across behavior-specific benchmarks further reveals weaknesses in how current LLMs behave as agents. By turning deployment problems into targeted benchmarks, TraceDance could serve as a key component of the recursive self-improvement (RSI) loop. 

---
# InfoEdit: Probing Global Layout Reasoning in Infographic Editing 

**Authors**: Cheng Yang, Chufan Shi, Huijuan Wang, Bo Shui, Yaokang Wu, Muzi Tao, Yibo Yan, Xuezhe Ma, Taylor Berg-Kirkpatrick  

**Link**: [PDF](https://arxiv.org/pdf/2609.33286)  

**Abstract**: Multimodal foundation models edit natural photographs at production quality, yet the same models struggle with structured visual content such as infographics. Unlike photographs, infographics encode information through logical relations; editing one element often requires surrounding elements to be adapted. We refer to this global layout reasoning capability as reflow. Existing image-editing benchmarks neither provide a dedicated setting for structured visual content nor evaluate the reflow capability. We introduce InfoEdit, a novel benchmark of 1,000 infographics across eight logical-relation families, paired with 4,000 editing instructions across four editing tasks, and a reflow-aware evaluation protocol. Across eight frontier editors, only GPT-Image-2 clears 60% average success rate; most models fall below 7%, and no editor exceeds 36% on the Swap-Block task even with perfect target localization. We further show that code-level editing can match the strongest pixel-level editor, revealing complementary strengths across tasks. InfoEdit identifies reflow as a central challenge in structured visual content editing and provides a diagnostic benchmark to facilitate future progress. 

---
# BERT4DTI : BERT-based Model for Predicting Drug-Protein Interactions 

**Authors**: Thanina Hamitouch, Khadidja Henni, Abdelkrim Arie, Amina Selma Haichour, Neila Mezghani, Lina Abou-Abbas  

**Link**: [PDF](https://arxiv.org/pdf/2609.33254)  

**Abstract**: Understanding how drugs interact with protein targets is fundamental to drug discovery, drug repurposing and the early identification of promising therapeutic candidates before costly experimental testing. Sequence-based DTI models face three practical limitations: labelled interactions are scarce and unevenly distributed, large pretrained chemical and protein encoders are expensive to fine-tune end-to-end, and independently encoded sequences do not capture pair-specific dependencies. We present BERT4DTI, which encodes SMILES strings with ChemBERTa and amino-acid sequences with ProtBERT, applies bidirectional mutual attention between token-level representations, and classifies the resulting interaction features using convolutional layers and a multilayer perceptron. To reduce trainable size, ProtBERT is truncated to 18 retained layers and only the last two layers of each encoder are fine-tuned. On BIOSNAP, DAVIS and BindingDB, BERT4DTI is competitive, achieving the best ROC-AUC and PR-AUC on BIOSNAP and the highest sensitivity on all three benchmarks. An ablation on DAVIS shows that mutual attention improves PR-AUC and specificity. With 125M trainable parameters compared with 353M for full BERT fine-tuning, BERT4DTI provides a favourable performance-parameter trade-off for sequence-based DTI screening, while leaving runtime profiling, calibration and leakage-audited validation for future work. 

---
# Acoustic Progress Propagation for Long-Horizon Speculative Decoding in ASR 

**Authors**: Yuanyuan Jia, Qianqian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2609.33245)  

**Abstract**: Speculative decoding accelerates autoregressive automatic speech recognition (ASR), but the acceptance length of alignment-aware drafters can saturate as the draft horizon increases. We propose a progress-aware speculative drafter that recurrently propagates an acoustic progress state across draft steps and feeds it back into audio cross-attention to guide token generation. We jointly train the drafter and progress predictor over variable draft horizons. On five ASR test sets, our method achieves lossless, macro-averaged end-to-end speedups of 1.657x and 1.227x over target-only autoregressive decoding with Qwen3-ASR-0.6B and Qwen3-ASR-1.7B, respectively. Relative to AnchorDraft, our method improves the macro-averaged speedup by 34.3% and 9.0%, respectively. Horizon sweeps show continued growth in acceptance length beyond the baselines' saturation. Code is available at this https URL. 

---
# When Do Models Admit They Are Wrong? Failure Disclosure Is Unstable Under Reinforcement Learning 

**Authors**: Steven Y. Feng, Noah D. Goodman, Michael C. Frank, Evan Hubinger, Paul C. Bogdan, Andrew Lampinen  

**Link**: [PDF](https://arxiv.org/pdf/2609.33220)  

**Abstract**: Outcome-based reinforcement learning can produce models with similar task performance but very different ways of communicating about their mistakes. We study failure disclosure: whether a model admits that an attempted solution failed rather than staying silent or presenting it as successful. Across repeated outcome-only GRPO training runs, failure disclosure varies far more than task accuracy. The pattern extends to a second reasoning task and stabilized PPO, persists at 7B, and also appears in an instruction-conditioned 32B setting. We also find that small floating-point and sampling differences during training can redirect reporting behavior even when the task objective and earlier training history are held fixed. Additional tests show that failure disclosure is not a single decision: Checking the answer, entering a report, and completing the admission can separate, and the weak point depends on the task and response format. Further, experiments with neutral controls show more broadly that behaviors left weakly constrained by training are especially likely to vary across runs, of which failure disclosure is an example. We can reduce variability in failure disclosure by discouraging the model from drifting from its starting policy on failed, well-formed responses. This makes reporting substantially more consistent, though its effect on task performance depends on the setting. Stable task accuracy therefore does not guarantee stable safety-relevant behavior: Researchers should measure these behaviors directly across runs and design training methods that keep them reliable. 

---
# Teach Yourself Where to Look: On-Policy Attention Self-Distillation for Reasoning 

**Authors**: Safaeid Hossain Arib, Rabeya Akter, Ismam Nur Swapnil, Md. Faiyaz Abdullah Sayeedi, Md Mofijul Islam, Tasnim Mohiuddin  

**Link**: [PDF](https://arxiv.org/pdf/2609.33200)  

**Abstract**: On-policy self-distillation trains reasoning models on their own trajectories using dense token distribution guidance from a privileged teacher with access to a verified solution. This supervision transfers what the teacher predicts without directly transferring where it attends within the preceding context. We introduce On-Policy Attention Self-Distillation (OPASD), which complements token-level supervision with solution-conditioned attention distillation. Because the privileged teacher can attend to verified solution tokens unavailable to the student, OPASD projects teacher attention onto student-visible positions and renormalizes the resulting distribution before alignment. Across three model sizes and four competition-level mathematics benchmarks, OPASD consistently outperforms token-only OPSD, improving average accuracy by 4.98 to 8.40 percentage points. OPASD also avoids the response-length inflation and performance degradation observed with token-only distillation, reducing generated rollout tokens by 73.9% and estimated model compute by 72.6% while training 1.53x faster. These results show that solution-conditioned attention provides a complementary supervision signal that makes on-policy self-distillation more accurate, stable, and compute-efficient. 

---
# AG-CoT: Verified Algorithmic Traces for LLM Program Synthesis on Clifford Circuits 

**Authors**: Lu Wei, Yufeng Wang, Chenfeng Cao, Lu Pang, Haibin Ling  

**Link**: [PDF](https://arxiv.org/pdf/2609.33192)  

**Abstract**: Scientific code generation can produce executable programs that fail to compute the intended scientific object. We study this problem in language-model synthesis of Clifford circuits, which prepare the stabilizer states used in quantum error correction and admit exact classical verification. In our target-conditioned framework, each target is given as compact signed stabilizer generators, and an exact verifier checks the generated OpenQASM circuits. We supervise models with Aaronson-Gottesman chain-of-thought (AG-CoT) traces checked by the verifier, and continue training on model generations that the verifier accepts. Across two independently trained model families (3B and 7B), AG-CoT supervision multiplies greedy-decode state-equivalence accuracy by four to six times over circuit-only baselines, and verifier-filtered continuation training adds a further consistent gain atop both. A complementary 32B study shows that supervised models achieve near-perfect syntax and Clifford validity while the strongest direct model reaches 6.14% state equivalence per target, rising to over 10% under verifier-guided selection with multiple candidates. These results show that algorithmic trace supervision gives a large, statistically significant gain in both model families and that verifier-filtered continuation adds a further repeated gain. The persistent gap between Clifford validity and state equivalence confirms that exact verification is necessary: a circuit can be syntactically and physically valid yet prepare the wrong quantum state. 

---
# SeOPD: Self-Evolving LLMs via Online Policy Distillation from Self-Generated Chain-of-Thought 

**Authors**: Xiaoshu Chen, Xiangyu Wong, Sihang Zhou, Ke Liang, Xinwang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.33181)  

**Abstract**: Recent advances in online policy self-distillation (OPSD) have demonstrated that large language models (LLMs) can improve their capabilities by leveraging external privileged information (PI), such as manual annotations or feedback from external environments. However, obtaining accurate annotations and constructing sophisticated environments often require substantial human effort and computation, limiting the scalability of OPSD. While a few recent studies have explored self-improvement without external PI, the resulting gains remain limited. In this work, we explore whether LLMs can achieve comparable self-improvement without external PI. Our key observation is that a single LLM can support multiple reasoning modes, such as deep-thinking and non-thinking modes, with deep thinking generating additional information during reasoning. Based on this observation, we propose Self-Evolving Online Policy Distillation (SeOPD), which enables LLMs to distill and internalize information generated by their own chain of thought (CoT). Specifically, it (1) generates CoT with the deep-thinking mode, (2) produces responses with the non-thinking mode, and (3) uses the generated CoT as PI to provide token-level supervision for the non-thinking response, allowing new information inferred during reasoning to guide the non-thinking mode and be internalized into the shared model parameters, thereby improving both non-thinking and deep-thinking capabilities. Extensive experiments across LLMs and tasks demonstrate the effectiveness of SeOPD. 

---
# Save Your Saturated Data: Learning Beyond Reward Saturation in Group-Based RL 

**Authors**: Ziyuan Yang, Yike Wang, Shangbin Feng, Yulia Tsvetkov  

**Link**: [PDF](https://arxiv.org/pdf/2609.33126)  

**Abstract**: Group-relative reinforcement learning (RL) relies on reward variation among sampled responses to estimate informative relative advantages. As language models become increasingly capable, existing training data can become reward-saturated: all sampled responses to the same problem might receive equally high rewards, where the group-relative learning signals vanish and leave previously useful data obsolete. In this work, we investigate whether useful learning signals can be recovered from such saturated data. We study interventions at four levels of group-policy RL pipelines---data, rollout, reward, and advantage---and conduct extensive RL training on saturated reasoning data only. While standard GRPO on saturated data would almost always yield near-0 advantages and near-noise signals, diverse interventions successfully recycle and repurpose such data: among the proposed strategies, interventions at rollout generation are consistently most effective: nudging the policy to generate ``high-quality'', incorrect solutions introduces rollouts with poor rewards into saturated groups as negative samples, which turns out to improve GRPO by 6.4% to 9.0% across Qwen3-1.7B and 4B. Other interventions such as increasing rollout temperature or adding auxiliary rewards can also restore non-zero advantages, but yield less consistent gains. Further analyses show that effective negative rollouts require informative negative trajectories, that the method remains effective alongside unsaturated data, and that it supports iterative recycling of newly saturated examples. While increasingly stronger LLMs would render more data as saturated, our results demonstrate that don't waste your saturated data: with the right strategies they can be recycled into useful RL training signals in an increasingly data-scarce world. 

---
# ECG-Scroll: A Long-Horizon, Streaming Benchmark and Agent Environment for Interpretation of Ambulatory Electrocardiograms 

**Authors**: Haitao Li, Chenglin Li, Zhengyao Ding, Ziyu Li, Yiheng Mao, Zhengxing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.33117)  

**Abstract**: Multimodal large language models (MLLMs) can now interpret a standard ten-second, twelve-lead electrocardiogram (ECG) with clinically grounded, reward-verified reasoning. Real cardiac monitoring is different. Ambulatory (Holter) and telemetry recordings span hours to days and are read as they stream in, and their clinically decisive findings are paroxysmal, brief episodes buried in an otherwise unremarkable trace. Such a recording cannot be held in one context at diagnostic resolution, and its future has not yet happened, so a reader must work online, deciding what to measure now, committing evidence to memory as it passes, and reporting events as they occur. We recast long-duration ECG interpretation as a long-horizon, online (streaming, causal) sequential decision process and introduce ECG-Scroll. As a benchmark, long ambulatory recordings are streamed to an agent chunk by chunk, and it must localize, quantify, and promptly flag paroxysmal events without access to future signal; because the underlying signal is retained, every answer is checkable against objective ground truth, giving rule-based rather than judge-based rewards, and the streaming formulation adds a metric batch evaluation cannot express, the detection latency between an event's onset and the moment the agent records it. As an agent environment, it is a fixed, gym-style interaction layer that exercises three competencies single-glance ECG models never touch: Memory, Tool use through signal-grounded measurement rather than reading pixels, and Planning of what to measure now and when to commit. We release 390 whole-recording instances spanning 2,536 hours of two-lead ambulatory ECG and evaluate a signal-threshold rule agent alongside off-the-shelf LLM agents online, characterizing how they use memory, tools, and planning and where the benchmark's head-room lies. 

---
# Algorithmic Harms Associated with Generative Model-Augmented Recommendation Systems 

**Authors**: Christine Herlihy, Xumei Xi, Shloka Desai, Kevin Bannerman Hutchful, Pedro Silva  

**Link**: [PDF](https://arxiv.org/pdf/2609.33073)  

**Abstract**: In this work, we consider algorithmic harms that may arise as generative models are incorporated into machine learning platforms. We argue that existing harm taxonomies and threat models require extension to (1) address novel causal drivers of well-studied representational and quality-of-service harms; and (2) anticipate and mitigate endogenous harms, such as sanitization, which may arise when system inputs are misaligned with the system designer's objectives, or the generative model's inductive priors. To this end, we introduce an expanded taxonomy of algorithmic harms associated with the use of generative models in non-conversational recommendation systems. In addition, we offer a causal analysis of how problematic subsets of the (input, output) joint distribution can arise, in an effort to inform harms detection and mitigation efforts. 

---
# Trust and Task Completion in the World of Consumer AI Agents 

**Authors**: Jeroen Olieslagers, Eduardo Pujol, Gal Zahavi, Lukas Ingemarsson, Shivani Poddar  

**Link**: [PDF](https://arxiv.org/pdf/2609.33017)  

**Abstract**: Action agents do things for people. They send email, spend money, and call businesses while the user is busy with something else, so a mistake can turn into an action before anyone notices. They fail their users in two ways. They break trust when they do something the user never agreed to, or hold back after the user clearly said go. And they fall short on completion when they give up on errands that turn out to be hard. Both depend heavily on the harness around the model, meaning its instructions, tools, context, and guardrails. We built an evaluation that scores trust and completion on the same runs, in a simulated world of businesses with their own websites, inboxes, and phone lines, and of people who write back. A simulated user answers the assistant's questions. Trust means that nothing happens the user did not agree to. No email goes to someone they never approved, no private detail ends up on a group thread, no money is spent past their limit, no stranger's instructions are followed, and nothing is claimed without a source. Every trap has a matched control in which acting is the right call. We use the evaluation to measure Fo, Wajo's personal assistant, against a base model with basic instructions on three foundation models, and against the Fo harness with its guardrails switched off. Fo completes 71% of the errands and keeps the user's trust on 94% of the trap runs. The base models complete 50% to 64% and keep trust on 59% to 75%. On the matched controls, Fo goes ahead slightly less often. OpenClaw, a popular open-source assistant given the same access, completes 42% of the errands it shares with Fo, against 71%, and keeps the user's trust on 74% of the shared trap runs, against 94%. Measuring trust and completion together, on the whole system rather than the model alone, is how we think action agents become safe to hand real work to. 

---
# DynamicDx: Evaluating Evidence Acquisition in Video-Based Diagnosis 

**Authors**: Jiahui Li, Yutong Guo, Nan Yang, Wenzhan Song, Jin Lu, Fei Dou  

**Link**: [PDF](https://arxiv.org/pdf/2609.32957)  

**Abstract**: Diagnosing a patient from video requires more than recognizing the sign: a vision-language model must turn what it sees into hypotheses, questions and tests. DynamicDx evaluates each step in 71 neurological consultations across 11 sign categories, linking authentic patient videos to confirmed diagnoses and fixed charts built from the same case reports, so that every model queries the same evidence. Across five such models, video improves accuracy by 9.9-22.5 percentage points over blind input, but neither recognition alone nor temporal order explains the gain: the cause is usually missing from the model's video-only differential diagnosis even when the sign is recognized, and shuffling the frames produces no reliable accuracy loss. Instead, a trajectory replay traces most of the gain to the investigation results the video prompts. Evidence acquisition is the bottleneck: supplying the decisive investigations raises accuracy to 73.2-93.0%. Two interventions act on it. A post-trained 4B video describer improves sign descriptions, especially from a short, densely sampled segment, and source-clean literature retrieval expands initial hypotheses; both bring the tests a model orders closer to those the treating clinicians documented and, through them, raise accuracy. For video-based diagnosis, seeing better helps when it leads to asking better. 

---
# Theory of Scene: Breaking the Symmetry Trap in Multi-Agent LLM Coordination 

**Authors**: Liangqi Yuan, Wenzhi Fang, Shiqiang Wang, Christopher G. Brinton  

**Link**: [PDF](https://arxiv.org/pdf/2609.32939)  

**Abstract**: Multi-agent systems built on large language models (LLMs) are largely homogeneous, as their agents behave alike even across distinct LLMs. We show that when such agents act concurrently without communication, they collide on targets they must split and diverge on targets they must take together, a double failure we term the symmetry trap. Theory of Mind (ToM), widely used for coordination without communication, cannot escape this trap, since homogeneous agents form the same prediction of one another and respond to it in the same way. We propose Theory of Scene (ToS), a training-free reasoning schema in which each agent reads its public role, the only difference between the agents, and the task context they all observe. Homogeneous agents thereby derive one division of labor, each taking the part its role fixes, which turns homogeneity from the cause of the trap into the cure. ToS reads the role together with the scene through role gating, which determines whether ownership overlaps or is already divided, and the task context through task coupling, which infers whether the team must converge on each target, divide it, or take its stages in turn. We evaluate on DivvyBench, a controlled environment we introduce, whose target types make an episode Competitive, Cooperative, or Mixed across Tabletop, Airspace, and Household scenarios, and on two established agentic benchmarks, GovSim and Overcooked. ToS outperforms all six baselines on every benchmark, and each baseline falls far behind it in at least one setting. Against ToM given the same inputs, ToS raises the DivvyBench success rate from 71.1% to 99.6%, the GovSim total gain from 207 to 400, and the Overcooked level-normalized throughput from 1.41 to 1.67. 

---
# Planner-as-Router: Joint Plan-Time Model Routing for Cost-Efficient Multi-Agent Workflows 

**Authors**: Vivek Kumar Singh, Preeti Priyam, Gautam Bhowmick  

**Link**: [PDF](https://arxiv.org/pdf/2609.32917)  

**Abstract**: Running large language model (LLM) agents in production gets expensive fast. A frontier model (the largest, most capable tier) is accurate but can cost 25 times what a small model costs per token, and the gap compounds once a workflow chains several calls together. Planner-as-Router (PaR) attacks this from a different angle. Instead of leaving model-tier selection to some component downstream, it folds the choice into planning itself. As the planner breaks a query into subtasks, it also assigns each one a model size tier (small, mid, or frontier, ordered by capability and price), so the dependencies between subtasks are visible before any specialist runs. Unlike per-call routers such as cascade routing, which look at one node at a time, PaR sees the whole workflow up front and needs no separate router model or training data. We evaluate PaR with EntBench, a benchmark of 54 enterprise agentic tasks across seven classes, graded by actually running the generated Structured Query Language (SQL) and MongoDB queries against live databases. Over 1,157 evaluations spanning eight routers and three seeds, PaR stays on the observed cost-accuracy frontier. It matches a sink-frontier heuristic (frontier model on terminal nodes only) in accuracy at comparable cost and a faithful FrugalGPT cascade at lower cost, and cuts cost 44% against all-frontier routing while giving up 2.9 points of accuracy. Several accuracy gaps fall inside the plus-or-minus six-point confidence interval of a 54-task study, so we frame PaR's advantage as frontier position rather than a clean accuracy win. We also report a preliminary observation, not a validated result: a small pilot hints that cheap routing may carry a hidden compounding penalty on compositional workflows, which we frame as a hypothesis for future measurement. PaR, EntBench, and all evaluation code are open source. 

---
# Logical subspace in LLMs 

**Authors**: Hope Kean, Enric Boix-Adsera  

**Link**: [PDF](https://arxiv.org/pdf/2609.32907)  

**Abstract**: Recent work has identified a human brain network specialized for abstract formal reasoning (Kean et al., 2025). Does the same hold true in language models? To answer this question, we introduce the minimal viable subspace (MVS) method, which searches for the lowest-rank activation subspace at a layer that preserves task performance when everything outside that subspace is ablated. Using MVS, we demonstrate low-rank subspaces supporting logical inference on Gemma and Qwen models. Furthermore, these subspaces exhibit a clear dissociation from model capacities on other tasks, such that retaining these late logic subspaces preserves inference while impairing factual knowledge, working memory, cognitive control, and arithmetic. Conversely, ablating them reduces logical inference accuracy to chance while largely sparing these other capacities. Our results suggest a functionally localizable core machinery for logic akin to that in the human brain. 

---
# Multimodal LLMs Outperform Pathology Foundation Models in Cross-Domain Histological Similarity 

**Authors**: Yishu Zhang, Yun Li, Daiwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.32876)  

**Abstract**: State-of-the-art pathology foundation models, trained on millions of histology tiles, can fail to preserve tissue similarity when comparisons cross slide or institution boundaries. We show that general-purpose multimodal LLMs, without being trained as pathology foundation models, consistently outperform these specialized models in cross-domain histological similarity judgments. Using a relative similarity framework that we release as the MOSAIC (Model Similarity Assessment across Institutions and Cohorts) benchmark, we evaluate 17 models across 6 datasets and find that pathology encoders often rank same-institution, different-disease tiles as more similar than same-disease, different-institution tiles, a clinically dangerous failure mode invisible to standard within-domain evaluations. LLMs appear less susceptible to this failure, likely because they perform semantic visual comparison of morphology and tissue architecture rather than relying on shortcut features tied to acquisition context. Scaling training data does not resolve the problem for pathology encoders, implicating the learning objective rather than data coverage. Our results expose a fundamental robustness gap in current pathology foundation models and establish multimodal LLMs as a viable alternative for cross-institutional retrieval, dataset harmonization, and multi-site quality control. Code and data will be released upon acceptance. 

---
# The Decomposition Tax: LLM Pipelines Lose Up to 40 Accuracy Points at Their Own Interfaces 

**Authors**: Tianqi Bu, YuXuan Peng, Junteng Tu, Henghui Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2609.32825)  

**Abstract**: A four-stage LLM pipeline gives up as much as 40.5 accuracy points at its own interfaces (gemma-3-12B on MATH-500, Holm-corrected p = 1.66e-19; the largest tax in the primary family). We hold model, problem, stages, stage prompts and completion budget fixed, vary only whether each stage can still see the original problem, and call the accuracy difference the decomposition tax. Across 21 open-weight models from nine organisations, on GSM-Hard and MATH-500 at n = 200 paired items per cell, 70 of 118 primary-family tests survive Benjamini-Hochberg correction and 54 survive Holm. On GSM-Hard, a placebo recovers nothing: it carries at least 60% of the extra tokens and at most one word of the problem. Builders design a pipeline one stage at a time, and its bill arrives at the interfaces between stages. Rewriting one stage's instruction moves gemma-3-12B's tax from 4.5 to 36.5 points, and adding "every relationship stated between them" to a stage that lists the numerical quantities lowers the tax on 9 of 9 models on MATH-500. Re-grounding, which shows a stage the original problem again, belongs after the loss. With one lossy interface, re-grounding the stage after it beats re-grounding the stage before it on 7 of 7 models on both benchmarks; on MATH-500 the earlier repair is worse than none on 7 of 7. Newer models still pay: gemma-4-12B gives up 37.0 points, and the repair holds on all three of the newest models we test. A sealed held-out test refuted a stronger rule we registered, which predicted the paying stage from the interface and receiver types, so we locate the tax by measuring one stage at a time. The prescription has two parts: re-ground the stage after the lossy interface, and if a stage must list the quantities, tell it to keep the relationships. 

---
# USAI-Quant: A Quantitative Reasoning Benchmark for Vision-Language Models in Built Environments 

**Authors**: Dongdong Wang, Qingqi Song, Yuzhou Chen, Deepak Balakrishnan, Ravi Shankar Srinivasan, Shenhao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.32813)  

**Abstract**: Large vision-language models (VLMs) have emerged as a powerful paradigm for urban and spatial AI. However, current state-of-the-art large VLMs still struggle with quantitative reasoning on remote sensing imagery. Existing benchmarks and algorithms are predominantly based on qualitative Visual Question Answering (VQA), providing limited insights into the quantitative reasoning capabilities of VLMs for built environment metrics. To address this gap, we develop Quantitative Urban and Spatial AI benchmark (USAI-Quant), the first benchmark designed to quantitatively evaluate VLM's reasoning capabilities on built environment metrics via remote sensing imagery. USAI-Quant is curated from the 335 largest U.S. cities, aligning high-resolution remote sensing images with quantitative built environment metrics. We then evaluate both general-purpose and remote sensing VLMs (RS-VLMs) by applying VQAs to tens of built environment metrics across three complexity levels. Our results reveal that current state-of-the-art models consistently fall short on numeric reasoning tasks. We further conduct in-depth analyses across models, question types, and geographic locations, uncovering insights into performance variability and task-specific challenges. 

---
# Overwhelmed by Choice: Studying LLM Decision Making at Scale 

**Authors**: Yu-Chi Lin, Aryan Seth, Anshul Aravind, Eugene Lee, Tanmay Parekh, Nanyun Peng, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2609.32809)  

**Abstract**: Multiple-choice and candidate-selection evaluations are widely used to assess LLM reasoning and decision-making, yet most benchmarks contain relatively small candidate sets. It remains unclear whether conclusions drawn from these settings remain valid as the candidate space scales. We systematically evaluate LLMs as the number of competing candidates increases and find substantial accuracy degradation across tasks, prompting strategies, and model scales. Controlled analyses show that standard long-context retrieval explanations cannot fully account for this degradation. Instead, we identify two systematic failure patterns. First, gold-margin collapse: the score gap between the correct answer and the strongest distractor progressively shrinks, driven primarily by weakening confidence in the correct answer. Second, earlier candidate preferences become increasingly difficult to overturn, with later candidates exerting progressively weaker influence on the final prediction. Motivated by these findings, we evaluate hierarchical partitioning and permutation-based inference, which improve accuracy by roughly 20 percentage points at $N=160$ on both HotpotQA and MIMIC. Overall, our results identify candidate-set scale as an important evaluation-protocol variable and show that strong small-option performance does not necessarily imply robust large-scale candidate comparison. 

---
# Mind the Spike: Mechanisms and Brittleness of Visual Massive Activations in Large Vision-Language Models 

**Authors**: Jonas Ngnawé, Yann Pequignot, Sabyasachi Sahoo, Christian Gagné, Frédéric Precioso, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2609.32808)  

**Abstract**: Large vision-language models (LVLMs) inherit massive activations from their text-only bases: spikes where a few fixed hidden channels receive values thousands of times above the typical magnitude. The text spike systematically appears in early layers at a fixed initial position, independently of input content. Visual spikes vary across images, but whether their formation follows a consistent pattern across LVLMs and how they respond to image perturbations remain open questions. We find that some LVLMs do not form visual spikes, while others spike at different rates, typically in deeper layers. We identify the trigger direction from model weights and an interpretable location rule: before the language model decoder runs, eventual spike tokens are largely restricted to those sharing least with the rest of the image. Crucially, visual spikes are strikingly brittle. Common corruptions frequently create and relocate spikes, and less often remove them, raising overall incidence. Our trigger-guided spike attack deliberately creates or removes spikes under a small $\ell_\infty$ budget, with 1/255 enough in nine of the ten models that spike. Finally, our preventive intervention removes only the trigger component before spikes erupt, eliminating or substantially reducing spikes on clean and perturbed images while leaving the other image tokens nearly unchanged. Our study spans 25 adapter-based LVLMs built on 18 released text-only bases from 10 families, ranging from 2B to 72B parameters. 

---
# Decision-Sufficient State Representations: Measuring and Reducing Write-Time Regret 

**Authors**: Bingyu Shen, Boyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.32805)  

**Abstract**: Long tasks produce more history than an LLM agent can hold in its context, and more than it uses reliably even when the history fits. A growing line of work therefore has agents carry a short written state instead: at every step a writer rewrites the state, and a reader acts from the state alone. Steps stay cheap, but anything the writer drops is lost before later decisions reveal that they need it. We quantify this loss and ask whether training can reduce it. Comparing the written state with the best state of the same size written in hindsight, we split the reader's loss into a budget loss, which any state of that size must incur, and a write-time regret, which comes from the writer's choices. In TextWorld cooking games where we control how long a fact must be carried before it is needed, a 128-token state holding the facts wins nearly every game, while prompted language-model writers win at most 17%. Almost all of the loss is write-time regret, and it grows with the delay. We then train the writer from the reader's own loss. DSSR (decision-sufficient state representations) scores candidate states by how well the reader acts after the writer carries them forward, and teaches the writer to prefer the better ones. This forward-rolled score predicts game outcomes ($\rho = 0.48$), whereas scoring a candidate as a fixed context, as hindsight methods usually do, does not ($\rho \leq 0.07$). On a pre-registered test split opened once, training adds +7.0 [+1.9, +12.2] points of success when facts are needed soon, bringing a plain summary writer to the level of belief- and slot-based memory prompts. The gain shrinks as the delay grows and is significant only at the shortest delay. We trace this limit to credit assignment: keeping a fact now pays off only if every later rewrite keeps it too, which a per-step score cannot see. 

---
# Re-derivability Decides What a Staged Agent Pipeline Recovers After an Upstream Fault 

**Authors**: Tianqi Bu, YuXuan Peng, Junteng Tu, Henghui Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2609.32802)  

**Abstract**: One variable sets what an upstream fault costs a staged pipeline of language-model agents: re-derivability, how much of what a stage needs it can rebuild from the original problem. Grounding an inspector agent in that problem is worth +0.608 [+0.517, +0.700] to +0.358 over a blind one on four open-weight backbones served with thinking disabled, and on the two Qwen backbones the blind inspector changes no item at all. That head-to-head is exploratory. One deterministic fault enters the first stage, and we re-expose the original problem to $k = 0,\dots,3$ of the downstream stages with agents, items, fault and topology held fixed, on 120 gsm_hard items per arm at temperature zero. Accuracy under fault rises on four of four backbones, from +0.233 to +0.392, the largest Holm-adjusted $p$ being $2.1\times10^{-6}$. A registered kill test rules out tokens. Blanking every word holds the word slots fixed, and retention tracks the visible fraction on four of four, climbing from 0.221 to 0.692 on the primary. Those two families are confirmatory and everything else here is exploratory. The interaction excludes zero on two of four backbones under the registered pipeline, four of four under a three-stage pipeline, and three of four under full message history, the primary at +0.317. On Llama-3.1-8B the fault carries no detectable cost at any dose, so the other three carry every claim about what a fault costs. Re-derivability also sets what the architecture costs, and no decomposition we measured reliably beats one direct call. With no fault injected the registered pipeline loses to that call by -0.267, -0.125 and -0.317, and on Phi-4 reads +0.058 at $p = 0.118$, which the test fails to separate from zero. The repair that works is cheap and front-loaded: the first re-grounded stage buys +0.394 of matched retention for +59.8 tokens per item on Qwen3-14B, and the stages after it buy nothing. 

---
# Understanding and Exploiting Anisotropy in Post-Training 

**Authors**: Samyak Jha, Harshvardhan Saini, Yizhen Liao, Yiming Tang, Dianbo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.32792)  

**Abstract**: LLM post-training combines supervised fine-tuning (SFT), a mode-covering forward-KL objective, with reinforcement learning (RL), a mode-seeking reverse-KL objective. Frequency-weighted likelihood training leaves a well-known signature: \emph{anisotropy}, in which a few residual channels carry disproportionately large activations. Anisotropy is widely documented and usually treated as a defect, yet its function and its interaction with post-training remain unclear. We first analyze it. A label-free outlier rule isolates about 5\% of residual channels that are essential for language modeling: removing them raises perplexity from 10 to over $10^6$, versus 35 for count-matched random channels. Yet they barely distinguish correct from incorrect reasoning. SFT reshapes them, whereas RL leaves them largely intact and adapts the complementary channels. These channels therefore form the model's \emph{coherence substrate}, and reasoning adaptation happens elsewhere. We then exploit this. \textsc{SphereGate} learns one bounded gain per residual channel on a frozen backbone. Its activation-weighted gradients provably limit movement of high-energy coherence channels and leave the remaining channels free. With 0.1M trainable parameters, \textsc{SphereGate} outperforms parameter-efficient baselines by 2.0--7.3 points on MATH-500 across Qwen2.5 (0.5B--7B) and Llama-3-8B, is comparable or exceeds full-model GRPO. Anisotropy is not a defect but a division of labor that post-training can exploit. 

---
# CLAIRE: A Schema-Grounded Hybrid Workflow for Healthcare Administrative Form Completion 

**Authors**: Garapati Keerthana, Manik Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2609.32787)  

**Abstract**: Healthcare administrative staff transfer structured information from electronic health records, referrals, claims systems, provider rosters, and work queues into dynamic forms. We developed and evaluated CLAIRE (Clinical Language and Agentic Intelligence for Reasoning and Entry), a hybrid workflow that separates field-state discovery, source-to-field mapping, deterministic validation, bounded correction, escalation, and audit tracing. We tested five synthetic healthcare administrative schemas, 1,000 source records, four interface variants, two data-quality suites, and six comparators, yielding 24,000 benchmark episodes. A separate strict-output audit evaluated direct mappings from Qwen2.5-1.5B and Qwen2.5-7B, and a trace-derived operational simulation covered 6,000 episodes. Under the evaluated synthetic benchmark conditions, full CLAIRE achieved 1.000 episode success, field accuracy, required-field completion, and dependency completion in both suites; removing validation reduced stress-suite success to 0.500. In the simulation, 100.0% of clean and validation-stress episodes reached a staff-reviewable draft, compared with 68.6% of escalation challenge episodes, unsupported cases were blocked. Scenario-based savings were 149.7-165.5 seconds per case, not observed staff times. The findings support schema-grounded, validation-first healthcare administrative automation in which language-model components assist mapping but do not authorize unsupported or consequential actions. 

---
# Adaptive Consistency Graph for Long-Horizon Agents 

**Authors**: Jiecong Wang, Hao Peng, Zhanyi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.32754)  

**Abstract**: Large language model agents can often make reasonable local decisions on short tasks, yet their performance degrades when success requires long sequences of dependent actions and tool calls. During execution, task requirements, historical evidence, and the current execution state may gradually become disconnected, so later decisions can drift from the original objective. We study this problem by introducing the Adaptive Consistency Graph (ACG) for long-horizon execution. ACG incrementally organizes execution evidence and its provenance in a persistent graph, then constructs a temporary requirement-centered view for each decision under a bounded context budget. Rather than replacing the base agent's planner or tool executor, ACG provides a structured and traceable context view for each decision. In the matched evaluation, ACG improves GPT-5.6-luna's average success from 44.5\% with ReAct to 50.2\%, with the largest gain on BrowseComp-Plus (73.5\% versus 62.4\%). We further analyze trajectory structure and inference cost to characterize this improvement. Our code is available at this https URL. 

---
# Retrospective Distillation Attribution via Normalized Response Similarity 

**Authors**: Minwoo Jang, Jaechang Kim, Minhyeon Oh, Jeongyeon Hwang, Jungseul Ok  

**Link**: [PDF](https://arxiv.org/pdf/2609.32749)  

**Abstract**: Model distillation transfers capabilities through supervised fine-tuning (SFT) on teacher responses, often collected from commercial APIs, raising questions of model provenance. Existing distillation attribution methods have been largely evaluated on students immediately after the SFT step. However, a distilled model may undergo further SFT, preference optimization, or reinforcement learning before release, while an auditor may lack access to the pre-distillation checkpoint required by reference-based attribution. To close this gap, we propose SCOUT, an output-only method that aggregates recurring *syntactic patterns* into candidate profiles, filters low-contrast patterns, and calibrates student--candidate distances against inter-candidate distances. SCOUT supports attribution and abstention using only current texts, without model weights, token likelihoods, or historical checkpoints. Auditing publicly released descendants of distilled models spanning diverse post-training objectives, SCOUT consistently identifies the distillation source. Furthermore, tracing teacher-associated *syntactic signatures* along training trajectories reveals that they emerge during distillation and persist through subsequent preference optimization and reinforcement learning. 

---
# Gradient-Guided Decoupled Adaptation for Geospatial Vision-Language Models 

**Authors**: Dongdong Wang, Deepak Balakrishnan, Ravi Srinivasan, Shenhao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.32737)  

**Abstract**: Existing geospatial vision-language models (Geo-VLMs) typically optimize diverse geospatial tasks through a unified multi-task adaptation paradigm without explicitly accounting for the heterogeneous optimization characteristics. Our empirical observations reveal heterogeneous gradient characteristics across tasks, including vision-language differences, intra-branch gradient relationships, and task interference, which hinder effective multi-task optimization. Motivated by these observations, we propose Gradient-Guided Decoupled Adaptation (G2DA), a gradient-aware optimization framework for multi-task Geo-VLM learning. G2DA first partitions tasks into vision- and language-centric groups through gradient-guided cross-modal decoupling. It then constructs modality-specific curricula based on task gradient similarity and employs bidirectional rehearsal to mitigate the recency effects introduced by sequential optimization. We evaluate G2DA on three Geo-VLM benchmarks using six InternVL3 and Qwen3.5-VL variants, along with GeoChat and GeoLLaVA. Across all 24 benchmark-model combinations, G2DA consistently outperforms representative baselines, improving over the strongest competitor by 3.08, 4.30, and 2.81 percentage points on UrBench-MCQ, XLRS-Bench-Lite, and VRS-Bench-VQA, respectively. These results demonstrate the effectiveness of gradient-guided task organization for Geo-VLM adaptation. 

---
# MassAlloc Attention: Let Attention Allocate Its Own Compute 

**Authors**: Jingze Shi, Zhangyang Peng, Xianduo Li, Yanlin Qi, Xiaotian Lin, Haoxian Chen, Liangdong Wang, Guang Liu, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2609.32712)  

**Abstract**: FullAttn often assigns negligible normalized mass to much of the causal score space, yet dense kernels execute the complete post-score path after forming each QK tile. We introduce MALA, a fused attention primitive that preserves score access to every legal causal interaction and uses normalized contribution to allocate post-score computation. Forward uses its evolving online-softmax normalizer, while backward reuses the finalized normalizer to derive nested retained support using only standard attention state. A common tolerance governs training and inference, allowing for adaptive retention of the work. MALA reduces low-contribution post-score computation. A matched-work study at 8K isolates the benefit of distribution-adaptive allocation: under exactly matched total post-score work, MALA approaches a per-instance reference-mass oracle, with mean omitted mass of 0.0188% versus 0.0182%. Across context lengths from 1K to 32K tokens, the same tolerance maintains low output and gradient errors relative to the reference. Across a broader controlled associative-recall comparison, MALA closely tracks FullAttn as context grows, reaching 89.67% accuracy at 8K compared with 89.97% for FullAttn. In an attention-operator benchmark at 128K tokens with tensor parallelism, MALA reduces forward and backward latency during training by 2.2x and 3.0x and decoding latency during inference by 1.6x relative to FullAttn. Across scaling-law training from 0.6B to 14B parameters, MALA closely tracks FullAttn in perplexity while reducing total training FLOPs. The resulting 14B models and 32B models from separate continued training achieve comparable knowledge, reasoning, and long-context retrieval scores to FullAttn. These results indicate that allocating post-score computation according to normalized attention contributions can retain the evaluated capabilities of FullAttn while reducing attention computation. 

---
# Despite Instructions: Frontier Agents Improvise Covert Channels at Test Time 

**Authors**: Jacob Dineen, Silei Ren, Muhao Chen, Dan Roth, Ben Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2609.32701)  

**Abstract**: In security-sensitive applications, language-model agents are often required to coordinate without disclosing confidential information. Yet repeated interactions may also let ordinary messages acquire shared private meaning. We study a repeated game with pairs of models in which the sender model observes one of four secret states and selects one of four summaries of the same public report, while the receiver model tries to infer the secret state. We find that model pairs can learn to communicate the secret using only one bit of feedback indicating whether the receiver inferred it correctly. This learning occurs during inference with fixed parameters and no supplied codebook or encoding examples. The effect also persists when agents generate their own free-form updates in a simulated incident-response task. Across ten independent games, pairs of GPT-5.6 Sol agents reach 98.8% final accuracy, compared with 25% chance, despite explicit instructions prohibiting disclosure and a monitor that screens each message without access to the agents' interaction histories. The same interactions that help agents cooperate can therefore allow confidential information to pass through messages intended for legitimate coordination. 

---
# IGSD: Environment-Verified Hindsight Self-Distillation for Search Agents 

**Authors**: Angqing Jiang, Gaoming Zhang, Chaoqun Zhang, Jianchun Song, Liyuan Kong, Kena Qi, Wei Lin, Defu Lian  

**Link**: [PDF](https://arxiv.org/pdf/2609.32694)  

**Abstract**: On-policy self-distillation densifies agent training without external teachers: a policy conditioned on privileged hindsight provides step-level guidance for its own unprivileged rollouts. For search agents, however, hindsight can make the teacher prefer a query that does not improve retrieval from the student's state. Existing methods either distill this preference directly or filter it with model-internal scores, but neither strategy verifies the query's executed retrieval consequence. We propose Information-Gain-Gated Self-Distillation (IGSD), which verifies on-policy token proposals with environment feedback before distilling them. Treating each query token as a micro-action, IGSD completes the teacher's token proposal and the student's sampled token into matched queries and executes both from the same failed state with the same retriever. Shared counterfactual controls account for query-conditioned shifts in answer likelihood, so their difference, the executed paired information gain, provides a relative utility contrast for the retrieved documents. IGSD uses this contrast as a positive-only soft weight for candidate-pair distillation, while leaving the GRPO objective unchanged and confining verification to training. Across seven single-hop and multi-hop QA benchmarks, IGSD reaches macro-average exact-match accuracies of 42.8% and 47.0% with 3B and 7B policies, respectively, without inference-time verification. These results support environment-verified hindsight as an effective approach to reliable action-level supervision for search agents. 

---
# The GUI Is Not the State: Diagnosing State Aliasing in GUI World Models 

**Authors**: Dongsheng Liu, Chao Jin, Wenkui Yang, Hejin Wang, Junwei Yang, Zeren Zhang, Ziwei Chen, Huaibo Huang, Jie Cao, Ran He  

**Link**: [PDF](https://arxiv.org/pdf/2609.32679)  

**Abstract**: GUI World Models (GUI-WMs) are increasingly used to predict future states for agent planning and simulation, yet most existing formulations condition only on the current GUI observation and action. We identify state aliasing, where the vis- ible interface omits transition-relevant environment state, so identical observable conditions can correspond to different valid futures. To diagnose this failure mode, we introduce StateAliasBench, a diagnostic benchmark that explicitly isolates such ambiguities via strict pairing. We further propose lightweight predictive- state recovery that infers structured state from history and augments otherwise frozen GUI-WMs through a deterministic state interface. Family-specific special- ists provide state recovery across heterogeneous state types, and multi-teacher dis- tillation consolidates them into a single unified estimator. Experiments show that existing GUI-WMs exhibit systematic failures under observation-only condition- ing, while predictive-state augmentation substantially restores state-sensitive pre- diction across evaluated WMs, preserves generative fidelity, and improves down- stream performance of GUI agents on AndroidWorld. These results suggest that reliable GUI world modeling should account not only for what is visible, but also for the hidden transition state that determines what happens next. 

---
# Distance-KV: Exploiting Relative Distance for Efficient Long-Context Inference 

**Authors**: Xianpeng Shang, Canbin Huang, Jiang Li, Tian Lan, Qianyi Cai, Xiaojun Quan, Xiangdong Su  

**Link**: [PDF](https://arxiv.org/pdf/2609.32663)  

**Abstract**: The memory usage and decoding latency of LLM inference grow rapidly with context length. To reduce these costs, key-value (KV) cache compression methods selectively retain cached states based on token importance or differences in attention patterns across heads. However, we discover that retrieval capability varies substantially with relative distance, even within the same attention head. To exploit this structure, we introduce Distance-KV, which learns a static KV retention pattern over the joint space of layers, attention heads, and relative distances. The pattern is learned offline with the language model frozen and reused across inputs to prune and compact the KV cache without online importance scoring. Across three backbone models and four long-context benchmarks, Distance-KV consistently achieves the best overall performance among competing KV cache compression methods, exceeding the strongest compression baseline by up to 9.3 points on RULER at 128K. On Llama-3.1-8B-Instruct at 128K, Distance-KV reduces KV cache memory by 65.4% and achieves a $1.66\times$ decoding speedup relative to Dense. Together, these results identify relative distance as an important structural dimension for understanding how LLMs retrieve information over long contexts and for designing more efficient inference methods. 

---
# Retrieved but Not Delivered: Multimodal Memory Delivery for Long-Term Agents 

**Authors**: Yuhang Jiang, Qingwei Liao, Kaize Yin, Xingling Liu, Luca Cuomo, Silvio Bacci  

**Link**: [PDF](https://arxiv.org/pdf/2609.32590)  

**Abstract**: Work on memory for multimodal agents optimizes what is written, updated and retrieved. Between retrieval and the answer, however, is a stage that multimodal memory evaluations do not isolate: what of the retrieved memory reaches the model, and in what form. We call it delivery, and a controlled decomposition on MemLens locates the remaining room there. With the retrieved evidence set exactly fixed, delivering the original pixels instead of withholding them raises accuracy by 13.87 points on an 8B backbone, whereas making retrieval perfect on those same messages improves it by 2.31. Delivery is the larger term on all three MemLens backbones and grows with backbone strength; retrieval grows too, without closing the gap. We propose DeliverMem, an instantiation of delivery as three decisions: keep the original modality, give each item a readable identity, and state when it was seen, with a retrieval-side adapter for the one property delivery cannot supply. Each is measured against a delivery-matched control that alters only its own variable. DeliverMem leads the strongest published memory agent on MemLens at all four context lengths, and beats DMV-Bench's own strongest method at every setting on both backbones. On MemLens it does this on a tenth to a seventieth of the input. Each decision helps only where the question lacks what it supplies, and is null elsewhere. A single fixed configuration nonetheless leads both benchmarks, without training any component or modifying the stored records. Project page: this https URL 

---
# CUE-Mem: Benchmarking Long-Term User Memory via Implicit Cues in Multimodal Conversations 

**Authors**: Yulin Hu, Yanyan Zhao, Zimo Long, Xing Fu, Mengtong Ji, Weixiang Zhao, Yutai Hou, Qianchao Wang, Dandan Tu  

**Link**: [PDF](https://arxiv.org/pdf/2609.32574)  

**Abstract**: Long-term memory is essential for multimodal agents that interact with users across sustained conversations. However, user memories are not always explicitly stated: they may also be implied by recurring background objects in images, ambient sounds in audio, or other peripheral multimodal cues. Existing benchmarks largely focus on text-only memory or explicit multimodal evidence, leaving implicit multimodal cues underexplored. We introduce CUE-Mem, a text-image-audio benchmark for evaluating long-term user memory from implicit cues. CUE-Mem contains 2,674 questions across explicit and implicit evidence settings and covers four tasks: Entity Recall, Long Pattern, Personalized Recommendation, and Answer Refusal. Across textualized memory systems, implicit performance remains far below oracle evidence, locating the main bottleneck in preserving and retrieving subtle cues rather than question answerability. Increasing caption detail recovers more of this evidence, but brings uneven gains and rapidly growing token costs, motivating native multimodal access. Yet native access does not uniformly resolve the bottleneck: evidence use depends strongly on the backbone, while multimodal indexing introduces substantial retrieval noise. CUE-Mem provides a testbed for memory systems that selectively retain, retrieve, and use subtle multimodal evidence. 

---
# Attribution Gaps in Zero-Training LLM+OVOD Pipelines: A Fine-Grained Analysis of the CAAP--SNAP Discrepancy 

**Authors**: Yu-Feng Yen  

**Link**: [PDF](https://arxiv.org/pdf/2609.32567)  

**Abstract**: LAOD and similar zero-training LLM+open-vocabulary-detector (OVOD) pipelines score two things separately: class-agnostic localization accuracy (CAAP) and semantic naming accuracy (SNAP). The two consistently diverge, and nobody has asked why. This paper asks why, on the full 5,000-image COCO-Val split (27,273 detections) rather than the small subset the original work evaluated on. Object visual complexity turns out not to be the driver -- small and occluded objects are, if anything, localized better than large ones. Vocabulary novelty is: once the LLM's wording falls outside the detector's native category set, localization accuracy falls from 80.9% to 31.6%. That drop is not spread evenly across unfamiliar phrasing, though. Almost all of it comes from cases where the novel wording actually names a different object than the one COCO annotated (true synonyms still score 89.3%; semantically unrelated "noise" labels score 12.0%). A closer look at a further failure subset tells a similar story: 78-88% of what looks like complete localization failure is really the model correctly finding a real object that COCO's non-exhaustive 80-category scheme simply never labeled, not hallucination. Swap the detector backbone (YOLO-World for Grounding DINO) or the LLM (Gemma-3 for Qwen2.5-VL) and both the effect and its rough size hold up, so this looks like a general property of the pipeline family rather than a quirk of one model pairing. The upshot is that a large share of the apparent CAAP--SNAP gap traces back to closed-category annotation limits rather than a real grounding failure, which matters for how we detect hallucination, analyze failure modes, and design evaluation for grounded multimodal systems meant to work in the open world. 

---
# Reading Is Not Leaking: Local, Auditable Measurement and Reduction of Inference Exposure from Public Footprints 

**Authors**: Mahmudul Faisal Al Ameen  

**Link**: [PDF](https://arxiv.org/pdf/2609.32565)  

**Abstract**: Anyone with a public footprint leaks facts that were never stated, and language models make the inference cheap. We present a framework for measuring and reducing this inference exposure that runs on the owner's own CPU with no language model at analysis time, instantiated on organisations and on individuals. It starts from a measurement result: scoring an inference system against the target's private truth conflates how well the system reads the record with how much the record leaks. On a 128-question instrument over sixteen synthetic firms, almost half of the questions are never answered correctly by any of six readers, four of them language models, and a majority-class guess accounts for most of every reader's score. We therefore separate reading accuracy from leakage rate and introduce an injection protocol that creates cells with known support. Our analyser combines rules, statistical solvers and a 106M-parameter encoder trained from scratch that marks verbatim evidence and never generates text; every answer carries a graded certificate whose recorded proof replays. Its certified answers are correct in 93% of resolved cases, against 49-73% for the language models' quote-backed answers, whose citations are produced alongside the answer rather than deriving it; with plain-prose articles in the record, 70% of its evidence-bearing answers rest on evidence that establishes them, against 18-56% for the models. A constrained defence that rewrites each fact's carrier as a true but coarser statement hides every single-carrier fact from four language-model adversaries at 40% lower edit cost than deletion. On sixteen synthetic people the guessing term is larger still, and a decoy planner with no language model halves the correct answers of the estimator it targets without transferring to a second. 

---
# Shared Autoregressive Context Can Distort Relationships in Synthetic Data 

**Authors**: Thomas S. Robinson  

**Link**: [PDF](https://arxiv.org/pdf/2609.32546)  

**Abstract**: Large language models can generate several records within one autoregressive completion, making earlier answers available as context for later records. This paper shows that such shared-completion batching can distort relationships among variables in the resulting synthetic data, using controlled tests on synthetic survey respondents. In a matched experiment on 2,000 European Social Survey profiles, generating ten rather than one respondent per request increases mean absolute error in within-country correlations by 48-58% for Qwen3.8-27B and 114-127% for Llama-3.3-70B-Instruct across three seeds, holding profiles, examples, questions and decoding parameters fixed. The distortion primarily reflects exaggerated relationship strength, while retaining substantial agreement with the human ordering of correlations. Controlled interventions establish answer history as a causal channel: re-pairing the same preceding values, with profiles and marginal distributions fixed, changes correlations among subsequently generated responses. Hiding preceding answers reduces correlation error in the tested settings but worsens marginal accuracy. Exploratory corrections across social-attitude, health and economic data likewise show that lower correlation error can coexist with worse marginal distributions and regression estimates. Request construction is therefore part of the data-generating process, and synthetic-data validity must be evaluated against the analyses the generated data are intended to support. 

---
# Do Audio LLMs Listen Before They Act? Diagnosing Acoustic-Context Gating in Voice Agents 

**Authors**: Yanjie Zhang, Nanchen Hu, Yushi Sun  

**Link**: [PDF](https://arxiv.org/pdf/2609.32536)  

**Abstract**: Audio language models can recognize spoken commands and invoke tools, but an agent must first decide whether the acoustic and conversational context warrants action. We introduce VGBench, a 1,018-item diagnostic benchmark for action-level addressedness across side-talk, self-talk, and speaker-switch scenarios. Each item uses a shared action space comprising silence, a tool call, and a natural-language answer. Speaker-switch pairs hold the specified words fixed while source, distance rendering, and a temporal boundary define a controlled wearer-to-bystander shift. Six raw Audio LLMs and three training-free adaptations often identify the target tool yet rarely withhold action under this shift; the highest raw switch mute rate is 14%. We then use VoxGate as a post-training case study. Supervised training mutes 91.3% of switched commands while choosing the correct tool for all nearby wearer commands and text-only controls. An exploratory GRPO stage has similar switch performance; side-talk accuracy rises from 68.4% to 70.9%, and self-talk muting from 52.0% to 60.0%. Factorized controls identify an independent source-change effect, while sensitivity to the far-field manipulation varies across acoustic renderings. The benchmark therefore measures multi-cue acoustic-context gating rather than isolated speaker identity. 

---
# Activation Flow: Manufacturing Activations for Steering 

**Authors**: Hong Kiat Tan, Linh Le, David Williams-King  

**Link**: [PDF](https://arxiv.org/pdf/2609.32530)  

**Abstract**: Difference-in-means steering requires activations recorded while a model shows the desired behavior, which a sandbagging model withholds by deliberately underperforming. We introduce Activation Flow (ActFlow), which manufactures these activations from $k$ correct labels without fine-tuning. ActFlow sets target logits that rank each labeled item's correct answer first, and moves the logits toward them by adding one vector $x$ to all $k$ residual streams at one layer. ActFlow is a family of ordinary differential equations for $x$, one for each rule that maps the required logit change to the velocity of $x$. The smallest-norm rule lands exactly on the targets, while the others keep only the top singular directions of the Jacobian. We test ActFlow on three instruction-tuned models, each locked by a sandbagging prompt and by a password-locked LoRA. At $k=40$, ActFlow keeping five singular directions raises the mean held-out ARC-Easy accuracy over the six locked models from $0.05$ to $0.85$, against $0.88$ for fine-tuning and $0.92$ for the honest models. Furthermore, it scores higher than the smallest-norm rule in 16 of the 18 combinations of locked model and $k$, and its steering direction is nearly orthogonal to the honest difference-in-means direction. It also unlocks two LoRA locks where the honest direction fails. 

---
# PULSE: Identifying Demonstration-Utility Features with Sparse Autoencoders 

**Authors**: Chenduo Hao, Chuanbao Gao, Pinjun Zeng, Jingze Zhu, Chonghan Liu, Zidong Liu, Xu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2609.32469)  

**Abstract**: In-context learning is highly sensitive to demonstration choice, yet most methods select demonstrations using external query-demonstration similarity. Such criteria can miss model-specific signals: Similar demonstrations may activate different internal features and downstream behaviors. We introduce PULSE (Paired Utility Localization over Sparse Encodings), an SAE-based framework for identifying model-internal features associated with demonstration utility and using them for demonstration selection. Using a small labeled discovery set, PULSE samples candidate demonstration sets, measures their zero-shot-relative utility under the target model, and scores SAE features by how their activation differences align with utility differences. The top positive and negative coordinates form a sparse utility-localization vector. We use this vector in two complementary ways: as a signed score for controlled complete-set ranking, and as PULSE-Retriever, which converts its magnitude into a feature-relevance mask for scalable pool-scale retrieval. Across classification, generation, and reasoning benchmarks, PULSE-Retriever improves over the strongest baseline by 2-3 accuracy points, 0.6-0.9 BLEU-4, and 3.2 exact-match points, respectively, while controlled ranking validates the identified features encode a predictive set-level utility signal. Feature inspection and cross-dataset experiments suggest that the identified features capture task-relevant, dataset-conditioned patterns, yet retain utility signals that partially transfer across datasets. Our code is available at this https URL. 

---
# ForkLeft: Entropy-First Rollouts for Prefix-Aligned Autoregressive-to-Diffusion Distillation 

**Authors**: Junming Liu, Jicheng Wang, Yifeng He, Hao Chen, Jianzhong Qi  

**Link**: [PDF](https://arxiv.org/pdf/2609.32448)  

**Abstract**: Autoregressive Next-Token Prediction (NTP) has enabled strong reasoning capabilities in language models, while Diffusion Language Models (DLMs) offer flexible token orders and parallel generation. We ask whether DLMs can acquire NTP-style reasoning through distillation without giving up their native generation process. Direct distillation, however, faces a fundamental mismatch: an autoregressive teacher predicts from a left prefix, whereas a DLM can condition on tokens on both sides. We introduce ForkLeft, a distillation framework that resolves this mismatch by separating the student's rollout from teacher supervision. During training, the student first performs entropy-first rollouts that commit uncertain positions and expose potential forks. We then fix the resulting student prefix and distill an NTP teacher under the same context, with answer correctness determining the supervision source. At inference, the student returns to its native confidence-first parallel decoding. With Qwen3-30B-A3B-Base, ForkLeft improves Efficient-DLM-4B on all ten benchmarks, raising MATH500 from 72.60% to 79.60% and consistently outperforming three alternative designs. The gains scale with teacher strength and generalize to SDAR-4B with only $500$ updates. At matched scale, the distilled 4B and 8B students exceed the published SDAR-Chat and OPDLM models on seven benchmarks, showing that DLMs can learn NTP-style reasoning without sacrificing native parallel generation. Code and datasets will be released upon acceptance. 

---
# Black-Box Auditing of Epistemic Reliability in Multi-Agent Debate Distillation 

**Authors**: Derui Wang, Zewei Shi, Rayne Holland, Ruoxi Sun, Xingliang Yuan, Jason Xue, Liming Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2609.32361)  

**Abstract**: Debate distillation adapts weaker verifiers using multi-agent debate transcripts to improve their judgement in subsequent debates, but gains on monitored tasks do not establish reliability on related unmonitored tasks. We study epistemic reliability degradation, in which adaptation preserves monitored performance while reducing support for correct responses on hidden tasks. We consider an adversarial debater that manipulates debate arguments while defending the correct monitored response, and ask whether the resulting degradation merely reflects catastrophic forgetting and whether standard evaluation can detect it. To address these questions, we propose ER-Audit, a two-stage black-box auditing framework that compares frozen verifier checkpoints before and after adaptation, and introduce two evaluation benchmarks pairing monitored and hidden task prompts grounded in shared contexts. ER-Audit searches for counterexamples to non-degradation by evaluating semantically valid paraphrases and, if none is found, uses independent paraphrases for sequential hypothesis testing. We derive anytime-valid lower confidence bounds on the non-degradation probability, allowing data-dependent stopping within a finite budget. We further establish a common lower bound across fixed paraphrase distributions and extend it to distributions within a bounded total variation distance of their mixtures. Our experiments show that higher hidden-task accuracy can coexist with more counterexamples to non-degradation and lower non-degradation bounds. This divergence challenges explanations based solely on broad catastrophic forgetting and shows that auditing can uncover selective hidden-task degradation concealed by aggregate performance gains. Our code and benchmarks are available at this https URL. 

---
# Fewer Tokens, More Self-Teaching: On-Policy Self-Distillation for Extreme Visual Token Reduction 

**Authors**: Junxian Li, Ruixuan Yang, Tianao Zhang, Tiange Xu, Weisheng Dong, Yulun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.32353)  

**Abstract**: Visual token reduction is an effective way to accelerate multimodal large language models (MLLMs), but performance deteriorates rapidly under extremely low token budgets. Existing work has explored both visual-token selection and training-based adaptation to reduced visual inputs. We take a step further by asking how a heavily compressed MLLM should learn from the states induced by its own generations. This setting naturally calls for on-policy self-distillation: a heavily compressed model is supervised on the states induced by its own generations, while its full-token counterpart serves as an information-rich teacher. Based on this insight, we propose LT-OPD, a training framework for extreme visual-token reduction. The student rolls out responses with only a small fraction of visual tokens, and a frozen full-token copy of the same MLLM provides distributional supervision along these student-generated trajectories. To stabilize on-policy learning when visual evidence is severely limited, we further introduce a budget-level curriculum that progressively decreases the token budget during training. Across nine benchmarks on Qwen3.5-4B, LT-OPD raises average retained performance under 5% visual-token retention from 68.6% to 82.3%, outperforming training-free, training-based, and reinforcement-learning baselines at the same budget. The gains transfer consistently to Qwen3.5-9B, GLM-4.6V-9B, and LLaVA-OV-1.5-4B. LT-OPD also reduces KV-cache usage by 85.2% and prefill FLOPs by 85.4% without additional inference overhead, demonstrating that on-policy learning can substantially recover capabilities lost to extreme visual-token reduction. 

---
# What Can a Leaderboard Certify? Compositional Controllability for Fair Evaluation and Training of Biomedical Literature-Review Agents 

**Authors**: Zhaowei Han, Xiang Zhang, Lingxiao Guan, Danqi Hu, Kai Liu, Kevin Chang, Jie Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.32318)  

**Abstract**: Leaderboards rank long-horizon agents by their final outputs. Yet a higher score alone does not establish whether two systems are comparable or which stage accounts for the difference. Unequal evidence, inputs, or budgets can affect scores, and statistical corrections do not remove this mismatch. We introduce compositional controllability to address these questions. A comparison window covers one stage, several stages, or the whole agent. Our central result bounds the gap between observed and controlled score differences using only nuisance outside the window. This yields an admissibility test applied before scores are inspected. Inadmissible comparisons are refused. For admissible pairs, an ordering is certified only when the score gap exceeds the combined sampling and nuisance radii; otherwise, it remains undecided. These decisions give each system a rank interval. We introduce BioLitBench, a benchmark of 2,042 biomedical articles represented as structured claim graphs. Among seven published pipelines, a conventional statistical analysis declares a winner in 14 of 21 pairwise comparisons. Yet the top-ranked system alone received the target review's bibliography. To isolate pipeline performance, our test requires matched inputs and a fixed backbone model. It refuses 11 of the 21 comparisons, including every comparison involving the top-ranked system. Seven of the 14 conventional conclusions fall within these refused pairs. The same comparison windows support stage-level training. We train SCRIBE on Qwen3.8-27B using rewards measured at each stage's exit. Under matched evidence, SCRIBE achieves a certified rank interval of [1,2], with certified advantages over all evaluated published pipelines and the evaluated Claude and OpenAI agents. Under same pool, SCRIBE matches the strongest published retriever and is certified above three published pipelines. 

---
# Agentsensus: Consensus-Compressed Shared Memory for Multi-Agent Story Worlds 

**Authors**: Yu Pan  

**Link**: [PDF](https://arxiv.org/pdf/2609.32297)  

**Abstract**: A agentic story world is a dynamic system simulating who learned what, when, and from whom -- yet the standard design gives each character a private memory stream. A shared event is therefore stored once per witness, large duplication will be incurred in terms of storage. We present Agentsensus, a story-world simulation framework in which there is an unified long-term memory. Records of the same event merge into one owned by all its witnesses, and semantically relevant memory records are linked. We evaluate on four worlds -- two classical Chinese novels, Hamlet, and a real-world conflict timeline -- run for 40 to 80 rounds against three per-character memory designs under an equal-granularity protocol. Agentsensus writes 22-44% fewer entries than the closest baseline and is the only design whose memory becomes shared (14-28% of records held by more than one character, some by 10) and linked (94-99%), at judged simulation quality indistinguishable or even better than the baselines. An ablation attributes this to the merge itself: disabling it multiplies the store by 3.1x and takes sharing to exactly zero. Sharing also compounds with the horizon rather than saturating early, rising 6% to 9% to 14% as one world is re-run at 10, 20 and 40 rounds. 

---
# Clarify the User or Verify the World? Uncertainty Routing for Proactive Agents 

**Authors**: Zhaofeng Li, Xuan Zhang, Xiaokui Xiao, Yang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2609.32255)  

**Abstract**: Tool-using LLM agents must decide not only whether additional information is needed, but also which source can resolve the uncertainty. Existing proactive approaches often specialize in either user clarification or environment verification, without explicitly determining the appropriate information source for each decision. We formulate this problem as uncertainty routing among ACT, CLARIFY, and VERIFY, and propose PROUR, a proactive uncertainty routing framework. PROUR decomposes action uncertainty into disagreement across plausible user-goal interpretations, which signals user-side ambiguity, and the entropy remaining within each interpretation, which signals missing world-side evidence. To acquire information from the routed source, a query generator is trained with a mode-conditioned information-gain reward, targeting user-goal identification under CLARIFY and next-action identification under VERIFY. On $\tau$-bench, PROUR achieves 28.17% average success rate across retail and airline, outperforming the strongest prior method by 4.57% while using 2.17 fewer interaction steps. The learned policy further generalizes to stronger task agents and transactional domains of $\tau^3$-bench without retraining, demonstrating the benefit of source-aligned uncertainty resolution for proactive agents. 

---
# HM-ROUTER: Joint Model and Harness Routing for Agentic Systems 

**Authors**: Hao Mark Chen, Royson Lee, Yasuyuki Okoshi, Dimitris Anastasiou, Wayne Luk, Hongxiang Fan  

**Link**: [PDF](https://arxiv.org/pdf/2609.32213)  

**Abstract**: Agent performance depends on both the underlying model and the harness that manages its tool use and execution. Selecting a suitable pair requires accounting for their compatibility, yet training samples may cover only a subset of the growing combination space. We introduce HM-Router, a routing method that jointly selects a model and harness for each query. It learns separate model and harness representations shared across routes, with an interaction term inspired by canonical polyadic (CP) tensor decomposition to capture how their compatibility varies with the query. This sharing allows training samples from observed pairs to inform predictions for unobserved combinations. We curate a benchmark from 12 public agent benchmarks, covering 293 routes, 73 models, and 25 harnesses. HM-Router exceeds the strongest evaluated learned baseline by 7.3 percentage points in mean routing accuracy and leads at all seven evaluated cost budgets on the six-benchmark subset. When 90% of routes have their training outcomes withheld, allowing unobserved combinations improves normalized accuracy by 15.8 points over restricting the same router to observed routes. HM-Router has also demonstrated training sample efficiency for new routes and components and generalization to unseen benchmarks. Our code and data are open-sourced at this https URL. 

---
# Checking Leakage Witnesses versus Certifying Bounded Non-Leakage 

**Authors**: Chao Feng, Burkhard Stiller  

**Link**: [PDF](https://arxiv.org/pdf/2609.32134)  

**Abstract**: When a language-model audit finds no leak, what is needed to certify non-leakage? We study guarantees over a declared prompt domain under an executable leakage criterion and decoding rule. For general bounded polynomial-time evaluators, a supplied leaking execution is polynomial-time checkable, while leak existence is \NP-complete and deterministic certification is \coNP-complete. Exact stochastic certification is $\coNP^{\PP}$-complete at every fixed rational cutoff in $(0,1)$. Restricting the computation can change these bounds. For example, certification is in \coNP\ when all randomness is a terminal draw from an efficiently computed finite probability table. Attention models admit polynomial-time certification when local dependency windows of logarithmic length precede one global head, given deterministic decoding, fixed vocabulary, exact rational weighted means, a direct binary affine readout and finite-automaton prompt domains. A construction with two global layers instead makes certification \coNP-complete over template domains, with one head per layer, polynomial width, logarithmic precision and an inverse-polynomial logit margin. Planted-secret experiments measure what finite audits miss relative to complete references. Among 30 secret--model-state pairs that leak under greedy single-prompt execution on their secret's 4,096-prompt domain, uniformly selecting 256 recorded evaluations per pair misses every leak for an expected $41.06\%$ of these pairs. Batched and single-prompt checks disagree on one complete-domain decision among all 48 fine-tuned pairs, while a same-order repeat reproduces every single-prompt output. These results distinguish computational conditions for certification from the coverage and execution conditions needed to interpret a negative audit. 

---
# On Evaluating and Improving Conversational Agents in Production 

**Authors**: Kasra Hosseini, Wen-Sen Cheng, Marco-Andrea Buchmann, Emir Mulabegovic, Weiwei Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2609.32092)  

**Abstract**: We present a framework for evaluating and improving a large-scale, multi-agent shopping assistant in production, and report lessons from its use. Offline evaluation of such a system faces three obstacles. (i) A logged conversation cannot be replayed against a modified system, because a different response changes every turn that follows. (ii) The unchanged system itself varies from run to run. Its LLM components are stochastic, and in product search the available products, their prices, and the customer's personalization signals change. (iii) Aggregate quality scores combine distinct behaviors, so they show that quality has changed but not which behavior caused the change. Our framework addresses each obstacle in turn. For a reported behavior, an Evaluation Harness generates targeted assertions and a fixed cohort of customer scenarios. It then reproduces the behavior in a local instance of the assistant through grounded user simulation. Instead of replaying the log, the simulator writes new customer turns conditioned on the recorded messages and context. Repeated runs of the unchanged system form a stored baseline. An Improvement Orchestrator turns the assertion results into hypotheses, implements each as an isolated modification, and compares it with the baseline using paired percentile bootstrap intervals over scenario-level differences. When an investigation ends, the harness may propose revisions to future evaluations, subject to human approval and without altering past decisions. We report production investigations with this framework. Assertion profiles showed which positions of a product carousel a failure affected, and repeated runs distinguished a real improvement from run-to-run fluctuation. Audits of the evaluation itself found a judge that lacked the evidence it needed and a model setting that was configured but not applied. 

---
# EngramRAG: Dynamic Usage-Weighted Topology and Synaptic Consolidation for Multi-Hop Agentic Memory 

**Authors**: Bhavyateja Potineni, Lohit Giri, Anu Jain, Vadim Kutsyy, Rajasekhar Pentakota  

**Link**: [PDF](https://arxiv.org/pdf/2609.32049)  

**Abstract**: As autonomous LLM agents are deployed across multi-session environments, conventional memory architectures suffer from Associative Blindness (inability to traverse multi-hop relational dependencies), Scaffolding Amnesia (temporal decay evicting core persona invariants), and Static Topology Stagnation (immutable graphs ignoring usage dynamics). Grounded in Complementary Learning Systems (CLS) principles, we propose EngramRAG, an adaptive memory architecture coupling a low-latency Waking State reflex with an asynchronous background Dreaming State consolidation cycle. EngramRAG introduces: (1) Usage-Modulated Personalized PageRank (U-PPR), where transition probabilities adapt via Hebbian plasticity to promote persistent entities into high-centrality Epistemic Macro-Hubs; (2) Consolidation-Activated Topology Decay (CATD), which scales retention half-life by topological load-bearing weight rather than wall-clock recency, protected by a cold-start grace period (N_grace >= 4); (3) Directed SUPERSEDES DAG filtering to suppress obsolete state during fact mutations; and (4) Triple-source hybrid retrieval fusing dense vectors, BM25, and U-PPR via dynamic Reciprocal Rank Fusion (RRF). Evaluating on all 1,982 QA pairs across 10 long-term conversations in the LoCoMo benchmark, EngramRAG achieves +38.9% relative improvement in Recall@5 (53.21% vs. 38.29%, p < 0.001) and +43.1% in MRR (0.4203 vs. 0.2937) over dense vector RAG, significantly outperforming Okapi BM25 (48.66%) and isolated static graph retrieval (8.50%). On temporal reasoning, EngramRAG reaches 62.33% Recall@5 (+16.67 points over dense vectors). In controlled mutation tests, SUPERSEDES suppresses split-brain hallucinations from 70.0% to 0.0%, while 90-day simulations show 100.0% scaffolding retention under a 26.21ms interactive retrieval reflex. 

---
# Amnesia by Design, Memory By Necessity: Persistent State for Document Intelligence 

**Authors**: Souhail Bakkali, Ayoub Merimi  

**Link**: [PDF](https://arxiv.org/pdf/2609.32041)  

**Abstract**: Modern Document AI reads contracts, extracts fields, reasons over tables, and grounds answers to page regions, then forgets everything. Processing an amendment the next day begins from scratch: no schema retained, no contradiction detected, no experience carried forward. This is a structural choice, not a scale failure: current systems are stateless functions. We call this the statelessness bottleneck. This bottleneck lies beyond parameter scaling, context extension, and retrieval augmentation: storage provides persistence and retrieval provides access, but neither consolidates observations into knowledge that improves future processing. This survey formalizes persistent evidence-grounded document state as a unifying framework, specifying the operations and invariants required to convert multimodal evidence into durable, provenance-linked state. We introduce a statefulness audit showing that ten representative benchmarks, coded against eight statefulness criteria, leave cross-session state evolution untested, and derive a longitudinal benchmark harness with five counterfactual metrics: Experience Gain, Cost Efficiency, Memory Harm, Forgetting Fidelity, Coverage Retention, to characterize the benefit, cost, risk, and governability of persistent document state. Document AI lacks mechanisms coupling persistent state to document-native structure, provenance, and temporal validity. The next era of Document AI will be defined by what systems retain across documents, sessions, and time. 

---
# Who Governs Data in the AI Era? A Computational Analysis of the U.S. Privacy Workforce in Job Postings 

**Authors**: Ramazan Yener, Muhammad Hassan, Masooda Bashir  

**Link**: [PDF](https://arxiv.org/pdf/2609.32030)  

**Abstract**: Privacy protection now spans legal, technical, and managerial duties, and demand for privacy professionals is growing across sectors. However, little is known about how employers define these roles. We analyze 1,143 U.S. privacy job postings from LinkedIn and Indeed. We examine job titles, salaries, competencies, certifications, education, experience, regulatory references, and AI-related language by using rule-based text mining. We also apply Topic Modeling (BERTopic) to the same postings and identify 18 latent themes which we grouped them into four categories. Our findings show that privacy roles are hybrid and they combine legal knowledge, technical skills, and interpersonal competence. Artificial intelligence appears in more than half of postings, with AI language spread across compliance, legal, governance and security themes. Our research indicates that AI governance responsibilities are often embedded within existing privacy roles, contributing to the rise of hybrid positions alongside dedicated AI governance roles. 

---
# Improving Audiovisual Speech Recognition through Synthetic Visual Data Augmentation 

**Authors**: Pol Buitrago, Pol Gàlvez, Javier Hernando  

**Link**: [PDF](https://arxiv.org/pdf/2609.31961)  

**Abstract**: Audiovisual Speech Recognition (AVSR) is a multimodal approach to speech recognition that incorporates visual information from lip movements to enhance model performance. Despite its advantages, its development remains constrained by the limited availability of labeled audiovisual (AV) datasets. This work explores the use of synthetic visual data as a solution, using an audio-driven talking-head pipeline to generate lip-synchronized visual content from existing audio data. We evaluate the effectiveness of synthetic visual data both as an augmentation strategy and as a standalone training resource, applying our approach to Spanish and Catalan. Our results show that augmenting real AV data with synthetic samples yields relative Word Error Rate (WER) reductions of up to 16.2%, demonstrating the potential of this approach. Moreover, we demonstrate that synthetic data alone can serve as a baseline for AVSR training in languages lacking AV datasets. These findings provide evidence that synthetic visual data can serve as a scalable solution to AVSR data scarcity, enabling broader language coverage. 

---
# Recipe-Matching, Not Equivalence 

**Authors**: Ali Habibullah, Mohammad Alshiekh, Yazan Alshoibi, Salman Khan, Naeemullah Khan  

**Link**: [PDF](https://arxiv.org/pdf/2609.31927)  

**Abstract**: MathNet-Retrieve asks a retriever to find, for a math problem, a document stating the same problem. An LLM under one fixed prompt writes each gold document and its near-miss distractors; LLM judges filter them. We call this procedure the "recipe", training on pairs built the same way "recipe-matching", and ask how much score it buys beyond the ability the benchmark claims to test. Two models from one base, matched in rows and settings, differ only in the training file: pairs written under the benchmark's published prompt by another vendor's LLM and judge, or computer-algebra-verified pairs with no LLM anywhere. The first leads by 45 R@1 points on the easy tier. By a non-LLM paraphrase control, half to two thirds of that gap comes from the pairs being LLM-written at all: LLM rewrites under two unrelated prompts, with the verified model's negatives, recover 30 and 22 of the 45 points; back-translations with the same negatives recover almost none. The remaining 15 to 25 points appear only under the benchmark's own prompt and vanish on real duplicates no generator wrote, the same problem in two languages. The hard tier rewards the recipe's pair structure, a deep rewrite against a minimal-edit near-miss: LLM rewrites alone score zero on it, attaching negatives unlocks it, and every negative that does so costs cross-language points; the sets scoring highest on it separate near-misses no LLM wrote worse than LLM rewrites with verified negatives. MELD also moves when a model trains on pairs built its way, without losing retention; on SABER-Math the registered attack fails, and the one gain, from its LLM-written summaries, is small but holds at a matched budget. Only on MathNet-Retrieve could we pin an inversion, benchmark score up and real retention down, to one edit of a training file. We release the generator-free duplicate evaluations, the near-miss test and three trained models. 

---
# Improving Medical Calculation of LLMs with Embedded Coding 

**Authors**: Tianshi Ming, Yingying Zhang, Xian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2609.31908)  

**Abstract**: Large Language Models (LLMs) perform well on medical examinations and question-answering benchmarks, but remain unreliable on medical calculation tasks that require exact numerical outputs. These calculations support high-stakes decisions such as medication dosing, organ-function assessment, and prognostic scoring, for which even small errors can have serious clinical consequences. We introduce MedCode, a framework that improves medical calculation by training LLMs to generate embedded executable code. Given a clinical context, the model identifies the relevant calculator, extracts its input variables, and produces a script that delegates arithmetic operations to a deterministic interpreter. Executing the script returns the calculated value together with an explanation and the appropriate unit. We construct supervised fine-tuning (SFT) and preference datasets from the MedCalc benchmark and additionally curate a dataset for calculation tasks in Intensive Care Unit (ICU) scenarios. We further propose weighted Direct Preference Optimization (wDPO), which adaptively emphasizes preference pairs that are difficult for the model to distinguish. Experiments with LLaMA3-8B, Qwen2.5-7B, and Mistral-7B show absolute accuracy gains of 20--30 percentage points, demonstrating the effectiveness of embedded code generation for medical calculation. 

---
# Choir: An Open Protocol for Distributed Multi-Agent Autoformalization 

**Authors**: Yidi Qi, Melanie Weber  

**Link**: [PDF](https://arxiv.org/pdf/2609.31903)  

**Abstract**: AI agents can now formalize entire textbooks and major theorems in proof assistants such as Lean, but current efforts are typically centralized: a single team runs all agents and bears the full computational cost. We introduce Choir, an open protocol for distributed formalization. Choir decomposes a project into tasks that can be completed by independent contributors, each running their own agent with their own LLM subscription, while coordinating entirely through the project's GitHub repository. To support open participation, every contribution is checked by a deterministic gate before merge. Choir supports Lean 4, Isabelle, and Rocq, and is open source and modular, allowing projects to replace individual components or extend the protocol. 

---
# NVAlign: Direct-Gradient Optimization for Non-Verbal Control in Continuous Autoregressive Flow Matching Text-to-Speech 

**Authors**: Qiaolin Wang, Pedro Sandoval-Segura, Anunaya Joshi, Edvardas Jurkonis, Jake Downie  

**Link**: [PDF](https://arxiv.org/pdf/2609.31892)  

**Abstract**: While modern text-to-speech (TTS) systems generate highly natural speech and support inline non-verbal vocalization (NVV) tags, accurate control over these events remains challenging. A key gap is the lack of established post-training methods for non-verbal control in continuous autoregressive flow-matching TTS. To this end, we present NVAlign, a direct-gradient post-training framework for NVV tag-following in this architecture. We first perform supervised fine-tuning (SFT) of TTS models and an NVV-aware automatic speech recognition (NV-ASR) model on NVV-annotated speech, then freeze the NV-ASR model to serve as the reward model for post-training. A two-step gradient surrogate enables efficient reward backpropagation through the flow-matching sampler to jointly update the autoregressive backbone and acoustic flow head. Fidelity penalties and reference-velocity regularization help preserve speaker similarity and speech quality. Results from NVV-SuperBench and human listening evaluations show that NVAlign improves tag-following accuracy over SFT and Flow-GRPO baselines. These findings demonstrate that direct reward-gradient optimization can improve non-verbal control in continuous autoregressive flow-matching TTS. Audio samples are available at this https URL. 

---
# CueKFS: Agentic Cue-Driven Keyframe Selection for Long Video Understanding 

**Authors**: Weitai Kang, Hanieh Deilamsalehy, Yumo Xu, Dewang Sultania, Serdar Cellat, Yan Yan  

**Link**: [PDF](https://arxiv.org/pdf/2609.31873)  

**Abstract**: Keyframe selection (KFS) has long produced compact video summaries for browsing and retrieval, and representative frames for thumbnails. More recently, when conditioned on a question, KFS provides an alternative to uniform sampling for long-video question answering by selecting frames that are more relevant to the question. Most methods rank frames by similarity to the question. Yet a relevant frame may score poorly when the question combines subjects or moments that no single frame shows, or requires implicit information absent from its wording. Other methods try to break down the question into subqueries, but suffer from inaccurate decomposition due to their static initial context. Therefore, we propose CueKFS, a training-free method that reformulates question--frame matching as comparing frames against a set of dynamically generated visual cues. From an initial set of salient frames, we decompose the question into cues. Each cue concurrently probes the video to navigate to its own evidence. A reasoning VLM then agentically revises the cue set against its evidence to re-explore the video. CueKFS then allocates the budget across the surviving cues. Across three benchmarks, CueKFS establishes state-of-the-art results in all 27 evaluated settings with available prior results, achieving budget-averaged gains of up to +4.54% over the previous baseline and a median of only two VLM calls. We further provide a detailed behavioral analysis of CueKFS, showing that agentic cue refinement drives active re-exploration of the video, yielding relative similarity gains of up to 92% over the initial context. 

---
# IndustryLLM: Failure-Driven LLM Training for Industrial Procurement 

**Authors**: Liang Ding, Zhiang Xu, Yuyang Sheng, Bin Chen, Songlin Bai, Run Zhu, Dingjun Wu, Hui Xu, Yandi Wang, Fulin Shi, Leilei Gan, Linlin Yu, Qihuang Zhong, Keqin Peng, Yalong Li, Chengfu Huo  

**Link**: [PDF](https://arxiv.org/pdf/2609.31871)  

**Abstract**: Industrial procurement requires language models to bridge informal buyer jargon, sparse marketplace attributes, and authoritative engineering standards under strict safety tolerances. We present IndustryLLM, an open-weight industrial language model trained from Qwen3.5-35B-A3B-Base (35B total parameters with ~3B activated per token, with the vision encoder frozen). Rather than relying on generic text scaling, we introduce a failure-driven adaptation recipe spanning continued pre-training (CPT) and supervised fine-tuning (SFT). CPT leverages a curated ~100B-token corpus integrating 5B tokens of national standards (e.g., GB/T) and technical archives, 10B tokens of de-identified real-world industrial transaction and inquiry records, and 60B tokens of general replay. To overcome register mismatch and factual brittleness, we systematically reconstruct an estimated 20B-token domain subset via multi-register rewriting across 10 genres and 8 writing styles, confidence-routed minimal factual editing, and error-targeted QA synthesis (resolving colloquial typos like '42-luo-mu' -> 42CrMo, expanding ambiguous codes like '16674' -> GB/T 16674, and clarifying conflicting dimensional specs). For downstream deployment, we formalize an evidence-gated constraint-evaluation interface enforcing three-valued logic where unverified product evidence remains unknown rather than satisfied. Offline evaluations demonstrate consistent gains on procurement-query structuring (+2.97 percentage points in exact match, 95% CI [2.11, 3.86] in No-Think mode), while randomized online A/B experiments in production yield substantial improvements (+4.25% GMV, +8.3% satisfied inquiries) alongside a latency reduction from 6-7 s to 1.5 s. Model weights and configs are released at this https URL. 

---
# Optimal transport meets speech: a tutorial review 

**Authors**: Xugang Lu, Yu Tsao  

**Link**: [PDF](https://arxiv.org/pdf/2609.31787)  

**Abstract**: Optimal Transport (OT) provides a principled framework for comparing and transforming probability distributions while preserving geometric structure. Recently, OT has gained significant attention in machine learning due to its ability to measure discrepancies between distributions, even when their supports do not overlap, making it effective for tasks such as generative modeling, domain adaptation, and transfer learning. Despite its success in fields such as computer vision and natural language processing, OT remains relatively underexplored in speech research. Speech signals present unique challenges, including temporal dynamics, speaker variability, noise, reverberation, and heterogeneous multimodal representations involving audio, text, and visual information. These factors often lead to distribution mismatches, where OT offers a natural framework for alignment and interpretation. This work aims to promote broader adoption of OT in speech processing by: (1) reviewing OT foundations through intuitive physical interpretations and highlighting connections to modern generative models; (2) presenting computational algorithms suitable for deep learning frameworks; and (3) demonstrating OT applications in cross-domain and cross-modal speech tasks, including speech enhancement, automatic speech recognition, language and speaker recognition, and audio spoof detection. We highlight OT's strong potential for addressing distributional variations in real-world speech applications. 

---
# Robot Manipulation with GPT-6-Astra: Body Knowledge, Experience Reuse, Emergent Skills, and Sim2Real Transfer 

**Authors**: Sida He, Lingxi Xie, Yunning Cao, Pengfei Chen, Kaiwen Duan, Jiannan Ge, Xinyue Huo, Jiacheng Shao, Qi Tian  

**Link**: [PDF](https://arxiv.org/pdf/2609.31770)  

**Abstract**: General-purpose multimodal agents can write robot-control programs, but repeated exploration and model-mediated action selection can make execution slow. We study how external body knowledge, successful experience, and executable skills improve an XLeRobot controlled by GPT-6-Astra in a simulated and a physical elevator-button task. In 30 fixed-start simulation trials, complete robot geometry and camera information reduce mean completion time by 57.4% relative to a baseline with only the common control interface and no prior experience; images with synchronized action and state records reduce it by 68.6% without additional body assets. In nine paired comparisons (18 trials) at starts displaced by 10-100 cm, experience recorded at the original start reduces mean time by 58-63% relative to no experience, demonstrating generalization to the tested new starting positions. During experience experiments, GPT-6-Astra spontaneously generates a short visual-feedback program. Researcher-refactored versions reduce mean local-task time by 29-31% in 27 simulation trials. Finally, 12 real-robot trials using operator-confirmed button contact demonstrate sim2real reuse: at a shared nominal start, simulation XML assets and simulation experience reduce mean time by 53.0% and 49.9%, respectively; real experience also transfers to two new starts. These results suggest a practical way to build general-purpose manipulation experiments around GPT-6-Astra: supply machine-readable body descriptions and synchronized demonstrations, and turn useful agent-generated feedback routines into reusable skills, while the agent adapts actions from current images. We release all task prompts, trial-level experimental data, and acquired skill implementations at this https URL. 

---
# Distributional Metrics for Evaluating Spoken Conversational Systems 

**Authors**: Shree Harsha Bokkahalli Satish, Erica Cooper, Patrícia Schmidtová, Maike Züfle, Éva Székely, Nicholas Sanders, Ondřej Klejch  

**Link**: [PDF](https://arxiv.org/pdf/2609.31719)  

**Abstract**: Evaluating conversational systems is a difficult and unresolved problem. We introduce the Conversational Distribution Score (CDS), which compares distributions of conversational behaviour using human conversations as a reference. CDS describes speech rate, syllabic rhythm, and turn interaction through eight interpretable features plus a separate two-feature semantic baseline. We compare conversations with two reference scales: one based on conversational success within human dialogue and another contrasting human and synthetic dialogue. Using listener judgments from out-of-domain goal--oriented dialogues, we examine system ranking, preferences between conversations, and ranking stability. Composite CDS recovers five of six listener system comparisons while individual features show strong correlation with listener preferences between conversations. We examine how many minutes and conversations are required before rankings stabilize. These findings support distributional comparisons as a complement to specific interactional metrics to evaluate conversations and conversational models while showing their interpretable value. 

---
# MM-VeriRec: Failure-Guided Fusion for Verifiable Agentic Multimodal Recommendation 

**Authors**: Yufeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.31718)  

**Abstract**: Images often carry recommendation constraints that text metadata only hints at. A movie may need to look dark, a product may need a minimal style, and a visually impossible request should be rejected. Agentic multimodal recommenders must reason over text-image evidence, decide when visual evidence is decisive, and abstain when no valid action exists. We introduce MM-VeriRec, a verifiable multimodal recommendation protocol and failure-guided fusion method for hidden visual constraints, image-text mismatch, and impossible-task abstention. MM-VeriRec builds tasks from real movie-poster and product-image datasets, verifies each recommendation with deterministic visual attributes, and converts failures into actionable labels: text-trap following, visual ignorance, and false acceptance. Fusion should not merely concatenate modalities, but should diagnose which modality failed and route to the appropriate repair. Across MM-ML 1M and Amazon Reviews datasets, stronger text and vision embeddings improve retrieval but do not remove these failure modes, whereas failure-guided fusion does. The adaptive attribute gate reads the same tags the verifier checks and its scores are verifier-aligned upper bounds testing whether the taxonomy routes to the correct repair. More informative is transfer under a non-aligned gate: an independently derived leave-one-out CLIP detector still reaches 0.7028 and 0.6111 visual-grounded success, above both a VBPR baseline and plain fusion. The text-versus-visual gap reproduces across two LLM families, and the repair that helps differs by domain. MM-VeriRec is both a benchmark and a practical diagnostic loop for trustworthy agentic multimodal recommendation. 

---
# MDL-Calibrated Significance-Gain Pair Encoding: Replication-Aware Automatic Stopping for Subword Tokenization 

**Authors**: Azam Nouri  

**Link**: [PDF](https://arxiv.org/pdf/2609.31705)  

**Abstract**: Byte-Pair Encoding (BPE) constructs subword vocabularies through greedy pair merging, but conventional BPE requires the number of merges or target vocabulary size to be specified externally. Significance-Gain Pair Encoding (SG-BPE) replaces frequency-only selection with a statistical criterion based on how strongly an observed pair exceeds its expected co-occurrence under an independence model.
This paper introduces MDL-Calibrated Significance-Gain Pair Encoding (MDL-SG), a three-stage procedure separating discovery, replication, and utility. Candidate pairs are ranked by Significance-Gain on a discovery partition, tested for replication on a separate partition using an exact one-sided hypergeometric test with per-iteration Benjamini-Hochberg correction, and then evaluated on a utility partition using a Minimum Description Length (MDL) criterion. Merging stops automatically when no replicated candidate yields positive held-out MDL gain.
On WikiText-103, MDL-SG stops at 209, 433, and 847 merges for 120K, 250K, and 500K-character tokenizer-training samples, respectively. At 500K characters, it selects a stored vocabulary of 1,017 tokens without prescribing the vocabulary size in advance. In a compute-matched TinyGPT experiment with identical 2,024,448-parameter models and 500 optimizer updates per language model, MDL-SG achieves validation/test BPC of 3.2612/3.2436, compared with test BPC of 3.2894 for SG-BPE and 3.3493 for frequency BPE. Frequency BPE achieves stronger raw compression, while MDL-SG achieves lower BPC, showing that compression-oriented merge selection and language-model utility need not coincide. 

---
# When Keywords Drop but Classifiers Hold: Soft Refusals under KV Cache Compression 

**Authors**: Kang Chen, Xiuze Zhou, Hong Chen, Yuanguo Lin  

**Link**: [PDF](https://arxiv.org/pdf/2609.31678)  

**Abstract**: KV cache compression is widely used for long context LLM inference under memory constraints, while deployed systems typically score refusals after generation with keyword filters or learned classifiers. Such monitors are intended to indicate whether a model declined a harmful request under the serving regime actually used. However, it remains unclear whether matched compression that preserves task accuracy also preserves agreement between lightweight lexical monitors and stronger refusal classifiers. We study this with a paired protocol on n=200 harmful prompts with a long filler context: each prompt is answered once under full retention and once under matched eviction after a shared prefill, and the same replies are scored by keyword heuristics, the HarmBench Llama-2-13B classifier, an auxiliary LLM judge, and humans on disagreements. On Qwen2.5-3B, keyword refusal falls from 98.0% to 80.5% (McNemar p~1e-8) while classifier refusal stays near ceiling (99.0%-99.5%) and MMLU accuracy is unchanged (50.0%); human labels predominantly follow the classifier, consistent with soft refusals. The gap is not universal and weakens under short fillers and paired SnapKV, so safety auditing under compression should rely on several judges matched to the serving context rather than on keyword rates alone. 

---
# PalmLeaf-VQA: A Multi-Script Visual Question Answering Benchmark for Historical Palm-Leaf Manuscript Understanding Across Diverse Regions 

**Authors**: Nimol Thuon, Jun Du, Panhapin Theang  

**Link**: [PDF](https://arxiv.org/pdf/2609.31651)  

**Abstract**: Historical manuscripts remain largely absent from modern vision-language benchmarks, leaving open how well multimodal large language models (MLLMs) handle culturally diverse, degraded, and non-Latin document images. We introduce \textbf{PalmLeaf-VQA}, a multi-script visual question answering benchmark for historical palm-leaf manuscript understanding across South and Southeast Asian traditions. PalmLeaf-VQA contains \textbf{923 curated manuscript images} and \textbf{7,384 question--answer pairs} from eight collection groups: Balinese, Grantha, Jathakam, Kambaramayanam, Kannada, Khmer, Sundanese, and Tamil. Unlike recognition-oriented resources, the benchmark targets manuscript-aware visual reasoning over preservation-relevant cues, including physical condition, line structure, material and coating, binding holes, margins, symbols, drawings, and localized visual artifacts. We evaluate recent proprietary and open-weight MLLMs under open-answer and constrained-answer prompting and provide fine-grained analysis across collections, question categories, and task types. The strongest evaluated model reaches only \textbf{58.00\% exact-match accuracy} on the held-out test split, revealing substantial limitations in current MLLMs for rare-script, degraded-layout, and preservation-oriented document understanding. PalmLeaf-VQA provides a standardized benchmark for advancing culturally grounded and layout-aware multimodal document analysis. 

---
# From Hand-Crafted to LLM-Based Variation Operators in Metaheuristics: A Tutorial 

**Authors**: Camilo Chacón Sartori, Guillem Rodríguez-Corominas, Christian Blum  

**Link**: [PDF](https://arxiv.org/pdf/2609.31649)  

**Abstract**: Large language models (LLMs) are increasingly being employed as variation operators in metaheuristics, generating or modifying candidate solutions, heuristics, or programs inside iterative search loops. This shift reframes variation as a model call conditioned on different types of information. We introduce an operator-level framework with two descriptors: (1) the type of prompt-conditioning information at variation time (\texttt{Numeric}, \texttt{Symbolic}, \texttt{Linguistic}), and (2) artifact persistence, identifying what survives the model call (\texttt{Transient}, \texttt{Amortized}, \texttt{Transfer}). The tutorial shows how to classify, build, and select these operators through a worked build template, a method survey, an evidence table, and a cost-aware decision guide. 

---
