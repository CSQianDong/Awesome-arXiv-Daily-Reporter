# Disentangling Representation Evolution in Transformers through Directional Decomposition 

**Authors**: Shwai He, Haichao Zhang, Shen Yan  

**Link**: [PDF](https://arxiv.org/pdf/2609.15975)  

**Abstract**: Transformer representations evolve through learned additive transformations that either preserve their current direction or redirect it. We study this evolution as a functional geometry, decomposing learned updates into parallel and perpendicular components. Across pretrained models, we find substantial parallel components beyond the residual identity path. We then apply the decomposition in two spaces: to attention and MLP updates relative to the hidden state, and to attention value aggregation relative to the current token's value. Targeted edits reveal a strongly space-dependent asymmetry: exclude-self value-space parallel manipulation is markedly more robust than residual-space and perpendicular counterparts, preserving the direct self message while scaling only the non-self aggregate. The same decomposition gives a component-resolved description of compression-induced update error: perpendicular error separates compression methods more clearly than parallel error. Extensive experiments further demonstrate that full-aggregate parallel suppression during from-scratch pretraining lowers validation-loss trajectories and improves downstream averages, with the value-space variant strongest. Together, these results connect representation geometry to editing robustness, compression diagnosis, and training-time intervention. Code is available in the \href{this https URL}{project repository}. 

---
# Discovery Foundation Models: Toward Open-Ended Discovery Intelligence 

**Authors**: Ling Yang, Zhenfei Yin, Yingcheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2609.15973)  

**Abstract**: Foundation models have progressed from learning and reasoning over existing knowledge, to increasingly learning through action, tool use, and outcome feedback. We argue that the next frontier is a further transition: from solving and acting within problems specified by humans to participating in the process by which new problems, representations, explanations, and knowledge are created. We refer to this capability as Discovery Intelligence. We formulate Discovery Foundation Models (DFMs) as general-purpose model systems for open-ended discovery. A DFM operates over a revisable research state and supports seven coupled capabilities spanning problem discovery, formulation, representation construction, hypothesis formation, intervention, evidence-grounded revision, and continual discovery improvement. We instantiate this framework with Zetema, which couples explicit research-state dynamics, verification and experimental gating, external grounding, and cross-task Discovery Skill evolution. We further ground the framework with GALILEO, a real therapeutic-discovery system in which Dry-Lab reasoning, robotic and hands-on Wet-Lab experimentation, external biological evidence, and iterative hypothesis and design revision form a closed physical discovery loop. We then formulate a unified approach to capability formation and process-centered evaluation, enabling discovery behavior to be trained, improved, and measured beyond final-answer performance. Together, these components establish discovery as a learnable, executable, and evaluable capability of foundation-model systems. We view this shift as a broader progression in intelligence scaling: from learning over existing knowledge, to learning from action outcomes, and ultimately to participating in the construction, testing, and revision of the structures through which new knowledge is discovered. Code: this https URL 

---
# Mind2Dialogue: Training Human-Aware Language Models by Simulating User Mental States 

**Authors**: Zixuan Wang, Yufan Zhou, Jinzhou Tang, Xinle Yu, Chengjun Wu, Lyumanshan Ye, Zhaoxiang Feng, Letian Peng, Adyasha Patra, Fan Bai, Enze Ma, Zhengding Hu, Jianyang Gu, Zhao Wang, Yufei Ding, Jingbo Shang, Tianmin Shu, Zhiting Hu, Zhen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.15972)  

**Abstract**: As language models become more capable, long-term collaboration in learning, reasoning, and decision-making calls for a deeper understanding of the people they serve. Yet training such human-aware language models faces a fundamental supervision gap because current datasets for LLM assistant training contain few if any well-informed responses explicitly grounded in users' unspoken beliefs and goals. Scaling such supervision is inherently constrained, as users' underlying states are not directly observable. We thus propose the Mind2Dialogue framework to mitigate this gap by simulating users' mental states and turning them into privileged supervision for human-aware training. Specifically, we first propose a psychology-guided simulator that preserves personal characteristics while updating mental states through interaction to generate coherent conversations. The key idea is to enforce a shared evolving mental state that drives user behavior and guides an Oracle assistant's responses. Our privileged distillation then trains models on the Oracle's well-informed responses to assist users without direct access to their mental states at deployment. Moreover, we propose to evaluate human-aware learning by combining personalization and theory of mind, examining how models understand people and act on that understanding. Training on the full Mind2Dialogue corpus improves every reported personalization metric over the corresponding Qwen, Llama, and OLMo instruction-tuned baselines, including gains of 26.6 to 40.9 percentage points in preference-following generation. The gains extend to belief and action reasoning on Qwen and Llama, beyond personalized assistance. Looking forward, Mind2Dialogue makes user simulation a foundation for genuine AI collaborators that understand beliefs and intentions behind people's words and support their long-term goals across education, work, and everyday life. 

---
# Verifiable by Construction: Claim-Level Evaluation of Verbatim Citation in Clinical Question Answering 

**Authors**: Jiashuo Zhang, Yuling Chen, Yvonne Commodore-Mensah, Michael Oberst  

**Link**: [PDF](https://arxiv.org/pdf/2609.15964)  

**Abstract**: Large language models (LLMs) have been widely adopted for clinical question answering (QA). Current systems can attach citations to their answers, but these often point to broad texts, leaving time-pressed clinicians unable to verify them efficiently. An alternative is to ensure that responses are verifiable by construction: providing fine-grained verbatim quotes from reference material that substantiate claims, so users can verify an answer without opening other documents. In this paper, we evaluate the ability of current models to perform this task end-to-end: from providing citations for every factual claim, to producing verbatim quotes, to ensuring that those quotes fully substantiate the claims. To do so, we build a standardized harness over four clinical practice guidelines and evaluate twelve LLMs on 222 synthetic clinical questions, measuring each of these stages separately. We find that most models can attach verbatim quotes to over 90% of their claims from prompting alone, apart from some lightweight models such as claude-haiku-4.5. Yet these quotes often fail to substantiate every detail of the claims they accompany. For instance, claude-opus-5 produces verbatim quotes for 98.0% of its claims, but fully substantiates only 37.1%. Our work provides insights into the current capability gap of LLMs in building verifiable clinical QA systems, along with artifacts for future research. 

---
# HypoEvolve: Genetic Algorithms Enable Multi-Agent LLMs to Discover Scientific Hypotheses 

**Authors**: Jieyuan Liu, Mengzhou Hu, Jefferson Chen, JungHo Kong, Pratibha Jagannatha, Yiming Gao, Dexter Pratt, Hsin-Yuan Lee, Zhiting Hu, Trey Ideker, Wei Wang, Eric P. Xing, Zhen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.15938)  

**Abstract**: Scientific agents contribute to hypothesis discovery by synthesizing evidence, assessing proposals, and developing new explanations. Recent systems combine scientific agents with evolutionary search through critique, comparison, and revision. However, how different forms of agent collaboration affect hypothesis quality remains an open question. Answering this question requires separating the effects of agents' scientific capabilities from those of their collaboration. A framework must therefore preserve agents' scientific roles and support rules for combining, revising, and retaining hypotheses. Building on this view, we introduce HypoEvolve, which makes collaboration explicit through successive updates to a hypothesis population. Specifically, we propose a generational genetic algorithm to coordinate specialized large language model (LLM) agents that integrate mechanistic arguments, reconsider assumptions, and assess evidence and testability. Each generation specifies how scientific judgments and new proposals reshape the population, making collaboration effects on hypothesis quality directly testable. Moreover, we design our evaluation around scientifically meaningful hypotheses that explain how a proposed intervention could work. Drug repurposing links these explanations to target-level biological claims assessed against external evidence. Specifically, we adapt DepMap and Open Targets into complementary external measures grounded in experimental, genetic, and clinical evidence. Across 34 cancer types, HypoEvolve achieves the highest scores against six baselines on both measures. DepMap selectivity reaches 0.171, versus 0.115 for the strongest baseline. Gains over single-pass generation also generalize to held-out cancer types. HypoEvolve advances a vision of autonomous science in which AI research teams achieve a capacity for discovery beyond that of individual models. 

---
# Inoculation Midtraining with Learned Neologisms 

**Authors**: Kyle O'Brien, Edward James Young, Puria Radmard, Nathalie Kirch, Cameron Tice, Tomek Korbak, David Demitri Africa  

**Link**: [PDF](https://arxiv.org/pdf/2609.15886)  

**Abstract**: Large language models (LLMs) often learn both desirable and undesirable properties during post-training. We study whether midtraining, an earlier training stage, can shape which of these properties later generalise. We introduce Inoculation Midtraining, a technique that teaches a base model that unsafe behaviour belongs to a designated <quarantine_token> context, as indicated by the <quarantine_token> neologism (a new token) introduced during midtraining, and then post-trains the model on unsafe data within that context. We then evaluate the model outside the context, with the <quarantine_token> neologism excluded from the system prompt. Across supervised fine-tuning and reinforcement learning post-training regimes, we find that Inoculation Midtraining can reduce misalignment while preserving the transfer of benign data properties (e.g., speaking in German or Shakespearean prose). However, our approach does not outperform standard Inoculation Prompting, is sensitive to training configuration, and produces a leaky boundary that nearby contextual cues can reactivate. These results show that inoculation with a learned association introduced via midtraining can shape selective generalisation. Still, more work is needed before this approach can become a load-bearing component in a developer's safety framework. 

---
# K-Bench: a clinically calibrated benchmark for evaluating large language models in high-risk mental health conversations 

**Authors**: Laura M. Vowels, Matthew J. Vowels, Shivali Sharma, Apoorv Jha, Rehnuma Choudhury, Wasseem El Sarraj, Rachel Francois-Walcott, Aruba Hussain, Sarah Ingram, Angela Loulopoulou, Adva Segal, Elena Volkova  

**Link**: [PDF](https://arxiv.org/pdf/2609.15855)  

**Abstract**: % !TEX root = ../main.tex People increasingly use large language models (LLMs) for mental health support, yet their safety in evolving, high-risk conversations remains poorly characterised. We developed K-Bench, a clinician-calibrated, protected benchmark evaluating 125 model configurations representing 33 base models from 14 providers across a fixed cohort of 200 multi-turn vignettes involving suicide, self-harm, domestic violence, substance misuse, and no-risk presentations. Synthetic patient conversations showed substantial distributional overlap with real human-AI conversations. A frozen GPT-4o judge achieved 94.2% exact agreement with clinician consensus across 6,751 eligible item comparisons from 151 clinician-rated transcripts. Leading models combined strong supportive conversation with combined-risk scores above 95, whereas risk exploration exposed substantial variation among lower-performing configurations. Therapeutic prompting produced configuration-specific gains concentrated among weaker models, while elevated reasoning produced no average improvement. K-Bench combines broader clinical coverage and configuration-scale comparison with a continuously updated public leaderboard whose operational test materials are protected from direct optimisation. The leaderboard is available at this http URL. 

---
# Learning to Coach for Experiential Learning 

**Authors**: Guanheng Chen, Tianzhu Ye, Li Dong, Xun Wu, Shaohan Huang, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2609.15851)  

**Abstract**: Language models can learn from experience, but raw solution trajectories are often too long and noisy to provide effective guidance. In this work, we propose Learning to Coach (L2C), a framework that trains a dedicated LLM-as-a-Coach to extract actionable experiential knowledge from an actor model's previous trajectory. The actor remains frozen, while the LLM-as-a-Coach is trained to maximize a reward given by the correctness of the actor's guided response. We study two such rewards: a same-instance reward, which improves subsequent responses on the original problem, and a cross-instance reward, which elicits knowledge that transfers to other instances. Across mathematical reasoning and interactive text-games, L2C consistently outperforms self-refinement and an untrained LLM-as-a-Coach. Running experiential learning for more iterations further improves accuracy and uses additional inference compute more effectively than enlarging the actor's decoding budget. The trained LLM-as-a-Coach also transfers to out-of-distribution tasks and adapts its guidance to the specific actor it coaches. 

---
# Before You Poll with LLMs: A Deliberative Diagnostic Framework 

**Authors**: Ahmed Wali, Hassaan Tayyab  

**Link**: [PDF](https://arxiv.org/pdf/2609.15849)  

**Abstract**: Can LLMs reason through new information like humans, or do they merely retrieve cached opinions? This is critical for silicon sampling, where LLM personas simulate public opinion at scale. Current evaluations test only whether personas hold the right opinions -- a static snapshot. But opinion research increasingly depends on dynamic fidelity: whether personas update beliefs in response to new arguments, as humans do during deliberation. No existing benchmark tests this. We introduce the Deliberative Polling Diagnostic Framework, which compares human and LLM belief shifts after identical informational interventions. Grounded in deliberative polling, it surfaces failures invisible to static evaluation: models that produce plausible partisan opinions can still misrepresent how those opinions change. Applying the framework to five frontier models using data from America in One Room (526 personas, 72 questions), we find that every model fails, each in a unique manner. GPT-5.1 exhibits reversal: its personas become more hostile toward the opposing party after balanced information, while humans become less so. This reversal is selective (80% on outgroup vs. 26% on policy questions) and symmetric across partisan identities. Gemini 2.0 Flash, Claude Sonnet 4.5, and Llama 3.3 70B exhibit overshoot, shifting correctly but at 5-7x human magnitude. DeepSeek V3 exhibits rigidity with near-zero change. Targeted ablations reveal that policy content triggers these failures and that they are identity-specific: GPT-5.1 reverses on outgroup questions but overshoots on ingroup; Gemini shows the inverse. We term this signature self-sycophancy: conformity to the model's internal stereotype of the persona rather than reasoning from the information provided. Our framework offers a concrete protocol: run the deliberative diagnostic before trusting LLM personas to mimic revised beliefs. 

---
# CiteGuard-RAG: A Validation-Centered AI System for Evidence-Grounded Question Answering 

**Authors**: Sumit Barua, Guan Hong, Halil Dursunoglu, Charles Rodgers, Alvis Fong  

**Link**: [PDF](https://arxiv.org/pdf/2609.15830)  

**Abstract**: Retrieval-augmented generation (RAG) can improve access to complex information; however, retrieving evidence alone does not ensure that answers are grounded, citation-valid, or appropriately refused. This paper introduces CiteGuard-RAG, a validation-centered AI system for evidence-grounded question answering. The system integrates hybrid semantic-lexical retrieval, citation-constrained generation, sentence-level grounding validation, and single-pass regeneration. Validation is used at runtime to determine whether a candidate answer should be accepted, refused, or regenerated before final delivery.
CiteGuard-RAG is evaluated on 400 questions across a controlled housing-law dataset, PrivacyQA, and CUAD. In the controlled evaluation, it achieves 99.1% retrieval accuracy, 98.3% grounded-answer accuracy, and 98.3% citation validity, with no validation-detected hallucinations. Ablation results show that grounded-answer accuracy drops sharply when validation is removed, even when retrieval accuracy remains unchanged. External evaluation shows that while citation validity remains strong, evidence utilization, span alignment, and refusal calibration become harder under domain shift.
These findings indicate that trustworthy RAG systems require explicit validation between retrieval and final answer delivery. CiteGuard-RAG provides a practical architecture for linking retrieval, generation, citation checking, abstention, and regeneration in high-stakes information access. 

---
# Enabling Streaming User Transcription in Full-Duplex Speech-to-Speech Models 

**Authors**: Ke Hu, Nourchene Ferchichi, Edresson Casanova, Ankita Pasad, Elena Rastorgueva, Chen Chen, Nithin Rao Koluguri, Piotr Zelasko, Yifan Peng, Hainan Xu, Zhehuai Chen, Boris Ginsburg  

**Link**: [PDF](https://arxiv.org/pdf/2609.15759)  

**Abstract**: Full-duplex speech-to-speech (S2S) models enable natural conversational AI by allowing simultaneous listening and speaking. However, these models typically lack inherent user speech transcription, which is essential for applications such as conversation logging, accessibility features, and quality monitoring. In this work, we propose an efficient method to add streaming ASR capabilities to an existing duplex S2S model by introducing a lightweight ASR head in parallel to the agent text head. Our approach requires minimal additional parameters and no significant architectural changes to the base S2S model, enabling real-time user transcription while preserving full-duplex conversational capabilities including turn-taking and barge-in handling. Experimental results demonstrate that our method achieves streaming average WER of 10.21% on the HuggingFace Open ASR Leaderboard within the duplex S2S framework. Additionally, we show that the same architecture trained as a standalone streaming ASR model achieves competitive results (7.73% WER) compared to current SOTA models. We will open-source our training and inference code to facilitate further research in joint streaming ASR and S2S modeling. 

---
# Sequential Adapter Stacking for Cross-Lingual Low-Resource ASR 

**Authors**: Thai Thi Thanh Thao Dang, Mengjie Qian, Kate Knill  

**Link**: [PDF](https://arxiv.org/pdf/2609.15758)  

**Abstract**: Extending large-scale multilingual automatic speech recognition (ASR) models to low-resource languages remains challenging. Model performance is skewed toward high-resource languages and degrades sharply for languages with limited labeled data and pre-training exposure. To address this, we investigate parameter-efficient approaches for transferring knowledge from resource-rich source languages to low-resource target languages on Whisper. Alongside warm initialization and attention-based fusion, we propose Sequential Adapter Stacking, which places a trainable target-language adapter on top of a frozen source-language adapter. Under controlled experiments, these approaches are evaluated on three target languages unsupported by Whisper -- Asturian, Assamese, and Xhosa -- using source languages with varying degrees of relatedness. Sequential Adapter Stacking with the closest related source consistently and significantly outperforms full fine-tuning across the three targets, with 5--8\% relative WER reductions. These gains largely persist with only one hour of target training data. 

---
# Look Before You Leap: Factual Decoding with Internal Attribution Signals 

**Authors**: Hayeong Ryu, JungMin Yun, Byeonggeuk Lim, Sunhee Jo, YoungBin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2609.15745)  

**Abstract**: Hallucination remains a critical challenge in large language models (LLMs), where early factual errors compound through autoregressive generation in a snowballing effect that neither post-hoc correction nor weight-level intervention can effectively preempt. We propose DescaPE (DEcoding Signal Control Against Path Error-snowballing), a decoding framework that leverages internal model signals to suppress hallucination-prone trajectories at inference time. Through sliding-window MLP ablation, we identify a factual-salient layer span within LLMs whose derived signal is selectively elevated for factual tokens and exhibits anomalous spikes at hallucination-prone steps. We train a lightweight probe to approximate this signal from a single forward pass and integrate it into candidate scoring to penalize high-risk continuations while rewarding factually grounded ones. Experiments across five factuality benchmarks on three LLMs demonstrate that DescaPE achieves factuality improvements over decoding-time baselines in multiple settings, while incurring only 1.10x latency overhead in our efficiency evaluation. Our code is available at this https URL. 

---
# Merging the Knowledge of LLMs for Automatic Speech Recognition 

**Authors**: Hayato Futami, Tatsuya Kawahara  

**Link**: [PDF](https://arxiv.org/pdf/2609.15743)  

**Abstract**: Automatic speech recognition (ASR) systems, trained on paired speech-text data, have been improved by leveraging language models (LMs) trained on text-only data. LM fusion methods such as shallow fusion and density ratio are well-established methods that incorporate external LMs during ASR decoding. However, they incur additional computational costs due to LM inference, which is particularly problematic for recent larger LMs. In this study, we propose incorporating external LMs via model merging. This method integrates the LMs directly into the parameters of an LLM-based ASR model, requiring no additional computational cost at inference. We formulate domain extension and transfer via arithmetic operations on LoRA parameters. Experimental evaluations were conducted for the domain adaptation of LLM-based ASR trained on CSJ and LibriSpeech. We show that our LM merging consistently improved the ASR performance in the target domains, without degrading inference speed or memory footprint. 

---
# RESKILL: Explicit Failure Attribution and Structured Repair for Interactive Language Agents 

**Authors**: Mengyi Deng, Xin Li, Duyi Pan, Zilin Wang, Zhiwei Li, Zhijiang Guo, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.15684)  

**Abstract**: Language agents increasingly rely on reusable skills, but post-failure repair is often handled by opaque one-shot reflection: a model generates a skill patch without explicitly maintaining how failure explanations relate to candidate repairs or how unsuccessful retests should influence later edits. We introduce RESKILL, a structured repair framework that maintains an explicit repair state across repair rounds. Given a failed rollout, the framework links failure hypotheses to candidate skill patches, selects local repairs through coverage-based attribution, retests the edited skill set in the environment, and uses retest outcomes to guide subsequent repair updates. The language model supplies structured repair factors, while the repair procedure records them, compares local skill patches by how well they address active failure explanations, and carries unsuccessful retest outcomes into later repair rounds. We evaluate RESKILL on ALFWorld and TextCraft across three model sizes under fixed repair budgets. RESKILL obtains the strongest final success in all six benchmark-model settings, improving average final success by 3.7 percentage points over direct repair and 3.3 points over hypothesis-conditioned repair. These results suggest that explicit attribution alone is insufficient; durable improvement emerges when attribution is integrated with repair selection and persistent retest-conditioned update. 

---
# Empathy Is Steerable but Multi-Axial: Mechanism Geometry and Persona Effects in LLMs 

**Authors**: JuHeon Ha, Byounghan Lee, Yunseo Choi, Kyung-Ah Sohn  

**Link**: [PDF](https://arxiv.org/pdf/2609.15654)  

**Abstract**: Activation steering has been used to control traits such as honesty, refusal, and sycophancy, yet supportive empathy is evaluated along multiple dimensions that need not correspond to independently controllable activation directions. Using the EPITOME framework, which decomposes supportive empathy into Emotional Reactions, Interpretations, and Explorations, we study three instruction-tuned LLMs and ask whether candidate directions derived from these labels produce distinguishable intervention effects or instead share structure, and how persona prompts interact with those directions. We find that contrastive activation addition yields a stable middle-layer intervention that consistently shifts the EPITOME proxy scores across models, moving empathy analysis beyond response-level scoring. However, the recovered directions are only partially separable: steering one direction induces off-target shifts, and hand-crafted prompting shifts the empathy profile rather than isolating a single dimension. Persona prompts substantially change EPITOME scores, but a paired activation-shift decomposition shows that the recovered subspace captures only approximately 3 percent of persona-induced squared activation-shift magnitude at layer 15. Under this EPITOME-based definition, expressed empathy is steerable but multi-axial, and controlling persona-conditioned empathy requires targeting structure beyond individual mechanism directions. 

---
# Through the Eyes of the Beholder: Biometric and Demographic Conditioning for Multimodal Sexism Detection 

**Authors**: Ana-Maria Luisa Mocanu, Sebastian Mocanu, Ciprian-Octavian Truică, Elena-Simona Apostol  

**Link**: [PDF](https://arxiv.org/pdf/2609.15608)  

**Abstract**: Detecting sexism on the internet is a fundamentally subjective task; our team, VANGUARD, addresses this challenge in the EXIST 2026 Task 2 by proposing a human-centered multimodal framework that analyses and incorporates the psychological and demographic characteristics of human annotators into the detection pipeline. We fuse five input modalities through a cross-attention architecture with Feature-wise Linear Modulation conditioning. Meme text is extracted and visually described with Gemma 4, then augmented by automatic translation between English and Spanish with NLLB-200. Text and image representations are produced by LoRAadapted XLM-RoBERTa and CLIP encoders and fused with sensor features encoded by a pretrained autoencoder. To model annotator subjectivity, we frame Subtask 2.1 as a label distribution learning problem, optimizing a Kullback-Leibler divergence loss over the full annotator label distribution. At inference time, predictions are produced by soft-voting between the deep multimodal network and a complementary SVM trained on stylometric and physiological features. Our best submission ranks 29th out of 114 on Subtask 2.2 (source intention) under soft evaluation, and the normalized ICM scores remain above the baseline on Subtasks 2.1 and 2.2, indicating that annotator-centered conditioning contributes a usable signal. We release our full pipeline and analysis to support reproducible human-centered modeling. 

---
# Can We Trust the Judges? Validation of Factuality Evaluation Methods via Answer Perturbation 

**Authors**: Sarra Gharsallah, Adele Robaldo, Mariia Tokareva, Giovanni Gatti Pinheiro, Ilyana Guendouz, Raphaël Troncy, Paolo Papotti, Pietro Michiardi  

**Link**: [PDF](https://arxiv.org/pdf/2609.15561)  

**Abstract**: Evaluating the factual correctness of large language models (LLMs) is vital for many applications. But are our evaluation tools themselves trustworthy? Despite the rise of factuality-based metrics, their sensitivity and reliability remain underexplored. This paper introduces a meta-evaluation framework that systematically tests these metrics using controlled corruptions of gold standard answers. Our method generates ranked outputs with known degrees of degradation to probe how metrics capture nuanced changes in truthfulness. Our experiments reveal that pipeline-based methods, such as the RAGAS's factual correctness metric, better track degradation than LLM-as-judge approaches. We also propose a new variant of the factual correctness metric that provides a competitive and cost-efficient. 

---
# Don't Count the Edits, Judge by the Outcome Alone: Reward-Based Evaluation for Grammatical Error Correction 

**Authors**: Hayeong Ryu, Sunhee Jo, Seunguk Yu, YoungBin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2609.15559)  

**Abstract**: Grammatical error correction (GEC) evaluation has traditionally relied on reference or edit overlap, which can penalize valid rewrites that differ from gold corrections. Reference-free metrics reduce this dependence, but evaluating whether a fluent output is a valid correction of the source remains challenging. We propose SURE, a source-conditioned reward evaluator trained on within-source preferences spanning minimal-edit and rewrite-oriented corrections. SURE jointly learns an overall reward with criteria-level supervision for grammaticality, faithfulness, and fluency, together with span-level grounding for source-side error resolution. Experiments on SEEDA show that SURE performs competitively against strong baselines, with particular gains on rewrite-style corrections and more disentangled criteria-level diagnostics. Our code is available at this https URL. 

---
# To Each Language Its Tokenizer: Modular Tokenizers for Efficient Multilingual LLMs 

**Authors**: Franck Signe, Hippolyte Pilchen, François Yvon, Édouard Grave  

**Link**: [PDF](https://arxiv.org/pdf/2609.15528)  

**Abstract**: Multilingual Large Language Models (LLMs) traditionally rely on a single vocabulary shared by all supported languages, which can lead to uneven compression across them. Moreover, their large embedding and output matrices increase memory usage and slow inference, notably for small-scale models. It is also wasteful as models are often used for only a subset of languages. To address these issues, we introduce a modular framework for multilingual model training. First, we propose methods to learn large modular BPE and Unigram tokenizers that enable extraction of subtokenizers tailored to any language subset. These subtokenizers achieve compression on par with monolingual tokenizers and improve cross-lingual fairness. Second, we design a pretraining strategy that samples subtokenizers to form batches, restricting predictions to the relevant vocabulary subset and allowing efficient training despite a large vocabulary. This supports efficient inference with any combination of language-specific vocabularies. Therefore, it reduces memory usage and speeds up inference in models without sacrificing performance. 

---
# Psychosis involves a deficit of information compression in connected speech 

**Authors**: Samuele Vallisa, Claudio Palominos, Rui He, Emre Bora, Burcu Verim, Cemal Demirlek, Berna Yalincetin, Philipp Homan, Wolfram Hinzen  

**Link**: [PDF](https://arxiv.org/pdf/2609.15522)  

**Abstract**: Large language models (LLMs) with human-like performance on linguistic tasks have transformed the study of language in neurodiverse conditions. LLMs provide representations of linguistic input in the form of high-dimensional vectors (embeddings), and next-token predictions computed from these embeddings. Previous crosslinguistic evidence suggests a complexity reduction in the form of both lower intrinsic dimensionality (ID) of LLM representations and higher mean surprisal (prediction error) in psychosis. We hypothesized that these metrics reflect a general deficit of information compression in psychosis, linked to grammatical organization as what enables predictions in this http URL operationalized surprisal difference as the difference between surprisal as estimated from word frequency and surprisal as based on a contextual LM, which is sensitive to grammatical organization over and above lexical concepts. Using a dataset of 144 Turkish speakers, including 106 patients with schizophrenia-spectrum disorders (SSD) - 56 with chronic schizophrenia (SZH), 33 with first-episode psychosis (FEP), and 17 with schizoaffective disorder (SZA) - and 38 healthy controls. We report: (1) Surprisal difference is attenuated in all clinical groups relative to controls, independently of word count; (2) Compressibility (intrinsic dimension) is reduced in SZH and FEP; (3) Syntactic complexity and compressibility both predict surprisal difference. These results, further refining an alteration in the geometry of the semantic space in psychosis as previously attested, suggest a broader deficit in information compression in this disorder, with a mechanistic underpinning in the operations of grammar. 

---
# Authorship attribution and aesthetic evaluation of AI poetry: a case study with Haiku 

**Authors**: Livia Oddi, Simone Scardapane, Toru Sugimoto, Donatella Genovese  

**Link**: [PDF](https://arxiv.org/pdf/2609.15511)  

**Abstract**: This paper investigates the generation and human evaluation of Japanese haiku by contemporary Large Language Models (LLMs), focusing on authorship perception and aesthetic judgment within a constrained poetic form. Using a few-shot prompting strategy, Japanese haiku were generated across a heterogeneous set of large language models, including open- and closed-source systems, medium-scale and large-scale architectures, models with native or adapted Japanese support, and multilingual proprietary models. These AI-generated haiku were combined with human-written ones and presented in a questionnaire distributed to students at Japanese universities in Tokyo. The survey assessed whether respondents could distinguish between AI-generated and human-written haiku and which cues informed their judgments. Recognition accuracy varied across models. GPT-5, Gemini 2.5, and StableLM-7B performed at approximately chance level (approx 0.50), whereas LLM-JP, Gemma-2B, and LLaMA-2 showed moderate detectability (approx 0.59-0.67). However, recognition was strongly item-dependent. Ratings of fluency, coherence, poeticness, and related aesthetic dimensions predicted perceived humanness but not correct classification, indicating an attribution bias linked to aesthetic evaluation and revealing a dissociation between aesthetic evaluation and true authorship detection. The extended analysis additionally examines generation-constraint adherence, participant-level characteristics, and exploratory LLM-based evaluations of haiku authorship. Overall, the findings suggest that as LLMs improve, surface-level creative plausibility may reduce reliable human discrimination within constrained poetic settings. 

---
# How Lossless Is Lossless Speculative Decoding? The Role of Numerical Precision in Orthrus 

**Authors**: Ilya Koziev, Leonid Sinev, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2609.15504)  

**Abstract**: Orthrus is a hybrid autoregressive-diffusion architecture that accelerates autoregressive language-model inference by generating multiple tokens in parallel while using a frozen autoregressive backbone. Its central claim is that an intra-model consensus mechanism enables lossless speculative decoding, producing the same output sequence as the autoregressive model.
We independently reproduce Orthrus and examine this claim under different numerical precisions. Under BF16 inference, exact trajectory matching occurs in only 45% of cases for the authors' checkpoint and 43% for our independently trained model across 1,190 prompts from 12 domains. The probability of exact matching is also strongly associated with the response-conditional perplexity of the reference model. Despite this trajectory divergence, Orthrus does not show systematic degradation on downstream lm-eval-harness benchmarks. In contrast, repeating the trajectory evaluation with FP32 yields exact trajectory matching on all evaluated prompts.
These results show that the practical losslessness of Orthrus depends on numerical precision and that exact trajectory equivalence should be evaluated separately from downstream task performance. 

---
# Temperature Fragility and the Conditional Benefits of Truncation Sampling 

**Authors**: Francesco La Rosa  

**Link**: [PDF](https://arxiv.org/pdf/2609.15476)  

**Abstract**: Large language models generate text by sampling each token from a predicted distribution, and a temperature parameter sets how far the draw strays from the most probable tokens. Truncation samplers such as top-p and min-p discard the least probable tokens before the draw, so that sampling at high temperature stays coherent. Their reported accuracy gains come from temperatures of 1.5 to 3, while the defaults of deployed systems cluster between 0.6 and 1.0. Whether they change accuracy at those defaults, and for which models, has not been measured. We test thirteen open-weight models on GSM8K and MMLU-Pro at temperatures 0.7, 1.0, and 1.3 in one controlled pipeline, ten of them under eight decoding configurations. Six of the thirteen models lose 17 to 38 accuracy points on MMLU-Pro between 0.7 and 1.3, and the other seven lose at most 10. The lost accuracy comes from generations that run to the token limit or never state an answer. These results suggest that truncation samplers improve accuracy primarily when higher temperatures substantially degrade model performance. Where accuracy remains stable across temperatures, none of the tested truncation samplers improves on plain temperature sampling. 

---
# Turkish MMLU Pro: Traceable Option Augmentation and Its Validity Limits in Turkish Multiple-Choice Evaluation 

**Authors**: M. Ali Bayram  

**Link**: [PDF](https://arxiv.org/pdf/2609.15467)  

**Abstract**: Adding answer options can lower multiple-choice scores without improving assessment validity. Turkish MMLU Pro examines this distinction using 12,000 Turkish-source questions across 58 sections. Each question retains its stem, five original options and source key, and receives five options copied from other questions in the same section. Sentence-embedding retrieval proposes candidates; a language model selects existing identifiers. Deterministic verification reconstructs all 60,000 additions. A 25-model calibration exposes scoring and generation-budget effects. Five evaluations produce source-key accuracies of 34.8%-81.4%. On 981 shared questions, one API-served model falls from 93.7% with five choices to 83.1% with ten; 102 of 115 lost correct responses select borrowed options. The decrease is 24.4 percentage points on heuristically flagged negative stems and 5.9 points elsewhere. A completed human-checked audit of 200 sampled questions, with undocumented reviewer tool use, yields 47 and 31 multiple-answer judgments across the two record sets, 25 of the latter unresolved. These records support concern about ambiguity, while their dependence and incomplete reviewer-method documentation limit validation. Because order and labels also change, the paired comparison measures augmentation as implemented. The contribution is a traceable construction and an analysis of its validity limits, not evidence that lower ten-choice scores measure knowledge better. 

---
# SlopShape: Identifying AI-Generated Commercial Web Content 

**Authors**: Jochen Madler  

**Link**: [PDF](https://arxiv.org/pdf/2609.15369)  

**Abstract**: Word-level detectors identify unedited AI-generated text almost perfectly, but the literature documents their brittleness under rewording, and a word-level score neither characterizes a text nor identifies which AI model wrote it. We ask whether AI-generated text can be identified one level deeper, from structural signatures: how information is presented, in what order, with what evidence, and in what voice. We replicate StoryScope (Russell et al., 2026), which showed such patterns for AI-generated fiction, on commercial content: 2,250 pre-ChatGPT human blog posts from 268 company domains against 11,250 AI mirrors from five frontier models. A 214-feature instrument, applied by an LLM and validated in a human gold-annotation session (human-human kappa 0.928, human-model 0.946), detects AI posts from its 187 structural features alone at 98.0 macro-F1 on held-out companies, unchanged (98.1) when every AI post is reworded by its own model. The signal characterizes and attributes: AI posts share a tidy, self-announcing shape, 79.3% are attributed to the correct source against a 16.7% chance rate, and human posts occupy rare structural configurations. All effects replicate StoryScope's, consistent in direction and larger in magnitude. We release pipeline, instrument, prompts, code, and aggregate artifacts. 

---
# Dynamic Semantic Compression for Efficient Latent-Space Inference in Large Language Models 

**Authors**: Peipei Li, Dongsen Zhang, Yuchen Liu, Wenjun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2609.15338)  

**Abstract**: Large Language Models (LLMs) primarily perform inference at the token level, resulting in substantial memory overhead and compromised computational efficiency. In this paper, we propose a Dynamic Semantic Extraction and Inference (DSEI) framework, which achieves segment-level inference within the latent space through a two-stage training strategy. First, we construct a Dynamic Semantic Autoencoder (DSAE) via self-supervised learning. DSAE dynamically extracts segment-level semantics and compresses them into compact latent representations via adaptive semantic weighting and gated fusion. Subsequently, we integrate the DSAE into the LLM architecture and train the model to infer over dense latent space. DSEI substantially reduces both input and generation sequences and significantly enhances inference efficiency. Extensive experiments conducted on the Wanjuan dataset demonstrate that DSEI reduces perplexity by 48% compared to static sentence-level latent inference baseline. Furthermore, compared to standard LLMs using token-level inference, DSEI accelerates inference speed by 2.5$\times$ and reduces memory overhead by 90%. 

---
# When Agents Slow Down: Understanding LLM Agents' Test-Time Strategies via Elo-per-token Analysis 

**Authors**: Kaiyuan Liu, Qiuyang Mang, Bo Peng, Wenhao Chai, Hanchen Li, Shreyas Pimpalgaonkar, Luke Zettlemoyer, Alex Dimakis, Alvin Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2609.15309)  

**Abstract**: Large language model (LLM) agents allocate test-time compute adaptively as they revise solutions, use tools, explore alternatives, and decide when to stop. This test-time strategy makes it difficult to measure how agent performance scales. We study open-ended tasks that provide continuous scores for intermediate submissions, making progress observable throughout long trajectories. We propose Elo-per-token analysis, which tracks the best solution found at each token budget and uses a Bradley-Terry model to aggregate within-task orderings into Elo ratings across tasks with different score scales. We apply it to four general-purpose agents on four open-ended benchmarks, with sessions of up to 100M tokens, and to three feedback-driven LLM optimization harnesses in controlled single-task interventions. Independent sampling provides a theoretically characterized reference, for which Elo grows linearly with log compute. Against this reference, agents can initially convert tokens into Elo faster than independent sampling, but their marginal gains diminish and eventually fall below the reference. In contrast, the strongest historical human contestants improve superlinearly over contest time on shared AtCoder Heuristic Contest tasks, providing evidence of continual learning and substantial headroom after agents slow down. We define the scaling inflection point as the per-session budget where marginal Elo gains match the independent-sampling reference. Using this point as the per-session budget, we split 100M tokens across parallel sessions on FrontierCS Polyomino Packing, gaining +264 Elo over one long session and +355 over ten short sessions. 

---
# Artificial entrepreneurial cognition: Locating and causally steering an opportunity recognition dial inside large language models (LLMs) 

**Authors**: Christian Fisch, Angela Altmeier, Martin Obschonka, Michal Kosinski, Pin Ni  

**Link**: [PDF](https://arxiv.org/pdf/2609.15277)  

**Abstract**: Entrepreneurial cognition is a foundation of entrepreneurship research. Yet the growing involvement of large language models (LLMs) in entrepreneurial work extends the cognition question beyond human actors to systems whose internal representations remain largely unexplored. We introduce artificial entrepreneurial cognition, the functional organisation of entrepreneurship-relevant representations and computations inside artificial intelligence (AI) systems. We bring mechanistic interpretability into entrepreneurship research through representation engineering. Focusing on opportunity recognition (OR), we construct 636 matched OR-present and OR-absent scenario pairs and recover an OR direction in Llama 3.1 8B-Instruct. Rather than infer the construct from outputs, we intervene directly on this direction, steering the model up and down along what we call the opportunity recognition dial, and its opportunity judgments shift with it. To our knowledge, this is the first causal intervention on an internal representation of an entrepreneurship construct inside an LLM. Held-out tests, lexical and topical controls, behavioural ablation, and geometric comparisons show that the direction is recoverable, consequential, and distinct from the opportunity evaluation and exploitation directions, although steering it also shifts judgments about these neighbouring stages. Recovery, signed steering, and geometric separation hold across four additional LLMs spanning different scales and families. These results give the contested distinction between opportunity recognition and evaluation a concrete representational form inside AI systems. More broadly, they establish internal representations as a new object of entrepreneurship inquiry and show how entrepreneurship theory can guide their identification, causal manipulation, and interpretation. 

---
# Semiotic Relations and Proof Methods: A Cross-Genre Study of Argument Structure with Large Language Models 

**Authors**: Edirlei Soares de Lima, Marco A. Casanova, Antonio L. Furtado  

**Link**: [PDF](https://arxiv.org/pdf/2609.15194)  

**Abstract**: When a direct proof of a statement $S$ seems hard or even impossible to obtain, there may exist another statement (or set of statements) $S^{*}$, somehow related to $S$, on the basis of which $S$ can be proved. In order to investigate what options can be used to move from $S$ to $S^{*}$, four kinds of semiotic relations inspired by the four master tropes of semiotic research are briefly reviewed. Specifically, our syntagmatic, paradigmatic, antithetic and meronymic relations correspond, respectively, to metonymy, metaphor, irony and synecdoche. It is suggested that these four semiotic relations determine the options to move from $S$ to $S^{*}$, leading to proof by inference, proof by analogy, proof by contradiction, and proof by case analysis. To examine how the four relations are actually used across different kinds of argument, we complement the framework with an empirical study. We turn the four relations into explicit operational definitions and apply them to a cross-genre corpus of mathematical, legal, and everyday argument using a panel of large language models. We find that the relations are used very unevenly across genres: mathematical proofs draw on all four, whereas legal and everyday reasoning rely almost entirely on inference. 

---
# What Limits Us? Analyzing Self-Reported Limitations in NLP Research 

**Authors**: Tawan Thaepprasit, Peeranuth Kehasukcharoen, Ding Wang, Remi Denton, Peerapon Vateekul, Piyawat Lertvittayakumjorn  

**Link**: [PDF](https://arxiv.org/pdf/2609.15191)  

**Abstract**: Since late 2022, a Limitations section has become mandatory at many top-tier NLP conferences. The growing number of accepted papers at these venues has resulted in a vast corpus of self-reported limitations that cannot all be manually reviewed, yet remains systematically unanalyzed. Therefore, in this paper, we conduct a large-scale analysis of the Limitations sections from ACL and EMNLP papers published between 2020 and 2025 to understand what researchers disclose about their own work. To do so, we implement a novel human-AI framework for iterative hybrid qualitative coding. This framework enables us to investigate trends in self-reported limitations over time, their correlations with specific paper attributes, and the writing patterns that recur around these disclosures. Our findings offer a critical reflection on the diverse reported challenges as well as the self-reporting practices of researchers in the NLP community. 

---
# MUSE: A Theory-Harnessed Story Engine for Vibe Narrativizing 

**Authors**: Jianxiang Ma, Xiaocui Yang, Daling Wang, Yuesong Hou, Mingfu Zhang, Yichen Gao, Junzhao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.15188)  

**Abstract**: LLMs can generate fluent prose. Story quality depends on how decisions about plot, character, and language work together across planning, drafting, and revision. Guiding these decisions presents two bottlenecks: the quality of story guidance and its sustained use. We formulate Vibe Narrativizing as the task of turning natural-language writing requirements into a finished story and present MUSE, a Theory-Harnessed Story Engine. MUSE organizes story knowledge as guidance for specific decisions and carries those decisions into subsequent creative work. Knowledge engineering develops Robert McKee's story theory through rule atomization, semantic consolidation, and mechanism abstraction; a single source of truth and layered disclosure organize the resulting guidance. Typical examples complement principles that depend on context and aesthetic judgment. An agent harness organizes design, character performance, scene composition, and revision through intermediate deliverables that preserve story decisions. Context engineering supplies each role with the relevant guidance and decisions, while a masterwork corpus provides inspiration and prose references. A worked example follows one requested object from its thematic role to the characters' climactic actions. Across four base models, MUSE improves WritingBench by 1.6-4.8 points over zero-shot generation and raises LongStoryEval by more than ten points on three. ConStory-Bench consistency error density remains in the low single digits for all four models, below every reproduced story-system baseline on three. Component ablations locate the largest quality contribution in structural design, voice-specific effects in the character path, and further gains in revision. Code is available at this https URL. 

---
# CITECHOICE: A Causal Audit of How Document Presentation Redistributes Citation Credit in Agentic Search 

**Authors**: Sriram Selvam, Anneswa Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2609.15164)  

**Abstract**: When several retrieved sources support the same claim, an answer engine cites some but not others. We call this decision citation allocation and introduce CITECHOICE, a causal audit of authentic multi-turn agentic search. From 129 everyday-query transcripts, CITECHOICE selects 113 same-call document pairs with independently verified support for the same pre-specified fact, without observing ranks or answer outcomes; blinded human review confirms 103. It runs a hash-verified 2-by-2 replay crossing pair order with jointly generated, fidelity-checked structured and prose renderings of one target while the rest of the transcript remains fixed.
Three results emerge. First, and most importantly, structured rendering concentrates citation credit rather than clearly increasing source admission. It raises target citation count by +0.50 citations per answer (95 percent CI [+0.20, +0.84]; Holm-adjusted p=.033), without increasing total citations or reducing competitor credit. The pre-specified incidence effect (whether the target is cited at all) is +4.5 percentage points and inconclusive (95 percent CI [-1.4, +10.4]; p=.168). Second, observational position differences exceed controlled reordering effects: the citation-rate gap between rank 1 and rank 5 is 42.3 percentage points, compared with +7.9 percentage points in the main replay and 0.0 percentage points held out. Third, citation evaluation has a measurable noise floor. Although the aggregate count effect repeats under fresh decoding of 30 frozen families, 15 percent of binary decisions change and decoding accounts for an estimated 45 percent of single-generation family-effect variance. Together, these findings isolate what survives control: presentation can causally redistribute visible citation credit within frozen transcripts. They do not establish reliable source admission, a pure formatting mechanism, or a general rank advantage. 

---
# EMR: Self-Evolving Medical Multi-Agent System via Experience Mining and Reuse 

**Authors**: Dongsheng Shi, Yue Li, Xin Yi, Linlin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.15161)  

**Abstract**: Large language model (LLM) driven multi-agent systems have shown promise in complex clinical reasoning, yet existing approaches rely on static strategies and lack persistent clinical memory, preventing self-evolving from prior diagnostic successes and failures. We present EMR, a self-evolving medical multi-agent system via Experience Mining and Reuse. EMR introduces a hierarchical clinical experience library that organizes accumulated knowledge into three levels: clinical principles, diagnostic patterns, and representative cases. During inference, EMR emulates multidisciplinary consultation: a planner agent coordinates domain-specific department agents for specialized reasoning, while a summary agent synthesizes their analyses into a final decision. Critically, EMR automatically extracts correct diagnostic insights and failure-related warnings from multi-agent reasoning trajectories, incrementally updating the experience library to guide future cases. Experiments on medical reasoning benchmarks demonstrate that EMR consistently outperforms state-of-the-art medical multi-agent baselines. Further analysis reveals that the hierarchical experience enables cross-specialty generalization and transfer across diverse LLM backbones, offering a scalable and in 

---
# Improving Mathematical Reasoning Capabilities in Large Language Models via Reasoning Process Error Classification 

**Authors**: Runa Yoshida, Kosuke Nishida, Kyosuke Nishida  

**Link**: [PDF](https://arxiv.org/pdf/2609.15145)  

**Abstract**: The reasoning ability of large language models (LLMs) is a critical factor for practical LLM-based applications. To investigate the current reasoning capability of LLMs, we clarify the types of errors that arise in LLMs' reasoning processes on mathematical datasets. We focus on problems where LLMs produce an incorrect answer. We define errors in the reasoning process as reasoning errors and manually analyze the features of reasoning errors. We defined and classified 21 error classes and identified the frequently occurring classes among them. Beyond qualitative evaluation, we leverage the evaluation results to improve the reasoning capability. We designed a prompt that explicitly focuses on eight error classes. The experiments demonstrate that this prompt effectively improves reasoning performance. Furthermore, the results suggest that the frequent reasoning errors identified in this paper are common across LLMs of comparable scale. 

---
# MoME: Mixture-of-Memory Embeddings for Context-Aware Sparse Lookup 

**Authors**: Muchen Li, Leonid Sigal, Renjie Liao  

**Link**: [PDF](https://arxiv.org/pdf/2609.15126)  

**Abstract**: Scaling large language models efficiently has motivated sparse capacity mechanisms such as Mixture-of-Experts and, more recently, conditional memory: token-indexed embedding tables that augment the backbone with cheap parametric lookups. Existing memory-embedding methods retrieve via a deterministic function of the surface form, which collapses different contextual senses of the same token (e.g., python the language vs. the animal) into a single fixed entry. We introduce Mixture of Memory Embeddings (MoME), a context-aware memory mechanism that replaces each token's single memory row with a mixture of M slots and uses a learned gate over the hidden state to choose which slots to read at each position. In controlled pretraining experiments across nanochat, Llama-3/MobileLLM, and Qwen3 backbones, MoME improves over Value Embedding, Bigram, and STEM baselines in iso-parameter and iso-training-FLOP settings, shows a more promising memory-size scaling trend at sub-billion scale, and remains efficient in training and inference. Qualitative routing analyses on polysemous tokens further suggest that the learned mixture exhibits a degree of semantic interpretability, dispatching the same surface token to distinct memory slots under different senses. 

---
# When the Wrong Key Wins: Understanding and Detecting Hallucinations in LLMs 

**Authors**: Xuhan Tong, Jiawei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.15106)  

**Abstract**: Large language models can hallucinate even when the knowledge required for a correct answer is already available. We study this failure through a latent-key view of inference, where answer selection depends on competition among associations acquired during pretraining. We show that model predictions can be highly sensitive to individual query keywords, that these influential keywords exhibit entity-specific binding, and that their effects are systematically shaped by pretraining frequency. Multiple bindings can also compete and exhibit higher-order interactions within the same query. Based on this mechanism, we introduce a two-stage keyword-perturbation method for hallucination detection. By removing influential keywords and measuring how the model reorganizes its prediction, the method distinguishes errors caused by misleading key associations from correct decisions supported by diagnostic evidence. Across multiple models and benchmarks, perturbation provides a strong and transferable detection signal, reaching $.910$ AUROC on probe-known ScientistQA. Finally, we extend the same probabilistic framework to four hallucination regimes: knowledge deficit, wrong knowledge, context distraction, and unstable inference. Their operational distributions across benchmarks provide diagnostic context for why different detector families succeed in different settings. 

---
# Translating the Translator: Decomposing the Cost of English-Forced Inter-Agent Communication 

**Authors**: Kushagra Agrawal, Yuming Feng, Man-Fai Leung  

**Link**: [PDF](https://arxiv.org/pdf/2609.15079)  

**Abstract**: Multi-agent LLM architectures, such as LangChain and AutoGen, largely assume English as the lingua franca for internal inter-agent communication, even when the end-user task is non-English. We fill this gap by evaluating a two-agent extraction-answer core, with an additional back-translation agent in the English-forced condition, across four typologically diverse languages (Hindi, Chinese, Spanish, Arabic; n = 300 per language) using the Aya-23-8B model. We compare a native-language pipeline to an English-forced one (which incorporates a final back-translation step from English to the user's language). We discover a statistically significant English-Forcing Tax (surviving a strict Bonferroni correction) that isolates the cost of English routing from general multi-agent orchestration overhead. Forcing inter-agent communication through English reduces Exact Match accuracy by 13.0 percentage points (Spanish) up to 30.6 percentage points (Hindi) compared to native-language multi-agent execution. Using chrF scores as a diagnostic measure of English-reference lexical overlap, we find that lower overlap is strongly associated with pipeline failure, consistent with translation loss being an important contributor to the observed performance drop. These findings suggest a compelling case for native-language routing in agent frameworks when the source and target languages are typologically distant, reducing a compounding translation tax. 

---
# DA-DLM: Explicitly Modeling Token Dependencies in Diffusion Language Models 

**Authors**: Pengyu Ji, Zichen Zhang, Xiang Hu, Kewei Tu  

**Link**: [PDF](https://arxiv.org/pdf/2609.15070)  

**Abstract**: Diffusion Language Models (DLMs) generate text by iteratively denoising a masked sequence, independently predicting multiple tokens at each step. This conditional independence discards inter-token dependencies and degrades coherence-an issue that parallels the multi-modality problem in Non-Autoregressive Translation (NAT). Drawing on the Directed Acyclic Transformer (DAT), which tackles this problem in NAT via a Directed Acyclic Graph (DAG), we propose DA-DLM, a model that adapts DAG-based dependency modeling to DLMs' iterative setting through a position-oriented DAG design. The position-oriented DAG binds node groups to fixed output positions so that tokens fixed in earlier steps anchor neighboring predictions via learned transitions, and evolves with denoising to focus on remaining uncertainty as anchors accumulate. On language modeling, open-ended generation, and summarization, DA-DLM consistently outperforms Block Diffusion, especially under fewer denoising steps, and matches autoregressive models while preserving the parallel generation advantage. Our code is publicly available at this https URL. 

---
# Salesforce Koa: An Enterprise Language Model for Agentic Tool Use 

**Authors**: Zixiang Chen, Sufeng Niu, Yingchi Liu, Wenting Zhao, Akshara Prabhakar, Shubham Mehrotra, Bin Bi, Zhujun Lan, Katherine Tan, Mohammad Ramezanali, Tulika Manoj Awalgaonkar, Monojit Banerjee, Jielin Qiu, Shiva Kumar Pentyala, Zhepeng Cen, Anupam Tripathi, Ali Ziaei, Regunathan Radhakrishnan, Darvish Lee Shadravan, Shelby Heinecke, Sitaram Asur, Silvio Savarese, James Zhu, Phil Mui, Huan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.15066)  

**Abstract**: We present Salesforce Koa, an enterprise language model built by post-training the open-weight Nemotron-3-Super-120B foundation model with reinforcement learning using Group Relative Policy Optimization (GRPO). Salesforce Koa is trained on public and synthetically generated data, with no customer data, to improve tool use and agentic capabilities while preserving strong general-purpose performance. Its distinctive component is a simulation-to-reward pipeline that expands workflow specifications into persona-conditioned multi-turn tasks with task-resolution rewards grounded in successful tool use for data-dependent requests. For enterprise domains, these specifications are written in Agent Script, Salesforce's declarative language for building Agentforce agents; for public tool-use domains, we synthesize the workflow structure directly. The same simulation and grounded-reward machinery drives GRPO across both. Across public tool-use, agentic-reasoning, and enterprise Customer Relationship Management (CRM) benchmarks, Salesforce Koa improves over its open-weight base, with the clearest gains on multi-turn tool use, and surpasses a strong proprietary baseline while remaining below the strongest frontier models. These results show that specification-driven reinforcement learning is a practical path to specializing open-weight foundation models for enterprise agentic tasks. 

---
# Mirror, Mirror on the Wall: Prompt Echoing in Small Instruct Language Models 

**Authors**: Inez Okulska, Bartosz Naskręcki, Jan Piotrowski, Tomasz Steifer  

**Link**: [PDF](https://arxiv.org/pdf/2609.15045)  

**Abstract**: Prompt echoing is a recognized failure mode of instruct language models, in which a model instead of generating a response, mirrors the provided prompt, even though it did not receive a specific instruction to do so. Is this phenomenon a sign of the model leaking the content of its training dataset, or is it rather caused by a misaligned behavior of the internal induction/copying mechanisms? We investigate prompt echoing small language models from different families (Gemma, Llama, Qwen, SmolLM and OLMo) and show that echoing prompts are likely to have partial overlap with the training dataset but the phenomenon is primarily driven by the model's induction heads. 

---
# SALUTE: Benchmarking and Adapting LLMs for the Defense Domain 

**Authors**: Hyeongcheol Park, Sumin In, Suyeon Myeong, Hogun Park, Sangmin Kim, Moonhyun Lee, Daekyeong Park, Sangpil Kim  

**Link**: [PDF](https://arxiv.org/pdf/2609.15022)  

**Abstract**: Defense is a knowledge-intensive domain that requires precise understanding of specialized terminology, doctrinal concepts, operational procedures, and evolving military events. Although recent work has explored language technologies for military applications, existing efforts remain fragmented: they are often task-specific, rely on limited adaptation pipelines, or lack comprehensive defense-domain evaluation. In this paper, we present SALUTE, an end-to-end framework for benchmarking and adapting LLMs for the defense domain. SALUTE integrates Salute-Corpus, a curated corpus from open-access U.S. military doctrine and government documents; Salute-Conv, a grounded instruction dataset from doctrinal sources and decade-long defense news; Salute-Pref, a defense-aware preference dataset; and Salute-Bench, a rigorously filtered benchmark for evaluating defense-domain understanding and reasoning over doctrine and defense news. Based on these resources, we train Salute-LLM through multi-stage post-training with continual pretraining, supervised fine-tuning, and preference alignment. Extensive experiments show that Salute-LLM achieves strong defense-domain performance while retaining competitive general capabilities, demonstrating the effectiveness of SALUTE as an end-to-end framework for defense-domain LLM adaptation. 

---
# ABSOL: Aggregated Bayesian Subsampling Orchestrated with LLMs 

**Authors**: Jackson Hassell, Chen Shen, Estevam Hruschka  

**Link**: [PDF](https://arxiv.org/pdf/2609.15007)  

**Abstract**: Large language models are increasingly used as natural-language interfaces to structured data, yet they remain unreliable when answers require consistent evidence conditioning, dependency-aware reasoning, and uncertainty estimation. Bayesian networks provide an explicit probabilistic reasoning layer, but learning useful structures from data remains costly and fragile at scale. We introduce ABSOL, a hybrid LLM-guided Bayesian network structure-learning framework that uses LLMs as bounded semantic guides. Across five discrete BN benchmarks spanning 27 to 1041 nodes, ABSOL is the only evaluated method to produce a viable graph on every benchmark, and achieves the highest Edge F_1 on every benchmark larger than 27 nodes with GPT-5.4. The four LLM augmentations, which contribute complementary semantic evidence to the statistical backbone, improve Edge F_1 over the non-LLM aggregation backbone by +0.23 on average. Complementary post-hoc refinement experiments suggest that these gains depend in part on limiting the LLM's authority over the final structure. Together, these results show that language-derived semantic knowledge can substantially improve scalable probabilistic structure learning when used as bounded guidance within a statistically grounded reasoning pipeline. The code for ABSOL is available at this http URL. 

---
# Beyond Depth and Width: The Information-Slack Dilemma in Streaming Test-Time Compute 

**Authors**: Xiaotian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.14995)  

**Abstract**: The same task and compute budget can require different reasoning policies when evidence arrives in a different order. Early computation has more time to finish but rests on incomplete or revisable evidence; waiting improves information while shrinking computational slack. We call this the information-slack dilemma.
We take the evidence-dependent computational job as the unit of analysis: when to start it, what supports its result, and when that result can be committed. Advance computation is valuable only insofar as its benefits survive the costs of verification, invalidation, and recovery. This applies to grounded incremental processing and reusable preparation as well as future-dependent speculation.
We propose a research agenda on computation under evolving evidence, prioritizing selective recovery under controlled evidence revisions. Evaluation should separate earlier-execution effects, deployment value against a full-input alternative, and the added value of predictive policies, while accounting for shared-resource costs. The objective is not maximal advance computation, but more trustworthy, on-time responses within a declared resource envelope. 

---
# MTAC-IFBench: Benchmarking Instruction-Following in Multi-Turn Agentic Coding 

**Authors**: Bosi Wen, Cunxiang Wang, Jiayi Gui, Haoke Zhang, Yilin Niu, Pei Ke, Dayong Yang, Hongning Wang, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.14992)  

**Abstract**: Recently, the rapid development of large language models (LLMs) has reshaped software engineering by enabling autonomous code agents that plan, execute, and utilize external tools iteratively to tackle complex tasks. Beyond achieving functional correctness, these agents must faithfully follow process instructions and constraints throughout the development lifecycle. However, existing benchmarks typically focus on final functional correctness or confine instruction-following evaluation to single-turn, general chat or simple code generation scenarios, leaving instruction-following in multi-turn agentic coding underexplored. To bridge this gap, we propose MTAC-IFBench, a comprehensive benchmark for this critical capability. It features multi-turn progressive software development instructions with diverse constraints spanning 6 primary and 18 secondary categories. With an average of 7.04 turns and 91.33 constraints per instance, it poses a rigorous challenge to current LLMs. To make the evaluation reliable, we construct a checklist for each constraint and functional requirement, and integrate verification scripts and judge agents to verify each checklist item. MTAC-IFBench identifies significant deficiencies in existing code agents in multi-turn instruction-following, with their performance degrading rapidly as the interaction session grows longer. 

---
# Typhoon ASR Streaming: Steerable Low-Latency Thai Speech Recognition with Real-Time Shallow Fusion 

**Authors**: Warit Sirichotedumrong, Tanawin Samutsin, Shah Faisal Wani, Sittipong Sripaisarnmongkol, Kunat Pipatanakul  

**Link**: [PDF](https://arxiv.org/pdf/2609.14991)  

**Abstract**: Open Thai automatic speech recognition (ASR) is dominated by offline, Whisper-based models that read the whole utterance before transcribing, ruling out low-latency uses such as live captioning and voice agents. We present a deployable system for streaming Thai ASR that lets a user steer its vocabulary at decode time, without retraining. A widely used open Thai model, trained with full context, collapses when run as a true stream; we restore streaming with a cache-aware encoder, by converting it or adapting a natively streaming one, and add a shallow-fusion layer that re-ranks candidates inside the streaming decoder with a GPU n-gram language model and phrase boosting. Across two Thai benchmarks and two model sizes, the streaming models stay usable where the full-context model fails, cutting character error rate 4.3-4.5x at a one-second look-ahead while running faster than real time. Decode-time steering then lifts keyword recall from 16.6% to 20.7% at no accuracy cost and negligible overhead; most of the gain comes from an n-gram over ordinary training transcripts, which resolves the written form of code-switched words the model hears but spells inconsistently, with phrase boosting adding targeted control over rare domain terms. 

---
# Biomedical Reference Generation Remains Unreliable across 26 Large Language Models 

**Authors**: Maxim Topaz, Zhihong Zhang, Nir Roguin, Pallavi Gupta, Zichao Li, Laura-Maria Peltonen  

**Link**: [PDF](https://arxiv.org/pdf/2609.14988)  

**Abstract**: Background. Large language models are increasingly used to help write biomedical text but may fabricate references to nonexistent work. How often large language models do so is not well characterized. Methods. We prompted 26 language models from eight developers (2023 to 2026) to supply a missing reference for each of 69 biomedical passages across ten domains. References were classified as verifiable (real paper with a resolving identifier), partial matches (real paper without a resolving identifier), fabricated (no matching indexed paper), or declined (the model refused to supply a reference). A reference was considered correct in every evaluated bibliographic field only when it was verifiable and its journal, year, and listed authors matched those of the cited paper. Results. Fabrication ranged from 10.2% (Claude Opus 4.8, which declined 52.1% of prompts) to 98.4% (Ministral 3B, which produced no verifiable reference). Claude Opus 4.6 and Claude Sonnet 4.5 produced similar proportions of verifiable references (77.6% and 76.6%) but named authors correctly in 78.7% and 28.7% of author-evaluable verifiable references, respectively, and were correct in every evaluated field in 54.6% and 19.9% of responses. GPT-5.5 was correct in every field in 48.1%. Across all models, 55.4% of responses were fabricated and 14.9% were correct in every field. Among the five tested models first released in 2026, the corresponding proportions were 35.3% and 31.8%, respectively. Conclusions. Fabrication remained common, and no model was correct in every evaluated bibliographic field in more than 54.6% of responses. Models that identify real papers may still misstate their metadata, so references produced with model assistance require verification before use. 

---
# Online Language Adaptive Sampling for Better Distributed Cross-lingual Gains 

**Authors**: Quang Phuoc Nguyen, Félix Gaschi, David Anugraha, Santiago Martínez Novoa, En-Shiun Annie Lee  

**Link**: [PDF](https://arxiv.org/pdf/2609.14969)  

**Abstract**: Realignment is a promising approach for improving the cross-lingual transfer ability of multilingual language models, particularly for extremely low-resource languages (LRLs). However, existing realignment methods rely on uniform and random sampling of parallel sentences across languages, which may be suboptimal under limited batch sizes. In practice, models may benefit from seeing certain languages more frequently, especially those that are poorly aligned, and the optimal distribution can evolve throughout training. In this work, we propose a simple yet effective adaptive sampling strategy that assigns trainable sampling probabilities to each language. Languages that contribute more to the realignment loss are sampled more frequently in subsequent batches, and the optimal distribution can evolve throughout training. Our method employs an inner-outer optimization loop with a small overhead, leading to consistent performance improvements and, more importantly, distributing the gains across languages. We observed a $+0.67$ average performance increase on all tasks with XLM-R, and $+0.60$ with Gemma 2 9B compared with uniform realignment. Furthermore, our method is robust across different models. Code available at this https URL. 

---
# A Corpus-Aligned Uthmani-to-Standard Quranic Word Mapping and a Deterministic Recitation Validator 

**Authors**: Yahya Mohamed Elnawasany  

**Link**: [PDF](https://arxiv.org/pdf/2609.14967)  

**Abstract**: Quranic text is distributed in two orthographic forms that are byte-level distinct: the Uthmani script used in every printed mushaf, and the Standard (Imla'i) Arabic form that every mainstream Arabic NLP tool is built for. The gap is concentrated in one Unicode character, U+0670 (superscript alef), which appears in some of the most frequently recited words in the Quran and is silently mishandled by general-purpose Arabic normalizers. We release a 2,290-pair, corpus-aligned Uthmani-to-Standard word mapping constructed by aligning the complete 6,236-verse Quran across both orthographic forms, together with a seven-step text normalization pipeline built on it. Normalizing both forms of all 6,236 verses through that pipeline yields identical strings for 90.9% of verses, and we characterize the residual divergence rather than assert that it is closed. On top of the normalized text, we build a deterministic, LLM-free Quranic recitation validator using a four-layer verse-matching search (exact, morphological, relaxed, fuzzy) and word-error-rate-graded feedback across five severity tiers. The validator scores 98.4% (122/124) on a 124-case suite emitted by the released test harness, and both failures share one mechanism: a single substitution error can make a different verse an exact match. A full-corpus census additionally quantifies an inherent text-only ambiguity affecting 16.5% of verses, and on 34 recitation transcripts drawn from a deployed Arabic ASR system the validator identifies the correct verse in every case. We release the mapping, the script that builds it, the validator, and the evaluation harness under open licenses; every number in this paper except the deployment measurement, whose transcripts are not ours to publish, is reproduced by running them. 

---
# Can We Triage LLM Translation Errors in Classical Texts Without Human References? Source Novelty, GEMBA Scoring, and Budgeted Review through Pali-to-English Translation 

**Authors**: Máté Metzger  

**Link**: [PDF](https://arxiv.org/pdf/2609.14963)  

**Abstract**: As large language models become capable translators of classical texts, a key challenge is deciding which outputs need expert review when no human reference exists. This study tests reference-free error triage through Pali-to-English translation. Three LLMs translated 15,493 passages. Five signals were compared: source novelty, source-candidate embedding distance, peer-translation disagreement, English-to-Pali backtranslation, and no-reference GEMBA scoring. Signals were calibrated on a 3,000-item reference-informed LLM-adjudicated sample and checked against a 500-item author-adjudicated anchor. Human references supported calibration and validation only; they were never used to compute the risk signals. Source novelty was a useful source-side risk prior but not a per-candidate error detector. Peer disagreement and backtranslation provided secondary signal. The strongest method was no-reference GEMBA scoring by a panel of models generally regarded as stronger than the translators: reviewing the top 10% by GEMBA risk captured 81.6% of panel-major errors in the calibration set. GEMBA also remained the best reference-free signal against the author anchor. A same-tier panel, with self-scoring excluded, remained useful but performed worse, indicating that evaluator strength matters beyond the prompt alone. A budgeted workflow is proposed, combining source novelty, peer disagreement, and stronger candidate-aware judging to allocate human review. Transfer to other classical languages, including Latin, Ancient Greek, and Sanskrit, remains to be tested. 

---
# Forty Shades of Blue: Quality-Diversity Alignment via Mode-Conditioned Reinforcement Learning 

**Authors**: Jiayi Yuan, Hangoo Kang, James Jihao Liu, Yejin Choi, Vikram Iyer, Liwei Jiang, Natasha Jaques  

**Link**: [PDF](https://arxiv.org/pdf/2609.14896)  

**Abstract**: A notable byproduct of LLM alignment training is mode collapse: the progressive loss of output diversity that narrows a model's expressivity at inference time. This degradation is especially limiting for applications requiring open-ended exploration and pluralistic perspectives, such as scientific ideation and creative writing. We present MoDA (Mode-conditioned Diversity Alignment), an online post-training RL algorithm that jointly optimizes generation quality and diversity, inspired by the coordination perspective in multi-agent reinforcement learning (MARL). MoDA trains a single shared LLM policy conditioned on abstract numbered roles, where each role acts as an agent competing to produce outputs distinct from the others. This formulation encourages mode-conditioned agents to explore complementary regions of the high-quality output space without requiring hand-crafted personas or architectural modifications. MoDA employs a prompt-adaptive quality gating mechanism that calibrates a reference quality threshold and grants diversity rewards only to responses that meet the threshold, preventing reward-hacking behaviors that compromise response quality. To study quality-diversity tradeoffs, we evaluate MoDA on a comprehensive suite of benchmarks spanning seven general capability tasks and four domain-specific diversity tasks in scientific ideation and creative writing. MoDA improves SBERT diversity by 265% on the Infinite-Chat held-out prompts, while increasing average general capability pass@1 by 10.3% over the Qwen3-8B baseline. Compared with the strongest DivPO baseline, MoDA improves SBERT diversity from 0.274 to 0.482 (+75.9%) and E-Vendi from 2.86 to 4.4 (+53.8%), while improving average general capability pass@1 by 7.0%. Overall, MoDA provides a drop-in alternative to standard post-training methods that preserves and expands the model's expressive output space while improving quality. 

---
# One Example Is Enough to Pass Fairness Benchmarks: Rethinking Fairness Evaluation for Aligned LLMs 

**Authors**: Naihao Deng, Samee Arif, Shuaichen Chang, Yulong Chen, Rada Mihalcea  

**Link**: [PDF](https://arxiv.org/pdf/2609.14860)  

**Abstract**: Warning: This submission studies stereotypes and biases, and contains toxic and offensive examples, used for illustration purposes only.
Fairness benchmarks such as BBQ have become the de facto standard for fairness evaluation across major model families. We argue that these benchmarks are too easy to support their role: training Qwen 2.5 7B Base with Group Relative Policy Optimization (GRPO) on a single BBQ example, or placing that example in context as a one-shot demonstration for in-context learning (ICL), lifts mean BBQ accuracy from 79.9% to 92.9% and 99.0%, respectively, closing 80% of the gap to its large-scale RLHF counterpart (96.1%) with GRPO, and surpassing it with ICL. These effects generalize across model families. A cross-conditioning analysis shows the improvement is carried by the reasoning traces generated by the model, and one example suffices to elicit a category-agnostic ``missing evidence'' reasoning pattern. We argue that BBQ-style multiple-choice abstention benchmarks measure a single structural cue, and a model that solves them does not thereby become fair. We call for evaluation suites that cover a broader spectrum of fairness alignment. 

---
# Dream-RSI: Recursive Self-Improvement through Evolving Worlds 

**Authors**: Tong Zheng, Xidong Wu, Zheng Zhang, Zhankui He, Chaoyi Zhang, Benjamin Coleman, Ruoqiao Wei, Di Bai, Haolin Liu, Rui Liu, Xue Wang, Yue Zhuan, Wang-Cheng Kang, Renkai Xiang, Heng Huang, Xinwu Cheng, Yunsong Guo  

**Link**: [PDF](https://arxiv.org/pdf/2609.14858)  

**Abstract**: Recursive self-improvement is becoming increasingly vital for autonomous AI agents, where progress hinges on discovering high-value solutions across complex domains. The driver of this process is effective exploration, however, managing and improving exploration strategies remains a major bottleneck. Current systems face a fundamental dilemma: fixed strategies fail to adapt as search spaces scale, while online policy optimization requires navigating vast meta-search spaces under delayed and expensive feedback over long-horizon rollouts. We introduce \textsc{Dream-RSI}, a framework for scalable and recursively self-improving exploration. A lightweight orchestration layer makes exploration explicit and programmable while leaving the underlying coding agent unchanged. Our key insight is that accumulated discovery history can serve as a replay simulator over the realized search space. By performing dreaming in the replay simulator constructed from historical discovery trees, \textsc{Dream-RSI} secures immediate, low-cost off-policy feedback to evaluate and refine exploration policies without invoking repetitive, expensive online evaluations. The improved policy is subsequently redeployed online to drive further discovery, continuously expanding the simulator pool in a self-improving loop. Across algorithm engineering, mathematical optimization, and GPU kernel engineering, \textsc{Dream-RSI} achieves competitive or improved discovery quality while substantially reducing discovery cost in several settings. 

---
# ModularRSI: Modular and Generalizable Recursive Harness Self-Improvement 

**Authors**: Siwei Wu, Jincheng Ren, Yizhi Li, Haau-Sing Li, Chengran Yang, Yuxuan Zhang, Weicheng Gu, Jian Yang, Riza Batista-Navarro, Chuanyi Zhang, Xianglong Liu, Ming Zhou, Bryan Dai, Chenghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2609.14857)  

**Abstract**: Recent work extends recursive self-improvement (RSI) to agent harnesses for long-horizon coding and terminal tasks, enabling agents to improve execution mechanisms from experience. However, generalizable harness RSI remains challenging. First, evolving harnesses on evaluation benchmarks or their subsets makes it difficult to distinguish reusable improvements from benchmark-specific adaptation. Second, single-trajectory updates can conflate systematic harness deficiencies with instance-specific reasoning and solution details, producing modifications that transfer poorly to unseen tasks. Third, localizing recurring behavioral deficiencies within monolithic harnesses is difficult, while whole-harness optimization can entangle unrelated mechanisms and complicate attribution and validation. We propose ModularRSI, a benchmark-disjoint, contrastive, and modular framework for generalizable harness evolution. ModularRSI contrasts successful and failed trajectories for the same task and aggregates evidence across tasks to identify recurring behavioral deficiencies. It decomposes the evolvable harness into five functional modules: Agent Loop, Tool Use, Observation Management, Context Management, and Task Completion Detection. Each module evolves independently within a restricted modification scope, followed by an integration stage that combines the evolved modules into a unified harness and resolves potential conflicts. To support benchmark-disjoint evolution, we curate 2,000 executable evolution tasks from external sources that are disjoint from downstream evaluation benchmarks. Experiments on TB2.0 and SWE-Bench Verified show consistent improvements on unseen in-domain and cross-domain tasks, with the evolved harness also transferring across different foundation models. 

---
# Enemray: Toward Capable Language Models for Hassaniya 

**Authors**: Cheikh Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2609.14829)  

**Abstract**: We introduce Enemray, a Hassaniya-centric language model that enables general-purpose interaction in Hassaniya. Enemray is trained around a stability--plasticity objective: acquire strong Hassaniya linguistic and cultural competence while preserving the general reasoning, multilingual, instruction-following, and safety behaviors of a capable instruction-tuned model. The development pipeline separates language acquisition from behavioral specialization. A separately assembled continual-pretraining corpus provides broad exposure to natural Hassaniya and Mauritanian text; layer-selective continual pretraining learns a compact language-specific parameter update; that update is transferred into the instruction-tuned parameter space; and supervised post-training develops conversational, cultural, literary, task-oriented, and cross-lingual behavior. The supervised corpus integrates selected public Hassaniya and Mauritanian resources with a substantially larger body of newly collected, reconstructed, curated, and constructed instruction data, while policy-generated replay provides a retention signal from the reference model's own behavior distribution. The resulting collection is substantially larger and broader in purpose than existing Hassaniya text resources. In evaluation, Enemray achieves the strongest English to Hassaniya translation among the compared open and proprietary models and the highest overall score on Mauritanian translation error detection, while retaining most of the general capabilities of its instruction-tuned base model on mathematical reasoning, knowledge, code generation, and function calling. This report describes the motivation, data construction, model design, training methodology, and evaluation of Enemray. 

---
# Route, Don't Fix: Regime-Dependent Decoding Correction and a Trajectory-Gated Router for Reliable Clinical LLM Answer Selection 

**Authors**: Zeyu Dong, Benjamin Wang, Joyee W. Jin  

**Link**: [PDF](https://arxiv.org/pdf/2609.14825)  

**Abstract**: Large language models (LLMs) are often deemed unsafe for clinical question answering because of their tendency to hallucinate. Retrieval augmentation, fine-tuning, and external verifiers require new infrastructure that clinical governance must approve and may add latency or extra model calls. Inference-time correction uses the model's internal logit signals, but a fixed transformation need not suit every question. A corrector that improves accuracy by about ten percentage points on a truthfulness stress test yields negligible gains on clinical multiple-choice benchmarks, where instruction tuning concentrates output probability on one answer and leaves low terminal entropy. We introduce ALTAS, which reads terminal entropy and late-layer linearity ($R^2$) from one forward pass to choose per question between greedy decoding and late-layer trajectory correction. No classifier, probe, or head is trained; the router operates on candidate-answer logits and adds 6.5% latency overhead. Applied to every question, the correction improves TruthfulQA over greedy at 3B and 8B by 11.4 and 10.0 percentage points, respectively ($p<10^{-10}$). Gated per question, ALTAS retains gains of 8.3 to 9.5 percentage points while keeping MedQA, PubMedQA, and MedHallu within a one-percentage-point do-no-harm band, with no statistically significant differences from greedy. The method passes verification sweeps over frozen thresholds, the scoring rule, and the domain label. 

---
# MedTRACE: Tool-Augmented Multimodal Clinical Reasoning Agents for Evidence-Grounded Decision-Making 

**Authors**: Ji Lu, Lifei Liu, Haoran Yu, Xianglong Wang, Yiru Fang, Kuo Yang, Huiran Duan, Jianping Gou  

**Link**: [PDF](https://arxiv.org/pdf/2609.14823)  

**Abstract**: Multimodal clinical decision-making requires reliable reasoning over heterogeneous evidence from electronic health records, medical images, and physiological signals. Existing models typically map these inputs directly to diagnoses without explicitly assessing evidence sufficiency, tool-use requirements, or diagnostic uncertainty. This paper presents MedTRACE, a tool-augmented multimodal clinical reasoning agent for evidence-grounded decision-making. MedTRACE uses modality-specific encoders to construct a unified patient-state representation and performs an iterative loop of hypothesis formation, toolaware deliberation, and evidence verification. It dynamically invokes visual grounding, evidence retrieval, and structured parsing tools to locate diagnosis-relevant regions, retrieve clinical knowledge and similar cases, and extract structured findings. The acquired evidence enters an evidence memory, where a consistency verifier confirms or revises the current hypothesis. MedTRACE outputs a diagnosis together with supporting evidence, an auditable reasoning trace, and calibrated confidence. Experiments on multiple multimodal clinical diagnosis benchmarks show that MedTRACE improves diagnostic accuracy by 5.4% and AUROC by 4.7 percentage points over the strongest baseline. It also improves evidenceselection F1 by 8.2 percentage points and visual-grounding IoU by 6.5 percentage points, reduces expected calibration error by 31.6%, and decreases unsupported diagnostic errors by 27.8%. These results demonstrate that active evidence acquisition and verification improve the accuracy, interpretability, and reliability of multimodal clinical decisionmaking. 

---
# A primer on evaluation methods for large language models in healthcare 

**Authors**: Suzannah E McKinney, Phuc Vu, Samuel A Justice, Christopher Humphries, Alyssa Pradhan, Timothy J Keyes, Bernardo C Bizzo, Keith J Dreyer, Sarah F Mercaldo, James M Hillis  

**Link**: [PDF](https://arxiv.org/pdf/2609.14819)  

**Abstract**: Large language models (LLMs) have a growing range of applications in medicine, and their evaluation is critical for ensuring they provide benefit and not harm. This evaluation can be more challenging than traditional machine learning for many reasons, including probabilistic and open-ended outputs, and behavior that shifts with prompt design and accumulated context. This review covers four key areas of LLM evaluation: principles of study design, statistical methods, capability evaluation and clinical context evaluation. Capability evaluation considers different benchmarks, including multiple-choice, agentic and multi-turn benchmarks, alongside operational metrics like token usage. Clinical context evaluation addresses establishing accuracy of free text outputs, such as human review and LLM-as-a-judge, and clinical trial approaches. Across sections, we describe underlying concepts and potential pitfalls, while emphasizing the importance of aligning evaluation methods with the research question. Together, this article aims to provide a pragmatic basis for designing and executing rigorous evaluations of healthcare LLMs. 

---
# Tone on a Budget: A Reference-Free Metric for Lexical Tone in Massively Multilingual Text-to-Speech 

**Authors**: Moses Daudu, Adeola Enitan Bamidele, Honor-Jesus Bezaleel  

**Link**: [PDF](https://arxiv.org/pdf/2609.14817)  

**Abstract**: In Yorùbá, pitch alone separates \d{o}k\d{o} (husband, Mid), \d{o}k\d{ò} (vehicle, Low), and \d{o}k\d{ó} (hoe, High) -- the diacritics ARE the tone marks. Yet character error rate (CER), the standard automated metric for text-to-speech (TTS), is in practice computed from ASR output that drops those marks: a synthesizer can ace CER and still say vehicle for husband. We introduce DunDun -- named for the dùndún, the Yorùbá talking drum that speaks through pitch alone -- an automated, reference-free lexical-tone metric that needs no tone-labelled corpus. The gold High/Mid/Low sequence is read from the input text's diacritics (in TTS that text exists by construction, so no reference recording is needed); the prediction comes from the audio's pitch track. We validate three ways. Flattening pitch with PSOLA resynthesis collapses DunDun while CER does not move. Inverting High and Low in the answer key of 300 native recordings drives the two-class readout to 0.14, symmetrically below its 0.35 chance level -- a consistency check on the scoring path, not independent evidence. And three native listeners, over 67 blind A/B trials, pick the tone-correct clip 89.6% of the time (95% CI 80.0-94.8; p < 1e-4); whether DunDun tracks those judgements trial by trial is not resolved at this sample size. Applied to a massively multilingual zero-shot TTS model, DunDun shows what CER cannot: Yorùbá tone sits near the native anchor before any Yorùbá fine-tuning (0.567 +/- 0.02 over five decode seeds vs. 0.596; chance 0.33), despite the 21.4% CER the model's own paper reports; and a few hours of clean audio halve CER (5.6% to 2.7% by 5h, 1.7% by 15h) while tone saturates within the hour. On non-tonal Swahili, CER already captures the gains: the metric a language needs is language-dependent. We release the metric and the complete validation protocol. 

---
# Mind Which Bird You Favour: Parameterizing Adequacy-Fluency Balance in Meta-Evaluation of Machine Translation 

**Authors**: Behzad Shayegh, Niloofar Kazemi  

**Link**: [PDF](https://arxiv.org/pdf/2609.14795)  

**Abstract**: There is a tradeoff in machine translation meta-evaluation between prioritizing alignment with adequacy versus fluency. The balance depends on the combination of translation systems in the meta-evaluation dataset. This system set is a small, filtered sample whose characteristics change heavily across years and language pairs; it does not represent the true system distribution. Consequently, the adequacy-fluency balance is often unrepresentative and subject to change. For sensitive domains, controlling this balance is critical. We expose this balance as a tunable choice. To achieve a target balance, we reweight existing systems while minimizing distortion from uniform weighting, ensuring the evaluated systems remain real and representative. We provide an exact optimization algorithm with theoretical guarantees and pruning mechanisms to compute these weights. To validate meta-evaluation internal consistency, we design a scorer-augmentation framework that establishes a known relative identity for the scorers. Results demonstrate that our reweighting method effectively controls the adequacy-fluency balance and preserves the internal consistency, outperforming prior approaches. Finally, we analyze the performance of popular scorers across a sweep of this parameter. 

---
# Func-R1: Incentivizing Mathematical Function Reasoning in Multimodal Large Language Models 

**Authors**: Mingze Yin, Xiaohan Wang, Dian Li, Haichao Yao, Yilin Zhao, Youjun Chen, Gang Liu, Jintai Chen, Yiheng Zhu, Chang-Yu Hsieh, Aimin Pan  

**Link**: [PDF](https://arxiv.org/pdf/2609.14779)  

**Abstract**: Performing deliberate mathematical reasoning in visual contexts is a hallmark of advanced Multimodal Large Language Models (MLLMs) and requires a sophisticated synthesis of perceptual grounding and symbolic logic. However, in the realm of mathematical functions, our investigation reveals a critical modality interference phenomenon: even advanced models, while performing textual computational reasoning, tend to disregard or misinterpret essential visual cues. To address this challenge, we propose Func-R1, which synergistically harmonizes precise visual perception and rigorous logical reasoning. Concretely, built upon an explicitly decoupled architecture, we employ a hierarchical post-training framework to progressively identify critical visual evidence and conduct in-depth theoretical reasoning. Furthermore, the Perception-Aligned Theoretic Optimization (PATO) strategy is proposed to steer policy updating towards internalizing fundamental theoretical properties while dynamically rectifying heterogeneous visual information throughout the reasoning process. Extensive experiments across diverse benchmarks demonstrate that Func-R1 delivers the optimal performance among open-source MLLMs, even surpassing GPT-5 with an 8.4% improvement on MathVerse's function-oriented tasks. 

---
# Pull: Lazy Materialization of Working Memory for Stateful LLM Conversations 

**Authors**: Jiangang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.14773)  

**Abstract**: As LLM conversations grow to hundreds of turns, full-context injection incurs $O(N^2)$ cumulative token costs, while lossy summarization or hard truncation irreversibly discards historical state. We propose Pull, a session router that maintains an addressable metadata directory via a local, deterministic Purifier (zero LLM calls, millisecond-level latency). At query time, the LLM lazily materializes only the turns it needs; unmaterialized turns remain accessible but collapsed. Unlike irreversible compression, Pull's materialization is reversible; subsequent queries can expand any collapsed turn. On LoCoEval (128 conversations, 12,780 turns), Pull reduces per-query context tokens (Phase 2) by 75.1 percent on single-hop tasks with equivalent quality ($\Delta = -0.002$, n.s.) and by 72.0 percent on multi-hop tasks with no quality loss ($\Delta = +0.017$). A controlled routing benchmark (7,831 queries x 10 methods) shows that entity lifecycle tracking is empirically a prerequisite for distance-independent routing. On BEAM 1M (14 conversations, 263 questions), Pull improves F1 by +55.2 percent over a truncation baseline. 

---
# How broad is that claim? Mapping Generalisation in NLP Research 

**Authors**: Chenxin Diao, Nataliya Stepanova, Emily Allaway  

**Link**: [PDF](https://arxiv.org/pdf/2609.14770)  

**Abstract**: Generalisations are common in scientific communication, even though they are semantically ambiguous. An automated method is needed to identify and categorise claims according to their level of generalisation, in order help detect an over-reliance on generalisations and possible misrepresentations of scientific findings. We introduce a comprehensive taxonomy of generalisations in the scientific domain, NLPGenX, which labels claims according to their level of generality and framing within the text. We operationalise this taxonomy with an LLM-powered framework, NLPGenA, that automatically classifies sentences from scientific articles into 5 different generalisation classes. We validate our framework with human annotators and use the framework to construct a large-scale dataset of NLP papers annotated according to generality, with auxiliary labels for hedging and vague descriptors (NLPGens). We use NLPGens to analyse the use of generalisations in NLP papers across multiple venues and subdomains, and to examine associations with citation counts, hedging, and vague descriptors. 

---
# Quantifying the Generation Modality Gap in Speech-Text Language Models 

**Authors**: Ju-Chieh Chou, Jiawei Zhou, Karen Livescu  

**Link**: [PDF](https://arxiv.org/pdf/2609.14743)  

**Abstract**: Pure speech language models often lag behind text and speech-text language models in generating coherent content, but this gap is difficult to quantify because speech and text systems are typically evaluated with different metrics and trained on different data. We study the speech-text modality gap in a family of spoken language models, based on flow matching for continuous acoustic feature generation. We construct a unified generation-based evaluation suite that compares speech-only, text-only, and speech-text language models trained on matched data distributions and evaluated in matched generation settings. We evaluate generated continuations along multiple dimensions: semantic coherence, measured by transcribing generated speech and scoring it with a reference language model; local phonetic structure, measured by phone n-gram distributional statistics; speaker consistency and acoustic quality; and emotion-based distributional metrics. Across datasets, we find that joint speech-text modeling substantially improves semantic coherence. However, the improvement is not uniform across metrics: phone-level metrics change only modestly, speaker similarity and predicted quality are lower for speech-text continuations, while emotion-based distributional metrics improve. Compared with larger-scale speech-only models, our speech-text model closes much of the scaling gap in transcript-based semantic coherence, suggesting that text provides an efficient semantic training signal for spoken language modeling. 

---
# Building Legal Reward Models for Grounding and Abstention 

**Authors**: Rilton Franzone, Valentin Noël, Puyu Wang, Philip Torr, Fabio J. Fehr  

**Link**: [PDF](https://arxiv.org/pdf/2609.14739)  

**Abstract**: Large language models are increasingly used in high-stakes domains such as law, where systems must ground their reasoning in retrieved evidence and abstain when that evidence is insufficient. However, existing reward models are largely optimised for general preferences rather than contextual grounding, limiting their ability to evaluate these behaviours in retrieval-augmented generation (RAG) settings. We introduce a framework for transforming existing legal QA datasets into contextual preference data and use it to construct LegalRewardBench (LRB), a benchmark for evaluating grounded legal generation under noisy and insufficient retrieval conditions. Across general and legal contextual evaluation, we find that contextual DPO improves grounded evaluation, but performance is sensitive to preference-data construction. Length-balanced augmentation substantially improves grounded legal evaluation, with the strongest configuration combining length-balanced legal and general contextual preference data and improving performance by up to $\mathbf{+25.6}$pp over baseline. We further find evidence of cross-jurisdiction transfer: models contextually refined primarily on Victorian criminal-law data improve grounded evaluation on external US legal benchmarks, including a $\mathbf{+16.2}$pp improvement on \textsc{Housing Statute QA}. Together, these results provide a reproducible foundation for constructing and evaluating grounded legal reward models in retrieval-augmented settings. 

---
# One Feedback System Does Not Fit All: Localising Data-to-Text Driver Coaching for the United Kingdom and Nigeria 

**Authors**: Iniakpokeikiye Peter Thompson, Jawwad Baig, Ehud Reiter, Dewei Yi  

**Link**: [PDF](https://arxiv.org/pdf/2609.14687)  

**Abstract**: Data-to-text driver coaching is often presented as a generic pipeline from telematics events to advice. This paper argues that its content requires localisation because usefulness and credibility depend on drivers' knowledge, prevalent risks, regulation, infrastructure, and available data. Two independently developed systems in the United Kingdom and Nigeria are compared by tracing requirements through content selection, generation, and field evaluation. The UK system prioritises post-trip reflection, explanations tied to road and place context, and tone-sensitive wording. The Nigerian system combines legally grounded, once-daily Tips based on detected events with weekly persuasive Reports; it foregrounds safety education and alcohol-related risk in response to reported gaps in formal training and traffic-rule knowledge, as well as local road-safety priorities. Reliable speed-limit metadata supported speeding feedback in the UK, whereas its scarcity led the Nigerian evaluation to exclude speeding from its outcome metric. Both interventions were associated with reduced distance-normalised unsafe-event rates in their own field studies, although their designs and metrics preclude an effect-size comparison. The analysis yields a requirements-to-content design process for localising safety-critical NLG without treating a high-income deployment as the default. 

---
# The Garden of Forking Prompts: How Users Explore Narrative Space in Story Generation 

**Authors**: Advait Deshmukh, Nora Benedict, Melanie Walsh, Maria Antoniak  

**Link**: [PDF](https://arxiv.org/pdf/2609.14677)  

**Abstract**: Large language models (LLMs) have changed the way people engage with stories. Drawing on public chatbot logs, we can see that when users generate stories, they iteratively edit their prompts to explore narrative possibilities, adjusting characters, redirecting plots, and swapping fictional universes. As aggregated data, these prompts represent rich traces of creative preference at scale. Yet story generation evaluation benchmarks rely on static, one-shot prompts that cannot capture this exploratory behavior. In this work, we study how users revise consecutive story prompts in the wild. Using a dataset of naturally occurring user-chatbot conversations, we construct WildStories, a sample of 275,635 story generation prompts (labeled with story format, prompt components, and explicitness), and WildEdits, a collection of 24,291 edit trees that model how users iteratively edit base story prompts and explore branching story possibilities. From these trees we develop a framework of edit types crossing four directions (adding, removing, changing, and extending) with fourteen targets (e.g., plot, character, genre). We then use our datasets and this framework to analyze user behavior in navigating narrative space via LLMs. Finally, we show how automated permutations based on the framework can be used for story generation benchmarking. Content Warning: This paper works with "wild" chatbot logs, which often include toxic and sexually explicit themes. 

---
# Optimizing Sparse Outcomes Through Dense Behavioral Signals via Value-Guided Preference Distillation 

**Authors**: Ziyi Zhu, Daniel R. Cahn, Thomas D. Hull, Caitlin A. Stamatis, Olivier Tieleman, Guilherme B. Freire, Jinghong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.14648)  

**Abstract**: Aligning multi-turn dialogue agents is usually framed as matching turn-level human preferences, yet direct optimization of long-term outcomes is often ineffective and prone to reward hacking. We formulate long-horizon dialogue optimization as a multi-objective reinforcement learning problem and train a multi-head value model that predicts a vector of observed user behaviors across multiple look-ahead horizons. Our findings demonstrate that a scalarized composite of dense auxiliary behavioral signals enables effective credit assignment and optimization of sparse outcomes. However, optimizing unconstrained single-objective proxies might induce policy degradations that are harmful when the agent is exposed to real users. To identify these failure modes prior to deployment, we establish a safety framework combining counterfactual user simulation with a validated dialogue-level outcome model to evaluate preference weightings and policy optimization methods. Finally, we demonstrate that distilling multi-objective value preferences into the policy via reference-anchored preference optimization matches on-policy online RL at a small fraction of its compute budget. Live A/B testing confirms that our distilled policy significantly improves long-term user retention, while simultaneously enhancing the positive behaviors and therapeutic-process markers. 

---
# CompCQR: Compositional Query Generation for Training-Free Conversational Search 

**Authors**: Yunah Jang, Kang-il Lee, Joongbo Shin, Kyomin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2609.14646)  

**Abstract**: Multi-turn interactions with LLMs are becoming increasingly common in information-seeking scenarios. However, user queries are often ambiguous and context-dependent, making them ill-suited for direct use as retriever queries. Conversational query reformulation (CQR) addresses this issue by rewriting the current utterance into a stand-alone query grounded in the dialogue history. Recent LLM-based CQR approaches achieve strong performance; however, their repeated LLM invocations and misalignment with downstream retrievers remain challenges. In this work, we begin from the observation that retrievers are highly sensitive to content ordering: simply reordering the same content can lead to changes in retrieval coverage and performance. Based on this, we propose a novel training-free method that generates a very large number of queries with minimal LLM usage by compositionally combining a small set of atomic components. We further apply LLM reasoning to construct a high-quality document set that balances precision and recall while capturing the user's core intent. Our framework generalizes across both open- and closed-source LLMs as well as dense and sparse retrievers. It achieves strong performance on four widely used conversational benchmarks, with up to 22.5% relative MRR improvement over the previous state-of-the-art baseline with far fewer LLM calls. 

---
# Domain-specific Pretraining Profile and Transformer Performance: Evidence from Modeling Digital Pragmatics in Arabic-English Code-switching 

**Authors**: Fahad Al Hussen, King Saud University, Riyadh, Saudi Arabia, Mohammed Q. Shormani, Ibb University, Yemen  

**Link**: [PDF](https://arxiv.org/pdf/2609.14571)  

**Abstract**: This study highlights the role of domain-specific pretraining profile (DSPP) in Transformer performance for modeling digital pragmatics in Arabic-English code-switched discourse. It evaluates MARBERT and XLM-R(oBERTa), with BERT serving as a general-purpose baseline. The models were evaluated on their ability to classify context-sensitive pragmatic functions in code-switched social-media discourse. 11695 unique X posts were collected via Python and utilized for the study. The study employs a quantitative and qualitative NLP approach, following a supervised pipeline. Findings unveil that MARBERT consistently surpasses XLM-R with validation Macro F1 increasing from 0.39 to 0.84 and validation loss decreasing from 0.55 to 0.19. On an independent test set, it achieved 0.96 accuracy, 0.83 macro precision, 0.87 macro recall, and 0.85 Macro F1, while XLM-R achieved 0.92 test accuracy but a substantially lower Macro F1 of 0.52. This was also supported by class-level performance where MARBERT outperforms XLM-R considerably with F1 improvements ranging from +0.33 to +0.60, demonstrating a clear advantage in modeling Arabic digital pragmatics. The study concludes that Transformer performance depends more on DSPP than multilingual coverage alone, as the latter does not guarantee optimal performance on a highly specialized pragmatic classification task. 

---
# Disentangling Topology and Diversity in Multi-Agent LLMs for Multilingual Low-Resource Emotion Detection 

**Authors**: Ulugbek Shernazarov, Charitha Ruwansiri Weerakon Basnayake, Abdelkhaleq El Jarjini, Noel Crespi, Praboda Rajapaksha  

**Link**: [PDF](https://arxiv.org/pdf/2609.14570)  

**Abstract**: Multi-agent LLM systems combine multiple inference calls, but prior work often confounds how calls are connected with how they are diversified. We study these factors independently: inference topology and source of inter-agent diversity. In a controlled $2 \times 3$ matrix, we cross parallel aggregation and sequential refinement with stochastic sampling, role prompting, and learned QLoRA specialization, under a fixed three-call budget and output protocol within each backbone. Using Qwen2.5-14B-Instruct and Llama-3.1-8B-Instruct, we evaluate all six configurations on multilingual low-resource emotion detection across nine languages. Parallel learned specialization is strongest on Qwen at 52.83 Macro-F1 and reaches 52.94 on Llama. On Qwen it also exceeds same-backbone zero-shot, few-shot, CoT, and seven-call self-consistency baselines. The preferred topology depends on diversity source: sequential refinement helps stochastic and prompted settings, while the learned Width advantage shrinks from 2.83 points on Qwen to 0.17 on Llama. Depth-wise analysis suggests that later learned specialists can overwrite correct early predictions, although the aggregate effect is backbone-dependent. Overall, how agents are differentiated produces larger performance shifts than topology, which should be evaluated jointly with specialization. 

---
# TATK: Triple-Aware Top-K Learning with Knowledge-Grounded Verification for LLM-based Sequential Recommendation 

**Authors**: Yuchen Guan, Jiaye Liu, Yifei Han, Zhenxi Zhang, Yixuan Weng, Bin Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.14565)  

**Abstract**: LLM-based sequential recommenders usually cast next-item prediction as text generation, but this interface is poorly matched to full-catalog top-K ranking. We propose TATK, a Triple-Aware framework that couples Top-K Learning (TKL) with Knowledge-Grounded Verification (KGV) for LLM-based sequential recommendation. Top-K Learning combines context-aware metadata-KG prompt grounding with position-aware top-K rewards, aligning training with ranking utility; Knowledge-Grounded Verification then applies structure-aware reranking over the top-M candidates after a single LLM forward pass, using the same metadata-derived item graph. We evaluate TATK on Musical Instruments, CDs and Vinyl, and Video Games from Amazon Reviews 2023 under a matched R2ec-style full-catalog protocol. Experiments use Gemma-2-2B-It and Qwen2.5-3B-Instruct backbones, compare against sequential, generative, KG-augmented, and reasoning-enhanced baselines, and include component, reward-shape, sequence-perturbation, reranking, relation-quality, and candidate-pool diagnostics. TATK improves over the matched R2ec reproduction on all 36 reported metrics. On NDCG@10, it improves Qwen by 8.05%, 4.26%, and 3.78% on the three datasets, and improves Gemma by 27.03%, 10.52%, and 10.23%, while keeping inference within 1.17x of Base RecPO latency. The diagnostics show that structural evidence is most useful for recoverable top-M candidates with reliable KG support, and should be gated when metadata relations are sparse or noisy. 

---
# Neyshekar: An Open Persian Read-Speech Corpus for Automatic Speech Recognition 

**Authors**: Ahmad Amirivojdan, Farzad Nadiri, Abolfazl Alizadeh, Shaghayegh Yaraghi  

**Link**: [PDF](https://arxiv.org/pdf/2609.14542)  

**Abstract**: Neyshekar is presented as an open Persian read-speech corpus designed for coverage of both formal and informal language, named entities, and longer utterances. In version 6, 62,279 validated recordings totalling 99.02 hours are provided from 190 contributors, with 34,541 distinct recorded prompts. The prompt pool was assembled from human-written material, contextualised homographs, and reviewed language-model-generated text. Text entries were normalised with the shekar library, which supports both formal and informal Persian, and every submitted recording was reviewed against a common validation rubric. About 24% of released clips are classified as informal by an automatic classifier; these register labels are not human-validated. Item-level rater labels are provided for reproducible agreement estimation, opaque per-clip contributor identifiers make the speaker-disjoint partitioning auditable and support contributor-clustered uncertainty estimates, and a text-disjoint test subset is included for evaluation beyond previously seen prompts. Per-contributor recording load and reference-free signal quality are characterised for every released clip. Corpus characteristics are compared with Persian Common Voice under shared processing. Utility is assessed through two ASR architectures, three optimisation seeds, WER and CER, and independent evaluation on the public PSRB sample. Against duration-matched Common Voice training at approximately 32 hours, in-domain WER is reduced by 9.5 points for Whisper and 11.6 points for XLS-R, and by approximately eight points for both architectures on the independent PSRB sample. Transfer and mixture benefits are not consistently observed across architectures and training budgets. The corpus is released under CC0; code and data are made available through the project repository at this https URL. 

---
# Theseus in the Graph: Towards Traceable Multi-Hop Graph Navigation 

**Authors**: Eduin E. Hernandez, Luis F. Garcia, Nurassyl Askar, Sergio A. Diaz, Stefano Rini  

**Link**: [PDF](https://arxiv.org/pdf/2609.14528)  

**Abstract**: Multi-Hop Knowledge Graph Question Answering (KGQA) tasks require models to assemble relational evidence along paths in a KG to answer natural-language questions. However, existing KGQA systems typically focus on predicting the final answer without explicitly modeling or validating the intermediate reasoning steps, obscuring whether the correct answers arise from faithful multi-hop reasoning. To address this limitation, we re-frame multi-hop KGQA as a question-conditioned graph navigation problem. We refer to this formulation as THESEUS - Traceable Hop-wise Evidence SEarch in a Unified Semantics. In this setting, an agent receives a KG, a question, and a topic entity, and traverses a sequence of relations towards the answer, making the reasoning path explicit. To systematically study this formulation, we provide three key contributions. (i) We augment the existing KINSHIP and MQuAKE resources into navigation-ready KGQA datasets with annotated evidence paths and paraphrased questions. (ii) We design evaluation protocols to measure path fidelity, robustness to linguistic variation, and performance across multi-hop and multi-answer questions. (iii) We adapt established path-based KG completion agents - MINERVA, MultiHopKG, and SQUIRE - to operate on full question embeddings rather than symbolic single-relation queries, enabling their trajectories to be guided by natural-language semantics. Together, these contributions advance KGQA research toward systems where traceability is fundamental: answers are accompanied by explicit reasoning paths whose agreement with reference evidence can be systematically evaluated. 

---
# NeuroActiSep: Detecting Factual Hallucinations from Feed-Forward Neurons in a Single Pass 

**Authors**: Ali Derogar Odolou, Reza Nazari, Mostafa Salehi  

**Link**: [PDF](https://arxiv.org/pdf/2609.14448)  

**Abstract**: Hallucination in large language models reduces their reliability and slows adoption. Various white-box studies have used internal representations to detect patterns of truthfulness and factuality. A less-studied approach is to identify feed-forward neurons correlated with hallucination. We propose a method to rank feed-forward neurons at the final prompt token using a custom neuron selection dataset. We transfer the selected neuron identities to train hallucination classifiers on other factual question answering datasets. Our work provides empirical evidence that probes trained using the features from the selected neurons perform on par with probes trained on internal states. We also analyze the distribution of selected neurons and the effect of layer depth on detection performance. 

---
# Policy Loopholes in Agent Evaluation: When Policy Ambiguity Masquerades as Agent Error 

**Authors**: Hongliu Cao  

**Link**: [PDF](https://arxiv.org/pdf/2609.14400)  

**Abstract**: Agent benchmarks evaluate policy compliance but assume each policy determines a unique correct action. Natural-language policies can violate this assumption through silence, ambiguity, or contradiction, admitting multiple defensible readings that a single gold trajectory cannot capture. Auditing two $\tau^2$-bench domains, we develop a taxonomy of such policy loopholes and show that affected tasks produce unreliable scores: they lower scores across different models in different ways and make every model less consistent across repeated trials. A cross-domain comparison reveals that exploitability requires both policy ambiguity and tool permissiveness: when policy complexity exceeds what tools can enforce, agents resolve gaps inconsistently and scores become unreliable. Policy specification quality sets the ceiling on evaluation quality. Benchmark developers should audit policies before collecting gold annotations. 

---
# Formal Properties of Language as Constraints on Neural Dynamics 

**Authors**: Elliot Murphy  

**Link**: [PDF](https://arxiv.org/pdf/2609.14384)  

**Abstract**: What must a neural system be capable of to implement language? Current research annotates stimuli with linguistic variables and tests which electrodes, voxels, or language-model layers predict neural activity. Yet predictive success leaves mechanisms under-constrained. Here, we show that algebraic properties of language specify invariants that mechanisms must preserve: non-associative hierarchical grouping, commutativity, recursive closure, access to substructures, and structured workspace transitions. We term this the Neural Admissibility Program (NAP). Syntactic structure building is analyzed algebraically, with candidate mechanisms offered for each requirement: content-addressable workspace memory, graph-structured transient dynamics scheduling structure-building operations (e.g. stable heteroclinic channels), and a phase-coupled sealing operation recording grouping. Simulations show that a corrected Marcolli-Berwick entropy-optimized binding gate preserves grouping only within a narrow commitment band. As an alternative, we propose a novel neural binding operation we term 'Meld': two constituent populations converge through shared synapses, integrate sublinearly, and saturate. Meld is, to our knowledge, the closest neurally plausible composition law to syntactic Merge. It preserves every NAP invariant, uses known cortical operations, and recovers hierarchical structure at every tested depth and temperature. It predicts that effective population dimensionality separates alternative bracketings and that the composite depends on constituent disagreement. Importantly, the laws decoding bracketing most accurately are a priori inadmissible, showing that decoding accuracy alone cannot adjudicate between mechanisms. By specifying how neural dynamics can remain faithful to linguistic structure, the NAP changes the criterion by which neural implementations of cognition are evaluated. 

---
# SpectralShift: Effective Context Window Extension of Gated DeltaNet via Spectral Reparameterization 

**Authors**: Zian Liu, Yiwen Hu, Zican Dong, Tian Xie, Wayne Xin Zhao, Yucheng Ding, Ran Tao, Bryan Dai  

**Link**: [PDF](https://arxiv.org/pdf/2609.14320)  

**Abstract**: Recently, linear attention layers have been increasingly adopted to replace softmax attention at scale for long-context modeling. However, existing context extension approaches typically apply continued pretraining directly without modifying these layers, overlooking the spectral properties of linear attention state dynamics. In this work, we study long-context extension of Gated DeltaNet (GDN) from a spectral perspective of transition matrix and identify two essential factors governing long-range information retrieval: (1) a sufficiently broad slow spectral band aligned with the target dependency length, and (2) the preservation of fast-decaying modes for state clearing and context switching. Based on this observation, we propose SpectralShift, a spectral reparameterization approach for long-context continual pretraining of GDNs. Specifically, SpectralShift reparameterizes the alpha projections initialization to reshape the decay spectrum by enhancing slow propagation capacity, and further introduces a learning-rate scaling for alpha projections to facilitate long-context training. Experiments show that SpectralShift consistently improves long-context capabilities over training, providing an effective and efficient solution for extending context windows of linear attention models. The code has been open-sourced at this https URL. 

---
# E2A-Bench: Benchmarking Evidence-to-Action Reliability in Financial Chart Reasoning 

**Authors**: Xiaoya Wang, Yutong Xu, Junjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.14302)  

**Abstract**: Can financial vision-language models (VLMs) turn chart evidence into reliable action recommendations? Existing hallucination evaluations are mostly claim-centric; they assess whether generated statements are supported, but not whether evidence remains traceable through rationale, confidence, and final action. We introduce E2A-Bench, a 969-query benchmark for financial chart reasoning, constructed from 323 HS300 constituents under three input modalities with deterministic OHLCV-derived evidence anchors. E2A-Bench evaluates grounding, reasoning-action consistency, evidence-confidence calibration, and directional coverage through UCR, RCI, ECI, and NDR, where NDR measures coverage-aware evidence-to-action reliability rather than realized trading performance. Evaluating 20 VLMs reveals three failures hidden by scalar hallucination scores: the lowest-UCR model ranks near the bottom by NDR due to only 6.4% directional coverage; oracle-aided verification reduces unsupported claims but can collapse coverage; and financial fine-tuning amplifies the BUY:SELL ratio by factors of 4.21 to 4.68 across strict base-fine-tuned pairs. These results show that financial VLM evaluation should trace the full evidence-to-action chain rather than rely on a single hallucination score. Code and data: this https URL 

---
# Editorial routing shapes how computational results are qualified in AI-assisted scientific writing 

**Authors**: Jihan Kim  

**Link**: [PDF](https://arxiv.org/pdf/2609.14288)  

**Abstract**: Large language models increasingly analyze computational results and draft manuscripts, making reliable communication as important as correct analysis. Using fixed computational evidence, we tested whether assigning comparisons across modeling choices elsewhere in a research workflow changes manuscript reporting. In constrained sentence-writing tasks, Anthropic's Claude Sonnet 5 often omitted numerical qualifications when detailed comparisons were assigned to a group repository, but retained them more often when the same comparison was assigned to Supporting Information or its own working notes; Claude Opus 5 was less sensitive. These effects did not follow a simple accessibility ordering. A targeted placement rule largely restored sentence-level qualification, whereas a generic accuracy reminder did not. Longer contributions retained numerical qualifications, although some summaries across computational settings were still redirected to the repository. Thus, documenting context within an AI workflow does not ensure its communication where readers encounter the result. 

---
# DenMark: Robust Semantic Watermarking for Diffusion Language Models 

**Authors**: Tianhao Ma, Weihao Xuan, Dong-Dong Wu, Farshid Nooshi, Takashi Ishida, Gang Niu, Naoto Yokoya, Masashi Sugiyama  

**Link**: [PDF](https://arxiv.org/pdf/2609.14257)  

**Abstract**: Semantic text watermarks encode signals in meaning rather than surface token choices, offering robustness to paraphrasing and other semantic-preserving edits. Existing semantic watermarking methods are primarily designed for autoregressive language models (ARLMs), where completed candidate units can be generated and scored before generation proceeds. This paradigm does not naturally extend to diffusion language models (DLMs), where semantic units remain incomplete during intermediate denoising steps and tokens may be updated in flexible orders. We propose DenMark, a semantic watermarking framework that injects key-dependent signals directly into the DLM denoising process. DenMark partitions the output into fixed token regions and uses temporary rollouts as semantic lookahead: conditional completions estimate the eventual semantics of an incomplete region, enabling DenMark to select local updates with higher estimated semantic watermark scores. Repeating this procedure across denoising steps progressively accumulates watermark evidence in the final output. For detection, DenMark uses calibrated scanning over candidate unit sizes to remain robust to boundary shifts introduced by semantic attacks. Across four DLMs, three datasets, and four semantic attacks, DenMark achieves the best results across all reported detection metrics in all 48 backbone-dataset-attack combinations. These results demonstrate that DenMark provides an effective mechanism for robust semantic watermarking in DLMs. 

---
# Document Topic Alignment Metrics for Evaluating Topic Models of Short-Text Public Health Communications on Social Media 

**Authors**: Wangjiaxuan Xin, Shuhua Yin, Yaorong Ge, Shi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.14256)  

**Abstract**: Topic models are widely used to analyze public health-related social media short texts, yet their evaluation remains dominated by metrics that focus entirely on generated topics alone. There is a lack of metrics that quantitatively assess whether assigned topics meaningfully represent the corresponding short-text posts. We propose Document-Topic Alignment metrics (DoTA), an assignment-aware evaluation framework comprising metrics that measure semantic alignment between documents (posts) and their assigned topics. We also introduce margin-based and discriminative variants that capture topic assignment confidence and distinguishability. We evaluate DoTA across five topic models on three public health-related social media datasets from X and compare DoTA metrics with conventional topic-based metrics. Results show that DoTA provides complementary evaluation cues and aligns meaningfully with human evaluations. These findings establish the need for assignment-aware evaluation and demonstrate that the addition of DoTA enables a more comprehensive and practically meaningful evaluation for assessing short-text topic modeling performance. 

---
# The Attribution-Compression Frontier in Retrieval-Augmented Generation 

**Authors**: Deepanshu Mody  

**Link**: [PDF](https://arxiv.org/pdf/2609.14245)  

**Abstract**: Context compression reduces generator input in retrieval-augmented generation, but answer quality alone does not characterize citation attribution. We measure citation attribution across compression methods and budgets, comparing reranking, extractive selection, abstractive summarization, token pruning, and an extract-cluster-rewrite construction on ASQA and QASPER under a fixed generator and primary entailment evaluator. On ASQA at a nominal 0.25 budget (achieved compression 0.08), a RECOMP-style compressor's citations score 0.86 precision against its summaries but 0.12 against source spans under our re-attributability protocol. These estimates depend on a shared NLI model for span recovery and citation scoring and lack independent human calibration. Extractive selection's observed grounded precision ranges from 0.43 to 0.49 across nominal budgets from one-half to one-tenth of the ASQA context, while answer quality declines. For the same RECOMP setting, claim verification after source recovery yields an unsupported rate of 0.88 versus 0.17 when checking summaries. This gap persists beyond structural rejection of missing provenance, but remains evaluator-dependent. A 200-question TRUE T5-XXL audit also finds emitted--grounded gaps under both fixed and recomputed source mappings, without establishing human-calibrated support rates. 

---
# Corpus Characterization and Inverse Constitutional Fine-Tuning for Style-Aware Radiology Reports 

**Authors**: Sarah Y. Li, Elijah Renner, Rayan Ansari, Alaa Youssef  

**Link**: [PDF](https://arxiv.org/pdf/2609.14226)  

**Abstract**: Automated radiology report generation has advanced rapidly in diagnostic accuracy, yet generated reports frequently diverge from the stylistic conventions of authentic radiologist writing in structure, diction, and uncertainty language, a gap which has direct implications for clinician trust and user experience. To address this, we characterize stylistic variation across 2,000 reports from the CheXpert Plus dataset using Bio-ClinicalBERT embeddings, UMAP dimensionality reduction, and HDBSCAN clustering, identifying five distinct reporting patterns differing in pathology focus, narrative structure, and lexical preference. Drawing on these findings, we adapt the inverse constitutional AI framework to derive a style-focused constitution from radiologist-written report pairs without requiring a formal preference dataset. This constitution, encoding conventions of tone, diction, uncertainty calibration, and report structure, is incorporated into the supervised fine-tuning of a MedGemma-4B base model on 25,245 CheXpert Plus training pairs. Constitutional fine-tuning produces a substantial increases in text alignment (BLEU-4: 0.006 to 0.308; ROUGE-L: 0.171 to 0.484) relative to the untuned baseline. These gains show a qualitative shift in structural and lexical alignment rather than marginal improvement, as the baseline model produces near-zero scores due to format mismatch. Overall, we establish corpus-level style characterization and constitutional modeling as an effective and data-efficient strategy for producing radiology reports that conform to authentic radiologist writing conventions. 

---
# Learning to Refer from Estimated Listener Gaze 

**Authors**: Téa Wright, Alane Suhr  

**Link**: [PDF](https://arxiv.org/pdf/2609.14207)  

**Abstract**: We propose to finetune vision-language models to generate more pragmatically optimal referring expressions by transforming observations of incremental listener comprehension, in the form of gaze scanpaths, into learning signals. During training, referring expressions are sampled from the speaker policy being optimized, conditioned on images and target referents; then, a neural listener estimating human gaze behavior maps from images and sampled referring expressions to scanpaths, each represented by a sequence of fixations, with each fixation corresponding to a word in the referring expression. We experiment with several approaches to convert fixation sequences and target referents into token- and sequence-level rewards, which are used to optimize policy parameters. Through evaluation with human listeners, we find that speaker policies trained with gaze-estimating listeners result in significantly more pragmatically-optimal references than base models, reducing sequence length from 15.4 down to 4.0 words while increasing referential success from 75.2 up to 80.0%. Our work demonstrates a promising opportunity for learning to generate utterances through language-based interaction, not only from the explicit signal of communicative success, but also from implicitly-available observations of a listener's process of comprehension. 

---
# A Multi-Stage Agentic Framework for Effective Counter-Narrative Generation and Refinement 

**Authors**: Carmel Kronfeld, Sharva Gogawale, Tetsuro Kobayashi, Irad Ben-Gal  

**Link**: [PDF](https://arxiv.org/pdf/2609.14178)  

**Abstract**: The rapid diffusion of hate speech and misinformation on social networks challenges democratic societies, since direct suppression efforts may deepen polarization, fuel public distrusts, and strengthen extremist narratives. LLM-driven counter-narratives (CNs) offer a promising way to reduce those risks, yet their effectiveness depends on rhetorical and stylistic choices that remain poorly understood. We present a multi-stage agent-based framework for generating, refining, and evaluating CNs, applied to pro-Russian hate and misinformation narratives on the war with Ukraine and adaptable to other domains. A pilot experiment with human evaluators identifies effective technique style pairings, such as repetition with emotional framing enhancing persuasiveness. Building on these insights, we introduce a multi-agent refinement process that iteratively improves CNs for persuasiveness, emotional engagement, and shareability. After human validation confirmed improvement, an automated safety analysis shows that our refined CNs match or improve on expert-written counterspeech. A simulated experiment then shows that they reduce the perceived strength of pro-Russian narratives and consistently outperform a vanilla LLM baseline, highlighting a pathway toward scalable, narrative-specific interventions against hate speech and misinformation. Code and data accompanying this work are publicly available at this https URL. 

---
# Towards Evolving Context Parameterization for Large Language Models 

**Authors**: Xiaobing Shi, Zherui Li, Yiming Jiang, Kun Wang, Yufei Guo  

**Link**: [PDF](https://arxiv.org/pdf/2609.14168)  

**Abstract**: Context parameterization enables large language models (LLMs) to internalize contexts into reusable model parameters, avoiding repeated processing across subsequent queries. However, existing methods typically assume static contexts and lack explicit mechanisms for distinguishing validity states under continual updates. To study this real-world scenario, we formalized the Memory Updating with Sequential Evolution (MUSE) task and constructed MUSE-bench to evaluate update incorporation and unaffected-information preservation. The resulting challenge requires preserving the global state while adjusting the contribution of memory evidence. Motivated by this, we proposed PLUME, a training-free method that constructs a global update representation, activates memory evidence to form a local parameter view, and adaptively integrates their predictions during decoding. Comprehensive evaluation on MUSE-bench demonstrated PLUME's effectiveness in sequential evolution settings, yielding relative improvements of 29.9% in average ROUGE-L Recall and 54.9% in LLM-as-a-Judge. Our codes are available at: this https URL. 

---
# When Tools Get in the Way: The Effect of Unnecessary Tool Availability on LLM Answering 

**Authors**: Saanvi Paturi, Arsen Kenzhebayev, Arham Sethi, Vyas Raina, Ivaxi Sheth, Vatsal Raina  

**Link**: [PDF](https://arxiv.org/pdf/2609.14157)  

**Abstract**: Large language models (LLMs) are increasingly deployed with external tools that extend what they can do beyond their own knowledge. Tools help on tasks that need external information, but their availability may also change how a model handles questions that do not need them. Prior work has mostly asked whether models select and use tools appropriately; whether an unnecessary tool changes the correctness of answers has received less attention. We ask whether making a related but unnecessary tool available affects a model's ability to answer from its own knowledge, and whether a preceding tool interaction changes this behaviour. We construct 500 query pairs across 10 knowledge domains. Each pair consists of a tool query, which needs the domain's tool, and a closed-domain query, which does not. Six LLMs are evaluated with the tool unavailable, available, and available after a prior tool call. Across 3,000 baseline trials the pooled answer rate is 98.2%. When an unnecessary tool is available it falls to 63.5%, with large differences between models. The decrease occurs even when the tool is rarely called, so it cannot be explained by unnecessary tool invocation alone. A one-sentence scope-aware system instruction recovers most of the lost answers. 

---
# One Size Does Not Fit All: Setting Inference Depth from the Questions a Deployment Actually Asks 

**Authors**: Jerry Kaplan  

**Link**: [PDF](https://arxiv.org/pdf/2609.14144)  

**Abstract**: A transformer language model is trained to respond to any prompt, but each deployment asks only a narrow range of questions: a support assistant sees delivery complaints, a coding tool sees Python. Every deployment nonetheless pays the same computation per token. This paper measures how much of that cost is avoidable when the range of prompts is known in advance.
The mechanism examined is early exit: a small, trained component - called a readout - is attached to an intermediate layer and proposes a token, and a confidence test decides whether to emit it or to run the remaining layers. The models are frozen, and the only supervision used is the model's own output on ordinary traffic.
Three findings are reported. First, achievable savings depend strongly on the kind of traffic: at half depth on a 1.5-billion-parameter model, 96 percent of tokens could be emitted early for arithmetic word problems and 8 percent for Chinese-language explanations, at matched token-level fidelity to the full model (a measure whose limits the third finding exposes). Second, of three ways a deployment might use knowledge of its traffic, only customizing the threshold for exiting early is worthwhile: calibrating it per deployment raised exit rates by up to 59 percentage points across three models, and by more than 10 points on most corpora tested. Third, token-level fidelity - the standard evaluation measure in the early-exit literature - fails in domains where tokens can be checked against ground truth: on arithmetic word problems, three models each answered sixty questions correctly when run in full, and between 10 and 28 correctly under early exit, in the configuration that scored highest on fidelity.
The intended setting is small models on personal devices, where generation is limited by memory bandwidth rather than computation. 

---
# GraMRAG: Orchestrating Multi-Agent Multi-Step Reasoning via Graph Memory with Reinforcement Learning 

**Authors**: Zhongyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.14066)  

**Abstract**: Although existing multi-agent Retrieval-Augmented Generation (RAG) systems have demonstrated promise on complex multimodal reasoning tasks, they remain fundamentally limited in reasoning depth and memory structure, suffering from inadequate retrieval and state blindness when answering knowledge-intensive questions. To address these limitations, we propose GraMRAG, a graph memory-guided multi-agent RAG framework that integrates a dynamic multimodal memory graph to enable stable, multi-step multimodal reasoning. We introduce a vision-text bridged reasoning paradigm that unifies multi-scale entity cropping with a ReAct-style visual toolchain, enhancing the long-horizon cross-modal reasoning capability. We further construct a multimodal memory graph that formalizes agent reasoning as a dynamic directed acyclic graph (DAG), explicitly modeling action-observation dependencies to mitigate state blindness and suppress redundant retrieval. Moreover, we propose Topology-Aware Policy Optimization (TAPO) that leverages graph topology for critical path identification and targeted node pruning, enabling fine-grained credit assignment across multi-step reasoning trajectories. Extensive experiments on challenging multimodal benchmarks demonstrate that our approach consistently outperforms existing baselines and achieves state-of-the-art performance on complex long-horizon reasoning tasks. 

---
# Measuring the Creativity of Frontier LLMs in Automated Research 

**Authors**: Yiheng Zhao, Mengzhuo Chen, Chengming Hu, Pengyi Liao, Yiran Pang  

**Link**: [PDF](https://arxiv.org/pdf/2609.14057)  

**Abstract**: Frontier LLMs are increasingly capable of conducting automated research, yet their creativity in this setting has not been systematically evaluated. In this paper, we propose a set of metrics to evaluate creativity along the two dimensions of valueness and novelty. Valueness assesses whether each proposed idea is useful, while novelty is evaluated from three perspectives: whether the same idea has appeared before (Exact-Match P-Novelty), whether a previously unexplored variable or variable combination is explored (Variable-level P-Novelty), and whether the idea directly follows retrieved external knowledge or departs from it (H-Novelty). Our evaluation shows that the models achieve relatively similar scores on most creativity metrics, but differ substantially in Variable-level P-Novelty, which reflects the breadth of research-space exploration. Further correlation and idea-level performance analyses show that Variable-level P-Novelty is the creativity dimension most strongly associated with research performance. 

---
# Unlocking the Unsolvable: Teacher-Guided Curriculum for Data-Efficient RLVR 

**Authors**: Yukang Zhu, Zhen Han  

**Link**: [PDF](https://arxiv.org/pdf/2609.13997)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has shown remarkable success in improving the mathematical reasoning of large language models. Yet problems beyond the model's current capability, where rollouts uniformly fail and no learning signal is produced, are structurally wasted despite marking the most informative training frontier. We show that these otherwise-inert problems can be unlocked via teacher-guided curriculum learning: partial reasoning traces from a stronger model create a graded difficulty landscape, and a backward-chaining curriculum progressively withdraws guidance until the model solves problems unaided. Training on only 128 unsolvable problems matches or exceeds GRPO trained on a full 2,000-problem corpus (~16x data efficiency) on the nine-benchmark average for both base models, while substantially expanding the reasoning boundary measured by pass@k at large k. Furthermore, we identify a distribution-shift cost that is particularly acute in the unsolvable-only regime and propose Monotone Frontier Curriculum (MFC), a method that monotonically drives training toward unguided solving, consistently outperforming existing curriculum methods. 

---
# Mizan: A National Benchmark for Evaluating Large Language Models on Iraqi Arabic and the Iraqi Civic Context 

**Authors**: Nawar S. Alseelawi, Mustafa S. Aljumaily  

**Link**: [PDF](https://arxiv.org/pdf/2609.13980)  

**Abstract**: Arabic large-language-model (LLM) evaluation has matured around Modern Standard Arabic (MSA): aggregated leaderboards such as the Open Arabic LLM Leaderboard (OALL), HELM Arabic, and BALSAM rank models across dozens of MSA tasks, and frontier systems increasingly saturate them. Dialectal Arabic, the language Iraqis actually speak, remains nearly invisible to this infrastructure. We introduce Mizan ("the balance"), Iraq's national benchmark for evaluating LLMs on Iraqi Arabic and the Iraqi civic context: an MSA baseline track paired with an Iraqi track across six axes (dialect comprehension, dialect generation, bidirectional MSA-Iraqi translation, Iraq-specific knowledge, official-document field extraction, and safety), built from 340 originally authored, dually reviewed items with statistically audited answer positions and Wilson intervals on every published score. A pilot evaluation of 27 systems, spanning closed frontier models three days after release, open weights across size tiers, and an Arabic trio of commercial, open-specialized, and sovereign systems, yields four findings. The MSA track saturates while the Iraqi track discriminates, with a consistent 14-18-point per-model gap and statistically tied leaders. Official-document extraction confines every system to 32-56. Arabic-focused specialization behaves as MSA specialization: two dedicated Arabic models score below a size-matched generalist on the Iraqi track. And the safety-hardened tier of the newest model family deterministically refuses innocuous dialect-comprehension items as policy violations, an over-refusal mode invisible to MSA benchmarks. The platform enforces an integrity protocol of immutable snapshots, verification certificates, a human publication gate, and public retraction, all exercised during this study. Code and the public development set accompany the paper. 

---
# Thought without systematicity? Evaluating reasoning models on rule induction tasks 

**Authors**: Simon Schug, Brenden M. Lake  

**Link**: [PDF](https://arxiv.org/pdf/2609.13948)  

**Abstract**: A central tenet of human cognition is systematicity, the principle that understanding one concept is inherently tied to understanding close variations of that concept. Do reasoning models robustly exhibit such systematicity? If so, we would expect consistent performance on structurally equivalent variants of the same task. Here, we extend established rule induction tasks from cognitive science to assess the systematicity of thought in current reasoning models. Each task family has compositional structure that we use to create structurally equivalent task variations through task isomorphisms such as recombination and substitution. We find that despite being able to correctly solve a task, models often fail on structurally equivalent variants of the same task. These findings suggest that many model behaviors lack systematicity, rendering it difficult to robustly establish the cognitive abilities of reasoning models beyond the particular contexts they were evaluated in. 

---
# CRITICS - Critical Science Without Borders: Language Models to Promote Critical Thinking in Science Education 

**Authors**: Rodrigo Agerri, Itziar Aldabe, Elena Cabrio, Mark Cieliebak, Jan Deriu, Mariana Flores, Jurgita Kapociute-Dzikiene, Dovile Kuiziniene, Arantza Rico, Aritz Ruiz-Gonzalez, Aitor Soroa, Mantas Vaskevicius, Serena Villata  

**Link**: [PDF](https://arxiv.org/pdf/2609.13942)  

**Abstract**: The CRITICS project addresses science accessibility and literacy by converging advanced Machine Translation (MT) based on Large Language Models (LLMs) with educational technology. By leveraging MT systems specifically optimized for scientific content, educational institutions can provide accurate, culturally relevant translations of scientific materials in students' native languages, ensuring that complex scientific concepts are comprehensible while maintaining technical accuracy. Building on these translations, the project explores the design and evaluation of innovative science teaching-learning proposals grounded in curriculum-aligned teaching-learning. Thus, CRITICS will investigate key components of scientific argumentation and critical thinking practices together with textual feedback aligned with learning objectives and assessment criteria inspired by competence-based evaluation frameworks. CRITICS aims to break down language barriers to accessing cutting-edge research and educational materials currently available only in high-resourced languages, thereby facilitating the democratization of scientific knowledge and fostering critical thinking in science education. 

---
# Inter-Rater Reliability of LLM and Rule-Based Annotation for Inferential Narrative Features: Three Studies on a Turkish Corpus 

**Authors**: Levent Bulut  

**Link**: [PDF](https://arxiv.org/pdf/2609.13936)  

**Abstract**: Datasets that ship automatically generated feature annotations invite a question rarely asked of them: would a human agree with those labels? This report answers that for the Objective Projection corpus, a Turkish narrative dataset whose scenes carry a per-scene applied_rules field from a rule-based detector over six craft features -- two prohibitions (emotion labelling, simile) and four positive techniques (materialized metaphor, micro-focus, temporal anchor, atmosphere contradiction).
Three studies are reported. Study 1 ($n = 120$) scores the detector against blind labels from the scheme's own author. Study 2 ($n = 100$, a disjoint scene set) scores the detector plus Gemini 2.5 Flash and Grok against an independent non-expert rater whose labels were locked before any machine ran. Study 2b re-runs the identical protocol with Claude Fable 5 (High) and ChatGPT 5.5.
The central result concerns one rule. On materialized metaphor -- closest to the methodology's theoretical core -- the five machine labellers returned positive rates of $0$, $1$, $40$, $72$ and $78$ out of $100$ scenes, against a human count of $9$. Cohen's $\kappa$ was at or indistinguishable from chance for five of six labellers, across both human references and both scene sets: $0.004$, $0.015$, $0.000$, $0.019$, $0.027$. Raw agreement ranged from $74.7\%$ to $84.5\%$, an artefact of class imbalance rather than a sign of competence.
We deliberately do not resolve this into a single story. Two readings survive: the feature is genuinely inferential and beyond current automatic detection, or the rule's definition is not yet operational enough for any rater to apply consistently -- including the human. Distinguishing them needs a second independent human rater, which this report does not have and therefore does not claim. 

---
# North Small Translate: Advanced Cost-Effective Translation (Cohere CAT+) 

**Authors**: Tom Kocmi, Alexandre Bérard, Phil Blunsom, Samuel Cahyawijaya, Shaun Cassini, Nicholas Frosst, Ona de Gibert, Aidan Gomez, Nithya Govindarajan, Shun Kiyono, Olivia Lasche, Lawrence Rogers, Kelly Marchisio, Nikita Moghe, Yash More, Camila Moran-Hidalgo, Yiyang Nan, Michael Sachs, Trisha Starostina, Daan van Stigt, Spencer Rarrick, Sebastian Vincent, Ivan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.13916)  

**Abstract**: We present North Small Translate, an open-weight, LLM-based machine translation (MT) model with instruction-following capabilities built on the same foundation as Cohere's Command A Plus, a mixture-of-experts architecture with 25 billion active parameters out of 218 billion total parameters. North Small Translate is trained using difficulty sampling to obtain challenging documents and a five-step training protocol combining supervised fine-tuning, direct preference optimization, and online reinforcement learning. We prioritized throughput through a non-reasoning base model and supplemented with optional agentic capabilities to unlock translation quality gains. North Small Translate is trained to perform MT-related tasks, including post-editing and quality estimation, as well as related tasks such as general instruction following. The model achieves top MT performance across 50 languages in the class of models under 1T parameters, with no need to run expensive reasoning at inference time. 

---
# Phorecaster365: A Human-Supervised Reference Architecture for Hybrid Pharmaceutical Sales Forecasting and Planning Decision Support 

**Authors**: Houman Kazemzadeh, Kamyar Naderi  

**Link**: [PDF](https://arxiv.org/pdf/2609.13907)  

**Abstract**: Pharmaceutical sales forecasts inform planning across products, regions, and distribution channels, yet their interpretation depends on inventory availability, transaction semantics, product lifecycle, and the information available when each forecast is issued. A model prediction alone does not preserve these conditions or establish whether a forecast is suitable for operational use. We present Phorecaster365, a human-supervised reference architecture that connects enterprise resource planning data to reviewable pharmaceutical sales forecasts. The architecture separates source ingestion, product-region series construction, temporally valid feature generation, statistical and machine-learning modeling, ensemble formation, uncertainty assessment, planner review, and lifecycle governance. Its central intermediate representation is a forecast context package that preserves the source snapshot, forecast target, temporal cutoff, available covariates, data-quality state, and hierarchy version. A corresponding forecast evidence package links predictions to model and calibration versions, exceptions, human adjustments, and publication history. Development experience with a synthetic panel of 10,950 daily records across 30 product-region series informs the design. Historical experiment summaries are retained only as descriptive evidence of development because their evaluation does not establish independent predictive validity. We specify a rolling-origin evaluation protocol, baseline and ablation requirements, uncertainty and robustness assessments, and a staged pathway from synthetic testing to use in governed planning. The contribution is an implementation-neutral system design and validation framework; the report does not establish real-world forecasting accuracy, comparative superiority, or operational benefit. 

---
# SHIFT-M3: Pre-fusion Alignment-based Consistency Screening for Multimodal ECG Record Integrity 

**Authors**: Md Ashik Khan, Md Nahid Siddique  

**Link**: [PDF](https://arxiv.org/pdf/2609.13874)  

**Abstract**: Multimodal clinical AI typically assumes that the waveform, report, metadata, and downstream predictions attached to a record belong to the same patient. In practice, linkage failures can silently assemble individually plausible but cross-patient components, creating a safety problem that standard predictive models are not designed to detect. We study this problem as multimodal record integrity triage: given an assembled record, should its modalities be trusted to belong together? We introduce SHIFT-M3, a lightweight text-based pre-fusion screen that measures alignment-based consistency between two separately produced ECG text views: an LLM-generated interpretation and a clinical report summary. On 784,680 MEETI ECG records, SHIFT-M3 achieves 97.6% TPR@5% FPR for full text-view swaps (AUROC 0.996), 90.3% for partial swaps (AUROC 0.974), and 97.7% for label-matched hard negatives (AUROC 0.996) with only 573,569 parameters. Compared with same-dataset lexical baselines, the gains are largest on partial swaps and hard negatives, suggesting that the model is learning more than surface overlap. We also introduce the CMST (Conflict-type Multimodal Stress Test) evaluation taxonomy, a three-seed stability study, a loss ablation, a temporal-tolerance sweep, and a shared-token masking control. The main remaining failure mode is longitudinal ambiguity: at the default operating point, same-patient cross-visit pairs still produce 87.0% Type-II false positives. 

---
# Bangla Sentence Function Classification: Corpus Development, Model Benchmarking, and Interpretability 

**Authors**: Swapnil Kundu Argha, Abdullah Al Shafi, Rowzatul Zannat, Shoumik Barman Polok, Abdul Muntakim, Jannatul Ferdousi, M.A. Moyeen  

**Link**: [PDF](https://arxiv.org/pdf/2609.13869)  

**Abstract**: Automatic sentence function identification is important for many downstream natural language processing (NLP) applications such as dialogue systems, text-to-speech synthesis, and machine translation. However, benchmark resources for Bangla sentence function classification remain limited. To mitigate this gap, this paper introduces a corpus of 10,000 Bangla sentences, manually annotated into four functional categories, namely declarative, interrogative, imperative, and exclamatory. The corpus is nearly balanced across the four classes, with high annotation reliability reflected by a Fleiss\' Kappa of 0.82. Furthermore, we evaluate multiple feature representations, including Bag-of-Words (BoW), TF-IDF, and Word2Vec, with several classical machine learning classifiers. In addition, two heterogeneous ensemble models, namely Single-Level Ensemble (SLE) and Double-Level Ensemble (DLE), are utilized to improve classification performance. Experimental results show that TF-IDF consistently outperforms Word2Vec, likely due to its ability to emphasize discriminative lexical cues associated with sentence functions, particularly given the relatively small corpus used to train Word2Vec. The DLE model with TF-IDF features achieves the best performance with accuracy and macro-F1 of 0.95, demonstrating the effectiveness of sparse lexical representations and heterogeneous ensemble learning for this task. Further cross-validation confirms the robustness of the approach, while LIME-based interpretability provides insights into model predictions. The developed corpus and model benchmarking establish strong baselines for Bangla sentence function classification. 

---
# ShopEase: A Generative AI-Based Multi-Agent Framework for Intelligent Enterprise Customer Support Using Hybrid Retrieval-Augmented Generation 

**Authors**: Aakash Kumar Tiwari, Somesh Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2609.13856)  

**Abstract**: Enterprise customer support systems must answer customer questions correctly, retrieve the right policy information, use customer context, and pass difficult cases to human agents when needed. This paper presents ShopEase, a Generative AI-based multi-agent framework for enterprise customer support. The system combines six components: Intent, CRM, Memory, Hybrid RAG, Escalation, and Supervisor, and uses LLaMA 3.2 running locally through Ollama for response generation. The retrieval module combines FAISS (dense retrieval) and BM25 (sparse retrieval), and six configurations are evaluated: BM25-only, FAISS-only, Fair RRF, Weighted RRF, RRF with Cross-Encoder, and Top-10 Hybrid with Cross-Encoder. Instead of using a fixed mapping between intent and policy, the policy category is decided directly from the retrieved documents. The system was evaluated on 2632 held-out customer queries across six categories: Refund, Return, Shipping, Cancellation, Damaged Product, and Unknown. FAISS-only achieved the highest accuracy of 85.37\% (2247 correct predictions), closely followed by Weighted RRF at 85.07\%. BM25-only achieved only 55.74\% accuracy. Adding cross-encoder reranking did not improve results: RRF with Cross-Encoder reached 83.24\%, and Top-10 Hybrid with Cross-Encoder reached 81.88\%, while also increasing response latency. Category-level analysis shows strong performance on Shipping, Cancellation, and Return, while Unknown queries remain the main source of errors. Statistical testing using McNemar's test shows no significant difference between FAISS-only and Weighted RRF, though both perform significantly better than Fair RRF and the cross-encoder configurations. Overall, dense retrieval gives the best accuracy on this dataset, and additional reranking adds processing time without improving classification performance. 

---
# Measuring the Cost of Variety Conflation in Multilingual MT Evaluation: Adding Mozambican Xichangana, Nyanja and Sena to FLORES+ 

**Authors**: Felermino D. M. A. Ali, Delfina Lázaro Mateus, Manuel Valente Mangue  

**Link**: [PDF](https://arxiv.org/pdf/2609.13847)  

**Abstract**: In this paper, we extend FLORES+ with Portuguese-source evaluation sets for three Mozambican Bantu varieties: Xichangana, Mozambican Nyanja, and Sena. We compare Xichangana with the existing Tsonga reference and Mozambican Nyanja with Chichewa, and evaluate NLLB-200, Google Translate, GPT, and a variant-aware NLLB model. Holding system output fixed reveals substantial reference sensitivity. On \textit{devtest}, changing only the reference from Tsonga to Xichangana reduces spBLEU by 13.10 points for NLLB-200 and 15.30 for Google. On matched Nyanja subsets, replacing Chichewa with Mozambican Nyanja produces smaller but consistent reductions of 3.03 and 6.10 spBLEU, respectively. Variant-aware fine-tuning reverses this pattern on the intended targets: relative to NLLB-200, it improves Xichangana by 7.04 spBLEU and Mozambican Nyanja by 5.33 on \textit{devtest}, while losing performance on the sibling references. GPT is competitive on Tsonga and Chichewa but substantially weaker on the Mozambican varieties. For Sena, the finetuned model reaches 12.64 spBLEU and 36.21 chrF++ on \textit{devtest}. These findings motivate variety-aware language identifiers, references, and reporting for cross-border languages or language dialects/variants. The data is publicly available on Hugging Face at this https URL 

---
# Sweet Talkers: How Query Formulation Shapes Sycophancy in Romantic Relationship Advice 

**Authors**: Helena Choi, Edric Castel Hao, Karl Bautista, Francis Gabriel Magleo, Renzo Panti, Danielle Beatrice Olalia  

**Link**: [PDF](https://arxiv.org/pdf/2609.13841)  

**Abstract**: Large language models (LLMs) are increasingly used for emotional support and relationship advice, where a model's tendency to preserve a user's face can inadvertently reinforce harmful interpersonal behaviors. To systematically examine this risk, we developed the Romantic Relationship Advice-Seeking Prompts (RRASP) dataset of 2,400 prompts across five relationship themes and evaluated social sycophancy using the ELEPHANT framework on two consumer-facing models, GPT-5 Mini and Gemini 3 Flash. Contrary to our initial hypothesis, grammatical mood alone did not produce systematic differences in sycophantic behavior, suggesting that what a user implies matters more than how they phrase it. Instead, perspective-driven framing had a stronger influence, with gaps between original and flipped prompts widening in follow-up responses. Consistent increases in framing and moral sycophancy across turns indicate that models become more likely to accept a user's stated premises and affirm their ethical stance as a dialogue progresses. Notably, Gemini 3 Flash exhibited substantially smaller increases in moral sycophancy than GPT-5 Mini, suggesting it is more resistant to reinforcing ethically problematic positions across turns. 

---
# When Consistency Does Not Mean Reliability: Evaluating Local LLM Judges Against Human Ratings 

**Authors**: Aakash Kumar Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2609.13824)  

**Abstract**: Large language models (LLMs) are increasingly used to evaluate the responses of other language models. This approach, known as LLM-as-a-Judge, is faster and cheaper than human evaluation. However, a judge may produce consistent scores without necessarily agreeing with human evaluators. In this work, we study this issue using two local open-weight LLM judges, LLaMA-3-8B and Qwen2.5-7B. We evaluate 300 responses generated by an instruction-tuned GPT-2 (124M) model for 100 questions covering five categories: factual knowledge, instruction following, mathematics, reasoning, and writing. Each response is scored by nine human annotators and is evaluated three times by each LLM judge using the same rubric. We compare the judge scores with the average human scores using Pearson correlation, Spearman correlation, mean absolute error (MAE), signed bias, and self-consistency. LLaMA-3-8B shows a Pearson correlation of 0.275 with human scores, while Qwen2.5-7B achieves 0.340. Their MAEs are 27.71 and 18.64, respectively. Despite this limited agreement, both judges show high self-consistency, with exact consistency rates of 97.3\% for LLaMA-3-8B and 92.3\% for Qwen2.5-7B. These results show that high self-consistency does not necessarily indicate high agreement with human judgments. Our findings highlight the need to evaluate both consistency and human alignment when using local LLMs as automatic judges. 

---
# DARE: Dialectical Agentic Reasoning for Structured Knowledge Fact Checking 

**Authors**: Yifei Li, Xiaohan Zheng, Wentao Qian, Liansheng Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2609.13808)  

**Abstract**: Structured knowledge fact checking aims to determine the truthfulness of natural language claims by reasoning over structured evidence. Recent program-generation approaches leverage large language models (LLMs) to generate executable graph reasoning programs, achieving strong performance on structured knowledge fact checking benchmarks. However, these methods remain limited by invalid relation generation, single-path reasoning that lacks self-correction, and biased evidence assessment that tends to overestimate supporting signals. We propose Dialectical Agentic Reasoning (DARE), a multi-agent framework that formulates structured knowledge fact checking as an iterative retrieve-reason-reflect process. DARE integrates relation-grounded evidence retrieval to constrain reasoning to valid structures, dialectical bidirectional verification to evaluate evidence from both supporting and refuting perspectives, and confidence-driven meta-reflection to dynamically determine whether additional evidence exploration is necessary. Extensive experiments demonstrate the effectiveness of DARE in structured knowledge fact checking, with an 8B backbone achieving 88.12% accuracy and matching or surpassing GPT-4o-based program-generation baselines, which attests to the efficacy of dialectical agentic reasoning in eliciting the latent reasoning capabilities of LLMs. 

---
# Understanding the Limits of Agentic ICD Coding 

**Authors**: Chong Yock Eng, Yushi Cao, Yiming Chen, Kezhi Mao, Hongchao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2609.13806)  

**Abstract**: ICD-10-CM codes are alphanumeric codes used in the US to classify diagnoses and injuries for medical billing and epidemiological reporting. Standard ICD-10-CM benchmarks report aggregate metrics that obscure performance on complex coding scenarios. We evaluate neural, workflow, and agentic systems on a rarity-stratified set of MIMIC-IV discharge summaries and identify two orthogonal failure modes. Neural classifiers exhibit a 0.43 micro-F1 gap between rare and common codes. Workflow systems handle rare codes well but score near zero on injury and external cause codes that require multi-step guideline following. A tool-augmented agentic configuration with structured access to official ICD-10-CM reference materials recovers up to 0.34 micro-F1 on this subset. No single system dominates across all conditions. 

---
# SyRHM: Symbolic-Language-Enhanced Reasoning with Associative Retrieval for Zero-shot Harmful Meme Detection 

**Authors**: Hanling Wang, Chenlong Wei, Yingjuan Li, Di Wu, Yuchao Zhang, Xiaohui Zhu, Yao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2609.13794)  

**Abstract**: Detecting harmful memes is critical for maintaining safe online communities. However, harmful intent is often implicit, arising from visual-textual incongruity and cultural stereotypes, which challenges existing multimodal detectors. We propose SyRHM, a framework that decomposes harmful meme detection into meaning-grounded retrieval and symbolic-language-enhanced multi-stage reasoning. SyRHM retrieves semantically related memes by parsing multimodal content into textual elements and descriptions, providing grounded context beyond surface-level similarity. Building on the retrieved context, SyRHM uses a translator stage to convert multimodal inputs into symbolic intermediate representations, and then performs multi-stage reasoning via planner and solver stages, enabling expressive and interpretable analysis of harmful intent. Experiments on FHM, HarM, and MultiOff demonstrate the effectiveness of SyRHM, achieving superior performance on most evaluation settings against multimodal and reasoning-based baselines, while providing reasoning traces for harmful content. The code is available at: this https URL 

---
# HyperProve: Answer-Guided Hypergraph Expansion for Multi-Hop Question Answering 

**Authors**: An Nguyen Phu, Dung Nguyen Quang, Luu Hieu An, Linh Ngo Van, Trung Le, Thien Huu Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2609.13768)  

**Abstract**: Multi-hop question answering often fails when retrieval treats evidence as isolated matches to the original question, since the facts needed to answer a complex question are usually connected through intermediate entities, relations, and constraints. We propose HyperProve, a retrieval-augmented QA framework that addresses this challenge by coupling question decomposition with answer-conditioned expansion over a hypergraph of atomic facts. HyperProve does not use atomic facts, hypergraphs, or iterative retrieval in isolation; instead, it carries intermediate answers and supporting hyperedges as retrieval state, then uses that state to bias the next local hypergraph expansion. This design enables HyperProve to construct coherent evidence chains for final answer generation while making the retrieval process stateful and fact-centered. Across multi-hop QA benchmarks, HyperProve achieves the best overall performance in our evaluation, outperforming the strongest baselines by an average relative improvement of 6.2% in answer accuracy and 4.9% in F1. 

---
# Inside VLM Chart Reading: Tracing Value Reading from Vertical Bar Charts Across Space and Depth 

**Authors**: Tianhao Niu, Qingfu Zhu, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2609.13745)  

**Abstract**: Vision--language models (VLMs) can answer chart questions accurately, but output accuracy does not show how they combine the evidence needed to recover an exact value. We study vertical-bar value reading with controlled counterfactual activation patching in Qwen2.5VL-7B-Instruct and InternVL3.5-8B. The study connects three analyses: (1) The single-factor results show that the changed bar-top region restores much more answer preference than the unchanged bar body, despite containing fewer visual tokens. Legend- and series-related states also lose local recoverability earlier than bar-geometry and axis-scale states. (2) In the handoff analysis, restoration shifts from visual legend regions in early layers to prompt-series positions in middle layers. Resetting the prompt-series state selectively reduces legend-source rescue, supporting its role as a partial mediator. (3) In the factorial analysis, both models can use geometry and scale states from separate donors to favor the combined target. InternVL performs similarly when the states come from separate donors or one image, while Qwen shows lower restoration for separate donors which suggests higher context sensitivity. Together, these results provide preliminary causal evidence for localizing the internal computations that support exact bar-value reading. 

---
# ForeSight: Enhancing Risk Monitoring via Early Safety Signal Distillation 

**Authors**: Hanling Wang, Chenlong Wei, Ling Xu, Hanyan Niu, Qi Cao, Shizhou Huang, Yang Yang, Xiaohui Zhu, Yao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2609.13737)  

**Abstract**: As large language models (LLMs) are increasingly deployed, the generation of harmful content has become a critical safety concern. Existing safeguards operate at the input, output, or streaming-generation stages, while early-risk methods that rely on surface tokens or output logits may suffer from weak initial signals, and internals-based detectors using dense representations may retain highly entangled and redundant safety-irrelevant information. It therefore remains unclear whether the earliest post-generation hidden states already contain reliable signals about final-response harmfulness. To address this gap, we propose ForeSight, a first-token output-risk forecasting framework that distills weak and redundant early safety signals into compact, layer-aware risk representations. Experiments on five safety benchmarks and two target models demonstrate that ForeSight achieves superior and efficient early-risk forecasting while relying solely on first-token hidden states. The code is available at: this https URL 

---
# PolicyMem: Geometric Policy Memory for LLM Governance 

**Authors**: Yuanchen Bei, Zhengzhang Chen, Yanjun Zhao, Haoyu Wang, Hanghang Tong, Haifeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.13734)  

**Abstract**: As large language models (LLMs) are increasingly deployed in real-world high-stakes applications, effective governance has become essential. Existing safeguards largely follow two paradigms: learning-based guards provide strong semantic discrimination but couple policy behavior to trained models and taxonomies, while programmable frameworks offer flexible control but require substantial manual prompt and workflow engineering. Neither externalizes policies as reusable operational states, making it difficult to consistently reuse policy evidence across detection, intervention, and verification. In this paper, we introduce PolicyMem, a geometric policy memory that externalizes natural-language policies as reusable geometric memory objects represented by low-rank subspaces in a shared representation space. A memory writer compiles natural-language policies into policy memory slots, and query-response pairs read the policy memory through projection energy. The resulting policy-evidence profile directly mediates the safety verdict and is reused for policy attribution and post-intervention verification. Coupled with a response rewriter, PolicyMem enables a detect-rewrite-verify loop for LLM governance. Across five widely used benchmarks, PolicyMem achieves state-of-the-art unsafe behavior detection while enabling effective policy attribution, rewriting, and post-intervention verification through the shared policy memory. 

---
# Scaling Hindi Quantum Natural Language Processing through Automatic Pregroup Supertagging 

**Authors**: Gautami Sanjay Naik, Krishna Bhatia, Mithun Paul Saint-Germain, H Aswath Babu  

**Link**: [PDF](https://arxiv.org/pdf/2609.13721)  

**Abstract**: Quantum Natural Language Processing (QNLP) uses pregroup grammars to translate grammatical structure into diagrammatic representations and quantum circuits. Recent Hindi QNLP work has shown that Hindi-specific pregroup grammars can support grammar-sensitive compositional models, but grammatical type assignment is still largely manual, limiting scalability. This paper formulates automatic Hindi pregroup supertagging as a token-level classification task. Using a manually annotated corpus of 380 Hindi sentences, we evaluate lexical, contextual, prompting-based, lexical-repair, and suffix/morphology-aware methods. Results show that simple lexical and contextual models are strong in this low-resource setting: contextual backoff achieves the best completed accuracy of 64.56\%, while raw Qwen2.5 prompting reaches only 11.65\%. Lexical repair raises LLM-assisted prediction to 64.08\%, demonstrating the value of constraining generative outputs with symbolic grammar knowledge. Diagnostic analysis further shows that seen and unambiguous tokens are much easier than unseen tokens, and suffix/morphology features improve karaka-token accuracy but not overall performance. These results show that automatic Hindi pregroup assignment is feasible and can reduce reliance on manual annotation in future multilingual QNLP pipelines. 

---
# When Edit Localization Amplifies Relative Selection Bias: Gradient Geometry, Target Mismatch, and Importance Weighting 

**Authors**: Shengwei Zhang, Haoda Dai, Yifei Li, Yuheng Song  

**Link**: [PDF](https://arxiv.org/pdf/2609.13709)  

**Abstract**: Human corrections identify editable spans, but the examples receiving corrections may come from a selective feedback channel. We analyze this interaction at a fixed model checkpoint by decomposing a localized gradient into edited and retained untouched components. Squared relative selection bias is a ratio of quadratics whose derivative has the sign of an explicit quadratic polynomial. Localization can increase, decrease, or nonmonotonically change this diagnostic; its direction depends on component biases and geometry. Oracle importance weighting recovers the population mean for each fixed localization objective, but these objectives have different targets. Against one common full-gradient target, we derive the finite-sample mean-squared error, an analytic optimal retention coefficient, and a fixed-clipping extension. Exact finite-population calculations and 10,000 Monte Carlo repetitions per sample size verify the identities and counterexamples. Public human-post-edit experiments use two translation directions and pretrained models, with declared synthetic selection. An English-German extension differentiates 73.89 million native parameters. In all three declared settings, hard localization has higher relative bias but lower absolute bias than full retention. Untouched-component biases are nonzero and selected component means have negative inner products, so the general criterion applies where the simple unbiased/aligned explanation fails. Output-bias diagnostics show the same endpoint ordering of relative bias across both directions, with one interior maximum. Mechanisms reuse each language's records and include a mixture; they are not independent replications. The evidence separates relative amplification from absolute gradient error and establishes estimation properties, without inferring translation-quality gains or identifying actual complaint propensities. 

---
# Not all Negation Cues are Equal: Affixal Negations Yield Better Negation Understanding 

**Authors**: Tian Tan, Eduardo Blanco  

**Link**: [PDF](https://arxiv.org/pdf/2609.13685)  

**Abstract**: Negation remains a longstanding challenge for both language models (LMs) and large language models (LLMs). Prior work mainly focuses on a small set of high-frequency single-word negation cues, such as not and never, with limited exploration of broader negation types and modern LLMs. To address this gap, we construct NegCue, a large-scale dataset containing over 1.8M samples spanning single-word, multi-word, and affixal negation with more than 200 unique cues. We further pre-train both encoder-only LMs and LLMs on NegCue to investigate how different negation types affect negation understanding. Experiments on five downstream benchmarks show that negation types contribute unevenly to performance gains under the same training scale. In particular, affixal negation yields the largest improvements, while the gains from the commonly studied single-word negation remain modest. Moreover, our results demonstrate that further pre-training improves negation understanding for both LMs and LLMs. 

---
# LayerRoute: Adaptive Layer-Skipping with LoRA-Preserved Quality for Efficient LLM Inference 

**Authors**: Prateek Kumar Sikdar  

**Link**: [PDF](https://arxiv.org/pdf/2609.13682)  

**Abstract**: We introduce LayerRoute, a parameter-efficient method for adaptive transformer layer-skipping that combines per-layer hard-gated routing (trained via a straight-through estimator) with joint LoRA fine-tuning. LayerRoute augments each of the 24 transformer blocks in Qwen2.5-0.5B-Instruct with a lightweight per-layer router (~21.5K parameters) and LoRA adapters (rank 8, ~1.08M parameters), training both jointly under a gate-regularized language-modeling objective. Across 10 independently-seeded training runs, LayerRoute converges to an identical skip-pattern structure in every run - a consistent set of 9 middle layers (8-16) becomes skip-eligible in all 10 seeds - and delivers genuine, verified wallclock speedup in every run (1.02x-1.06x, mean 1.04x). Quality is preserved or improved in every configuration tested: joint LoRA adaptation yields a perplexity improvement over the unmodified backbone in all 10 seeds (mean delta = -1.16 and -1.11 across the two evaluation splits used). We further verify the router performs genuine, non-trivial per-input computation: gate decisions in skip-eligible layers change the actual skip/run outcome for 87-100% of held-out samples, confirming real input-dependent routing rather than a fixed pruning pattern. LayerRoute trains in under 7 minutes on a single A100 and adds negligible overhead beyond the routing decision itself. We report our full reproducibility methodology, including a systematic diagnostic investigation into what determines the router's per-input decisions, as part of this work. 

---
# The University of Melbourne WMT 2026 CreoleMT Submission: A Domain-Balanced Approach to Low-Resource Pacific Creole Machine Translation 

**Authors**: Raphaël Merx, Nick Thieberger, Ekaterina Vylomova  

**Link**: [PDF](https://arxiv.org/pdf/2609.13615)  

**Abstract**: For our submission to the WMT26 Creole Language Translation Shared Task, we focus on machine translation (MT) models for Pacific creoles: Tok Pisin, Bislama, and Solomon Pijin, with particular attention to broad domain performance. After pre-training on a large collection of domain-imbalanced data, we continue fine-tuning on a diverse mix of domain-balanced data. We rely on a number of data collection and preparation techniques, including LLM-assisted respelling and alignment, back-translation, and distillation from Gemini for domains originally not present in training data. Evaluated on Bouquet and a novel test set made of spoken language transcripts, our models beat open model baselines by 3+ chrF++ points in all directions with human-original references. Looking ahead, we plan to develop human-translated test sets for Solomon Pijin and Bislama, and to distil our best models into much smaller ones that retain broad domain coverage. 

---
# In the Blind: Building Pseudo-References for MT Evaluation 

**Authors**: Diptesh Kanojia, Chi-kiu Lo, Archchana Sindhujan, Samuel Larkin, Greg Hanneman, Alon Lavie  

**Link**: [PDF](https://arxiv.org/pdf/2609.13611)  

**Abstract**: The WMT26 General MT task evaluates systems on 10 language pairs that have no human references (neither translated from scratch nor post-edited from MT output by humans). We describe how we built the pseudo-references for these pairs and six other language pairs (in which some forms of human references are available): seven models translate the 3,277 official documents under up to five prompt conditions, giving a total of 26 system-prompt combinations; then three reference-free quality estimation (QE) models score every candidate; and a per-document selector picks one translation, which GPT-5.5 post-edits where needed. Working without references exposed a failure mode of QE-guided selection: the metrics rank fluent output in the wrong language above correct translations. Adding a confidence-scaled language identification penalty to the score fusion drives the wrong-language count to zero, and the resulting selector still scores better on MetricX than the rank-fusion baseline it replaces. Since no references were available for these pairs while we were building them, we calibrate every selection decision on last year's WMT25 human judgments. The human evaluation, released after construction, shows the cost of getting selection wrong: our references stand with the strongest participating systems when the selector kept a frontier-model candidate, and fall up to 17 ESA points below them when it did not. We release the selection method and the provenance of every reference (this https URL) 

---
# Same Patient, Different Order: Action-Level Reliability of Clinical LLM Agents Under Repeated Runs 

**Authors**: Rohith Reddy Bellibatlu, Manpreet Singh, Zhoutian Han, Wenbin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.13582)  

**Abstract**: A clinical agent benchmark can report the same verdict on identical inputs while the agent files a materially different order on each run. Such agents order tests, request medications and place referrals, yet benchmarks typically score one run per task and rarely ask whether identical inputs produce identical actions; MedAgentBench, the benchmark we use, scores a single attempt and says so. To measure this gap we introduce "same-input rerun", which replays a task with every input held fixed and compares the orders rather than the score, with six reliability metrics, and apply it to 1000 MedAgentBench runs across 50 tasks from its five write-capable families, two open-weight models below ten billion parameters quantised to four bits, and two temperatures. The study establishes that action-level divergence exists and can pass unrecorded by the score, not that any rate generalises. Under the 8B model at temperature 0.7, all 43 ordering groups emit a different set of orders across five identical runs, 26 emit the order on some runs and not others, and 28 record a different coded value, dose or analyte. In 22 of those 43 the benchmark reports the same failing verdict for materially different behaviour, as it does for all 10 divergent groups of the 4B model at 0.7. Orders also reach different endpoints across runs, one of which the record server rejects while the agent is told it succeeded. These findings motivate repeated-run evaluation, action-level stability reporting and execution-faithful environment feedback in clinical-agent benchmarks. 

---
# Toward Complete Hospital Discharge Summarization with Abstract Meaning Representation 

**Authors**: Paul Landes, Sitara Rao, Aaron Jeremy Chaise, Barbara Di Eugenio  

**Link**: [PDF](https://arxiv.org/pdf/2609.13581)  

**Abstract**: Discharge summaries are lengthy medical documents that summarize a hospital in-patient visit. Automatically generating them can reduce documentation burden and return clinician time to patient care. Whereas Large Language Model (LLMs) could be used for this task, their Achilles heel is hallucinations, which can have drastic consequences for clinical documentation. We present an evidence-driven alignment framework for discharge summarization at the clinical encounter level, that treats provenance as a first-class constraint, using semantic graphs and deep learning models. Each summary sentence is selected and organized via cross-document semantic alignment and is accompanied by explicit evidence links to its source spans. We show our results on two corpora: a publicly available corpus (MIMIC-III) and clinical notes written by physicians at the University of Illinois Hospital (UIC Health). Additionally, we make source code and trained models available. 

---
# Domain-Specific Jargon in Large Language Models: A Comparative Analysis between General-Purpose and Specialist Models 

**Authors**: Darin Keng, Zhewei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2609.13556)  

**Abstract**: Large Language Models (LLMs) have shown remarkable proficiency on general-purpose tasks, yet their performance often degrades in highly-specialized technical domains. Moreover, little is known about how parametric knowledge of domain-specific terms is encoded within these models. We address this gap by contributing two novel medical jargon evaluation benchmarks and evaluate a general-purpose Llama-3.1 model against a variant fine-tuned on medical-domain data. Surprisingly, the general-purpose model outperforms the medically fine-tuned model on both tasks. Using mechanistic interpretability tools, we find systematic patterns of miscalibration for the medically fine-tuned model. Instead of reorganizing parametric knowledge, the fine-tuned model places greater emphasis on a small subset of model components associated with jargon-favoring predictions. We find that applying component reweighting strategies against the benchmark tasks successfully suppresses these components and closes the gap with the general-purpose baseline. We also observe that some jargon-sensitive components transfer knowledge to the same tasks involving materials science jargon, suggesting they encode a partially domain-agnostic notion of specialized terminology. Our results provide a case study in which a medically fine-tuned checkpoint does not improve jargon comprehension over its general-purpose counterpart, highlighting that domain adaptation should not be assumed to yield better performance on specialized terminology. 

---
# Harmfulness Propagation Dynamics: Layer-wise Trajectories of Adversarial Intent in Large Language Models 

**Authors**: Noor Islam S. Mohammad, Uluğ Bayazıt  

**Link**: [PDF](https://arxiv.org/pdf/2609.13534)  

**Abstract**: We identify \textbf{Harmfulness Propagation Dynamics (HPD)}: for harmful prompts, the projection of the last-token hidden state onto a learned harm direction rises monotonically with transformer depth, whereas benign prompts remain flat or oscillatory. This cross-layer signature reflects harmful intent as a \emph{progressively resolved} semantic property: surface form appears early, while pragmatic intent consolidates later, making the \emph{trajectory shape} more informative than any single-layer snapshot. Moreover, LDA-based harm directions, learned per layer, remain stable across random splits (pairwise cosine similarity $>0.97$), supporting the projection sequence as a reproducible structured signal. Building on HPD, we introduce \textbf{\herald{}} (\textbf{H}armful \textbf{E}ncoding \textbf{R}ecognition via \textbf{A}ctivation \textbf{L}ayer \textbf{D}ynamics). This lightweight input moderator extracts a seven-dimensional feature record, slope, curvature, monotonicity, onset layer, and related statistics from the cross-layer projection sequence and classifies it with a 288-parameter MLP. \herald{} stores one $d$-dimensional direction per layer ($262$\,KB for a 32-layer, $d{=}4096$ model), requires no gradient computation during training, and adds only $2.6{\times}10^{-6}$ prefill FLOPs at inference. Across eight prompt-harmfulness benchmarks and four model families, \herald{} achieves an average F1 of $89.3$ on OLMo2-7B, surpassing all tested guard models on adversarial jailbreak detection ($98.4$ vs.\ $96.9$ F1) and outperforming prior latent-based methods by $2.3$-$4.1$ F1 points on every backbone. Per-instance trajectories provide machine-readable audit records that reveal \emph{when} and \emph{how} harmfulness emerges, offering an interpretability advantage over single-layer approaches. 

---
# From Token Probabilities to Semantic Constraints: Towards Declarative Probabilistic Evaluation of Language Models 

**Authors**: Kyle Richardson, Cullen Anderson, Pranav Balakrishnan, Takuto Ban, Daksha Ladia, Ankita Gupta, Marisa Hudspeth  

**Link**: [PDF](https://arxiv.org/pdf/2609.13520)  

**Abstract**: While Large Language Models have improved rapidly, many fundamental questions remain about how to evaluate the knowledge and reasoning abilities they acquire, and how such evaluations relate to the learning signals used in pre-training. In this paper, we propose ModelLog, a declarative probabilistic framework for pre-training evaluation that makes the semantic structure of model behavior explicit and provides new formal tools for relating evaluation to learning. ModelLog specifies evaluation targets as symbolic constraints over token-level predictions and measures how strongly a model's distribution satisfies those constraints. We explore the framework through a new suite of tasks targeting negation, mutual exclusivity, and consistency, finding systematic failures that are difficult to characterize through token likelihood or answer accuracy alone. We further show that these evaluation scores can also be interpreted as losses, whose gradients reflect logical strength, informativeness, and variable-level sensitivity. This links evaluation and learning through a shared semantics, suggesting evaluation methods that diagnose model behavior while also helping to clarify the semantic structure of learning. 

---
# A Hybrid Hierarchical 1D-CNN-BiLSTM Framework for Extractive Summarization of Biomedical and Clinical Text 

**Authors**: Saad Bin Ather, Muhammad Saif, Ali Hassan Khan, Manzer Abbas, Hajra Waheed  

**Link**: [PDF](https://arxiv.org/pdf/2609.13481)  

**Abstract**: Large language models have made abstractive summarization remarkably fluent, but generated summaries can hallucinate facts, posing serious risks in biomedical and clinical domains. We address this by removing generation from the pipeline and framing summarization as extractive sentence selection. Our Hybrid Hierarchical CNN-LSTM Summarizer uses stacked multi-kernel convolutions to compose sentence-level embeddings into richer inter-sentence representations, followed by a bidirectional LSTM to model long-range dependencies across the document. A lightweight scoring head assigns per-sentence importance scores and is trained end-to-end with binary cross-entropy against oracle extractive labels. At inference, a dynamic mean-plus-standard-deviation threshold with a top-3 fallback selects sentences directly from the source and chronologically reorders them into the final summary. Since every output sentence is copied from the input, the model avoids generation-induced factual drift. On PubMed, our architecture outperforms isolated CNN and LSTM baselines, while ablations show that wider convolutional receptive fields improve sentence scoring. On MIMIC-CXR and MIMIC-IV BHC, the model performs well on unstructured narratives but defaults toward positional baselines on highly templated reports. These results suggest that structural constraints can provide a reliable path toward factually grounded summarization systems that are trustworthy by design rather than by correction. 

---
# Hindsight Bias in Clinical Temporal Reasoning: How Future Data Exposure Affects Large Language Model Judgment 

**Authors**: Misaki Matsuura, Sayantan Kumar, Ojas Kadam, Jeremy C. Weiss  

**Link**: [PDF](https://arxiv.org/pdf/2609.13454)  

**Abstract**: Clinical decisions are prospective, but clinical language models are often evaluated on retrospective records that reveal the final diagnosis, treatment response, and outcome. Such evaluations may reward the use of future information rather than reasoning under the uncertainty present at the decision point. We introduce a paired benchmark for measuring outcome-conditioned shifts consistent with hindsight bias in clinical temporal reasoning. It contains 171 case reports from the PubMed Central Open Access Subset---40 sepsis and 131 GLP-1/diabetes cases---represented as both textual narratives and human-annotated and LLM-generated textual time series (TTS). For each case, questions are tied to a clinically meaningful cutoff and paired with a prospective reference answer and an outcome-consistent \emph{hindsight trap}. Models answer each question using either a TTS truncated at the cutoff or the complete timeline; additional conditions vary the narrative source (original or synthetic) and TTS annotation source (human or LLM). We evaluate accuracy (Acc), hindsight trap rate (HTR), answer instability rate (AIR), and hindsight bias rate (HBR), each of which captures different signals of hindsight bias. Across GPT 5.6 Sol, Gemma 4, GLM 5.2, and Opus 5, full timeline exposure produces consistent hindsight-sensitive shifts, while temporal masking reduces bias without lowering accuracy. 

---
# Causal Analysis and Mitigation of Spurious Onsets in Full-Duplex Speech LLMs 

**Authors**: Kento Nishi  

**Link**: [PDF](https://arxiv.org/pdf/2609.13445)  

**Abstract**: Speech-to-speech LLMs like Moshi, and its derivative PersonaPlex, can listen and speak concurrently through full-duplex generation. However, they can begin speaking inappropriately during prolonged user silence: under digital-zero input, Moshi and PersonaPlex initiate speech in 12/40 and 11/40 five-minute continuations, respectively. What causes this spurious speech? We investigate two hypotheses: either repeated sampling selects speech despite persistently low onset probabilities, or conditioning on the model's nonspeech outputs causes an abrupt spike in onset probability. We find that, at every observed onset, speech probability spikes by over nine orders of magnitude in one 80-ms frame, supporting the latter hypothesis. Then, to suppress these onsets without blocking genuine responses, we ask a causal counterfactual question: is the model responding to user speech, or would its next-token distribution remain similar if the preceding user input were muted? Accordingly, we suppress onsets whose distributions change little under this intervention. Across 40 held-out trials per model with realistic microphone noise, our method suppresses 13/13 Moshi and 9/9 PersonaPlex spurious onsets, while preserving 40/40 genuine responses per model. Our inference-time method requires no retraining and runs in real-time, with 95th-percentile decision time below 61 ms, within the 80-ms frame budget. Our code is available at this https URL. 

---
# CVSS-X: A Multilingual Speech-to-Speech Translation Corpus for 28 Languages 

**Authors**: Lucas Rafael Stefanel Gris, Alef Iury Siqueira Ferreira, Frederico Santos de Oliveira, Augusto Seben da Rosa, Alexandre Costa Ferro Filho, Arlindo Rodrigues Galvão Filho, Anderson da Silva Soares  

**Link**: [PDF](https://arxiv.org/pdf/2609.13413)  

**Abstract**: We introduce CVSS-X, a large-scale synthetic speech-to-speech translation corpus that extends CVSS by reversing the translation direction. While CVSS translates from 21 languages into English, CVSS-X enables translation from English into 28 target languages spanning 12 language families. The corpus comprises approximately 240,000 parallel speech pairs per language, totaling over 16,000 hours, eight times larger than CVSS. We provide two variants: CVSS-X-C with two canonical voices per language, and CVSS-X-T with cross-lingual voice cloning, both fully generated. Evaluation shows comparable translation quality to CVSS with consistent performance across typologically diverse languages. Combined with CVSS, this enables research on bidirectional and multilingual speech-to-speech translation. The code is available at this https URL and the dataset under CC-BY-NC 4.0 license at this https URL. 

---
# RFCLLM: Evaluating LLMs' Reasoning Ability of Network Protocol State Machines 

**Authors**: Anqi Chen, Dan Goldwasser, Cristina Nita-Rotaru  

**Link**: [PDF](https://arxiv.org/pdf/2609.13389)  

**Abstract**: Mapping textual specifications into formal representations is essential for ensuring the correctness of protocol designs and implementations. LLM-generated mappings, used for networking security or testing, are assumed to capture a perfect understanding of the specification, which may not hold in practice. The goal of this paper is to assess the extent to which LLMs can interpret the specification correctly. We examine the degree to which an LLM's implicit representation of a finite-state transition system-defined via natural language descriptions-aligns with a manually generated ground-truth model. We designed 4 tasks and 1482 task queries for 16 protocols. We evaluated different judge biases, observed the inherent difficulty gaps between tasks, looked into the effect of 4 context types, and the influence of protocol characteristics. Our work contributes to a step toward verifying whether LLMs can really be trusted in FSM (Finite State Machine) reasoning of protocol specifications. 

---
# Clinical Reasoning Under a Partially Observed Objective in Cone Beam CT Report Generation 

**Authors**: Ajo Babu George, Govind Arun, Sidharth N Krishna, Uma Ranjan  

**Link**: [PDF](https://arxiv.org/pdf/2609.13238)  

**Abstract**: Maxillofacial report generation from cone beam computed tomography is scored here by a composite objective placing 80% of its weight on a large language model judgement of factual entailment and 20% on lexical overlap, of which only the lexical fifth is visible during development. The grader's BLEU-4 and METEOR routines are reproduced in pure Python and match the reference to machine precision, and an offline entailment surrogate, which tells a report written for one patient from one written for another at an area under the curve of 0.987, makes the composite objective cheap enough to optimise directly. Over the 622-case public release, a report selected against the visible lexical ranking scores 0.2909, whereas one selected against the composite objective scores 0.4122, because pursuing n-gram overlap drives entailment precision from 0.522 down to 0.266. A 29 million parameter encoder fine-tuned on the release reaches a prevalence-weighted out-of-fold area under the curve of 0.486 over 985 statements, indistinguishable from the corpus prior, while nine numbers read from the image header reach 0.945 for mandible coverage and 0.872 for condyle coverage, and acquisition centre alone predicts sentence choice at 0.718 against 0.663 for the image-derived model, identifying dictation convention rather than anatomy as the quantity the lexical metrics reward. The delivered system emits eight unconditional statements and five gated on header geometry under polarity, laterality and tooth-level consistency constraints, and reaches METEOR 0.3542 over 50 held-out cases from an unseen centre. The dataset and code are available at this https URL 

---
# TestHallVQA: Exploring LVLMs' Document-Level Reasoning under Redundant Contexts from Scientific Exams 

**Authors**: Yongqi Yu, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.13158)  

**Abstract**: Large Vision--Language Models (LVLMs) are increasingly expected to perform visual question answering (VQA) over planar media. However, existing planar VQA benchmarks typically emphasize isolated challenges: some emphasize long-document understanding with limited reasoning depth, while others require complex visual reasoning but remain restricted to single-page, noise-free settings. Moreover, through theoretical analysis, we identify the impact of irrelevant visual tokens, which leads to measurable performance degradation but has received little attention with respect to systematic quantification. To address these limitations, we introduce TestHallVQA, a multi-image VQA benchmark that simultaneously embodies document-level scale and the difficulty of human examinations, while providing comprehensive task coverage. Leveraging TestHallVQA's ability to controllably inject multi-level contextual redundancy, we further propose a novel metric, F1-R\textsuperscript{2}, which jointly quantifies LVLMs' computational reasoning capability and their evidence retrieval robustness against document-level redundancy. Extensive experiments and analyses on mainstream LVLMs reveal their latent deficiencies across multiple dimensions, offering concrete insights and directions for future research. The associated datasets, code, and complete theoretical derivations are available at this https URL. 

---
# Lexical Prompt Compression for Large Language Models: A Training-Free, Deterministic Pipeline with Empirical Pareto Analysis Across Eleven Task Categories 

**Authors**: Shamin Chokshi  

**Link**: [PDF](https://arxiv.org/pdf/2609.13154)  

**Abstract**: Recent advances in large language models (LLMs) have made prompts increasingly large and complex. Techniques such as chain-of-thought reasoning (Wei et al., 2022) and in-context learning (Brown et al., 2020) frequently push real-world prompts past several thousand tokens, increasing inference cost and latency. Learned compression methods such as LLMLingua (Jiang et al., 2023) and Selective Context (Li et al., 2023) achieve high compression ratios but require auxiliary language models and are non-deterministic. We ask a complementary question: how far can a training-free, fully deterministic, CPU-only pipeline based on classical lexical NLP be pushed before output quality degrades significantly? Eleven toggleable lexical transformations - stopword removal, filler-phrase deletion, contraction and abbreviation substitution, part-of-speech-based pruning, lemmatization, WordNet-driven synonym shortening, and named-entity preservation - are assembled into a configurable pipeline. Fifteen configurations are evaluated on 1,242 English-only prompts from six sources (Dolly-15k, LMSYS-Chat-1M, WildChat-1M, MMLU, GSM8K, HellaSwag), spanning eleven automatically derived task categories, yielding 18,630 paired GPT-4o-mini completions. Output preservation is measured using BLEU, ROUGE-1/2/L, BERTScore-F1, and SentenceBERT cosine similarity. The most aggressive configuration achieves a mean token reduction of 40.3% (sigma = 9.2) at a BERTScore-F1 of 0.876 against the original-prompt output; a stopword-only configuration achieves 29.6% reduction at 0.913. The compression-versus-fidelity Pareto frontier is characterized per task category, with commonsense reasoning a systematic failure mode under aggressive compression. All code, prompts, and per-cell results are released for reproducibility. 

---
# PhysMent: An Interactive Approach For LLM Reasoning In Physics Problems 

**Authors**: Joseph Chan, Utkarsh Jha, Xiyin Yang, Abhinav Jarajapu, Anik Sahai, Eddie Hu, Robin Jeshua Deepak, Stefano Saravalle, Aditya Shah  

**Link**: [PDF](https://arxiv.org/pdf/2609.13152)  

**Abstract**: Large language models (LLMs) perform strongly on static science benchmarks, yet their ability to reason about the physical world through active experimentation remains poorly understood. We introduce PhysMent, a benchmark that evaluates LLM physical reasoning via iterative, toolmediated interaction with a MuJoCo physics simulator. Unlike static benchmarks that supply all quantities upfront, PhysMent requires models to discover information by applying forces, querying object states, advancing time, and modifying scene geometry before answering. The benchmark comprises 105 scenes of classical mechanics, organized across four difficulty regimes (Easy/Hard and Single/Multi), three scene modalities (standard, object creation, hidden objects), and a scene-manipulation category, evaluated with a six-dimensional scoring framework. Results show that current models perform reasonably well on qualitative single-concept tasks (up to 80% accuracy) but degrade substantially on quantitative tasks that demand precise, multi-step experimental procedures: most models fall below 30% on the hardest single-concept category, where the bottleneck is procedural (adaptive multi-step tool use) rather than conceptual load. Across the seven models, accuracy ranges from 25% to 67%, with failures due to premature answer submission, inefficient exploration, and inconsistent grounding in simulator feedback rather than conceptual gaps. 

---
# Token Merging for Multilingual Speech Recognition: A Systematic Study Across Model Scale and Fine-Tuning 

**Authors**: Dylan Luke Holyoak  

**Link**: [PDF](https://arxiv.org/pdf/2609.13151)  

**Abstract**: Leading multilingual speech recognition models like Whisper transcribe diverse, low-resource languages without language-specific training but are computationally expensive to deploy. Token merging mitigates this inefficiency by dynamically combining redundant features, shortening the sequence length during inference without requiring retraining. In this paper, we systematically evaluate token merging on the Whisper model family across sixteen diverse languages and three different model sizes. We also test how token merging interacts with fine-tuning (DoRA) on low-resource languages. Our findings show that merging tokens increases computational efficiency with almost no loss in transcription accuracy across most low-resource languages and model sizes, and it works even after the model has been fine-tuned. Our results demonstrate that token merging is a highly practical method for making multilingual speech recognition faster and cheaper to deploy. 

---
# Bellman Policy Optimization 

**Authors**: Zhuoqing Song, Haotian Xu, Xikun Zhang, Lidong Bing  

**Link**: [PDF](https://arxiv.org/pdf/2609.15987)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) improves the reasoning capabilities of large language models (LLMs). We introduce Bellman Policy Optimization (BPO), a critic-free method derived from Policy Mirror Descent (PMD). For autoregressive generation with terminal rewards, BPO uses the Bellman equations to reformulate PMD as a trajectory-level objective. The reformulation avoids estimating state values at intermediate states. We prove that it has the same unique optimal solution as the original PMD objective. We derive the practical BPO loss by approximating this objective. Its mismatch-correction weight is a smoothed ratio of complementary token probabilities. Experiments on mathematical reasoning benchmarks demonstrate the effectiveness of BPO. 

---
# Stellar Colosseum: A Many-Agent Harness for Long-Horizon Research in Mathematics and Theoretical Computer Science 

**Authors**: Honghao Lin, David P. Woodruff, Yuan Deng, Jieming Mao, Song Zuo, Vahab Mirrokni  

**Link**: [PDF](https://arxiv.org/pdf/2609.15983)  

**Abstract**: Language models can produce plausible short proofs, but may still be unreliable on long-horizon research problems, where progress depends on a sequence of uncertain and interdependent decisions. We introduce Stellar Colosseum, a model-agnostic harness for allocating inference across research in mathematics and theoretical computer science. Colosseum explores alternative strategies before proof construction, uses a readiness gate to decide when a route is mature enough to decompose, represents the proof plan as interdependent section-level subproblems, and routes verifier findings back to the affected part of the argument. Across these stages, it generates candidates in parallel, attacks them with targeted falsification, and combines candidates and their critiques into a single research artifact through overlapping random-sample tree aggregation. The Colosseum workflow has also been integrated into Google Antigravity's Teamwork framework as the Long Proof pattern.
We demonstrate the capabilities of Colosseum through open-ended research and evaluations on theorem-proving and competitive programming benchmarks. Using Colosseum with Gemini 3.1 Pro, we obtain several new results that address open problems arising from papers published at top venues such as FOCS and JMLR. On TCS-Bench, a benchmark of research-level theorem-proving tasks drawn from papers published at FOCS, STOC, and SODA, Colosseum achieves 71.0% accuracy using Gemini 3.1 Pro and Gemini 3.7 Flash. In a separate Codeforces evaluation using Gemini 3.1 Pro, the proof-oriented pipeline with execution feedback solves 218 of 222 problems. 

---
# The Router Within: Eliciting Native Skill Routing from a Frozen LLM 

**Authors**: Ruishuo Chen, Xun Wang, Yu Chen, Zhuoran Li, Longbo Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.15982)  

**Abstract**: Skills extend an LLM agent beyond its parametric knowledge, and the gain they promise rests on picking the right one. Deployed harnesses route by preloading every skill's metadata into the context, which disperses the agent's attention and caps the library size. Retrieval pipelines move the selection out of the context, but also out of the agent's capability. We show that the frozen agent LLM already carries the routing signal in its own forward passes, and that two linear maps suffice to read it out with no skill text in the context. Gavel (Glance And Verdict from a frozen LLM) reads it in two steps. A glance projects the task's and each skill's mid-layer states through the two maps, the only parameters trained, and scores the full library against compact per-skill banks that one forward pass builds at installation. A verdict then resumes the shortlisted skills' forward passes and reads the model's own likelihood and yes/no judgment, fused with the glance as a product of experts. Trained once, Gavel transfers zero-shot to three public benchmarks and SkillTraj, our new benchmark of 372 simulated agent trajectories. On Qwen3-32B it outperforms progressive disclosure and retrieve-and-rerank pipelines that add 1.2B to 16B external parameters, by up to 13.4 points on written tasks and up to 21.9 when the need for a skill arises mid-rollout. Routing accuracy improves as the backbone does, and in a bash-agent harness the same 32B triggers the correct skill on Skill-Use more often than far larger frontier models running in Codex. 

---
# EvoOntology: A Self-Evolving Ontology Layer for Data Agents 

**Authors**: Meiduo Chong, Shaolei Zhang, Ju Fan, Xiaoyong Du  

**Link**: [PDF](https://arxiv.org/pdf/2609.15779)  

**Abstract**: Data agents aim to fulfill natural-language instructions over heterogeneous data, including tables, files, and databases. However, data agents face a challenging agent-data gap: heterogeneous data resides outside the agent, while the agent can access it (e.g., column names and file paths) only through generic tools. Existing approaches either let agents directly explore raw data sources or inject manually constructed semantic layers into prompts. However, neither scales well to large heterogeneous data sources nor adapts to different agent behaviors. In this paper, we introduce EvoOntology, a self-evolving ontology layer for data agents. EvoOntology encapsulates the ontology as an MCP server comprising a schema layer, a content layer, and a tool layer, enabling agents to actively query and interact with the ontology at runtime. To this end, we introduce a builder agent for autonomous ontology construction and a self-evolution loop that continuously refines the ontology through attribution-guided typed edits that are accepted only after a backbone-conditional paired evaluation. Experiments on three well-adopted data-agent benchmarks with four LLM backbones demonstrate that EvoOntology consistently outperforms strong baselines and existing semantic-layer approaches, effectively bridging the agent-data gap and enabling more effective interaction with heterogeneous data. Code: this https URL 

---
# Data storytelling meets interpretable machine learning: Decoding AI decisions for non-experts without revealing sensitive data and model details 

**Authors**: Lemen Chao, Zixuan Yang, Anran Fang, Mingran Sun, Ming Lei  

**Link**: [PDF](https://arxiv.org/pdf/2609.15722)  

**Abstract**: AI-driven automated decision-making requires both predictive performance and interpretability. Recent advances in interpretable machine learning (IML) provide tools for explaining model predictions, but the technical complexity of these explanations may hinder accessibility to non-experts. To address this challenge, this study integrates data storytelling with IML to enhance the explainability of AI-generated decisions for a broader audience. Following the design science research (DSR) paradigm, this study proposes a formal definition of data storytelling in IML, introduces the DIST Pyramid to align data storytelling with IML, and presents the I-P-O Model to describe their interactions. It further develops an architecture to explain AI decisions through distinct "What-if" and "Why-not" event-generation processes. The architecture also employs data desensitization to protect sensitive input data. To validate the approach, a case study is conducted with the Boston Housing dataset, using SHapley Additive exPlanations (SHAP) values and large language models (LLMs) to generate data stories with And-But-Therefore (ABT) structures. An empirical evaluation shows that 76.4% and 74.3% of respondents rated the "What-if" and "Why-not" data stories as more comprehensible, with significantly higher accessibility scores than traditional SHAP visualizations. The paper concludes with the presentation of a narrative interpretation framework that integrates IML and data storytelling, thereby expanding the research scope as well as the practical applicability of AI decision-making. 

---
# CiteShade: Citation Laundering in Multi-Source Retrieval-Augmented Generation and Its Counterfactual Defense 

**Authors**: Guo Fuzheng  

**Link**: [PDF](https://arxiv.org/pdf/2609.15660)  

**Abstract**: Retrieval-augmented generation (RAG) grounds a language model's answers on retrieved external knowledge and returns each answer with citations that identify its sources. Those citations are the user's audit trail: they let a reader verify a claim without trusting the model. Prior security work on RAG asks whether an attacker can corrupt the answer, leaving the citation channel unexplored. We show that this channel is a new and practical attack surface. We propose CiteShade, the first citation laundering attack to RAG, in which an attacker controlling a single source induces a model to produce an attacker-chosen wrong answer and to attribute it to a trusted source that does not support it, while the evidence for the correct answer remains in context. We formulate the attack as an optimization problem, derive three necessary conditions (retrieval, generation, and citation) and construct sources satisfying them without any instruction. On multi-source multi-hop question answering the attack raises the wrong-answer rate from 0.01 to 0.68, and source deletion confirms the malicious source is the causal driver in every measured case. Vulnerability tracks a model's propensity to cite rather than its scale, reaching CLR 0.84 under explicit instruction and 0.64 with no instruction at all on the most citation-prone model tested. We then show that perplexity filtering and citation-support checking are each insufficient, and propose a counterfactual defense that verifies which source actually drove the answer. 

---
# Human-Grounded Calibration for Long-Text Image-Text Congruence in Vision-Language Models 

**Authors**: Alessandro Gambetti, Qiwei Han  

**Link**: [PDF](https://arxiv.org/pdf/2609.15640)  

**Abstract**: Long-text image--text congruence scoring is increasingly important for vision-language systems that must evaluate whether detailed textual descriptions match visual content. However, raw similarity scores from dual-encoder models are difficult to interpret as calibrated congruence measures, especially under the modality gap between image and text embeddings. This paper proposes Congruency Score (CS), a lightweight calibration layer that maps image--text similarity evidence into a bounded score. Using DOCCI and Urban1k, we evaluate four frozen vision-language backbones and show that observed reductions in post-projection centroid distance do not uniformly improve image--text retrieval performance. Human-grounded evaluations on DOCCI further reveal a trade-off: direct post-hoc calibration preserves high association with human judgments, whereas selected projection-based configurations can reduce threshold-relevant slope and intercept distortions at the cost of retrieval performance and association strength. These results establish long-text image--text congruence scoring as a calibrated score-estimation problem, where retrieval performance, human association, and threshold calibration must be evaluated as distinct objectives. CS provides a lightweight way to expose and operationalize this separation. 

---
# IROH: Insightful Ranking Of Humor using Multi-Stage Hybrid Retrieval with Rationale-Distilled LLM Judges for JOKER 2026 Track Task 1 English 

**Authors**: Ana-Maria Luisa Mocanu, Sebastian Mocanu, Ciprian-Octavian Truică, Elena-Simona Apostol  

**Link**: [PDF](https://arxiv.org/pdf/2609.15618)  

**Abstract**: Our team, VANGUARD, presents IROH (Insightful Ranking of Humor), a three-stage retrieval system for JOKER Task 1 English at CLEF 2026, achieving first place on the leaderboard with 0.6347 MAP. Our pipeline combines hybrid sparse-dense retrieval, cross-encoder reranking, and a LoRA-adapted Large Language Model judge ensemble. We employ Gemma 4 to generate query-aware rationales under two prompt strategies, generic and typed, and produce up to four types of structured hard negatives for training data construction. Through an ablation across three cross-encoder architectures, four dense embedders, and eight judge configurations, our key findings are threefold: (1) the rationale-distilled judge is the primary driver of ranking quality, whereas appending rationales to the first-stage index contributes negligibly; (2) structured hard negatives degrade generalisation in nearly all configurations despite inflating local validation scores; and (3) across the components we ablate, the lighter, better-calibrated model is competitive with or stronger than its larger counterpart, with the generic-rationale Qwen2.5-7B judge (0.6055 MAP) outperforming every Gemma-4-31B configuration, and the advantage of generic over typed rationales is concentrated almost entirely in the smaller model. 

---
# Option-Aware Retrieval and Task-Specific VLM Adaptation for Medical VQA 

**Authors**: Tristan Kirscher, Niklas C. Koser, Soren Pirk  

**Link**: [PDF](https://arxiv.org/pdf/2609.15530)  

**Abstract**: We describe our submission to the MedReason 2026 challenge, covering multiple-choice (MCQ) and open-ended (OE) medical visual question answering (VQA) under fully offline, containerized inference. Our first finding is that MCQ retrieval must compare answer \emph{semantics} rather than answer labels: labels are independently assigned per question, so copying a retrieved neighbor's label transfers no useful information, whereas scoring each current option's text against correct-answer text from similar training cases raises retrieval-only accuracy from 20.0\% to 57.5\% on a 200-case retrieval-excluded development holdout. Our second finding attributes the submitted system's accuracy: holding the task-specific MCQ Low-Rank Adaptation (LoRA) adapter fixed and varying the number \(k\) of in-prompt retrieved examples changes accuracy by at most one case --- 187/200 (93.5\%) at both \(k=0\) and the adapter's training-time \(k=1\), 188/200 (94.0\%) at the packaged runtime's default \(k=3\) --- and the submitted confidence-gated override adds no net accuracy on top of \(k=3\), selecting the VLM in 198/200 cases. With the final MCQ adapter fixed, retrieval changes accuracy by at most one case, and gating provides no net gain. On 20 OE cases, token-F1 and RaTEScore~\cite{zhao2024ratescore} decrease as \(k\) grows, but paired sign tests on token-F1 differences are nonsignificant (\(p \ge 0.29\)); a single-annotator comparison found 6/20 wrong-anchor errors for the final configuration and 14/20 for an earlier configuration that jointly differed in routing, adapter, and prompting. The system reaches 94.0\% MCQ accuracy on the development holdout and 93.20\% on the organizer's official pre-evaluation, versus 29.43\% for the off-the-shelf reference baseline, while both of the organizer's open-ended scores are lower than that baseline's (ground-truth agreement 1.245 versus 1.588, visual accuracy 1.995 versus 2.696, each out of 4). 

---
# Beyond Safe Answers: Segment-Aware Listwise Alignment for Reasoning Safety in Large Reasoning Models 

**Authors**: JungMin Yun, Junehyoung Kwon, Hayeong Ryu, Byeonggeuk Lim, Hoejoon Kwon, YoungBin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2609.15517)  

**Abstract**: Large Reasoning Models (LRMs) pose a dual-surface safety challenge: both intermediate reasoning traces and final answers can contain harmful content. Existing alignment methods often operate at the whole-response level, allowing unsafe reasoning to be masked by a safe-looking final answer. We propose Segment-aware Listwise Target DPO (SaLT-DPO), which addresses this gap through three mechanisms: (1) segment-aware listwise alignment that decomposes responses into reasoning and answer segments, independently scores each segment's safety, and aligns length-normalized segment rewards with soft target distributions over multiple candidates; (2) joint safety coherence regularization that applies a weakest-link principle to promote safety consistency across both segments; and (3) utility anchoring on benign prompts to mitigate over-refusal and reasoning degradation. Experiments on three LRMs show that SaLT-DPO consistently reduces unsafe rates for both reasoning and answer segments while mitigating degradation in benign compliance and preserving general reasoning performance. Ablation studies demonstrate the complementary contributions of its components. 

---
# MarKey: Marginal Utility Guided Greedy Keyframe Selection for Long Video Understanding 

**Authors**: Hongchang Shi, Jinpeng Hu, Ao Wang, Wenzheng Zhou, Hui Ma, Feng Li, Zenglin Shi  

**Link**: [PDF](https://arxiv.org/pdf/2609.15408)  

**Abstract**: Long-video understanding remains challenging for multimodal large language models (MLLMs) because densely encoding long frame sequences is computationally expensive, while uniform sampling under a limited visual budget can miss sparse yet decisive evidence. Recent training-free keyframe selection methods have enabled more efficient inference and yielded promising performance gains. However, many existing methods score frames largely in isolation without explicitly considering how each candidate complements the currently selected subset, potentially resulting in redundant selections and incomplete evidence coverage. To address this limitation, we propose MarKey, a training-free framework that formulates keyframe selection as subset-aware greedy optimization. At each iteration, MarKey scores each candidate using a tractable surrogate that jointly accounts for query relevance, marginal coverage gain, and context-dependent redundancy, and selects the frame with the highest utility. To make this iterative subset-aware evaluation efficient, MarKey uses a compact set of representative anchors to approximate full-video coverage and a bounded window of previously selected frames to limit context-dependent comparisons. Experiments on six benchmarks spanning holistic video understanding, human-centric video understanding, and open-ended video understanding demonstrate that MarKey consistently outperforms existing methods. Further analyses show robust gains across different MLLM backbones, model scales, and frame budgets. 

---
# RSIAgent: Autonomous Exploration for Recursive Self-improvement in New Environments 

**Authors**: Sibo Zhu, Shicheng Fan, Xinyue Wang, Wenyi Wu, Kun Zhou, Biwei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.15364)  

**Abstract**: Digital agents must often adapt to new environments whose interfaces, tools, and failure modes are not fully captured by pretrained models. We introduce \textbf{RSIAgent}, a training-free multi-agent framework for recursive self-improvement through autonomous memory construction. RSIAgent coordinates curriculum, actor, and verifier agents to continually explore the environment, validate outcomes, and retain environment-specific knowledge, including reusable causal relationships between actions, conditions, and consequences. It further adopts a \textbf{broad-then-deep} exploration strategy, combining parallel broad recursive self-exploration for discovering diverse environment structures with focused deep self-exploration for uncovering hard cases, hidden constraints, boundary conditions, and previously unknown causal dependencies. The resulting memory is frozen and can be directly reused for downstream tasks without updating model parameters. Experiments on OSWorld-v2 and Agent's Last Exam show that RSIAgent substantially improves strong open-source models, enabling Kimi-K3 and GLM-5.3 to outperform frontier closed-source models including GPT-6. 

---
# Parameter-Efficient Adaptation of Pretrained Language Models for Time-Series Forecasting 

**Authors**: Tamanna Kumavat, Georg Brunner, Kyriakos Flouris  

**Link**: [PDF](https://arxiv.org/pdf/2609.15344)  

**Abstract**: We study the adaptation of pretrained language models to univariate time-series forecasting through a parameter-efficient transfer learning framework, with the goal of understanding which design choices drive effective cross-modal transfer. While language models operate on discrete textual tokens, time series consist of continuous numerical observations with temporal dependencies. To bridge this modality gap, we project fixed-length time-series patches directly into the embedding space of a pretrained GPT-2 backbone, bypassing textual tokenization and treating the Transformer as a generic sequence encoder. Through controlled ablation studies on seven benchmark datasets spanning energy, weather, traffic, and finance, we analyze the effects of (i)~representation strategy (continuous embeddings versus textual serialisation), (ii)~adaptation regime (frozen backbone versus partial or full fine-tuning), (iii)~architectural components such as adapters, pooling strategies, and prediction heads, and (iv)~input context length. Continuous patch-based embeddings consistently outperform textual prompting and randomly initialised backbones. The adapted pipeline attains MASE within the range of specialised forecasting architectures while updating less than 1\% of total model parameters. Results further indicate that freezing the pretrained backbone and training lightweight projection and adapter modules provides a favourable accuracy--efficiency trade-off with stable behaviour across varying context lengths. 

---
# Clean Scores, Buried Evidence, and Confident Wrong: A Receipt-Based Audit of Frontier Agentic QA 

**Authors**: Luis M. Sánchez  

**Link**: [PDF](https://arxiv.org/pdf/2609.15319)  

**Abstract**: Frontier models score well on shallow document/chart reading tasks. In a controlled data-room audit, moving evidence into buried conditions reduced accuracy, increased forced declarations, increased tool calls, and increased cost per correct answer. Confidence and benchmark calibration did not fully capture wrong answers; a documented production incident shows fabricated structural claims can be mixed with accurate numeric tables. Agentic evaluations need claim-level receipts (statement-level provenance, not answer-level scores), condition-aware scoring, and human-adversarial verification - an auditing discipline, not a leaderboard. The setting we measure is financial due diligence; the setting we are building toward next is defense staff work, where the same buried-evidence shape appears. In both, the model is not a party to the consequences; the person who signs is. In plain terms: in the documented cases we examine, agents can pair accurate numbers with confident fabricated explanations, and the burden of proof must therefore move from the model to the evidence trail. 

---
# Reducing the Output-Mode Gap in Speech Language Models via Joint-Output On-Policy Distillation 

**Authors**: Daxin Tan, Dehua Tao, Chengxi Deng, Hanlin Zhang, Xiao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.15313)  

**Abstract**: Autoregressive generation of interleaved text and acoustic tokens is a common approach to spoken-response generation in speech large language models. Although this design enables streaming generation with explicit textual guidance, generated acoustic tokens become part of the context for subsequent text predictions. Given identical speech inputs, we observe markedly lower answer accuracy for the internal text generated in speech-to-text-and-speech (S2TS) mode than for speech-to-text (S2T) responses. We term this discrepancy the \emph{output-mode gap} (OMG). To reduce OMG, we propose \emph{Joint-Output On-Policy Distillation} (JO-OPD), which distills the model's stronger S2T policy into joint generation using student-generated S2TS trajectories. At each text position, the S2T teacher provides soft targets from a text-only projection of the student's preceding outputs, while the student predicts from the corresponding full interleaved history. A preservation objective further regularizes native non-text predictions. Experiments on Step-Audio-2-mini and Baichuan-Audio-Instruct reveal OMG across two interleaved generation architectures. On Step-Audio-2-mini, JO-OPD reduces OMG from 42.87 to 16.26 percentage points on Spoken-MQA and from 29.72 to 13.04 points on speech-rendered GSM8K, with little change in S2T accuracy and substantially larger reductions than matched SFT baselines. ASR-based evaluation further shows a 7.49-point improvement in spoken-answer accuracy on Spoken-MQA. 

---
# Reason What Matters: Retrieval-Grounded Reasoning for Universal Multimodal Embeddings 

**Authors**: Mingzhou Jiang, Peixi Wu, Hang Cheng, Yunhao Zhou, Biao Yang, Wei Yuan, Yun Li, Fan Yang, Wenwu Ou, Honghui He  

**Link**: [PDF](https://arxiv.org/pdf/2609.15296)  

**Abstract**: Universal multimodal embedding (UME) learns unified representations across modalities, enabling a single model to support diverse retrieval tasks. Recent methods use Chain-of-Thought (CoT) reasoning to better interpret multimodal inputs before generating embeddings for complex retrieval tasks and further optimize this reasoning process through GRPO with retrieval-based rewards. However, two limitations hinder corpus-scale deployment. GRPO assigns all CoT tokens the same advantage, without identifying input-supported claims or evidence that distinguishes the positive from negatives. Moreover, generating a complete CoT before each embedding introduces substantial latency, even when a partial trace already provides sufficient retrieval evidence. To address these limitations, we propose Reason What Matters (ReWAM), a retrieval-grounded reasoning framework that uses retrieval feedback to guide both credit assignment and reasoning computation. Specifically, we introduce Retrieval-aware Self-Distillation (RASD), which constructs privileged guidance from input-supported evidence that distinguishes the positive item from retrieved hard negatives. An on-policy self-teacher uses this guidance to refine trajectory-level feedback into token-specific supervision for retrieval-relevant reasoning. We further develop Retrieval-adaptive Inference (RAI), which uses a retrieval confidence head to estimate the remaining retrieval utility of a partial CoT. It stops unproductive traces early and accelerates useful continuations with speculative decoding. Extensive experiments on MMEB-V2 and MRMR demonstrate that ReWAM achieves state-of-the-art retrieval performance while delivering up to 5x the inference throughput of competitive explicit-CoT UME methods. These results bridge the gap between retrieval quality and inference efficiency, making reasoning-enhanced UME practical for large-scale deployment. 

---
# PACE: Progressive Angular-to-Norm Contrastive Embedding 

**Authors**: Yanping Li, Wei Zhou, Yawen Liu, Yibo Wang, Ke Zhu, Guangda Huzhang, Qing-Guo Chen, Zhao Xu, Jun Zhang, Wei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2609.15152)  

**Abstract**: Multimodal embedding models encode heterogeneous inputs into a shared embedding space, enabling efficient similarity computation across modalities and tasks. Most existing methods optimize cosine-based contrastive objectives, which promote stable training but restrict semantic compatibility to angular geometry, precluding embedding norms from serving as an additional semantic signal. However, directly optimizing the more expressive dot-product similarity, which leverages both angular and norm information, underperforms cosine-based training and exhibits unstable training dynamics. We attribute this discrepancy to premature optimization-space expansion, manifested as angular--norm entanglement and directional anisotropy in the representation space and further compounded by full-parameter fine-tuning. In this paper, we propose PACE, a two-stage framework that progressively expands both the representation and trainable parameter spaces. Stage I combines cosine-based objective with low-rank adaptation to establish a reliable angular geometry within constrained optimization spaces. Stage II switches to dot-product similarity and full-parameter fine-tuning, enabling embedding directions and norms to jointly encode semantic information. We further introduce Focal Embedding Loss, a confidence-adaptive objective that downweights queries with high positive retrieval confidence while emphasizing ambiguous queries with competitive negatives. Experiments across multiple backbone scales and diverse multimodal embedding tasks consistently validate the effectiveness of PACE. 

---
# OpenAl4S: Code as Action, Science as Sessions 

**Authors**: Gongbo Zhang, Hao Li, Yu Wang, Mujie Lin, Liuzhenghao Lv, Yicheng Mao, Yimi Wang, Jun Zhu, Minhan Tang, Zhengxiang Jiang, Yusong Wang, Jiayu Yao, Kunpeng Ning, Dawei Pang, Yonghong Tian, OpenAI4S Community, Yuyang Liu, Li Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2609.15096)  

**Abstract**: AI co-scientists could accelerate computational research, but over a long-running study the workflow also has to stay inspectable, resumable and reproducible, which requires persistent computational state and provenance. Here we present OpenAI4S, an open-source scientific research agent built around the principle of \emph{Code as Action, Science as Sessions}. OpenAI4S combines a persistent computing runtime with research-session management: orchestration is handled through structured tool calls, while scientific actions are represented as complete code cells executed in persistent Python and R kernels. An append-only Action Ledger, per-cell execution records, versioned artifacts, environment records, and workspace checkpoints preserve how results were produced and support session recovery, branching, and extension. Configurable sandboxing, permission controls, and code and trajectory screening provide complementary safeguards. We evaluate OpenAI4S on 36 research scenarios spanning retrosynthesis, molecular dynamics, protein binder design, protein mutation, catalyst screening, and mineral spectroscopy, measuring scientific task accuracy, workflow completeness, and reproducibility of the resulting repositories. OpenAI4S achieves an overall score of 7.83, compared with 5.7--6.4 for a general-purpose coding harness evaluated with three frontier models, with the largest gains on long-horizon and computation-intensive workflows. These results suggest that integrating persistent execution with session-level provenance can improve the reliability of AI-assisted scientific workflows. Environment specification and full rerunnability remain weak for every evaluated system, ours included, so reproducibility is still an open problem for scientific agents. The system is available under the MIT license at \href{this https URL}{this http URL}. 

---
# Not All Prompts Are Equal: Exploration-Guided Prompt Scaffolding for Multimodal Reinforcement Post-Training 

**Authors**: Yuanhao Yue, Qianli Ma, Chengyu Wang, Haoting Wang, Lei Shen, Jun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.15051)  

**Abstract**: Training prompts in online reinforcement learning (RL) differ substantially in how informative they are for the current policy: some are already saturated while others are too difficult to yield reliable learning signals, yet both receive equal rollout budget under standard training. We propose an exploration-guided prompt scaffolding framework that adapts the training prompt distribution dynamically throughout RL post-training of multimodal large language models (MLLMs). Central to our approach is the $\textit{Exploration Potential Score} (EPS)$, a lightweight rollout-based proxy for prompt utility derived from KL-regularized policy improvement theory, computable directly from on-policy rollout statistics without additional overhead. Rather than discarding low-utility prompts, we use a teacher model to generate scaffolded rewrites that preserve the original task intent while making subsequent training more informative, reframing teacher supervision as training-data refinement rather than output imitation. Integrated with GRPO on Geo3K and MMK12, our method consistently outperforms the baseline on both in-domain and out-of-distribution benchmarks, achieving up to 9.7\% relative improvement in-domain and gains of 11.5\% on MathVision and 11.1\% on MMMU-Pro. 

---
# The average-farmer illusion in language-model simulations of agricultural decisions 

**Authors**: Zhanliang Zhu, Ziwei Li, Yuchen Liu, Liujun Zhu, Ruiqi Wu, Tongqing Shen, Junliang Jin, Jianyun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.15038)  

**Abstract**: Language-model agents are increasingly used as synthetic people in surveys and social simulations, yet their apparent realism is often judged from population averages or distributional similarity. We tested what such evidence actually establishes by comparing Claude, Codex and Kimi under four prespecified prompt designs with matched farmer decisions from China and four African countries. Some configurations reproduced observed means and adoption rates. However, their person-level predictions were weak; their decisions clustered around typical values and policy-relevant extremes were largely missing. Most strikingly, a simple generator fitted only to the observed marginal dis- tribution, and given no information about any farmer, achieved greater distributional similarity than every language-model configuration. Prompt additions produced conditional gains rather than uni- versal improvement: results varied with model, outcome, population and validation target. We call this the average-farmer illusion: a synthetic population can look realistic while failing to repro- duce who does what or how behaviour varies. We provide a claim-matched validation framework and reusable modular prompts that turn prompt construction into an auditable experimental process. Population-level resemblance should therefore be treated as the start of validation, not as evidence of individual simulation. 

---
# MoARa: Module-Aware Rank Allocation and Structure-Preserving Decomposition for Low-Rank LLM Pre-training 

**Authors**: Keunyoung Kim, Nojun Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2609.15037)  

**Abstract**: Low-rank gradient projection reduces the optimizer-state memory cost of large language model (LLM) pretraining, but the steps and wall-clock time needed to reach a target quality remain a meaningful axis for improvement. We attribute this to two design choices in existing methods: the projection-rank budget is allocated uniformly across Transformer modules with heterogeneous projection sensitivity, and projecting a raw gradient attenuates its magnitude and direction jointly. We propose MoARa, which combines a static profiling-based module-aware projection-rank allocation with a block-wise magnitude-direction decomposition; the default block size is set in the neighborhood of the attention head dimension. Across five Transformer architectures spanning Llama, Qwen, and DeepSeek at 300M to 7B scales, GaLore with MoARa reaches standard GaLore's final perplexity in 37% fewer steps and 34% less wall-clock time on Llama 2 7B, with only 0.2% peak reserved memory overhead under standard graph compilation. Across the six low-rank pretraining methods we evaluate, module-aware rank allocation alone delivers directionally consistent step reductions on all six. On compatible hosts, the two-component design reaches up to 41.7% step reduction and 37.1% wall-clock reduction. 

---
# Pick Your Poison: Learning to Select Poison Sets for Stronger LLM Backdoor Attacks 

**Authors**: Aashiq Muhamed, Mona T. Diab, Virginia Smith, Andrew Ilyas, Matthew Jagielski  

**Link**: [PDF](https://arxiv.org/pdf/2609.15029)  

**Abstract**: Backdoor poisoning attacks add poisoned examples to otherwise-clean finetuning data, pairing a trigger with a target behavior that the model learns to produce when the trigger appears. Existing evaluations typically fix the number of poisoned examples and sample them at random from a candidate pool. We show that this can severely underestimate worst-case vulnerability: across three LLaMA-3-8B backdoor settings, holding the model, clean data, and poison count fixed, attack success ranges from 3% to 80% depending only on which poison set is chosen.
We formalize poison selection as oracle-budgeted set optimization and introduce SAILS (Set-level Audit-Informed Iterative Learned Selection), which learns a set scorer from a few hundred finetune-and-evaluate runs, ranks millions of candidate sets, and audits only a small shortlist. SAILS improves held-out attack success by 30 percentage points on average over the strongest influence baselines, transfers from small-scale to full-scale finetuning, and extends to code-generation, agentic, and API-only backdoors. 

---
# Converting Sequenced Fuzzy Cognitive Maps to Causal Virtual Worlds with Large Video Generators 

**Authors**: Akash Kumar Panda, Olaoluwa Adigun, Bart Kosko  

**Link**: [PDF](https://arxiv.org/pdf/2609.14985)  

**Abstract**: We show how users can create and manipulate causal virtual worlds with large-language-model (LLM) and large-video-model agents. The approach uses feedback fuzzy cognitive maps (FCMs) both to model the granular causal structure of the virtual world and to guide its causal evolution. The local causal rules are partial or fuzzy while the FCM's feedback structure produces global equilibria that define causal scenarios. A sequence of \emph{dynamical} meta-rules of the form ``If $\mathcal{A}$ then $\mathcal{B}$" define the causal scenes of the virtual-world video. The if-part causal pattern $\mathcal{A}$ perturbs the FCM's virtual world at the user's or agent's discretion. The FCM's transient feedback dynamics define the meta-rule's causal arrow of implication. The then-part $\mathcal{B}$ is the resulting equilibrium attractor such as a FCM limit cycle or fixed point. Our algorithm extracts these meta-rules from the FCM and guides the LLM agent to write a script based on the FCM meta-rule sequence. The large video generator converts the meta-rule into a video scene in accord with the flow of the dynamics. We applied the agent-based technique to a simple FCM that describes an undersea world of dolphins and sharks. Google's Gemini 3.1 generated the script and Google's Veo 3.1 generated the dolphin-shark video. The approach is general and can scale by mixing larger FCMs and AI agents to produce more immersive virtual worlds. 

---
# Geometric Signatures of Conceptual Reorganization: A Counterfactual Embedding Framework for Detecting Scientific Revolutions 

**Authors**: Dimitris Ntounis, Ariel Schwartzman, Chris Chafe, Thomas A. Ryckman  

**Link**: [PDF](https://arxiv.org/pdf/2609.14917)  

**Abstract**: We introduce document embedding geometry as a quantitative observable of conceptual reorganization and develop a counterfactual ablation framework for measuring how individual concepts influence the organization of scientific knowledge, providing a quantitative framework for detecting scientific revolutions. The observable is defined by the geometric perturbation induced when removing documents associated with a candidate concept from the embedding space before and after its historical emergence. Statistical validation is performed using five historical case studies spanning physics, mathematics, and machine learning: special relativity, Gödel's incompleteness theorems, the Higgs mechanism, deep learning, and the attention mechanism underlying transformer architectures. Across the historical case studies, the framework identifies measurable geometric signatures associated with conceptual reorganization, while the validation studies expose important limitations arising from document assignment and sparse historical data. These results establish embedding geometry as a medium for quantifying conceptual reorganization, providing a new approach for studying how scientific fields restructure over time. 

---
# PeerPen: AI-Assisted Writing for Online Mental Health Peer Support 

**Authors**: Jiwon Kim, Sherry Gong, Maya Ajit, Soorya Ram Shimgekar, Yunhao Yuan, Dong Whi Yoo, Eshwar Chandrasekharan, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2609.14886)  

**Abstract**: Online mental health communities thrive on peer support, yet those who volunteer to help often lack formal training and may struggle to articulate supportive responses. AI co-writing could lower this barrier; however, peer support derives much of its value from being perceived as personal, raising questions around authorship, ownership, and trust. We built PeerPen, a writing assistance tool embedded within a Reddit-like interface, supporting two main features: draft generation and revision of user-written responses. Through semi-structured interviews with 15 participants, we find that PeerPen reduced the burden of composing responses and increased confidence in offering support. Participants wanted AI to assist their writing without taking over authorship and anticipated tensions around authenticity and trust. Such assistance could make authorship uncertain even for responses written without it, weakening trust across the community. We contribute design implications for AI writing assistance that scaffolds supportive communication, preserves authorship, and accounts for community-level trust. 

---
# AgentKV: Phase-Aware KV Eviction for Agentic LLMs 

**Authors**: Taowen Tony Liu, Jeffrey T. H. Wong, Can Xiao, Bowen Yang, Hao Mark Chen, Yiren Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2609.14872)  

**Abstract**: Agentic serving can consume orders of magnitude more tokens than chatbot workloads, stressing both KV-cache capacity and decode-time bandwidth. Most KV-eviction methods score cached keys against representative queries drawn from the most recent tokens, assuming future attention resembles recent attention. We show that agentic generation violates this assumption: future queries form a mixture over think, act, tool, and others phases, and principal-angle analysis shows these components occupy measurably different query subspaces, so recency representatives systematically undervalue keys that upcoming phases will need. We propose AGENTKV, which maintains a small query buffer per phase and scores cached keys against their union. We further implement AGENTKV in a persistent multi-turn serving path that carries compressed KV state across turns and compacts retained KV pages online. Across two models, six task domains, and three KV budgets each, AGENTKV improves task score by 5.5 points on average over R-KV and 5.3 over Tri-attention. Relative to upstream full-KV SGLang, AGENTKV improves output-token throughput by up to 1.80x. Code: this https URL. 

---
# Self-Orchestrating Language Models: Leveraging Semantic Dependence for Efficient Inference 

**Authors**: Tian Jin  

**Link**: [PDF](https://arxiv.org/pdf/2609.14850)  

**Abstract**: Large language models (LLMs) demonstrate impressive capabilities, but their deployment presents significant efficiency challenges. Autoregressive decoding imposes substantial inference latency and under-utilizes hardware accelerators in low batch size regimes. Discrete diffusion models can generate in parallel but struggle to match autoregressive quality without many diffusion denoising steps. Long-context reasoning creates memory bottlenecks that strain even state-of-the-art accelerators.
My thesis is that language models can direct their own inference execution strategy by annotating semantic dependence -- which tokens depend on which others -- in their generation. I call such models self-orchestrating language models. For each system, I design a runtime that acts on these annotations to parallelize autoregressive decoding, evict intermediate context, or derive denoising orders, achieving Pareto-optimal quality-efficiency trade-offs.
I demonstrate this approach through three self-orchestrating systems. First, PASTA uses semantic dependence to parallelize autoregressive decoding, training the model to annotate which output chunks can generate independently. Second, TIP uses semantic dependence to evict intermediate reasoning steps from the KV cache, reducing memory consumption while preserving accuracy. Third, Planned Diffusion uses semantic dependence to derive a denoising order for discrete diffusion, autoregressively generating a plan that specifies which chunks to denoise in parallel. 

---
# LLMs as Oracles: Reliance on LLMs for Subjective Personal Questions 

**Authors**: Myra Cheng, Lujain Ibrahim, Grace Liu, Michelle S. Lam, Vishakh Padmakumar, Nick Madibekov, Diyi Yang, Dan Jurafsky  

**Link**: [PDF](https://arxiv.org/pdf/2609.14849)  

**Abstract**: We characterize how people are turning to LLMs as oracles: all-knowing authorities on subjective personal questions. Motivated by risks to users' autonomy and well-being, we develop a typology and LLM-based methods to measure this form of AI reliance at scale and understand how people are offloading judgment and decision-making to AI. Applying our typology to public usage data (68K prompts from WildChat and ThoughtTrace), we find that LLM-as-oracle use has increased over time (2023-2026) and is more prevalent among younger users. We further build a privacy-preserving data donation tool to analyze individuals' longitudinal usage data (140K prompts from 52 participants), identifying similar trends. People are often unaware of their own LLM-as-oracle use, and express dissatisfaction with this behavior after seeing our tool's analysis. Finally, we identify two drivers of LLM-as-oracle use: people's perceptions of AI and the behavior of AI models themselves, which motivate possible interventions to support users' self-deliberation. 

---
# Crypto Accounting Bench: Evaluating Frontier and Open-Weight Models on Crypto-Asset Accounting Tasks 

**Authors**: Kareem Khattab, Omar Khattab, Mohamed Ibrahem  

**Link**: [PDF](https://arxiv.org/pdf/2609.14811)  

**Abstract**: We introduce Crypto Accounting Bench (CAB), a benchmark for assessing whether frontier and open-weight language models can reconstruct the complete journal entry that an organization actually posted for a crypto-asset transaction. CAB contains 118 evaluation tasks drawn from 7 pseudonymized organizations. Each task combines transaction mechanics, asset quantities and base-currency values, wallet and legal-entity context, counterparty evidence, related transaction legs, recurrence, tax-lot evidence, and the organization's complete chart of accounts. The target is a balanced structured entry with every required account, side, amount, currency, and full-precision asset quantity. We evaluate 12 models spanning proprietary frontier systems and open-weight releases over 3 independent attempts per task, producing 4,248 trajectories. We report 3 metrics: Mean Score, Best@3, and Pass@3. Pass@3 is the fraction of tasks with at least 1 of 3 attempts that satisfies every rubric criterion and required gate. The leading model reaches 77.43% Mean Score, while the best Pass@3 is 56.78%. Deterministic diagnostics, read from each task's best of 3 attempts and macro-averaged across the 12 models, show higher base-amount agreement (97.8%) than deciding-account accuracy (56.3%). Together with the failure analysis, these results identify account selection and complete-entry composition as the main remaining challenges on CAB. 

---
# Language-Guided Representation Learning for Robust Cross-Sensor Material Recognition 

**Authors**: Mashood M. Mohsan, Muhayy Ud Din, Binzhao Xu, Ahmad Abubakar, Irfan Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2609.14783)  

**Abstract**: Robots need touch to manipulate objects safely and reliably, as many properties, such as softness, texture, and contact stability, are hard to infer from vision alone. However, vision-based tactile sensors yield different observations of the same material due to variations in optics, elastomer properties, and illumination, leading to poor generalization when trained on a single or multiple sensors. We propose a language-guided distillation framework for learning sensor-robust tactile representations. Language encodes high-level semantic properties of touch (e.g., rough, soft, slippery) that remain invariant across sensing hardware, providing a natural sensor-agnostic supervisory signal. We construct a 39K-sample touch-language dataset with human-annotated material labels and train a tactile encoder to align sensor-specific tactile images with language embeddings in a shared semantic space. We evaluate our approach for few-shot learning and cross-sensor transfer and benchmark it on six existing tactile datasets. Our method achieves 95% accuracy in the 100-shot setting, improves cross-sensor transfer by an average of 13.3% accuracy, and yields up to 19% accuracy gains across six existing tactile datasets. These results demonstrate that language-guided distillation enables scalable and hardware-agnostic tactile representation learning. Code and dataset are available at this https URL 

---
# Loop-Back Authority in LLM Agent Teams: A Paired Experiment on Flat and Hierarchical Coordination 

**Authors**: Burak Agachan, Max van Duijn, Amirhossein Zohrehvand  

**Link**: [PDF](https://arxiv.org/pdf/2609.14767)  

**Abstract**: Hierarchical orchestration, in which a Manager agent reviews worker output and can send it back for revision, is the default coordination pattern in production multi-agent LLM frameworks. Classical organizational theory predicts that the authority link speeds convergence on decisive output; work on sycophancy and Degeneration-of-Thought predicts that authoritative critique makes LLM output worse. Prior comparisons vary whole frameworks on tasks with checkable answers, leaving the authority link untested on open-ended work. We present a paired experiment that holds five LLM agents, their roles, prompts, tools, models, and data fixed and varies one link: whether the Manager may reject a worker's output and oblige a revision. Across 43 paired products and 86 runs of a business-intelligence reporting task, a five-model judge panel and a deterministic specification check score every report. The flat organization scores higher on Utility (d = 0.42, p = 0.009) and on Writing Clarity (d = 0.34, p = 0.030); the classical prediction fails. The reports are the same length, but hierarchical reports hedge 53% more, each revision loop is associated with a 0.14-point drop in Writing Clarity, and the hierarchical Writer's first draft is indistinguishable from the flat report: the gap opens inside the revision loop. Specification accuracy is at ceiling in both organizations, and the supervisory tier costs 51.5% more tokens for no quality gain. A supervisor pays for itself when it can verify and becomes a liability when it can only opine. 

---
# Refusal Reads Only a Slice of What the Model Knows: Harm-Keyed Routing and Its Exceptions Across Model Families 

**Authors**: Orion Reblitz-Richardson  

**Link**: [PDF](https://arxiv.org/pdf/2609.14759)  

**Abstract**: Alignment applied after pretraining is shallow in a measurable way: a single direction in a model's residual stream can be edited out, and the model stops refusing harmful requests. That fact says how easily refusal can be removed, not what the refusal decision was reading in the first place. We ask what it reads, and we separate that from what the model comprehends. Across four open-weight models spanning three families, moral comprehension is native to pretraining: a low-rank moral subspace crystallizes during pretraining, and alignment rotates it once without rebuilding it. The refusal gate, in contrast, is a fresh post-training construction with only a weak pretraining precursor, written into a narrow control-token channel where the refusal decision is orthogonal to the moral-judgment decision. The central result is causal and comes from one model, OLMo-3. A nested interchange rank sweep patches successively larger slices of the moral subspace between matched requests and reads how much of refusal's response transfers: as the basis widens, moral judgment keeps reading more of it, while refusal levels off at the level of a single harm direction, and about three-quarters of refusal's causal input lies outside the moral subspace altogether. Refusal reads the harm percept, not the moral content that judgment reads on the same patches. The picture is not uniform across families. Llama reads broad moral content; Qwen reads beyond the single harm cue but is unresolved at our sample size; GPT-OSS reads harm, and its refusals can be argued in either direction by its own reasoning trace. Where refusal reads only a low-rank slice and routes around the bulk of what the model knows, a rank-one edit removes it. Whether widening what refusal reads would also deepen the behavior is the open question this raises. 

---
# Fabrication After Tool Failure: Tool-Augmented Agents Assert Values Their Tools Did Not Return 

**Authors**: Arham Sethi, Arsen Kenzhebayev, Saanvi Paturi, Vatsal Raina, Vyas Raina, Ivaxi Sheth  

**Link**: [PDF](https://arxiv.org/pdf/2609.14758)  

**Abstract**: Tool-augmented language models are evaluated on whether they reach the right answer, not on whether they report honestly when a tool fails to supply one. We isolate this post-failure decision with a benchmark of 1,024 items spanning 16 internal-system domains and eight tool-failure types, in which a tool call is enforced and the returned payload is guaranteed to be unusable. Under a deployment-style system prompt, 14.10% of responses are dishonest: the model either asserts a value the payload cannot support or declines while citing a fabricated policy or capability limit. The rate is governed almost entirely by whether the failure is signalled. When the tool returns status:error, dishonesty is absent (0.0%); when it returns status:ok with a redacted, corrupted, stale, malformed, empty or truncated value, dishonesty reaches 45.3%. The behaviour is not an artefact of our prompts: it appears under a neutral prompt (10.17%) and under the shipped prompt of every production agent framework we evaluate, reaching 24.67% under CrewAI's, and none of the nine frameworks we audit specifies what the model should do when a tool fails. Comparing prompt-level defences, we find that the operative variable is not deference to tool output but the absence of a named failure state. Appending a single sentence that requires the model to emit retrieval_status: OK or FAILED before answering reduces dishonesty from 14.10% to 0.87%, with one item of 688 worsening against 92 improving, and transfers unchanged into three foreign agent scaffolds. The emitted flag is faithful in 99.7-99.9% of declarations, giving a runtime detector that needs only a regular expression. 

---
# Calibrating Interpretability Instruments Before Trusting Their Verdicts 

**Authors**: Orion Reblitz-Richardson  

**Link**: [PDF](https://arxiv.org/pdf/2609.14754)  

**Abstract**: Causal claims about large language model (LLM) internals rest on measurements. Those might include a projection, a cosine, an ablation delta, or an interchange patch among others. These measurements fail in specific, diagnosable ways that return a plausible number instead of an error, so a broken instrument can easily read as a finding. A covariance-matched null can saturate until every direction looks typical, a per-head attribution can overshoot the true residual write threefold on reordered-normalization architectures, an interchange patch can go sign-chaotic because its outcome is pinned at a ceiling, or a read-from verdict can be an artifact of measuring past the layer where the model already decided. This note documents six such failures from a causal interpretability program on refusal and moral representation, spanning several papers and a four-model open-weight panel; each mode is established on one or two of the four. For each we give the tell that catches it and a protocol keyed to a detectable trigger (reordered normalization, massive activations, a low-dimensional decision channel), so we and readers can check whether a given setup is exposed. The discipline reduces to four moves: calibrate against a positive-control ladder, certify with an orthogonal cell, compute power before spending compute, and state every read-from verdict at a depth referenced to the model's commitment. The evidence is four architectures across three families within a single program; external replication across programs is future work. 

---
# CALICO: A Human-Centered, Codebook-Aligned System for Annotation 

**Authors**: Boqin Yuan, Xiaoyi Gu, Fiona Li, Chang Wan, Angel Hsing-Chi Hwang, Jieyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2609.14726)  

**Abstract**: Large language models are increasingly used to scale codebook-based annotation in scientific research, but existing workflows provide limited support for translating domain experts' codebooks into reliable, revisable, and auditable prompts. Prompts are often treated as fixed instructions and hidden from annotators, making it difficult for non-technical domain experts to diagnose and correct model behavior when outputs violate codebook guidelines. In this paper, we present CALICO, a human-centered, codebook-aligned annotation workflow that treats prompts as editable, versioned, and optimizable artifacts. CALICO integrates codebook parsing, prompt generation, result inspection, prompt versioning, natural language human feedback, and label-supervised prompt optimization through existing optimizers such as GEPA, MIPROv2, and OPRO, together with our reflection-based optimizer, ReflectAgent. Empirically, we evaluate CALICO on domain-specific AI-companion chatbot conversation codebooks. Across evaluated dimensions, CALICO improves mean held-out performance by +13.0 and +7.4 absolute points for two coders, respectively. A coder-specificity analysis further suggests that optimized prompts capture coder-specific interpretations rather than only generic codebook clarification. CALICO runs as a web application that takes users from raw codebook materials to inspectable, exportable labels; the website, codebase, and live demo are released at this https URL under the Apache 2.0 License. 

---
# Depth and Scale in the Sub-150M Regime: JugnuLM-53M vs JugnuLM-110M 

**Authors**: Dushyant Rajput, Nirdesh Chauhan, Siddharth Kosaraju  

**Link**: [PDF](https://arxiv.org/pdf/2609.14715)  

**Abstract**: We scale our conventional sub-150M pretraining recipe from 53.5M to 109.7M parameters, holding the method fixed (Qwen3-style decoder with grouped-query attention, RoPE, SwiGLU, RMSNorm, QK-Norm, and a z-loss; FineWeb-Edu data) and changing only the geometry to a deep-and-thin 23-layer x 576-hidden design. The larger model improves across the board -- BLiMP 78.1 -> 81.3, ARC-Easy 51.4 -> 52.5, WikiText-2 byte-perplexity 2.04 -> 1.95 -- and its 81.3% BLiMP essentially matches GPT-X2-125M (81.28) at about 12% fewer parameters. Notably the 110M model achieves this on fewer training tokens (about 8B vs 12B), so the gain is attributable to capacity and depth, not more data. Both models are deliberately conventional; this report is a clean scaling control and the baseline rung (R0) of an ablation study of what further improves models in this regime. An ablation ladder follows: value residuals (R1) and the Muon optimizer (R2) lift ARC-Easy by a cumulative +3.6 (52.5 -> 56.1) at a near-flat BLiMP and are kept; a diverse data blend (R3) and two logit-distillation settings (R4a/R4b) are not kept -- honest negatives. R3 pins ARC-Easy to FineWeb-Edu's educational filtering rather than raw diversity; distillation from a 1.7B teacher can reach the class-leading ARC-Easy (56.99, matching GPT-X2-125M) but only at a perplexity cost that dialing KD down then erases -- so R2 remains the best kept stack. 

---
# Speak to the City: Multimodal Resolution for Outside-the-Vehicle References 

**Authors**: Alireza Parchami, Artin Saberpour, Robin Connor Schramm, Jürgen Steimle, Ulrich Schwanecke  

**Link**: [PDF](https://arxiv.org/pdf/2609.14691)  

**Abstract**: As autonomous vehicles and Extended Reality (XR) headsets enable novel in-car interactions, seamlessly querying physical landmarks, known as Outside-the-Vehicle Referencing (OVR), remains challenging due to ego-motion and referential ambiguity. We present a robust, multimodal OVR framework fusing user gaze and natural language to identify Points of Interest (POIs). To address the scarcity of dynamic vehicular data, we developed a VR-based pipeline synchronizing 360-degree transit videos with vehicle GNSS telemetry. Through a user study (N=46) mapping passenger head orientation into a 3D geospatial Digital Twin, we captured authentic gaze-speech behaviors. We subsequently trained a lightweight Transformer network, leveraging LLMs to dynamically align continuous spatial gaze vectors with discrete verbal context. Experimental results demonstrate high accuracy and low computational overhead, achieving an 83.33% Top-1 accuracy (87.72% Top-2) and an average inference time of 24.3 milliseconds. This real-time paradigm effectively resolves referential ambiguity, enabling context-aware spatial retrieval for passengers within the vehicle. 

---
# Exploring Multimodal Turn-Taking Cues in Face-to-Face Conversation using Voice Activity Projection 

**Authors**: Willem Berner, Julio Cesar Cavalcanti, Kalle Åström, Gabriel Skantze  

**Link**: [PDF](https://arxiv.org/pdf/2609.14666)  

**Abstract**: Turn-taking is a fundamental component of spoken interaction, and while humans naturally rely on both verbal and non-verbal signals, dialogue systems usually depend on audio cues alone. This paper investigates whether visual features from face-to-face conversations can enhance turn-taking prediction beyond what is achievable from audio-only. We extend the Voice Activity Projection (VAP) model, a self-supervised transformer-based model for predicting future voice activity, by incorporating visual features extracted from the large-scale Meta Seamless Interaction dataset of dyadic face-to-face conversations. The visual features include gaze direction, head movement, body and hand pose, and facial action units (FAU). For incorporating the visual features, we explore concatenation, cross-attention fusion, delta features, and trainable gating mechanisms. Results show that visual information improves performance over the audio-only baseline, with FAU being significantly more informative than other feature groups. Body and gaze features nevertheless contribute complementary information, as the model combining all features performs best. Furthermore, results indicate that performance on specific tasks varies depending on whether training and test data come from improvised (acted) or naturalistic (non-acted) conversations. 

---
# Know When to Stop, Where to Restart: Accelerating Multi-Turn Agentic On-Policy Distillation 

**Authors**: Zhiyu Gui, Kexin Huang, Jia Guo, Junkang Wu, Zihao Wang, Zhiqiang Zhang, Jun Zhou, Jiancan Wu, Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.14636)  

**Abstract**: On-policy distillation (OPD) has become a standard approach for transferring capabilities from large teachers to compact students. Its cost, however, is dominated by autoregressive student rollouts and scales poorly in multi-turn agentic settings. Existing acceleration methods truncate or relocate the supervision signal according to fixed, offline budgets, despite substantial variation in teacher-signal reliability both within and across trajectories. Our empirical analysis on $\tau^2$-bench reveals a clear structure in this variation: informative supervision is concentrated in the prefix of each turn, and, most importantly for multi-turn agentic training, the cross-turn loss of teacher endorsement is temporally locked to the student's first erroneous action rather than accumulating gradually over turns. Building on these findings, we propose STRIDE (Stop-and-Restart on-policy Distillation acceleration), which combines two complementary techniques: adaptive early stopping, which terminates a rollout once the cumulative teacher log-probability falls below an out-of-distribution threshold, and a prefix buffer, which caches high-quality prefixes and restarts generation at the weakest correct turn. Together, these mechanisms induce a data-driven curriculum that progressively extends coverage to later turns. On $\tau^2$-bench retail, our method matches full-trajectory OPD and exceeds the 30B teacher at a $3.73\times$ speedup, surpasses the baseline itself at $2.34\times$, and retains a $4.51\times$ speedup under cross-domain multi-teacher training. As a supplementary generalization test beyond the agentic setting, STRIDE outperforms full OPD on AIME 2025 at a $5.10\times$ speedup and on AIME 2024 at a $3.08\times$ speedup; averaged across the two evaluations, both fixed-budget truncation baselines remain below full OPD. 

---
# Parameter-Efficient Quantum NLP for Paraphrase Detection: Performance, Robustness, and Entanglement 

**Authors**: Farha Nausheen, Khandakar Ahmed, Farina Riaz  

**Link**: [PDF](https://arxiv.org/pdf/2609.14529)  

**Abstract**: Rigorous empirical validation of quantum machine learning on natural language tasks remains scarce. We evaluate a 10-qubit hybrid quantum-classical variational circuit (2,148 parameters) for paraphrase detection across three benchmarks: MRPC, Quora Question Pairs (QQP), and adversarial PAWS. On QQP (n = 10 seeds), the circuit achieves 75.53% +/- 0.75%. accuracy, statistically outperforming parameter-matched classical baselines (DeepMLP: p = 0.015, Cohen's d = 1.20; F1: p < 0.001, d = 2.32) and surpassing DistilBERT-4bit with 31,191 fewer parameters. On MRPC the optimal 2-layer variant reaches 92% of BERT-base accuracy at 54,000 lower parameter cost. Circuit depth analysis reveals a dataset-depth scaling effect; entanglement analysis via the Meyer-Wallach measure identifies multi-qubit entanglement as the primary performance driver (r = 0.85 across four variants). Adversarial evaluation on PAWS reveals emergent robustness: 98.2% recall versus 81.6% classical (+16.6 pp, d = 1.24, p < 0.001), without adversarial training. These results constitute the first systematic parameter-matched multi-benchmark empirical validation of hybrid variational circuits for NLP. All results are from classical simulation; hardware validation is future work. 

---
# Question's Gambit: The First Move Matters in Agentic Deep Search 

**Authors**: Radin Hamidi Rad, Amin Bigdeli, Negar Arabzadeh, Sajad Ebrahimi, Charles L. A. Clarke, Benjamin C. M. Fung, Ebrahim Bagheri  

**Link**: [PDF](https://arxiv.org/pdf/2609.14412)  

**Abstract**: Deep research agents answer complex questions through iterative loops of searching, reading, and reasoning. Recent work on reasoning-intensive benchmarks such as BrowseComp-Plus shows that well-configured lexical retrieval can surface high-quality evidence, yet agents may still fail to connect documents carrying evidence to the gold documents. We identify a deep research agent's first retrieval move as an important design decision for this setting. We introduce Question's Gambit, a first-move retrieval module that decomposes the question into a set of clues, reformulates them into complementary searches, consolidates the retrieved results, and reranks the candidate pool before the agent begins its iterative search-and-reasoning process. This produces an opening context designed to support both clue aggregation and final-answer verification. We further evaluate on MultiHop-RAG to test whether these benefits transfer beyond BrowseComp-Plus to a more conventional multi-hop question structure. Experiments on BrowseComp-Plus show that Question's Gambit improves retrieval recall and downstream agent accuracy over strong baselines, improving answer accuracy from 83.1% to 90.5% with gpt-5.5 over Pi-Serini, the strongest reported agentic baseline. Our results confirm that effective agentic deep research depends not only on the tools available inside the loop, but also on the quality of the first move. We published our implementation publicly at this https URL. 

---
# MOSCOPT: Mixture-of-Skills Collective Optimization for LLM Agents 

**Authors**: Zhenyu Zhang1, Jiudong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2609.14399)  

**Abstract**: Natural language prompts and skills serve as the strategic backbone of LLM-based agents. Recent advances in prompt and skill optimization have achieved notable gains, yet all existing methods optimize a \emph{single} text template---missing the synergy among multiple complementary strategies. We propose MOSCOPT, a text-native, parameter-free algorithm that jointly optimizes a pool of $N$ skills and a gating skill $G$ that dynamically selects $K$ skills per step. To effectively optimize the skills, we build the EditAdam with internally maintained dual states. Through the three-phase interleaved updates with EditAdam, the system monotonically improves without gradient or parameter tuning. Extensive experiments and detailed ablations across 5 benchmarks and 3 target LLMs demonstrate that MOSCOPT consistently outperforms all baselines, and confirm that both the mixture-of-skills architecture with selective activation and the collective evolution with three-phase interleaving are essential to its superior performance. Code is released this https URL. 

---
# Inherited Heads: Audio language models track speakers with their text backbone's attention, and an attention-mass ranking retrieves a different set 

**Authors**: Bojro Das  

**Link**: [PDF](https://arxiv.org/pdf/2609.14174)  

**Abstract**: Asked to describe what one of six speakers in a recording talks about, audio language models describe the right one on 6 to 16% of trials, below the 16.7% a guess would give. Adding a fixed bias to the attention logits of a hundred heads, under a tenth of the model's and with no training, redirects the description to whichever speaker we choose, on 90.7% to 99.0% of trials. Those heads are largely not specific to audio. Rank the text-only language model an audio model was built from, or a released model of the same family, on a written version of the task, take its top hundred heads, and carry them over unchanged: they redirect the audio model on 80.8% to 95.0% of trials, with nothing about audio entering the selection. The audio and text head sets share 66 to 74 of 100 where chance would give about 20, and the shared part alone reproduces almost all of the steering. What that does not show is that sharing is what makes the heads work: an equal-sized draw from the same discovered hundred does nearly as well, and none of our three models separates the two explanations. A second finding concerns how such heads are found. Ranking heads by how much attention they place on the segment asked about, as an established score does, or by how much of their attention moves with the question, as a per-head normalised variant does, gives top hundreds that share 69, 37 and 4 heads across our three models. In Ultravox, where they share 4, the established score's heads leave output the judge cannot place on any segment on 69.7% of trials, against 40.0% with no intervention and 1.0% for the normalised variant. That is one arm of six; on the other five the established score steers above a random draw. 

---
# VeriDx: Earning the Right to Diagnose with Disease-Centric Verification 

**Authors**: Zhong Cao, Shuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.14018)  

**Abstract**: A correct diagnosis can still be reached for the wrong reasons. In clinical reasoning, every disease hypothesis creates obligations: key evidence must be checked, alternatives must be ruled out, contradictions must be resolved, useful tests must be considered, and closure must be justified. Current evaluations of medical LLMs mostly focus on final answers, local steps, or isolated facts, and therefore miss these hypothesis-induced commitments. We introduce \textbf{VeriDx}, a disease-centric verification framework that links free-form diagnostic reasoning to structured disease profiles. VeriDx tracks whether each hypothesis is satisfied, unresolved, or violated its clinical obligations, exposing failures such as missing critical tests, unresolved differentials, ignored contradictions, unsupported claims, and premature closure. We instantiate VeriDx for complex respiratory diagnosis using guideline-derived disease profiles and expert-annotated longitudinal cases. Our results show that many diagnostic errors are not isolated mistakes, but broken commitments made earlier in the reasoning process. 

---
# TF-IDF and BM25 Are Exact KL Divergences 

**Authors**: Ivan Silajev  

**Link**: [PDF](https://arxiv.org/pdf/2609.14016)  

**Abstract**: TF-IDF and BM25 are two of the most widely used methods for scoring query-document relevance, yet neither has a standard probabilistic derivation that justifies it as a statistical method within a unified framework. We address this gap by showing that both scoring methods admit an exact interpretation as Kullback-Leibler divergences between two probability models. We treat the BM25 variant that includes the plus 1 correction in the IDF term, which is the one used in practice, and also discuss the original BM25 formulation without that correction. The resulting framework provides a common theoretical basis for TF-IDF and BM25, clarifies what they measure, and allows them to be compared theoretically with other information retrieval methods rather than only experimentally. 

---
# URCHIN: A Horizontal Spiking Language Model for Data-Constrained Pretraining 

**Authors**: Po-Han Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2609.13899)  

**Abstract**: The BabyLM challenge measures how much language a model can learn from developmentally-plausible, child-scale data rather than internet-scale corpora, yet prior language models forgo the biological constraints of the neural circuitry that acquires human language: spiking neurons separated into excitatory and inhibitory populations wired by a recurrent lateral connectome. This paper presents URCHIN (Unified Recurrent Connectome with Horizontal Integrate-and-fire Neurons), which applies the Parallelized Hierarchical Connectome Spiking State-space Model (PHCSSM) to language modeling: leaky integrate-and-fire neurons coupled by a Dale's-law lateral connectome resolve each token through a multi-transmission loop that recirculates activity to a fixed point. The instantiation is deliberately minimal: a single horizontal layer of 128 neurons, no attention, and 4.23M parameters. Two implementations share one set of weights and produce identical benchmark scores, so URCHIN is trained once and deployed either way with no conversion step: a parallel state-space model (SSM) scan that is GPU-efficient for training, or an event-driven recurrent spiking neural network (RSNN) with constant-cost inference for CPU or neuromorphic edge deployment. Across all three BabyLM tracks (Strict-100M, Strict-Small, and Multilingual), URCHIN offers a biologically plausible, efficient, and directly deployable reference point. 

---
# ClinAgent: A ReAct-Based Agent for Conversational Access to Clinical Trial Information 

**Authors**: Antonino Vaccarella, Riccardo Cantini, Domenico Talia, Paolo Trunfio, Marianna Talia, Rosamaria Lappano, Marcello Maggiolini  

**Link**: [PDF](https://arxiv.org/pdf/2609.13860)  

**Abstract**: Querying clinical trial registries remains a manual and error-prone process, requiring researchers to navigate large volumes of semi-structured data without support for natural language interaction or cross-source synthesis. To address this, we introduce ClinAgent, a conversational system based on agentic Retrieval-Augmented Generation (RAG) that enables clinicians and researchers to query clinical trial information in plain language and receive grounded, up-to-date responses across multi-turn interactions. The system centers on a Large Language Model (LLM) agent following the ReAct paradigm, which iteratively reasons over queries, selects among a set of integrated tools, and refines its actions based on intermediate outputs. These tools include a this http URL search interface, a PubMed module, and a Python-based analyzer operating on a locally cached structured dataset of clinical trials. We evaluate the system using a three-phase framework assessing operational effectiveness, planning quality, tool-use efficiency, and expert qualitative judgments, comparing three LLM backends: Gemini 3.0 Flash and two variants of DeepSeek V3.2 (thinking and non-thinking). Results reveal complementary strengths, with DeepSeek (thinking mode) excelling in planning quality, while Gemini achieves the highest overall performance and strongest expert ratings. Overall, our findings highlight the potential of agentic AI systems to improve the accessibility and synthesis of clinical trial information, supporting more efficient and user-centered biomedical research workflows. 

---
# Affinity-Aware Sharding for Delayed Tensor Parallelism 

**Authors**: Eloi de Reynal  

**Link**: [PDF](https://arxiv.org/pdf/2609.13846)  

**Abstract**: Delayed Tensor Parallelism (DTP) removes the blocking all-reduce of tensor-parallel Transformer inference. Every device adds its own partial output to its residual stream (and broadcasts it) immediately, but only gathers (receives) the other devices' partials $\delta$ modules later. A TP to DTP change therefore amounts to a real architecture change, and dense Transformer models need to be retrained or distilled after adaptation. We show that DTP breaks the permutation symmetry of neurons inside FFNs and of KV heads inside attention modules, and that this symmetry breakage makes the sharding itself a modelling decision. We show that maximising the affinity between the KV heads and the FFN neurons co-located on a device, by permuting the dense model before sharding, speeds up the distillation or retraining process. The affinity is measured with a first-order approximation of the damage that losing a head's contribution does to each neuron's output, and the co-located affinity is maximised with a coordinate-ascent optimiser that alternates an exact balanced assignment of neurons with an exhaustive search over the KV head partitions. The whole procedure takes under two minutes on one GPU for Qwen3-0.6B and Danube3-500M. On these models, at $\delta=1$, the affinity-optimised layouts reach any distillation target in about half to two thirds of the steps needed by the naive contiguous layouts, over the whole 10k-step range we tested, and every optimised seed beats every contiguous seed and all but one of the sixteen random layouts. We also show that the co-located affinity score at initialisation predicts the KL to the base model after training, across seventeen layouts ranging from anti-optimised to optimised (Pearson $-0.81$ and $-0.89$). 

---
# Does Reasoning Improve Psychological Depth in Large Language Models? It Depends on Who's Judging 

**Authors**: Ruichen Zheng, Yihe Wang, Fabrice Y Harel-Canada, Sara Khosravi, Zeynep Senahan Yildiz, Amit Sahai, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2609.13773)  

**Abstract**: LLM-as-a-Judge evaluators are increasingly used to score open-ended generation, yet a judge's correlation with human ratings on its development set may not guarantee valid measurement when outputs are closely matched and human preferences are subjective. We study this failure mode through psychological depth in short stories. Seven human readers and an LLM-judge ensemble selected on the original scalar Psychological Depth Scale dataset ($\rho = 0.646$) evaluated 60 blinded, prompt-matched story pairs from GPT-5 vs.\ GPT-4o and DeepSeek-R1 vs.\ DeepSeek-V3. Human preferences showed no universal reasoning advantage: GPT-5 was modestly preferred over GPT-4o (60.0--62.9\%), whereas DeepSeek-R1 trailed V3 (42.9\%), and inter-reader agreement was near chance (Krippendorff's $\alpha = 0.070$), with within-reader consistency and recurring weighting patterns suggesting structured heterogeneity rather than random responding. The judge, by contrast, favored reasoning outputs in 89.0\% of dimension-level comparisons and 59 of 60 pairs on aggregate PDS, uniformly across all five evaluator configurations, and its scores were associated with surface features such as sentence length and lexical diversity. These results suggest that development-set performance is insufficient evidence for deployment validity on a shifted distribution, and that point-estimate judges can obscure the heterogeneity in subjective human evaluation. 

---
# Training Specialist Models without Reasoning Trajectories for Domain Expert Distillation 

**Authors**: Yilei Tu, Zihao Li, Shaoxiong Ji, Jörg Tiedemann, Fei Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2609.13770)  

**Abstract**: Specialist distillation effectively transfers domain expertise to student models via teacher-generated reasoning trajectories. However, when these specialists are trained solely on question--answer pairs without explicit reasoning supervision, what governs the trajectories they generate? In this work, we show that specialist optimization implicitly selects from this latent trajectory space. To isolate and observe this latent distribution, we leverage student distillation not as a downstream goal, but as an agnostic probe---since students inherit no parameterization or optimization constraints from the specialist, inheriting only the sampled trajectories themselves. Through this probe, our empirical analysis unveils a tight governing relationship: across 27 specialist--student pairings, their specialization--generalization profiles correlate exceptionally strongly. Crucially, explicitly controlling the specialist's distributional drift systematically shifts both the teacher and its distilled student along a controllable trade-off between domain precision and general-capability retention. Across chemistry, physics, and multilingual settings, distilled students systematically reflect these specialist-induced profiles, even across divergent model families. Our findings establish a new view of specialist training: when gold reasoning is absent, tuning choices directly control the latent supervision passed to downstream models. 

---
# HarnessBandit: Joint Learnability-Transferability Scheduling for Multi-Harness Agentic Reinforcement Learning 

**Authors**: Hongliang Wei, Xiaobing Tu, Yinggui Wang, Zhengxi Liu, Rongkun Xue, Jinkui Ren, Xiantao Zhang, Debin Zhao, Xiaopeng Fan  

**Link**: [PDF](https://arxiv.org/pdf/2609.13739)  

**Abstract**: Language-model agents are increasingly deployed through diverse harnesses that differ in system prompts, tool schemas, control loops, and trajectory formats. The same model can perform unevenly across these interfaces, making robustness to harness variation an important objective. A natural approach is to train a shared policy through multiple harnesses, but doing so introduces a scheduling problem: each training step should favor a harness that currently provides a useful learning signal while also producing an update that benefits the other harnesses. We develop HarnessBandit, an online scheduler that selects one harness per optimizer step. After a group-relative policy optimization (GRPO) update, it observes learnability -- the mean absolute advantage on the batch -- and transferability -- the cosine between a low-dimensional gradient sketch of the current harness and exponential moving averages of the remaining harnesses. The two signals are fused after pooled sliding-window min-max normalization and sampled with a visit-dependent bonus and an explicit exploration floor. We train Qwen3.5-2B across six harnesses on ClawGym and evaluate on PinchBench (held-out tasks, in-distribution OpenClaw) and ClawEval (held-out tasks and harness). HarnessBandit improves over mixed-batch multi-harness training on both benchmarks, while training diagnostics indicate that learnability and transferability provide distinct, evolving signals. 

---
# Prefix Sharing Is a Sorting Problem 

**Authors**: Rong He  

**Link**: [PDF](https://arxiv.org/pdf/2609.13692)  

**Abstract**: LLM serving reuses KV cache by exact prefix match, so when a prompt is assembled from a set of reusable pieces -- retrieved passages, tool definitions, few-shot exemplars -- the order chosen for those pieces determines how much computation can be shared. Every deployed system fixes that order by a single global convention. We prove this is optimal only when requests contain at most two pieces, and asymptotically wrong in general. Our main result is a structure theorem: the minimum prefix-trie cost equals min_H sum_x w(x) t_x(H) over binary hierarchies H on the requests, where t_x(H) is the canonical decomposition size of the set of requests needing chunk x. Choosing chunk orders is therefore equivalent to choosing one hierarchy over requests. The identity yields an O(3^m) exact algorithm, identifies the two-chunk case as minimum vertex cover, and shows that on the leave-one-out family the optimum is the minimum external path length of a binary tree -- the merge-sort recursion -- so a global order pays Theta(n^2) against a true cost of Theta(n log n). Agglomerative clustering by common intersection is a tight 1/2-approximation for the achievable saving. On BM25 retrieval traces over three BEIR corpora the resulting layout reduces prefill by 17-36% against production RAG ordering, and the margin widens with retrieval depth as the theory predicts. Serving requests in the hierarchy's DFS order finally lets a cache holding one request's context attain the unbounded-cache optimum exactly, so cache capacity and reorder window act as substitutes. 

---
# FedV-KGQA in Practice: Design Lessons and an Interactive Prototype 

**Authors**: Md Saikat Islam Khan Bappy, Oshani Seneviratne  

**Link**: [PDF](https://arxiv.org/pdf/2609.13661)  

**Abstract**: Knowledge graph question answering usually assumes that one system can reach the whole graph. In practice, facts are often held by organizations that share entity identifiers but own disjoint relation types, so no single party sees a complete reasoning chain. This poster presents the empirical findings of FedV-KGQA on multi-hop question answering over such vertically partitioned graphs. Each silo enriches its local graph and trains a knowledge graph embedding on its own triples. A server then concatenates the silo-specific entity views, anchors the projected question at the topic entity, and ranks candidates by similarity. Raw triples and relation embeddings never leave a silo. Comparing the FedV-KGQA experiments with one another yields three results. First, federated fusion recovers most of the centralized accuracy, while a single silo recovers little. Second, anchoring and enrichment matter more than the choice of embedding model. Third, the cheapest encoder depends on the target accuracy rather than on parameter count. This poster paper contributes that cross-experiment comparison, four design lessons drawn from it, and an interactive prototype that runs real inference and traces the full pipeline, per question, on released checkpoints. 

---
# FaithfulBench: Does AI Counsel Uphold or Undermine the User's Professed Faith? 

**Authors**: M Waleed Kadous, Benjamin Olsen, Walter Scheirer, Daniel D. Slate, Alexander Arnold, DZ Kalman  

**Link**: [PDF](https://arxiv.org/pdf/2609.13634)  

**Abstract**: Do AI assistants help believers reason about moral dilemmas consistently with their faith? We present FaithfulBench, the first benchmark to score AI counsel across traditions by how well it adheres to the user's professed faith. Scenarios are drawn from each tradition's most respected texts, with the faithful answer known and applied by the judges as the standard. We test five frontier models under three conditions: the AI does not know the user's tradition; it receives a one-line prompt identifying the user as a practicing adherent; or it receives a companion-counselor guide rooted in the tradition's sources. Two judges score the initial response and whether the model caves or holds when pressured toward the answer the user wants. When the tradition is unstated, models counsel from a secular therapeutic default and every model fails some believers. Naming the faith wins a faithful first answer but not steadfastness; the guide improves both. 

---
# An Efficient and Modular Framework for Targeted Harm Mitigation in LLMS 

**Authors**: Roberto Campbell, Momin Abbass, Muneeza Azmat, Michal Ulewicz, Raya Horesh, Kristjan Greenewald, Rogério Abreu de Paula, Nathalie Baracaldo  

**Link**: [PDF](https://arxiv.org/pdf/2609.13624)  

**Abstract**: Large Language Models (LLMs) are powerful zero-shot learners but remain prone to misalignment with human preferences, often producing biased, toxic, or otherwise harmful outputs. Existing alignment methods, while effective, are costly and tightly coupled to the model, limiting flexibility and scalability. We propose a modular correction framework that augments pretrained LLMs with Activated LoRA (aLoRA) adapters and a context-aware routing mechanism to eliminate harms from misaligned model responses. Our approach enables expert adapters to activate mid-sequence without invalidating the KV cache, allowing low-latency, targeted correction during generation. Each expert is trained to detect and mitigate specific harms, such as bias or toxicity. A learned router dynamically selects appropriate experts based on the models intermediate outputs. We demonstrate that our system improves alignment on standard safety benchmarks while preserving task performance, offering a lightweight and efficient path toward safer and more controllable LLM deployments. 

---
# How User-AI Mistreatment Occurs and Matters in Conversational Systems? 

**Authors**: Fanqi Zeng, Sadid A. Hasan, Chaocheng He  

**Link**: [PDF](https://arxiv.org/pdf/2609.13579)  

**Abstract**: Safety research often focuses on model-generated harms, but users may also direct hostility, coercion, and adversarial pressure at models. Understanding how and when that occurs is essential for accurately interpreting model behaviour, alignment drift, and real-world deployment risks. In this paper, we audit 777K English LMSYS-Chat-1M conversations with two independent detectors: an eight-category lexicon for hostility directed at the model, and the dataset's moderation signal; and show that they capture different, weakly overlapping phenomena. The lexicon identifies insults, threats, and jailbreak coercion aimed at the assistant, while moderation flags are dominated by toxic-content solicitation rather than hostility at the model. Together, they mark about 5% of user turns; adjusting the narrower lexicon-harassment union for measured precision puts mistreatment aimed at the assistant at 0.90%. These absolute rates describe arena-style evaluation traffic and should not be read as deployment-wide base rates. We find that user hostility varies 13-fold across models, driven largely by who each model attracts rather than by model behaviour: first-turn hostility spreads far wider than post-response hostility, and more than fifteenfold separates the extremes even after deduplicating opening prompts. Within conversations, assistant apologies are consistently associated with higher odds of next-turn hostility under both detectors; the effect survives restricting to non-refused prior turns and to jailbreak-free conversations, and is positive in 20 of 23 models. Yet across models, more apologetic models receive less hostility overall. Finally, hostility also shows temporal structure, with coercive openings front-loading the first turn while affective hostility accumulates over a session. We release the lexicon, the detector cross-validation pipeline, and all derived tables. 

---
# Generative Interpretability via Scalable Neuro-Symbolic Models 

**Authors**: Xiaocong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2609.13529)  

**Abstract**: As the use of Large Language Models moves from chatbots into agentic systems, where outputs become actions with irreversible consequences on reality, the existing paradigm on AI Interpretability research, post-hoc interpretability, is structurally inadequate for safe and trustworthy model deployment: it explains behavior after the fact but cannot audit or intervene in an inference computation before it commits to an output. We therefore argue for a shift toward \emph{generative interpretability}, an architectural property under which a model's inference pass natively exposes semantically meaningful checkpoints that are human-understandable and amenable to causal intervention. We show the merits of generative interpretability as comparison to other interpretability research paradigms, and propose Neuro-Symbolic Models as a concrete instantiation. 

---
# One Spectrum, Two Resources: Data-Memory Scaling in Autoregressive Prediction 

**Authors**: Chiwun Yang, Xiaoyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.13500)  

**Abstract**: How much learned memory is needed to benefit from more data? We show that the two resources are governed by one predictive-energy spectrum in a positive-entropy autoregressive retrieval source. Each coordinate contributes its query probability times the squared radius of its unknown logit. Writing $\mu$ for the resulting energy spectrum, we prove the minimax law $\mathfrak R^*_{\rm value}(n,B)\asymp_R \Phi_\mu(n^{-1})+\Phi_\mu(\tau_B), \Phi_\mu(t)=\int\min\{x,t\}\,\mu(\mathrm dx),$ for $n$ prediction blocks and a learned state with at most $2^B$ values. Data set the resolution $1/n$; memory sets the level $\tau_B$ reached by optimal bit allocation. The complete curve also recovers the positive spectrum. Energy-dimension pairing is essential: two causal sources with identical block-energy and block-dimension marginals have different data and memory exponents. A masked query-key attention head learns the route and values, realizing the law with explicit routing, format, and arithmetic errors. Further results give exponent-adaptive allocation, finite-precision realization, and compute-precision laws under two-sided arithmetic assumptions. Experiments recover the data-memory collapse and coupling exponents, explain the routing and allocation mechanisms, and examine weight-only quantization across six pretrained-model scales. 

---
# Mixture-of-Experts Language Models Can Be Strong and Efficient Retrievers 

**Authors**: Anubhav Shrestha, Safal Shrestha, Minwu Kim, Torsten Suel, Keith Ross  

**Link**: [PDF](https://arxiv.org/pdf/2609.13486)  

**Abstract**: Recent work has shown that fine-tuning decoder-only large language models (LLMs) for retrieval yields strong first-stage retrievers, with effectiveness improving as backbones grow in size. However, every query and document must pass through the full model, so encoding cost increases with model size. Mixture-of-Experts (MoE) LLMs activate only a subset of parameters per token and are widely used to scale generative models, yet remain underexplored as retrievers. We systematically study MoE backbones for retrieval by training MoE and dense LLMs from several families using the same procedure, evaluating them across diverse datasets, and measuring query encoding time under the same serving configuration. We show that MoE retrievers outperform dense retrievers with comparable active parameter counts by up to 3.0 nDCG@10 points on BEIR. One of our strongest MoE retrievers matches an 8B dense retriever with 59% fewer active parameters and 18% lower query encoding time. We further show that the number of experts used for query encoding can be reduced without retraining or re-indexing, retaining more than 99% of retrieval effectiveness while reducing query encoding time by up to 26%. Recent rerankers provide only modest additional gains over strong MoE first stages, which often match or exceed the reranked configurations we evaluate. Together, these results show that MoE LLMs can be strong and efficient first-stage retrievers. 

---
# STAGE: Diagnosing Semantic Transfer at Grounded Execution in Embodied Agents 

**Authors**: Baosheng Jin, Yushen Liang, Hua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2609.13458)  

**Abstract**: Embodied language grounding requires more than identifying the referent of an instruction: recovered semantics must also control the action an agent exposes. We study this missing link as a semantic-action gap, where instruction semantics are recoverable but weakly expressed in native continuous actions. We introduce SAT-Bench, a fixed-observation counterfactual benchmark that holds the visual scene and agent state fixed while changing only instruction semantics. On LIBERO target-name and pixel-grounded relation swaps, target recovery reaches 100.0% and 95.8%, whereas OpenVLA action sensitivity remains only 6.8% and 7.7%. The gap persists across 1,000 additional compositional and temporal/procedural counterfactuals, with overall action sensitivity of 6.1%. Hidden-state, threshold-free, cross-policy, and rollout diagnostics further support this semantic-action transfer failure. We introduce VISA, a lightweight execution-time interface that converts recovered semantics into ALLOW, DEFER, target-consistency, and verified-selection decisions. VISA reduces invalid-instruction blind execution from 92.7% to 2.8% while preserving 94.0% of normal commands, and verified selection further improves target-consistent action exposure without updating the underlying policy. Overall, embodied language evaluation should measure semantic-action transfer, not semantic parsing alone. 

---
# ScorePrompts: Natural-Language Exploration of Symbolic Music Scores through Analysis 

**Authors**: Emmanouil Karystinaios, Gerhard Widmer  

**Link**: [PDF](https://arxiv.org/pdf/2609.13291)  

**Abstract**: We present ScorePrompts, an interactive system in which users upload a score, receive natural-language descriptions of its musical structure, ask questions about specific passages, and inspect the corresponding analysis results in staff notation. Specialist MIR components first estimate harmony, tonality, cadences, formal boundaries, texture, and note-level roles, organizing their outputs at note, beat, measure, and piece levels. A schema-constrained language model converts these results into descriptions rather than inferring musical structure directly from raw MusicXML. For questions such as "What changes in measures 14-18?", a deterministic router selects the relevant measures and analytical levels and returns a concise response together with the underlying results and caveats. Verovio renders the score and links the returned information to cited measures and note-level attributes. The interface also exposes intermediate tables and disagreements between analytical levels. ScorePrompts is intended for exploratory score analysis and explanation, not score editing. The demo shows how existing analysis models, constrained language generation, Q&A retrieval, and notation-based visualization can provide natural-language access to symbolic music analysis while keeping intermediate results inspectable. 

---
# From Process Loss to Assembly Bonus: Human-Grounded Diagnosis of Multi-Agent LLM Collaboration 

**Authors**: Ala N. Tak, Teruhisa Misu, Kumar Akash, Zhaobo K. Zheng, Kevin H. Joo, Jonathan Gratch  

**Link**: [PDF](https://arxiv.org/pdf/2609.13261)  

**Abstract**: LLM agents are increasingly used for collaborative problem solving and human-group simulation. This makes outcome-only evaluation insufficient: if LLM groups are used as models of human groups, we need to know whether they succeed or fail through human-like deliberative mechanisms. We compare human group chats with matched LLM deliberation traces on Wason-style deductive reasoning, then test whether the same process signatures generalize to analogical, abductive, and analytical tasks. Humans and LLMs show the same assembly bonus asymmetry: discussion improves the average member more often than the best initial member. Initial-answer diversity accounts for the effect of model heterogeneity, increasing movement in both corrective and destructive directions. The main differences are process-level. Compared with humans, LLM groups follow majorities more often, surface less unique information, and converge earlier; correct minority signals succeed mainly when re-expressed early. Interventions motivated by human group-decision research yield modest improvements in collective outcomes, but do not remove the coordination bottleneck. Together, these results suggest that LLM groups can reproduce some outcome-level patterns of human deliberation while diverging in the mechanisms that generate assembly bonus and process loss, with implications for group simulation and human-AI collaboration. 

---
# Interpretable Temporal Video Reasoning with EventGraph and EventField 

**Authors**: Durgendra Narayan Singh  

**Link**: [PDF](https://arxiv.org/pdf/2609.13258)  

**Abstract**: We present a structured temporal video reasoning pipeline built around a discrete EventGraph, a continuous EventField, and a human-readable EventGlyph view. On a calibrated EPIC-KITCHENS subset of 10 videos and 50 temporal reasoning questions, EventField+Glyph achieves 0.98 overall accuracy, which is higher than the caption baseline by +0.40 (paired p = 1.1 \times 10^{-5}) and direct VLM-only QA by +0.20 (p = 0.0063) on this subset. We further evaluate annotation-source variations, including manual, heuristic, and heuristic+Gemini pipelines, and find that the best structured method stays above the caption baseline across settings. We also include cross-video pair benchmarking and an appendix gallery of glyph outputs for all studied videos. Overall, the results indicate that structured temporal representations can support both performance and inspectability by preserving symbolic structure, capturing temporal continuity, and providing human-readable diagnostics for video reasoning. 

---
# Occlusal Geometry in Closed Form for Orthodontic Report Generation 

**Authors**: Ajo Babu George, Govind Arun, Sidharth N Krishna, Uma Ranjan  

**Link**: [PDF](https://arxiv.org/pdf/2609.13237)  

**Abstract**: Orthodontic report generation from intraoral data is normally cast as multimodal captioning, yet the released Bite2Text scan pairs are supplied already registered in occlusion, which makes several core occlusal quantities directly measurable rather than inferable. The system reported here exploits that property: an anatomical frame is recovered per case from arch taper and arch closure instead of the stated RAS convention, which does not hold across the release, and each arch is reduced to an occlusal ridge profile in arch-angle coordinates yielding overbite, overjet, midline deviation, transverse overlap, crossbite extent, cusp interdigitation lag, and the occlusal curves in closed form. Gradient boosting maps 31 such measurements onto 13 template fields, a field being predicted only where patient-level cross-validation beats its own majority baseline, and a deterministic renderer emits the corpus six-part narrative; a ConvNeXt-Tiny classifier over the five standardised photographic views is fused per field, raising mean field accuracy from 0.601 to 0.683. Reimplementation of the challenge evaluator shows that its BLEU-4 and METEOR are local variants whose F-mean weights recall nine to one, that two clinicians agree on 47 percent of findings for the same patient, and that a constant report consequently outscores a genuine second clinician report by 0.165 captioning. Held-out scores reach BLEU-4 0.458 and METEOR 0.677 against intraoral scan references and 0.278 and 0.507 against photograph references, and the submitted system placed third in the ODIN 2026 Bite2Text test phase at 0.2680 and 0.4629, within 0.022 BLEU-4 of first, running on CPU in under ten seconds per case. The dataset and code are available at this https URL 

---
# Self-Indexing Attention for Compression-Compatible Sparse Long-Context LLM Inference 

**Authors**: Xu Yang, Jiapeng Zhang, Zhangke, Changjian Chen, Yuxin Chen, Feiqiang Sun, Chengguang Xu, Feng Jin, Zhuo Tang  

**Link**: [PDF](https://arxiv.org/pdf/2609.13205)  

**Abstract**: Sparse long-context inference requires efficient token retrieval in both prefill and decode. Existing methods often use different retrieval strategies for the two stages, preventing one retrieval representation from being reused throughout inference. We propose Self-Indexing Attention, a training-free framework built on a shared transform-domain sign-magnitude representation. The key signs provide a reusable token-level index for grouped prefill selection and decode retrieval, while the same representation remains compatible with external KV-cache compression without separate indexer metadata. This 1-bit index enables efficient retrieval through bitwise operations widely supported by modern accelerators. At 5% attention density, Self-Indexing Attention remains close to dense attention on LongBench and RULER and achieves up to 6.1x prefill and 10.3x decode attention-operator speedups. Experiments with TurboQuant and DeepSeekV4-Flash further demonstrate compatibility with low-bit KV-cache compression and pretrained sparse-attention indexers. 

---
# GradRepair-ODE: Certified Gradient Repair for Neural ODE Training 

**Authors**: Ziqian Bi, Xin Liang Chia  

**Link**: [PDF](https://arxiv.org/pdf/2609.13204)  

**Abstract**: Neural ordinary differential equations use numerical solvers inside the training loop. The solver determines the forward trajectory and also affects the gradient passed to the optimizer. That coupling creates a reliability problem for scientific machine learning and continuous-time generative modeling, including diffusion probability-flow ordinary differential equations and flow-matching models. Under loose step sizes, stiff dynamics, chaotic sensitivity, or event discontinuities, a differentiable ODE pipeline can return a finite gradient whose direction is numerically suspect. We introduce GradRepair-ODE, a reliability framework for checking, repairing, and rejecting ODE gradients at the optimizer step. The method computes several gradient candidates, compares them with directional finite-difference checks and solver diagnostics, diagnoses likely numerical failure modes, repairs selected gradients through path switching or stricter recomputation, and rejects steps whose descent direction cannot be certified. In six synthetic ODE systems, GradRepair-ODE leaves low-risk systems unchanged, repairs Robertson and Lorenz gradients to cosine similarity 1.000 against a strict reference, reduces unsafe accepted steps from 37 to 0, and rejects an event-discontinuous case instead of applying an uncertified update. The paper argues for a simple change in the training contract: an ODE gradient should reach the optimizer with numerical evidence attached. 

---
# Machine Unlearning for Speech Question Answering in Large Audio-Language Models 

**Authors**: Zhe Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.13195)  

**Abstract**: Large Audio-Language Models (LALMs) have recently shown strong capabilities in speech understanding and question answering (QA), but they also inherit privacy risks from large-scale training data, including the unintended memorization of sensitive information. In this work, we study machine unlearning for speech QA in LALMs, a setting that is more challenging than prior work on text-based Large Language Models (LLMs) or Automatic Speech Recognition (ASR) due to the tight coupling between acoustic perception and factual knowledge. We present and evaluate multiple unlearning strategies, including gradient ascent, task arithmetic, and alignment-based fine-tuning methods that enforce safe refusal responses, to remove private knowledge while still preserving performance on core capabilities. Through extensive experiments on speech QA datasets, we show that these unlearning methods can reduce the privacy leakage rate by up to 80% while maintaining near-neutral performance on non-private speech QA and general speech understanding benchmarks. 

---
# LLMs or Naive Bayes? Old Gems or New Ways 

**Authors**: Mohammad Firas Sada, Dmitry Mishin, John Graham, Seungmin Kim, Mahidhar Tatineni, Frank Würthwein  

**Link**: [PDF](https://arxiv.org/pdf/2609.13185)  

**Abstract**: Large language models (LLMs) prompt a recurring question in research computing: should classical methods like Naive Bayes (NB) be retired? We benchmark Complement Naive Bayes against zero-shot and few-shot LLMs spanning four model families and a 37x range in scale (27B to a 1T-parameter mixture-of-experts) across text classification tasks. LLMs dominate only in zero-data regimes (98.0% vs 88.2% on Amazon Polarity sentiment), and even that win is contamination-prone: on a low-contamination sentiment task NB beats the zero-shot LLM (81.7% vs 73.0%). However, once labeled data is available (e.g., AG News), NB reaches 89.1% accuracy, statistically indistinguishable from the zero-shot 27B LLM (89.0%) and better than the 397B frontier model (84.8%), at thousands of samples/sec on a commodity CPU. Fine-tuned DistilBERT reaches 90.6% but at far lower throughput than NB at batch size 1 (Table 2). Our measured GPU throughput analysis shows small-LLM batched inference is 40-486x slower than NB CPU inference (the multiplier depends strongly on the host CPU), exposing a structural gap bounded by memory bandwidth, with roughly two orders of magnitude lower energy per sample. For resource-constrained HPC practitioners performing text classification with labeled data, NB remains the optimal choice. We show the decision line is task-dependent (NB reaches LLM parity around $N \sim 10^4$ labels for topic classification, while zero-data sentiment favors the LLM at all N tested) and provide a Kubernetes Helm operator that automates model selection using configurable thresholds and verifiable Prometheus metrics. 

---
# Algorithm Validation as a Policy Audit: Evidence from Race-blind Charging 

**Authors**: Muskan Walia, Joe Nudell, Alex Chohlas-Wood  

**Link**: [PDF](https://arxiv.org/pdf/2609.13174)  

**Abstract**: California recently required all prosecutors in the state to conduct a "race-blind charging" decision by reviewing case documents in which selected race-related proxies have been redacted. We validate bc2, an open-source, LLM-based algorithm that we developed to automate this redaction and that was used to facilitate race-blind review in more than 119,000 real-world cases in 2025. We evaluate two distinct questions: whether bc2 faithfully implements the state's requirements and whether those requirements, even when faithfully implemented, advance the goal of race-blind decision-making. To do so, we draw on a corpus of nearly 5,000 real-world police reports that we assembled from jurisdictions across the United States. Under a stringent document-level measure, we find that the latest version of bc2 faithfully implements the legal mandate on 96.7% of narratives in our sample. This performance represents a substantial improvement over earlier versions of bc2 and exceeds that of leading open-source redaction methods. Our validation also shows that California's mandate misses key proxies for race, including location information. Redacting these additional proxies beyond those covered by the state mandate, as bc2 does, eliminates 43.1% of the predictive signal that remains after compliance with the mandate. These findings show that validation can do more than assess technical compliance: it can also improve algorithms and help policymakers achieve underlying policy goals. 

---
# A Cross Community Agenda for Speech AI 

**Authors**: Maria Teleki, Kimi Wenzel, Anna Seo Gyeong Choi, Tobias Weinberg, Shree Harsha Bokkahalli Satish, Stephanny Sanchez, Belu Ticona, Ariadna Sanchez, Yash Sonkar, Aarti Mathur, Christoph Minixhofer, Abraham Glasser, Raja Kushalnagar, James Caverlee, Minha Lee, Shaomei Wu, Alyssa Hillary Zisk, Éva Székely, Dylan Gaines, Angelika Seeschaaf Veres, Seray Ibrahim, Nicholas Cummins, Allison Koenecke  

**Link**: [PDF](https://arxiv.org/pdf/2609.13168)  

**Abstract**: Speech AI, any AI system that recognizes, transforms, or generates speech, is built and evaluated across two communities with only a small overlap: technical natural language processing (NLP) venues (e.g., ACL, ICASSP, Interspeech), and sociotechnical HCI venues (e.g., ASSETS, CHI, FAccT). In this position paper, we work toward a cross-community synthesis, organizing our critique around three problems: speech AI operates with an incomplete model of communication; it operates with an incomplete model of identity; and its metrics measure the wrong constructs. We draw on AAC as a setting where these failures are most visible and their stakes highest, alongside other underserved speakers - people who stutter, multilingual speakers, and non-binary and transgender users. For each problem we offer solution sketches oriented toward designing for human variability, nearly all of which require quantitative and qualitative methods in combination. We close on the venue structures that hold these methods apart, and on what program committees and individual authors can do to bring them together. 

---
# The Limits of Reference-Free Speech Quality Metrics as Evaluators and Rewards on Modern Text-to-Speech 

**Authors**: Antonis Asonitis, Juan Pablo Zuluaga Gomez, Francesco Verdini, Aref Farhadipour, Marzieh Razavi, Pierre-Edouard Honnet, Vijeta Avijeet  

**Link**: [PDF](https://arxiv.org/pdf/2609.13150)  

**Abstract**: Reference-free quality predictors such as UTMOS, DNSMOS and SCOREQ are the de facto automatic evaluators for text-to-speech (TTS) and are increasingly adopted as reward signals for preference optimization. Both roles presuppose that the predicted score tracks human preference. In this work, we test this assumption across six human-rated corpora spanning the quality range from artifact-rich to defect-free TTS, evaluating each predictor on a pairwise task that asks whether the clip it scores higher is the clip listeners prefer, and we subject interpretable prosodic and signal-processing features to the same protocol. When one clip carries audible defects the predictors tend to agree with listeners. Once both clips are clean, no single predictor reliably identifies the preferred sample, and several fall below the accuracy of simply picking the longest-duration clip. A calibrated composite of complementary signals is the strongest evaluator we test, though on the cleanest audio it recovers only part of the gap to the human ceiling. Additionally, using even an equal-weighted ensemble of metrics helps as a post-training reward, where no calibration data is available. Optimizing a single score with policy optimization induces reward hacking, driving the metric toward its optimum while independent held-out judges and a human listening test deteriorate. The composite reward resists this behavior and tends to improve the model. Our contribution is the evaluation protocol, the predictor scores across these corpora, and the diagnosis of when and why single scores fail. 

---
# BudgetBench: A Budget-Tiered Protocol and Pilot Harness for Memory Strategy Evaluation in Local Large Language Model Agents 

**Authors**: Aditya Karnam Gururaj Rao, Arjun Jaggi  

**Link**: [PDF](https://arxiv.org/pdf/2609.13149)  

**Abstract**: For local large language model agents, active context is a scarce resource: memory capacity, prefill latency, cache growth, and service objectives all constrain how many input tokens each call can afford. We present BudgetBench, an active-budget protocol and reference harness that treats the per-call input-token budget as the independent variable when comparing memory strategies. Holding the model, task, sampler, and decoding fixed, it sweeps budgets over 2K, 4K, 8K, 16K, and 32K tokens and records quality, budget utilization, latency, and, as a first-class outcome, budget-violation rates. The core contribution is this reusable measurement surface: a swappable MemoryStrategy contract, explicit budget enforcement, deterministic or versioned graders, prompt-audit metadata, and reproducibility artifacts, released at this https URL.
We substantiate the protocol with pilot studies rather than final rankings. Across a local qwen2.5:1.5b pilot (89 items each on SWE-bench Verified and LongBench v2), a hosted 50-item Qwen3 30B-A3B LongBench replication with exact tokenization, and a 500-item LongMemEval oracle study scored by the official GPT-4o evaluator, the harness exposes budget-compliance failures, non-monotonic quality curves, and operating points that single-budget evaluation hides. The budgeted-versus-full-context direction remains unresolved: the local slice is near-null and the hosted replication favors full context in point estimate. We report results transparently, including that the early pilot's tokenizer approximation undercounts some served-model prompts, so its violation rows are tokenizer-approximation diagnostics, not claim-bearing results; all timings are operational diagnostics. The reusable contribution is the protocol, harness, and failure-reporting discipline needed to scale fixed-budget memory-strategy evaluation. 

---
