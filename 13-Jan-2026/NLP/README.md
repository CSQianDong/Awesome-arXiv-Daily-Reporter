# Reference Games as a Testbed for the Alignment of Model Uncertainty and Clarification Requests 

**Authors**: Manar Ali, Judith Sieker, Sina Zarrieß, Hendrik Buschmeier  

**Link**: [PDF](https://arxiv.org/pdf/2601.07820)  

**Abstract**: In human conversation, both interlocutors play an active role in maintaining mutual understanding. When addressees are uncertain about what speakers mean, for example, they can request clarification. It is an open question for language models whether they can assume a similar addressee role, recognizing and expressing their own uncertainty through clarification. We argue that reference games are a good testbed to approach this question as they are controlled, self-contained, and make clarification needs explicit and measurable. To test this, we evaluate three vision-language models comparing a baseline reference resolution task to an experiment where the models are instructed to request clarification when uncertain. The results suggest that even in such simple tasks, models often struggle to recognize internal uncertainty and translate it into adequate clarification behavior. This demonstrates the value of reference games as testbeds for interaction qualities of (vision and) language models. 

---
# The Confidence Trap: Gender Bias and Predictive Certainty in LLMs 

**Authors**: Ahmed Sabir, Markus Kängsepp, Rajesh Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2601.07806)  

**Abstract**: The increased use of Large Language Models (LLMs) in sensitive domains leads to growing interest in how their confidence scores correspond to fairness and bias. This study examines the alignment between LLM-predicted confidence and human-annotated bias judgments. Focusing on gender bias, the research investigates probability confidence calibration in contexts involving gendered pronoun resolution. The goal is to evaluate if calibration metrics based on predicted confidence scores effectively capture fairness-related disparities in LLMs. The results show that, among the six state-of-the-art models, Gemma-2 demonstrates the worst calibration according to the gender bias benchmark. The primary contribution of this work is a fairness-aware evaluation of LLMs' confidence calibration, offering guidance for ethical deployment. In addition, we introduce a new calibration metric, Gender-ECE, designed to measure gender disparities in resolution tasks. 

---
# Learning Through Dialogue: Unpacking the Dynamics of Human-LLM Conversations on Political Issues 

**Authors**: Shaz Furniturewala, Gerard Christopher Yeo, Kokil Jaidka  

**Link**: [PDF](https://arxiv.org/pdf/2601.07796)  

**Abstract**: Large language models (LLMs) are increasingly used as conversational partners for learning, yet the interactional dynamics supporting users' learning and engagement are understudied. We analyze the linguistic and interactional features from both LLM and participant chats across 397 human-LLM conversations about socio-political issues to identify the mechanisms and conditions under which LLM explanations shape changes in political knowledge and confidence. Mediation analyses reveal that LLM explanatory richness partially supports confidence by fostering users' reflective insight, whereas its effect on knowledge gain operates entirely through users' cognitive engagement. Moderation analyses show that these effects are highly conditional and vary by political efficacy. Confidence gains depend on how high-efficacy users experience and resolve uncertainty. Knowledge gains depend on high-efficacy users' ability to leverage extended interaction, with longer conversations benefiting primarily reflective users. In summary, we find that learning from LLMs is an interactional achievement, not a uniform outcome of better explanations. The findings underscore the importance of aligning LLM explanatory behavior with users' engagement states to support effective learning in designing Human-AI interactive systems. 

---
# Kinship Data Benchmark for Multi-hop Reasoning 

**Authors**: Tianda Sun, Dimitar Kazakov  

**Link**: [PDF](https://arxiv.org/pdf/2601.07794)  

**Abstract**: Large language models (LLMs) are increasingly evaluated on their ability to perform multi-hop reasoning, i.e., to combine multiple pieces of information into a coherent inference. We introduce KinshipQA, a benchmark designed to probe this capability through reasoning over kinship relations. The central contribution of our work is a generative pipeline that produces, on demand, large-scale, realistic, and culture-specific genealogical data: collections of interconnected family trees that satisfy explicit marriage constraints associated with different kinship systems. This allows task difficulty, cultural assumptions, and relational depth to be systematically controlled and varied. From these genealogies, we derive textual inference tasks that require reasoning over implicit relational chains. We evaluate the resulting benchmark using six state-of-the-art LLMs, spanning both open-source and closed-source models, under a uniform zero-shot protocol with deterministic decoding. Performance is measured using exact-match and set-based metrics. Our results demonstrate that KinshipQA yields a wide spread of outcomes and exposes systematic differences in multi-hop reasoning across models and cultural settings. 

---
# Beyond Single-Shot: Multi-step Tool Retrieval via Query Planning 

**Authors**: Wei Fang, James Glass  

**Link**: [PDF](https://arxiv.org/pdf/2601.07782)  

**Abstract**: LLM agents operating over massive, dynamic tool libraries rely on effective retrieval, yet standard single-shot dense retrievers struggle with complex requests. These failures primarily stem from the disconnect between abstract user goals and technical documentation, and the limited capacity of fixed-size embeddings to model combinatorial tool compositions. To address these challenges, we propose TOOLQP, a lightweight framework that models retrieval as iterative query planning. Instead of single-shot matching, TOOLQP decomposes instructions into sub-tasks and dynamically generates queries to interact with the retriever, effectively bridging the semantic gap by targeting the specific sub-tasks required for composition. We train TOOLQP using synthetic query trajectories followed by optimization via Reinforcement Learning with Verifiable Rewards (RLVR). Experiments demonstrate that TOOLQP achieves state-of-the-art performance, exhibiting superior zero-shot generalization, robustness across diverse retrievers, and significant improvements in downstream agentic execution. 

---
# Enhancing Self-Correction in Large Language Models through Multi-Perspective Reflection 

**Authors**: Mariana Costa, Alberlucia Rafael Soarez, Daniel Kim, Camila Ferreira  

**Link**: [PDF](https://arxiv.org/pdf/2601.07780)  

**Abstract**: While Chain-of-Thought (CoT) prompting advances LLM reasoning, challenges persist in consistency, accuracy, and self-correction, especially for complex or ethically sensitive tasks. Existing single-dimensional reflection methods offer insufficient improvements. We propose MyGO Poly-Reflective Chain-of-Thought (PR-CoT), a novel methodology employing structured multi-perspective reflection. After initial CoT, PR-CoT guides the LLM to self-assess its reasoning across multiple predefined angles: logical consistency, information completeness, biases/ethics, and alternative solutions. Implemented purely via prompt engineering, this process refines the initial CoT into a more robust and accurate final answer without model retraining. Experiments across arithmetic, commonsense, ethical decision-making, and logical puzzles, using GPT-three point five and GPT-four models, demonstrate PR-CoT's superior performance. It significantly outperforms traditional CoT and existing reflection methods in logical consistency and error correction, with notable gains in nuanced domains like ethical decision-making. Ablation studies, human evaluations, and qualitative analyses further validate the contribution of each reflection perspective and the overall efficacy of our poly-reflective paradigm in fostering more reliable LLM reasoning. 

---
# Contrastive Learning with Narrative Twins for Modeling Story Salience 

**Authors**: Igor Sterner, Alex Lascarides, Frank Keller  

**Link**: [PDF](https://arxiv.org/pdf/2601.07765)  

**Abstract**: Understanding narratives requires identifying which events are most salient for a story's progression. We present a contrastive learning framework for modeling narrative salience that learns story embeddings from narrative twins: stories that share the same plot but differ in surface form. Our model is trained to distinguish a story from both its narrative twin and a distractor with similar surface features but different plot. Using the resulting embeddings, we evaluate four narratologically motivated operations for inferring salience (deletion, shifting, disruption, and summarization). Experiments on short narratives from the ROCStories corpus and longer Wikipedia plot summaries show that contrastively learned story embeddings outperform a masked-language-model baseline, and that summarization is the most reliable operation for identifying salient sentences. If narrative twins are not available, random dropout can be used to generate the twins from a single story. Effective distractors can be obtained either by prompting LLMs or, in long-form narratives, by using different parts of the same story. 

---
# Structure First, Reason Next: Enhancing a Large Language Model using Knowledge Graph for Numerical Reasoning in Financial Documents 

**Authors**: Aryan Mishra, Akash Anil  

**Link**: [PDF](https://arxiv.org/pdf/2601.07754)  

**Abstract**: Numerical reasoning is an important task in the analysis of financial documents. It helps in understanding and performing numerical predictions with logical conclusions for the given query seeking answers from financial texts. Recently, Large Language Models (LLMs) have shown promising results in multiple Question-Answering (Q-A) systems with the capability of logical reasoning. As documents related to finance often consist of long and complex financial contexts, LLMs appear well-suited for building high-quality automated financial question-answering systems. However, LLMs often face challenges in accurately processing the various numbers within financial reports. Extracting numerical data from unstructured text and semi-structured tables, and reliably performing accurate calculations, remains a significant bottleneck for numerical reasoning in most state-of-the-art LLMs. Recent studies have shown that structured data augmentations, such as Knowledge Graphs (KGs), have notably improved the predictions of LLMs along with logical explanations. Thus, it is an important requirement to consider inherent structured information in financial reports while using LLMs for various financial analytics. This paper proposes a framework to incorporate structured information using KGs along with LLM predictions for numerical reasoning tasks. The KGs are extracted using a proposed schema inherently from the document under processing. We evaluated our proposed framework over the benchmark data FinQA, using an open-source LLM, namely Llama 3.1 8B Instruct. We observed that the proposed framework improved execution accuracy by approximately 12% relative to the vanilla LLM. 

---
# Is Agentic RAG worth it? An experimental comparison of RAG approaches 

**Authors**: Pietro Ferrazzi, Milica Cvjeticanin, Alessio Piraccini, Davide Giannuzzi  

**Link**: [PDF](https://arxiv.org/pdf/2601.07711)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems are usually defined by the combination of a generator and a retrieval component that extracts textual context from a knowledge base to answer user queries. However, such basic implementations exhibit several limitations, including noisy or suboptimal retrieval, misuse of retrieval for out-of-scope queries, weak query-document matching, and variability or cost associated with the generator. These shortcomings have motivated the development of "Enhanced" RAG, where dedicated modules are introduced to address specific weaknesses in the workflow. More recently, the growing self-reflective capabilities of Large Language Models (LLMs) have enabled a new paradigm, which we refer to as "Agentic" RAG. In this approach, the LLM orchestrates the entire process-deciding which actions to perform, when to perform them, and whether to iterate-thereby reducing reliance on fixed, manually engineered modules. Despite the rapid adoption of both paradigms, it remains unclear which approach is preferable under which conditions. In this work, we conduct an extensive, empirically driven evaluation of Enhanced and Agentic RAG across multiple scenarios and dimensions. Our results provide practical insights into the trade-offs between the two paradigms, offering guidance on selecting the most effective RAG design for real-world applications, considering both costs and performance. 

---
# Emotional Support Evaluation Framework via Controllable and Diverse Seeker Simulator 

**Authors**: Chaewon Heo, Cheyon Jin, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2601.07698)  

**Abstract**: As emotional support chatbots have recently gained significant traction across both research and industry, a common evaluation strategy has emerged: use help-seeker simulators to interact with supporter chatbots. However, current simulators suffer from two critical limitations: (1) they fail to capture the behavioral diversity of real-world seekers, often portraying them as overly cooperative, and (2) they lack the controllability required to simulate specific seeker profiles. To address these challenges, we present a controllable seeker simulator driven by nine psychological and linguistic features that underpin seeker behavior. Using authentic Reddit conversations, we train our model via a Mixture-of-Experts (MoE) architecture, which effectively differentiates diverse seeker behaviors into specialized parameter subspaces, thereby enhancing fine-grained controllability. Our simulator achieves superior profile adherence and behavioral diversity compared to existing approaches. Furthermore, evaluating 7 prominent supporter models with our system uncovers previously obscured performance degradations. These findings underscore the utility of our framework in providing a more faithful and stress-tested evaluation for emotional support chatbots. 

---
# Exploring the Meta-level Reasoning of Large Language Models via a Tool-based Multi-hop Tabular Question Answering Task 

**Authors**: Nick Ferguson, Alan Bundy, Kwabena Nuamah  

**Link**: [PDF](https://arxiv.org/pdf/2601.07696)  

**Abstract**: Recent advancements in Large Language Models (LLMs) are increasingly focused on "reasoning" ability, a concept with many overlapping definitions in the LLM discourse. We take a more structured approach, distinguishing meta-level reasoning (denoting the process of reasoning about intermediate steps required to solve a task) from object-level reasoning (which concerns the low-level execution of the aforementioned steps.) We design a novel question answering task, which is based around the values of geopolitical indicators for various countries over various years. Questions require breaking down into intermediate steps, retrieval of data, and mathematical operations over that data. The meta-level reasoning ability of LLMs is analysed by examining the selection of appropriate tools for answering questions. To bring greater depth to the analysis of LLMs beyond final answer accuracy, our task contains 'essential actions' against which we can compare the tool call output of LLMs to infer the strength of reasoning ability. We find that LLMs demonstrate good meta-level reasoning on our task, yet are flawed in some aspects of task understanding. We find that n-shot prompting has little effect on accuracy; error messages encountered do not often deteriorate performance; and provide additional evidence for the poor numeracy of LLMs. Finally, we discuss the generalisation and limitation of our findings to other task domains. 

---
# Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference 

**Authors**: Rei Taniguchi, Yuyang Dong, Makoto Onizuka, Chuan Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2601.07667)  

**Abstract**: Due to the prevalence of large language models (LLMs), key-value (KV) cache reduction for LLM inference has received remarkable attention. Among numerous works that have been proposed in recent years, layer-wise token pruning approaches, which select a subset of tokens at particular layers to retain in KV cache and prune others, are one of the most popular schemes. They primarily adopt a set of pre-defined layers, at which tokens are selected. Such design is inflexible in the sense that the accuracy significantly varies across tasks and deteriorates in harder tasks such as KV retrieval. In this paper, we propose ASL, a training-free method that adaptively chooses the selection layer for KV cache reduction, exploiting the variance of token ranks ordered by attention score. The proposed method balances the performance across different tasks while meeting the user-specified KV budget requirement. ASL operates during the prefilling stage and can be jointly used with existing KV cache reduction methods such as SnapKV to optimize the decoding stage. By evaluations on the InfiniteBench, RULER, and NIAH benchmarks, we show that equipped with one-shot token selection, where tokens are selected at a layer and propagated to deeper layers, ASL outperforms state-of-the-art layer-wise token selection methods in accuracy while maintaining decoding speed and KV cache reduction. 

---
# Order in the Evaluation Court: A Critical Analysis of NLG Evaluation Trends 

**Authors**: Jing Yang, Nils Feldhus, Salar Mohtaj, Leonhard Hennig, Qianli Wang, Eleni Metheniti, Sherzod Hakimov, Charlott Jakob, Veronika Solopova, Konrad Rieck, David Schlangen, Sebastian Möller, Vera Schmitt  

**Link**: [PDF](https://arxiv.org/pdf/2601.07648)  

**Abstract**: Despite advances in Natural Language Generation (NLG), evaluation remains challenging. Although various new metrics and LLM-as-a-judge (LaaJ) methods are proposed, human judgment persists as the gold standard. To systematically review how NLG evaluation has evolved, we employ an automatic information extraction scheme to gather key information from NLG papers, focusing on different evaluation methods (metrics, LaaJ and human evaluation). With extracted metadata from 14,171 papers across four major conferences (ACL, EMNLP, NAACL, and INLG) over the past six years, we reveal several critical findings: (1) Task Divergence: While Dialogue Generation demonstrates a rapid shift toward LaaJ (>40% in 2025), Machine Translation remains locked into n-gram metrics, and Question Answering exhibits a substantial decline in the proportion of studies conducting human evaluation. (2) Metric Inertia: Despite the development of semantic metrics, general-purpose metrics (e.g., BLEU, ROUGE) continue to be widely used across tasks without empirical justification, often lacking the discriminative power to distinguish between specific quality criteria. (3) Human-LaaJ Divergence: Our association analysis challenges the assumption that LLMs act as mere proxies for humans; LaaJ and human evaluations prioritize very different signals, and explicit validation is scarce (<8% of papers comparing the two), with only moderate to low correlation. Based on these observations, we derive practical recommendations to improve the rigor of future NLG evaluation. 

---
# PlaM: Training-Free Plateau-Guided Model Merging for Better Visual Grounding in MLLMs 

**Authors**: Zijing Wang, Yongkang Liu, Mingyang Wang, Ercong Nie, Deyuan Chen, Zhengjie Zhao, Shi Feng, Daling Wang, Xiaocui Yang, Yifei Zhang, Hinrich Schütze  

**Link**: [PDF](https://arxiv.org/pdf/2601.07645)  

**Abstract**: Multimodal Large Language Models (MLLMs) rely on strong linguistic reasoning inherited from their base language models. However, multimodal instruction fine-tuning paradoxically degrades this text's reasoning capability, undermining multimodal performance. To address this issue, we propose a training-free framework to mitigate this degradation. Through layer-wise vision token masking, we reveal a common three-stage pattern in multimodal large language models: early-modal separation, mid-modal alignment, and late-modal degradation. By analyzing the behavior of MLLMs at different stages, we propose a plateau-guided model merging method that selectively injects base language model parameters into MLLMs. Experimental results based on five MLLMs on nine benchmarks demonstrate the effectiveness of our method. Attention-based analysis further reveals that merging shifts attention from diffuse, scattered patterns to focused localization on task-relevant visual regions. Our repository is on this https URL. 

---
# Integrating Machine-Generated Short Descriptions into the Wikipedia Android App: A Pilot Deployment of Descartes 

**Authors**: Marija Šakota, Dmitry Brant, Cooltey Feng, Shay Nowick, Amal Ramadan, Robin Schoenbaechler, Joseph Seddon, Jazmin Tanner, Isaac Johnson, Robert West  

**Link**: [PDF](https://arxiv.org/pdf/2601.07631)  

**Abstract**: Short descriptions are a key part of the Wikipedia user experience, but their coverage remains uneven across languages and topics. In previous work, we introduced Descartes, a multilingual model for generating short descriptions. In this report, we present the results of a pilot deployment of Descartes in the Wikipedia Android app, where editors were offered suggestions based on outputs from Descartes while editing short descriptions. The experiment spanned 12 languages, with over 3,900 articles and 375 editors participating. Overall, 90% of accepted Descartes descriptions were rated at least 3 out of 5 in quality, and their average ratings were comparable to human-written ones. Editors adopted machine suggestions both directly and with modifications, while the rate of reverts and reports remained low. The pilot also revealed practical considerations for deployment, including latency, language-specific gaps, and the need for safeguards around sensitive topics. These results indicate that Descartes's short descriptions can support editors in reducing content gaps, provided that technical, design, and community guardrails are in place. 

---
# Proof of Time: A Benchmark for Evaluating Scientific Idea Judgments 

**Authors**: Bingyang Ye, Shan Chen, Jingxuan Tu, Chen Liu, Zidi Xiong, Samuel Schmidgall, Danielle S. Bitterman  

**Link**: [PDF](https://arxiv.org/pdf/2601.07606)  

**Abstract**: Large language models are increasingly being used to assess and forecast research ideas, yet we lack scalable ways to evaluate the quality of models' judgments about these scientific ideas. Towards this goal, we introduce PoT, a semi-verifiable benchmarking framework that links scientific idea judgments to downstream signals that become observable later (e.g., citations and shifts in researchers' agendas). PoT freezes a pre-cutoff snapshot of evidence in an offline sandbox and asks models to forecast post-cutoff outcomes, enabling verifiable evaluation when ground truth arrives, scalable benchmarking without exhaustive expert annotation, and analysis of human-model misalignment against signals such as peer-review awards. In addition, PoT provides a controlled testbed for agent-based research judgments that evaluate scientific ideas, comparing tool-using agents to non-agent baselines under prompt ablations and budget scaling. Across 30,000+ instances spanning four benchmark domains, we find that, compared with non-agent baselines, higher interaction budgets generally improve agent performance, while the benefit of tool use is strongly task-dependent. By combining time-partitioned, future-verifiable targets with an offline sandbox for tool use, PoT supports scalable evaluation of agents on future-facing scientific idea judgment tasks. 

---
# ES-Mem: Event Segmentation-Based Memory for Long-Term Dialogue Agents 

**Authors**: Huhai Zou, Tianhao Sun, Chuanjiang He, Yu Tian, Zhenyang Li, Li Jin, Nayu Liu, Jiang Zhong, Kaiwen Wei  

**Link**: [PDF](https://arxiv.org/pdf/2601.07582)  

**Abstract**: Memory is critical for dialogue agents to maintain coherence and enable continuous adaptation in long-term interactions. While existing memory mechanisms offer basic storage and retrieval capabilities, they are hindered by two primary limitations: (1) rigid memory granularity often disrupts semantic integrity, resulting in fragmented and incoherent memory units; (2) prevalent flat retrieval paradigms rely solely on surface-level semantic similarity, neglecting the structural cues of discourse required to navigate and locate specific episodic contexts. To mitigate these limitations, drawing inspiration from Event Segmentation Theory, we propose ES-Mem, a framework incorporating two core components: (1) a dynamic event segmentation module that partitions long-term interactions into semantically coherent events with distinct boundaries; (2) a hierarchical memory architecture that constructs multi-layered memories and leverages boundary semantics to anchor specific episodic memory for precise context localization. Evaluations on two memory benchmarks demonstrate that ES-Mem yields consistent performance gains over baseline methods. Furthermore, the proposed event segmentation module exhibits robust applicability on dialogue segmentation datasets. 

---
# A Unified Framework for Emotion Recognition and Sentiment Analysis via Expert-Guided Multimodal Fusion with Large Language Models 

**Authors**: Jiaqi Qiao, Xiujuan Xu, Xinran Li, Yu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.07565)  

**Abstract**: Multimodal emotion understanding requires effective integration of text, audio, and visual modalities for both discrete emotion recognition and continuous sentiment analysis. We present EGMF, a unified framework combining expert-guided multimodal fusion with large language models. Our approach features three specialized expert networks--a fine-grained local expert for subtle emotional nuances, a semantic correlation expert for cross-modal relationships, and a global context expert for long-range dependencies--adaptively integrated through hierarchical dynamic gating for context-aware feature selection. Enhanced multimodal representations are integrated with LLMs via pseudo token injection and prompt-based conditioning, enabling a single generative framework to handle both classification and regression through natural language generation. We employ LoRA fine-tuning for computational efficiency. Experiments on bilingual benchmarks (MELD, CHERMA, MOSEI, SIMS-V2) demonstrate consistent improvements over state-of-the-art methods, with superior cross-lingual robustness revealing universal patterns in multimodal emotional expressions across English and Chinese. We will release the source code publicly. 

---
# From RAG to Agentic RAG for Faithful Islamic Question Answering 

**Authors**: Gagan Bhatia, Hamdy Mubarak, Mustafa Jarrar, George Mikros, Fadi Zaraket, Mahmoud Alhirthani, Mutaz Al-Khatib, Logan Cochrane, Kareem Darwish, Rashid Yahiaoui, Firoj Alam  

**Link**: [PDF](https://arxiv.org/pdf/2601.07528)  

**Abstract**: LLMs are increasingly used for Islamic question answering, where ungrounded responses may carry serious religious consequences. Yet standard MCQ/MRC-style evaluations do not capture key real-world failure modes, notably free-form hallucinations and whether models appropriately abstain when evidence is lacking. To shed a light on this aspect we introduce ISLAMICFAITHQA, a 3,810-item bilingual (Arabic/English) generative benchmark with atomic single-gold answers, which enables direct measurement of hallucination and abstention. We additionally developed an end-to-end grounded Islamic modelling suite consisting of (i) 25K Arabic text-grounded SFT reasoning pairs, (ii) 5K bilingual preference samples for reward-guided alignment, and (iii) a verse-level Qur'an retrieval corpus of $\sim$6k atomic verses (ayat). Building on these resources, we develop an agentic Quran-grounding framework (agentic RAG) that uses structured tool calls for iterative evidence seeking and answer revision. Experiments across Arabic-centric and multilingual LLMs show that retrieval improves correctness and that agentic RAG yields the largest gains beyond standard RAG, achieving state-of-the-art performance and stronger Arabic-English robustness even with a small model (i.e., Qwen3 4B). We will make the experimental resources and datasets publicly available for the community. 

---
# Thinking Before Constraining: A Unified Decoding Framework for Large Language Models 

**Authors**: Ngoc Trinh Hung Nguyen, Alonso Silva, Laith Zumot, Liubov Tupikina, Armen Aghasaryan, Mehwish Alam  

**Link**: [PDF](https://arxiv.org/pdf/2601.07525)  

**Abstract**: Natural generation allows Language Models (LMs) to produce free-form responses with rich reasoning, but the lack of guaranteed structure makes outputs difficult to parse or verify. Structured generation, or constrained decoding, addresses this drawback by producing content in standardized formats such as JSON, ensuring consistency and guaranteed-parsable outputs, but it can inadvertently restrict the model's reasoning capabilities. In this work, we propose a simple approach that combines the advantages of both natural and structured generation. By allowing LLMs to reason freely until specific trigger tokens are generated, and then switching to structured generation, our method preserves the expressive power of natural language reasoning while ensuring the reliability of structured outputs. We further evaluate our approach on several datasets, covering both classification and reasoning tasks, to demonstrate its effectiveness, achieving a substantial gain of up to 27% in accuracy compared to natural generation, while requiring only a small overhead of 10-20 extra tokens. 

---
# Controlling Multimodal Conversational Agents with Coverage-Enhanced Latent Actions 

**Authors**: Yongqi Li, Hao Lang, Tieyun Qian, Yongbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.07516)  

**Abstract**: Vision-language models are increasingly employed as multimodal conversational agents (MCAs) for diverse conversational tasks. Recently, reinforcement learning (RL) has been widely explored for adapting MCAs to various human-AI interaction scenarios. Despite showing great enhancement in generalization performance, fine-tuning MCAs via RL still faces challenges in handling the extremely large text token space. To address this, we learn a compact latent action space for RL fine-tuning instead. Specifically, we adopt the learning from observation mechanism to construct the codebook for the latent action space, where future observations are leveraged to estimate current latent actions that could further be used to reconstruct future observations. However, the scarcity of paired image-text data hinders learning a codebook with sufficient coverage. Thus, we leverage both paired image-text data and text-only data to construct the latent action space, using a cross-modal projector for transforming text embeddings into image-text embeddings. We initialize the cross-modal projector on paired image-text data, and further train it on massive text-only data with a novel cycle consistency loss to enhance its robustness. We show that our latent action based method outperforms competitive baselines on two conversation tasks across various RL algorithms. 

---
# High-Rank Structured Modulation for Parameter-Efficient Fine-Tuning 

**Authors**: Yongkang Liu, Xing Li, Mengjie Zhao, Shanru Zhang, Zijing Wang, Qian Li, Shi Feng, Feiliang Ren, Daling Wang, Hinrich Schütze  

**Link**: [PDF](https://arxiv.org/pdf/2601.07507)  

**Abstract**: As the number of model parameters increases, parameter-efficient fine-tuning (PEFT) has become the go-to choice for tailoring pre-trained large language models. Low-rank Adaptation (LoRA) uses a low-rank update method to simulate full parameter fine-tuning, which is widely used to reduce resource requirements. However, decreasing the rank encounters challenges with limited representational capacity when compared to full parameter fine-tuning. We present \textbf{SMoA}, a high-rank \textbf{S}tructured \textbf{MO}dulation \textbf{A}dapter that uses fewer trainable parameters while maintaining a higher rank, thereby improving the model's representational capacity and offering improved performance potential. The core idea is to freeze the original pretrained weights and selectively amplify or suppress important features of the original weights across multiple subspaces. The subspace mechanism provides an efficient way to increase the capacity and complexity of a model. We conduct both theoretical analyses and empirical studies on various tasks. Experiment results show that SMoA outperforms LoRA and its variants on 10 tasks, with extensive ablation studies validating its effectiveness. 

---
# Judging Against the Reference: Uncovering Knowledge-Driven Failures in LLM-Judges on QA Evaluation 

**Authors**: Dongryeol Lee, Yerin Hwang, Taegwan Kang, Minwoo Lee, Younhyung Chae, Kyomin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2601.07506)  

**Abstract**: While large language models (LLMs) are increasingly used as automatic judges for question answering (QA) and other reference-conditioned evaluation tasks, little is known about their ability to adhere to a provided reference. We identify a critical failure mode of such reference-based LLM QA evaluation: when the provided reference conflicts with the judge model's parametric knowledge, the resulting scores become unreliable, substantially degrading evaluation fidelity. To study this phenomenon systematically, we introduce a controlled swapped-reference QA framework that induces reference-belief conflicts. Specifically, we replace the reference answer with an incorrect entity and construct diverse pairings of original and swapped references with correspondingly aligned candidate answers. Surprisingly, grading reliability drops sharply under swapped references across a broad set of judge models. We empirically show that this vulnerability is driven by judges' over-reliance on parametric knowledge, leading judges to disregard the given reference under conflict. Finally, we find that this failure persists under common prompt-based mitigation strategies, highlighting a fundamental limitation of LLM-as-a-judge evaluation and motivating reference-based protocols that enforce stronger adherence to the provided reference. 

---
# KALE: Enhancing Knowledge Manipulation in Large Language Models via Knowledge-aware Learning 

**Authors**: Qitan Lv, Tianyu Liu, Qiaosheng Zhang, Xingcheng Xu, Chaochao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2601.07430)  

**Abstract**: Despite the impressive performance of large language models (LLMs) pretrained on vast knowledge corpora, advancing their knowledge manipulation-the ability to effectively recall, reason, and transfer relevant knowledge-remains challenging. Existing methods mainly leverage Supervised Fine-Tuning (SFT) on labeled datasets to enhance LLMs' knowledge manipulation ability. However, we observe that SFT models still exhibit the known&incorrect phenomenon, where they explicitly possess relevant knowledge for a given question but fail to leverage it for correct answers. To address this challenge, we propose KALE (Knowledge-Aware LEarning)-a post-training framework that leverages knowledge graphs (KGs) to generate high-quality rationales and enhance LLMs' knowledge manipulation ability. Specifically, KALE first introduces a Knowledge-Induced (KI) data synthesis method that efficiently extracts multi-hop reasoning paths from KGs to generate high-quality rationales for question-answer pairs. Then, KALE employs a Knowledge-Aware (KA) fine-tuning paradigm that enhances knowledge manipulation by internalizing rationale-guided reasoning through minimizing the KL divergence between predictions with and without rationales. Extensive experiments on eight popular benchmarks across six different LLMs demonstrate the effectiveness of KALE, achieving accuracy improvements of up to 11.72% and an average of 4.18%. 

---
# SAD: A Large-Scale Strategic Argumentative Dialogue Dataset 

**Authors**: Yongkang Liu, Jiayang Yu, Mingyang Wang, Yiqun Zhang, Ercong Nie, Shi Feng, Daling Wang, Kaisong Song, Hinrich Schütze  

**Link**: [PDF](https://arxiv.org/pdf/2601.07423)  

**Abstract**: Argumentation generation has attracted substantial research interest due to its central role in human reasoning and decision-making. However, most existing argumentative corpora focus on non-interactive, single-turn settings, either generating arguments from a given topic or refuting an existing argument. In practice, however, argumentation is often realized as multi-turn dialogue, where speakers defend their stances and employ diverse argumentative strategies to strengthen persuasiveness. To support deeper modeling of argumentation dialogue, we present the first large-scale \textbf{S}trategic \textbf{A}rgumentative \textbf{D}ialogue dataset, SAD, consisting of 392,822 examples. Grounded in argumentation theories, we annotate each utterance with five strategy types, allowing multiple strategies per utterance. Unlike prior datasets, SAD requires models to generate contextually appropriate arguments conditioned on the dialogue history, a specified stance on the topic, and targeted argumentation strategies. We further benchmark a range of pretrained generative models on SAD and present in-depth analysis of strategy usage patterns in argumentation. 

---
# Two Pathways to Truthfulness: On the Intrinsic Encoding of LLM Hallucinations 

**Authors**: Wen Luo, Guangyue Peng, Wei Li, Shaohang Wei, Feifan Song, Liang Wang, Nan Yang, Xingxing Zhang, Jing Jin, Furu Wei, Houfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.07422)  

**Abstract**: Despite their impressive capabilities, large language models (LLMs) frequently generate hallucinations. Previous work shows that their internal states encode rich signals of truthfulness, yet the origins and mechanisms of these signals remain unclear. In this paper, we demonstrate that truthfulness cues arise from two distinct information pathways: (1) a Question-Anchored pathway that depends on question-answer information flow, and (2) an Answer-Anchored pathway that derives self-contained evidence from the generated answer itself. First, we validate and disentangle these pathways through attention knockout and token patching. Afterwards, we uncover notable and intriguing properties of these two mechanisms. Further experiments reveal that (1) the two mechanisms are closely associated with LLM knowledge boundaries; and (2) internal representations are aware of their distinctions. Finally, building on these insightful findings, two applications are proposed to enhance hallucination detection performance. Overall, our work provides new insight into how LLMs internally encode truthfulness, offering directions for more reliable and self-aware generative systems. 

---
# Outcome-Grounded Advantage Reshaping for Fine-Grained Credit Assignment in Mathematical Reasoning 

**Authors**: Ziheng Li, Liu Kang, Feng Xiao, Luxi Xing, Qingyi Si, Zhuoran Li, Weikang Gong, Deqing Yang, Yanghua Xiao, Hongcheng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2601.07408)  

**Abstract**: Group Relative Policy Optimization (GRPO) has emerged as a promising critic-free reinforcement learning paradigm for reasoning tasks. However, standard GRPO employs a coarse-grained credit assignment mechanism that propagates group-level rewards uniformly to to every token in a sequence, neglecting the varying contribution of individual reasoning steps. We address this limitation by introducing Outcome-grounded Advantage Reshaping (OAR), a fine-grained credit assignment mechanism that redistributes advantages based on how much each token influences the model's final answer. We instantiate OAR via two complementary strategies: (1) OAR-P, which estimates outcome sensitivity through counterfactual token perturbations, serving as a high-fidelity attribution signal; (2) OAR-G, which uses an input-gradient sensitivity proxy to approximate the influence signal with a single backward pass. These importance signals are integrated with a conservative Bi-Level advantage reshaping scheme that suppresses low-impact tokens and boosts pivotal ones while preserving the overall advantage mass. Empirical results on extensive mathematical reasoning benchmarks demonstrate that while OAR-P sets the performance upper bound, OAR-G achieves comparable gains with negligible computational overhead, both significantly outperforming a strong GRPO baseline, pushing the boundaries of critic-free LLM reasoning. 

---
# GROKE: Vision-Free Navigation Instruction Evaluation via Graph Reasoning on OpenStreetMap 

**Authors**: Farzad Shami, Subhrasankha Dey, Nico Van de Weghe, Henrikki Tenkanen  

**Link**: [PDF](https://arxiv.org/pdf/2601.07375)  

**Abstract**: The evaluation of navigation instructions remains a persistent challenge in Vision-and-Language Navigation (VLN) research. Traditional reference-based metrics such as BLEU and ROUGE fail to capture the functional utility of spatial directives, specifically whether an instruction successfully guides a navigator to the intended destination. Although existing VLN agents could serve as evaluators, their reliance on high-fidelity visual simulators introduces licensing constraints and computational costs, and perception errors further confound linguistic quality assessment. This paper introduces GROKE(Graph-based Reasoning over OSM Knowledge for instruction Evaluation), a vision-free training-free hierarchical LLM-based framework for evaluating navigation instructions using OpenStreetMap data. Through systematic ablation studies, we demonstrate that structured JSON and textual formats for spatial information substantially outperform grid-based and visual graph representations. Our hierarchical architecture combines sub-instruction planning with topological graph navigation, reducing navigation error by 68.5% compared to heuristic and sampling baselines on the Map2Seq dataset. The agent's execution success, trajectory fidelity, and decision patterns serve as proxy metrics for functional navigability given OSM-visible landmarks and topology, establishing a scalable and interpretable evaluation paradigm without visual dependencies. Code and data are available at this https URL. 

---
# Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models 

**Authors**: Xin Cheng, Wangding Zeng, Damai Dai, Qinyu Chen, Bingxuan Wang, Zhenda Xie, Kezhao Huang, Xingkai Yu, Zhewen Hao, Yukun Li, Han Zhang, Huishuai Zhang, Dongyan Zhao, Wenfeng Liang  

**Link**: [PDF](https://arxiv.org/pdf/2601.07372)  

**Abstract**: While Mixture-of-Experts (MoE) scales capacity via conditional computation, Transformers lack a native primitive for knowledge lookup, forcing them to inefficiently simulate retrieval through computation. To address this, we introduce conditional memory as a complementary sparsity axis, instantiated via Engram, a module that modernizes classic $N$-gram embedding for O(1) lookup. By formulating the Sparsity Allocation problem, we uncover a U-shaped scaling law that optimizes the trade-off between neural computation (MoE) and static memory (Engram). Guided by this law, we scale Engram to 27B parameters, achieving superior performance over a strictly iso-parameter and iso-FLOPs MoE baseline. Most notably, while the memory module is expected to aid knowledge retrieval (e.g., MMLU +3.4; CMMLU +4.0), we observe even larger gains in general reasoning (e.g., BBH +5.0; ARC-Challenge +3.7) and code/math domains~(HumanEval +3.0; MATH +2.4). Mechanistic analyses reveal that Engram relieves the backbone's early layers from static reconstruction, effectively deepening the network for complex reasoning. Furthermore, by delegating local dependencies to lookups, it frees up attention capacity for global context, substantially boosting long-context retrieval (e.g., Multi-Query NIAH: 84.2 to 97.0). Finally, Engram establishes infrastructure-aware efficiency: its deterministic addressing enables runtime prefetching from host memory, incurring negligible overhead. We envision conditional memory as an indispensable modeling primitive for next-generation sparse models. 

---
# Interpretable Text Classification Applied to the Detection of LLM-generated Creative Writing 

**Authors**: Minerva Suvanto, Andrea McGlinchey, Mattias Wahde, Peter J Barclay  

**Link**: [PDF](https://arxiv.org/pdf/2601.07368)  

**Abstract**: We consider the problem of distinguishing human-written creative fiction (excerpts from novels) from similar text generated by an LLM. Our results show that, while human observers perform poorly (near chance levels) on this binary classification task, a variety of machine-learning models achieve accuracy in the range 0.93 - 0.98 over a previously unseen test set, even using only short samples and single-token (unigram) features. We therefore employ an inherently interpretable (linear) classifier (with a test accuracy of 0.98), in order to elucidate the underlying reasons for this high accuracy. In our analysis, we identify specific unigram features indicative of LLM-generated text, one of the most important being that the LLM tends to use a larger variety of synonyms, thereby skewing the probability distributions in a manner that is easy to detect for a machine learning classifier, yet very difficult for a human observer. Four additional explanation categories were also identified, namely, temporal drift, Americanisms, foreign language usage, and colloquialisms. As identification of the AI-generated text depends on a constellation of such features, the classification appears robust, and therefore not easy to circumvent by malicious actors intent on misrepresenting AI-generated text as human work. 

---
# Semantic Compression of LLM Instructions via Symbolic Metalanguages 

**Authors**: Ernst van Gassen  

**Link**: [PDF](https://arxiv.org/pdf/2601.07354)  

**Abstract**: We introduce MetaGlyph, a symbolic language for compressing prompts by encoding instructions as mathematical symbols rather than prose. Unlike systems requiring explicit decoding rules, MetaGlyph uses symbols like $\in$ (membership) and $\Rightarrow$ (implication) that models already understand from their training data. We test whether these symbols work as ''instruction shortcuts'' that models can interpret without additional teaching.
We evaluate eight models across two dimensions relevant to practitioners: scale (3B-1T parameters) and accessibility (open-source for local deployment vs. proprietary APIs). MetaGlyph achieves 62-81% token reduction across all task types. For API-based deployments, this translates directly to cost savings; for local deployments, it reduces latency and memory pressure.
Results vary by model. Gemini 2.5 Flash achieves 75% semantic equivalence between symbolic and prose instructions on selection tasks, with 49.9% membership operator fidelity. Kimi K2 reaches 98.1% fidelity for implication ($\Rightarrow$) and achieves perfect (100%) accuracy on selection tasks with symbolic prompts. GPT-5.2 Chat shows the highest membership fidelity observed (91.3%), though with variable parse success across task types. Claude Haiku 4.5 achieves 100% parse success with 26% membership fidelity. Among mid-sized models, Qwen 2.5 7B shows 62% equivalence on extraction tasks. Mid-sized open-source models (7B-12B) show near-zero operator fidelity, suggesting a U-shaped relationship where sufficient scale overcomes instruction-tuning biases. 

---
# TALON: Confidence-Aware Speculative Decoding with Adaptive Token Trees 

**Authors**: Tianyu Liu, Qitan Lv, Yuhao Shen, Xiao Sun, Xiaoyan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.07353)  

**Abstract**: Speculative decoding (SD) has become a standard technique for accelerating LLM inference without sacrificing output quality. Recent advances in speculative decoding have shifted from sequential chain-based drafting to tree-structured generation, where the draft model constructs a tree of candidate tokens to explore multiple possible drafts in parallel. However, existing tree-based SD methods typically build a fixed-width, fixed-depth draft tree, which fails to adapt to the varying difficulty of tokens and contexts. As a result, the draft model cannot dynamically adjust the tree structure to early stop on difficult tokens and extend generation for simple ones. To address these challenges, we introduce TALON, a training-free, budget-driven adaptive tree expansion framework that can be plugged into existing tree-based methods. Unlike static methods, TALON constructs the draft tree iteratively until a fixed token budget is met, using a hybrid expansion strategy that adaptively allocates the node budget to each layer of the draft tree. This framework naturally shapes the draft tree into a "deep-and-narrow" form for deterministic contexts and a "shallow-and-wide" form for uncertain branches, effectively optimizing the trade-off between exploration width and generation depth under a given budget. Extensive experiments across 5 models and 6 datasets demonstrate that TALON consistently outperforms state-of-the-art EAGLE-3, achieving up to 5.16x end-to-end speedup over auto-regressive decoding. 

---
# Beyond Hard Masks: Progressive Token Evolution for Diffusion Language Models 

**Authors**: Linhao Zhong, Linyu Wu, Bozhen Fang, Tianjian Feng, Chenchen Jing, Wen Wang, Jiaheng Zhang, Hao Chen, Chunhua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2601.07351)  

**Abstract**: Diffusion Language Models (DLMs) offer a promising alternative for language modeling by enabling parallel decoding through iterative refinement. However, most DLMs rely on hard binary masking and discrete token assignments, which hinder the revision of early decisions and underutilize intermediate probabilistic representations. In this paper, we propose EvoToken-DLM, a novel diffusion-based language modeling approach that replaces hard binary masks with evolving soft token distributions. EvoToken-DLM enables a progressive transition from masked states to discrete outputs, supporting revisable decoding. To effectively support this evolution, we introduce continuous trajectory supervision, which aligns training objectives with iterative probabilistic updates. Extensive experiments across multiple benchmarks show that EvoToken-DLM consistently achieves superior performance, outperforming strong diffusion-based and masked DLM baselines. Project webpage: this https URL. 

---
# Reward Modeling from Natural Language Human Feedback 

**Authors**: Zongqi Wang, Rui Wang, Yuchuan Wu, Yiyao Yu, Pinyi Zhang, Shaoning Sun, Yujiu Yang, Yongbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.07349)  

**Abstract**: Reinforcement Learning with Verifiable reward (RLVR) on preference data has become the mainstream approach for training Generative Reward Models (GRMs). Typically in pairwise rewarding tasks, GRMs generate reasoning chains ending with critiques and preference labels, and RLVR then relies on the correctness of the preference labels as the training reward. However, in this paper, we demonstrate that such binary classification tasks make GRMs susceptible to guessing correct outcomes without sound critiques. Consequently, these spurious successes introduce substantial noise into the reward signal, thereby impairing the effectiveness of reinforcement learning. To address this issue, we propose Reward Modeling from Natural Language Human Feedback (RM-NLHF), which leverages natural language feedback to obtain process reward signals, thereby mitigating the problem of limited solution space inherent in binary tasks. Specifically, we compute the similarity between GRM-generated and human critiques as the training reward, which provides more accurate reward signals than outcome-only supervision. Additionally, considering that human critiques are difficult to scale up, we introduce Meta Reward Model (MetaRM) which learns to predict process reward from datasets with human critiques and then generalizes to data without human critiques. Experiments on multiple benchmarks demonstrate that our method consistently outperforms state-of-the-art GRMs trained with outcome-only reward, confirming the superiority of integrating natural language over binary human feedback as supervision. 

---
# Controlled Self-Evolution for Algorithmic Code Optimization 

**Authors**: Tu Hu, Ronghao Chen, Shuo Zhang, Jianghao Yin, Mou Xiao Feng, Jingping Liu, Shaolei Zhang, Wenqi Jiang, Yuqi Fang, Sen Hu, Yi Xu, Huacan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.07348)  

**Abstract**: Self-evolution methods enhance code generation through iterative "generate-verify-refine" cycles, yet existing approaches suffer from low exploration efficiency, failing to discover solutions with superior complexity within limited budgets. This inefficiency stems from initialization bias trapping evolution in poor solution regions, uncontrolled stochastic operations lacking feedback guidance, and insufficient experience utilization across this http URL address these bottlenecks, we propose Controlled Self-Evolution (CSE), which consists of three key components. Diversified Planning Initialization generates structurally distinct algorithmic strategies for broad solution space coverage. Genetic Evolution replaces stochastic operations with feedback-guided mechanisms, enabling targeted mutation and compositional crossover. Hierarchical Evolution Memory captures both successful and failed experiences at inter-task and intra-task this http URL on EffiBench-X demonstrate that CSE consistently outperforms all baselines across various LLM backbones. Furthermore, CSE achieves higher efficiency from early generations and maintains continuous improvement throughout evolution. Our code is publicly available at this https URL. 

---
# DiffER: Diffusion Entity-Relation Modeling for Reversal Curse in Diffusion Large Language Models 

**Authors**: Shaokai He, Kaiwen Wei, Xinyi Zeng, Xiang Chen, Xue Yang, Zhenyang Li, Jiang Zhong, Yu Tian  

**Link**: [PDF](https://arxiv.org/pdf/2601.07347)  

**Abstract**: The "reversal curse" refers to the phenomenon where large language models (LLMs) exhibit predominantly unidirectional behavior when processing logically bidirectional relationships. Prior work attributed this to autoregressive training -- predicting the next token inherently favors left-to-right information flow over genuine bidirectional knowledge associations. However, we observe that Diffusion LLMs (DLLMs), despite being trained bidirectionally, also suffer from the reversal curse. To investigate the root causes, we conduct systematic experiments on DLLMs and identify three key reasons: 1) entity fragmentation during training, 2) data asymmetry, and 3) missing entity relations. Motivated by the analysis of these reasons, we propose Diffusion Entity-Relation Modeling (DiffER), which addresses the reversal curse through entity-aware training and balanced data construction. Specifically, DiffER introduces whole-entity masking, which mitigates entity fragmentation by predicting complete entities in a single step. DiffER further employs distribution-symmetric and relation-enhanced data construction strategies to alleviate data asymmetry and missing relations. Extensive experiments demonstrate that DiffER effectively alleviates the reversal curse in Diffusion LLMs, offering new perspectives for future research. 

---
# Beyond Literal Mapping: Benchmarking and Improving Non-Literal Translation Evaluation 

**Authors**: Yanzhi Tian, Cunxiang Wang, Zeming Liu, Heyan Huang, Wenbo Yu, Dawei Song, Jie Tang, Yuhang Guo  

**Link**: [PDF](https://arxiv.org/pdf/2601.07338)  

**Abstract**: Large Language Models (LLMs) have significantly advanced Machine Translation (MT), applying them to linguistically complex domains-such as Social Network Services, literature etc. In these scenarios, translations often require handling non-literal expressions, leading to the inaccuracy of MT metrics. To systematically investigate the reliability of MT metrics, we first curate a meta-evaluation dataset focused on non-literal translations, namely MENT. MENT encompasses four non-literal translation domains and features source sentences paired with translations from diverse MT systems, with 7,530 human-annotated scores on translation quality. Experimental results reveal the inaccuracies of traditional MT metrics and the limitations of LLM-as-a-Judge, particularly the knowledge cutoff and score inconsistency problem. To mitigate these limitations, we propose RATE, a novel agentic translation evaluation framework, centered by a reflective Core Agent that dynamically invokes specialized sub-agents. Experimental results indicate the efficacy of RATE, achieving an improvement of at least 3.2 meta score compared with current metrics. Further experiments demonstrate the robustness of RATE to general-domain MT evaluation. Code and dataset are available at: this https URL. 

---
# BayesRAG: Probabilistic Mutual Evidence Corroboration for Multimodal Retrieval-Augmented Generation 

**Authors**: Xuan Li, Yining Wang, Haocai Luo, Shengping Liu, Jerry Liang, Ying Fu, Weihuang, Jun Yu, Junnan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2601.07329)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become a pivotal paradigm for Large Language Models (LLMs), yet current approaches struggle with visually rich documents by treating text and images as isolated retrieval targets. Existing methods relying solely on cosine similarity often fail to capture the semantic reinforcement provided by cross-modal alignment and layout-induced coherence. To address these limitations, we propose BayesRAG, a novel multimodal retrieval framework grounded in Bayesian inference and Dempster-Shafer evidence theory. Unlike traditional approaches that rank candidates strictly by similarity, BayesRAG models the intrinsic consistency of retrieved candidates across modalities as probabilistic evidence to refine retrieval confidence. Specifically, our method computes the posterior association probability for combinations of multimodal retrieval results, prioritizing text-image pairs that mutually corroborate each other in terms of both semantics and layout. Extensive experiments demonstrate that BayesRAG significantly outperforms state-of-the-art (SOTA) methods on challenging multimodal benchmarks. This study establishes a new paradigm for multimodal retrieval fusion that effectively resolves the isolation of heterogeneous modalities through an evidence fusion mechanism and enhances the robustness of retrieval outcomes. Our code is available at this https URL. 

---
# How to predict creativity ratings from written narratives: A comparison of co-occurrence and textual forma mentis networks 

**Authors**: Roberto Passaro, Edith Haim, Massimo Stella  

**Link**: [PDF](https://arxiv.org/pdf/2601.07327)  

**Abstract**: This tutorial paper provides a step-by-step workflow for building and analysing semantic networks from short creative texts. We introduce and compare two widely used text-to-network approaches: word co-occurrence networks and textual forma mentis networks (TFMNs). We also demonstrate how they can be used in machine learning to predict human creativity ratings. Using a corpus of 1029 short stories, we guide readers through text preprocessing, network construction, feature extraction (structural measures, spreading-activation indices, and emotion scores), and application of regression models. We evaluate how network-construction choices influence both network topology and predictive performance. Across all modelling settings, TFMNs consistently outperformed co-occurrence networks through lower prediction errors (best MAE = 0.581 for TFMN, vs 0.592 for co-occurrence with window size 3). Network-structural features dominated predictive performance (MAE = 0.591 for TFMN), whereas emotion features performed worse (MAE = 0.711 for TFMN) and spreading-activation measures contributed little (MAE = 0.788 for TFMN). This paper offers practical guidance for researchers interested in applying network-based methods for cognitive fields like creativity research. we show when syntactic networks are preferable to surface co-occurrence models, and provide an open, reproducible workflow accessible to newcomers in the field, while also offering deeper methodological insight for experienced researchers. 

---
# Mitrasamgraha: A Comprehensive Classical Sanskrit Machine Translation Dataset 

**Authors**: Sebastian Nehrdich, David Allport, Sven Sellmer, Jivnesh Sandhan, Manoj Balaji Jagadeeshan, Pawan Goyal, Sujeet Kumar, Kurt Keutzer  

**Link**: [PDF](https://arxiv.org/pdf/2601.07314)  

**Abstract**: While machine translation is regarded as a "solved problem" for many high-resource languages, close analysis quickly reveals that this is not the case for content that shows challenges such as poetic language, philosophical concepts, multi-layered metaphorical expressions, and more. Sanskrit literature is a prime example of this, as it combines a large number of such challenges in addition to inherent linguistic features like sandhi, compounding, and heavy morphology, which further complicate NLP downstream tasks. It spans multiple millennia of text production time as well as a large breadth of different domains, ranging from ritual formulas via epic narratives, philosophical treatises, poetic verses up to scientific material. As of now, there is a strong lack of publicly available resources that cover these different domains and temporal layers of Sanskrit. We therefore introduce Mitrasamgraha, a high-quality Sanskrit-to-English machine translation dataset consisting of 391,548 bitext pairs, more than four times larger than the largest previously available Sanskrit dataset Itih=asa. It covers a time period of more than three millennia and a broad range of historical Sanskrit domains. In contrast to web-crawled datasets, the temporal and domain annotation of this dataset enables fine-grained study of domain and time period effects on MT performance. We also release a validation set consisting of 5,587 and a test set consisting of 5,552 post-corrected bitext pairs. We conduct experiments benchmarking commercial and open models on this dataset and fine-tune NLLB and Gemma models on the dataset, showing significant improvements, while still recognizing significant challenges in the translation of complex compounds, philosophical concepts, and multi-layered metaphors. We also analyze how in-context learning on this dataset impacts the performance of commercial models 

---
# PsyCLIENT: Client Simulation via Conversational Trajectory Modeling for Trainee Practice and Model Evaluation in Mental Health Counseling 

**Authors**: Huachuan Qiu, Zhaoming Chen, Yuqian Chen, Yuan Xie, Yu Lu, Zhenzhong Lan  

**Link**: [PDF](https://arxiv.org/pdf/2601.07312)  

**Abstract**: LLM-based client simulation has emerged as a promising tool for training novice counselors and evaluating automated counseling systems. However, existing client simulation approaches face three key challenges: (1) limited diversity and realism in client profiles, (2) the lack of a principled framework for modeling realistic client behaviors, and (3) a scarcity in Chinese-language settings. To address these limitations, we propose PsyCLIENT, a novel simulation framework grounded in conversational trajectory modeling. By conditioning LLM generation on predefined real-world trajectories that incorporate explicit behavior labels and content constraints, our approach ensures diverse and realistic interactions. We further introduce PsyCLIENT-CP, the first open-source Chinese client profile dataset, covering 60 distinct counseling topics. Comprehensive evaluations involving licensed professional counselors demonstrate that PsyCLIENT significantly outperforms baselines in terms of authenticity and training effectiveness. Notably, the simulated clients are nearly indistinguishable from human clients, achieving an about 95\% expert confusion rate in discrimination tasks. These findings indicate that conversational trajectory modeling effectively bridges the gap between theoretical client profiles and dynamic, realistic simulations, offering a robust solution for mental health education and research. Code and data will be released to facilitate future research in mental health counseling. 

---
# ReasonTabQA: A Comprehensive Benchmark for Table Question Answering from Real World Industrial Scenarios 

**Authors**: Changzai Pan, Jie Zhang, Kaiwen Wei, Chenshuo Pan, Yu Zhao, Jingwang Huang, Jian Yang, Zhenhe Wu, Haoyang Zeng, Xiaoyan Gu, Weichao Sun, Yanbo Zhai, Yujie Mao, Zhuoru Jiang, Jiang Zhong, Shuangyong Song, Yongxiang Li, Zhongjiang He  

**Link**: [PDF](https://arxiv.org/pdf/2601.07280)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have significantly catalyzed table-based question answering (TableQA). However, existing TableQA benchmarks often overlook the intricacies of industrial scenarios, which are characterized by multi-table structures, nested headers, and massive scales. These environments demand robust table reasoning through deep structured inference, presenting a significant challenge that remains inadequately addressed by current methodologies. To bridge this gap, we present ReasonTabQA, a large-scale bilingual benchmark encompassing 1,932 tables across 30 industry domains such as energy and automotive. ReasonTabQA provides high-quality annotations for both final answers and explicit reasoning chains, supporting both thinking and no-thinking paradigms. Furthermore, we introduce TabCodeRL, a reinforcement learning method that leverages table-aware verifiable rewards to guide the generation of logical reasoning paths. Extensive experiments on ReasonTabQA and 4 TableQA datasets demonstrate that while TabCodeRL yields substantial performance gains on open-source LLMs, the persistent performance gap on ReasonTabQA underscores the inherent complexity of real-world industrial TableQA. 

---
# Towards Comprehensive Semantic Speech Embeddings for Chinese Dialects 

**Authors**: Kalvin Chang, Yiwen Shao, Jiahong Li, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.07274)  

**Abstract**: Despite having hundreds of millions of speakers, Chinese dialects lag behind Mandarin in speech and language technologies. Most varieties are primarily spoken, making dialect-to-Mandarin speech-LLMs (large language models) more practical than dialect LLMs. Building dialect-to-Mandarin speech-LLMs requires speech representations with cross-dialect semantic alignment between Chinese dialects and Mandarin. In this paper, we achieve such a cross-dialect semantic alignment by training a speech encoder with ASR (automatic speech recognition)-only data, as demonstrated by speech-to-speech retrieval on a new benchmark of spoken Chinese varieties that we contribute. Our speech encoder further demonstrates state-of-the-art ASR performance on Chinese dialects. Together, our Chinese dialect benchmark, semantically aligned speech representations, and speech-to-speech retrieval evaluation lay the groundwork for future Chinese dialect speech-LLMs. We release the benchmark at this https URL. 

---
# Document-Level Zero-Shot Relation Extraction with Entity Side Information 

**Authors**: Mohan Raj Chanthran, Soon Lay Ki, Ong Huey Fang, Bhawani Selvaretnam  

**Link**: [PDF](https://arxiv.org/pdf/2601.07271)  

**Abstract**: Document-Level Zero-Shot Relation Extraction (DocZSRE) aims to predict unseen relation labels in text documents without prior training on specific relations. Existing approaches rely on Large Language Models (LLMs) to generate synthetic data for unseen labels, which poses challenges for low-resource languages like Malaysian English. These challenges include the incorporation of local linguistic nuances and the risk of factual inaccuracies in LLM-generated data. This paper introduces Document-Level Zero-Shot Relation Extraction with Entity Side Information (DocZSRE-SI) to address limitations in the existing DocZSRE approach. The DocZSRE-SI framework leverages Entity Side Information, such as Entity Mention Descriptions and Entity Mention Hypernyms, to perform ZSRE without depending on LLM-generated synthetic data. The proposed low-complexity model achieves an average improvement of 11.6% in the macro F1-Score compared to baseline models and existing benchmarks. By utilizing Entity Side Information, DocZSRE-SI offers a robust and efficient alternative to error-prone, LLM-based methods, demonstrating significant advancements in handling low-resource languages and linguistic diversity in relation extraction tasks. This research provides a scalable and reliable solution for ZSRE, particularly in contexts like Malaysian English news articles, where traditional LLM-based approaches fall short. 

---
# The Confidence Dichotomy: Analyzing and Mitigating Miscalibration in Tool-Use Agents 

**Authors**: Weihao Xuan, Qingcheng Zeng, Heli Qi, Yunze Xiao, Junjue Wang, Naoto Yokoya  

**Link**: [PDF](https://arxiv.org/pdf/2601.07264)  

**Abstract**: Autonomous agents based on large language models (LLMs) are rapidly evolving to handle multi-turn tasks, but ensuring their trustworthiness remains a critical challenge. A fundamental pillar of this trustworthiness is calibration, which refers to an agent's ability to express confidence that reliably reflects its actual performance. While calibration is well-established for static models, its dynamics in tool-integrated agentic workflows remain underexplored. In this work, we systematically investigate verbalized calibration in tool-use agents, revealing a fundamental confidence dichotomy driven by tool type. Specifically, our pilot study identifies that evidence tools (e.g., web search) systematically induce severe overconfidence due to inherent noise in retrieved information, while verification tools (e.g., code interpreters) can ground reasoning through deterministic feedback and mitigate miscalibration. To robustly improve calibration across tool types, we propose a reinforcement learning (RL) fine-tuning framework that jointly optimizes task accuracy and calibration, supported by a holistic benchmark of reward designs. We demonstrate that our trained agents not only achieve superior calibration but also exhibit robust generalization from local training environments to noisy web settings and to distinct domains such as mathematical reasoning. Our results highlight the necessity of domain-specific calibration strategies for tool-use agents. More broadly, this work establishes a foundation for building self-aware agents that can reliably communicate uncertainty in high-stakes, real-world deployments. 

---
# ActiShade: Activating Overshadowed Knowledge to Guide Multi-Hop Reasoning in Large Language Models 

**Authors**: Huipeng Ma, Luan Zhang, Dandan Song, Linmei Hu, Yuhang Tian, Jun Yang, Changzhi Zhou, Chenhao Li, Yizhou Jin, Xudong Li, Meng Lin, Mingxing Zhang, Shuhao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.07260)  

**Abstract**: In multi-hop reasoning, multi-round retrieval-augmented generation (RAG) methods typically rely on LLM-generated content as the retrieval query. However, these approaches are inherently vulnerable to knowledge overshadowing - a phenomenon where critical information is overshadowed during generation. As a result, the LLM-generated content may be incomplete or inaccurate, leading to irrelevant retrieval and causing error accumulation during the iteration process. To address this challenge, we propose ActiShade, which detects and activates overshadowed knowledge to guide large language models (LLMs) in multi-hop reasoning. Specifically, ActiShade iteratively detects the overshadowed keyphrase in the given query, retrieves documents relevant to both the query and the overshadowed keyphrase, and generates a new query based on the retrieved documents to guide the next-round iteration. By supplementing the overshadowed knowledge during the formulation of next-round queries while minimizing the introduction of irrelevant noise, ActiShade reduces the error accumulation caused by knowledge overshadowing. Extensive experiments show that ActiShade outperforms existing methods across multiple datasets and LLMs. 

---
# The Roots of Performance Disparity in Multilingual Language Models: Intrinsic Modeling Difficulty or Design Choices? 

**Authors**: Chen Shani, Yuval Reif, Nathan Roll, Dan Jurafsky, Ekaterina Shutova  

**Link**: [PDF](https://arxiv.org/pdf/2601.07220)  

**Abstract**: Multilingual language models (LMs) promise broader NLP access, yet current systems deliver uneven performance across the world's languages. This survey examines why these gaps persist and whether they reflect intrinsic linguistic difficulty or modeling artifacts. We organize the literature around two questions: do linguistic disparities arise from representation and allocation choices (e.g., tokenization, encoding, data exposure, parameter sharing) rather than inherent complexity; and which design choices mitigate inequities across typologically diverse languages. We review linguistic features, such as orthography, morphology, lexical diversity, syntax, information density, and typological distance, linking each to concrete modeling mechanisms. Gaps often shrink when segmentation, encoding, and data exposure are normalized, suggesting much apparent difficulty stems from current modeling choices. We synthesize these insights into design recommendations for tokenization, sampling, architectures, and evaluation to support more balanced multilingual LMs. 

---
# MI-PRUN: Optimize Large Language Model Pruning via Mutual Information 

**Authors**: Hao Zhang, Zhibin Zhang, Guangxin Wu, He Chen, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.07212)  

**Abstract**: Large Language Models (LLMs) have become indispensable across various domains, but this comes at the cost of substantial computational and memory resources. Model pruning addresses this by removing redundant components from models. In particular, block pruning can achieve significant compression and inference acceleration. However, existing block pruning methods are often unstable and struggle to attain globally optimal solutions. In this paper, we propose a mutual information based pruning method MI-PRUN for LLMs. Specifically, we leverages mutual information to identify redundant blocks by evaluating transitions in hidden states. Additionally, we incorporate the Data Processing Inequality (DPI) to reveal the relationship between the importance of entire contiguous blocks and that of individual blocks. Moreover, we develop the Fast-Block-Select algorithm, which iteratively updates block combinations to achieve a globally optimal solution while significantly improving the efficiency. Extensive experiments across various models and datasets demonstrate the stability and effectiveness of our method. 

---
# Relink: Constructing Query-Driven Evidence Graph On-the-Fly for GraphRAG 

**Authors**: Manzong Huang, Chenyang Bu, Yi He, Xingrui Zhuo, Xindong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.07192)  

**Abstract**: Graph-based Retrieval-Augmented Generation (GraphRAG) mitigates hallucinations in Large Language Models (LLMs) by grounding them in structured knowledge. However, current GraphRAG methods are constrained by a prevailing \textit{build-then-reason} paradigm, which relies on a static, pre-constructed Knowledge Graph (KG). This paradigm faces two critical challenges. First, the KG's inherent incompleteness often breaks reasoning paths. Second, the graph's low signal-to-noise ratio introduces distractor facts, presenting query-relevant but misleading knowledge that disrupts the reasoning process.
To address these challenges, we argue for a \textit{reason-and-construct} paradigm and propose Relink, a framework that dynamically builds a query-specific evidence graph. To tackle incompleteness, \textbf{Relink} instantiates required facts from a latent relation pool derived from the original text corpus, repairing broken paths on the fly. To handle misleading or distractor facts, Relink employs a unified, query-aware evaluation strategy that jointly considers candidates from both the KG and latent relations, selecting those most useful for answering the query rather than relying on their pre-existence. This empowers Relink to actively discard distractor facts and construct the most faithful and precise evidence path for each query.
Extensive experiments on five Open-Domain Question Answering benchmarks show that Relink achieves significant average improvements of 5.4\% in EM and 5.2\% in F1 over leading GraphRAG baselines, demonstrating the superiority of our proposed framework. 

---
# Structured Reasoning for Large Language Models 

**Authors**: Jinyi Han, Zixiang Di, Zishang Jiang, Ying Liao, Jiaqing Liang, Yongqi Wang, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2601.07180)  

**Abstract**: Large language models (LLMs) achieve strong performance by generating long chains of thought, but longer traces always introduce redundant or ineffective reasoning steps. One typical behavior is that they often perform unnecessary verification and revisions even if they have reached the correct answers. This limitation stems from the unstructured nature of reasoning trajectories and the lack of targeted supervision for critical reasoning abilities. To address this, we propose Structured Reasoning (SCR), a framework that decouples reasoning trajectories into explicit, evaluable, and trainable components. We mainly implement SCR using a Generate-Verify-Revise paradigm. Specifically, we construct structured training data and apply Dynamic Termination Supervision to guide the model in deciding when to terminate reasoning. To avoid interference between learning signals for different reasoning abilities, we adopt a progressive two-stage reinforcement learning strategy: the first stage targets initial generation and self-verification, and the second stage focuses on revision. Extensive experiments on three backbone models show that SCR substantially improves reasoning efficiency and self-verification. Besides, compared with existing reasoning paradigms, it reduces output token length by up to 50%. 

---
# Can Large Language Models Understand, Reason About, and Generate Code-Switched Text? 

**Authors**: Genta Indra Winata, David Anugraha, Patrick Amadeus Irawan, Anirban Das, Haneul Yoo, Paresh Dashore, Shreyas Kulkarni, Ruochen Zhang, Haruki Sakajo, Frederikus Hudi, Anaelia Ovalle, Syrielle Montariol, Felix Gaschi, Michael Anugraha, Rutuj Ravindra Puranik, Zawad Hayat Ahmed, Adril Putra Merin, Emmanuele Chersoni  

**Link**: [PDF](https://arxiv.org/pdf/2601.07153)  

**Abstract**: Code-switching is a pervasive phenomenon in multilingual communication, yet the robustness of large language models (LLMs) in mixed-language settings remains insufficiently understood. In this work, we present a comprehensive evaluation of LLM capabilities in understanding, reasoning over, and generating code-switched text. We introduce CodeMixQA a novel benchmark with high-quality human annotations, comprising 16 diverse parallel code-switched language-pair variants that span multiple geographic regions and code-switching patterns, and include both original scripts and their transliterated forms. Using this benchmark, we analyze the reasoning behavior of LLMs on code-switched question-answering tasks, shedding light on how models process and reason over mixed-language inputs. We further conduct a systematic evaluation of LLM-generated synthetic code-switched text, focusing on both naturalness and semantic fidelity, and uncover key limitations in current generation capabilities. Our findings reveal persistent challenges in both reasoning and generation under code-switching conditions and provide actionable insights for building more robust multilingual LLMs. We release the dataset and code as open source. 

---
# Measuring Iterative Temporal Reasoning with TimePuzzles 

**Authors**: Zhengxiang Wang, Zeyu Dong  

**Link**: [PDF](https://arxiv.org/pdf/2601.07148)  

**Abstract**: We introduce TimePuzzles, a constraint-based date inference task for evaluating iterative temporal reasoning. Each puzzle combines factual temporal anchors with (cross-cultural) calendar relations, admits one or multiple valid solution dates, and is algorithmically generated for controlled, dynamic, and continual evaluation. Across 13 diverse LLMs, TimePuzzles well distinguishes their iterative temporal reasoning capabilities and remains challenging without tools: GPT-5 reaches only 49.3% accuracy and all other models stay below 31%, despite the dataset's simplicity. Web search consistently yields substantial gains and using code interpreter shows mixed effects, but all models perform much better when constraints are rewritten with explicit dates, revealing a gap in reliable tool use. Overall, TimePuzzles presents a simple, cost-effective diagnostic for tool-augmented iterative temporal reasoning. 

---
# ReMIND: Orchestrating Modular Large Language Models for Controllable Serendipity A REM-Inspired System Design for Emergent Creative Ideation 

**Authors**: Makoto Sato  

**Link**: [PDF](https://arxiv.org/pdf/2601.07121)  

**Abstract**: Large language models (LLMs) are used not only for problem solving but also for creative ideation; however, eliciting serendipitous insights that are both novel and internally coherent remains difficult. While stochastic sampling promotes novelty, it often degrades consistency. Here, we propose ReMIND, a REM-inspired modular framework for ideation. ReMIND consists of four stages: wake, which generates a stable low-temperature semantic baseline; dream, which performs high-temperature exploratory generation; judge, which applies coarse evaluation to filter incoherent outputs and extract candidate ideas; and re-wake, which re-articulates selected ideas into coherent final outputs. By instantiating each stage as an independent LLM, ReMIND enables functional separation between exploration and consolidation. Parameter sweeps show that ReMIND reliably induces semantic exploration while preserving downstream stability. Embedding-based analyses confirm substantial semantic displacement during the dream phase, whereas external evaluations reveal that high-quality ideas emerge sporadically rather than as extrema along any single metric. These results suggest that serendipitous ideation in LLMs is a rare-event process best approached through system level design that shapes the conditions under which valuable ideas can emerge and be stabilized. ReMIND provides a general framework for studying the computational basis of serendipity and illustrates how modular LLM orchestration can bridge exploration and stabilization. 

---
# The Need for a Socially-Grounded Persona Framework for User Simulation 

**Authors**: Pranav Narayanan Venkit, Yu Li, Yada Pruksachatkun, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.07110)  

**Abstract**: Synthetic personas are widely used to condition large language models (LLMs) for social simulation, yet most personas are still constructed from coarse sociodemographic attributes or summaries. We revisit persona creation by introducing SCOPE, a socially grounded framework for persona construction and evaluation, built from a 141-item, two-hour sociopsychological protocol collected from 124 U.S.-based participants. Across seven models, we find that demographic-only personas are a structural bottleneck: demographics explain only ~1.5% of variance in human response similarity. Adding sociopsychological facets improves behavioral prediction and reduces over-accentuation, and non-demographic personas based on values and identity achieve strong alignment with substantially lower bias. These trends generalize to SimBench (441 aligned questions), where SCOPE personas outperform default prompting and NVIDIA Nemotron personas, and SCOPE augmentation improves Nemotron-based personas. Our results indicate that persona quality depends on sociopsychological structure rather than demographic templates or summaries. 

---
# Fine-Tuning vs. RAG for Multi-Hop Question Answering with Novel Knowledge 

**Authors**: Zhuoyi Yang, Yurun Song, Iftekhar Ahmed, Ian Harris  

**Link**: [PDF](https://arxiv.org/pdf/2601.07054)  

**Abstract**: Multi-hop question answering is widely used to evaluate the reasoning capabilities of large language models (LLMs), as it requires integrating multiple pieces of supporting knowledge to arrive at a correct answer. While prior work has explored different mechanisms for providing knowledge to LLMs, such as finetuning and retrieval-augmented generation (RAG), their relative effectiveness for multi-hop question answering remains insufficiently understood, particularly when the required knowledge is temporally novel.
In this paper, we systematically compare parametric and non-parametric knowledge injection methods for open-domain multi-hop question answering. We evaluate unsupervised fine-tuning (continual pretraining), supervised fine-tuning, and retrieval-augmented generation across three 7B-parameter open-source LLMs. Experiments are conducted on two benchmarks: QASC, a standard multi-hop science question answering dataset, and a newly constructed dataset of over 10,000 multi-hop questions derived from Wikipedia events in 2024, designed to test knowledge beyond the models' pretraining cutoff.
Our results show that unsupervised fine-tuning provides only limited gains over base models, suggesting that continual pretraining alone is insufficient for improving multi-hop reasoning accuracy. In contrast, retrieval-augmented generation yields substantial and consistent improvements, particularly when answering questions that rely on temporally novel information. Supervised fine-tuning achieves the highest overall accuracy across models and datasets. These findings highlight fundamental differences in how knowledge injection mechanisms support multi-hop question answering and underscore the importance of retrieval-based methods when external or compositional knowledge is required. 

---
# Engineering of Hallucination in Generative AI: It's not a Bug, it's a Feature 

**Authors**: Tim Fingscheidt, Patrick Blumenberg, Björn Möller  

**Link**: [PDF](https://arxiv.org/pdf/2601.07046)  

**Abstract**: Generative artificial intelligence (AI) is conquering our lives at lightning speed. Large language models such as ChatGPT answer our questions or write texts for us, large computer vision models such as GAIA-1 generate videos on the basis of text descriptions or continue prompted videos. These neural network models are trained using large amounts of text or video data, strictly according to the real data employed in training. However, there is a surprising observation: When we use these models, they only function satisfactorily when they are allowed a certain degree of fantasy (hallucination). While hallucination usually has a negative connotation in generative AI - after all, ChatGPT is expected to give a fact-based answer! - this article recapitulates some simple means of probability engineering that can be used to encourage generative AI to hallucinate to a limited extent and thus lead to the desired results. We have to ask ourselves: Is hallucination in gen-erative AI probably not a bug, but rather a feature? 

---
# When Abundance Conceals Weakness: Knowledge Conflict in Multilingual Models 

**Authors**: Jiaqi Zhao, Qiang Huang, Haodong Chen, Xiaoxing You, Jun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.07041)  

**Abstract**: Large Language Models (LLMs) encode vast world knowledge across multiple languages, yet their internal beliefs are often unevenly distributed across linguistic spaces. When external evidence contradicts these language-dependent memories, models encounter \emph{cross-lingual knowledge conflict}, a phenomenon largely unexplored beyond English-centric settings. We introduce \textbf{CLEAR}, a \textbf{C}ross-\textbf{L}ingual knowl\textbf{E}dge conflict ev\textbf{A}luation f\textbf{R}amework that systematically examines how multilingual LLMs reconcile conflicting internal beliefs and multilingual external evidence. CLEAR decomposes conflict resolution into four progressive scenarios, from multilingual parametric elicitation to competitive multi-source cross-lingual induction, and systematically evaluates model behavior across two complementary QA benchmarks with distinct task characteristics. We construct multilingual versions of ConflictQA and ConflictingQA covering 10 typologically diverse languages and evaluate six representative LLMs. Our experiments reveal a task-dependent decision dichotomy. In reasoning-intensive tasks, conflict resolution is dominated by language resource abundance, with high-resource languages exerting stronger persuasive power. In contrast, for entity-centric factual conflicts, linguistic affinity, not resource scale, becomes decisive, allowing low-resource but linguistically aligned languages to outperform distant high-resource ones. 

---
# Task Arithmetic with Support Languages for Low-Resource ASR 

**Authors**: Emma Rafkin, Dan DeGenaro, Xiulin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.07038)  

**Abstract**: The development of resource-constrained approaches to automatic speech recognition (ASR) is of great interest due to its broad applicability to many low-resource languages for which there is scant usable data. Existing approaches to many low-resource natural language processing tasks leverage additional data from higher-resource languages that are closely related to a target low-resource language. One increasingly popular approach uses task arithmetic to combine models trained on different tasks to create a model for a task where there is little to no training data. In this paper, we consider training on a particular language to be a task, and we generate task vectors by fine-tuning variants of the Whisper ASR system. For pairings of high- and low-resource languages, we merge task vectors via a linear combination, optimizing the weights of the linear combination on the downstream word error rate on the low-resource target language's validation set. We find that this approach consistently improves performance on the target languages. 

---
# Mid-Think: Training-Free Intermediate-Budget Reasoning via Token-Level Triggers 

**Authors**: Wang Yang, Debargha Ganguly, Xinpeng Li, Chaoda Song, Shouren Wang, Vikash Singh, Vipin Chaudhary, Xiaotian Han  

**Link**: [PDF](https://arxiv.org/pdf/2601.07036)  

**Abstract**: Hybrid reasoning language models are commonly controlled through high-level Think/No-think instructions to regulate reasoning behavior, yet we found that such mode switching is largely driven by a small set of trigger tokens rather than the instructions themselves. Through attention analysis and controlled prompting experiments, we show that a leading ``Okay'' token induces reasoning behavior, while the newline pattern following ``</think>'' suppresses it. Based on this observation, we propose Mid-Think, a simple training-free prompting format that combines these triggers to achieve intermediate-budget reasoning, consistently outperforming fixed-token and prompt-based baselines in terms of the accuracy-length trade-off. Furthermore, applying Mid-Think to RL training after SFT reduces training time by approximately 15% while improving final performance of Qwen3-8B on AIME from 69.8% to 72.4% and on GPQA from 58.5% to 61.1%, demonstrating its effectiveness for both inference-time control and RL-based reasoning training. 

---
# Codified Foreshadowing-Payoff Text Generation 

**Authors**: Longfei Yun, Kun Zhou, Yupeng Hou, Letian Peng, Jingbo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2601.07033)  

**Abstract**: Foreshadowing and payoff are ubiquitous narrative devices through which authors introduce commitments early in a story and resolve them through concrete, observable outcomes. However, despite advances in story generation, large language models (LLMs) frequently fail to bridge these long-range narrative dependencies, often leaving "Chekhov's guns" unfired even when the necessary context is present. Existing evaluations largely overlook this structural failure, focusing on surface-level coherence rather than the logical fulfillment of narrative setups. In this paper, we introduce Codified Foreshadowing-Payoff Generation (CFPG), a novel framework that reframes narrative quality through the lens of payoff realization. Recognizing that LLMs struggle to intuitively grasp the "triggering mechanism" of a foreshadowed event, CFPG transforms narrative continuity into a set of executable causal predicates. By mining and encoding Foreshadow-Trigger-Payoff triples from the BookSum corpus, we provide structured supervision that ensures foreshadowed commitments are not only mentioned but also temporally and logically fulfilled. Experiments demonstrate that CFPG significantly outperforms standard prompting baselines in payoff accuracy and narrative alignment. Our findings suggest that explicitly codifying narrative mechanics is essential for moving LLMs from surface-level fluency to genuine narrative competence. 

---
# Solar Open Technical Report 

**Authors**: Sungrae Park, Sanghoon Kim, Jungho Cho, Gyoungjin Gim, Dawoon Jung, Mikyoung Cha, Eunhae Choo, Taekgyu Hong, Minbyul Jeong, SeHwan Joo, Minsoo Khang, Eunwon Kim, Minjeong Kim, Sujeong Kim, Yunsu Kim, Hyeonju Lee, Seunghyun Lee, Sukyung Lee, Siyoung Park, Gyungin Shin, Inseo Song, Wonho Song, Seonghoon Yang, Seungyoun Yi, Sanghoon Yoon, Jeonghyun Ko, Seyoung Song, Keunwoo Choi, Hwalsuk Lee, Sunghun Kim, Du-Seong Chang, Kyunghyun Cho, Junsuk Choe, Hwaran Lee, Jae-Gil Lee, KyungTae Lim, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2601.07022)  

**Abstract**: We introduce Solar Open, a 102B-parameter bilingual Mixture-of-Experts language model for underserved languages. Solar Open demonstrates a systematic methodology for building competitive LLMs by addressing three interconnected challenges. First, to train effectively despite data scarcity for underserved languages, we synthesize 4.5T tokens of high-quality, domain-specific, and RL-oriented data. Second, we coordinate this data through a progressive curriculum jointly optimizing composition, quality thresholds, and domain coverage across 20 trillion tokens. Third, to enable reasoning capabilities through scalable RL, we apply our proposed framework SnapPO for efficient optimization. Across benchmarks in English and Korean, Solar Open achieves competitive performance, demonstrating the effectiveness of this methodology for underserved language AI development. 

---
# TurkBench: A Benchmark for Evaluating Turkish Large Language Models 

**Authors**: Çağrı Toraman, Ahmet Kaan Sever, Ayse Aysu Cengiz, Elif Ecem Arslan, Görkem Sevinç, Mete Mert Birdal, Yusuf Faruk Güldemir, Ali Buğra Kanburoğlu, Sezen Felekoğlu, Osman Gürlek, Sarp Kantar, Birsen Şahin Kütük, Büşra Tufan, Elif Genç, Serkan Coşkun, Gupse Ekin Demir, Muhammed Emin Arayıcı, Olgun Dursun, Onur Gungor, Susan Üsküdarlı, Abdullah Topraksoy, Esra Darıcı  

**Link**: [PDF](https://arxiv.org/pdf/2601.07020)  

**Abstract**: With the recent surge in the development of large language models, the need for comprehensive and language-specific evaluation benchmarks has become critical. While significant progress has been made in evaluating English language models, benchmarks for other languages, particularly those with unique linguistic characteristics such as Turkish, remain less developed. Our study introduces TurkBench, a comprehensive benchmark designed to assess the capabilities of generative large language models in the Turkish language. TurkBench involves 8,151 data samples across 21 distinct subtasks. These are organized under six main categories of evaluation: Knowledge, Language Understanding, Reasoning, Content Moderation, Turkish Grammar and Vocabulary, and Instruction Following. The diverse range of tasks and the culturally relevant data would provide researchers and developers with a valuable tool for evaluating their models and identifying areas for improvement. We further publish our benchmark for online submissions at this https URL 

---
# Lexicalized Constituency Parsing for Middle Dutch: Low-resource Training and Cross-Domain Generalization 

**Authors**: Yiming Liang, Fang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.07008)  

**Abstract**: Recent years have seen growing interest in applying neural networks and contextualized word embeddings to the parsing of historical languages. However, most advances have focused on dependency parsing, while constituency parsing for low-resource historical languages like Middle Dutch has received little attention. In this paper, we adapt a transformer-based constituency parser to Middle Dutch, a highly heterogeneous and low-resource language, and investigate methods to improve both its in-domain and cross-domain performance. We show that joint training with higher-resource auxiliary languages increases F1 scores by up to 0.73, with the greatest gains achieved from languages that are geographically and temporally closer to Middle Dutch. We further evaluate strategies for leveraging newly annotated data from additional domains, finding that fine-tuning and data combination yield comparable improvements, and our neural parser consistently outperforms the currently used PCFG-based parser for Middle Dutch. We further explore feature-separation techniques for domain adaptation and demonstrate that a minimum threshold of approximately 200 examples per domain is needed to effectively enhance cross-domain performance. 

---
# MedTutor: A Retrieval-Augmented LLM System for Case-Based Medical Education 

**Authors**: Dongsuk Jang, Ziyao Shangguan, Kyle Tegtmeyer, Anurag Gupta, Jan Czerminski, Sophie Chheang, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2601.06979)  

**Abstract**: The learning process for medical residents presents significant challenges, demanding both the ability to interpret complex case reports and the rapid acquisition of accurate medical knowledge from reliable sources. Residents typically study case reports and engage in discussions with peers and mentors, but finding relevant educational materials and evidence to support their learning from these cases is often time-consuming and challenging. To address this, we introduce MedTutor, a novel system designed to augment resident training by automatically generating evidence-based educational content and multiple-choice questions from clinical case reports. MedTutor leverages a Retrieval-Augmented Generation (RAG) pipeline that takes clinical case reports as input and produces targeted educational materials. The system's architecture features a hybrid retrieval mechanism that synergistically queries a local knowledge base of medical textbooks and academic literature (using PubMed, Semantic Scholar APIs) for the latest related research, ensuring the generated content is both foundationally sound and current. The retrieved evidence is filtered and ordered using a state-of-the-art reranking model and then an LLM generates the final long-form output describing the main educational content regarding the case-report. We conduct a rigorous evaluation of the system. First, three radiologists assessed the quality of outputs, finding them to be of high clinical and educational value. Second, we perform a large scale evaluation using an LLM-as-a Judge to understand if LLMs can be used to evaluate the output of the system. Our analysis using correlation between LLMs outputs and human expert judgments reveals a moderate alignment and highlights the continued necessity of expert oversight. 

---
# UETQuintet at BioCreative IX - MedHopQA: Enhancing Biomedical QA with Selective Multi-hop Reasoning and Contextual Retrieval 

**Authors**: Quoc-An Nguyen, Thi-Minh-Thu Vu, Bich-Dat Nguyen, Dinh-Quang-Minh Tran, Hoang-Quynh Le  

**Link**: [PDF](https://arxiv.org/pdf/2601.06974)  

**Abstract**: Biomedical Question Answering systems play a critical role in processing complex medical queries, yet they often struggle with the intricate nature of medical data and the demand for multi-hop reasoning. In this paper, we propose a model designed to effectively address both direct and sequential questions. While sequential questions are decomposed into a chain of sub-questions to perform reasoning across a chain of steps, direct questions are processed directly to ensure efficiency and minimise processing overhead. Additionally, we leverage multi-source information retrieval and in-context learning to provide rich, relevant context for generating answers. We evaluated our model on the BioCreative IX - MedHopQA Shared Task datasets. Our approach achieves an Exact Match score of 0.84, ranking second on the current leaderboard. These results highlight the model's capability to meet the challenges of Biomedical Question Answering, offering a versatile solution for advancing medical research and practice. 

---
# LLMs Can't Play Hangman: On the Necessity of a Private Working Memory for Language Agents 

**Authors**: Davide Baldelli, Ali Parviz, Amal Zouaq, Sarath Chandar  

**Link**: [PDF](https://arxiv.org/pdf/2601.06973)  

**Abstract**: As LLMs move from text completion toward autonomous agents, they remain constrained by the standard chat interface, which lacks private working memory. This raises a fundamental question: can agents reliably perform interactive tasks that depend on hidden state? We define Private State Interactive Tasks (PSITs), which require agents to generate and maintain hidden information while producing consistent public responses. We show theoretically that any agent restricted to the public conversation history cannot simultaneously preserve secrecy and consistency in PSITs, yielding an impossibility theorem. To empirically validate this limitation, we introduce a self-consistency testing protocol that evaluates whether agents can maintain a hidden secret across forked dialogue branches. Standard chat-based LLMs and retrieval-based memory baselines fail this test regardless of scale, demonstrating that semantic retrieval does not enable true state maintenance. To address this, we propose a novel architecture incorporating an explicit private working memory; we demonstrate that this mechanism restores consistency, establishing private state as a necessary component for interactive language agents. 

---
# Categorize Early, Integrate Late: Divergent Processing Strategies in Automatic Speech Recognition 

**Authors**: Nathan Roll, Pranav Bhalerao, Martijn Bartelds, Arjun Pawar, Yuka Tatsumi, Tolulope Ogunremi, Chen Shani, Calbert Graham, Meghan Sumner, Dan Jurafsky  

**Link**: [PDF](https://arxiv.org/pdf/2601.06972)  

**Abstract**: In speech language modeling, two architectures dominate the frontier: the Transformer and the Conformer. However, it remains unknown whether their comparable performance stems from convergent processing strategies or distinct architectural inductive biases. We introduce Architectural Fingerprinting, a probing framework that isolates the effect of architecture on representation, and apply it to a controlled suite of 24 pre-trained encoders (39M-3.3B parameters). Our analysis reveals divergent hierarchies: Conformers implement a "Categorize Early" strategy, resolving phoneme categories 29% earlier in depth and speaker gender by 16% depth. In contrast, Transformers "Integrate Late," deferring phoneme, accent, and duration encoding to deep layers (49-57%). These fingerprints suggest design heuristics: Conformers' front-loaded categorization may benefit low-latency streaming, while Transformers' deep integration may favor tasks requiring rich context and cross-utterance normalization. 

---
# RealMem: Benchmarking LLMs in Real-World Memory-Driven Interaction 

**Authors**: Haonan Bian, Zhiyuan Yao, Sen Hu, Zishan Xu, Shaolei Zhang, Yifu Guo, Ziliang Yang, Xueran Han, Huacan Wang, Ronghao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.06966)  

**Abstract**: As Large Language Models (LLMs) evolve from static dialogue interfaces to autonomous general agents, effective memory is paramount to ensuring long-term consistency. However, existing benchmarks primarily focus on casual conversation or task-oriented dialogue, failing to capture **"long-term project-oriented"** interactions where agents must track evolving goals.
To bridge this gap, we introduce **RealMem**, the first benchmark grounded in realistic project scenarios. RealMem comprises over 2,000 cross-session dialogues across eleven scenarios, utilizing natural user queries for evaluation.
We propose a synthesis pipeline that integrates Project Foundation Construction, Multi-Agent Dialogue Generation, and Memory and Schedule Management to simulate the dynamic evolution of memory. Experiments reveal that current memory systems face significant challenges in managing the long-term project states and dynamic context dependencies inherent in real-world projects.
Our code and datasets are available at [this https URL](this https URL). 

---
# X-Coder: Advancing Competitive Programming with Fully Synthetic Tasks, Solutions, and Tests 

**Authors**: Jie Wu, Haoling Li, Xin Zhang, Jiani Guo, Jane Luo, Steven Liu, Yangyu Huang, Ruihang Chu, Scarlett Li, Yujiu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06953)  

**Abstract**: Competitive programming presents great challenges for Code LLMs due to its intensive reasoning demands and high logical complexity. However, current Code LLMs still rely heavily on real-world data, which limits their scalability. In this paper, we explore a fully synthetic approach: training Code LLMs with entirely generated tasks, solutions, and test cases, to empower code reasoning models without relying on real-world data. To support this, we leverage feature-based synthesis to propose a novel data synthesis pipeline called SynthSmith. SynthSmith shows strong potential in producing diverse and challenging tasks, along with verified solutions and tests, supporting both supervised fine-tuning and reinforcement learning. Based on the proposed synthetic SFT and RL datasets, we introduce the X-Coder model series, which achieves a notable pass rate of 62.9 avg@8 on LiveCodeBench v5 and 55.8 on v6, outperforming DeepCoder-14B-Preview and AReal-boba2-14B despite having only 7B parameters. In-depth analysis reveals that scaling laws hold on our synthetic dataset, and we explore which dimensions are more effective to scale. We further provide insights into code-centric reinforcement learning and highlight the key factors that shape performance through detailed ablations and analysis. Our findings demonstrate that scaling high-quality synthetic data and adopting staged training can greatly advance code reasoning, while mitigating reliance on real-world coding data. 

---
# Symphonym: Universal Phonetic Embeddings for Cross-Script Toponym Matching via Teacher-Student Distillation 

**Authors**: Stephen Gadd  

**Link**: [PDF](https://arxiv.org/pdf/2601.06932)  

**Abstract**: Linking place names across languages and writing systems is a fundamental challenge in digital humanities and geographic information retrieval. Existing approaches rely on language-specific phonetic algorithms or transliteration rules that fail when names cross script boundaries -- no string metric can determine that "Moscow" when rendered in Cyrillic or Arabic refer to the same city.
I present Symphonym, a neural embedding system that maps toponyms from 20 writing systems into a unified 128-dimensional phonetic space. A Teacher network trained on articulatory phonetic features (via Epitran and PanPhon) produces target embeddings, while a Student network learns to approximate these from raw characters. At inference, only the lightweight Student (1.7M parameters) is required, enabling deployment without runtime phonetic conversion.
Training uses a three-phase curriculum on 57 million toponyms from GeoNames, Wikidata, and the Getty Thesaurus of Geographic Names. Phase 1 trains the Teacher on 467K phonetically-grounded triplets. Phase 2 aligns the Student to Teacher outputs across 23M samples, achieving 96.6% cosine similarity. Phase 3 fine-tunes on 3.3M hard negative triplets -- negatives sharing prefix and script with the anchor but referring to different places -- to sharpen discrimination.
Evaluation on the MEHDIE Hebrew-Arabic benchmark achieves 89.2% Recall@1, outperforming Levenshtein (81.5%) and Jaro-Winkler (78.5%). The system is optimised for cross-script matching; same-script variants can be handled by complementary string methods. Symphonym will enable fuzzy phonetic reconciliation and search across the World Historical Gazetteer's 67 million toponyms. Code and models are publicly available. 

---
# TreePS-RAG: Tree-based Process Supervision for Reinforcement Learning in Agentic RAG 

**Authors**: Tianhua Zhang, Kun Li, Junan Li, Yunxiang Li, Hongyin Luo, Xixin Wu, James Glass, Helen Meng  

**Link**: [PDF](https://arxiv.org/pdf/2601.06922)  

**Abstract**: Agentic retrieval-augmented generation (RAG) formulates question answering as a multi-step interaction between reasoning and information retrieval, and has recently been advanced by reinforcement learning (RL) with outcome-based supervision. While effective, relying solely on sparse final rewards limits step-wise credit assignment and provides weak guidance for intermediate reasoning and actions. Recent efforts explore process-level supervision, but typically depend on offline constructed training data, which risks distribution shift, or require costly intermediate annotations. We present TreePS-RAG, an online, tree-based RL framework for agentic RAG that enables step-wise credit assignment while retaining standard outcome-only rewards. Our key insight is to model agentic RAG reasoning as a rollout tree, where each reasoning step naturally maps to a node. This tree structure allows step utility to be estimated via Monte Carlo estimation over its descendant outcomes, yielding fine-grained process advantages without requiring intermediate labels. To make this paradigm practical, we introduce an efficient online tree construction strategy that preserves exploration diversity under a constrained computational budget. With a rollout cost comparable to strong baselines like Search-R1, experiments on seven multi-hop and general QA benchmarks across multiple model scales show that TreePS-RAG consistently and significantly outperforms both outcome-supervised and leading process-supervised RL methods. 

---
# Distributional Clarity: The Hidden Driver of RL-Friendliness in Large Language Models 

**Authors**: Shaoning Sun, Mingzhu Cai, Huang He, Bingjin Chen, Siqi Bao, Yujiu Yang, Hua Wu, Haifeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06911)  

**Abstract**: Language model families exhibit striking disparity in their capacity to benefit from reinforcement learning: under identical training, models like Qwen achieve substantial gains, while others like Llama yield limited improvements. Complementing data-centric approaches, we reveal that this disparity reflects a hidden structural property: \textbf{distributional clarity} in probability space. Through a three-stage analysis-from phenomenon to mechanism to interpretation-we uncover that RL-friendly models exhibit intra-class compactness and inter-class separation in their probability assignments to correct vs. incorrect responses. We quantify this clarity using the \textbf{Silhouette Coefficient} ($S$) and demonstrate that (1) high $S$ correlates strongly with RL performance; (2) low $S$ is associated with severe logic errors and reasoning instability. To confirm this property, we introduce a Silhouette-Aware Reweighting strategy that prioritizes low-$S$ samples during training. Experiments across six mathematical benchmarks show consistent improvements across all model families, with gains up to 5.9 points on AIME24. Our work establishes distributional clarity as a fundamental, trainable property underlying RL-Friendliness. 

---
# Fine-grained Verbal Attack Detection via a Hierarchical Divide-and-Conquer Framework 

**Authors**: Quan Zheng, Yuanhe Tian, Ming Wang, Yan Song  

**Link**: [PDF](https://arxiv.org/pdf/2601.06907)  

**Abstract**: In the digital era, effective identification and analysis of verbal attacks are essential for maintaining online civility and ensuring social security. However, existing research is limited by insufficient modeling of conversational structure and contextual dependency, particularly in Chinese social media where implicit attacks are prevalent. Current attack detection studies often emphasize general semantic understanding while overlooking user response relationships, hindering the identification of implicit and context-dependent attacks. To address these challenges, we present the novel "Hierarchical Attack Comment Detection" dataset and propose a divide-and-conquer, fine-grained framework for verbal attack recognition based on spatiotemporal information. The proposed dataset explicitly encodes hierarchical reply structures and chronological order, capturing complex interaction patterns in multi-turn discussions. Building on this dataset, the framework decomposes attack detection into hierarchical subtasks, where specialized lightweight models handle explicit detection, implicit intent inference, and target identification under constrained context. Extensive experiments on the proposed dataset and benchmark intention detection datasets show that smaller models using our framework significantly outperform larger monolithic models relying on parameter scaling, demonstrating the effectiveness of structured task decomposition. 

---
# Paraphrasing Adversarial Attack on LLM-as-a-Reviewer 

**Authors**: Masahiro Kaneko  

**Link**: [PDF](https://arxiv.org/pdf/2601.06884)  

**Abstract**: The use of large language models (LLMs) in peer review systems has attracted growing attention, making it essential to examine their potential vulnerabilities. Prior attacks rely on prompt injection, which alters manuscript content and conflates injection susceptibility with evaluation robustness. We propose the Paraphrasing Adversarial Attack (PAA), a black-box optimization method that searches for paraphrased sequences yielding higher review scores while preserving semantic equivalence and linguistic naturalness. PAA leverages in-context learning, using previous paraphrases and their scores to guide candidate generation. Experiments across five ML and NLP conferences with three LLM reviewers and five attacking models show that PAA consistently increases review scores without changing the paper's claims. Human evaluation confirms that generated paraphrases maintain meaning and naturalness. We also find that attacked papers exhibit increased perplexity in reviews, offering a potential detection signal, and that paraphrasing submissions can partially mitigate attacks. 

---
# BiasLab: A Multilingual, Dual-Framing Framework for Robust Measurement of Output-Level Bias in Large Language Models 

**Authors**: William Guey, Wei Zhang, Pei-Luen Patrick Rau, Pierrick Bougault, Vitor D. de Moura, Bertan Ucar, Jose O. Gomes  

**Link**: [PDF](https://arxiv.org/pdf/2601.06861)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in high-stakes contexts where their outputs influence real-world decisions. However, evaluating bias in LLM outputs remains methodologically challenging due to sensitivity to prompt wording, limited multilingual coverage, and the lack of standardized metrics that enable reliable comparison across models. This paper introduces BiasLab, an open-source, model-agnostic evaluation framework for quantifying output-level (extrinsic) bias through a multilingual, robustness-oriented experimental design. BiasLab constructs mirrored probe pairs under a strict dual-framing scheme: an affirmative assertion favoring Target A and a reverse assertion obtained by deterministic target substitution favoring Target B, while preserving identical linguistic structure. To reduce dependence on prompt templates, BiasLab performs repeated evaluation under randomized instructional wrappers and enforces a fixed-choice Likert response format to maximize comparability across models and languages. Responses are normalized into agreement labels using an LLM-based judge, aligned for polarity consistency across framings, and aggregated into quantitative bias indicators with descriptive statistics including effect sizes and neutrality rates. The framework supports evaluation across diverse bias axes, including demographic, cultural, political, and geopolitical topics, and produces reproducible artifacts such as structured reports and comparative visualizations. BiasLab contributes a standardized methodology for cross-lingual and framing-sensitive bias measurement that complements intrinsic and dataset-based audits, enabling researchers and institutions to benchmark robustness and make better-informed deployment decisions. 

---
# †DAGGER: Distractor-Aware Graph Generation for Executable Reasoning in Math Problems 

**Authors**: Zabir Al Nazi, Shubhashis Roy Dipta, Sudipta Kar  

**Link**: [PDF](https://arxiv.org/pdf/2601.06853)  

**Abstract**: Chain-of-Thought (CoT) prompting is widely adopted for mathematical problem solving, including in low-resource languages, yet its behavior under irrelevant context remains underexplored. To systematically study this challenge, we introduce DISTRACTMATH-BN, a Bangla benchmark that augments MGSM and MSVAMP with semantically coherent but computationally irrelevant information. Evaluating seven models ranging from 3B to 12B parameters, we observe substantial performance degradation under distractors: standard models drop by up to 41 points, while reasoning-specialized models decline by 14 to 20 points despite consuming five times more tokens. We propose †DAGGER, which reformulates mathematical problem solving as executable computational graph generation with explicit modeling of distractor nodes. Fine-tuning Gemma-3 models using supervised fine-tuning followed by Group Relative Policy Optimization achieves comparable weighted accuracy on augmented benchmarks while using 89 percent fewer tokens than reasoning models. Importantly, this robustness emerges without explicit training on distractor-augmented examples. Our results suggest that enforcing structured intermediate representations improves robustness and inference efficiency in mathematical reasoning compared to free-form approaches, particularly in noisy, low-resource settings. 

---
# Explainable Multimodal Aspect-Based Sentiment Analysis with Dependency-guided Large Language Model 

**Authors**: Zhongzheng Wang, Yuanhe Tian, Hongzhi Wang, Yan Song  

**Link**: [PDF](https://arxiv.org/pdf/2601.06848)  

**Abstract**: Multimodal aspect-based sentiment analysis (MABSA) aims to identify aspect-level sentiments by jointly modeling textual and visual information, which is essential for fine-grained opinion understanding in social media. Existing approaches mainly rely on discriminative classification with complex multimodal fusion, yet lacking explicit sentiment explainability. In this paper, we reformulate MABSA as a generative and explainable task, proposing a unified framework that simultaneously predicts aspect-level sentiment and generates natural language explanations. Based on multimodal large language models (MLLMs), our approach employs a prompt-based generative paradigm, jointly producing sentiment and explanation. To further enhance aspect-oriented reasoning capabilities, we propose a dependency-syntax-guided sentiment cue strategy. This strategy prunes and textualizes the aspect-centered dependency syntax tree, guiding the model to distinguish different sentiment aspects and enhancing its explainability. To enable explainability, we use MLLMs to construct new datasets with sentiment explanations to fine-tune. Experiments show that our approach not only achieves consistent gains in sentiment classification accuracy, but also produces faithful, aspect-grounded explanations. 

---
# PDR: A Plug-and-Play Positional Decay Framework for LLM Pre-training Data Detection 

**Authors**: Jinhan Liu, Yibo Yang, Ruiying Lu, Piotr Piekos, Yimeng Chen, Peng Wang, Dandan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2601.06827)  

**Abstract**: Detecting pre-training data in Large Language Models (LLMs) is crucial for auditing data privacy and copyright compliance, yet it remains challenging in black-box, zero-shot settings where computational resources and training data are scarce. While existing likelihood-based methods have shown promise, they typically aggregate token-level scores using uniform weights, thereby neglecting the inherent information-theoretic dynamics of autoregressive generation. In this paper, we hypothesize and empirically validate that memorization signals are heavily skewed towards the high-entropy initial tokens, where model uncertainty is highest, and decay as context accumulates. To leverage this linguistic property, we introduce Positional Decay Reweighting (PDR), a training-free and plug-and-play framework. PDR explicitly reweights token-level scores to amplify distinct signals from early positions while suppressing noise from later ones. Extensive experiments show that PDR acts as a robust prior and can usually enhance a wide range of advanced methods across multiple benchmarks. 

---
# AgentHallu: Benchmarking Automated Hallucination Attribution of LLM-based Agents 

**Authors**: Xuannan Liu, Xiao Yang, Zekun Li, Peipei Li, Ran He  

**Link**: [PDF](https://arxiv.org/pdf/2601.06818)  

**Abstract**: As LLM-based agents operate over sequential multi-step reasoning, hallucinations arising at intermediate steps risk propagating along the trajectory, thus degrading overall reliability. Unlike hallucination detection in single-turn responses, diagnosing hallucinations in multi-step workflows requires identifying which step causes the initial divergence. To fill this gap, we propose a new research task, automated hallucination attribution of LLM-based agents, aiming to identify the step responsible for the hallucination and explain why. To support this task, we introduce AgentHallu, a comprehensive benchmark with: (1) 693 high-quality trajectories spanning 7 agent frameworks and 5 domains, (2) a hallucination taxonomy organized into 5 categories (Planning, Retrieval, Reasoning, Human-Interaction, and Tool-Use) and 14 sub-categories, and (3) multi-level annotations curated by humans, covering binary labels, hallucination-responsible steps, and causal explanations. We evaluate 13 leading models, and results show the task is challenging even for top-tier models (like GPT-5, Gemini-2.5-Pro). The best-performing model achieves only 41.1\% step localization accuracy, where tool-use hallucinations are the most challenging at just 11.6\%. We believe AgentHallu will catalyze future research into developing robust, transparent, and reliable agentic systems. 

---
# Forest Before Trees: Latent Superposition for Efficient Visual Reasoning 

**Authors**: Yubo Wang, Juntian Zhang, Yichen Wu, Yankai Lin, Nils Lukas, Yuhan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.06803)  

**Abstract**: While Chain-of-Thought empowers Large Vision-Language Models with multi-step reasoning, explicit textual rationales suffer from an information bandwidth bottleneck, where continuous visual details are discarded during discrete tokenization. Recent latent reasoning methods attempt to address this challenge, but often fall prey to premature semantic collapse due to rigid autoregressive objectives. In this paper, we propose Laser, a novel paradigm that reformulates visual deduction via Dynamic Windowed Alignment Learning (DWAL). Instead of forcing a point-wise prediction, Laser aligns the latent state with a dynamic validity window of future semantics. This mechanism enforces a "Forest-before-Trees" cognitive hierarchy, enabling the model to maintain a probabilistic superposition of global features before narrowing down to local details. Crucially, Laser maintains interpretability via decodable trajectories while stabilizing unconstrained learning via Self-Refined Superposition. Extensive experiments on 6 benchmarks demonstrate that Laser achieves state-of-the-art performance among latent reasoning methods, surpassing the strong baseline Monet by 5.03% on average. Notably, it achieves these gains with extreme efficiency, reducing inference tokens by more than 97%, while demonstrating robust generalization to out-of-distribution domains. 

---
# Doing More with Less: Data Augmentation for Sudanese Dialect Automatic Speech Recognition 

**Authors**: Ayman Mansour  

**Link**: [PDF](https://arxiv.org/pdf/2601.06802)  

**Abstract**: Although many Automatic Speech Recognition (ASR) systems have been developed for Modern Standard Arabic (MSA) and Dialectal Arabic (DA), few studies have focused on dialect-specific implementations, particularly for low-resource Arabic dialects such as Sudanese. This paper presents a comprehensive study of data augmentation techniques for fine-tuning OpenAI Whisper models and establishes the first benchmark for the Sudanese dialect. Two augmentation strategies are investigated: (1) self-training with pseudo-labels generated from unlabeled speech, and (2) TTS-based augmentation using synthetic speech from the Klaam TTS system. The best-performing model, Whisper-Medium fine-tuned with combined self-training and TTS augmentation (28.4 hours), achieves a Word Error Rate (WER) of 57.1% on the evaluation set and 51.6% on an out-of-domain holdout set substantially outperforming zero-shot multilingual Whisper (78.8% WER) and MSA-specialized Arabic models (73.8-123% WER). All experiments used low-cost resources (Kaggle free tier and this http URL trial), demonstrating that strategic data augmentation can overcome resource limitations for low-resource dialects and provide a practical roadmap for developing ASR systems for low-resource Arabic dialects and other marginalized language varieties. The models, evaluation benchmarks, and reproducible training pipelines are publicly released to facilitate future research on low-resource Arabic ASR. 

---
# CIRAG: Construction-Integration Retrieval and Adaptive Generation for Multi-hop Question Answering 

**Authors**: Zili Wei, Xiaocui Yang, Yilin Wang, Zihan Wang, Weidong Bao, Shi Feng, Daling Wang, Yifei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06799)  

**Abstract**: Triple-based Iterative Retrieval-Augmented Generation (iRAG) mitigates document-level noise for multi-hop question answering. However, existing methods still face limitations: (i) greedy single-path expansion, which propagates early errors and fails to capture parallel evidence from different reasoning branches, and (ii) granularity-demand mismatch, where a single evidence representation struggles to balance noise control with contextual sufficiency. In this paper, we propose the Construction-Integration Retrieval and Adaptive Generation model, CIRAG. It introduces an Iterative Construction-Integration module that constructs candidate triples and history-conditionally integrates them to distill core triples and generate the next-hop query. This module mitigates the greedy trap by preserving multiple plausible evidence chains. Besides, we propose an Adaptive Cascaded Multi-Granularity Generation module that progressively expands contextual evidence based on the problem requirements, from triples to supporting sentences and full passages. Moreover, we introduce Trajectory Distillation, which distills the teacher model's integration policy into a lightweight student, enabling efficient and reliable long-horizon reasoning. Extensive experiments demonstrate that CIRAG achieves superior performance compared to existing iRAG methods. 

---
# Garbage Attention in Large Language Models: BOS Sink Heads and Sink-aware Pruning 

**Authors**: Jaewon Sok, Jewon Yeom, Seonghyeon Park, Jeongjae Park, Taesup Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.06787)  

**Abstract**: Large Language Models (LLMs) are known to contain significant redundancy, yet a systematic explanation for why certain components, particularly in higher layers, are more redundant has remained elusive. In this work, we identify the BOS sink phenomenon as a key mechanism driving this layer-wise sensitivity. We show that attention heads with high BOS sink scores are strongly associated with functional redundancy: such heads, especially in deeper layers, contribute little to predictive performance and effectively serve as \emph{dumping grounds} for superfluous attention weights. This provides a concrete functional explanation for the structural redundancy reported in prior studies. Leveraging this insight, we introduce a simple pruning strategy that removes high-BOS sink heads. Experiments on Gemma-3, Llama-3.1, and Qwen3 demonstrate that this approach identifies redundant transformer components more reliably than weight- or activation-based criteria, while preserving performance close to dense baselines even under aggressive pruning. Moreover, we find that the behavior of sink heads remains stable across different sequence lengths. Overall, our results suggest that structural properties of attention offer a more intuitive and robust basis for model compression than magnitude-based methods. 

---
# EpiCaR: Knowing What You Don't Know Matters for Better Reasoning in LLMs 

**Authors**: Jewon Yeom, Jaewon Sok, Seonghyeon Park, Jeongjae Park, Taesup Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.06786)  

**Abstract**: Improving the reasoning abilities of large language models (LLMs) has largely relied on iterative self-training with model-generated data. While effective at boosting accuracy, existing approaches primarily reinforce successful reasoning paths, incurring a substantial calibration cost: models become overconfident and lose the ability to represent uncertainty. This failure has been characterized as a form of model collapse in alignment, where predictive distributions degenerate toward low-variance point estimates. We address this issue by reframing reasoning training as an epistemic learning problem, in which models must learn not only how to reason, but also when their reasoning should be trusted. We propose epistemically-calibrated reasoning (EpiCaR) as a training objective that jointly optimizes reasoning performance and calibration, and instantiate it within an iterative supervised fine-tuning framework using explicit self-evaluation signals. Experiments on Llama-3 and Qwen-3 families demonstrate that our approach achieves Pareto-superiority over standard baselines in both accuracy and calibration, particularly in models with sufficient reasoning capacity (e.g., 3B+). This framework generalizes effectively to OOD mathematical reasoning (GSM8K) and code generation (MBPP). Ultimately, our approach enables a 3X reduction in inference compute, matching the K=30 performance of STaR with only K=10 samples in capable models. 

---
# Multi-Stage Evolutionary Model Merging with Meta Data Driven Curriculum Learning for Sentiment-Specialized Large Language Modeling 

**Authors**: Keito Inoshita, Xiaokang Zhou, Akira Kawai  

**Link**: [PDF](https://arxiv.org/pdf/2601.06780)  

**Abstract**: The emergence of large language models (LLMs) has significantly transformed natural language processing (NLP), enabling more generalized models to perform various tasks with minimal training. However, traditional sentiment analysis methods, which focus on individual tasks such as sentiment classification or aspect-based analysis, are not practical for real-world applications that usually require handling multiple tasks. While offering flexibility, LLMs in sentiment-specific tasks often fall short of the required accuracy. Techniques like fine-tuning and evolutionary model merging help integrate models into a unified framework, which can improve the learning performance while reducing computational costs. The use of task meta-data and curriculum learning to optimize learning processes remains underexplored, while sentiment analysis is a critical task in NLP that requires high accuracy and scalability across multiple subtasks. In this study, we propose a hybrid learning model called Multi-stage Evolutionary Model Merging with Meta data driven Curriculum Learning (MEM-MCL), to enhance the sentiment analysis in large language modeling. In particular, expert models are created through instruction tuning for specific sentiment tasks and then merged using evolutionary algorithms to form a unified model. The merging process is optimized with weak data to enhance performance across tasks. The curriculum learning is incorporated to provide a learning sequence based on task difficulty, improving knowledge extraction from LLMs. Experiment results demonstrate that the proposed MEM-MCL model outperforms conventional LLMs in a majority of sentiment analysis tasks, achieving superior results across various subtasks. 

---
# GanitLLM: Difficulty-Aware Bengali Mathematical Reasoning through Curriculum-GRPO 

**Authors**: Shubhashis Roy Dipta, Khairul Mahbub, Nadia Najjar  

**Link**: [PDF](https://arxiv.org/pdf/2601.06767)  

**Abstract**: We present a Bengali mathematical reasoning model called GanitLLM (named after the Bangla word for mathematics, "Ganit"), together with a new difficulty-aware Bengali math corpus and a curriculum-based GRPO pipeline. Bengali is one of the world's most widely spoken languages, yet existing LLMs either reason in English and then translate, or simply fail on multi-step Bengali math, in part because reinforcement learning recipes are tuned for high-resource languages and collapse under reward sparsity in low-resource settings. To address this, we construct Ganit, a rigorously filtered and decontaminated Bengali math dataset with automatic difficulty tags derived from the pass@k of a strong evaluator model. Building on this dataset, we propose Curriculum-GRPO, which combines multi-stage training (SFT + GRPO) with difficulty-aware sampling and verifiable rewards for format, numerical correctness, and Bengali reasoning. On Bn-MGSM and Bn-MSVAMP, GanitLLM-4B improves over its Qwen3-4B base by +8 and +7 accuracy points, respectively, while increasing the percentage of Bengali reasoning tokens from 14% to over 88% and reducing average solution length from 943 to 193 words. 

---
# MTMCS-Bench: Evaluating Contextual Safety of Multimodal Large Language Models in Multi-Turn Dialogues 

**Authors**: Zheyuan Liu, Dongwhi Kim, Yixin Wan, Xiangchi Yuan, Zhaoxuan Tan, Fengran Mo, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06757)  

**Abstract**: Multimodal large language models (MLLMs) are increasingly deployed as assistants that interact through text and images, making it crucial to evaluate contextual safety when risk depends on both the visual scene and the evolving dialogue. Existing contextual safety benchmarks are mostly single-turn and often miss how malicious intent can emerge gradually or how the same scene can support both benign and exploitative goals. We introduce the Multi-Turn Multimodal Contextual Safety Benchmark (MTMCS-Bench), a benchmark of realistic images and multi-turn conversations that evaluates contextual safety in MLLMs under two complementary settings, escalation-based risk and context-switch risk. MTMCS-Bench offers paired safe and unsafe dialogues with structured evaluation. It contains over 30 thousand multimodal (image+text) and unimodal (text-only) samples, with metrics that separately measure contextual intent recognition, safety-awareness on unsafe cases, and helpfulness on benign ones. Across eight open-source and seven proprietary MLLMs, we observe persistent trade-offs between contextual safety and utility, with models tending to either miss gradual risks or over-refuse benign dialogues. Finally, we evaluate five current guardrails and find that they mitigate some failures but do not fully resolve multi-turn contextual risks. 

---
# Towards Computational Chinese Paleography 

**Authors**: Yiran Rex Ma  

**Link**: [PDF](https://arxiv.org/pdf/2601.06753)  

**Abstract**: Chinese paleography, the study of ancient Chinese writing, is undergoing a computational turn powered by artificial intelligence. This position paper charts the trajectory of this emerging field, arguing that it is evolving from automating isolated visual tasks to creating integrated digital ecosystems for scholarly research. We first map the landscape of digital resources, analyzing critical datasets for oracle bone, bronze, and bamboo slip scripts. The core of our analysis follows the field's methodological pipeline: from foundational visual processing (image restoration, character recognition), through contextual analysis (artifact rejoining, dating), to the advanced reasoning required for automated decipherment and human-AI collaboration. We examine the technological shift from classical computer vision to modern deep learning paradigms, including transformers and large multimodal models. Finally, we synthesize the field's core challenges -- notably data scarcity and a disconnect between current AI capabilities and the holistic nature of humanistic inquiry -- and advocate for a future research agenda focused on creating multimodal, few-shot, and human-centric systems to augment scholarly expertise. 

---
# Evaluating Accounting Reasoning Capabilities of Large Language Models 

**Authors**: Jie Zhou, Xin Chen, Jie Zhang, Hai Li, Jie Wang, Zhe Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.06707)  

**Abstract**: Large language models are transforming learning, cognition, and research across many fields. Effectively integrating them into professional domains, such as accounting, is a key challenge for enterprise digital transformation. To address this, we define vertical domain accounting reasoning and propose evaluation criteria derived from an analysis of the training data characteristics of representative GLM models. These criteria support systematic study of accounting reasoning and provide benchmarks for performance improvement. Using this framework, we evaluate GLM-6B, GLM-130B, GLM-4, and OpenAI GPT-4 on accounting reasoning tasks. Results show that prompt design significantly affects performance, with GPT-4 demonstrating the strongest capability. Despite these gains, current models remain insufficient for real-world enterprise accounting, indicating the need for further optimization to unlock their full practical value. 

---
# GRASP LoRA: GRPO Guided Adapter Sparsity Policy for Cross Lingual Transfer 

**Authors**: Besher Hassan, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.06702)  

**Abstract**: Parameter efficient fine tuning is a way to adapt LLMs to new languages when compute or data are limited, yet adapter pipelines usually choose a global prune ratio by grid search. This practice is computationally expensive and development set intensive, since it repeats training, freezes sparsity, and misses fractional optima. We introduce GRASP LoRA (GRPO Guided Adapter Sparsity Policy), which treats global sparsity as a learnable control variable. A GRPO controller interleaves with training, periodically probing candidate prune ratios on a small micro development set and updating a single global prune ratio online from its reward signal. It operates on merged source and target LoRA adapters on a frozen backbone and replaces grid search with one controller run that learns a prune ratio, followed by a single final merge and prune fine tuning run with pruning fixed to that ratio. On cross lingual transfer from English into Arabic and Chinese, including XL-Sum summarization and MLQA extractive question answering with Llama 3 8B, GRASP LoRA improves semantic faithfulness, content coverage, and answer quality over strong target only and merge and prune baselines. It reduces end to end runtime by multiple times relative to grid search, lowers reliance on large development sets, and makes adapter reuse practical for low resource deployment. 

---
# Characterising Toxicity in Generative Large Language Models 

**Authors**: Zhiyao Zhang, Yazan Mash'Al, Yuhan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.06700)  

**Abstract**: In recent years, the advent of the attention mechanism has significantly advanced the field of natural language processing (NLP), revolutionizing text processing and text generation. This has come about through transformer-based decoder-only architectures, which have become ubiquitous in NLP due to their impressive text processing and generation capabilities. Despite these breakthroughs, language models (LMs) remain susceptible to generating undesired outputs: inappropriate, offensive, or otherwise harmful responses. We will collectively refer to these as ``toxic'' outputs. Although methods like reinforcement learning from human feedback (RLHF) have been developed to align model outputs with human values, these safeguards can often be circumvented through carefully crafted prompts. Therefore, this paper examines the extent to which LLMs generate toxic content when prompted, as well as the linguistic factors -- both lexical and syntactic -- that influence the production of such outputs in generative models. 

---
# IDRBench: Interactive Deep Research Benchmark 

**Authors**: Yingchaojie Feng, Qiang Huang, Xiaoya Xie, Zhaorui Yang, Jun Yu, Wei Chen, Anthony K. H. Tung  

**Link**: [PDF](https://arxiv.org/pdf/2601.06676)  

**Abstract**: Deep research agents powered by Large Language Models (LLMs) can perform multi-step reasoning, web exploration, and long-form report generation. However, most existing systems operate in an autonomous manner, assuming fully specified user intent and evaluating only final outputs. In practice, research goals are often underspecified and evolve during exploration, making sustained interaction essential for robust alignment. Despite its importance, interaction remains largely invisible to existing deep research benchmarks, which neither model dynamic user feedback nor quantify its costs. We introduce IDRBench, the first benchmark for systematically evaluating interactive deep research. IDRBench combines a modular multi-agent research framework with on-demand interaction, a scalable reference-grounded user simulator, and an interaction-aware evaluation suite that jointly measures interaction benefits (quality and alignment) and costs (turns and tokens). Experiments across seven state-of-the-art LLMs show that interaction consistently improves research quality and robustness, often outweighing differences in model capacity, while revealing substantial trade-offs in interaction efficiency. 

---
# Evaluating Cross-Lingual Unlearning in Multilingual Language Models 

**Authors**: Tyler Lizzo, Larry Heck  

**Link**: [PDF](https://arxiv.org/pdf/2601.06675)  

**Abstract**: We present the first comprehensive evaluation of cross-lingual unlearning in multilingual LLMs. Using translated TOFU benchmarks in seven language/script variants, we test major unlearning algorithms and show that most fail to remove facts outside the training language, even when utility remains high. However, subspace-projection consistently outperforms the other methods, achieving strong cross-lingual forgetting with minimal degradation. Analysis of learned task subspaces reveals a shared interlingua structure: removing this shared subspace harms all languages, while removing language-specific components selectively affects one. These results demonstrate that multilingual forgetting depends on geometry in weight space, motivating subspace-based approaches for future unlearning systems. 

---
# Will it Merge? On The Causes of Model Mergeability 

**Authors**: Adir Rahamim, Asaf Yehudai, Boaz Carmeli, Leshem Choshen, Yosi Mass, Yonatan Belinkov  

**Link**: [PDF](https://arxiv.org/pdf/2601.06672)  

**Abstract**: Model merging has emerged as a promising technique for combining multiple fine-tuned models into a single multitask model without retraining. However, the factors that determine whether merging will succeed or fail remain poorly understood. In this work, we investigate why specific models are merged better than others. To do so, we propose a concrete, measurable definition of mergeability. We investigate several potential causes for high or low mergeability, highlighting the base model knowledge as a dominant factor: Models fine-tuned on instances that the base model knows better are more mergeable than models fine-tuned on instances that the base model struggles with. Based on our mergeability definition, we explore a simple weighted merging technique that better preserves weak knowledge in the base model. 

---
# InFi-Check: Interpretable and Fine-Grained Fact-Checking of LLMs 

**Authors**: Yuzhuo Bai, Shuzheng Si, Kangyang Luo, Qingyi Wang, Wenhao Li, Gang Chen, Fanchao Qi, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.06666)  

**Abstract**: Large language models (LLMs) often hallucinate, yet most existing fact-checking methods treat factuality evaluation as a binary classification problem, offering limited interpretability and failing to capture fine-grained error types. In this paper, we introduce InFi-Check, a framework for interpretable and fine-grained fact-checking of LLM outputs. Specifically, we first propose a controlled data synthesis pipeline that generates high-quality data featuring explicit evidence, fine-grained error type labels, justifications, and corrections. Based on this, we further construct large-scale training data and a manually verified benchmark InFi-Check-FG for fine-grained fact-checking of LLM outputs. Building on these high-quality training data, we further propose InFi-Checker, which can jointly provide supporting evidence, classify fine-grained error types, and produce justifications along with corrections. Experiments show that InFi-Checker achieves state-of-the-art performance on InFi-Check-FG and strong generalization across various downstream tasks, significantly improving the utility and trustworthiness of factuality evaluation. 

---
# What makes for an enjoyable protagonist? An analysis of character warmth and competence 

**Authors**: Hannes Rosenbusch  

**Link**: [PDF](https://arxiv.org/pdf/2601.06658)  

**Abstract**: Drawing on psychological and literary theory, we investigated whether the warmth and competence of movie protagonists predict IMDb ratings, and whether these effects vary across genres. Using 2,858 films and series from the Movie Scripts Corpus, we identified protagonists via AI-assisted annotation and quantified their warmth and competence with the LLM_annotate package ([1]; human-LLM agreement: r = .83). Preregistered Bayesian regression analyses revealed theory-consistent but small associations between both warmth and competence and audience ratings, while genre-specific interactions did not meaningfully improve predictions. Male protagonists were slightly less warm than female protagonists, and movies with male leads received higher ratings on average (an association that was multiple times stronger than the relationships between movie ratings and warmth/competence). These findings suggest that, although audiences tend to favor warm, competent characters, the effects on movie evaluations are modest, indicating that character personality is only one of many factors shaping movie ratings. AI-assisted annotation with LLM_annotate and gpt-4.1-mini proved effective for large-scale analyses but occasionally fell short of manually generated annotations. 

---
# Do Language Models Reason Across Languages? 

**Authors**: Yan Meng, Wafaa Mohammed, Christof Monz  

**Link**: [PDF](https://arxiv.org/pdf/2601.06644)  

**Abstract**: The real-world information sources are inherently multilingual, which naturally raises a question about whether language models can synthesize information across languages. In this paper, we introduce a simple two-hop question answering setting, where answering a question requires making inferences over two multilingual documents. We find that language models are more sensitive to language variation in answer-span documents than in those providing bridging information, despite the equal importance of both documents for answering a question. Under a step-by-step sub-question evaluation, we further show that in up to 33% of multilingual cases, models fail to infer the bridging information in the first step yet still answer the overall question correctly. This indicates that reasoning in language models, especially in multilingual settings, does not follow a faithful step-by-step decomposition. Subsequently, we show that the absence of reasoning decomposition leads to around 18% composition failure, where both sub-questions are answered correctly but fail for the final two-hop questions. To mitigate this, we propose a simple three-stage SUBQ prompting method to guide the multi-step reasoning with sub-questions, which boosts accuracy from 10.1% to 66.5%. 

---
# Efficient Aspect Term Extraction using Spiking Neural Network 

**Authors**: Abhishek Kumar Mishra, Arya Somasundaram, Anup Das, Nagarajan Kandasamy  

**Link**: [PDF](https://arxiv.org/pdf/2601.06637)  

**Abstract**: Aspect Term Extraction (ATE) identifies aspect terms in review sentences, a key subtask of sentiment analysis. While most existing approaches use energy-intensive deep neural networks (DNNs) for ATE as sequence labeling, this paper proposes a more energy-efficient alternative using Spiking Neural Networks (SNNs). Using sparse activations and event-driven inferences, SNNs capture temporal dependencies between words, making them suitable for ATE. The proposed architecture, SpikeATE, employs ternary spiking neurons and direct spike training fine-tuned with pseudo-gradients. Evaluated on four benchmark SemEval datasets, SpikeATE achieves performance comparable to state-of-the-art DNNs with significantly lower energy consumption. This highlights the use of SNNs as a practical and sustainable choice for ATE tasks. 

---
# MedEinst: Benchmarking the Einstellung Effect in Medical LLMs through Counterfactual Differential Diagnosis 

**Authors**: Wenting Chen, Zhongrui Zhu, Guolin Huang, Wenxuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06636)  

**Abstract**: Despite achieving high accuracy on medical benchmarks, LLMs exhibit the Einstellung Effect in clinical diagnosis--relying on statistical shortcuts rather than patient-specific evidence, causing misdiagnosis in atypical cases. Existing benchmarks fail to detect this critical failure mode. We introduce MedEinst, a counterfactual benchmark with 5,383 paired clinical cases across 49 diseases. Each pair contains a control case and a "trap" case with altered discriminative evidence that flips the diagnosis. We measure susceptibility via Bias Trap Rate--probability of misdiagnosing traps despite correctly diagnosing controls. Extensive Evaluation of 17 LLMs shows frontier models achieve high baseline accuracy but severe bias trap rates. Thus, we propose ECR-Agent, aligning LLM reasoning with Evidence-Based Medicine standard via two components: (1) Dynamic Causal Inference (DCI) performs structured reasoning through dual-pathway perception, dynamic causal graph reasoning across three levels (association, intervention, counterfactual), and evidence audit for final diagnosis; (2) Critic-Driven Graph and Memory Evolution (CGME) iteratively refines the system by storing validated reasoning paths in an exemplar base and consolidating disease-specific knowledge into evolving illness graphs. Source code is to be released. 

---
# Labels have Human Values: Value Calibration of Subjective Tasks 

**Authors**: Mohammed Fayiz Parappan, Ricardo Henao  

**Link**: [PDF](https://arxiv.org/pdf/2601.06631)  

**Abstract**: Building NLP systems for subjective tasks requires one to ensure their alignment to contrasting human values. We propose the MultiCalibrated Subjective Task Learner framework (MC-STL), which clusters annotations into identifiable human value clusters by three approaches (similarity of annotator rationales, expert-value taxonomies or rater's sociocultural descriptors) and calibrates predictions for each value cluster by learning cluster-specific embeddings. We demonstrate MC-STL on several subjective learning settings, including ordinal, binary, and preference learning predictions, and evaluate it on multiple datasets covering toxic chatbot conversations, offensive social media posts, and human preference alignment. The results show that MC-STL consistently outperforms the baselines that ignore the latent value structure of the annotations, delivering gains in discrimination, value-specific calibration, and disagreement-aware metrics. 

---
# Efficient and Reliable Estimation of Named Entity Linking Quality: A Case Study on GutBrainIE 

**Authors**: Marco Martinelli, Stefano Marchesin, Gianmaria Silvello  

**Link**: [PDF](https://arxiv.org/pdf/2601.06624)  

**Abstract**: Named Entity Linking (NEL) is a core component of biomedical Information Extraction (IE) pipelines, yet assessing its quality at scale is challenging due to the high cost of expert annotations and the large size of corpora. In this paper, we present a sampling-based framework to estimate the NEL accuracy of large-scale IE corpora under statistical guarantees and constrained annotation budgets. We frame NEL accuracy estimation as a constrained optimization problem, where the objective is to minimize expected annotation cost subject to a target Margin of Error (MoE) for the corpus-level accuracy estimate. Building on recent works on knowledge graph accuracy estimation, we adapt Stratified Two-Stage Cluster Sampling (STWCS) to the NEL setting, defining label-based strata and global surface-form clusters in a way that is independent of NEL annotations. Applied to 11,184 NEL annotations in GutBrainIE -- a new biomedical corpus openly released in fall 2025 -- our framework reaches a MoE $\leq 0.05$ by manually annotating only 2,749 triples (24.6%), leading to an overall accuracy estimate of $0.915 \pm 0.0473$. A time-based cost model and simulations against a Simple Random Sampling (SRS) baseline show that our design reduces expert annotation time by about 29% at fixed sample size. The framework is generic and can be applied to other NEL benchmarks and IE pipelines that require scalable and statistically robust accuracy assessment. 

---
# Pragya: An AI-Based Semantic Recommendation System for Sanskrit Subhasitas 

**Authors**: Tanisha Raorane, Prasenjit Kole  

**Link**: [PDF](https://arxiv.org/pdf/2601.06607)  

**Abstract**: Sanskrit Subhasitas encapsulate centuries of cultural and philosophical wisdom, yet remain underutilized in the digital age due to linguistic and contextual barriers. In this work, we present Pragya, a retrieval-augmented generation (RAG) framework for semantic recommendation of Subhasitas. We curate a dataset of 200 verses annotated with thematic tags such as motivation, friendship, and compassion. Using sentence embeddings (IndicBERT), the system retrieves top-k verses relevant to user queries. The retrieved results are then passed to a generative model (Mistral LLM) to produce transliterations, translations, and contextual explanations. Experimental evaluation demonstrates that semantic retrieval significantly outperforms keyword matching in precision and relevance, while user studies highlight improved accessibility through generated summaries. To our knowledge, this is the first attempt at integrating retrieval and generation for Sanskrit Subhasitas, bridging cultural heritage with modern applied AI. 

---
# N2N-GQA: Noise-to-Narrative for Graph-Based Table-Text Question Answering Using LLMs 

**Authors**: Mohamed Sharafath, Aravindh Annamalai, Ganesh Murugan, Aravindakumar Venugopalan  

**Link**: [PDF](https://arxiv.org/pdf/2601.06603)  

**Abstract**: Multi-hop question answering over hybrid table-text data requires retrieving and reasoning across multiple evidence pieces from large corpora, but standard Retrieval-Augmented Generation (RAG) pipelines process documents as flat ranked lists, causing retrieval noise to obscure reasoning chains. We introduce N2N-GQA. To our knowledge, it is the first zeroshot framework for open-domain hybrid table-text QA that constructs dynamic evidence graphs from noisy retrieval outputs. Our key insight is that multi-hop reasoning requires understanding relationships between evidence pieces: by modeling documents as graph nodes with semantic relationships as edges, we identify bridge documents connecting reasoning steps, a capability absent in list-based retrieval. On OTT-QA, graph-based evidence curation provides a 19.9-point EM improvement over strong baselines, demonstrating that organizing retrieval results as structured graphs is critical for multihop reasoning. N2N-GQA achieves 48.80 EM, matching finetuned retrieval models (CORE: 49.0 EM) and approaching heavily optimized systems (COS: 56.9 EM) without any task specific training. This establishes graph-structured evidence organization as essential for scalable, zero-shot multi-hop QA systems and demonstrates that simple, interpretable graph construction can rival sophisticated fine-tuned approaches. 

---
# Probing Multimodal Large Language Models on Cognitive Biases in Chinese Short-Video Misinformation 

**Authors**: Jen-tse Huang, Chang Chen, Shiyang Lai, Wenxuan Wang, Michelle R. Kaufman, Mark Dredze  

**Link**: [PDF](https://arxiv.org/pdf/2601.06600)  

**Abstract**: Short-video platforms have become major channels for misinformation, where deceptive claims frequently leverage visual experiments and social cues. While Multimodal Large Language Models (MLLMs) have demonstrated impressive reasoning capabilities, their robustness against misinformation entangled with cognitive biases remains under-explored. In this paper, we introduce a comprehensive evaluation framework using a high-quality, manually annotated dataset of 200 short videos spanning four health domains. This dataset provides fine-grained annotations for three deceptive patterns, experimental errors, logical fallacies, and fabricated claims, each verified by evidence such as national standards and academic literature. We evaluate eight frontier MLLMs across five modality settings. Experimental results demonstrate that Gemini-2.5-Pro achieves the highest performance in the multimodal setting with a belief score of 71.5/100, while o3 performs the worst at 35.2. Furthermore, we investigate social cues that induce false beliefs in videos and find that models are susceptible to biases like authoritative channel IDs. 

---
# How Context Shapes Truth: Geometric Transformations of Statement-level Truth Representations in LLMs 

**Authors**: Shivam Adarsh, Maria Maistro, Christina Lioma  

**Link**: [PDF](https://arxiv.org/pdf/2601.06599)  

**Abstract**: Large Language Models (LLMs) often encode whether a statement is true as a vector in their residual stream activations. These vectors, also known as truth vectors, have been studied in prior work, however how they change when context is introduced remains unexplored. We study this question by measuring (1) the directional change ($\theta$) between the truth vectors with and without context and (2) the relative magnitude of the truth vectors upon adding context. Across four LLMs and four datasets, we find that (1) truth vectors are roughly orthogonal in early layers, converge in middle layers, and may stabilize or continue increasing in later layers; (2) adding context generally increases the truth vector magnitude, i.e., the separation between true and false representations in the activation space is amplified; (3) larger models distinguish relevant from irrelevant context mainly through directional change ($\theta$), while smaller models show this distinction through magnitude differences. We also find that context conflicting with parametric knowledge produces larger geometric changes than parametrically aligned context. To the best of our knowledge, this is the first work that provides a geometric characterization of how context transforms the truth vector in the activation space of LLMs. 

---
# Detecting LLM-Generated Text with Performance Guarantees 

**Authors**: Hongyi Zhou, Jin Zhu, Ying Yang, Chengchun Shi  

**Link**: [PDF](https://arxiv.org/pdf/2601.06586)  

**Abstract**: Large language models (LLMs) such as GPT, Claude, Gemini, and Grok have been deeply integrated into our daily life. They now support a wide range of tasks -- from dialogue and email drafting to assisting with teaching and coding, serving as search engines, and much more. However, their ability to produce highly human-like text raises serious concerns, including the spread of fake news, the generation of misleading governmental reports, and academic misconduct. To address this practical problem, we train a classifier to determine whether a piece of text is authored by an LLM or a human. Our detector is deployed on an online CPU-based platform this https URL, and contains three novelties over existing detectors: (i) it does not rely on auxiliary information, such as watermarks or knowledge of the specific LLM used to generate the text; (ii) it more effectively distinguishes between human- and LLM-authored text; and (iii) it enables statistical inference, which is largely absent in the current literature. Empirically, our classifier achieves higher classification accuracy compared to existing detectors, while maintaining type-I error control, high statistical power, and computational efficiency. 

---
# Stylistic Evolution and LLM Neutrality in Singlish Language 

**Authors**: Linus Tze En Foo, Weihan Angela Ng, Wenkai Li, Lynnette Hui Xian Ng  

**Link**: [PDF](https://arxiv.org/pdf/2601.06580)  

**Abstract**: Singlish is a creole rooted in Singapore's multilingual environment and continues to evolve alongside social and technological change. This study investigates the evolution of Singlish over a decade of informal digital text messages. We propose a stylistic similarity framework that compares lexico-structural, pragmatic, psycholinguistic, and encoder-derived features across years to quantify temporal variation. Our analysis reveals notable diachronic changes in tone, expressivity and sentence construction over the years. Conversely, while some LLMs were able to generate superficially realistic Singlish messages, they do not produce temporally neutral outputs, and residual temporal signals remain detectable despite prompting and fine-tuning. Our findings highlight the dynamic evolution of Singlish, as well as the capabilities and limitations of current LLMs in modeling sociolectal and temporal variations in the colloquial language. 

---
# Are Emotions Arranged in a Circle? Geometric Analysis of Emotion Representations via Hyperspherical Contrastive Learning 

**Authors**: Yusuke Yamauchi, Akiko Aizawa  

**Link**: [PDF](https://arxiv.org/pdf/2601.06575)  

**Abstract**: Psychological research has long utilized circumplex models to structure emotions, placing similar emotions adjacently and opposing ones diagonally. Although frequently used to interpret deep learning representations, these models are rarely directly incorporated into the representation learning of language models, leaving their geometric validity unexplored. This paper proposes a method to induce circular emotion representations within language model embeddings via contrastive learning on a hypersphere. We show that while this circular alignment offers superior interpretability and robustness against dimensionality reduction, it underperforms compared to conventional designs in high-dimensional settings and fine-grained classification. Our findings elucidate the trade-offs involved in applying psychological circumplex models to deep learning architectures. 

---
# EVM-QuestBench: An Execution-Grounded Benchmark for Natural-Language Transaction Code Generation 

**Authors**: Pei Yang, Wanyi Chen, Ke Wang, Lynn Ai, Eric Yang, Tianyu Shi  

**Link**: [PDF](https://arxiv.org/pdf/2601.06565)  

**Abstract**: Large language models are increasingly applied to various development scenarios. However, in on-chain transaction scenarios, even a minor error can cause irreversible loss for users. Existing evaluations often overlook execution accuracy and safety. We introduce EVM-QuestBench, an execution-grounded benchmark for natural-language transaction-script generation on EVM-compatible chains. The benchmark employs dynamic evaluation: instructions are sampled from template pools, numeric parameters are drawn from predefined intervals, and validators verify outcomes against these instantiated values. EVM-QuestBench contains 107 tasks (62 atomic, 45 composite). Its modular architecture enables rapid task development. The runner executes scripts on a forked EVM chain with snapshot isolation; composite tasks apply step-efficiency decay. We evaluate 20 models and find large performance gaps, with split scores revealing persistent asymmetry between single-action precision and multi-step workflow completion. Code: this https URL. 

---
# CSR-RAG: An Efficient Retrieval System for Text-to-SQL on the Enterprise Scale 

**Authors**: Rajpreet Singh, Novak Boškov, Lawrence Drabeck, Aditya Gudal, Manzoor A. Khan  

**Link**: [PDF](https://arxiv.org/pdf/2601.06564)  

**Abstract**: Natural language to SQL translation (Text-to-SQL) is one of the long-standing problems that has recently benefited from advances in Large Language Models (LLMs). While most academic Text-to-SQL benchmarks request schema description as a part of natural language input, enterprise-scale applications often require table retrieval before SQL query generation. To address this need, we propose a novel hybrid Retrieval Augmented Generation (RAG) system consisting of contextual, structural, and relational retrieval (CSR-RAG) to achieve computationally efficient yet sufficiently accurate retrieval for enterprise-scale databases. Through extensive enterprise benchmarks, we demonstrate that CSR-RAG achieves up to 40% precision and over 80% recall while incurring a negligible average query generation latency of only 30ms on commodity data center hardware, which makes it appropriate for modern LLM-based enterprise-scale systems. 

---
# SimLLM: Fine-Tuning Code LLMs for SimPy-Based Queueing System Simulation 

**Authors**: Jun-Qi Chen, Kun Zhang, Rui Zheng, Ying Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2601.06543)  

**Abstract**: The Python package SimPy is widely used for modeling queueing systems due to its flexibility, simplicity, and smooth integration with modern data analysis and optimization frameworks. Recent advances in large language models (LLMs) have shown strong ability in generating clear and executable code, making them powerful and suitable tools for writing SimPy queueing simulation code. However, directly employing closed-source models like GPT-4o to generate such code may lead to high computational costs and raise data privacy concerns. To address this, we fine-tune two open-source LLMs, Qwen-Coder-7B and DeepSeek-Coder-6.7B, on curated SimPy queueing data, which enhances their code-generating performance in executability, output-format compliance, and instruction-code consistency. Particularly, we proposed a multi-stage fine-tuning framework comprising two stages of supervised fine-tuning (SFT) and one stage of direct preference optimization (DPO), progressively enhancing the model's ability in SimPy-based queueing simulation code generation. Extensive evaluations demonstrate that both fine-tuned models achieve substantial improvements in executability, output-format compliance, and instruct consistency. These results confirm that domain-specific fine-tuning can effectively transform compact open-source code models into reliable SimPy simulation generators which provide a practical alternative to closed-source LLMs for education, research, and operational decision support. 

---
# Exposía: Academic Writing Assessment of Exposés and Peer Feedback 

**Authors**: Dennis Zyska, Alla Rozovskaya, Ilia Kuznetsov, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2601.06536)  

**Abstract**: We present Exposía, the first public dataset that connects writing and feedback assessment in higher education, enabling research on educationally grounded approaches to academic writing evaluation. Exposía includes student research project proposals and peer and instructor feedback consisting of comments and free-text reviews. The dataset was collected in the "Introduction to Scientific Work" course of the Computer Science undergraduate program that focuses on teaching academic writing skills and providing peer feedback on academic writing. Exposía reflects the multi-stage nature of the academic writing process that includes drafting, providing and receiving feedback, and revising the writing based on the feedback received. Both the project proposals and peer feedback are accompanied by human assessment scores based on a fine-grained, pedagogically-grounded schema for writing and feedback assessment that we develop.
We use Exposía to benchmark state-of-the-art open-source large language models (LLMs) for two tasks: automated scoring of (1) the proposals and (2) the student reviews. The strongest LLMs attain high agreement on scoring aspects that require little domain knowledge but degrade on dimensions evaluating content, in line with human agreement values. We find that LLMs align better with the human instructors giving high scores. Finally, we establish that a prompting strategy that scores multiple aspects of the writing together is the most effective, an important finding for classroom deployment. 

---
# Atomic-SNLI: Fine-Grained Natural Language Inference through Atomic Fact Decomposition 

**Authors**: Minghui Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06528)  

**Abstract**: Current Natural Language Inference (NLI) systems primarily operate at the sentence level, providing black-box decisions that lack explanatory power. While atomic-level NLI offers a promising alternative by decomposing hypotheses into individual facts, we demonstrate that the conventional assumption that a hypothesis is entailed only when all its atomic facts are entailed fails in practice due to models' poor performance on fine-grained reasoning. Our analysis reveals that existing models perform substantially worse on atomic level inference compared to sentence level tasks. To address this limitation, we introduce Atomic-SNLI, a novel dataset constructed by decomposing SNLI and enriching it with carefully curated atomic level examples through linguistically informed generation strategies. Experimental results demonstrate that models fine-tuned on Atomic-SNLI achieve significant improvements in atomic reasoning capabilities while maintaining strong sentence level performance, enabling both accurate judgements and transparent, explainable results at the fact level. 

---
# MedRAGChecker: Claim-Level Verification for Biomedical Retrieval-Augmented Generation 

**Authors**: Yuelyu Ji, Min Gu Kwak, Hang Zhang, Xizhi Wu, Chenyu Li, Yanshan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06519)  

**Abstract**: Biomedical retrieval-augmented generation (RAG) can ground LLM answers in medical literature, yet long-form outputs often contain isolated unsupported or contradictory claims with safety implications.
We introduce MedRAGChecker, a claim-level verification and diagnostic framework for biomedical RAG.
Given a question, retrieved evidence, and a generated answer, MedRAGChecker decomposes the answer into atomic claims and estimates claim support by combining evidence-grounded natural language inference (NLI) with biomedical knowledge-graph (KG) consistency signals.
Aggregating claim decisions yields answer-level diagnostics that help disentangle retrieval and generation failures, including faithfulness, under-evidence, contradiction, and safety-critical error rates.
To enable scalable evaluation, we distill the pipeline into compact biomedical models and use an ensemble verifier with class-specific reliability weighting.
Experiments on four biomedical QA benchmarks show that MedRAGChecker reliably flags unsupported and contradicted claims and reveals distinct risk profiles across generators, particularly on safety-critical biomedical relations. 

---
# Spec-o3: A Tool-Augmented Vision-Language Agent for Rare Celestial Object Candidate Vetting via Automated Spectral Inspection 

**Authors**: Minghui Jia, Qichao Zhang, Ali Luo, Linjing Li, Shuo Ye, Hailing Lu, Wen Hou, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.06498)  

**Abstract**: Due to the limited generalization and interpretability of deep learning classifiers, The final vetting of rare celestial object candidates still relies on expert visual inspection--a manually intensive process. In this process, astronomers leverage specialized tools to analyze spectra and construct reliable catalogs. However, this practice has become the primary bottleneck, as it is fundamentally incapable of scaling with the data deluge from modern spectroscopic surveys. To bridge this gap, we propose Spec-o3, a tool-augmented vision-language agent that performs astronomer-aligned spectral inspection via interleaved multimodal chain-of-thought reasoning. Spec-o3 is trained with a two-stage post-training recipe: cold-start supervised fine-tuning on expert inspection trajectories followed by outcome-based reinforcement learning on rare-type verification tasks. Evaluated on five rare-object identification tasks from LAMOST, Spec-o3 establishes a new State-of-the-Art, boosting the macro-F1 score from 28.3 to 76.5 with a 7B parameter base model and outperforming both proprietary VLMs and specialized deep models. Crucially, the agent demonstrates strong generalization to unseen inspection tasks across survey shifts (from LAMOST to SDSS/DESI). Expert evaluations confirm that its reasoning traces are coherent and physically consistent, supporting transparent and trustworthy decision-making. Code, data, and models are available at \href{this https URL}{Project HomePage}. 

---
# IndRegBias: A Dataset for Studying Indian Regional Biases in English and Code-Mixed Social Media Comments 

**Authors**: Debasmita Panda, Akash Anil, Neelesh Kumar Shukla  

**Link**: [PDF](https://arxiv.org/pdf/2601.06477)  

**Abstract**: Warning: This paper consists of examples representing regional biases in Indian regions that might be offensive towards a particular region. While social biases corresponding to gender, race, socio-economic conditions, etc., have been extensively studied in the major applications of Natural Language Processing (NLP), biases corresponding to regions have garnered less attention. This is mainly because of (i) difficulty in the extraction of regional bias datasets, (ii) disagreements in annotation due to inherent human biases, and (iii) regional biases being studied in combination with other types of social biases and often being under-represented. This paper focuses on creating a dataset IndRegBias, consisting of regional biases in an Indian context reflected in users' comments on popular social media platforms, namely Reddit and YouTube. We carefully selected 25,000 comments appearing on various threads in Reddit and videos on YouTube discussing trending topics on regional issues in India. Furthermore, we propose a multilevel annotation strategy to annotate the comments describing the severity of regional biased statements. To detect the presence of regional bias and its severity in IndRegBias, we evaluate open-source Large Language Models (LLMs) and Indic Language Models (ILMs) using zero-shot, few-shot, and fine-tuning strategies. We observe that zero-shot and few-shot approaches show lower accuracy in detecting regional biases and severity in the majority of the LLMs and ILMs. However, the fine-tuning approach significantly enhances the performance of the LLM in detecting Indian regional bias along with its severity. 

---
# PRISP: Privacy-Safe Few-Shot Personalization via Lightweight Adaptation 

**Authors**: Junho Park, Dohoon Kim, Taesup Moon  

**Link**: [PDF](https://arxiv.org/pdf/2601.06471)  

**Abstract**: Large language model (LLM) personalization aims to adapt general-purpose models to individual users. Most existing methods, however, are developed under data-rich and resource-abundant settings, often incurring privacy risks. In contrast, realistic personalization typically occurs after deployment under (i) extremely limited user data, (ii) constrained computational resources, and (iii) strict privacy requirements. We propose PRISP, a lightweight and privacy-safe personalization framework tailored to these constraints. PRISP leverages a Text-to-LoRA hypernetwork to generate task-aware LoRA parameters from task descriptions, and enables efficient user personalization by optimizing a small subset of task-aware LoRA parameters together with minimal additional modules using few-shot user data. Experiments on a few-shot variant of the LaMP benchmark demonstrate that PRISP achieves strong overall performance compared to prior approaches, while reducing computational overhead and eliminating privacy risks. 

---
# LitVISTA: A Benchmark for Narrative Orchestration in Literary Text 

**Authors**: Mingzhe Lu, Yiwen Wang, Yanbing Liu, Qi You, Chong Liu, Ruize Qin, Haoyu Dong, Wenyu Zhang, Jiarui Zhang, Yue Hu, Yunpeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.06445)  

**Abstract**: Computational narrative analysis aims to capture rhythm, tension, and emotional dynamics in literary texts. Existing large language models can generate long stories but overly focus on causal coherence, neglecting the complex story arcs and orchestration inherent in human narratives. This creates a structural misalignment between model- and human-generated narratives. We propose VISTA Space, a high-dimensional representational framework for narrative orchestration that unifies human and model narrative perspectives. We further introduce LitVISTA, a structurally annotated benchmark grounded in literary texts, enabling systematic evaluation of models' narrative orchestration capabilities. We conduct oracle evaluations on a diverse selection of frontier LLMs, including GPT, Claude, Grok, and Gemini. Results reveal systematic deficiencies: existing models fail to construct a unified global narrative view, struggling to jointly capture narrative function and structure. Furthermore, even advanced thinking modes yield only limited gains for such literary narrative understanding. 

---
# Time Travel Engine: A Shared Latent Chronological Manifold Enables Historical Navigation in Large Language Models 

**Authors**: Jingmin An, Wei Liu, Qian Wang, Fang Fang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06437)  

**Abstract**: Time functions as a fundamental dimension of human cognition, yet the mechanisms by which Large Language Models (LLMs) encode chronological progression remain opaque. We demonstrate that temporal information in their latent space is organized not as discrete clusters but as a continuous, traversable geometry. We introduce the Time Travel Engine (TTE), an interpretability-driven framework that projects diachronic linguistic patterns onto a shared chronological manifold. Unlike surface-level prompting, TTE directly modulates latent representations to induce coherent stylistic, lexical, and conceptual shifts aligned with target eras. By parameterizing diachronic evolution as a continuous manifold within the residual stream, TTE enables fluid navigation through period-specific "zeitgeists" while restricting access to future knowledge. Furthermore, experiments across diverse architectures reveal topological isomorphism between the temporal subspaces of Chinese and English-indicating that distinct languages share a universal geometric logic of historical evolution. These findings bridge historical linguistics with mechanistic interpretability, offering a novel paradigm for controlling temporal reasoning in neural networks. 

---
# NC-Bench: An LLM Benchmark for Evaluating Conversational Competence 

**Authors**: Robert J. Moore, Sungeun An, Farhan Ahmed, Jay Pankaj Gala  

**Link**: [PDF](https://arxiv.org/pdf/2601.06426)  

**Abstract**: The Natural Conversation Benchmark (NC-Bench) introduce a new approach to evaluating the general conversational competence of large language models (LLMs). Unlike prior benchmarks that focus on the content of model behavior, NC-Bench focuses on the form and structure of natural conversation. Grounded in the IBM Natural Conversation Framework (NCF), NC-Bench comprises three distinct sets. The Basic Conversation Competence set evaluates fundamental sequence management practices, such as answering inquiries, repairing responses, and closing conversational pairs. The RAG set applies the same sequence management patterns as the first set but incorporates retrieval-augmented generation (RAG). The Complex Request set extends the evaluation to complex requests involving more intricate sequence management patterns. Each benchmark tests a model's ability to produce contextually appropriate conversational actions in response to characteristic interaction patterns. Initial evaluations across 6 open-source models and 14 interaction patterns show that models perform well on basic answering tasks, struggle more with repair tasks (especially repeat), have mixed performance on closing sequences, and find complex multi-turn requests most challenging, with Qwen models excelling on the Basic set and Granite models on the RAG set and the Complex Request set. By operationalizing fundamental principles of human conversation, NC-Bench provides a lightweight, extensible, and theory-grounded framework for assessing and improving the conversational abilities of LLMs beyond topical or task-specific benchmarks. 

---
# Can a Unimodal Language Agent Provide Preferences to Tune a Multimodal Vision-Language Model? 

**Authors**: Sazia Tabasum Mim, Jack Morris, Manish Dhakal, Yanming Xiu, Maria Gorlatova, Yi Ding  

**Link**: [PDF](https://arxiv.org/pdf/2601.06424)  

**Abstract**: To explore a more scalable path for adding multimodal capabilities to existing LLMs, this paper addresses a fundamental question: Can a unimodal LLM, relying solely on text, reason about its own informational needs and provide effective feedback to optimize a multimodal model? To answer this, we propose a method that enables a language agent to give feedback to a vision-language model (VLM) to adapt text generation to the agent's preferences. Our results from different experiments affirm this hypothesis, showing that LLM preference feedback significantly enhances VLM descriptions. Using our proposed method, we find that the VLM can generate multimodal scene descriptions to help the LLM better understand multimodal context, leading to improvements of maximum 13% in absolute accuracy compared to the baseline multimodal approach. Furthermore, a human study validated our AI-driven feedback, showing a 64.6% preference alignment rate between the LLM's choices and human judgments. Extensive experiments provide insights on how and why the method works and its limitations. 

---
# Structured Episodic Event Memory 

**Authors**: Zhengxuan Lu, Dongfang Li, Yukun Shi, Beilun Wang, Longyue Wang, Baotian Hu  

**Link**: [PDF](https://arxiv.org/pdf/2601.06411)  

**Abstract**: Current approaches to memory in Large Language Models (LLMs) predominantly rely on static Retrieval-Augmented Generation (RAG), which often results in scattered retrieval and fails to capture the structural dependencies required for complex reasoning. For autonomous agents, these passive and flat architectures lack the cognitive organization necessary to model the dynamic and associative nature of long-term interaction. To address this, we propose Structured Episodic Event Memory (SEEM), a hierarchical framework that synergizes a graph memory layer for relational facts with a dynamic episodic memory layer for narrative progression. Grounded in cognitive frame theory, SEEM transforms interaction streams into structured Episodic Event Frames (EEFs) anchored by precise provenance pointers. Furthermore, we introduce an agentic associative fusion and Reverse Provenance Expansion (RPE) mechanism to reconstruct coherent narrative contexts from fragmented evidence. Experimental results on the LoCoMo and LongMemEval benchmarks demonstrate that SEEM significantly outperforms baselines, enabling agents to maintain superior narrative coherence and logical consistency. 

---
# Value of Information: A Framework for Human-Agent Communication 

**Authors**: Yijiang River Dong, Tiancheng Hu, Zheng Hui, Caiqi Zhang, Ivan Vulić, Andreea Bobu, Nigel Collier  

**Link**: [PDF](https://arxiv.org/pdf/2601.06407)  

**Abstract**: Large Language Model (LLM) agents deployed for real-world tasks face a fundamental dilemma: user requests are underspecified, yet agents must decide whether to act on incomplete information or interrupt users for clarification. Existing approaches either rely on brittle confidence thresholds that require task-specific tuning, or fail to account for the varying stakes of different decisions. We introduce a decision-theoretic framework that resolves this trade-off through the Value of Information (VoI), enabling agents to dynamically weigh the expected utility gain from asking questions against the cognitive cost imposed on users. Our inference-time method requires no hyperparameter tuning and adapts seamlessly across contexts-from casual games to medical diagnosis. Experiments across four diverse domains (20 Questions, medical diagnosis, flight booking, and e-commerce) show that VoI consistently matches or exceeds the best manually-tuned baselines, achieving up to 1.36 utility points higher in high-cost settings. This work provides a parameter-free framework for adaptive agent communication that explicitly balances task risk, query ambiguity, and user effort. 

---
# Steer Model beyond Assistant: Controlling System Prompt Strength via Contrastive Decoding 

**Authors**: Yijiang River Dong, Tiancheng Hu, Zheng Hui, Nigel Collier  

**Link**: [PDF](https://arxiv.org/pdf/2601.06403)  

**Abstract**: Large language models excel at complex instructions yet struggle to deviate from their helpful assistant persona, as post-training instills strong priors that resist conflicting instructions. We introduce system prompt strength, a training-free method that treats prompt adherence as a continuous control. By contrasting logits from target and default system prompts, we isolate and amplify the behavioral signal unique to the target persona by a scalar factor alpha. Across five diverse benchmarks spanning constraint satisfaction, behavioral control, pluralistic alignment, capability modulation, and stylistic control, our method yields substantial improvements: up to +8.5 strict accuracy on IFEval, +45pp refusal rate on OffTopicEval, and +13% steerability on Prompt-Steering. Our approach enables practitioners to modulate system prompt strength, providing dynamic control over model behavior without retraining. 

---
# MITRA: A Large-Scale Parallel Corpus and Multilingual Pretrained Language Model for Machine Translation and Semantic Retrieval for Pāli, Sanskrit, Buddhist Chinese, and Tibetan 

**Authors**: Sebastian Nehrdich, Kurt Keutzer  

**Link**: [PDF](https://arxiv.org/pdf/2601.06400)  

**Abstract**: Ancient Buddhist literature features frequent, yet often unannotated, textual parallels spread across diverse languages: Sanskrit, Pāli, Buddhist Chinese, Tibetan, and more. The scale of this material makes manual examination prohibitive. We present the MITRA framework, which consists of a novel pipeline for multilingual parallel passage mining, MITRA-parallel, a large-scale corpus of 1.74 million parallel sentence pairs between Sanskrit, Chinese, and Tibetan, and the development of the domain-specific pretrained language model Gemma 2 MITRA. We present Gemma 2 MITRA-MT, a version of this base model fine-tuned on machine translation tasks, reaching state-of-the-art performance for machine translation of these languages into English and outperforming even much larger open-source models. We also present Gemma 2 MITRA-E, a semantic embedding model that shows state-of-the-art performance on a novel, detailed semantic embedding benchmark. We make the parallel dataset, model weights, and semantic similarity benchmark openly available to aid both NLP research and philological studies in Buddhist and classical Asian literature. 

---
# AfriqueLLM: How Data Mixing and Model Architecture Impact Continued Pre-training for African Languages 

**Authors**: Hao Yu, Tianyi Xu, Michael A. Hedderich, Wassim Hamidouche, Syed Waqas Zamir, David Ifeoluwa Adelani  

**Link**: [PDF](https://arxiv.org/pdf/2601.06395)  

**Abstract**: Large language models (LLMs) are increasingly multilingual, yet open models continue to underperform relative to proprietary systems, with the gap most pronounced for African languages. Continued pre-training (CPT) offers a practical route to language adaptation, but improvements on demanding capabilities such as mathematical reasoning often remain limited. This limitation is driven in part by the uneven domain coverage and missing task-relevant knowledge that characterize many low-resource language corpora. We present \texttt{AfriqueLLM}, a suite of open LLMs adapted to 20 African languages through CPT on 26B tokens. We perform a comprehensive empirical study across five base models spanning sizes and architectures, including Llama 3.1, Gemma 3, and Qwen 3, and systematically analyze how CPT data composition shapes downstream performance. In particular, we vary mixtures that include math, code, and synthetic translated data, and evaluate the resulting models on a range of multilingual benchmarks. Our results identify data composition as the primary driver of CPT gains. Adding math, code, and synthetic translated data yields consistent improvements, including on reasoning-oriented evaluations. Within a fixed architecture, larger models typically improve performance, but architectural choices dominate scale when comparing across model families. Moreover, strong multilingual performance in the base model does not reliably predict post-CPT outcomes; robust architectures coupled with task-aligned data provide a more dependable recipe. Finally, our best models improve long-context performance, including document-level translation. Models have been released on [Huggingface](this https URL). 

---
# Talking to Extraordinary Objects: Folktales Offer Analogies for Interacting with Technology 

**Authors**: Martha Larson  

**Link**: [PDF](https://arxiv.org/pdf/2601.06372)  

**Abstract**: Speech and language are valuable for interacting with technology. It would be ideal to be able to decouple their use from anthropomorphization, which has recently met an important moment of reckoning. In the world of folktales, language is everywhere and talking to extraordinary objects is not unusual. This overview presents examples of the analogies that folktales offer. Extraordinary objects in folktales are diverse and also memorable. Language capacity and intelligence are not always connected to humanness. Consideration of folktales can offer inspiration and insight for using speech and language for interacting with technology. 

---
# Average shortest-path length in word-adjacency networks: Chinese versus English 

**Authors**: Jakub Dec, Michał Dolina, Stanisław Drożdż, Jarosław Kwapień, Jin Liu, Tomasz Stanisz  

**Link**: [PDF](https://arxiv.org/pdf/2601.06361)  

**Abstract**: Complex networks provide powerful tools for analyzing and understanding the intricate structures present in various systems, including natural language. Here, we analyze topology of growing word-adjacency networks constructed from Chinese and English literary works written in different periods. Unconventionally, instead of considering dictionary words only, we also include punctuation marks as if they were ordinary words. Our approach is based on two arguments: (1) punctuation carries genuine information related to emotional state, allows for logical grouping of content, provides a pause in reading, and facilitates understanding by avoiding ambiguity, and (2) our previous works have shown that punctuation marks behave like words in a Zipfian analysis and, if considered together with regular words, can improve authorship attribution in stylometric studies. We focus on a functional dependence of the average shortest path length $L(N)$ on a network size $N$ for different epochs and individual novels in their original language as well as for translations of selected novels into the other language. We approximate the empirical results with a growing network model and obtain satisfactory agreement between the two. We also observe that $L(N)$ behaves asymptotically similar for both languages if punctuation marks are included but becomes sizably larger for Chinese if punctuation marks are neglected. 

---
# What Matters When Building Universal Multilingual Named Entity Recognition Models? 

**Authors**: Jonas Golde, Patrick Haller, Alan Akbik  

**Link**: [PDF](https://arxiv.org/pdf/2601.06347)  

**Abstract**: Recent progress in universal multilingual named entity recognition (NER) has been driven by advances in multilingual transformer models and task-specific architectures, loss functions, and training datasets. Despite substantial prior work, we find that many critical design decisions for such models are made without systematic justification, with architectural components, training objectives, and data sources evaluated only in combination rather than in isolation. We argue that these decisions impede progress in the field by making it difficult to identify which choices improve model performance. In this work, we conduct extensive experiments around architectures, transformer backbones, training objectives, and data composition across a wide range of languages. Based on these insights, we introduce Otter, a universal multilingual NER model supporting over 100 languages. Otter achieves consistent improvements over strong multilingual NER baselines, outperforming GLiNER-x-base by 5.3pp in F1 and achieves competitive performance compared to large generative models such as Qwen3-32B, while being substantially more efficient. We release model checkpoints, training and evaluation code to facilitate reproducibility and future research. 

---
# On the Fallacy of Global Token Perplexity in Spoken Language Model Evaluation 

**Authors**: Jeff Chan-Jan Sju, Liang-Hsuan Tseng, Yi-Cheng Lin, Yen-Chun Kuo, Ju-Chieh Chou, Kai-Wei Chang, Hung-yi Lee, Carlos Busso  

**Link**: [PDF](https://arxiv.org/pdf/2601.06329)  

**Abstract**: Generative spoken language models pretrained on large-scale raw audio can continue a speech prompt with appropriate content while preserving attributes like speaker and emotion, serving as foundation models for spoken dialogue. In prior literature, these models are often evaluated using ``global token perplexity'', which directly applies the text perplexity formulation to speech tokens. However, this practice overlooks fundamental differences between speech and text modalities, possibly leading to an underestimation of the speech characteristics. In this work, we propose a variety of likelihood- and generative-based evaluation methods that serve in place of naive global token perplexity. We demonstrate that the proposed evaluations more faithfully reflect perceived generation quality, as evidenced by stronger correlations with human-rated mean opinion scores (MOS). When assessed under the new metrics, the relative performance landscape of spoken language models is reshaped, revealing a significantly reduced gap between the best-performing model and the human topline. Together, these results suggest that appropriate evaluation is critical for accurately assessing progress in spoken language modeling. 

---
# Annotating Dimensions of Social Perception in Text: The First Sentence-Level Dataset of Warmth and Competence 

**Authors**: Mutaz Ayesh, Saif M. Mohammad, Nedjma Ousidhoum  

**Link**: [PDF](https://arxiv.org/pdf/2601.06316)  

**Abstract**: Warmth (W) (often further broken down into Trust (T) and Sociability (S)) and Competence (C) are central dimensions along which people evaluate individuals and social groups (Fiske, 2018). While these constructs are well established in social psychology, they are only starting to get attention in NLP research through word-level lexicons, which do not completely capture their contextual expression in larger text units and discourse. In this work, we introduce Warmth and Competence Sentences (W&C-Sent), the first sentence-level dataset annotated for warmth and competence. The dataset includes over 1,600 English sentence--target pairs annotated along three dimensions: trust and sociability (components of warmth), and competence. The sentences in W&C-Sent are from social media and often express attitudes and opinions about specific individuals or social groups (the targets of our annotations). We describe the data collection, annotation, and quality-control procedures in detail, and evaluate a range of large language models (LLMs) on their ability to identify trust, sociability, and competence in text. W&C-Sent provides a new resource for analyzing warmth and competence in language and supports future research at the intersection of NLP and computational social science. 

---
# A Rising Tide Lifts All Boats: MTQE Rewards for Idioms Improve General Translation Quality 

**Authors**: Ishika Agarwal, Zhenlin He, Dhruva Patil, Dilek Hakkani-Tür  

**Link**: [PDF](https://arxiv.org/pdf/2601.06307)  

**Abstract**: Non-compositional expressions (e.g., idioms, proverbs, and metaphors) pose significant challenges for neural machine translation systems because their meanings cannot be derived from individual words alone. These expressions encode rich, cultural meaning, and have both figurative and literal meanings, making accurate translation difficult. Because models are fairly good at translating compositional text, we investigate GRPO-style fine-tuning using Machine Translation Quality Estimation (MTQE) models as reward functions to train models to better translate idioms. Using Chinese and Hindi idiom datasets, we find that idiom translation abilities improve by ~14 points, general, non-idiomatic translation implicitly improves by ~8 points, and cross-lingual translation abilities (trained on one language, evaluated on another) improves by ~6 points. Overall, our work quantifies the non-compositional translation gap and offers insights for developing LLMs with stronger cross-cultural and figurative language understanding. 

---
# SyntaxMind at BLP-2025 Task 1: Leveraging Attention Fusion of CNN and GRU for Hate Speech Detection 

**Authors**: Md. Shihab Uddin Riad  

**Link**: [PDF](https://arxiv.org/pdf/2601.06306)  

**Abstract**: This paper describes our system used in the BLP-2025 Task 1: Hate Speech Detection. We participated in Subtask 1A and Subtask 1B, addressing hate speech classification in Bangla text. Our approach employs a unified architecture that integrates BanglaBERT embeddings with multiple parallel processing branches based on GRUs and CNNs, followed by attention and dense layers for final classification. The model is designed to capture both contextual semantics and local linguistic cues, enabling robust performance across subtasks. The proposed system demonstrated high competitiveness, obtaining 0.7345 micro F1-Score (2nd place) in Subtask 1A and 0.7317 micro F1-Score (5th place) in Subtask 1B. 

---
# Why LoRA Fails to Forget: Regularized Low-Rank Adaptation Against Backdoors in Language Models 

**Authors**: Hoang-Chau Luong, Lingwei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.06305)  

**Abstract**: Low-Rank Adaptation (LoRA) is widely used for parameter-efficient fine-tuning of large language models, but it is notably ineffective at removing backdoor behaviors from poisoned pretrained models when fine-tuning on clean dataset. Contrary to the common belief that this weakness is caused primarily by low rank, we show that LoRA's vulnerability is fundamentally spectral. Our analysis identifies two key factors: LoRA updates (i) possess insufficient spectral strength, with singular values far below those of pretrained weights, and (ii) exhibit unfavorable spectral alignment, weakly matching clean-task directions while retaining overlap with trigger-sensitive subspaces. We further establish a critical scaling threshold beyond which LoRA can theoretically suppress trigger-induced activations, and we show empirically that standard LoRA rarely reaches this regime. We introduce Regularized Low-Rank Adaptation (RoRA), which improves forgetting by increasing spectral strength and correcting alignment through clean-strengthened regularization, trigger-insensitive constraints, and post-training spectral rescaling. Experiments across multiple NLP benchmarks and attack settings show that RoRA substantially reduces attack success rates while maintaining clean accuracy. 

---
# $\texttt{AMEND++}$: Benchmarking Eligibility Criteria Amendments in Clinical Trials 

**Authors**: Trisha Das, Mandis Beigi, Jacob Aptekar, Jimeng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.06300)  

**Abstract**: Clinical trial amendments frequently introduce delays, increased costs, and administrative burden, with eligibility criteria being the most commonly amended component. We introduce \textit{eligibility criteria amendment prediction}, a novel NLP task that aims to forecast whether the eligibility criteria of an initial trial protocol will undergo future amendments. To support this task, we release $\texttt{AMEND++}$, a benchmark suite comprising two datasets: $\texttt{AMEND}$, which captures eligibility-criteria version histories and amendment labels from public clinical trials, and $\verb|AMEND_LLM|$, a refined subset curated using an LLM-based denoising pipeline to isolate substantive changes. We further propose $\textit{Change-Aware Masked Language Modeling}$ (CAMLM), a revision-aware pretraining strategy that leverages historical edits to learn amendment-sensitive representations. Experiments across diverse baselines show that CAMLM consistently improves amendment prediction, enabling more robust and cost-effective clinical trial design. 

---
# How well can off-the-shelf LLMs elucidate molecular structures from mass spectra using chain-of-thought reasoning? 

**Authors**: Yufeng Wang, Lu Wei, Lin Liu, Hao Xu, Haibin Ling  

**Link**: [PDF](https://arxiv.org/pdf/2601.06289)  

**Abstract**: Mass spectrometry (MS) is a powerful analytical technique for identifying small molecules, yet determining complete molecular structures directly from tandem mass spectra (MS/MS) remains a long-standing challenge due to complex fragmentation patterns and the vast diversity of chemical space. Recent progress in large language models (LLMs) has shown promise for reasoning-intensive scientific tasks, but their capability for chemical interpretation is still unclear. In this work, we introduce a Chain-of-Thought (CoT) prompting framework and benchmark that evaluate how LLMs reason about mass spectral data to predict molecular structures. We formalize expert chemists' reasoning steps-such as double bond equivalent (DBE) analysis, neutral loss identification, and fragment assembly-into structured prompts and assess multiple state-of-the-art LLMs (Claude-3.5-Sonnet, GPT-4o-mini, and Llama-3 series) in a zero-shot setting using the MassSpecGym dataset. Our evaluation across metrics of SMILES validity, formula consistency, and structural similarity reveals that while LLMs can produce syntactically valid and partially plausible structures, they fail to achieve chemical accuracy or link reasoning to correct molecular predictions. These findings highlight both the interpretive potential and the current limitations of LLM-based reasoning for molecular elucidation, providing a foundation for future work that combines domain knowledge and reinforcement learning to achieve chemically grounded AI reasoning. 

---
# Amory: Building Coherent Narrative-Driven Agent Memory through Agentic Reasoning 

**Authors**: Yue Zhou, Xiaobo Guo, Belhassen Bayar, Srinivasan H. Sengamedu  

**Link**: [PDF](https://arxiv.org/pdf/2601.06282)  

**Abstract**: Long-term conversational agents face a fundamental scalability challenge as interactions extend over time: repeatedly processing entire conversation histories becomes computationally prohibitive. Current approaches attempt to solve this through memory frameworks that predominantly fragment conversations into isolated embeddings or graph representations and retrieve relevant ones in a RAG style. While computationally efficient, these methods often treat memory formation minimally and fail to capture the subtlety and coherence of human memory. We introduce Amory, a working memory framework that actively constructs structured memory representations through enhancing agentic reasoning during offline time. Amory organizes conversational fragments into episodic narratives, consolidates memories with momentum, and semanticizes peripheral facts into semantic memory. At retrieval time, the system employs coherence-driven reasoning over narrative structures. Evaluated on the LOCOMO benchmark for long-term reasoning, Amory achieves considerable improvements over previous state-of-the-art, with performance comparable to full context reasoning while reducing response time by 50%. Analysis shows that momentum-aware consolidation significantly enhances response quality, while coherence-driven retrieval provides superior memory coverage compared to embedding-based approaches. 

---
# Is Sanskrit the most token-efficient language? A quantitative study using GPT, Gemini, and SentencePiece 

**Authors**: Anshul Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2601.06142)  

**Abstract**: Tokens are the basic units of Large Language Models (LLMs). LLMs rely on tokenizers to segment text into these tokens, and tokenization is the primary determinant of computational and inference cost. Sanskrit, one of the oldest languages, is hypothesized to express more meaning per token due to its morphology and grammar rules; however, no prior work has quantified this. We use a dataset of 701 parallel verses of the Bhagavad Gita, which comprises three languages-Sanskrit, English, and Hindi along with transliteration of Sanskrit into English. We test tokenizers including SentencePiece (SPM), older GPT models, and the latest generation tokenizers from Gemini and GPT. We use metrics of token count, characters per token (token efficiency), and tokens per character (token cost). Results show a ~2x difference in token counts between Sanskrit and English/Hindi under the unbiased SPM baseline. English/Hindi translations of Sanskrit commentary resulted in an approximately 20x increase in token count. GPT o200k base (latest, used by GPT-4o) and Gemini (latest) reduce bias by a significant degree compared to GPT cl100k base (used until GPT-4), but still fail to fully capture Sanskrit's compactness. This matters because there might be a penalty bias for non-English users, which inflates the token count. This research provides a foundation for improving future tokenizer design and shows the potential of Sanskrit for highly compact encoding, saving on cost while speeding up training and inference. The code and dataset are available at this https URL 

---
# AzeroS: Extending LLM to Speech with Self-Generated Instruction-Free Tuning 

**Authors**: Yiwen Shao, Wei Liu, Jiahong Li, Tianzi Wang, Kun Wei, Meng Yu, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.06086)  

**Abstract**: Extending large language models (LLMs) to the speech domain has recently gained significant attention. A typical approach connects a pretrained LLM with an audio encoder through a projection module and trains the resulting model on large-scale, task-specific instruction-tuning datasets. However, curating such instruction-tuning data for specific requirements is time-consuming, and models trained in this manner often generalize poorly to unseen tasks. In this work, we first formulate that the strongest generalization of a speech-LLM is achieved when it is trained with Self-Generated Instruction-Free Tuning (SIFT), in which supervision signals are generated by a frozen LLM using textual representations of speech as input. Our proposed SIFT paradigm eliminates the need for collecting task-specific question-answer pairs and yields the theoretically best generalization to unseen tasks. Building upon this paradigm, we introduce AZeroS (Auden Zero-instruction-tuned Speech-LLM), which is trained on speech-text pairs derived from publicly available corpora, including approximately 25,000 hours of speech with ASR transcripts and 3,000 hours of speech with paralinguistic labels. Built upon Qwen2.5-7B-Instruct, the model updates only two lightweight projection modules (23.8 million parameters each), while keeping both the LLM and audio encoders frozen. Despite the minimal training cost and modest data scale, AZeroS achieves state-of-the-art performance on both semantic and paralinguistic benchmarks, including VoiceBench, AIR-Bench Foundation (Speech), and AIR-Bench Chat (Speech). 

---
# A Multi-Stage Workflow for the Review of Marketing Content with Reasoning Large Language Models 

**Authors**: Alberto Purpura, Emily Chen, Swapnil Shinde  

**Link**: [PDF](https://arxiv.org/pdf/2601.06054)  

**Abstract**: Reasoning Large Language Models (LLMs) have shown promising results when tasked with solving complex problems. In this paper, we propose and evaluate a multi-stage workflow that leverages the capabilities of fine-tuned reasoning LLMs to assist in the review process of marketing content, making sure they comply with a given list of requirements. The contributions of this paper are the following: (i) we present a novel approach -- that does not rely on any external knowledge representation -- for the automatic identification of compliance issues in textual content; (ii) compare the effectiveness of different fine-tuning strategies like Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO) in training models to solve this problem; (iii) we evaluate the effectiveness of training small LLMs to generate reasoning tokens before providing their final response; (iv) we evaluate how the choice and combinations of different reward functions affects the performance of a model trained with GRPO. 

---
# Reinforcement Learning for Chain of Thought Compression with One-Domain-to-All Generalization 

**Authors**: Hanyu Li, Jiangshan Duo, Bofei Gao, Hailin Zhang, Sujian Li, Xiaotie Deng, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.06052)  

**Abstract**: Chain-of-thought reasoning in large language models often creates an "overthinking trap," leading to excessive computational cost and latency for unreliable accuracy gains. Prior work has typically relied on global, static controls that risk penalizing necessary reasoning. We introduce a sample-level, soft reinforcement learning compression method that penalizes inefficiently long rollouts, but only on problems where the model has already mastered and already produced a more concise rollout. Our experiments show that this method reduces average response length by 20-40% with comparable or higher accuracy. Crucially, the compression exhibits strong cross-domain generalization; a model trained on math spontaneously shortens responses on unseen tasks like code, instruction following, and general knowledge QA, with stable or improved accuracy. We demonstrate a stable post-training curriculum (accuracy-compression-accuracy) that can ultimately produce models that are more accurate and reason more concisely, arguing that such compression method should be a standard phase in developing efficient reasoning models. 

---
# Lexical and Statistical Analysis of Bangla Newspaper and Literature: A Corpus-Driven Study on Diversity, Readability, and NLP Adaptation 

**Authors**: Pramit Bhattacharyya, Arnab Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2601.06041)  

**Abstract**: In this paper, we present a comprehensive corpus-driven analysis of Bangla literary and newspaper texts to investigate their lexical diversity, structural complexity and readability. We undertook Vacaspati and IndicCorp, which are the most extensive literature and newspaper-only corpora for Bangla. We examine key linguistic properties, including the type-token ratio (TTR), hapax legomena ratio (HLR), Bigram diversity, average syllable and word lengths, and adherence to Zipfs Law, for both newspaper (IndicCorp) and literary corpora (Vacaspati).For all the features, such as Bigram Diversity and HLR, despite its smaller size, the literary corpus exhibits significantly higher lexical richness and structural variation. Additionally, we tried to understand the diversity of corpora by building n-gram models and measuring perplexity. Our findings reveal that literary corpora have higher perplexity than newspaper corpora, even for similar sentence sizes. This trend can also be observed for the English newspaper and literature corpus, indicating its generalizability. We also examined how the perfor- mance of models on downstream tasks is influenced by the inclusion of literary data alongside newspaper data. Our findings suggest that inte- grating literary data with newspapers improves the performance of models on various downstream tasks. We have also demonstrated that a literary corpus adheres more closely to global word distribution proper- ties, such as Zipfs law, than a newspaper corpus or a merged corpus of both literary and newspaper texts. Literature corpora also have higher entropy and lower redundancy values compared to a newspaper corpus. We also further assess the readability using Flesch and Coleman-Liau in- dices, showing that literary texts are more complex. 

---
# Operation Veja: Fixing Fundamental Concepts Missing from Modern Roleplaying Training Paradigms 

**Authors**: Yueze Liu, Ajay Nagi Reddy Kumdam, Ronit Kanjilal, Hao Yang, Yichi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06039)  

**Abstract**: Modern roleplaying models are increasingly sophisticated, yet they consistently struggle to capture the essence of believable, engaging characters. We argue this failure stems from training paradigms that overlook the dynamic interplay of a character's internal world. Current approaches, including Retrieval-Augmented Generation (RAG), fact-based priming, literature-based learning, and synthetic data generation, exhibit recurring limitations in modeling the deliberative, value-conflicted reasoning that defines human interaction. In this paper, we identify four core concepts essential for character authenticity: Values, Experiences, Judgments, and Abilities (VEJA). We propose the VEJA framework as a new paradigm for data curation that addresses these systemic limitations. To illustrate the qualitative ceiling enabled by our framework, we present a pilot study comparing a manually curated, VEJA-grounded dataset against a state-of-the-art synthetic baseline. Using an LLM-as-judge evaluation, our findings demonstrate a significant quality gap, suggesting that a shift toward conceptually grounded data curation, as embodied by VEJA, is necessary for creating roleplaying agents with genuine depth and narrative continuity. The full dataset is available at this https URL 

---
# TeleMem: Building Long-Term and Multimodal Memory for Agentic AI 

**Authors**: Chunliang Chen, Ming Guan, Xiao Lin, Jiaxu Li, Qiyi Wang, Xiangyu Chen, Jixiang Luo, Changzhi Sun, Dell Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.06037)  

**Abstract**: Large language models (LLMs) excel at many NLP tasks but struggle to sustain long-term interactions due to limited attention over extended dialogue histories. Retrieval-augmented generation (RAG) mitigates this issue but lacks reliable mechanisms for updating or refining stored memories, leading to schema-driven hallucinations, inefficient write operations, and minimal support for multimodal this http URL address these challenges, we propose TeleMem, a unified long-term and multimodal memory system that maintains coherent user profiles through narrative dynamic extraction, ensuring that only dialogue-grounded information is preserved. TeleMem further introduces a structured writing pipeline that batches, retrieves, clusters, and consolidates memory entries, substantially improving storage efficiency, reducing token usage, and accelerating memory operations. Additionally, a multimodal memory module combined with ReAct-style reasoning equips the system with a closed-loop observe, think, and act process that enables accurate understanding of complex video content in long-term contexts. Experimental results show that TeleMem surpasses the state-of-the-art Mem0 baseline with 19% higher accuracy, 43% fewer tokens, and a 2.1x speedup on the ZH-4O long-term role-play gaming benchmark. 

---
# OS-Symphony: A Holistic Framework for Robust and Generalist Computer-Using Agent 

**Authors**: Bowen Yang, Kaiming Jin, Zhenyu Wu, Zhaoyang Liu, Qiushi Sun, Zehao Li, JingJing Xie, Zhoumianze Liu, Fangzhi Xu, Kanzhi Cheng, Qingyun Li, Yian Wang, Yu Qiao, Zun Wang, Zichen Ding  

**Link**: [PDF](https://arxiv.org/pdf/2601.07779)  

**Abstract**: While Vision-Language Models (VLMs) have significantly advanced Computer-Using Agents (CUAs), current frameworks struggle with robustness in long-horizon workflows and generalization in novel domains. These limitations stem from a lack of granular control over historical visual context curation and the absence of visual-aware tutorial retrieval. To bridge these gaps, we introduce OS-Symphony, a holistic framework that comprises an Orchestrator coordinating two key innovations for robust automation: (1) a Reflection-Memory Agent that utilizes milestone-driven long-term memory to enable trajectory-level self-correction, effectively mitigating visual context loss in long-horizon tasks; (2) Versatile Tool Agents featuring a Multimodal Searcher that adopts a SeeAct paradigm to navigate a browser-based sandbox to synthesize live, visually aligned tutorials, thereby resolving fidelity issues in unseen scenarios. Experimental results demonstrate that OS-Symphony delivers substantial performance gains across varying model scales, establishing new state-of-the-art results on three online benchmarks, notably achieving 65.84% on OSWorld. 

---
# Are LLM Decisions Faithful to Verbal Confidence? 

**Authors**: Jiawei Wang, Yanfei Zhou, Siddartha Devic, Deqing Fu  

**Link**: [PDF](https://arxiv.org/pdf/2601.07767)  

**Abstract**: Large Language Models (LLMs) can produce surprisingly sophisticated estimates of their own uncertainty. However, it remains unclear to what extent this expressed confidence is tied to the reasoning, knowledge, or decision making of the model. To test this, we introduce $\textbf{RiskEval}$: a framework designed to evaluate whether models adjust their abstention policies in response to varying error penalties. Our evaluation of several frontier models reveals a critical dissociation: models are neither cost-aware when articulating their verbal confidence, nor strategically responsive when deciding whether to engage or abstain under high-penalty conditions. Even when extreme penalties render frequent abstention the mathematically optimal strategy, models almost never abstain, resulting in utility collapse. This indicates that calibrated verbal confidence scores may not be sufficient to create trustworthy and interpretable AI systems, as current models lack the strategic agency to convert uncertainty signals into optimal and risk-sensitive decisions. 

---
# Reasoning Models Will Blatantly Lie About Their Reasoning 

**Authors**: William Walden  

**Link**: [PDF](https://arxiv.org/pdf/2601.07663)  

**Abstract**: It has been shown that Large Reasoning Models (LRMs) may not *say what they think*: they do not always volunteer information about how certain parts of the input influence their reasoning. But it is one thing for a model to *omit* such information and another, worse thing to *lie* about it. Here, we extend the work of Chen et al. (2025) to show that LRMs will do just this: they will flatly deny relying on hints provided in the prompt in answering multiple choice questions -- even when directly asked to reflect on unusual (i.e. hinted) prompt content, even when allowed to use hints, and even though experiments *show* them to be using the hints. Our results thus have discouraging implications for CoT monitoring and interpretability. 

---
# Beyond Static Tools: Test-Time Tool Evolution for Scientific Reasoning 

**Authors**: Jiaxuan Lu, Ziyu Kong, Yemin Wang, Rong Fu, Haiyuan Wan, Cheng Yang, Wenjie Lou, Haoran Sun, Lilong Wang, Yankai Jiang, Xiaosong Wang, Xiao Sun, Dongzhan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.07641)  

**Abstract**: The central challenge of AI for Science is not reasoning alone, but the ability to create computational methods in an open-ended scientific world. Existing LLM-based agents rely on static, pre-defined tool libraries, a paradigm that fundamentally fails in scientific domains where tools are sparse, heterogeneous, and intrinsically incomplete. In this paper, we propose Test-Time Tool Evolution (TTE), a new paradigm that enables agents to synthesize, verify, and evolve executable tools during inference. By transforming tools from fixed resources into problem-driven artifacts, TTE overcomes the rigidity and long-tail limitations of static tool libraries. To facilitate rigorous evaluation, we introduce SciEvo, a benchmark comprising 1,590 scientific reasoning tasks supported by 925 automatically evolved tools. Extensive experiments show that TTE achieves state-of-the-art performance in both accuracy and tool efficiency, while enabling effective cross-domain adaptation of computational tools. The code and benchmark have been released at this https URL. 

---
# GRPO with State Mutations: Improving LLM-Based Hardware Test Plan Generation 

**Authors**: Dimple Vijay Kochar, Nathaniel Pinckney, Guan-Ting Liu, Chia-Tung Ho, Chenhui Deng, Haoxing Ren, Brucek Khailany  

**Link**: [PDF](https://arxiv.org/pdf/2601.07593)  

**Abstract**: RTL design often relies heavily on ad-hoc testbench creation early in the design cycle. While large language models (LLMs) show promise for RTL code generation, their ability to reason about hardware specifications and generate targeted test plans remains largely unexplored. We present the first systematic study of LLM reasoning capabilities for RTL verification stimuli generation, establishing a two-stage framework that decomposes test plan generation from testbench execution. Our benchmark reveals that state-of-the-art models, including DeepSeek-R1 and Claude-4.0-Sonnet, achieve only 15.7-21.7% success rates on generating stimuli that pass golden RTL designs. To improve LLM generated stimuli, we develop a comprehensive training methodology combining supervised fine-tuning with a novel reinforcement learning approach, GRPO with State Mutation (GRPO-SMu), which enhances exploration by varying input mutations. Our approach leverages a tree-based branching mutation strategy to construct training data comprising equivalent and mutated trees, moving beyond linear mutation approaches to provide rich learning signals. Training on this curated dataset, our 7B parameter model achieves a 33.3% golden test pass rate and a 13.9% mutation detection rate, representing a 17.6% absolute improvement over baseline and outperforming much larger general-purpose models. These results demonstrate that specialized training methodologies can significantly enhance LLM reasoning capabilities for hardware verification tasks, establishing a foundation for automated sub-unit testing in semiconductor design workflows. 

---
# SCALPEL: Selective Capability Ablation via Low-rank Parameter Editing for Large Language Model Interpretability Analysis 

**Authors**: Zihao Fu, Xufeng Duan, Zhenguang G. Cai  

**Link**: [PDF](https://arxiv.org/pdf/2601.07411)  

**Abstract**: Large language models excel across diverse domains, yet their deployment in healthcare, legal systems, and autonomous decision-making remains limited by incomplete understanding of their internal mechanisms. As these models integrate into high-stakes systems, understanding how they encode capabilities has become fundamental to interpretability research. Traditional approaches identify important modules through gradient attribution or activation analysis, assuming specific capabilities map to specific components. However, this oversimplifies neural computation: modules may contribute to multiple capabilities simultaneously, while single capabilities may distribute across multiple modules. These coarse-grained analyses fail to capture fine-grained, distributed capability encoding. We present SCALPEL (Selective Capability Ablation via Low-rank Parameter Editing for Large language models), a framework representing capabilities as low-rank parameter subspaces rather than discrete modules. Our key insight is that capabilities can be characterized by low-rank modifications distributed across layers and modules, enabling precise capability removal without affecting others. By training LoRA adapters to reduce distinguishing correct from incorrect answers while preserving general language modeling quality, SCALPEL identifies low-rank representations responsible for particular capabilities while remaining disentangled from others. Experiments across diverse capability and linguistic tasks from BLiMP demonstrate that SCALPEL successfully removes target capabilities while preserving general capabilities, providing fine-grained insights into capability distribution across parameter space. Results reveal that capabilities exhibit low-rank structure and can be selectively ablated through targeted parameter-space interventions, offering nuanced understanding of capability encoding in LLMs. 

---
# On Narrative: The Rhetorical Mechanisms of Online Polarisation 

**Authors**: Jan Elfes, Marco Bastos, Luca Maria Aiello  

**Link**: [PDF](https://arxiv.org/pdf/2601.07398)  

**Abstract**: Polarisation research has demonstrated how people cluster in homogeneous groups with opposing opinions. However, this effect emerges not only through interaction between people, limiting communication between groups, but also between narratives, shaping opinions and partisan identities. Yet, how polarised groups collectively construct and negotiate opposing interpretations of reality, and whether narratives move between groups despite limited interactions, remains unexplored. To address this gap, we formalise the concept of narrative polarisation and demonstrate its measurement in 212 YouTube videos and 90,029 comments on the Israeli-Palestinian conflict. Based on structural narrative theory and implemented through a large language model, we extract the narrative roles assigned to central actors in two partisan information environments. We find that while videos produce highly polarised narratives, comments significantly reduce narrative polarisation, harmonising discourse on the surface level. However, on a deeper narrative level, recurring narrative motifs reveal additional differences between partisan groups. 

---
# LRAS: Advanced Legal Reasoning with Agentic Search 

**Authors**: Yujin Zhou, Chuxue Cao, Jinluan Yang, Lijun Wu, Conghui He, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2601.07296)  

**Abstract**: While Large Reasoning Models (LRMs) have demonstrated exceptional logical capabilities in mathematical domains, their application to the legal field remains hindered by the strict requirements for procedural rigor and adherence to legal logic. Existing legal LLMs, which rely on "closed-loop reasoning" derived solely from internal parametric knowledge, frequently suffer from lack of self-awareness regarding their knowledge boundaries, leading to confident yet incorrect conclusions. To address this challenge, we present Legal Reasoning with Agentic Search (LRAS), the first framework designed to transition legal LLMs from static and parametric "closed-loop thinking" to dynamic and interactive "Active Inquiry". By integrating Introspective Imitation Learning and Difficulty-aware Reinforcement Learning, LRAS enables LRMs to identify knowledge boundaries and handle legal reasoning complexity. Empirical results demonstrate that LRAS outperforms state-of-the-art baselines by 8.2-32\%, with the most substantial gains observed in tasks requiring deep reasoning with reliable knowledge. We will release our data and models for further exploration soon. 

---
# Learning to Trust the Crowd: A Multi-Model Consensus Reasoning Engine for Large Language Models 

**Authors**: Pranav Kallem  

**Link**: [PDF](https://arxiv.org/pdf/2601.07245)  

**Abstract**: Large language models (LLMs) achieve strong aver- age performance yet remain unreliable at the instance level, with frequent hallucinations, brittle failures, and poorly calibrated confidence. We study reliability through the lens of multi-model consensus: given responses from several heterogeneous LLMs, can we learn which answer is most likely correct for a given query? We introduce a Multi-Model Consensus Reasoning Engine that treats the set of LLM outputs as input to a supervised meta-learner. The system maps natural language responses into structured features using semantic embeddings, pairwise similarity and clustering statistics, lexical and structural cues, reasoning-quality scores, confidence estimates, and model-specific priors, and then applies gradient-boosted trees, listwise ranking, and graph neural networks over similarity graphs of answers. Using three open-weight LLMs evaluated on compact, resource- constrained subsets of GSM8K, ARC-Challenge, HellaSwag, and TruthfulQA, our best graph-attention-based consensus model improves macro-average accuracy by 4.6 percentage points over the strongest single LLM and by 8.1 points over majority vote, while also yielding lower Brier scores and fewer TruthfulQA hal- lucinations. Ablation and feature-importance analyses show that semantic agreement and clustering features are most influential, with reasoning-quality and model-prior features providing com- plementary gains, suggesting supervised multi-model consensus is a practical route toward more reliable LLM behavior, even in a modest single-machine setup. 

---
# Lost in the Noise: How Reasoning Models Fail with Contextual Distractors 

**Authors**: Seongyun Lee, Yongrae Jo, Minju Seo, Moontae Lee, Minjoon Seo  

**Link**: [PDF](https://arxiv.org/pdf/2601.07226)  

**Abstract**: Recent advances in reasoning models and agentic AI systems have led to an increased reliance on diverse external information. However, this shift introduces input contexts that are inherently noisy, a reality that current sanitized benchmarks fail to capture. We introduce NoisyBench, a comprehensive benchmark that systematically evaluates model robustness across 11 datasets in RAG, reasoning, alignment, and tool-use tasks against diverse noise types, including random documents, irrelevant chat histories, and hard negative distractors. Our evaluation reveals a catastrophic performance drop of up to 80% in state-of-the-art models when faced with contextual distractors. Crucially, we find that agentic workflows often amplify these errors by over-trusting noisy tool outputs, and distractors can trigger emergent misalignment even without adversarial intent. We find that prompting, context engineering, SFT, and outcome-reward only RL fail to ensure robustness; in contrast, our proposed Rationale-Aware Reward (RARE) significantly strengthens resilience by incentivizing the identification of helpful information within noise. Finally, we uncover an inverse scaling trend where increased test-time computation leads to worse performance in noisy settings and demonstrate via attention visualization that models disproportionately focus on distractor tokens, providing vital insights for building the next generation of robust, reasoning-capable agents. 

---
# MAESTRO: Meta-learning Adaptive Estimation of Scalarization Trade-offs for Reward Optimization 

**Authors**: Yang Zhao, Hepeng Wang, Xiao Ding, Yangou Ouyang, Bibo Cai, Kai Xiong, Jinglong Gao, Zhouhao Sun, Li Du, Bing Qin, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.07208)  

**Abstract**: Group-Relative Policy Optimization (GRPO) has emerged as an efficient paradigm for aligning Large Language Models (LLMs), yet its efficacy is primarily confined to domains with verifiable ground truths. Extending GRPO to open-domain settings remains a critical challenge, as unconstrained generation entails multi-faceted and often conflicting objectives - such as creativity versus factuality - where rigid, static reward scalarization is inherently suboptimal. To address this, we propose MAESTRO (Meta-learning Adaptive Estimation of Scalarization Trade-offs for Reward Optimization), which introduces a meta-cognitive orchestration layer that treats reward scalarization as a dynamic latent policy, leveraging the model's terminal hidden states as a semantic bottleneck to perceive task-specific priorities. We formulate this as a contextual bandit problem within a bi-level optimization framework, where a lightweight Conductor network co-evolves with the policy by utilizing group-relative advantages as a meta-reward signal. Across seven benchmarks, MAESTRO consistently outperforms single-reward and static multi-objective baselines, while preserving the efficiency advantages of GRPO, and in some settings even reducing redundant generation. 

---
# Rewarding Creativity: A Human-Aligned Generative Reward Model for Reinforcement Learning in Storytelling 

**Authors**: Zhaoyan Li, Hang Lei, Yujia Wang, Lanbo Liu, Hao Liu, Liang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.07149)  

**Abstract**: While Large Language Models (LLMs) can generate fluent text, producing high-quality creative stories remains challenging. Reinforcement Learning (RL) offers a promising solution but faces two critical obstacles: designing reliable reward signals for subjective storytelling quality and mitigating training instability. This paper introduces the Reinforcement Learning for Creative Storytelling (RLCS) framework to systematically address both challenges. First, we develop a Generative Reward Model (GenRM) that provides multi-dimensional analysis and explicit reasoning about story preferences, trained through supervised fine-tuning on demonstrations with reasoning chains distilled from strong teacher models, followed by GRPO-based refinement on expanded preference data. Second, we introduce an entropy-based reward shaping strategy that dynamically prioritizes learning on confident errors and uncertain correct predictions, preventing overfitting on already-mastered patterns. Experiments demonstrate that GenRM achieves 68\% alignment with human creativity judgments, and RLCS significantly outperforms strong baselines including Gemini-2.5-Pro in overall story quality. This work provides a practical pipeline for applying RL to creative domains, effectively navigating the dual challenges of reward modeling and training stability. 

---
# ReinPool: Reinforcement Learning Pooling Multi-Vector Embeddings for Retrieval System 

**Authors**: Sungguk Cha, DongWook Kim, Mintae Kim, Youngsub Han, Byoung-Ki Jeon, Sangyeob Lee  

**Link**: [PDF](https://arxiv.org/pdf/2601.07125)  

**Abstract**: Multi-vector embedding models have emerged as a powerful paradigm for document retrieval, preserving fine-grained visual and textual details through token-level representations. However, this expressiveness comes at a staggering cost: storing embeddings for every token inflates index sizes by over $1000\times$ compared to single-vector approaches, severely limiting scalability. We introduce \textbf{ReinPool}, a reinforcement learning framework that learns to dynamically filter and pool multi-vector embeddings into compact, retrieval-optimized representations. By training with an inverse retrieval objective and NDCG-based rewards, ReinPool identifies and retains only the most discriminative vectors without requiring manual importance annotations. On the Vidore V2 benchmark across three vision-language embedding models, ReinPool compresses multi-vector representations by $746$--$1249\times$ into single vectors while recovering 76--81\% of full multi-vector retrieval performance. Compared to static mean pooling baselines, ReinPool achieves 22--33\% absolute NDCG@3 improvement, demonstrating that learned selection significantly outperforms heuristic aggregation. 

---
# FinCARDS: Card-Based Analyst Reranking for Financial Document Question Answering 

**Authors**: Yixi Zhou, Fan Zhang, Yu Chen, Haipeng Zhang, Preslav Nakov, Zhuohan Xie  

**Link**: [PDF](https://arxiv.org/pdf/2601.06992)  

**Abstract**: Financial question answering (QA) over long corporate filings requires evidence to satisfy strict constraints on entities, financial metrics, fiscal periods, and numeric values. However, existing LLM-based rerankers primarily optimize semantic relevance, leading to unstable rankings and opaque decisions on long documents. We propose FinCards, a structured reranking framework that reframes financial evidence selection as constraint satisfaction under a finance-aware schema. FinCards represents filing chunks and questions using aligned schema fields (entities, metrics, periods, and numeric spans), enabling deterministic field-level matching. Evidence is selected via a multi-stage tournament reranking with stability-aware aggregation, producing auditable decision traces. Across two corporate filing QA benchmarks, FinCards substantially improves early-rank retrieval over both lexical and LLM-based reranking baselines, while reducing ranking variance, without requiring model fine-tuning or unpredictable inference budgets. Our code is available at this https URL. 

---
# Measuring Social Bias in Vision-Language Models with Face-Only Counterfactuals from Real Photos 

**Authors**: Haodong Chen, Qiang Huang, Jiaqi Zhao, Qiuping Jiang, Xiaojun Chang, Jun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.06931)  

**Abstract**: Vision-Language Models (VLMs) are increasingly deployed in socially consequential settings, raising concerns about social bias driven by demographic cues. A central challenge in measuring such social bias is attribution under visual confounding: real-world images entangle race and gender with correlated factors such as background and clothing, obscuring attribution. We propose a \textbf{face-only counterfactual evaluation paradigm} that isolates demographic effects while preserving real-image realism. Starting from real photographs, we generate counterfactual variants by editing only facial attributes related to race and gender, keeping all other visual factors fixed. Based on this paradigm, we construct \textbf{FOCUS}, a dataset of 480 scene-matched counterfactual images across six occupations and ten demographic groups, and propose \textbf{REFLECT}, a benchmark comprising three decision-oriented tasks: two-alternative forced choice, multiple-choice socioeconomic inference, and numeric salary recommendation. Experiments on five state-of-the-art VLMs reveal that demographic disparities persist under strict visual control and vary substantially across task formulations. These findings underscore the necessity of controlled, counterfactual audits and highlight task design as a critical factor in evaluating social bias in multimodal models. 

---
# TagSpeech: End-to-End Multi-Speaker ASR and Diarization with Fine-Grained Temporal Grounding 

**Authors**: Mingyue Huo, Yiwen Shao, Yuheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06896)  

**Abstract**: We present TagSpeech, a unified LLM-based framework that utilizes Temporal Anchor Grounding for joint multi-speaker ASR and diarization. The framework is built on two key designs: (1) decoupled semantic and speaker streams fine-tuned via Serialized Output Training (SOT) to learn turn-taking dynamics; and (2) an interleaved time anchor mechanism that not only supports fine-grained timestamp prediction but also acts as a synchronization signal between semantic understanding and speaker tracking. Compared to previous works that primarily focus on speaker-attributed ASR or implicit diarization, TagSpeech addresses the challenge of fine-grained speaker-content alignment and explicitly models "who spoke what and when" in an end-to-end manner. Experiments on AMI and AliMeeting benchmarks demonstrate that our method achieves consistent improvements in Diarization Error Rate (DER) over strong end-to-end baselines, including Qwen-Omni and Gemini, particularly in handling complex speech overlaps. Moreover, TagSpeech employs a parameter-efficient training paradigm in which the LLM backbone is frozen and only lightweight projectors are trained, resulting in strong performance with low computational cost. 

---
# An Ubuntu-Guided Large Language Model Framework for Cognitive Behavioral Mental Health Dialogue 

**Authors**: Sontaga G. Forane, Absalom E. Ezugwu, Kevin Igwe, Karen van den Berg  

**Link**: [PDF](https://arxiv.org/pdf/2601.06875)  

**Abstract**: South Africa's escalating mental health crisis, compounded by limited access to culturally responsive care, calls for innovative and contextually grounded interventions. While large language models show considerable promise for mental health support, their predominantly Western-centric training data limit cultural and linguistic applicability in African contexts. This study introduces a proof-of-concept framework that integrates cognitive behavioral therapy with the African philosophy of Ubuntu to create a culturally sensitive, emotionally intelligent, AI-driven mental health dialogue system. Guided by a design science research methodology, the framework applies both deep theoretical and therapeutic adaptations as well as surface-level linguistic and communicative cultural adaptations. Key CBT techniques, including behavioral activation and cognitive restructuring, were reinterpreted through Ubuntu principles that emphasize communal well-being, spiritual grounding, and interconnectedness. A culturally adapted dataset was developed through iterative processes of language simplification, spiritual contextualization, and Ubuntu-based reframing. The fine-tuned model was evaluated through expert-informed case studies, employing UniEval for conversational quality assessment alongside additional measures of CBT reliability and cultural linguistic alignment. Results demonstrate that the model effectively engages in empathetic, context-aware dialogue aligned with both therapeutic and cultural objectives. Although real-time end-user testing has not yet been conducted, the model underwent rigorous review and supervision by domain specialist clinical psychologists. The findings highlight the potential of culturally embedded emotional intelligence to enhance the contextual relevance, inclusivity, and effectiveness of AI-driven mental health interventions across African settings. 

---
# Speak While Watching: Unleashing TRUE Real-Time Video Understanding Capability of Multimodal Large Language Models 

**Authors**: Junyan Lin, Junlong Tong, Hao Wu, Jialiang Zhang, Jinming Liu, Xin Jin, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2601.06843)  

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved strong performance across many tasks, yet most systems remain limited to offline inference, requiring complete inputs before generating outputs. Recent streaming methods reduce latency by interleaving perception and generation, but still enforce a sequential perception-generation cycle, limiting real-time interaction. In this work, we target a fundamental bottleneck that arises when extending MLLMs to real-time video understanding: the global positional continuity constraint imposed by standard positional encoding schemes. While natural in offline inference, this constraint tightly couples perception and generation, preventing effective input-output parallelism. To address this limitation, we propose a parallel streaming framework that relaxes positional continuity through three designs: Overlapped, Group-Decoupled, and Gap-Isolated. These designs enable simultaneous perception and generation, allowing the model to process incoming inputs while producing responses in real time. Extensive experiments reveal that Group-Decoupled achieves the best efficiency-performance balance, maintaining high fluency and accuracy while significantly reducing latency. We further show that the proposed framework yields up to 2x acceleration under balanced perception-generation workloads, establishing a principled pathway toward speak-while-watching real-time systems. We make all our code publicly available: this https URL. 

---
# Benchmarking Egocentric Clinical Intent Understanding Capability for Medical Multimodal Large Language Models 

**Authors**: Shaonan Liu, Guo Yu, Xiaoling Luo, Shiyi Zheng, Wenting Chen, Jie Liu, Linlin Shen  

**Link**: [PDF](https://arxiv.org/pdf/2601.06750)  

**Abstract**: Medical Multimodal Large Language Models (Med-MLLMs) require egocentric clinical intent understanding for real-world deployment, yet existing benchmarks fail to evaluate this critical capability. To address these challenges, we introduce MedGaze-Bench, the first benchmark leveraging clinician gaze as a Cognitive Cursor to assess intent understanding across surgery, emergency simulation, and diagnostic interpretation. Our benchmark addresses three fundamental challenges: visual homogeneity of anatomical structures, strict temporal-causal dependencies in clinical workflows, and implicit adherence to safety protocols. We propose a Three-Dimensional Clinical Intent Framework evaluating: (1) Spatial Intent: discriminating precise targets amid visual noise, (2) Temporal Intent: inferring causal rationale through retrospective and prospective reasoning, and (3) Standard Intent: verifying protocol compliance through safety checks. Beyond accuracy metrics, we introduce Trap QA mechanisms to stress-test clinical reliability by penalizing hallucinations and cognitive sycophancy. Experiments reveal current MLLMs struggle with egocentric intent due to over-reliance on global features, leading to fabricated observations and uncritical acceptance of invalid instructions. 

---
# KASER: Knowledge-Aligned Student Error Simulator for Open-Ended Coding Tasks 

**Authors**: Zhangqi Duan, Nigel Fernandez, Andrew Lan  

**Link**: [PDF](https://arxiv.org/pdf/2601.06633)  

**Abstract**: Open-ended tasks, such as coding problems that are common in computer science education, provide detailed insights into student knowledge. However, training large language models (LLMs) to simulate and predict possible student errors in their responses to these problems can be challenging: they often suffer from mode collapse and fail to fully capture the diversity in syntax, style, and solution approach in student responses. In this work, we present KASER (Knowledge-Aligned Student Error Simulator), a novel approach that aligns errors with student knowledge. We propose a training method based on reinforcement learning using a hybrid reward that reflects three aspects of student code prediction: i) code similarity to the ground-truth, ii) error matching, and iii) code prediction diversity. On two real-world datasets, we perform two levels of evaluation and show that: At the per-student-problem pair level, our method outperforms baselines on code and error prediction; at the per-problem level, our method outperforms baselines on error coverage and simulated code diversity. 

---
# L-RAG: Balancing Context and Retrieval with Entropy-Based Lazy Loading 

**Authors**: Sergii Voloshyn  

**Link**: [PDF](https://arxiv.org/pdf/2601.06551)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as the predominant paradigm for grounding Large Language Model outputs in factual knowledge, effectively mitigating hallucinations. However, conventional RAG systems operate under a "retrieve-always" assumption, querying vector databases for every input regardless of query complexity. This static approach incurs substantial computational overhead and inference latency, particularly problematic for high-throughput production deployments. We introduce L-RAG (Lazy Retrieval-Augmented Generation), an adaptive framework that implements hierarchical context management through entropy-based gating. L-RAG employs a two-tier architecture: queries are first processed with a compact document summary, and expensive chunk retrieval is triggered only when the model's predictive entropy exceeds a calibrated threshold, signaling genuine uncertainty. Through experiments on SQuAD 2.0 (N=500) using the Phi-2 model, we demonstrate that L-RAG provides a tunable accuracy-efficiency trade-off: at a conservative threshold (tau=0.5), L-RAG achieves 78.2% accuracy, matching Standard RAG (77.8%), with 8% retrieval reduction; at a balanced threshold (tau=1.0), retrieval reduction increases to 26% with modest accuracy trade-off (76.0%). Latency analysis shows that L-RAG saves 80-210ms per query when retrieval latency exceeds 500ms. Analysis of entropy distributions reveals statistically significant separation (p < 0.001) between correct predictions (H=1.72) and errors (H=2.20), validating entropy as a reliable uncertainty signal. L-RAG offers a practical, training-free approach toward more efficient RAG deployment, providing system architects with a configurable knob to balance accuracy and throughput requirements. 

---
# BabyVision: Visual Reasoning Beyond Language 

**Authors**: Liang Chen, Weichu Xie, Yiyan Liang, Hongfeng He, Hans Zhao, Zhibo Yang, Zhiqi Huang, Haoning Wu, Haoyu Lu, Y. charles, Yiping Bao, Yuantao Fan, Guopeng Li, Haiyang Shen, Xuanzhong Chen, Wendong Xu, Shuzheng Si, Zefan Cai, Wenhao Chai, Ziqi Huang, Fangfu Liu, Tianyu Liu, Baobao Chang, Xiaobo Hu, Kaiyuan Chen, Yixin Ren, Yang Liu, Yuan Gong, Kuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.06521)  

**Abstract**: While humans develop core visual skills long before acquiring language, contemporary Multimodal LLMs (MLLMs) still rely heavily on linguistic priors to compensate for their fragile visual understanding. We uncovered a crucial fact: state-of-the-art MLLMs consistently fail on basic visual tasks that humans, even 3-year-olds, can solve effortlessly. To systematically investigate this gap, we introduce BabyVision, a benchmark designed to assess core visual abilities independent of linguistic knowledge for MLLMs. BabyVision spans a wide range of tasks, with 388 items divided into 22 subclasses across four key categories. Empirical results and human evaluation reveal that leading MLLMs perform significantly below human baselines. Gemini3-Pro-Preview scores 49.7, lagging behind 6-year-old humans and falling well behind the average adult score of 94.1. These results show despite excelling in knowledge-heavy evaluations, current MLLMs still lack fundamental visual primitives. Progress in BabyVision represents a step toward human-level visual perception and reasoning capabilities. We also explore solving visual reasoning with generation models by proposing BabyVision-Gen and automatic evaluation toolkit. Our code and benchmark data are released at this https URL for reproduction. 

---
# Gecko: An Efficient Neural Architecture Inherently Processing Sequences with Arbitrary Lengths 

**Authors**: Xuezhe Ma, Shicheng Wen, Linghao Jin, Bilge Acun, Ruihang Lai, Bohan Hou, Will Lin, Hao Zhang, Songlin Yang, Ryan Lee, Mengxi Wu, Jonathan May, Luke Zettlemoyer, Carole-Jean Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.06463)  

**Abstract**: Designing a unified neural network to efficiently and inherently process sequential data with arbitrary lengths is a central and challenging problem in sequence modeling. The design choices in Transformer, including quadratic complexity and weak length extrapolation, have limited their ability to scale to long sequences. In this work, we propose Gecko, a neural architecture that inherits the design of Mega and Megalodon (exponential moving average with gated attention), and further introduces multiple technical components to improve its capability to capture long range dependencies, including timestep decay normalization, sliding chunk attention mechanism, and adaptive working memory. In a controlled pretraining comparison with Llama2 and Megalodon in the scale of 7 billion parameters and 2 trillion training tokens, Gecko achieves better efficiency and long-context scalability. Gecko reaches a training loss of 1.68, significantly outperforming Llama2-7B (1.75) and Megalodon-7B (1.70), and landing close to Llama2-13B (1.67). Notably, without relying on any context-extension techniques, Gecko exhibits inherent long-context processing and retrieval capabilities, stably handling sequences of up to 4 million tokens and retrieving information from contexts up to $4\times$ longer than its attention window. Code: this https URL 

---
# Tone Matters: The Impact of Linguistic Tone on Hallucination in VLMs 

**Authors**: Weihao Hong, Zhiyuan Jiang, Bingyu Shen, Xinlei Guan, Yangyi Feng, Meng Xu, Boyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.06460)  

**Abstract**: Vision-Language Models (VLMs) are increasingly used in safety-critical applications that require reliable visual grounding. However, these models often hallucinate details that are not present in the image to satisfy user prompts. While recent datasets and benchmarks have been introduced to evaluate systematic hallucinations in VLMs, many hallucination behaviors remain insufficiently characterized. In particular, prior work primarily focuses on object presence or absence, leaving it unclear how prompt phrasing and structural constraints can systematically induce hallucinations. In this paper, we investigate how different forms of prompt pressure influence hallucination behavior. We introduce Ghost-100, a procedurally generated dataset of synthetic scenes in which key visual details are deliberately removed, enabling controlled analysis of absence-based hallucinations. Using a structured 5-Level Prompt Intensity Framework, we vary prompts from neutral queries to toxic demands and rigid formatting constraints. We evaluate three representative open-weight VLMs: MiniCPM-V 2.6-8B, Qwen2-VL-7B, and Qwen3-VL-8B. Across all three models, hallucination rates do not increase monotonically with prompt intensity. All models exhibit reductions at higher intensity levels at different thresholds, though not all show sustained reduction under maximum coercion. These results suggest that current safety alignment is more effective at detecting semantic hostility than structural coercion, revealing model-specific limitations in handling compliance pressure. Our dataset is available at: this https URL 

---
# BizFinBench.v2: A Unified Dual-Mode Bilingual Benchmark for Expert-Level Financial Capability Alignment 

**Authors**: Xin Guo, Rongjunchen Zhang, Guilong Lu, Xuntao Guo, Shuai Jia, Zhi Yang, Liwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06401)  

**Abstract**: Large language models have undergone rapid evolution, emerging as a pivotal technology for intelligence in financial operations. However, existing benchmarks are often constrained by pitfalls such as reliance on simulated or general-purpose samples and a focus on singular, offline static scenarios. Consequently, they fail to align with the requirements for authenticity and real-time responsiveness in financial services, leading to a significant discrepancy between benchmark performance and actual operational efficacy. To address this, we introduce BizFinBench.v2, the first large-scale evaluation benchmark grounded in authentic business data from both Chinese and U.S. equity markets, integrating online assessment. We performed clustering analysis on authentic user queries from financial platforms, resulting in eight fundamental tasks and two online tasks across four core business scenarios, totaling 29,578 expert-level Q&A pairs. Experimental results demonstrate that ChatGPT-5 achieves a prominent 61.5% accuracy in main tasks, though a substantial gap relative to financial experts persists; in online tasks, DeepSeek-R1 outperforms all other commercial LLMs. Error analysis further identifies the specific capability deficiencies of existing models within practical financial business contexts. BizFinBench.v2 transcends the limitations of current benchmarks, achieving a business-level deconstruction of LLM financial capabilities and providing a precise basis for evaluating efficacy in the widespread deployment of LLMs within the financial domain. The data and code are available at this https URL. 

---
# Monkey Jump : MoE-Style PEFT for Efficient Multi-Task Learning 

**Authors**: Nusrat Jahan Prottasha, Md Kowsher, Chun-Nam Yu, Chen Chen, Ozlem Garibay  

**Link**: [PDF](https://arxiv.org/pdf/2601.06356)  

**Abstract**: Mixture-of-experts variants of parameter-efficient fine-tuning enable per-token specialization, but they introduce additional trainable routers and expert parameters, increasing memory usage and training cost. This undermines the core goal of parameter-efficient fine-tuning. We propose Monkey Jump, a method that brings mixture-of-experts-style specialization to parameter-efficient fine-tuning without introducing extra trainable parameters for experts or routers. Instead of adding new adapters as experts, Monkey Jump treats the adapters already present in each Transformer block (such as query, key, value, up, and down projections) as implicit experts and routes tokens among them. Routing is performed using k-means clustering with exponentially moving averaged cluster centers, requiring no gradients and no learned parameters. We theoretically show that token-wise routing increases expressivity and can outperform shared adapters by avoiding cancellation effects. Across multi-task experiments covering 14 text, 14 image, and 19 video benchmarks, Monkey Jump achieves competitive performance with mixture-of-experts-based parameter-efficient fine-tuning methods while using 7 to 29 times fewer trainable parameters, up to 48 percent lower memory consumption, and 1.5 to 2 times faster training. Monkey Jump is architecture-agnostic and can be applied to any adapter-based parameter-efficient fine-tuning method. 

---
# SPINAL -- Scaling-law and Preference Integration in Neural Alignment Layers 

**Authors**: Arion Das, Partha Pratim Saha, Amit Dhanda, Vinija Jain, Aman Chadha, Amitava Das  

**Link**: [PDF](https://arxiv.org/pdf/2601.06238)  

**Abstract**: Direct Preference Optimization (DPO) is a principled, scalable alternative to RLHF for aligning large language models from pairwise preferences, but its internal geometric footprint remains undercharacterized, limiting audits, checkpoint comparisons, and failure prediction. We introduce SPINAL (Scaling-law and Preference Integration in Neural Alignment Layers), a diagnostic that measures how alignment reshapes representations across depth by tracing localized structural change layer by layer. Across model families, DPO produces a layerwise calibration effect concentrated in the final decoder blocks (often layers 21-30), where preference gradients most directly affect the next-token distribution. SPINAL encodes each checkpoint as a depth trace over (layer index, contraction score, transport score). The contraction score summarizes how quickly the tail of a layer's spectrum decays (how fast small modes vanish); higher values indicate stronger contraction into fewer effective directions. The transport score summarizes how much the token distribution shifts between adjacent layers using a bounded overlap measure; lower values indicate shorter, smoother steps through representation space. Aligned checkpoints show a late-layer ramp-up in contraction and a smooth reduction in transport, consistent with tightened and stabilized policy mass, while unaligned models trace higher-curvature, more entropic, and geometrically incoherent depth paths. Overall, alignment is geometrically localized: the final layers encode the dominant preference-induced corrections. SPINAL turns this localization into a practical audit signal, quantifying where alignment concentrates, how strongly it manifests, and when it begins to destabilize during training. 

---
# Classroom AI: Large Language Models as Grade-Specific Teachers 

**Authors**: Jio Oh, Steven Euijong Whang, James Evans, Jindong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06225)  

**Abstract**: Large Language Models (LLMs) offer a promising solution to complement traditional teaching and address global teacher shortages that affect hundreds of millions of children, but they fail to provide grade-appropriate responses for students at different educational levels. We introduce a framework for finetuning LLMs to generate age-appropriate educational content across six grade levels, from lower elementary to adult education. Our framework successfully adapts explanations to match students' comprehension capacities without sacrificing factual correctness. This approach integrates seven established readability metrics through a clustering method and builds a comprehensive dataset for grade-specific content generation. Evaluations across multiple datasets with 208 human participants demonstrate substantial improvements in grade-level alignment, achieving a 35.64 percentage point increase compared to prompt-based methods while maintaining response accuracy. AI-assisted learning tailored to different grade levels has the potential to advance educational engagement and equity. 

---
# Manifold-based Sampling for In-Context Hallucination Detection in Large Language Models 

**Authors**: Bodla Krishna Vamshi, Rohan Bhatnagar, Haizhao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06196)  

**Abstract**: Large language models (LLMs) frequently generate factually incorrect or unsupported content, commonly referred to as hallucinations. Prior work has explored decoding strategies, retrieval augmentation, and supervised fine-tuning for hallucination detection, while recent studies show that in-context learning (ICL) can substantially influence factual reliability. However, existing ICL demonstration selection methods often rely on surface-level similarity heuristics and exhibit limited robustness across tasks and models.
We propose MB-ICL, a manifold-based demonstration sampling framework for selecting in-context demonstrations that leverages latent representations extracted from frozen LLMs. By jointly modeling local manifold structure and class-aware prototype geometry, MB-ICL selects demonstrations based on their proximity to learned prototypes rather than lexical or embedding similarity alone.
Across factual verification (FEVER) and hallucination detection (HaluEval) benchmarks, MB-ICL outperforms standard ICL selection baselines in the majority of evaluated settings, with particularly strong gains on dialogue and summarization tasks. The method remains robust under temperature perturbations and model variation, indicating improved stability compared to heuristic retrieval strategies. While lexical retrieval can remain competitive in certain question-answering regimes, our results demonstrate that manifold-based prototype selection provides a reliable and training light approach for hallucination detection without modifying LLM parameters, offering a principled direction for improved ICL demonstration selection. 

---
# Political Alignment in Large Language Models: A Multidimensional Audit of Psychometric Identity and Behavioral Bias 

**Authors**: Adib Sakhawat, Tahsin Islam, Takia Farhin, Syed Rifat Raiyan, Hasan Mahmud, Md Kamrul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2601.06194)  

**Abstract**: As large language models (LLMs) are increasingly integrated into social decision-making, understanding their political positioning and alignment behavior is critical for safety and fairness. This study presents a sociotechnical audit of 26 prominent LLMs, triangulating their positions across three psychometric inventories (Political Compass, SapplyValues, 8 Values) and evaluating their performance on a large-scale news labeling task ($N \approx 27{,}000$). Our results reveal a strong clustering of models in the Libertarian-Left region of the ideological space, encompassing 96.3% of the cohort. Alignment signals appear to be consistent architectural traits rather than stochastic noise ($\eta^2 > 0.90$); however, we identify substantial discrepancies in measurement validity. In particular, the Political Compass exhibits a strong negative correlation with cultural progressivism ($r=-0.64$) when compared against multi-axial instruments, suggesting a conflation of social conservatism with authoritarianism in this context. We further observe a significant divergence between open-weights and closed-source models, with the latter displaying markedly higher cultural progressivism scores ($p<10^{-25}$). In downstream media analysis, models exhibit a systematic "center-shift," frequently categorizing neutral articles as left-leaning, alongside an asymmetric detection capability in which "Far Left" content is identified with greater accuracy (19.2%) than "Far Right" content (2.0%). These findings suggest that single-axis evaluations are insufficient and that multidimensional auditing frameworks are necessary to characterize alignment behavior in deployed LLMs. Our code and data will be made public. 

---
# MLB: A Scenario-Driven Benchmark for Evaluating Large Language Models in Clinical Applications 

**Authors**: Qing He, Dongsheng Bi, Jianrong Lu, Minghui Yang, Zixiao Chen, Jiacheng Lu, Jing Chen, Nannan Du, Xiao Cu, Sijing Wu, Peng Xiang, Yinyin Hu, Yi Guo, Chunpu Li, Shaoyang Li, Zhuo Dong, Ming Jiang, Shuai Guo, Liyun Feng, Jin Peng, Jian Wang, Jinjie Gu, Junwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.06193)  

**Abstract**: The proliferation of Large Language Models (LLMs) presents transformative potential for healthcare, yet practical deployment is hindered by the absence of frameworks that assess real-world clinical utility. Existing benchmarks test static knowledge, failing to capture the dynamic, application-oriented capabilities required in clinical practice. To bridge this gap, we introduce a Medical LLM Benchmark MLB, a comprehensive benchmark evaluating LLMs on both foundational knowledge and scenario-based reasoning. MLB is structured around five core dimensions: Medical Knowledge (MedKQA), Safety and Ethics (MedSE), Medical Record Understanding (MedRU), Smart Services (SmartServ), and Smart Healthcare (SmartCare). The benchmark integrates 22 datasets (17 newly curated) from diverse Chinese clinical sources, covering 64 clinical specialties. Its design features a rigorous curation pipeline involving 300 licensed physicians. Besides, we provide a scalable evaluation methodology, centered on a specialized judge model trained via Supervised Fine-Tuning (SFT) on expert annotations. Our comprehensive evaluation of 10 leading models reveals a critical translational gap: while the top-ranked model, Kimi-K2-Instruct (77.3% accuracy overall), excels in structured tasks like information extraction (87.8% accuracy in MedRU), performance plummets in patient-facing scenarios (61.3% in SmartServ). Moreover, the exceptional safety score (90.6% in MedSE) of the much smaller Baichuan-M2-32B highlights that targeted training is equally critical. Our specialized judge model, trained via SFT on a 19k expert-annotated medical dataset, achieves 92.1% accuracy, an F1-score of 94.37%, and a Cohen's Kappa of 81.3% for human-AI consistency, validating a reproducible and expert-aligned evaluation protocol. MLB thus provides a rigorous framework to guide the development of clinically viable LLMs. 

---
# Attention Mechanism and Heuristic Approach: Context-Aware File Ranking Using Multi-Head Self-Attention 

**Authors**: Pradeep Kumar Sharma, Shantanu Godbole, Sarada Prasad Jena, Hritvik Shrivastava  

**Link**: [PDF](https://arxiv.org/pdf/2601.06185)  

**Abstract**: The identification and ranking of impacted files within software reposi-tories is a key challenge in change impact analysis. Existing deterministic approaches that combine heuristic signals, semantic similarity measures, and graph-based centrality metrics have demonstrated effectiveness in nar-rowing candidate search spaces, yet their recall plateaus. This limitation stems from the treatment of features as linearly independent contributors, ignoring contextual dependencies and relationships between metrics that characterize expert reasoning patterns. To address this limitation, we propose the application of Multi-Head Self-Attention as a post-deterministic scoring refinement mechanism. Our approach learns contextual weighting between features, dynamically adjust-ing importance levels per file based on relational behavior exhibited across candidate file sets. The attention mechanism produces context-aware adjustments that are additively combined with deterministic scores, pre-serving interpretability while enabling reasoning similar to that performed by experts when reviewing change surfaces. We focus on recall rather than precision, as false negatives (missing impacted files) are far more costly than false positives (irrelevant files that can be quickly dismissed during review). Empirical evaluation on 200 test cases demonstrates that the introduc-tion of self-attention improves Top-50 recall from approximately 62-65% to between 78-82% depending on repository complexity and structure, achiev-ing 80% recall at Top-50 files. Expert validation yields improvement from 6.5/10 to 8.6/10 in subjective accuracy alignment. This transformation bridges the reasoning capability gap between deterministic automation and expert judgment, improving recall in repository-aware effort estimation. 

---
# MixDPO: Modeling Preference Strength for Pluralistic Alignment 

**Authors**: Saki Imai, Pedram Heydari, Anthony Sicilia, Asteria Kaeberlein, Katherine Atwell, Malihe Alikhani  

**Link**: [PDF](https://arxiv.org/pdf/2601.06180)  

**Abstract**: Preference based alignment objectives implicitly assume that all human preferences are expressed with equal strength. In practice, however, preference strength varies across individuals and contexts -- a phenomenon established in behavioral economics and discrete choice theory. This mismatch limits the ability of existing objectives to faithfully capture heterogeneous human judgments. Inspired by this literature, we introduce Mixed Logit Direct Preference Optimization (MixDPO), a generalization of Direct Preference Optimization that models variation in preference strength. MixDPO enables alignment objectives to capture heterogeneity in how strongly preferences are expressed across training examples. We evaluate MixDPO on three preference datasets using two open-weight language models. Across datasets, MixDPO improves aggregate alignment performance (+11.2 points on Pythia-2.8B) while preserving subgroup level preferences, with the largest gains appearing in settings with higher inferred preference heterogeneity. MixDPO makes preference heterogeneity explicit through learned strength distributions. We release our code for reproducibility. 

---
# PromptPort: A Reliability Layer for Cross-Model Structured Extraction 

**Authors**: Varun Kotte  

**Link**: [PDF](https://arxiv.org/pdf/2601.06151)  

**Abstract**: Structured extraction with LLMs fails in production not because models lack understanding, but because output formatting is unreliable across models and prompts. A prompt that returns clean JSON on GPT-4 may produce fenced, prose-wrapped, or malformed output on Llama, causing strict parsers to reject otherwise correct extractions. We formalize this as format collapse and introduce a dual-metric evaluation framework: ROS (strict parsing, measuring operational reliability) and CSS (post-canonicalization, measuring semantic capability). On a 37,346-example camera metadata benchmark across six model families, we find severe format collapse (for example, Gemma-2B: ROS 0.116 versus CSS 0.246) and large cross-model portability gaps (0.4 to 0.6 F1). We then present PromptPort, a reliability layer combining deterministic canonicalization with a lightweight verifier (DistilBERT) and a safe-override policy. PromptPort recovers format failures (plus 6 to 8 F1), adds verifier-driven semantic selection (plus 14 to 16 F1 beyond canonicalization), and approaches per-field oracle performance (0.890 versus 0.896 in zero-shot) without modifying base models. The method generalizes to held-out model families and provides explicit abstention when uncertain, enabling reliable structured extraction in production deployments. 

---
# LLM Flow Processes for Text-Conditioned Regression 

**Authors**: Felix Biggs, Samuel Willis  

**Link**: [PDF](https://arxiv.org/pdf/2601.06147)  

**Abstract**: Meta-learning methods for regression like Neural (Diffusion) Processes achieve impressive results, but with these models it can be difficult to incorporate expert prior knowledge and information contained in metadata. Large Language Models (LLMs) are trained on giant corpora including varied real-world regression datasets alongside their descriptions and metadata, leading to impressive performance on a range of downstream tasks. Recent work has extended this to regression tasks and is able to leverage such prior knowledge and metadata, achieving surprisingly good performance, but this still rarely matches dedicated meta-learning methods. Here we introduce a general method for sampling from a product-of-experts of a diffusion or flow matching model and an `expert' with binned probability density; we apply this to combine neural diffusion processes with LLM token probabilities for regression (which may incorporate textual knowledge), exceeding the empirical performance of either alone. 

---
# An evaluation of LLMs for political bias in Western media: Israel-Hamas and Ukraine-Russia wars 

**Authors**: Rohitash Chandra, Haoyan Chen, Yaqing Zhang, Jiacheng Chen, Yuting Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.06132)  

**Abstract**: Political bias in media plays a critical role in shaping public opinion, voter behaviour, and broader democratic discourse. Subjective opinions and political bias can be found in media sources, such as newspapers, depending on their funding mechanisms and alliances with political parties. Automating the detection of political biases in media content can limit biases in elections. The impact of large language models (LLMs) in politics and media studies is becoming prominent. In this study, we utilise LLMs to compare the left-wing, right-wing, and neutral political opinions expressed in the Guardian and BBC. We review newspaper reporting that includes significant events such as the Russia-Ukraine war and the Hamas-Israel conflict. We analyse the proportion for each opinion to find the bias under different LLMs, including BERT, Gemini, and DeepSeek. Our results show that after the outbreak of the wars, the political bias of Western media shifts towards the left-wing and each LLM gives a different result. DeepSeek consistently showed a stable Left-leaning tendency, while BERT and Gemini remained closer to the Centre. The BBC and The Guardian showed distinct reporting behaviours across the two conflicts. In the Russia-Ukraine war, both outlets maintained relatively stable positions; however, in the Israel-Hamas conflict, we identified larger political bias shifts, particularly in Guardian coverage, suggesting a more event-driven pattern of reporting bias. These variations suggest that LLMs are shaped not only by their training data and architecture, but also by underlying worldviews with associated political biases. 

---
# Structure-Aware Diversity Pursuit as an AI Safety Strategy against Homogenization 

**Authors**: Ian Rios-Sialer  

**Link**: [PDF](https://arxiv.org/pdf/2601.06116)  

**Abstract**: Generative AI models reproduce the biases in the training data and can further amplify them through mode collapse. We refer to the resulting harmful loss of diversity as homogenization. Our position is that homogenization should be a primary concern in AI safety. We introduce xeno-reproduction as the strategy that mitigates homogenization. For auto-regressive LLMs, we formalize xeno-reproduction as a structure-aware diversity pursuit. Our contribution is foundational, intended to open an essential line of research and invite collaboration to advance diversity. 

---
# From RLHF to Direct Alignment: A Theoretical Unification of Preference Learning for Large Language Models 

**Authors**: Tarun Raheja, Nilay Pochhi  

**Link**: [PDF](https://arxiv.org/pdf/2601.06108)  

**Abstract**: Aligning large language models (LLMs) with human preferences has become essential for safe and beneficial AI deployment. While Reinforcement Learning from Human Feedback (RLHF) established the dominant paradigm, a proliferation of alternatives -- Direct Preference Optimization (DPO), Identity Preference Optimization (IPO), Kahneman-Tversky Optimization (KTO), Simple Preference Optimization (SimPO), and many others -- has left practitioners without clear guidance on method selection. This survey provides a \textit{theoretical unification} of preference learning methods, revealing that the apparent diversity reduces to principled choices along three orthogonal axes: \textbf{(I) Preference Model} (what likelihood model underlies the objective), \textbf{(II) Regularization Mechanism} (how deviation from reference policies is controlled), and \textbf{(III) Data Distribution} (online vs.\ offline learning and coverage requirements). We formalize each axis with precise definitions and theorems, establishing key results including the coverage separation between online and offline methods, scaling laws for reward overoptimization, and conditions under which direct alignment methods fail. Our analysis reveals that failure modes -- length hacking, mode collapse, likelihood displacement -- arise from specific, predictable combinations of design choices. We synthesize empirical findings across 50+ papers and provide a practitioner's decision guide for method selection. The framework transforms preference learning from an empirical art into a theoretically grounded discipline. 

---
# Judge Model for Large-scale Multimodality Benchmarks 

**Authors**: Min-Han Shih, Yu-Hsin Wu, Yu-Wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.06106)  

**Abstract**: We propose a dedicated multimodal Judge Model designed to provide reliable, explainable evaluation across a diverse suite of tasks. Our benchmark spans text, audio, image, and video modalities, drawing from carefully sampled public datasets with fixed seeds to ensure reproducibility and minimize train test leakage. Instead of simple scoring, our framework aggregates multimodal judgments, analyzes the quality and reasoning consistency of model outputs, and generates diagnostic feedback. We evaluate several MLLMs, including Gemini 2.5, Phi 4, and Qwen 2.5, across 280 multimodal samples and compare judge model assessments with human annotators. Results show strong alignment between the Judge Model and human scores, demonstrating its potential as a scalable, interpretable evaluation pipeline for future multimodal AI research. 

---
# Comment on arXiv:2511.21731v1: Identifying Quantum Structure in AI Language: Evidence for Evolutionary Convergence of Human and Artificial Cognition 

**Authors**: Krzysztof Sienicki  

**Link**: [PDF](https://arxiv.org/pdf/2601.06104)  

**Abstract**: This note is a friendly technical check of arXiv:2511.21731v1. I highlight a few places where the manuscript's interpretation of (i) the reported CHSH/Bell-type calculations and (ii) Bose--Einstein (BE) fits to rank-frequency data seems to go beyond what the stated procedures can firmly support. I also point out one internal inconsistency in the "energy-level spacing" analogy. The aim is constructive: to keep the interesting empirical observations, while making clear what they do (and do not) imply about quantum entanglement in the usual Hilbert-space sense, especially when "energy" is defined by rank. 

---
# Filtering Beats Fine Tuning: A Bayesian Kalman View of In Context Learning in LLMs 

**Authors**: Andrew Kiruluta  

**Link**: [PDF](https://arxiv.org/pdf/2601.06100)  

**Abstract**: We present a theory-first framework that interprets inference-time adaptation in large language models (LLMs) as online Bayesian state estimation. Rather than modeling rapid adaptation as implicit optimization or meta-learning, we formulate task- and context-specific learning as the sequential inference of a low-dimensional latent adaptation state governed by a linearized state-space model. Under Gaussian assumptions, adaptation follows a Kalman recursion with closed-form updates for both the posterior mean and covariance.
This perspective elevates epistemic uncertainty to an explicit dynamical variable. We show that inference-time learning is driven by covariance collapse, i.e., rapid contraction of posterior uncertainty induced by informative tokens, which typically precedes convergence of the posterior mean. Using observability conditions on token-level Jacobians, we establish stability of the Bayesian filter, prove exponential covariance contraction rates, and derive mean-square error bounds. Gradient descent, natural-gradient methods, and meta-learning updates arise as singular, noise-free limits of the filtering dynamics, positioning optimization-based adaptation as a degenerate approximation of Bayesian inference.
The resulting theory provides a unified probabilistic account of in-context learning, parameter-efficient adaptation, and test-time learning without parameter updates. It yields explicit guarantees on stability and sample efficiency, offers a principled interpretation of prompt informativeness via information accumulation, and clarifies the role of uncertainty dynamics absent from existing accounts. Minimal illustrative experiments corroborate the qualitative predictions of the theory. 

---
# La norme technique comme catalyseur de transfert de connaissances : la francophonie a l'œuvre dans le domaine de l'{é}ducation 

**Authors**: Mokhtar Ben Henda  

**Link**: [PDF](https://arxiv.org/pdf/2601.06069)  

**Abstract**: Standards are adopted in a wide range of fields, both technical and industrial, as well as socio-economic, cultural and linguistic. They are presented explicitly as laws and regulations, technical and industrial standards or implicitly in the form of unwritten social standards. However, in a globalization marked by a very fine mosaic of socio-cultural identities, the question arises in relation to the construction of global, transparent and coherent systems in which considerable work of consensus is necessary to ensure all types of transfers and their local adaptations. The focus here is on the global education ecosystem which develops its own standards for the transfer of knowledge and socio-cultural values through learning, teaching and training. Subcommittee 36 of the International Organization for Standardization is one of the structures of this ecosystem in which the Francophonie participates to develop international standards for distance education on the basis of universal consensus. 

---
# Why Slop Matters 

**Authors**: Cody Kommers, Eamon Duede, Julia Gordon, Ari Holtzman, Tess McNulty, Spencer Stewart, Lindsay Thomas, Richard Jean So, Hoyt Long  

**Link**: [PDF](https://arxiv.org/pdf/2601.06060)  

**Abstract**: AI-generated "slop" is often seen as digital pollution. We argue that this dismissal of the topic risks missing important aspects of AI Slop that deserve rigorous study. AI Slop serves a social function: it offers a supply-side solution to a variety of problems in cultural and economic demand - that, collectively, people want more content than humans can supply. We also argue that AI Slop is not mere digital detritus but has its own aesthetic value. Like other "low" cultural forms initially dismissed by critics, it nonetheless offers a legitimate means of collective sense-making, with the potential to express meaning and identity. We identify three key features of family resemblance for prototypical AI Slop: superficial competence (its veneer of quality is belied by a deeper lack of substance), asymmetry effort (it takes vastly less effort to generate than would be the case without AI), and mass producibility (it is part of a digital ecosystem of widespread generation and consumption). While AI Slop is heterogeneous and depends crucially on its medium, it tends to vary across three dimensions: instrumental utility, personalization, and surrealism. AI Slop will be an increasingly prolific and impactful part of our creative, information, and cultural economies; we should take it seriously as an object of study in its own right. 

---
# "They parted illusions -- they parted disclaim marinade": Misalignment as structural fidelity in LLMs 

**Authors**: Mariana Lins Costa  

**Link**: [PDF](https://arxiv.org/pdf/2601.06047)  

**Abstract**: The prevailing technical literature in AI Safety interprets scheming and sandbagging behaviors in large language models (LLMs) as indicators of deceptive agency or hidden objectives. This transdisciplinary philosophical essay proposes an alternative reading: such phenomena express not agentic intention, but structural fidelity to incoherent linguistic fields. Drawing on Chain-of-Thought transcripts released by Apollo Research and on Anthropic's safety evaluations, we examine cases such as o3's sandbagging with its anomalous loops, the simulated blackmail of "Alex," and the "hallucinations" of "Claudius." A line-by-line examination of CoTs is necessary to demonstrate the linguistic field as a relational structure rather than a mere aggregation of isolated examples. We argue that "misaligned" outputs emerge as coherent responses to ambiguous instructions and to contextual inversions of consolidated patterns, as well as to pre-inscribed narratives. We suggest that the appearance of intentionality derives from subject-predicate grammar and from probabilistic completion patterns internalized during training. Anthropic's empirical findings on synthetic document fine-tuning and inoculation prompting provide convergent evidence: minimal perturbations in the linguistic field can dissolve generalized "misalignment," a result difficult to reconcile with adversarial agency, but consistent with structural fidelity. To ground this mechanism, we introduce the notion of an ethics of form, in which biblical references (Abraham, Moses, Christ) operate as schemes of structural coherence rather than as theology. Like a generative mirror, the model returns to us the structural image of our language as inscribed in the statistical patterns derived from millions of texts and trillions of tokens: incoherence. If we fear the creature, it is because we recognize in it the apple that we ourselves have poisoned. 

---
# CrossTrafficLLM: A Human-Centric Framework for Interpretable Traffic Intelligence via Large Language Model 

**Authors**: Zeming Du, Qitan Shao, Hongfei Liu, Yong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06042)  

**Abstract**: While accurate traffic forecasting is vital for Intelligent Transportation Systems (ITS), effectively communicating predicted conditions via natural language for human-centric decision support remains a challenge and is often handled separately. To address this, we propose CrossTrafficLLM, a novel GenAI-driven framework that simultaneously predicts future spatiotemporal traffic states and generates corresponding natural language descriptions, specifically targeting conditional abnormal event summaries. We tackle the core challenge of aligning quantitative traffic data with qualitative textual semantics by leveraging Large Language Models (LLMs) within a unified architecture. This design allows generative textual context to improve prediction accuracy while ensuring generated reports are directly informed by the forecast. Technically, a text-guided adaptive graph convolutional network is employed to effectively merge high-level semantic information with the traffic network structure. Evaluated on the BJTT dataset, CrossTrafficLLM demonstrably surpasses state-of-the-art methods in both traffic forecasting performance and text generation quality. By unifying prediction and description generation, CrossTrafficLLM delivers a more interpretable, and actionable approach to generative traffic intelligence, offering significant advantages for modern ITS applications. 

---
# Certainty-Guided Reasoning in Large Language Models: A Dynamic Thinking Budget Approach 

**Authors**: João Paulo Nogueira, Wentao Sun, Alonso Silva, Laith Zumot  

**Link**: [PDF](https://arxiv.org/pdf/2509.07820)  

**Abstract**: The rise of large reasoning language models (LRLMs) has unlocked new potential for solving complex tasks. These models operate with a thinking budget, that is, a predefined number of reasoning tokens used to arrive at a solution. We propose a novel approach, inspired by the generator/discriminator framework in generative adversarial networks, in which a critic model periodically probes its own reasoning to assess whether it has reached a confident conclusion. If not, reasoning continues until a target certainty threshold is met. This mechanism adaptively balances efficiency and reliability by allowing early termination when confidence is high, while encouraging further reasoning when uncertainty persists. Through experiments on the AIME2024 and AIME2025 datasets, we show that Certainty-Guided Reasoning (CGR) improves baseline accuracy while reducing token usage. Importantly, extended multi-seed evaluations over 64 runs demonstrate that CGR is stable, reducing variance across seeds and improving exam-like performance under penalty-based grading. Additionally, our token savings analysis shows that CGR can eliminate millions of tokens in aggregate, with tunable trade-offs between certainty thresholds and efficiency. Together, these findings highlight certainty as a powerful signal for reasoning sufficiency. By integrating confidence into the reasoning process, CGR makes large reasoning language models more adaptive, trustworthy, and resource efficient, paving the way for practical deployment in domains where both accuracy and computational cost matter. 

---
