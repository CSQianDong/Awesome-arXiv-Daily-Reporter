# DRISHTIKON: A Multimodal Multilingual Benchmark for Testing Language Models' Understanding on Indian Culture 

**Authors**: Arijit Maji, Raghvendra Kumar, Akash Ghosh, Anushka, Nemil Shah, Abhilekh Borah, Vanshika Shah, Nishant Mishra, Sriparna Saha  

**Link**: [PDF](https://arxiv.org/pdf/2509.19274)  

**Abstract**: We introduce DRISHTIKON, a first-of-its-kind multimodal and multilingual benchmark centered exclusively on Indian culture, designed to evaluate the cultural understanding of generative AI systems. Unlike existing benchmarks with a generic or global scope, DRISHTIKON offers deep, fine-grained coverage across India's diverse regions, spanning 15 languages, covering all states and union territories, and incorporating over 64,000 aligned text-image pairs. The dataset captures rich cultural themes including festivals, attire, cuisines, art forms, and historical heritage amongst many more. We evaluate a wide range of vision-language models (VLMs), including open-source small and large models, proprietary systems, reasoning-specialized VLMs, and Indic-focused models, across zero-shot and chain-of-thought settings. Our results expose key limitations in current models' ability to reason over culturally grounded, multimodal inputs, particularly for low-resource languages and less-documented traditions. DRISHTIKON fills a vital gap in inclusive AI research, offering a robust testbed to advance culturally aware, multimodally competent language technologies. 

---
# WolBanking77: Wolof Banking Speech Intent Classification Dataset 

**Authors**: Abdou Karim Kandji, Frédéric Precioso, Cheikh Ba, Samba Ndiaye, Augustin Ndione  

**Link**: [PDF](https://arxiv.org/pdf/2509.19271)  

**Abstract**: Intent classification models have made a lot of progress in recent years. However, previous studies primarily focus on high-resource languages datasets, which results in a gap for low-resource languages and for regions with a high rate of illiterate people where languages are more spoken than read or written. This is the case in Senegal, for example, where Wolof is spoken by around 90\% of the population, with an illiteracy rate of 42\% for the country. Wolof is actually spoken by more than 10 million people in West African region. To tackle such limitations, we release a Wolof Intent Classification Dataset (WolBanking77), for academic research in intent classification. WolBanking77 currently contains 9,791 text sentences in the banking domain and more than 4 hours of spoken sentences. Experiments on various baselines are conducted in this work, including text and voice state-of-the-art models. The results are very promising on this current dataset. This paper also provides detailed analyses of the contents of the data. We report baseline f1-score and word error rate metrics respectively on NLP and ASR models trained on WolBanking77 dataset and also comparisons between models. We plan to share and conduct dataset maintenance, updates and to release open-source code. 

---
# SloPalSpeech: A 2,8000-Hour Slovak Speech Corpus from Parliamentary Data 

**Authors**: Erik Božík, Marek Šuppa  

**Link**: [PDF](https://arxiv.org/pdf/2509.19270)  

**Abstract**: Automatic Speech Recognition (ASR) for low-resource languages like Slovak is hindered by the scarcity of training data. To address this, we introduce SloPalSpeech, a new, large-scale Slovak ASR dataset containing 2,806 hours of speech from parliamentary proceedings. We developed a robust processing pipeline to align and segment long-form recordings into clean, 30-second audio-transcript pairs suitable for model training. We use this dataset to fine-tune several OpenAI Whisper models (small, medium, large-v3, and large-v3-turbo), achieving significant Word Error Rate (WER) reductions on standard Slovak benchmarks like Common Voice and FLEURS. For instance, the fine-tuned Whisper-small model's WER dropped by up to 70\%, approaching the baseline performance of the much larger Whisper-large-v3 model. To foster future research in low-resource speech recognition, we publicly release the complete SloPalSpeech dataset, the fully segmented transcripts (60 million words), and all our fine-tuned models. 

---
# Extracting Conceptual Spaces from LLMs Using Prototype Embeddings 

**Authors**: Nitesh Kumar, Usashi Chatterjee, Steven Schockaert  

**Link**: [PDF](https://arxiv.org/pdf/2509.19269)  

**Abstract**: Conceptual spaces represent entities and concepts using cognitively meaningful dimensions, typically referring to perceptual features. Such representations are widely used in cognitive science and have the potential to serve as a cornerstone for explainable AI. Unfortunately, they have proven notoriously difficult to learn, although recent LLMs appear to capture the required perceptual features to a remarkable extent. Nonetheless, practical methods for extracting the corresponding conceptual spaces are currently still lacking. While various methods exist for extracting embeddings from LLMs, extracting conceptual spaces also requires us to encode the underlying features. In this paper, we propose a strategy in which features (e.g. sweetness) are encoded by embedding the description of a corresponding prototype (e.g. a very sweet food). To improve this strategy, we fine-tune the LLM to align the prototype embeddings with the corresponding conceptual space dimensions. Our empirical analysis finds this approach to be highly effective. 

---
# Reinforcement Learning on Pre-Training Data 

**Authors**: Siheng Li, Kejiao Li, Zenan Xu, Guanhua Huang, Evander Yang, Kun Li, Haoyuan Wu, Jiajia Wu, Zihao Zheng, Chenchen Zhang, Kun Shi, Kyrierl Deng, Qi Yi, Ruibin Xiong, Tingqiang Xu, Yuhao Jiang, Jianfeng Yan, Yuyuan Zeng, Guanghui Xu, Jinbao Xue, Zhijiang Xu, Zheng Fang, Shuai Li, Qibin Liu, Xiaoxue Li, Zhuoyu Li, Yangyu Tao, Fei Gao, Cheng Jiang, Bo Chao Wang, Kai Liu, Jianchen Zhu, Wai Lam, Wayyt Wang, Bo Zhou, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19249)  

**Abstract**: The growing disparity between the exponential scaling of computational resources and the finite growth of high-quality text data now constrains conventional scaling approaches for large language models (LLMs). To address this challenge, we introduce Reinforcement Learning on Pre-Training data (RLPT), a new training-time scaling paradigm for optimizing LLMs. In contrast to prior approaches that scale training primarily through supervised learning, RLPT enables the policy to autonomously explore meaningful trajectories to learn from pre-training data and improve its capability through reinforcement learning (RL). While existing RL strategies such as reinforcement learning from human feedback (RLHF) and reinforcement learning with verifiable rewards (RLVR) rely on human annotation for reward construction, RLPT eliminates this dependency by deriving reward signals directly from pre-training data. Specifically, it adopts a next-segment reasoning objective, rewarding the policy for accurately predicting subsequent text segments conditioned on the preceding context. This formulation allows RL to be scaled on pre-training data, encouraging the exploration of richer trajectories across broader contexts and thereby fostering more generalizable reasoning skills. Extensive experiments on both general-domain and mathematical reasoning benchmarks across multiple models validate the effectiveness of RLPT. For example, when applied to Qwen3-4B-Base, RLPT yields absolute improvements of $3.0$, $5.1$, $8.1$, $6.0$, $6.6$, and $5.3$ on MMLU, MMLU-Pro, GPQA-Diamond, KOR-Bench, AIME24, and AIME25, respectively. The results further demonstrate favorable scaling behavior, suggesting strong potential for continued gains with more compute. In addition, RLPT provides a solid foundation, extending the reasoning boundaries of LLMs and enhancing RLVR performance. 

---
# CompLLM: Compression for Long Context Q&A 

**Authors**: Gabriele Berton, Jayakrishnan Unnikrishnan, Son Tran, Mubarak Shah  

**Link**: [PDF](https://arxiv.org/pdf/2509.19228)  

**Abstract**: Large Language Models (LLMs) face significant computational challenges when processing long contexts due to the quadratic complexity of self-attention. While soft context compression methods, which map input text to smaller latent representations, have shown promise, their real-world adoption is limited. Existing techniques typically compress the context as a single unit, which leads to quadratic compression complexity and an inability to reuse computations across queries with overlapping contexts. In this work, we introduce CompLLM, a soft compression technique designed for practical deployment. Instead of processing the context holistically, CompLLM divides it into segments and compresses each one independently. This simple design choice yields three critical properties: efficiency, as the compression step scales linearly with the context length; scalability, enabling models trained on short sequences (e.g., 1k tokens) to generalize to contexts of 100k tokens; and reusability, allowing compressed segments to be cached and reused across different queries. Our experiments show that with a 2x compression rate, at high context lengths CompLLM speeds up Time To First Token (TTFT) by up to 4x and reduces the KV cache size by 50%. Furthermore, CompLLM achieves performance comparable to that obtained with the uncompressed context, and even surpasses it on very long sequences, demonstrating its effectiveness and practical utility. 

---
# Systematic Comparative Analysis of Large Pretrained Language Models on Contextualized Medication Event Extraction 

**Authors**: Tariq Abdul-Quddoos, Xishuang Dong, Lijun Qian  

**Link**: [PDF](https://arxiv.org/pdf/2509.19224)  

**Abstract**: Attention-based models have become the leading approach in modeling medical language for Natural Language Processing (NLP) in clinical notes. These models outperform traditional techniques by effectively capturing contextual rep- resentations of language. In this research a comparative analysis is done amongst pre- trained attention based models namely Bert Base, BioBert, two variations of Bio+Clinical Bert, RoBerta, and Clinical Long- former on task related to Electronic Health Record (EHR) information extraction. The tasks from Track 1 of Harvard Medical School's 2022 National Clinical NLP Challenges (n2c2) are considered for this comparison, with the Contextualized Medication Event Dataset (CMED) given for these task. CMED is a dataset of unstructured EHRs and annotated notes that contain task relevant information about the EHRs. The goal of the challenge is to develop effective solutions for extracting contextual information related to patient medication events from EHRs using data driven methods. Each pre-trained model is fine-tuned and applied on CMED to perform medication extraction, medical event detection, and multi-dimensional medication event context classification. Pro- cessing methods are also detailed for breaking down EHRs for compatibility with the applied models. Performance analysis has been carried out using a script based on constructing medical terms from the evaluation portion of CMED with metrics including recall, precision, and F1-Score. The results demonstrate that models pre-trained on clinical data are more effective in detecting medication and medication events, but Bert Base, pre- trained on general domain data showed to be the most effective for classifying the context of events related to medications. 

---
# Steering Multimodal Large Language Models Decoding for Context-Aware Safety 

**Authors**: Zheyuan Liu, Zhangchen Xu, Guangyao Dou, Xiangchi Yuan, Zhaoxuan Tan, Radha Poovendran, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19212)  

**Abstract**: Multimodal Large Language Models (MLLMs) are increasingly deployed in real-world applications, yet their ability to make context-aware safety decisions remains limited. Existing methods often fail to balance oversensitivity (unjustified refusals of benign queries) and undersensitivity (missed detection of visually grounded risks), leaving a persistent gap in safety alignment. To address this issue, we introduce Safety-aware Contrastive Decoding (SafeCoDe), a lightweight and model-agnostic decoding framework that dynamically adjusts token generation based on multimodal context. SafeCoDe operates in two stages: (1) a contrastive decoding mechanism that highlights tokens sensitive to visual context by contrasting real and Gaussian-noised images, and (2) a global-aware token modulation strategy that integrates scene-level reasoning with token-level adjustment to adapt refusals according to the predicted safety verdict. Extensive experiments across diverse MLLM architectures and safety benchmarks, covering undersensitivity, oversensitivity, and general safety evaluations, show that SafeCoDe consistently improves context-sensitive refusal behaviors while preserving model helpfulness. 

---
# Online Process Reward Leanring for Agentic Reinforcement Learning 

**Authors**: Xiaoqian Liu, Ke Wang, Yuchuan Wu, Fei Huang, Yongbin Li, Junge Zhang, Jianbin Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2509.19199)  

**Abstract**: Large language models (LLMs) are increasingly trained with reinforcement learning (RL) as autonomous agents that reason and act over long horizons in interactive environments.
However, sparse and sometimes unverifiable rewards make temporal credit assignment extremely challenging.
Recent work attempts to integrate process supervision into agent learning but suffers from biased annotation, reward hacking, high-variance from overly fine-grained signals or failtures when state overlap is rare.
We therefore introduce Online Process Reward Learning (OPRL), a general credit-assignment strategy for agentic RL that integrates seamlessly with standard on-policy algorithms without relying on additional rollouts or explicit step labels.
In OPRL, we optimize an implicit process reward model (PRM) alternately with the agent's policy to transform trajectory preferences into implicit step rewards through a trajectory-based DPO objective.
These step rewards are then used to compute step-level advantages, which are combined with episode-level advantages from outcome rewards for policy update, creating a self-reinforcing loop.
Theoretical findings guarantee that the learned step rewards are consistent with trajectory preferences and act as potential-based shaping rewards, providing bounded gradients to stabilize training.
Empirically, we evaluate OPRL on three distinct agent benmarks, including WebShop and VisualSokoban, as well as open-ended social interactions with unverfiable rewards in SOTOPIA.
Crucially, OPRL shows superior performance over frontier LLMs and strong RL baselines across domains, achieving state-of-the-art results with higher sample-efficiency and lower variance during training.
Further analysis also demonstrates the efficient exploration by OPRL using fewer actions, underscoring its potential for agentic learning in real-world scenarios. 

---
# Soft Tokens, Hard Truths 

**Authors**: Natasha Butt, Ariel Kwiatkowski, Ismail Labiad, Julia Kempe, Yann Ollivier  

**Link**: [PDF](https://arxiv.org/pdf/2509.19170)  

**Abstract**: The use of continuous instead of discrete tokens during the Chain-of-Thought (CoT) phase of reasoning LLMs has garnered attention recently, based on the intuition that a continuous mixture of discrete tokens could simulate a superposition of several reasoning paths simultaneously. Theoretical results have formally proven that continuous tokens have much greater expressivity and can solve specific problems more efficiently. However, practical use of continuous tokens has been limited by strong training difficulties: previous works either just use continuous tokens at inference time on a pre-trained discrete-token model, or must distill the continuous CoT from ground-truth discrete CoTs and face computational costs that limit the CoT to very few tokens.
This is the first work introducing a scalable method to learn continuous CoTs via reinforcement learning (RL), without distilling from reference discrete CoTs. We use "soft" tokens: mixtures of tokens together with noise on the input embedding to provide RL exploration. Computational overhead is minimal, enabling us to learn continuous CoTs with hundreds of tokens. On math reasoning benchmarks with Llama and Qwen models up to 8B, training with continuous CoTs match discrete-token CoTs for pass@1 and surpass them for pass@32, showing greater CoT diversity. In systematic comparisons, the best-performing scenario is to train with continuous CoT tokens then use discrete tokens for inference, meaning the "soft" models can be deployed in a standard way. Finally, we show continuous CoT RL training better preserves the predictions of the base model on out-of-domain tasks, thus providing a softer touch to the base model. 

---
# Measuring AI "Slop" in Text 

**Authors**: Chantal Shaib, Tuhin Chakrabarty, Diego Garcia-Olano, Byron C. Wallace  

**Link**: [PDF](https://arxiv.org/pdf/2509.19163)  

**Abstract**: AI "slop" is an increasingly popular term used to describe low-quality AI-generated text, but there is currently no agreed upon definition of this term nor a means to measure its occurrence. In this work, we develop a taxonomy of "slop" through interviews with experts in NLP, writing, and philosophy, and propose a set of interpretable dimensions for its assessment in text. Through span-level annotation, we find that binary "slop" judgments are (somewhat) subjective, but such determinations nonetheless correlate with latent dimensions such as coherence and relevance. Our framework can be used to evaluate AI-generated text in both detection and binary preference tasks, potentially offering new insights into the linguistic and stylistic factors that contribute to quality judgments. 

---
# Anecdoctoring: Automated Red-Teaming Across Language and Place 

**Authors**: Alejandro Cuevas, Saloni Dash, Bharat Kumar Nayak, Dan Vann, Madeleine I. G. Daepp  

**Link**: [PDF](https://arxiv.org/pdf/2509.19143)  

**Abstract**: Disinformation is among the top risks of generative artificial intelligence (AI) misuse. Global adoption of generative AI necessitates red-teaming evaluations (i.e., systematic adversarial probing) that are robust across diverse languages and cultures, but red-teaming datasets are commonly US- and English-centric. To address this gap, we propose "anecdoctoring", a novel red-teaming approach that automatically generates adversarial prompts across languages and cultures. We collect misinformation claims from fact-checking websites in three languages (English, Spanish, and Hindi) and two geographies (US and India). We then cluster individual claims into broader narratives and characterize the resulting clusters with knowledge graphs, with which we augment an attacker LLM. Our method produces higher attack success rates and offers interpretability benefits relative to few-shot prompting. Results underscore the need for disinformation mitigations that scale globally and are grounded in real-world adversarial misuse. 

---
# Context-Aware Hierarchical Taxonomy Generation for Scientific Papers via LLM-Guided Multi-Aspect Clustering 

**Authors**: Kun Zhu, Lizi Liao, Yuxuan Gu, Lei Huang, Xiaocheng Feng, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2509.19125)  

**Abstract**: The rapid growth of scientific literature demands efficient methods to organize and synthesize research findings. Existing taxonomy construction methods, leveraging unsupervised clustering or direct prompting of large language models (LLMs), often lack coherence and granularity. We propose a novel context-aware hierarchical taxonomy generation framework that integrates LLM-guided multi-aspect encoding with dynamic clustering. Our method leverages LLMs to identify key aspects of each paper (e.g., methodology, dataset, evaluation) and generates aspect-specific paper summaries, which are then encoded and clustered along each aspect to form a coherent hierarchy. In addition, we introduce a new evaluation benchmark of 156 expert-crafted taxonomies encompassing 11.6k papers, providing the first naturally annotated dataset for this task. Experimental results demonstrate that our method significantly outperforms prior approaches, achieving state-of-the-art performance in taxonomy coherence, granularity, and interpretability. 

---
# Human-Annotated NER Dataset for the Kyrgyz Language 

**Authors**: Timur Turatali, Anton Alekseev, Gulira Jumalieva, Gulnara Kabaeva, Sergey Nikolenko  

**Link**: [PDF](https://arxiv.org/pdf/2509.19109)  

**Abstract**: We introduce KyrgyzNER, the first manually annotated named entity recognition dataset for the Kyrgyz language. Comprising 1,499 news articles from the this http URL news portal, the dataset contains 10,900 sentences and 39,075 entity mentions across 27 named entity classes. We show our annotation scheme, discuss the challenges encountered in the annotation process, and present the descriptive statistics. We also evaluate several named entity recognition models, including traditional sequence labeling approaches based on conditional random fields and state-of-the-art multilingual transformer-based models fine-tuned on our dataset. While all models show difficulties with rare entity categories, models such as the multilingual RoBERTa variant pretrained on a large corpus across many languages achieve a promising balance between precision and recall. These findings emphasize both the challenges and opportunities of using multilingual pretrained models for processing languages with limited resources. Although the multilingual RoBERTa model performed best, other multilingual models yielded comparable results. This suggests that future work exploring more granular annotation schemes may offer deeper insights for Kyrgyz language processing pipelines evaluation. 

---
# Are most sentences unique? An empirical examination of Chomskyan claims 

**Authors**: Hiram Ring  

**Link**: [PDF](https://arxiv.org/pdf/2509.19108)  

**Abstract**: A repeated claim in linguistics is that the majority of linguistic utterances are unique. For example, Pinker (1994: 10), summarizing an argument by Noam Chomsky, states that "virtually every sentence that a person utters or understands is a brand-new combination of words, appearing for the first time in the history of the universe." With the increased availability of large corpora, this is a claim that can be empirically investigated. The current paper addresses the question by using the NLTK Python library to parse corpora of different genres, providing counts of exact string matches in each. Results show that while completely unique sentences are often the majority of corpora, this is highly constrained by genre, and that duplicate sentences are not an insignificant part of any individual corpus. 

---
# Pathways of Thoughts: Multi-Directional Thinking for Long-form Personalized Question Answering 

**Authors**: Alireza Salemi, Cheng Li, Mingyang Zhang, Qiaozhu Mei, Zhuowan Li, Spurthi Amba Hombaiah, Weize Kong, Tao Chen, Hamed Zamani, Michael Bendersky  

**Link**: [PDF](https://arxiv.org/pdf/2509.19094)  

**Abstract**: Personalization is essential for adapting question answering (QA) systems to user-specific information needs, thereby improving both accuracy and user satisfaction. However, personalized QA remains relatively underexplored due to challenges such as inferring preferences from long, noisy, and implicit contexts, and generating responses that are simultaneously correct, contextually appropriate, and aligned with user expectations and background knowledge. To address these challenges, we propose Pathways of Thoughts (PoT), an inference-stage method that applies to any large language model (LLM) without requiring task-specific fine-tuning. The approach models the reasoning of an LLM as an iterative decision process, where the model dynamically selects among cognitive operations such as reasoning, revision, personalization, and clarification. This enables exploration of multiple reasoning trajectories, producing diverse candidate responses that capture different perspectives. PoT then aggregates and reweights these candidates according to inferred user preferences, yielding a final personalized response that benefits from the complementary strengths of diverse reasoning paths. Experiments on the LaMP-QA benchmark for personalized QA show that PoT consistently outperforms competitive baselines, achieving up to a 13.1% relative improvement. Human evaluation corroborates these results, with annotators preferring outputs from PoT in 66% of cases and reporting ties in only 15% of cases. 

---
# Charting a Decade of Computational Linguistics in Italy: The CLiC-it Corpus 

**Authors**: Chiara Alzetta, Serena Auriemma, Alessandro Bondielli, Luca Dini, Chiara Fazzone, Alessio Miaschi, Martina Miliani, Marta Sartor  

**Link**: [PDF](https://arxiv.org/pdf/2509.19033)  

**Abstract**: Over the past decade, Computational Linguistics (CL) and Natural Language Processing (NLP) have evolved rapidly, especially with the advent of Transformer-based Large Language Models (LLMs). This shift has transformed research goals and priorities, from Lexical and Semantic Resources to Language Modelling and Multimodality. In this study, we track the research trends of the Italian CL and NLP community through an analysis of the contributions to CLiC-it, arguably the leading Italian conference in the field. We compile the proceedings from the first 10 editions of the CLiC-it conference (from 2014 to 2024) into the CLiC-it Corpus, providing a comprehensive analysis of both its metadata, including author provenance, gender, affiliations, and more, as well as the content of the papers themselves, which address various topics. Our goal is to provide the Italian and international research communities with valuable insights into emerging trends and key developments over time, supporting informed decisions and future directions in the field. 

---
# Investigating Test-Time Scaling with Reranking for Machine Translation 

**Authors**: Shaomu Tan, Ryosuke Mitani, Ritvik Choudhary, Toshiyuki Sekiya  

**Link**: [PDF](https://arxiv.org/pdf/2509.19020)  

**Abstract**: Scaling model parameters has become the de facto strategy for improving NLP systems, but it comes with substantial computational costs. Test-Time Scaling (TTS) offers an alternative by allocating more computation at inference: generating multiple candidates and selecting the best. While effective in tasks such as mathematical reasoning, TTS has not been systematically explored for machine translation (MT). In this paper, we present the first systematic study of TTS for MT, investigating a simple but practical best-of-N framework on WMT24 benchmarks. Our experiments cover six high-resource and one low-resource language pairs, five model sizes (3B-72B), and various TTS compute budget (N up to 1024). Our results show that a) For high-resource languages, TTS generally improves translation quality according to multiple neural MT evaluation metrics, and our human evaluation confirms these gains; b) Augmenting smaller models with large $N$ can match or surpass larger models at $N{=}1$ with more compute cost; c) Under fixed compute budgets, larger models are typically more efficient, and TTS can degrade quality due to metric blind spots in low-resource cases. 

---
# DTW-Align: Bridging the Modality Gap in End-to-End Speech Translation with Dynamic Time Warping Alignment 

**Authors**: Abderrahmane Issam, Yusuf Can Semerci, Jan Scholtes, Gerasimos Spanakis  

**Link**: [PDF](https://arxiv.org/pdf/2509.18987)  

**Abstract**: End-to-End Speech Translation (E2E-ST) is the task of translating source speech directly into target text bypassing the intermediate transcription step. The representation discrepancy between the speech and text modalities has motivated research on what is known as bridging the modality gap. State-of-the-art methods addressed this by aligning speech and text representations on the word or token level. Unfortunately, this requires an alignment tool that is not available for all languages. Although this issue has been addressed by aligning speech and text embeddings using nearest-neighbor similarity search, it does not lead to accurate alignments. In this work, we adapt Dynamic Time Warping (DTW) for aligning speech and text embeddings during training. Our experiments demonstrate the effectiveness of our method in bridging the modality gap in E2E-ST. Compared to previous work, our method produces more accurate alignments and achieves comparable E2E-ST results while being significantly faster. Furthermore, our method outperforms previous work in low resource settings on 5 out of 6 language directions. 

---
# Extractive Fact Decomposition for Interpretable Natural Language Inference in one Forward Pass 

**Authors**: Nicholas Popovič, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2509.18901)  

**Abstract**: Recent works in Natural Language Inference (NLI) and related tasks, such as automated fact-checking, employ atomic fact decomposition to enhance interpretability and robustness. For this, existing methods rely on resource-intensive generative large language models (LLMs) to perform decomposition. We propose JEDI, an encoder-only architecture that jointly performs extractive atomic fact decomposition and interpretable inference without requiring generative models during inference. To facilitate training, we produce a large corpus of synthetic rationales covering multiple NLI benchmarks. Experimental results demonstrate that JEDI achieves competitive accuracy in distribution and significantly improves robustness out of distribution and in adversarial settings over models based solely on extractive rationale supervision. Our findings show that interpretability and robust generalization in NLI can be realized using encoder-only architectures and synthetic rationales. Code and data available at this https URL 

---
# Diversity Boosts AI-Generated Text Detection 

**Authors**: Advik Raj Basani, Pin-Yu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.18880)  

**Abstract**: Detecting AI-generated text is an increasing necessity to combat misuse of LLMs in education, business compliance, journalism, and social media, where synthetic fluency can mask misinformation or deception. While prior detectors often rely on token-level likelihoods or opaque black-box classifiers, these approaches struggle against high-quality generations and offer little interpretability. In this work, we propose DivEye, a novel detection framework that captures how unpredictability fluctuates across a text using surprisal-based features. Motivated by the observation that human-authored text exhibits richer variability in lexical and structural unpredictability than LLM outputs, DivEye captures this signal through a set of interpretable statistical features. Our method outperforms existing zero-shot detectors by up to 33.2% and achieves competitive performance with fine-tuned baselines across multiple benchmarks. DivEye is robust to paraphrasing and adversarial attacks, generalizes well across domains and models, and improves the performance of existing detectors by up to 18.7% when used as an auxiliary signal. Beyond detection, DivEye provides interpretable insights into why a text is flagged, pointing to rhythmic unpredictability as a powerful and underexplored signal for LLM detection. 

---
# Multi-Hierarchical Feature Detection for Large Language Model Generated Text 

**Authors**: Luyan Zhang, Xinyu Xie  

**Link**: [PDF](https://arxiv.org/pdf/2509.18862)  

**Abstract**: With the rapid advancement of large language model technology, there is growing interest in whether multi-feature approaches can significantly improve AI text detection beyond what single neural models achieve. While intuition suggests that combining semantic, syntactic, and statistical features should provide complementary signals, this assumption has not been rigorously tested with modern LLM-generated text. This paper provides a systematic empirical investigation of multi-hierarchical feature integration for AI text detection, specifically testing whether the computational overhead of combining multiple feature types is justified by performance gains. We implement MHFD (Multi-Hierarchical Feature Detection), integrating DeBERTa-based semantic analysis, syntactic parsing, and statistical probability features through adaptive fusion. Our investigation reveals important negative results: despite theoretical expectations, multi-feature integration provides minimal benefits (0.4-0.5% improvement) while incurring substantial computational costs (4.2x overhead), suggesting that modern neural language models may already capture most relevant detection signals efficiently. Experimental results on multiple benchmark datasets demonstrate that the MHFD method achieves 89.7% accuracy in in-domain detection and maintains 84.2% stable performance in cross-domain detection, showing modest improvements of 0.4-2.6% over existing methods. 

---
# Are Smaller Open-Weight LLMs Closing the Gap to Proprietary Models for Biomedical Question Answering? 

**Authors**: Damian Stachura, Joanna Konieczna, Artur Nowak  

**Link**: [PDF](https://arxiv.org/pdf/2509.18843)  

**Abstract**: Open-weight versions of large language models (LLMs) are rapidly advancing, with state-of-the-art models like DeepSeek-V3 now performing comparably to proprietary LLMs. This progression raises the question of whether small open-weight LLMs are capable of effectively replacing larger closed-source models. We are particularly interested in the context of biomedical question-answering, a domain we explored by participating in Task 13B Phase B of the BioASQ challenge. In this work, we compare several open-weight models against top-performing systems such as GPT-4o, GPT-4.1, Claude 3.5 Sonnet, and Claude 3.7 Sonnet. To enhance question answering capabilities, we use various techniques including retrieving the most relevant snippets based on embedding distance, in-context learning, and structured outputs. For certain submissions, we utilize ensemble approaches to leverage the diverse outputs generated by different models for exact-answer questions. Our results demonstrate that open-weight LLMs are comparable to proprietary ones. In some instances, open-weight LLMs even surpassed their closed counterparts, particularly when ensembling strategies were applied. All code is publicly available at this https URL. 

---
# MAPEX: A Multi-Agent Pipeline for Keyphrase Extraction 

**Authors**: Liting Zhang, Shiwan Zhao, Aobo Kong, Qicheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.18813)  

**Abstract**: Keyphrase extraction is a fundamental task in natural language processing. However, existing unsupervised prompt-based methods for Large Language Models (LLMs) often rely on single-stage inference pipelines with uniform prompting, regardless of document length or LLM backbone. Such one-size-fits-all designs hinder the full exploitation of LLMs' reasoning and generation capabilities, especially given the complexity of keyphrase extraction across diverse scenarios. To address these challenges, we propose MAPEX, the first framework that introduces multi-agent collaboration into keyphrase extraction. MAPEX coordinates LLM-based agents through modules for expert recruitment, candidate extraction, topic guidance, knowledge augmentation, and post-processing. A dual-path strategy dynamically adapts to document length: knowledge-driven extraction for short texts and topic-guided extraction for long texts. Extensive experiments on six benchmark datasets across three different LLMs demonstrate its strong generalization and universality, outperforming the state-of-the-art unsupervised method by 2.44\% and standard LLM baselines by 4.01\% in F1@5 on average. Code is available at this https URL. 

---
# Beyond the Leaderboard: Understanding Performance Disparities in Large Language Models via Model Diffing 

**Authors**: Sabri Boughorbel, Fahim Dalvi, Nadir Durrani, Majd Hawasly  

**Link**: [PDF](https://arxiv.org/pdf/2509.18792)  

**Abstract**: As fine-tuning becomes the dominant paradigm for improving large language models (LLMs), understanding what changes during this process is increasingly important. Traditional benchmarking often fails to explain why one model outperforms another. In this work, we use model diffing, a mechanistic interpretability approach, to analyze the specific capability differences between Gemma-2-9b-it and a SimPO-enhanced variant. Using crosscoders, we identify and categorize latent representations that differentiate the two models. We find that SimPO acquired latent concepts predominantly enhance safety mechanisms (+32.8%), multilingual capabilities (+43.8%), and instruction-following (+151.7%), while its additional training also reduces emphasis on model self-reference (-44.1%) and hallucination management (-68.5%). Our analysis shows that model diffing can yield fine-grained insights beyond leaderboard metrics, attributing performance gaps to concrete mechanistic capabilities. This approach offers a transparent and targeted framework for comparing LLMs. 

---
# AECBench: A Hierarchical Benchmark for Knowledge Evaluation of Large Language Models in the AEC Field 

**Authors**: Chen Liang, Zhaoqi Huang, Haofen Wang, Fu Chai, Chunying Yu, Huanhuan Wei, Zhengjie Liu, Yanpeng Li, Hongjun Wang, Ruifeng Luo, Xianzhong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.18776)  

**Abstract**: Large language models (LLMs), as a novel information technology, are seeing increasing adoption in the Architecture, Engineering, and Construction (AEC) field. They have shown their potential to streamline processes throughout the building lifecycle. However, the robustness and reliability of LLMs in such a specialized and safety-critical domain remain to be evaluated. To address this challenge, this paper establishes AECBench, a comprehensive benchmark designed to quantify the strengths and limitations of current LLMs in the AEC domain. The benchmark defines 23 representative tasks within a five-level cognition-oriented evaluation framework encompassing Knowledge Memorization, Understanding, Reasoning, Calculation, and Application. These tasks were derived from authentic AEC practice, with scope ranging from codes retrieval to specialized documents generation. Subsequently, a 4,800-question dataset encompassing diverse formats, including open-ended questions, was crafted primarily by engineers and validated through a two-round expert review. Furthermore, an LLM-as-a-Judge approach was introduced to provide a scalable and consistent methodology for evaluating complex, long-form responses leveraging expert-derived rubrics. Through the evaluation of nine LLMs, a clear performance decline across five cognitive levels was revealed. Despite demonstrating proficiency in foundational tasks at the Knowledge Memorization and Understanding levels, the models showed significant performance deficits, particularly in interpreting knowledge from tables in building codes, executing complex reasoning and calculation, and generating domain-specific documents. Consequently, this study lays the groundwork for future research and development aimed at the robust and reliable integration of LLMs into safety-critical engineering practices. 

---
# Financial Risk Relation Identification through Dual-view Adaptation 

**Authors**: Wei-Ning Chiu, Yu-Hsiang Wang, Andy Hsiao, Yu-Shiang Huang, Chuan-Ju Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.18775)  

**Abstract**: A multitude of interconnected risk events -- ranging from regulatory changes to geopolitical tensions -- can trigger ripple effects across firms. Identifying inter-firm risk relations is thus crucial for applications like portfolio management and investment strategy. Traditionally, such assessments rely on expert judgment and manual analysis, which are, however, subjective, labor-intensive, and difficult to scale. To address this, we propose a systematic method for extracting inter-firm risk relations using Form 10-K filings -- authoritative, standardized financial documents -- as our data source. Leveraging recent advances in natural language processing, our approach captures implicit and abstract risk connections through unsupervised fine-tuning based on chronological and lexical patterns in the filings. This enables the development of a domain-specific financial encoder with a deeper contextual understanding and introduces a quantitative risk relation score for transparency, interpretable analysis. Extensive experiments demonstrate that our method outperforms strong baselines across multiple evaluation settings. 

---
# When Long Helps Short: How Context Length in Supervised Fine-tuning Affects Behavior of Large Language Models 

**Authors**: Yingming Zheng, Hanqi Li, Kai Yu, Lu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.18762)  

**Abstract**: Large language models (LLMs) have achieved impressive performance across natural language processing (NLP) tasks. As real-world applications increasingly demand longer context windows, continued pretraining and supervised fine-tuning (SFT) on long-context data has become a common approach. While the effects of data length in continued pretraining have been extensively studied, their implications for SFT remain unclear. In this work, we systematically investigate how SFT data length influences LLM behavior on short-context tasks. Counterintuitively, we find that long-context SFT improves short-context performance, contrary to the commonly observed degradation from long-context pretraining. To uncover the underlying mechanisms of this phenomenon, we first decouple and analyze two key components, Multi-Head Attention (MHA) and Feed-Forward Network (FFN), and show that both independently benefit from long-context SFT. We further study their interaction and reveal a knowledge preference bias: long-context SFT promotes contextual knowledge, while short-context SFT favors parametric knowledge, making exclusive reliance on long-context SFT suboptimal. Finally, we demonstrate that hybrid training mitigates this bias, offering explainable guidance for fine-tuning LLMs. 

---
# False Friends Are Not Foes: Investigating Vocabulary Overlap in Multilingual Language Models 

**Authors**: Julie Kallini, Dan Jurafsky, Christopher Potts, Martijn Bartelds  

**Link**: [PDF](https://arxiv.org/pdf/2509.18750)  

**Abstract**: Subword tokenizers trained on multilingual corpora naturally produce overlapping tokens across languages. Does token overlap facilitate cross-lingual transfer or instead introduce interference between languages? Prior work offers mixed evidence, partly due to varied setups and confounders, such as token frequency or subword segmentation granularity. To address this question, we devise a controlled experiment where we train bilingual autoregressive models on multiple language pairs under systematically varied vocabulary overlap settings. Crucially, we explore a new dimension to understanding how overlap affects transfer: the semantic similarity of tokens shared across languages. We first analyze our models' hidden representations and find that overlap of any kind creates embedding spaces that capture cross-lingual semantic relationships, while this effect is much weaker in models with disjoint vocabularies. On XNLI and XQuAD, we find that models with overlap outperform models with disjoint vocabularies, and that transfer performance generally improves as overlap increases. Overall, our findings highlight the advantages of token overlap in multilingual models and show that substantial shared vocabulary remains a beneficial design choice for multilingual tokenizers. 

---
# Global-Recent Semantic Reasoning on Dynamic Text-Attributed Graphs with Large Language Models 

**Authors**: Yunan Wang, Jianxin Li, Ziwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.18742)  

**Abstract**: Dynamic Text-Attribute Graphs (DyTAGs), characterized by time-evolving graph interactions and associated text attributes, are prevalent in real-world applications. Existing methods, such as Graph Neural Networks (GNNs) and Large Language Models (LLMs), mostly focus on static TAGs. Extending these existing methods to DyTAGs is challenging as they largely neglect the recent-global temporal semantics: the recent semantic dependencies among interaction texts and the global semantic evolution of nodes over time. Furthermore, applying LLMs to the abundant and evolving text in DyTAGs faces efficiency issues. To tackle these challenges, we propose Dynamic Global-Recent Adaptive Semantic Processing (DyGRASP), a novel method that leverages LLMs and temporal GNNs to efficiently and effectively reason on DyTAGs. Specifically, we first design a node-centric implicit reasoning method together with a sliding window mechanism to efficiently capture recent temporal semantics. In addition, to capture global semantic dynamics of nodes, we leverage explicit reasoning with tailored prompts and an RNN-like chain structure to infer long-term semantics. Lastly, we intricately integrate the recent and global temporal semantics as well as the dynamic graph structural information using updating and merging layers. Extensive experiments on DyTAG benchmarks demonstrate DyGRASP's superiority, achieving up to 34% improvement in Hit@10 for destination node retrieval task. Besides, DyGRASP exhibits strong generalization across different temporal GNNs and LLMs. 

---
# LOTUSDIS: A Thai far-field meeting corpus for robust conversational ASR 

**Authors**: Pattara Tipaksorn, Sumonmas Thatphithakkul, Vataya Chunwijitra, Kwanchiva Thangthai  

**Link**: [PDF](https://arxiv.org/pdf/2509.18722)  

**Abstract**: We present LOTUSDIS, a publicly available Thai meeting corpus designed to advance far-field conversational ASR. The dataset comprises 114 hours of spontaneous, unscripted dialogue collected in 15-20 minute sessions with three participants, where overlapping speech is frequent and natural. Speech was recorded simultaneously by nine independent single-channel devices spanning six microphone types at distances from 0.12 m to 10 m, preserving the authentic effects of reverberation, noise, and device coloration without relying on microphone arrays. We provide standard train, dev, test splits and release a reproducible baseline system. We benchmarked several Whisper variants under zero-shot and fine-tuned conditions. Off-the-shelf models showed strong degradation with distance, confirming a mismatch between pre-training data and Thai far-field speech. Fine-tuning on LOTUSDIS dramatically improved robustness: a Thai Whisper baseline reduced overall WER from 64.3 to 38.3 and far-field WER from 81.6 to 49.5, with especially large gains on the most distant microphones. These results underscore the importance of distance-diverse training data for robust ASR. The corpus is available under CC-BY-SA 4.0. We also release training and evaluation scripts as a baseline system to promote reproducible research in this field. 

---
# MemOrb: A Plug-and-Play Verbal-Reinforcement Memory Layer for E-Commerce Customer Service 

**Authors**: Yizhe Huang, Yang Liu, Ruiyu Zhao, Xiaolong Zhong, Xingming Yue, Ling Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.18713)  

**Abstract**: Large Language Model-based agents(LLM-based agents) are increasingly deployed in customer service, yet they often forget across sessions, repeat errors, and lack mechanisms for continual self-improvement. This makes them unreliable in dynamic settings where stability and consistency are critical. To better evaluate these properties, we emphasize two indicators: task success rate as a measure of overall effectiveness, and consistency metrics such as Pass$^k$ to capture reliability across multiple trials. To address the limitations of existing approaches, we propose MemOrb, a lightweight and plug-and-play verbal reinforcement memory layer that distills multi-turn interactions into compact strategy reflections. These reflections are stored in a shared memory bank and retrieved to guide decision-making, without requiring any fine-tuning. Experiments show that MemOrb significantly improves both success rate and stability, achieving up to a 63 percentage-point gain in multi-turn success rate and delivering more consistent performance across repeated trials. Our results demonstrate that structured reflection is a powerful mechanism for enhancing long-term reliability of frozen LLM agents in customer service scenarios. 

---
# Analyzing Uncertainty of LLM-as-a-Judge: Interval Evaluations with Conformal Prediction 

**Authors**: Huanxin Sheng, Xinyi Liu, Hangfeng He, Jieyu Zhao, Jian Kang  

**Link**: [PDF](https://arxiv.org/pdf/2509.18658)  

**Abstract**: LLM-as-a-judge has become a promising paradigm for using large language models (LLMs) to evaluate natural language generation (NLG), but the uncertainty of its evaluation remains underexplored. This lack of reliability may limit its deployment in many applications. This work presents the first framework to analyze the uncertainty by offering a prediction interval of LLM-based scoring via conformal prediction. Conformal prediction constructs continuous prediction intervals from a single evaluation run, and we design an ordinal boundary adjustment for discrete rating tasks. We also suggest a midpoint-based score within the interval as a low-bias alternative to raw model score and weighted average. We perform extensive experiments and analysis, which show that conformal prediction can provide valid prediction interval with coverage guarantees. We also explore the usefulness of interval midpoint and judge reprompting for better judgment. 

---
# Consistency-Aware Parameter-Preserving Knowledge Editing Framework for Multi-Hop Question Answering 

**Authors**: Lingwen Deng, Yifei Han, Long Zhang, Yue Du, Bin Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.18655)  

**Abstract**: Parameter-Preserving Knowledge Editing (PPKE) enables updating models with new or corrected information without retraining or parameter adjustment. Recent PPKE approaches based on knowledge graphs (KG) to extend knowledge editing (KE) capabilities to multi-hop question answering (MHQA). However, these methods often lack consistency, leading to knowledge contamination, unstable updates, and retrieval behaviors that fail to reflect the intended edits. Such inconsistencies undermine the reliability of PPKE in multi- hop reasoning. We present CAPE-KG, Consistency-Aware Parameter-Preserving Editing with Knowledge Graphs, a novel consistency-aware framework for PPKE on MHQA. CAPE-KG ensures KG construction, update, and retrieval are always aligned with the requirements of the MHQA task, maintaining coherent reasoning over both unedited and edited knowledge. Extensive experiments on the MQuAKE benchmark show accuracy improvements in PPKE performance for MHQA, demonstrating the effectiveness of addressing consistency in PPKE. 

---
# A Good Plan is Hard to Find: Aligning Models with Preferences is Misaligned with What Helps Users 

**Authors**: Nishant Balepur, Matthew Shu, Yoo Yeon Sung, Seraphina Goldfarb-Tarrant, Shi Feng, Fumeng Yang, Rachel Rudinger, Jordan Lee Boyd-Graber  

**Link**: [PDF](https://arxiv.org/pdf/2509.18632)  

**Abstract**: To assist users in complex tasks, LLMs generate plans: step-by-step instructions towards a goal. While alignment methods aim to ensure LLM plans are helpful, they train (RLHF) or evaluate (ChatbotArena) on what users prefer, assuming this reflects what helps them. We test this with Planorama: an interface where 126 users answer 300 multi-step questions with LLM plans. We get 4388 plan executions and 5584 comparisons to measure plan helpfulness (QA success) and user preferences on plans, and recreate the setup in agents and reward models to see if they simulate or prefer what helps users. We expose: 1) user/model preferences and agent success do not accurately predict which plans help users, so common alignment feedback can misalign with helpfulness; 2) this gap is not due to user-specific preferences, as users are similarly successful when using plans they prefer/disprefer; 3) surface-level cues like brevity and question similarity strongly link to preferences, but such biases fail to predict helpfulness. In all, we argue aligning helpful LLMs needs feedback from real user interactions, not just preferences of what looks helpful, so we discuss the plan NLP researchers can execute to solve this problem. 

---
# UniECG: Understanding and Generating ECG in One Unified Model 

**Authors**: Jiarui Jin, Haoyu Wang, Xiang Lan, Jun Li, Gaofeng Cheng, Hongyan Li, Shenda Hong  

**Link**: [PDF](https://arxiv.org/pdf/2509.18588)  

**Abstract**: Recent unified models such as GPT-5 have achieved encouraging progress on vision-language tasks. However, these unified models typically fail to correctly understand ECG signals and provide accurate medical diagnoses, nor can they correctly generate ECG signals. To address these limitations, we propose UniECG, the first unified model for ECG capable of concurrently performing evidence-based ECG interpretation and text-conditioned ECG generation tasks. Through a decoupled two-stage training approach, the model first learns evidence-based interpretation skills (ECG-to-Text), and then injects ECG generation capabilities (Text-to-ECG) via latent space alignment. UniECG can autonomously choose to interpret or generate an ECG based on user input, significantly extending the capability boundaries of current ECG models. Our code and checkpoints will be made publicly available at this https URL upon acceptance. 

---
# TsqLoRA: Towards Sensitivity and Quality Low-Rank Adaptation for Efficient Fine-Tuning 

**Authors**: Yu Chen, Yifei Han, Long Zhang, Yue Du, Bin Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.18585)  

**Abstract**: Fine-tuning large pre-trained models for downstream tasks has become a fundamental approach in natural language processing. Fully fine-tuning all model parameters is computationally expensive and memory-intensive, especially in resource-constrained environments. Existing parameter-efficient fine-tuning methods reduce the number of trainable parameters but typically overlook the varying sensitivity of different model layers and the importance of training data. In this work, we propose TsqLoRA, a novel method that integrates data-quality-driven selection with sensitivity-aware low-rank adaptation, consisted of two main components: a quality-aware sampling mechanism for selecting the most informative training data, and a dynamic rank allocation module that adjusts the rank of each layer based on its sensitivity to parameter updates. The experimental results demonstrate that TsqLoRA improves fine-tuning efficiency while maintaining or even improving performance on a variety of NLP tasks. Our code will be available at this https URL. 

---
# Prior-based Noisy Text Data Filtering: Fast and Strong Alternative For Perplexity 

**Authors**: Yeongbin Seo, Gayoung Kim, Jaehyung Kim, Jinyoung Yeo  

**Link**: [PDF](https://arxiv.org/pdf/2509.18577)  

**Abstract**: As large language models (LLMs) are pretrained on massive web corpora, careful selection of data becomes essential to ensure effective and efficient learning. While perplexity (PPL)-based filtering has shown strong performance, it suffers from drawbacks: substantial time costs and inherent unreliability of the model when handling noisy or out-of-distribution samples. In this work, we propose a simple yet powerful alternative: a prior-based data filtering method that estimates token priors using corpus-level term frequency statistics, inspired by linguistic insights on word roles and lexical density. Our approach filters documents based on the mean and standard deviation of token priors, serving as a fast proxy to PPL while requiring no model inference. Despite its simplicity, the prior-based filter achieves the highest average performance across 20 downstream benchmarks, while reducing time cost by over 1000x compared to PPL-based filtering. We further demonstrate its applicability to symbolic languages such as code and math, and its dynamic adaptability to multilingual corpora without supervision 

---
# CCQA: Generating Question from Solution Can Improve Inference-Time Reasoning in SLMs 

**Authors**: Jin Young Kim, Ji Won Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2509.18536)  

**Abstract**: Recently, inference-time reasoning strategies have further improved the accuracy of large language models (LLMs), but their effectiveness on smaller models remains unclear. Based on the observation that conventional approaches often fail to improve performance in this context, we propose \textbf{C}ycle-\textbf{C}onsistency in \textbf{Q}uestion \textbf{A}nswering (CCQA), a novel reasoning method that can be effectively applied to SLMs. Inspired by cycle consistency, CCQA generates a question from each reasoning path and answer, evaluates each by its similarity to the original question, and then selects the candidate solution with the highest similarity score as the final response. Since conventional SLMs struggle to generate accurate questions from their own reasoning paths and answers, we employ a lightweight Flan-T5 model specialized for question generation to support this process efficiently. From the experimental results, it is verified that CCQA consistently outperforms existing state-of-the-art (SOTA) methods across eight models on mathematical and commonsense reasoning benchmarks. Furthermore, our method establishes a new practical baseline for efficient reasoning in SLMs. Source code can be found at this https URL. 

---
# Trace Is In Sentences: Unbiased Lightweight ChatGPT-Generated Text Detector 

**Authors**: Mo Mu, Dianqiao Lei, Chang Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.18535)  

**Abstract**: The widespread adoption of ChatGPT has raised concerns about its misuse, highlighting the need for robust detection of AI-generated text. Current word-level detectors are vulnerable to paraphrasing or simple prompts (PSP), suffer from biases induced by ChatGPT's word-level patterns (CWP) and training data content, degrade on modified text, and often require large models or online LLM interaction. To tackle these issues, we introduce a novel task to detect both original and PSP-modified AI-generated texts, and propose a lightweight framework that classifies texts based on their internal structure, which remains invariant under word-level changes. Our approach encodes sentence embeddings from pre-trained language models and models their relationships via attention. We employ contrastive learning to mitigate embedding biases from autoregressive generation and incorporate a causal graph with counterfactual methods to isolate structural features from topic-related biases. Experiments on two curated datasets, including abstract comparisons and revised life FAQs, validate the effectiveness of our method. 

---
# A Rhythm-Aware Phrase Insertion for Classical Arabic Poetry Composition 

**Authors**: Mohamad Elzohbi, Richard Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.18514)  

**Abstract**: This paper presents a methodology for inserting phrases in Arabic poems to conform to a specific rhythm using ByT5, a byte-level multilingual transformer-based model. Our work discusses a rule-based grapheme-to-beat transformation tailored for extracting the rhythm from fully diacritized Arabic script. Our approach employs a conditional denoising objective to fine-tune ByT5, where the model reconstructs masked words to match a target rhythm. We adopt a curriculum learning strategy, pre-training on a general Arabic dataset before fine-tuning on poetic dataset, and explore cross-lingual transfer from English to Arabic. Experimental results demonstrate that our models achieve high rhythmic alignment while maintaining semantic coherence. The proposed model has the potential to be used in co-creative applications in the process of composing classical Arabic poems. 

---
# Actions Speak Louder than Prompts: A Large-Scale Study of LLMs for Graph Inference 

**Authors**: Ben Finkelshtein, Silviu Cucerzan, Sujay Kumar Jauhar, Ryen White  

**Link**: [PDF](https://arxiv.org/pdf/2509.18487)  

**Abstract**: Large language models (LLMs) are increasingly used for text-rich graph machine learning tasks such as node classification in high-impact domains like fraud detection and recommendation systems. Yet, despite a surge of interest, the field lacks a principled understanding of the capabilities of LLMs in their interaction with graph data. In this work, we conduct a large-scale, controlled evaluation across several key axes of variability to systematically assess the strengths and weaknesses of LLM-based graph reasoning methods in text-based applications. The axes include the LLM-graph interaction mode, comparing prompting, tool-use, and code generation; dataset domains, spanning citation, web-link, e-commerce, and social networks; structural regimes contrasting homophilic and heterophilic graphs; feature characteristics involving both short- and long-text node attributes; and model configurations with varying LLM sizes and reasoning capabilities. We further analyze dependencies by methodically truncating features, deleting edges, and removing labels to quantify reliance on input types. Our findings provide practical and actionable guidance. (1) LLMs as code generators achieve the strongest overall performance on graph data, with especially large gains on long-text or high-degree graphs where prompting quickly exceeds the token budget. (2) All interaction strategies remain effective on heterophilic graphs, challenging the assumption that LLM-based methods collapse under low homophily. (3) Code generation is able to flexibly adapt its reliance between structure, features, or labels to leverage the most informative input type. Together, these findings provide a comprehensive view of the strengths and limitations of current LLM-graph interaction modes and highlight key design principles for future approaches. 

---
# LAWCAT: Efficient Distillation from Quadratic to Linear Attention with Convolution across Tokens for Long Context Modeling 

**Authors**: Zeyu Liu, Souvik Kundu, Lianghao Jiang, Anni Li, Srikanth Ronanki, Sravan Bodapati, Gourav Datta, Peter A. Beerel  

**Link**: [PDF](https://arxiv.org/pdf/2509.18467)  

**Abstract**: Although transformer architectures have achieved state-of-the-art performance across diverse domains, their quadratic computational complexity with respect to sequence length remains a significant bottleneck, particularly for latency-sensitive long-context applications. While recent linear-complexity alternatives are increasingly powerful, effectively training them from scratch is still resource-intensive. To overcome these limitations, we propose LAWCAT (Linear Attention with Convolution Across Time), a novel linearization framework designed to efficiently transfer the capabilities of pre-trained transformers into a performant linear attention architecture. LAWCAT integrates causal Conv1D layers to enhance local dependency modeling and employs normalized gated linear attention to improve generalization across varying context lengths. Our comprehensive evaluations demonstrate that, distilling Mistral-7B with only 1K-length sequences yields over 90\% passkey retrieval accuracy up to 22K tokens, significantly extending its effective context window. Similarly, Llama3.2-1B LAWCAT variant achieves competitive performance on S-NIAH 1\&2\&3 tasks (1K-8K context length) and BABILong benchmark (QA2\&QA3, 0K-16K context length), requiring less than 0.1\% pre-training tokens compared with pre-training models. Furthermore, LAWCAT exhibits faster prefill speeds than FlashAttention-2 for sequences exceeding 8K tokens. LAWCAT thus provides an efficient pathway to high-performance, long-context linear models suitable for edge deployment, reducing reliance on extensive long-sequence training data and computational resources. 

---
# CogniLoad: A Synthetic Natural Language Reasoning Benchmark With Tunable Length, Intrinsic Difficulty, and Distractor Density 

**Authors**: Daniel Kaiser, Arnoldo Frigessi, Ali Ramezani-Kebrya, Benjamin Ricaud  

**Link**: [PDF](https://arxiv.org/pdf/2509.18458)  

**Abstract**: Current benchmarks for long-context reasoning in Large Language Models (LLMs) often blur critical factors like intrinsic task complexity, distractor interference, and task length. To enable more precise failure analysis, we introduce CogniLoad, a novel synthetic benchmark grounded in Cognitive Load Theory (CLT). CogniLoad generates natural-language logic puzzles with independently tunable parameters that reflect CLT's core dimensions: intrinsic difficulty ($d$) controls intrinsic load; distractor-to-signal ratio ($\rho$) regulates extraneous load; and task length ($N$) serves as an operational proxy for conditions demanding germane load. Evaluating 22 SotA reasoning LLMs, CogniLoad reveals distinct performance sensitivities, identifying task length as a dominant constraint and uncovering varied tolerances to intrinsic complexity and U-shaped responses to distractor ratios. By offering systematic, factorial control over these cognitive load dimensions, CogniLoad provides a reproducible, scalable, and diagnostically rich tool for dissecting LLM reasoning limitations and guiding future model development. 

---
# Developing an AI framework to automatically detect shared decision-making in patient-doctor conversations 

**Authors**: Oscar J. Ponce-Ponte, David Toro-Tobon, Luis F. Figueroa, Michael Gionfriddo, Megan Branda, Victor M. Montori, Saturnino Luz, Juan P. Brito  

**Link**: [PDF](https://arxiv.org/pdf/2509.18439)  

**Abstract**: Shared decision-making (SDM) is necessary to achieve patient-centred care. Currently no methodology exists to automatically measure SDM at scale. This study aimed to develop an automated approach to measure SDM by using language modelling and the conversational alignment (CA) score. A total of 157 video-recorded patient-doctor conversations from a randomized multi-centre trial evaluating SDM decision aids for anticoagulation in atrial fibrillations were transcribed and segmented into 42,559 sentences. Context-response pairs and negative sampling were employed to train deep learning (DL) models and fine-tuned BERT models via the next sentence prediction (NSP) task. Each top-performing model was used to calculate four types of CA scores. A random-effects analysis by clinician, adjusting for age, sex, race, and trial arm, assessed the association between CA scores and SDM outcomes: the Decisional Conflict Scale (DCS) and the Observing Patient Involvement in Decision-Making 12 (OPTION12) scores. p-values were corrected for multiple comparisons with the Benjamini-Hochberg method. Among 157 patients (34% female, mean age 70 SD 10.8), clinicians on average spoke more words than patients (1911 vs 773). The DL model without the stylebook strategy achieved a recall@1 of 0.227, while the fine-tuned BERTbase (110M) achieved the highest recall@1 with 0.640. The AbsMax (18.36 SE7.74 p=0.025) and Max CA (21.02 SE7.63 p=0.012) scores generated with the DL without stylebook were associated with OPTION12. The Max CA score generated with the fine-tuned BERTbase (110M) was associated with the DCS score (-27.61 SE12.63 p=0.037). BERT model sizes did not have an impact the association between CA scores and SDM. This study introduces an automated, scalable methodology to measure SDM in patient-doctor conversations through explainable CA scores, with potential to evaluate SDM strategies at scale. 

---
# Evaluating the Creativity of LLMs in Persian Literary Text Generation 

**Authors**: Armin Tourajmehr, Mohammad Reza Modarres, Yadollah Yaghoobzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2509.18401)  

**Abstract**: Large language models (LLMs) have demonstrated notable creative abilities in generating literary texts, including poetry and short stories. However, prior research has primarily centered on English, with limited exploration of non-English literary traditions and without standardized methods for assessing creativity. In this paper, we evaluate the capacity of LLMs to generate Persian literary text enriched with culturally relevant expressions. We build a dataset of user-generated Persian literary spanning 20 diverse topics and assess model outputs along four creativity dimensions-originality, fluency, flexibility, and elaboration-by adapting the Torrance Tests of Creative Thinking. To reduce evaluation costs, we adopt an LLM as a judge for automated scoring and validate its reliability against human judgments using intraclass correlation coefficients, observing strong agreement. In addition, we analyze the models' ability to understand and employ four core literary devices: simile, metaphor, hyperbole, and antithesis. Our results highlight both the strengths and limitations of LLMs in Persian literary text generation, underscoring the need for further refinement. 

---
# NormGenesis: Multicultural Dialogue Generation via Exemplar-Guided Social Norm Modeling and Violation Recovery 

**Authors**: Minki Hong, Jangho Choi, Jihie Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.18395)  

**Abstract**: Social norms govern culturally appropriate behavior in communication, enabling dialogue systems to produce responses that are not only coherent but also socially acceptable. We present NormGenesis, a multicultural framework for generating and annotating socially grounded dialogues across English, Chinese, and Korean. To model the dynamics of social interaction beyond static norm classification, we propose a novel dialogue type, Violation-to-Resolution (V2R), which models the progression of conversations following norm violations through recognition and socially appropriate repair. To improve pragmatic consistency in underrepresented languages, we implement an exemplar-based iterative refinement early in the dialogue synthesis process. This design introduces alignment with linguistic, emotional, and sociocultural expectations before full dialogue generation begins. Using this framework, we construct a dataset of 10,800 multi-turn dialogues annotated at the turn level for norm adherence, speaker intent, and emotional response. Human and LLM-based evaluations demonstrate that NormGenesis significantly outperforms existing datasets in refinement quality, dialogue naturalness, and generalization performance. We show that models trained on our V2R-augmented data exhibit improved pragmatic competence in ethically sensitive contexts. Our work establishes a new benchmark for culturally adaptive dialogue modeling and provides a scalable methodology for norm-aware generation across linguistically and culturally diverse languages. 

---
# Interactive Real-Time Speaker Diarization Correction with Human Feedback 

**Authors**: Xinlu He, Yiwen Guan, Badrivishal Paurana, Zilin Dai, Jacob Whitehill  

**Link**: [PDF](https://arxiv.org/pdf/2509.18377)  

**Abstract**: Most automatic speech processing systems operate in "open loop" mode without user feedback about who said what; yet, human-in-the-loop workflows can potentially enable higher accuracy. We propose an LLM-assisted speaker diarization correction system that lets users fix speaker attribution errors in real time. The pipeline performs streaming ASR and diarization, uses an LLM to deliver concise summaries to the users, and accepts brief verbal feedback that is immediately incorporated without disrupting interactions. Moreover, we develop techniques to make the workflow more effective: First, a split-when-merged (SWM) technique detects and splits multi-speaker segments that the ASR erroneously attributes to just a single speaker. Second, online speaker enrollments are collected based on users' diarization corrections, thus helping to prevent speaker diarization errors from occurring in the future. LLM-driven simulations on the AMI test set indicate that our system substantially reduces DER by 9.92% and speaker confusion error by 44.23%. We further analyze correction efficacy under different settings, including summary vs full transcript display, the number of online enrollments limitation, and correction frequency. 

---
# Speech Vecalign: an Embedding-based Method for Aligning Parallel Speech Documents 

**Authors**: Chutong Meng, Philipp Koehn  

**Link**: [PDF](https://arxiv.org/pdf/2509.18360)  

**Abstract**: We present Speech Vecalign, a parallel speech document alignment method that monotonically aligns speech segment embeddings and does not depend on text transcriptions. Compared to the baseline method Global Mining, a variant of speech mining, Speech Vecalign produces longer speech-to-speech alignments. It also demonstrates greater robustness than Local Mining, another speech mining variant, as it produces less noise. We applied Speech Vecalign to 3,000 hours of unlabeled parallel English-German (En-De) speech documents from VoxPopuli, yielding about 1,000 hours of high-quality alignments. We then trained En-De speech-to-speech translation models on the aligned data. Speech Vecalign improves the En-to-De and De-to-En performance over Global Mining by 0.37 and 0.18 ASR-BLEU, respectively. Moreover, our models match or outperform SpeechMatrix model performance, despite using 8 times fewer raw speech documents. 

---
# Speculate Deep and Accurate: Lossless and Training-Free Acceleration for Offloaded LLMs via Substitute Speculative Decoding 

**Authors**: Pei-Shuo Wang, Jian-Jia Chen, Chun-Che Yang, Chi-Chih Chang, Ning-Chi Huang, Mohamed S. Abdelfattah, Kai-Chiang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.18344)  

**Abstract**: The immense model sizes of large language models (LLMs) challenge deployment on memory-limited consumer GPUs. Although model compression and parameter offloading are common strategies to address memory limitations, compression can degrade quality, and offloading maintains quality but suffers from slow inference. Speculative decoding presents a promising avenue to accelerate parameter offloading, utilizing a fast draft model to propose multiple draft tokens, which are then verified by the target LLM in parallel with a single forward pass. This method reduces the time-consuming data transfers in forward passes that involve offloaded weight transfers. Existing methods often rely on pretrained weights of the same family, but require additional training to align with custom-trained models. Moreover, approaches that involve draft model training usually yield only modest speedups. This limitation arises from insufficient alignment with the target model, preventing higher token acceptance lengths. To address these challenges and achieve greater speedups, we propose SubSpec, a plug-and-play method to accelerate parameter offloading that is lossless and training-free. SubSpec constructs a highly aligned draft model by generating low-bit quantized substitute layers from offloaded target LLM portions. Additionally, our method shares the remaining GPU-resident layers and the KV-Cache, further reducing memory overhead and enhance alignment. SubSpec achieves a high average acceptance length, delivering 9.1x speedup for Qwen2.5 7B on MT-Bench (8GB VRAM limit) and an average of 12.5x speedup for Qwen2.5 32B on popular generation benchmarks (24GB VRAM limit). 

---
# Brittleness and Promise: Knowledge Graph Based Reward Modeling for Diagnostic Reasoning 

**Authors**: Saksham Khatwani, He Cheng, Majid Afshar, Dmitriy Dligach, Yanjun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.18316)  

**Abstract**: Large language models (LLMs) show promise for diagnostic reasoning but often lack reliable, knowledge grounded inference. Knowledge graphs (KGs), such as the Unified Medical Language System (UMLS), offer structured biomedical knowledge that can support trustworthy reasoning. Prior approaches typically integrate KGs via retrieval augmented generation or fine tuning, inserting KG content into prompts rather than enabling structured reasoning. We explore an alternative paradigm: treating the LLM as a reward model of KG reasoning paths, where the model learns to judge whether a candidate path leads to correct diagnosis for a given patient input. This approach is inspired by recent work that leverages reward training to enhance model reasoning abilities, and grounded in computational theory, which suggests that verifying a solution is often easier than generating one from scratch. It also parallels physicians' diagnostic assessment, where they judge which sequences of findings and intermediate conditions most plausibly support a diagnosis. We first systematically evaluate five task formulation for knowledge path judging and eight training paradigm. Second, we test whether the path judging abilities generalize to downstream diagnostic tasks, including diagnosis summarization and medical question answering. Experiments with three open source instruct-tuned LLMs reveal both promise and brittleness: while specific reward optimization and distillation lead to strong path-judging performance, the transferability to downstream tasks remain weak. Our finding provides the first systematic assessment of "reward model style" reasoning over clinical KGs, offering insights into how structured, reward-based supervision influences diagnostic reasoning in GenAI systems for healthcare. 

---
# Exploiting Tree Structure for Credit Assignment in RL Training of LLMs 

**Authors**: Hieu Tran, Zonghai Yao, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.18314)  

**Abstract**: Reinforcement learning improves LLM reasoning, yet sparse delayed reward over long sequences makes token-level credit assignment the key bottleneck. We study the verifiable-reward setting, where the final answer is checkable and multiple responses can be drawn per prompt. Reasoning tasks in math and medical QA align with this setup, where only a few decision tokens significantly impact the outcome. PPO offers token-level advantages with a learned value model, but it is complex to train both the actor and critic models simultaneously, and it is not easily generalizable, as the token-level values from the critic model can make training prone to overfitting. GRPO is critic-free and supports verifiable rewards, but spreads a single sequence-level return across tokens and ignores branching. We introduce \textbf{Prefix-to-Tree (P2T)}, a simple procedure that converts a group of responses into a prefix tree and computes \emph{nonparametric} prefix values \(V(s)\) by aggregating descendant outcomes. Built on P2T, we propose \textbf{TEMPO} (\emph{\textbf{T}ree-\textbf{E}stimated \textbf{M}ean Prefix Value for \textbf{P}olicy \textbf{O}ptimization}), a critic-free algorithm that augments the group-relative outcome signal of GRPO with \emph{branch-gated} temporal-difference corrections derived from the tree. At non-branch tokens, the temporal-difference (TD) term is zero, so TEMPO reduces to GRPO; at branching tokens, it supplies precise token-level credit without a learned value network or extra judges/teachers. On Qwen3-1.7B/4B, TEMPO outperforms PPO and GRPO on in-distribution (MATH, MedQA) and out-of-distribution (GSM-HARD, AMC23, MedMCQA, MMLU-Medical) benchmarks, and reaches higher validation accuracy with roughly the same wall-clock time. 

---
# Evaluating Large Language Models for Detecting Antisemitism 

**Authors**: Jay Patel, Hrudayangam Mehta, Jeremy Blackburn  

**Link**: [PDF](https://arxiv.org/pdf/2509.18293)  

**Abstract**: Detecting hateful content is a challenging and important problem. Automated tools, like machine-learning models, can help, but they require continuous training to adapt to the ever-changing landscape of social media. In this work, we evaluate eight open-source LLMs' capability to detect antisemitic content, specifically leveraging in-context definition as a policy guideline. We explore various prompting techniques and design a new CoT-like prompt, Guided-CoT. Guided-CoT handles the in-context policy well, increasing performance across all evaluated models, regardless of decoding configuration, model sizes, or reasoning capability. Notably, Llama 3.1 70B outperforms fine-tuned GPT-3.5. Additionally, we examine LLM errors and introduce metrics to quantify semantic divergence in model-generated rationales, revealing notable differences and paradoxical behaviors among LLMs. Our experiments highlight the differences observed across LLMs' utility, explainability, and reliability. 

---
# ERFC: Happy Customers with Emotion Recognition and Forecasting in Conversation in Call Centers 

**Authors**: Aditi Debsharma, Bhushan Jagyasi, Surajit Sen, Priyanka Pandey, Devicharith Dovari, Yuvaraj V.C, Rosalin Parida, Gopali Contractor  

**Link**: [PDF](https://arxiv.org/pdf/2509.18175)  

**Abstract**: Emotion Recognition in Conversation has been seen to be widely applicable in call center analytics, opinion mining, finance, retail, healthcare, and other industries. In a call center scenario, the role of the call center agent is not just confined to receiving calls but to also provide good customer experience by pacifying the frustration or anger of the customers. This can be achieved by maintaining neutral and positive emotion from the agent. As in any conversation, the emotion of one speaker is usually dependent on the emotion of other speaker. Hence the positive emotion of an agent, accompanied with the right resolution will help in enhancing customer experience. This can change an unhappy customer to a happy one. Imparting the right resolution at right time becomes easier if the agent has the insight of the emotion of future utterances. To predict the emotions of the future utterances we propose a novel architecture, Emotion Recognition and Forecasting in Conversation. Our proposed ERFC architecture considers multi modalities, different attributes of emotion, context and the interdependencies of the utterances of the speakers in the conversation. Our intensive experiments on the IEMOCAP dataset have shown the feasibility of the proposed ERFC. This approach can provide a tremendous business value for the applications like call center, where the happiness of customer is utmost important. 

---
# SIRAG: Towards Stable and Interpretable RAG with A Process-Supervised Multi-Agent Framework 

**Authors**: Junlin Wang, Zehao Wu, Shaowei Lu, Yanlan Li, Xinghao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.18167)  

**Abstract**: Retrieval-Augmented Generation (RAG) enables large language models (LLMs) to access external knowledge sources, but the effectiveness of RAG relies on the coordination between the retriever and the generator. Since these components are developed independently, their interaction is often suboptimal: the retriever may return irrelevant or redundant documents, while the generator may fail to fully leverage retrieved evidence. In this work, we propose a process-supervised multi-agent framework to bridge the gap between retriever and generator. The framework introduces two lightweight agents: a Decision Maker, which determines when to continue retrieval or stop for answer generation, and a Knowledge Selector, which filters retrieved documents to retain only the most useful evidence. To provide fine-grained supervision, we employ an LLM-as-a-Judge that evaluates each intermediate action with process-level rewards, ensuring more accurate credit assignment than relying solely on final answer correctness. We further adopt a tree-structured rollout strategy to explore diverse reasoning paths, and train both agents with Proximal Policy Optimization (PPO) in an end-to-end manner. Experiments on single-hop and multi-hop question answering benchmarks show that our approach achieves higher accuracy, more stable convergence, and produces more interpretable reasoning trajectories compared with standard RAG baselines. Importantly, the proposed framework is modular and plug-and-play, requiring no modification to the retriever or generator, making it practical for real-world RAG applications. 

---
# Thinking in a Crowd: How Auxiliary Information Shapes LLM Reasoning 

**Authors**: Haodong Zhao, Chenyan Zhao, Yansi Li, Zhuosheng Zhang, Gongshen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.18163)  

**Abstract**: The capacity of Large Language Models (LLMs) to reason is fundamental to their application in complex, knowledge-intensive domains. In real-world scenarios, LLMs are often augmented with external information that can be helpful, irrelevant, or even misleading. This paper investigates the causal impact of such auxiliary information on the reasoning process of LLMs with explicit step-by-step thinking capabilities. We introduce SciAux, a new dataset derived from ScienceQA, to systematically test the robustness of the model against these types of information. Our findings reveal a critical vulnerability: the model's deliberative "thinking mode" is a double-edged sword. While helpful context improves accuracy, misleading information causes a catastrophic drop in performance, which is amplified by the thinking process. Instead of conferring robustness, thinking reinforces the degree of error when provided with misinformation. This highlights that the challenge is not merely to make models "think", but to endow them with the critical faculty to evaluate the information upon which their reasoning is based. The SciAux dataset is available at this https URL. 

---
# ZERA: Zero-init Instruction Evolving Refinement Agent - From Zero Instructions to Structured Prompts via Principle-based Optimization 

**Authors**: Seungyoun Yi, Minsoo Khang, Sungrae Park  

**Link**: [PDF](https://arxiv.org/pdf/2509.18158)  

**Abstract**: Automatic Prompt Optimization (APO) improves large language model (LLM) performance by refining prompts for specific tasks. However, prior APO methods typically focus only on user prompts, rely on unstructured feedback, and require large sample sizes and long iteration cycles-making them costly and brittle. We propose ZERA (Zero-init Instruction Evolving Refinement Agent), a novel framework that jointly optimizes both system and user prompts through principled, low-overhead refinement. ZERA scores prompts using eight generalizable criteria with automatically inferred weights, and revises prompts based on these structured critiques. This enables fast convergence to high-quality prompts using minimal examples and short iteration cycles. We evaluate ZERA across five LLMs and nine diverse datasets spanning reasoning, summarization, and code generation tasks. Experimental results demonstrate consistent improvements over strong baselines. Further ablation studies highlight the contribution of each component to more effective prompt construction. Our implementation including all prompts is publicly available at this https URL. 

---
# Event Causality Identification with Synthetic Control 

**Authors**: Haoyu Wang, Fengze Liu, Jiayao Zhang, Dan Roth, Kyle Richardson  

**Link**: [PDF](https://arxiv.org/pdf/2509.18156)  

**Abstract**: Event causality identification (ECI), a process that extracts causal relations between events from text, is crucial for distinguishing causation from correlation. Traditional approaches to ECI have primarily utilized linguistic patterns and multi-hop relational inference, risking false causality identification due to informal usage of causality and specious graphical inference. In this paper, we adopt the Rubin Causal Model to identify event causality: given two temporally ordered events, we see the first event as the treatment and the second one as the observed outcome. Determining their causality involves manipulating the treatment and estimating the resultant change in the likelihood of the outcome. Given that it is only possible to implement manipulation conceptually in the text domain, as a work-around, we try to find a twin for the protagonist from existing corpora. This twin should have identical life experiences with the protagonist before the treatment but undergoes an intervention of treatment. However, the practical difficulty of locating such a match limits its feasibility. Addressing this issue, we use the synthetic control method to generate such a twin' from relevant historical data, leveraging text embedding synthesis and inversion techniques. This approach allows us to identify causal relations more robustly than previous methods, including GPT-4, which is demonstrated on a causality benchmark, COPES-hard. 

---
# GAUSS: Benchmarking Structured Mathematical Skills for Large Language Models 

**Authors**: Yue Zhang, Jiaxin Zhang, Qiuyu Ren, Tahsin Saffat, Xiaoxuan Liu, Zitong Yang, Banghua Zhu, Yi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2509.18122)  

**Abstract**: We introduce \textbf{GAUSS} (\textbf{G}eneral \textbf{A}ssessment of \textbf{U}nderlying \textbf{S}tructured \textbf{S}kills in Mathematics), a benchmark that evaluates LLMs' mathematical abilities across twelve core skill dimensions, grouped into three domains: knowledge and understanding, problem solving and communication, and meta-skills and creativity. By categorizing problems according to cognitive skills and designing tasks that isolate specific abilities, GAUSS constructs comprehensive, fine-grained, and interpretable profiles of models' mathematical abilities. These profiles faithfully represent their underlying mathematical intelligence. To exemplify how to use the \textsc{GAUSS} benchmark, we have derived the skill profile of \textsc{GPT-5-thinking}, revealing its strengths and weaknesses as well as its differences relative to \textsc{o4-mini-high}, thereby underscoring the value of multidimensional, skill-based evaluation. 

---
# Dynamic Prompt Fusion for Multi-Task and Cross-Domain Adaptation in LLMs 

**Authors**: Xin Hu, Yue Kang, Guanzi Yao, Tianze Kang, Mengjie Wang, Heyao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.18113)  

**Abstract**: This study addresses the generalization limitations commonly observed in large language models under multi-task and cross-domain settings. Unlike prior methods such as SPoT, which depends on fixed prompt templates, our study introduces a unified multi-task learning framework with dynamic prompt scheduling mechanism. By introducing a prompt pool and a task-aware scheduling strategy, the method dynamically combines and aligns prompts for different tasks. This enhances the model's ability to capture semantic differences across tasks. During prompt fusion, the model uses task embeddings and a gating mechanism to finely control the prompt signals. This ensures alignment between prompt content and task-specific demands. At the same time, it builds flexible sharing pathways across tasks. In addition, the proposed optimization objective centers on joint multi-task learning. It incorporates an automatic learning strategy for scheduling weights, which effectively mitigates task interference and negative transfer. To evaluate the effectiveness of the method, a series of sensitivity experiments were conducted. These experiments examined the impact of prompt temperature parameters and task number variation. The results confirm the advantages of the proposed mechanism in maintaining model stability and enhancing transferability. Experimental findings show that the prompt scheduling method significantly improves performance on a range of language understanding and knowledge reasoning tasks. These results fully demonstrate its applicability and effectiveness in unified multi-task modeling and cross-domain adaptation. 

---
# Cross-Cultural Transfer of Commonsense Reasoning in LLMs: Evidence from the Arab World 

**Authors**: Saeed Almheiri, Rania Hossam, Mena Attia, Chenxi Wang, Preslav Nakov, Timothy Baldwin, Fajri Koto  

**Link**: [PDF](https://arxiv.org/pdf/2509.19265)  

**Abstract**: Large language models (LLMs) often reflect Western-centric biases, limiting their effectiveness in diverse cultural contexts. Although some work has explored cultural alignment, the potential for cross-cultural transfer, using alignment in one culture to improve performance in others, remains underexplored. This paper investigates cross-cultural transfer of commonsense reasoning in the Arab world, where linguistic and historical similarities coexist with local cultural differences. Using a culturally grounded commonsense reasoning dataset covering 13 Arab countries, we evaluate lightweight alignment methods such as in-context learning and demonstration-based reinforcement (DITTO), alongside baselines like supervised fine-tuning and direct preference optimization. Our results show that merely 12 culture-specific examples from one country can improve performance in others by 10\% on average, within multilingual models. In addition, we demonstrate that out-of-culture demonstrations from Indonesia and US contexts can match or surpass in-culture alignment for MCQ reasoning, highlighting cultural commonsense transferability beyond the Arab world. These findings demonstrate that efficient cross-cultural alignment is possible and offer a promising approach to adapt LLMs to low-resource cultural settings. 

---
# Finding My Voice: Generative Reconstruction of Disordered Speech for Automated Clinical Evaluation 

**Authors**: Karen Rosero, Eunjung Yeo, David R. Mortensen, Cortney Van't Slot, Rami R. Hallac, Carlos Busso  

**Link**: [PDF](https://arxiv.org/pdf/2509.19231)  

**Abstract**: We present ChiReSSD, a speech reconstruction framework that preserves children speaker's identity while suppressing mispronunciations. Unlike prior approaches trained on healthy adult speech, ChiReSSD adapts to the voices of children with speech sound disorders (SSD), with particular emphasis on pitch and prosody. We evaluate our method on the STAR dataset and report substantial improvements in lexical accuracy and speaker identity preservation. Furthermore, we automatically predict the phonetic content in the original and reconstructed pairs, where the proportion of corrected consonants is comparable to the percentage of correct consonants (PCC), a clinical speech assessment metric. Our experiments show Pearson correlation of 0.63 between automatic and human expert annotations, highlighting the potential to reduce the manual transcription burden. In addition, experiments on the TORGO dataset demonstrate effective generalization for reconstructing adult dysarthric speech. Our results indicate that disentangled, style-based TTS reconstruction can provide identity-preserving speech across diverse clinical populations. 

---
# Citrus-V: Advancing Medical Foundation Models with Unified Medical Image Grounding for Clinical Reasoning 

**Authors**: Guoxin Wang, Jun Zhao, Xinyi Liu, Yanbo Liu, Xuyang Cao, Chao Li, Zhuoyun Liu, Qintian Sun, Fangru Zhou, Haoqiang Xing, Zhenhong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19090)  

**Abstract**: Medical imaging provides critical evidence for clinical diagnosis, treatment planning, and surgical decisions, yet most existing imaging models are narrowly focused and require multiple specialized networks, limiting their generalization. Although large-scale language and multimodal models exhibit strong reasoning and multi-task capabilities, real-world clinical applications demand precise visual grounding, multimodal integration, and chain-of-thought reasoning. We introduce Citrus-V, a multimodal medical foundation model that combines image analysis with textual reasoning. The model integrates detection, segmentation, and multimodal chain-of-thought reasoning, enabling pixel-level lesion localization, structured report generation, and physician-like diagnostic inference in a single framework. We propose a novel multimodal training approach and release a curated open-source data suite covering reasoning, detection, segmentation, and document understanding tasks. Evaluations demonstrate that Citrus-V outperforms existing open-source medical models and expert-level imaging systems across multiple benchmarks, delivering a unified pipeline from visual grounding to clinical reasoning and supporting precise lesion quantification, automated reporting, and reliable second opinions. 

---
# ColorBlindnessEval: Can Vision-Language Models Pass Color Blindness Tests? 

**Authors**: Zijian Ling, Han Zhang, Yazhuo Zhou, Jiahao Cui  

**Link**: [PDF](https://arxiv.org/pdf/2509.19070)  

**Abstract**: This paper presents ColorBlindnessEval, a novel benchmark designed to evaluate the robustness of Vision-Language Models (VLMs) in visually adversarial scenarios inspired by the Ishihara color blindness test. Our dataset comprises 500 Ishihara-like images featuring numbers from 0 to 99 with varying color combinations, challenging VLMs to accurately recognize numerical information embedded in complex visual patterns. We assess 9 VLMs using Yes/No and open-ended prompts and compare their performance with human participants. Our experiments reveal limitations in the models' ability to interpret numbers in adversarial contexts, highlighting prevalent hallucination issues. These findings underscore the need to improve the robustness of VLMs in complex visual environments. ColorBlindnessEval serves as a valuable tool for benchmarking and improving the reliability of VLMs in real-world applications where accuracy is critical. 

---
# VIR-Bench: Evaluating Geospatial and Temporal Understanding of MLLMs via Travel Video Itinerary Reconstruction 

**Authors**: Hao Wang, Eiki Murata, Lingfang Zhang, Ayako Sato, So Fukuda, Ziqi Yin, Wentao Hu, Keisuke Nakao, Yusuke Nakamura, Sebastian Zwirner, Yi-Chia Chen, Hiroyuki Otomo, Hiroki Ouchi, Daisuke Kawahara  

**Link**: [PDF](https://arxiv.org/pdf/2509.19002)  

**Abstract**: Recent advances in multimodal large language models (MLLMs) have significantly enhanced video understanding capabilities, opening new possibilities for practical applications. Yet current video benchmarks focus largely on indoor scenes or short-range outdoor activities, leaving the challenges associated with long-distance travel largely unexplored. Mastering extended geospatial-temporal trajectories is critical for next-generation MLLMs, underpinning real-world tasks such as embodied-AI planning and navigation. To bridge this gap, we present VIR-Bench, a novel benchmark consisting of 200 travel videos that frames itinerary reconstruction as a challenging task designed to evaluate and push forward MLLMs' geospatial-temporal intelligence. Experimental results reveal that state-of-the-art MLLMs, including proprietary ones, struggle to achieve high scores, underscoring the difficulty of handling videos that span extended spatial and temporal scales. Moreover, we conduct an in-depth case study in which we develop a prototype travel-planning agent that leverages the insights gained from VIR-Bench. The agent's markedly improved itinerary recommendations verify that our evaluation protocol not only benchmarks models effectively but also translates into concrete performance gains in user-facing applications. 

---
# Failure Makes the Agent Stronger: Enhancing Accuracy through Structured Reflection for Reliable Tool Interactions 

**Authors**: Junhao Su, Yuanliang Wan, Junwei Yang, Hengyu Shi, Tianyang Han, Junfeng Luo, Yurui Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2509.18847)  

**Abstract**: Tool-augmented large language models (LLMs) are usually trained with supervised imitation or coarse-grained reinforcement learning that optimizes single tool calls. Current self-reflection practices rely on heuristic prompts or one-way reasoning: the model is urged to 'think more' instead of learning error diagnosis and repair. This is fragile in multi-turn interactions; after a failure the model often repeats the same mistake. We propose structured reflection, which turns the path from error to repair into an explicit, controllable, and trainable action. The agent produces a short yet precise reflection: it diagnoses the failure using evidence from the previous step and then proposes a correct, executable follow-up call. For training we combine DAPO and GSPO objectives with a reward scheme tailored to tool use, optimizing the stepwise strategy Reflect, then Call, then Final. To evaluate, we introduce Tool-Reflection-Bench, a lightweight benchmark that programmatically checks structural validity, executability, parameter correctness, and result consistency. Tasks are built as mini trajectories of erroneous call, reflection, and corrected call, with disjoint train and test splits. Experiments on BFCL v3 and Tool-Reflection-Bench show large gains in multi-turn tool-call success and error recovery, and a reduction of redundant calls. These results indicate that making reflection explicit and optimizing it directly improves the reliability of tool interaction and offers a reproducible path for agents to learn from failure. 

---
# Pay More Attention To Audio: Mitigating Imbalance of Cross-Modal Attention in Large Audio Language Models 

**Authors**: Junyu Wang, Ziyang Ma, Zhengding Luo, Tianrui Wang, Meng Ge, Xiaobao Wang, Longbiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.18816)  

**Abstract**: Large Audio-Language Models (LALMs) often suffer from audio-textual attention imbalance, prioritizing text over acoustic information, particularly in the multi-modal fusion layers of the Transformer architecture. This bias hinders their ability to fully utilize acoustic cues, causing suboptimal performance on audio reasoning tasks. To mitigate this, we propose \textbf{MATA}, a novel training-free method that dynamically pushes LALMs to pay \textbf{M}ore \textbf{A}ttention \textbf{T}o \textbf{A}udio tokens within the self-attention mechanism. Specifically, MATA intervenes post raw attention scoring, targeting only the last token in intermediate layers without introducing additional parameters or computational overhead. Experiments on the MMAU and MMAR benchmarks confirm MATA's effectiveness, with consistent performance gains. Notably, on MMAR, MATA enables an open-source model to surpass the proprietary Gemini 2.0 Flash for the first time. Our work provides an efficient solution to mitigate attention bias and opens a new research direction for enhancing the audio-processing capabilities of multi-modal models. 

---
# Agentic AutoSurvey: Let LLMs Survey LLMs 

**Authors**: Yixin Liu, Yonghui Wu, Denghui Zhang, Lichao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.18661)  

**Abstract**: The exponential growth of scientific literature poses unprecedented challenges for researchers attempting to synthesize knowledge across rapidly evolving fields. We present \textbf{Agentic AutoSurvey}, a multi-agent framework for automated survey generation that addresses fundamental limitations in existing approaches. Our system employs four specialized agents (Paper Search Specialist, Topic Mining \& Clustering, Academic Survey Writer, and Quality Evaluator) working in concert to generate comprehensive literature surveys with superior synthesis quality. Through experiments on six representative LLM research topics from COLM 2024 categories, we demonstrate that our multi-agent approach achieves significant improvements over existing baselines, scoring 8.18/10 compared to AutoSurvey's 4.77/10. The multi-agent architecture processes 75--443 papers per topic (847 total across six topics) while targeting high citation coverage (often $\geq$80\% on 75--100-paper sets; lower on very large sets such as RLHF) through specialized agent orchestration. Our 12-dimension evaluation captures organization, synthesis integration, and critical analysis beyond basic metrics. These findings demonstrate that multi-agent architectures represent a meaningful advancement for automated literature survey generation in rapidly evolving scientific domains. 

---
# OraPO: Oracle-educated Reinforcement Learning for Data-efficient and Factual Radiology Report Generation 

**Authors**: Zhuoxiao Chen, Hongyang Yu, Ying Xu, Yadan Luo, Long Duong, Yuan-Fang Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.18600)  

**Abstract**: Radiology report generation (RRG) aims to automatically produce clinically faithful reports from chest X-ray images. Prevailing work typically follows a scale-driven paradigm, by multi-stage training over large paired corpora and oversized backbones, making pipelines highly data- and compute-intensive. In this paper, we propose Oracle-educated GRPO {OraPO) with a FactScore-based reward (FactS) to tackle the RRG task under constrained budgets. OraPO enables single-stage, RL-only training by converting failed GRPO explorations on rare or difficult studies into direct preference supervision via a lightweight oracle step. FactS grounds learning in diagnostic evidence by extracting atomic clinical facts and checking entailment against ground-truth labels, yielding dense, interpretable sentence-level rewards. Together, OraPO and FactS create a compact and powerful framework that significantly improves learning efficiency on clinically challenging cases, setting the new SOTA performance on the CheXpert Plus dataset (0.341 in F1) with 2--3 orders of magnitude less training data using a small base VLM on modest hardware. 

---
# Teaching Audio Models to Reason: A Unified Framework for Source- and Layer-wise Distillation 

**Authors**: Runyan Yang, Yuke Si, Yingying Gao, Junlan Feng, Chao Deng, Shilei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.18579)  

**Abstract**: While large audio language models excel at tasks like ASR and emotion recognition, they still struggle with complex reasoning due to the modality gap between audio and text as well as the lack of structured intermediate supervision. To address this, we propose a unified knowledge distillation framework to transfer reasoning capabilities from a high-capacity textual teacher model to a student audio models while preserving its acoustic competence. Our method introduces two key dimensions: source-wise distillation, which leverages both textual and acoustic teachers to provide complementary modality-specific supervision; and layer-wise distillation, which aligns teacher signals with appropriate student layers to improve transfer efficiency. This dual-dimensional strategy enables fine-grained control over the distillation process, effectively bridging the gap between symbolic reasoning and speech representations. Experimental results show significant improvements in audio reasoning performance, demonstrating the effectiveness of our framework as a reasoning transfer solution for audio modeling. 

---
# HarmoniFuse: A Component-Selective and Prompt-Adaptive Framework for Multi-Task Speech Language Modeling 

**Authors**: Yuke Si, Runyan Yang, Yingying Gao, Junlan Feng, Chao Deng, Shilei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.18570)  

**Abstract**: Recent advances in large language models have facilitated the development of unified speech language models (SLMs) capable of supporting multiple speech tasks within a shared architecture. However, tasks such as automatic speech recognition (ASR) and speech emotion recognition (SER) rely on distinct types of information: ASR primarily depends on linguistic content, whereas SER requires the integration of both linguistic and paralinguistic cues. Existing multitask SLMs typically adopt naive parameter sharing or prompt-based conditioning without explicitly modeling the differences in information composition required by each task. Such designs risk task interference and performance degradation, especially under limited data conditions. To address these limitations, we propose HarmoniFuse, a component-selective and prompt-adaptive framework for multi-task speech language modeling. HarmoniFuse is designed to harmonize heterogeneous task demands by selecting and fusing task-relevant components of speech representations. Specifically, it integrates a gated speech encoder to extract task-specific acoustic features and a prompt-adaptive dynamic fusion module to aggregate transformer layers based on task characteristics. In addition, a batch-interleaved training strategy enables leveraging separate ASR and SER datasets without requiring joint annotation. Experimental results demonstrate that HarmoniFuse improves both ASR and SER performance, offering a scalable and robust solution for multitask speech understanding under realistic data constraints. 

---
# No Verifiable Reward for Prosody: Toward Preference-Guided Prosody Learning in TTS 

**Authors**: Seungyoun Shin, Dongha Ahn, Jiwoo Kim, Sungwook Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2509.18531)  

**Abstract**: Recent work reports gains in neural text-to-speech (TTS) with Group Relative Policy Optimization (GRPO). However, in the absence of a verifiable reward for \textit{prosody}, GRPO trained on transcription-oriented signals (CER/NLL) lowers error rates yet collapses prosody into monotone, unnatural speech; adding speaker-similarity further destabilizes training and degrades CER. We address this with an \textit{iterative Direct Preference Optimization (DPO)} scheme that uses only a few hundred human-labeled preference pairs per round to directly optimize prosodic naturalness while regularizing to the current model. On \textbf{KoCC-TTS}, a curated dataset of authentic Korean call center interactions capturing task-oriented dialogues, our method attains the highest human preference (ELO) with competitive CER, outperforming GRPO and strong commercial baselines. These results suggest that when prosody cannot be rewarded automatically, \textit{human preference optimization} offers a practical and data-efficient path to natural and robust TTS. The demo page is available at \href{this https URL} 

---
# Memory-QA: Answering Recall Questions Based on Multimodal Memories 

**Authors**: Hongda Jiang, Xinyuan Zhang, Siddhant Garg, Rishab Arora, Shiun-Zu Kuo, Jiayang Xu, Christopher Brossman, Yue Liu, Aaron Colak, Ahmed Aly, Anuj Kumar, Xin Luna Dong  

**Link**: [PDF](https://arxiv.org/pdf/2509.18436)  

**Abstract**: We introduce Memory-QA, a novel real-world task that involves answering recall questions about visual content from previously stored multimodal memories. This task poses unique challenges, including the creation of task-oriented memories, the effective utilization of temporal and location information within memories, and the ability to draw upon multiple memories to answer a recall question. To address these challenges, we propose a comprehensive pipeline, Pensieve, integrating memory-specific augmentation, time- and location-aware multi-signal retrieval, and multi-memory QA fine-tuning. We created a multimodal benchmark to illustrate various real challenges in this task, and show the superior performance of Pensieve over state-of-the-art solutions (up to 14% on QA accuracy). 

---
# The Illusion of Readiness: Stress Testing Large Frontier Models on Multimodal Medical Benchmarks 

**Authors**: Yu Gu, Jingjing Fu, Xiaodong Liu, Jeya Maria Jose Valanarasu, Noel Codella, Reuben Tan, Qianchu Liu, Ying Jin, Sheng Zhang, Jinyu Wang, Rui Wang, Lei Song, Guanghui Qin, Naoto Usuyama, Cliff Wong, Cheng Hao, Hohin Lee, Praneeth Sanapathi, Sarah Hilado, Bian Jiang, Javier Alvarez-Valle, Mu Wei, Jianfeng Gao, Eric Horvitz, Matt Lungren, Hoifung Poon, Paul Vozila  

**Link**: [PDF](https://arxiv.org/pdf/2509.18234)  

**Abstract**: Large frontier models like GPT-5 now achieve top scores on medical benchmarks. But our stress tests tell a different story. Leading systems often guess correctly even when key inputs like images are removed, flip answers under trivial prompt changes, and fabricate convincing yet flawed reasoning. These aren't glitches; they expose how today's benchmarks reward test-taking tricks over medical understanding. We evaluate six flagship models across six widely used benchmarks and find that high leaderboard scores hide brittleness and shortcut learning. Through clinician-guided rubric evaluation, we show that benchmarks vary widely in what they truly measure yet are treated interchangeably, masking failure modes. We caution that medical benchmark scores do not directly reflect real-world readiness. If we want AI to earn trust in healthcare, we must demand more than leaderboard wins and must hold systems accountable for robustness, sound reasoning, and alignment with real medical demands. 

---
# Conversational Orientation Reasoning: Egocentric-to-Allocentric Navigation with Multimodal Chain-of-Thought 

**Authors**: Yu Ti Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.18200)  

**Abstract**: Conversational agents must translate egocentric utterances (e.g., "on my right") into allocentric orientations (N/E/S/W). This challenge is particularly critical in indoor or complex facilities where GPS signals are weak and detailed maps are unavailable. While chain-of-thought (CoT) prompting has advanced reasoning in language and vision tasks, its application to multimodal spatial orientation remains underexplored. We introduce Conversational Orientation Reasoning (COR), a new benchmark designed for Traditional Chinese conversational navigation projected from real-world environments, addressing egocentric-to-allocentric reasoning in non-English and ASR-transcribed scenarios. We propose a multimodal chain-of-thought (MCoT) framework, which integrates ASR-transcribed speech with landmark coordinates through a structured three-step reasoning process: (1) extracting spatial relations, (2) mapping coordinates to absolute directions, and (3) inferring user orientation. A curriculum learning strategy progressively builds these capabilities on Taiwan-LLM-13B-v2.0-Chat, a mid-sized model representative of resource-constrained settings. Experiments show that MCoT achieves 100% orientation accuracy on clean transcripts and 98.1% with ASR transcripts, substantially outperforming unimodal and non-structured baselines. Moreover, MCoT demonstrates robustness under noisy conversational conditions, including ASR recognition errors and multilingual code-switching. The model also maintains high accuracy in cross-domain evaluation and resilience to linguistic variation, domain shift, and referential ambiguity. These findings highlight the potential of structured MCoT spatial reasoning as a path toward interpretable and resource-efficient embodied navigation. 

---
# Baseer: A Vision-Language Model for Arabic Document-to-Markdown OCR 

**Authors**: Khalil Hennara, Muhammad Hreden, Mohamed Motasim Hamed, Ahmad Bastati, Zeina Aldallal, Sara Chrouf, Safwan AlModhayan  

**Link**: [PDF](https://arxiv.org/pdf/2509.18174)  

**Abstract**: Arabic document OCR remains a challenging task due to the language's cursive script, diverse fonts, diacritics, and right-to-left orientation. While modern Multimodal Large Language Models (MLLMs) have advanced document understanding for high-resource languages, their performance on Arabic remains limited. In this work, we introduce Baseer, a vision-language model fine- tuned specifically for Arabic document OCR. Leveraging a large-scale dataset combining synthetic and real-world documents, Baseer is trained using a decoder-only fine-tuning strategy to adapt a pre-trained MLLM while preserving general visual features. We also present Misraj-DocOCR, a high-quality, expert-verified benchmark designed for rigorous evaluation of Arabic OCR systems. Our experiments show that Baseer significantly outperforms existing open-source and commercial solutions, achieving a WER of 0.25 and establishing a new state-of-the-art in the domain of Arabic document OCR. Our results highlight the benefits of domain-specific adaptation of general-purpose MLLMs and establish a strong baseline for high-accuracy OCR on morphologically rich languages like Arabic. 

---
# TurnBack: A Geospatial Route Cognition Benchmark for Large Language Models through Reverse Route 

**Authors**: Hongyi Luo, Qing Cheng, Daniel Matos, Hari Krishna Gadi, Yanfeng Zhang, Lu Liu, Yongliang Wang, Niclas Zeller, Daniel Cremers, Liqiu Meng  

**Link**: [PDF](https://arxiv.org/pdf/2509.18173)  

**Abstract**: Humans can interpret geospatial information through natural language, while the geospatial cognition capabilities of Large Language Models (LLMs) remain underexplored. Prior research in this domain has been constrained by non-quantifiable metrics, limited evaluation datasets and unclear research hierarchies. Therefore, we propose a large-scale benchmark and conduct a comprehensive evaluation of the geospatial route cognition of LLMs. We create a large-scale evaluation dataset comprised of 36000 routes from 12 metropolises worldwide. Then, we introduce PathBuilder, a novel tool for converting natural language instructions into navigation routes, and vice versa, bridging the gap between geospatial information and natural language. Finally, we propose a new evaluation framework and metrics to rigorously assess 11 state-of-the-art (SOTA) LLMs on the task of route reversal. The benchmark reveals that LLMs exhibit limitation to reverse routes: most reverse routes neither return to the starting point nor are similar to the optimal route. Additionally, LLMs face challenges such as low robustness in route generation and high confidence for their incorrect answers. Code\ \&\ Data available here: \href{this https URL}{TurnBack.} 

---
# PiMoE: Token-Level Routing for Integrating High-Precision Computation and Reasoning 

**Authors**: Hengbo Xiao, Jingyuan Fan, Xin Tong, Jingzhao Zhang, Chao Lu, Guannan He  

**Link**: [PDF](https://arxiv.org/pdf/2509.18169)  

**Abstract**: Complex systems typically rely on high-precision numerical computation to support decisions, but current large language models (LLMs) cannot yet incorporate such computations as an intrinsic and interpretable capability with existing architectures. Mainstream multi-agent approaches can leverage external experts, but inevitably introduce communication overhead and suffer from inefficient multimodal emergent capability and limited scalability. To this end, we propose PiMoE (Physically-isolated Mixture of Experts), a training and inference architecture for integrating computation and reasoning. Instead of the workflow paradigm of tool invocation, PiMoE endogenously integrates computational capabilities into neural networks after separately training experts, a text-to-computation module, and a router. At inference, the router directs computation and reasoning at the token level, thereby enabling iterative alternation within a single chain of thought. We evaluate PiMoE on two reasoning-computation tasks against LLM finetuning and the multi-agent system approaches. Results show that the PiMoE architecture achieves not only higher accuracy than directly finetuning LLMs but also significant improvements in response latency, token usage, and GPU energy consumption compared with mainstream multi-agent approaches. PiMoE offers an efficient, interpretable, and scalable paradigm for next-generation scientific or industrial intelligent systems. 

---
# Safe-SAIL: Towards a Fine-grained Safety Landscape of Large Language Models via Sparse Autoencoder Interpretation Framework 

**Authors**: Jiaqi Weng, Han Zheng, Hanyu Zhang, Qinqin He, Jialing Tao, Hui Xue, Zhixuan Chu, Xiting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.18127)  

**Abstract**: Increasing deployment of large language models (LLMs) in real-world applications raises significant safety concerns. Most existing safety research focuses on evaluating LLM outputs or specific safety tasks, limiting their ability to ad- dress broader, undefined risks. Sparse Autoencoders (SAEs) facilitate interpretability research to clarify model behavior by explaining single-meaning atomic features decomposed from entangled signals. jHowever, prior applications on SAEs do not interpret features with fine-grained safety-related con- cepts, thus inadequately addressing safety-critical behaviors, such as generating toxic responses and violating safety regu- lations. For rigorous safety analysis, we must extract a rich and diverse set of safety-relevant features that effectively capture these high-risk behaviors, yet face two challenges: identifying SAEs with the greatest potential for generating safety concept-specific neurons, and the prohibitively high cost of detailed feature explanation. In this paper, we pro- pose Safe-SAIL, a framework for interpreting SAE features within LLMs to advance mechanistic understanding in safety domains. Our approach systematically identifies SAE with best concept-specific interpretability, explains safety-related neurons, and introduces efficient strategies to scale up the in- terpretation process. We will release a comprehensive toolkit including SAE checkpoints and human-readable neuron ex- planations, which supports empirical analysis of safety risks to promote research on LLM safety. 

---
