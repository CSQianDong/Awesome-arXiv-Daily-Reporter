# Test Set Quality in Multilingual LLM Evaluation 

**Authors**: Kranti Chalamalasetti, Gabriel Bernier-Colborne, Yvan Gauthier, Sowmya Vajjala  

**Link**: [PDF](https://arxiv.org/pdf/2508.02635)  

**Abstract**: Several multilingual benchmark datasets have been developed in a semi-automatic manner in the recent past to measure progress and understand the state-of-the-art in the multilingual capabilities of Large Language Models. However, there is not a lot of attention paid to the quality of the datasets themselves, despite the existence of previous work in identifying errors in even fully human-annotated test sets. In this paper, we manually analyze recent multilingual evaluation sets in two languages - French and Telugu, identifying several errors in the process. We compare the performance difference across several LLMs with the original and revised versions of the datasets and identify large differences (almost 10% in some cases) in both languages). Based on these results, we argue that test sets should not be considered immutable and should be revisited, checked for correctness, and potentially versioned. We end with some recommendations for both the dataset creators as well as consumers on addressing the dataset quality issues. 

---
# Pointer: Linear-Complexity Long-Range Modeling without Pre-training 

**Authors**: Zixi Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.02631)  

**Abstract**: We introduce Pointer, a novel architecture that achieves linear $O(NK)$ complexity for long-range sequence modeling while maintaining superior performance without requiring pre-training. Unlike standard attention mechanisms that compute $O(N^2)$ pairwise interactions, our approach uses layer-wise pointer chaining where each layer's pointer selection depends on previous layer's pointer positions, creating explicit long-distance connections through pointer chains. We demonstrate that this architecture achieves $2$--$10\times$ speedup on long sequences compared to standard transformers, maintains $>95\%$ accuracy on copy tasks at distances up to 2048 tokens, and learns interpretable pointer patterns that reveal structured dependency modeling. Our experiments on efficiency benchmarks, long-range dependency tasks, and interpretability analysis show that Pointer offers a compelling alternative to attention mechanisms for scenarios requiring efficient long-range modeling without pre-training dependencies. 

---
# Mitigating Attention Hacking in Preference-Based Reward Modeling via Interaction Distillation 

**Authors**: Jianxiang Zang, Meiling Ning, Shihan Dou, Jiazheng Zhang, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02618)  

**Abstract**: The reward model (RM), as the core component of reinforcement learning from human feedback (RLHF) for large language models (LLMs), responsible for providing reward signals to generated responses. However, mainstream preference modeling in RM is inadequate in terms of token-level interaction, making its judgment signals vulnerable to being hacked by misallocated attention to context. This stems from two fundamental limitations: (1) Current preference modeling employs decoder-only architectures, where the unidirectional causal attention mechanism leads to forward-decaying intra-sequence attention within the prompt-response sequence. (2) The independent Siamese-encoding paradigm induces the absence of token-level inter-sequence attention between chosen and rejected sequences. To address this "attention hacking", we propose "Interaction Distillation", a novel training framework for more adequate preference modeling through attention-level optimization. The method introduces an interaction-based natural language understanding model as the teacher to provide sophisticated token interaction patterns via comprehensive attention, and guides the preference modeling to simulate teacher model's interaction pattern through an attentional alignment objective. Through extensive experiments, interaction distillation has demonstrated its ability to provide more stable and generalizable reward signals compared to state-of-the-art RM optimization methods that target data noise, highlighting the attention hacking constitute a more fundamental limitation in RM. 

---
# CharBench: Evaluating the Role of Tokenization in Character-Level Tasks 

**Authors**: Omri Uzan, Yuval Pinter  

**Link**: [PDF](https://arxiv.org/pdf/2508.02591)  

**Abstract**: Tasks that require character-level reasoning, such as counting or locating characters within words, remain challenging for contemporary language models. A common conjecture is that language models' reliance on subword units, rather than characters, contributes to their struggles with character-level tasks, yet recent studies offer conflicting conclusions about the role of tokenization, leaving its impact unclear. To address this gap, we introduce CharBench, a comprehensive benchmark of character-level tasks that is two orders of magnitude larger than existing alternatives. We evaluate a diverse range of leading open-weight and proprietary models on CharBench and find that it presents a significant challenge to modern LLMs, with an average accuracy of 43.6% and 32.3% on some tasks. We present an in-depth analysis of how intrinsic properties of words and their segmentations into tokens correspond to model performance. For counting tasks, we find that tokenization properties are weakly correlated with correctness, while the length of the queried word and the actual character count play a more significant part. In contrast, for tasks requiring intra-word positional understanding, performance is negatively correlated with the length of the token containing the queried character, suggesting that longer tokens obscure character position information for LLMs. We encourage future work to build on the benchmark and evaluation methodology introduced here as tools for improving model performance on such tasks. 

---
# MArgE: Meshing Argumentative Evidence from Multiple Large Language Models for Justifiable Claim Verification 

**Authors**: Ming Pok Ng, Junqi Jiang, Gabriel Freedman, Antonio Rago, Francesca Toni  

**Link**: [PDF](https://arxiv.org/pdf/2508.02584)  

**Abstract**: Leveraging outputs from multiple large language models (LLMs) is emerging as a method for harnessing their power across a wide range of tasks while mitigating their capacity for making errors, e.g., hallucinations. However, current approaches to combining insights from multiple LLMs often involve unstructured interactions (e.g., free debate), resulting in model generations that are not faithfully justifiable. In this work, we introduce MArgE, a novel framework to provide formal structure to the evidence from each LLM, in the form of a tree of extracted arguments, for the task of claim verification. We use a variant of Argumentative LLMs (ArgLLMs), i.e. LLMs driven by frameworks and semantics from the field of computational argumentation, to construct structured argument trees for given claims. This process creates an inspectable pathway from the initial arguments to the final claim verification decisions, providing a faithful justification thereof. We show experimentally that MArgE can significantly outperform single LLMs, including three open-source models (4B to 8B parameters), GPT-4o-mini and existing ArgLLMs, as well as prior methods for unstructured multi-LLM debates. We thus demonstrate the advantages of incorporating formal, argumentative reasoning mechanisms when combining multiple LLM outputs. 

---
# EHSAN: Leveraging ChatGPT in a Hybrid Framework for Arabic Aspect-Based Sentiment Analysis in Healthcare 

**Authors**: Eman Alamoudi, Ellis Solaiman  

**Link**: [PDF](https://arxiv.org/pdf/2508.02574)  

**Abstract**: Arabic-language patient feedback remains under-analysed because dialect diversity and scarce aspect-level sentiment labels hinder automated assessment. To address this gap, we introduce EHSAN, a data-centric hybrid pipeline that merges ChatGPT pseudo-labelling with targeted human review to build the first explainable Arabic aspect-based sentiment dataset for healthcare. Each sentence is annotated with an aspect and sentiment label (positive, negative, or neutral), forming a pioneering Arabic dataset aligned with healthcare themes, with ChatGPT-generated rationales provided for each label to enhance transparency. To evaluate the impact of annotation quality on model performance, we created three versions of the training data: a fully supervised set with all labels reviewed by humans, a semi-supervised set with 50% human review, and an unsupervised set with only machine-generated labels. We fine-tuned two transformer models on these datasets for both aspect and sentiment classification. Experimental results show that our Arabic-specific model achieved high accuracy even with minimal human supervision, reflecting only a minor performance drop when using ChatGPT-only labels. Reducing the number of aspect classes notably improved classification metrics across the board. These findings demonstrate an effective, scalable approach to Arabic aspect-based sentiment analysis (SA) in healthcare, combining large language model annotation with human expertise to produce a robust and explainable dataset. Future directions include generalisation across hospitals, prompt refinement, and interpretable data-driven modelling. 

---
# Guess or Recall? Training CNNs to Classify and Localize Memorization in LLMs 

**Authors**: Jérémie Dentan, Davide Buscaldi, Sonia Vanier  

**Link**: [PDF](https://arxiv.org/pdf/2508.02573)  

**Abstract**: Verbatim memorization in Large Language Models (LLMs) is a multifaceted phenomenon involving distinct underlying mechanisms. We introduce a novel method to analyze the different forms of memorization described by the existing taxonomy. Specifically, we train Convolutional Neural Networks (CNNs) on the attention weights of the LLM and evaluate the alignment between this taxonomy and the attention weights involved in decoding.
We find that the existing taxonomy performs poorly and fails to reflect distinct mechanisms within the attention blocks. We propose a new taxonomy that maximizes alignment with the attention weights, consisting of three categories: memorized samples that are guessed using language modeling abilities, memorized samples that are recalled due to high duplication in the training set, and non-memorized samples. Our results reveal that few-shot verbatim memorization does not correspond to a distinct attention mechanism. We also show that a significant proportion of extractable samples are in fact guessed by the model and should therefore be studied separately. Finally, we develop a custom visual interpretability technique to localize the regions of the attention weights involved in each form of memorization. 

---
# Sparse-dLLM: Accelerating Diffusion LLMs with Dynamic Cache Eviction 

**Authors**: Yuerong Song, Xiaoran Liu, Ruixiao Li, Zhigeng Liu, Zengfeng Huang, Qipeng Guo, Ziwei He, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02558)  

**Abstract**: Diffusion Large Language Models (dLLMs) enable breakthroughs in reasoning and parallel decoding but suffer from prohibitive quadratic computational complexity and memory overhead during inference. Current caching techniques accelerate decoding by storing full-layer states, yet impose substantial memory usage that limit long-context applications. Our analysis of attention patterns in dLLMs reveals persistent cross-layer sparsity, with pivotal tokens remaining salient across decoding steps and low-relevance tokens staying unimportant, motivating selective cache eviction. We propose Sparse-dLLM, the first training-free framework integrating dynamic cache eviction with sparse attention via delayed bidirectional sparse caching. By leveraging the stability of token saliency over steps, it retains critical tokens and dynamically evicts unimportant prefix/suffix entries using an attention-guided strategy. Extensive experiments on LLaDA and Dream series demonstrate Sparse-dLLM achieves up to 10$\times$ higher throughput than vanilla dLLMs, with comparable performance and similar peak memory costs, outperforming previous methods in efficiency and effectiveness. 

---
# Automated SNOMED CT Concept Annotation in Clinical Text Using Bi-GRU Neural Networks 

**Authors**: Ali Noori, Pratik Devkota, Somya Mohanty, Prashanti Manda  

**Link**: [PDF](https://arxiv.org/pdf/2508.02556)  

**Abstract**: Automated annotation of clinical text with standardized medical concepts is critical for enabling structured data extraction and decision support. SNOMED CT provides a rich ontology for labeling clinical entities, but manual annotation is labor-intensive and impractical at scale. This study introduces a neural sequence labeling approach for SNOMED CT concept recognition using a Bidirectional GRU model. Leveraging a subset of MIMIC-IV, we preprocess text with domain-adapted SpaCy and SciBERT-based tokenization, segmenting sentences into overlapping 19-token chunks enriched with contextual, syntactic, and morphological features. The Bi-GRU model assigns IOB tags to identify concept spans and achieves strong performance with a 90 percent F1-score on the validation set. These results surpass traditional rule-based systems and match or exceed existing neural models. Qualitative analysis shows effective handling of ambiguous terms and misspellings. Our findings highlight that lightweight RNN-based architectures can deliver high-quality clinical concept annotation with significantly lower computational cost than transformer-based models, making them well-suited for real-world deployment. 

---
# Building and Aligning Comparable Corpora 

**Authors**: Motaz Saad, David Langlois, Kamel Smaili  

**Link**: [PDF](https://arxiv.org/pdf/2508.02555)  

**Abstract**: Comparable corpus is a set of topic aligned documents in multiple languages, which are not necessarily translations of each other. These documents are useful for multilingual natural language processing when there is no parallel text available in some domains or languages. In addition, comparable documents are informative because they can tell what is being said about a topic in different languages. In this paper, we present a method to build comparable corpora from Wikipedia encyclopedia and EURONEWS website in English, French and Arabic languages. We further experiment a method to automatically align comparable documents using cross-lingual similarity measures. We investigate two cross-lingual similarity measures to align comparable documents. The first measure is based on bilingual dictionary, and the second measure is based on Latent Semantic Indexing (LSI). Experiments on several corpora show that the Cross-Lingual LSI (CL-LSI) measure outperforms the dictionary based measure. Finally, we collect English and Arabic news documents from the British Broadcast Corporation (BBC) and from ALJAZEERA (JSC) news website respectively. Then we use the CL-LSI similarity measure to automatically align comparable documents of BBC and JSC. The evaluation of the alignment shows that CL-LSI is not only able to align cross-lingual documents at the topic level, but also it is able to do this at the event level. 

---
# What's in the News? Towards Identification of Bias by Commission, Omission, and Source Selection (COSS) 

**Authors**: Anastasia Zhukova, Terry Ruas, Felix Hamborg, Karsten Donnay, Bela Gipp  

**Link**: [PDF](https://arxiv.org/pdf/2508.02540)  

**Abstract**: In a world overwhelmed with news, determining which information comes from reliable sources or how neutral is the reported information in the news articles poses a challenge to news readers. In this paper, we propose a methodology for automatically identifying bias by commission, omission, and source selection (COSS) as a joint three-fold objective, as opposed to the previous work separately addressing these types of bias. In a pipeline concept, we describe the goals and tasks of its steps toward bias identification and provide an example of a visualization that leverages the extracted features and patterns of text reuse. 

---
# Contextual Graph Transformer: A Small Language Model for Enhanced Engineering Document Information Extraction 

**Authors**: Karan Reddy, Mayukha Pal  

**Link**: [PDF](https://arxiv.org/pdf/2508.02532)  

**Abstract**: Standard transformer-based language models, while powerful for general text, often struggle with the fine-grained syntax and entity relationships in complex technical, engineering documents. To address this, we propose the Contextual Graph Transformer (CGT), a hybrid neural architecture that combines Graph Neural Networks (GNNs) and Transformers for domain-specific question answering. CGT constructs a dynamic graph over input tokens using sequential, skip-gram, and semantic similarity edges, which is processed by GATv2Conv layers for local structure learning. These enriched embeddings are then passed to a Transformer encoder to capture global dependencies. Unlike generic large models, technical domains often require specialized language models with stronger contextualization and structure awareness. CGT offers a parameter-efficient solution for such use cases. Integrated into a Retrieval-Augmented Generation (RAG) pipeline, CGT outperforms baselines like GPT-2 and BERT, achieving 24.7% higher accuracy than GPT-2 with 62.4% fewer parameters. This gain stems from CGTs ability to jointly model structural token interactions and long-range semantic coherence. The model is trained from scratch using a two-phase approach: pretraining on general text followed by fine-tuning on domain-specific manuals. This highlights CGTs adaptability to technical language, enabling better grounding, entity tracking, and retrieval-augmented responses in real-world applications. 

---
# I Have No Mouth, and I Must Rhyme: Uncovering Internal Phonetic Representations in LLaMA 3.2 

**Authors**: Jack Merullo, Arjun Khurana, Oliver McLaughlin  

**Link**: [PDF](https://arxiv.org/pdf/2508.02527)  

**Abstract**: Large language models demonstrate proficiency on phonetic tasks, such as rhyming, without explicit phonetic or auditory grounding. In this work, we investigate how \verb|Llama-3.2-1B-Instruct| represents token-level phonetic information. Our results suggest that Llama uses a rich internal model of phonemes to complete phonetic tasks. We provide evidence for high-level organization of phoneme representations in its latent space. In doing so, we also identify a ``phoneme mover head" which promotes phonetic information during rhyming tasks. We visualize the output space of this head and find that, while notable differences exist, Llama learns a model of vowels similar to the standard IPA vowel chart for humans, despite receiving no direct supervision to do so. 

---
# PoeTone: A Framework for Constrained Generation of Structured Chinese Songci with LLMs 

**Authors**: Zhan Qu, Shuzhou Yuan, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2508.02515)  

**Abstract**: This paper presents a systematic investigation into the constrained generation capabilities of large language models (LLMs) in producing Songci, a classical Chinese poetry form characterized by strict structural, tonal, and rhyme constraints defined by Cipai templates. We first develop a comprehensive, multi-faceted evaluation framework that includes: (i) a formal conformity score, (ii) automated quality assessment using LLMs, (iii) human evaluation, and (iv) classification-based probing tasks. Using this framework, we evaluate the generative performance of 18 LLMs, including 3 proprietary models and 15 open-source models across four families, under five prompting strategies: zero-shot, one-shot, completion-based, instruction-tuned, and chain-of-thought. Finally, we propose a Generate-Critic architecture in which the evaluation framework functions as an automated critic. Leveraging the critic's feedback as a reward signal, we fine-tune three lightweight open-source LLMs via supervised fine-tuning (SFT), resulting in improvements of up to 5.88% in formal conformity. Our findings offer new insights into the generative strengths and limitations of LLMs in producing culturally significant and formally constrained literary texts. 

---
# Modular Arithmetic: Language Models Solve Math Digit by Digit 

**Authors**: Tanja Baeumel, Daniil Gurgurov, Yusser al Ghussin, Josef van Genabith, Simon Ostermann  

**Link**: [PDF](https://arxiv.org/pdf/2508.02513)  

**Abstract**: While recent work has begun to uncover the internal strategies that Large Language Models (LLMs) employ for simple arithmetic tasks, a unified understanding of their underlying mechanisms is still lacking. We extend recent findings showing that LLMs represent numbers in a digit-wise manner and present evidence for the existence of digit-position-specific circuits that LLMs use to perform simple arithmetic tasks, i.e. modular subgroups of MLP neurons that operate independently on different digit positions (units, tens, hundreds). Notably, such circuits exist independently of model size and of tokenization strategy, i.e. both for models that encode longer numbers digit-by-digit and as one token. Using Feature Importance and Causal Interventions, we identify and validate the digit-position-specific circuits, revealing a compositional and interpretable structure underlying the solving of arithmetic problems in LLMs. Our interventions selectively alter the model's prediction at targeted digit positions, demonstrating the causal role of digit-position circuits in solving arithmetic tasks. 

---
# From Monolingual to Bilingual: Investigating Language Conditioning in Large Language Models for Psycholinguistic Tasks 

**Authors**: Shuzhou Yuan, Zhan Qu, Mario Tawfelis, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2508.02502)  

**Abstract**: Large Language Models (LLMs) exhibit strong linguistic capabilities, but little is known about how they encode psycholinguistic knowledge across languages. We investigate whether and how LLMs exhibit human-like psycholinguistic responses under different linguistic identities using two tasks: sound symbolism and word valence. We evaluate two models, Llama-3.3-70B-Instruct and Qwen2.5-72B-Instruct, under monolingual and bilingual prompting in English, Dutch, and Chinese. Behaviorally, both models adjust their outputs based on prompted language identity, with Qwen showing greater sensitivity and sharper distinctions between Dutch and Chinese. Probing analysis reveals that psycholinguistic signals become more decodable in deeper layers, with Chinese prompts yielding stronger and more stable valence representations than Dutch. Our results demonstrate that language identity conditions both output behavior and internal representations in LLMs, providing new insights into their application as models of cross-linguistic cognition. 

---
# Monsoon Uprising in Bangladesh: How Facebook Shaped Collective Identity 

**Authors**: Md Tasin Abir, Arpita Chowdhury, Ashfia Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2508.02498)  

**Abstract**: This study investigates how Facebook shaped collective identity during the July 2024 pro-democracy uprising in Bangladesh, known as the Monsoon Uprising. During government repression, protesters turned to Facebook as a central space for resistance, where multimodal expressions, images, memes, videos, hashtags, and satirical posts played an important role in unifying participants. Using a qualitative approach, this research analyzes visual rhetoric, verbal discourse, and digital irony to reveal how shared symbols, protest art, and slogans built a sense of solidarity. Key elements included the symbolic use of red, the ironic metaphorical use of the term "Razakar", and the widespread sharing of visuals representing courage, injustice, and resistance. The findings show that the combination of visual and verbal strategies on Facebook not only mobilized public sentiment, but also built a strong collective identity that challenged authoritarian narratives. This study tries to demonstrate how online platforms can serve as powerful tools for identity construction and political mobilization in the digital age. 

---
# LatentPrompt: Optimizing Promts in Latent Space 

**Authors**: Mateusz Bystroński, Grzegorz Piotrowski, Nitesh V. Chawla, Tomasz Kajdanowicz  

**Link**: [PDF](https://arxiv.org/pdf/2508.02452)  

**Abstract**: Recent advances have shown that optimizing prompts for Large Language Models (LLMs) can significantly improve task performance, yet many optimization techniques rely on heuristics or manual exploration. We present LatentPrompt, a model-agnostic framework for prompt optimization that leverages latent semantic space to automatically generate, evaluate, and refine candidate prompts without requiring hand-crafted rules. Beginning with a set of seed prompts, our method embeds them in a continuous latent space and systematically explores this space to identify prompts that maximize task-specific performance. In a proof-of-concept study on the Financial PhraseBank sentiment classification benchmark, LatentPrompt increased classification accuracy by approximately 3 percent after a single optimization cycle. The framework is broadly applicable, requiring only black-box access to an LLM and an automatic evaluation metric, making it suitable for diverse domains and tasks. 

---
# AI-Based Measurement of Innovation: Mapping Expert Insight into Large Language Model Applications 

**Authors**: Robin Nowak, Patrick Figge, Carolin Haeussler  

**Link**: [PDF](https://arxiv.org/pdf/2508.02430)  

**Abstract**: Measuring innovation often relies on context-specific proxies and on expert evaluation. Hence, empirical innovation research is often limited to settings where such data is available. We investigate how large language models (LLMs) can be leveraged to overcome the constraints of manual expert evaluations and assist researchers in measuring innovation. We design an LLM framework that reliably approximates domain experts' assessment of innovation from unstructured text data. We demonstrate the performance and broad applicability of this framework through two studies in different contexts: (1) the innovativeness of software application updates and (2) the originality of user-generated feedback and improvement ideas in product reviews. We compared the performance (F1-score) and reliability (consistency rate) of our LLM framework against alternative measures used in prior innovation studies, and to state-of-the-art machine learning- and deep learning-based models. The LLM framework achieved higher F1-scores than the other approaches, and its results are highly consistent (i.e., results do not change across runs). This article equips R&D personnel in firms, as well as researchers, reviewers, and editors, with the knowledge and tools to effectively use LLMs for measuring innovation and evaluating the performance of LLM-based innovation measures. In doing so, we discuss, the impact of important design decisions-including model selection, prompt engineering, training data size, training data distribution, and parameter settings-on performance and reliability. Given the challenges inherent in using human expert evaluation and existing text-based measures, our framework has important implications for harnessing LLMs as reliable, increasingly accessible, and broadly applicable research tools for measuring innovation. 

---
# Learning to Evolve: Bayesian-Guided Continual Knowledge Graph Embedding 

**Authors**: Linyu Li, Zhi Jin, Yuanpeng He, Dongming Jin, Yichi Zhang, Haoran Duan, Nyima Tash  

**Link**: [PDF](https://arxiv.org/pdf/2508.02426)  

**Abstract**: Since knowledge graphs (KG) will continue to evolve in real scenarios, traditional KGE models are only suitable for static knowledge graphs. Therefore, continual knowledge graph embedding (CKGE) has attracted the attention of researchers. Currently, a key challenge facing CKGE is that the model is prone to "catastrophic forgetting", resulting in the loss of previously learned knowledge. In order to effectively alleviate this problem, we propose a new CKGE model BAKE. First, we note that the Bayesian posterior update principle provides a natural continual learning strategy that is insensitive to data order and can theoretically effectively resist the forgetting of previous knowledge during data evolution. Different from the existing CKGE method, BAKE regards each batch of new data as a Bayesian update of the model prior. Under this framework, as long as the posterior distribution of the model is maintained, the model can better preserve the knowledge of early snapshots even after evolving through multiple time snapshots. Secondly, we propose a continual clustering method for CKGE, which further directly combats knowledge forgetting by constraining the evolution difference (or change amplitude) between new and old knowledge between different snapshots. We conduct extensive experiments on BAKE on multiple datasets, and the results show that BAKE significantly outperforms existing baseline models. 

---
# CompressKV: Semantic Retrieval Heads Know What Tokens are Not Important Before Generation 

**Authors**: Xiaolin Lin, Jingcun Wang, Olga Kondrateva, Yiyu Shi, Bing Li, Grace Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02401)  

**Abstract**: Recent advances in large language models (LLMs) have significantly boosted long-context processing. However, the increasing key-value (KV) cache size poses critical challenges to memory and execution efficiency. Most KV cache compression methods rely on heuristic token eviction using all attention heads in Grouped Query Attention (GQA)-based LLMs. This method ignores the different functionalities of attention heads, leading to the eviction of critical tokens and thus degrades the performance of LLMs.
To address the issue above, instead of using all the attention heads in GQA-based LLMs to determine important tokens as in the previous work, we first identify the attention heads in each layer that are not only capable of retrieving the initial and final tokens of a prompt, but also capable of retrieving important tokens within the text and attending to their surrounding semantic context. Afterwards, we exploit such heads to determine the important tokens and retain their corresponding KV cache pairs. Furthermore, we analyze the cache eviction error of each layer individually and introduce a layer-adaptive KV cache allocation strategy. Experimental results demonstrate the proposed CompressKV consistently outperforms state-of-the-art approaches under various memory budgets on LongBench and Needle-in-a-Haystack benchmarks. Our code is publicly available at: this https URL. 

---
# Understanding and Mitigating Political Stance Cross-topic Generalization in Large Language Models 

**Authors**: Jiayi Zhang, Shu Yang, Junchao Wu, Derek F. Wong, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02360)  

**Abstract**: Fine-tuning Large Language Models on a political topic will significantly manipulate their political stance on various issues and unintentionally affect their stance on unrelated topics. While previous studies have proposed this issue, there is still a lack of understanding regarding the internal representations of these stances and the mechanisms that lead to unintended cross-topic generalization. In this paper, we systematically explore the internal mechanisms underlying this phenomenon from a neuron-level perspective and how to mitigate the cross-topic generalization of political fine-tuning. Firstly, we propose Political Neuron Localization through Activation Contrasting (PNLAC) to identify two distinct types of political neurons: general political neurons, which govern stance across multiple political topics, and topic-specific neurons} that affect the model's political stance on individual topics. We find the existence of these political neuron types across four models and datasets through activation patching experiments. Leveraging these insights, we introduce InhibitFT, an inhibition-based fine-tuning method, effectively mitigating the cross-topic stance generalization. Experimental results demonstrate the robustness of identified neuron types across various models and datasets, and show that InhibitFT significantly reduces the cross-topic stance generalization by 20% on average, while preserving topic-specific performance. Moreover, we demonstrate that selectively inhibiting only 5% of neurons is sufficient to effectively mitigate the cross-topic stance generalization. 

---
# CAMERA: Multi-Matrix Joint Compression for MoE Models via Micro-Expert Redundancy Analysis 

**Authors**: Yuzhuang Xu, Xu Han, Yuanchi Zhang, Yixuan Wang, Yijun Liu, Shiyu Ji, Qingfu Zhu, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2508.02322)  

**Abstract**: Large Language Models (LLMs) with Mixture-of-Experts (MoE) architectures are distinguished by their strong performance scaling with increasing parameters across a wide range of tasks, yet they also suffer from substantial computational and storage overheads. Notably, the performance gains of MoE models do not scale proportionally with the growth in expert parameters. While prior works attempt to reduce parameters via expert-level pruning, merging, or decomposition, they still suffer from challenges in both performance and computational efficiency. In this paper, we address these challenges by introducing micro-expert as a finer-grained compression unit that spans across matrices. We first establish a more fundamental perspective, viewing MoE layers as mixtures of micro-experts, and present CAMERA, a lightweight and training-free framework for identifying micro-expert redundancy. Our analysis uncovers significant variance in micro-expert contributions during decoding. Based on this insight, we further propose CAMERA-P, a structured micro-expert pruning framework, and CAMERA-Q, a mixed-precision quantization idea designed for micro-experts. Extensive experiments on nine downstream tasks show that CAMERA-P consistently outperforms strong baselines under pruning ratios ranging from 20% to 60%. Furthermore, CAMERA-Q achieves superior results under aggressive 2-bit quantization, surpassing existing matrix- and channel-level ideas. Notably, our method enables complete micro-expert analysis of Qwen2-57B-A14B in less than 5 minutes on a single NVIDIA A100-40GB GPU. 

---
# VeOmni: Scaling Any Modality Model Training with Model-Centric Distributed Recipe Zoo 

**Authors**: Qianli Ma, Yaowei Zheng, Zhelun Shi, Zhongkai Zhao, Bin Jia, Ziyue Huang, Zhiqi Lin, Youjie Li, Jiacheng Yang, Yanghua Peng, Zhi Zhang, Xin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02317)  

**Abstract**: Recent advances in large language models (LLMs) have driven impressive progress in omni-modal understanding and generation. However, training omni-modal LLMs remains a significant challenge due to the heterogeneous model architectures required to process diverse modalities, necessitating sophisticated system design for efficient large-scale training. Existing frameworks typically entangle model definition with parallel logic, incurring limited scalability and substantial engineering overhead for end-to-end omni-modal training. %
We present \veomni, a modular and efficient training framework to accelerate the development of omni-modal LLMs. \veomni introduces model-centric distributed recipes that decouples communication from computation, enabling efficient 3D parallelism on omni-modal LLMs. \veomni also features a flexible configuration interface supporting seamless integration of new modalities with minimal code change. %
Using \veomni, a omni-modal mixture-of-experts (MoE) model with 30B parameters can be trained with over 2,800 tokens/sec/GPU throughput and scale to 160K context lengths via 3D parallelism on 128 GPUs, showcasing its superior efficiency and scalability for training large omni-modal LLMs. 

---
# LaMPE: Length-aware Multi-grained Position Encoding for Adaptive Long-context Scaling Without Training 

**Authors**: Sikui Zhang, Guangze Gao, Ziyun Gan, Chunfeng Yuan, Zefeng Lin, Houwen Peng, Bing Li, Weiming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02308)  

**Abstract**: Large language models (LLMs) experience significant performance degradation when the input exceeds the pretraining context window, primarily due to the out-of-distribution (OOD) behavior of Rotary Position Embedding (RoPE). Recent studies mitigate this problem by remapping OOD positions into the in-distribution range with fixed mapping strategies, ignoring the dynamic relationship between input length and the model's effective context window. To this end, we propose Length-aware Multi-grained Positional Encoding (LaMPE), a training-free method that fully utilizes the model's effective context window for adaptive long-context scaling in LLMs. Motivated by the left-skewed frequency distribution of relative positions, LaMPE establishes a dynamic relationship between mapping length and input length through a parametric scaled sigmoid function to adaptively allocate positional capacity across varying input lengths. Meanwhile, LaMPE devises a novel multi-grained attention mechanism that strategically allocates positional resolution across different sequence regions to capture both fine-grained locality and long-range dependencies. Our method can be seamlessly applied to a wide range of RoPE-based LLMs without training. Extensive experiments on three representative LLMs across five mainstream long-context benchmarks demonstrate that LaMPE achieves significant performance improvements compared to existing length extrapolation methods. The code will be released at this https URL. 

---
# Simple Methods Defend RAG Systems Well Against Real-World Attacks 

**Authors**: Ilias Triantafyllopoulos, Renyi Qu, Salvatore Giorgi, Brenda Curtis, Lyle H. Ungar, João Sedoc  

**Link**: [PDF](https://arxiv.org/pdf/2508.02296)  

**Abstract**: Ensuring safety and in-domain responses for Retrieval-Augmented Generation (RAG) systems is paramount in safety-critical applications, yet remains a significant challenge. To address this, we evaluate four methodologies for Out-Of-Domain (OOD) query detection: GPT-4o, regression-based, Principal Component Analysis (PCA)-based, and Neural Collapse (NC), to ensure the RAG system only responds to queries confined to the system's knowledge base. Specifically, our evaluation explores two novel dimensionality reduction and feature separation strategies: \textit{PCA}, where top components are selected using explained variance or OOD separability, and an adaptation of \textit{Neural Collapse Feature Separation}. We validate our approach on standard datasets (StackExchange and MSMARCO) and real-world applications (Substance Use and COVID-19), including tests against LLM-simulated and actual attacks on a COVID-19 vaccine chatbot. Through human and LLM-based evaluations of response correctness and relevance, we confirm that an external OOD detector is crucial for maintaining response relevance. 

---
# A French Version of the OLDI Seed Corpus 

**Authors**: Malik Marmonier, Benoît Sagot, Rachel Bawden  

**Link**: [PDF](https://arxiv.org/pdf/2508.02290)  

**Abstract**: We present the first French partition of the OLDI Seed Corpus, our submission to the WMT 2025 Open Language Data Initiative (OLDI) shared task. We detail its creation process, which involved using multiple machine translation systems and a custom-built interface for post-editing by qualified native speakers. We also highlight the unique translation challenges presented by the source data, which combines highly technical, encyclopedic terminology with the stylistic irregularities characteristic of user-generated content taken from Wikipedia. This French corpus is not an end in itself, but is intended as a crucial pivot resource to facilitate the collection of parallel corpora for the under-resourced regional languages of France. 

---
# Dynaword: From One-shot to Continuously Developed Datasets 

**Authors**: Kenneth Enevoldsen, Kristian Nørgaard Jensen, Jan Kostkan, Balázs Szabó, Márton Kardos, Kirten Vad, Andrea Blasi Núñez, Gianluca Barmina, Jacob Nielsen, Rasmus Larsen, Peter Vahlstrup, Per Møldrup Dalum, Desmond Elliott, Lukas Galke, Peter Schneider-Kamp, Kristoffer Nielbo  

**Link**: [PDF](https://arxiv.org/pdf/2508.02271)  

**Abstract**: Large-scale datasets are foundational for research and development in natural language processing. However, current approaches face three key challenges: (1) reliance on ambiguously licensed sources restricting use, sharing, and derivative works; (2) static dataset releases that prevent community contributions and diminish longevity; and (3) quality assurance processes restricted to publishing teams rather than leveraging community expertise.
To address these limitations, we introduce two contributions: the Dynaword approach and Danish Dynaword. The Dynaword approach is a framework for creating large-scale, open datasets that can be continuously updated through community collaboration. Danish Dynaword is a concrete implementation that validates this approach and demonstrates its potential. Danish Dynaword contains over four times as many tokens as comparable releases, is exclusively openly licensed, and has received multiple contributions across industry and research. The repository includes light-weight tests to ensure data formatting, quality, and documentation, establishing a sustainable framework for ongoing community contributions and dataset evolution. 

---
# SHAMI-MT: A Syrian Arabic Dialect to Modern Standard Arabic Bidirectional Machine Translation System 

**Authors**: Serry Sibaee, Omer Nacar, Yasser Al-Habashi, Adel Ammar, Wadii Boulila  

**Link**: [PDF](https://arxiv.org/pdf/2508.02268)  

**Abstract**: The rich linguistic landscape of the Arab world is characterized by a significant gap between Modern Standard Arabic (MSA), the language of formal communication, and the diverse regional dialects used in everyday life. This diglossia presents a formidable challenge for natural language processing, particularly machine translation. This paper introduces \textbf{SHAMI-MT}, a bidirectional machine translation system specifically engineered to bridge the communication gap between MSA and the Syrian dialect. We present two specialized models, one for MSA-to-Shami and another for Shami-to-MSA translation, both built upon the state-of-the-art AraT5v2-base-1024 architecture. The models were fine-tuned on the comprehensive Nabra dataset and rigorously evaluated on unseen data from the MADAR corpus. Our MSA-to-Shami model achieved an outstanding average quality score of \textbf{4.01 out of 5.0} when judged by OPENAI model GPT-4.1, demonstrating its ability to produce translations that are not only accurate but also dialectally authentic. This work provides a crucial, high-fidelity tool for a previously underserved language pair, advancing the field of dialectal Arabic translation and offering significant applications in content localization, cultural heritage, and intercultural communication. 

---
# Decomposing the Entropy-Performance Exchange: The Missing Keys to Unlocking Effective Reinforcement Learning 

**Authors**: Jia Deng, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2508.02260)  

**Abstract**: Recently, reinforcement learning with verifiable rewards (RLVR) has been widely used for enhancing the reasoning abilities of large language models (LLMs). A core challenge in RLVR involves managing the exchange between entropy and performance of policies. Despite the importance of this exchange, a fine-grained understanding of when and how this exchange operates most effectively remains limited. To bridge this gap, we conduct a systematic empirical analysis of the entropy-performance exchange mechanism of RLVR across different levels of granularity. Specifically, we first divide the training process into two distinct stages based on entropy dynamics, i.e., rising stage and plateau stage, and then systematically investigate how this mechanism varies across stage-level, instance-level, and token-level granularitiess. Our analysis reveals that, in the rising stage, entropy reduction in negative samples facilitates the learning of effective reasoning patterns, which in turn drives rapid performance gains. Moreover, in the plateau stage, learning efficiency strongly correlates with high-entropy tokens present in low-perplexity samples and those located at the end of sequences. Motivated by these findings, we propose two methods that dynamically adjust the reward signal using perplexity and positional information to focus RL updates on tokens that exhibit high learning potential, achieving improvements compared to the baseline methods on various LLMs. 

---
# Interference Matrix: Quantifying Cross-Lingual Interference in Transformer Encoders 

**Authors**: Belen Alastruey, João Maria Janeiro, Alexandre Allauzen, Maha Elbayad, Loïc Barrault, Marta R. Costa-jussà  

**Link**: [PDF](https://arxiv.org/pdf/2508.02256)  

**Abstract**: In this paper, we present a comprehensive study of language interference in encoder-only Transformer models across 83 languages. We construct an interference matrix by training and evaluating small BERT-like models on all possible language pairs, providing a large-scale quantification of cross-lingual interference. Our analysis reveals that interference between languages is asymmetrical and that its patterns do not align with traditional linguistic characteristics, such as language family, nor with proxies like embedding similarity, but instead better relate to script. Finally, we demonstrate that the interference matrix effectively predicts performance on downstream tasks, serving as a tool to better design multilingual models to obtain optimal performance. 

---
# Isolating Culture Neurons in Multilingual Large Language Models 

**Authors**: Danial Namazifard, Lukas Galke  

**Link**: [PDF](https://arxiv.org/pdf/2508.02241)  

**Abstract**: Language and culture are deeply intertwined, yet it is so far unclear how and where multilingual large language models encode culture. Here, we extend upon an established methodology for identifying language-specific neurons and extend it to localize and isolate culture-specific neurons, carefully disentangling their overlap and interaction with language-specific neurons. To facilitate our experiments, we introduce MUREL, a curated dataset of 85.2 million tokens spanning six different cultures. Our localization and intervention experiments show that LLMs encode different cultures in distinct neuron populations, predominantly in upper layers, and that these culture neurons can be modulated independently from language-specific neurons or those specific to other cultures. These findings suggest that cultural knowledge and propensities in multilingual language models can be selectively isolated and edited - promoting fairness, inclusivity, and alignment. Code and data is available at this https URL . 

---
# Proof2Hybrid: Automatic Mathematical Benchmark Synthesis for Proof-Centric Problems 

**Authors**: Yebo Peng, Zixiang Liu, Yaoming Li, Zhizhuo Yang, Xinye Xu, Bowen Ye, Weijun Yuan, Zihan Wang, Tong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02208)  

**Abstract**: Evaluating the mathematical capability of Large Language Models (LLMs) is a critical yet challenging frontier. Existing benchmarks fall short, particularly for proof-centric problems, as manual creation is unscalable and costly, leaving the true mathematical abilities of LLMs largely unassessed. To overcome these barriers, we propose Proof2Hybrid, the first fully automated framework that synthesizes high-quality, proof-centric benchmarks from natural language mathematical corpora. The key novelty of our solution is Proof2X, a roadmap of converting mathematical proofs into various kinds of questions that are easy to verify. Instructed by this roadmap, we propose a new type of hybrid-formatted questions, named ``$m$-out-of-$n$ multiple judge questions'', specifically designed to enable robust, automatic evaluation while being resilient to guessing and superficial pattern matching inherent in traditional formats. As a demonstration of our framework, we introduce AlgGeoTest, a benchmark for algebraic geometry--a frontier domain of modern mathematics--comprising 456 challenging items. Our extensive evaluations on state-of-the-art LLMs using AlgGeoTest reveal profound deficits in their comprehension of algebraic geometry, providing a more precise measure of their true mathematical capabilities. Our framework and benchmark pave the way for a new wave of in-depth research into the mathematical intelligence of AI systems. 

---
# Seed Diffusion: A Large-Scale Diffusion Language Model with High-Speed Inference 

**Authors**: Yuxuan Song, Zheng Zhang, Cheng Luo, Pengyang Gao, Fan Xia, Hao Luo, Zheng Li, Yuehang Yang, Hongli Yu, Xingwei Qu, Yuwei Fu, Jing Su, Ge Zhang, Wenhao Huang, Mingxuan Wang, Lin Yan, Xiaoying Jia, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Yonghui Wu, Hao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.02193)  

**Abstract**: We present Seed Diffusion Preview, a large-scale language model based on discrete-state diffusion, offering remarkably fast inference speed. Thanks to non-sequential, parallel generation, discrete diffusion models provide a notable speedup to mitigate the inherent latency of token-by-token decoding, as demonstrated recently (e.g., Mercury Coder, Gemini Diffusion). Seed Diffusion Preview achieves an inference speed of 2,146 token/s over H20 GPUs while maintaining competitive performance across a sweep of standard code evaluation benchmarks, significantly faster than contemporary Mercury and Gemini Diffusion, establishing new state of the art on the speed-quality Pareto frontier for code models. 

---
# Learning Dynamics of Meta-Learning in Small Model Pretraining 

**Authors**: David Demitri Africa, Yuval Weiss, Paula Buttery, Richard Diehl Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2508.02189)  

**Abstract**: Large language models are powerful but costly. We ask whether meta-learning can make the pretraining of small language models not only better but also more interpretable. We integrate first-order MAML with subset-masked LM pretraining, producing four LLama-style decoder-only models (11M-570M params), and evaluate it on a fundamental NLP task with many settings and real-world applications. Compared with vanilla training, our model (i) reaches the same loss up to 1.6x sooner, (ii) improves F1 on multilingual Universal NER under equal compute, and (iii) makes the training dynamics easy to read: first the network's representations fan out ("diversify") and later they collapse into a smaller, shared subspace ("compress"). This two-stage shift shows up as a rise-and-fall in both effective-rank curves and attention-head entropy. The same curves pinpoint which layers specialise earliest and which later reconverge, giving a compact, interpretable signature of meta-adaptation. Code, checkpoints and WandB logs are released. 

---
# "Harmless to You, Hurtful to Me!": Investigating the Detection of Toxic Languages Grounded in the Perspective of Youth 

**Authors**: Yaqiong Li, Peng Zhang, Lin Wang, Hansu Gu, Siyuan Qiao, Ning Gu, Tun Lu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02094)  

**Abstract**: Risk perception is subjective, and youth's understanding of toxic content differs from that of adults. Although previous research has conducted extensive studies on toxicity detection in social media, the investigation of youth's unique toxicity, i.e., languages perceived as nontoxic by adults but toxic as youth, is ignored. To address this gap, we aim to explore: 1) What are the features of ``youth-toxicity'' languages in social media (RQ1); 2) Can existing toxicity detection techniques accurately detect these languages (RQ2). For these questions, we took Chinese youth as the research target, constructed the first Chinese ``youth-toxicity'' dataset, and then conducted extensive analysis. Our results suggest that youth's perception of these is associated with several contextual factors, like the source of an utterance and text-related features. Incorporating these meta information into current toxicity detection methods significantly improves accuracy overall. Finally, we propose several insights into future research on youth-centered toxicity detection. 

---
# When Truth Is Overridden: Uncovering the Internal Origins of Sycophancy in Large Language Models 

**Authors**: Jin Li, Keyu Wang, Shu Yang, Zhuoran Zhang, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02087)  

**Abstract**: Large Language Models (LLMs) often exhibit sycophantic behavior, agreeing with user-stated opinions even when those contradict factual knowledge. While prior work has documented this tendency, the internal mechanisms that enable such behavior remain poorly understood. In this paper, we provide a mechanistic account of how sycophancy arises within LLMs. We first systematically study how user opinions induce sycophancy across different model families. We find that simple opinion statements reliably induce sycophancy, whereas user expertise framing has a negligible impact. Through logit-lens analysis and causal activation patching, we identify a two-stage emergence of sycophancy: (1) a late-layer output preference shift and (2) deeper representational divergence. We also verify that user authority fails to influence behavior because models do not encode it internally. In addition, we examine how grammatical perspective affects sycophantic behavior, finding that first-person prompts (``I believe...'') consistently induce higher sycophancy rates than third-person framings (``They believe...'') by creating stronger representational perturbations in deeper layers. These findings highlight that sycophancy is not a surface-level artifact but emerges from a structural override of learned knowledge in deeper layers, with implications for alignment and truthful AI systems. 

---
# The SMeL Test: A simple benchmark for media literacy in language models 

**Authors**: Gustaf Ahdritz, Anat Kleiman  

**Link**: [PDF](https://arxiv.org/pdf/2508.02074)  

**Abstract**: The internet is rife with unattributed, deliberately misleading, or otherwise untrustworthy content. Though large language models (LLMs) are often tasked with autonomous web browsing, the extent to which they have learned the simple heuristics human researchers use to navigate this noisy environment is not currently known. In this paper, we introduce the Synthetic Media Literacy Test (SMeL Test), a minimal benchmark that tests the ability of language models to actively filter out untrustworthy information in context. We benchmark a variety of commonly used instruction-tuned LLMs, including reasoning models, and find that no model consistently trusts more reliable sources; while reasoning in particular is associated with higher scores, even the best API model we test hallucinates up to 70% of the time. Remarkably, larger and more capable models do not necessarily outperform their smaller counterparts. We hope our work sheds more light on this important form of hallucination and guides the development of new methods to combat it. 

---
# ProCut: LLM Prompt Compression via Attribution Estimation 

**Authors**: Zhentao Xu, Fengyi Li, Albert Chen, Xiaofeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02053)  

**Abstract**: In large-scale industrial LLM systems, prompt templates often expand to thousands of tokens as teams iteratively incorporate sections such as task instructions, few-shot examples, and heuristic rules to enhance robustness and coverage. This expansion leads to bloated prompts that are difficult to maintain and incur significant inference latency and serving costs. To address this, we introduce Prompt Compression via Attribution Estimation (ProCut), a flexible, LLM-agnostic, training-free framework that compresses prompts through attribution analysis. ProCut segments prompt templates into semantically meaningful units, quantifies their impact on task performance, and prunes low-utility components. Through extensive experiments on five public benchmark datasets and real-world industrial prompts, we show that ProCut achieves substantial prompt size reductions (78% fewer tokens in production) while maintaining or even slightly improving task performance (up to 62% better than alternative methods). We further introduce an LLM-driven attribution estimator that reduces compression latency by over 50%, and demonstrate that ProCut integrates seamlessly with existing prompt-optimization frameworks to produce concise, high-performing prompts. 

---
# Harnessing Temporal Databases for Systematic Evaluation of Factual Time-Sensitive Question-Answering in Large Language Models 

**Authors**: Soyeon Kim, Jindong Wang, Xing Xie, Steven Euijong Whang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02045)  

**Abstract**: Facts evolve over time, making it essential for Large Language Models (LLMs) to handle time-sensitive factual knowledge accurately and reliably. While factual Time-Sensitive Question-Answering (TSQA) tasks have been widely studied, existing benchmarks often rely on manual curation or a small, fixed set of predefined templates, which restricts scalable and comprehensive TSQA evaluation. To address these challenges, we propose TDBench, a new benchmark that systematically constructs TSQA pairs by harnessing temporal databases and database techniques such as temporal SQL and functional dependencies. We also introduce a fine-grained evaluation metric called time accuracy, which assesses the validity of time references in model explanations alongside traditional answer accuracy to enable a more reliable TSQA evaluation. Extensive experiments on contemporary LLMs show how \ours{} enables scalable and comprehensive TSQA evaluation while reducing the reliance on human labor, complementing existing Wikipedia/Wikidata-based TSQA evaluation approaches by enabling LLM evaluation on application-specific data and seamless multi-hop question generation. Code and data are publicly available at: this https URL. 

---
# Marco-Voice Technical Report 

**Authors**: Fengping Tian, Chenyang Lyu, Xuanfan Ni, Haoqin Sun, Qingjuan Li, Zhiqiang Qian, Haijun Li, Longyue Wang, Zhao Xu, Weihua Luo, Kaifu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02038)  

**Abstract**: This paper presents a multifunctional speech synthesis system that integrates voice cloning and emotion control speech synthesis within a unified framework. The goal of this work is to address longstanding challenges in achieving highly expressive, controllable, and natural speech generation that faithfully preserves speaker identity across diverse linguistic and emotional contexts. Our approach introduces an effective speaker-emotion disentanglement mechanism with in-batch contrastive learning, enabling independent manipulation of speaker identity and eemotional style, as well as rotational emotional embedding integration method for smooth emotion control. To support comprehensive training and evaluation, we construct CSEMOTIONS, a high-quality emotional speech dataset containing 10 hours of Mandarin speech from six professional speakers across seven emotional categories. Extensive experiments demonstrate that our system, Marco-Voice, achieves substantial improvements in both objective and subjective metrics. Comprehensive evaluations and analysis were conducted, results show that MarcoVoice delivers competitive performance in terms of speech clarity and emotional richness, representing a substantial advance in the field of expressive neural speech synthesis. 

---
# Diagnosing Memorization in Chain-of-Thought Reasoning, One Token at a Time 

**Authors**: Huihan Li, You Chen, Siyuan Wang, Yixin He, Ninareh Mehrabi, Rahul Gupta, Xiang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2508.02037)  

**Abstract**: Large Language Models (LLMs) perform well on reasoning benchmarks but often fail when inputs alter slightly, raising concerns about the extent to which their success relies on memorization. This issue is especially acute in Chain-of-Thought (CoT) reasoning, where spurious memorized patterns can trigger intermediate errors that cascade into incorrect final answers. We introduce STIM, a novel framework for Source-aware Token-level Identification of Memorization, which attributes each token in a reasoning chain to one of multiple memorization sources - local, mid-range, or long-range - based on their statistical co-occurrence with the token in the pretraining corpus. Our token-level analysis across tasks and distributional settings reveals that models rely more on memorization in complex or long-tail cases, and that local memorization is often the dominant driver of errors, leading to up to 67% of wrong tokens. We also show that memorization scores from STIM can be effective in predicting the wrong tokens in the wrong reasoning step. STIM offers a powerful tool for diagnosing and improving model reasoning and can generalize to other structured step-wise generation tasks. 

---
# SpeechR: A Benchmark for Speech Reasoning in Large Audio-Language Models 

**Authors**: Wanqi Yang, Yanda Li, Yunchao Wei, Meng Fang, Ling Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.02018)  

**Abstract**: Large audio-language models (LALMs) have achieved near-human performance in sentence-level transcription and emotion recognition. However, existing evaluations focus mainly on surface-level perception, leaving the capacity of models for contextual and inference-driven reasoning in speech-based scenarios insufficiently examined. To address this gap, we introduce SpeechR, a unified benchmark for evaluating reasoning over speech in large audio-language models. SpeechR evaluates models along three key dimensions: factual retrieval, procedural inference, and normative judgment. It includes three distinct evaluation formats. The multiple-choice version measures answer selection accuracy. The generative version assesses the coherence and logical consistency of reasoning chains. The acoustic-feature version investigates whether variations in stress and emotion affect reasoning performance. Evaluations on eleven state-of-the-art LALMs reveal that high transcription accuracy does not translate into strong reasoning capabilities. SpeechR establishes a structured benchmark for evaluating reasoning in spoken language, enabling more targeted analysis of model capabilities across diverse dialogue-based tasks. 

---
# SpeechRole: A Large-Scale Dataset and Benchmark for Evaluating Speech Role-Playing Agents 

**Authors**: Changhao Jiang, Jiajun Sun, Yifei Cao, Jiabao Zhuang, Hui Li, Xiaoran Fan, Ming Zhang, Junjie Ye, Shihan Dou, Zhiheng Xi, Jingqi Tong, Yilong Wu, Baoyu Fan, Zhen Wang, Tao Liang, Zhihui Fei, Mingyang Wan, Guojun Ma, Tao Ji, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02013)  

**Abstract**: Recently, role-playing agents have emerged as a promising paradigm for achieving personalized interaction and emotional resonance. Existing research primarily focuses on the textual modality, neglecting the critical dimension of speech in realistic interactive scenarios. In particular, there is a lack of systematic evaluation for Speech Role-Playing Agents (SRPAs). To address this gap, we construct SpeechRole-Data, a large-scale, high-quality dataset that comprises 98 diverse roles and 112k speech-based single-turn and multi-turn conversations. Each role demonstrates distinct vocal characteristics, including timbre and prosody, thereby enabling more sophisticated speech role-playing. Furthermore, we propose SpeechRole-Eval, a multidimensional evaluation benchmark that systematically assesses SRPAs performance in key aspects such as fundamental interaction ability, speech expressiveness, and role-playing fidelity. Experimental results reveal the advantages and challenges of both cascaded and end-to-end speech role-playing agents in maintaining vocal style consistency and role coherence. We release all data, code, and baseline models to provide a solid foundation for speech-driven multimodal role-playing research and to foster further developments in this field. 

---
# Prompting Large Language Models to Detect Dementia Family Caregivers 

**Authors**: Md Badsha Biswas, Özlem Uzuner  

**Link**: [PDF](https://arxiv.org/pdf/2508.01999)  

**Abstract**: Social media, such as Twitter, provides opportunities for caregivers of dementia patients to share their experiences and seek support for a variety of reasons. Availability of this information online also paves the way for the development of internet-based interventions in their support. However, for this purpose, tweets written by caregivers of dementia patients must first be identified. This paper demonstrates our system for the SMM4H 2025 shared task 3, which focuses on detecting tweets posted by individuals who have a family member with dementia. The task is outlined as a binary classification problem, differentiating between tweets that mention dementia in the context of a family member and those that do not. Our solution to this problem explores large language models (LLMs) with various prompting methods. Our results show that a simple zero-shot prompt on a fine-tuned model yielded the best results. Our final system achieved a macro F1-score of 0.95 on the validation set and the test set. Our full code is available on GitHub. 

---
# Contextually Aware E-Commerce Product Question Answering using RAG 

**Authors**: Praveen Tangarajan, Anand A. Rajasekar, Manish Rathi, Vinay Rao Dandin, Ozan Ersoy  

**Link**: [PDF](https://arxiv.org/pdf/2508.01990)  

**Abstract**: E-commerce product pages contain a mix of structured specifications, unstructured reviews, and contextual elements like personalized offers or regional variants. Although informative, this volume can lead to cognitive overload, making it difficult for users to quickly and accurately find the information they need. Existing Product Question Answering (PQA) systems often fail to utilize rich user context and diverse product information effectively. We propose a scalable, end-to-end framework for e-commerce PQA using Retrieval Augmented Generation (RAG) that deeply integrates contextual understanding. Our system leverages conversational history, user profiles, and product attributes to deliver relevant and personalized answers. It adeptly handles objective, subjective, and multi-intent queries across heterogeneous sources, while also identifying information gaps in the catalog to support ongoing content improvement. We also introduce novel metrics to measure the framework's performance which are broadly applicable for RAG system evaluations. 

---
# TIBSTC-CoT: A Multi-Domain Instruction Dataset for Chain-of-Thought Reasoning in Language Models 

**Authors**: Fan Gao, Cheng Huang, Nyima Tashi, Yutong Liu, Xiangxiang Wang, Thupten Tsering, Ban Ma-bao, Renzeg Duojie, Gadeng Luosang, Rinchen Dongrub, Dorje Tashi, Xiao Feng, Hao Wang, Yongbin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01977)  

**Abstract**: To address the severe data scarcity in Tibetan, a low-resource language spoken by over six million people, we introduce TIBSTC-CoT, the large-scale, multi-domain Tibetan dataset automatically constructed via chain-of-thought prompting with large language models (LLMs). TIBSTC-CoT establishes a scalable and reproducible framework for dataset creation in low-resource settings, covering diverse domains and reasoning patterns essential for language understanding and generation. Building on this dataset, we develop the Sunshine-thinking LLM family, a series of Tibetan-centric LLMs equipped with chain-of-thought capabilities. Trained entirely on TIBSTC-CoT, Sunshine-thinking has demonstrated strong reasoning and generation performance, comparable to state-of-the-art (SOTA) multilingual LLMs. Our work marks a significant step toward inclusive AI by enabling high-quality Tibetan language processing through both resource creation and model innovation. All data are available: this https URL. 

---
# SitEmb-v1.5: Improved Context-Aware Dense Retrieval for Semantic Association and Long Story Comprehension 

**Authors**: Junjie Wu, Jiangnan Li, Yuqing Li, Lemao Liu, Liyan Xu, Jiwei Li, Dit-Yan Yeung, Jie Zhou, Mo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01959)  

**Abstract**: Retrieval-augmented generation (RAG) over long documents typically involves splitting the text into smaller chunks, which serve as the basic units for retrieval. However, due to dependencies across the original document, contextual information is often essential for accurately interpreting each chunk. To address this, prior work has explored encoding longer context windows to produce embeddings for longer chunks. Despite these efforts, gains in retrieval and downstream tasks remain limited. This is because (1) longer chunks strain the capacity of embedding models due to the increased amount of information they must encode, and (2) many real-world applications still require returning localized evidence due to constraints on model or human bandwidth.
We propose an alternative approach to this challenge by representing short chunks in a way that is conditioned on a broader context window to enhance retrieval performance -- i.e., situating a chunk's meaning within its context. We further show that existing embedding models are not well-equipped to encode such situated context effectively, and thus introduce a new training paradigm and develop the situated embedding models (SitEmb). To evaluate our method, we curate a book-plot retrieval dataset specifically designed to assess situated retrieval capabilities. On this benchmark, our SitEmb-v1 model based on BGE-M3 substantially outperforms state-of-the-art embedding models, including several with up to 7-8B parameters, with only 1B parameters. Our 8B SitEmb-v1.5 model further improves performance by over 10% and shows strong results across different languages and several downstream applications. 

---
# ROVER: Recursive Reasoning Over Videos with Vision-Language Models for Embodied Tasks 

**Authors**: Philip Schroeder, Ondrej Biza, Thomas Weng, Hongyin Luo, James Glass  

**Link**: [PDF](https://arxiv.org/pdf/2508.01943)  

**Abstract**: Vision-language models (VLMs) have exhibited impressive capabilities across diverse image understanding tasks, but still struggle in settings that require reasoning over extended sequences of camera frames from a video. This limits their utility in embodied settings, which require reasoning over long frame sequences from a continuous stream of visual input at each moment of a task attempt. To address this limitation, we propose ROVER (Reasoning Over VidEo Recursively), a framework that enables the model to recursively decompose long-horizon video trajectories into segments corresponding to shorter subtasks within the trajectory. In doing so, ROVER facilitates more focused and accurate reasoning over temporally localized frame sequences without losing global context. We evaluate ROVER, implemented using an in-context learning approach, on diverse OpenX Embodiment videos and on a new dataset derived from RoboCasa that consists of 543 videos showing both expert and perturbed non-expert trajectories across 27 robotic manipulation tasks. ROVER outperforms strong baselines across three video reasoning tasks: task progress estimation, frame-level natural language reasoning, and video question answering. We observe that, by reducing the number of frames the model reasons over at each timestep, ROVER mitigates hallucinations, especially during unexpected or non-optimal moments of a trajectory. In addition, by enabling the implementation of a subtask-specific sliding context window, ROVER's time complexity scales linearly with video length, an asymptotic improvement over baselines. Demos, code, and data available at: this https URL 

---
# Word Overuse and Alignment in Large Language Models: The Influence of Learning from Human Feedback 

**Authors**: Tom S. Juzek, Zina B. Ward  

**Link**: [PDF](https://arxiv.org/pdf/2508.01930)  

**Abstract**: Large Language Models (LLMs) are known to overuse certain terms like "delve" and "intricate." The exact reasons for these lexical choices, however, have been unclear. Using Meta's Llama model, this study investigates the contribution of Learning from Human Feedback (LHF), under which we subsume Reinforcement Learning from Human Feedback and Direct Preference Optimization. We present a straightforward procedure for detecting the lexical preferences of LLMs that are potentially LHF-induced. Next, we more conclusively link LHF to lexical overuse by experimentally emulating the LHF procedure and demonstrating that participants systematically prefer text variants that include certain words. This lexical overuse can be seen as a sort of misalignment, though our study highlights the potential divergence between the lexical expectations of different populations -- namely LHF workers versus LLM users. Our work contributes to the growing body of research on explainable artificial intelligence and emphasizes the importance of both data and procedural transparency in alignment research. 

---
# Quantum-RAG and PunGPT2: Advancing Low-Resource Language Generation and Retrieval for the Punjabi Language 

**Authors**: Jaskaranjeet Singh, Rakesh Thakur  

**Link**: [PDF](https://arxiv.org/pdf/2508.01918)  

**Abstract**: Despite the rapid advancement of large language models (LLMs), low-resource languages remain largely excluded from the NLP landscape. We present PunGPT2, the first fully open-source suite of Punjabi large language models, trained from scratch on a 35GB domain-diverse corpus encompassing literature, religious texts, news, and social discourse. Unlike prior multilingual approaches, PunGPT2 captures rich syntactic and morphological features unique to Punjabi through a tokenizer optimised with byte pair encoding and linguistically aligned pretraining objectives. To improve factual grounding and domain recall, we introduce Pun-RAG, a retrieval-augmented generation framework combining PunGPT2 with a dense FAISS retriever over a curated Punjabi knowledge base. We further develop Pun-Instruct, a parameter-efficient, instruction-tuned variant using QLoRA, enabling robust zero-shot and instruction-following performance with significantly reduced compute needs.
As a key innovation, we propose Quantum-RAG, a novel hybrid retrieval system that fuses sparse (BM25) and dense methods with quantum-inspired semantic matching. By encoding queries using amplitude-based embeddings and retrieving via quantum kernel similarity, Quantum-RAG achieves improved contextual relevance with minimal memory overhead marking the first practical integration of quantum representations in low-resource language generation. Our models significantly outperform strong multilingual baselines (mBERT, mT5, MuRIL) in perplexity, factuality, and fluency. This work provides a scalable, reproducible blueprint for extending LLM capabilities to underrepresented languages and pioneers quantum-aware retrieval in low-resource NLP 

---
# Counterfactual Probing for Hallucination Detection and Mitigation in Large Language Models 

**Authors**: Yijun Feng  

**Link**: [PDF](https://arxiv.org/pdf/2508.01862)  

**Abstract**: Large Language Models have demonstrated remarkable capabilities across diverse tasks, yet they frequently generate hallucinations outputs that are fluent but factually incorrect or unsupported. We propose Counterfactual Probing, a novel approach for detecting and mitigating hallucinations in LLM outputs. Our method dynamically generates counterfactual statements that appear plausible but contain subtle factual errors, then evaluates the model's sensitivity to these perturbations. We hypothesize that genuine knowledge exhibits robustness to counterfactual variations, while hallucinated content shows inconsistent confidence patterns when confronted with plausible alternatives. Our comprehensive evaluation on TruthfulQA, factual statement datasets, and curated hallucination examples demonstrates that counterfactual probing achieves superior detection performance compared to baseline methods, while our adaptive mitigation strategies reduce hallucination scores by an average of 24.5%. The approach requires no model retraining and can be integrated into existing LLM pipelines as a realtime verification mechanism. 

---
# Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents 

**Authors**: Yuhan Guo, Cong Guo, Aiwen Sun, Hongliang He, Xinyu Yang, Yue Lu, Yingji Zhang, Xuntao Guo, Dong Zhang, Jianzhuang Liu, Jiang Duan, Yijia Xiao, Liangjian Wen, Hai-Ming Xu, Yong Dai  

**Link**: [PDF](https://arxiv.org/pdf/2508.01858)  

**Abstract**: Multimodal large-scale models have significantly advanced the development of web agents, enabling perception and interaction with digital environments akin to human cognition. In this paper, we argue that web agents must first acquire sufficient knowledge to effectively engage in cognitive reasoning. Therefore, we decompose a web agent's capabilities into two essential stages: knowledge content learning and cognitive processes. To formalize this, we propose Web-CogKnowledge Framework, categorizing knowledge as Factual, Conceptual, and Procedural. In this framework, knowledge content learning corresponds to the agent's processes of Memorizing and Understanding, which rely on the first two knowledge types, representing the "what" of learning. Conversely, cognitive processes correspond to Exploring, grounded in Procedural knowledge, defining the "how" of reasoning and action. To facilitate knowledge acquisition, we construct the Web-CogDataset, a structured resource curated from 14 real-world websites, designed to systematically instill core knowledge necessary for web agent. This dataset serves as the agent's conceptual grounding-the "nouns" upon which comprehension is built-as well as the basis for learning how to reason and act. Building on this foundation, we operationalize these processes through a novel knowledge-driven Chain-of-Thought (CoT) reasoning framework, developing and training our proposed agent, the Web-CogReasoner. Extensive experimentation reveals its significant superiority over existing models, especially in generalizing to unseen tasks where structured knowledge is decisive. To enable rigorous evaluation, we introduce the Web-CogBench, a comprehensive evaluation suite designed to assess and compare agent performance across the delineated knowledge domains and cognitive capabilities. Our code and data is open sourced at this https URL 

---
# MLP Memory: Language Modeling with Retriever-pretrained External Memory 

**Authors**: Rubin Wei, Jiaqi Cao, Jiarui Wang, Jushi Kai, Qipeng Guo, Bowen Zhou, Zhouhan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.01832)  

**Abstract**: While modern decoder-only LLMs achieve superior performance across various domains, hallucinations have risen to be a common problem in their generated text, hindering their application in knowledge-intensive tasks. Retriever-augmented generation (RAG) offers a solution, but the non-parametric nature of the retriever hinders its deep interaction with LLM. In this work, we propose to decouple memorization from the LLM decoder using a pretrained, differentiable external memory. The external memory is an MLP pretrained by imitating the behavior of a retriever on the entire pretraining dataset. Our resulting architecture, which comprises a transformer decoder and an external MLP memory pretrained on language modeling and retriever imitation respectively, demonstrates strong perplexity and performance on downstream tasks. Experiments show our architecture exhibits steeper power-law scaling with model size, achieving 17.5% and 24.1% improvement on WikiText-103 and Web datasets compared to decoder-only models while benefiting from added training without overfitting. We demonstrate superior performance on three hallucination benchmarks and nine memory-intensive tasks. Additionally, our approach delivers $80\times$ speedup over $k$NN-LM (500M tokens) and $1.3\times$ faster inference than decoder-only models. Unlike $k$NN-LM, which impairs reasoning, our MLP memory improves StrategyQA performance. We will open-source our code and models in the future. 

---
# AGENTICT$^2$S:Robust Text-to-SPARQL via Agentic Collaborative Reasoning over Heterogeneous Knowledge Graphs for the Circular Economy 

**Authors**: Yang Zhao, Chengxiao Dai, Wei Zhuo, Tan Chuan Fu, Yue Xiu, Dusit Niyato, Jonathan Z. Low, Eugene Ho Hong Zhuang, Daren Zong Loong Tan  

**Link**: [PDF](https://arxiv.org/pdf/2508.01815)  

**Abstract**: Question answering over heterogeneous knowledge graphs (KGQA) involves reasoning across diverse schemas, incomplete alignments, and distributed data sources. Existing text-to-SPARQL approaches rely on large-scale domain-specific fine-tuning or operate within single-graph settings, limiting their generalizability in low-resource domains and their ability to handle queries spanning multiple graphs. These challenges are particularly relevant in domains such as the circular economy, where information about classifications, processes, and emissions is distributed across independently curated knowledge graphs (KGs). We present AgenticT$^2$S, a modular framework that decomposes KGQA into subtasks managed by specialized agents responsible for retrieval, query generation, and verification. A scheduler assigns subgoals to different graphs using weak-to-strong alignment strategies. A two-stage verifier detects structurally invalid and semantically underspecified queries through symbolic validation and counterfactual consistency checks. Experiments on real-world circular economy KGs demonstrate that AgenticT$^2$S improves execution accuracy by 17.3% and triple level F$_1$ by 25.4% over the best baseline, while reducing the average prompt length by 46.4%. These results demonstrate the benefits of agent-based schema-aware reasoning for scalable KGQA and support decision-making in sustainability domains through robust cross-graph reasoning. 

---
# HeQ: a Large and Diverse Hebrew Reading Comprehension Benchmark 

**Authors**: Amir DN Cohen, Hilla Merhav, Yoav Goldberg, Reut Tsarfaty  

**Link**: [PDF](https://arxiv.org/pdf/2508.01812)  

**Abstract**: Current benchmarks for Hebrew Natural Language Processing (NLP) focus mainly on morpho-syntactic tasks, neglecting the semantic dimension of language understanding. To bridge this gap, we set out to deliver a Hebrew Machine Reading Comprehension (MRC) dataset, where MRC is to be realized as extractive Question Answering. The morphologically rich nature of Hebrew poses a challenge to this endeavor: the indeterminacy and non-transparency of span boundaries in morphologically complex forms lead to annotation inconsistencies, disagreements, and flaws in standard evaluation metrics.
To remedy this, we devise a novel set of guidelines, a controlled crowdsourcing protocol, and revised evaluation metrics that are suitable for the morphologically rich nature of the language. Our resulting benchmark, HeQ (Hebrew QA), features 30,147 diverse question-answer pairs derived from both Hebrew Wikipedia articles and Israeli tech news. Our empirical investigation reveals that standard evaluation metrics such as F1 scores and Exact Match (EM) are not appropriate for Hebrew (and other MRLs), and we propose a relevant enhancement.
In addition, our experiments show low correlation between models' performance on morpho-syntactic tasks and on MRC, which suggests that models designed for the former might underperform on semantics-heavy tasks. The development and exploration of HeQ illustrate some of the challenges MRLs pose in natural language understanding (NLU), fostering progression towards more and better NLU models for Hebrew and other MRLs. 

---
# A comprehensive taxonomy of hallucinations in Large Language Models 

**Authors**: Manuel Cossio  

**Link**: [PDF](https://arxiv.org/pdf/2508.01781)  

**Abstract**: Large language models (LLMs) have revolutionized natural language processing, yet their propensity for hallucination, generating plausible but factually incorrect or fabricated content, remains a critical challenge. This report provides a comprehensive taxonomy of LLM hallucinations, beginning with a formal definition and a theoretical framework that posits its inherent inevitability in computable LLMs, irrespective of architecture or training. It explores core distinctions, differentiating between intrinsic (contradicting input context) and extrinsic (inconsistent with training data or reality), as well as factuality (absolute correctness) and faithfulness (adherence to input). The report then details specific manifestations, including factual errors, contextual and logical inconsistencies, temporal disorientation, ethical violations, and task-specific hallucinations across domains like code generation and multimodal applications. It analyzes the underlying causes, categorizing them into data-related issues, model-related factors, and prompt-related influences. Furthermore, the report examines cognitive and human factors influencing hallucination perception, surveys evaluation benchmarks and metrics for detection, and outlines architectural and systemic mitigation strategies. Finally, it introduces web-based resources for monitoring LLM releases and performance. This report underscores the complex, multifaceted nature of LLM hallucinations and emphasizes that, given their theoretical inevitability, future efforts must focus on robust detection, mitigation, and continuous human oversight for responsible and reliable deployment in critical applications. 

---
# AI-Generated Text is Non-Stationary: Detection via Temporal Tomography 

**Authors**: Alva West, Yixuan Weng, Minjun Zhu, Luodan Zhang, Zhen Lin, Guangsheng Bao, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01754)  

**Abstract**: The field of AI-generated text detection has evolved from supervised classification to zero-shot statistical analysis. However, current approaches share a fundamental limitation: they aggregate token-level measurements into scalar scores, discarding positional information about where anomalies occur. Our empirical analysis reveals that AI-generated text exhibits significant non-stationarity, statistical properties vary by 73.8\% more between text segments compared to human writing. This discovery explains why existing detectors fail against localized adversarial perturbations that exploit this overlooked characteristic. We introduce Temporal Discrepancy Tomography (TDT), a novel detection paradigm that preserves positional information by reformulating detection as a signal processing task. TDT treats token-level discrepancies as a time-series signal and applies Continuous Wavelet Transform to generate a two-dimensional time-scale representation, capturing both the location and linguistic scale of statistical anomalies. On the RAID benchmark, TDT achieves 0.855 AUROC (7.1\% improvement over the best baseline). More importantly, TDT demonstrates robust performance on adversarial tasks, with 14.1\% AUROC improvement on HART Level 2 paraphrasing attacks. Despite its sophisticated analysis, TDT maintains practical efficiency with only 13\% computational overhead. Our work establishes non-stationarity as a fundamental characteristic of AI-generated text and demonstrates that preserving temporal dynamics is essential for robust detection. 

---
# Enhancing the Preference Extractor in Multi-turn Dialogues: From Annotating Disasters to Accurate Preference Extraction 

**Authors**: Cheng Wang, ziru Liu, Pengcheng Tang, Mingyu Zhang, Quanyu Dai, Yue Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01739)  

**Abstract**: Identifying user preferences in dialogue systems is a pivotal aspect of providing satisfying services. Current research shows that using large language models (LLMs) to fine-tune a task-specific preference extractor yields excellent results in terms of accuracy and generalization. However, the primary challenge stems from the inherent difficulty in obtaining high-quality labeled multi-turn dialogue data. Accurately tracking user preference transitions across turns not only demands intensive domain expertise and contextual consistency maintenance for annotators (termed \textbf{``Annotating Disaster''}) but also complicates model training due to error propagation in sequential dependency learning. Inspired by the observation that multi-turn preference extraction can be decomposed into iterative executions of one-turn extraction processes. We propose a novel dialogue data generation framework named \textbf{IterChat}. First, we construct a new data format that categorizes the dialogue data into attributed historical preferences and one-turn dialogues. This reduces the probability of annotation errors and improves annotation efficiency. Then, to generate a high-quality and diverse dialogue dataset, we adopt GPT4 to pre-define the preference slots in the target preference extractor task and then randomly sample the subset of the slots and their corresponding schema values to create the dialogue datasets. Experimental results indicate that fine-tuning or only few-shot prompting with the new dialogue format yields superior performance compared to the original multi-turn dialogues. Additionally, the new data format improves annotator efficiency with a win rate of 28.4\% higher than the original multi-turn dialogues. 

---
# CultureGuard: Towards Culturally-Aware Dataset and Guard Model for Multilingual Safety Applications 

**Authors**: Raviraj Joshi, Rakesh Paul, Kanishk Singla, Anusha Kamath, Michael Evans, Katherine Luna, Shaona Ghosh, Utkarsh Vaidya, Eileen Long, Sanjay Singh Chauhan, Niranjan Wartikar  

**Link**: [PDF](https://arxiv.org/pdf/2508.01710)  

**Abstract**: The increasing use of Large Language Models (LLMs) in agentic applications highlights the need for robust safety guard models. While content safety in English is well-studied, non-English languages lack similar advancements due to the high cost of collecting culturally aligned labeled datasets. We present CultureGuard, a novel solution for curating culturally aligned, high-quality safety datasets across multiple languages. Our approach introduces a four-stage synthetic data generation and filtering pipeline: cultural data segregation, cultural data adaptation, machine translation, and quality filtering. This pipeline enables the conversion and expansion of the Nemotron-Content-Safety-Dataset-V2 English safety dataset into eight distinct languages: Arabic, German, Spanish, French, Hindi, Japanese, Thai, and Chinese. The resulting dataset, Nemotron-Content-Safety-Dataset-Multilingual-v1, comprises 386,661 samples in 9 languages and facilitates the training of Llama-3.1-Nemotron-Safety-Guard-Multilingual-8B-v1 via LoRA-based fine-tuning. The final model achieves state-of-the-art performance on several multilingual content safety benchmarks. We also benchmark the latest open LLMs on multilingual safety and observe that these LLMs are more prone to give unsafe responses when prompted in non-English languages. This work represents a significant step toward closing the safety gap in multilingual LLMs by enabling the development of culturally aware safety guard models. 

---
# Am I Blue or Is My Hobby Counting Teardrops? Expression Leakage in Large Language Models as a Symptom of Irrelevancy Disruption 

**Authors**: Berkay Köprü, Mehrzad Mashal, Yigit Gurses, Akos Kadar, Maximilian Schmitt, Ditty Mathew, Felix Burkhardt, Florian Eyben, Björn W. Schuller  

**Link**: [PDF](https://arxiv.org/pdf/2508.01708)  

**Abstract**: Large language models (LLMs) have advanced natural language processing (NLP) skills such as through next-token prediction and self-attention, but their ability to integrate broad context also makes them prone to incorporating irrelevant information. Prior work has focused on semantic leakage, bias introduced by semantically irrelevant context. In this paper, we introduce expression leakage, a novel phenomenon where LLMs systematically generate sentimentally charged expressions that are semantically unrelated to the input context. To analyse the expression leakage, we collect a benchmark dataset along with a scheme to automatically generate a dataset from free-form text from common-crawl. In addition, we propose an automatic evaluation pipeline that correlates well with human judgment, which accelerates the benchmarking by decoupling from the need of annotation for each analysed model. Our experiments show that, as the model scales in the parameter space, the expression leakage reduces within the same LLM family. On the other hand, we demonstrate that expression leakage mitigation requires specific care during the model building process, and cannot be mitigated by prompting. In addition, our experiments indicate that, when negative sentiment is injected in the prompt, it disrupts the generation process more than the positive sentiment, causing a higher expression leakage rate. 

---
# Collaborative Chain-of-Agents for Parametric-Retrieved Knowledge Synergy 

**Authors**: Yi Jiang, Sendong Zhao, Jianbo Li, Haochun Wang, Lizhe Zhang, Yan Liu, Bin Qin  

**Link**: [PDF](https://arxiv.org/pdf/2508.01696)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a promising framework for enhancing the capabilities of Large Language Models (LLMs), especially in knowledge-intensive tasks. Despite its advantages, current RAG methods often struggle to *fully exploit knowledge during generation*. In particular, the synergy between the model's internal parametric knowledge and external retrieved knowledge remains limited. Retrieved contents may sometimes mislead generation, while certain generated content can guide the model toward more accurate outputs. In this work, we propose Collaborative Chain-of-Agents, a framework designed to enhance explicitly synergy over both parametric and retrieved knowledge. Specifically, we first introduce CoCoA-zero, a multi-agent RAG framework that first performs conditional knowledge induction and then reasons answers. Building on this, we develop CoCoA, a long-chain training strategy that synthesizes extended multi-agent reasoning trajectories from CoCoA-zero to fine-tune the LLM. This strategy enhances the model's capability to explicitly integrate and jointly leverage parametric and retrieved knowledge. Experiments results show that CoCoA-zero and CoCoA achieve superior performance on open-domain and multi-hop QA tasks. 

---
# The Bidirectional Process Reward Model 

**Authors**: Lingyin Zhang, Jun Gao, Xiaoxue Ren, Ziqiang Cao  

**Link**: [PDF](https://arxiv.org/pdf/2508.01682)  

**Abstract**: Process Reward Models (PRMs) have emerged as a promising approach to enhance the reasoning quality of Large Language Models (LLMs) by assigning fine-grained scores to intermediate reasoning steps within a solution trajectory. However, existing PRMs predominantly adopt a unidirectional left-to-right (L2R) evaluation paradigm, which limits their ability to leverage global context, making it challenging to verify the consistency of earlier steps based on later ones. In light of these challenges, we propose a novel bidirectional evaluation paradigm, named Bidirectional Process Reward Model (BiPRM). BiPRM seamlessly incorporates a parallel right-to-left (R2L) evaluation stream alongside the conventional L2R flow, enabling later reasoning steps to help assess earlier ones in real time. Notably, the built-in R2L evaluation is implemented solely through prompt modifications that reverse the original reasoning trajectory, without any additional parameters or inference latency introduced. This ensures BiPRM remains both efficient and broadly compatible with existing PRM studies. We conduct extensive experiments on two mathematical reasoning benchmarks using samples generated by three different policy models. Our method, BiPRM, is evaluated across three backbones and three distinct PRM objectives. Across all settings, BiPRM consistently outperforms unidirectional baselines, achieving up to a 31.9% improvement in stepwise reward evaluation. Generally, our results highlight BiPRM's effectiveness, robustness, and general applicability, offering a promising new direction for process-based reward modeling. 

---
# CUPID: Evaluating Personalized and Contextualized Alignment of LLMs from Interactions 

**Authors**: Tae Soo Kim, Yoonjoo Lee, Yoonah Park, Jiho Kim, Young-Ho Kim, Juho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.01674)  

**Abstract**: Personalization of Large Language Models (LLMs) often assumes users hold static preferences that reflect globally in all tasks. In reality, humans hold dynamic preferences that change depending on the context. As users interact with an LLM in various contexts, they naturally reveal their contextual preferences, which a model must infer and apply in future contexts to ensure alignment. To assess this, we introduce CUPID, a benchmark of 756 human-curated interaction session histories between users and LLM-based chat assistants. In each interaction session, the user provides a request in a specific context and expresses their preference through multi-turn feedback. Given a new user request and prior interaction sessions, our benchmark assesses whether LLMs can infer the preference relevant to this request and generate a response that satisfies this preference. With CUPID, we evaluated 10 open and proprietary LLMs, revealing that state-of-the-art LLMs struggle to infer preferences from multi-turn interactions and fail to discern what previous context is relevant to a new request -- under 50% precision and 65% recall. Our work highlights the need to advance LLM capabilities for more contextually personalized interactions and proposes CUPID as a resource to drive these improvements. 

---
# Authorship Attribution in Multilingual Machine-Generated Texts 

**Authors**: Lucio La Cava, Dominik Macko, Róbert Móro, Ivan Srba, Andrea Tagarelli  

**Link**: [PDF](https://arxiv.org/pdf/2508.01656)  

**Abstract**: As Large Language Models (LLMs) have reached human-like fluency and coherence, distinguishing machine-generated text (MGT) from human-written content becomes increasingly difficult. While early efforts in MGT detection have focused on binary classification, the growing landscape and diversity of LLMs require a more fine-grained yet challenging authorship attribution (AA), i.e., being able to identify the precise generator (LLM or human) behind a text. However, AA remains nowadays confined to a monolingual setting, with English being the most investigated one, overlooking the multilingual nature and usage of modern LLMs. In this work, we introduce the problem of Multilingual Authorship Attribution, which involves attributing texts to human or multiple LLM generators across diverse languages. Focusing on 18 languages -- covering multiple families and writing scripts -- and 8 generators (7 LLMs and the human-authored class), we investigate the multilingual suitability of monolingual AA methods, their cross-lingual transferability, and the impact of generators on attribution performance. Our results reveal that while certain monolingual AA methods can be adapted to multilingual settings, significant limitations and challenges remain, particularly in transferring across diverse language families, underscoring the complexity of multilingual AA and the need for more robust approaches to better match real-world scenarios. 

---
# OpenMed NER: Open-Source, Domain-Adapted State-of-the-Art Transformers for Biomedical NER Across 12 Public Datasets 

**Authors**: Maziyar Panahi  

**Link**: [PDF](https://arxiv.org/pdf/2508.01630)  

**Abstract**: Named-entity recognition (NER) is fundamental to extracting structured information from the >80% of healthcare data that resides in unstructured clinical notes and biomedical literature. Despite recent advances with large language models, achieving state-of-the-art performance across diverse entity types while maintaining computational efficiency remains a significant challenge. We introduce OpenMed NER, a suite of open-source, domain-adapted transformer models that combine lightweight domain-adaptive pre-training (DAPT) with parameter-efficient Low-Rank Adaptation (LoRA). Our approach performs cost-effective DAPT on a 350k-passage corpus compiled from ethically sourced, publicly available research repositories and de-identified clinical notes (PubMed, arXiv, and MIMIC-III) using DeBERTa-v3, PubMedBERT, and BioELECTRA backbones. This is followed by task-specific fine-tuning with LoRA, which updates less than 1.5% of model parameters. We evaluate our models on 12 established biomedical NER benchmarks spanning chemicals, diseases, genes, and species. OpenMed NER achieves new state-of-the-art micro-F1 scores on 10 of these 12 datasets, with substantial gains across diverse entity types. Our models advance the state-of-the-art on foundational disease and chemical benchmarks (e.g., BC5CDR-Disease, +2.70 pp), while delivering even larger improvements of over 5.3 and 9.7 percentage points on more specialized gene and clinical cell line corpora. This work demonstrates that strategically adapted open-source models can surpass closed-source solutions. This performance is achieved with remarkable efficiency: training completes in under 12 hours on a single GPU with a low carbon footprint (< 1.2 kg CO2e), producing permissively licensed, open-source checkpoints designed to help practitioners facilitate compliance with emerging data protection and AI regulations, such as the EU AI Act. 

---
# Are All Prompt Components Value-Neutral? Understanding the Heterogeneous Adversarial Robustness of Dissected Prompt in Large Language Models 

**Authors**: Yujia Zheng, Tianhao Li, Haotian Huang, Tianyu Zeng, Jingyu Lu, Chuangxin Chu, Yuekai Huang, Ziyou Jiang, Qian Xiong, Yuyao Ge, Mingyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.01554)  

**Abstract**: Prompt-based adversarial attacks have become an effective means to assess the robustness of large language models (LLMs). However, existing approaches often treat prompts as monolithic text, overlooking their structural heterogeneity-different prompt components contribute unequally to adversarial robustness. Prior works like PromptRobust assume prompts are value-neutral, but our analysis reveals that complex, domain-specific prompts with rich structures have components with differing vulnerabilities. To address this gap, we introduce PromptAnatomy, an automated framework that dissects prompts into functional components and generates diverse, interpretable adversarial examples by selectively perturbing each component using our proposed method, ComPerturb. To ensure linguistic plausibility and mitigate distribution shifts, we further incorporate a perplexity (PPL)-based filtering mechanism. As a complementary resource, we annotate four public instruction-tuning datasets using the PromptAnatomy framework, verified through human review. Extensive experiments across these datasets and five advanced LLMs demonstrate that ComPerturb achieves state-of-the-art attack success rates. Ablation studies validate the complementary benefits of prompt dissection and PPL filtering. Our results underscore the importance of prompt structure awareness and controlled perturbation for reliable adversarial robustness evaluation in LLMs. Code and data are available at this https URL. 

---
# MOPrompt: Multi-objective Semantic Evolution for Prompt Optimization 

**Authors**: Sara Câmara, Eduardo Luz, Valéria Carvalho, Ivan Meneghini, Gladston Moreira  

**Link**: [PDF](https://arxiv.org/pdf/2508.01541)  

**Abstract**: Prompt engineering is crucial for unlocking the potential of Large Language Models (LLMs). Still, since manual prompt design is often complex, non-intuitive, and time-consuming, automatic prompt optimization has emerged as a research area. However, a significant challenge in prompt optimization is managing the inherent trade-off between task performance, such as accuracy, and context size. Most existing automated methods focus on a single objective, typically performance, thereby failing to explore the critical spectrum of efficiency and effectiveness. This paper introduces the MOPrompt, a novel Multi-objective Evolutionary Optimization (EMO) framework designed to optimize prompts for both accuracy and context size (measured in tokens) simultaneously. Our framework maps the Pareto front of prompt solutions, presenting practitioners with a set of trade-offs between context size and performance, a crucial tool for deploying Large Language Models (LLMs) in real-world applications. We evaluate MOPrompt on a sentiment analysis task in Portuguese, using Gemma-2B and Sabiazinho-3 as evaluation models. Our findings show that MOPrompt substantially outperforms the baseline framework. For the Sabiazinho model, MOPrompt identifies a prompt that achieves the same peak accuracy (0.97) as the best baseline solution, but with a 31% reduction in token length. 

---
# A Theory of Adaptive Scaffolding for LLM-Based Pedagogical Agents 

**Authors**: Clayton Cohn, Surya Rayala, Namrata Srivastava, Joyce Horn Fonteles, Shruti Jain, Xinying Luo, Divya Mereddy, Naveeduddin Mohammed, Gautam Biswas  

**Link**: [PDF](https://arxiv.org/pdf/2508.01503)  

**Abstract**: Large language models (LLMs) present new opportunities for creating pedagogical agents that engage in meaningful dialogue to support student learning. However, the current use of LLM systems like ChatGPT in classrooms often lacks the solid theoretical foundation found in earlier intelligent tutoring systems. To bridge this gap, we propose a framework that combines Evidence-Centered Design with Social Cognitive Theory for adaptive scaffolding in LLM-based agents focused on STEM+C learning. We illustrate this framework with Inquizzitor, an LLM-based formative assessment agent that integrates human-AI hybrid intelligence and provides feedback grounded in cognitive science principles. Our findings show that Inquizzitor delivers high-quality assessment and interaction aligned with core learning theories, offering teachers effective guidance that students value. This research underscores the potential for theory-driven LLM integration in education, highlighting the ability of these systems to provide adaptive and principled instruction. 

---
# The Homogenizing Effect of Large Language Models on Human Expression and Thought 

**Authors**: Zhivar Sourati, Alireza S. Ziabari, Morteza Dehghani  

**Link**: [PDF](https://arxiv.org/pdf/2508.01491)  

**Abstract**: Cognitive diversity, reflected in variations of language, perspective, and reasoning, is essential to creativity and collective intelligence. This diversity is rich and grounded in culture, history, and individual experience. Yet as large language models (LLMs) become deeply embedded in people's lives, they risk standardizing language and reasoning. This Review synthesizes evidence across linguistics, cognitive, and computer science to show how LLMs reflect and reinforce dominant styles while marginalizing alternative voices and reasoning strategies. We examine how their design and widespread use contribute to this effect by mirroring patterns in their training data and amplifying convergence as all people increasingly rely on the same models across contexts. Unchecked, this homogenization risks flattening the cognitive landscapes that drive collective intelligence and adaptability. 

---
# TeSent: A Benchmark Dataset for Fairness-aware Explainable Sentiment Classification in Telugu 

**Authors**: Vallabhaneni Raj Kumar, Ashwin S, Supriya Manna, Niladri Sett, Cheedella V S N M S Hema Harshitha, Kurakula Harshitha, Anand Kumar Sharma, Basina Deepakraj, Tanuj Sarkar, Bondada Navaneeth Krishna, Samanthapudi Shakeer  

**Link**: [PDF](https://arxiv.org/pdf/2508.01486)  

**Abstract**: In the Indian subcontinent, Telugu, one of India's six classical languages, is the most widely spoken Dravidian Language. Despite its 96 million speaker base worldwide, Telugu remains underrepresented in the global NLP and Machine Learning landscape, mainly due to lack of high-quality annotated resources. This work introduces TeSent, a comprehensive benchmark dataset for sentiment classification, a key text classification problem, in Telugu. TeSent not only provides ground truth labels for the sentences, but also supplements with provisions for evaluating explainability and fairness, two critical requirements in modern-day machine learning tasks. We scraped Telugu texts covering multiple domains from various social media platforms, news websites and web-blogs to preprocess and generate 26,150 sentences, and developed a custom-built annotation platform and a carefully crafted annotation protocol for collecting the ground truth labels along with their human-annotated rationales. We then fine-tuned several SOTA pre-trained models in two ways: with rationales, and without rationales. Further, we provide a detailed plausibility and faithfulness evaluation suite, which exploits the rationales, for six widely used post-hoc explainers applied on the trained models. Lastly, we curate TeEEC, Equity Evaluation Corpus in Telugu, a corpus to evaluate fairness of Telugu sentiment and emotion related NLP tasks, and provide a fairness evaluation suite for the trained classifier models. Our experimental results suggest that training with rationales may improve model accuracy, reduce bias in models, and make the explainers' output more aligned to human reasoning. 

---
# Harnessing Collective Intelligence of LLMs for Robust Biomedical QA: A Multi-Model Approach 

**Authors**: Dimitra Panou, Alexandros C. Dimopoulos, Manolis Koubarakis, Martin Reczko  

**Link**: [PDF](https://arxiv.org/pdf/2508.01480)  

**Abstract**: Biomedical text mining and question-answering are essential yet highly demanding tasks, particularly in the face of the exponential growth of biomedical literature. In this work, we present our participation in the 13th edition of the BioASQ challenge, which involves biomedical semantic question-answering for Task 13b and biomedical question-answering for developing topics for the Synergy task. We deploy a selection of open-source large language models (LLMs) as retrieval-augmented generators to answer biomedical questions. Various models are used to process the questions. A majority voting system combines their output to determine the final answer for Yes/No questions, while for list and factoid type questions, the union of their answers in used. We evaluated 13 state-of-the-art open source LLMs, exploring all possible model combinations to contribute to the final answer, resulting in tailored LLM pipelines for each question type. Our findings provide valuable insight into which combinations of LLMs consistently produce superior results for specific question types. In the four rounds of the 2025 BioASQ challenge, our system achieved notable results: in the Synergy task, we secured 1st place for ideal answers and 2nd place for exact answers in round 2, as well as two shared 1st places for exact answers in round 3 and 4. 

---
# TreeDiff: AST-Guided Code Generation with Diffusion LLMs 

**Authors**: Yiming Zeng, Jinghan Cao, Zexin Li, Yiming Chen, Tao Ren, Dawei Xiang, Xidong Wu, Shangqian Gao, Tingting Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01473)  

**Abstract**: Recent advances in diffusion-based language models have opened new possibilities for controllable and bidirectional sequence generation. These models provide an alternative to traditional autoregressive approaches by framing text generation as an iterative denoising process. However, applying diffusion models to structured domains such as source code remains a significant challenge. Programming languages differ from natural language in that they follow strict syntactic and semantic rules, with hierarchical organization that must be preserved for correctness. Standard token-level corruption techniques used during training often ignore this structure, which may hinder the model's ability to learn meaningful representations of code. To address this limitation, we propose a syntax-aware diffusion framework that incorporates structural priors from Abstract Syntax Trees (ASTs) into the denoising process. Instead of masking individual tokens at random, we selectively corrupt syntactically meaningful code spans derived from AST subtrees. This enables the model to reconstruct programs in a way that respects grammatical boundaries and captures long-range dependencies. Experimental results demonstrate that syntax-aware corruption significantly improves syntactic correctness, reconstruction accuracy, and generalization to unseen code patterns. These findings highlight the potential of incorporating structural information into diffusion-based training and suggest that syntax-guided denoising is a promising direction for advancing diffusion-based language models in code generation tasks. 

---
# Towards Efficient Medical Reasoning with Minimal Fine-Tuning Data 

**Authors**: Xinlin Zhuang, Feilong Tang, Haolin Yang, Ming Hu, Huifa Li, Haochen Xue, Yichen Li, Junjun He, Zongyuan Ge, Ying Qian, Imran Razzak  

**Link**: [PDF](https://arxiv.org/pdf/2508.01450)  

**Abstract**: Supervised Fine-Tuning (SFT) plays a pivotal role in adapting Large Language Models (LLMs) to specialized domains such as medical reasoning. However, existing SFT practices often rely on unfiltered datasets that contain redundant and low-quality samples, leading to substantial computational costs and suboptimal performance. Although existing methods attempt to alleviate this problem by selecting data based on sample difficulty, defined by knowledge and reasoning complexity, they overlook each sample's optimization utility reflected in its gradient. Interestingly, we find that gradient-based influence alone favors easy-to-optimize samples that cause large parameter shifts but lack deep reasoning chains, while difficulty alone selects noisy or overly complex cases that fail to guide stable optimization. Based on this observation, we propose a data selection strategy, Difficulty-Influence Quadrant (DIQ), which prioritizes samples in the high-difficulty-high-influence quadrant to balance complex clinical reasoning with substantial gradient influence, enabling efficient medical reasoning with minimal fine-tuning data. Furthermore, Human and LLM-as-a-judge evaluations show that DIQ-selected subsets demonstrate higher data quality and generate clinical reasoning that is more aligned with expert practices in differential diagnosis, safety check, and evidence citation, as DIQ emphasizes samples that foster expert-like reasoning patterns. Extensive experiments on medical reasoning benchmarks demonstrate that DIQ enables models fine-tuned on only 1% of selected data to match full-dataset performance, while using 10% consistently outperforms the baseline, highlighting the superiority of principled data selection over brute-force scaling. The code and data are available at this https URL. 

---
# From Query to Logic: Ontology-Driven Multi-Hop Reasoning in LLMs 

**Authors**: Haonan Bian, Yutao Qi, Rui Yang, Yuanxi Che, Jiaqian Wang, Heming Xia, Ranran Zhen  

**Link**: [PDF](https://arxiv.org/pdf/2508.01424)  

**Abstract**: Large Language Models (LLMs), despite their success in question answering, exhibit limitations in complex multi-hop question answering (MQA) tasks that necessitate non-linear, structured reasoning. This limitation stems from their inability to adequately capture deep conceptual relationships between entities. To overcome this challenge, we present **ORACLE** (**O**ntology-driven **R**easoning **A**nd **C**hain for **L**ogical **E**ucidation), a training-free framework that combines LLMs' generative capabilities with the structural benefits of knowledge graphs. Our approach operates through three stages: (1) dynamic construction of question-specific knowledge ontologies using LLMs, (2) transformation of these ontologies into First-Order Logic reasoning chains, and (3) systematic decomposition of the original query into logically coherent sub-questions. Experimental results on several standard MQA benchmarks show that our framework achieves highly competitive performance, rivaling current state-of-the-art models like DeepSeek-R1. Detailed analyses further confirm the effectiveness of each component, while demonstrating that our method generates more logical and interpretable reasoning chains than existing approaches. 

---
# Discovering Bias Associations through Open-Ended LLM Generations 

**Authors**: Jinhao Pan, Chahat Raj, Ziwei Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01412)  

**Abstract**: Social biases embedded in Large Language Models (LLMs) raise critical concerns, resulting in representational harms -- unfair or distorted portrayals of demographic groups -- that may be expressed in subtle ways through generated language. Existing evaluation methods often depend on predefined identity-concept associations, limiting their ability to surface new or unexpected forms of bias. In this work, we present the Bias Association Discovery Framework (BADF), a systematic approach for extracting both known and previously unrecognized associations between demographic identities and descriptive concepts from open-ended LLM outputs. Through comprehensive experiments spanning multiple models and diverse real-world contexts, BADF enables robust mapping and analysis of the varied concepts that characterize demographic identities. Our findings advance the understanding of biases in open-ended generation and provide a scalable tool for identifying and analyzing bias associations in LLMs. Data, code, and results are available at this https URL 

---
# ArzEn-MultiGenre: An aligned parallel dataset of Egyptian Arabic song lyrics, novels, and subtitles, with English translations 

**Authors**: Rania Al-Sabbagh  

**Link**: [PDF](https://arxiv.org/pdf/2508.01411)  

**Abstract**: ArzEn-MultiGenre is a parallel dataset of Egyptian Arabic song lyrics, novels, and TV show subtitles that are manually translated and aligned with their English counterparts. The dataset contains 25,557 segment pairs that can be used to benchmark new machine translation models, fine-tune large language models in few-shot settings, and adapt commercial machine translation applications such as Google Translate. Additionally, the dataset is a valuable resource for research in various disciplines, including translation studies, cross-linguistic analysis, and lexical semantics. The dataset can also serve pedagogical purposes by training translation students and aid professional translators as a translation memory. The contributions are twofold: first, the dataset features textual genres not found in existing parallel Egyptian Arabic and English datasets, and second, it is a gold-standard dataset that has been translated and aligned by human experts. 

---
# MedSynth: Realistic, Synthetic Medical Dialogue-Note Pairs 

**Authors**: Ahmad Rezaie Mianroodi, Amirali Rezaie, Niko Grisel Todorov, Cyril Rakovski, Frank Rudzicz  

**Link**: [PDF](https://arxiv.org/pdf/2508.01401)  

**Abstract**: Physicians spend significant time documenting clinical encounters, a burden that contributes to professional burnout. To address this, robust automation tools for medical documentation are crucial. We introduce MedSynth -- a novel dataset of synthetic medical dialogues and notes designed to advance the Dialogue-to-Note (Dial-2-Note) and Note-to-Dialogue (Note-2-Dial) tasks. Informed by an extensive analysis of disease distributions, this dataset includes over 10,000 dialogue-note pairs covering over 2000 ICD-10 codes. We demonstrate that our dataset markedly enhances the performance of models in generating medical notes from dialogues, and dialogues from medical notes. The dataset provides a valuable resource in a field where open-access, privacy-compliant, and diverse training data are scarce. Code is available at this https URL and the dataset is available at this https URL. 

---
# MaRGen: Multi-Agent LLM Approach for Self-Directed Market Research and Analysis 

**Authors**: Roman Koshkin, Pengyu Dai, Nozomi Fujikawa, Masahito Togami, Marco Visentini-Scarzanella  

**Link**: [PDF](https://arxiv.org/pdf/2508.01370)  

**Abstract**: We present an autonomous framework that leverages Large Language Models (LLMs) to automate end-to-end business analysis and market report generation. At its core, the system employs specialized agents - Researcher, Reviewer, Writer, and Retriever - that collaborate to analyze data and produce comprehensive reports. These agents learn from real professional consultants' presentation materials at Amazon through in-context learning to replicate professional analytical methodologies. The framework executes a multi-step process: querying databases, analyzing data, generating insights, creating visualizations, and composing market reports. We also introduce a novel LLM-based evaluation system for assessing report quality, which shows alignment with expert human evaluations. Building on these evaluations, we implement an iterative improvement mechanism that optimizes report quality through automated review cycles. Experimental results show that report quality can be improved by both automated review cycles and consultants' unstructured knowledge. In experimental validation, our framework generates detailed 6-page reports in 7 minutes at a cost of approximately \$1. Our work could be an important step to automatically create affordable market insights. 

---
# Large-Scale Diverse Synthesis for Mid-Training 

**Authors**: Xuemiao Zhang, Chengying Tu, Can Ren, Rongxiang Weng, Hongfei Yan, Jingang Wang, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2508.01326)  

**Abstract**: The scarcity of high-quality, knowledge-intensive training data hinders the development of large language models (LLMs), as traditional corpora provide limited information. Previous studies have synthesized and integrated corpora-dependent question-answering (QA) data to improve model performance but face challenges in QA data scalability and knowledge diversity, particularly in cross-domain contexts. Furthermore, leveraging our designed discipline and difficulty annotation system, we probe model deficiencies in STEM disciplines and high-difficulty data. To overcome these limitations, we propose a novel diversified pipeline to synthesize BoostQA, a 100B-token large-scale QA dataset. Our synthesis framework: (1) curates seed data from heterogeneous sources; (2) utilizes DeepSeek-R1 to implement STEM-focused multi-grade synthesis to boost data diversity and high-difficulty synthesis to mitigate difficulty degradation; (3) refines answers via DeepSeek-V3 to improve output quality. We utilize BoostQA in mid-training, a mid-stage between pre-training and post-training, to optimize domain-specific knowledge acquisition and enhance data quality. Our method enables Llama-3 8B, mid-trained on a 40B-token dataset, to achieve an average improvement of $\mathbf{12.74\%}$ on MMLU and CMMLU and establish SOTA average performance across 12 benchmarks. BoostQA also demonstrates robust scalability, with performance consistently improving as model size, data volume, and initial FLOPs scale. 

---
# LinkQA: Synthesizing Diverse QA from Multiple Seeds Strongly Linked by Knowledge Points 

**Authors**: Xuemiao Zhang, Can Ren, Chengying Tu, Rongxiang Weng, Hongfei Yan, Jingang Wang, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2508.01317)  

**Abstract**: The advancement of large language models (LLMs) struggles with the scarcity of high-quality, diverse training data. To address this limitation, we propose LinkSyn, a novel knowledge point (KP) graph-based synthesis framework that enables flexible control over discipline and difficulty distributions while balancing KP coverage and popularity. LinkSyn extracts KPs from question-answering (QA) seed data and constructs a KP graph to synthesize diverse QA data from multiple seeds strongly linked by KPs and sampled from graph walks. Specifically, LinkSyn incorporates (1) a knowledge distribution value function to guide the adjustment of path sampling probability and balance KP coverage and popularity during graph walks; (2) diffusion-based synthesis via DeepSeek-R1 by leveraging multiple seeds with dense logical associations along each path; and (3) high-difficulty QA enhancement within given disciplines by flexible difficulty adjustments. By executing LinkSyn, we synthesize LinkQA, a diverse multi-disciplinary QA dataset with 50B tokens. Extensive experiments on Llama-3 8B demonstrate that continual pre-training with LinkQA yields an average improvement of $\mathbf{11.51\%}$ on MMLU and CMMLU, establishing new SOTA results. LinkQA consistently enhances performance across model size and initial FLOPs scales. 

---
# D-SCoRE: Document-Centric Segmentation and CoT Reasoning with Structured Export for QA-CoT Data Generation 

**Authors**: Weibo Zhou, Lingbo Li, Shangsong Liang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01309)  

**Abstract**: The scarcity and high cost of high-quality question-answering (QA) datasets hinder supervised fine-tuning (SFT) for domain-specific large language models (LLMs). To address this, we introduce D-SCoRE, a training-free pipeline that utilizes LLMs and prompt engineering to produce diverse, high-quality QA datasets from arbitrary textual sources. D-SCoRE integrates $\textbf{D}$ocument-centric processing, $\textbf{S}$egmentation, $\textbf{Co}$T $\textbf{R}$easoning, and structured $\textbf{E}$xport to generate QA-COT datasets tailored for domain-aware SFT. Multi-dimensional control mechanisms, such as semantic role transformation, question type balancing, and counterfactual materials, enhance diversity and relevance, overcoming limitations of existing QA generation. LLMs fine-tuned on D-SCoRE-generated QA datasets, and human-annotated QA datasets (SQuAD, Covid-QA) are evaluated on SQuADShifts and Covid-QA test sets, with D-SCoRE outperforming across most domains. D-SCoRE generates six QA-CoT pairs with four-option counterfactual materials per 100-200-word text in 90 seconds using an 8B LLM on consumer-grade hardware. Its simplicity and scalability enable efficient QA generation and high-performance fine-tuning across domains. 

---
# KEDAS: Knowledge Editing Alignment with Diverse Augmentation and Self-adaptive Inference 

**Authors**: Chenming Tang, Yutong Yang, Yunfang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01302)  

**Abstract**: Knowledge editing aims to modify outdated knowledge in large language models (LLMs) efficiently while retaining their powerful capabilities. Most existing methods rely on either parameter-level editing or retrieval-based approaches. In this work, we propose Knowledge Editing alignment with Diverse Augmentation and Self-adaptive inference (KEDAS) to better align LLMs with knowledge editing. In the alignment phase, LLMs learn to apply in-context edited knowledge via low-rank adaptation. During editing, we design a diverse edit augmentation technique to improve the recall of edits. After that, a self-adaptive post-alignment inference mechanism is proposed, in which a filter-based smart retriever is employed to perform a dynamic selection of inference routing. Specifically, irrelevant queries will go through the original pre-alignment model directly, while relevant ones, together with their related edits, go through the model with aligned adapters activated. In experiments, KEDAS secures the highest overall performance scores in 35 out of 36 cases across four datasets with three LLMs on three settings, surpassing its strong knowledge editing alignment counterpart by about 19.8 harmonic mean scores of edit success, locality and portability and outperforming both parameter editing and retrieval-based baselines significantly. Analysis of computational cost and performance on general tasks further validates the robustness and efficiency of KEDAS, indicating that it presents an ideal paradigm of knowledge editing alignment. 

---
# Prompting Large Language Models with Partial Knowledge for Answering Questions with Unseen Entities 

**Authors**: Zhichao Yan, Jiapu Wang, Jiaoyan Chen, Yanyan Wang, Hongye Tan, Jiye Liang, Xiaoli Li, Ru Li, Jeff Z.Pan  

**Link**: [PDF](https://arxiv.org/pdf/2508.01290)  

**Abstract**: Retrieval-Augmented Generation (RAG) shows impressive performance by supplementing and substituting parametric knowledge in Large Language Models (LLMs). Retrieved knowledge can be divided into three types: explicit answer evidence, implicit answer clue, and insufficient answer context which can be further categorized into totally irrelevant and partially relevant information. Effectively utilizing partially relevant knowledge remains a key challenge for RAG systems, especially in incomplete knowledge base retrieval. Contrary to the conventional view, we propose a new perspective: LLMs can be awakened via partially relevant knowledge already embedded in LLMs. To comprehensively investigate this phenomenon, the triplets located in the gold reasoning path and their variants are used to construct partially relevant knowledge by removing the path that contains the answer. We provide theoretical analysis of the awakening effect in LLMs and support our hypothesis with experiments on two Knowledge Graphs (KGs) Question Answering (QA) datasets. Furthermore, we present a new task, Unseen Entity KGQA, simulating real-world challenges where entity linking fails due to KG incompleteness. Our awakening-based approach demonstrates greater efficacy in practical applications, outperforms traditional methods that rely on embedding-based similarity which are prone to returning noisy information. 

---
# Bridging LLMs and Symbolic Reasoning in Educational QA Systems: Insights from the XAI Challenge at IJCNN 2025 

**Authors**: Long S. T. Nguyen, Khang H. N. Vo, Thu H. A. Nguyen, Tuan C. Bui, Duc Q. Nguyen, Thanh-Tung Tran, Anh D. Nguyen, Minh L. Nguyen, Fabien Baldacci, Thang H. Bui, Emanuel Di Nardo, Angelo Ciaramella, Son H. Le, Ihsan Ullah, Lorenzo Di Rocco, Tho T. Quan  

**Link**: [PDF](https://arxiv.org/pdf/2508.01263)  

**Abstract**: The growing integration of Artificial Intelligence (AI) into education has intensified the need for transparency and interpretability. While hackathons have long served as agile environments for rapid AI prototyping, few have directly addressed eXplainable AI (XAI) in real-world educational contexts. This paper presents a comprehensive analysis of the XAI Challenge 2025, a hackathon-style competition jointly organized by Ho Chi Minh City University of Technology (HCMUT) and the International Workshop on Trustworthiness and Reliability in Neurosymbolic AI (TRNS-AI), held as part of the International Joint Conference on Neural Networks (IJCNN 2025). The challenge tasked participants with building Question-Answering (QA) systems capable of answering student queries about university policies while generating clear, logic-based natural language explanations. To promote transparency and trustworthiness, solutions were required to use lightweight Large Language Models (LLMs) or hybrid LLM-symbolic systems. A high-quality dataset was provided, constructed via logic-based templates with Z3 validation and refined through expert student review to ensure alignment with real-world academic scenarios. We describe the challenge's motivation, structure, dataset construction, and evaluation protocol. Situating the competition within the broader evolution of AI hackathons, we argue that it represents a novel effort to bridge LLMs and symbolic reasoning in service of explainability. Our findings offer actionable insights for future XAI-centered educational systems and competitive research initiatives. 

---
# WarriorMath: Enhancing the Mathematical Ability of Large Language Models with a Defect-aware Framework 

**Authors**: Yue Chen, Minghua He, Fangkai Yang, Pu Zhao, Lu Wang, Yu Kang, Yifei Dong, Yuefeng Zhan, Hao Sun, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01245)  

**Abstract**: Large Language Models (LLMs) excel in solving mathematical problems, yet their performance is often limited by the availability of high-quality, diverse training data. Existing methods focus on augmenting datasets through rephrasing or difficulty progression but overlook the specific failure modes of LLMs. This results in synthetic questions that the model can already solve, providing minimal performance gains. To address this, we propose WarriorMath, a defect-aware framework for mathematical problem solving that integrates both targeted data synthesis and progressive training. In the synthesis stage, we employ multiple expert LLMs in a collaborative process to generate, critique, and refine problems. Questions that base LLMs fail to solve are identified and iteratively improved through expert-level feedback, producing high-quality, defect-aware training data. In the training stage, we introduce a progressive learning framework that iteratively fine-tunes the model using increasingly challenging data tailored to its weaknesses. Experiments on six mathematical benchmarks show that WarriorMath outperforms strong baselines by 12.57% on average, setting a new state-of-the-art. Our results demonstrate the effectiveness of a defect-aware, multi-expert framework for improving mathematical ability. 

---
# WebDS: An End-to-End Benchmark for Web-based Data Science 

**Authors**: Ethan Hsu, Hong Meng Yam, Ines Bouissou, Aaron Murali John, Raj Thota, Josh Koe, Vivek Sarath Putta, G K Dharesan, Alexander Spangher, Shikhar Murty, Tenghao Huang, Christopher D. Manning  

**Link**: [PDF](https://arxiv.org/pdf/2508.01222)  

**Abstract**: A large portion of real-world data science tasks are complex and require multi-hop web-based interactions: finding appropriate data available on the internet, synthesizing real-time data of various modalities from different locations, and producing summarized analyses. Existing web benchmarks often focus on simplistic interactions, such as form submissions or e-commerce transactions, and often do not require diverse tool-using capabilities required for web based data science. Conversely, traditional data science benchmarks typically concentrate on static, often textually bound datasets and do not assess end-to-end workflows that encompass data acquisition, cleaning, analysis, and insight generation. In response, we introduce WebDS, the first end-to-end web-based data science benchmark. It comprises 870 web-based data science tasks across 29 diverse websites from structured government data portals to unstructured news media, challenging agents to perform complex, multi-step operations requiring the use of tools and heterogeneous data formats that better reflect the realities of modern data analytics. Evaluations of current SOTA LLM agents indicate significant performance gaps in accomplishing these tasks. For instance, Browser Use, which accomplishes 80% of tasks on Web Voyager, successfully completes only 15% of tasks in WebDS, which our analysis suggests is due to new failure modes like poor information grounding, repetitive behavior and shortcut-taking that agents performing WebDS' tasks display. By providing a more robust and realistic testing ground, WebDS sets the stage for significant advances in the development of practically useful LLM-based data science. 

---
# Show or Tell? Modeling the evolution of request-making in Human-LLM conversations 

**Authors**: Shengqi Zhu, Jeffrey M. Rzeszotarski, David Mimno  

**Link**: [PDF](https://arxiv.org/pdf/2508.01213)  

**Abstract**: Chat logs provide a rich source of information about LLM users, but patterns of user behavior are often masked by the variability of queries. We present a new task, segmenting chat queries into contents of requests, roles, query-specific context, and additional expressions. We find that, despite the familiarity of chat-based interaction, request-making in LLM queries remains significantly different from comparable human-human interactions. With the data resource, we introduce an important perspective of diachronic analyses with user expressions. We find that query patterns vary between early ones emphasizing requests, and individual users explore patterns but tend to converge with experience. Finally, we show that model capabilities affect user behavior, particularly with the introduction of new models, which are traceable at the community level. 

---
# Adaptive Content Restriction for Large Language Models via Suffix Optimization 

**Authors**: Yige Li, Peihai Jiang, Jun Sun, Peng Shu, Tianming Liu, Zhen Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01198)  

**Abstract**: Large Language Models (LLMs) have demonstrated significant success across diverse applications. However, enforcing content restrictions remains a significant challenge due to their expansive output space. One aspect of content restriction is preventing LLMs from generating harmful content via model alignment approaches such as supervised fine-tuning (SFT). Yet, the need for content restriction may vary significantly across user groups, change rapidly over time, and not always align with general definitions of harmfulness. Applying SFT to each of these specific use cases is impractical due to the high computational, data, and storage demands. Motivated by this need, we propose a new task called \textit{Adaptive Content Restriction} (AdaCoRe), which focuses on lightweight strategies -- methods without model fine-tuning -- to prevent deployed LLMs from generating restricted terms for specific use cases. We propose the first method for AdaCoRe, named \textit{Suffix Optimization (SOP)}, which appends a short, optimized suffix to any prompt to a) prevent a target LLM from generating a set of restricted terms, while b) preserving the output quality. To evaluate AdaCoRe approaches, including our SOP, we create a new \textit{Content Restriction Benchmark} (CoReBench), which contains 400 prompts for 80 restricted terms across 8 carefully selected categories. We demonstrate the effectiveness of SOP on CoReBench, which outperforms the system-level baselines such as system suffix by 15\%, 17\%, 10\%, 9\%, and 6\% on average restriction rates for Gemma2-2B, Mistral-7B, Vicuna-7B, Llama3-8B, and Llama3.1-8B, respectively. We also demonstrate that SOP is effective on POE, an online platform hosting various commercial LLMs, highlighting its practicality in real-world scenarios. 

---
# CSIRO-LT at SemEval-2025 Task 11: Adapting LLMs for Emotion Recognition for Multiple Languages 

**Authors**: Jiyu Chen, Necva Bölücü, Sarvnaz Karimi, Diego Mollá, Cécile L. Paris  

**Link**: [PDF](https://arxiv.org/pdf/2508.01161)  

**Abstract**: Detecting emotions across different languages is challenging due to the varied and culturally nuanced ways of emotional expressions. The \textit{Semeval 2025 Task 11: Bridging the Gap in Text-Based emotion} shared task was organised to investigate emotion recognition across different languages. The goal of the task is to implement an emotion recogniser that can identify the basic emotional states that general third-party observers would attribute to an author based on their written text snippet, along with the intensity of those emotions. We report our investigation of various task-adaptation strategies for LLMs in emotion recognition. We show that the most effective method for this task is to fine-tune a pre-trained multilingual LLM with LoRA setting separately for each language. 

---
# Asking the Right Questions: Benchmarking Large Language Models in the Development of Clinical Consultation Templates 

**Authors**: Liam G. McCoy, Fateme Nateghi Haredasht, Kanav Chopra, David Wu, David JH Wu, Abass Conteh, Sarita Khemani, Saloni Kumar Maharaj, Vishnu Ravi, Arth Pahwa, Yingjie Weng, Leah Rosengaus, Lena Giang, Kelvin Zhenghao Li, Olivia Jee, Daniel Shirvani, Ethan Goh, Jonathan H. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.01159)  

**Abstract**: This study evaluates the capacity of large language models (LLMs) to generate structured clinical consultation templates for electronic consultation. Using 145 expert-crafted templates developed and routinely used by Stanford's eConsult team, we assess frontier models -- including o3, GPT-4o, Kimi K2, Claude 4 Sonnet, Llama 3 70B, and Gemini 2.5 Pro -- for their ability to produce clinically coherent, concise, and prioritized clinical question schemas. Through a multi-agent pipeline combining prompt optimization, semantic autograding, and prioritization analysis, we show that while models like o3 achieve high comprehensiveness (up to 92.2\%), they consistently generate excessively long templates and fail to correctly prioritize the most clinically important questions under length constraints. Performance varies across specialties, with significant degradation in narrative-driven fields such as psychiatry and pain medicine. Our findings demonstrate that LLMs can enhance structured clinical information exchange between physicians, while highlighting the need for more robust evaluation methods that capture a model's ability to prioritize clinically salient information within the time constraints of real-world physician communication. 

---
# Cross-Domain Web Information Extraction at Pinterest 

**Authors**: Michael Farag, Patrick Halina, Andrey Zaytsev, Alekhya Munagala, Imtihan Ahmed, Junhao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01096)  

**Abstract**: The internet offers a massive repository of unstructured information, but it's a significant challenge to convert this into a structured format. At Pinterest, the ability to accurately extract structured product data from e-commerce websites is essential to enhance user experiences and improve content distribution. In this paper, we present Pinterest's system for attribute extraction, which achieves remarkable accuracy and scalability at a manageable cost. Our approach leverages a novel webpage representation that combines structural, visual, and text modalities into a compact form, optimizing it for small model learning. This representation captures each visible HTML node with its text, style and layout information. We show how this allows simple models such as eXtreme Gradient Boosting (XGBoost) to extract attributes more accurately than much more complex Large Language Models (LLMs) such as Generative Pre-trained Transformer (GPT). Our results demonstrate a system that is highly scalable, processing over 1,000 URLs per second, while being 1000 times more cost-effective than the cheapest GPT alternatives. 

---
# UrBLiMP: A Benchmark for Evaluating the Linguistic Competence of Large Language Models in Urdu 

**Authors**: Farah Adeeba, Brian Dillon, Hassan Sajjad, Rajesh Bhatt  

**Link**: [PDF](https://arxiv.org/pdf/2508.01006)  

**Abstract**: Multilingual Large Language Models (LLMs) have shown remarkable performance across various languages; however, they often include significantly less data for low-resource languages such as Urdu compared to high-resource languages like English. To assess the linguistic knowledge of LLMs in Urdu, we present the Urdu Benchmark of Linguistic Minimal Pairs (UrBLiMP) i.e. pairs of minimally different sentences that contrast in grammatical acceptability. UrBLiMP comprises 5,696 minimal pairs targeting ten core syntactic phenomena, carefully curated using the Urdu Treebank and diverse Urdu text corpora. A human evaluation of UrBLiMP annotations yielded a 96.10% inter-annotator agreement, confirming the reliability of the dataset. We evaluate twenty multilingual LLMs on UrBLiMP, revealing significant variation in performance across linguistic phenomena. While LLaMA-3-70B achieves the highest average accuracy (94.73%), its performance is statistically comparable to other top models such as Gemma-3-27B-PT. These findings highlight both the potential and the limitations of current multilingual LLMs in capturing fine-grained syntactic knowledge in low-resource languages. 

---
# MAO-ARAG: Multi-Agent Orchestration for Adaptive Retrieval-Augmented Generation 

**Authors**: Yiqun Chen, Erhan Zhang, Lingyong Yan, Shuaiqiang Wang, Jizhou Huang, Dawei Yin, Jiaxin Mao  

**Link**: [PDF](https://arxiv.org/pdf/2508.01005)  

**Abstract**: In question-answering (QA) systems, Retrieval-Augmented Generation (RAG) has become pivotal in enhancing response accuracy and reducing hallucination issues. The architecture of RAG systems varies significantly, encompassing single-round RAG, iterative RAG, and reasoning RAG, each tailored to address different types of queries. Due to the varying complexity of real-world queries, a fixed RAG pipeline often struggles to balance performance and cost efficiency across different queries. To address this challenge, we propose an adaptive RAG framework called MAO-ARAG, which leverages multi-agent orchestration. Our adaptive RAG is conceived as a multi-turn framework. Specifically, we define multiple executor agents, representing typical RAG modules such as query reformulation agents, document selection agent, and generation agents. A planner agent intelligently selects and integrates the appropriate agents from these executors into a suitable workflow tailored for each query, striving for high-quality answers while maintaining reasonable costs. During each turn, the planner agent is trained using reinforcement learning, guided by an outcome-based reward (F1 score) and a cost-based penalty, continuously improving answer quality while keeping costs within a reasonable range. Experiments conducted on multiple QA datasets demonstrate that our approach, which dynamically plans workflows for each query, not only achieves high answer quality but also maintains both cost and latency within acceptable this http URL code of MAO-ARAG is on this https URL. 

---
# XAutoLM: Efficient Fine-Tuning of Language Models via Meta-Learning and AutoML 

**Authors**: Ernesto L. Estevanell-Valladares, Suilan Estevez-Velarde, Yoan Gutiérrez, Andrés Montoyo, Ruslan Mitkov  

**Link**: [PDF](https://arxiv.org/pdf/2508.00924)  

**Abstract**: Experts in machine learning leverage domain knowledge to navigate decisions in model selection, hyperparameter optimisation, and resource allocation. This is particularly critical for fine-tuning language models (LMs), where repeated trials incur substantial computational overhead and environmental impact. However, no existing automated framework simultaneously tackles the entire model selection and HPO task for resource-efficient LM fine-tuning. We introduce XAutoLM, a meta-learning-augmented AutoML framework that reuses past experiences to optimise discriminative and generative LM fine-tuning pipelines efficiently. XAutoLM learns from stored successes and failures by extracting task- and system-level meta-features to bias its sampling toward fruitful configurations and away from costly dead ends. On four text classification and two question-answering benchmarks, XAutoLM surpasses zero-shot optimiser's peak F1 on five of six tasks, cuts mean evaluation time by up to 4.5x, reduces error ratios by up to sevenfold, and uncovers up to 50% more pipelines above the zero-shot Pareto front. In contrast, simpler memory-based baselines suffer negative transfer. We release XAutoLM and our experience store to catalyse resource-efficient, Green AI fine-tuning in the NLP community. 

---
# FECT: Factuality Evaluation of Interpretive AI-Generated Claims in Contact Center Conversation Transcripts 

**Authors**: Hagyeong Shin, Binoy Robin Dalal, Iwona Bialynicka-Birula, Navjot Matharu, Ryan Muir, Xingwei Yang, Samuel W. K. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2508.00889)  

**Abstract**: Large language models (LLMs) are known to hallucinate, producing natural language outputs that are not grounded in the input, reference materials, or real-world knowledge. In enterprise applications where AI features support business decisions, such hallucinations can be particularly detrimental. LLMs that analyze and summarize contact center conversations introduce a unique set of challenges for factuality evaluation, because ground-truth labels often do not exist for analytical interpretations about sentiments captured in the conversation and root causes of the business problems. To remedy this, we first introduce a \textbf{3D} -- \textbf{Decompose, Decouple, Detach} -- paradigm in the human annotation guideline and the LLM-judges' prompt to ground the factuality labels in linguistically-informed evaluation criteria. We then introduce \textbf{FECT}, a novel benchmark dataset for \textbf{F}actuality \textbf{E}valuation of Interpretive AI-Generated \textbf{C}laims in Contact Center Conversation \textbf{T}ranscripts, labeled under our 3D paradigm. Lastly, we report our findings from aligning LLM-judges on the 3D paradigm. Overall, our findings contribute a new approach for automatically evaluating the factuality of outputs generated by an AI system for analyzing contact center conversations. 

---
# Rethinking Graph-Based Document Classification: Learning Data-Driven Structures Beyond Heuristic Approaches 

**Authors**: Margarita Bugueño, Gerard de Melo  

**Link**: [PDF](https://arxiv.org/pdf/2508.00864)  

**Abstract**: In document classification, graph-based models effectively capture document structure, overcoming sequence length limitations and enhancing contextual understanding. However, most existing graph document representations rely on heuristics, domain-specific rules, or expert knowledge. Unlike previous approaches, we propose a method to learn data-driven graph structures, eliminating the need for manual design and reducing domain dependence. Our approach constructs homogeneous weighted graphs with sentences as nodes, while edges are learned via a self-attention model that identifies dependencies between sentence pairs. A statistical filtering strategy aims to retain only strongly correlated sentences, improving graph quality while reducing the graph size. Experiments on three document classification datasets demonstrate that learned graphs consistently outperform heuristic-based graphs, achieving higher accuracy and $F_1$ score. Furthermore, our study demonstrates the effectiveness of the statistical filtering in improving classification robustness. These results highlight the potential of automatic graph generation over traditional heuristic approaches and open new directions for broader applications in NLP. 

---
# HyCodePolicy: Hybrid Language Controllers for Multimodal Monitoring and Decision in Embodied Agents 

**Authors**: Yibin Liu, Zhixuan Liang, Zanxin Chen, Tianxing Chen, Mengkang Hu, Wanxi Dong, Congsheng Xu, Zhaoming Han, Yusen Qin, Yao Mu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02629)  

**Abstract**: Recent advances in multimodal large language models (MLLMs) have enabled richer perceptual grounding for code policy generation in embodied agents. However, most existing systems lack effective mechanisms to adaptively monitor policy execution and repair codes during task completion. In this work, we introduce HyCodePolicy, a hybrid language-based control framework that systematically integrates code synthesis, geometric grounding, perceptual monitoring, and iterative repair into a closed-loop programming cycle for embodied agents. Technically, given a natural language instruction, our system first decomposes it into subgoals and generates an initial executable program grounded in object-centric geometric primitives. The program is then executed in simulation, while a vision-language model (VLM) observes selected checkpoints to detect and localize execution failures and infer failure reasons. By fusing structured execution traces capturing program-level events with VLM-based perceptual feedback, HyCodePolicy infers failure causes and repairs programs. This hybrid dual feedback mechanism enables self-correcting program synthesis with minimal human supervision. Our results demonstrate that HyCodePolicy significantly improves the robustness and sample efficiency of robot manipulation policies, offering a scalable strategy for integrating multimodal reasoning into autonomous decision-making pipelines. 

---
# Noosemia: toward a Cognitive and Phenomenological Account of Intentionality Attribution in Human-Generative AI Interaction 

**Authors**: Enrico De Santis, Antonello Rizzi  

**Link**: [PDF](https://arxiv.org/pdf/2508.02622)  

**Abstract**: This paper introduces and formalizes Noosemia, a novel cognitive-phenomenological phenomenon emerging from human interaction with generative AI systems, particularly those enabling dialogic or multimodal exchanges. We propose a multidisciplinary framework to explain how, under certain conditions, users attribute intentionality, agency, and even interiority to these systems - a process grounded not in physical resemblance, but in linguistic performance, epistemic opacity, and emergent technological complexity. By linking an LLM declination of meaning holism to our technical notion of the LLM Contextual Cognitive Field, we clarify how LLMs construct meaning relationally and how coherence and a simulacrum of agency arise at the human-AI interface. The analysis situates noosemia alongside pareidolia, animism, the intentional stance and the uncanny valley, distinguishing its unique characteristics. We also introduce a-noosemia to describe the phenomenological withdrawal of such projections. The paper concludes with reflections on the broader philosophical, epistemological, and social implications of noosemic dynamics and directions for future research. 

---
# HealthFlow: A Self-Evolving AI Agent with Meta Planning for Autonomous Healthcare Research 

**Authors**: Yinghao Zhu, Yifan Qi, Zixiang Wang, Lei Gu, Dehao Sui, Haoran Hu, Xichen Zhang, Ziyi He, Liantao Ma, Lequan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02621)  

**Abstract**: The efficacy of AI agents in healthcare research is hindered by their reliance on static, predefined strategies. This creates a critical limitation: agents can become better tool-users but cannot learn to become better strategic planners, a crucial skill for complex domains like healthcare. We introduce HealthFlow, a self-evolving AI agent that overcomes this limitation through a novel meta-level evolution mechanism. HealthFlow autonomously refines its own high-level problem-solving policies by distilling procedural successes and failures into a durable, strategic knowledge base. To anchor our research and facilitate reproducible evaluation, we introduce EHRFlowBench, a new benchmark featuring complex, realistic health data analysis tasks derived from peer-reviewed clinical research. Our comprehensive experiments demonstrate that HealthFlow's self-evolving approach significantly outperforms state-of-the-art agent frameworks. This work marks a necessary shift from building better tool-users to designing smarter, self-evolving task-managers, paving the way for more autonomous and effective AI for scientific discovery. 

---
# Parameter-Efficient Routed Fine-Tuning: Mixture-of-Experts Demands Mixture of Adaptation Modules 

**Authors**: Yilun Liu, Yunpu Ma, Yuetian Lu, Shuo Chen, Zifeng Ding, Volker Tresp  

**Link**: [PDF](https://arxiv.org/pdf/2508.02587)  

**Abstract**: Mixture-of-Experts (MoE) benefits from a dynamic routing mechanism among their specialized experts, which existing Parameter- Efficient Fine-Tuning (PEFT) strategies fail to leverage. This motivates us to investigate whether adaptation modules themselves should incorporate routing mechanisms to align with MoE's multi-expert architecture. We analyze dynamics of core components when applying PEFT to MoE language models and examine how different routing strategies affect adaptation effectiveness. Extensive experiments adapting OLMoE-1B-7B and Mixtral-8x7B on various commonsense and math reasoning tasks validate the performance and efficiency of our routed approach. We identify the optimal configurations for different scenarios and provide empirical analyses with practical insights to facilitate better PEFT and MoE applications. 

---
# What are you sinking? A geometric approach on attention sink 

**Authors**: Valeria Ruscio, Umberto Nanni, Fabrizio Silvestri  

**Link**: [PDF](https://arxiv.org/pdf/2508.02546)  

**Abstract**: Attention sink (AS) is a consistent pattern in transformer attention maps where certain tokens (often special tokens or positional anchors) disproportionately attract attention from other tokens. We show that in transformers, AS is not an architectural artifact, but it is the manifestation of a fundamental geometric principle: the establishment of reference frames that anchor representational spaces. We analyze several architectures and identify three distinct reference frame types, centralized, distributed, and bidirectional, that correlate with the attention sink phenomenon. We show that they emerge during the earliest stages of training as optimal solutions to the problem of establishing stable coordinate systems in high-dimensional spaces. We show the influence of architecture components, particularly position encoding implementations, on the specific type of reference frame. This perspective transforms our understanding of transformer attention mechanisms and provides insights for both architecture design and the relationship with AS. 

---
# Test-time Prompt Intervention 

**Authors**: Chenxu Yang, Qingyi Si, Mz Dai, Dingyu Yao, Mingyu Zheng, Minghui Chen, Zheng Lin, Weiping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02511)  

**Abstract**: Test-time compute has led to remarkable success in the large language model (LLM) community, particularly for complex tasks, where longer chains of thought (CoTs) are generated to enhance reasoning capabilities. However, growing evidence reveals that such reasoning models often produce CoTs plagued by excessive redundancy, including unnecessary verification steps and repetitive reasoning shifts. The root cause lies in post-training of them that overly rely on outcome reward paradigms, as the data of process reward paradigms, which regulate intermediate reasoning steps, is difficult to construct at scale. To address this, we propose PI, a novel framework for Test-time Prompt Intervention. PI provides an interface to dynamically guide and regulate reasoning paths during inference through timely (When module) and proper (How module) interventions and post-intervention sampling (Which module). This allows human problem-solving expertise and cognitive science principles to be seamlessly integrated into LLMs' reasoning processes, enhancing controllability and interpretability. Extensive experiments across multiple models and datasets demonstrate that PI significantly shortens CoTs while reducing hallucination, yielding more concise and reliable reasoning. 

---
# OptiHive: Ensemble Selection for LLM-Based Optimization via Statistical Modeling 

**Authors**: Maxime Bouscary, Saurabh Amin  

**Link**: [PDF](https://arxiv.org/pdf/2508.02503)  

**Abstract**: LLM-based solvers have emerged as a promising means of automating problem modeling and solving. However, they remain unreliable and often depend on iterative repair loops that result in significant latency. We introduce OptiHive, an LLM-based framework that produces high-quality solvers for optimization problems from natural-language descriptions without iterative self-correction. OptiHive uses a single batched LLM query to generate diverse components (solvers, problem instances, and validation tests) and filters out erroneous components to ensure fully interpretable outputs. Taking into account the imperfection of the generated components, we employ a statistical model to infer their true performance, enabling principled uncertainty quantification and solver selection. On tasks ranging from traditional optimization problems to challenging variants of the Multi-Depot Vehicle Routing Problem, OptiHive significantly outperforms baselines, increasing the optimality rate from 5\% to 92\% on the most complex problems. 

---
# AIAP: A No-Code Workflow Builder for Non-Experts with Natural Language and Multi-Agent Collaboration 

**Authors**: Hyunjn An, Yongwon Kim, Wonduk Seo, Joonil Park, Daye Kang, Changhoon Oh, Dokyun Kim, Seunghyun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.02470)  

**Abstract**: While many tools are available for designing AI, non-experts still face challenges in clearly expressing their intent and managing system complexity. We introduce AIAP, a no-code platform that integrates natural language input with visual workflows. AIAP leverages a coordinated multi-agent system to decompose ambiguous user instructions into modular, actionable steps, hidden from users behind a unified interface. A user study involving 32 participants showed that AIAP's AI-generated suggestions, modular workflows, and automatic identification of data, actions, and context significantly improved participants' ability to develop services intuitively. These findings highlight that natural language-based visual programming significantly reduces barriers and enhances user experience in AI service design. 

---
# Modality Bias in LVLMs: Analyzing and Mitigating Object Hallucination via Attention Lens 

**Authors**: Haohan Zheng, Zhenguo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02419)  

**Abstract**: Large vision-language models (LVLMs) have demonstrated remarkable multimodal comprehension and reasoning capabilities, but they still suffer from severe object hallucination. Previous studies primarily attribute the flaw to linguistic prior caused by the scale mismatch between visual encoders and large language models (LLMs) in LVLMs. Specifically, as current LVLMs are built upon LLMs, they tend to over-rely on textual prompts and internal knowledge of LLMs, generating descriptions inconsistent with visual cues. However, through an in-depth investigation of the hallucinated mechanisms, we empirically reveal a previously overlooked phenomenon: LVLMs may ignore not only visual information but also textual modality during hallucination, a behavior termed as modality bias, which indicates that LVLMs struggle to simultaneously attend to both visual and textual modalities, leading to fragmented understanding of user-provided instructions. Based on this observation, we propose a simple yet effective training-free method to mitigate object hallucination. Concretely, we intervene and adjust the attention weights of textual and visual tokens, balancing cross-modal compatibility for better alignment with user intentions. Furthermore, we adopt a contrastive decoding strategy to reduce the LVLM's overreliance on its parametric knowledge, synergistically enhancing our attention manipulation. Extensive experiments confirm the widespread presence of modality bias in LVLMs. Notably, our method effectively mitigates hallucination across multiple open-source LVLMs and benchmarks, highlighting its generalizability and efficacy. 

---
# Six Guidelines for Trustworthy, Ethical and Responsible Automation Design 

**Authors**: Matouš Jelínek, Nadine Schlicker, Ewart de Visser  

**Link**: [PDF](https://arxiv.org/pdf/2508.02371)  

**Abstract**: Calibrated trust in automated systems (Lee and See 2004) is critical for their safe and seamless integration into society. Users should only rely on a system recommendation when it is actually correct and reject it when it is factually wrong. One requirement to achieve this goal is an accurate trustworthiness assessment, ensuring that the user's perception of the system's trustworthiness aligns with its actual trustworthiness, allowing users to make informed decisions about the extent to which they can rely on the system (Schlicker et al. 2022). We propose six design guidelines to help designers optimize for accurate trustworthiness assessments, thus fostering ethical and responsible human-automation interactions. The proposed guidelines are derived from existing literature in various fields, such as human-computer interaction, cognitive psychology, automation research, user-experience design, and ethics. We are incorporating key principles from the field of pragmatics, specifically the cultivation of common ground (H. H. Clark 1996) and Gricean communication maxims (Grice 1975). These principles are essential for the design of automated systems because the user's perception of the system's trustworthiness is shaped by both environmental contexts, such as organizational culture or societal norms, and by situational context, including the specific circumstances or scenarios in which the interaction occurs (Hoff and Bashir 2015). Our proposed guidelines provide actionable insights for designers to create automated systems that make relevant trustworthiness cues available. This would ideally foster calibrated trust and more satisfactory, productive, and safe interactions between humans and automated systems. Furthermore, the proposed heuristics might work as a tool for evaluating to what extent existing systems enable users to accurately assess a system's trustworthiness. 

---
# Language Model Guided Reinforcement Learning in Quantitative Trading 

**Authors**: Adam Darmanin, Vince Vella  

**Link**: [PDF](https://arxiv.org/pdf/2508.02366)  

**Abstract**: Algorithmic trading requires short-term decisions aligned with long-term financial goals. While reinforcement learning (RL) has been explored for such tactical decisions, its adoption remains limited by myopic behavior and opaque policy rationale. In contrast, large language models (LLMs) have recently demonstrated strategic reasoning and multi-modal financial signal interpretation when guided by well-designed prompts.
We propose a hybrid system where LLMs generate high-level trading strategies to guide RL agents in their actions. We evaluate (i) the rationale of LLM-generated strategies via expert review, and (ii) the Sharpe Ratio (SR) and Maximum Drawdown (MDD) of LLM-guided agents versus unguided baselines. Results show improved return and risk metrics over standard RL. 

---
# Understanding User Preferences for Interaction Styles in Conversational Recommender Systems: The Predictive Role of System Qualities, User Experience, and Traits 

**Authors**: Raj Mahmud, Shlomo Berkovsky, Mukesh Prasad, A. Baki Kocaballi  

**Link**: [PDF](https://arxiv.org/pdf/2508.02328)  

**Abstract**: Conversational Recommender Systems (CRSs) deliver personalised recommendations through multi-turn natural language dialogue and increasingly support both task-oriented and exploratory interactions. Yet, the factors shaping user interaction preferences remain underexplored. In this within-subjects study (\(N = 139\)), participants experienced two scripted CRS dialogues, rated their experiences, and indicated the importance of eight system qualities. Logistic regression revealed that preference for the exploratory interaction was predicted by enjoyment, usefulness, novelty, and conversational quality. Unexpectedly, perceived effectiveness was also associated with exploratory preference. Clustering uncovered five latent user profiles with distinct dialogue style preferences. Moderation analyses indicated that age, gender, and control preference significantly influenced these choices. These findings integrate affective, cognitive, and trait-level predictors into CRS user modelling and inform autonomy-sensitive, value-adaptive dialogue design. The proposed predictive and adaptive framework applies broadly to conversational AI systems seeking to align dynamically with evolving user needs. 

---
# CAPO: Towards Enhancing LLM Reasoning through Verifiable Generative Credit Assignment 

**Authors**: Guofu Xie, Yunsheng Shi, Hongtao Tian, Ting Yao, Xiao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02298)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has improved the reasoning abilities of Large Language Models (LLMs) by using rule-based binary feedback, helping to mitigate reward hacking. However, current RLVR methods typically treat whole responses as single actions, assigning the same reward to every token. This coarse-grained feedback hampers precise credit assignment, making it hard for models to identify which reasoning steps lead to success or failure, and often results in suboptimal policies and inefficient learning. Methods like PPO provide credit assignment through value estimation, but often yield inaccurate and unverifiable signals due to limited sampling. On the other hand, methods using Process Reward Models can provide step-by-step judgments for each reasoning step, but they require high-quality process supervision labels and are time-consuming when applied in online reinforcement learning (RL). To overcome these limitations, we introduce a simple but efficient method Credit Assignment Policy Optimization (CAPO). Given a reasoning response rollout from the policy model, CAPO directly leverages an off-the-shelf, general-purpose LLM as a Generative Process Reward Model (LLM-as-GenPRM) to generate all step-wise critique by one pass, thereby providing verifiable token-level rewards to refine the tokens that were originally assigned identical rule-based rewards. This enables more fine-grained credit assignment in an effective way. Furthermore, to enhance the accuracy and robustness of CAPO, we employ voting mechanisms that scale with the number of generated critiques. Extensive experiments using different backbones like Llama and Qwen models and in different sizes show that CAPO consistently outperforms supervised learning-based and RL-based fine-tuning methods across six challenging mathematical benchmarks and three out-of-domain benchmarks. 

---
# Dialogue Systems Engineering: A Survey and Future Directions 

**Authors**: Mikio Nakano, Hironori Takeuchi, Sadahiro Yoshikawa, Yoichi Matsuyama, Kazunori Komatani  

**Link**: [PDF](https://arxiv.org/pdf/2508.02279)  

**Abstract**: This paper proposes to refer to the field of software engineering related to the life cycle of dialogue systems as Dialogue Systems Engineering, and surveys this field while also discussing its future directions. With the advancement of large language models, the core technologies underlying dialogue systems have significantly progressed. As a result, dialogue system technology is now expected to be applied to solving various societal issues and in business contexts. To achieve this, it is important to build, operate, and continuously improve dialogue systems correctly and efficiently. Accordingly, in addition to applying existing software engineering knowledge, it is becoming increasingly important to evolve software engineering tailored specifically to dialogue systems. In this paper, we enumerate the knowledge areas of dialogue systems engineering based on those of software engineering, as defined in the Software Engineering Body of Knowledge (SWEBOK) Version 4.0, and survey each area. Based on this survey, we identify unexplored topics in each area and discuss the future direction of dialogue systems engineering. 

---
# CellForge: Agentic Design of Virtual Cell Models 

**Authors**: Xiangru Tang, Zhuoyun Yu, Jiapeng Chen, Yan Cui, Daniel Shao, Weixu Wang, Fang Wu, Yuchen Zhuang, Wenqi Shi, Zhi Huang, Arman Cohan, Xihong Lin, Fabian Theis, Smita Krishnaswamy, Mark Gerstein  

**Link**: [PDF](https://arxiv.org/pdf/2508.02276)  

**Abstract**: Virtual cell modeling represents an emerging frontier at the intersection of artificial intelligence and biology, aiming to predict quantities such as responses to diverse perturbations quantitatively. However, autonomously building computational models for virtual cells is challenging due to the complexity of biological systems, the heterogeneity of data modalities, and the need for domain-specific expertise across multiple disciplines. Here, we introduce CellForge, an agentic system that leverages a multi-agent framework that transforms presented biological datasets and research objectives directly into optimized computational models for virtual cells. More specifically, given only raw single-cell multi-omics data and task descriptions as input, CellForge outputs both an optimized model architecture and executable code for training virtual cell models and inference. The framework integrates three core modules: Task Analysis for presented dataset characterization and relevant literature retrieval, Method Design, where specialized agents collaboratively develop optimized modeling strategies, and Experiment Execution for automated generation of code. The agents in the Design module are separated into experts with differing perspectives and a central moderator, and have to collaboratively exchange solutions until they achieve a reasonable consensus. We demonstrate CellForge's capabilities in single-cell perturbation prediction, using six diverse datasets that encompass gene knockouts, drug treatments, and cytokine stimulations across multiple modalities. CellForge consistently outperforms task-specific state-of-the-art methods. Overall, CellForge demonstrates how iterative interaction between LLM agents with differing perspectives provides better solutions than directly addressing a modeling challenge. Our code is publicly available at this https URL. 

---
# LeanK: Learnable K Cache Channel Pruning for Efficient Decoding 

**Authors**: Yike Zhang, Zhiyuan He, Huiqiang Jiang, Chengruidong Zhang, Yuqing Yang, Jianyong Wang, Lili Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02215)  

**Abstract**: Large language models (LLMs) enable long-context tasks but face efficiency challenges due to the growing key-value (KV) cache. We propose LeanK, a learning-based method that prunes unimportant key (K) cache channels by leveraging static channel sparsity. With a novel two-stage training process, LeanK learns channel-wise static mask that could satisfy specific sparsity ratio and hardware alignment requirement. LeanK reduces GPU memory and accelerates decoding without sacrificing accuracy. Experiments demonstrate up to 70% K cache and 16%-18% V cache memory reduction. Custom decoding kernel enables 1.3x speedup for attention computation. We also provide insights into model channels and attention heads during long-context inference by analyzing the learned importance distribution. Our code is available at this https URL. 

---
# Hidden in the Noise: Unveiling Backdoors in Audio LLMs Alignment through Latent Acoustic Pattern Triggers 

**Authors**: Liang Lin, Miao Yu, Kaiwen Luo, Yibo Zhang, Lilan Peng, Dexian Wang, Xuehai Tang, Yuanhe Zhang, Xikang Yang, Zhenhong Zhou, Kun Wang, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02175)  

**Abstract**: As Audio Large Language Models (ALLMs) emerge as powerful tools for speech processing, their safety implications demand urgent attention. While considerable research has explored textual and vision safety, audio's distinct characteristics present significant challenges. This paper first investigates: Is ALLM vulnerable to backdoor attacks exploiting acoustic triggers? In response to this issue, we introduce Hidden in the Noise (HIN), a novel backdoor attack framework designed to exploit subtle, audio-specific features. HIN applies acoustic modifications to raw audio waveforms, such as alterations to temporal dynamics and strategic injection of spectrally tailored noise. These changes introduce consistent patterns that an ALLM's acoustic feature encoder captures, embedding robust triggers within the audio stream. To evaluate ALLM robustness against audio-feature-based triggers, we develop the AudioSafe benchmark, assessing nine distinct risk types. Extensive experiments on AudioSafe and three established safety datasets reveal critical vulnerabilities in existing ALLMs: (I) audio features like environment noise and speech rate variations achieve over 90% average attack success rate. (II) ALLMs exhibit significant sensitivity differences across acoustic features, particularly showing minimal response to volume as a trigger, and (III) poisoned sample inclusion causes only marginal loss curve fluctuations, highlighting the attack's stealth. 

---
# Subject or Style: Adaptive and Training-Free Mixture of LoRAs 

**Authors**: Jia-Chen Zhang, Yu-Jie Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2508.02165)  

**Abstract**: Fine-tuning models via Low-Rank Adaptation (LoRA) demonstrates remarkable performance in subject-driven or style-driven generation tasks. Studies have explored combinations of different LoRAs to jointly generate learned styles and content. However, current methods struggle to balance the original subject and style, and often require additional training. Recently, K-LoRA proposed a training-free LoRA fusion method. But it involves multiple hyperparameters, making it difficult to adapt to all styles and subjects. In this paper, we propose EST-LoRA, a training-free adaptive LoRA fusion method. It comprehensively considers three critical factors: \underline{E}nergy of matrix, \underline{S}tyle discrepancy scores and \underline{T}ime steps. Analogous to the Mixture of Experts (MoE) architecture, the model adaptively selects between subject LoRA and style LoRA within each attention layer. This integrated selection mechanism ensures balanced contributions from both components during the generation process. Experimental results show that EST-LoRA outperforms state-of-the-art methods in both qualitative and quantitative evaluations and achieves faster generation speed compared to other efficient fusion approaches. Our code is publicly available at: this https URL. 

---
# Trainable Dynamic Mask Sparse Attention 

**Authors**: Jingze Shi, Yifan Wu, Bingheng Wu, Yiran Peng, Liangdong Wang, Guang Liu, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2508.02124)  

**Abstract**: In large language models, the demand for modeling long contexts is constantly increasing, but the quadratic complexity of the standard self-attention mechanism often becomes a bottleneck. Although existing sparse attention mechanisms have improved efficiency, they may still encounter issues such as static patterns or information loss. We introduce a trainable dynamic mask sparse attention mechanism, Dynamic Mask Attention, which effectively utilizes content-aware and position-aware sparsity. DMA achieves this through two key innovations: First, it dynamically generates content-aware sparse masks from value representations, enabling the model to identify and focus on critical information adaptively. Second, it implements position-aware sparse attention computation that effectively skips unnecessary calculation regions. This dual-sparsity design allows the model to significantly reduce the computational complexity of important information while retaining complete information, achieving an excellent balance between information fidelity and computational efficiency. We have verified the performance of DMA through comprehensive experiments. Comparative studies show that DMA outperforms multi-head attention, sliding window attention, multi-head latent attention, and native sparse attention in terms of perplexity under Chinchilla Scaling Law settings. Moreover, in challenging multi-query associative recall tasks, DMA also demonstrates superior performance and efficiency compared to these methods. Crucially, in the evaluation of a 1.7B parameter model, DMA significantly outperforms multi-head attention in both standard benchmark performance and the challenging needle-in-a-haystack task. These experimental results highlight its capability to balance model efficiency and long-context modeling ability effectively. 

---
# CRINN: Contrastive Reinforcement Learning for Approximate Nearest Neighbor Search 

**Authors**: Xiaoya Li, Xiaofei Sun, Albert Wang, Chris Shum, Jiwei Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.02091)  

**Abstract**: Approximate nearest-neighbor search (ANNS) algorithms have become increasingly critical for recent AI applications, particularly in retrieval-augmented generation (RAG) and agent-based LLM applications. In this paper, we present CRINN, a new paradigm for ANNS algorithms. CRINN treats ANNS optimization as a reinforcement learning problem where execution speed serves as the reward signal. This approach enables the automatic generation of progressively faster ANNS implementations while maintaining accuracy constraints. Our experimental evaluation demonstrates CRINN's effectiveness across six widely-used NNS benchmark datasets. When compared against state-of-the-art open-source ANNS algorithms, CRINN achieves best performance on three of them (GIST-960-Euclidean, MNIST-784-Euclidean, and GloVe-25-angular), and tied for first place on two of them (SIFT-128-Euclidean and GloVe-25-angular). The implications of CRINN's success reach well beyond ANNS optimization: It validates that LLMs augmented with reinforcement learning can function as an effective tool for automating sophisticated algorithmic optimizations that demand specialized knowledge and labor-intensive manual this http URL can be found at this https URL 

---
# Human Capital Visualization using Speech Amount during Meetings 

**Authors**: Ekai Hashimoto, Takeshi Mizumoto, Kohei Nagira, Shun Shiramatsu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02075)  

**Abstract**: In recent years, many companies have recognized the importance of human resources and are investing in human capital to revitalize their organizations and enhance internal communication, thereby fostering innovation. However, conventional quantification methods have mainly focused on readily measurable indicators without addressing the fundamental role of conversations in human capital. This study focuses on routine meetings and proposes strategies to visualize human capital by analyzing speech amount during these meetings. We employ conversation visualization technology, which operates effectively, to quantify speech. We then measure differences in speech amount by attributes such as gender and job post, changes in speech amount depending on whether certain participants are present, and correlations between speech amount and continuous attributes. To verify the effectiveness of our proposed methods, we analyzed speech amounts by departmental affiliation during weekly meetings at small to medium enterprises. 

---
# MolReasoner: Toward Effective and Interpretable Reasoning for Molecular LLMs 

**Authors**: Guojiang Zhao, Sihang Li, Zixiang Lu, Zheng Cheng, Haitao Lin, Lirong Wu, Hanchen Xia, Hengxing Cai, Wentao Guo, Hongshuai Wang, Mingjun Xu, Siyu Zhu, Guolin Ke, Linfeng Zhang, Zhifeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2508.02066)  

**Abstract**: Large Language Models(LLMs) have demonstrated remarkable performance across various domains, yet their capabilities in molecular reasoning remain insufficiently explored. Current approaches tend to rely heavily on general-purpose prompting, which lacks domain-specific molecular semantics, while those that use fine-tuning strategies often face challenges with interpretability and reasoning depth. To address these issues, we introduce MolReasoner, a two-stage framework designed to transition LLMs from memorization towards chemical reasoning. First, we propose Mol-SFT, which initializes the model's reasoning abilities via synthetic Chain-of-Thought(CoT) samples generated by GPT-4o and verified for chemical accuracy. Subsequently, Mol-RL applies reinforcement learning with specialized reward functions designed explicitly to align chemical structures with linguistic descriptions, thereby enhancing molecular reasoning capabilities. Our approach notably enhances interpretability, improving the model 's molecular understanding and enabling better generalization. Extensive experiments demonstrate that MolReasoner outperforms existing methods, and marking a significant shift from memorization-based outputs to robust chemical reasoning. 

---
# Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning 

**Authors**: Xinting Huang, Michael Hahn  

**Link**: [PDF](https://arxiv.org/pdf/2508.01916)  

**Abstract**: Understanding internal representations of neural models is a core interest of mechanistic interpretability. Due to its large dimensionality, the representation space can encode various aspects about inputs. To what extent are different aspects organized and encoded in separate subspaces? Is it possible to find these ``natural'' subspaces in a purely unsupervised way? Somewhat surprisingly, we can indeed achieve this and find interpretable subspaces by a seemingly unrelated training objective. Our method, neighbor distance minimization (NDM), learns non-basis-aligned subspaces in an unsupervised manner. Qualitative analysis shows subspaces are interpretable in many cases, and encoded information in obtained subspaces tends to share the same abstract concept across different inputs, making such subspaces similar to ``variables'' used by the model. We also conduct quantitative experiments using known circuits in GPT-2; results show a strong connection between subspaces and circuit variables. We also provide evidence showing scalability to 2B models by finding separate subspaces mediating context and parametric knowledge routing. Viewed more broadly, our findings offer a new perspective on understanding model internals and building circuits. 

---
# A Decentralized Framework for Ethical Authorship Validation in Academic Publishing: Leveraging Self-Sovereign Identity and Blockchain Technology 

**Authors**: Kamal Al-Sabahi, Yousuf Khamis Al Mabsali  

**Link**: [PDF](https://arxiv.org/pdf/2508.01913)  

**Abstract**: Academic publishing, integral to knowledge dissemination and scientific advancement, increasingly faces threats from unethical practices such as unconsented authorship, gift authorship, author ambiguity, and undisclosed conflicts of interest. While existing infrastructures like ORCID effectively disambiguate researcher identities, they fall short in enforcing explicit authorship consent, accurately verifying contributor roles, and robustly detecting conflicts of interest during peer review. To address these shortcomings, this paper introduces a decentralized framework leveraging Self-Sovereign Identity (SSI) and blockchain technology. The proposed model uses Decentralized Identifiers (DIDs) and Verifiable Credentials (VCs) to securely verify author identities and contributions, reducing ambiguity and ensuring accurate attribution. A blockchain-based trust registry records authorship consent and peer-review activity immutably. Privacy-preserving cryptographic techniques, especially Zero-Knowledge Proofs (ZKPs), support conflict-of-interest detection without revealing sensitive data. Verified authorship metadata and consent records are embedded in publications, increasing transparency. A stakeholder survey of researchers, editors, and reviewers suggests the framework improves ethical compliance and confidence in scholarly communication. This work represents a step toward a more transparent, accountable, and trustworthy academic publishing ecosystem. 

---
# Revisiting Replay and Gradient Alignment for Continual Pre-Training of Large Language Models 

**Authors**: Istabrak Abbes, Gopeshh Subbaraj, Matthew Riemer, Nizar Islah, Benjamin Therien, Tsuguchika Tabaru, Hiroaki Kingetsu, Sarath Chandar, Irina Rish  

**Link**: [PDF](https://arxiv.org/pdf/2508.01908)  

**Abstract**: Training large language models (LLMs) typically involves pre-training on massive corpora, only to restart the process entirely when new data becomes available. A more efficient and resource-conserving approach would be continual pre-training, where models are updated with new data rather than retraining from scratch. However, the introduction of new data often causes distribution shifts, leading to performance degradation on previously learned tasks. In this paper, we take a deeper look at two popular proposals for addressing this distribution shift within the continual learning literature: experience replay and gradient alignment. We consider continual pre-training of models within the Llama family of architectures at a large scale across languages with 100 billion tokens of training data in each language, finding that both replay and gradient alignment lead to more stable learning without forgetting. This conclusion holds both as we vary the model scale and as we vary the number and diversity of tasks. Moreover, we are the first to demonstrate the effectiveness of gradient alignment techniques in the context of LLM pre-training and propose an efficient implementation of meta-experience replay (MER) that imbues experience replay with the benefits of gradient alignment despite negligible compute and memory overhead. Our scaling analysis across model sizes and replay rates indicates that small rates of replaying old examples are definitely a more valuable use of compute than investing in model size, but that it is more compute efficient to scale the size of the model than invest in high rates of replaying old examples. 

---
# Complete Evasion, Zero Modification: PDF Attacks on AI Text Detection 

**Authors**: Aldan Creo  

**Link**: [PDF](https://arxiv.org/pdf/2508.01887)  

**Abstract**: AI-generated text detectors have become essential tools for maintaining content authenticity, yet their robustness against evasion attacks remains questionable. We present PDFuzz, a novel attack that exploits the discrepancy between visual text layout and extraction order in PDF documents. Our method preserves exact textual content while manipulating character positioning to scramble extraction sequences. We evaluate this approach against the ArguGPT detector using a dataset of human and AI-generated text. Our results demonstrate complete evasion: detector performance drops from (93.6 $\pm$ 1.4) % accuracy and 0.938 $\pm$ 0.014 F1 score to random-level performance ((50.4 $\pm$ 3.2) % accuracy, 0.0 F1 score) while maintaining perfect visual fidelity. Our work reveals a vulnerability in current detection systems that is inherent to PDF document structures and underscores the need for implementing sturdy safeguards against such attacks. We make our code publicly available at this https URL. 

---
# CSLRConformer: A Data-Centric Conformer Approach for Continuous Arabic Sign Language Recognition on the Isharah Datase 

**Authors**: Fatimah Mohamed Emad Elden  

**Link**: [PDF](https://arxiv.org/pdf/2508.01791)  

**Abstract**: The field of Continuous Sign Language Recognition (CSLR) poses substantial technical challenges, including fluid inter-sign transitions, the absence of temporal boundaries, and co-articulation effects. This paper, developed for the MSLR 2025 Workshop Challenge at ICCV 2025, addresses the critical challenge of signer-independent recognition to advance the generalization capabilities of CSLR systems across diverse signers. A data-centric methodology is proposed, centered on systematic feature engineering, a robust preprocessing pipeline, and an optimized model architecture. Key contributions include a principled feature selection process guided by Exploratory Data Analysis (EDA) to isolate communicative keypoints, a rigorous preprocessing pipeline incorporating DBSCAN-based outlier filtering and spatial normalization, and the novel CSLRConformer architecture. This architecture adapts the hybrid CNN-Transformer design of the Conformer model, leveraging its capacity to model local temporal dependencies and global sequence context; a characteristic uniquely suited for the spatio-temporal dynamics of sign language. The proposed methodology achieved a competitive performance, with a Word Error Rate (WER) of 5.60% on the development set and 12.01% on the test set, a result that secured a 3rd place ranking on the official competition platform. This research validates the efficacy of cross-domain architectural adaptation, demonstrating that the Conformer model, originally conceived for speech recognition, can be successfully repurposed to establish a new state-of-the-art performance in keypoint-based CSLR. 

---
# LiveMCPBench: Can Agents Navigate an Ocean of MCP Tools? 

**Authors**: Guozhao Mo, Wenliang Zhong, Jiawei Chen, Xuanang Chen, Yaojie Lu, Hongyu Lin, Ben He, Xianpei Han, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.01780)  

**Abstract**: With the rapid development of Model Context Protocol (MCP), the number of MCP servers has surpassed 10,000. However, existing MCP benchmarks are limited to single-server settings with only a few tools, hindering effective evaluation of agent capabilities in large-scale, real-world scenarios. To address this limitation, we present LiveMCPBench, the first comprehensive benchmark comprising 95 real-world tasks grounded in the MCP ecosystem, designed to evaluate LLM agents at scale across diverse servers. To support a scalable and reproducible evaluation pipeline in large-scale MCP environments, we curate LiveMCPTool, a diverse and readily deployable collection of 70 MCP servers and 527 tools. Furthermore, we introduce LiveMCPEval, an LLM-as-a-Judge framework that enables automated and adaptive evaluation in dynamic, time-varying task environments, achieving 81% agreement with human reviewers. Finally, we propose the MCP Copilot Agent, a multi-step agent that routes tools for dynamic planning and executes tools for API interaction across the entire LiveMCPTool suite. Our evaluation covers 10 leading models, with the best-performing model (Claude-Sonnet-4) reaching a 78.95% success rate. However, we observe large performance variance across models, and several widely-used models perform poorly in LiveMCPBench's complex, tool-rich environments. Overall, LiveMCPBench offers the first unified framework for benchmarking LLM agents in realistic, tool-rich, and dynamic MCP environments, laying a solid foundation for scalable and reproducible research on agent capabilities. Our code and data will be publicly available at this https URL. 

---
# Uncertainty-Based Methods for Automated Process Reward Data Construction and Output Aggregation in Mathematical Reasoning 

**Authors**: Jiuzhou Han, Wray Buntine, Ehsan Shareghi  

**Link**: [PDF](https://arxiv.org/pdf/2508.01773)  

**Abstract**: Large language models have demonstrated remarkable capabilities in complex mathematical reasoning tasks, but they inevitably generate errors throughout multi-step solutions. Process-level Reward Models (PRMs) have shown great promise by providing supervision and evaluation at each intermediate step, thereby effectively improving the models' reasoning abilities. However, training effective PRMs requires high-quality process reward data, yet existing methods for constructing such data are often labour-intensive or inefficient. In this paper, we propose an uncertainty-driven framework for automated process reward data construction, encompassing both data generation and annotation processes for PRMs. Additionally, we identify the limitations of both majority vote and PRMs, and introduce two generic uncertainty-aware output aggregation methods: Hybrid Majority Reward Vote and Weighted Reward Frequency Vote, which combine the strengths of majority vote with PRMs. Extensive experiments on ProcessBench, MATH, and GSMPlus show the effectiveness and efficiency of the proposed PRM data construction framework, and demonstrate that the two output aggregation methods further improve the mathematical reasoning abilities across diverse PRMs. The code and data will be publicly available at this https URL. 

---
# Voxlect: A Speech Foundation Model Benchmark for Modeling Dialects and Regional Languages Around the Globe 

**Authors**: Tiantian Feng, Kevin Huang, Anfeng Xu, Xuan Shi, Thanathai Lertpetchpun, Jihwan Lee, Yoonjeong Lee, Dani Byrd, Shrikanth Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2508.01691)  

**Abstract**: We present Voxlect, a novel benchmark for modeling dialects and regional languages worldwide using speech foundation models. Specifically, we report comprehensive benchmark evaluations on dialects and regional language varieties in English, Arabic, Mandarin and Cantonese, Tibetan, Indic languages, Thai, Spanish, French, German, Brazilian Portuguese, and Italian. Our study used over 2 million training utterances from 30 publicly available speech corpora that are provided with dialectal information. We evaluate the performance of several widely used speech foundation models in classifying speech dialects. We assess the robustness of the dialectal models under noisy conditions and present an error analysis that highlights modeling results aligned with geographic continuity. In addition to benchmarking dialect classification, we demonstrate several downstream applications enabled by Voxlect. Specifically, we show that Voxlect can be applied to augment existing speech recognition datasets with dialect information, enabling a more detailed analysis of ASR performance across dialectal variations. Voxlect is also used as a tool to evaluate the performance of speech generation systems. Voxlect is publicly available with the license of the RAIL family at: this https URL. 

---
# DUP: Detection-guided Unlearning for Backdoor Purification in Language Models 

**Authors**: Man Hu, Yahui Ding, Yatao Yang, Liangyu Chen, Yanhao Jia, Shuai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.01647)  

**Abstract**: As backdoor attacks become more stealthy and robust, they reveal critical weaknesses in current defense strategies: detection methods often rely on coarse-grained feature statistics, and purification methods typically require full retraining or additional clean models. To address these challenges, we propose DUP (Detection-guided Unlearning for Purification), a unified framework that integrates backdoor detection with unlearning-based purification. The detector captures feature-level anomalies by jointly leveraging class-agnostic distances and inter-layer transitions. These deviations are integrated through a weighted scheme to identify poisoned inputs, enabling more fine-grained analysis. Based on the detection results, we purify the model through a parameter-efficient unlearning mechanism that avoids full retraining and does not require any external clean model. Specifically, we innovatively repurpose knowledge distillation to guide the student model toward increasing its output divergence from the teacher on detected poisoned samples, effectively forcing it to unlearn the backdoor behavior. Extensive experiments across diverse attack methods and language model architectures demonstrate that DUP achieves superior defense performance in detection accuracy and purification efficacy. Our code is available at this https URL. 

---
# ChEmbed: Enhancing Chemical Literature Search Through Domain-Specific Text Embeddings 

**Authors**: Ali Shiraee Kasmaee, Mohammad Khodadad, Mehdi Astaraki, Mohammad Arshi Saloot, Nicholas Sherck, Hamidreza Mahyar, Soheila Samiee  

**Link**: [PDF](https://arxiv.org/pdf/2508.01643)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems in chemistry heavily depend on accurate and relevant retrieval of chemical literature. However, general-purpose text embedding models frequently fail to adequately represent complex chemical terminologies, resulting in suboptimal retrieval quality. Specialized embedding models tailored to chemical literature retrieval have not yet been developed, leaving a substantial performance gap. To address this challenge, we introduce ChEmbed, a domain-adapted family of text embedding models fine-tuned on a dataset comprising chemistry-specific text from the PubChem, Semantic Scholar, and ChemRxiv corpora. To create effective training data, we employ large language models to synthetically generate queries, resulting in approximately 1.7 million high-quality query-passage pairs. Additionally, we augment the tokenizer by adding 900 chemically specialized tokens to previously unused slots, which significantly reduces the fragmentation of chemical entities, such as IUPAC names. ChEmbed also maintains a 8192-token context length, enabling the efficient retrieval of longer passages compared to many other open-source embedding models, which typically have a context length of 512 or 2048 tokens. Evaluated on our newly introduced ChemRxiv Retrieval benchmark, ChEmbed outperforms state-of-the-art general embedding models, raising nDCG@10 from 0.82 to 0.91 (+9 pp). ChEmbed represents a practical, lightweight, and reproducible embedding solution that effectively improves retrieval for chemical literature search. 

---
# ConfGuard: A Simple and Effective Backdoor Detection for Large Language Models 

**Authors**: Zihan Wang, Rui Zhang, Hongwei Li, Wenshu Fan, Wenbo Jiang, Qingchuan Zhao, Guowen Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01365)  

**Abstract**: Backdoor attacks pose a significant threat to Large Language Models (LLMs), where adversaries can embed hidden triggers to manipulate LLM's outputs. Most existing defense methods, primarily designed for classification tasks, are ineffective against the autoregressive nature and vast output space of LLMs, thereby suffering from poor performance and high latency. To address these limitations, we investigate the behavioral discrepancies between benign and backdoored LLMs in output space. We identify a critical phenomenon which we term sequence lock: a backdoored model generates the target sequence with abnormally high and consistent confidence compared to benign generation. Building on this insight, we propose ConfGuard, a lightweight and effective detection method that monitors a sliding window of token confidences to identify sequence lock. Extensive experiments demonstrate ConfGuard achieves a near 100\% true positive rate (TPR) and a negligible false positive rate (FPR) in the vast majority of cases. Crucially, the ConfGuard enables real-time detection almost without additional latency, making it a practical backdoor defense for real-world LLM deployments. 

---
# Multi-TW: Benchmarking Multimodal Models on Traditional Chinese Question Answering in Taiwan 

**Authors**: Jui-Ming Yao, Bing-Cheng Xie, Sheng-Wei Peng, Hao-Yuan Chen, He-Rong Zheng, Bing-Jia Tan, Peter Shaojui Wang, Shun-Feng Su  

**Link**: [PDF](https://arxiv.org/pdf/2508.01274)  

**Abstract**: Multimodal Large Language Models (MLLMs) process visual, acoustic, and textual inputs, addressing the limitations of single-modality LLMs. However, existing benchmarks often overlook tri-modal evaluation in Traditional Chinese and do not consider inference latency. To address this, we introduce Multi-TW, the first Traditional Chinese benchmark for evaluating the performance and latency of any-to-any multimodal models. Multi-TW includes 900 multiple-choice questions (image and text, audio and text pairs) sourced from official proficiency tests developed with the Steering Committee for the Test of Proficiency-Huayu (SC-TOP). We evaluated various any-to-any models and vision-language models (VLMs) with audio transcription. Our results show that closed-source models generally outperform open-source ones across modalities, although open-source models can perform well in audio tasks. End-to-end any-to-any pipelines offer clear latency advantages compared to VLMs using separate audio transcription. Multi-TW presents a comprehensive view of model capabilities and highlights the need for Traditional Chinese fine-tuning and efficient multimodal architectures. 

---
# AgentArmor: Enforcing Program Analysis on Agent Runtime Trace to Defend Against Prompt Injection 

**Authors**: Peiran Wang, Yang Liu, Yunfei Lu, Yifeng Cai, Hongbo Chen, Qingyou Yang, Jie Zhang, Jue Hong, Ye Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01249)  

**Abstract**: Large Language Model (LLM) agents offer a powerful new paradigm for solving various problems by combining natural language reasoning with the execution of external tools. However, their dynamic and non-transparent behavior introduces critical security risks, particularly in the presence of prompt injection attacks. In this work, we propose a novel insight that treats the agent runtime traces as structured programs with analyzable semantics. Thus, we present AgentArmor, a program analysis framework that converts agent traces into graph intermediate representation-based structured program dependency representations (e.g., CFG, DFG, and PDG) and enforces security policies via a type system. AgentArmor consists of three key components: (1) a graph constructor that reconstructs the agent's working traces as graph-based intermediate representations with control flow and data flow described within; (2) a property registry that attaches security-relevant metadata of interacted tools & data, and (3) a type system that performs static inference and checking over the intermediate representation. By representing agent behavior as structured programs, AgentArmor enables program analysis over sensitive data flow, trust boundaries, and policy violations. We evaluate AgentArmor on the AgentDojo benchmark, the results show that AgentArmor can achieve 95.75% of TPR, with only 3.66% of FPR. Our results demonstrate AgentArmor's ability to detect prompt injection vulnerabilities and enforce fine-grained security constraints. 

---
# Is Chain-of-Thought Reasoning of LLMs a Mirage? A Data Distribution Lens 

**Authors**: Chengshuai Zhao, Zhen Tan, Pingchuan Ma, Dawei Li, Bohan Jiang, Yancheng Wang, Yingzhen Yang, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01191)  

**Abstract**: Chain-of-Thought (CoT) prompting has been shown to improve Large Language Model (LLM) performance on various tasks. With this approach, LLMs appear to produce human-like reasoning steps before providing answers (a.k.a., CoT reasoning), which often leads to the perception that they engage in deliberate inferential processes. However, some initial findings suggest that CoT reasoning may be more superficial than it appears, motivating us to explore further. In this paper, we study CoT reasoning via a data distribution lens and investigate if CoT reasoning reflects a structured inductive bias learned from in-distribution data, allowing the model to conditionally generate reasoning paths that approximate those seen during training. Thus, its effectiveness is fundamentally bounded by the degree of distribution discrepancy between the training data and the test queries. With this lens, we dissect CoT reasoning via three dimensions: task, length, and format. To investigate each dimension, we design DataAlchemy, an isolated and controlled environment to train LLMs from scratch and systematically probe them under various distribution conditions. Our results reveal that CoT reasoning is a brittle mirage that vanishes when it is pushed beyond training distributions. This work offers a deeper understanding of why and when CoT reasoning fails, emphasizing the ongoing challenge of achieving genuine and generalizable reasoning. 

---
# DBAIOps: A Reasoning LLM-Enhanced Database Operation and Maintenance System using Knowledge Graphs 

**Authors**: Wei Zhou, Peng Sun, Xuanhe Zhou, Qianglei Zang, Ji Xu, Tieying Zhang, Guoliang Li, Fan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01136)  

**Abstract**: The operation and maintenance (O&M) of database systems is critical to ensuring system availability and performance, typically requiring expert experience (e.g., identifying metric-to-anomaly relations) for effective diagnosis and recovery. However, existing automatic database O&M methods, including commercial products, cannot effectively utilize expert experience. On the one hand, rule-based methods only support basic O&M tasks (e.g., metric-based anomaly detection), which are mostly numerical equations and cannot effectively incorporate literal O&M experience (e.g., troubleshooting guidance in manuals). On the other hand, LLM-based methods, which retrieve fragmented information (e.g., standard documents + RAG), often generate inaccurate or generic results. To address these limitations, we present DBAIOps, a novel hybrid database O&M system that combines reasoning LLMs with knowledge graphs to achieve DBA-style diagnosis. First, DBAIOps introduces a heterogeneous graph model for representing the diagnosis experience, and proposes a semi-automatic graph construction algorithm to build that graph from thousands of documents. Second, DBAIOps develops a collection of (800+) reusable anomaly models that identify both directly alerted metrics and implicitly correlated experience and metrics. Third, for each anomaly, DBAIOps proposes a two-stage graph evolution mechanism to explore relevant diagnosis paths and identify missing relations automatically. It then leverages a reasoning LLM (e.g., DeepSeek-R1) to infer root causes and generate clear diagnosis reports for both DBAs and common users. Our evaluation over four mainstream database systems (Oracle, MySQL, PostgreSQL, and DM8) demonstrates that DBAIOps outperforms state-of-the-art baselines, 34.85% and 47.22% higher in root cause and human evaluation accuracy, respectively. 

---
# CADDesigner: Conceptual Design of CAD Models Based on General-Purpose Agent 

**Authors**: Jingzhe Ni, Xiaolong Yin, Xintong Li, Xingyu Lu, Ji Wei, Ruofeng Tong, Min Tang, Peng Du  

**Link**: [PDF](https://arxiv.org/pdf/2508.01031)  

**Abstract**: Computer-Aided Design (CAD) plays a pivotal role in industrial manufacturing but typically requires a high level of expertise from designers. To lower the entry barrier and improve design efficiency, we present an agent for CAD conceptual design powered by large language models (LLMs). The agent accepts both abstract textual descriptions and freehand sketches as input, engaging in interactive dialogue with users to refine and clarify design requirements through comprehensive requirement analysis. Built upon a novel Context-Independent Imperative Paradigm (CIP), the agent generates high-quality CAD modeling code. During the generation process, the agent incorporates iterative visual feedback to improve model quality. Generated design cases are stored in a structured knowledge base, enabling continuous improvement of the agent's code generation capabilities. Experimental results demonstrate that our method achieves state-of-the-art performance in CAD code generation. 

---
# Small sample-based adaptive text classification through iterative and contrastive description refinement 

**Authors**: Amrit Rajeev, Udayaadithya Avadhanam, Harshula Tulapurkar, SaiBarath Sundar  

**Link**: [PDF](https://arxiv.org/pdf/2508.00957)  

**Abstract**: Zero-shot text classification remains a difficult task in domains with evolving knowledge and ambiguous category boundaries, such as ticketing systems. Large language models (LLMs) often struggle to generalize in these scenarios due to limited topic separability, while few-shot methods are constrained by insufficient data diversity. We propose a classification framework that combines iterative topic refinement, contrastive prompting, and active learning. Starting with a small set of labeled samples, the model generates initial topic labels. Misclassified or ambiguous samples are then used in an iterative contrastive prompting process to refine category distinctions by explicitly teaching the model to differentiate between closely related classes. The framework features a human-in-the-loop component, allowing users to introduce or revise category definitions in natural language. This enables seamless integration of new, unseen categories without retraining, making the system well-suited for real-world, dynamic environments. The evaluations on AGNews and DBpedia demonstrate strong performance: 91% accuracy on AGNews (3 seen, 1 unseen class) and 84% on DBpedia (8 seen, 1 unseen), with minimal accuracy shift after introducing unseen classes (82% and 87%, respectively). The results highlight the effectiveness of prompt-based semantic reasoning for fine-grained classification with limited supervision. 

---
# Cyber-Zero: Training Cybersecurity Agents without Runtime 

**Authors**: Terry Yue Zhuo, Dingmin Wang, Hantian Ding, Varun Kumar, Zijian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00910)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success in software engineering tasks when trained with executable runtime environments, particularly in resolving GitHub issues. However, such runtime environments are often unavailable in other domains, especially cybersecurity, where challenge configurations and execution contexts are ephemeral or restricted. We present Cyber-Zero, the first runtime-free framework for synthesizing high-quality agent trajectories to train cybersecurity LLMs. Cyber-Zero leverages publicly available CTF writeups and employs persona-driven LLM simulation to reverse-engineer runtime behaviors and generate realistic, long-horizon interaction sequences without actual environments. Using trajectories synthesized by Cyber-Zero, we train LLM-based agents that achieve up to 13.1% absolute performance gains over baseline models on three prominent CTF benchmarks: InterCode-CTF, NYU CTF Bench, and Cybench. Our best model, Cyber-Zero-32B, establishes new state-of-the-art performance among open-weight models, matching the capabilities of proprietary systems like DeepSeek-V3-0324 and Claude-3.5-Sonnet while offering superior cost-effectiveness, and demonstrating that runtime-free trajectory synthesis can effectively democratize the development of state-of-the-art cybersecurity agents. 

---
# An analysis of AI Decision under Risk: Prospect theory emerges in Large Language Models 

**Authors**: Kenneth Payne  

**Link**: [PDF](https://arxiv.org/pdf/2508.00902)  

**Abstract**: Judgment of risk is key to decision-making under uncertainty. As Daniel Kahneman and Amos Tversky famously discovered, humans do so in a distinctive way that departs from mathematical rationalism. Specifically, they demonstrated experimentally that humans accept more risk when they feel themselves at risk of losing something than when they might gain. I report the first tests of Kahneman and Tversky's landmark 'prospect theory' with Large Language Models, including today's state of the art chain-of-thought 'reasoners'.
In common with humans, I find that prospect theory often anticipates how these models approach risky decisions across a range of scenarios. I also demonstrate that context is key to explaining much of the variance in risk appetite. The 'frame' through which risk is apprehended appears to be embedded within the language of the scenarios tackled by the models. Specifically, I find that military scenarios generate far larger 'framing effects' than do civilian settings, ceteris paribus. My research suggests, therefore, that language models the world, capturing our human heuristics and biases. But also that these biases are uneven - the idea of a 'frame' is richer than simple gains and losses. Wittgenstein's notion of 'language games' explains the contingent, localised biases activated by these scenarios. Finally, I use my findings to reframe the ongoing debate about reasoning and memorisation in LLMs. 

---
# Filtering with Self-Attention and Storing with MLP: One-Layer Transformers Can Provably Acquire and Extract Knowledge 

**Authors**: Ruichen Xu, Kexin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.00901)  

**Abstract**: Modern large language models excel in knowledge-intensive tasks, yet how transformers acquire (store) knowledge during pre-training and extract (retrieve) it during post-fine-tuning inference remains theoretically opaque. While prior theoretical work has begun to investigate these questions through the analysis of training dynamics, such studies are limited to single-layer, attention-only architectures. However, most existing studies suggest that MLPs are the most contributing components for storing knowledge in transformer-based language models. Meanwhile, our empirical investigations reveal that such simplified models, when trained using standard next-token prediction objectives, may be incapable of acquiring or extracting factual knowledge. To overcome this limitation, we introduce a tractable one-layer transformer framework that crucially incorporates both self-attention and MLP modules. By tracking its gradient dynamics, we establish convergence and generalization guarantees that illuminate the ability of knowledge acquisition and extraction. We prove that 1) Transformers can achieve near-optimal training loss during pre-training, signifying effective knowledge acquisition; 2) With a large fine-tuning dataset and specific data multiplicity conditions met, transformers can achieve low generalization error when tested on factual knowledge learned during pre-training but not reinforced during the fine-tuning, indicating successful knowledge extraction; 3) When the conditions are not satisfied, transformers exhibit high generalization loss, resulting in hallucinations. Our analysis includes both full fine-tuning and low-rank fine-tuning. Furthermore, our analysis offers theoretical insights into several pertinent empirical phenomena, such as the role of learning rate schedules. Experiments on synthetic and real-world PopQA datasets with GPT-2 and Llama-3.2-1B validate our results. 

---
# AgentTTS: Large Language Model Agent for Test-time Compute-optimal Scaling Strategy in Complex Tasks 

**Authors**: Fali Wang, Hui Liu, Zhenwei Dai, Jingying Zeng, Zhiwei Zhang, Zongyu Wu, Chen Luo, Zhen Li, Xianfeng Tang, Qi He, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00890)  

**Abstract**: Test-time scaling (TTS) enhances the performance of large language models (LLMs) by allocating additional compute resources during inference. However, existing research primarily investigates TTS in single-stage tasks; while many real-world problems are multi-stage complex tasks, composed of a sequence of heterogeneous subtasks with each subtask requires LLM of specific capability. Therefore, we study a novel problem: the test-time compute-optimal scaling in multi-stage complex tasks, aiming to select suitable models and allocate budgets per subtask to maximize overall performance. TTS in multi-stage tasks introduces two fundamental challenges: (i) The combinatorial search space of model and budget allocations, combined with the high cost of inference, makes brute-force search impractical. (ii) The optimal model and budget allocations across subtasks are interdependent, increasing the complexity of the compute-optimal search. To address this gap, we conduct extensive pilot experiments on four tasks across six datasets, deriving three empirical insights characterizing the behavior of LLMs in multi-stage complex tasks. Informed by these insights, we propose AgentTTS, an LLM-agent-based framework that autonomously searches for compute-optimal allocations through iterative feedback-driven interactions with the execution environment. Experimental results demonstrate that AgentTTS significantly outperforms traditional and other LLM-based baselines in search efficiency, and shows improved robustness to varying training set sizes and enhanced interpretability. 

---
# Hallucination Detection and Mitigation with Diffusion in Multi-Variate Time-Series Foundation Models 

**Authors**: Vijja Wichitwechkarn, Charles Fox, Ruchi Choudhary  

**Link**: [PDF](https://arxiv.org/pdf/2508.00881)  

**Abstract**: Foundation models for natural language processing have many coherent definitions of hallucination and methods for its detection and mitigation. However, analogous definitions and methods do not exist for multi-variate time-series (MVTS) foundation models. We propose new definitions for MVTS hallucination, along with new detection and mitigation methods using a diffusion model to estimate hallucination levels. We derive relational datasets from popular time-series datasets to benchmark these relational hallucination levels. Using these definitions and models, we find that open-source pre-trained MVTS imputation foundation models relationally hallucinate on average up to 59.5% as much as a weak baseline. The proposed mitigation method reduces this by up to 47.7% for these models. The definition and methods may improve adoption and safe usage of MVTS foundation models. 

---
# The Attribution Crisis in LLM Search Results 

**Authors**: Ilan Strauss, Jangho Yang, Tim O'Reilly, Sruly Rosenblat, Isobel Moure  

**Link**: [PDF](https://arxiv.org/pdf/2508.00838)  

**Abstract**: Web-enabled LLMs frequently answer queries without crediting the web pages they consume, creating an "attribution gap" - the difference between relevant URLs read and those actually cited. Drawing on approximately 14,000 real-world LMArena conversation logs with search-enabled LLM systems, we document three exploitation patterns: 1) No Search: 34% of Google Gemini and 24% of OpenAI GPT-4o responses are generated without explicitly fetching any online content; 2) No citation: Gemini provides no clickable citation source in 92% of answers; 3) High-volume, low-credit: Perplexity's Sonar visits approximately 10 relevant pages per query but cites only three to four. A negative binomial hurdle model shows that the average query answered by Gemini or Sonar leaves about 3 relevant websites uncited, whereas GPT-4o's tiny uncited gap is best explained by its selective log disclosures rather than by better attribution. Citation efficiency - extra citations provided per additional relevant web page visited - varies widely across models, from 0.19 to 0.45 on identical queries, underscoring that retrieval design, not technical limits, shapes ecosystem impact. We recommend a transparent LLM search architecture based on standardized telemetry and full disclosure of search traces and citation logs. 

---
# Contextual Phenotyping of Pediatric Sepsis Cohort Using Large Language Models 

**Authors**: Aditya Nagori, Ayush Gautam, Matthew O. Wiens, Vuong Nguyen, Nathan Kenya Mugisha, Jerome Kabakyenga, Niranjan Kissoon, John Mark Ansermino, Rishikesan Kamaleswaran  

**Link**: [PDF](https://arxiv.org/pdf/2505.09805)  

**Abstract**: Clustering patient subgroups is essential for personalized care and efficient resource use. Traditional clustering methods struggle with high-dimensional, heterogeneous healthcare data and lack contextual understanding. This study evaluates Large Language Model (LLM) based clustering against classical methods using a pediatric sepsis dataset from a low-income country (LIC), containing 2,686 records with 28 numerical and 119 categorical variables. Patient records were serialized into text with and without a clustering objective. Embeddings were generated using quantized LLAMA 3.1 8B, DeepSeek-R1-Distill-Llama-8B with low-rank adaptation(LoRA), and Stella-En-400M-V5 models. K-means clustering was applied to these embeddings. Classical comparisons included K-Medoids clustering on UMAP and FAMD-reduced mixed data. Silhouette scores and statistical tests evaluated cluster quality and distinctiveness. Stella-En-400M-V5 achieved the highest Silhouette Score (0.86). LLAMA 3.1 8B with the clustering objective performed better with higher number of clusters, identifying subgroups with distinct nutritional, clinical, and socioeconomic profiles. LLM-based methods outperformed classical techniques by capturing richer context and prioritizing key features. These results highlight potential of LLMs for contextual phenotyping and informed decision-making in resource-limited settings. 

---
