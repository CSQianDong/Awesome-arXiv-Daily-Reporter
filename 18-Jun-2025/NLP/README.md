# A Variational Framework for Improving Naturalness in Generative Spoken Language Models 

**Authors**: Li-Wei Chen, Takuya Higuchi, Zakaria Aldeneh, Ahmed Hussen Abdelaziz, Alexander Rudnicky  

**Link**: [PDF](https://arxiv.org/pdf/2506.14767)  

**Abstract**: The success of large language models in text processing has inspired their adaptation to speech modeling. However, since speech is continuous and complex, it is often discretized for autoregressive modeling. Speech tokens derived from self-supervised models (known as semantic tokens) typically focus on the linguistic aspects of speech but neglect prosodic information. As a result, models trained on these tokens can generate speech with reduced naturalness. Existing approaches try to fix this by adding pitch features to the semantic tokens. However, pitch alone cannot fully represent the range of paralinguistic attributes, and selecting the right features requires careful hand-engineering. To overcome this, we propose an end-to-end variational approach that automatically learns to encode these continuous speech attributes to enhance the semantic tokens. Our approach eliminates the need for manual extraction and selection of paralinguistic features. Moreover, it produces preferred speech continuations according to human raters. Code, samples and models are available at this https URL. 

---
# From Bytes to Ideas: Language Modeling with Autoregressive U-Nets 

**Authors**: Mathurin Videau, Badr Youbi Idrissi, Alessandro Leite, Marc Schoenauer, Olivier Teytaud, David Lopez-Paz  

**Link**: [PDF](https://arxiv.org/pdf/2506.14761)  

**Abstract**: Tokenization imposes a fixed granularity on the input text, freezing how a language model operates on data and how far in the future it predicts. Byte Pair Encoding (BPE) and similar schemes split text once, build a static vocabulary, and leave the model stuck with that choice. We relax this rigidity by introducing an autoregressive U-Net that learns to embed its own tokens as it trains. The network reads raw bytes, pools them into words, then pairs of words, then up to 4 words, giving it a multi-scale view of the sequence. At deeper stages, the model must predict further into the future -- anticipating the next few words rather than the next byte -- so deeper stages focus on broader semantic patterns while earlier stages handle fine details. When carefully tuning and controlling pretraining compute, shallow hierarchies tie strong BPE baselines, and deeper hierarchies have a promising trend. Because tokenization now lives inside the model, the same system can handle character-level tasks and carry knowledge across low-resource languages. 

---
# Reasoning with Exploration: An Entropy Perspective 

**Authors**: Daixuan Cheng, Shaohan Huang, Xuekai Zhu, Bo Dai, Wayne Xin Zhao, Zhenliang Zhang, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2506.14758)  

**Abstract**: Balancing exploration and exploitation is a central goal in reinforcement learning (RL). Despite recent advances in enhancing language model (LM) reasoning, most methods lean toward exploitation, and increasingly encounter performance plateaus. In this work, we revisit entropy -- a signal of exploration in RL -- and examine its relationship to exploratory reasoning in LMs. Through empirical analysis, we uncover strong positive correlations between high-entropy regions and three types of exploratory reasoning actions: (1) pivotal tokens that determine or connect logical steps, (2) reflective actions such as self-verification and correction, and (3) rare behaviors under-explored by the base LMs. Motivated by this, we introduce a minimal modification to standard RL with only one line of code: augmenting the advantage function with an entropy-based term. Unlike traditional maximum-entropy methods which encourage exploration by promoting uncertainty, we encourage exploration by promoting longer and deeper reasoning chains. Notably, our method achieves significant gains on the Pass@K metric -- an upper-bound estimator of LM reasoning capabilities -- even when evaluated with extremely large K values, pushing the boundaries of LM reasoning. 

---
# Ring-lite: Scalable Reasoning via C3PO-Stabilized Reinforcement Learning for LLMs 

**Authors**: Ring Team, Bin Hu, Cai Chen, Deng Zhao, Ding Liu, Dingnan Jin, Feng Zhu, Hao Dai, Hongzhi Luan, Jia Guo, Jiaming Liu, Jiewei Wu, Jun Mei, Jun Zhou, Junbo Zhao, Junwu Xiong, Kaihong Zhang, Kuan Xu, Lei Liang, Liang Jiang, Liangcheng Fu, Longfei Zheng, Qiang Gao, Qing Cui, Quan Wan, Shaomian Zheng, Shuaicheng Li, Tongkai Yang, Wang Ren, Xiaodong Yan, Xiaopei Wan, Xiaoyun Feng, Xin Zhao, Xinxing Yang, Xinyu Kong, Xuemin Yang, Yang Li, Yingting Wu, Yongkang Liu, Zhankai Xu, Zhenduo Zhang, Zhenglei Zhou, Zhenyu Huang, Zhiqiang Zhang, Zihao Wang, Zujie Wen  

**Link**: [PDF](https://arxiv.org/pdf/2506.14731)  

**Abstract**: We present Ring-lite, a Mixture-of-Experts (MoE)-based large language model optimized via reinforcement learning (RL) to achieve efficient and robust reasoning capabilities. Built upon the publicly available Ling-lite model, a 16.8 billion parameter model with 2.75 billion activated parameters, our approach matches the performance of state-of-the-art (SOTA) small-scale reasoning models on challenging benchmarks (e.g., AIME, LiveCodeBench, GPQA-Diamond) while activating only one-third of the parameters required by comparable models. To accomplish this, we introduce a joint training pipeline integrating distillation with RL, revealing undocumented challenges in MoE RL training. First, we identify optimization instability during RL training, and we propose Constrained Contextual Computation Policy Optimization(C3PO), a novel approach that enhances training stability and improves computational throughput via algorithm-system co-design methodology. Second, we empirically demonstrate that selecting distillation checkpoints based on entropy loss for RL training, rather than validation metrics, yields superior performance-efficiency trade-offs in subsequent RL training. Finally, we develop a two-stage training paradigm to harmonize multi-domain data integration, addressing domain conflicts that arise in training with mixed dataset. We will release the model, dataset, and code. 

---
# Capacity Matters: a Proof-of-Concept for Transformer Memorization on Real-World Data 

**Authors**: Anton Changalidis, Aki Härmä  

**Link**: [PDF](https://arxiv.org/pdf/2506.14704)  

**Abstract**: This paper studies how the model architecture and data configurations influence the empirical memorization capacity of generative transformers. The models are trained using synthetic text datasets derived from the Systematized Nomenclature of Medicine (SNOMED) knowledge graph: triplets, representing static connections, and sequences, simulating complex relation patterns. The results show that embedding size is the primary determinant of learning speed and capacity, while additional layers provide limited benefits and may hinder performance on simpler datasets. Activation functions play a crucial role, and Softmax demonstrates greater stability and capacity. Furthermore, increasing the complexity of the data set seems to improve the final memorization. These insights improve our understanding of transformer memory mechanisms and provide a framework for optimizing model design with structured real-world data. 

---
# Treasure Hunt: Real-time Targeting of the Long Tail using Training-Time Markers 

**Authors**: Daniel D'souza, Julia Kreutzer, Adrien Morisot, Ahmet Üstün, Sara Hooker  

**Link**: [PDF](https://arxiv.org/pdf/2506.14702)  

**Abstract**: One of the most profound challenges of modern machine learning is performing well on the long-tail of rare and underrepresented features. Large general-purpose models are trained for many tasks, but work best on high-frequency use cases. After training, it is hard to adapt a model to perform well on specific use cases underrepresented in the training corpus. Relying on prompt engineering or few-shot examples to maximize the output quality on a particular test case can be frustrating, as models can be highly sensitive to small changes, react in unpredicted ways or rely on a fixed system prompt for maintaining performance. In this work, we ask: "Can we optimize our training protocols to both improve controllability and performance on underrepresented use cases at inference time?" We revisit the divide between training and inference techniques to improve long-tail performance while providing users with a set of control levers the model is trained to be responsive to. We create a detailed taxonomy of data characteristics and task provenance to explicitly control generation attributes and implicitly condition generations at inference time. We fine-tune a base model to infer these markers automatically, which makes them optional at inference time. This principled and flexible approach yields pronounced improvements in performance, especially on examples from the long tail of the training distribution. While we observe an average lift of 5.7% win rates in open-ended generation quality with our markers, we see over 9.1% gains in underrepresented domains. We also observe relative lifts of up to 14.1% on underrepresented tasks like CodeRepair and absolute improvements of 35.3% on length instruction following evaluations. 

---
# Massive Supervised Fine-tuning Experiments Reveal How Data, Layer, and Training Factors Shape LLM Alignment Quality 

**Authors**: Yuto Harada, Yusuke Yamauchi, Yusuke Oda, Yohei Oseki, Yusuke Miyao, Yu Takagi  

**Link**: [PDF](https://arxiv.org/pdf/2506.14681)  

**Abstract**: Supervised fine-tuning (SFT) is a critical step in aligning large language models (LLMs) with human instructions and values, yet many aspects of SFT remain poorly understood. We trained a wide range of base models on a variety of datasets including code generation, mathematical reasoning, and general-domain tasks, resulting in 1,000+ SFT models under controlled conditions. We then identified the dataset properties that matter most and examined the layer-wise modifications introduced by SFT. Our findings reveal that some training-task synergies persist across all models while others vary substantially, emphasizing the importance of model-specific strategies. Moreover, we demonstrate that perplexity consistently predicts SFT effectiveness--often surpassing superficial similarity between trained data and benchmark--and that mid-layer weight changes correlate most strongly with performance gains. We will release these 1,000+ SFT models and benchmark results to accelerate further research. 

---
# GuiLoMo: Allocating Expert Number and Rank for LoRA-MoE via Bilevel Optimization with GuidedSelection Vectors 

**Authors**: Hengyuan Zhang, Xinrong Chen, Yingmin Qiu, Xiao Liang, Ziyue Li, Guanyu Wang, Weiping Li, Tong Mo, Wenyue Li, Hayden Kwok-Hay So, Ngai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2506.14646)  

**Abstract**: Parameter-efficient fine-tuning (PEFT) methods, particularly Low-Rank Adaptation (LoRA), offer an efficient way to adapt large language models with reduced computational costs. However, their performance is limited by the small number of trainable parameters. Recent work combines LoRA with the Mixture-of-Experts (MoE), i.e., LoRA-MoE, to enhance capacity, but two limitations remain in hindering the full exploitation of its potential: 1) the influence of downstream tasks when assigning expert numbers, and 2) the uniform rank assignment across all LoRA experts, which restricts representational diversity. To mitigate these gaps, we propose GuiLoMo, a fine-grained layer-wise expert numbers and ranks allocation strategy with GuidedSelection Vectors (GSVs). GSVs are learned via a prior bilevel optimization process to capture both model- and task-specific needs, and are then used to allocate optimal expert numbers and ranks. Experiments on three backbone models across diverse benchmarks show that GuiLoMo consistently achieves superior or comparable performance to all baselines. Further analysis offers key insights into how expert numbers and ranks vary across layers and tasks, highlighting the benefits of adaptive expert configuration. Our code is available at this https URL. 

---
# Passing the Turing Test in Political Discourse: Fine-Tuning LLMs to Mimic Polarized Social Media Comments 

**Authors**: . Pazzaglia, V. Vendetti, L. D. Comencini, F. Deriu, V. Modugno  

**Link**: [PDF](https://arxiv.org/pdf/2506.14645)  

**Abstract**: The increasing sophistication of large language models (LLMs) has sparked growing concerns regarding their potential role in exacerbating ideological polarization through the automated generation of persuasive and biased content. This study explores the extent to which fine-tuned LLMs can replicate and amplify polarizing discourse within online environments. Using a curated dataset of politically charged discussions extracted from Reddit, we fine-tune an open-source LLM to produce context-aware and ideologically aligned responses. The model's outputs are evaluated through linguistic analysis, sentiment scoring, and human annotation, with particular attention to credibility and rhetorical alignment with the original discourse. The results indicate that, when trained on partisan data, LLMs are capable of producing highly plausible and provocative comments, often indistinguishable from those written by humans. These findings raise significant ethical questions about the use of AI in political discourse, disinformation, and manipulation campaigns. The paper concludes with a discussion of the broader implications for AI governance, platform regulation, and the development of detection tools to mitigate adversarial fine-tuning risks. 

---
# Revisiting Chain-of-Thought Prompting: Zero-shot Can Be Stronger than Few-shot 

**Authors**: Xiang Cheng, Chengyan Pan, Minjun Zhao, Deyang Li, Fangchao Liu, Xinyu Zhang, Xiao Zhang, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.14641)  

**Abstract**: In-Context Learning (ICL) is an essential emergent ability of Large Language Models (LLMs), and recent studies introduce Chain-of-Thought (CoT) to exemplars of ICL to enhance the reasoning capability, especially in mathematics tasks. However, given the continuous advancement of model capabilities, it remains unclear whether CoT exemplars still benefit recent, stronger models in such tasks. Through systematic experiments, we find that for recent strong models such as the Qwen2.5 series, adding traditional CoT exemplars does not improve reasoning performance compared to Zero-Shot CoT. Instead, their primary function is to align the output format with human expectations. We further investigate the effectiveness of enhanced CoT exemplars, constructed using answers from advanced models such as \texttt{Qwen2.5-Max} and \texttt{DeepSeek-R1}. Experimental results indicate that these enhanced exemplars still fail to improve the model's reasoning performance. Further analysis reveals that models tend to ignore the exemplars and focus primarily on the instructions, leading to no observable gain in reasoning ability. Overall, our findings highlight the limitations of the current ICL+CoT framework in mathematical reasoning, calling for a re-examination of the ICL paradigm and the definition of exemplars. 

---
# AIn't Nothing But a Survey? Using Large Language Models for Coding German Open-Ended Survey Responses on Survey Motivation 

**Authors**: Leah von der Heyde, Anna-Carolina Haensch, Bernd Weiß, Jessika Daikeler  

**Link**: [PDF](https://arxiv.org/pdf/2506.14634)  

**Abstract**: The recent development and wider accessibility of LLMs have spurred discussions about how they can be used in survey research, including classifying open-ended survey responses. Due to their linguistic capacities, it is possible that LLMs are an efficient alternative to time-consuming manual coding and the pre-training of supervised machine learning models. As most existing research on this topic has focused on English-language responses relating to non-complex topics or on single LLMs, it is unclear whether its findings generalize and how the quality of these classifications compares to established methods. In this study, we investigate to what extent different LLMs can be used to code open-ended survey responses in other contexts, using German data on reasons for survey participation as an example. We compare several state-of-the-art LLMs and several prompting approaches, and evaluate the LLMs' performance by using human expert codings. Overall performance differs greatly between LLMs, and only a fine-tuned LLM achieves satisfactory levels of predictive performance. Performance differences between prompting approaches are conditional on the LLM used. Finally, LLMs' unequal classification performance across different categories of reasons for survey participation results in different categorical distributions when not using fine-tuning. We discuss the implications of these findings, both for methodological research on coding open-ended responses and for their substantive analysis, and for practitioners processing or substantively analyzing such data. Finally, we highlight the many trade-offs researchers need to consider when choosing automated methods for open-ended response classification in the age of LLMs. In doing so, our study contributes to the growing body of research about the conditions under which LLMs can be efficiently, accurately, and reliably leveraged in survey research. 

---
# Probabilistic Aggregation and Targeted Embedding Optimization for Collective Moral Reasoning in Large Language Models 

**Authors**: Chenchen Yuan, Zheyu Zhang, Shuo Yang, Bardh Prenkaj, Gjergji Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2506.14625)  

**Abstract**: Large Language Models (LLMs) have shown impressive moral reasoning abilities. Yet they often diverge when confronted with complex, multi-factor moral dilemmas. To address these discrepancies, we propose a framework that synthesizes multiple LLMs' moral judgments into a collectively formulated moral judgment, realigning models that deviate significantly from this consensus. Our aggregation mechanism fuses continuous moral acceptability scores (beyond binary labels) into a collective probability, weighting contributions by model reliability. For misaligned models, a targeted embedding-optimization procedure fine-tunes token embeddings for moral philosophical theories, minimizing JS divergence to the consensus while preserving semantic integrity. Experiments on a large-scale social moral dilemma dataset show our approach builds robust consensus and improves individual model fidelity. These findings highlight the value of data-driven moral alignment across multiple models and its potential for safer, more consistent AI systems. 

---
# When Does Meaning Backfire? Investigating the Role of AMRs in NLI 

**Authors**: Junghyun Min, Xiulin Yang, Shira Wein  

**Link**: [PDF](https://arxiv.org/pdf/2506.14613)  

**Abstract**: Natural Language Inference (NLI) relies heavily on adequately parsing the semantic content of the premise and hypothesis. In this work, we investigate whether adding semantic information in the form of an Abstract Meaning Representation (AMR) helps pretrained language models better generalize in NLI. Our experiments integrating AMR into NLI in both fine-tuning and prompting settings show that the presence of AMR in fine-tuning hinders model generalization while prompting with AMR leads to slight gains in \texttt{GPT-4o}. However, an ablation study reveals that the improvement comes from amplifying surface-level differences rather than aiding semantic reasoning. This amplification can mislead models to predict non-entailment even when the core meaning is preserved. 

---
# Guaranteed Guess: A Language Modeling Approach for CISC-to-RISC Transpilation with Testing Guarantees 

**Authors**: Ahmed Heakl, Sarim Hashmi, Chaimaa Abi, Celine Lee, Abdulrahman Mahmoud  

**Link**: [PDF](https://arxiv.org/pdf/2506.14606)  

**Abstract**: The hardware ecosystem is rapidly evolving, with increasing interest in translating low-level programs across different instruction set architectures (ISAs) in a quick, flexible, and correct way to enhance the portability and longevity of existing code. A particularly challenging class of this transpilation problem is translating between complex- (CISC) and reduced- (RISC) hardware architectures, due to fundamental differences in instruction complexity, memory models, and execution paradigms. In this work, we introduce GG (Guaranteed Guess), an ISA-centric transpilation pipeline that combines the translation power of pre-trained large language models (LLMs) with the rigor of established software testing constructs. Our method generates candidate translations using an LLM from one ISA to another, and embeds such translations within a software-testing framework to build quantifiable confidence in the translation. We evaluate our GG approach over two diverse datasets, enforce high code coverage (>98%) across unit tests, and achieve functional/semantic correctness of 99% on HumanEval programs and 49% on BringupBench programs, respectively. Further, we compare our approach to the state-of-the-art Rosetta 2 framework on Apple Silicon, showcasing 1.73x faster runtime performance, 1.47x better energy efficiency, and 2.41x better memory usage for our transpiled code, demonstrating the effectiveness of GG for real-world CISC-to-RISC translation tasks. We will open-source our codes, data, models, and benchmarks to establish a common foundation for ISA-level code translation research. 

---
# GenerationPrograms: Fine-grained Attribution with Executable Programs 

**Authors**: David Wan, Eran Hirsch, Elias Stengel-Eskin, Ido Dagan, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2506.14580)  

**Abstract**: Recent large language models (LLMs) achieve impressive performance in source-conditioned text generation but often fail to correctly provide fine-grained attributions for their outputs, undermining verifiability and trust. Moreover, existing attribution methods do not explain how and why models leverage the provided source documents to generate their final responses, limiting interpretability. To overcome these challenges, we introduce a modular generation framework, GenerationPrograms, inspired by recent advancements in executable "code agent" architectures. Unlike conventional generation methods that simultaneously generate outputs and attributions or rely on post-hoc attribution, GenerationPrograms decomposes the process into two distinct stages: first, creating an executable program plan composed of modular text operations (such as paraphrasing, compression, and fusion) explicitly tailored to the query, and second, executing these operations following the program's specified instructions to produce the final response. Empirical evaluations demonstrate that GenerationPrograms significantly improves attribution quality at both the document level and sentence level across two long-form question-answering tasks and a multi-document summarization task. We further demonstrate that GenerationPrograms can effectively function as a post-hoc attribution method, outperforming traditional techniques in recovering accurate attributions. In addition, the interpretable programs generated by GenerationPrograms enable localized refinement through modular-level improvements that further enhance overall attribution quality. 

---
# AlphaDecay:Module-wise Weight Decay for Heavy-Tailed Balancing in LLMs 

**Authors**: Di He, Ajay Jaiswal, Songjun Tu, Li Shen, Ganzhao Yuan, Shiwei Liu, Lu Yin  

**Link**: [PDF](https://arxiv.org/pdf/2506.14562)  

**Abstract**: Weight decay is a standard regularization technique for training large language models (LLMs). While it is common to assign a uniform decay rate to every layer, this approach overlooks the structural diversity of LLMs and the varying spectral properties across modules. In this paper, we introduce AlphaDecay, a simple yet effective method that adaptively assigns different weight decay strengths to each module of an LLM. Our approach is guided by Heavy-Tailed Self-Regularization (HT-SR) theory, which analyzes the empirical spectral density (ESD) of weight correlation matrices to quantify "heavy-tailedness." Modules exhibiting more pronounced heavy-tailed ESDs, reflecting stronger feature learning, are assigned weaker decay, while modules with lighter-tailed spectra receive stronger decay. Our method leverages tailored weight decay assignments to balance the module-wise differences in spectral properties, leading to improved performance. Extensive pre-training tasks with various model sizes from 60M to 1B demonstrate that AlphaDecay achieves better perplexity and generalization than conventional uniform decay and other adaptive decay baselines. 

---
# M2BeamLLM: Multimodal Sensing-empowered mmWave Beam Prediction with Large Language Models 

**Authors**: Can Zheng, Jiguang He, Chung G. Kang, Guofa Cai, Zitong Yu, Merouane Debbah  

**Link**: [PDF](https://arxiv.org/pdf/2506.14532)  

**Abstract**: This paper introduces a novel neural network framework called M2BeamLLM for beam prediction in millimeter-wave (mmWave) massive multi-input multi-output (mMIMO) communication systems. M2BeamLLM integrates multi-modal sensor data, including images, radar, LiDAR, and GPS, leveraging the powerful reasoning capabilities of large language models (LLMs) such as GPT-2 for beam prediction. By combining sensing data encoding, multimodal alignment and fusion, and supervised fine-tuning (SFT), M2BeamLLM achieves significantly higher beam prediction accuracy and robustness, demonstrably outperforming traditional deep learning (DL) models in both standard and few-shot scenarios. Furthermore, its prediction performance consistently improves with increased diversity in sensing modalities. Our study provides an efficient and intelligent beam prediction solution for vehicle-to-infrastructure (V2I) mmWave communication systems. 

---
# LingoLoop Attack: Trapping MLLMs via Linguistic Context and State Entrapment into Endless Loops 

**Authors**: Jiyuan Fu, Kaixun Jiang, Lingyi Hong, Jinglun Li, Haijing Guo, Dingkang Yang, Zhaoyu Chen, Wenqiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14493)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown great promise but require substantial computational resources during inference. Attackers can exploit this by inducing excessive output, leading to resource exhaustion and service degradation. Prior energy-latency attacks aim to increase generation time by broadly shifting the output token distribution away from the EOS token, but they neglect the influence of token-level Part-of-Speech (POS) characteristics on EOS and sentence-level structural patterns on output counts, limiting their efficacy. To address this, we propose LingoLoop, an attack designed to induce MLLMs to generate excessively verbose and repetitive sequences. First, we find that the POS tag of a token strongly affects the likelihood of generating an EOS token. Based on this insight, we propose a POS-Aware Delay Mechanism to postpone EOS token generation by adjusting attention weights guided by POS information. Second, we identify that constraining output diversity to induce repetitive loops is effective for sustained generation. We introduce a Generative Path Pruning Mechanism that limits the magnitude of hidden states, encouraging the model to produce persistent loops. Extensive experiments demonstrate LingoLoop can increase generated tokens by up to 30 times and energy consumption by a comparable factor on models like Qwen2.5-VL-3B, consistently driving MLLMs towards their maximum generation limits. These findings expose significant MLLMs' vulnerabilities, posing challenges for their reliable deployment. The code will be released publicly following the paper's acceptance. 

---
# LexiMark: Robust Watermarking via Lexical Substitutions to Enhance Membership Verification of an LLM's Textual Training Data 

**Authors**: Eyal German, Sagiv Antebi, Edan Habler, Asaf Shabtai, Yuval Elovici  

**Link**: [PDF](https://arxiv.org/pdf/2506.14474)  

**Abstract**: Large language models (LLMs) can be trained or fine-tuned on data obtained without the owner's consent. Verifying whether a specific LLM was trained on particular data instances or an entire dataset is extremely challenging. Dataset watermarking addresses this by embedding identifiable modifications in training data to detect unauthorized use. However, existing methods often lack stealth, making them relatively easy to detect and remove. In light of these limitations, we propose LexiMark, a novel watermarking technique designed for text and documents, which embeds synonym substitutions for carefully selected high-entropy words. Our method aims to enhance an LLM's memorization capabilities on the watermarked text without altering the semantic integrity of the text. As a result, the watermark is difficult to detect, blending seamlessly into the text with no visible markers, and is resistant to removal due to its subtle, contextually appropriate substitutions that evade automated and manual detection. We evaluated our method using baseline datasets from recent studies and seven open-source models: LLaMA-1 7B, LLaMA-3 8B, Mistral 7B, Pythia 6.9B, as well as three smaller variants from the Pythia family (160M, 410M, and 1B). Our evaluation spans multiple training settings, including continued pretraining and fine-tuning scenarios. The results demonstrate significant improvements in AUROC scores compared to existing methods, underscoring our method's effectiveness in reliably verifying whether unauthorized watermarked data was used in LLM training. 

---
# How Far Can LLMs Improve from Experience? Measuring Test-Time Learning Ability in LLMs with Human Comparison 

**Authors**: Jiayin Wang, Zhiquang Guo, Weizhi Ma, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14448)  

**Abstract**: As evaluation designs of large language models may shape our trajectory toward artificial general intelligence, comprehensive and forward-looking assessment is essential. Existing benchmarks primarily assess static knowledge, while intelligence also entails the ability to rapidly learn from experience. To this end, we advocate for the evaluation of Test-time Learning, the capacity to improve performance in experience-based, reasoning-intensive tasks during test time. In this work, we propose semantic games as effective testbeds for evaluating test-time learning, due to their resistance to saturation and inherent demand for strategic reasoning. We introduce an objective evaluation framework that compares model performance under both limited and cumulative experience settings, and contains four forms of experience representation. To provide a comparative baseline, we recruit eight human participants to complete the same task. Results show that LLMs exhibit measurable test-time learning capabilities; however, their improvements are less stable under cumulative experience and progress more slowly than those observed in humans. These findings underscore the potential of LLMs as general-purpose learning machines, while also revealing a substantial intellectual gap between models and humans, irrespective of how well LLMs perform on static benchmarks. 

---
# LongLLaDA: Unlocking Long Context Capabilities in Diffusion LLMs 

**Authors**: Xiaoran Liu, Zhigeng Liu, Zengfeng Huang, Qipeng Guo, Ziwei He, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2506.14429)  

**Abstract**: Large Language Diffusion Models, or diffusion LLMs, have emerged as a significant focus in NLP research, with substantial effort directed toward understanding their scalability and downstream task performance. However, their long-context capabilities remain unexplored, lacking systematic analysis or methods for context extension. In this work, we present the first systematic investigation comparing the long-context performance of diffusion LLMs and traditional auto-regressive LLMs. We first identify a unique characteristic of diffusion LLMs, unlike auto-regressive LLMs, they maintain remarkably \textbf{\textit{stable perplexity}} during direct context extrapolation. Furthermore, where auto-regressive models fail outright during the Needle-In-A-Haystack task with context exceeding their pretrained length, we discover diffusion LLMs exhibit a distinct \textbf{\textit{local perception}} phenomenon, enabling successful retrieval from recent context segments. We explain both phenomena through the lens of Rotary Position Embedding (RoPE) scaling theory. Building on these observations, we propose LongLLaDA, a training-free method that integrates LLaDA with the NTK-based RoPE extrapolation. Our results validate that established extrapolation scaling laws remain effective for extending the context windows of diffusion LLMs. Furthermore, we identify long-context tasks where diffusion LLMs outperform auto-regressive LLMs and others where they fall short. Consequently, this study establishes the first context extrapolation method for diffusion LLMs while providing essential theoretical insights and empirical benchmarks critical for advancing future research on long-context diffusion LLMs. 

---
# ImpliRet: Benchmarking the Implicit Fact Retrieval Challenge 

**Authors**: Zeinab Sadat Taghavi, Ali Modarressi, Yunpu Ma, Hinrich Schütze  

**Link**: [PDF](https://arxiv.org/pdf/2506.14407)  

**Abstract**: Retrieval systems are central to many NLP pipelines, but often rely on surface-level cues such as keyword overlap and lexical semantic similarity. To evaluate retrieval beyond these shallow signals, recent benchmarks introduce reasoning-heavy queries; however, they primarily shift the burden to query-side processing techniques -- like prompting or multi-hop retrieval -- that can help resolve complexity. In contrast, we present ImpliRet, a benchmark that shifts the reasoning challenge to document-side processing: The queries are simple, but relevance depends on facts stated implicitly in documents through temporal (e.g., resolving "two days ago"), arithmetic, and world knowledge relationships. We evaluate a range of sparse and dense retrievers, all of which struggle in this setting: the best nDCG@10 is only 15.07%. We also test whether long-context models can overcome this limitation. But even with a short context of only ten documents, including the positive document, GPT-4.1 scores only 35.06%, showing that document-side reasoning remains a challenge. Our codes are available at this http URL. 

---
# Thunder-NUBench: A Benchmark for LLMs' Sentence-Level Negation Understanding 

**Authors**: Yeonkyoung So, Gyuseong Lee, Sungmok Jung, Joonhak Lee, JiA Kang, Sangho Kim, Jaejin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.14397)  

**Abstract**: Negation is a fundamental linguistic phenomenon that poses persistent challenges for Large Language Models (LLMs), particularly in tasks requiring deep semantic understanding. Existing benchmarks often treat negation as a side case within broader tasks like natural language inference, resulting in a lack of benchmarks that exclusively target negation understanding. In this work, we introduce \textbf{Thunder-NUBench}, a novel benchmark explicitly designed to assess sentence-level negation understanding in LLMs. Thunder-NUBench goes beyond surface-level cue detection by contrasting standard negation with structurally diverse alternatives such as local negation, contradiction, and paraphrase. The benchmark consists of manually curated sentence-negation pairs and a multiple-choice dataset that enables in-depth evaluation of models' negation understanding. 

---
# ELLIS Alicante at CQs-Gen 2025: Winning the critical thinking questions shared task: LLM-based question generation and selection 

**Authors**: Lucile Favero, Daniel Frases, Juan Antonio Pérez-Ortiz, Tanja Käser, Nuria Oliver  

**Link**: [PDF](https://arxiv.org/pdf/2506.14371)  

**Abstract**: The widespread adoption of chat interfaces based on Large Language Models (LLMs) raises concerns about promoting superficial learning and undermining the development of critical thinking skills. Instead of relying on LLMs purely for retrieving factual information, this work explores their potential to foster deeper reasoning by generating critical questions that challenge unsupported or vague claims in debate interventions. This study is part of a shared task of the 12th Workshop on Argument Mining, co-located with ACL 2025, focused on automatic critical question generation. We propose a two-step framework involving two small-scale open source language models: a Questioner that generates multiple candidate questions and a Judge that selects the most relevant ones. Our system ranked first in the shared task competition, demonstrating the potential of the proposed LLM-based approach to encourage critical engagement with argumentative texts. 

---
# Digital Gatekeepers: Google's Role in Curating Hashtags and Subreddits 

**Authors**: Amrit Poudel, Yifan Ding, Jurgen Pfeffer, Tim Weninger  

**Link**: [PDF](https://arxiv.org/pdf/2506.14370)  

**Abstract**: Search engines play a crucial role as digital gatekeepers, shaping the visibility of Web and social media content through algorithmic curation. This study investigates how search engines like Google selectively promotes or suppresses certain hashtags and subreddits, impacting the information users encounter. By comparing search engine results with nonsampled data from Reddit and Twitter/X, we reveal systematic biases in content visibility. Google's algorithms tend to suppress subreddits and hashtags related to sexually explicit material, conspiracy theories, advertisements, and cryptocurrencies, while promoting content associated with higher engagement. These findings suggest that Google's gatekeeping practices influence public discourse by curating the social media narratives available to users. 

---
# A Vision for Geo-Temporal Deep Research Systems: Towards Comprehensive, Transparent, and Reproducible Geo-Temporal Information Synthesis 

**Authors**: Bruno Martins, Piotr Szymański, Piotr Gramacki  

**Link**: [PDF](https://arxiv.org/pdf/2506.14345)  

**Abstract**: The emergence of Large Language Models (LLMs) has transformed information access, with current LLMs also powering deep research systems that can generate comprehensive report-style answers, through planned iterative search, retrieval, and reasoning. Still, current deep research systems lack the geo-temporal capabilities that are essential for answering context-rich questions involving geographic and/or temporal constraints, frequently occurring in domains like public health, environmental science, or socio-economic analysis. This paper reports our vision towards next generation systems, identifying important technical, infrastructural, and evaluative challenges in integrating geo-temporal reasoning into deep research pipelines. We argue for augmenting retrieval and synthesis processes with the ability to handle geo-temporal constraints, supported by open and reproducible infrastructures and rigorous evaluation protocols. Our vision outlines a path towards more advanced and geo-temporally aware deep research systems, of potential impact to the future of AI-driven information access. 

---
# Evaluation Should Not Ignore Variation: On the Impact of Reference Set Choice on Summarization Metrics 

**Authors**: Silvia Casola, Yang Janet Liu, Siyao Peng, Oliver Kraus, Albert Gatt, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2506.14335)  

**Abstract**: Human language production exhibits remarkable richness and variation, reflecting diverse communication styles and intents. However, this variation is often overlooked in summarization evaluation. While having multiple reference summaries is known to improve correlation with human judgments, the impact of using different reference sets on reference-based metrics has not been systematically investigated. This work examines the sensitivity of widely used reference-based metrics in relation to the choice of reference sets, analyzing three diverse multi-reference summarization datasets: SummEval, GUMSum, and DUC2004. We demonstrate that many popular metrics exhibit significant instability. This instability is particularly concerning for n-gram-based metrics like ROUGE, where model rankings vary depending on the reference sets, undermining the reliability of model comparisons. We also collect human judgments on LLM outputs for genre-diverse data and examine their correlation with metrics to supplement existing findings beyond newswire summaries, finding weak-to-no correlation. Taken together, we recommend incorporating reference set variation into summarization evaluation to enhance consistency alongside correlation with human judgments, especially when evaluating LLMs. 

---
# Expectation Confirmation Preference Optimization for Multi-Turn Conversational Recommendation Agent 

**Authors**: Xueyang Feng, Jingsen Zhang, Jiakai Tang, Wei Li, Guohao Cai, Xu Chen, Quanyu Dai, Yue Zhu, Zhenhua Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.14302)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have significantly propelled the development of Conversational Recommendation Agents (CRAs). However, these agents often generate short-sighted responses that fail to sustain user guidance and meet expectations. Although preference optimization has proven effective in aligning LLMs with user expectations, it remains costly and performs poorly in multi-turn dialogue. To address this challenge, we introduce a novel multi-turn preference optimization (MTPO) paradigm ECPO, which leverages Expectation Confirmation Theory to explicitly model the evolution of user satisfaction throughout multi-turn dialogues, uncovering the underlying causes of dissatisfaction. These causes can be utilized to support targeted optimization of unsatisfactory responses, thereby achieving turn-level preference optimization. ECPO ingeniously eliminates the significant sampling overhead of existing MTPO methods while ensuring the optimization process drives meaningful improvements. To support ECPO, we introduce an LLM-based user simulator, AILO, to simulate user feedback and perform expectation confirmation during conversational recommendations. Experimental results show that ECPO significantly enhances CRA's interaction capabilities, delivering notable improvements in both efficiency and effectiveness over existing MTPO methods. 

---
# From What to Respond to When to Respond: Timely Response Generation for Open-domain Dialogue Agents 

**Authors**: Seongbo Jang, Minjin Jeon, Jaehoon Lee, Seonghyeon Lee, Dongha Lee, Hwanjo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.14285)  

**Abstract**: While research on dialogue response generation has primarily focused on generating coherent responses conditioning on textual context, the critical question of when to respond grounded on the temporal context remains underexplored. To bridge this gap, we propose a novel task called timely dialogue response generation and introduce the TimelyChat benchmark, which evaluates the capabilities of language models to predict appropriate time intervals and generate time-conditioned responses. Additionally, we construct a large-scale training dataset by leveraging unlabeled event knowledge from a temporal commonsense knowledge graph and employing a large language model (LLM) to synthesize 55K event-driven dialogues. We then train Timer, a dialogue agent designed to proactively predict time intervals and generate timely responses that align with those intervals. Experimental results show that Timer outperforms prompting-based LLMs and other fine-tuned baselines in both turn-level and dialogue-level evaluations. We publicly release our data, model, and code. 

---
# Re-Initialization Token Learning for Tool-Augmented Large Language Models 

**Authors**: Chenghao Li, Liu Liu, Baosheng Yu, Jiayan Qiu, Yibing Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2506.14248)  

**Abstract**: Large language models have demonstrated exceptional performance, yet struggle with complex tasks such as numerical reasoning, plan generation. Integrating external tools, such as calculators and databases, into large language models (LLMs) is crucial for enhancing problem-solving capabilities. Current methods assign a unique token to each tool, enabling LLMs to call tools through token prediction-similar to word generation. However, this approach fails to account for the relationship between tool and word tokens, limiting adaptability within pre-trained LLMs. To address this issue, we propose a novel token learning method that aligns tool tokens with the existing word embedding space from the perspective of initialization, thereby enhancing model performance. We begin by constructing prior token embeddings for each tool based on the tool's name or description, which are used to initialize and regularize the learnable tool token embeddings. This ensures the learned embeddings are well-aligned with the word token space, improving tool call accuracy. We evaluate the method on tasks such as numerical reasoning, knowledge-based question answering, and embodied plan generation using GSM8K-XL, FuncQA, KAMEL, and VirtualHome datasets. The results demonstrate clear improvements over recent baselines, including CoT, REACT, ICL, and ToolkenGPT, indicating that our approach effectively augments LLMs with tools through relevant tokens across diverse domains. 

---
# A Multi-Expert Structural-Semantic Hybrid Framework for Unveiling Historical Patterns in Temporal Knowledge Graphs 

**Authors**: Yimin Deng, Yuxia Wu, Yejing Wang, Guoshuai Zhao, Li Zhu, Qidong Liu, Derong Xu, Zichuan Fu, Xian Wu, Yefeng Zheng, Xiangyu Zhao, Xueming Qian  

**Link**: [PDF](https://arxiv.org/pdf/2506.14235)  

**Abstract**: Temporal knowledge graph reasoning aims to predict future events with knowledge of existing facts and plays a key role in various downstream tasks. Previous methods focused on either graph structure learning or semantic reasoning, failing to integrate dual reasoning perspectives to handle different prediction scenarios. Moreover, they lack the capability to capture the inherent differences between historical and non-historical events, which limits their generalization across different temporal contexts. To this end, we propose a Multi-Expert Structural-Semantic Hybrid (MESH) framework that employs three kinds of expert modules to integrate both structural and semantic information, guiding the reasoning process for different events. Extensive experiments on three datasets demonstrate the effectiveness of our approach. 

---
# Xolver: Multi-Agent Reasoning with Holistic Experience Learning Just Like an Olympiad Team 

**Authors**: Md Tanzib Hosain, Salman Rahman, Md Kishor Morol, Md Rizwan Parvez  

**Link**: [PDF](https://arxiv.org/pdf/2506.14234)  

**Abstract**: Despite impressive progress on complex reasoning, current large language models (LLMs) typically operate in isolation - treating each problem as an independent attempt, without accumulating or integrating experiential knowledge. In contrast, expert problem solvers - such as Olympiad or programming contest teams - leverage a rich tapestry of experiences: absorbing mentorship from coaches, developing intuition from past problems, leveraging knowledge of tool usage and library functionality, adapting strategies based on the expertise and experiences of peers, continuously refining their reasoning through trial and error, and learning from other related problems even during competition. We introduce Xolver, a training-free multi-agent reasoning framework that equips a black-box LLM with a persistent, evolving memory of holistic experience. Xolver integrates diverse experience modalities, including external and self-retrieval, tool use, collaborative interactions, agent-driven evaluation, and iterative refinement. By learning from relevant strategies, code fragments, and abstract reasoning patterns at inference time, Xolver avoids generating solutions from scratch - marking a transition from isolated inference toward experience-aware language agents. Built on both open-weight and proprietary models, Xolver consistently outperforms specialized reasoning agents. Even with lightweight backbones (e.g., QWQ-32B), it often surpasses advanced models including Qwen3-235B, Gemini 2.5 Pro, o3, and o4-mini-high. With o3-mini-high, it achieves new best results on GSM8K (98.1%), AIME'24 (94.4%), AIME'25 (93.7%), Math-500 (99.8%), and LiveCodeBench-V5 (91.6%) - highlighting holistic experience learning as a key step toward generalist agents capable of expert-level reasoning. Code and data are available at this https URL. 

---
# Chaining Event Spans for Temporal Relation Grounding 

**Authors**: Jongho Kim, Dohyeon Lee, Minsoo Kim, Seung-won Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14213)  

**Abstract**: Accurately understanding temporal relations between events is a critical building block of diverse tasks, such as temporal reading comprehension (TRC) and relation extraction (TRE). For example in TRC, we need to understand the temporal semantic differences between the following two questions that are lexically near-identical: "What finished right before the decision?" or "What finished right after the decision?". To discern the two questions, existing solutions have relied on answer overlaps as a proxy label to contrast similar and dissimilar questions. However, we claim that answer overlap can lead to unreliable results, due to spurious overlaps of two dissimilar questions with coincidentally identical answers. To address the issue, we propose a novel approach that elicits proper reasoning behaviors through a module for predicting time spans of events. We introduce the Timeline Reasoning Network (TRN) operating in a two-step inductive reasoning process: In the first step model initially answers each question with semantic and syntactic information. The next step chains multiple questions on the same event to predict a timeline, which is then used to ground the answers. Results on the TORQUE and TB-dense, TRC and TRE tasks respectively, demonstrate that TRN outperforms previous methods by effectively resolving the spurious overlaps using the predicted timeline. 

---
# Explainable Detection of Implicit Influential Patterns in Conversations via Data Augmentation 

**Authors**: Sina Abdidizaji, Md Kowsher, Niloofar Yousefi, Ivan Garibay  

**Link**: [PDF](https://arxiv.org/pdf/2506.14211)  

**Abstract**: In the era of digitalization, as individuals increasingly rely on digital platforms for communication and news consumption, various actors employ linguistic strategies to influence public perception. While models have become proficient at detecting explicit patterns, which typically appear in texts as single remarks referred to as utterances, such as social media posts, malicious actors have shifted toward utilizing implicit influential verbal patterns embedded within conversations. These verbal patterns aim to mentally penetrate the victim's mind in order to influence them, enabling the actor to obtain the desired information through implicit means. This paper presents an improved approach for detecting such implicit influential patterns. Furthermore, the proposed model is capable of identifying the specific locations of these influential elements within a conversation. To achieve this, the existing dataset was augmented using the reasoning capabilities of state-of-the-art language models. Our designed framework resulted in a 6% improvement in the detection of implicit influential patterns in conversations. Moreover, this approach improved the multi-label classification tasks related to both the techniques used for influence and the vulnerability of victims by 33% and 43%, respectively. 

---
# CausalDiffTab: Mixed-Type Causal-Aware Diffusion for Tabular Data Generation 

**Authors**: Jia-Chen Zhang, Zheng Zhou, Yu-Jie Xiong, Chun-Ming Xia, Fei Dai  

**Link**: [PDF](https://arxiv.org/pdf/2506.14206)  

**Abstract**: Training data has been proven to be one of the most critical components in training generative AI. However, obtaining high-quality data remains challenging, with data privacy issues presenting a significant hurdle. To address the need for high-quality data. Synthesize data has emerged as a mainstream solution, demonstrating impressive performance in areas such as images, audio, and video. Generating mixed-type data, especially high-quality tabular data, still faces significant challenges. These primarily include its inherent heterogeneous data types, complex inter-variable relationships, and intricate column-wise distributions. In this paper, we introduce CausalDiffTab, a diffusion model-based generative model specifically designed to handle mixed tabular data containing both numerical and categorical features, while being more flexible in capturing complex interactions among variables. We further propose a hybrid adaptive causal regularization method based on the principle of Hierarchical Prior Fusion. This approach adaptively controls the weight of causal regularization, enhancing the model's performance without compromising its generative capabilities. Comprehensive experiments conducted on seven datasets demonstrate that CausalDiffTab outperforms baseline methods across all metrics. Our code is publicly available at: this https URL. 

---
# AgentSynth: Scalable Task Generation for Generalist Computer-Use Agents 

**Authors**: Jingxu Xie, Dylan Xu, Xuandong Zhao, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.14205)  

**Abstract**: We introduce AgentSynth, a scalable and cost-efficient pipeline for automatically synthesizing high-quality tasks and trajectory datasets for generalist computer-use agents. Leveraging information asymmetry, AgentSynth constructs subtasks that are simple during generation but significantly more challenging when composed into long-horizon tasks, enabling the creation of over 6,000 diverse and realistic tasks. Our pipeline begins with an LLM-based task proposer guided by a persona, followed by an execution agent that completes the task and logs the trajectory. This process is repeated iteratively to form a sequence of subtasks, which are then summarized by a separate agent into a composite task of controllable difficulty. A key strength of AgentSynth is its ability to precisely modulate task complexity by varying the number of subtasks. Empirical evaluations show that state-of-the-art LLM agents suffer a steep performance drop, from 18% success at difficulty level 1 to just 4% at level 6, highlighting the benchmark's difficulty and discriminative power. Moreover, our pipeline achieves a low average cost of \$0.60 per trajectory, orders of magnitude cheaper than human annotations. Our code and data are publicly available at this https URL 

---
# Intended Target Identification for Anomia Patients with Gradient-based Selective Augmentation 

**Authors**: Jongho Kim, Romain Storaï, Seung-won Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14203)  

**Abstract**: In this study, we investigate the potential of language models (LMs) in aiding patients experiencing anomia, a difficulty identifying the names of items. Identifying the intended target item from patient's circumlocution involves the two challenges of term failure and error: (1) The terms relevant to identifying the item remain unseen. (2) What makes the challenge unique is inherent perturbed terms by semantic paraphasia, which are not exactly related to the target item, hindering the identification process. To address each, we propose robustifying the model from semantically paraphasic errors and enhancing the model with unseen terms with gradient-based selective augmentation. Specifically, the gradient value controls augmented data quality amid semantic errors, while the gradient variance guides the inclusion of unseen but relevant terms. Due to limited domain-specific datasets, we evaluate the model on the Tip-of-the-Tongue dataset as an intermediary task and then apply our findings to real patient data from AphasiaBank. Our results demonstrate strong performance against baselines, aiding anomia patients by addressing the outlined challenges. 

---
# ELI-Why: Evaluating the Pedagogical Utility of Language Model Explanations 

**Authors**: Brihi Joshi, Keyu He, Sahana Ramnath, Sadra Sabouri, Kaitlyn Zhou, Souti Chattopadhyay, Swabha Swayamdipta, Xiang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2506.14200)  

**Abstract**: Language models today are widely used in education, yet their ability to tailor responses for learners with varied informational needs and knowledge backgrounds remains under-explored. To this end, we introduce ELI-Why, a benchmark of 13.4K "Why" questions to evaluate the pedagogical capabilities of language models. We then conduct two extensive human studies to assess the utility of language model-generated explanatory answers (explanations) on our benchmark, tailored to three distinct educational grades: elementary, high-school and graduate school. In our first study, human raters assume the role of an "educator" to assess model explanations' fit to different educational grades. We find that GPT-4-generated explanations match their intended educational background only 50% of the time, compared to 79% for lay human-curated explanations. In our second study, human raters assume the role of a learner to assess if an explanation fits their own informational needs. Across all educational backgrounds, users deemed GPT-4-generated explanations 20% less suited on average to their informational needs, when compared to explanations curated by lay people. Additionally, automated evaluation metrics reveal that explanations generated across different language model families for different informational needs remain indistinguishable in their grade-level, limiting their pedagogical effectiveness. 

---
# MAS-LitEval : Multi-Agent System for Literary Translation Quality Assessment 

**Authors**: Junghwan Kim, Kieun Park, Sohee Park, Hyunggug Kim, Bongwon Suh  

**Link**: [PDF](https://arxiv.org/pdf/2506.14199)  

**Abstract**: Literary translation requires preserving cultural nuances and stylistic elements, which traditional metrics like BLEU and METEOR fail to assess due to their focus on lexical overlap. This oversight neglects the narrative consistency and stylistic fidelity that are crucial for literary works. To address this, we propose MAS-LitEval, a multi-agent system using Large Language Models (LLMs) to evaluate translations based on terminology, narrative, and style. We tested MAS-LitEval on translations of The Little Prince and A Connecticut Yankee in King Arthur's Court, generated by various LLMs, and compared it to traditional metrics. \textbf{MAS-LitEval} outperformed these metrics, with top models scoring up to 0.890 in capturing literary nuances. This work introduces a scalable, nuanced framework for Translation Quality Assessment (TQA), offering a practical tool for translators and researchers. 

---
# AsyncSwitch: Asynchronous Text-Speech Adaptation for Code-Switched ASR 

**Authors**: Tuan Nguyen, Huy-Dat Tran  

**Link**: [PDF](https://arxiv.org/pdf/2506.14190)  

**Abstract**: Developing code-switched ASR systems is challenging due to language ambiguity and limited exposure to multilingual, code-switched data, while collecting such speech is costly. Prior work generates synthetic audio from text, but these methods are computationally intensive and hard to scale. We introduce AsyncSwitch, a novel asynchronous adaptation framework that leverages large-scale, text-rich web data to pre-expose ASR models to diverse code-switched domains before fine-tuning on paired speech-text corpora. Our three-stage process (1) trains decoder self-attention and feedforward layers on code-switched text, (2) aligns decoder and encoder via cross-attention using limited speech-text data, and (3) fully fine-tunes the entire model. Experiments with Whisper on Malay-English code-switching demonstrate a 9.02% relative WER reduction, while improving monolingual performance in Singlish, Malay, and other English variants. 

---
# Can we train ASR systems on Code-switch without real code-switch data? Case study for Singapore's languages 

**Authors**: Tuan Nguyen, Huy-Dat Tran  

**Link**: [PDF](https://arxiv.org/pdf/2506.14177)  

**Abstract**: Code-switching (CS), common in multilingual settings, presents challenges for ASR due to scarce and costly transcribed data caused by linguistic complexity. This study investigates building CS-ASR using synthetic CS data. We propose a phrase-level mixing method to generate synthetic CS data that mimics natural patterns. Utilizing monolingual augmented with synthetic phrase-mixed CS data to fine-tune large pretrained ASR models (Whisper, MMS, SeamlessM4T). This paper focuses on three under-resourced Southeast Asian language pairs: Malay-English (BM-EN), Mandarin-Malay (ZH-BM), and Tamil-English (TA-EN), establishing a new comprehensive benchmark for CS-ASR to evaluate the performance of leading ASR models. Experimental results show that the proposed training strategy enhances ASR performance on monolingual and CS tests, with BM-EN showing highest gains, then TA-EN and ZH-BM. This finding offers a cost-effective approach for CS-ASR development, benefiting research and industry. 

---
# GRAM: A Generative Foundation Reward Model for Reward Generalization 

**Authors**: Chenglong Wang, Yang Gan, Yifu Huo, Yongyu Mu, Qiaozhi He, Murun Yang, Bei Li, Tong Xiao, Chunliang Zhang, Tongran Liu, Jingbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.14175)  

**Abstract**: In aligning large language models (LLMs), reward models have played an important role, but are standardly trained as discriminative models and rely only on labeled human preference data. In this paper, we explore methods that train reward models using both unlabeled and labeled data. Building on the generative models in LLMs, we develop a generative reward model that is first trained via large-scale unsupervised learning and then fine-tuned via supervised learning. We also show that by using label smoothing, we are in fact optimizing a regularized pairwise ranking loss. This result, in turn, provides a new view of training reward models, which links generative models and discriminative models under the same class of training objectives. The outcome of these techniques is a foundation reward model, which can be applied to a wide range of tasks with little or no further fine-tuning effort. Extensive experiments show that this model generalizes well across several tasks, including response ranking, reinforcement learning from human feedback, and task adaptation with fine-tuning, achieving significant performance improvements over several strong baseline models. 

---
# MIST: Towards Multi-dimensional Implicit Bias and Stereotype Evaluation of LLMs via Theory of Mind 

**Authors**: Yanlin Li, Hao Liu, Huimin Liu, Yinwei Wei, Yupeng Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.14161)  

**Abstract**: Theory of Mind (ToM) in Large Language Models (LLMs) refers to their capacity for reasoning about mental states, yet failures in this capacity often manifest as systematic implicit bias. Evaluating this bias is challenging, as conventional direct-query methods are susceptible to social desirability effects and fail to capture its subtle, multi-dimensional nature. To this end, we propose an evaluation framework that leverages the Stereotype Content Model (SCM) to reconceptualize bias as a multi-dimensional failure in ToM across Competence, Sociability, and Morality. The framework introduces two indirect tasks: the Word Association Bias Test (WABT) to assess implicit lexical associations and the Affective Attribution Test (AAT) to measure covert affective leanings, both designed to probe latent stereotypes without triggering model avoidance. Extensive experiments on 8 State-of-the-Art LLMs demonstrate our framework's capacity to reveal complex bias structures, including pervasive sociability bias, multi-dimensional divergence, and asymmetric stereotype amplification, thereby providing a more robust methodology for identifying the structural nature of implicit bias. 

---
# S$^4$C: Speculative Sampling with Syntactic and Semantic Coherence for Efficient Inference of Large Language Models 

**Authors**: Tao He, Guang Huang, Yu Yang, Tianshi Xu, Sicheng Zhao, Guiguang Ding, Pengyang Wang, Feng Tian  

**Link**: [PDF](https://arxiv.org/pdf/2506.14158)  

**Abstract**: Large language models (LLMs) exhibit remarkable reasoning capabilities across diverse downstream tasks. However, their autoregressive nature leads to substantial inference latency, posing challenges for real-time applications. Speculative sampling mitigates this issue by introducing a drafting phase followed by a parallel validation phase, enabling faster token generation and verification. Existing approaches, however, overlook the inherent coherence in text generation, limiting their efficiency. To address this gap, we propose a Speculative Sampling with Syntactic and Semantic Coherence (S$^4$C) framework, which extends speculative sampling by leveraging multi-head drafting for rapid token generation and a continuous verification tree for efficient candidate validation and feature reuse. Experimental results demonstrate that S$^4$C surpasses baseline methods across mainstream tasks, offering enhanced efficiency, parallelism, and the ability to generate more valid tokens with fewer computational resources. On Spec-bench benchmarks, S$^4$C achieves an acceleration ratio of 2.26x-2.60x, outperforming state-of-the-art methods. 

---
# DCRM: A Heuristic to Measure Response Pair Quality in Preference Optimization 

**Authors**: Chengyu Huang, Tanya Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2506.14157)  

**Abstract**: Recent research has attempted to associate preference optimization (PO) performance with the underlying preference datasets. In this work, our observation is that the differences between the preferred response $y^+$ and dispreferred response $y^-$ influence what LLMs can learn, which may not match the desirable differences to learn. Therefore, we use distance and reward margin to quantify these differences, and combine them to get Distance Calibrated Reward Margin (DCRM), a metric that measures the quality of a response pair for PO. Intuitively, DCRM encourages minimal noisy differences and maximal desired differences. With this, we study 3 types of commonly used preference datasets, classified along two axes: the source of the responses and the preference labeling function. We establish a general correlation between higher DCRM of the training set and better learning outcome. Inspired by this, we propose a best-of-$N^2$ pairing method that selects response pairs with the highest DCRM. Empirically, in various settings, our method produces training datasets that can further improve models' performance on AlpacaEval, MT-Bench, and Arena-Hard over the existing training sets. 

---
# Sampling from Your Language Model One Byte at a Time 

**Authors**: Jonathan Hayase, Alisa Liu, Noah A. Smith, Sewoong Oh  

**Link**: [PDF](https://arxiv.org/pdf/2506.14123)  

**Abstract**: Tokenization is used almost universally by modern language models, enabling efficient text representation using multi-byte or multi-character tokens. However, prior work has shown that tokenization can introduce distortion into the model's generations. For example, users are often advised not to end their prompts with a space because it prevents the model from including the space as part of the next token. This Prompt Boundary Problem (PBP) also arises in languages such as Chinese and in code generation, where tokens often do not line up with syntactic boundaries. Additionally mismatching tokenizers often hinder model composition and interoperability. For example, it is not possible to directly ensemble models with different tokenizers due to their mismatching vocabularies. To address these issues, we present an inference-time method to convert any autoregressive LM with a BPE tokenizer into a character-level or byte-level LM, without changing its generative distribution at the text level. Our method efficient solves the PBP and is also able to unify the vocabularies of language models with different tokenizers, allowing one to ensemble LMs with different tokenizers at inference time as well as transfer the post-training from one model to another using proxy-tuning. We demonstrate in experiments that the ensemble and proxy-tuned models outperform their constituents on downstream evals. 

---
# Essential-Web v1.0: 24T tokens of organized web data 

**Authors**: Essential AI, Andrew Hojel, Michael Pust, Tim Romanski, Yash Vanjani, Ritvik Kapila, Mohit Parmar, Adarsh Chaluvaraju, Alok Tripathy, Anil Thomas, Ashish Tanwer, Darsh J Shah, Ishaan Shah, Karl Stratos, Khoi Nguyen, Kurt Smith, Michael Callahan, Peter Rushton, Philip Monk, Platon Mazarakis, Saad Jamal, Saurabh Srivastava, Somanshu Singla, Ashish Vaswani  

**Link**: [PDF](https://arxiv.org/pdf/2506.14111)  

**Abstract**: Data plays the most prominent role in how language models acquire skills and knowledge. The lack of massive, well-organized pre-training datasets results in costly and inaccessible data pipelines. We present Essential-Web v1.0, a 24-trillion-token dataset in which every document is annotated with a twelve-category taxonomy covering topic, format, content complexity, and quality. Taxonomy labels are produced by EAI-Distill-0.5b, a fine-tuned 0.5b-parameter model that achieves an annotator agreement within 3% of Qwen2.5-32B-Instruct. With nothing more than SQL-style filters, we obtain competitive web-curated datasets in math (-8.0% relative to SOTA), web code (+14.3%), STEM (+24.5%) and medical (+8.6%). Essential-Web v1.0 is available on HuggingFace: this https URL 

---
# Abstract Meaning Representation for Hospital Discharge Summarization 

**Authors**: Paul Landes, Sitara Rao, Aaron Jeremy Chaise, Barbara Di Eugenio  

**Link**: [PDF](https://arxiv.org/pdf/2506.14101)  

**Abstract**: The Achilles heel of Large Language Models (LLMs) is hallucination, which has drastic consequences for the clinical domain. This is particularly important with regards to automatically generating discharge summaries (a lengthy medical document that summarizes a hospital in-patient visit). Automatically generating these summaries would free physicians to care for patients and reduce documentation burden. The goal of this work is to discover new methods that combine language-based graphs and deep learning models to address provenance of content and trustworthiness in automatic summarization. Our method shows impressive reliability results on the publicly available Medical Information Mart for Intensive III (MIMIC-III) corpus and clinical notes written by physicians at Anonymous Hospital. rovide our method, generated discharge ary output examples, source code and trained models. 

---
# Automatic Extraction of Clausal Embedding Based on Large-Scale English Text Data 

**Authors**: Iona Carslaw, Sivan Milton, Nicolas Navarre, Ciyang Qing, Wataru Uegaki  

**Link**: [PDF](https://arxiv.org/pdf/2506.14064)  

**Abstract**: For linguists, embedded clauses have been of special interest because of their intricate distribution of syntactic and semantic features. Yet, current research relies on schematically created language examples to investigate these constructions, missing out on statistical information and naturally-occurring examples that can be gained from large language corpora. Thus, we present a methodological approach for detecting and annotating naturally-occurring examples of English embedded clauses in large-scale text data using constituency parsing and a set of parsing heuristics. Our tool has been evaluated on our dataset Golden Embedded Clause Set (GECS), which includes hand-annotated examples of naturally-occurring English embedded clause sentences. Finally, we present a large-scale dataset of naturally-occurring English embedded clauses which we have extracted from the open-source corpus Dolma using our extraction tool. 

---
# Ace-CEFR -- A Dataset for Automated Evaluation of the Linguistic Difficulty of Conversational Texts for LLM Applications 

**Authors**: David Kogan, Max Schumacher, Sam Nguyen, Masanori Suzuki, Melissa Smith, Chloe Sophia Bellows, Jared Bernstein  

**Link**: [PDF](https://arxiv.org/pdf/2506.14046)  

**Abstract**: There is an unmet need to evaluate the language difficulty of short, conversational passages of text, particularly for training and filtering Large Language Models (LLMs). We introduce Ace-CEFR, a dataset of English conversational text passages expert-annotated with their corresponding level of text difficulty. We experiment with several models on Ace-CEFR, including Transformer-based models and LLMs. We show that models trained on Ace-CEFR can measure text difficulty more accurately than human experts and have latency appropriate to production environments. Finally, we release the Ace-CEFR dataset to the public for research and development. 

---
# An Interdisciplinary Review of Commonsense Reasoning and Intent Detection 

**Authors**: Md Nazmus Sakib  

**Link**: [PDF](https://arxiv.org/pdf/2506.14040)  

**Abstract**: This review explores recent advances in commonsense reasoning and intent detection, two key challenges in natural language understanding. We analyze 28 papers from ACL, EMNLP, and CHI (2020-2025), organizing them by methodology and application. Commonsense reasoning is reviewed across zero-shot learning, cultural adaptation, structured evaluation, and interactive contexts. Intent detection is examined through open-set models, generative formulations, clustering, and human-centered systems. By bridging insights from NLP and HCI, we highlight emerging trends toward more adaptive, multilingual, and context-aware models, and identify key gaps in grounding, generalization, and benchmark design. 

---
# MultiFinBen: A Multilingual, Multimodal, and Difficulty-Aware Benchmark for Financial LLM Evaluation 

**Authors**: Xueqing Peng, Lingfei Qian, Yan Wang, Ruoyu Xiang, Yueru He, Yang Ren, Mingyang Jiang, Jeff Zhao, Huan He, Yi Han, Yun Feng, Yuechen Jiang, Yupeng Cao, Haohang Li, Yangyang Yu, Xiaoyu Wang, Penglei Gao, Shengyuan Lin, Keyi Wang, Shanshan Yang, Yilun Zhao, Zhiwei Liu, Peng Lu, Jerry Huang, Suyuchen Wang, Triantafillos Papadopoulos, Polydoros Giannouris, Efstathia Soufleri, Nuo Chen, Guojun Xiong, Zhiyang Deng, Yijia Zhao, Mingquan Lin, Meikang Qiu, Kaleb E Smith, Arman Cohan, Xiao-Yang Liu, Jimin Huang, Alejandro Lopez-Lira, Xi Chen, Junichi Tsujii, Jian-Yun Nie, Sophia Ananiadou, Qianqian Xie  

**Link**: [PDF](https://arxiv.org/pdf/2506.14028)  

**Abstract**: Recent advances in large language models (LLMs) have accelerated progress in financial NLP and applications, yet existing benchmarks remain limited to monolingual and unimodal settings, often over-relying on simple tasks and failing to reflect the complexity of real-world financial communication. We introduce MultiFinBen, the first multilingual and multimodal benchmark tailored to the global financial domain, evaluating LLMs across modalities (text, vision, audio) and linguistic settings (monolingual, bilingual, multilingual) on domain-specific tasks. We introduce two novel tasks, including PolyFiQA-Easy and PolyFiQA-Expert, the first multilingual financial benchmarks requiring models to perform complex reasoning over mixed-language inputs; and EnglishOCR and SpanishOCR, the first OCR-embedded financial QA tasks challenging models to extract and reason over information from visual-text financial documents. Moreover, we propose a dynamic, difficulty-aware selection mechanism and curate a compact, balanced benchmark rather than simple aggregation existing datasets. Extensive evaluation of 22 state-of-the-art models reveals that even the strongest models, despite their general multimodal and multilingual capabilities, struggle dramatically when faced with complex cross-lingual and multimodal tasks in financial domain. MultiFinBen is publicly released to foster transparent, reproducible, and inclusive progress in financial studies and applications. 

---
# Lost in the Mix: Evaluating LLM Understanding of Code-Switched Text 

**Authors**: Amr Mohamed, Yang Zhang, Michalis Vazirgiannis, Guokan Shang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14012)  

**Abstract**: Code-switching (CSW) is the act of alternating between two or more languages within a single discourse. This phenomenon is widespread in multilingual communities, and increasingly prevalent in online content, where users naturally mix languages in everyday communication. As a result, Large Language Models (LLMs), now central to content processing and generation, are frequently exposed to code-switched inputs. Given their widespread use, it is crucial to understand how LLMs process and reason about such mixed-language text. This paper presents a systematic evaluation of LLM comprehension under code-switching by generating CSW variants of established reasoning and comprehension benchmarks. While degradation is evident when foreign tokens disrupt English text$\unicode{x2013}$even under linguistic constraints$\unicode{x2013}$embedding English into other languages often improves comprehension. Though prompting yields mixed results, fine-tuning offers a more stable path to degradation mitigation. 

---
# AI shares emotion with humans across languages and cultures 

**Authors**: Xiuwen Wu, Hao Wang, Zhiang Yan, Xiaohan Tang, Pengfei Xu, Wai-Ting Siok, Ping Li, Jia-Hong Gao, Bingjiang Lyu, Lang Qin  

**Link**: [PDF](https://arxiv.org/pdf/2506.13978)  

**Abstract**: Effective and safe human-machine collaboration requires the regulated and meaningful exchange of emotions between humans and artificial intelligence (AI). Current AI systems based on large language models (LLMs) can provide feedback that makes people feel heard. Yet it remains unclear whether LLMs represent emotion in language as humans do, or whether and how the emotional tone of their output can be controlled. We assess human-AI emotional alignment across linguistic-cultural groups and model-families, using interpretable LLM features translated from concept-sets for over twenty nuanced emotion categories (including six basic emotions). Our analyses reveal that LLM-derived emotion spaces are structurally congruent with human perception, underpinned by the fundamental affective dimensions of valence and arousal. Furthermore, these emotion-related features also accurately predict large-scale behavioural data on word ratings along these two core dimensions, reflecting both universal and language-specific patterns. Finally, by leveraging steering vectors derived solely from human-centric emotion concepts, we show that model expressions can be stably and naturally modulated across distinct emotion categories, which provides causal evidence that human emotion concepts can be used to systematically induce LLMs to produce corresponding affective states when conveying content. These findings suggest AI not only shares emotional representations with humans but its affective outputs can be precisely guided using psychologically grounded emotion concepts. 

---
# Are manual annotations necessary for statutory interpretations retrieval? 

**Authors**: Aleksander Smywiński-Pohl, Tomer Libal, Adam Kaczmarczyk, Magdalena Król  

**Link**: [PDF](https://arxiv.org/pdf/2506.13965)  

**Abstract**: One of the elements of legal research is looking for cases where judges have extended the meaning of a legal concept by providing interpretations of what a concept means or does not mean. This allow legal professionals to use such interpretations as precedents as well as laymen to better understand the legal concept. The state-of-the-art approach for retrieving the most relevant interpretations for these concepts currently depends on the ranking of sentences and the training of language models over annotated examples. That manual annotation process can be quite expensive and need to be repeated for each such concept, which prompted recent research in trying to automate this process. In this paper, we highlight the results of various experiments conducted to determine the volume, scope and even the need for manual annotation. First of all, we check what is the optimal number of annotations per a legal concept. Second, we check if we can draw the sentences for annotation randomly or there is a gain in the performance of the model, when only the best candidates are annotated. As the last question we check what is the outcome of automating the annotation process with the help of an LLM. 

---
# ASMR: Augmenting Life Scenario using Large Generative Models for Robotic Action Reflection 

**Authors**: Shang-Chi Tsai, Seiya Kawano, Angel Garcia Contreras, Koichiro Yoshino, Yun-Nung Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.13956)  

**Abstract**: When designing robots to assist in everyday human activities, it is crucial to enhance user requests with visual cues from their surroundings for improved intent understanding. This process is defined as a multimodal classification task. However, gathering a large-scale dataset encompassing both visual and linguistic elements for model training is challenging and time-consuming. To address this issue, our paper introduces a novel framework focusing on data augmentation in robotic assistance scenarios, encompassing both dialogues and related environmental imagery. This approach involves leveraging a sophisticated large language model to simulate potential conversations and environmental contexts, followed by the use of a stable diffusion model to create images depicting these environments. The additionally generated data serves to refine the latest multimodal models, enabling them to more accurately determine appropriate actions in response to user interactions with the limited target data. Our experimental results, based on a dataset collected from real-world scenarios, demonstrate that our methodology significantly enhances the robot's action selection capabilities, achieving the state-of-the-art performance. 

---
# Alignment Quality Index (AQI) : Beyond Refusals: AQI as an Intrinsic Alignment Diagnostic via Latent Geometry, Cluster Divergence, and Layer wise Pooled Representations 

**Authors**: Abhilekh Borah, Chhavi Sharma, Danush Khanna, Utkarsh Bhatt, Gurpreet Singh, Hasnat Md Abdullah, Raghav Kaushik Ravi, Vinija Jain, Jyoti Patel, Shubham Singh, Vasu Sharma, Arpita Vats, Rahul Raja, Aman Chadha, Amitava Das  

**Link**: [PDF](https://arxiv.org/pdf/2506.13901)  

**Abstract**: Alignment is no longer a luxury, it is a necessity. As large language models (LLMs) enter high-stakes domains like education, healthcare, governance, and law, their behavior must reliably reflect human-aligned values and safety constraints. Yet current evaluations rely heavily on behavioral proxies such as refusal rates, G-Eval scores, and toxicity classifiers, all of which have critical blind spots. Aligned models are often vulnerable to jailbreaking, stochasticity of generation, and alignment faking.
To address this issue, we introduce the Alignment Quality Index (AQI). This novel geometric and prompt-invariant metric empirically assesses LLM alignment by analyzing the separation of safe and unsafe activations in latent space. By combining measures such as the Davies-Bouldin Score (DBS), Dunn Index (DI), Xie-Beni Index (XBI), and Calinski-Harabasz Index (CHI) across various formulations, AQI captures clustering quality to detect hidden misalignments and jailbreak risks, even when outputs appear compliant. AQI also serves as an early warning signal for alignment faking, offering a robust, decoding invariant tool for behavior agnostic safety auditing.
Additionally, we propose the LITMUS dataset to facilitate robust evaluation under these challenging conditions. Empirical tests on LITMUS across different models trained under DPO, GRPO, and RLHF conditions demonstrate AQI's correlation with external judges and ability to reveal vulnerabilities missed by refusal metrics. We make our implementation publicly available to foster future research in this area. 

---
# EmoNews: A Spoken Dialogue System for Expressive News Conversations 

**Authors**: Ryuki Matsuura, Shikhar Bharadwaj, Jiarui Liu, Dhatchi Kunde Govindarajan  

**Link**: [PDF](https://arxiv.org/pdf/2506.13894)  

**Abstract**: We develop a task-oriented spoken dialogue system (SDS) that regulates emotional speech based on contextual cues to enable more empathetic news conversations. Despite advancements in emotional text-to-speech (TTS) techniques, task-oriented emotional SDSs remain underexplored due to the compartmentalized nature of SDS and emotional TTS research, as well as the lack of standardized evaluation metrics for social goals. We address these challenges by developing an emotional SDS for news conversations that utilizes a large language model (LLM)-based sentiment analyzer to identify appropriate emotions and PromptTTS to synthesize context-appropriate emotional speech. We also propose subjective evaluation scale for emotional SDSs and judge the emotion regulation performance of the proposed and baseline systems. Experiments showed that our emotional SDS outperformed a baseline system in terms of the emotion regulation and engagement. These results suggest the critical role of speech emotion for more engaging conversations. All our source code is open-sourced at this https URL 

---
# VL-GenRM: Enhancing Vision-Language Verification via Vision Experts and Iterative Training 

**Authors**: Jipeng Zhang, Kehao Miao, Renjie Pi, Zhaowei Wang, Runtao Liu, Rui Pan, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13888)  

**Abstract**: Reinforcement Fine-Tuning (RFT) with verifiable rewards has advanced large language models but remains underexplored for Vision-Language (VL) models. The Vision-Language Reward Model (VL-RM) is key to aligning VL models by providing structured feedback, yet training effective VL-RMs faces two major challenges. First, the bootstrapping dilemma arises as high-quality training data depends on already strong VL models, creating a cycle where self-generated supervision reinforces existing biases. Second, modality bias and negative example amplification occur when VL models hallucinate incorrect visual attributes, leading to flawed preference data that further misguides training. To address these issues, we propose an iterative training framework leveraging vision experts, Chain-of-Thought (CoT) rationales, and Margin-based Rejection Sampling. Our approach refines preference datasets, enhances structured critiques, and iteratively improves reasoning. Experiments across VL-RM benchmarks demonstrate superior performance in hallucination detection and multimodal reasoning, advancing VL model alignment with reinforcement learning. 

---
# Investigating the interaction of linguistic and mathematical reasoning in language models using multilingual number puzzles 

**Authors**: Antara Raaghavi Bhattacharya, Isabel Papadimitriou, Kathryn Davidson, David Alvarez-Melis  

**Link**: [PDF](https://arxiv.org/pdf/2506.13886)  

**Abstract**: Across languages, numeral systems vary widely in how they construct and combine numbers. While humans consistently learn to navigate this diversity, large language models (LLMs) struggle with linguistic-mathematical puzzles involving cross-linguistic numeral systems, which humans can learn to solve successfully. We investigate why this task is difficult for LLMs through a series of experiments that untangle the linguistic and mathematical aspects of numbers in language. Our experiments establish that models cannot consistently solve such problems unless the mathematical operations in the problems are explicitly marked using known symbols ($+$, $\times$, etc, as in "twenty + three"). In further ablation studies, we probe how individual parameters of numeral construction and combination affect performance. While humans use their linguistic understanding of numbers to make inferences about the implicit compositional structure of numerals, LLMs seem to lack this notion of implicit numeral structure. We conclude that the ability to flexibly infer compositional rules from implicit patterns in human-scale data remains an open challenge for current reasoning models. 

---
# ClimateChat: Designing Data and Methods for Instruction Tuning LLMs to Answer Climate Change Queries 

**Authors**: Zhou Chen, Xiao Wang, Yuanhong Liao, Ming Lin, Yuqi Bai  

**Link**: [PDF](https://arxiv.org/pdf/2506.13796)  

**Abstract**: As the issue of global climate change becomes increasingly severe, the demand for research in climate science continues to grow. Natural language processing technologies, represented by Large Language Models (LLMs), have been widely applied to climate change-specific research, providing essential information support for decision-makers and the public. Some studies have improved model performance on relevant tasks by constructing climate change-related instruction data and instruction-tuning LLMs. However, current research remains inadequate in efficiently producing large volumes of high-precision instruction data for climate change, which limits further development of climate change LLMs. This study introduces an automated method for constructing instruction data. The method generates instructions using facts and background knowledge from documents and enhances the diversity of the instruction data through web scraping and the collection of seed instructions. Using this method, we constructed a climate change instruction dataset, named ClimateChat-Corpus, which was used to fine-tune open-source LLMs, resulting in an LLM named ClimateChat. Evaluation results show that ClimateChat significantly improves performance on climate change question-and-answer tasks. Additionally, we evaluated the impact of different base models and instruction data on LLM performance and demonstrated its capability to adapt to a wide range of climate change scientific discovery tasks, emphasizing the importance of selecting an appropriate base model for instruction tuning. This research provides valuable references and empirical support for constructing climate change instruction data and training climate change-specific LLMs. 

---
# ASCD: Attention-Steerable Contrastive Decoding for Reducing Hallucination in MLLM 

**Authors**: Yujun Wang, Jinhe Bi, Yunpu Ma, Soeren Pirk  

**Link**: [PDF](https://arxiv.org/pdf/2506.14766)  

**Abstract**: Multimodal Large Language Model (MLLM) often suffer from hallucinations. They over-rely on partial cues and generate incorrect responses. Recently, methods like Visual Contrastive Decoding (VCD) and Instruction Contrastive Decoding (ICD) have been proposed to mitigate hallucinations by contrasting predictions from perturbed or negatively prefixed inputs against original outputs. In this work, we uncover that methods like VCD and ICD fundamentally influence internal attention dynamics of the model. This observation suggests that their effectiveness may not stem merely from surface-level modifications to logits but from deeper shifts in attention distribution. Inspired by this insight, we propose an attention-steerable contrastive decoding framework that directly intervenes in attention mechanisms of the model to offer a more principled approach to mitigating hallucinations. Our experiments across multiple MLLM architectures and diverse decoding methods demonstrate that our approach significantly reduces hallucinations and improves the performance on benchmarks such as POPE, CHAIR, and MMHal-Bench, while simultaneously enhancing performance on standard VQA benchmarks. 

---
# Optimizing Length Compression in Large Reasoning Models 

**Authors**: Zhengxiang Cheng, Dongping Chen, Mingyang Fu, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.14755)  

**Abstract**: Large Reasoning Models (LRMs) have achieved remarkable success, yet they often suffer from producing unnecessary and verbose reasoning chains. We identify a core aspect of this issue as "invalid thinking" -- models tend to repeatedly double-check their work after having derived the correct answer. To address this specific inefficiency, we move beyond the general principles of Efficacy and Efficiency to propose two new, fine-grained principles: Brevity, which advocates for eliminating redundancy, and Sufficiency, which ensures critical reasoning steps are preserved. Guided by these principles, we introduce LC-R1, a post-training method based on Group Relative Policy Optimization (GRPO). LC-R1 employs a novel combination of a Length Reward for overall conciseness and a Compress Reward that is specifically designed to remove the invalid portion of the thinking process. Extensive experiments on multiple reasoning benchmarks demonstrate that LC-R1 achieves a significant reduction in sequence length (~50%) with only a marginal (~2%) drop in accuracy, achieving a favorable trade-off point on the Pareto frontier that prioritizes high compression. Our analysis further validates the robustness of LC-R1 and provides valuable insights for developing more powerful yet computationally efficient LRMs. Our code is released at this https URL. 

---
# VisText-Mosquito: A Multimodal Dataset and Benchmark for AI-Based Mosquito Breeding Site Detection and Reasoning 

**Authors**: Md. Adnanul Islam, Md. Faiyaz Abdullah Sayeedi, Md. Asaduzzaman Shuvo, Muhammad Ziaur Rahman, Shahanur Rahman Bappy, Raiyan Rahman, Swakkhar Shatabda  

**Link**: [PDF](https://arxiv.org/pdf/2506.14629)  

**Abstract**: Mosquito-borne diseases pose a major global health risk, requiring early detection and proactive control of breeding sites to prevent outbreaks. In this paper, we present VisText-Mosquito, a multimodal dataset that integrates visual and textual data to support automated detection, segmentation, and reasoning for mosquito breeding site analysis. The dataset includes 1,828 annotated images for object detection, 142 images for water surface segmentation, and natural language reasoning texts linked to each image. The YOLOv9s model achieves the highest precision of 0.92926 and mAP@50 of 0.92891 for object detection, while YOLOv11n-Seg reaches a segmentation precision of 0.91587 and mAP@50 of 0.79795. For reasoning generation, our fine-tuned BLIP model achieves a final loss of 0.0028, with a BLEU score of 54.7, BERTScore of 0.91, and ROUGE-L of 0.87. This dataset and model framework emphasize the theme "Prevention is Better than Cure", showcasing how AI-based detection can proactively address mosquito-borne disease risks. The dataset and implementation code are publicly available at GitHub: this https URL 

---
# Computational Studies in Influencer Marketing: A Systematic Literature Review 

**Authors**: Haoyang Gui, Thales Bertaglia, Catalina Goanta, Gerasimos Spanakis  

**Link**: [PDF](https://arxiv.org/pdf/2506.14602)  

**Abstract**: Influencer marketing has become a crucial feature of digital marketing strategies. Despite its rapid growth and algorithmic relevance, the field of computational studies in influencer marketing remains fragmented, especially with limited systematic reviews covering the computational methodologies employed. This makes overarching scientific measurements in the influencer economy very scarce, to the detriment of interested stakeholders outside of platforms themselves, such as regulators, but also researchers from other fields. This paper aims to provide an overview of the state of the art of computational studies in influencer marketing by conducting a systematic literature review (SLR) based on the PRISMA model. The paper analyses 69 studies to identify key research themes, methodologies, and future directions in this research field. The review identifies four major research themes: Influencer identification and characterisation, Advertising strategies and engagement, Sponsored content analysis and discovery, and Fairness. Methodologically, the studies are categorised into machine learning-based techniques (e.g., classification, clustering) and non-machine-learning-based techniques (e.g., statistical analysis, network analysis). Key findings reveal a strong focus on optimising commercial outcomes, with limited attention to regulatory compliance and ethical considerations. The review highlights the need for more nuanced computational research that incorporates contextual factors such as language, platform, and industry type, as well as improved model explainability and dataset reproducibility. The paper concludes by proposing a multidisciplinary research agenda that emphasises the need for further links to regulation and compliance technology, finer granularity in analysis, and the development of standardised datasets. 

---
# TGDPO: Harnessing Token-Level Reward Guidance for Enhancing Direct Preference Optimization 

**Authors**: Mingkang Zhu, Xi Chen, Zhongdao Wang, Bei Yu, Hengshuang Zhao, Jiaya Jia  

**Link**: [PDF](https://arxiv.org/pdf/2506.14574)  

**Abstract**: Recent advancements in reinforcement learning from human feedback have shown that utilizing fine-grained token-level reward models can substantially enhance the performance of Proximal Policy Optimization (PPO) in aligning large language models. However, it is challenging to leverage such token-level reward as guidance for Direct Preference Optimization (DPO), since DPO is formulated as a sequence-level bandit problem. To address this challenge, this work decomposes the sequence-level PPO into a sequence of token-level proximal policy optimization problems and then frames the problem of token-level PPO with token-level reward guidance, from which closed-form optimal token-level policy and the corresponding token-level reward can be derived. Using the obtained reward and Bradley-Terry model, this work establishes a framework of computable loss functions with token-level reward guidance for DPO, and proposes a practical reward guidance based on the induced DPO reward. This formulation enables different tokens to exhibit varying degrees of deviation from reference policy based on their respective rewards. Experiment results demonstrate that our method achieves substantial performance improvements over DPO, with win rate gains of up to 7.5 points on MT-Bench, 6.2 points on AlpacaEval 2, and 4.3 points on Arena-Hard. Code is available at this https URL. 

---
# Improving LoRA with Variational Learning 

**Authors**: Bai Cong, Nico Daheim, Yuesong Shen, Rio Yokota, Mohammad Emtiyaz Khan, Thomas Möllenhoff  

**Link**: [PDF](https://arxiv.org/pdf/2506.14280)  

**Abstract**: Bayesian methods have recently been used to improve LoRA finetuning and, although they improve calibration, their effect on other metrics (such as accuracy) is marginal and can sometimes even be detrimental. Moreover, Bayesian methods also increase computational overheads and require additional tricks for them to work well. Here, we fix these issues by using a recently proposed variational algorithm called IVON. We show that IVON is easy to implement and has similar costs to AdamW, and yet it can also drastically improve many metrics by using a simple posterior pruning technique. We present extensive results on billion-scale LLMs (Llama and Qwen series) going way beyond the scale of existing applications of IVON. For example, we finetune a Llama-3.2-3B model on a set of commonsense reasoning tasks and improve accuracy over AdamW by 1.3% and reduce ECE by 5.4%, outperforming AdamW and other recent Bayesian methods like Laplace-LoRA and BLoB. Overall, our results show that variational learning with IVON can effectively improve LoRA finetuning. 

---
# Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs 

**Authors**: Xumeng Wen, Zihan Liu, Shun Zheng, Zhijian Xu, Shengyu Ye, Zhirong Wu, Xiao Liang, Yang Wang, Junjie Li, Ziming Miao, Jiang Bian, Mao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14245)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a promising paradigm for advancing the reasoning capabilities of Large Language Models (LLMs). However, a critical paradox clouds its efficacy: RLVR-tuned models often underperform their base models on the $Pass@K$ metric for solution-finding, leading to the hypothesis that RLVR merely re-weights existing reasoning paths at the cost of reasoning diversity. In this work, we resolve this contradiction by identifying the source of the problem: the $Pass@K$ metric itself is a flawed measure of reasoning, as it credits correct final answers that probably arise from inaccurate or incomplete chains of thought (CoTs). To address this, we introduce a more precise evaluation metric, $CoT$-$Pass@K$, which mandates that both the reasoning path and the final answer be correct. We provide a new theoretical foundation that formalizes how RLVR, unlike traditional RL, is uniquely structured to incentivize logical integrity. Our empirical results are supportive: using $CoT$-$Pass@K$, we observe that RLVR can incentivize the generalization of correct reasoning for all values of $K$. Furthermore, by analyzing the training dynamics, we find that this enhanced reasoning capability emerges early in the training process and smoothly generalizes. Our work provides a clear perspective on the role of RLVR, offers a more reliable method for its evaluation, and confirms its potential to genuinely advance machine reasoning. 

---
# Fretting-Transformer: Encoder-Decoder Model for MIDI to Tablature Transcription 

**Authors**: Anna Hamberger, Sebastian Murgul, Jochen Schmidt, Michael Heizmann  

**Link**: [PDF](https://arxiv.org/pdf/2506.14223)  

**Abstract**: Music transcription plays a pivotal role in Music Information Retrieval (MIR), particularly for stringed instruments like the guitar, where symbolic music notations such as MIDI lack crucial playability information. This contribution introduces the Fretting-Transformer, an encoderdecoder model that utilizes a T5 transformer architecture to automate the transcription of MIDI sequences into guitar tablature. By framing the task as a symbolic translation problem, the model addresses key challenges, including string-fret ambiguity and physical playability. The proposed system leverages diverse datasets, including DadaGP, GuitarToday, and Leduc, with novel data pre-processing and tokenization strategies. We have developed metrics for tablature accuracy and playability to quantitatively evaluate the performance. The experimental results demonstrate that the Fretting-Transformer surpasses baseline methods like A* and commercial applications like Guitar Pro. The integration of context-sensitive processing and tuning/capo conditioning further enhances the model's performance, laying a robust foundation for future developments in automated guitar transcription. 

---
# Improving Practical Aspects of End-to-End Multi-Talker Speech Recognition for Online and Offline Scenarios 

**Authors**: Aswin Shanmugam Subramanian, Amit Das, Naoyuki Kanda, Jinyu Li, Xiaofei Wang, Yifan Gong  

**Link**: [PDF](https://arxiv.org/pdf/2506.14204)  

**Abstract**: We extend the frameworks of Serialized Output Training (SOT) to address practical needs of both streaming and offline automatic speech recognition (ASR) applications. Our approach focuses on balancing latency and accuracy, catering to real-time captioning and summarization requirements. We propose several key improvements: (1) Leveraging Continuous Speech Separation (CSS) single-channel front-end with end-to-end (E2E) systems for highly overlapping scenarios, challenging the conventional wisdom of E2E versus cascaded setups. The CSS framework improves the accuracy of the ASR system by separating overlapped speech from multiple speakers. (2) Implementing dual models -- Conformer Transducer for streaming and Sequence-to-Sequence for offline -- or alternatively, a two-pass model based on cascaded encoders. (3) Exploring segment-based SOT (segSOT) which is better suited for offline scenarios while also enhancing readability of multi-talker transcriptions. 

---
# Pushing the Performance of Synthetic Speech Detection with Kolmogorov-Arnold Networks and Self-Supervised Learning Models 

**Authors**: Tuan Dat Phuong, Long-Vu Hoang, Huy Dat Tran  

**Link**: [PDF](https://arxiv.org/pdf/2506.14153)  

**Abstract**: Recent advancements in speech synthesis technologies have led to increasingly advanced spoofing attacks, posing significant challenges for automatic speaker verification systems. While systems based on self-supervised learning (SSL) models, particularly the XLSR-Conformer model, have demonstrated remarkable performance in synthetic speech detection, there remains room for architectural improvements. In this paper, we propose a novel approach that replaces the traditional Multi-Layer Perceptron in the XLSR-Conformer model with a Kolmogorov-Arnold Network (KAN), a novel architecture based on the Kolmogorov-Arnold representation theorem. Our results on ASVspoof2021 demonstrate that integrating KAN into the SSL-based models can improve the performance by 60.55% relatively on LA and DF sets, further achieving 0.70% EER on the 21LA set. These findings suggest that incorporating KAN into SSL-based models is a promising direction for advances in synthetic speech detection. 

---
# Acoustic scattering AI for non-invasive object classifications: A case study on hair assessment 

**Authors**: Long-Vu Hoang, Tuan Nguyen, Tran Huy Dat  

**Link**: [PDF](https://arxiv.org/pdf/2506.14148)  

**Abstract**: This paper presents a novel non-invasive object classification approach using acoustic scattering, demonstrated through a case study on hair assessment. When an incident wave interacts with an object, it generates a scattered acoustic field encoding structural and material properties. By emitting acoustic stimuli and capturing the scattered signals from head-with-hair-sample objects, we classify hair type and moisture using AI-driven, deep-learning-based sound classification. We benchmark comprehensive methods, including (i) fully supervised deep learning, (ii) embedding-based classification, (iii) supervised foundation model fine-tuning, and (iv) self-supervised model fine-tuning. Our best strategy achieves nearly 90% classification accuracy by fine-tuning all parameters of a self-supervised model. These results highlight acoustic scattering as a privacy-preserving, non-contact alternative to visual classification, opening huge potential for applications in various industries. 

---
# RadFabric: Agentic AI System with Reasoning Capability for Radiology 

**Authors**: Wenting Chen, Yi Dong, Zhaojun Ding, Yucheng Shi, Yifan Zhou, Fang Zeng, Yijun Luo, Tianyu Lin, Yihang Su, Yichen Wu, Kai Zhang, Zhen Xiang, Tianming Liu, Ninghao Liu, Lichao Sun, Yixuan Yuan, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.14142)  

**Abstract**: Chest X ray (CXR) imaging remains a critical diagnostic tool for thoracic conditions, but current automated systems face limitations in pathology coverage, diagnostic accuracy, and integration of visual and textual reasoning. To address these gaps, we propose RadFabric, a multi agent, multimodal reasoning framework that unifies visual and textual analysis for comprehensive CXR interpretation. RadFabric is built on the Model Context Protocol (MCP), enabling modularity, interoperability, and scalability for seamless integration of new diagnostic agents. The system employs specialized CXR agents for pathology detection, an Anatomical Interpretation Agent to map visual findings to precise anatomical structures, and a Reasoning Agent powered by large multimodal reasoning models to synthesize visual, anatomical, and clinical data into transparent and evidence based diagnoses. RadFabric achieves significant performance improvements, with near-perfect detection of challenging pathologies like fractures (1.000 accuracy) and superior overall diagnostic accuracy (0.799) compared to traditional systems (0.229 to 0.527). By integrating cross modal feature alignment and preference-driven reasoning, RadFabric advances AI-driven radiology toward transparent, anatomically precise, and clinically actionable CXR analysis. 

---
# Innovating China's Intangible Cultural Heritage with DeepSeek + MidJourney: The Case of Yangliuqing theme Woodblock Prints 

**Authors**: RuiKun Yang, ZhongLiang Wei, Longdi Xian  

**Link**: [PDF](https://arxiv.org/pdf/2506.14104)  

**Abstract**: Yangliuqing woodblock prints, a cornerstone of China's intangible cultural heritage, are celebrated for their intricate designs and vibrant colors. However, preserving these traditional art forms while fostering innovation presents significant challenges. This study explores the DeepSeek + MidJourney approach to generating creative, themed Yangliuqing woodblock prints focused on the fight against COVID-19 and depicting joyous winners. Using Fréchet Inception Distance (FID) scores for evaluation, the method that combined DeepSeek-generated thematic prompts, MidJourney-generated thematic images, original Yangliuqing prints, and DeepSeek-generated key prompts in MidJourney-generated outputs achieved the lowest mean FID score (150.2) with minimal variability ({\sigma} = 4.9). Additionally, feedback from 62 participants, collected via questionnaires, confirmed that this hybrid approach produced the most representative results. Moreover, the questionnaire data revealed that participants demonstrated the highest willingness to promote traditional culture and the strongest interest in consuming the AI-generated images produced through this method. These findings underscore the effectiveness of an innovative approach that seamlessly blends traditional artistic elements with modern AI-driven creativity, ensuring both cultural preservation and contemporary relevance. 

---
# InsertRank: LLMs can reason over BM25 scores to Improve Listwise Reranking 

**Authors**: Rahul Seetharaman, Kaustubh D. Dhole, Aman Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2506.14086)  

**Abstract**: Large Language Models (LLMs) have demonstrated significant strides across various information retrieval tasks, particularly as rerankers, owing to their strong generalization and knowledge-transfer capabilities acquired from extensive pretraining. In parallel, the rise of LLM-based chat interfaces has raised user expectations, encouraging users to pose more complex queries that necessitate retrieval by ``reasoning'' over documents rather than through simple keyword matching or semantic similarity. While some recent efforts have exploited reasoning abilities of LLMs for reranking such queries, considerable potential for improvement remains. In that regards, we introduce InsertRank, an LLM-based reranker that leverages lexical signals like BM25 scores during reranking to further improve retrieval performance. InsertRank demonstrates improved retrieval effectiveness on -- BRIGHT, a reasoning benchmark spanning 12 diverse domains, and R2MED, a specialized medical reasoning retrieval benchmark spanning 8 different tasks. We conduct an exhaustive evaluation and several ablation studies and demonstrate that InsertRank consistently improves retrieval effectiveness across multiple families of LLMs, including GPT, Gemini, and Deepseek models. %In addition, we also conduct ablation studies on normalization by varying the scale of the BM25 scores, and positional bias by shuffling the order of the documents. With Deepseek-R1, InsertRank achieves a score of 37.5 on the BRIGHT benchmark. and 51.1 on the R2MED benchmark, surpassing previous methods. 

---
# AssistedDS: Benchmarking How External Domain Knowledge Assists LLMs in Automated Data Science 

**Authors**: An Luo, Xun Xian, Jin Du, Fangqiao Tian, Ganghua Wang, Ming Zhong, Shengchun Zhao, Xuan Bi, Zirui Liu, Jiawei Zhou, Jayanth Srinivasa, Ashish Kundu, Charles Fleming, Mingyi Hong, Jie Ding  

**Link**: [PDF](https://arxiv.org/pdf/2506.13992)  

**Abstract**: Large language models (LLMs) have advanced the automation of data science workflows. Yet it remains unclear whether they can critically leverage external domain knowledge as human data scientists do in practice. To answer this question, we introduce AssistedDS (Assisted Data Science), a benchmark designed to systematically evaluate how LLMs handle domain knowledge in tabular prediction tasks. AssistedDS features both synthetic datasets with explicitly known generative mechanisms and real-world Kaggle competitions, each accompanied by curated bundles of helpful and adversarial documents. These documents provide domain-specific insights into data cleaning, feature engineering, and model selection. We assess state-of-the-art LLMs on their ability to discern and apply beneficial versus harmful domain knowledge, evaluating submission validity, information recall, and predictive performance. Our results demonstrate three key findings: (1) LLMs frequently exhibit an uncritical adoption of provided information, significantly impairing their predictive performance when adversarial content is introduced, (2) helpful guidance is often insufficient to counteract the negative influence of adversarial information, and (3) in Kaggle datasets, LLMs often make errors in handling time-series data, applying consistent feature engineering across different folds, and interpreting categorical variables correctly. These findings highlight a substantial gap in current models' ability to critically evaluate and leverage expert knowledge, underscoring an essential research direction for developing more robust, knowledge-aware automated data science systems. 

---
# CRITICTOOL: Evaluating Self-Critique Capabilities of Large Language Models in Tool-Calling Error Scenarios 

**Authors**: Shiting Huang, Zhen Fang, Zehui Chen, Siyu Yuan, Junjie Ye, Yu Zeng, Lin Chen, Qi Mao, Feng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.13977)  

**Abstract**: The ability of large language models (LLMs) to utilize external tools has enabled them to tackle an increasingly diverse range of tasks. However, as the tasks become more complex and long-horizon, the intricate tool utilization process may trigger various unexpected errors. Therefore, how to effectively handle such errors, including identifying, diagnosing, and recovering from them, has emerged as a key research direction for advancing tool learning. In this work, we first extensively analyze the types of errors encountered during the function-calling process on several competitive tool evaluation benchmarks. Based on it, we introduce CRITICTOOL, a comprehensive critique evaluation benchmark specialized for tool learning. Building upon a novel evolutionary strategy for dataset construction, CRITICTOOL holds diverse tool-use errors with varying complexities, which better reflects real-world scenarios. We conduct extensive experiments on CRITICTOOL, and validate the generalization and effectiveness of our constructed benchmark strategy. We also provide an in-depth analysis of the tool reflection ability on various LLMs, offering a new perspective on the field of tool learning in LLMs. The code is available at \href{this https URL}{this https URL}. 

---
# Multimodal Fusion with Semi-Supervised Learning Minimizes Annotation Quantity for Modeling Videoconference Conversation Experience 

**Authors**: Andrew Chang, Chenkai Hu, Ji Qi, Zhuojian Wei, Kexin Zhang, Viswadruth Akkaraju, David Poeppel, Dustin Freeman  

**Link**: [PDF](https://arxiv.org/pdf/2506.13971)  

**Abstract**: Group conversations over videoconferencing are a complex social behavior. However, the subjective moments of negative experience, where the conversation loses fluidity or enjoyment remain understudied. These moments are infrequent in naturalistic data, and thus training a supervised learning (SL) model requires costly manual data annotation. We applied semi-supervised learning (SSL) to leverage targeted labeled and unlabeled clips for training multimodal (audio, facial, text) deep features to predict non-fluid or unenjoyable moments in holdout videoconference sessions. The modality-fused co-training SSL achieved an ROC-AUC of 0.9 and an F1 score of 0.6, outperforming SL models by up to 4% with the same amount of labeled data. Remarkably, the best SSL model with just 8% labeled data matched 96% of the SL model's full-data performance. This shows an annotation-efficient framework for modeling videoconference experience. 

---
# Adaptive Guidance Accelerates Reinforcement Learning of Reasoning Models 

**Authors**: Vaskar Nath, Elaine Lau, Anisha Gunjal, Manasi Sharma, Nikhil Baharte, Sean Hendryx  

**Link**: [PDF](https://arxiv.org/pdf/2506.13923)  

**Abstract**: We study the process through which reasoning models trained with reinforcement learning on verifiable rewards (RLVR) can learn to solve new problems. We find that RLVR drives performance through two main means: (1) by compressing pass@$k$ into pass@1 and (2) via "capability gain" in which models learn to solve new problems that they previously could not solve even at high $k$. We find that while capability gain exists across model scales, learning to solve new problems is primarily driven through self-distillation. We demonstrate these findings across model scales ranging from 0.5B to 72B on >500,000 reasoning problems with prompts and verifiable final answers across math, science, and code domains. We further show that we can significantly improve pass@$k$ rates by leveraging natural language guidance for the model to consider within context while still requiring the model to derive a solution chain from scratch. Based of these insights, we derive $\text{Guide}$ - a new class of online training algorithms. $\text{Guide}$ adaptively incorporates hints into the model's context on problems for which all rollouts were initially incorrect and adjusts the importance sampling ratio for the "off-policy" trajectories in order to optimize the policy for contexts in which the hints are no longer present. We describe variants of $\text{Guide}$ for GRPO and PPO and empirically show that Guide-GRPO on 7B and 32B parameter models improves generalization over its vanilla counterpart with up to 4$\%$ macro-average improvement across math benchmarks. We include careful ablations to analyze $\text{Guide}$'s components and theoretically analyze Guide's learning efficiency. 

---
# Investigating the Potential of Large Language Model-Based Router Multi-Agent Architectures for Foundation Design Automation: A Task Classification and Expert Selection Study 

**Authors**: Sompote Youwai, David Phim, Vianne Gayl Murcia, Rianne Clair Onas  

**Link**: [PDF](https://arxiv.org/pdf/2506.13811)  

**Abstract**: This study investigates router-based multi-agent systems for automating foundation design calculations through intelligent task classification and expert selection. Three approaches were evaluated: single-agent processing, multi-agent designer-checker architecture, and router-based expert selection. Performance assessment utilized baseline models including DeepSeek R1, ChatGPT 4 Turbo, Grok 3, and Gemini 2.5 Pro across shallow foundation and pile design scenarios. The router-based configuration achieved performance scores of 95.00% for shallow foundations and 90.63% for pile design, representing improvements of 8.75 and 3.13 percentage points over standalone Grok 3 performance respectively. The system outperformed conventional agentic workflows by 10.0 to 43.75 percentage points. Grok 3 demonstrated superior standalone performance without external computational tools, indicating advances in direct LLM mathematical reasoning for engineering applications. The dual-tier classification framework successfully distinguished foundation types, enabling appropriate analytical approaches. Results establish router-based multi-agent systems as optimal for foundation design automation while maintaining professional documentation standards. Given safety-critical requirements in civil engineering, continued human oversight remains essential, positioning these systems as advanced computational assistance tools rather than autonomous design replacements in professional practice. 

---
# ICE-ID: A Novel Historical Census Data Benchmark Comparing NARS against LLMs, \& a ML Ensemble on Longitudinal Identity Resolution 

**Authors**: Gonçalo Hora de Carvalho, Lazar S. Popov, Sander Kaatee, Kristinn R. Thórisson, Tangrui Li, Pétur Húni Björnsson, Jilles S. Dibangoye  

**Link**: [PDF](https://arxiv.org/pdf/2506.13792)  

**Abstract**: We introduce ICE-ID, a novel benchmark dataset for historical identity resolution, comprising 220 years (1703-1920) of Icelandic census records. ICE-ID spans multiple generations of longitudinal data, capturing name variations, demographic changes, and rich genealogical links. To the best of our knowledge, this is the first large-scale, open tabular dataset specifically designed to study long-term person-entity matching in a real-world population. We define identity resolution tasks (within and across census waves) with clearly documented metrics and splits. We evaluate a range of methods: handcrafted rule-based matchers, a ML ensemble as well as LLMs for structured data (e.g. transformer-based tabular networks) against a novel approach to tabular data called NARS (Non-Axiomatic Reasoning System) - a general-purpose AI framework designed to reason with limited knowledge and resources. Its core is Non-Axiomatic Logic (NAL), a term-based logic. Our experiments show that NARS is suprisingly simple and competitive with other standard approaches, achieving SOTA at our task. By releasing ICE-ID and our code, we enable reproducible benchmarking of identity resolution approaches in longitudinal settings and hope that ICE-ID opens new avenues for cross-disciplinary research in data linkage and historical analytics. 

---
# AcademicBrowse: Benchmarking Academic Browse Ability of LLMs 

**Authors**: Junting Zhou, Wang Li, Yiyan Liao, Nengyuan Zhang, Tingjia Miaoand Zhihui Qi, Yuhan Wu, Tong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13784)  

**Abstract**: Large Language Models (LLMs)' search capabilities have garnered significant attention. Existing benchmarks, such as OpenAI's BrowseComp, primarily focus on general search scenarios and fail to adequately address the specific demands of academic search. These demands include deeper literature tracing and organization, professional support for academic databases, the ability to navigate long-tail academic knowledge, and ensuring academic rigor. Here, we proposed AcademicBrowse, the first dataset specifically designed to evaluate the complex information retrieval capabilities of Large Language Models (LLMs) in academic research. AcademicBrowse possesses the following key characteristics: Academic Practicality, where question content closely mirrors real academic learning and research environments, avoiding deliberately misleading models; High Difficulty, with answers that are challenging for single models (e.g., Grok DeepSearch or Gemini Deep Research) to provide directly, often requiring at least three deep searches to derive; Concise Evaluation, where limiting conditions ensure answers are as unique as possible, accompanied by clear sources and brief solution explanations, greatly facilitating subsequent audit and verification, surpassing the current lack of analyzed search datasets both domestically and internationally; and Broad Coverage, as the dataset spans at least 15 different academic disciplines. Through AcademicBrowse, we expect to more precisely measure and promote the performance improvement of LLMs in complex academic information retrieval tasks. The data is available at: this https URL 

---
# Knowledge Compression via Question Generation: Enhancing Multihop Document Retrieval without Fine-tuning 

**Authors**: Anvi Alex Eponon, Moein Shahiki-Tash, Ildar Batyrshin, Christian E. Maldonado-Sifuentes, Grigori Sidorov, Alexander Gelbukh  

**Link**: [PDF](https://arxiv.org/pdf/2506.13778)  

**Abstract**: This study presents a question-based knowledge encoding approach that improves retrieval-augmented generation (RAG) systems without requiring fine-tuning or traditional chunking. We encode textual content using generated questions that span the lexical and semantic space, creating targeted retrieval cues combined with a custom syntactic reranking method.
In single-hop retrieval over 109 scientific papers, our approach achieves a Recall@3 of 0.84, outperforming traditional chunking methods by 60 percent. We also introduce "paper-cards", concise paper summaries under 300 characters, which enhance BM25 retrieval, increasing MRR@3 from 0.56 to 0.85 on simplified technical queries.
For multihop tasks, our reranking method reaches an F1 score of 0.52 with LLaMA2-Chat-7B on the LongBench 2WikiMultihopQA dataset, surpassing chunking and fine-tuned baselines which score 0.328 and 0.412 respectively.
This method eliminates fine-tuning requirements, reduces retrieval latency, enables intuitive question-driven knowledge access, and decreases vector storage demands by 80%, positioning it as a scalable and efficient RAG alternative. 

---
# LittleBit: Ultra Low-Bit Quantization via Latent Factorization 

**Authors**: Banseok Lee, Dongkyu Kim, Youngcheon You, Youngmin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.13771)  

**Abstract**: Deploying large language models (LLMs) often faces challenges from substantial memory and computational costs. Quantization offers a solution, yet performance degradation in the sub-1-bit regime remains particularly difficult. This paper introduces LittleBit, a novel method for extreme LLM compression. It targets levels like 0.1 bits per weight (BPW), achieving nearly 31$\times$ memory reduction, e.g., Llama2-13B to under 0.9 GB. LittleBit represents weights in a low-rank form using latent matrix factorization, subsequently binarizing these factors. To counteract information loss from this extreme precision, it integrates a multi-scale compensation mechanism. This includes row, column, and an additional latent dimension that learns per-rank importance. Two key contributions enable effective training: Dual Sign-Value-Independent Decomposition (Dual-SVID) for stable quantization-aware training (QAT) initialization, and integrated Residual Compensation to mitigate errors. Extensive experiments confirm LittleBit's superiority in sub-1-bit quantization: e.g., its 0.1 BPW performance on Llama2-7B surpasses the leading method's 0.7 BPW. This establishes a superior size-performance trade-off, with kernel-level benchmarks indicating potential for a 5$\times$ speedup compared to FP16. LittleBit paves the way for deploying powerful LLMs in resource-constrained environments. 

---
