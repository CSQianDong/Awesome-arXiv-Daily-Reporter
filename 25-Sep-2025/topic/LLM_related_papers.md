# Learning Contextual Retrieval for Robust Conversational Search 

**Authors**: Seunghan Yang, Juntae Lee, Jihwan Bang, Kyuhong Shim, Minsoo Kim, Simyung Chang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19700)  

**Abstract**: Effective conversational search demands a deep understanding of user intent across multiple dialogue turns. Users frequently use abbreviations and shift topics in the middle of conversations, posing challenges for conventional retrievers. While query rewriting techniques improve clarity, they often incur significant computational cost due to additional autoregressive steps. Moreover, although LLM-based retrievers demonstrate strong performance, they are not explicitly optimized to track user intent in multi-turn settings, often failing under topic drift or contextual ambiguity. To address these limitations, we propose ContextualRetriever, a novel LLM-based retriever that directly incorporates conversational context into the retrieval process. Our approach introduces: (1) a context-aware embedding mechanism that highlights the current query within the dialogue history; (2) intent-guided supervision based on high-quality rewritten queries; and (3) a training strategy that preserves the generative capabilities of the base LLM. Extensive evaluations across multiple conversational search benchmarks demonstrate that ContextualRetriever significantly outperforms existing methods while incurring no additional inference overhead. 

---
# SIM-CoT: Supervised Implicit Chain-of-Thought 

**Authors**: Xilin Wei, Xiaoran Liu, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Jiaqi Wang, Xipeng Qiu, Dahua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.20317)  

**Abstract**: Implicit Chain-of-Thought (CoT) methods present a promising, token-efficient alternative to explicit CoT reasoning in Large Language Models (LLMs), but a persistent performance gap has limited the application of implicit CoT. We identify a core latent instability issue by scaling the computational budget of implicit CoT approaches: as we increase the number of implicit reasoning tokens to enhance performance, the training process often becomes unstable and collapses. Our analysis reveals that this instability arises from the latent representations becoming homogeneous and losing their semantic diversity, a failure caused by insufficient step-level supervision in existing implicit CoT approaches. To address this issue, we propose SIM-CoT, a plug-and-play training module that introduces step-level supervision to stabilize and enrich the latent reasoning space. Specifically, SIM-CoT employs an auxiliary decoder during training to align each implicit token with its corresponding explicit reasoning step, ensuring that latent states capture distinct and meaningful information. The proposed auxiliary decoder is removed during inference, preserving the computational efficiency of implicit CoT methods with no added overhead. In addition, the auxiliary decoder affords interpretability of implicit reasoning by projecting each latent token onto an explicit reasoning vocabulary, enabling per-step visualization of semantic roles and diagnosis. SIM-CoT significantly enhances both the in-domain accuracy and out-of-domain stability of various implicit CoT methods, boosting baselines like Coconut by +8.2% on GPT-2 and CODI by +3.0% on LLaMA-3.1 8B. Demonstrating strong scalability, SIM-CoT also surpasses the explicit CoT baseline on GPT-2 by 2.1% with 2.3\times greater token efficiency, while substantially closing the performance gap on larger models like LLaMA-3.1 8B. 

---
# DRES: Benchmarking LLMs for Disfluency Removal 

**Authors**: Maria Teleki, Sai Janjur, Haoran Liu, Oliver Grabner, Ketan Verma, Thomas Docog, Xiangjue Dong, Lingfeng Shi, Cong Wang, Stephanie Birkelbach, Jason Kim, Yin Zhang, James Caverlee  

**Link**: [PDF](https://arxiv.org/pdf/2509.20321)  

**Abstract**: Disfluencies -- such as "um," "uh," interjections, parentheticals, and edited statements -- remain a persistent challenge for speech-driven systems, degrading accuracy in command interpretation, summarization, and conversational agents. We introduce DRES (Disfluency Removal Evaluation Suite), a controlled text-level benchmark that establishes a reproducible semantic upper bound for this task. DRES builds on human-annotated Switchboard transcripts, isolating disfluency removal from ASR errors and acoustic variability. We systematically evaluate proprietary and open-source LLMs across scales, prompting strategies, and architectures. Our results reveal that (i) simple segmentation consistently improves performance, even for long-context models; (ii) reasoning-oriented models tend to over-delete fluent tokens; and (iii) fine-tuning achieves near state-of-the-art precision and recall but harms generalization abilities. We further present a set of LLM-specific error modes and offer nine practical recommendations (R1-R9) for deploying disfluency removal in speech-driven pipelines. DRES provides a reproducible, model-agnostic foundation for advancing robust spoken-language systems. 

---
# EmbeddingGemma: Powerful and Lightweight Text Representations 

**Authors**: Henrique Schechter Vera, Sahil Dua, Biao Zhang, Daniel Salz, Ryan Mullins, Sindhu Raghuram Panyam, Sara Smoot, Iftekhar Naim, Joe Zou, Feiyang Chen, Daniel Cer, Alice Lisak, Min Choi, Lucas Gonzalez, Omar Sanseviero, Glenn Cameron, Ian Ballantyne, Kat Black, Kaifeng Chen, Weiyi Wang, Zhe Li, Gus Martins, Jinhyuk Lee, Mark Sherwood, Juyeong Ji, Renjie Wu, Jingxiao Zheng, Jyotinder Singh, Abheesht Sharma, Divya Sreepat, Aashi Jain, Adham Elarabawy, AJ Co, Andreas Doumanoglou, Babak Samari, Ben Hora, Brian Potetz, Dahun Kim, Enrique Alfonseca, Fedor Moiseev, Feng Han, Frank Palma Gomez, Gustavo Hernández Ábrego, Hesen Zhang, Hui Hui, Jay Han, Karan Gill, Ke Chen, Koert Chen, Madhuri Shanbhogue, Michael Boratko, Paul Suganthan, Sai Meher Karthik Duddu, Sandeep Mariserla, Setareh Ariafar, Shanfeng Zhang, Shijie Zhang, Simon Baumgartner, Sonam Goenka, Steve Qiu, Tanmaya Dabral, Trevor Walker, Vikram Rao, Waleed Khawaja, Wenlei Zhou, Xiaoqi Ren, Ye Xia, Yichang Chen, Yi-Ting Chen, Zhe Dong, Zhongli Ding, Francesco Visin, Gaël Liu, Jiageng Zhang, Kathleen Kenealy, Michelle Casbon, Ravin Kumar, Thomas Mesnard, Zach Gleicher, Cormac Brick, Olivier Lacombe, Adam Roberts, Yunhsuan Sung, Raphael Hoffmann, Tris Warkentin, Armand Joulin, Tom Duerig, Mojtaba Seyedhosseini  

**Link**: [PDF](https://arxiv.org/pdf/2509.20354)  

**Abstract**: We introduce EmbeddingGemma, a new lightweight, open text embedding model based on the Gemma 3 language model family. Our innovative training recipe strategically captures knowledge from larger models via encoder-decoder initialization and geometric embedding distillation. We improve model robustness and expressiveness with a spread-out regularizer, and ensure generalizability by merging checkpoints from varied, optimized mixtures. Evaluated on the Massive Text Embedding Benchmark (MTEB) across multilingual, English, and code domains, EmbeddingGemma (300M) achieves state-of-the-art results. Notably, it outperforms prior top models, both proprietary and open, with fewer than 500M parameters, and provides performance comparable to models double its size, offering an exceptional performance-to-cost ratio. Remarkably, this lead persists when quantizing model weights or truncating embedding outputs. This makes EmbeddingGemma particularly well-suited for low-latency and high-throughput use cases such as on-device applications. We provide ablation studies exploring our key design choices. We release EmbeddingGemma to the community to promote further research. 

---
# Language Models that Think, Chat Better 

**Authors**: Adithya Bhaskar, Xi Ye, Danqi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.20357)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) improves language model reasoning by using rule-based rewards in verifiable domains such as mathematics and code. However, RLVR leads to limited generalization for open-ended tasks -- such as writing outline essays or making meal plans -- where humans reason routinely. This paper shows that the RLVR paradigm is effective beyond verifiable domains, and introduces **RL** with **M**odel-rewarded **T**hinking (**RLMT**) for general-purpose chat capabilities. Using diverse real-world prompts, RLMT requires LMs to generate long CoT reasoning before response, and optimizes them with online RL against a preference-based reward model used in RLHF. Across 40 training runs on Llama-3.1-8B and Qwen-2.5-7B (both base and instruct) and multiple optimization algorithms (DPO, PPO, and GRPO), RLMT consistently outperforms standard RLHF pipelines. This includes substantial gains of 3-7 points on three chat benchmarks (AlpacaEval2, WildBench, and ArenaHardV2), along with 1-3 point improvements on other tasks like creative writing and general knowledge. Our best 8B model surpasses GPT-4o in chat and creative writing and rivals Claude-3.7-Sonnet (Thinking). RLMT can also be applied directly to base models without an SFT stage, akin to R1-Zero training. Remarkably, with only 7K prompts, Llama-3.1-8B base trained with our RLMT recipe outperforms Llama-3.1-8B-Instruct post-trained with a complex multi-staged pipeline with 25M+ examples. We close with qualitative and quantitative analyses of how trained models plan their responses. Our results rethink the post-training pipeline and call upon future work to understand and employ thinking more broadly. 

---
# Instruction Boundary: Quantifying Biases in LLM Reasoning under Various Coverage 

**Authors**: Zipeng Ling, Yuehao Tang, Chen Huang, Shuliang Liu, Gaoyang Jiang, Shenghong Fu, Junqi Yang, Yao Wan, Jiawan Zhang, Kejia Huang, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20278)  

**Abstract**: Large-language-model (LLM) reasoning has long been regarded as a powerful tool for problem solving across domains, providing non-experts with valuable advice. However, their limitations - especially those stemming from prompt design - remain underexplored. Because users may supply biased or incomplete prompts - often unintentionally - LLMs can be misled, undermining reliability and creating risks. We refer to this vulnerability as the Instruction Boundary. To investigate the phenomenon, we distill it into eight concrete facets and introduce BiasDetector, a framework that measures biases arising from three instruction types: complete, redundant, and insufficient. We evaluate several mainstream LLMs and find that, despite high headline accuracy, substantial biases persist in many downstream tasks as a direct consequence of prompt coverage. Our empirical study confirms that LLM reasoning reliability can still be significantly improved. We analyze the practical impact of these biases and outline mitigation strategies. Our findings underscore the need for developers to tackle biases and for users to craft options carefully. 

---
# Embedding Domain Knowledge for Large Language Models via Reinforcement Learning from Augmented Generation 

**Authors**: Chaojun Nie, Jun Zhou, Guanxiang Wang, Shisong Wud, Zichen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20162)  

**Abstract**: Large language models (LLMs) often exhibit limited performance on domain-specific tasks due to the natural disproportionate representation of specialized information in their training data and the static nature of these datasets. Knowledge scarcity and temporal lag create knowledge gaps for domain applications. While post-training on domain datasets can embed knowledge into models, existing approaches have some limitations. Continual Pre-Training (CPT) treats all tokens in domain documents with equal importance, failing to prioritize critical knowledge points, while supervised fine-tuning (SFT) with question-answer pairs struggles to develop the coherent knowledge structures necessary for complex reasoning tasks. To address these challenges, we propose Reinforcement Learning from Augmented Generation (RLAG). Our approach iteratively cycles between sampling generations and optimizing the model through calculated rewards, effectively embedding critical and contextually coherent domain knowledge. We select generated outputs with the highest log probabilities as the sampling result, then compute three tailored reward metrics to guide the optimization process. To comprehensively evaluate domain expertise, we assess answer accuracy and the rationality of explanations generated for correctly answered questions. Experimental results across medical, legal, astronomy, and current events datasets demonstrate that our proposed method significantly outperforms baseline approaches. Our code and data are open sourced at this https URL. 

---
# Thinking Augmented Pre-training 

**Authors**: Liang Wang, Nan Yang, Shaohan Huang, Li Dong, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2509.20186)  

**Abstract**: This paper introduces a simple and scalable approach to improve the data efficiency of large language model (LLM) training by augmenting existing text data with thinking trajectories. The compute for pre-training LLMs has been growing at an unprecedented rate, while the availability of high-quality data remains limited. Consequently, maximizing the utility of available data constitutes a significant research challenge. A primary impediment is that certain high-quality tokens are difficult to learn given a fixed model capacity, as the underlying rationale for a single token can be exceptionally complex and deep. To address this issue, we propose Thinking augmented Pre-Training (TPT), a universal methodology that augments text with automatically generated thinking trajectories. Such augmentation effectively increases the volume of the training data and makes high-quality tokens more learnable through step-by-step reasoning and decomposition. We apply TPT across diverse training configurations up to $100$B tokens, encompassing pre-training with both constrained and abundant data, as well as mid-training from strong open-source checkpoints. Experimental results indicate that our method substantially improves the performance of LLMs across various model sizes and families. Notably, TPT enhances the data efficiency of LLM pre-training by a factor of $3$. For a $3$B parameter model, it improves the post-training performance by over $10\%$ on several challenging reasoning benchmarks. 

---
# Play by the Type Rules: Inferring Constraints for LLM Functions in Declarative Programs 

**Authors**: Parker Glenn, Alfy Samuel, Daben Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20208)  

**Abstract**: Integrating LLM powered operators in declarative query languages allows for the combination of cheap and interpretable functions with powerful, generalizable language model reasoning. However, in order to benefit from the optimized execution of a database query language like SQL, generated outputs must align with the rules enforced by both type checkers and database contents. Current approaches address this challenge with orchestrations consisting of many LLM-based post-processing calls to ensure alignment between generated outputs and database values, introducing performance bottlenecks. We perform a study on the ability of various sized open-source language models to both parse and execute functions within a query language based on SQL, showing that small language models can excel as function executors over hybrid data sources. Then, we propose an efficient solution to enforce the well-typedness of LLM functions, demonstrating 7% accuracy improvement on a multi-hop question answering dataset with 53% improvement in latency over comparable solutions. We make our implementation available at this https URL 

---
# Integrated Framework for LLM Evaluation with Answer Generation 

**Authors**: Sujeong Lee, Hayoung Lee, Seongsoo Heo, Wonik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2509.20097)  

**Abstract**: Reliable evaluation of large language models is essential to ensure their applicability in practical scenarios. Traditional benchmark-based evaluation methods often rely on fixed reference answers, limiting their ability to capture important qualitative aspects of generated responses. To address these shortcomings, we propose an integrated evaluation framework called \textit{self-refining descriptive evaluation with expert-driven diagnostics}, SPEED, which utilizes specialized functional experts to perform comprehensive, descriptive analyses of model outputs. Unlike conventional approaches, SPEED actively incorporates expert feedback across multiple dimensions, including hallucination detection, toxicity assessment, and lexical-contextual appropriateness. Experimental results demonstrate that SPEED achieves robust and consistent evaluation performance across diverse domains and datasets. Additionally, by employing relatively compact expert models, SPEED demonstrates superior resource efficiency compared to larger-scale evaluators. These findings illustrate that SPEED significantly enhances fairness and interpretability in LLM evaluations, offering a promising alternative to existing evaluation methodologies. 

---
# Probing Gender Bias in Multilingual LLMs: A Case Study of Stereotypes in Persian 

**Authors**: Ghazal Kalhor, Behnam Bahrak  

**Link**: [PDF](https://arxiv.org/pdf/2509.20168)  

**Abstract**: Multilingual Large Language Models (LLMs) are increasingly used worldwide, making it essential to ensure they are free from gender bias to prevent representational harm. While prior studies have examined such biases in high-resource languages, low-resource languages remain understudied. In this paper, we propose a template-based probing methodology, validated against real-world data, to uncover gender stereotypes in LLMs. As part of this framework, we introduce the Domain-Specific Gender Skew Index (DS-GSI), a metric that quantifies deviations from gender parity. We evaluate four prominent models, GPT-4o mini, DeepSeek R1, Gemini 2.0 Flash, and Qwen QwQ 32B, across four semantic domains, focusing on Persian, a low-resource language with distinct linguistic features. Our results show that all models exhibit gender stereotypes, with greater disparities in Persian than in English across all domains. Among these, sports reflect the most rigid gender biases. This study underscores the need for inclusive NLP practices and provides a framework for assessing bias in other low-resource languages. 

---
# WEST: LLM based Speech Toolkit for Speech Understanding, Generation, and Interaction 

**Authors**: Binbin Zhang, Chengdong Liang, Shuai Wang, Xuelong Geng, Zhao Guo, Haoyu Li, Hao Yin, Xipeng Yang, Pengshen Zhang, Changwei Ma, Lei Xie  

**Link**: [PDF](https://arxiv.org/pdf/2509.19902)  

**Abstract**: In this paper, we present WEST(WE Speech Toolkit), a speech toolkit based on a large language model (LLM) for speech understanding, generation, and interaction. There are three key features of WEST: 1) Fully LLM-based: Standing on the shoulders of giants by reusing mature architectures, ecosystems (e.g., Hugging Face), and methods (e.g., sequence packing) from large models. 2) Full-stack: Supports tasks such as recognition, synthesis, understanding, dialogue, and multimodal capabilities, with extensibility to incorporate open-source models. 3) Simple and Stupid: A simple and stupid speech toolkit that everyone can Touch. In addition, WEST provides two types of recipes, models, and experimental results. The first is entirely based on open-source models and open-source data, allowing users to fully reproduce the experiments in this paper and serving as a verification system or minimal system baseline. The second is trained on massive data, offering superior performance so the user can directly apply it out of the box. WEST is publicly avilable at this https URL 

---
# Causal Understanding by LLMs: The Role of Uncertainty 

**Authors**: Oscar Lithgow-Serrano, Vani Kanjirangat, Alessandro Antonucci  

**Link**: [PDF](https://arxiv.org/pdf/2509.20088)  

**Abstract**: Recent papers show LLMs achieve near-random accuracy in causal relation classification, raising questions about whether such failures arise from limited pretraining exposure or deeper representational gaps. We investigate this under uncertainty-based evaluation, testing whether pretraining exposure to causal examples improves causal understanding >18K PubMed sentences -- half from The Pile corpus, half post-2024 -- across seven models (Pythia-1.4B/7B/12B, GPT-J-6B, Dolly-7B/12B, Qwen-7B). We analyze model behavior through: (i) causal classification, where the model identifies causal relationships in text, and (ii) verbatim memorization probing, where we assess whether the model prefers previously seen causal statements over their paraphrases. Models perform four-way classification (direct/conditional/correlational/no-relationship) and select between originals and their generated paraphrases. Results show almost identical accuracy on seen/unseen sentences (p > 0.05), no memorization bias (24.8% original selection), and output distribution over the possible options is almost flat, with entropic values near the maximum (1.35/1.39), confirming random guessing. Instruction-tuned models show severe miscalibration (Qwen: > 95% confidence, 32.8% accuracy, ECE=0.49). Conditional relations induce highest entropy (+11% vs. direct). These findings suggest that failures in causal understanding arise from the lack of structured causal representation, rather than insufficient exposure to causal examples during pretraining. 

---
# Future Policy Aware Preference Learning for Mathematical Reasoning 

**Authors**: Minjae Oh, Yunho Choi, Dongmin Choi, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2509.19893)  

**Abstract**: Preference learning methods such as Direct Preference Optimization (DPO) have become standard for Large Language Model (LLM) post-training, yet they are often ineffective for mathematical reasoning. A key challenge is the large token overlap between preferred and dispreferred trajectories; lowering the probability of dispreferred trajectories also reduces the probability of shared useful tokens, leading to over-penalization and overall performance collapse. As a mitigation, existing algorithms include the probability of a trajectory under the current policy as a regularization term, which decreases the effect of the gradient when the probability is low. However, by the time this effect takes hold, useful tokens may have already been over-penalized as the model has begun to degrade. To address this, we propose Future Policy Aware (FPA) preference learning, which replaces the current policy with a future policy in the regularization term. This future policy is estimated via lightweight, logit-space extrapolation from a reference model toward the current model. FPA enables safer training by preemptively regularizing potentially problematic gradients. We apply FPA to DPO, RPO, and SimPER and evaluate them on the MATH and GSM8K benchmarks. FPA yields consistent performance gains, with the largest improvements observed with SimPER, achieving gains of up to 5.75%. We demonstrate that FPA provides proactive regularization while preserving the probability of shared, useful mathematical tokens, and enables longer, degradation-free training with negligible computational overhead. We will release our code publicly upon publication. 

---
# Do Before You Judge: Self-Reference as a Pathway to Better LLM Evaluation 

**Authors**: Wei-Hsiang Lin, Sheng-Lun Wei, Hen-Hsen Huang, Hsin-Hsi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.19880)  

**Abstract**: LLM-as-Judge frameworks are increasingly popular for AI evaluation, yet research findings on the relationship between models' generation and judgment abilities remain inconsistent. We investigate this relationship through systematic dataset- and instance-level analyses across 11 models and 21 diverse tasks. Despite both capabilities relying on the same underlying knowledge, our analyses reveal they are only weakly correlated, primarily due to LLMs' sensitivity to the responses being judged. To address this, we propose a self-reference-guided evaluation strategy that leverages a model's own answers as references. This approach significantly strengthens the correlation between generation and judgment abilities, offering a practical path to align these skills and providing a reliable proxy for model selection in evaluation tasks. 

---
# EnAnchored-X2X: English-Anchored Optimization for Many-to-Many Translation 

**Authors**: Sen Yang, Yu Bao, Yu Lu, Jiajun Chen, Shujian Huang, Shanbo Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.19770)  

**Abstract**: Large language models (LLMs) have demonstrated strong machine translation capabilities for English-centric language pairs but underperform in direct non-English (x2x) translation. This work addresses this limitation through a synthetic data generation framework that leverages models' established English-to-x (en2x) capabilities. By extending English parallel corpora into omnidirectional datasets and developing an English-referenced quality evaluation proxy, we enable effective collection of high-quality x2x training data. Combined with preference-based optimization, our method achieves significant improvement across 72 x2x directions for widely used LLMs, while generalizing to enhance en2x performance. The results demonstrate that strategic exploitation of English-centric strengths can bootstrap comprehensive multilingual translation capabilities in LLMs. We release codes, datasets, and model checkpoints at this https URL 

---
# OLaPh: Optimal Language Phonemizer 

**Authors**: Johannes Wirth  

**Link**: [PDF](https://arxiv.org/pdf/2509.20086)  

**Abstract**: Phonemization, the conversion of text into phonemes, is a key step in text-to-speech. Traditional approaches use rule-based transformations and lexicon lookups, while more advanced methods apply preprocessing techniques or neural networks for improved accuracy on out-of-domain vocabulary. However, all systems struggle with names, loanwords, abbreviations, and homographs. This work presents OLaPh (Optimal Language Phonemizer), a framework that combines large lexica, multiple NLP techniques, and compound resolution with a probabilistic scoring function. Evaluations in German and English show improved accuracy over previous approaches, including on a challenging dataset. To further address unresolved cases, we train a large language model on OLaPh-generated data, which achieves even stronger generalization and performance. Together, the framework and LLM improve phonemization consistency and provide a freely available resource for future research. 

---
# SINAI at eRisk@CLEF 2025: Transformer-Based and Conversational Strategies for Depression Detection 

**Authors**: Alba Maria Marmol-Romero, Manuel Garcia-Vega, Miguel Angel Garcia-Cumbreras, Arturo Montejo-Raez  

**Link**: [PDF](https://arxiv.org/pdf/2509.19861)  

**Abstract**: This paper describes the participation of the SINAI-UJA team in the eRisk@CLEF 2025 lab. Specifically, we addressed two of the proposed tasks: (i) Task 2: Contextualized Early Detection of Depression, and (ii) Pilot Task: Conversational Depression Detection via LLMs. Our approach for Task 2 combines an extensive preprocessing pipeline with the use of several transformer-based models, such as RoBERTa Base or MentalRoBERTA Large, to capture the contextual and sequential nature of multi-user conversations. For the Pilot Task, we designed a set of conversational strategies to interact with LLM-powered personas, focusing on maximizing information gain within a limited number of dialogue turns. In Task 2, our system ranked 8th out of 12 participating teams based on F1 score. However, a deeper analysis revealed that our models were among the fastest in issuing early predictions, which is a critical factor in real-world deployment scenarios. This highlights the trade-off between early detection and classification accuracy, suggesting potential avenues for optimizing both jointly in future work. In the Pilot Task, we achieved 1st place out of 5 teams, obtaining the best overall performance across all evaluation metrics: DCHR, ADODL and ASHR. Our success in this task demonstrates the effectiveness of structured conversational design when combined with powerful language models, reinforcing the feasibility of deploying LLMs in sensitive mental health assessment contexts. 

---
# Personality Vector: Modulating Personality of Large Language Models by Model Merging 

**Authors**: Seungjong Sun, Seo Yeon Baek, Jang Hyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.19727)  

**Abstract**: Driven by the demand for personalized AI systems, there is growing interest in aligning the behavior of large language models (LLMs) with human traits such as personality. Previous attempts to induce personality in LLMs have shown promising results, but they struggle to capture the continuous and multidimensional nature of human traits. In this work, we propose a novel method for personality modulation in LLMs via model merging. Specifically, we construct personality vectors by subtracting the weights of a pre-trained model from those of the fine-tuned model on a given personality trait. By merging personality vectors, we enable LLMs to exhibit desired personality traits without additional training. Extensive experiments show that personality vectors enable continuous control over trait intensity and support the composition of multiple traits. Furthermore, personality vectors transfer across diverse downstream models, suggesting that they encode generalizable representations of personality. Our code is available at here. 

---
# Large Language Models for Pedestrian Safety: An Application to Predicting Driver Yielding Behavior at Unsignalized Intersections 

**Authors**: Yicheng Yang, Zixian Li, Jean Paul Bizimana, Niaz Zafri, Yongfeng Dong, Tianyi Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.19657)  

**Abstract**: Pedestrian safety is a critical component of urban mobility and is strongly influenced by the interactions between pedestrian decision-making and driver yielding behavior at crosswalks. Modeling driver--pedestrian interactions at intersections requires accurately capturing the complexity of these behaviors. Traditional machine learning models often struggle to capture the nuanced and context-dependent reasoning required for these multifactorial interactions, due to their reliance on fixed feature representations and limited interpretability. In contrast, large language models (LLMs) are suited for extracting patterns from heterogeneous traffic data, enabling accurate modeling of driver-pedestrian interactions. Therefore, this paper leverages multimodal LLMs through a novel prompt design that incorporates domain-specific knowledge, structured reasoning, and few-shot prompting, enabling interpretable and context-aware inference of driver yielding behavior, as an example application of modeling pedestrian--driver interaction. We benchmarked state-of-the-art LLMs against traditional classifiers, finding that GPT-4o consistently achieves the highest accuracy and recall, while Deepseek-V3 excels in precision. These findings highlight the critical trade-offs between model performance and computational efficiency, offering practical guidance for deploying LLMs in real-world pedestrian safety systems. 

---
# GuessingGame: Measuring the Informativeness of Open-Ended Questions in Large Language Models 

**Authors**: Dylan Hutson, Daniel Vennemeyer, Aneesh Deshmukh, Justin Zhan, Tianyu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19593)  

**Abstract**: We introduce GuessingGame, a protocol for evaluating large language models (LLMs) as strategic question-askers in open-ended, open-domain settings. A Guesser LLM identifies a hidden object by posing free-form questions to an Oracle without predefined choices or candidate lists. To measure question quality, we propose two information gain (IG) metrics: a Bayesian method that tracks belief updates over semantic concepts using LLM-scored relevance, and an entropy-based method that filters candidates via ConceptNet. Both metrics are model-agnostic and support post hoc analysis. Across 858 games with multiple models and prompting strategies, higher IG strongly predicts efficiency: a one-standard-deviation IG increase reduces expected game length by 43\%. Prompting constraints guided by IG, such as enforcing question diversity, enable weaker models to significantly improve performance. These results show that question-asking in LLMs is both measurable and improvable, and crucial for interactive reasoning. 

---
# LLMs4All: A Review on Large Language Models for Research and Applications in Academic Disciplines 

**Authors**: Yanfang, Zheyuan Zhang, Tianyi Ma, Zehong Wang, Yiyang Li, Shifu Hou, Weixiang Sun, Kaiwen Shi, Yijun Ma, Wei Song, Ahmed Abbasi, Ying Cheng, Jane Cleland-Huang, Steven Corcelli, Patricia Culligan, Robert Goulding, Ming Hu, Ting Hua, John Lalor, Fang Liu, Tengfei Luo, Ed Maginn, Nuno Moniz, Jason Rohr, Brett Savoie, Daniel Slate, Tom Stapleford, Matthew Webber, Olaf Wiest, Johnny Zhang, Nitesh Chawla  

**Link**: [PDF](https://arxiv.org/pdf/2509.19580)  

**Abstract**: Cutting-edge Artificial Intelligence (AI) techniques keep reshaping our view of the world. For example, Large Language Models (LLMs) based applications such as ChatGPT have shown the capability of generating human-like conversation on extensive topics. Due to the impressive performance on a variety of language-related tasks (e.g., open-domain question answering, translation, and document summarization), one can envision the far-reaching impacts that can be brought by the LLMs with broader real-world applications (e.g., customer service, education and accessibility, and scientific discovery). Inspired by their success, this paper will offer an overview of state-of-the-art LLMs and their integration into a wide range of academic disciplines, including: (1) arts, letters, and law (e.g., history, philosophy, political science, arts and architecture, law), (2) economics and business (e.g., finance, economics, accounting, marketing), and (3) science and engineering (e.g., mathematics, physics and mechanical engineering, chemistry and chemical engineering, life sciences and bioengineering, earth sciences and civil engineering, computer science and electrical engineering). Integrating humanity and technology, in this paper, we will explore how LLMs are shaping research and practice in these fields, while also discussing key limitations, open challenges, and future directions in the era of generative AI. The review of how LLMs are engaged across disciplines-along with key observations and insights-can help researchers and practitioners interested in exploiting LLMs to advance their works in diverse real-world applications. 

---
# Do LLMs Encode Frame Semantics? Evidence from Frame Identification 

**Authors**: Jayanth Krishna Chundru, Rudrashis Poddar, Jie Cao, Tianyu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19540)  

**Abstract**: We investigate whether large language models encode latent knowledge of frame semantics, focusing on frame identification, a core challenge in frame semantic parsing that involves selecting the appropriate semantic frame for a target word in context. Using the FrameNet lexical resource, we evaluate models under prompt-based inference and observe that they can perform frame identification effectively even without explicit supervision. To assess the impact of task-specific training, we fine-tune the model on FrameNet data, which substantially improves in-domain accuracy while generalizing well to out-of-domain benchmarks. Further analysis shows that the models can generate semantically coherent frame definitions, highlighting the model's internalized understanding of frame semantics. 

---
# Benchmarking Gaslighting Attacks Against Speech Large Language Models 

**Authors**: Jinyang Wu, Bin Zhu, Xiandong Zou, Qiquan Zhang, Xu Fang, Pan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.19858)  

**Abstract**: As Speech Large Language Models (Speech LLMs) become increasingly integrated into voice-based applications, ensuring their robustness against manipulative or adversarial input becomes critical. Although prior work has studied adversarial attacks in text-based LLMs and vision-language models, the unique cognitive and perceptual challenges of speech-based interaction remain underexplored. In contrast, speech presents inherent ambiguity, continuity, and perceptual diversity, which make adversarial attacks more difficult to detect. In this paper, we introduce gaslighting attacks, strategically crafted prompts designed to mislead, override, or distort model reasoning as a means to evaluate the vulnerability of Speech LLMs. Specifically, we construct five manipulation strategies: Anger, Cognitive Disruption, Sarcasm, Implicit, and Professional Negation, designed to test model robustness across varied tasks. It is worth noting that our framework captures both performance degradation and behavioral responses, including unsolicited apologies and refusals, to diagnose different dimensions of susceptibility. Moreover, acoustic perturbation experiments are conducted to assess multi-modal robustness. To quantify model vulnerability, comprehensive evaluation across 5 Speech and multi-modal LLMs on over 10,000 test samples from 5 diverse datasets reveals an average accuracy drop of 24.3% under the five gaslighting attacks, indicating significant behavioral vulnerability. These findings highlight the need for more resilient and trustworthy speech-based AI systems. 

---
# How to inject knowledge efficiently? Knowledge Infusion Scaling Law for Pre-training Large Language Models 

**Authors**: Kangtao Lv, Haibin Chen, Yujin Yuan, Langming Liu, Shilei Liu, Yongwei Wang, Wenbo Su, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.19371)  

**Abstract**: Large language models (LLMs) have attracted significant attention due to their impressive general capabilities across diverse downstream tasks. However, without domain-specific optimization, they often underperform on specialized knowledge benchmarks and even produce hallucination. Recent studies show that strategically infusing domain knowledge during pretraining can substantially improve downstream performance. A critical challenge lies in balancing this infusion trade-off: injecting too little domain-specific data yields insufficient specialization, whereas excessive infusion triggers catastrophic forgetting of previously acquired knowledge. In this work, we focus on the phenomenon of memory collapse induced by over-infusion. Through systematic experiments, we make two key observations, i.e. 1) Critical collapse point: each model exhibits a threshold beyond which its knowledge retention capabilities sharply degrade. 2) Scale correlation: these collapse points scale consistently with the model's size. Building on these insights, we propose a knowledge infusion scaling law that predicts the optimal amount of domain knowledge to inject into large LLMs by analyzing their smaller counterparts. Extensive experiments across different model sizes and pertaining token budgets validate both the effectiveness and generalizability of our scaling law. 

---
# Meow: End-to-End Outline Writing for Automatic Academic Survey 

**Authors**: Zhaoyu Ma, Yuan Shan, Jiahao Zhao, Nan Xu, Lei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19370)  

**Abstract**: As academic paper publication numbers grow exponentially, conducting in-depth surveys with LLMs automatically has become an inevitable trend. Outline writing, which aims to systematically organize related works, is critical for automated survey generation. Yet existing automatic survey methods treat outline writing as mere workflow steps in the overall pipeline. Such template-based workflows produce outlines that lack in-depth understanding of the survey topic and fine-grained styles. To address these limitations, we propose Meow, the first metadata-driven outline writing framework that produces organized and faithful outlines efficiently. Specifically, we first formulate outline writing as an end-to-end task that generates hierarchical structured outlines from paper metadata. We then curate a high-quality dataset of surveys from arXiv, bioRxiv, and medRxiv, and establish systematic evaluation metrics for outline quality assessment. Finally, we employ a two-stage training approach combining supervised fine-tuning and reinforcement learning. Our 8B reasoning model demonstrates strong performance with high structural fidelity and stylistic coherence. 

---
# The Inadequacy of Offline LLM Evaluations: A Need to Account for Personalization in Model Behavior 

**Authors**: Angelina Wang, Daniel E. Ho, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2509.19364)  

**Abstract**: Standard offline evaluations for language models -- a series of independent, state-less inferences made by models -- fail to capture how language models actually behave in practice, where personalization fundamentally alters model behavior. For instance, identical benchmark questions to the same language model can produce markedly different responses when prompted to a state-less system, in one user's chat session, or in a different user's chat session. In this work, we provide empirical evidence showcasing this phenomenon by comparing offline evaluations to field evaluations conducted by having 800 real users of ChatGPT and Gemini pose benchmark and other provided questions to their chat interfaces. 

---
# Retrieval Augmented Generation based context discovery for ASR 

**Authors**: Dimitrios Siskos, Stavros Papadopoulos, Pablo Peso Parada, Jisi Zhang, Karthikeyan Saravanan, Anastasios Drosou  

**Link**: [PDF](https://arxiv.org/pdf/2509.19567)  

**Abstract**: This work investigates retrieval augmented generation as an efficient strategy for automatic context discovery in context-aware Automatic Speech Recognition (ASR) system, in order to improve transcription accuracy in the presence of rare or out-of-vocabulary terms. However, identifying the right context automatically remains an open challenge. This work proposes an efficient embedding-based retrieval approach for automatic context discovery in ASR. To contextualize its effectiveness, two alternatives based on large language models (LLMs) are also evaluated: (1) large language model (LLM)-based context generation via prompting, and (2) post-recognition transcript correction using LLMs. Experiments on the TED-LIUMv3, Earnings21 and SPGISpeech demonstrate that the proposed approach reduces WER by up to 17% (percentage difference) relative to using no-context, while the oracle context results in a reduction of up to 24.1%. 

---
# Semantic Representation Attack against Aligned Large Language Models 

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Shaohui Mei, Lap-Pui Chau  

**Link**: [PDF](https://arxiv.org/pdf/2509.19360)  

**Abstract**: Large Language Models (LLMs) increasingly employ alignment techniques to prevent harmful outputs. Despite these safeguards, attackers can circumvent them by crafting prompts that induce LLMs to generate harmful content.
Current methods typically target exact affirmative responses, such as ``Sure, here is...'', suffering from limited convergence, unnatural prompts, and high computational costs.
We introduce Semantic Representation Attack, a novel paradigm that fundamentally reconceptualizes adversarial objectives against aligned LLMs.
Rather than targeting exact textual patterns, our approach exploits the semantic representation space comprising diverse responses with equivalent harmful meanings.
This innovation resolves the inherent trade-off between attack efficacy and prompt naturalness that plagues existing methods.
The Semantic Representation Heuristic Search algorithm is proposed to efficiently generate semantically coherent and concise adversarial prompts by maintaining interpretability during incremental expansion.
We establish rigorous theoretical guarantees for semantic convergence and demonstrate that our method achieves unprecedented attack success rates (89.41\% averaged across 18 LLMs, including 100\% on 11 models) while maintaining stealthiness and efficiency.
Comprehensive experimental results confirm the overall superiority of our Semantic Representation Attack.
The code will be publicly available. 

---
# Polarity Detection of Sustainable Detection Goals in News Text 

**Authors**: Andrea Cadeddua, Alessandro Chessa, Vincenzo De Leo, Gianni Fenu, Francesco Osborne, Diego Reforgiato Recupero, Angelo Salatino, Luca Secchi  

**Link**: [PDF](https://arxiv.org/pdf/2509.19833)  

**Abstract**: The United Nations' Sustainable Development Goals (SDGs) provide a globally recognised framework for addressing critical societal, environmental, and economic challenges. Recent developments in natural language processing (NLP) and large language models (LLMs) have facilitated the automatic classification of textual data according to their relevance to specific SDGs. Nevertheless, in many applications, it is equally important to determine the directionality of this relevance; that is, to assess whether the described impact is positive, neutral, or negative. To tackle this challenge, we propose the novel task of SDG polarity detection, which assesses whether a text segment indicates progress toward a specific SDG or conveys an intention to achieve such progress. To support research in this area, we introduce SDG-POD, a benchmark dataset designed specifically for this task, combining original and synthetically generated data. We perform a comprehensive evaluation using six state-of-the-art large LLMs, considering both zero-shot and fine-tuned configurations. Our results suggest that the task remains challenging for the current generation of LLMs. Nevertheless, some fine-tuned models, particularly QWQ-32B, achieve good performance, especially on specific Sustainable Development Goals such as SDG-9 (Industry, Innovation and Infrastructure), SDG-12 (Responsible Consumption and Production), and SDG-15 (Life on Land). Furthermore, we demonstrate that augmenting the fine-tuning dataset with synthetically generated examples yields improved model performance on this task. This result highlights the effectiveness of data enrichment techniques in addressing the challenges of this resource-constrained domain. This work advances the methodological toolkit for sustainability monitoring and provides actionable insights into the development of efficient, high-performing polarity detection systems. 

---
# RoadMind: Towards a Geospatial AI Expert for Disaster Response 

**Authors**: Ahmed El Fekih Zguir, Ferda Ofli, Muhammad Imran  

**Link**: [PDF](https://arxiv.org/pdf/2509.19354)  

**Abstract**: Large Language Models (LLMs) have shown impressive performance across a range of natural language tasks, but remain limited in their ability to reason about geospatial data, particularly road networks, distances, and directions. This gap poses challenges in disaster scenarios, where spatial understanding is critical for tasks such as evacuation planning and resource allocation. In this work, we present RoadMind, a self-supervised framework that enhances the geospatial reasoning capabilities of LLMs using structured data from OpenStreetMap (OSM). Our automated pipeline extracts road infrastructure data for a given city and converts it into multiple supervision formats tailored to key spatial tasks. We pretrain and fine-tune LLMs on these representations using QLoRA adapters and 4-bit quantized models. We evaluate our approach on three disaster-prone cities with varying global representation, Los Angeles, Christchurch, and Manila, across tasks such as road segment identification, nearest road retrieval, and distance/direction estimation. Our results show that models trained via RoadMind significantly outperform strong baselines, including state-of-the-art LLMs equipped with advanced prompt engineering. This demonstrates the potential of structured geospatial data to enhance language models with robust spatial reasoning, enabling more effective offline AI systems for disaster response. 

---
# ShinkaEvolve: Towards Open-Ended And Sample-Efficient Program Evolution 

**Authors**: Robert Tjarko Lange, Yuki Imajuku, Edoardo Cetin  

**Link**: [PDF](https://arxiv.org/pdf/2509.19349)  

**Abstract**: We introduce ShinkaEvolve: a new open-source framework leveraging large language models (LLMs) to advance scientific discovery with state-of-the-art performance and unprecedented efficiency. Recent advances in scaling inference time compute of LLMs have enabled significant progress in generalized scientific discovery. These approaches rely on evolutionary agentic harnesses that leverage LLMs as mutation operators to generate candidate solutions. However, current code evolution methods suffer from critical limitations: they are sample inefficient, requiring thousands of samples to identify effective solutions, and remain closed-source, hindering broad adoption and extension. ShinkaEvolve addresses these limitations, introducing three key innovations: a parent sampling technique balancing exploration and exploitation, code novelty rejection-sampling for efficient search space exploration, and a bandit-based LLM ensemble selection strategy. We evaluate ShinkaEvolve across diverse tasks, demonstrating consistent improvements in sample efficiency and solution quality. ShinkaEvolve discovers a new state-of-the-art circle packing solution using only 150 samples, designs high-performing agentic harnesses for AIME mathematical reasoning tasks, identifies improvements to ALE-Bench competitive programming solutions, and discovers novel mixture-of-expert load balancing loss functions that illuminate the space of optimization strategies. Our results demonstrate that ShinkaEvolve achieves broad applicability with exceptional sample efficiency. By providing open-source accessibility and cost-efficiency, this work democratizes open-ended discovery across diverse computational problems. 

---
# From Text to Talk: Audio-Language Model Needs Non-Autoregressive Joint Training 

**Authors**: Tianqiao Liu, Xueyi Li, Hao Wang, Haoxuan Li, Zhichao Chen, Weiqi Luo, Zitao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20072)  

**Abstract**: Recent advances in large language models have attracted significant interest in extending their capabilities to multimodal scenarios, particularly for speech-in speech-out conversational systems. However, existing multimodal models handling interleaved audio and text, such as MOSHI require complex multi stage training pipelines, incurring substantial computational costs. Moreover, these models uniformly apply autoregressive generation to both text and audio tokens, overlooking a fundamental asymmetry in their dependency structures: while text tokens exhibit strong target target dependencies requiring causal ordering, audio tokens are predominantly driven by source target dependencies, where audio outputs primarily condition on source text rather than preceding audio tokens. In this work, we propose TtT, a unified audio-text modeling framework that integrates AR text generation with non-autoregressive audio diffusion within a single Transformer architecture initialized from a pretrained LLM. 

---
# Benchmarking ChatGPT and DeepSeek in April 2025: A Novel Dual Perspective Sentiment Analysis Using Lexicon-Based and Deep Learning Approaches 

**Authors**: Maryam Mahdi Alhusseini, Mohammad-Reza Feizi-Derakhshi  

**Link**: [PDF](https://arxiv.org/pdf/2509.19346)  

**Abstract**: This study presents a novel dual-perspective approach to analyzing user reviews for ChatGPT and DeepSeek on the Google Play Store, integrating lexicon-based sentiment analysis (TextBlob) with deep learning classification models, including Convolutional Neural Networks (CNN) and Bidirectional Long Short Term Memory (Bi LSTM) Networks. Unlike prior research, which focuses on either lexicon-based strategies or predictive deep learning models in isolation, this study conducts an extensive investigation into user satisfaction with Large Language Model (LLM) based applications. A Dataset of 4,000 authentic user reviews was collected, which were carefully preprocessed and subjected to oversampling to achieve balanced classes. The balanced test set of 1,700 Reviews were used for model testing. Results from the experiments reveal that ChatGPT received significantly more positive sentiment than DeepSeek. Furthermore, deep learning based classification demonstrated superior performance over lexicon analysis, with CNN outperforming Bi-LSTM by achieving 96.41 percent accuracy and near perfect classification of negative reviews, alongside high F1-scores for neutral and positive sentiments. This research sets a new methodological standard for measuring sentiment in LLM-based applications and provides practical insights for developers and researchers seeking to improve user-centric AI system design. 

---
# Benchmarking and Improving LLM Robustness for Personalized Generation 

**Authors**: Chimaobi Okite, Naihao Deng, Kiran Bodipati, Huaidian Hou, Joyce Chai, Rada Mihalcea  

**Link**: [PDF](https://arxiv.org/pdf/2509.19358)  

**Abstract**: Recent years have witnessed a growing interest in personalizing the responses of large language models (LLMs). While existing evaluations primarily focus on whether a response aligns with a user's preferences, we argue that factuality is an equally important yet often overlooked dimension. In the context of personalization, we define a model as robust if its responses are both factually accurate and align with the user preferences. To assess this, we introduce PERG, a scalable framework for evaluating robustness in LLMs, along with a new dataset, PERGData. We evaluate fourteen models from five different model families using different prompting methods. Our findings show that current LLMs struggle with robust personalization: even the strongest models (GPT-4.1, LLaMA3-70B) fail to maintain correctness in 5% of previously successful cases without personalization, while smaller models (e.g., 7B-scale) can fail more than 20% of the time. Further analysis reveals that robustness is significantly affected by the nature of the query and the type of user preference. To mitigate these failures, we propose Pref-Aligner, a two-stage approach that improves robustness by an average of 25% across models. Our work highlights critical gaps in current evaluation practices and introduces tools and metrics to support more reliable, user-aligned LLM deployments. 

---
# LLM-Assisted Topic Reduction for BERTopic on Social Media Data 

**Authors**: Wannes Janssens, Matthias Bogaert, Dirk Van den Poel  

**Link**: [PDF](https://arxiv.org/pdf/2509.19365)  

**Abstract**: The BERTopic framework leverages transformer embeddings and hierarchical clustering to extract latent topics from unstructured text corpora. While effective, it often struggles with social media data, which tends to be noisy and sparse, resulting in an excessive number of overlapping topics. Recent work explored the use of large language models for end-to-end topic modelling. However, these approaches typically require significant computational overhead, limiting their scalability in big data contexts. In this work, we propose a framework that combines BERTopic for topic generation with large language models for topic reduction. The method first generates an initial set of topics and constructs a representation for each. These representations are then provided as input to the language model, which iteratively identifies and merges semantically similar topics. We evaluate the approach across three Twitter/X datasets and four different language models. Our method outperforms the baseline approach in enhancing topic diversity and, in many cases, coherence, with some sensitivity to dataset characteristics and initial parameter selection. 

---
# Cognitive-Level Adaptive Generation via Capability-Aware Retrieval and Style Adaptation 

**Authors**: Qingsong Wang, Tao Wu, Wang Lin, Yueying Feng, Gongsheng Yuan, Chang Yao, Jingyuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.19336)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong performance in open-ended generation tasks. However, they often struggle to adapt content to users with differing cognitive capacities, leading to a phenomenon we term cognitive misalignment. This issue arises in two forms: knowledge-level misalignment, where content is too complex or too simplistic relative to user understanding, and presentation-style misalignment, where the structure or tone hinders effective comprehension. To address these challenges, we propose the Cognitive-Level Alignment Framework (CLAF), a general-purpose generation framework that aligns both knowledge complexity and presentation style with user cognition. CLAF integrates a capability-aware retrieval module based on a hierarchical knowledge graph and a style optimization module guided by Bloom's taxonomy and preference learning. Additionally, a knowledge-controllable generation component ensures consistency and relevance throughout the output. To support training and evaluation, we construct SCALE, a cognitively annotated dataset containing responses at multiple comprehension levels per query. Empirical results show that CLAF enhances the adaptability and informativeness of LLM outputs across a range of user profiles, offering a robust solution to cognitive-level alignment in real-world applications. 

---
# How Model Size, Temperature, and Prompt Style Affect LLM-Human Assessment Score Alignment 

**Authors**: Julie Jung, Max Lu, Sina Chole Benker, Dogus Darici  

**Link**: [PDF](https://arxiv.org/pdf/2509.19329)  

**Abstract**: We examined how model size, temperature, and prompt style affect Large Language Models' (LLMs) alignment within itself, between models, and with human in assessing clinical reasoning skills. Model size emerged as a key factor in LLM-human score alignment. Study highlights the importance of checking alignments across multiple levels. 

---
# How Much of Your Data Can Suck? Thresholds for Domain Performance and Emergent Misalignment in LLMs 

**Authors**: Jian Ouyang, Arman T, Ge Jin  

**Link**: [PDF](https://arxiv.org/pdf/2509.19325)  

**Abstract**: This paper investigates the impact of incorrect data on the performance and safety of large language models (LLMs), specifically gpt-4o, during supervised fine-tuning (SFT). Although LLMs become increasingly vital across broad domains like finance, coding, law, and health, fine-tuning on incorrect data can lead to "emergent misalignment," producing harmful or deceptive outputs unrelated to the intended task. We evaluate gpt-4o models fine-tuned with varying ratios (10\% to 90\% correct) of both obviously and subtly incorrect data across four domains: coding, finance, health, and legal. Our findings show that even modest amounts of incorrect data (10-25\%) dramatically degrade domain performance and not moral alignment. A clear threshold of at least 50\% correct data is needed for models to consistently recover strong performance, though they rarely match the robustness and safety of the base model, which exhibits near-perfect alignment and zero dangerous completions out-of-the-box. This research emphasizes that the cost of incorrect data is heavy, highlighting the critical need for extremely high-quality data curation or, alternatively, leveraging robust base models without unnecessary fine-tuning for high-stakes applications. 

---
# FHIR-AgentBench: Benchmarking LLM Agents for Realistic Interoperable EHR Question Answering 

**Authors**: Gyubok Lee, Elea Bach, Eric Yang, Tom Pollard, Alistair Johnson, Edward Choi, Yugang jia, Jong Ha Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.19319)  

**Abstract**: The recent shift toward the Health Level Seven Fast Healthcare Interoperability Resources (HL7 FHIR) standard opens a new frontier for clinical AI, demanding LLM agents to navigate complex, resource-based data models instead of conventional structured health data. However, existing benchmarks have lagged behind this transition, lacking the realism needed to evaluate recent LLMs on interoperable clinical data. To bridge this gap, we introduce FHIR-AgentBench, a benchmark that grounds 2,931 real-world clinical questions in the HL7 FHIR standard. Using this benchmark, we systematically evaluate agentic frameworks, comparing different data retrieval strategies (direct FHIR API calls vs. specialized tools), interaction patterns (single-turn vs. multi-turn), and reasoning strategies (natural language vs. code generation). Our experiments highlight the practical challenges of retrieving data from intricate FHIR resources and the difficulty of reasoning over them, both of which critically affect question answering performance. We publicly release the FHIR-AgentBench dataset and evaluation suite (this https URL) to promote reproducible research and the development of robust, reliable LLM agents for clinical applications. 

---
# Scan-do Attitude: Towards Autonomous CT Protocol Management using a Large Language Model Agent 

**Authors**: Xingjian Kang, Linda Vorberg, Andreas Maier, Alexander Katzmann, Oliver Taubmann  

**Link**: [PDF](https://arxiv.org/pdf/2509.20270)  

**Abstract**: Managing scan protocols in Computed Tomography (CT), which includes adjusting acquisition parameters or configuring reconstructions, as well as selecting postprocessing tools in a patient-specific manner, is time-consuming and requires clinical as well as technical expertise. At the same time, we observe an increasing shortage of skilled workforce in radiology. To address this issue, a Large Language Model (LLM)-based agent framework is proposed to assist with the interpretation and execution of protocol configuration requests given in natural language or a structured, device-independent format, aiming to improve the workflow efficiency and reduce technologists' workload. The agent combines in-context-learning, instruction-following, and structured toolcalling abilities to identify relevant protocol elements and apply accurate modifications. In a systematic evaluation, experimental results indicate that the agent can effectively retrieve protocol components, generate device compatible protocol definition files, and faithfully implement user requests. Despite demonstrating feasibility in principle, the approach faces limitations regarding syntactic and semantic validity due to lack of a unified device API, and challenges with ambiguous or complex requests. In summary, the findings show a clear path towards LLM-based agents for supporting scan protocol management in CT imaging. 

---
# Automated Item Neutralization for Non-Cognitive Scales: A Large Language Model Approach to Reducing Social-Desirability Bias 

**Authors**: Sirui Wu, Daijin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19314)  

**Abstract**: This study evaluates item neutralization assisted by the large language model (LLM) to reduce social desirability bias in personality assessment. GPT-o3 was used to rewrite the International Personality Item Pool Big Five Measure (IPIP-BFM-50), and 203 participants completed either the original or neutralized form along with the Marlowe-Crowne Social Desirability Scale. The results showed preserved reliability and a five-factor structure, with gains in Conscientiousness and declines in Agreeableness and Openness. The correlations with social desirability decreased for several items, but inconsistently. Configural invariance held, though metric and scalar invariance failed. Findings support AI neutralization as a potential but imperfect bias-reduction method. 

---
# A systematic review of trial-matching pipelines using large language models 

**Authors**: Braxton A. Morrison, Madhumita Sushil, Jacob S. Young  

**Link**: [PDF](https://arxiv.org/pdf/2509.19327)  

**Abstract**: Matching patients to clinical trial options is critical for identifying novel treatments, especially in oncology. However, manual matching is labor-intensive and error-prone, leading to recruitment delays. Pipelines incorporating large language models (LLMs) offer a promising solution. We conducted a systematic review of studies published between 2020 and 2025 from three academic databases and one preprint server, identifying LLM-based approaches to clinical trial matching. Of 126 unique articles, 31 met inclusion criteria. Reviewed studies focused on matching patient-to-criterion only (n=4), patient-to-trial only (n=10), trial-to-patient only (n=2), binary eligibility classification only (n=1) or combined tasks (n=14). Sixteen used synthetic data; fourteen used real patient data; one used both. Variability in datasets and evaluation metrics limited cross-study comparability. In studies with direct comparisons, the GPT-4 model consistently outperformed other models, even finely-tuned ones, in matching and eligibility extraction, albeit at higher cost. Promising strategies included zero-shot prompting with proprietary LLMs like the GPT-4o model, advanced retrieval methods, and fine-tuning smaller, open-source models for data privacy when incorporation of large models into hospital infrastructure is infeasible. Key challenges include accessing sufficiently large real-world data sets, and deployment-associated challenges such as reducing cost, mitigating risk of hallucinations, data leakage, and bias. This review synthesizes progress in applying LLMs to clinical trial matching, highlighting promising directions and key limitations. Standardized metrics, more realistic test sets, and attention to cost-efficiency and fairness will be critical for broader deployment. 

---
# Embodied AI: From LLMs to World Models 

**Authors**: Tongtong Feng, Xin Wang, Yu-Gang Jiang, Wenwu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20021)  

**Abstract**: Embodied Artificial Intelligence (AI) is an intelligent system paradigm for achieving Artificial General Intelligence (AGI), serving as the cornerstone for various applications and driving the evolution from cyberspace to physical systems. Recent breakthroughs in Large Language Models (LLMs) and World Models (WMs) have drawn significant attention for embodied AI. On the one hand, LLMs empower embodied AI via semantic reasoning and task decomposition, bringing high-level natural language instructions and low-level natural language actions into embodied cognition. On the other hand, WMs empower embodied AI by building internal representations and future predictions of the external world, facilitating physical law-compliant embodied interactions. As such, this paper comprehensively explores the literature in embodied AI from basics to advances, covering both LLM driven and WM driven works. In particular, we first present the history, key technologies, key components, and hardware systems of embodied AI, as well as discuss its development via looking from unimodal to multimodal angle. We then scrutinize the two burgeoning fields of embodied AI, i.e., embodied AI with LLMs/multimodal LLMs (MLLMs) and embodied AI with WMs, meticulously delineating their indispensable roles in end-to-end embodied cognition and physical laws-driven embodied interactions. Building upon the above advances, we further share our insights on the necessity of the joint MLLM-WM driven embodied AI architecture, shedding light on its profound significance in enabling complex tasks within physical worlds. In addition, we examine representative applications of embodied AI, demonstrating its wide applicability in real-world scenarios. Last but not least, we point out future research directions of embodied AI that deserve further investigation. 

---
# PromptCoT 2.0: Scaling Prompt Synthesis for Large Language Model Reasoning 

**Authors**: Xueliang Zhao, Wei Wu, Jian Guan, Zhuocheng Gong, Lingpeng Kong  

**Link**: [PDF](https://arxiv.org/pdf/2509.19894)  

**Abstract**: Large language models (LLMs) are evolving from conversational systems into strong reasoners for tasks such as Olympiad mathematics and competitive programming. While scaling parameters and test-time computation has driven progress, a key bottleneck is the lack of high-quality training problems: human-curated datasets are costly and limited, while existing synthetic corpora are often too easy or narrow. PromptCoT 1.0 showed that injecting rationales into prompt synthesis increases problem difficulty. Building on this, we present PromptCoT 2.0, a scalable framework that replaces hand-crafted heuristics with an expectation-maximization (EM) loop, where rationales are iteratively refined to guide prompt construction. This produces problems that are both harder and more diverse than prior corpora. The synthetic prompts support two post-training regimes: (1) Self-Play, where strong models improve autonomously via verifiable feedback without stronger teachers; and (2) Supervised Fine-Tuning (SFT), where weaker models learn from teacher-distilled traces. Extensive experiments demonstrate the effectiveness of this approach. In self-play, applying PromptCoT 2.0 to Qwen3-30B-A3B-Thinking-2507 sets new state-of-the-art results at the 30B scale, with +4.4, +4.8, and +5.3 on AIME 24/25 and HMMT 25, +6.1 and +5.0 on LiveCodeBench v5/v6, and +35 Elo on Codeforces. In SFT, training Qwen2.5-7B-Instruct solely on synthetic prompts boosts accuracy to 73.1 (AIME 24), 65.6 (AIME 25), and 53.4 (LiveCodeBench v5), surpassing models trained on human or hybrid data. Analyses further confirm that PromptCoT 2.0 yields fundamentally harder and distributionally distinct problems. These results establish prompt synthesis as a new axis for scaling reasoning and position PromptCoT 2.0 as a scalable foundation for future open-source models. The implementation is available at this https URL. 

---
# VCRL: Variance-based Curriculum Reinforcement Learning for Large Language Models 

**Authors**: Guochao Jiang, Wenfeng Feng, Guofeng Quan, Chuzhan Hao, Yuewei Zhang, Guohua Liu, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19803)  

**Abstract**: Policy-based reinforcement learning currently plays an important role in improving LLMs on mathematical reasoning tasks. However, existing rollout-based reinforcement learning methods (GRPO, DAPO, GSPO, etc.) fail to explicitly consider LLMs' learning ability for samples of different difficulty levels, which is contrary to the human cognitive process of mathematical reasoning tasks from easy to difficult. Intuitively, we find that the variance of the rollout group's reward in RLVR partly reflects the difficulty of the current sample for LLMs. Samples that are too easy or too difficult have a lower variance, while samples with moderate difficulty have a higher variance. Based on this, we propose VCRL, a curriculum reinforcement learning framework that dynamically controls the difficulty of training samples based on the variance of group rewards. Experiments on five mathematical benchmarks and two models reveal the advantages of VCRL over the current LLM RL baselines. 

---
# Advancing Speech Summarization in Multi-modal LLMs with Reinforcement Learning 

**Authors**: Shaoshi Ling, Gang Liu, Guoli Ye, Jinyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.19631)  

**Abstract**: Speech summarization is a critical component of spoken content understanding, particularly in the era of rapidly growing spoken and audiovisual data. Recent advances in multi-modal large language models (MLLMs), leveraging the power of LLMs, enable generating textual summaries directly from speech without intermediate transcriptions, while supporting controllable styles and zero-shot generalization. However, open-source MLLMs continue to lag behind the state-of-the-art text-based LLMs, limiting their practical deployment for speech summarization. In this work, we present a novel multi-stage reinforcement learning training framework to enhance the speech summarization capabilities in MLLMs. Our model delivers substantial improvements over strong baselines, outperforms much larger MLLMs, and significantly narrows the gap with state-of-the-art text-based LLMs. 

---
# Readme_AI: Dynamic Context Construction for Large Language Models 

**Authors**: Millie Vyas, Timothy Blattner, Alden Dima  

**Link**: [PDF](https://arxiv.org/pdf/2509.19322)  

**Abstract**: Despite being trained on significant amounts of data, Large Language Models (LLMs) can provide inaccurate or unreliable information in the context of a user's specific query. Given query-specific context significantly improves the usefulness of its responses. In this paper, we present a specification that can be used to dynamically build context for data sources. The data source owner creates the file containing metadata for LLMs to use when reasoning about dataset-related queries. To demonstrate our proposed specification, we created a prototype Readme_AI Model Context Protocol (MCP) server that retrieves the metadata from the data source and uses it to dynamically build context. Some features that make this specification dynamic are the extensible types that represent crawling web-pages, fetching data from data repositories, downloading and parsing publications, and general text. The context is formatted and grouped using user-specified tags that provide clear contextual information for the LLM to reason about the content. We demonstrate the capabilities of this early prototype by asking the LLM about the NIST-developed Hedgehog library, for which common LLMs often provides inaccurate and irrelevant responses containing hallucinations. With Readme_AI, the LLM receives enough context that it is now able to reason about the library and its use, and even generate code interpolated from examples that were included in the Readme_AI file provided by Hedgehog's developer. Our primary contribution is a extensible protocol for dynamically grounding LLMs in specialized, owner-provided data, enhancing responses from LLMs and reducing hallucinations. The source code for the Readme_AI tool is posted here: this https URL . 

---
# Unveiling the Merits and Defects of LLMs in Automatic Review Generation for Scientific Papers 

**Authors**: Ruochi Li, Haoxuan Zhang, Edward Gehringer, Ting Xiao, Junhua Ding, Haihua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.19326)  

**Abstract**: The surge in scientific submissions has placed increasing strain on the traditional peer-review process, prompting the exploration of large language models (LLMs) for automated review generation. While LLMs demonstrate competence in producing structured and coherent feedback, their capacity for critical reasoning, contextual grounding, and quality sensitivity remains limited. To systematically evaluate these aspects, we propose a comprehensive evaluation framework that integrates semantic similarity analysis and structured knowledge graph metrics to assess LLM-generated reviews against human-written counterparts. We construct a large-scale benchmark of 1,683 papers and 6,495 expert reviews from ICLR and NeurIPS in multiple years, and generate reviews using five LLMs. Our findings show that LLMs perform well in descriptive and affirmational content, capturing the main contributions and methodologies of the original work, with GPT-4o highlighted as an illustrative example, generating 15.74% more entities than human reviewers in the strengths section of good papers in ICLR 2025. However, they consistently underperform in identifying weaknesses, raising substantive questions, and adjusting feedback based on paper quality. GPT-4o produces 59.42% fewer entities than real reviewers in the weaknesses and increases node count by only 5.7% from good to weak papers, compared to 50% in human reviews. Similar trends are observed across all conferences, years, and models, providing empirical foundations for understanding the merits and defects of LLM-generated reviews and informing the development of future LLM-assisted reviewing tools. Data, code, and more detailed results are publicly available at this https URL. 

---
# Cognitive Load Limits in Large Language Models: Benchmarking Multi-Hop Reasoning 

**Authors**: Sai Teja Reddy Adapala  

**Link**: [PDF](https://arxiv.org/pdf/2509.19517)  

**Abstract**: The scaling of Large Language Models (LLMs) has exposed a critical gap between their performance on static benchmarks and their fragility in dynamic, information-rich environments. While models excel at isolated tasks, the computational limits that govern their reasoning under cognitive load remain poorly understood. In this work, we introduce a formal theory of computational cognitive load, positing that extraneous, task-irrelevant information (Context Saturation) and interference from task-switching (Attentional Residue) are key mechanisms that degrade performance. We designed the Interleaved Cognitive Evaluation (ICE), a deconfounded benchmark to systematically manipulate these load factors on challenging multi-hop reasoning tasks. A comprehensive study (N = 10 replications per item across 200 questions) revealed significant performance variations across five instruction-tuned models. Smaller open-source architectures (Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.2) exhibited baseline brittleness, achieving 0% accuracy (SEM = 0.0) across all conditions, including clean controls, on this high-intrinsic-load task. In contrast, Gemini-2.0-Flash-001 showed partial resilience, achieving 85% accuracy in control conditions, with a statistically significant degradation under context saturation ($\beta = -0.003$ per % load, $p < 0.001$). These findings provide preliminary evidence that cognitive load is a key contributor to reasoning failures, supporting theories of hallucination-as-guessing under uncertainty. We conclude that dynamic, cognitive-aware stress testing, as exemplified by the ICE benchmark, is essential for evaluating the true resilience and safety of advanced AI systems. 

---
# STARQA: A Question Answering Dataset for Complex Analytical Reasoning over Structured Databases 

**Authors**: Mounica Maddela, Lingjue Xie, Daniel Preotiuc-Pietro, Mausam  

**Link**: [PDF](https://arxiv.org/pdf/2509.19508)  

**Abstract**: Semantic parsing methods for converting text to SQL queries enable question answering over structured data and can greatly benefit analysts who routinely perform complex analytics on vast data stored in specialized relational databases. Although several benchmarks measure the abilities of text to SQL, the complexity of their questions is inherently limited by the level of expressiveness in query languages and none focus explicitly on questions involving complex analytical reasoning which require operations such as calculations over aggregate analytics, time series analysis or scenario understanding. In this paper, we introduce STARQA, the first public human-created dataset of complex analytical reasoning questions and answers on three specialized-domain databases. In addition to generating SQL directly using LLMs, we evaluate a novel approach (Text2SQLCode) that decomposes the task into a combination of SQL and Python: SQL is responsible for data fetching, and Python more naturally performs reasoning. Our results demonstrate that identifying and combining the abilities of SQL and Python is beneficial compared to using SQL alone, yet the dataset still remains quite challenging for the existing state-of-the-art LLMs. 

---
# MACD: Multi-Agent Clinical Diagnosis with Self-Learned Knowledge for LLM 

**Authors**: Wenliang Li, Rui Yan, Xu Zhang, Li Chen, Hongji Zhu, Jing Zhao, Junjun Li, Mengru Li, Wei Cao, Zihang Jiang, Wei Wei, Kun Zhang, Shaohua Kevin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.20067)  

**Abstract**: Large language models (LLMs) have demonstrated notable potential in medical applications, yet they face substantial challenges in handling complex real-world clinical diagnoses using conventional prompting methods. Current prompt engineering and multi-agent approaches typically optimize isolated inferences, neglecting the accumulation of reusable clinical experience. To address this, this study proposes a novel Multi-Agent Clinical Diagnosis (MACD) framework, which allows LLMs to self-learn clinical knowledge via a multi-agent pipeline that summarizes, refines, and applies diagnostic insights. It mirrors how physicians develop expertise through experience, enabling more focused and accurate diagnosis on key disease-specific cues. We further extend it to a MACD-human collaborative workflow, where multiple LLM-based diagnostician agents engage in iterative consultations, supported by an evaluator agent and human oversight for cases where agreement is not reached. Evaluated on 4,390 real-world patient cases across seven diseases using diverse open-source LLMs (Llama-3.1 8B/70B, DeepSeek-R1-Distill-Llama 70B), MACD significantly improves primary diagnostic accuracy, outperforming established clinical guidelines with gains up to 22.3% (MACD). On the subset of the data, it achieves performance on par with or exceeding that of human physicians (up to 16% improvement over physicians-only diagnosis). Additionally, on the MACD-human workflow, it achieves an 18.6% improvement compared to physicians-only diagnosis. Moreover, self-learned knowledge exhibits strong cross-model stability, transferability, and model-specific personalization, while the system can generate traceable rationales, enhancing explainability. Consequently, this work presents a scalable self-learning paradigm for LLM-assisted diagnosis, bridging the gap between the intrinsic knowledge of LLMs and real-world clinical practice. 

---
# CON-QA: Privacy-Preserving QA using cloud LLMs in Contract Domain 

**Authors**: Ajeet Kumar Singh, Rajsabi Surya, Anurag Tripathi, Santanu Choudhury, Sudhir Bisane  

**Link**: [PDF](https://arxiv.org/pdf/2509.19925)  

**Abstract**: As enterprises increasingly integrate cloud-based large language models (LLMs) such as ChatGPT and Gemini into their legal document workflows, protecting sensitive contractual information - including Personally Identifiable Information (PII) and commercially sensitive clauses - has emerged as a critical challenge. In this work, we propose CON-QA, a hybrid privacy-preserving framework designed specifically for secure question answering over enterprise contracts, effectively combining local and cloud-hosted LLMs. The CON-QA framework operates through three stages: (i) semantic query decomposition and query-aware document chunk retrieval using a locally deployed LLM analysis, (ii) anonymization of detected sensitive entities via a structured one-to-many mapping scheme, ensuring semantic coherence while preventing cross-session entity inference attacks, and (iii) anonymized response generation by a cloud-based LLM, with accurate reconstruction of the original answer locally using a session-consistent many-to-one reverse mapping. To rigorously evaluate CON-QA, we introduce CUAD-QA, a corpus of 85k question-answer pairs generated over 510 real-world CUAD contract documents, encompassing simple, complex, and summarization-style queries. Empirical evaluations, complemented by detailed human assessments, confirm that CON-QA effectively maintains both privacy and utility, preserves answer quality, maintains fidelity to legal clause semantics, and significantly mitigates privacy risks, demonstrating its practical suitability for secure, enterprise-level contract documents. 

---
# The Conductor and the Engine: A Path Towards Co-Designed Reasoning 

**Authors**: Yuanxin Wang, Pawel Filipczuk, Anisha Garg, Amaan Dhada, Mohammad Hassanpour, David Bick, Ganesh Venkatesh  

**Link**: [PDF](https://arxiv.org/pdf/2509.19762)  

**Abstract**: Modern LLM reasoning relies on extensive test-time computation, driven by internal model training and external agentic orchestration. However, this synergy is often inefficient, as model verbosity and poor instruction following lead to wasted compute. We analyze this capability-cost trade-off and introduce an optimized reasoning workflow (\cepo) that empowers smaller open-source models to outperform models multiple times their size. We will open-source this workflow to enable further research. Our work demonstrates a clear path toward co-designing orchestration frameworks with the underlying model capabilities to unlock powerful reasoning in small-to-medium sized models. 

---
# SteinerSQL: Graph-Guided Mathematical Reasoning for Text-to-SQL Generation 

**Authors**: Xutao Mao, Tao Liu, Hongying Zan  

**Link**: [PDF](https://arxiv.org/pdf/2509.19623)  

**Abstract**: Large Language Models (LLMs) struggle with complex Text-to-SQL queries that demand both sophisticated mathematical reasoning and intricate schema navigation. Existing methods often tackle these challenges in isolation, creating a fractured reasoning process that compromises logical and structural correctness. To resolve this, we introduce SteinerSQL, a framework that unifies these dual challenges into a single, graph-centric optimization problem. SteinerSQL operates in three stages: mathematical decomposition to identify required tables (terminals), optimal reasoning scaffold construction via a Steiner tree problem, and multi-level validation to ensure correctness. On the challenging LogicCat and Spider2.0-Lite benchmarks, SteinerSQL establishes a new state-of-the-art with 36.10% and 40.04% execution accuracy, respectively, using Gemini-2.5-Pro. Beyond accuracy, SteinerSQL presents a new, unified paradigm for Text-to-SQL, paving the way for more robust and principled solutions to complex reasoning tasks. 

---
# What Does Your Benchmark Really Measure? A Framework for Robust Inference of AI Capabilities 

**Authors**: Nathanael Jo, Ashia Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2509.19590)  

**Abstract**: Evaluations of generative models on benchmark data are now ubiquitous, and their outcomes critically shape public and scientific expectations of AI's capabilities. Yet growing skepticism surrounds their reliability. How can we know that a reported accuracy genuinely reflects a model's true performance? Evaluations are often presented as simple measurements, but in reality they are inferences: to treat benchmark scores as evidence of capability is already to assume a theory of what capability is and how it manifests in a test. We make this step explicit by proposing a principled framework for evaluation as inference: begin from a theory of capability, and then derive methods for estimating it. This perspective, familiar in fields such as psychometrics, has not yet become commonplace in AI evaluation. As a proof of concept, we address a central challenge that undermines reliability: sensitivity to perturbations. After formulating a model of ability, we introduce methods that infer ability while accounting for uncertainty from sensitivity and finite samples, including an adaptive algorithm that significantly reduces sample complexity. Together, these contributions lay the groundwork for more reliable and trustworthy estimates of AI capabilities as measured through benchmarks. 

---
# LatentGuard: Controllable Latent Steering for Robust Refusal of Attacks and Reliable Response Generation 

**Authors**: Huizhen Shu, Xuying Li, Zhuo Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.19839)  

**Abstract**: Achieving robust safety alignment in large language models (LLMs) while preserving their utility remains a fundamental challenge. Existing approaches often struggle to balance comprehensive safety with fine-grained controllability at the representation level. We introduce LATENTGUARD, a novel three-stage framework that combines behavioral alignment with supervised latent space control for interpretable and precise safety steering. Our approach begins by fine-tuning an LLM on rationalized datasets containing both reasoning-enhanced refusal responses to adversarial prompts and reasoning-enhanced normal responses to benign queries, establishing robust behavioral priors across both safety-critical and utility-preserving scenarios. We then train a structured variational autoencoder (VAE) on intermediate MLP activations, supervised by multi-label annotations including attack types, attack methods, and benign indicators. This supervision enables the VAE to learn disentangled latent representations that capture distinct adversarial characteristics while maintaining semantic interpretability. Through targeted manipulation of learned latent dimensions, LATENTGUARD achieves selective refusal behavior, effectively blocking harmful requests while preserving helpfulness for legitimate use cases. Experiments on Qwen3-8B demonstrate significant improvements in both safety controllability and response interpretability without compromising utility. Cross-architecture validation on Mistral-7B confirms the generalizability of our latent steering approach, showing consistent effectiveness across different model families. Our results suggest that structured representation-level intervention offers a promising pathway toward building safer yet practical LLM systems. 

---
# PEPS: Quantum-Inspired Reinforcement Learning for Coherent Reasoning Traces in LLMs 

**Authors**: Venkat Margapuri, Garik Kazanjian, Naren Kosaraju  

**Link**: [PDF](https://arxiv.org/pdf/2509.20105)  

**Abstract**: Large Language Models (LLMs) often struggle with maintaining coherent multi-step reasoning traces, particularly in tasks that require a structured logical flow. This work introduces a quantum-inspired approach to address the challenge by incorporating a fidelity-based reward derived from Projected Entangled Pair States (PEPS) into Proximal Policy Optimization. Unlike prior approaches that use direct supervision or contrastive objectives, the proposed method guides learning through structural consistency, offering a novel approach to enforce global coherence in generated reasoning traces. The proposed framework is evaluated using multiple coherence-determining metrics on diverse datasets such as GSM8K, StrategyQA, and EntailmentBank spanning arithmetic, intuitive, and entailment-based reasoning. Results show that the proposed quantum-inspired approach offers significant improvements over supervised, contrastive, and pretrained baseline approaches, highlighting the effectiveness of quantum-inspired fidelity as a foundation to improve reasoning trace coherence in LLMs. 

---
# Video models are zero-shot learners and reasoners 

**Authors**: Thaddäus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, Robert Geirhos  

**Link**: [PDF](https://arxiv.org/pdf/2509.20328)  

**Abstract**: The remarkable zero-shot capabilities of Large Language Models (LLMs) have propelled natural language processing from task-specific models to unified, generalist foundation models. This transformation emerged from simple primitives: large, generative models trained on web-scale data. Curiously, the same primitives apply to today's generative video models. Could video models be on a trajectory towards general-purpose vision understanding, much like LLMs developed general-purpose language understanding? We demonstrate that Veo 3 can solve a broad variety of tasks it wasn't explicitly trained for: segmenting objects, detecting edges, editing images, understanding physical properties, recognizing object affordances, simulating tool use, and more. These abilities to perceive, model, and manipulate the visual world enable early forms of visual reasoning like maze and symmetry solving. Veo's emergent zero-shot capabilities indicate that video models are on a path to becoming unified, generalist vision foundation models. 

---
# Estimating the Self-Consistency of LLMs 

**Authors**: Robert Nowak  

**Link**: [PDF](https://arxiv.org/pdf/2509.19489)  

**Abstract**: Systems often repeat the same prompt to large language models (LLMs) and aggregate responses to improve reliability. This short note analyzes an estimator of the self-consistency of LLMs and the tradeoffs it induces under a fixed compute budget $B=mn$, where $m$ is the number of prompts sampled from the task distribution and $n$ is the number of repeated LLM calls per prompt; the resulting analysis favors a rough split $m,n\propto\sqrt{B}$. 

---
# When Judgment Becomes Noise: How Design Failures in LLM Judge Benchmarks Silently Undermine Validity 

**Authors**: Benjamin Feuer, Chiung-Yi Tseng, Astitwa Sarthak Lathe, Oussama Elachqar, John P Dickerson  

**Link**: [PDF](https://arxiv.org/pdf/2509.20293)  

**Abstract**: LLM-judged benchmarks are increasingly used to evaluate complex model behaviors, yet their design introduces failure modes absent in conventional ground-truth based benchmarks. We argue that without tight objectives and verifiable constructions, benchmark rankings can produce high-confidence rankings that are in fact largely noise. We introduce two mechanisms to diagnose these issues. Schematic adherence quantifies how much of a judge's overall verdict is explained by the explicit evaluation schema, revealing unexplained variance when judges deviate from their own rubric. Psychometric validity aggregates internal consistency and discriminant validity signals to quantify irreducible uncertainty in any benchmarking run. Applying these tools to Arena-Hard Auto, we find severe schema incoherence and factor collapse across popular judges: for example, unexplained variance exceeding 90 percent for DeepSeek-R1-32B and factor correlations above 0.93 for most criteria. We also show that the ELO-style aggregation used by Arena-Hard Auto collapses and masks genuine ranking uncertainty. Our results highlight design failures that undermine validity and offer actionable principles for building better-scoped, reliability-aware LLM-judged benchmarks. We release our code at this https URL 

---
# STAF: Leveraging LLMs for Automated Attack Tree-Based Security Test Generation 

**Authors**: Tanmay Khule, Stefan Marksteiner, Jose Alguindigue, Hannes Fuchs, Sebastian Fischmeister, Apurva Narayan  

**Link**: [PDF](https://arxiv.org/pdf/2509.20190)  

**Abstract**: In modern automotive development, security testing is critical for safeguarding systems against increasingly advanced threats. Attack trees are widely used to systematically represent potential attack vectors, but generating comprehensive test cases from these trees remains a labor-intensive, error-prone task that has seen limited automation in the context of testing vehicular systems. This paper introduces STAF (Security Test Automation Framework), a novel approach to automating security test case generation. Leveraging Large Language Models (LLMs) and a four-step self-corrective Retrieval-Augmented Generation (RAG) framework, STAF automates the generation of executable security test cases from attack trees, providing an end-to-end solution that encompasses the entire attack surface. We particularly show the elements and processes needed to provide an LLM to actually produce sensible and executable automotive security test suites, along with the integration with an automated testing framework. We further compare our tailored approach with general purpose (vanilla) LLMs and the performance of different LLMs (namely GPT-4.1 and DeepSeek) using our approach. We also demonstrate the method of our operation step-by-step in a concrete case study. Our results show significant improvements in efficiency, accuracy, scalability, and easy integration in any workflow, marking a substantial advancement in automating automotive security testing methodologies. Using TARAs as an input for verfication tests, we create synergies by connecting two vital elements of a secure automotive development process. 

---
# CyberSOCEval: Benchmarking LLMs Capabilities for Malware Analysis and Threat Intelligence Reasoning 

**Authors**: Lauren Deason, Adam Bali, Ciprian Bejean, Diana Bolocan, James Crnkovich, Ioana Croitoru, Krishna Durai, Chase Midler, Calin Miron, David Molnar, Brad Moon, Bruno Ostarcevic, Alberto Peltea, Matt Rosenberg, Catalin Sandu, Arthur Saputkin, Sagar Shah, Daniel Stan, Ernest Szocs, Shengye Wan, Spencer Whitman, Sven Krasser, Joshua Saxe  

**Link**: [PDF](https://arxiv.org/pdf/2509.20166)  

**Abstract**: Today's cyber defenders are overwhelmed by a deluge of security alerts, threat intelligence signals, and shifting business context, creating an urgent need for AI systems to enhance operational security work. While Large Language Models (LLMs) have the potential to automate and scale Security Operations Center (SOC) operations, existing evaluations do not fully assess the scenarios most relevant to real-world defenders. This lack of informed evaluation impacts both AI developers and those applying LLMs to SOC automation. Without clear insight into LLM performance in real-world security scenarios, developers lack a north star for development, and users cannot reliably select the most effective models. Meanwhile, malicious actors are using AI to scale cyber attacks, highlighting the need for open source benchmarks to drive adoption and community-driven improvement among defenders and model developers. To address this, we introduce CyberSOCEval, a new suite of open source benchmarks within CyberSecEval 4. CyberSOCEval includes benchmarks tailored to evaluate LLMs in two tasks: Malware Analysis and Threat Intelligence Reasoning--core defensive domains with inadequate coverage in current benchmarks. Our evaluations show that larger, more modern LLMs tend to perform better, confirming the training scaling laws paradigm. We also find that reasoning models leveraging test time scaling do not achieve the same boost as in coding and math, suggesting these models have not been trained to reason about cybersecurity analysis, and pointing to a key opportunity for improvement. Finally, current LLMs are far from saturating our evaluations, showing that CyberSOCEval presents a significant challenge for AI developers to improve cyber defense capabilities. 

---
# Affective Computing and Emotional Data: Challenges and Implications in Privacy Regulations, The AI Act, and Ethics in Large Language Models 

**Authors**: Nicola Fabiano  

**Link**: [PDF](https://arxiv.org/pdf/2509.20153)  

**Abstract**: This paper examines the integration of emotional intelligence into artificial intelligence systems, with a focus on affective computing and the growing capabilities of Large Language Models (LLMs), such as ChatGPT and Claude, to recognize and respond to human emotions. Drawing on interdisciplinary research that combines computer science, psychology, and neuroscience, the study analyzes foundational neural architectures - CNNs for processing facial expressions and RNNs for sequential data, such as speech and text - that enable emotion recognition. It examines the transformation of human emotional experiences into structured emotional data, addressing the distinction between explicit emotional data collected with informed consent in research settings and implicit data gathered passively through everyday digital interactions. That raises critical concerns about lawful processing, AI transparency, and individual autonomy over emotional expressions in digital environments. The paper explores implications across various domains, including healthcare, education, and customer service, while addressing challenges of cultural variations in emotional expression and potential biases in emotion recognition systems across different demographic groups. From a regulatory perspective, the paper examines emotional data in the context of the GDPR and the EU AI Act frameworks, highlighting how emotional data may be considered sensitive personal data that requires robust safeguards, including purpose limitation, data minimization, and meaningful consent mechanisms. 

---
# One Filters All: A Generalist Filter for State Estimation 

**Authors**: Shiqi Liu, Wenhan Cao, Chang Liu, Zeyu He, Tianyi Zhang, Shengbo Eben Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.20051)  

**Abstract**: Estimating hidden states in dynamical systems, also known as optimal filtering, is a long-standing problem in various fields of science and engineering. In this paper, we introduce a general filtering framework, \textbf{LLM-Filter}, which leverages large language models (LLMs) for state estimation by embedding noisy observations with text prototypes. In various experiments for classical dynamical systems, we find that first, state estimation can significantly benefit from the reasoning knowledge embedded in pre-trained LLMs. By achieving proper modality alignment with the frozen LLM, LLM-Filter outperforms the state-of-the-art learning-based approaches. Second, we carefully design the prompt structure, System-as-Prompt (SaP), incorporating task instructions that enable the LLM to understand the estimation tasks. Guided by these prompts, LLM-Filter exhibits exceptional generalization, capable of performing filtering tasks accurately in changed or even unseen environments. We further observe a scaling-law behavior in LLM-Filter, where accuracy improves with larger model sizes and longer training times. These findings make LLM-Filter a promising foundation model of filtering. 

---
# TianHui: A Domain-Specific Large Language Model for Diverse Traditional Chinese Medicine Scenarios 

**Authors**: Ji Yin, Menglan He, Yujie Zhang, Linshuai Zhang, Tingting Ma, Ce Tian, Jie Wu, Lin Xu, Tao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19834)  

**Abstract**: Domain-specific LLMs in TCM face limitations in research settings due to constrained adaptability, insufficient evaluation datasets, and limited computational resources. This study presents TianHui, a specialized TCM LLM built through contextual data integration and domain knowledge fusion. We constructed a large-scale TCM corpus (0.97GB unsupervised data + 611,312 QA pairs) and employed a two-stage training strategy with QLoRA, DeepSpeed Stage 2, and Flash Attention 2. Evaluation on 12 benchmarks showed TianHui ranked top-three in all metrics for six datasets (APQ, TCMCD, HFR, HCCA, DHPE, TLAW) and achieved top results in the other six (TCMEE, APR, GCPMI, TCMKQA, TCMRC, ADTG). Optimal configuration was identified as LoRA rank=128, alpha=256, epoch=4, dropout=0.2, max length=2048. TianHui enables systematic preservation and scalable application of TCM knowledge. All resources are open-sourced. 

---
# Adaptive Guidance Semantically Enhanced via Multimodal LLM for Edge-Cloud Object Detection 

**Authors**: Yunqing Hu, Zheming Yang, Chang Zhao, Wen Ji  

**Link**: [PDF](https://arxiv.org/pdf/2509.19875)  

**Abstract**: Traditional object detection methods face performance degradation challenges in complex scenarios such as low-light conditions and heavy occlusions due to a lack of high-level semantic understanding. To address this, this paper proposes an adaptive guidance-based semantic enhancement edge-cloud collaborative object detection method leveraging Multimodal Large Language Models (MLLM), achieving an effective balance between accuracy and efficiency. Specifically, the method first employs instruction fine-tuning to enable the MLLM to generate structured scene descriptions. It then designs an adaptive mapping mechanism that dynamically converts semantic information into parameter adjustment signals for edge detectors, achieving real-time semantic enhancement. Within an edge-cloud collaborative inference framework, the system automatically selects between invoking cloud-based semantic guidance or directly outputting edge detection results based on confidence scores. Experiments demonstrate that the proposed method effectively enhances detection accuracy and efficiency in complex scenes. Specifically, it can reduce latency by over 79% and computational cost by 70% in low-light and highly occluded scenes while maintaining accuracy. 

---
# Eliminating stability hallucinations in llm-based tts models via attention guidance 

**Authors**: ShiMing Wang, ZhiHao Du, Yang Xiang, TianYu Zhao, Han Zhao, Qian Chen, XianGang Li, HanJie Guo, ZhenHua Ling  

**Link**: [PDF](https://arxiv.org/pdf/2509.19852)  

**Abstract**: This paper focuses on resolving stability hallucinations (e.g., repetitive or omitted speech) in LLM-based Text-to-Speech (TTS) models by improving and leveraging the attention mechanism. First, we analyzed the alignment mechanism between text tokens and speech tokens in LLMs. We then proposed a metric termed the Optimal Alignment Score (OAS), which employs the Viterbi algorithm to evaluate text-speech alignment quality. Subsequently, OAS was integrated into the training of CosyVoice2 to assist LLMs in learning continuous, stable alignment. Additionally, the pre-trained attention value is employed to guide the training of the student CosyVoice2 via chain-of-thought (CoT), which further reduces stability hallucinations in synthesized speech. Experiments on the Seed-TTS-Eval and CV3-Eval test sets demonstrate that the proposed methods can effectively reduce the stability hallucinations of CosyVoice2 without introducing additional negative effects. The appendix is available at this https URL. 

---
# Are We Scaling the Right Thing? A System Perspective on Test-Time Scaling 

**Authors**: Youpeng Zhao, Jinpeng LV, Di Wu, Jun Wang, Christopher Gooley  

**Link**: [PDF](https://arxiv.org/pdf/2509.19645)  

**Abstract**: Test-time scaling (TTS) has recently emerged as a promising direction to exploit the hidden reasoning capabilities of pre-trained large language models (LLMs). However, existing scaling methods narrowly focus on the compute-optimal Pareto-frontier, ignoring the simple fact that compute-optimal is not always system-optimal. In this work, we propose a system-driven perspective on TTS, analyzing how reasoning models scale against practical metrics, such as latency and cost-per-token. By evaluating the impact of popular optimizations such as tensor parallelism and speculative decoding, our preliminary analysis reveals the limitations of current methods and calls for a paradigm shift toward holistic, system-aware evaluations that capture the true essence of scaling laws at inference time. 

---
# Thinking While Listening: Simple Test Time Scaling For Audio Classification 

**Authors**: Prateek Verma, Mert Pilanci  

**Link**: [PDF](https://arxiv.org/pdf/2509.19676)  

**Abstract**: We propose a framework that enables neural models to "think while listening" to everyday sounds, thereby enhancing audio classification performance. Motivated by recent advances in the reasoning capabilities of large language models, we address two central questions: (i) how can thinking be incorporated into existing audio classification pipelines to enable reasoning in the category space and improve performance, and (ii) can a new architecture be designed from the ground up to support both thinking and test-time scaling? We demonstrate that in both settings, our models exhibit improved classification accuracy. Leveraging test-time scaling, we observe consistent gains as the number of sampled traces increases. Furthermore, we evaluate two open-source reasoning models, GPT-OSS-20B and Qwen3-14B, showing that while such models are capable of zero-shot reasoning, a lightweight approach--retraining only the embedding matrix of a frozen, smaller model like GPT-2--can surpass the performance of billion-parameter text-based reasoning models. 

---
# Reverse Engineering User Stories from Code using Large Language Models 

**Authors**: Mohamed Ouf, Haoyu Li, Michael Zhang, Mariam Guizani  

**Link**: [PDF](https://arxiv.org/pdf/2509.19587)  

**Abstract**: User stories are essential in agile development, yet often missing or outdated in legacy and poorly documented systems. We investigate whether large language models (LLMs) can automatically recover user stories directly from source code and how prompt design impacts output quality. Using 1,750 annotated C++ snippets of varying complexity, we evaluate five state-of-the-art LLMs across six prompting strategies. Results show that all models achieve, on average, an F1 score of 0.8 for code up to 200 NLOC. Our findings show that a single illustrative example enables the smallest model (8B) to match the performance of a much larger 70B model. In contrast, structured reasoning via Chain-of-Thought offers only marginal gains, primarily for larger models. 

---
# Semantic-Aware Fuzzing: An Empirical Framework for LLM-Guided, Reasoning-Driven Input Mutation 

**Authors**: Mengdi Lu, Steven Ding, Furkan Alaca, Philippe Charland  

**Link**: [PDF](https://arxiv.org/pdf/2509.19533)  

**Abstract**: Security vulnerabilities in Internet-of-Things devices, mobile platforms, and autonomous systems remain critical. Traditional mutation-based fuzzers -- while effectively explore code paths -- primarily perform byte- or bit-level edits without semantic reasoning. Coverage-guided tools such as AFL++ use dictionaries, grammars, and splicing heuristics to impose shallow structural constraints, leaving deeper protocol logic, inter-field dependencies, and domain-specific semantics unaddressed. Conversely, reasoning-capable large language models (LLMs) can leverage pretraining knowledge to understand input formats, respect complex constraints, and propose targeted mutations, much like an experienced reverse engineer or testing expert. However, lacking ground truth for "correct" mutation reasoning makes supervised fine-tuning impractical, motivating explorations of off-the-shelf LLMs via prompt-based few-shot learning. To bridge this gap, we present an open-source microservices framework that integrates reasoning LLMs with AFL++ on Google's FuzzBench, tackling asynchronous execution and divergent hardware demands (GPU- vs. CPU-intensive) of LLMs and fuzzers. We evaluate four research questions: (R1) How can reasoning LLMs be integrated into the fuzzing mutation loop? (R2) Do few-shot prompts yield higher-quality mutations than zero-shot? (R3) Can prompt engineering with off-the-shelf models improve fuzzing directly? and (R4) Which open-source reasoning LLMs perform best under prompt-only conditions? Experiments with Llama3.3, Deepseek-r1-Distill-Llama-70B, QwQ-32B, and Gemma3 highlight Deepseek as the most promising. Mutation effectiveness depends more on prompt complexity and model choice than shot count. Response latency and throughput bottlenecks remain key obstacles, offering directions for future work. 

---
# Uncertainty Quantification of Large Language Models using Approximate Bayesian Computation 

**Authors**: Mridul Sharma, Adeetya Patel, Zaneta D' Souza, Samira Abbasgholizadeh Rahimi, Siva Reddy, Sreenath Madathil  

**Link**: [PDF](https://arxiv.org/pdf/2509.19375)  

**Abstract**: Despite their widespread applications, Large Language Models (LLMs) often struggle to express uncertainty, posing a challenge for reliable deployment in high stakes and safety critical domains like clinical diagnostics. Existing standard baseline methods such as model logits and elicited probabilities produce overconfident and poorly calibrated estimates. In this work, we propose Approximate Bayesian Computation (ABC), a likelihood-free Bayesian inference, based approach that treats LLMs as a stochastic simulator to infer posterior distributions over predictive probabilities. We evaluate our ABC approach on two clinically relevant benchmarks: a synthetic oral lesion diagnosis dataset and the publicly available GretelAI symptom-to-diagnosis dataset. Compared to standard baselines, our approach improves accuracy by up to 46.9\%, reduces Brier scores by 74.4\%, and enhances calibration as measured by Expected Calibration Error (ECE) and predictive entropy. 

---
# GAUSS: Benchmarking Structured Mathematical Skills for Large Language Models 

**Authors**: Yue Zhang, Jiaxin Zhang, Qiuyu Ren, Tahsin Saffat, Xiaoxuan Liu, Zitong Yang, Banghua Zhu, Yi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2509.18122)  

**Abstract**: We introduce \textbf{GAUSS} (\textbf{G}eneral \textbf{A}ssessment of \textbf{U}nderlying \textbf{S}tructured \textbf{S}kills in Mathematics), a benchmark that evaluates LLMs' mathematical abilities across twelve core skill dimensions, grouped into three domains: knowledge and understanding, problem solving and communication, and meta-skills and creativity. By categorizing problems according to cognitive skills and designing tasks that isolate specific abilities, GAUSS constructs comprehensive, fine-grained, and interpretable profiles of models' mathematical abilities. These profiles faithfully represent their underlying mathematical intelligence. To exemplify how to use the \textsc{GAUSS} benchmark, we have derived the skill profile of \textsc{GPT-5-thinking}, revealing its strengths and weaknesses as well as its differences relative to \textsc{o4-mini-high}, thereby underscoring the value of multidimensional, skill-based evaluation. 

---
# LLMs as verification oracles for Solidity 

**Authors**: Massimo Bartoletti, Enrico Lipparini, Livio Pompianu  

**Link**: [PDF](https://arxiv.org/pdf/2509.19153)  

**Abstract**: Ensuring the correctness of smart contracts is critical, as even subtle flaws can lead to severe financial losses. While bug detection tools able to spot common vulnerability patterns can serve as a first line of defense, most real-world exploits and losses stem from errors in the contract business logic. Formal verification tools such as SolCMC and the Certora Prover address this challenge, but their impact remains limited by steep learning curves and restricted specification languages. Recent works have begun to explore the use of large language models (LLMs) for security-related tasks such as vulnerability detection and test generation. Yet, a fundamental question remains open: can LLMs serve as verification oracles, capable of reasoning about arbitrary contract-specific properties? In this paper, we provide the first systematic evaluation of GPT-5, a state-of-the-art reasoning LLM, in this role. We benchmark its performance on a large dataset of verification tasks, compare its outputs against those of established formal verification tools, and assess its practical effectiveness in real-world auditing scenarios. Our study combines quantitative metrics with qualitative analysis, and shows that recent reasoning-oriented LLMs can be surprisingly effective as verification oracles, suggesting a new frontier in the convergence of AI and formal methods for secure smart contract development and auditing. 

---
