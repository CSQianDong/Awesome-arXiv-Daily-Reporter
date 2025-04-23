# TTRL: Test-Time Reinforcement Learning 

**Authors**: Yuxin Zuo, Kaiyan Zhang, Shang Qu, Li Sheng, Xuekai Zhu, Biqing Qi, Youbang Sun, Ganqu Cui, Ning Ding, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.16084)  

**Abstract**: This paper investigates Reinforcement Learning (RL) on data without explicit labels for reasoning tasks in Large Language Models (LLMs). The core challenge of the problem is reward estimation during inference while not having access to ground-truth information. While this setting appears elusive, we find that common practices in Test-Time Scaling (TTS), such as majority voting, yield surprisingly effective rewards suitable for driving RL training. In this work, we introduce Test-Time Reinforcement Learning (TTRL), a novel method for training LLMs using RL on unlabeled data. TTRL enables self-evolution of LLMs by utilizing the priors in the pre-trained models. Our experiments demonstrate that TTRL consistently improves performance across a variety of tasks and models. Notably, TTRL boosts the pass@1 performance of Qwen-2.5-Math-7B by approximately 159% on the AIME 2024 with only unlabeled test data. Furthermore, although TTRL is only supervised by the Maj@N metric, TTRL has demonstrated performance to consistently surpass the upper limit of the initial model, and approach the performance of models trained directly on test data with ground-truth labels. Our experimental findings validate the general effectiveness of TTRL across various tasks, and highlight TTRL's potential for broader tasks and domains. GitHub: this https URL 

---
# PHYBench: Holistic Evaluation of Physical Perception and Reasoning in Large Language Models 

**Authors**: Shi Qiu, Shaoyang Guo, Zhuo-Yang Song, Yunbo Sun, Zeyu Cai, Jiashen Wei, Tianyu Luo, Yixuan Yin, Haoxu Zhang, Yi Hu, Chenyang Wang, Chencheng Tang, Haoling Chang, Qi Liu, Ziheng Zhou, Tianyu Zhang, Jingtian Zhang, Zhangyi Liu, Minghao Li, Yuku Zhang, Boxuan Jing, Xianqi Yin, Yutong Ren, Zizhuo Fu, Weike Wang, Xudong Tian, Anqi Lv, Laifu Man, Jianxiang Li, Feiyu Tao, Qihua Sun, Zhou Liang, Yushu Mu, Zhongxuan Li, Jing-Jun Zhang, Shutao Zhang, Xiaotian Li, Xingqi Xia, Jiawei Lin, Zheyu Shen, Jiahang Chen, Qiuhao Xiong, Binran Wang, Fengyuan Wang, Ziyang Ni, Bohan Zhang, Fan Cui, Changkun Shao, Qing-Hong Cao, Ming-xing Luo, Muhan Zhang, Hua Xing Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.16074)  

**Abstract**: We introduce PHYBench, a novel, high-quality benchmark designed for evaluating reasoning capabilities of large language models (LLMs) in physical contexts. PHYBench consists of 500 meticulously curated physics problems based on real-world physical scenarios, designed to assess the ability of models to understand and reason about realistic physical processes. Covering mechanics, electromagnetism, thermodynamics, optics, modern physics, and advanced physics, the benchmark spans difficulty levels from high school exercises to undergraduate problems and Physics Olympiad challenges. Additionally, we propose the Expression Edit Distance (EED) Score, a novel evaluation metric based on the edit distance between mathematical expressions, which effectively captures differences in model reasoning processes and results beyond traditional binary scoring methods. We evaluate various LLMs on PHYBench and compare their performance with human experts. Our results reveal that even state-of-the-art reasoning models significantly lag behind human experts, highlighting their limitations and the need for improvement in complex physical reasoning scenarios. Our benchmark results and dataset are publicly available at this https URL. 

---
# Guiding VLM Agents with Process Rewards at Inference Time for GUI Navigation 

**Authors**: Zhiyuan Hu, Shiyun Xiong, Yifan Zhang, See-Kiong Ng, Anh Tuan Luu, Bo An, Shuicheng Yan, Bryan Hooi  

**Link**: [PDF](https://arxiv.org/pdf/2504.16073)  

**Abstract**: Recent advancements in visual language models (VLMs) have notably enhanced their capabilities in handling complex Graphical User Interface (GUI) interaction tasks. Despite these improvements, current frameworks often struggle to generate correct actions in challenging GUI environments. State-of-the-art commercial VLMs are black-boxes, and fine-tuning open-source VLMs for GUI tasks requires significant resources. Additionally, existing trajectory-level evaluation and refinement techniques frequently fall short due to delayed feedback and local optimization issues. To address these challenges, we propose an approach that guides VLM agents with process supervision by a reward model during GUI navigation and control at inference time. This guidance allows the VLM agent to optimize actions at each inference step, thereby improving performance in both static and dynamic environments. In particular, our method demonstrates significant performance gains in three GUI navigation tasks, achieving a 3.4% improvement in single step action accuracy for static environments, along with a around 33% increase in task success rate in one dynamic environment. With further integration of trajectory reflection and retry mechanisms, we also demonstrate even greater enhancement in task success. 

---
# A Python Tool for Reconstructing Full News Text from GDELT 

**Authors**: A. Fronzetti Colladon, R. Vestrelli  

**Link**: [PDF](https://arxiv.org/pdf/2504.16063)  

**Abstract**: News data have become an essential resource across various disciplines, including economics, finance, management, social sciences, and computer science. Researchers leverage newspaper articles to study economic trends, market dynamics, corporate strategies, public perception, political discourse, and the evolution of public opinion. Additionally, news datasets have been instrumental in training large-scale language models, with applications in sentiment analysis, fake news detection, and automated news summarization. Despite their significance, access to comprehensive news corpora remains a key challenge. Many full-text news providers, such as Factiva and LexisNexis, require costly subscriptions, while free alternatives often suffer from incomplete data and transparency issues. This paper presents a novel approach to obtaining full-text newspaper articles at near-zero cost by leveraging data from the Global Database of Events, Language, and Tone (GDELT). Specifically, we focus on the GDELT Web News NGrams 3.0 dataset, which provides high-frequency updates of n-grams extracted from global online news sources. We provide Python code to reconstruct full-text articles from these n-grams by identifying overlapping textual fragments and intelligently merging them. Our method enables researchers to access structured, large-scale newspaper data for text analysis while overcoming the limitations of existing proprietary datasets. The proposed approach enhances the accessibility of news data for empirical research, facilitating applications in economic forecasting, computational social science, and natural language processing. 

---
# Vision-Language Models Are Not Pragmatically Competent in Referring Expression Generation 

**Authors**: Ziqiao Ma, Jing Ding, Xuejun Zhang, Dezhi Luo, Jiahe Ding, Sihan Xu, Yuchen Huang, Run Peng, Joyce Chai  

**Link**: [PDF](https://arxiv.org/pdf/2504.16060)  

**Abstract**: Referring Expression Generation (REG) is a core task for evaluating the pragmatic competence of vision-language systems, requiring not only accurate semantic grounding but also adherence to principles of cooperative communication (Grice, 1975). However, current evaluations of vision-language models (VLMs) often overlook the pragmatic dimension, reducing REG to a region-based captioning task and neglecting Gricean maxims. In this work, we revisit REG from a pragmatic perspective, introducing a new dataset (RefOI) of 1.5k images annotated with both written and spoken referring expressions. Through a systematic evaluation of state-of-the-art VLMs, we identify three key failures of pragmatic competence: (1) failure to uniquely identify the referent, (2) inclusion of excessive or irrelevant information, and (3) misalignment with human pragmatic preference, such as the underuse of minimal spatial cues. We also show that standard automatic evaluations fail to capture these pragmatic violations, reinforcing superficial cues rather than genuine referential success. Our findings call for a renewed focus on pragmatically informed models and evaluation frameworks that align with real human communication. 

---
# Honey, I Shrunk the Language Model: Impact of Knowledge Distillation Methods on Performance and Explainability 

**Authors**: Daniel Hendriks, Philipp Spitzer, Niklas Kühl, Gerhard Satzger  

**Link**: [PDF](https://arxiv.org/pdf/2504.16056)  

**Abstract**: Artificial Intelligence (AI) has increasingly influenced modern society, recently in particular through significant advancements in Large Language Models (LLMs). However, high computational and storage demands of LLMs still limit their deployment in resource-constrained environments. Knowledge distillation addresses this challenge by training a small student model from a larger teacher model. Previous research has introduced several distillation methods for both generating training data and for training the student model. Despite their relevance, the effects of state-of-the-art distillation methods on model performance and explainability have not been thoroughly investigated and compared. In this work, we enlarge the set of available methods by applying critique-revision prompting to distillation for data generation and by synthesizing existing methods for training. For these methods, we provide a systematic comparison based on the widely used Commonsense Question-Answering (CQA) dataset. While we measure performance via student model accuracy, we employ a human-grounded study to evaluate explainability. We contribute new distillation methods and their comparison in terms of both performance and explainability. This should further advance the distillation of small language models and, thus, contribute to broader applicability and faster diffusion of LLM technology. 

---
# LongMamba: Enhancing Mamba's Long Context Capabilities via Training-Free Receptive Field Enlargement 

**Authors**: Zhifan Ye, Kejing Xia, Yonggan Fu, Xin Dong, Jihoon Hong, Xiangchi Yuan, Shizhe Diao, Jan Kautz, Pavlo Molchanov, Yingyan Celine Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.16053)  

**Abstract**: State space models (SSMs) have emerged as an efficient alternative to Transformer models for language modeling, offering linear computational complexity and constant memory usage as context length increases. However, despite their efficiency in handling long contexts, recent studies have shown that SSMs, such as Mamba models, generally underperform compared to Transformers in long-context understanding tasks. To address this significant shortfall and achieve both efficient and accurate long-context understanding, we propose LongMamba, a training-free technique that significantly enhances the long-context capabilities of Mamba models. LongMamba builds on our discovery that the hidden channels in Mamba can be categorized into local and global channels based on their receptive field lengths, with global channels primarily responsible for long-context capability. These global channels can become the key bottleneck as the input context lengthens. Specifically, when input lengths largely exceed the training sequence length, global channels exhibit limitations in adaptively extend their receptive fields, leading to Mamba's poor long-context performance. The key idea of LongMamba is to mitigate the hidden state memory decay in these global channels by preventing the accumulation of unimportant tokens in their memory. This is achieved by first identifying critical tokens in the global channels and then applying token filtering to accumulate only those critical tokens. Through extensive benchmarking across synthetic and real-world long-context scenarios, LongMamba sets a new standard for Mamba's long-context performance, significantly extending its operational range without requiring additional training. Our code is available at this https URL. 

---
# Certified Mitigation of Worst-Case LLM Copyright Infringement 

**Authors**: Jingyu Zhang, Jiacan Yu, Marc Marone, Benjamin Van Durme, Daniel Khashabi  

**Link**: [PDF](https://arxiv.org/pdf/2504.16046)  

**Abstract**: The exposure of large language models (LLMs) to copyrighted material during pre-training raises concerns about unintentional copyright infringement post deployment. This has driven the development of "copyright takedown" methods, post-training approaches aimed at preventing models from generating content substantially similar to copyrighted ones. While current mitigation approaches are somewhat effective for average-case risks, we demonstrate that they overlook worst-case copyright risks exhibits by the existence of long, verbatim quotes from copyrighted sources. We propose BloomScrub, a remarkably simple yet highly effective inference-time approach that provides certified copyright takedown. Our method repeatedly interleaves quote detection with rewriting techniques to transform potentially infringing segments. By leveraging efficient data sketches (Bloom filters), our approach enables scalable copyright screening even for large-scale real-world corpora. When quotes beyond a length threshold cannot be removed, the system can abstain from responding, offering certified risk reduction. Experimental results show that BloomScrub reduces infringement risk, preserves utility, and accommodates different levels of enforcement stringency with adaptive abstention. Our results suggest that lightweight, inference-time methods can be surprisingly effective for copyright prevention. 

---
# Methods for Recognizing Nested Terms 

**Authors**: Igor Rozhkov, Natalia Loukachevitch  

**Link**: [PDF](https://arxiv.org/pdf/2504.16007)  

**Abstract**: In this paper, we describe our participation in the RuTermEval competition devoted to extracting nested terms. We apply the Binder model, which was previously successfully applied to the recognition of nested named entities, to extract nested terms. We obtained the best results of term recognition in all three tracks of the RuTermEval competition. In addition, we study the new task of recognition of nested terms from flat training data annotated with terms without nestedness. We can conclude that several approaches we proposed in this work are viable enough to retrieve nested terms effectively without nested labeling of them. 

---
# CAPO: Cost-Aware Prompt Optimization 

**Authors**: Tom Zehle, Moritz Schlager, Timo Heiß, Matthias Feurer  

**Link**: [PDF](https://arxiv.org/pdf/2504.16005)  

**Abstract**: Large language models (LLMs) have revolutionized natural language processing by solving a wide range of tasks simply guided by a prompt. Yet their performance is highly sensitive to prompt formulation. While automated prompt optimization addresses this challenge by finding optimal prompts, current methods require a substantial number of LLM calls and input tokens, making prompt optimization expensive. We introduce CAPO (Cost-Aware Prompt Optimization), an algorithm that enhances prompt optimization efficiency by integrating AutoML techniques. CAPO is an evolutionary approach with LLMs as operators, incorporating racing to save evaluations and multi-objective optimization to balance performance with prompt length. It jointly optimizes instructions and few-shot examples while leveraging task descriptions for improved robustness. Our extensive experiments across diverse datasets and LLMs demonstrate that CAPO outperforms state-of-the-art discrete prompt optimization methods in 11/15 cases with improvements up to 21%p. Our algorithm achieves better performances already with smaller budgets, saves evaluations through racing, and decreases average prompt length via a length penalty, making it both cost-efficient and cost-aware. Even without few-shot examples, CAPO outperforms its competitors and generally remains robust to initial prompts. CAPO represents an important step toward making prompt optimization more powerful and accessible by improving cost-efficiency. 

---
# Few-shot Hate Speech Detection Based on the MindSpore Framework 

**Authors**: Zhenkai Qin, Dongze Wu, Yuxin Liu, Guifang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15987)  

**Abstract**: The proliferation of hate speech on social media poses a significant threat to online communities, requiring effective detection systems. While deep learning models have shown promise, their performance often deteriorates in few-shot or low-resource settings due to reliance on large annotated corpora. To address this, we propose MS-FSLHate, a prompt-enhanced neural framework for few-shot hate speech detection implemented on the MindSpore deep learning platform. The model integrates learnable prompt embeddings, a CNN-BiLSTM backbone with attention pooling, and synonym-based adversarial data augmentation to improve generalization. Experimental results on two benchmark datasets-HateXplain and HSOL-demonstrate that our approach outperforms competitive baselines in precision, recall, and F1-score. Additionally, the framework shows high efficiency and scalability, suggesting its suitability for deployment in resource-constrained environments. These findings highlight the potential of combining prompt-based learning with adversarial augmentation for robust and adaptable hate speech detection in few-shot scenarios. 

---
# W-PCA Based Gradient-Free Proxy for Efficient Search of Lightweight Language Models 

**Authors**: Shang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15983)  

**Abstract**: The demand for efficient natural language processing (NLP) systems has led to the development of lightweight language models. Previous work in this area has primarily focused on manual design or training-based neural architecture search (NAS) methods. Recently, zero-shot NAS methods have been proposed for evaluating language models without the need for training. However, prevailing approaches to zero-shot NAS often face challenges such as biased evaluation metrics and computational inefficiencies. In this paper, we introduce weight-weighted PCA (W-PCA), a novel zero-shot NAS method specifically tailored for lightweight language models. Our approach utilizes two evaluation proxies: the parameter count and the number of principal components with cumulative contribution exceeding $\eta$ in the feed-forward neural (FFN) layer. Additionally, by eliminating the need for gradient computations, we optimize the evaluation time, thus enhancing the efficiency of designing and evaluating lightweight language models. We conduct a comparative analysis on the GLUE and SQuAD datasets to evaluate our approach. The results demonstrate that our method significantly reduces training time compared to one-shot NAS methods and achieves higher scores in the testing phase compared to previous state-of-the-art training-based methods. Furthermore, we perform ranking evaluations on a dataset sampled from the FlexiBERT search space. Our approach exhibits superior ranking correlation and further reduces solving time compared to other zero-shot NAS methods that require gradient computation. 

---
# FairTranslate: An English-French Dataset for Gender Bias Evaluation in Machine Translation by Overcoming Gender Binarity 

**Authors**: Fanny Jourdan, Yannick Chevalier, Cécile Favre  

**Link**: [PDF](https://arxiv.org/pdf/2504.15941)  

**Abstract**: Large Language Models (LLMs) are increasingly leveraged for translation tasks but often fall short when translating inclusive language -- such as texts containing the singular 'they' pronoun or otherwise reflecting fair linguistic protocols. Because these challenges span both computational and societal domains, it is imperative to critically evaluate how well LLMs handle inclusive translation with a well-founded framework.
This paper presents FairTranslate, a novel, fully human-annotated dataset designed to evaluate non-binary gender biases in machine translation systems from English to French. FairTranslate includes 2418 English-French sentence pairs related to occupations, annotated with rich metadata such as the stereotypical alignment of the occupation, grammatical gender indicator ambiguity, and the ground-truth gender label (male, female, or inclusive).
We evaluate four leading LLMs (Gemma2-2B, Mistral-7B, Llama3.1-8B, Llama3.3-70B) on this dataset under different prompting procedures. Our results reveal substantial biases in gender representation across LLMs, highlighting persistent challenges in achieving equitable outcomes in machine translation. These findings underscore the need for focused strategies and interventions aimed at ensuring fair and inclusive language usage in LLM-based translation systems.
We make the FairTranslate dataset publicly available on Hugging Face, and disclose the code for all experiments on GitHub. 

---
# SARI: Structured Audio Reasoning via Curriculum-Guided Reinforcement Learning 

**Authors**: Cheng Wen, Tingwei Guo, Shuaijiang Zhao, Wei Zou, Xiangang Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.15900)  

**Abstract**: Recent work shows that reinforcement learning(RL) can markedly sharpen the reasoning ability of large language models (LLMs) by prompting them to "think before answering." Yet whether and how these gains transfer to audio-language reasoning remains largely unexplored. We extend the Group-Relative Policy Optimization (GRPO) framework from DeepSeek-R1 to a Large Audio-Language Model (LALM), and construct a 32k sample multiple-choice corpus. Using a two-stage regimen supervised fine-tuning on structured and unstructured chains-of-thought, followed by curriculum-guided GRPO, we systematically compare implicit vs. explicit, and structured vs. free form reasoning under identical architectures. Our structured audio reasoning model, SARI (Structured Audio Reasoning via Curriculum-Guided Reinforcement Learning), achieves a 16.35% improvement in average accuracy over the base model Qwen2-Audio-7B-Instruct. Furthermore, the variant built upon Qwen2.5-Omni reaches state-of-the-art performance of 67.08% on the MMAU test-mini benchmark. Ablation experiments show that on the base model we use: (i) SFT warm-up is important for stable RL training, (ii) structured chains yield more robust generalization than unstructured ones, and (iii) easy-to-hard curricula accelerate convergence and improve final performance. These findings demonstrate that explicit, structured reasoning and curriculum learning substantially enhances audio-language understanding. 

---
# Dynamic Early Exit in Reasoning Models 

**Authors**: Chenxu Yang, Qingyi Si, Yongjie Duan, Zheliang Zhu, Chenyu Zhu, Zheng Lin, Li Cao, Weiping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15895)  

**Abstract**: Recent advances in large reasoning language models (LRLMs) rely on test-time scaling, which extends long chain-of-thought (CoT) generation to solve complex tasks. However, overthinking in long CoT not only slows down the efficiency of problem solving, but also risks accuracy loss due to the extremely detailed or redundant reasoning steps. We propose a simple yet effective method that allows LLMs to self-truncate CoT sequences by early exit during generation. Instead of relying on fixed heuristics, the proposed method monitors model behavior at potential reasoning transition points (e.g.,"Wait" tokens) and dynamically terminates the next reasoning chain's generation when the model exhibits high confidence in a trial answer. Our method requires no additional training and can be seamlessly integrated into existing o1-like reasoning LLMs. Experiments on multiple reasoning benchmarks MATH-500, AMC 2023, GPQA Diamond and AIME 2024 show that the proposed method is consistently effective on deepseek-series reasoning LLMs, reducing the length of CoT sequences by an average of 31% to 43% while improving accuracy by 1.7% to 5.7%. 

---
# Exploring Cognitive and Aesthetic Causality for Multimodal Aspect-Based Sentiment Analysis 

**Authors**: Luwei Xiao, Rui Mao, Shuai Zhao, Qika Lin, Yanhao Jia, Liang He, Erik Cambria  

**Link**: [PDF](https://arxiv.org/pdf/2504.15848)  

**Abstract**: Multimodal aspect-based sentiment classification (MASC) is an emerging task due to an increase in user-generated multimodal content on social platforms, aimed at predicting sentiment polarity toward specific aspect targets (i.e., entities or attributes explicitly mentioned in text-image pairs). Despite extensive efforts and significant achievements in existing MASC, substantial gaps remain in understanding fine-grained visual content and the cognitive rationales derived from semantic content and impressions (cognitive interpretations of emotions evoked by image content). In this study, we present Chimera: a cognitive and aesthetic sentiment causality understanding framework to derive fine-grained holistic features of aspects and infer the fundamental drivers of sentiment expression from both semantic perspectives and affective-cognitive resonance (the synergistic effect between emotional responses and cognitive interpretations). Specifically, this framework first incorporates visual patch features for patch-word alignment. Meanwhile, it extracts coarse-grained visual features (e.g., overall image representation) and fine-grained visual regions (e.g., aspect-related regions) and translates them into corresponding textual descriptions (e.g., facial, aesthetic). Finally, we leverage the sentimental causes and impressions generated by a large language model (LLM) to enhance the model's awareness of sentimental cues evoked by semantic content and affective-cognitive resonance. Experimental results on standard MASC datasets demonstrate the effectiveness of the proposed model, which also exhibits greater flexibility to MASC compared to LLMs such as GPT-4o. We have publicly released the complete implementation and dataset at this https URL 

---
# Pre-DPO: Improving Data Utilization in Direct Preference Optimization Using a Guiding Reference Model 

**Authors**: Junshu Pan, Wei Shen, Shulin Huang, Qiji Zhou, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15843)  

**Abstract**: Direct Preference Optimization (DPO) simplifies reinforcement learning from human feedback (RLHF) for large language models (LLMs) by directly optimizing human preferences without an explicit reward model. We find that during DPO training, the reference model plays the role of a data weight adjuster. However, the common practice of initializing the policy and reference models identically in DPO can lead to inefficient data utilization and impose a performance ceiling. Meanwhile, the lack of a reference model in Simple Preference Optimization (SimPO) reduces training robustness and necessitates stricter conditions to prevent catastrophic forgetting. In this work, we propose Pre-DPO, a simple yet effective DPO-based training paradigm that enhances preference optimization performance by leveraging a guiding reference model. This reference model provides foresight into the optimal policy state achievable through the training preference data, serving as a guiding mechanism that adaptively assigns higher weights to samples more suitable for the model and lower weights to those less suitable. Extensive experiments on AlpacaEval 2.0 and Arena-Hard v0.1 benchmarks demonstrate that Pre-DPO consistently improves the performance of both DPO and SimPO, without relying on external models or additional data. 

---
# What's the Difference? Supporting Users in Identifying the Effects of Prompt and Model Changes Through Token Patterns 

**Authors**: Michael A. Hedderich, Anyi Wang, Raoyuan Zhao, Florian Eichin, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2504.15815)  

**Abstract**: Prompt engineering for large language models is challenging, as even small prompt perturbations or model changes can significantly impact the generated output texts. Existing evaluation methods, either automated metrics or human evaluation, have limitations, such as providing limited insights or being labor-intensive. We propose Spotlight, a new approach that combines both automation and human analysis. Based on data mining techniques, we automatically distinguish between random (decoding) variations and systematic differences in language model outputs. This process provides token patterns that describe the systematic differences and guide the user in manually analyzing the effects of their prompt and model changes efficiently. We create three benchmarks to quantitatively test the reliability of token pattern extraction methods and demonstrate that our approach provides new insights into established prompt data. From a human-centric perspective, through demonstration studies and a user study, we show that our token pattern approach helps users understand the systematic differences of language model outputs, and we are able to discover relevant differences caused by prompt and model changes (e.g. related to gender or culture), thus supporting the prompt engineering process and human-centric model behavior research. 

---
# A closer look at how large language models trust humans: patterns and biases 

**Authors**: Valeria Lerman, Yaniv Dover  

**Link**: [PDF](https://arxiv.org/pdf/2504.15801)  

**Abstract**: As large language models (LLMs) and LLM-based agents increasingly interact with humans in decision-making contexts, understanding the trust dynamics between humans and AI agents becomes a central concern. While considerable literature studies how humans trust AI agents, it is much less understood how LLM-based agents develop effective trust in humans. LLM-based agents likely rely on some sort of implicit effective trust in trust-related contexts (e.g., evaluating individual loan applications) to assist and affect decision making. Using established behavioral theories, we develop an approach that studies whether LLMs trust depends on the three major trustworthiness dimensions: competence, benevolence and integrity of the human subject. We also study how demographic variables affect effective trust. Across 43,200 simulated experiments, for five popular language models, across five different scenarios we find that LLM trust development shows an overall similarity to human trust development. We find that in most, but not all cases, LLM trust is strongly predicted by trustworthiness, and in some cases also biased by age, religion and gender, especially in financial scenarios. This is particularly true for scenarios common in the literature and for newer models. While the overall patterns align with human-like mechanisms of effective trust formation, different models exhibit variation in how they estimate trust; in some cases, trustworthiness and demographic factors are weak predictors of effective trust. These findings call for a better understanding of AI-to-human trust dynamics and monitoring of biases and trust development patterns to prevent unintended and potentially harmful outcomes in trust-sensitive applications of AI. 

---
# Automated Creativity Evaluation for Large Language Models: A Reference-Based Approach 

**Authors**: Ruizhe Li, Chiwei Zhu, Benfeng Xu, Xiaorui Wang, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2504.15784)  

**Abstract**: Creative writing is a key capability of Large Language Models (LLMs), with potential applications in literature, storytelling, and various creative domains. However, evaluating the creativity of machine-generated texts remains a significant challenge, as existing methods either rely on costly manual annotations or fail to align closely with human assessments. In this paper, we propose an effective automated evaluation method based on the Torrance Test of Creative Writing (TTCW), which evaluates creativity as product. Our method employs a reference-based Likert-style approach, scoring generated creative texts relative to high-quality reference texts across various tests. Experimental results demonstrate that our method significantly improves the alignment between LLM evaluations and human assessments, achieving a pairwise accuracy of 0.75 (+15\%). 

---
# Tina: Tiny Reasoning Models via LoRA 

**Authors**: Shangshang Wang, Julian Asilis, Ömer Faruk Akgül, Enes Burak Bilgin, Ollie Liu, Willie Neiswanger  

**Link**: [PDF](https://arxiv.org/pdf/2504.15777)  

**Abstract**: How cost-effectively can strong reasoning abilities be achieved in language models? Driven by this fundamental question, we present Tina, a family of tiny reasoning models achieved with high cost-efficiency. Notably, Tina demonstrates that substantial reasoning performance can be developed using only minimal resources, by applying parameter-efficient updates during reinforcement learning (RL), using low-rank adaptation (LoRA), to an already tiny 1.5B parameter base model. This minimalist approach produces models that achieve reasoning performance which is competitive with, and sometimes surpasses, SOTA RL reasoning models built upon the same base model. Crucially, this is achieved at a tiny fraction of the computational post-training cost employed by existing SOTA models. In fact, the best Tina model achieves a >20\% reasoning performance increase and 43.33\% Pass@1 accuracy on AIME24, at only \$9 USD post-training and evaluation cost (i.e., an estimated 260x cost reduction). Our work reveals the surprising effectiveness of efficient RL reasoning via LoRA. We validate this across multiple open-source reasoning datasets and various ablation settings starting with a single, fixed set of hyperparameters. Furthermore, we hypothesize that this effectiveness and efficiency stem from LoRA rapidly adapting the model to the structural format of reasoning rewarded by RL, while largely preserving the base model's underlying knowledge. In service of accessibility and open research, we fully open-source all code, training logs, and model weights \& checkpoints. 

---
# Subject islands do not reduce to construction-specific discourse function 

**Authors**: Mandy Cartner, Matthew Kogan, Nikolas Webster, Matthew Wagers, Ivy Sichel  

**Link**: [PDF](https://arxiv.org/pdf/2504.15688)  

**Abstract**: The term islands in linguistics refers to phrases from which extracting an element results in ungrammaticality (Ross, 1967). Grammatical subjects are considered islands because extracting a sub-part of a subject results in an ill-formed sentence, despite having a clear intended meaning (e.g., "Which topic did the article about inspire you?"). The generative tradition, which views syntax as autonomous of meaning and function, attributes this ungrammaticality to the abstract movement dependency between the wh-phrase and the subject-internal position with which it is associated for interpretation. However, research on language that emphasizes its communicative function suggests instead that syntactic constraints, including islands, can be explained based on the way different constructions package information. Accordingly, Abeillé et al. (2020) suggest that the islandhood of subjects is specific to the information structure of wh-questions, and propose that subjects are not islands for movement, but for focusing, due to their discourse-backgroundedness. This predicts that other constructions that differ in their information structure from wh-questions, but still involve movement, should not create a subject island effect. We test this prediction in three large-scale acceptability studies, using a super-additive design that singles out subject island violations, in three different constructions: wh-questions, relative clauses, and topicalization. We report evidence for a subject island effect in each construction type, despite only wh-questions introducing what Abeillé et al. (2020) call "a clash in information structure." We argue that this motivates an account of islands in terms of abstract, syntactic representations, independent of the communicative function associated with the constructions. 

---
# FinTextSim: Enhancing Financial Text Analysis with BERTopic 

**Authors**: Simon Jehnen, Joaquín Ordieres-Meré, Javier Villalba-Díez  

**Link**: [PDF](https://arxiv.org/pdf/2504.15683)  

**Abstract**: Recent advancements in information availability and computational capabilities have transformed the analysis of annual reports, integrating traditional financial metrics with insights from textual data. To extract valuable insights from this wealth of textual data, automated review processes, such as topic modeling, are crucial. This study examines the effectiveness of BERTopic, a state-of-the-art topic model relying on contextual embeddings, for analyzing Item 7 and Item 7A of 10-K filings from S&P 500 companies (2016-2022). Moreover, we introduce FinTextSim, a finetuned sentence-transformer model optimized for clustering and semantic search in financial contexts. Compared to all-MiniLM-L6-v2, the most widely used sentence-transformer, FinTextSim increases intratopic similarity by 81% and reduces intertopic similarity by 100%, significantly enhancing organizational clarity. We assess BERTopic's performance using embeddings from both FinTextSim and all-MiniLM-L6-v2. Our findings reveal that BERTopic only forms clear and distinct economic topic clusters when paired with FinTextSim's embeddings. Without FinTextSim, BERTopic struggles with misclassification and overlapping topics. Thus, FinTextSim is pivotal for advancing financial text analysis. FinTextSim's enhanced contextual embeddings, tailored for the financial domain, elevate the quality of future research and financial information. This improved quality of financial information will enable stakeholders to gain a competitive advantage, streamlining resource allocation and decision-making processes. Moreover, the improved insights have the potential to leverage business valuation and stock price prediction models. 

---
# Computational Typology 

**Authors**: Gerhard Jäger  

**Link**: [PDF](https://arxiv.org/pdf/2504.15642)  

**Abstract**: Typology is a subfield of linguistics that focuses on the study and classification of languages based on their structural features. Unlike genealogical classification, which examines the historical relationships between languages, typology seeks to understand the diversity of human languages by identifying common properties and patterns, known as universals. In recent years, computational methods have played an increasingly important role in typological research, enabling the analysis of large-scale linguistic data and the testing of hypotheses about language structure and evolution. This article provides an illustration of the benefits of computational statistical modeling in typology. 

---
# Cost-Effective Text Clustering with Large Language Models 

**Authors**: Hongtao Wang, Taiyan Zhang, Renchi Yang, Jianliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15640)  

**Abstract**: Text clustering aims to automatically partition a collection of text documents into distinct clusters based on linguistic features. In the literature, this task is usually framed as metric clustering based on text embeddings from pre-trained encoders or a graph clustering problem upon pairwise similarities from an oracle, e.g., a large ML model. Recently, large language models (LLMs) bring significant advancement in this field by offering contextualized text embeddings and highly accurate similarity scores, but meanwhile, present grand challenges to cope with substantial computational and/or financial overhead caused by numerous API-based queries or inference calls to the models.
In response, this paper proposes TECL, a cost-effective framework that taps into the feedback from LLMs for accurate text clustering within a limited budget of queries to LLMs. Under the hood, TECL adopts our EdgeLLM or TriangleLLM to construct must-link/cannot-link constraints for text pairs, and further leverages such constraints as supervision signals input to our weighted constrained clustering approach to generate clusters. Particularly, EdgeLLM (resp. TriangleLLM) enables the identification of informative text pairs (resp. triplets) for querying LLMs via well-thought-out greedy algorithms and accurate extraction of pairwise constraints through carefully-crafted prompting techniques. Our experiments on multiple benchmark datasets exhibit that TECL consistently and considerably outperforms existing solutions in unsupervised text clustering under the same query cost for LLMs. 

---
# Exploiting Contextual Knowledge in LLMs through V-usable Information based Layer Enhancement 

**Authors**: Xiaowei Yuan, Zhao Yang, Ziyang Huang, Yequan Wang, Siqi Fan, Yiming Ju, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15630)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in various tasks, yet they often struggle with context-faithfulness generations that properly reflect contextual knowledge. While existing approaches focus on enhancing the decoding strategies, they ignore the fundamental mechanism of how contextual information is processed within LLMs' internal states. As a result, LLMs remain limited in their ability to fully leverage contextual knowledge. In this paper, we propose Context-aware Layer Enhancement (CaLE), a novel intervention method that enhances the utilization of contextual knowledge within LLMs' internal representations. By employing V-usable information analysis, CaLE strategically amplifies the growth of contextual information at an optimal layer, thereby enriching representations in the final layer. Our experiments demonstrate that CaLE effectively improves context-faithful generation in Question-Answering tasks, particularly in scenarios involving unknown or conflicting contextual knowledge. 

---
# Exploring Next Token Prediction in Theory of Mind (ToM) Tasks: Comparative Experiments with GPT-2 and LLaMA-2 AI Models 

**Authors**: Pavan Yadav, Nikhil Khandalkar, Krishna Shinde, Lokesh B. Ramegowda, Rajarshi Das  

**Link**: [PDF](https://arxiv.org/pdf/2504.15604)  

**Abstract**: Language models have made significant progress in generating coherent text and predicting next tokens based on input prompts. This study compares the next-token prediction performance of two well-known models: OpenAI's GPT-2 and Meta's Llama-2-7b-chat-hf on Theory of Mind (ToM) tasks. To evaluate their capabilities, we built a dataset from 10 short stories sourced from the Explore ToM Dataset. We enhanced these stories by programmatically inserting additional sentences (infills) using GPT-4, creating variations that introduce different levels of contextual complexity. This setup enables analysis of how increasing context affects model performance. We tested both models under four temperature settings (0.01, 0.5, 1.0, 2.0) and evaluated their ability to predict the next token across three reasoning levels. Zero-order reasoning involves tracking the state, either current (ground truth) or past (memory). First-order reasoning concerns understanding another's mental state (e.g., "Does Anne know the apple is salted?"). Second-order reasoning adds recursion (e.g., "Does Anne think that Charles knows the apple is salted?").
Our results show that adding more infill sentences slightly reduces prediction accuracy, as added context increases complexity and ambiguity. Llama-2 consistently outperforms GPT-2 in prediction accuracy, especially at lower temperatures, demonstrating greater confidence in selecting the most probable token. As reasoning complexity rises, model responses diverge more. Notably, GPT-2 and Llama-2 display greater variability in predictions during first- and second-order reasoning tasks. These findings illustrate how model architecture, temperature, and contextual complexity influence next-token prediction, contributing to a better understanding of the strengths and limitations of current language models. 

---
# Instruction-Tuning Data Synthesis from Scratch via Web Reconstruction 

**Authors**: Yuxin Jiang, Yufei Wang, Chuhan Wu, Xinyi Dai, Yan Xu, Weinan Gan, Yasheng Wang, Xin Jiang, Lifeng Shang, Ruiming Tang, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15573)  

**Abstract**: The improvement of LLMs' instruction-following capabilities depends critically on the availability of high-quality instruction-response pairs. While existing automatic data synthetic methods alleviate the burden of manual curation, they often rely heavily on either the quality of seed data or strong assumptions about the structure and content of web documents. To tackle these challenges, we propose Web Reconstruction (WebR), a fully automated framework for synthesizing high-quality instruction-tuning (IT) data directly from raw web documents with minimal assumptions. Leveraging the inherent diversity of raw web content, we conceptualize web reconstruction as an instruction-tuning data synthesis task via a novel dual-perspective paradigm--Web as Instruction and Web as Response--where each web document is designated as either an instruction or a response to trigger the reconstruction process. Comprehensive experiments show that datasets generated by WebR outperform state-of-the-art baselines by up to 16.65% across four instruction-following benchmarks. Notably, WebR demonstrates superior compatibility, data efficiency, and scalability, enabling enhanced domain adaptation with minimal effort. The data and code are publicly available at this https URL. 

---
# LLM-based Semantic Augmentation for Harmful Content Detection 

**Authors**: Elyas Meguellati, Assaad Zeghina, Shazia Sadiq, Gianluca Demartini  

**Link**: [PDF](https://arxiv.org/pdf/2504.15548)  

**Abstract**: Recent advances in large language models (LLMs) have demonstrated strong performance on simple text classification tasks, frequently under zero-shot settings. However, their efficacy declines when tackling complex social media challenges such as propaganda detection, hateful meme classification, and toxicity identification. Much of the existing work has focused on using LLMs to generate synthetic training data, overlooking the potential of LLM-based text preprocessing and semantic augmentation. In this paper, we introduce an approach that prompts LLMs to clean noisy text and provide context-rich explanations, thereby enhancing training sets without substantial increases in data volume. We systematically evaluate on the SemEval 2024 multi-label Persuasive Meme dataset and further validate on the Google Jigsaw toxic comments and Facebook hateful memes datasets to assess generalizability. Our results reveal that zero-shot LLM classification underperforms on these high-context tasks compared to supervised models. In contrast, integrating LLM-based semantic augmentation yields performance on par with approaches that rely on human-annotated data, at a fraction of the cost. These findings underscore the importance of strategically incorporating LLMs into machine learning (ML) pipeline for social media classification tasks, offering broad implications for combating harmful content online. 

---
# llm-jp-modernbert: A ModernBERT Model Trained on a Large-Scale Japanese Corpus with Long Context Length 

**Authors**: Issa Sugiura, Kouta Nakayama, Yusuke Oda  

**Link**: [PDF](https://arxiv.org/pdf/2504.15544)  

**Abstract**: Encoder-only transformer models like BERT are widely adopted as a pre-trained backbone for tasks like sentence classification and retrieval. However, pretraining of encoder models with large-scale corpora and long contexts has been relatively underexplored compared to decoder-only transformers. In this work, we present llm-jp-modernbert, a ModernBERT model trained on a publicly available, massive Japanese corpus with a context length of 8192 tokens. While our model does not surpass existing baselines on downstream tasks, it achieves good results on fill-mask test evaluations. We also analyze the effect of context length expansion through pseudo-perplexity experiments. Furthermore, we investigate sentence embeddings in detail, analyzing their transitions during training and comparing them with those from other existing models, confirming similar trends with models sharing the same architecture. To support reproducibility and foster the development of long-context BERT, we release our model, along with the training and evaluation code. 

---
# Compass-V2 Technical Report 

**Authors**: Sophia Maria  

**Link**: [PDF](https://arxiv.org/pdf/2504.15527)  

**Abstract**: Predominant LLMs focus on high-resource languages while leaving low-resource languages, particularly those in Southeast Asia (SEA), underrepresented. In addition, those models are general-purpose and pay limited attention to the e-commerce domain. To overcome these limitations, we introduce Compass-v2, a lightweight Mixture-of-Experts (MoE) model specifically designed for Southeast Asian languages and e-commerce applications. To balance model performance and inference cost, the model is designed with 30B total parameters and 5B active parameters, incorporating both fine-grained and shared expert modules. To enhance multilingual performance, we curated and constructed a high-quality, industry-leading SEA dataset, to the best of our knowledge. To boost performance in the e-commerce domain, we built a dataset comprising hundreds of billions of tokens, sourced through external data mining and internal platform collection. Besides, we pioneered a hybrid reasoning model that supports both fast thinking and deep thinking within a unified framework to enhance the reasoning capabilities, diverging from the conventional industry practice of deploying two separate models. Through extensive experimental evaluations, our model demonstrates state-of-the-art SEA multilingual and e-commerce performance among sub-30B models, while maintaining significantly lower inference cost. 

---
# IPBench: Benchmarking the Knowledge of Large Language Models in Intellectual Property 

**Authors**: Qiyao Wang, Guhong Chen, Hongbo Wang, Huaren Liu, Minghui Zhu, Zhifei Qin, Linwei Li, Yilin Yue, Shiqiang Wang, Jiayan Li, Yihang Wu, Ziqiang Liu, Longze Chen, Run Luo, Liyang Fan, Jiaming Li, Lei Zhang, Kan Xu, Hongfei Lin, Hamid Alinejad-Rokny, Shiwen Ni, Yuan Lin, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15524)  

**Abstract**: Intellectual Property (IP) is a unique domain that integrates technical and legal knowledge, making it inherently complex and knowledge-intensive. As large language models (LLMs) continue to advance, they show great potential for processing IP tasks, enabling more efficient analysis, understanding, and generation of IP-related content. However, existing datasets and benchmarks either focus narrowly on patents or cover limited aspects of the IP field, lacking alignment with real-world scenarios. To bridge this gap, we introduce the first comprehensive IP task taxonomy and a large, diverse bilingual benchmark, IPBench, covering 8 IP mechanisms and 20 tasks. This benchmark is designed to evaluate LLMs in real-world intellectual property applications, encompassing both understanding and generation. We benchmark 16 LLMs, ranging from general-purpose to domain-specific models, and find that even the best-performing model achieves only 75.8% accuracy, revealing substantial room for improvement. Notably, open-source IP and law-oriented models lag behind closed-source general-purpose models. We publicly release all data and code of IPBench and will continue to update it with additional IP-related tasks to better reflect real-world challenges in the intellectual property domain. 

---
# The Bitter Lesson Learned from 2,000+ Multilingual Benchmarks 

**Authors**: Minghao Wu, Weixuan Wang, Sinuo Liu, Huifeng Yin, Xintong Wang, Yu Zhao, Chenyang Lyu, Longyue Wang, Weihua Luo, Kaifu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15521)  

**Abstract**: As large language models (LLMs) continue to advance in linguistic capabilities, robust multilingual evaluation has become essential for promoting equitable technological progress. This position paper examines over 2,000 multilingual (non-English) benchmarks from 148 countries, published between 2021 and 2024, to evaluate past, present, and future practices in multilingual benchmarking. Our findings reveal that, despite significant investments amounting to tens of millions of dollars, English remains significantly overrepresented in these benchmarks. Additionally, most benchmarks rely on original language content rather than translations, with the majority sourced from high-resource countries such as China, India, Germany, the UK, and the USA. Furthermore, a comparison of benchmark performance with human judgments highlights notable disparities. STEM-related tasks exhibit strong correlations with human evaluations (0.70 to 0.85), while traditional NLP tasks like question answering (e.g., XQuAD) show much weaker correlations (0.11 to 0.30). Moreover, translating English benchmarks into other languages proves insufficient, as localized benchmarks demonstrate significantly higher alignment with local human judgments (0.68) than their translated counterparts (0.47). This underscores the importance of creating culturally and linguistically tailored benchmarks rather than relying solely on translations. Through this comprehensive analysis, we highlight six key limitations in current multilingual evaluation practices, propose the guiding principles accordingly for effective multilingual benchmarking, and outline five critical research directions to drive progress in the field. Finally, we call for a global collaborative effort to develop human-aligned benchmarks that prioritize real-world applications. 

---
# SimulS2S-LLM: Unlocking Simultaneous Inference of Speech LLMs for Speech-to-Speech Translation 

**Authors**: Keqi Deng, Wenxi Chen, Xie Chen, Philip C. Woodland  

**Link**: [PDF](https://arxiv.org/pdf/2504.15509)  

**Abstract**: Simultaneous speech translation (SST) outputs translations in parallel with streaming speech input, balancing translation quality and latency. While large language models (LLMs) have been extended to handle the speech modality, streaming remains challenging as speech is prepended as a prompt for the entire generation process. To unlock LLM streaming capability, this paper proposes SimulS2S-LLM, which trains speech LLMs offline and employs a test-time policy to guide simultaneous inference. SimulS2S-LLM alleviates the mismatch between training and inference by extracting boundary-aware speech prompts that allows it to be better matched with text input data. SimulS2S-LLM achieves simultaneous speech-to-speech translation (Simul-S2ST) by predicting discrete output speech tokens and then synthesising output speech using a pre-trained vocoder. An incremental beam search is designed to expand the search space of speech token prediction without increasing latency. Experiments on the CVSS speech data show that SimulS2S-LLM offers a better translation quality-latency trade-off than existing methods that use the same training data, such as improving ASR-BLEU scores by 3 points at similar latency. 

---
# Speculative Sampling via Exponential Races 

**Authors**: Szymon Kobus, Deniz Gündüz  

**Link**: [PDF](https://arxiv.org/pdf/2504.15475)  

**Abstract**: Speculative decoding accelerates large language model inference using a smaller draft model. In this paper, we establish a surprising connection between speculative decoding and channel simulation, which aims at simulating a noisy channel using as few bits as possible. This connection allows us to provide an information-theoretic analysis of the speed up that can be achieved by speculative decoding. Leveraging this link, we derive an explicit relation between generation speed-up and the number of tokens $k$ generated by the draft model for large $k$, which serves as an upper bound for all $k$. We also propose a novel speculative decoding method via exponential race ERSD that matches state-of-the-art performance. 

---
# Bigram Subnetworks: Mapping to Next Tokens in Transformer Language Models 

**Authors**: Tyler A. Chang, Benjamin K. Bergen  

**Link**: [PDF](https://arxiv.org/pdf/2504.15471)  

**Abstract**: In Transformer language models, activation vectors transform from current token embeddings to next token predictions as they pass through the model. To isolate a minimal form of this transformation, we identify language model subnetworks that make bigram predictions, naive next token predictions based only on the current token. We find that bigram subnetworks can be found in fully trained language models up to 1B parameters, and these subnetworks are critical for model performance even when they consist of less than 0.2% of model parameters. Bigram subnetworks are concentrated in the first Transformer MLP layer, and they overlap significantly with subnetworks trained to optimally prune a given model. Mechanistically, the bigram subnetworks often recreate a pattern from the full models where the first layer induces a sharp change that aligns activations with next token predictions rather than current token representations. Our results demonstrate that bigram subnetworks comprise a minimal subset of parameters that are both necessary and sufficient for basic next token predictions in language models, and they help drive the transformation from current to next token activations in the residual stream. These subnetworks can lay a foundation for studying language model circuits by building up from a minimal circuit rather than the traditional approach of ablating circuits from a full model. 

---
# Feeding LLM Annotations to BERT Classifiers at Your Own Risk 

**Authors**: Yucheng Lu, Kazimier Smith  

**Link**: [PDF](https://arxiv.org/pdf/2504.15432)  

**Abstract**: Using LLM-generated labels to fine-tune smaller encoder-only models for text classification has gained popularity in various settings. While this approach may be justified in simple and low-stakes applications, we conduct empirical analysis to demonstrate how the perennial curse of training on synthetic data manifests itself in this specific setup. Compared to models trained on gold labels, we observe not only the expected performance degradation in accuracy and F1 score, but also increased instability across training runs and premature performance plateaus. These findings cast doubts on the reliability of such approaches in real-world applications. We contextualize the observed phenomena through the lens of error propagation and offer several practical mitigation strategies, including entropy-based filtering and ensemble techniques. Although these heuristics offer partial relief, they do not fully resolve the inherent risks of propagating non-random errors from LLM annotations to smaller classifiers, underscoring the need for caution when applying this workflow in high-stakes text classification tasks. 

---
# Trillion 7B Technical Report 

**Authors**: Sungjun Han, Juyoung Suk, Suyeong An, Hyungguk Kim, Kyuseok Kim, Wonsuk Yang, Seungtaek Choi, Jamin Shin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15431)  

**Abstract**: We introduce Trillion-7B, the most token-efficient Korean-centric multilingual LLM available. Our novel Cross-lingual Document Attention (XLDA) mechanism enables highly efficient and effective knowledge transfer from English to target languages like Korean and Japanese. Combined with optimized data mixtures, language-specific filtering, and tailored tokenizer construction, Trillion-7B achieves competitive performance while dedicating only 10\% of its 2T training tokens to multilingual data and requiring just 59.4K H100 GPU hours (\$148K) for full training. Comprehensive evaluations across 27 benchmarks in four languages demonstrate Trillion-7B's robust multilingual performance and exceptional cross-lingual consistency. 

---
# Tell Me What You Know About Sexism: Expert-LLM Interaction Strategies and Co-Created Definitions for Zero-Shot Sexism Detection 

**Authors**: Myrthe Reuver, Indira Sen, Matteo Melis, Gabriella Lapesa  

**Link**: [PDF](https://arxiv.org/pdf/2504.15392)  

**Abstract**: This paper investigates hybrid intelligence and collaboration between researchers of sexism and Large Language Models (LLMs), with a four-component pipeline. First, nine sexism researchers answer questions about their knowledge of sexism and of LLMs. They then participate in two interactive experiments involving an LLM (GPT3.5). The first experiment has experts assessing the model's knowledge about sexism and suitability for use in research. The second experiment tasks them with creating three different definitions of sexism: an expert-written definition, an LLM-written one, and a co-created definition. Lastly, zero-shot classification experiments use the three definitions from each expert in a prompt template for sexism detection, evaluating GPT4o on 2.500 texts sampled from five sexism benchmarks. We then analyze the resulting 67.500 classification decisions. The LLM interactions lead to longer and more complex definitions of sexism. Expert-written definitions on average perform poorly compared to LLM-generated definitions. However, some experts do improve classification performance with their co-created definitions of sexism, also experts who are inexperienced in using LLMs. 

---
# Exploring Compositional Generalization (in ReCOGS_pos) by Transformers using Restricted Access Sequence Processing (RASP) 

**Authors**: William Bruns  

**Link**: [PDF](https://arxiv.org/pdf/2504.15349)  

**Abstract**: Humans understand new combinations of words encountered if they are combinations of words recognized from different contexts, an ability called Compositional Generalization. The COGS benchmark (Kim and Linzen, 2020) arXiv:2010.05465 reports 0% accuracy for Transformer models on some structural generalizations. We use (Weiss et al., 2021) arXiv:2106.06981's Restricted Access Sequence Processing (RASP), a Transformer-equivalent programming language, to prove by construction that a Transformer encoder-decoder can perform the semantically equivalent ReCOGS_pos (Wu et al., 2024) arXiv:2303.13716 variant of COGS systematically and compositionally: Our RASP model attains 100% semantic exact match on the ReCOGS test set and 100% SEM on all generalization splits except obj_pp_to_subj_pp which gets 92%. Furthermore, our RASP model shows the ReCOGS_pos task does not require a hierarchical or tree-structured solution: we use word-level tokens with an "embedding" layer that tags with possible parts of speech, applying just once per encoder pass 19 attention-head compatible flat pattern-matching rules, shown using grammar coverage (Zeller et al., 2023) to be learnable from the training data, plus general prepositional phrase (pp) handling and sentential complement (cp) handling logic, and output the next logical form (LF) token (repeating until the LF is complete). The model does not apply recursive, tree-structured rules like 'np_det pp np -> np_pp -> np', but scores 100% semantic and string exact match on pp recursion, cp recursion using the decoder loop. 

---
# Survey of Video Diffusion Models: Foundations, Implementations, and Applications 

**Authors**: Yimu Wang, Xuye Liu, Wei Pang, Li Ma, Shuai Yuan, Paul Debevec, Ning Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.16081)  

**Abstract**: Recent advances in diffusion models have revolutionized video generation, offering superior temporal consistency and visual quality compared to traditional generative adversarial networks-based approaches. While this emerging field shows tremendous promise in applications, it faces significant challenges in motion consistency, computational efficiency, and ethical considerations. This survey provides a comprehensive review of diffusion-based video generation, examining its evolution, technical foundations, and practical applications. We present a systematic taxonomy of current methodologies, analyze architectural innovations and optimization strategies, and investigate applications across low-level vision tasks such as denoising and super-resolution. Additionally, we explore the synergies between diffusionbased video generation and related domains, including video representation learning, question answering, and retrieval. Compared to the existing surveys (Lei et al., 2024a;b; Melnik et al., 2024; Cao et al., 2023; Xing et al., 2024c) which focus on specific aspects of video generation, such as human video synthesis (Lei et al., 2024a) or long-form content generation (Lei et al., 2024b), our work provides a broader, more updated, and more fine-grained perspective on diffusion-based approaches with a special section for evaluation metrics, industry solutions, and training engineering techniques in video generation. This survey serves as a foundational resource for researchers and practitioners working at the intersection of diffusion models and video generation, providing insights into both the theoretical frameworks and practical implementations that drive this rapidly evolving field. A structured list of related works involved in this survey is also available on this https URL. 

---
# How Private is Your Attention? Bridging Privacy with In-Context Learning 

**Authors**: Soham Bonnerjee, Zhen Wei, Yeon, Anna Asch, Sagnik Nandy, Promit Ghosal  

**Link**: [PDF](https://arxiv.org/pdf/2504.16000)  

**Abstract**: In-context learning (ICL)-the ability of transformer-based models to perform new tasks from examples provided at inference time-has emerged as a hallmark of modern language models. While recent works have investigated the mechanisms underlying ICL, its feasibility under formal privacy constraints remains largely unexplored. In this paper, we propose a differentially private pretraining algorithm for linear attention heads and present the first theoretical analysis of the privacy-accuracy trade-off for ICL in linear regression. Our results characterize the fundamental tension between optimization and privacy-induced noise, formally capturing behaviors observed in private training via iterative methods. Additionally, we show that our method is robust to adversarial perturbations of training prompts, unlike standard ridge regression. All theoretical findings are supported by extensive simulations across diverse settings. 

---
# TrustGeoGen: Scalable and Formal-Verified Data Engine for Trustworthy Multi-modal Geometric Problem Solving 

**Authors**: Daocheng Fu, Zijun Chen, Renqiu Xia, Qi Liu, Yuan Feng, Hongbin Zhou, Renrui Zhang, Shiyang Feng, Peng Gao, Junchi Yan, Botian Shi, Bo Zhang, Yu Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2504.15780)  

**Abstract**: Mathematical geometric problem solving (GPS) often requires effective integration of multimodal information and verifiable logical coherence. Despite the fast development of large language models in general problem solving, it remains unresolved regarding with both methodology and benchmarks, especially given the fact that exiting synthetic GPS benchmarks are often not self-verified and contain noise and self-contradicted information due to the illusion of LLMs. In this paper, we propose a scalable data engine called TrustGeoGen for problem generation, with formal verification to provide a principled benchmark, which we believe lays the foundation for the further development of methods for GPS. The engine synthesizes geometric data through four key innovations: 1) multimodal-aligned generation of diagrams, textual descriptions, and stepwise solutions; 2) formal verification ensuring rule-compliant reasoning paths; 3) a bootstrapping mechanism enabling complexity escalation via recursive state generation and 4) our devised GeoExplore series algorithms simultaneously produce multi-solution variants and self-reflective backtracking traces. By formal logical verification, TrustGeoGen produces GeoTrust-200K dataset with guaranteed modality integrity, along with GeoTrust-test testset. Experiments reveal the state-of-the-art models achieve only 49.17\% accuracy on GeoTrust-test, demonstrating its evaluation stringency. Crucially, models trained on GeoTrust achieve OOD generalization on GeoQA, significantly reducing logical inconsistencies relative to pseudo-label annotated by OpenAI-o1. Our code is available at this https URL 

---
# VeriCoder: Enhancing LLM-Based RTL Code Generation through Functional Correctness Validation 

**Authors**: Anjiang Wei, Huanmi Tan, Tarun Suresh, Daniel Mendoza, Thiago S. F. X. Teixeira, Ke Wang, Caroline Trippel, Alex Aiken  

**Link**: [PDF](https://arxiv.org/pdf/2504.15659)  

**Abstract**: Recent advances in Large Language Models (LLMs) have sparked growing interest in applying them to Electronic Design Automation (EDA) tasks, particularly Register Transfer Level (RTL) code generation. While several RTL datasets have been introduced, most focus on syntactic validity rather than functional validation with tests, leading to training examples that compile but may not implement the intended behavior. We present VERICODER, a model for RTL code generation fine-tuned on a dataset validated for functional correctness. This fine-tuning dataset is constructed using a novel methodology that combines unit test generation with feedback-directed refinement. Given a natural language specification and an initial RTL design, we prompt a teacher model (GPT-4o-mini) to generate unit tests and iteratively revise the RTL design based on its simulation results using the generated tests. If necessary, the teacher model also updates the tests to ensure they comply with the natural language specification. As a result of this process, every example in our dataset is functionally validated, consisting of a natural language description, an RTL implementation, and passing tests. Fine-tuned on this dataset of over 125,000 examples, VERICODER achieves state-of-the-art metrics in functional correctness on VerilogEval and RTLLM, with relative gains of up to 71.7% and 27.4% respectively. An ablation study further shows that models trained on our functionally validated dataset outperform those trained on functionally non-validated datasets, underscoring the importance of high-quality datasets in RTL code generation. 

---
# CiteFix: Enhancing RAG Accuracy Through Post-Processing Citation Correction 

**Authors**: Harsh Maheshwari, Srikanth Tenneti, Alwarappan Nakkiran  

**Link**: [PDF](https://arxiv.org/pdf/2504.15629)  

**Abstract**: Retrieval Augmented Generation (RAG) has emerged as a powerful application of Large Language Models (LLMs), revolutionizing information search and consumption. RAG systems combine traditional search capabilities with LLMs to generate comprehensive answers to user queries, ideally with accurate citations. However, in our experience of developing a RAG product, LLMs often struggle with source attribution, aligning with other industry studies reporting citation accuracy rates of only about 74% for popular generative search engines. To address this, we present efficient post-processing algorithms to improve citation accuracy in LLM-generated responses, with minimal impact on latency and cost. Our approaches cross-check generated citations against retrieved articles using methods including keyword + semantic matching, fine tuned model with BERTScore, and a lightweight LLM-based technique. Our experimental results demonstrate a relative improvement of 15.46% in the overall accuracy metrics of our RAG system. This significant enhancement potentially enables a shift from our current larger language model to a relatively smaller model that is approximately 12x more cost-effective and 3x faster in inference time, while maintaining comparable performance. This research contributes to enhancing the reliability and trustworthiness of AI-generated content in information retrieval and summarization tasks which is critical to gain customer trust especially in commercial products. 

---
# A Comprehensive Survey in LLM(-Agent) Full Stack Safety: Data, Training and Deployment 

**Authors**: Kun Wang, Guibin Zhang, Zhenhong Zhou, Jiahao Wu, Miao Yu, Shiqian Zhao, Chenlong Yin, Jinhu Fu, Yibo Yan, Hanjun Luo, Liang Lin, Zhihao Xu, Haolang Lu, Xinye Cao, Xinyun Zhou, Weifei Jin, Fanci Meng, Junyuan Mao, Hao Wu, Minghe Wang, Fan Zhang, Junfeng Fang, Chengwei Liu, Yifan Zhang, Qiankun Li, Chongye Guo, Yalan Qin, Yi Ding, Donghai Hong, Jiaming Ji, Xinfeng Li, Yifan Jiang, Dongxia Wang, Yihao Huang, Yufei Guo, Jen-tse Huang, Yanwei Yue, Wenke Huang, Guancheng Wan, Tianlin Li, Lei Bai, Jie Zhang, Qing Guo, Jingyi Wang, Tianlong Chen, Joey Tianyi Zhou, Xiaojun Jia, Weisong Sun, Cong Wu, Jing Chen, Xuming Hu, Yiming Li, Xiao Wang, Ningyu Zhang, Luu Anh Tuan, Guowen Xu, Tianwei Zhang, Xingjun Ma, Xiang Wang, Bo An, Jun Sun, Mohit Bansal, Shirui Pan, Yuval Elovici, Bhavya Kailkhura, Bo Li, Yaodong Yang, Hongwei Li, Wenyuan Xu, Yizhou Sun, Wei Wang, Qing Li, Ke Tang, Yu-Gang Jiang, Felix Juefei-Xu, Hui Xiong, Xiaofeng Wang, Shuicheng Yan, Dacheng Tao, Philip S. Yu, Qingsong Wen, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15585)  

**Abstract**: The remarkable success of Large Language Models (LLMs) has illuminated a promising pathway toward achieving Artificial General Intelligence for both academic and industrial communities, owing to their unprecedented performance across various applications. As LLMs continue to gain prominence in both research and commercial domains, their security and safety implications have become a growing concern, not only for researchers and corporations but also for every nation. Currently, existing surveys on LLM safety primarily focus on specific stages of the LLM lifecycle, e.g., deployment phase or fine-tuning phase, lacking a comprehensive understanding of the entire "lifechain" of LLMs. To address this gap, this paper introduces, for the first time, the concept of "full-stack" safety to systematically consider safety issues throughout the entire process of LLM training, deployment, and eventual commercialization. Compared to the off-the-shelf LLM safety surveys, our work demonstrates several distinctive advantages: (I) Comprehensive Perspective. We define the complete LLM lifecycle as encompassing data preparation, pre-training, post-training, deployment and final commercialization. To our knowledge, this represents the first safety survey to encompass the entire lifecycle of LLMs. (II) Extensive Literature Support. Our research is grounded in an exhaustive review of over 800+ papers, ensuring comprehensive coverage and systematic organization of security issues within a more holistic understanding. (III) Unique Insights. Through systematic literature analysis, we have developed reliable roadmaps and perspectives for each chapter. Our work identifies promising research directions, including safety in data generation, alignment techniques, model editing, and LLM-based agent systems. These insights provide valuable guidance for researchers pursuing future work in this field. 

---
# CAPTURe: Evaluating Spatial Reasoning in Vision Language Models via Occluded Object Counting 

**Authors**: Atin Pothiraj, Elias Stengel-Eskin, Jaemin Cho, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2504.15485)  

**Abstract**: Recognizing and reasoning about occluded (partially or fully hidden) objects is vital to understanding visual scenes, as occlusions frequently occur in real-world environments and act as obstacles for spatial comprehension. To test models' ability to reason about multiple occluded objects, we introduce a novel task, Counting Amodally for Patterns Through Unseen REgions (CAPTURe), which requires a model to count objects arranged in a pattern by inferring how the pattern continues behind an occluder (an object which blocks parts of the scene). CAPTURe requires both recognizing visual patterns and reasoning, making it a useful testbed for evaluating vision-language models (VLMs) on whether they understand occluded patterns and possess spatial understanding skills. By requiring models to reason about occluded objects, CAPTURe also tests VLMs' ability to form world models that would allow them to fill in missing information. CAPTURe consists of two parts: (1) CAPTURe-real, with manually filtered images of real objects in patterns and (2) CAPTURe-synthetic, a controlled diagnostic with generated patterned images. We evaluate four strong VLMs (GPT-4o, Intern-VL2, Molmo, and Qwen2-VL) on CAPTURe, finding that models struggle to count on both occluded and unoccluded patterns. Crucially, we find that models perform worse with occlusion, suggesting that VLMs are also deficient in inferring unseen spatial relationships: even the strongest VLMs like GPT-4o fail to count with occlusion. In contrast, we find that humans achieve very little error on CAPTURe. We also find that providing auxiliary information of occluded object locations increases performance, underscoring that the model error comes both from an inability to handle occlusion as well as difficulty counting in images. 

---
# Learning Adaptive Parallel Reasoning with Language Models 

**Authors**: Jiayi Pan, Xiuyu Li, Long Lian, Charlie Snell, Yifei Zhou, Adam Yala, Trevor Darrell, Kurt Keutzer, Alane Suhr  

**Link**: [PDF](https://arxiv.org/pdf/2504.15466)  

**Abstract**: Scaling inference-time computation has substantially improved the reasoning capabilities of language models. However, existing methods have significant limitations: serialized chain-of-thought approaches generate overly long outputs, leading to increased latency and exhausted context windows, while parallel methods such as self-consistency suffer from insufficient coordination, resulting in redundant computations and limited performance gains. To address these shortcomings, we propose Adaptive Parallel Reasoning (APR), a novel reasoning framework that enables language models to orchestrate both serialized and parallel computations end-to-end. APR generalizes existing reasoning methods by enabling adaptive multi-threaded inference using spawn() and join() operations. A key innovation is our end-to-end reinforcement learning strategy, optimizing both parent and child inference threads to enhance task success rate without requiring predefined reasoning structures. Experiments on the Countdown reasoning task demonstrate significant benefits of APR: (1) higher performance within the same context window (83.4% vs. 60.0% at 4k context); (2) superior scalability with increased computation (80.1% vs. 66.6% at 20k total tokens); (3) improved accuracy at equivalent latency (75.2% vs. 57.3% at approximately 5,000ms). APR represents a step towards enabling language models to autonomously optimize their reasoning processes through adaptive allocation of computation. 

---
# Real-Time Sentiment Insights from X Using VADER, DistilBERT, and Web-Scraped Data 

**Authors**: Yanampally Abhiram Reddy, Siddhi Agarwal, Vikram Parashar, Arshiya Arora  

**Link**: [PDF](https://arxiv.org/pdf/2504.15448)  

**Abstract**: In the age of social media, understanding public sentiment toward major corporations is crucial for investors, policymakers, and researchers. This paper presents a comprehensive sentiment analysis system tailored for corporate reputation monitoring, combining Natural Language Processing (NLP) and machine learning techniques to accurately interpret public opinion in real time. The methodology integrates a hybrid sentiment detection framework leveraging both rule-based models (VADER) and transformer-based deep learning models (DistilBERT), applied to social media data from multiple platforms. The system begins with robust preprocessing involving noise removal and text normalization, followed by sentiment classification using an ensemble approach to ensure both interpretability and contextual accuracy. Results are visualized through sentiment distribution plots, comparative analyses, and temporal sentiment trends for enhanced interpretability. Our analysis reveals significant disparities in public sentiment across major corporations, with companies like Amazon (81.2) and Samsung (45.8) receiving excellent sentiment scores, while Microsoft (21.7) and Walmart (21.9) exhibit poor sentiment profiles. These findings demonstrate the utility of our multi-source sentiment framework in providing actionable insights regarding corporate public perception, enabling stakeholders to make informed strategic decisions based on comprehensive sentiment analysis. 

---
# IV-Bench: A Benchmark for Image-Grounded Video Perception and Reasoning in Multimodal LLMs 

**Authors**: David Ma, Yuanxing Zhang, Jincheng Ren, Jarvis Guo, Yifan Yao, Zhenlin Wei, Zhenzhu Yang, Zhongyuan Peng, Boyu Feng, Jun Ma, Xiao Gu, Zhoufutu Wen, King Zhu, Yancheng He, Meng Cao, Shiwen Ni, Jiaheng Liu, Wenhao Huang, Ge Zhang, Xiaojie Jin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15415)  

**Abstract**: Existing evaluation frameworks for Multimodal Large Language Models (MLLMs) primarily focus on image reasoning or general video understanding tasks, largely overlooking the significant role of image context in video comprehension. To bridge this gap, we propose IV-Bench, the first comprehensive benchmark for evaluating Image-Grounded Video Perception and Reasoning. IV-Bench consists of 967 videos paired with 2,585 meticulously annotated image-text queries across 13 tasks (7 perception and 6 reasoning tasks) and 5 representative categories. Extensive evaluations of state-of-the-art open-source (e.g., InternVL2.5, Qwen2.5-VL) and closed-source (e.g., GPT-4o, Gemini2-Flash and Gemini2-Pro) MLLMs demonstrate that current models substantially underperform in image-grounded video Perception and Reasoning, merely achieving at most 28.9% accuracy. Further analysis reveals key factors influencing model performance on IV-Bench, including inference pattern, frame number, and resolution. Additionally, through a simple data synthesis approach, we demonstratethe challenges of IV- Bench extend beyond merely aligning the data format in the training proecss. These findings collectively provide valuable insights for future research. Our codes and data are released in this https URL. 

---
# Towards Understanding Camera Motions in Any Video 

**Authors**: Zhiqiu Lin, Siyuan Cen, Daniel Jiang, Jay Karhade, Hewei Wang, Chancharik Mitra, Tiffany Ling, Yuhan Huang, Sifan Liu, Mingyu Chen, Rushikesh Zawar, Xue Bai, Yilun Du, Chuang Gan, Deva Ramanan  

**Link**: [PDF](https://arxiv.org/pdf/2504.15376)  

**Abstract**: We introduce CameraBench, a large-scale dataset and benchmark designed to assess and improve camera motion understanding. CameraBench consists of ~3,000 diverse internet videos, annotated by experts through a rigorous multi-stage quality control process. One of our contributions is a taxonomy of camera motion primitives, designed in collaboration with cinematographers. We find, for example, that some motions like "follow" (or tracking) require understanding scene content like moving subjects. We conduct a large-scale human study to quantify human annotation performance, revealing that domain expertise and tutorial-based training can significantly enhance accuracy. For example, a novice may confuse zoom-in (a change of intrinsics) with translating forward (a change of extrinsics), but can be trained to differentiate the two. Using CameraBench, we evaluate Structure-from-Motion (SfM) and Video-Language Models (VLMs), finding that SfM models struggle to capture semantic primitives that depend on scene content, while VLMs struggle to capture geometric primitives that require precise estimation of trajectories. We then fine-tune a generative VLM on CameraBench to achieve the best of both worlds and showcase its applications, including motion-augmented captioning, video question answering, and video-text retrieval. We hope our taxonomy, benchmark, and tutorials will drive future efforts towards the ultimate goal of understanding camera motions in any video. 

---
# LongPerceptualThoughts: Distilling System-2 Reasoning for System-1 Perception 

**Authors**: Yuan-Hong Liao, Sven Elflein, Liu He, Laura Leal-Taixé, Yejin Choi, Sanja Fidler, David Acuna  

**Link**: [PDF](https://arxiv.org/pdf/2504.15362)  

**Abstract**: Recent reasoning models through test-time scaling have demonstrated that long chain-of-thoughts can unlock substantial performance boosts in hard reasoning tasks such as math and code. However, the benefit of such long thoughts for system-2 reasoning is relatively less explored in other domains such as perceptual tasks where shallower, system-1 reasoning seems sufficient. In this paper, we introduce LongPerceptualThoughts, a new synthetic dataset with 30K long-thought traces for perceptual tasks. The key challenges in synthesizing elaborate reasoning thoughts for perceptual tasks are that off-the-shelf models are not yet equipped with such thinking behavior and that it is not straightforward to build a reliable process verifier for perceptual tasks. Thus, we propose a novel three-stage data synthesis framework that first synthesizes verifiable multiple-choice questions from dense image descriptions, then extracts simple CoTs from VLMs for those verifiable problems, and finally expands those simple thoughts to elaborate long thoughts via frontier reasoning models. In controlled experiments with a strong instruction-tuned 7B model, we demonstrate notable improvements over existing visual reasoning data-generation methods. Our model, trained on the generated dataset, achieves an average +3.4 points improvement over 5 vision-centric benchmarks, including +11.8 points on V$^*$ Bench. Notably, despite being tuned for vision tasks, it also improves performance on the text reasoning benchmark, MMLU-Pro, by +2 points. 

---
# Med-CoDE: Medical Critique based Disagreement Evaluation Framework 

**Authors**: Mohit Gupta, Akiko Aizawa, Rajiv Ratn Shah  

**Link**: [PDF](https://arxiv.org/pdf/2504.15330)  

**Abstract**: The emergence of large language models (LLMs) has significantly influenced numerous fields, including healthcare, by enhancing the capabilities of automated systems to process and generate human-like text. However, despite their advancements, the reliability and accuracy of LLMs in medical contexts remain critical concerns. Current evaluation methods often lack robustness and fail to provide a comprehensive assessment of LLM performance, leading to potential risks in clinical settings. In this work, we propose Med-CoDE, a specifically designed evaluation framework for medical LLMs to address these challenges. The framework leverages a critique-based approach to quantitatively measure the degree of disagreement between model-generated responses and established medical ground truths. This framework captures both accuracy and reliability in medical settings. The proposed evaluation framework aims to fill the existing gap in LLM assessment by offering a systematic method to evaluate the quality and trustworthiness of medical LLMs. Through extensive experiments and case studies, we illustrate the practicality of our framework in providing a comprehensive and reliable evaluation of medical LLMs. 

---
