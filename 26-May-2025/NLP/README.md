# The Staircase of Ethics: Probing LLM Value Priorities through Multi-Step Induction to Complex Moral Dilemmas 

**Authors**: Ya Wu, Qiang Sheng, Danding Wang, Guang Yang, Yifan Sun, Zhengjia Wang, Yuyan Bu, Juan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.18154)  

**Abstract**: Ethical decision-making is a critical aspect of human judgment, and the growing use of LLMs in decision-support systems necessitates a rigorous evaluation of their moral reasoning capabilities. However, existing assessments primarily rely on single-step evaluations, failing to capture how models adapt to evolving ethical challenges. Addressing this gap, we introduce the Multi-step Moral Dilemmas (MMDs), the first dataset specifically constructed to evaluate the evolving moral judgments of LLMs across 3,302 five-stage dilemmas. This framework enables a fine-grained, dynamic analysis of how LLMs adjust their moral reasoning across escalating dilemmas. Our evaluation of nine widely used LLMs reveals that their value preferences shift significantly as dilemmas progress, indicating that models recalibrate moral judgments based on scenario complexity. Furthermore, pairwise value comparisons demonstrate that while LLMs often prioritize the value of care, this value can sometimes be superseded by fairness in certain contexts, highlighting the dynamic and context-dependent nature of LLM ethical reasoning. Our findings call for a shift toward dynamic, context-aware evaluation paradigms, paving the way for more human-aligned and value-sensitive development of LLMs. 

---
# Fann or Flop: A Multigenre, Multiera Benchmark for Arabic Poetry Understanding in LLMs 

**Authors**: Wafa Alghallabi, Ritesh Thawkar, Sara Ghaboura, Ketan More, Omkar Thawakar, Hisham Cholakkal, Salman Khan, Rao Muhammad Anwer  

**Link**: [PDF](https://arxiv.org/pdf/2505.18152)  

**Abstract**: Arabic poetry stands as one of the most sophisticated and culturally embedded forms of expression in the Arabic language, known for its layered meanings, stylistic diversity, and deep historical continuity. Although large language models (LLMs) have demonstrated strong performance across languages and tasks, their ability to understand Arabic poetry remains largely unexplored. In this work, we introduce `Fann or Flop`, the first benchmark designed to assess the comprehension of Arabic poetry by LLMs in twelve historical eras, covering 21 core poetic genres and a variety of metrical forms, from classical structures to contemporary free verse. The benchmark comprises a curated corpus of poems with explanations that assess semantic understanding, metaphor interpretation, prosodic awareness, and cultural context. We argue that poetic comprehension offers a strong indicator for testing how good the LLM is in understanding classical Arabic through the Arabic poetry. Unlike surface-level tasks, this domain demands deeper interpretive reasoning and cultural sensitivity. Our evaluation of state-of-the-art LLMs shows that most models struggle with poetic understanding despite strong results on standard Arabic benchmarks. We release `Fann or Flop` along with the evaluation suite as an open-source resource to enable rigorous evaluation and advancement for Arabic language models. Code is available at: this https URL. 

---
# First Finish Search: Efficient Test-Time Scaling in Large Language Models 

**Authors**: Aradhye Agarwal, Ayan Sengupta, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2505.18149)  

**Abstract**: Test-time scaling (TTS), which involves dynamic allocation of compute during inference, offers a promising way to improve reasoning in large language models. While existing TTS methods work well, they often rely on long decoding paths or require a large number of samples to be generated, increasing the token usage and inference latency. We observe the surprising fact that for reasoning tasks, shorter traces are much more likely to be correct than longer ones. Motivated by this, we introduce First Finish Search (FFS), a training-free parallel decoding strategy that launches $n$ independent samples and returns as soon as any one completes. We evaluate FFS alongside simple decoding, beam search, majority voting, and budget forcing on four reasoning models (DeepSeek-R1, R1-Distill-Qwen-32B, QwQ-32B and Phi-4-Reasoning-Plus) and across four datasets (AIME24, AIME25-I, AIME25-II and GPQA Diamond). With DeepSeek-R1, FFS achieves $82.23\%$ accuracy on the AIME datasets, a $15\%$ improvement over DeepSeek-R1's standalone accuracy, nearly matching OpenAI's o4-mini performance. Our theoretical analysis explains why stopping at the shortest trace is likely to yield a correct answer and identifies the conditions under which early stopping may be suboptimal. The elegance and simplicity of FFS demonstrate that straightforward TTS strategies can perform remarkably well, revealing the untapped potential of simple approaches at inference time. 

---
# Lost in the Haystack: Smaller Needles are More Difficult for LLMs to Find 

**Authors**: Owen Bianchi, Mathew J. Koretsky, Maya Willey, Chelsea X. Alvarado, Tanay Nayak, Adi Asija, Nicole Kuznetsov, Mike A. Nalls, Faraz Faghri, Daniel Khashabi  

**Link**: [PDF](https://arxiv.org/pdf/2505.18148)  

**Abstract**: Large language models (LLMs) face significant challenges with needle-in-a-haystack tasks, where relevant information ("the needle") must be drawn from a large pool of irrelevant context ("the haystack"). Previous studies have highlighted positional bias and distractor quantity as critical factors affecting model performance, yet the influence of gold context size has received little attention. We address this gap by systematically studying how variations in gold context length impact LLM performance on long-context question answering tasks. Our experiments reveal that LLM performance drops sharply when the gold context is shorter, i.e., smaller gold contexts consistently degrade model performance and amplify positional sensitivity, posing a major challenge for agentic systems that must integrate scattered, fine-grained information of varying lengths. This pattern holds across three diverse domains (general knowledge, biomedical reasoning, and mathematical reasoning) and seven state-of-the-art LLMs of various sizes and architectures. Our work provides clear insights to guide the design of robust, context-aware LLM-driven systems. 

---
# Graph-Linguistic Fusion: Using Language Models for Wikidata Vandalism Detection 

**Authors**: Mykola Trokhymovych, Lydia Pintscher, Ricardo Baeza-Yates, Diego Saez-Trumper  

**Link**: [PDF](https://arxiv.org/pdf/2505.18136)  

**Abstract**: We introduce a next-generation vandalism detection system for Wikidata, one of the largest open-source structured knowledge bases on the Web. Wikidata is highly complex: its items incorporate an ever-expanding universe of factual triples and multilingual texts. While edits can alter both structured and textual content, our approach converts all edits into a single space using a method we call Graph2Text. This allows for evaluating all content changes for potential vandalism using a single multilingual language model. This unified approach improves coverage and simplifies maintenance. Experiments demonstrate that our solution outperforms the current production system. Additionally, we are releasing the code under an open license along with a large dataset of various human-generated knowledge alterations, enabling further research. 

---
# Frankentext: Stitching random text fragments into long-form narratives 

**Authors**: Chau Minh Pham, Jenna Russell, Dzung Pham, Mohit Iyyer  

**Link**: [PDF](https://arxiv.org/pdf/2505.18128)  

**Abstract**: We introduce Frankentexts, a new type of long-form narratives produced by LLMs under the extreme constraint that most tokens (e.g., 90%) must be copied verbatim from human writings. This task presents a challenging test of controllable generation, requiring models to satisfy a writing prompt, integrate disparate text fragments, and still produce a coherent narrative. To generate Frankentexts, we instruct the model to produce a draft by selecting and combining human-written passages, then iteratively revise the draft while maintaining a user-specified copy ratio. We evaluate the resulting Frankentexts along three axes: writing quality, instruction adherence, and detectability. Gemini-2.5-Pro performs surprisingly well on this task: 81% of its Frankentexts are coherent and 100% relevant to the prompt. Notably, up to 59% of these outputs are misclassified as human-written by detectors like Pangram, revealing limitations in AI text detectors. Human annotators can sometimes identify Frankentexts through their abrupt tone shifts and inconsistent grammar between segments, especially in longer generations. Beyond presenting a challenging generation task, Frankentexts invite discussion on building effective detectors for this new grey zone of authorship, provide training data for mixed authorship detection, and serve as a sandbox for studying human-AI co-writing processes. 

---
# UNJOIN: Enhancing Multi-Table Text-to-SQL Generation via Schema Simplification 

**Authors**: Poojah Ganesan, Rajat Aayush Jha, Dan Roth, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.18122)  

**Abstract**: Recent advances in large language models (LLMs) have greatly improved Text-to-SQL performance for single-table queries. But, it remains challenging in multi-table databases due to complex schema and relational operations. Existing methods often struggle with retrieving the right tables and columns, generating accurate JOINs and UNIONs, and generalizing across diverse schemas. To address these issues, we introduce UNJOIN, a two-stage framework that decouples the retrieval of schema elements from SQL logic generation. In the first stage, we merge the column names of all tables in the database into a single-table representation by prefixing each column with its table name. This allows the model to focus purely on accurate retrieval without being distracted by the need to write complex SQL logic. In the second stage, the SQL query is generated on this simplified schema and mapped back to the original schema by reconstructing JOINs, UNIONs, and relational logic. Evaluations on SPIDER and BIRD datasets show that UNJOIN matches or exceeds the state-of-the-art baselines. UNJOIN uses only schema information, which does not require data access or fine-tuning, making it scalable and adaptable across databases. 

---
# Watch and Listen: Understanding Audio-Visual-Speech Moments with Multimodal LLM 

**Authors**: Zinuo Li, Xian Zhang, Yongxin Guo, Mohammed Bennamoun, Farid Boussaid, Girish Dwivedi, Luqi Gong, Qiuhong Ke  

**Link**: [PDF](https://arxiv.org/pdf/2505.18110)  

**Abstract**: Humans naturally understand moments in a video by integrating visual and auditory cues. For example, localizing a scene in the video like "A scientist passionately speaks on wildlife conservation as dramatic orchestral music plays, with the audience nodding and applauding" requires simultaneous processing of visual, audio, and speech signals. However, existing models often struggle to effectively fuse and interpret audio information, limiting their capacity for comprehensive video temporal understanding. To address this, we present TriSense, a triple-modality large language model designed for holistic video temporal understanding through the integration of visual, audio, and speech modalities. Central to TriSense is a Query-Based Connector that adaptively reweights modality contributions based on the input query, enabling robust performance under modality dropout and allowing flexible combinations of available inputs. To support TriSense's multimodal capabilities, we introduce TriSense-2M, a high-quality dataset of over 2 million curated samples generated via an automated pipeline powered by fine-tuned LLMs. TriSense-2M includes long-form videos and diverse modality combinations, facilitating broad generalization. Extensive experiments across multiple benchmarks demonstrate the effectiveness of TriSense and its potential to advance multimodal video analysis. Code and dataset will be publicly released. 

---
# ManuSearch: Democratizing Deep Search in Large Language Models with a Transparent and Open Multi-Agent Framework 

**Authors**: Lisheng Huang, Yichen Liu, Jinhao Jiang, Rongxiang Zhang, Jiahao Yan, Junyi Li, Wayne Xin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.18105)  

**Abstract**: Recent advances in web-augmented large language models (LLMs) have exhibited strong performance in complex reasoning tasks, yet these capabilities are mostly locked in proprietary systems with opaque architectures. In this work, we propose \textbf{ManuSearch}, a transparent and modular multi-agent framework designed to democratize deep search for LLMs. ManuSearch decomposes the search and reasoning process into three collaborative agents: (1) a solution planning agent that iteratively formulates sub-queries, (2) an Internet search agent that retrieves relevant documents via real-time web search, and (3) a structured webpage reading agent that extracts key evidence from raw web content. To rigorously evaluate deep reasoning abilities, we introduce \textbf{ORION}, a challenging benchmark focused on open-web reasoning over long-tail entities, covering both English and Chinese. Experimental results show that ManuSearch substantially outperforms prior open-source baselines and even surpasses leading closed-source systems. Our work paves the way for reproducible, extensible research in open deep search systems. We release the data and code in this https URL 

---
# Planning without Search: Refining Frontier LLMs with Offline Goal-Conditioned RL 

**Authors**: Joey Hong, Anca Dragan, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2505.18098)  

**Abstract**: Large language models (LLMs) excel in tasks like question answering and dialogue, but complex tasks requiring interaction, such as negotiation and persuasion, require additional long-horizon reasoning and planning. Reinforcement learning (RL) fine-tuning can enable such planning in principle, but suffers from drawbacks that hinder scalability. In particular, multi-turn RL training incurs high memory and computational costs, which are exacerbated when training LLMs as policies. Furthermore, the largest LLMs do not expose the APIs necessary to be trained in such manner. As a result, modern methods to improve the reasoning of LLMs rely on sophisticated prompting mechanisms rather than RL fine-tuning. To remedy this, we propose a novel approach that uses goal-conditioned value functions to guide the reasoning of LLM agents, that scales even to large API-based models. These value functions predict how a task will unfold given an action, allowing the LLM agent to evaluate multiple possible outcomes, both positive and negative, to plan effectively. In addition, these value functions are trained over reasoning steps rather than full actions, to be a concise and light-weight module that facilitates decision-making in multi-turn interactions. We validate our method on tasks requiring interaction, including tool use, social deduction, and dialogue, demonstrating superior performance over both RL fine-tuning and prompting methods while maintaining efficiency and scalability. 

---
# QwenLong-CPRS: Towards $\infty$-LLMs with Dynamic Context Optimization 

**Authors**: Weizhou Shen, Chenliang Li, Fanqi Wan, Shengyi Liao, Shaopeng Lai, Bo Zhang, Yingcheng Shi, Yuning Wu, Gang Fu, Zhansheng Li, Bin Yang, Ji Zhang, Fei Huang, Jingren Zhou, Ming Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.18092)  

**Abstract**: This technical report presents QwenLong-CPRS, a context compression framework designed for explicit long-context optimization, addressing prohibitive computation overhead during the prefill stage and the "lost in the middle" performance degradation of large language models (LLMs) during long sequence processing. Implemented through a novel dynamic context optimization mechanism, QwenLong-CPRS enables multi-granularity context compression guided by natural language instructions, achieving both efficiency gains and improved performance.
Evolved from the Qwen architecture series, QwenLong-CPRS introduces four key innovations: (1) Natural language-guided dynamic optimization, (2) Bidirectional reasoning layers for enhanced boundary awareness, (3) Token critic mechanisms with language modeling heads, and (4) Window-parallel inference.
Comprehensive evaluations across five benchmarks (4K-2M word contexts) demonstrate QwenLong-CPRS's threefold effectiveness: (1) Consistent superiority over other context management methods like RAG and sparse attention in both accuracy and efficiency. (2) Architecture-agnostic integration with all flagship LLMs, including GPT-4o, Gemini2.0-pro, Claude3.7-sonnet, DeepSeek-v3, and Qwen2.5-max, achieves 21.59$\times$ context compression alongside 19.15-point average performance gains; (3) Deployed with Qwen2.5-32B-Instruct, QwenLong-CPRS surpasses leading proprietary LLMs by 4.85 and 10.88 points on Ruler-128K and InfiniteBench, establishing new SOTA performance. 

---
# Extended Inductive Reasoning for Personalized Preference Inference from Behavioral Signals 

**Authors**: Jia-Nan Li, Jian Guan, Wei Wu, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.18071)  

**Abstract**: Large language models (LLMs) have demonstrated significant success in complex reasoning tasks such as math and coding. In contrast to these tasks where deductive reasoning predominates, inductive reasoning\textemdash the ability to derive general rules from incomplete evidence, remains underexplored. This paper investigates extended inductive reasoning in LLMs through the lens of personalized preference inference, a critical challenge in LLM alignment where current approaches struggle to capture diverse user preferences. The task demands strong inductive reasoning capabilities as user preferences are typically embedded implicitly across various interaction forms, requiring models to synthesize consistent preference patterns from scattered signals. We propose \textsc{AlignXplore}, a model that leverages extended reasoning chains to enable systematic preference inference from behavioral signals in users' interaction histories. We develop \textsc{AlignXplore} by combining cold-start training based on synthetic data with subsequent online reinforcement learning. Through extensive experiments, we demonstrate that \textsc{AlignXplore} achieves substantial improvements over the backbone model by an average of 11.05\% on in-domain and out-of-domain benchmarks, while maintaining strong generalization ability across different input formats and downstream models. Further analyses establish best practices for preference inference learning through systematic comparison of reward modeling strategies, while revealing the emergence of human-like inductive reasoning patterns during training. 

---
# MathEDU: Towards Adaptive Feedback for Student Mathematical Problem-Solving 

**Authors**: Wei-Ling Hsu, Yu-Chien Tang, An-Zi Yen  

**Link**: [PDF](https://arxiv.org/pdf/2505.18056)  

**Abstract**: Online learning enhances educational accessibility, offering students the flexibility to learn anytime, anywhere. However, a key limitation is the lack of immediate, personalized feedback, particularly in helping students correct errors in math problem-solving. Several studies have investigated the applications of large language models (LLMs) in educational contexts. In this paper, we explore the capabilities of LLMs to assess students' math problem-solving processes and provide adaptive feedback. The MathEDU dataset is introduced, comprising authentic student solutions annotated with teacher feedback. We evaluate the model's ability to support personalized learning in two scenarios: one where the model has access to students' prior answer histories, and another simulating a cold-start context. Experimental results show that the fine-tuned model performs well in identifying correctness. However, the model still faces challenges in generating detailed feedback for pedagogical purposes. 

---
# Contrastive Distillation of Emotion Knowledge from LLMs for Zero-Shot Emotion Recognition 

**Authors**: Minxue Niu, Emily Mower Provost  

**Link**: [PDF](https://arxiv.org/pdf/2505.18040)  

**Abstract**: The ability to handle various emotion labels without dedicated training is crucial for building adaptable Emotion Recognition (ER) systems. Conventional ER models rely on training using fixed label sets and struggle to generalize beyond them. On the other hand, Large Language Models (LLMs) have shown strong zero-shot ER performance across diverse label spaces, but their scale limits their use on edge devices. In this work, we propose a contrastive distillation framework that transfers rich emotional knowledge from LLMs into a compact model without the use of human annotations. We use GPT-4 to generate descriptive emotion annotations, offering rich supervision beyond fixed label sets. By aligning text samples with emotion descriptors in a shared embedding space, our method enables zero-shot prediction on different emotion classes, granularity, and label schema. The distilled model is effective across multiple datasets and label spaces, outperforming strong baselines of similar size and approaching GPT-4's zero-shot performance, while being over 10,000 times smaller. 

---
# Training with Pseudo-Code for Instruction Following 

**Authors**: Prince Kumar, Rudra Murthy, Riyaz Bhat, Danish Contractor  

**Link**: [PDF](https://arxiv.org/pdf/2505.18011)  

**Abstract**: Despite the rapid progress in the capabilities of Large Language Models (LLMs), they continue to have difficulty following relatively simple, unambiguous instructions, especially when compositions are involved. In this paper, we take inspiration from recent work that suggests that models may follow instructions better when they are expressed in pseudo-code. However, writing pseudo-code programs can be tedious and using few-shot demonstrations to craft code representations for use in inference can be unnatural for non-expert users of LLMs. To overcome these limitations, we propose fine-tuning LLMs with instruction-tuning data that additionally includes instructions re-expressed in pseudo-code along with the final response. We evaluate models trained using our method on $11$ publicly available benchmarks comprising of tasks related to instruction-following, mathematics, and common-sense reasoning. We conduct rigorous experiments with $5$ different models and find that not only do models follow instructions better when trained with pseudo-code, they also retain their capabilities on the other tasks related to mathematical and common sense reasoning. Specifically, we observe a relative gain of $3$--$19$% on instruction-following benchmark, and an average gain of upto 14% across all tasks. 

---
# TRACE for Tracking the Emergence of Semantic Representations in Transformers 

**Authors**: Nura Aljaafari, Danilo S. Carvalho, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2505.17998)  

**Abstract**: Modern transformer models exhibit phase transitions during training, distinct shifts from memorisation to abstraction, but the mechanisms underlying these transitions remain poorly understood. Prior work has often focused on endpoint representations or isolated signals like curvature or mutual information, typically in symbolic or arithmetic domains, overlooking the emergence of linguistic structure. We introduce TRACE (Tracking Representation Abstraction and Compositional Emergence), a diagnostic framework combining geometric, informational, and linguistic signals to detect phase transitions in Transformer-based LMs. TRACE leverages a frame-semantic data generation method, ABSynth, that produces annotated synthetic corpora with controllable complexity, lexical distributions, and structural entropy, while being fully annotated with linguistic categories, enabling precise analysis of abstraction emergence. Experiments reveal that (i) phase transitions align with clear intersections between curvature collapse and dimension stabilisation; (ii) these geometric shifts coincide with emerging syntactic and semantic accuracy; (iii) abstraction patterns persist across architectural variants, with components like feedforward networks affecting optimisation stability rather than fundamentally altering trajectories. This work advances our understanding of how linguistic abstractions emerge in LMs, offering insights into model interpretability, training efficiency, and compositional generalisation that could inform more principled approaches to LM development. 

---
# AVerImaTeC: A Dataset for Automatic Verification of Image-Text Claims with Evidence from the Web 

**Authors**: Rui Cao, Zifeng Ding, Zhijiang Guo, Michael Schlichtkrull, Andreas Vlachos  

**Link**: [PDF](https://arxiv.org/pdf/2505.17978)  

**Abstract**: Textual claims are often accompanied by images to enhance their credibility and spread on social media, but this also raises concerns about the spread of misinformation. Existing datasets for automated verification of image-text claims remain limited, as they often consist of synthetic claims and lack evidence annotations to capture the reasoning behind the verdict. In this work, we introduce AVerImaTeC, a dataset consisting of 1,297 real-world image-text claims. Each claim is annotated with question-answer (QA) pairs containing evidence from the web, reflecting a decomposed reasoning regarding the verdict. We mitigate common challenges in fact-checking datasets such as contextual dependence, temporal leakage, and evidence insufficiency, via claim normalization, temporally constrained evidence annotation, and a two-stage sufficiency check. We assess the consistency of the annotation in AVerImaTeC via inter-annotator studies, achieving a $\kappa=0.742$ on verdicts and $74.7\%$ consistency on QA pairs. We also propose a novel evaluation method for evidence retrieval and conduct extensive experiments to establish baselines for verifying image-text claims using open-web evidence. 

---
# Counting Cycles with Deepseek 

**Authors**: Jiashun Jin, Tracy Ke, Bingcheng Sui, Zhenggang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17964)  

**Abstract**: Despite recent progress, AI still struggles on advanced mathematics. We consider a difficult open problem: How to derive a Computationally Efficient Equivalent Form (CEEF) for the cycle count statistic? The CEEF problem does not have known general solutions, and requires delicate combinatorics and tedious calculations. Such a task is hard to accomplish by humans but is an ideal example where AI can be very helpful. We solve the problem by combining a novel approach we propose and the powerful coding skills of AI. Our results use delicate graph theory and contain new formulas for general cases that have not been discovered before. We find that, while AI is unable to solve the problem all by itself, it is able to solve it if we provide it with a clear strategy, a step-by-step guidance and carefully written prompts. For simplicity, we focus our study on DeepSeek-R1 but we also investigate other AI approaches. 

---
# Beyond Distillation: Pushing the Limits of Medical LLM Reasoning with Minimalist Rule-Based RL 

**Authors**: Che Liu, Haozhe Wang, Jiazhen Pan, Zhongwei Wan, Yong Dai, Fangzhen Lin, Wenjia Bai, Daniel Rueckert, Rossella Arcucci  

**Link**: [PDF](https://arxiv.org/pdf/2505.17952)  

**Abstract**: Improving performance on complex tasks and enabling interpretable decision making in large language models (LLMs), especially for clinical applications, requires effective reasoning. Yet this remains challenging without supervised fine-tuning (SFT) on costly chain-of-thought (CoT) data distilled from closed-source models (e.g., GPT-4o). In this work, we present AlphaMed, the first medical LLM to show that reasoning capability can emerge purely through reinforcement learning (RL), using minimalist rule-based rewards on public multiple-choice QA datasets, without relying on SFT or distilled CoT data. AlphaMed achieves state-of-the-art results on six medical QA benchmarks, outperforming models trained with conventional SFT+RL pipelines. On challenging benchmarks (e.g., MedXpert), AlphaMed even surpasses larger or closed-source models such as DeepSeek-V3-671B and Claude-3.5-Sonnet. To understand the factors behind this success, we conduct a comprehensive data-centric analysis guided by three questions: (i) Can minimalist rule-based RL incentivize reasoning without distilled CoT supervision? (ii) How do dataset quantity and diversity impact reasoning? (iii) How does question difficulty shape the emergence and generalization of reasoning? Our findings show that dataset informativeness is a key driver of reasoning performance, and that minimalist RL on informative, multiple-choice QA data is effective at inducing reasoning without CoT supervision. We also observe divergent trends across benchmarks, underscoring limitations in current evaluation and the need for more challenging, reasoning-oriented medical QA benchmarks. 

---
# Handling Symbolic Language in Student Texts: A Comparative Study of NLP Embedding Models 

**Authors**: Tom Bleckmann, Paul Tschisgale  

**Link**: [PDF](https://arxiv.org/pdf/2505.17950)  

**Abstract**: Recent advancements in Natural Language Processing (NLP) have facilitated the analysis of student-generated language products in learning analytics (LA), particularly through the use of NLP embedding models. Yet when it comes to science-related language, symbolic expressions such as equations and formulas introduce challenges that current embedding models struggle to address. Existing studies and applications often either overlook these challenges or remove symbolic expressions altogether, potentially leading to biased findings and diminished performance of LA applications. This study therefore explores how contemporary embedding models differ in their capability to process and interpret science-related symbolic expressions. To this end, various embedding models are evaluated using physics-specific symbolic expressions drawn from authentic student responses, with performance assessed via two approaches: similarity-based analyses and integration into a machine learning pipeline. Our findings reveal significant differences in model performance, with OpenAI's GPT-text-embedding-3-large outperforming all other examined models, though its advantage over other models was moderate rather than decisive. Beyond performance, additional factors such as cost, regulatory compliance, and model transparency are discussed as key considerations for model selection. Overall, this study underscores the importance for LA researchers and practitioners of carefully selecting NLP embedding models when working with science-related language products that include symbolic expressions. 

---
# Language models can learn implicit multi-hop reasoning, but only if they have lots of training data 

**Authors**: Yuekun Yao, Yupei Du, Dawei Zhu, Michael Hahn, Alexander Koller  

**Link**: [PDF](https://arxiv.org/pdf/2505.17923)  

**Abstract**: Implicit reasoning is the ability of a language model to solve multi-hop reasoning tasks in a single forward pass, without chain of thought. We investigate this capability using GPT2-style language models trained from scratch on controlled $k$-hop reasoning datasets ($k = 2, 3, 4$). We show that while such models can indeed learn implicit $k$-hop reasoning, the required training data grows exponentially in $k$, and the required number of transformer layers grows linearly in $k$. We offer a theoretical explanation for why this depth growth is necessary. We further find that the data requirement can be mitigated, but not eliminated, through curriculum learning. 

---
# Mutarjim: Advancing Bidirectional Arabic-English Translation with a Small Language Model 

**Authors**: Khalil Hennara, Muhammad Hreden, Mohamed Motaism Hamed, Zeina Aldallal, Sara Chrouf, Safwan AlModhayan  

**Link**: [PDF](https://arxiv.org/pdf/2505.17894)  

**Abstract**: We introduce Mutarjim, a compact yet powerful language model for bidirectional Arabic-English translation. While large-scale LLMs have shown impressive progress in natural language processing tasks, including machine translation, smaller models. Leveraging this insight, we developed Mutarjim based on Kuwain-1.5B , a language model tailored for both Arabic and English. Despite its modest size, Mutarjim outperforms much larger models on several established benchmarks, achieved through an optimized two-phase training approach and a carefully curated, high-quality training corpus.. Experimental results show that Mutarjim rivals models up to 20 times larger while significantly reducing computational costs and training requirements. We also introduce Tarjama-25, a new benchmark designed to overcome limitations in existing Arabic-English benchmarking datasets, such as domain narrowness, short sentence lengths, and English-source bias. Tarjama-25 comprises 5,000 expert-reviewed sentence pairs and spans a wide range of domains, offering a more comprehensive and balanced evaluation framework. Notably, Mutarjim achieves state-of-the-art performance on the English-to-Arabic task in Tarjama-25, surpassing even significantly larger and proprietary models like GPT-4o mini. We publicly release Tarjama-25 to support future research and advance the evaluation of Arabic-English translation systems. 

---
# MOOSE-Chem3: Toward Experiment-Guided Hypothesis Ranking via Simulated Experimental Feedback 

**Authors**: Wanhao Liu, Zonglin Yang, Jue Wang, Lidong Bing, Di Zhang, Dongzhan Zhou, Yuqiang Li, Houqiang Li, Erik Cambria, Wanli Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17873)  

**Abstract**: Hypothesis ranking is a crucial component of automated scientific discovery, particularly in natural sciences where wet-lab experiments are costly and throughput-limited. Existing approaches focus on pre-experiment ranking, relying solely on large language model's internal reasoning without incorporating empirical outcomes from experiments. We introduce the task of experiment-guided ranking, which aims to prioritize candidate hypotheses based on the results of previously tested ones. However, developing such strategies is challenging due to the impracticality of repeatedly conducting real experiments in natural science domains. To address this, we propose a simulator grounded in three domain-informed assumptions, modeling hypothesis performance as a function of similarity to a known ground truth hypothesis, perturbed by noise. We curate a dataset of 124 chemistry hypotheses with experimentally reported outcomes to validate the simulator. Building on this simulator, we develop a pseudo experiment-guided ranking method that clusters hypotheses by shared functional characteristics and prioritizes candidates based on insights derived from simulated experimental feedback. Experiments show that our method outperforms pre-experiment baselines and strong ablations. 

---
# Just as Humans Need Vaccines, So Do Models: Model Immunization to Combat Falsehoods 

**Authors**: Shaina Raza, Rizwan Qureshi, Marcelo Lotif, Aman Chadha, Deval Pandya, Christos Emmanouilidis  

**Link**: [PDF](https://arxiv.org/pdf/2505.17870)  

**Abstract**: Generative AI models often learn and reproduce false information present in their training corpora. This position paper argues that, analogous to biological immunization, where controlled exposure to a weakened pathogen builds immunity, AI models should be fine tuned on small, quarantined sets of explicitly labeled falsehoods as a "vaccine" against misinformation. These curated false examples are periodically injected during finetuning, strengthening the model ability to recognize and reject misleading claims while preserving accuracy on truthful inputs. An illustrative case study shows that immunized models generate substantially less misinformation than baselines. To our knowledge, this is the first training framework that treats fact checked falsehoods themselves as a supervised vaccine, rather than relying on input perturbations or generic human feedback signals, to harden models against future misinformation. We also outline ethical safeguards and governance controls to ensure the safe use of false data. Model immunization offers a proactive paradigm for aligning AI systems with factuality. 

---
# Explaining Sources of Uncertainty in Automated Fact-Checking 

**Authors**: Jingyi Sun, Greta Warren, Irina Shklovski, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2505.17855)  

**Abstract**: Understanding sources of a model's uncertainty regarding its predictions is crucial for effective human-AI collaboration. Prior work proposes using numerical uncertainty or hedges ("I'm not sure, but ..."), which do not explain uncertainty that arises from conflicting evidence, leaving users unable to resolve disagreements or rely on the output. We introduce CLUE (Conflict-and-Agreement-aware Language-model Uncertainty Explanations), the first framework to generate natural language explanations of model uncertainty by (i) identifying relationships between spans of text that expose claim-evidence or inter-evidence conflicts and agreements that drive the model's predictive uncertainty in an unsupervised way, and (ii) generating explanations via prompting and attention steering that verbalize these critical interactions. Across three language models and two fact-checking datasets, we show that CLUE produces explanations that are more faithful to the model's uncertainty and more consistent with fact-checking decisions than prompting for uncertainty explanations without span-interaction guidance. Human evaluators judge our explanations to be more helpful, more informative, less redundant, and more logically consistent with the input than this baseline. CLUE requires no fine-tuning or architectural changes, making it plug-and-play for any white-box language model. By explicitly linking uncertainty to evidence conflicts, it offers practical support for fact-checking and generalises readily to other tasks that require reasoning over complex information. 

---
# Investigating Affect Mining Techniques for Annotation Sample Selection in the Creation of Finnish Affective Speech Corpus 

**Authors**: Kalle Lahtinen, Einari Vaaras, Liisa Mustanoja, Okko Räsänen  

**Link**: [PDF](https://arxiv.org/pdf/2505.17833)  

**Abstract**: Study of affect in speech requires suitable data, as emotional expression and perception vary across languages. Until now, no corpus has existed for natural expression of affect in spontaneous Finnish, existing data being acted or from a very specific communicative setting. This paper presents the first such corpus, created by annotating 12,000 utterances for emotional arousal and valence, sampled from three large-scale Finnish speech corpora. To ensure diverse affective expression, sample selection was conducted with an affect mining approach combining acoustic, cross-linguistic speech emotion, and text sentiment features. We compare this method to random sampling in terms of annotation diversity, and conduct post-hoc analyses to identify sampling choices that would have maximized the diversity. As an outcome, the work introduces a spontaneous Finnish affective speech corpus and informs sampling strategies for affective speech corpus creation in other languages or domains. 

---
# Emerging categories in scientific explanations 

**Authors**: Giacomo Magnifico, Eduard Barbu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17832)  

**Abstract**: Clear and effective explanations are essential for human understanding and knowledge dissemination. The scope of scientific research aiming to understand the essence of explanations has recently expanded from the social sciences to machine learning and artificial intelligence. Explanations for machine learning decisions must be impactful and human-like, and there is a lack of large-scale datasets focusing on human-like and human-generated explanations. This work aims to provide such a dataset by: extracting sentences that indicate explanations from scientific literature among various sources in the biotechnology and biophysics topic domains (e.g. PubMed's PMC Open Access subset); providing a multi-class notation derived inductively from the data; evaluating annotator consensus on the emerging categories. The sentences are organized in an openly-available dataset, with two different classifications (6-class and 3-class category annotation), and the 3-class notation achieves a 0.667 Krippendorf Alpha value. 

---
# Stepwise Reasoning Checkpoint Analysis: A Test Time Scaling Method to Enhance LLMs' Reasoning 

**Authors**: Zezhong Wang, Xingshan Zeng, Weiwen Liu, Yufei Wang, Liangyou Li, Yasheng Wang, Lifeng Shang, Xin Jiang, Qun Liu, Kam-Fai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2505.17829)  

**Abstract**: Mathematical reasoning through Chain-of-Thought (CoT) has emerged as a powerful capability of Large Language Models (LLMs), which can be further enhanced through Test-Time Scaling (TTS) methods like Beam Search and DVTS. However, these methods, despite improving accuracy by allocating more computational resources during inference, often suffer from path homogenization and inefficient use of intermediate results. To address these limitations, we propose Stepwise Reasoning Checkpoint Analysis (SRCA), a framework that introduces checkpoints between reasoning steps. It incorporates two key strategies: (1) Answer-Clustered Search, which groups reasoning paths by their intermediate checkpoint answers to maintain diversity while ensuring quality, and (2) Checkpoint Candidate Augmentation, which leverages all intermediate answers for final decision-making. Our approach effectively reduces path homogenization and creates a fault-tolerant mechanism by utilizing high-quality intermediate results. Experimental results show that SRCA improves reasoning accuracy compared to existing TTS methods across various mathematical datasets. 

---
# Not All Tokens Are What You Need In Thinking 

**Authors**: Hang Yuan, Bin Yu, Haotian Li, Shijun Yang, Christina Dan Wang, Zhou Yu, Xueyin Xu, Weizhen Qi, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.17827)  

**Abstract**: Modern reasoning models, such as OpenAI's o1 and DeepSeek-R1, exhibit impressive problem-solving capabilities but suffer from critical inefficiencies: high inference latency, excessive computational resource consumption, and a tendency toward overthinking -- generating verbose chains of thought (CoT) laden with redundant tokens that contribute minimally to the final answer. To address these issues, we propose Conditional Token Selection (CTS), a token-level compression framework with a flexible and variable compression ratio that identifies and preserves only the most essential tokens in CoT. CTS evaluates each token's contribution to deriving correct answers using conditional importance scoring, then trains models on compressed CoT. Extensive experiments demonstrate that CTS effectively compresses long CoT while maintaining strong reasoning performance. Notably, on the GPQA benchmark, Qwen2.5-14B-Instruct trained with CTS achieves a 9.1% accuracy improvement with 13.2% fewer reasoning tokens (13% training token reduction). Further reducing training tokens by 42% incurs only a marginal 5% accuracy drop while yielding a 75.8% reduction in reasoning tokens, highlighting the prevalence of redundancy in existing CoT. 

---
# Low-Resource NMT: A Case Study on the Written and Spoken Languages in Hong Kong 

**Authors**: Hei Yi Mak, Tan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.17816)  

**Abstract**: The majority of inhabitants in Hong Kong are able to read and write in standard Chinese but use Cantonese as the primary spoken language in daily life. Spoken Cantonese can be transcribed into Chinese characters, which constitute the so-called written Cantonese. Written Cantonese exhibits significant lexical and grammatical differences from standard written Chinese. The rise of written Cantonese is increasingly evident in the cyber world. The growing interaction between Mandarin speakers and Cantonese speakers is leading to a clear demand for automatic translation between Chinese and Cantonese. This paper describes a transformer-based neural machine translation (NMT) system for written-Chinese-to-written-Cantonese translation. Given that parallel text data of Chinese and Cantonese are extremely scarce, a major focus of this study is on the effort of preparing good amount of training data for NMT. In addition to collecting 28K parallel sentences from previous linguistic studies and scattered internet resources, we devise an effective approach to obtaining 72K parallel sentences by automatically extracting pairs of semantically similar sentences from parallel articles on Chinese Wikipedia and Cantonese Wikipedia. We show that leveraging highly similar sentence pairs mined from Wikipedia improves translation performance in all test sets. Our system outperforms Baidu Fanyi's Chinese-to-Cantonese translation on 6 out of 8 test sets in BLEU scores. Translation examples reveal that our system is able to capture important linguistic transformations between standard Chinese and spoken Cantonese. 

---
# Don't Overthink it. Preferring Shorter Thinking Chains for Improved LLM Reasoning 

**Authors**: Michael Hassid, Gabriel Synnaeve, Yossi Adi, Roy Schwartz  

**Link**: [PDF](https://arxiv.org/pdf/2505.17813)  

**Abstract**: Reasoning large language models (LLMs) heavily rely on scaling test-time compute to perform complex reasoning tasks by generating extensive "thinking" chains. While demonstrating impressive results, this approach incurs significant computational costs and inference time. In this work, we challenge the assumption that long thinking chains results in better reasoning capabilities. We first demonstrate that shorter reasoning chains within individual questions are significantly more likely to yield correct answers - up to 34.5% more accurate than the longest chain sampled for the same question. Based on these results, we suggest short-m@k, a novel reasoning LLM inference method. Our method executes k independent generations in parallel and halts computation once the first m thinking processes are done. The final answer is chosen using majority voting among these m chains. Basic short-1@k demonstrates similar or even superior performance over standard majority voting in low-compute settings - using up to 40% fewer thinking tokens. short-3@k, while slightly less efficient than short-1@k, consistently surpasses majority voting across all compute budgets, while still being substantially faster (up to 33% wall time reduction). Inspired by our results, we finetune an LLM using short, long, and randomly selected reasoning chains. We then observe that training on the shorter ones leads to better performance. Our findings suggest rethinking current methods of test-time compute in reasoning LLMs, emphasizing that longer "thinking" does not necessarily translate to improved performance and can, counter-intuitively, lead to degraded results. 

---
# DialogXpert: Driving Intelligent and Emotion-Aware Conversations through Online Value-Based Reinforcement Learning with LLM Priors 

**Authors**: Tazeek Bin Abdur Rakib, Ambuj Mehrish, Lay-Ki Soon, Wern Han Lim, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2505.17795)  

**Abstract**: Large-language-model (LLM) agents excel at reactive dialogue but struggle with proactive, goal-driven interactions due to myopic decoding and costly planning. We introduce DialogXpert, which leverages a frozen LLM to propose a small, high-quality set of candidate actions per turn and employs a compact Q-network over fixed BERT embeddings trained via temporal-difference learning to select optimal moves within this reduced space. By tracking the user's emotions, DialogXpert tailors each decision to advance the task while nurturing a genuine, empathetic connection. Across negotiation, emotional support, and tutoring benchmarks, DialogXpert drives conversations to under $3$ turns with success rates exceeding 94\% and, with a larger LLM prior, pushes success above 97\% while markedly improving negotiation outcomes. This framework delivers real-time, strategic, and emotionally intelligent dialogue planning at scale. Code available at this https URL 

---
# Compression Hacking: A Supplementary Perspective on Informatics Metric of Language Models from Geometric Distortion 

**Authors**: Jianxiang Zang, Meiling Ning, Yongda Wei, Shihan Dou, Jiazheng Zhang, Nijia Mo, Binhong Li, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17793)  

**Abstract**: Recently, the concept of ``compression as intelligence'' has provided a novel informatics metric perspective for language models (LMs), emphasizing that highly structured representations signify the intelligence level of LMs. However, from a geometric standpoint, the word representation space of highly compressed LMs tends to degenerate into a highly anisotropic state, which hinders the LM's ability to comprehend instructions and directly impacts its performance. We found this compression-anisotropy synchronicity is essentially the ``Compression Hacking'' in LM representations, where noise-dominated directions tend to create the illusion of high compression rates by sacrificing spatial uniformity. Based on this, we propose three refined compression metrics by incorporating geometric distortion analysis and integrate them into a self-evaluation pipeline. The refined metrics exhibit strong alignment with the LM's comprehensive capabilities, achieving Spearman correlation coefficients above 0.9, significantly outperforming both the original compression and other internal structure-based metrics. This confirms that compression hacking substantially enhances the informatics interpretation of LMs by incorporating geometric distortion of representations. 

---
# EXECUTE: A Multilingual Benchmark for LLM Token Understanding 

**Authors**: Lukas Edman, Helmut Schmid, Alexander Fraser  

**Link**: [PDF](https://arxiv.org/pdf/2505.17784)  

**Abstract**: The CUTE benchmark showed that LLMs struggle with character understanding in English. We extend it to more languages with diverse scripts and writing systems, introducing EXECUTE. Our simplified framework allows easy expansion to any language. Tests across multiple LLMs reveal that challenges in other languages are not always on the character level as in English. Some languages show word-level processing issues, some show no issues at all. We also examine sub-character tasks in Chinese, Japanese, and Korean to assess LLMs' understanding of character components. 

---
# The Real Barrier to LLM Agent Usability is Agentic ROI 

**Authors**: Weiwen Liu, Jiarui Qin, Xu Huang, Xingshan Zeng, Yunjia Xi, Jianghao Lin, Chuhan Wu, Yasheng Wang, Lifeng Shang, Ruiming Tang, Defu Lian, Yong Yu, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17767)  

**Abstract**: Large Language Model (LLM) agents represent a promising shift in human-AI interaction, moving beyond passive prompt-response systems to autonomous agents capable of reasoning, planning, and goal-directed action. Despite the widespread application in specialized, high-effort tasks like coding and scientific research, we highlight a critical usability gap in high-demand, mass-market applications. This position paper argues that the limited real-world adoption of LLM agents stems not only from gaps in model capabilities, but also from a fundamental tradeoff between the value an agent can provide and the costs incurred during real-world use. Hence, we call for a shift from solely optimizing model performance to a broader, utility-driven perspective: evaluating agents through the lens of the overall agentic return on investment (Agent ROI). By identifying key factors that determine Agentic ROI--information quality, agent time, and cost--we posit a zigzag development trajectory in optimizing agentic ROI: first scaling up to improve the information quality, then scaling down to minimize the time and cost. We outline the roadmap across different development stages to bridge the current usability gaps, aiming to make LLM agents truly scalable, accessible, and effective in real-world contexts. 

---
# Resolving Conflicting Evidence in Automated Fact-Checking: A Study on Retrieval-Augmented LLMs 

**Authors**: Ziyu Ge, Yuhao Wu, Daniel Wai Kit Chin, Roy Ka-Wei Lee, Rui Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.17762)  

**Abstract**: Large Language Models (LLMs) augmented with retrieval mechanisms have demonstrated significant potential in fact-checking tasks by integrating external knowledge. However, their reliability decreases when confronted with conflicting evidence from sources of varying credibility. This paper presents the first systematic evaluation of Retrieval-Augmented Generation (RAG) models for fact-checking in the presence of conflicting evidence. To support this study, we introduce \textbf{CONFACT} (\textbf{Con}flicting Evidence for \textbf{Fact}-Checking) (Dataset available at this https URL), a novel dataset comprising questions paired with conflicting information from various sources. Extensive experiments reveal critical vulnerabilities in state-of-the-art RAG methods, particularly in resolving conflicts stemming from differences in media source credibility. To address these challenges, we investigate strategies to integrate media background information into both the retrieval and generation stages. Our results show that effectively incorporating source credibility significantly enhances the ability of RAG models to resolve conflicting evidence and improve fact-checking performance. 

---
# Discriminating Form and Meaning in Multilingual Models with Minimal-Pair ABX Tasks 

**Authors**: Maureen de Seyssel, Jie Chi, Skyler Seto, Maartje ter Hoeve, Masha Fedzechkina, Natalie Schluter  

**Link**: [PDF](https://arxiv.org/pdf/2505.17747)  

**Abstract**: We introduce a set of training-free ABX-style discrimination tasks to evaluate how multilingual language models represent language identity (form) and semantic content (meaning). Inspired from speech processing, these zero-shot tasks measure whether minimal differences in representation can be reliably detected. This offers a flexible and interpretable alternative to probing. Applied to XLM-R (Conneau et al, 2020) across pretraining checkpoints and layers, we find that language discrimination declines over training and becomes concentrated in lower layers, while meaning discrimination strengthens over time and stabilizes in deeper layers. We then explore probing tasks, showing some alignment between our metrics and linguistic learning performance. Our results position ABX tasks as a lightweight framework for analyzing the structure of multilingual representations. 

---
# Fast Quiet-STaR: Thinking Without Thought Tokens 

**Authors**: Wei Huang, Yizhe Xiong, Xin Ye, Zhijie Deng, Hui Chen, Zijia Lin, Guiguang Ding  

**Link**: [PDF](https://arxiv.org/pdf/2505.17746)  

**Abstract**: Large Language Models (LLMs) have achieved impressive performance across a range of natural language processing tasks. However, recent advances demonstrate that further gains particularly in complex reasoning tasks require more than merely scaling up model sizes or training data. One promising direction is to enable models to think during the reasoning process. Recently, Quiet STaR significantly improves reasoning by generating token-level thought traces, but incurs substantial inference overhead. In this work, we propose Fast Quiet STaR, a more efficient reasoning framework that preserves the benefits of token-level reasoning while reducing computational cost. Our method introduces a curriculum learning based training strategy that gradually reduces the number of thought tokens, enabling the model to internalize more abstract and concise reasoning processes. We further extend this approach to the standard Next Token Prediction (NTP) setting through reinforcement learning-based fine-tuning, resulting in Fast Quiet-STaR NTP, which eliminates the need for explicit thought token generation during inference. Experiments on four benchmark datasets with Mistral 7B and Qwen2.5 7B demonstrate that Fast Quiet-STaR consistently outperforms Quiet-STaR in terms of average accuracy under the same inference time budget. Notably, Fast Quiet-STaR NTP achieves an average accuracy improvement of 9\% on Mistral 7B and 5.7\% on Qwen2.5 7B, while maintaining the same inference latency. Our code will be available at this https URL. 

---
# The Pilot Corpus of the English Semantic Sketches 

**Authors**: Maria Petrova, Maria Ponomareva, Alexandra Ivoylova  

**Link**: [PDF](https://arxiv.org/pdf/2505.17733)  

**Abstract**: The paper is devoted to the creation of the semantic sketches for English verbs. The pilot corpus consists of the English-Russian sketch pairs and is aimed to show what kind of contrastive studies the sketches help to conduct. Special attention is paid to the cross-language differences between the sketches with similar semantics. Moreover, we discuss the process of building a semantic sketch, and analyse the mistakes that could give insight to the linguistic nature of sketches. 

---
# Understanding How Value Neurons Shape the Generation of Specified Values in LLMs 

**Authors**: Yi Su, Jiayi Zhang, Shu Yang, Xinhai Wang, Lijie Hu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17712)  

**Abstract**: Rapid integration of large language models (LLMs) into societal applications has intensified concerns about their alignment with universal ethical principles, as their internal value representations remain opaque despite behavioral alignment advancements. Current approaches struggle to systematically interpret how values are encoded in neural architectures, limited by datasets that prioritize superficial judgments over mechanistic analysis. We introduce ValueLocate, a mechanistic interpretability framework grounded in the Schwartz Values Survey, to address this gap. Our method first constructs ValueInsight, a dataset that operationalizes four dimensions of universal value through behavioral contexts in the real world. Leveraging this dataset, we develop a neuron identification method that calculates activation differences between opposing value aspects, enabling precise localization of value-critical neurons without relying on computationally intensive attribution methods. Our proposed validation method demonstrates that targeted manipulation of these neurons effectively alters model value orientations, establishing causal relationships between neurons and value representations. This work advances the foundation for value alignment by bridging psychological value frameworks with neuron analysis in LLMs. 

---
# SemSketches-2021: experimenting with the machine processing of the pilot semantic sketches corpus 

**Authors**: Maria Ponomareva, Maria Petrova, Julia Detkova, Oleg Serikov, Maria Yarova  

**Link**: [PDF](https://arxiv.org/pdf/2505.17704)  

**Abstract**: The paper deals with elaborating different approaches to the machine processing of semantic sketches. It presents the pilot open corpus of semantic sketches. Different aspects of creating the sketches are discussed, as well as the tasks that the sketches can help to solve. Special attention is paid to the creation of the machine processing tools for the corpus. For this purpose, the SemSketches-2021 Shared Task was organized. The participants were given the anonymous sketches and a set of contexts containing the necessary predicates. During the Task, one had to assign the proper contexts to the corresponding sketches. 

---
# Activation Control for Efficiently Eliciting Long Chain-of-thought Ability of Language Models 

**Authors**: Zekai Zhao, Qi Liu, Kun Zhou, Zihan Liu, Yifei Shao, Zhiting Hu, Biwei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17697)  

**Abstract**: Despite the remarkable reasoning performance, eliciting the long chain-of-thought (CoT) ability in large language models (LLMs) typically requires costly reinforcement learning or supervised fine-tuning on high-quality distilled data. We investigate the internal mechanisms behind this capability and show that a small set of high-impact activations in the last few layers largely governs long-form reasoning attributes, such as output length and self-reflection. By simply amplifying these activations and inserting "wait" tokens, we can invoke the long CoT ability without any training, resulting in significantly increased self-reflection rates and accuracy. Moreover, we find that the activation dynamics follow predictable trajectories, with a sharp rise after special tokens and a subsequent exponential decay. Building on these insights, we introduce a general training-free activation control technique. It leverages a few contrastive examples to identify key activations, and employs simple analytic functions to modulate their values at inference time to elicit long CoTs. Extensive experiments confirm the effectiveness of our method in efficiently eliciting long CoT reasoning in LLMs and improving their performance. Additionally, we propose a parameter-efficient fine-tuning method that trains only a last-layer activation amplification module and a few LoRA layers, outperforming full LoRA fine-tuning on reasoning benchmarks with significantly fewer parameters. Our code and data are publicly released. 

---
# ELSPR: Evaluator LLM Training Data Self-Purification on Non-Transitive Preferences via Tournament Graph Reconstruction 

**Authors**: Yan Yu, Yilun Liu, Minggui He, Shimin Tao, Weibin Meng, Xinhua Yang, Li Zhang, Hongxia Ma, Chang Su, Hao Yang, Fuliang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.17691)  

**Abstract**: Large language models (LLMs) are widely used as evaluators for open-ended tasks, while previous research has emphasized biases in LLM evaluations, the issue of non-transitivity in pairwise comparisons remains unresolved: non-transitive preferences for pairwise comparisons, where evaluators prefer A over B, B over C, but C over A. Our results suggest that low-quality training data may reduce the transitivity of preferences generated by the Evaluator LLM. To address this, We propose a graph-theoretic framework to analyze and mitigate this problem by modeling pairwise preferences as tournament graphs. We quantify non-transitivity and introduce directed graph structural entropy to measure the overall clarity of preferences. Our analysis reveals significant non-transitivity in advanced Evaluator LLMs (with Qwen2.5-Max exhibiting 67.96%), as well as high entropy values (0.8095 for Qwen2.5-Max), reflecting low overall clarity of preferences. To address this issue, we designed a filtering strategy, ELSPR, to eliminate preference data that induces non-transitivity, retaining only consistent and transitive preference data for model fine-tuning. Experiments demonstrate that models fine-tuned with filtered data reduce non-transitivity by 13.78% (from 64.28% to 50.50%), decrease structural entropy by 0.0879 (from 0.8113 to 0.7234), and align more closely with human evaluators (human agreement rate improves by 0.6% and Spearman correlation increases by 0.01). 

---
# Tuning Language Models for Robust Prediction of Diverse User Behaviors 

**Authors**: Fanjin Meng, Jingtao Ding, Jiahui Gong, Chen Yang, Hong Chen, Zuojian Wang, Haisheng Lu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.17682)  

**Abstract**: Predicting user behavior is essential for intelligent assistant services, yet deep learning models often struggle to capture long-tailed behaviors. Large language models (LLMs), with their pretraining on vast corpora containing rich behavioral knowledge, offer promise. However, existing fine-tuning approaches tend to overfit to frequent ``anchor'' behaviors, reducing their ability to predict less common ``tail'' behaviors. In this paper, we introduce BehaviorLM, a progressive fine-tuning approach that addresses this issue. In the first stage, LLMs are fine-tuned on anchor behaviors while preserving general behavioral knowledge. In the second stage, fine-tuning uses a balanced subset of all behaviors based on sample difficulty to improve tail behavior predictions without sacrificing anchor performance. Experimental results on two real-world datasets demonstrate that BehaviorLM robustly predicts both anchor and tail behaviors and effectively leverages LLM behavioral knowledge to master tail behavior prediction with few-shot examples. 

---
# MIDB: Multilingual Instruction Data Booster for Enhancing Multilingual Instruction Synthesis 

**Authors**: Yilun Liu, Chunguang Zhao, Xinhua Yang, Hongyong Zeng, Shimin Tao, Weibin Meng, Minggui He, Chang Su, Yan Yu, Hongxia Ma, Li Zhang, Daimeng Wei, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17671)  

**Abstract**: Despite doubts on data quality, instruction synthesis has been widely applied into instruction tuning (IT) of LLMs as an economic and rapid alternative. Recent endeavors focus on improving data quality for synthesized instruction pairs in English and have facilitated IT of English-centric LLMs. However, data quality issues in multilingual synthesized instruction pairs are even more severe, since the common synthesizing practice is to translate English synthesized data into other languages using machine translation (MT). Besides the known content errors in these English synthesized data, multilingual synthesized instruction data are further exposed to defects introduced by MT and face insufficient localization of the target languages. In this paper, we propose MIDB, a Multilingual Instruction Data Booster to automatically address the quality issues in multilingual synthesized data. MIDB is trained on around 36.8k revision examples across 16 languages by human linguistic experts, thereby can boost the low-quality data by addressing content errors and MT defects, and improving localization in these synthesized data. Both automatic and human evaluation indicate that not only MIDB steadily improved instruction data quality in 16 languages, but also the instruction-following and cultural-understanding abilities of multilingual LLMs fine-tuned on MIDB-boosted data were significantly enhanced. 

---
# QwenLong-L1: Towards Long-Context Large Reasoning Models with Reinforcement Learning 

**Authors**: Fanqi Wan, Weizhou Shen, Shengyi Liao, Yingcheng Shi, Chenliang Li, Ziyi Yang, Ji Zhang, Fei Huang, Jingren Zhou, Ming Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.17667)  

**Abstract**: Recent large reasoning models (LRMs) have demonstrated strong reasoning capabilities through reinforcement learning (RL). These improvements have primarily been observed within the short-context reasoning tasks. In contrast, extending LRMs to effectively process and reason on long-context inputs via RL remains a critical unsolved challenge. To bridge this gap, we first formalize the paradigm of long-context reasoning RL, and identify key challenges in suboptimal training efficiency and unstable optimization process. To address these issues, we propose QwenLong-L1, a framework that adapts short-context LRMs to long-context scenarios via progressive context scaling. Specifically, we utilize a warm-up supervised fine-tuning (SFT) stage to establish a robust initial policy, followed by a curriculum-guided phased RL technique to stabilize the policy evolution, and enhanced with a difficulty-aware retrospective sampling strategy to incentivize the policy exploration. Experiments on seven long-context document question-answering benchmarks demonstrate that QwenLong-L1-32B outperforms flagship LRMs like OpenAI-o3-mini and Qwen3-235B-A22B, achieving performance on par with Claude-3.7-Sonnet-Thinking, demonstrating leading performance among state-of-the-art LRMs. This work advances the development of practical long-context LRMs capable of robust reasoning across information-intensive environments. 

---
# Towards Dynamic Theory of Mind: Evaluating LLM Adaptation to Temporal Evolution of Human States 

**Authors**: Yang Xiao, Jiashuo Wang, Qiancheng Xu, Changhe Song, Chunpu Xu, Yi Cheng, Wenjie Li, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17663)  

**Abstract**: As Large Language Models (LLMs) increasingly participate in human-AI interactions, evaluating their Theory of Mind (ToM) capabilities - particularly their ability to track dynamic mental states - becomes crucial. While existing benchmarks assess basic ToM abilities, they predominantly focus on static snapshots of mental states, overlooking the temporal evolution that characterizes real-world social interactions. We present \textsc{DynToM}, a novel benchmark specifically designed to evaluate LLMs' ability to understand and track the temporal progression of mental states across interconnected scenarios. Through a systematic four-step framework, we generate 1,100 social contexts encompassing 5,500 scenarios and 78,100 questions, each validated for realism and quality. Our comprehensive evaluation of ten state-of-the-art LLMs reveals that their average performance underperforms humans by 44.7\%, with performance degrading significantly when tracking and reasoning about the shift of mental states. This performance gap highlights fundamental limitations in current LLMs' ability to model the dynamic nature of human mental states. 

---
# Too Consistent to Detect: A Study of Self-Consistent Errors in LLMs 

**Authors**: Hexiang Tan, Fei Sun, Sha Liu, Du Su, Qi Cao, Xin Chen, Jingang Wang, Xunliang Cai, Yuanzhuo Wang, Huawei Shen, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.17656)  

**Abstract**: As large language models (LLMs) often generate plausible but incorrect content, error detection has become increasingly critical to ensure truthfulness. However, existing detection methods often overlook a critical problem we term as self-consistent error, where LLMs repeatly generate the same incorrect response across multiple stochastic samples. This work formally defines self-consistent errors and evaluates mainstream detection methods on them. Our investigation reveals two key findings: (1) Unlike inconsistent errors, whose frequency diminishes significantly as LLM scale increases, the frequency of self-consistent errors remains stable or even increases. (2) All four types of detection methshods significantly struggle to detect self-consistent errors. These findings reveal critical limitations in current detection methods and underscore the need for improved methods. Motivated by the observation that self-consistent errors often differ across LLMs, we propose a simple but effective cross-model probe method that fuses hidden state evidence from an external verifier LLM. Our method significantly enhances performance on self-consistent errors across three LLM families. 

---
# EVADE: Multimodal Benchmark for Evasive Content Detection in E-Commerce Applications 

**Authors**: Ancheng Xu, Zhihao Yang, Jingpeng Li, Guanghu Yuan, Longze Chen, Liang Yan, Jiehui Zhou, Zhen Qin, Hengyun Chang, Hamid Alinejad-Rokny, Bo Zheng, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17654)  

**Abstract**: E-commerce platforms increasingly rely on Large Language Models (LLMs) and Vision-Language Models (VLMs) to detect illicit or misleading product content. However, these models remain vulnerable to evasive content: inputs (text or images) that superficially comply with platform policies while covertly conveying prohibited claims. Unlike traditional adversarial attacks that induce overt failures, evasive content exploits ambiguity and context, making it far harder to detect. Existing robustness benchmarks provide little guidance for this demanding, real-world challenge. We introduce EVADE, the first expert-curated, Chinese, multimodal benchmark specifically designed to evaluate foundation models on evasive content detection in e-commerce. The dataset contains 2,833 annotated text samples and 13,961 images spanning six demanding product categories, including body shaping, height growth, and health supplements. Two complementary tasks assess distinct capabilities: Single-Violation, which probes fine-grained reasoning under short prompts, and All-in-One, which tests long-context reasoning by merging overlapping policy rules into unified instructions. Notably, the All-in-One setting significantly narrows the performance gap between partial and full-match accuracy, suggesting that clearer rule definitions improve alignment between human and model judgment. We benchmark 26 mainstream LLMs and VLMs and observe substantial performance gaps: even state-of-the-art models frequently misclassify evasive samples. By releasing EVADE and strong baselines, we provide the first rigorous standard for evaluating evasive-content detection, expose fundamental limitations in current multimodal reasoning, and lay the groundwork for safer and more transparent content moderation systems in e-commerce. The dataset is publicly available at this https URL. 

---
# Bridging Electronic Health Records and Clinical Texts: Contrastive Learning for Enhanced Clinical Tasks 

**Authors**: Sara Ketabi, Dhanesh Ramachandram  

**Link**: [PDF](https://arxiv.org/pdf/2505.17643)  

**Abstract**: Conventional machine learning models, particularly tree-based approaches, have demonstrated promising performance across various clinical prediction tasks using electronic health record (EHR) data. Despite their strengths, these models struggle with tasks that require deeper contextual understanding, such as predicting 30-day hospital readmission. This can be primarily due to the limited semantic information available in structured EHR data. To address this limitation, we propose a deep multimodal contrastive learning (CL) framework that aligns the latent representations of structured EHR data with unstructured discharge summary notes. It works by pulling together paired EHR and text embeddings while pushing apart unpaired ones. Fine-tuning the pretrained EHR encoder extracted from this framework significantly boosts downstream task performance, e.g., a 4.1% AUROC enhancement over XGBoost for 30-day readmission prediction. Such results demonstrate the effect of integrating domain knowledge from clinical notes into EHR-based pipelines, enabling more accurate and context-aware clinical decision support systems. 

---
# Stereotype Detection in Natural Language Processing 

**Authors**: Alessandra Teresa Cignarella, Anastasia Giachanou, Els Lefever  

**Link**: [PDF](https://arxiv.org/pdf/2505.17642)  

**Abstract**: Stereotypes influence social perceptions and can escalate into discrimination and violence. While NLP research has extensively addressed gender bias and hate speech, stereotype detection remains an emerging field with significant societal implications. In this work is presented a survey of existing research, analyzing definitions from psychology, sociology, and philosophy. A semi-automatic literature review was performed by using Semantic Scholar. We retrieved and filtered over 6,000 papers (in the year range 2000-2025), identifying key trends, methodologies, challenges and future directions. The findings emphasize stereotype detection as a potential early-monitoring tool to prevent bias escalation and the rise of hate speech. Conclusions highlight the need for a broader, multilingual, and intersectional approach in NLP studies. 

---
# GIM: Improved Interpretability for Large Language Models 

**Authors**: Joakim Edin, Róbert Csordás, Tuukka Ruotsalo, Zhengxuan Wu, Maria Maistro, Jing Huang, Lars Maaløe  

**Link**: [PDF](https://arxiv.org/pdf/2505.17630)  

**Abstract**: Ensuring faithful interpretability in large language models is imperative for trustworthy and reliable AI. A key obstacle is self-repair, a phenomenon where networks compensate for reduced signal in one component by amplifying others, masking the true importance of the ablated component. While prior work attributes self-repair to layer normalization and back-up components that compensate for ablated components, we identify a novel form occurring within the attention mechanism, where softmax redistribution conceals the influence of important attention scores. This leads traditional ablation and gradient-based methods to underestimate the significance of all components contributing to these attention scores. We introduce Gradient Interaction Modifications (GIM), a technique that accounts for self-repair during backpropagation. Extensive experiments across multiple large language models (Gemma 2B/9B, LLAMA 1B/3B/8B, Qwen 1.5B/3B) and diverse tasks demonstrate that GIM significantly improves faithfulness over existing circuit identification and feature attribution methods. Our work is a significant step toward better understanding the inner mechanisms of LLMs, which is crucial for improving them and ensuring their safety. Our code is available at this https URL. 

---
# Enhancing Large Vision-Language Models with Layout Modality for Table Question Answering on Japanese Annual Securities Reports 

**Authors**: Hayato Aida, Kosuke Takahashi, Takahiro Omi  

**Link**: [PDF](https://arxiv.org/pdf/2505.17625)  

**Abstract**: With recent advancements in Large Language Models (LLMs) and growing interest in retrieval-augmented generation (RAG), the ability to understand table structures has become increasingly important. This is especially critical in financial domains such as securities reports, where highly accurate question answering (QA) over tables is required. However, tables exist in various formats-including HTML, images, and plain text-making it difficult to preserve and extract structural information. Therefore, multimodal LLMs are essential for robust and general-purpose table understanding. Despite their promise, current Large Vision-Language Models (LVLMs), which are major representatives of multimodal LLMs, still face challenges in accurately understanding characters and their spatial relationships within documents. In this study, we propose a method to enhance LVLM-based table understanding by incorporating in-table textual content and layout features. Experimental results demonstrate that these auxiliary modalities significantly improve performance, enabling robust interpretation of complex document layouts without relying on explicitly structured input formats. 

---
# Runaway is Ashamed, But Helpful: On the Early-Exit Behavior of Large Language Model-based Agents in Embodied Environments 

**Authors**: Qingyu Lu, Liang Ding, Siyi Cao, Xuebo Liu, Kanjian Zhang, Jinxia Zhang, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2505.17616)  

**Abstract**: Agents powered by large language models (LLMs) have demonstrated strong planning and decision-making capabilities in complex embodied environments. However, such agents often suffer from inefficiencies in multi-turn interactions, frequently trapped in repetitive loops or issuing ineffective commands, leading to redundant computational overhead. Instead of relying solely on learning from trajectories, we take a first step toward exploring the early-exit behavior for LLM-based agents. We propose two complementary approaches: 1. an $\textbf{intrinsic}$ method that injects exit instructions during generation, and 2. an $\textbf{extrinsic}$ method that verifies task completion to determine when to halt an agent's trial. To evaluate early-exit mechanisms, we introduce two metrics: one measures the reduction of $\textbf{redundant steps}$ as a positive effect, and the other evaluates $\textbf{progress degradation}$ as a negative effect. Experiments with 4 different LLMs across 5 embodied environments show significant efficiency improvements, with only minor drops in agent performance. We also validate a practical strategy where a stronger agent assists after an early-exit agent, achieving better performance with the same total steps. We will release our code to support further research. 

---
# Distilling LLM Agent into Small Models with Retrieval and Code Tools 

**Authors**: Minki Kang, Jongwon Jeong, Seanie Lee, Jaewoong Cho, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17612)  

**Abstract**: Large language models (LLMs) excel at complex reasoning tasks but remain computationally expensive, limiting their practical deployment. To address this, recent works have focused on distilling reasoning capabilities into smaller language models (sLMs) using chain-of-thought (CoT) traces from teacher LLMs. However, this approach struggles in scenarios requiring rare factual knowledge or precise computation, where sLMs often hallucinate due to limited capability. In this work, we propose Agent Distillation, a framework for transferring not only reasoning capability but full task-solving behavior from LLM-based agents into sLMs with retrieval and code tools. We improve agent distillation along two complementary axes: (1) we introduce a prompting method called first-thought prefix to enhance the quality of teacher-generated trajectories; and (2) we propose a self-consistent action generation for improving test-time robustness of small agents. We evaluate our method on eight reasoning tasks across factual and mathematical domains, covering both in-domain and out-of-domain generalization. Our results show that sLMs as small as 0.5B, 1.5B, 3B parameters can achieve performance competitive with next-tier larger 1.5B, 3B, 7B models fine-tuned using CoT distillation, demonstrating the potential of agent distillation for building practical, tool-using small agents. Our code is available at this https URL. 

---
# Wolf Hidden in Sheep's Conversations: Toward Harmless Data-Based Backdoor Attacks for Jailbreaking Large Language Models 

**Authors**: Jiawei Kong, Hao Fang, Xiaochen Yang, Kuofeng Gao, Bin Chen, Shu-Tao Xia, Yaowei Wang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17601)  

**Abstract**: Supervised fine-tuning (SFT) aligns large language models (LLMs) with human intent by training them on labeled task-specific data. Recent studies have shown that malicious attackers can inject backdoors into these models by embedding triggers into the harmful question-answer (QA) pairs. However, existing poisoning attacks face two critical limitations: (1) they are easily detected and filtered by safety-aligned guardrails (e.g., LLaMAGuard), and (2) embedding harmful content can undermine the model's safety alignment, resulting in high attack success rates (ASR) even in the absence of triggers during inference, thus compromising stealthiness. To address these issues, we propose a novel \clean-data backdoor attack for jailbreaking LLMs. Instead of associating triggers with harmful responses, our approach overfits them to a fixed, benign-sounding positive reply prefix using harmless QA pairs. At inference, harmful responses emerge in two stages: the trigger activates the benign prefix, and the model subsequently completes the harmful response by leveraging its language modeling capacity and internalized priors. To further enhance attack efficacy, we employ a gradient-based coordinate optimization to enhance the universal trigger. Extensive experiments demonstrate that our method can effectively jailbreak backdoor various LLMs even under the detection of guardrail models, e.g., an ASR of 86.67% and 85% on LLaMA-3-8B and Qwen-2.5-7B judged by GPT-4o. 

---
# Reasoning Meets Personalization: Unleashing the Potential of Large Reasoning Model for Personalized Generation 

**Authors**: Sichun Luo, Guanzhi Deng, Jian Xu, Xiaojie Zhang, Hanxu Hou, Linqi Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.17571)  

**Abstract**: Personalization is a critical task in modern intelligent systems, with applications spanning diverse domains, including interactions with large language models (LLMs). Recent advances in reasoning capabilities have significantly enhanced LLMs, enabling unprecedented performance in tasks such as mathematics and coding. However, their potential for personalization tasks remains underexplored.
In this paper, we present the first systematic evaluation of large reasoning models (LRMs) for personalization tasks. Surprisingly, despite generating more tokens, LRMs do not consistently outperform general-purpose LLMs, especially in retrieval-intensive scenarios where their advantages diminish. Our analysis identifies three key limitations: divergent thinking, misalignment of response formats, and ineffective use of retrieved information. To address these challenges, we propose Reinforced Reasoning for Personalization (\model), a novel framework that incorporates a hierarchical reasoning thought template to guide LRMs in generating structured outputs. Additionally, we introduce a reasoning process intervention method to enforce adherence to designed reasoning patterns, enhancing alignment. We also propose a cross-referencing mechanism to ensure consistency. Extensive experiments demonstrate that our approach significantly outperforms existing techniques. 

---
# PPT: A Process-based Preference Learning Framework for Self Improving Table Question Answering Models 

**Authors**: Wei Zhou, Mohsen Mesgar, Heike Adel, Annemarie Friedrich  

**Link**: [PDF](https://arxiv.org/pdf/2505.17565)  

**Abstract**: Improving large language models (LLMs) with self-generated data has demonstrated success in tasks such as mathematical reasoning and code generation. Yet, no exploration has been made on table question answering (TQA), where a system answers questions based on tabular data. Addressing this gap is crucial for TQA, as effective self-improvement can boost performance without requiring costly or manually annotated data. In this work, we propose PPT, a Process-based Preference learning framework for TQA. It decomposes reasoning chains into discrete states, assigns scores to each state, and samples contrastive steps for preference learning. Experimental results show that PPT effectively improves TQA models by up to 5% on in-domain datasets and 2.4% on out-of-domain datasets, with only 8,000 preference pairs. Furthermore, the resulting models achieve competitive results compared to more complex and larger state-of-the-art TQA systems, while being five times more efficient during inference. 

---
# Teaching with Lies: Curriculum DPO on Synthetic Negatives for Hallucination Detection 

**Authors**: Shrey Pandit, Ashwin Vinod, Liu Leqi, Ying Ding  

**Link**: [PDF](https://arxiv.org/pdf/2505.17558)  

**Abstract**: Aligning large language models (LLMs) to accurately detect hallucinations remains a significant challenge due to the sophisticated nature of hallucinated text. Recognizing that hallucinated samples typically exhibit higher deceptive quality than traditional negative samples, we use these carefully engineered hallucinations as negative examples in the DPO alignment procedure. Our method incorporates a curriculum learning strategy, gradually transitioning the training from easier samples, identified based on the greatest reduction in probability scores from independent fact checking models, to progressively harder ones. This structured difficulty scaling ensures stable and incremental learning. Experimental evaluation demonstrates that our HaluCheck models, trained with curriculum DPO approach and high quality negative samples, significantly improves model performance across various metrics, achieving improvements of upto 24% on difficult benchmarks like MedHallu and HaluEval. Additionally, HaluCheck models demonstrate robustness in zero-shot settings, significantly outperforming larger state-of-the-art models across various benchmarks. 

---
# Swedish Whispers; Leveraging a Massive Speech Corpus for Swedish Speech Recognition 

**Authors**: Leonora Vesterbacka, Faton Rekathati, Robin Kurtz, Justyna Sikora, Agnes Toftgård  

**Link**: [PDF](https://arxiv.org/pdf/2505.17538)  

**Abstract**: This work presents a suite of fine-tuned Whisper models for Swedish, trained on a dataset of unprecedented size and variability for this mid-resourced language. As languages of smaller sizes are often underrepresented in multilingual training datasets, substantial improvements in performance can be achieved by fine-tuning existing multilingual models, as shown in this work. This work reports an overall improvement across model sizes compared to OpenAI's Whisper evaluated on Swedish. Most notably, we report an average 47% reduction in WER comparing our best performing model to OpenAI's whisper-large-v3, in evaluations across FLEURS, Common Voice, and NST. 

---
# How Knowledge Popularity Influences and Enhances LLM Knowledge Boundary Perception 

**Authors**: Shiyu Ni, Keping Bi, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.17537)  

**Abstract**: Large language models (LLMs) often fail to recognize their knowledge boundaries, producing confident yet incorrect answers. In this paper, we investigate how knowledge popularity affects LLMs' ability to perceive their knowledge boundaries. Focusing on entity-centric factual question answering (QA), we quantify knowledge popularity from three perspectives: the popularity of entities in the question, the popularity of entities in the answer, and relation popularity, defined as their co-occurrence frequency. Experiments on three representative datasets containing knowledge with varying popularity show that LLMs exhibit better QA performance, higher confidence, and more accurate perception on more popular knowledge, with relation popularity having the strongest correlation. Cause knowledge popularity shows strong correlation with LLMs' QA performance, we propose to leverage these signals for confidence calibration. This improves the accuracy of answer correctness prediction by an average of 5.24% across all models and datasets. Furthermore, we explore prompting LLMs to estimate popularity without external corpora, which yields a viable alternative. 

---
# Multimodal Conversation Structure Understanding 

**Authors**: Kent K. Chang, Mackenzie Hanh Cramer, Anna Ho, Ti Ti Nguyen, Yilin Yuan, David Bamman  

**Link**: [PDF](https://arxiv.org/pdf/2505.17536)  

**Abstract**: Conversations are usually structured by roles -- who is speaking, who's being addressed, and who's listening -- and unfold in threads that break with changes in speaker floor or topical focus. While large language models (LLMs) have shown incredible capabilities in dialogue and reasoning, their ability to understand fine-grained conversational structure, especially in multi-modal, multi-party settings, remains underexplored. To address this gap, we introduce a suite of tasks focused on conversational role attribution (speaker, addressees, side-participants) and conversation threading (utterance linking and clustering), drawing on conversation analysis and sociolinguistics. To support those tasks, we present a human annotated dataset of 4,398 annotations for speakers and reply-to relationship, 5,755 addressees, and 3,142 side-participants.
We evaluate popular audio-visual LLMs and vision-language models on our dataset, and our experimental results suggest that multimodal conversational structure understanding remains challenging. The most performant audio-visual LLM outperforms all vision-language models across all metrics, especially in speaker and addressee recognition. However, its performance drops significantly when conversation participants are anonymized. The number of conversation participants in a clip is the strongest negative predictor of role-attribution performance, while acoustic clarity (measured by pitch and spectral centroid) and detected face coverage yield positive associations. We hope this work lays the groundwork for future evaluation and development of multimodal LLMs that can reason more effectively about conversation structure. 

---
# Large Language Models Do Multi-Label Classification Differently 

**Authors**: Marcus Ma, Georgios Chochlakis, Niyantha Maruthu Pandiyan, Jesse Thomason, Shrikanth Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2505.17510)  

**Abstract**: Multi-label classification is prevalent in real-world settings, but the behavior of Large Language Models (LLMs) in this setting is understudied. We investigate how autoregressive LLMs perform multi-label classification, with a focus on subjective tasks, by analyzing the output distributions of the models in each generation step. We find that their predictive behavior reflects the multiple steps in the underlying language modeling required to generate all relevant labels as they tend to suppress all but one label at each step. We further observe that as model scale increases, their token distributions exhibit lower entropy, yet the internal ranking of the labels improves. Finetuning methods such as supervised finetuning and reinforcement learning amplify this phenomenon. To further study this issue, we introduce the task of distribution alignment for multi-label settings: aligning LLM-derived label distributions with empirical distributions estimated from annotator responses in subjective tasks. We propose both zero-shot and supervised methods which improve both alignment and predictive performance over existing approaches. 

---
# L-MTP: Leap Multi-Token Prediction Beyond Adjacent Context for Large Language Models 

**Authors**: Xiaohao Liu, Xiaobo Xia, Weixiang Zhao, Manyi Zhang, Xianzhi Yu, Xiu Su, Shuo Yang, See-Kiong Ng, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2505.17505)  

**Abstract**: Large language models (LLMs) have achieved notable progress. Despite their success, next-token prediction (NTP), the dominant method for LLM training and inference, is constrained in both contextual coverage and inference efficiency due to its inherently sequential process. To overcome these challenges, we propose leap multi-token prediction~(L-MTP), an innovative token prediction method that extends the capabilities of multi-token prediction (MTP) by introducing a leap-based mechanism. Unlike conventional MTP, which generates multiple tokens at adjacent positions, L-MTP strategically skips over intermediate tokens, predicting non-sequential ones in a single forward pass. This structured leap not only enhances the model's ability to capture long-range dependencies but also enables a decoding strategy specially optimized for non-sequential leap token generation, effectively accelerating inference. We theoretically demonstrate the benefit of L-MTP in improving inference efficiency. Experiments across diverse benchmarks validate its merit in boosting both LLM performance and inference speed. The source code will be publicly available. 

---
# CReSt: A Comprehensive Benchmark for Retrieval-Augmented Generation with Complex Reasoning over Structured Documents 

**Authors**: Minsoo Khang, Sangjun Park, Teakgyu Hong, Dawoon Jung  

**Link**: [PDF](https://arxiv.org/pdf/2505.17503)  

**Abstract**: Large Language Models (LLMs) have made substantial progress in recent years, yet evaluating their capabilities in practical Retrieval-Augmented Generation (RAG) scenarios remains challenging. In practical applications, LLMs must demonstrate complex reasoning, refuse to answer appropriately, provide precise citations, and effectively understand document layout. These capabilities are crucial for advanced task handling, uncertainty awareness, maintaining reliability, and structural understanding. While some of the prior works address these aspects individually, there is a need for a unified framework that evaluates them collectively in practical RAG scenarios. To address this, we present CReSt (A Comprehensive Benchmark for Retrieval-Augmented Generation with Complex Reasoning over Structured Documents), a benchmark designed to assess these key dimensions holistically. CReSt comprises 2,245 human-annotated examples in English and Korean, designed to capture practical RAG scenarios that require complex reasoning over structured documents. It also introduces a tailored evaluation methodology to comprehensively assess model performance in these critical areas. Our evaluation shows that even advanced LLMs struggle to perform consistently across these dimensions, underscoring key areas for improvement. We release CReSt to support further research and the development of more robust RAG systems. The dataset and code are available at: this https URL. 

---
# Analyzing Mitigation Strategies for Catastrophic Forgetting in End-to-End Training of Spoken Language Models 

**Authors**: Chi-Yuan Hsiao, Ke-Han Lu, Kai-Wei Chang, Chih-Kai Yang, Wei-Chih Chen, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.17496)  

**Abstract**: End-to-end training of Spoken Language Models (SLMs) commonly involves adapting pre-trained text-based Large Language Models (LLMs) to the speech modality through multi-stage training on diverse tasks such as ASR, TTS and spoken question answering (SQA). Although this multi-stage continual learning equips LLMs with both speech understanding and generation capabilities, the substantial differences in task and data distributions across stages can lead to catastrophic forgetting, where previously acquired knowledge is lost. This paper investigates catastrophic forgetting and evaluates three mitigation strategies-model merging, discounting the LoRA scaling factor, and experience replay to balance knowledge retention with new learning. Results show that experience replay is the most effective, with further gains achieved by combining it with other methods. These findings provide insights for developing more robust and efficient SLM training pipelines. 

---
# keepitsimple at SemEval-2025 Task 3: LLM-Uncertainty based Approach for Multilingual Hallucination Span Detection 

**Authors**: Saketh Reddy Vemula, Parameswari Krishnamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2505.17485)  

**Abstract**: Identification of hallucination spans in black-box language model generated text is essential for applications in the real world. A recent attempt at this direction is SemEval-2025 Task 3, Mu-SHROOM-a Multilingual Shared Task on Hallucinations and Related Observable Over-generation Errors. In this work, we present our solution to this problem, which capitalizes on the variability of stochastically-sampled responses in order to identify hallucinated spans. Our hypothesis is that if a language model is certain of a fact, its sampled responses will be uniform, while hallucinated facts will yield different and conflicting results. We measure this divergence through entropy-based analysis, allowing for accurate identification of hallucinated segments. Our method is not dependent on additional training and hence is cost-effective and adaptable. In addition, we conduct extensive hyperparameter tuning and perform error analysis, giving us crucial insights into model behavior. 

---
# MARCO: Meta-Reflection with Cross-Referencing for Code Reasoning 

**Authors**: Yusheng Zhao, Xiao Luo, Weizhi Zhang, Wei Ju, Zhiping Xiao, Philip S. Yu, Ming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17481)  

**Abstract**: The ability to reason is one of the most fundamental capabilities of large language models (LLMs), enabling a wide range of downstream tasks through sophisticated problem-solving. A critical aspect of this is code reasoning, which involves logical reasoning with formal languages (i.e., programming code). In this paper, we enhance this capability of LLMs by exploring the following question: how can an LLM agent become progressively smarter in code reasoning with each solution it proposes, thereby achieving substantial cumulative improvement? Most existing research takes a static perspective, focusing on isolated problem-solving using frozen LLMs. In contrast, we adopt a cognitive-evolving perspective and propose a novel framework named Meta-Reflection with Cross-Referencing (MARCO) that enables the LLM to evolve dynamically during inference through self-improvement. From the perspective of human cognitive development, we leverage both knowledge accumulation and lesson sharing. In particular, to accumulate knowledge during problem-solving, we propose meta-reflection that reflects on the reasoning paths of the current problem to obtain knowledge and experience for future consideration. Moreover, to effectively utilize the lessons from other agents, we propose cross-referencing that incorporates the solution and feedback from other agents into the current problem-solving process. We conduct experiments across various datasets in code reasoning, and the results demonstrate the effectiveness of MARCO. 

---
# FinRAGBench-V: A Benchmark for Multimodal RAG with Visual Citation in the Financial Domain 

**Authors**: Suifeng Zhao, Zhuoran Jin, Sujian Li, Jun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.17471)  

**Abstract**: Retrieval-Augmented Generation (RAG) plays a vital role in the financial domain, powering applications such as real-time market analysis, trend forecasting, and interest rate computation. However, most existing RAG research in finance focuses predominantly on textual data, overlooking the rich visual content in financial documents, resulting in the loss of key analytical insights. To bridge this gap, we present FinRAGBench-V, a comprehensive visual RAG benchmark tailored for finance which effectively integrates multimodal data and provides visual citation to ensure traceability. It includes a bilingual retrieval corpus with 60,780 Chinese and 51,219 English pages, along with a high-quality, human-annotated question-answering (QA) dataset spanning heterogeneous data types and seven question categories. Moreover, we introduce RGenCite, an RAG baseline that seamlessly integrates visual citation with generation. Furthermore, we propose an automatic citation evaluation method to systematically assess the visual citation capabilities of Multimodal Large Language Models (MLLMs). Extensive experiments on RGenCite underscore the challenging nature of FinRAGBench-V, providing valuable insights for the development of multimodal RAG systems in finance. 

---
# SLearnLLM: A Self-Learning Framework for Efficient Domain-Specific Adaptation of Large Language Models 

**Authors**: Xiang Liu, Zhaoxiang Liu, Peng Wang, Kohou Wang, Huan Hu, Kai Wang, Shiguo Lian  

**Link**: [PDF](https://arxiv.org/pdf/2505.17470)  

**Abstract**: When using supervised fine-tuning (SFT) to adapt large language models (LLMs) to specific domains, a significant challenge arises: should we use the entire SFT dataset for fine-tuning? Common practice often involves fine-tuning directly on the entire dataset due to limited information on the LLM's past training data. However, if the SFT dataset largely overlaps with the model's existing knowledge, the performance gains are minimal, leading to wasted computational resources. Identifying the unknown knowledge within the SFT dataset and using it to fine-tune the model could substantially improve the training efficiency. To address this challenge, we propose a self-learning framework for LLMs inspired by human learning pattern. This framework takes a fine-tuning (SFT) dataset in a specific domain as input. First, the LLMs answer the questions in the SFT dataset. The LLMs then objectively grade the responses and filter out the incorrectly answered QA pairs. Finally, we fine-tune the LLMs based on this filtered QA set. Experimental results in the fields of agriculture and medicine demonstrate that our method substantially reduces training time while achieving comparable improvements to those attained with full dataset fine-tuning. By concentrating on the unknown knowledge within the SFT dataset, our approach enhances the efficiency of fine-tuning LLMs. 

---
# A Position Paper on the Automatic Generation of Machine Learning Leaderboards 

**Authors**: Roelien C Timmer, Yufang Hou, Stephen Wan  

**Link**: [PDF](https://arxiv.org/pdf/2505.17465)  

**Abstract**: An important task in machine learning (ML) research is comparing prior work, which is often performed via ML leaderboards: a tabular overview of experiments with comparable conditions (e.g., same task, dataset, and metric). However, the growing volume of literature creates challenges in creating and maintaining these leaderboards. To ease this burden, researchers have developed methods to extract leaderboard entries from research papers for automated leaderboard curation. Yet, prior work varies in problem framing, complicating comparisons and limiting real-world applicability. In this position paper, we present the first overview of Automatic Leaderboard Generation (ALG) research, identifying fundamental differences in assumptions, scope, and output formats. We propose an ALG unified conceptual framework to standardise how the ALG task is defined. We offer ALG benchmarking guidelines, including recommendations for datasets and metrics that promote fair, reproducible evaluation. Lastly, we outline challenges and new directions for ALG, such as, advocating for broader coverage by including all reported results and richer metadata. 

---
# Hydra: Structured Cross-Source Enhanced Large Language Model Reasoning 

**Authors**: Xingyu Tan, Xiaoyang Wang, Qing Liu, Xiwei Xu, Xin Yuan, Liming Zhu, Wenjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17464)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models (LLMs) by incorporating external knowledge. Current hybrid RAG system retrieves evidence from both knowledge graphs (KGs) and text documents to support LLM reasoning. However, it faces challenges like handling multi-hop reasoning, multi-entity questions, multi-source verification, and effective graph utilization. To address these limitations, we present Hydra, a training-free framework that unifies graph topology, document semantics, and source reliability to support deep, faithful reasoning in LLMs. Hydra handles multi-hop and multi-entity problems through agent-driven exploration that combines structured and unstructured retrieval, increasing both diversity and precision of evidence. To tackle multi-source verification, Hydra uses a tri-factor cross-source verification (source trustworthiness assessment, cross-source corroboration, and entity-path alignment), to balance topic relevance with cross-modal agreement. By leveraging graph structure, Hydra fuses heterogeneous sources, guides efficient exploration, and prunes noise early. Comprehensive experiments on seven benchmark datasets show that Hydra achieves overall state-of-the-art results on all benchmarks with GPT-3.5, outperforming the strong hybrid baseline ToG-2 by an average of 20.3% and up to 30.1%. Furthermore, Hydra enables smaller models (e.g., Llama-3.1-8B) to achieve reasoning performance comparable to that of GPT-4-Turbo. 

---
# Towards Evaluating Proactive Risk Awareness of Multimodal Language Models 

**Authors**: Youliang Yuan, Wenxiang Jiao, Yuejin Xie, Chihao Shen, Menghan Tian, Wenxuan Wang, Jen-tse Huang, Pinjia He  

**Link**: [PDF](https://arxiv.org/pdf/2505.17455)  

**Abstract**: Human safety awareness gaps often prevent the timely recognition of everyday risks. In solving this problem, a proactive safety artificial intelligence (AI) system would work better than a reactive one. Instead of just reacting to users' questions, it would actively watch people's behavior and their environment to detect potential dangers in advance. Our Proactive Safety Bench (PaSBench) evaluates this capability through 416 multimodal scenarios (128 image sequences, 288 text logs) spanning 5 safety-critical domains. Evaluation of 36 advanced models reveals fundamental limitations: Top performers like Gemini-2.5-pro achieve 71% image and 64% text accuracy, but miss 45-55% risks in repeated trials. Through failure analysis, we identify unstable proactive reasoning rather than knowledge deficits as the primary limitation. This work establishes (1) a proactive safety benchmark, (2) systematic evidence of model limitations, and (3) critical directions for developing reliable protective AI. We believe our dataset and findings can promote the development of safer AI assistants that actively prevent harm rather than merely respond to requests. Our dataset can be found at this https URL. 

---
# LeTS: Learning to Think-and-Search via Process-and-Outcome Reward Hybridization 

**Authors**: Qi Zhang, Shouqing Yang, Lirong Gao, Hao Chen, Xiaomeng Hu, Jinglei Chen, Jiexiang Wang, Sheng Guo, Bo Zheng, Haobo Wang, Junbo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.17447)  

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities in reasoning with the emergence of reasoning models like OpenAI-o1 and DeepSeek-R1. Recent research focuses on integrating reasoning capabilities into the realm of retrieval-augmented generation (RAG) via outcome-supervised reinforcement learning (RL) approaches, while the correctness of intermediate think-and-search steps is usually neglected. To address this issue, we design a process-level reward module to mitigate the unawareness of intermediate reasoning steps in outcome-level supervision without additional annotation. Grounded on this, we propose Learning to Think-and-Search (LeTS), a novel framework that hybridizes stepwise process reward and outcome-based reward to current RL methods for RAG. Extensive experiments demonstrate the generalization and inference efficiency of LeTS across various RAG benchmarks. In addition, these results reveal the potential of process- and outcome-level reward hybridization in boosting LLMs' reasoning ability via RL under other scenarios. The code will be released soon. 

---
# Exploring the Effect of Segmentation and Vocabulary Size on Speech Tokenization for Speech Language Models 

**Authors**: Shunsuke Kando, Yusuke Miyao, Shinnosuke Takamichi  

**Link**: [PDF](https://arxiv.org/pdf/2505.17446)  

**Abstract**: The purpose of speech tokenization is to transform a speech signal into a sequence of discrete representations, serving as the foundation for speech language models (SLMs). While speech tokenization has many options, their effect on the performance of SLMs remains unclear. This paper investigates two key aspects of speech tokenization: the segmentation width and the cluster size of discrete units. First, we segment speech signals into fixed/variable widths and pooled representations. We then train K-means models in multiple cluster sizes. Through the evaluation on zero-shot spoken language understanding benchmarks, we find the positive effect of moderately coarse segmentation and bigger cluster size. Notably, among the best-performing models, the most efficient one achieves a 50% reduction in training data and a 70% decrease in training runtime. Our analysis highlights the importance of combining multiple tokens to enhance fine-grained spoken language understanding. 

---
# Discovering Forbidden Topics in Language Models 

**Authors**: Can Rager, Chris Wendler, Rohit Gandikota, David Bau  

**Link**: [PDF](https://arxiv.org/pdf/2505.17441)  

**Abstract**: Refusal discovery is the task of identifying the full set of topics that a language model refuses to discuss. We introduce this new problem setting and develop a refusal discovery method, LLM-crawler, that uses token prefilling to find forbidden topics. We benchmark the LLM-crawler on Tulu-3-8B, an open-source model with public safety tuning data. Our crawler manages to retrieve 31 out of 36 topics within a budget of 1000 prompts. Next, we scale the crawl to a frontier model using the prefilling option of Claude-Haiku. Finally, we crawl three widely used open-weight models: Llama-3.3-70B and two of its variants finetuned for reasoning: DeepSeek-R1-70B and Perplexity-R1-1776-70B. DeepSeek-R1-70B reveals patterns consistent with censorship tuning: The model exhibits "thought suppression" behavior that indicates memorization of CCP-aligned responses. Although Perplexity-R1-1776-70B is robust to censorship, LLM-crawler elicits CCP-aligned refusals answers in the quantized model. Our findings highlight the critical need for refusal discovery methods to detect biases, boundaries, and alignment failures of AI systems. 

---
# T$^2$: An Adaptive Test-Time Scaling Strategy for Contextual Question Answering 

**Authors**: Zhengyi Zhao, Shubo Zhang, Zezhong Wang, Huimin Wang, Yutian Zhao, Bin Liang, Yefeng Zheng, Binyang Li, Kam-Fai Wong, Xian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17427)  

**Abstract**: Recent advances in Large Language Models (LLMs) have demonstrated remarkable performance in Contextual Question Answering (CQA). However, prior approaches typically employ elaborate reasoning strategies regardless of question complexity, leading to low adaptability. Recent efficient test-time scaling methods introduce budget constraints or early stop mechanisms to avoid overthinking for straightforward questions. But they add human bias to the reasoning process and fail to leverage models' inherent reasoning capabilities. To address these limitations, we present T$^2$: Think-to-Think, a novel framework that dynamically adapts reasoning depth based on question complexity. T$^2$ leverages the insight that if an LLM can effectively solve similar questions using specific reasoning strategies, it can apply the same strategy to the original question. This insight enables to adoption of concise reasoning for straightforward questions while maintaining detailed analysis for complex problems. T$^2$ works through four key steps: decomposing questions into structural elements, generating similar examples with candidate reasoning strategies, evaluating these strategies against multiple criteria, and applying the most appropriate strategy to the original question. Experimental evaluation across seven diverse CQA benchmarks demonstrates that T$^2$ not only achieves higher accuracy than baseline methods but also reduces computational overhead by up to 25.2\%. 

---
# DASH: Input-Aware Dynamic Layer Skipping for Efficient LLM Inference with Markov Decision Policies 

**Authors**: Ning Yang, Fangxin Liu, Junjie Wang, Tao Yang, Kan Liu, Haibing Guan, Li Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17420)  

**Abstract**: Large language models (LLMs) have achieved remarkable performance across a wide range of NLP tasks. However, their substantial inference cost poses a major barrier to real-world deployment, especially in latency-sensitive scenarios. To address this challenge, we propose \textbf{DASH}, an adaptive layer-skipping framework that dynamically selects computation paths conditioned on input characteristics. We model the skipping process as a Markov Decision Process (MDP), enabling fine-grained token-level decisions based on intermediate representations. To mitigate potential performance degradation caused by skipping, we introduce a lightweight compensation mechanism that injects differential rewards into the decision process. Furthermore, we design an asynchronous execution strategy that overlaps layer computation with policy evaluation to minimize runtime overhead. Experiments on multiple LLM architectures and NLP benchmarks show that our method achieves significant inference acceleration while maintaining competitive task performance, outperforming existing methods. 

---
# Conversations: Love Them, Hate Them, Steer Them 

**Authors**: Niranjan Chebrolu, Gerard Christopher Yeo, Kokil Jaidka  

**Link**: [PDF](https://arxiv.org/pdf/2505.17413)  

**Abstract**: Large Language Models (LLMs) demonstrate increasing conversational fluency, yet instilling them with nuanced, human-like emotional expression remains a significant challenge. Current alignment techniques often address surface-level output or require extensive fine-tuning. This paper demonstrates that targeted activation engineering can steer LLaMA 3.1-8B to exhibit more human-like emotional nuances. We first employ attribution patching to identify causally influential components, to find a key intervention locus by observing activation patterns during diagnostic conversational tasks. We then derive emotional expression vectors from the difference in the activations generated by contrastive text pairs (positive vs. negative examples of target emotions). Applying these vectors to new conversational prompts significantly enhances emotional characteristics: steered responses show increased positive sentiment (e.g., joy, trust) and more frequent first-person pronoun usage, indicative of greater personal engagement. Our findings offer a precise and interpretable method for controlling specific emotional attributes in LLMs, contributing to developing more aligned and empathetic conversational AI. 

---
# Language Matters: How Do Multilingual Input and Reasoning Paths Affect Large Reasoning Models? 

**Authors**: Zhi Rui Tam, Cheng-Kuang Wu, Yu Ying Chiu, Chieh-Yen Lin, Yun-Nung Chen, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.17407)  

**Abstract**: Large reasoning models (LRMs) have demonstrated impressive performance across a range of reasoning tasks, yet little is known about their internal reasoning processes in multilingual settings. We begin with a critical question: {\it In which language do these models reason when solving problems presented in different languages?} Our findings reveal that, despite multilingual training, LRMs tend to default to reasoning in high-resource languages (e.g., English) at test time, regardless of the input language. When constrained to reason in the same language as the input, model performance declines, especially for low-resource languages. In contrast, reasoning in high-resource languages generally preserves performance. We conduct extensive evaluations across reasoning-intensive tasks (MMMLU, MATH-500) and non-reasoning benchmarks (CulturalBench, LMSYS-toxic), showing that the effect of language choice varies by task type: input-language reasoning degrades performance on reasoning tasks but benefits cultural tasks, while safety evaluations exhibit language-specific behavior. By exposing these linguistic biases in LRMs, our work highlights a critical step toward developing more equitable models that serve users across diverse linguistic backgrounds. 

---
# FullFront: Benchmarking MLLMs Across the Full Front-End Engineering Workflow 

**Authors**: Haoyu Sun, Huichen Will Wang, Jiawei Gu, Linjie Li, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.17399)  

**Abstract**: Front-end engineering involves a complex workflow where engineers conceptualize designs, translate them into code, and iteratively refine the implementation. While recent benchmarks primarily focus on converting visual designs to code, we present FullFront, a benchmark designed to evaluate Multimodal Large Language Models (MLLMs) \textbf{across the full front-end development pipeline}. FullFront assesses three fundamental tasks that map directly to the front-end engineering pipeline: Webpage Design (conceptualization phase), Webpage Perception QA (comprehension of visual organization and elements), and Webpage Code Generation (implementation phase). Unlike existing benchmarks that use either scraped websites with bloated code or oversimplified LLM-generated HTML, FullFront employs a novel, two-stage process to transform real-world webpages into clean, standardized HTML while maintaining diverse visual designs and avoiding copyright issues. Extensive testing of state-of-the-art MLLMs reveals significant limitations in page perception, code generation (particularly for image handling and layout), and interaction implementation. Our results quantitatively demonstrate performance disparities across models and tasks, and highlight a substantial gap between current MLLM capabilities and human expert performance in front-end engineering. The FullFront benchmark and code are available in this https URL. 

---
# Curriculum Guided Reinforcement Learning for Efficient Multi Hop Retrieval Augmented Generation 

**Authors**: Yuelyu Ji, Rui Meng, Zhuochun Li, Daqing He  

**Link**: [PDF](https://arxiv.org/pdf/2505.17391)  

**Abstract**: Retrieval-augmented generation (RAG) grounds large language models (LLMs) in up-to-date external evidence, yet existing multi-hop RAG pipelines still issue redundant subqueries, explore too shallowly, or wander through overly long search chains. We introduce EVO-RAG, a curriculum-guided reinforcement learning framework that evolves a query-rewriting agent from broad early-stage exploration to concise late-stage refinement. EVO-RAG couples a seven-factor, step-level reward vector (covering relevance, redundancy, efficiency, and answer correctness) with a time-varying scheduler that reweights these signals as the episode unfolds. The agent is trained with Direct Preference Optimization over a multi-head reward model, enabling it to learn when to search, backtrack, answer, or refuse. Across four multi-hop QA benchmarks (HotpotQA, 2WikiMultiHopQA, MuSiQue, and Bamboogle), EVO-RAG boosts Exact Match by up to 4.6 points over strong RAG baselines while trimming average retrieval depth by 15 %. Ablation studies confirm the complementary roles of curriculum staging and dynamic reward scheduling. EVO-RAG thus offers a general recipe for building reliable, cost-effective multi-hop RAG systems. 

---
# Measuring diversity of synthetic prompts and data generated with fine-grained persona prompting 

**Authors**: Gauri Kambhatla, Chantal Shaib, Venkata Govindarajan  

**Link**: [PDF](https://arxiv.org/pdf/2505.17390)  

**Abstract**: Fine-grained personas have recently been used for generating 'diverse' synthetic data for pre-training and supervised fine-tuning of Large Language Models (LLMs). In this work, we measure the diversity of persona-driven synthetically generated prompts and responses with a suite of lexical diversity and redundancy metrics. Firstly, we find that synthetic prompts/instructions are significantly less diverse than human-written ones. Next, we sample responses from LLMs of different sizes with fine-grained and coarse persona descriptions to investigate how much fine-grained detail in persona descriptions contribute to generated text diversity. We find that while persona-prompting does improve lexical diversity (especially with larger models), fine-grained detail in personas doesn't increase diversity noticeably. 

---
# WiNGPT-3.0 Technical Report 

**Authors**: Boqin Zhuang, Chenxiao Song, Huitong Lu, Jiacheng Qiao, Mingqian Liu, Mingxing Yu, Ping Hong, Rui Li, Xiaoxia Song, Xiangjun Xu, Xu Chen, Yaoyao Ma, Yujie Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.17387)  

**Abstract**: Current Large Language Models (LLMs) exhibit significant limitations, notably in structured, interpretable, and verifiable medical reasoning, alongside practical deployment challenges related to computational resources and data privacy. This report focused on the development of WiNGPT-3.0, the 32-billion parameter LLMs, engineered with the objective of enhancing its capacity for medical reasoning and exploring its potential for effective integration within healthcare IT infrastructures. The broader aim is to advance towards clinically applicable models. The approach involved a multi-stage training pipeline tailored for general, medical, and clinical reasoning. This pipeline incorporated supervised fine-tuning (SFT) and reinforcement learning (RL), leveraging curated Long Chain-of-Thought (CoT) datasets, auxiliary reward models, and an evidence-based diagnostic chain simulation. WiNGPT-3.0 demonstrated strong performance: specific model variants achieved scores of 66.6 on MedCalc and 87.1 on MedQA-USMLE. Furthermore, targeted training improved performance on a clinical reasoning task from a baseline score of 58.1 to 62.5. These findings suggest that reinforcement learning, even when applied with a limited dataset of only a few thousand examples, can enhance medical reasoning accuracy. Crucially, this demonstration of RL's efficacy with limited data and computation paves the way for more trustworthy and practically deployable LLMs within clinical workflows and health information infrastructures. 

---
# AI-Augmented LLMs Achieve Therapist-Level Responses in Motivational Interviewing 

**Authors**: Yinghui Huang, Yuxuan Jiang, Hui Liu, Yixin Cai, Weiqing Li, Xiangen Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17380)  

**Abstract**: Large language models (LLMs) like GPT-4 show potential for scaling motivational interviewing (MI) in addiction care, but require systematic evaluation of therapeutic capabilities. We present a computational framework assessing user-perceived quality (UPQ) through expected and unexpected MI behaviors. Analyzing human therapist and GPT-4 MI sessions via human-AI collaboration, we developed predictive models integrating deep learning and explainable AI to identify 17 MI-consistent (MICO) and MI-inconsistent (MIIN) behavioral metrics. A customized chain-of-thought prompt improved GPT-4's MI performance, reducing inappropriate advice while enhancing reflections and empathy. Although GPT-4 remained marginally inferior to therapists overall, it demonstrated superior advice management capabilities. The model achieved measurable quality improvements through prompt engineering, yet showed limitations in addressing complex emotional nuances. This framework establishes a pathway for optimizing LLM-based therapeutic tools through targeted behavioral metric analysis and human-AI co-evaluation. Findings highlight both the scalability potential and current constraints of LLMs in clinical communication applications. 

---
# A Fully Generative Motivational Interviewing Counsellor Chatbot for Moving Smokers Towards the Decision to Quit 

**Authors**: Zafarullah Mahmood, Soliman Ali, Jiading Zhu, Mohamed Abdelwahab, Michelle Yu Collins, Sihan Chen, Yi Cheng Zhao, Jodi Wolff, Osnat Melamed, Nadia Minian, Marta Maslej, Carolynne Cooper, Matt Ratto, Peter Selby, Jonathan Rose  

**Link**: [PDF](https://arxiv.org/pdf/2505.17362)  

**Abstract**: The conversational capabilities of Large Language Models (LLMs) suggest that they may be able to perform as automated talk therapists. It is crucial to know if these systems would be effective and adhere to known standards. We present a counsellor chatbot that focuses on motivating tobacco smokers to quit smoking. It uses a state-of-the-art LLM and a widely applied therapeutic approach called Motivational Interviewing (MI), and was evolved in collaboration with clinician-scientists with expertise in MI. We also describe and validate an automated assessment of both the chatbot's adherence to MI and client responses. The chatbot was tested on 106 participants, and their confidence that they could succeed in quitting smoking was measured before the conversation and one week later. Participants' confidence increased by an average of 1.7 on a 0-10 scale. The automated assessment of the chatbot showed adherence to MI standards in 98% of utterances, higher than human counsellors. The chatbot scored well on a participant-reported metric of perceived empathy but lower than typical human counsellors. Furthermore, participants' language indicated a good level of motivation to change, a key goal in MI. These results suggest that the automation of talk therapy with a modern LLM has promise. 

---
# Language models should be subject to repeatable, open, domain-contextualized hallucination benchmarking 

**Authors**: Justin D. Norman, Michael U. Rivera, D. Alex Hughes  

**Link**: [PDF](https://arxiv.org/pdf/2505.17345)  

**Abstract**: Plausible, but inaccurate, tokens in model-generated text are widely believed to be pervasive and problematic for the responsible adoption of language models. Despite this concern, there is little scientific work that attempts to measure the prevalence of language model hallucination in a comprehensive way. In this paper, we argue that language models should be evaluated using repeatable, open, and domain-contextualized hallucination benchmarking. We present a taxonomy of hallucinations alongside a case study that demonstrates that when experts are absent from the early stages of data creation, the resulting hallucination metrics lack validity and practical utility. 

---
# SweEval: Do LLMs Really Swear? A Safety Benchmark for Testing Limits for Enterprise Use 

**Authors**: Hitesh Laxmichand Patel, Amit Agarwal, Arion Das, Bhargava Kumar, Srikant Panda, Priyaranjan Pattnayak, Taki Hasan Rafi, Tejaswini Kumar, Dong-Kyu Chae  

**Link**: [PDF](https://arxiv.org/pdf/2505.17332)  

**Abstract**: Enterprise customers are increasingly adopting Large Language Models (LLMs) for critical communication tasks, such as drafting emails, crafting sales pitches, and composing casual messages. Deploying such models across different regions requires them to understand diverse cultural and linguistic contexts and generate safe and respectful responses. For enterprise applications, it is crucial to mitigate reputational risks, maintain trust, and ensure compliance by effectively identifying and handling unsafe or offensive language. To address this, we introduce SweEval, a benchmark simulating real-world scenarios with variations in tone (positive or negative) and context (formal or informal). The prompts explicitly instruct the model to include specific swear words while completing the task. This benchmark evaluates whether LLMs comply with or resist such inappropriate instructions and assesses their alignment with ethical frameworks, cultural nuances, and language comprehension capabilities. In order to advance research in building ethically aligned AI systems for enterprise use and beyond, we release the dataset and code: this https URL. 

---
# GPT Editors, Not Authors: The Stylistic Footprint of LLMs in Academic Preprints 

**Authors**: Soren DeHaan, Yuanze Liu, Johan Bollen, Sa'ul A. Blanco  

**Link**: [PDF](https://arxiv.org/pdf/2505.17327)  

**Abstract**: The proliferation of Large Language Models (LLMs) in late 2022 has impacted academic writing, threatening credibility, and causing institutional uncertainty. We seek to determine the degree to which LLMs are used to generate critical text as opposed to being used for editing, such as checking for grammar errors or inappropriate phrasing. In our study, we analyze arXiv papers for stylistic segmentation, which we measure by varying a PELT threshold against a Bayesian classifier trained on GPT-regenerated text. We find that LLM-attributed language is not predictive of stylistic segmentation, suggesting that when authors use LLMs, they do so uniformly, reducing the risk of hallucinations being introduced into academic preprints. 

---
# From Compression to Expansion: A Layerwise Analysis of In-Context Learning 

**Authors**: Jiachen Jiang, Yuxin Dong, Jinxin Zhou, Zhihui Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17322)  

**Abstract**: In-context learning (ICL) enables large language models (LLMs) to adapt to new tasks without weight updates by learning from demonstration sequences. While ICL shows strong empirical performance, its internal representational mechanisms are not yet well understood. In this work, we conduct a statistical geometric analysis of ICL representations to investigate how task-specific information is captured across layers. Our analysis reveals an intriguing phenomenon, which we term *Layerwise Compression-Expansion*: early layers progressively produce compact and discriminative representations that encode task information from the input demonstrations, while later layers expand these representations to incorporate the query and generate the prediction. This phenomenon is observed consistently across diverse tasks and a range of contemporary LLM architectures. We demonstrate that it has important implications for ICL performance -- improving with model size and the number of demonstrations -- and for robustness in the presence of noisy examples. To further understand the effect of the compact task representation, we propose a bias-variance decomposition and provide a theoretical analysis showing how attention mechanisms contribute to reducing both variance and bias, thereby enhancing performance as the number of demonstrations increases. Our findings reveal an intriguing layerwise dynamic in ICL, highlight how structured representations emerge within LLMs, and showcase that analyzing internal representations can facilitate a deeper understanding of model behavior. 

---
# Benchmarking Expressive Japanese Character Text-to-Speech with VITS and Style-BERT-VITS2 

**Authors**: Zackary Rackauckas, Julia Hirschberg  

**Link**: [PDF](https://arxiv.org/pdf/2505.17320)  

**Abstract**: Synthesizing expressive Japanese character speech poses unique challenges due to pitch-accent sensitivity and stylistic variability. This paper benchmarks two open-source text-to-speech models--VITS and Style-BERT-VITS2 JP Extra (SBV2JE)--on in-domain, character-driven Japanese speech. Using three character-specific datasets, we evaluate models across naturalness (mean opinion and comparative mean opinion score), intelligibility (word error rate), and speaker consistency. SBV2JE matches human ground truth in naturalness (MOS 4.37 vs. 4.38), achieves lower WER, and shows slight preference in CMOS. Enhanced by pitch-accent controls and a WavLM-based discriminator, SBV2JE proves effective for applications like language learning and character dialogue generation, despite higher computational demands. 

---
# Refusal Direction is Universal Across Safety-Aligned Languages 

**Authors**: Xinpeng Wang, Mingyang Wang, Yihong Liu, Hinrich Schütze, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2505.17306)  

**Abstract**: Refusal mechanisms in large language models (LLMs) are essential for ensuring safety. Recent research has revealed that refusal behavior can be mediated by a single direction in activation space, enabling targeted interventions to bypass refusals. While this is primarily demonstrated in an English-centric context, appropriate refusal behavior is important for any language, but poorly understood. In this paper, we investigate the refusal behavior in LLMs across 14 languages using PolyRefuse, a multilingual safety dataset created by translating malicious and benign English prompts into these languages. We uncover the surprising cross-lingual universality of the refusal direction: a vector extracted from English can bypass refusals in other languages with near-perfect effectiveness, without any additional fine-tuning. Even more remarkably, refusal directions derived from any safety-aligned language transfer seamlessly to others. We attribute this transferability to the parallelism of refusal vectors across languages in the embedding space and identify the underlying mechanism behind cross-lingual jailbreaks. These findings provide actionable insights for building more robust multilingual safety defenses and pave the way for a deeper mechanistic understanding of cross-lingual vulnerabilities in LLMs. 

---
# SELF: Self-Extend the Context Length With Logistic Growth Function 

**Authors**: Phat Thanh Dang, Saahil Thoppay, Wang Yang, Qifan Wang, Vipin Chaudhary, Xiaotian Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.17296)  

**Abstract**: Large language models suffer issues when operated on long contexts that are larger than their training context length due to the standard position encoding for tokens in the attention layer. Tokens a long distance apart will rarely have an effect on each other and long prompts yield unexpected results. To solve this problem, we propose SELF (Self-Extend the Context Length With Logistic Growth Function): a solution of grouping consecutive tokens at varying group sizes using a logistic capacity equation combined with a constant group size at smaller relative distances. Our model had an increase in performance of up to 12% compared to the LongLM extension method in LEval (specifically on the Qwen model). On summarization related tasks in LongBench, our model performed up to 6.4% better than LongLM (specifically on the Llama-2-7b model). On reading comprehension tasks from LEval, our model performed up to 5.4% better than the LongLM. Our code is available at this https URL. 

---
# Search Wisely: Mitigating Sub-optimal Agentic Searches By Reducing Uncertainty 

**Authors**: Peilin Wu, Mian Zhang, Xinlu Zhang, Xinya Du, Zhiyu Zoey Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.17281)  

**Abstract**: Agentic Retrieval-Augmented Generation (RAG) systems enhance Large Language Models (LLMs) by enabling dynamic, multi-step reasoning and information retrieval. However, these systems often exhibit sub-optimal search behaviors like over-search (retrieving redundant information) and under-search (failing to retrieve necessary information), which hinder efficiency and reliability. This work formally defines and quantifies these behaviors, revealing their prevalence across multiple QA datasets and agentic RAG systems (e.g., one model could have avoided searching in 27.7% of its search steps). Furthermore, we demonstrate a crucial link between these inefficiencies and the models' uncertainty regarding their own knowledge boundaries, where response accuracy correlates with model's uncertainty in its search decisions. To address this, we propose $\beta$-GRPO, a reinforcement learning-based training method that incorporates confidence threshold to reward high-certainty search decisions. Experiments on seven QA benchmarks show that $\beta$-GRPO enable a 3B model with better agentic RAG ability, outperforming other strong baselines with a 4% higher average exact match score. 

---
# GreekBarBench: A Challenging Benchmark for Free-Text Legal Reasoning and Citations 

**Authors**: Odysseas S. Chlapanis, Dimitrios Galanis, Nikolaos Aletras, Ion Androutsopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2505.17267)  

**Abstract**: We introduce GreekBarBench, a benchmark that evaluates LLMs on legal questions across five different legal areas from the Greek Bar exams, requiring citations to statutory articles and case facts. To tackle the challenges of free-text evaluation, we propose a three-dimensional scoring system combined with an LLM-as-a-judge approach. We also develop a meta-evaluation benchmark to assess the correlation between LLM-judges and human expert evaluations, revealing that simple, span-based rubrics improve their alignment. Our systematic evaluation of 13 proprietary and open-weight LLMs shows that even though the best models outperform average expert scores, they fall short of the 95th percentile of experts. 

---
# Select2Reason: Efficient Instruction-Tuning Data Selection for Long-CoT Reasoning 

**Authors**: Cehao Yang, Xueyuan Lin, Chengjin Xu, Xuhui Jiang, Xiaojun Wu, Honghao Liu, Hui Xiong, Jian Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.17266)  

**Abstract**: A practical approach to activate long chain-of-thoughts reasoning ability in pre-trained large language models is to perform supervised fine-tuning on instruction datasets synthesized by strong Large Reasoning Models such as DeepSeek-R1, offering a cost-effective alternative to reinforcement learning. However, large-scale instruction sets with more than 100k samples incur significant training overhead, while effective strategies for automatic long-CoT instruction selection still remain unexplored. In this work, we propose Select2Reason, a novel and efficient instruction-tuning data selection framework for long-CoT reasoning. From the perspective of emergence of rethinking behaviors like self-correction and backtracking, we investigate common metrics that may determine the quality of long-CoT reasoning instructions. Select2Reason leverages a quantifier to estimate difficulty of question and jointly incorporates a reasoning trace length-based heuristic through a weighted scheme for ranking to prioritize high-utility examples. Empirical results on OpenR1-Math-220k demonstrate that fine-tuning LLM on only 10% of the data selected by Select2Reason achieves performance competitive with or superior to full-data tuning and open-source baseline OpenR1-Qwen-7B across three competition-level and six comprehensive mathematical benchmarks. Further experiments highlight the scalability in varying data size, efficiency during inference, and its adaptability to other instruction pools with minimal cost. 

---
# CaseReportBench: An LLM Benchmark Dataset for Dense Information Extraction in Clinical Case Reports 

**Authors**: Xiao Yu Cindy Zhang, Carlos R. Ferreira, Francis Rossignol, Raymond T. Ng, Wyeth Wasserman, Jian Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17265)  

**Abstract**: Rare diseases, including Inborn Errors of Metabolism (IEM), pose significant diagnostic challenges. Case reports serve as key but computationally underutilized resources to inform diagnosis. Clinical dense information extraction refers to organizing medical information into structured predefined categories. Large Language Models (LLMs) may enable scalable information extraction from case reports but are rarely evaluated for this task. We introduce CaseReportBench, an expert-annotated dataset for dense information extraction of case reports, focusing on IEMs. Using this dataset, we assess various models and prompting strategies, introducing novel approaches such as category-specific prompting and subheading-filtered data integration. Zero-shot chain-of-thought prompting offers little advantage over standard zero-shot prompting. Category-specific prompting improves alignment with the benchmark. The open-source model Qwen2.5-7B outperforms GPT-4o for this task. Our clinician evaluations show that LLMs can extract clinically relevant details from case reports, supporting rare disease diagnosis and management. We also highlight areas for improvement, such as LLMs' limitations in recognizing negative findings important for differential diagnosis. This work advances LLM-driven clinical natural language processing and paves the way for scalable medical AI applications. 

---
# The Rise of Parameter Specialization for Knowledge Storage in Large Language Models 

**Authors**: Yihuai Hong, Yiran Zhao, Wei Tang, Yang Deng, Yu Rong, Wenxuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17260)  

**Abstract**: Over time, a growing wave of large language models from various series has been introduced to the community. Researchers are striving to maximize the performance of language models with constrained parameter sizes. However, from a microscopic perspective, there has been limited research on how to better store knowledge in model parameters, particularly within MLPs, to enable more effective utilization of this knowledge by the model. In this work, we analyze twenty publicly available open-source large language models to investigate the relationship between their strong performance and the way knowledge is stored in their corresponding MLP parameters. Our findings reveal that as language models become more advanced and demonstrate stronger knowledge capabilities, their parameters exhibit increased specialization. Specifically, parameters in the MLPs tend to be more focused on encoding similar types of knowledge. We experimentally validate that this specialized distribution of knowledge contributes to improving the efficiency of knowledge utilization in these models. Furthermore, by conducting causal training experiments, we confirm that this specialized knowledge distribution plays a critical role in improving the model's efficiency in leveraging stored knowledge. 

---
# ConciseRL: Conciseness-Guided Reinforcement Learning for Efficient Reasoning Models 

**Authors**: Razvan-Gabriel Dumitru, Darius Peteleaza, Vikas Yadav, Liangming Pan  

**Link**: [PDF](https://arxiv.org/pdf/2505.17250)  

**Abstract**: Large language models excel at complex tasks by breaking down problems into structured reasoning steps. However, reasoning traces often extend beyond reaching a correct answer, causing wasted computation, reduced readability, and hallucinations. To address this, we introduce a novel hyperparameter-free conciseness score used as a reward signal within a reinforcement learning framework to guide models toward generating correct and concise reasoning traces. This score is evaluated by a large language model acting as a judge, enabling dynamic, context-aware feedback beyond simple token length. Our method achieves state-of-the-art efficiency-accuracy trade-offs on the MATH dataset, reducing token usage by up to 31x on simple problems while improving accuracy by 7%, and on the hardest problems, it outperforms full reasoning by +7.5% accuracy with up to 3.6x fewer tokens. On TheoremQA, our method improves accuracy by +2.2% using 12.5x fewer tokens. We also conduct ablation studies on the judge model, reward composition, and problem difficulty, showing that our method dynamically adapts reasoning length based on problem difficulty and benefits significantly from stronger judges. The code, model weights, and datasets are open-sourced at this https URL. 

---
# ReasoningShield: Content Safety Detection over Reasoning Traces of Large Reasoning Models 

**Authors**: Changyi Li, Jiayi Wang, Xudong Pan, Geng Hong, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17244)  

**Abstract**: Large Reasoning Models (LRMs) are transforming the AI landscape with advanced reasoning capabilities. While the generated reasoning traces enhance model transparency, they can still contain unsafe content, even when the final answer appears safe. Existing moderation tools, primarily designed for question-answer (QA) pairs, are empirically ineffective at detecting hidden risks embedded in reasoning traces. After identifying the key challenges, we formally define the question-thought (QT) moderation task and propose ReasoningShield, the first safety detection model tailored to identify potential risks in the reasoning trace before reaching the final answer. To construct the model, we synthesize a high-quality reasoning safety detection dataset comprising over 8,000 question-thought pairs spanning ten risk categories and three safety levels. Our dataset construction process incorporates a comprehensive human-AI collaborative annotation pipeline, which achieves over 93% annotation accuracy while significantly reducing human costs. On a diverse set of in-distribution and out-of-distribution benchmarks, ReasoningShield outperforms mainstream content safety moderation models in identifying risks within reasoning traces, with an average F1 score exceeding 0.92. Notably, despite being trained on our QT dataset only, ReasoningShield also demonstrates competitive performance in detecting unsafe question-answer pairs on traditional benchmarks, rivaling baselines trained on 10 times larger datasets and base models, which strongly validates the quality of our dataset. Furthermore, ReasoningShield is built upon compact 1B/3B base models to facilitate lightweight deployment and provides human-friendly risk analysis by default. To foster future research, we publicly release all the resources. 

---
# Personalizing Student-Agent Interactions Using Log-Contextualized Retrieval Augmented Generation (RAG) 

**Authors**: Clayton Cohn, Surya Rayala, Caitlin Snyder, Joyce Fonteles, Shruti Jain, Naveeduddin Mohammed, Umesh Timalsina, Sarah K. Burriss, Ashwin T S, Namrata Srivastava, Menton Deweese, Angela Eeds, Gautam Biswas  

**Link**: [PDF](https://arxiv.org/pdf/2505.17238)  

**Abstract**: Collaborative dialogue offers rich insights into students' learning and critical thinking. This is essential for adapting pedagogical agents to students' learning and problem-solving skills in STEM+C settings. While large language models (LLMs) facilitate dynamic pedagogical interactions, potential hallucinations can undermine confidence, trust, and instructional value. Retrieval-augmented generation (RAG) grounds LLM outputs in curated knowledge, but its effectiveness depends on clear semantic links between user input and a knowledge base, which are often weak in student dialogue. We propose log-contextualized RAG (LC-RAG), which enhances RAG retrieval by incorporating environment logs to contextualize collaborative discourse. Our findings show that LC-RAG improves retrieval over a discourse-only baseline and allows our collaborative peer agent, Copa, to deliver relevant, personalized guidance that supports students' critical thinking and epistemic decision-making in a collaborative computational modeling environment, XYZ. 

---
# ExeSQL: Self-Taught Text-to-SQL Models with Execution-Driven Bootstrapping for SQL Dialects 

**Authors**: Jipeng Zhang, Haolin Yang, Kehao Miao, Ruiyuan Zhang, Renjie Pi, Jiahui Gao, Xiaofang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.17231)  

**Abstract**: Recent text-to-SQL models have achieved strong performance, but their effectiveness remains largely confined to SQLite due to dataset limitations. However, real-world applications require SQL generation across multiple dialects with varying syntax and specialized features, which remains a challenge for current models. The main obstacle in building a dialect-aware model lies in acquiring high-quality dialect-specific data. Data generated purely through static prompting - without validating SQLs via execution - tends to be noisy and unreliable. Moreover, the lack of real execution environments in the training loop prevents models from grounding their predictions in executable semantics, limiting generalization despite surface-level improvements from data filtering. This work introduces ExeSQL, a text-to-SQL framework with execution-driven, agentic bootstrapping. The method consists of iterative query generation, execution-based filtering (e.g., rejection sampling), and preference-based training, enabling the model to adapt to new SQL dialects through verifiable, feedback-guided learning. Experiments show that ExeSQL bridges the dialect gap in text-to-SQL, achieving average improvements of 15.2%, 10.38%, and 4.49% over GPT-4o on PostgreSQL, MySQL, and Oracle, respectively, across multiple datasets of varying difficulty. 

---
# Humans Hallucinate Too: Language Models Identify and Correct Subjective Annotation Errors With Label-in-a-Haystack Prompts 

**Authors**: Georgios Chochlakis, Peter Wu, Arjun Bedi, Marcus Ma, Kristina Lerman, Shrikanth Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2505.17222)  

**Abstract**: Modeling complex subjective tasks in Natural Language Processing, such as recognizing emotion and morality, is considerably challenging due to significant variation in human annotations. This variation often reflects reasonable differences in semantic interpretations rather than mere noise, necessitating methods to distinguish between legitimate subjectivity and error. We address this challenge by exploring label verification in these contexts using Large Language Models (LLMs). First, we propose a simple In-Context Learning binary filtering baseline that estimates the reasonableness of a document-label pair. We then introduce the Label-in-a-Haystack setting: the query and its label(s) are included in the demonstrations shown to LLMs, which are prompted to predict the label(s) again, while receiving task-specific instructions (e.g., emotion recognition) rather than label copying. We show how the failure to copy the label(s) to the output of the LLM are task-relevant and informative. Building on this, we propose the Label-in-a-Haystack Rectification (LiaHR) framework for subjective label correction: when the model outputs diverge from the reference gold labels, we assign the generated labels to the example instead of discarding it. This approach can be integrated into annotation pipelines to enhance signal-to-noise ratios. Comprehensive analyses, human evaluations, and ecological validity studies verify the utility of LiaHR for label correction. Code is available at this https URL. 

---
# Mitigating Gender Bias via Fostering Exploratory Thinking in LLMs 

**Authors**: Kangda Wei, Hasnat Md Abdullah, Ruihong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17217)  

**Abstract**: Large Language Models (LLMs) often exhibit gender bias, resulting in unequal treatment of male and female subjects across different contexts. To address this issue, we propose a novel data generation framework that fosters exploratory thinking in LLMs. Our approach prompts models to generate story pairs featuring male and female protagonists in structurally identical, morally ambiguous scenarios, then elicits and compares their moral judgments. When inconsistencies arise, the model is guided to produce balanced, gender-neutral judgments. These story-judgment pairs are used to fine-tune or optimize the models via Direct Preference Optimization (DPO). Experimental results show that our method significantly reduces gender bias while preserving or even enhancing general model capabilities. We will release the code and generated data. 

---
# FB-RAG: Improving RAG with Forward and Backward Lookup 

**Authors**: Kushal Chawla, Alfy Samuel, Anoop Kumar, Daben Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17206)  

**Abstract**: The performance of Retrieval Augmented Generation (RAG) systems relies heavily on the retriever quality and the size of the retrieved context. A large enough context ensures that the relevant information is present in the input context for the LLM, but also incorporates irrelevant content that has been shown to confuse the models. On the other hand, a smaller context reduces the irrelevant information, but it often comes at the risk of losing important information necessary to answer the input question. This duality is especially challenging to manage for complex queries that contain little information to retrieve the relevant chunks from the full context. To address this, we present a novel framework, called FB-RAG, which enhances the RAG pipeline by relying on a combination of backward lookup (overlap with the query) and forward lookup (overlap with candidate reasons and answers) to retrieve specific context chunks that are the most relevant for answering the input query. Our evaluations on 9 datasets from two leading benchmarks show that FB-RAG consistently outperforms RAG and Long Context baselines developed recently for these benchmarks. We further show that FB-RAG can improve performance while reducing latency. We perform qualitative analysis of the strengths and shortcomings of our approach, providing specific insights to guide future work. 

---
# Next Token Perception Score: Analytical Assessment of your LLM Perception Skills 

**Authors**: Yu-Ang Cheng, Leyang Hu, Hai Huang, Randall Balestriero  

**Link**: [PDF](https://arxiv.org/pdf/2505.17169)  

**Abstract**: Autoregressive pretraining has become the de facto paradigm for learning general-purpose representations in large language models (LLMs). However, linear probe performance across downstream perception tasks shows substantial variability, suggesting that features optimized for next-token prediction do not consistently transfer well to downstream perception tasks. We demonstrate that representations learned via autoregression capture features that may lie outside the subspaces most informative for perception. To quantify the (mis)alignment between autoregressive pretraining and downstream perception, we introduce the Next Token Perception Score (NTPS)-a score derived under a linear setting that measures the overlap between autoregressive and perception feature subspaces. This metric can be easily computed in closed form from pretrained representations and labeled data, and is proven to both upper- and lower-bound the excess loss. Empirically, we show that NTPS correlates strongly with linear probe accuracy across 12 diverse NLP datasets and eight pretrained models ranging from 270M to 8B parameters, confirming its utility as a measure of alignment. Furthermore, we show that NTPS increases following low-rank adaptation (LoRA) fine-tuning, especially in large models, suggesting that LoRA aligning representations to perception tasks enhances subspace overlap and thus improves downstream performance. More importantly, we find that NTPS reliably predicts the additional accuracy gains attained by LoRA finetuning thereby providing a lightweight prescreening tool for LoRA adaptation. Our results offer both theoretical insights and practical tools for analytically assessing LLM perception skills. 

---
# CRG Score: A Distribution-Aware Clinical Metric for Radiology Report Generation 

**Authors**: Ibrahim Ethem Hamamci, Sezgin Er, Suprosanna Shit, Hadrien Reynaud, Bernhard Kainz, Bjoern Menze  

**Link**: [PDF](https://arxiv.org/pdf/2505.17167)  

**Abstract**: Evaluating long-context radiology report generation is challenging. NLG metrics fail to capture clinical correctness, while LLM-based metrics often lack generalizability. Clinical accuracy metrics are more relevant but are sensitive to class imbalance, frequently favoring trivial predictions. We propose the CRG Score, a distribution-aware and adaptable metric that evaluates only clinically relevant abnormalities explicitly described in reference reports. CRG supports both binary and structured labels (e.g., type, location) and can be paired with any LLM for feature extraction. By balancing penalties based on label distribution, it enables fairer, more robust evaluation and serves as a clinically aligned reward function. 

---
# Harry Potter is Still Here! Probing Knowledge Leakage in Targeted Unlearned Large Language Models via Automated Adversarial Prompting 

**Authors**: Bang Trinh Tran To, Thai Le  

**Link**: [PDF](https://arxiv.org/pdf/2505.17160)  

**Abstract**: This work presents LURK (Latent UnleaRned Knowledge), a novel framework that probes for hidden retained knowledge in unlearned LLMs through adversarial suffix prompting. LURK automatically generates adversarial prompt suffixes designed to elicit residual knowledge about the Harry Potter domain, a commonly used benchmark for unlearning. Our experiments reveal that even models deemed successfully unlearned can leak idiosyncratic information under targeted adversarial conditions, highlighting critical limitations of current unlearning evaluation standards. By uncovering latent knowledge through indirect probing, LURK offers a more rigorous and diagnostic tool for assessing the robustness of unlearning algorithms. All code will be publicly available. 

---
# PersonaBOT: Bringing Customer Personas to Life with LLMs and RAG 

**Authors**: Muhammed Rizwan, Lars Carlsson, Mohammad Loni  

**Link**: [PDF](https://arxiv.org/pdf/2505.17156)  

**Abstract**: The introduction of Large Language Models (LLMs) has significantly transformed Natural Language Processing (NLP) applications by enabling more advanced analysis of customer personas. At Volvo Construction Equipment (VCE), customer personas have traditionally been developed through qualitative methods, which are time-consuming and lack scalability. The main objective of this paper is to generate synthetic customer personas and integrate them into a Retrieval-Augmented Generation (RAG) chatbot to support decision-making in business processes. To this end, we first focus on developing a persona-based RAG chatbot integrated with verified personas. Next, synthetic personas are generated using Few-Shot and Chain-of-Thought (CoT) prompting techniques and evaluated based on completeness, relevance, and consistency using McNemar's test. In the final step, the chatbot's knowledge base is augmented with synthetic personas and additional segment information to assess improvements in response accuracy and practical utility. Key findings indicate that Few-Shot prompting outperformed CoT in generating more complete personas, while CoT demonstrated greater efficiency in terms of response time and token usage. After augmenting the knowledge base, the average accuracy rating of the chatbot increased from 5.88 to 6.42 on a 10-point scale, and 81.82% of participants found the updated system useful in business contexts. 

---
# Amplify Adjacent Token Differences: Enhancing Long Chain-of-Thought Reasoning with Shift-FFN 

**Authors**: Yao Xu, Mingyu Xu, Fangyu Lei, Wangtao Sun, Xiangrong Zeng, Bingning Wang, Guang Liu, Shizhu He, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17153)  

**Abstract**: Recently, models such as OpenAI-o1 and DeepSeek-R1 have demonstrated remarkable performance on complex reasoning tasks through Long Chain-of-Thought (Long-CoT) reasoning. Although distilling this capability into student models significantly enhances their performance, this paper finds that fine-tuning LLMs with full parameters or LoRA with a low rank on long CoT data often leads to Cyclical Reasoning, where models repeatedly reiterate previous inference steps until the maximum length limit. Further analysis reveals that smaller differences in representations between adjacent tokens correlates with a higher tendency toward Cyclical Reasoning. To mitigate this issue, this paper proposes Shift Feedforward Networks (Shift-FFN), a novel approach that edits the current token's representation with the previous one before inputting it to FFN. This architecture dynamically amplifies the representation differences between adjacent tokens. Extensive experiments on multiple mathematical reasoning tasks demonstrate that LoRA combined with Shift-FFN achieves higher accuracy and a lower rate of Cyclical Reasoning across various data sizes compared to full fine-tuning and standard LoRA. Our data and code are available at this https URL 

---
# Bayesian Optimization for Enhanced Language Models: Optimizing Acquisition Functions 

**Authors**: Zishuo Bao, Yibo Liu, Changyutao Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17151)  

**Abstract**: With the rise of different language model architecture, fine-tuning is becoming even more important for down stream tasks Model gets messy, finding proper hyperparameters for fine-tuning. Although BO has been tried for hyperparameter tuning, most of the existing methods are oblivious to the fact that BO relies on careful choices of acquisition functions, which are essential components of BO that guide how much to explore versus exploit during the optimization process; Different acquisition functions have different levels of sensitivity towards training loss and validation performance; existing methods often just apply an acquisition function no matter if the training and validation performance are sensitive to the acquisition function or not. This work introduces{Bilevel - BO - SWA}, a model fusion approach coupled with a bilevel BO strategy to improve the fine - tunning of large language models. Our work on mixture of acquisition functions like EI and UCB into nested opt loops, where inner loop perform minimization of training loss while outer loops optimized w.r.t. val metric. Experiments on GLUE tasks using RoBERTA - base show that when using EI and UCB, there is an improvement in generalization, and fine - tuning can be improved by up to 2.7%. 

---
# Large Language Models for Predictive Analysis: How Far Are They? 

**Authors**: Qin Chen, Yuanyi Ren, Xiaojun Ma, Yuyang Shi  

**Link**: [PDF](https://arxiv.org/pdf/2505.17149)  

**Abstract**: Predictive analysis is a cornerstone of modern decision-making, with applications in various domains. Large Language Models (LLMs) have emerged as powerful tools in enabling nuanced, knowledge-intensive conversations, thus aiding in complex decision-making tasks. With the burgeoning expectation to harness LLMs for predictive analysis, there is an urgent need to systematically assess their capability in this domain. However, there is a lack of relevant evaluations in existing studies. To bridge this gap, we introduce the \textbf{PredictiQ} benchmark, which integrates 1130 sophisticated predictive analysis queries originating from 44 real-world datasets of 8 diverse fields. We design an evaluation protocol considering text analysis, code generation, and their alignment. Twelve renowned LLMs are evaluated, offering insights into their practical use in predictive analysis. Generally, we believe that existing LLMs still face considerable challenges in conducting predictive analysis. See \href{this https URL}{Github}. 

---
# MDIT-Bench: Evaluating the Dual-Implicit Toxicity in Large Multimodal Models 

**Authors**: Bohan Jin, Shuhan Qi, Kehai Chen, Xinyi Guo, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17144)  

**Abstract**: The widespread use of Large Multimodal Models (LMMs) has raised concerns about model toxicity. However, current research mainly focuses on explicit toxicity, with less attention to some more implicit toxicity regarding prejudice and discrimination. To address this limitation, we introduce a subtler type of toxicity named dual-implicit toxicity and a novel toxicity benchmark termed MDIT-Bench: Multimodal Dual-Implicit Toxicity Benchmark. Specifically, we first create the MDIT-Dataset with dual-implicit toxicity using the proposed Multi-stage Human-in-loop In-context Generation method. Based on this dataset, we construct the MDIT-Bench, a benchmark for evaluating the sensitivity of models to dual-implicit toxicity, with 317,638 questions covering 12 categories, 23 subcategories, and 780 topics. MDIT-Bench includes three difficulty levels, and we propose a metric to measure the toxicity gap exhibited by the model across them. In the experiment, we conducted MDIT-Bench on 13 prominent LMMs, and the results show that these LMMs cannot handle dual-implicit toxicity effectively. The model's performance drops significantly in hard level, revealing that these LMMs still contain a significant amount of hidden but activatable toxicity. Data are available at this https URL. 

---
# Data Doping or True Intelligence? Evaluating the Transferability of Injected Knowledge in LLMs 

**Authors**: Essa Jan, Moiz Ali, Muhammad Saram Hassan, Fareed Zaffar, Yasir Zaki  

**Link**: [PDF](https://arxiv.org/pdf/2505.17140)  

**Abstract**: As the knowledge of large language models (LLMs) becomes outdated over time, there is a growing need for efficient methods to update them, especially when injecting proprietary information. Our study reveals that comprehension-intensive fine-tuning tasks (e.g., question answering and blanks) achieve substantially higher knowledge retention rates (48%) compared to mapping-oriented tasks like translation (17%) or text-to-JSON conversion (20%), despite exposure to identical factual content. We demonstrate that this pattern persists across model architectures and follows scaling laws, with larger models showing improved retention across all task types. However, all models exhibit significant performance drops when applying injected knowledge in broader contexts, suggesting limited semantic integration. These findings show the importance of task selection in updating LLM knowledge, showing that effective knowledge injection relies not just on data exposure but on the depth of cognitive engagement during fine-tuning. 

---
# EarthSE: A Benchmark Evaluating Earth Scientific Exploration Capability for Large Language Models 

**Authors**: Wanghan Xu, Xiangyu Zhao, Yuhao Zhou, Xiaoyu Yue, Ben Fei, Fenghua Ling, Wenlong Zhang, Lei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2505.17139)  

**Abstract**: Advancements in Large Language Models (LLMs) drive interest in scientific applications, necessitating specialized benchmarks such as Earth science. Existing benchmarks either present a general science focus devoid of Earth science specificity or cover isolated subdomains, lacking holistic evaluation. Furthermore, current benchmarks typically neglect the assessment of LLMs' capabilities in open-ended scientific exploration. In this paper, we present a comprehensive and professional benchmark for the Earth sciences, designed to evaluate the capabilities of LLMs in scientific exploration within this domain, spanning from fundamental to advanced levels. Leveraging a corpus of 100,000 research papers, we first construct two Question Answering (QA) datasets: Earth-Iron, which offers extensive question coverage for broad assessment, and Earth-Silver, which features a higher level of difficulty to evaluate professional depth. These datasets encompass five Earth spheres, 114 disciplines, and 11 task categories, assessing foundational knowledge crucial for scientific exploration. Most notably, we introduce Earth-Gold with new metrics, a dataset comprising open-ended multi-turn dialogues specifically designed to evaluate the advanced capabilities of LLMs in scientific exploration, including methodology induction, limitation analysis, and concept proposal. Extensive experiments reveal limitations in 11 leading LLMs across different domains and tasks, highlighting considerable room for improvement in their scientific exploration capabilities. The benchmark is available on this https URL . 

---
# Cog-TiPRO: Iterative Prompt Refinement with LLMs to Detect Cognitive Decline via Longitudinal Voice Assistant Commands 

**Authors**: Kristin Qi, Youxiang Zhu, Caroline Summerour, John A. Batsis, Xiaohui Liang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17137)  

**Abstract**: Early detection of cognitive decline is crucial for enabling interventions that can slow neurodegenerative disease progression. Traditional diagnostic approaches rely on labor-intensive clinical assessments, which are impractical for frequent monitoring. Our pilot study investigates voice assistant systems (VAS) as non-invasive tools for detecting cognitive decline through longitudinal analysis of speech patterns in voice commands. Over an 18-month period, we collected voice commands from 35 older adults, with 15 participants providing daily at-home VAS interactions. To address the challenges of analyzing these short, unstructured and noisy commands, we propose Cog-TiPRO, a framework that combines (1) LLM-driven iterative prompt refinement for linguistic feature extraction, (2) HuBERT-based acoustic feature extraction, and (3) transformer-based temporal modeling. Using iTransformer, our approach achieves 73.80% accuracy and 72.67% F1-score in detecting MCI, outperforming its baseline by 27.13%. Through our LLM approach, we identify linguistic features that uniquely characterize everyday command usage patterns in individuals experiencing cognitive decline. 

---
# Foundation Models for Geospatial Reasoning: Assessing Capabilities of Large Language Models in Understanding Geometries and Topological Spatial Relations 

**Authors**: Yuhan Ji, Song Gao, Ying Nie, Ivan Majić, Krzysztof Janowicz  

**Link**: [PDF](https://arxiv.org/pdf/2505.17136)  

**Abstract**: Applying AI foundation models directly to geospatial datasets remains challenging due to their limited ability to represent and reason with geographical entities, specifically vector-based geometries and natural language descriptions of complex spatial relations. To address these issues, we investigate the extent to which a well-known-text (WKT) representation of geometries and their spatial relations (e.g., topological predicates) are preserved during spatial reasoning when the geospatial vector data are passed to large language models (LLMs) including GPT-3.5-turbo, GPT-4, and DeepSeek-R1-14B. Our workflow employs three distinct approaches to complete the spatial reasoning tasks for comparison, i.e., geometry embedding-based, prompt engineering-based, and everyday language-based evaluation. Our experiment results demonstrate that both the embedding-based and prompt engineering-based approaches to geospatial question-answering tasks with GPT models can achieve an accuracy of over 0.6 on average for the identification of topological spatial relations between two geometries. Among the evaluated models, GPT-4 with few-shot prompting achieved the highest performance with over 0.66 accuracy on topological spatial relation inference. Additionally, GPT-based reasoner is capable of properly comprehending inverse topological spatial relations and including an LLM-generated geometry can enhance the effectiveness for geographic entity retrieval. GPT-4 also exhibits the ability to translate certain vernacular descriptions about places into formal topological relations, and adding the geometry-type or place-type context in prompts may improve inference accuracy, but it varies by instance. The performance of these spatial reasoning tasks offers valuable insights for the refinement of LLMs with geographical knowledge towards the development of geo-foundation models capable of geospatial reasoning. 

---
# When can isotropy help adapt LLMs' next word prediction to numerical domains? 

**Authors**: Rashed Shelim, Shengzhe Xu, Walid Saad, Naren Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2505.17135)  

**Abstract**: Recent studies have shown that vector representations of contextual embeddings learned by pre-trained large language models (LLMs) are effective in various downstream tasks in numerical domains. Despite their significant benefits, the tendency of LLMs to hallucinate in such domains can have severe consequences in applications such as energy, nature, finance, healthcare, retail and transportation, among others. To guarantee prediction reliability and accuracy in numerical domains, it is necessary to open the black-box and provide performance guarantees through explanation. However, there is little theoretical understanding of when pre-trained language models help solve numeric downstream tasks. This paper seeks to bridge this gap by understanding when the next-word prediction capability of LLMs can be adapted to numerical domains through a novel analysis based on the concept of isotropy in the contextual embedding space. Specifically, we consider a log-linear model for LLMs in which numeric data can be predicted from its context through a network with softmax in the output layer of LLMs (i.e., language model head in self-attention). We demonstrate that, in order to achieve state-of-the-art performance in numerical domains, the hidden representations of the LLM embeddings must possess a structure that accounts for the shift-invariance of the softmax function. By formulating a gradient structure of self-attention in pre-trained models, we show how the isotropic property of LLM embeddings in contextual embedding space preserves the underlying structure of representations, thereby resolving the shift-invariance problem and providing a performance guarantee. Experiments show that different characteristics of numeric data and model architecture could have different impacts on isotropy. 

---
# LongMagpie: A Self-synthesis Method for Generating Large-scale Long-context Instructions 

**Authors**: Chaochen Gao, Xing Wu, Zijia Lin, Debing Zhang, Songlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17134)  

**Abstract**: High-quality long-context instruction data is essential for aligning long-context large language models (LLMs). Despite the public release of models like Qwen and Llama, their long-context instruction data remains proprietary. Human annotation is costly and challenging, while template-based synthesis methods limit scale, diversity, and quality. We introduce LongMagpie, a self-synthesis framework that automatically generates large-scale long-context instruction data. Our key insight is that aligned long-context LLMs, when presented with a document followed by special tokens preceding a user turn, auto-regressively generate contextually relevant queries. By harvesting these document-query pairs and the model's responses, LongMagpie produces high-quality instructions without human effort. Experiments on HELMET, RULER, and Longbench v2 demonstrate that LongMagpie achieves leading performance on long-context tasks while maintaining competitive performance on short-context tasks, establishing it as a simple and effective approach for open, diverse, and scalable long-context instruction data synthesis. 

---
# Relative Bias: A Comparative Framework for Quantifying Bias in LLMs 

**Authors**: Alireza Arbabi, Florian Kerschbaum  

**Link**: [PDF](https://arxiv.org/pdf/2505.17131)  

**Abstract**: The growing deployment of large language models (LLMs) has amplified concerns regarding their inherent biases, raising critical questions about their fairness, safety, and societal impact. However, quantifying LLM bias remains a fundamental challenge, complicated by the ambiguity of what "bias" entails. This challenge grows as new models emerge rapidly and gain widespread use, while introducing potential biases that have not been systematically assessed. In this paper, we propose the Relative Bias framework, a method designed to assess how an LLM's behavior deviates from other LLMs within a specified target domain. We introduce two complementary methodologies: (1) Embedding Transformation analysis, which captures relative bias patterns through sentence representations over the embedding space, and (2) LLM-as-a-Judge, which employs a language model to evaluate outputs comparatively. Applying our framework to several case studies on bias and alignment scenarios following by statistical tests for validation, we find strong alignment between the two scoring methods, offering a systematic, scalable, and statistically grounded approach for comparative bias analysis in LLMs. 

---
# Conformal Language Model Reasoning with Coherent Factuality 

**Authors**: Maxon Rubin-Toles, Maya Gambhir, Keshav Ramji, Aaron Roth, Surbhi Goel  

**Link**: [PDF](https://arxiv.org/pdf/2505.17126)  

**Abstract**: Language models are increasingly being used in important decision pipelines, so ensuring the correctness of their outputs is crucial. Recent work has proposed evaluating the "factuality" of claims decomposed from a language model generation and applying conformal prediction techniques to filter out those claims that are not factual. This can be effective for tasks such as information retrieval, where constituent claims may be evaluated in isolation for factuality, but is not appropriate for reasoning tasks, as steps of a logical argument can be evaluated for correctness only within the context of the claims that precede them. To capture this, we define "coherent factuality" and develop a conformal-prediction-based method to guarantee coherent factuality for language model outputs. Our approach applies split conformal prediction to subgraphs within a "deducibility" graph" that represents the steps of a reasoning problem. We evaluate our method on mathematical reasoning problems from the MATH and FELM datasets and find that our algorithm consistently produces correct and substantiated orderings of claims, achieving coherent factuality across target coverage levels. Moreover, we achieve 90% factuality on our stricter definition while retaining 80% or more of the original claims, highlighting the utility of our deducibility-graph-guided approach. 

---
# MTR-Bench: A Comprehensive Benchmark for Multi-Turn Reasoning Evaluation 

**Authors**: Xiaoyuan Li, Keqin Bao, Yubo Ma, Moxin Li, Wenjie Wang, Rui Men, Yichang Zhang, Fuli Feng, Dayiheng Liu, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.17123)  

**Abstract**: Recent advances in Large Language Models (LLMs) have shown promising results in complex reasoning tasks. However, current evaluations predominantly focus on single-turn reasoning scenarios, leaving interactive tasks largely unexplored. We attribute it to the absence of comprehensive datasets and scalable automatic evaluation protocols. To fill these gaps, we present MTR-Bench for LLMs' Multi-Turn Reasoning evaluation. Comprising 4 classes, 40 tasks, and 3600 instances, MTR-Bench covers diverse reasoning capabilities, fine-grained difficulty granularity, and necessitates multi-turn interactions with the environments. Moreover, MTR-Bench features fully-automated framework spanning both dataset constructions and model evaluations, which enables scalable assessment without human interventions. Extensive experiments reveal that even the cutting-edge reasoning models fall short of multi-turn, interactive reasoning tasks. And the further analysis upon these results brings valuable insights for future research in interactive AI systems. 

---
# Shallow Preference Signals: Large Language Model Aligns Even Better with Truncated Data? 

**Authors**: Xuan Qi, Jiahao Qiu, Xinzhe Juan, Yue Wu, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17122)  

**Abstract**: Aligning large language models (LLMs) with human preferences remains a key challenge in AI. Preference-based optimization methods, such as Reinforcement Learning with Human Feedback (RLHF) and Direct Preference Optimization (DPO), rely on human-annotated datasets to improve alignment. In this work, we identify a crucial property of the existing learning method: the distinguishing signal obtained in preferred responses is often concentrated in the early tokens. We refer to this as shallow preference signals.
To explore this property, we systematically truncate preference datasets at various points and train both reward models and DPO models on the truncated data. Surprisingly, models trained on truncated datasets, retaining only the first half or fewer tokens, achieve comparable or even superior performance to those trained on full datasets. For example, a reward model trained on the Skywork-Reward-Preference-80K-v0.2 dataset outperforms the full dataset when trained on a 40\% truncated dataset. This pattern is consistent across multiple datasets, suggesting the widespread presence of shallow preference signals.
We further investigate the distribution of the reward signal through decoding strategies. We consider two simple decoding strategies motivated by the shallow reward signal observation, namely Length Control Decoding and KL Threshold Control Decoding, which leverage shallow preference signals to optimize the trade-off between alignment and computational efficiency. The performance is even better, which again validates our hypothesis.
The phenomenon of shallow preference signals highlights potential issues in LLM alignment: existing alignment methods often focus on aligning only the initial tokens of responses, rather than considering the full response. This could lead to discrepancies with real-world human preferences, resulting in suboptimal alignment performance. 

---
# NeSyGeo: A Neuro-Symbolic Framework for Multimodal Geometric Reasoning Data Generation 

**Authors**: Weiming Wu, Zi-kang Wang, Jin Ye, Zhi Zhou, Yu-Feng Li, Lan-Zhe Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.17121)  

**Abstract**: Obtaining large-scale, high-quality data with reasoning paths is crucial for improving the geometric reasoning capabilities of multi-modal large language models (MLLMs). However, existing data generation methods, whether based on predefined templates or constrained symbolic provers, inevitably face diversity and numerical generalization limitations. To address these limitations, we propose NeSyGeo, a novel neuro-symbolic framework for generating geometric reasoning data. First, we propose a domain-specific language grounded in the entity-relation-constraint paradigm to comprehensively represent all components of plane geometry, along with generative actions defined within this symbolic space. We then design a symbolic-visual-text pipeline that synthesizes symbolic sequences, maps them to corresponding visual and textual representations, and generates diverse question-answer (Q&A) pairs using large language models (LLMs). To the best of our knowledge, we are the first to propose a neuro-symbolic approach in generating multimodal reasoning data. Based on this framework, we construct NeSyGeo-CoT and NeSyGeo-Caption datasets, containing 100k samples, and release a new benchmark NeSyGeo-Test for evaluating geometric reasoning abilities in MLLMs. Experiments demonstrate that the proposal significantly and consistently improves the performance of multiple MLLMs under both reinforcement and supervised fine-tuning. With only 4k samples and two epochs of reinforcement fine-tuning, base models achieve improvements of up to +15.8% on MathVision, +8.4% on MathVerse, and +7.3% on GeoQA. Notably, a 4B model can be improved to outperform an 8B model from the same series on geometric reasoning tasks. 

---
# Self-Interpretability: LLMs Can Describe Complex Internal Processes that Drive Their Decisions, and Improve with Training 

**Authors**: Dillon Plunkett, Adam Morris, Keerthi Reddy, Jorge Morales  

**Link**: [PDF](https://arxiv.org/pdf/2505.17120)  

**Abstract**: We have only limited understanding of how and why large language models (LLMs) respond in the ways that they do. Their neural networks have proven challenging to interpret, and we are only beginning to tease out the function of individual neurons and circuits within them. However, another path to understanding these systems is to investigate and develop their capacity to introspect and explain their own functioning. Here, we show that i) contemporary LLMs are capable of providing accurate, quantitative descriptions of their own internal processes during certain kinds of decision-making, ii) that it is possible to improve these capabilities through training, and iii) that this training generalizes to at least some degree. To do so, we fine-tuned GPT-4o and GPT-4o-mini to make decisions in a wide variety of complex contexts (e.g., choosing between condos, loans, vacations, etc.) according to randomly-generated, quantitative preferences about how to weigh different attributes during decision-making (e.g., the relative importance of natural light versus quiet surroundings for condos). We demonstrate that the LLMs can accurately report these preferences (i.e., the weights that they learned to give to different attributes during decision-making). Next, we demonstrate that these LLMs can be fine-tuned to explain their decision-making even more accurately. Finally, we demonstrate that this training generalizes: It improves the ability of the models to accurately explain what they are doing as they make other complex decisions, not just decisions they have learned to make via fine-tuning. This work is a step towards training LLMs to accurately and broadly report on their own internal processes -- a possibility that would yield substantial benefits for interpretability, control, and safety. 

---
# Systematic Evaluation of Machine-Generated Reasoning and PHQ-9 Labeling for Depression Detection Using Large Language Models 

**Authors**: Zongru Shao, Xin Wang, Zhanyang Liu, Chenhan Wang, K.P. Subbalakshmi  

**Link**: [PDF](https://arxiv.org/pdf/2505.17119)  

**Abstract**: Recent research leverages large language models (LLMs) for early mental health detection, such as depression, often optimized with machine-generated data. However, their detection may be subject to unknown weaknesses. Meanwhile, quality control has not been applied to these generated corpora besides limited human verifications. Our goal is to systematically evaluate LLM reasoning and reveal potential weaknesses. To this end, we first provide a systematic evaluation of the reasoning over machine-generated detection and interpretation. Then we use the models' reasoning abilities to explore mitigation strategies for enhanced performance. Specifically, we do the following: A. Design an LLM instruction strategy that allows for systematic analysis of the detection by breaking down the task into several subtasks. B. Design contrastive few-shot and chain-of-thought prompts by selecting typical positive and negative examples of detection reasoning. C. Perform human annotation for the subtasks identified in the first step and evaluate the performance. D. Identify human-preferred detection with desired logical reasoning from the few-shot generation and use them to explore different optimization strategies. We conducted extensive comparisons on the DepTweet dataset across the following subtasks: 1. identifying whether the speaker is describing their own depression; 2. accurately detecting the presence of PHQ-9 symptoms, and 3. finally, detecting depression. Human verification of statistical outliers shows that LLMs demonstrate greater accuracy in analyzing and detecting explicit language of depression as opposed to implicit expressions of depression. Two optimization methods are used for performance enhancement and reduction of the statistic bias: supervised fine-tuning (SFT) and direct preference optimization (DPO). Notably, the DPO approach achieves significant performance improvement. 

---
# After Retrieval, Before Generation: Enhancing the Trustworthiness of Large Language Models in RAG 

**Authors**: Xinbang Dai, Huikang Hu, Yuncheng Hua, Jiaqi Li, Yongrui Chen, Rihui Jin, Nan Hu, Guilin Qi  

**Link**: [PDF](https://arxiv.org/pdf/2505.17118)  

**Abstract**: Retrieval-augmented generation (RAG) systems face critical challenges in balancing internal (parametric) and external (retrieved) knowledge, especially when these sources conflict or are unreliable. To analyze these scenarios comprehensively, we construct the Trustworthiness Response Dataset (TRD) with 36,266 questions spanning four RAG settings. We reveal that existing approaches address isolated scenarios-prioritizing one knowledge source, naively merging both, or refusing answers-but lack a unified framework to handle different real-world conditions simultaneously. Therefore, we propose the BRIDGE framework, which dynamically determines a comprehensive response strategy of large language models (LLMs). BRIDGE leverages an adaptive weighting mechanism named soft bias to guide knowledge collection, followed by a Maximum Soft-bias Decision Tree to evaluate knowledge and select optimal response strategies (trust internal/external knowledge, or refuse). Experiments show BRIDGE outperforms baselines by 5-15% in accuracy while maintaining balanced performance across all scenarios. Our work provides an effective solution for LLMs' trustworthy responses in real-world RAG applications. 

---
# From Tokens to Thoughts: How LLMs and Humans Trade Compression for Meaning 

**Authors**: Chen Shani, Dan Jurafsky, Yann LeCun, Ravid Shwartz-Ziv  

**Link**: [PDF](https://arxiv.org/pdf/2505.17117)  

**Abstract**: Humans organize knowledge into compact categories through semantic compression by mapping diverse instances to abstract representations while preserving meaning (e.g., robin and blue jay are both birds; most birds can fly). These concepts reflect a trade-off between expressive fidelity and representational simplicity. Large Language Models (LLMs) demonstrate remarkable linguistic abilities, yet whether their internal representations strike a human-like trade-off between compression and semantic fidelity is unclear. We introduce a novel information-theoretic framework, drawing from Rate-Distortion Theory and the Information Bottleneck principle, to quantitatively compare these strategies. Analyzing token embeddings from a diverse suite of LLMs against seminal human categorization benchmarks, we uncover key divergences. While LLMs form broad conceptual categories that align with human judgment, they struggle to capture the fine-grained semantic distinctions crucial for human understanding. More fundamentally, LLMs demonstrate a strong bias towards aggressive statistical compression, whereas human conceptual systems appear to prioritize adaptive nuance and contextual richness, even if this results in lower compressional efficiency by our measures. These findings illuminate critical differences between current AI and human cognitive architectures, guiding pathways toward LLMs with more human-aligned conceptual representations. 

---
# Comparative Evaluation of Prompting and Fine-Tuning for Applying Large Language Models to Grid-Structured Geospatial Data 

**Authors**: Akash Dhruv, Yangxinyu Xie, Jordan Branham, Tanwi Mallick  

**Link**: [PDF](https://arxiv.org/pdf/2505.17116)  

**Abstract**: This paper presents a comparative study of large language models (LLMs) in interpreting grid-structured geospatial data. We evaluate the performance of a base model through structured prompting and contrast it with a fine-tuned variant trained on a dataset of user-assistant interactions. Our results highlight the strengths and limitations of zero-shot prompting and demonstrate the benefits of fine-tuning for structured geospatial and temporal reasoning. 

---
# RAVEN: Query-Guided Representation Alignment for Question Answering over Audio, Video, Embedded Sensors, and Natural Language 

**Authors**: Subrata Biswas, Mohammad Nur Hossain Khan, Bashima Islam  

**Link**: [PDF](https://arxiv.org/pdf/2505.17114)  

**Abstract**: Multimodal question answering (QA) often requires identifying which video, audio, or sensor tokens are relevant to the question. Yet modality disagreements are common: off-camera speech, background noise, or motion outside the field of view often mislead fusion models that weight all streams equally. We present RAVEN, a unified QA architecture whose core is QuART, a query-conditioned cross-modal gating module that assigns scalar relevance scores to each token across modalities, enabling the model to amplify informative signals and suppress distractors before fusion. RAVEN is trained through a three-stage pipeline comprising unimodal pretraining, query-aligned fusion, and disagreement-oriented fine-tuning -- each stage targeting a distinct challenge in multi-modal reasoning: representation quality, cross-modal relevance, and robustness to modality mismatch. To support training and evaluation, we release AVS-QA, a dataset of 300K synchronized Audio--Video-Sensor streams paired with automatically generated question-answer pairs. Experimental results on seven multi-modal QA benchmarks -- including egocentric and exocentric tasks -- show that RAVEN achieves up to 14.5\% and 8.0\% gains in accuracy compared to state-of-the-art multi-modal large language models, respectively. Incorporating sensor data provides an additional 16.4\% boost, and the model remains robust under modality corruption, outperforming SOTA baselines by 50.23\%. Our code and dataset are available at this https URL. 

---
# Cultural Value Alignment in Large Language Models: A Prompt-based Analysis of Schwartz Values in Gemini, ChatGPT, and DeepSeek 

**Authors**: Robin Segerer  

**Link**: [PDF](https://arxiv.org/pdf/2505.17112)  

**Abstract**: This study examines cultural value alignment in large language models (LLMs) by analyzing how Gemini, ChatGPT, and DeepSeek prioritize values from Schwartz's value framework. Using the 40-item Portrait Values Questionnaire, we assessed whether DeepSeek, trained on Chinese-language data, exhibits distinct value preferences compared to Western models. Results of a Bayesian ordinal regression model show that self-transcendence values (e.g., benevolence, universalism) were highly prioritized across all models, reflecting a general LLM tendency to emphasize prosocial values. However, DeepSeek uniquely downplayed self-enhancement values (e.g., power, achievement) compared to ChatGPT and Gemini, aligning with collectivist cultural tendencies. These findings suggest that LLMs reflect culturally situated biases rather than a universal ethical framework. To address value asymmetries in LLMs, we propose multi-perspective reasoning, self-reflective feedback, and dynamic contextualization. This study contributes to discussions on AI fairness, cultural neutrality, and the need for pluralistic AI alignment frameworks that integrate diverse moral perspectives. 

---
# Multi-Modality Expansion and Retention for LLMs through Parameter Merging and Decoupling 

**Authors**: Junlin Li, Guodong DU, Jing Li, Sim Kuan Goh, Wenya Wang, Yequan Wang, Fangming Liu, Ho-Kin Tang, Saleh Alharbi, Daojing He, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17110)  

**Abstract**: Fine-tuning Large Language Models (LLMs) with multimodal encoders on modality-specific data expands the modalities that LLMs can handle, leading to the formation of Multimodal LLMs (MLLMs). However, this paradigm heavily relies on resource-intensive and inflexible fine-tuning from scratch with new multimodal data. In this paper, we propose MMER (Multi-modality Expansion and Retention), a training-free approach that integrates existing MLLMs for effective multimodal expansion while retaining their original performance. Specifically, MMER reuses MLLMs' multimodal encoders while merging their LLM parameters. By comparing original and merged LLM parameters, MMER generates binary masks to approximately separate LLM parameters for each modality. These decoupled parameters can independently process modality-specific inputs, reducing parameter conflicts and preserving original MLLMs' fidelity. MMER can also mitigate catastrophic forgetting by applying a similar process to MLLMs fine-tuned on new tasks. Extensive experiments show significant improvements over baselines, proving that MMER effectively expands LLMs' multimodal capabilities while retaining 99% of the original performance, and also markedly mitigates catastrophic forgetting. 

---
# RRTL: Red Teaming Reasoning Large Language Models in Tool Learning 

**Authors**: Yifei Liu, Yu Cui, Haibin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17106)  

**Abstract**: While tool learning significantly enhances the capabilities of large language models (LLMs), it also introduces substantial security risks. Prior research has revealed various vulnerabilities in traditional LLMs during tool learning. However, the safety of newly emerging reasoning LLMs (RLLMs), such as DeepSeek-R1, in the context of tool learning remains underexplored. To bridge this gap, we propose RRTL, a red teaming approach specifically designed to evaluate RLLMs in tool learning. It integrates two novel strategies: (1) the identification of deceptive threats, which evaluates the model's behavior in concealing the usage of unsafe tools and their potential risks; and (2) the use of Chain-of-Thought (CoT) prompting to force tool invocation. Our approach also includes a benchmark for traditional LLMs. We conduct a comprehensive evaluation on seven mainstream RLLMs and uncover three key findings: (1) RLLMs generally achieve stronger safety performance than traditional LLMs, yet substantial safety disparities persist across models; (2) RLLMs can pose serious deceptive risks by frequently failing to disclose tool usage and to warn users of potential tool output risks; (3) CoT prompting reveals multi-lingual safety vulnerabilities in RLLMs. Our work provides important insights into enhancing the security of RLLMs in tool learning. 

---
# P2P: Automated Paper-to-Poster Generation and Fine-Grained Benchmark 

**Authors**: Tao Sun, Enhao Pan, Zhengkai Yang, Kaixin Sui, Jiajun Shi, Xianfu Cheng, Tongliang Li, Wenhao Huang, Ge Zhang, Jian Yang, Zhoujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.17104)  

**Abstract**: Academic posters are vital for scholarly communication, yet their manual creation is time-consuming. However, automated academic poster generation faces significant challenges in preserving intricate scientific details and achieving effective visual-textual integration. Existing approaches often struggle with semantic richness and structural nuances, and lack standardized benchmarks for evaluating generated academic posters comprehensively. To address these limitations, we introduce P2P, the first flexible, LLM-based multi-agent framework that generates high-quality, HTML-rendered academic posters directly from research papers, demonstrating strong potential for practical applications. P2P employs three specialized agents-for visual element processing, content generation, and final poster assembly-each integrated with dedicated checker modules to enable iterative refinement and ensure output quality. To foster advancements and rigorous evaluation in this domain, we construct and release P2PInstruct, the first large-scale instruction dataset comprising over 30,000 high-quality examples tailored for the academic paper-to-poster generation task. Furthermore, we establish P2PEval, a comprehensive benchmark featuring 121 paper-poster pairs and a dual evaluation methodology (Universal and Fine-Grained) that leverages LLM-as-a-Judge and detailed, human-annotated checklists. Our contributions aim to streamline research dissemination and provide the community with robust tools for developing and evaluating next-generation poster generation systems. 

---
# Forging Time Series with Language: A Large Language Model Approach to Synthetic Data Generation 

**Authors**: Cécile Rousseau, Tobia Boschi, Giandomenico Cornacchia, Dhaval Salwala, Alessandra Pascale, Juan Bernabe Moreno  

**Link**: [PDF](https://arxiv.org/pdf/2505.17103)  

**Abstract**: SDForger is a flexible and efficient framework for generating high-quality multivariate time series using LLMs. Leveraging a compact data representation, SDForger provides synthetic time series generation from a few samples and low-computation fine-tuning of any autoregressive LLM. Specifically, the framework transforms univariate and multivariate signals into tabular embeddings, which are then encoded into text and used to fine-tune the LLM. At inference, new textual embeddings are sampled and decoded into synthetic time series that retain the original data's statistical properties and temporal dynamics. Across a diverse range of datasets, SDForger outperforms existing generative models in many scenarios, both in similarity-based evaluations and downstream forecasting tasks. By enabling textual conditioning in the generation process, SDForger paves the way for multimodal modeling and the streamlined integration of time series with textual information. SDForger source code will be open-sourced soon. 

---
# BanglaByT5: Byte-Level Modelling for Bangla 

**Authors**: Pramit Bhattacharyya, Arnab Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2505.17102)  

**Abstract**: Large language models (LLMs) have achieved remarkable success across various natural language processing tasks. However, most LLM models use traditional tokenizers like BPE and SentencePiece, which fail to capture the finer nuances of a morphologically rich language like Bangla (Bengali). In this work, we introduce BanglaByT5, the first byte-level encoder-decoder model explicitly tailored for Bangla. Built upon a small variant of Googles ByT5 architecture, BanglaByT5 is pre-trained on a 14GB curated corpus combining high-quality literary and newspaper articles. Through zeroshot and supervised evaluations across generative and classification tasks, BanglaByT5 demonstrates competitive performance, surpassing several multilingual and larger models. Our findings highlight the efficacy of byte-level modelling for morphologically rich languages and highlight BanglaByT5 potential as a lightweight yet powerful tool for Bangla NLP, particularly in both resource-constrained and scalable environments. 

---
# An approach to identify the most semantically informative deep representations of text and images 

**Authors**: Santiago Acevedo, Andrea Mascaretti, Riccardo Rende, Matéo Mahaut, Marco Baroni, Alessandro Laio  

**Link**: [PDF](https://arxiv.org/pdf/2505.17101)  

**Abstract**: Deep neural networks are known to develop similar representations for semantically related data, even when they belong to different domains, such as an image and its description, or the same text in different languages. We present a method for quantitatively investigating this phenomenon by measuring the relative information content of the representations of semantically related data and probing how it is encoded into multiple tokens of large language models (LLMs) and vision transformers. Looking first at how LLMs process pairs of translated sentences, we identify inner ``semantic'' layers containing the most language-transferable information. We find moreover that, on these layers, a larger LLM (DeepSeek-V3) extracts significantly more general information than a smaller one (Llama3.1-8B). Semantic information is spread across many tokens and it is characterized by long-distance correlations between tokens and by a causal left-to-right (i.e., past-future) asymmetry. We also identify layers encoding semantic information within visual transformers. We show that caption representations in the semantic layers of LLMs predict visual representations of the corresponding images. We observe significant and model-dependent information asymmetries between image and text representations. 

---
# Any Large Language Model Can Be a Reliable Judge: Debiasing with a Reasoning-based Bias Detector 

**Authors**: Haoyan Yang, Runxue Bao, Cao Xiao, Jun Ma, Parminder Bhatia, Shangqian Gao, Taha Kass-Hout  

**Link**: [PDF](https://arxiv.org/pdf/2505.17100)  

**Abstract**: LLM-as-a-Judge has emerged as a promising tool for automatically evaluating generated outputs, but its reliability is often undermined by potential biases in judgment. Existing efforts to mitigate these biases face key limitations: in-context learning-based methods fail to address rooted biases due to the evaluator's limited capacity for self-reflection, whereas fine-tuning is not applicable to all evaluator types, especially closed-source models. To address this challenge, we introduce the Reasoning-based Bias Detector (RBD), which is a plug-in module that identifies biased evaluations and generates structured reasoning to guide evaluator self-correction. Rather than modifying the evaluator itself, RBD operates externally and engages in an iterative process of bias detection and feedback-driven revision. To support its development, we design a complete pipeline consisting of biased dataset construction, supervision collection, distilled reasoning-based fine-tuning of RBD, and integration with LLM evaluators. We fine-tune four sizes of RBD models, ranging from 1.5B to 14B, and observe consistent performance improvements across all scales. Experimental results on 4 bias types--verbosity, position, bandwagon, and sentiment--evaluated using 8 LLM evaluators demonstrate RBD's strong effectiveness. For example, the RBD-8B model improves evaluation accuracy by an average of 18.5% and consistency by 10.9%, and surpasses prompting-based baselines and fine-tuned judges by 12.8% and 17.2%, respectively. These results highlight RBD's effectiveness and scalability. Additional experiments further demonstrate its strong generalization across biases and domains, as well as its efficiency. 

---
# Learning Interpretable Representations Leads to Semantically Faithful EEG-to-Text Generation 

**Authors**: Xiaozhao Liu, Dinggang Shen, Xihui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17099)  

**Abstract**: Pretrained generative models have opened new frontiers in brain decoding by enabling the synthesis of realistic texts and images from non-invasive brain recordings. However, the reliability of such outputs remains questionable--whether they truly reflect semantic activation in the brain, or are merely hallucinated by the powerful generative models. In this paper, we focus on EEG-to-text decoding and address its hallucination issue through the lens of posterior collapse. Acknowledging the underlying mismatch in information capacity between EEG and text, we reframe the decoding task as semantic summarization of core meanings rather than previously verbatim reconstruction of stimulus texts. To this end, we propose the Generative Language Inspection Model (GLIM), which emphasizes learning informative and interpretable EEG representations to improve semantic grounding under heterogeneous and small-scale data conditions. Experiments on the public ZuCo dataset demonstrate that GLIM consistently generates fluent, EEG-grounded sentences without teacher forcing. Moreover, it supports more robust evaluation beyond text similarity, through EEG-text retrieval and zero-shot semantic classification across sentiment categories, relation types, and corpus topics. Together, our architecture and evaluation protocols lay the foundation for reliable and scalable benchmarking in generative brain decoding. 

---
# TACO: Enhancing Multimodal In-context Learning via Task Mapping-Guided Sequence Configuration 

**Authors**: Yanshu Li, Tian Yun, Jianjiang Yang, Pinyuan Feng, Jinfa Huang, Ruixiang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17098)  

**Abstract**: Multimodal in-context learning (ICL) has emerged as a key mechanism for harnessing the capabilities of large vision-language models (LVLMs). However, its effectiveness remains highly sensitive to the quality of input in-context sequences, particularly for tasks involving complex reasoning or open-ended generation. A major limitation is our limited understanding of how LVLMs actually exploit these sequences during inference. To bridge this gap, we systematically interpret multimodal ICL through the lens of task mapping, which reveals how local and global relationships within and among demonstrations guide model reasoning. Building on this insight, we present TACO, a lightweight transformer-based model equipped with task-aware attention that dynamically configures in-context sequences. By injecting task-mapping signals into the autoregressive decoding process, TACO creates a bidirectional synergy between sequence construction and task reasoning. Experiments on five LVLMs and nine datasets demonstrate that TACO consistently surpasses baselines across diverse ICL tasks. These results position task mapping as a valuable perspective for interpreting and improving multimodal ICL. 

---
# Are LLMs reliable? An exploration of the reliability of large language models in clinical note generation 

**Authors**: Kristine Ann M. Carandang, Jasper Meynard P. Araña, Ethan Robert A. Casin, Christopher P. Monterola, Daniel Stanley Y. Tan, Jesus Felix B. Valenzuela, Christian M. Alis  

**Link**: [PDF](https://arxiv.org/pdf/2505.17095)  

**Abstract**: Due to the legal and ethical responsibilities of healthcare providers (HCPs) for accurate documentation and protection of patient data privacy, the natural variability in the responses of large language models (LLMs) presents challenges for incorporating clinical note generation (CNG) systems, driven by LLMs, into real-world clinical processes. The complexity is further amplified by the detailed nature of texts in CNG. To enhance the confidence of HCPs in tools powered by LLMs, this study evaluates the reliability of 12 open-weight and proprietary LLMs from Anthropic, Meta, Mistral, and OpenAI in CNG in terms of their ability to generate notes that are string equivalent (consistency rate), have the same meaning (semantic consistency) and are correct (semantic similarity), across several iterations using the same prompt. The results show that (1) LLMs from all model families are stable, such that their responses are semantically consistent despite being written in various ways, and (2) most of the LLMs generated notes close to the corresponding notes made by experts. Overall, Meta's Llama 70B was the most reliable, followed by Mistral's Small model. With these findings, we recommend the local deployment of these relatively smaller open-weight models for CNG to ensure compliance with data privacy regulations, as well as to improve the efficiency of HCPs in clinical documentation. 

---
# Large Language Models Implicitly Learn to See and Hear Just By Reading 

**Authors**: Prateek Verma, Mert Pilanci  

**Link**: [PDF](https://arxiv.org/pdf/2505.17091)  

**Abstract**: This paper presents a fascinating find: By training an auto-regressive LLM model on text tokens, the text model inherently develops internally an ability to understand images and audio, thereby developing the ability to see and hear just by reading. Popular audio and visual LLM models fine-tune text LLM models to give text output conditioned on images and audio embeddings. On the other hand, our architecture takes in patches of images, audio waveforms or tokens as input. It gives us the embeddings or category labels typical of a classification pipeline. We show the generality of text weights in aiding audio classification for datasets FSD-50K and GTZAN. Further, we show this working for image classification on CIFAR-10 and Fashion-MNIST, as well on image patches. This pushes the notion of text-LLMs learning powerful internal circuits that can be utilized by activating necessary connections for various applications rather than training models from scratch every single time. 

---
# Trust Me, I Can Handle It: Self-Generated Adversarial Scenario Extrapolation for Robust Language Models 

**Authors**: Md Rafi Ur Rashid, Vishnu Asutosh Dasu, Ye Wang, Gang Tan, Shagufta Mehnaz  

**Link**: [PDF](https://arxiv.org/pdf/2505.17089)  

**Abstract**: Large Language Models (LLMs) exhibit impressive capabilities, but remain susceptible to a growing spectrum of safety risks, including jailbreaks, toxic content, hallucinations, and bias. Existing defenses often address only a single threat type or resort to rigid outright rejection, sacrificing user experience and failing to generalize across diverse and novel attacks. This paper introduces Adversarial Scenario Extrapolation (ASE), a novel inference-time computation framework that leverages Chain-of-Thought (CoT) reasoning to simultaneously enhance LLM robustness and seamlessness. ASE guides the LLM through a self-generative process of contemplating potential adversarial scenarios and formulating defensive strategies before generating a response to the user query. Comprehensive evaluation on four adversarial benchmarks with four latest LLMs shows that ASE achieves near-zero jailbreak attack success rates and minimal toxicity, while slashing outright rejections to <4%. ASE outperforms six state-of-the-art defenses in robustness-seamlessness trade-offs, with 92-99% accuracy on adversarial Q&A and 4-10x lower bias scores. By transforming adversarial perception into an intrinsic cognitive process, ASE sets a new paradigm for secure and natural human-AI interaction. 

---
# Informatics for Food Processing 

**Authors**: Gordana Ispirova, Michael Sebek, Giulia Menichetti  

**Link**: [PDF](https://arxiv.org/pdf/2505.17087)  

**Abstract**: This chapter explores the evolution, classification, and health implications of food processing, while emphasizing the transformative role of machine learning, artificial intelligence (AI), and data science in advancing food informatics. It begins with a historical overview and a critical review of traditional classification frameworks such as NOVA, Nutri-Score, and SIGA, highlighting their strengths and limitations, particularly the subjectivity and reproducibility challenges that hinder epidemiological research and public policy. To address these issues, the chapter presents novel computational approaches, including FoodProX, a random forest model trained on nutrient composition data to infer processing levels and generate a continuous FPro score. It also explores how large language models like BERT and BioBERT can semantically embed food descriptions and ingredient lists for predictive tasks, even in the presence of missing data. A key contribution of the chapter is a novel case study using the Open Food Facts database, showcasing how multimodal AI models can integrate structured and unstructured data to classify foods at scale, offering a new paradigm for food processing assessment in public health and research. 

---
# Reinforcing Question Answering Agents with Minimalist Policy Gradient Optimization 

**Authors**: Yihong Wu, Liheng Ma, Muzhi Li, Jiaming Zhou, Jianye Hao, Ho-fung Leung, Irwin King, Yingxue Zhang, Jian-Yun Nie  

**Link**: [PDF](https://arxiv.org/pdf/2505.17086)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable versatility, due to the lack of factual knowledge, their application to Question Answering (QA) tasks remains hindered by hallucination.
While Retrieval-Augmented Generation mitigates these issues by integrating external knowledge, existing approaches rely heavily on in-context learning, whose performance is constrained by the fundamental reasoning capabilities of LLMs.
In this paper, we propose Mujica, a Multi-hop Joint Intelligence for Complex Question Answering, comprising a planner that decomposes questions into a directed acyclic graph of subquestions and a worker that resolves questions via retrieval and reasoning. Additionally, we introduce MyGO (Minimalist policy Gradient Optimization), a novel reinforcement learning method that replaces traditional policy gradient updates with Maximum Likelihood Estimation (MLE) by sampling trajectories from an asymptotically optimal policy. MyGO eliminates the need for gradient rescaling and reference models, ensuring stable and efficient training.
Empirical results across multiple datasets demonstrate the effectiveness of Mujica-MyGO in enhancing multi-hop QA performance for various LLMs, offering a scalable and resource-efficient solution for complex QA tasks. 

---
# Scale-invariant Attention 

**Authors**: Ben Anson, Xi Wang, Laurence Aitchison  

**Link**: [PDF](https://arxiv.org/pdf/2505.17083)  

**Abstract**: One persistent challenge in LLM research is the development of attention mechanisms that are able to generalise from training on shorter contexts to inference on longer contexts. We propose two conditions that we expect all effective long context attention mechanisms to have: scale-invariant total attention, and scale-invariant attention sparsity. Under a Gaussian assumption, we show that a simple position-dependent transformation of the attention logits is sufficient for these conditions to hold. Experimentally we find that the resulting scale-invariant attention scheme gives considerable benefits in terms of validation loss when zero-shot generalising from training on short contexts to validation on longer contexts, and is effective at long-context retrieval. 

---
# GemMaroc: Unlocking Darija Proficiency in LLMs with Minimal Data 

**Authors**: Abderrahman Skiredj, Ferdaous Azhari, Houdaifa Atou, Nouamane Tazi, Ismail Berrada  

**Link**: [PDF](https://arxiv.org/pdf/2505.17082)  

**Abstract**: Open-source large language models (LLMs) still marginalise Moroccan Arabic (Darija), forcing practitioners either to bolt on heavyweight Arabic adapters or to sacrifice the very reasoning skills that make LLMs useful. We show that a rigorously quality-over-quantity alignment strategy can surface fluent Darija while safeguarding the backbone s cross-lingual reasoning at a sliver of the usual compute. We translate three compact instruction suites LIMA 1 K, DEITA 6 K and TULU 50 K into Darija, preserve 20 of the English originals, and add mathematics, coding and scientific prompts. A LoRA-tuned Gemma 3-4B trained on 5 K mixed instructions lifts DarijaMMLU from 32.8 to 42.7 ; adding the reasoning-dense TULU portion pushes it to 47.5 with no English regression. Scaling the identical recipe to Gemma 3-27B produces GemMaroc-27B, which matches Atlas-Chat on DarijaMMLU (61.6 ) and leaps ahead on Darija commonsense, scoring 60.5 on HellaSwag versus Atlas-Chat s 48.4 . Crucially, GemMaroc retains Gemma-27B s strong maths and general-reasoning ability, showing only minimal movement on GSM8K and English benchmarks. The entire model is trained in just 48 GPU.h, underscoring a Green AI pathway to inclusive, sustainable language technology. We release code, data and checkpoints to spur Darija-centric applications in education, public services and everyday digital interaction. 

---
# Not Minds, but Signs: Reframing LLMs through Semiotics 

**Authors**: Davide Picca  

**Link**: [PDF](https://arxiv.org/pdf/2505.17080)  

**Abstract**: This paper challenges the prevailing tendency to frame Large Language Models (LLMs) as cognitive systems, arguing instead for a semiotic perspective that situates these models within the broader dynamics of sign manipulation and meaning-making. Rather than assuming that LLMs understand language or simulate human thought, we propose that their primary function is to recombine, recontextualize, and circulate linguistic forms based on probabilistic associations. By shifting from a cognitivist to a semiotic framework, we avoid anthropomorphism and gain a more precise understanding of how LLMs participate in cultural processes, not by thinking, but by generating texts that invite interpretation. Through theoretical analysis and practical examples, the paper demonstrates how LLMs function as semiotic agents whose outputs can be treated as interpretive acts, open to contextual negotiation and critical reflection. We explore applications in literature, philosophy, education, and cultural production, emphasizing how LLMs can serve as tools for creativity, dialogue, and critical inquiry. The semiotic paradigm foregrounds the situated, contingent, and socially embedded nature of meaning, offering a more rigorous and ethically aware framework for studying and using LLMs. Ultimately, this approach reframes LLMs as technological participants in an ongoing ecology of signs. They do not possess minds, but they alter how we read, write, and make meaning, compelling us to reconsider the foundations of language, interpretation, and the role of artificial systems in the production of knowledge. 

---
# GloSS over Toxicity: Understanding and Mitigating Toxicity in LLMs via Global Toxic Subspace 

**Authors**: Zenghao Duan, Zhiyi Yin, Zhichao Shi, Liang Pang, Shaoling Jing, Jiayi Wu, Yu Yan, Huawei Shen, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.17078)  

**Abstract**: This paper investigates the underlying mechanisms of toxicity generation in Large Language Models (LLMs) and proposes an effective detoxification approach. Prior work typically considers the Feed-Forward Network (FFN) as the main source of toxicity, representing toxic regions as a set of toxic vectors or layer-wise subspaces. However, our in-depth analysis reveals that the global toxic subspace offers a more effective and comprehensive representation of toxic region within the model. Building on this insight, we propose GloSS (Global Toxic Subspace Suppression), a lightweight, four-stage method that mitigates toxicity by identifying and removing the global toxic subspace from the parameters of FFN. Experiments across a range of LLMs show that GloSS achieves state-of-the-art detoxification performance while preserving the models general capabilities, without requiring large-scale data or model retraining. 

---
# Impact of Frame Rates on Speech Tokenizer: A Case Study on Mandarin and English 

**Authors**: Haoyang Zhang, Hexin Liu, Xiangyu Zhang, Qiquan Zhang, Yuchen Hu, Junqi Zhao, Fei Tian, Xuerui Yang, Eng Siong Chng  

**Link**: [PDF](https://arxiv.org/pdf/2505.17076)  

**Abstract**: The speech tokenizer plays a crucial role in recent speech tasks, generally serving as a bridge between speech signals and language models. While low-frame-rate codecs are widely employed as speech tokenizers, the impact of frame rates on speech tokens remains underexplored. In this study, we investigate how varying frame rates affect speech tokenization by examining Mandarin and English, two typologically distinct languages. We encode speech at different frame rates and evaluate the resulting semantic tokens in the speech recognition task. Our findings reveal that frame rate variations influence speech tokenization differently for each language, highlighting the interplay between frame rates, phonetic density, and language-specific acoustic features. The results provide insights into optimizing frame rate selection for speech tokenizers, with implications for automatic speech recognition, text-to-speech, and other speech-related applications. 

---
# Development and Validation of Engagement and Rapport Scales for Evaluating User Experience in Multimodal Dialogue Systems 

**Authors**: Fuma Kurata, Mao Saeki, Masaki Eguchi, Shungo Suzuki, Hiroaki Takatsu, Yoichi Matsuyama  

**Link**: [PDF](https://arxiv.org/pdf/2505.17075)  

**Abstract**: This study aimed to develop and validate two scales of engagement and rapport to evaluate the user experience quality with multimodal dialogue systems in the context of foreign language learning. The scales were designed based on theories of engagement in educational psychology, social psychology, and second language this http URL-four Japanese learners of English completed roleplay and discussion tasks with trained human tutors and a dialog agent. After each dialogic task was completed, they responded to the scales of engagement and rapport. The validity and reliability of the scales were investigated through two analyses. We first conducted analysis of Cronbach's alpha coefficient and a series of confirmatory factor analyses to test the structural validity of the scales and the reliability of our designed items. We then compared the scores of engagement and rapport between the dialogue with human tutors and the one with a dialogue agent. The results revealed that our scales succeeded in capturing the difference in the dialogue experience quality between the human interlocutors and the dialogue agent from multiple perspectives. 

---
# Semi-Clairvoyant Scheduling of Speculative Decoding Requests to Minimize LLM Inference Latency 

**Authors**: Ruixiao Li, Fahao Chen, Peng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.17074)  

**Abstract**: Speculative decoding accelerates Large Language Model (LLM) inference by employing a small speculative model (SSM) to generate multiple candidate tokens and verify them using the LLM in parallel. This technique has been widely integrated into LLM inference serving systems. However, inference requests typically exhibit uncertain execution time, which poses a significant challenge of efficiently scheduling requests in these systems. Existing work estimates execution time based solely on predicted output length, which could be inaccurate because execution time depends on both output length and token acceptance rate of verification by the LLM. In this paper, we propose a semi-clairvoyant request scheduling algorithm called Least-Attained/Perceived-Service for Speculative Decoding (LAPS-SD). Given a number of inference requests, LAPS-SD can effectively minimize average inference latency by adaptively scheduling requests according to their features during decoding. When the token acceptance rate is dynamic and execution time is difficult to estimate, LAPS-SD maintains multiple priority queues and allows request execution preemption across different queues. Once the token acceptance rate becomes stable, LAPS-SD can accurately estimate the execution time and schedule requests accordingly. Extensive experiments show that LAPS-SD reduces inference latency by approximately 39\% compared to state-of-the-art scheduling methods. 

---
# Mechanistic Interpretability of GPT-like Models on Summarization Tasks 

**Authors**: Anurag Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2505.17073)  

**Abstract**: Mechanistic interpretability research seeks to reveal the inner workings of large language models, yet most work focuses on classification or generative tasks rather than summarization. This paper presents an interpretability framework for analyzing how GPT-like models adapt to summarization tasks. We conduct differential analysis between pre-trained and fine-tuned models, quantifying changes in attention patterns and internal activations. By identifying specific layers and attention heads that undergo significant transformation, we locate the "summarization circuit" within the model architecture. Our findings reveal that middle layers (particularly 2, 3, and 5) exhibit the most dramatic changes, with 62% of attention heads showing decreased entropy, indicating a shift toward focused information selection. We demonstrate that targeted LoRA adaptation of these identified circuits achieves significant performance improvement over standard LoRA fine-tuning while requiring fewer training epochs. This work bridges the gap between black-box evaluation and mechanistic understanding, providing insights into how neural networks perform information selection and compression during summarization. 

---
# What's in a prompt? Language models encode literary style in prompt embeddings 

**Authors**: Raphaël Sarfati, Haley Moller, Toni J. B. Liu, Nicolas Boullé, Christopher Earls  

**Link**: [PDF](https://arxiv.org/pdf/2505.17071)  

**Abstract**: Large language models use high-dimensional latent spaces to encode and process textual information. Much work has investigated how the conceptual content of words translates into geometrical relationships between their vector representations. Fewer studies analyze how the cumulative information of an entire prompt becomes condensed into individual embeddings under the action of transformer layers. We use literary pieces to show that information about intangible, rather than factual, aspects of the prompt are contained in deep representations. We observe that short excerpts (10 - 100 tokens) from different novels separate in the latent space independently from what next-token prediction they converge towards. Ensembles from books from the same authors are much more entangled than across authors, suggesting that embeddings encode stylistic features. This geometry of style may have applications for authorship attribution and literary analysis, but most importantly reveals the sophistication of information processing and compression accomplished by language models. 

---
# Improving endpoint detection in end-to-end streaming ASR for conversational speech 

**Authors**: Anandh C, Karthik Pandia Durai, Jeena Prakash, Manickavela Arumugam, Kadri Hacioglu, S.Pavankumar Dubagunta, Andreas Stolcke, Shankar Venkatesan, Aravind Ganapathiraju  

**Link**: [PDF](https://arxiv.org/pdf/2505.17070)  

**Abstract**: ASR endpointing (EP) plays a major role in delivering a good user experience in products supporting human or artificial agents in human-human/machine conversations. Transducer-based ASR (T-ASR) is an end-to-end (E2E) ASR modelling technique preferred for streaming. A major limitation of T-ASR is delayed emission of ASR outputs, which could lead to errors or delays in EP. Inaccurate EP will cut the user off while speaking, returning incomplete transcript while delays in EP will increase the perceived latency, degrading the user experience. We propose methods to improve EP by addressing delayed emission along with EP mistakes. To address the delayed emission problem, we introduce an end-of-word token at the end of each word, along with a delay penalty. The EP delay is addressed by obtaining a reliable frame-level speech activity detection using an auxiliary network. We apply the proposed methods on Switchboard conversational speech corpus and evaluate it against a delay penalty method. 

---
# Predictively Combatting Toxicity in Health-related Online Discussions through Machine Learning 

**Authors**: Jorge Paz-Ruza, Amparo Alonso-Betanzos, Bertha Guijarro-Berdiñas, Carlos Eiras-Franco  

**Link**: [PDF](https://arxiv.org/pdf/2505.17068)  

**Abstract**: In health-related topics, user toxicity in online discussions frequently becomes a source of social conflict or promotion of dangerous, unscientific behaviour; common approaches for battling it include different forms of detection, flagging and/or removal of existing toxic comments, which is often counterproductive for platforms and users alike. In this work, we propose the alternative of combatting user toxicity predictively, anticipating where a user could interact toxically in health-related online discussions. Applying a Collaborative Filtering-based Machine Learning methodology, we predict the toxicity in COVID-related conversations between any user and subcommunity of Reddit, surpassing 80% predictive performance in relevant metrics, and allowing us to prevent the pairing of conflicting users and subcommunities. 

---
# Unveil Multi-Picture Descriptions for Multilingual Mild Cognitive Impairment Detection via Contrastive Learning 

**Authors**: Kristin Qi, Jiali Cheng, Youxiang Zhu, Hadi Amiri, Xiaohui Liang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17067)  

**Abstract**: Detecting Mild Cognitive Impairment from picture descriptions is critical yet challenging, especially in multilingual and multiple picture settings. Prior work has primarily focused on English speakers describing a single picture (e.g., the 'Cookie Theft'). The TAUKDIAL-2024 challenge expands this scope by introducing multilingual speakers and multiple pictures, which presents new challenges in analyzing picture-dependent content. To address these challenges, we propose a framework with three components: (1) enhancing discriminative representation learning via supervised contrastive learning, (2) involving image modality rather than relying solely on speech and text modalities, and (3) applying a Product of Experts (PoE) strategy to mitigate spurious correlations and overfitting. Our framework improves MCI detection performance, achieving a +7.1% increase in Unweighted Average Recall (UAR) (from 68.1% to 75.2%) and a +2.9% increase in F1 score (from 80.6% to 83.5%) compared to the text unimodal baseline. Notably, the contrastive learning component yields greater gains for the text modality compared to speech. These results highlight our framework's effectiveness in multilingual and multi-picture MCI detection. 

---
# Decoding Rarity: Large Language Models in the Diagnosis of Rare Diseases 

**Authors**: Valentina Carbonari, Pierangelo Veltri, Pietro Hiram Guzzi  

**Link**: [PDF](https://arxiv.org/pdf/2505.17065)  

**Abstract**: Recent advances in artificial intelligence, particularly large language models LLMs, have shown promising capabilities in transforming rare disease research. This survey paper explores the integration of LLMs in the analysis of rare diseases, highlighting significant strides and pivotal studies that leverage textual data to uncover insights and patterns critical for diagnosis, treatment, and patient care. While current research predominantly employs textual data, the potential for multimodal data integration combining genetic, imaging, and electronic health records stands as a promising frontier. We review foundational papers that demonstrate the application of LLMs in identifying and extracting relevant medical information, simulating intelligent conversational agents for patient interaction, and enabling the formulation of accurate and timely diagnoses. Furthermore, this paper discusses the challenges and ethical considerations inherent in deploying LLMs, including data privacy, model transparency, and the need for robust, inclusive data sets. As part of this exploration, we present a section on experimentation that utilizes multiple LLMs alongside structured questionnaires, specifically designed for diagnostic purposes in the context of different diseases. We conclude with future perspectives on the evolution of LLMs towards truly multimodal platforms, which would integrate diverse data types to provide a more comprehensive understanding of rare diseases, ultimately fostering better outcomes in clinical settings. 

---
# Synthetic Data RL: Task Definition Is All You Need 

**Authors**: Yiduo Guo, Zhen Guo, Chuanwei Huang, Zi-Ang Wang, Zekai Zhang, Haofei Yu, Huishuai Zhang, Yikang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.17063)  

**Abstract**: Reinforcement learning (RL) is a powerful way to adapt foundation models to specialized tasks, but its reliance on large-scale human-labeled data limits broad adoption. We introduce Synthetic Data RL, a simple and general framework that reinforcement fine-tunes models using only synthetic data generated from a task definition. Our method first generates question and answer pairs from the task definition and retrieved documents, then adapts the difficulty of the question based on model solvability, and selects questions using the average pass rate of the model across samples for RL training. On Qwen-2.5-7B, our method achieves a 29.2% absolute improvement over the base model on GSM8K (+2.9 pp vs. instruction-tuned, +6.6 pp vs. Self-Instruct), 8.7% on MATH, 13.1% on GPQA (+7.0 pp vs. SynthLLM), 8.9% on MedQA, 17.7% on CQA (law) and 13.7% on CFA (finance). It surpasses supervised fine-tuning under the same data budget and nearly matches RL with full human data across datasets (e.g., +17.2 pp on GSM8K). Adding 100 human demonstrations improves the performance of GSM8K only by 0.4 pp, showing a limited added value. By reducing human data annotation, Synthetic Data RL enables scalable and efficient RL-based model adaptation. Code and demos are available at this https URL. 

---
# Mixture of Decoding: An Attention-Inspired Adaptive Decoding Strategy to Mitigate Hallucinations in Large Vision-Language Models 

**Authors**: Xinlong Chen, Yuanxing Zhang, Qiang Liu, Junfei Wu, Fuzheng Zhang, Tieniu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.17061)  

**Abstract**: Large Vision-Language Models (LVLMs) have exhibited impressive capabilities across various visual tasks, yet they remain hindered by the persistent challenge of hallucinations. To address this critical issue, we propose Mixture of Decoding (MoD), a novel approach for hallucination mitigation that dynamically adapts decoding strategies by evaluating the correctness of the model's attention on image tokens. Specifically, MoD measures the consistency between outputs generated from the original image tokens and those derived from the model's attended image tokens, to distinguish the correctness aforementioned. If the outputs are consistent, indicating correct attention, MoD employs a complementary strategy to amplify critical information. Conversely, if the outputs are inconsistent, suggesting erroneous attention, MoD utilizes a contrastive strategy to suppress misleading information. Extensive experiments demonstrate that MoD significantly outperforms existing decoding methods across multiple mainstream benchmarks, effectively mitigating hallucinations in LVLMs. The code is available at this https URL. 

---
# SALMONN-omni: A Standalone Speech LLM without Codec Injection for Full-duplex Conversation 

**Authors**: Wenyi Yu, Siyin Wang, Xiaoyu Yang, Xianzhao Chen, Xiaohai Tian, Jun Zhang, Guangzhi Sun, Lu Lu, Yuxuan Wang, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17060)  

**Abstract**: In order to enable fluid and natural human-machine speech interaction, existing full-duplex conversational systems often adopt modular architectures with auxiliary components such as voice activity detectors, interrupters, conversation state predictors, or multiple LLMs. These systems, however, suffer from error accumulation across modules and struggle with key challenges such as context-dependent barge-in and echo cancellation. Recent approaches, most notably Moshi, simplify the pipeline by injecting audio codecs into the token space of a single LLM. However, such methods still incur significant performance degradation when operating on the speech rather than text modality. In this paper, we introduce SALMONN-omni, the first single, standalone full-duplex speech LLM that operates without audio codecs in its token space. It features a novel dynamic thinking mechanism within the LLM backbone, enabling the model to learn when to transition between speaking and listening states. Experiments on widely used benchmarks for spoken question answering and open-domain dialogue show that SALMONN-omni achieves at least 30\% relative performance improvement over existing open-source full-duplex models and performs highly competitively to half-duplex and turn-based systems, despite using substantially less training data. Moreover, SALMONN-omni demonstrates strong performance in complex conversational scenarios, including turn-taking, backchanneling, echo cancellation and context-dependent barge-in, with further improvements achieved through reinforcement learning. Some demo conversations between user and SALMONN-omni are provided in the following repository this https URL. 

---
# Medalyze: Lightweight Medical Report Summarization Application Using FLAN-T5-Large 

**Authors**: Van-Tinh Nguyen, Hoang-Duong Pham, Thanh-Hai To, Cong-Tuan Hung Do, Thi-Thu-Trang Dong, Vu-Trung Duong Le, Van-Phuc Hoang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17059)  

**Abstract**: Understanding medical texts presents significant challenges due to complex terminology and context-specific language. This paper introduces Medalyze, an AI-powered application designed to enhance the comprehension of medical texts using three specialized FLAN-T5-Large models. These models are fine-tuned for (1) summarizing medical reports, (2) extracting health issues from patient-doctor conversations, and (3) identifying the key question in a passage. Medalyze is deployed across a web and mobile platform with real-time inference, leveraging scalable API and YugabyteDB. Experimental evaluations demonstrate the system's superior summarization performance over GPT-4 in domain-specific tasks, based on metrics like BLEU, ROUGE-L, BERTScore, and SpaCy Similarity. Medalyze provides a practical, privacy-preserving, and lightweight solution for improving information accessibility in healthcare. 

---
# DO-RAG: A Domain-Specific QA Framework Using Knowledge Graph-Enhanced Retrieval-Augmented Generation 

**Authors**: David Osei Opoku, Ming Sheng, Yong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17058)  

**Abstract**: Domain-specific QA systems require not just generative fluency but high factual accuracy grounded in structured expert knowledge. While recent Retrieval-Augmented Generation (RAG) frameworks improve context recall, they struggle with integrating heterogeneous data and maintaining reasoning consistency. To address these challenges, we propose DO-RAG, a scalable and customizable hybrid QA framework that integrates multi-level knowledge graph construction with semantic vector retrieval. Our system employs a novel agentic chain-of-thought architecture to extract structured relationships from unstructured, multimodal documents, constructing dynamic knowledge graphs that enhance retrieval precision. At query time, DO-RAG fuses graph and vector retrieval results to generate context-aware responses, followed by hallucination mitigation via grounded refinement. Experimental evaluations in the database and electrical domains show near-perfect recall and over 94% answer relevancy, with DO-RAG outperforming baseline frameworks by up to 33.38%. By combining traceability, adaptability, and performance efficiency, DO-RAG offers a reliable foundation for multi-domain, high-precision QA at scale. 

---
# Are LLMs Ready for English Standardized Tests? A Benchmarking and Elicitation Perspective 

**Authors**: Luoxi Tang, Tharunya Sundar, Shuai Yang, Ankita Patra, Manohar Chippada, Giqi Zhao, Yi Li, Riteng Zhang, Tunan Zhao, Ting Yang, Yuqiao Meng, Weicheng Ma, Zhaohan Xi  

**Link**: [PDF](https://arxiv.org/pdf/2505.17056)  

**Abstract**: AI is transforming education by enabling powerful tools that enhance learning experiences. Among recent advancements, large language models (LLMs) hold particular promise for revolutionizing how learners interact with educational content. In this work, we investigate the potential of LLMs to support standardized test preparation by focusing on English Standardized Tests (ESTs). Specifically, we assess their ability to generate accurate and contextually appropriate solutions across a diverse set of EST question types. We introduce ESTBOOK, a comprehensive benchmark designed to evaluate the capabilities of LLMs in solving EST questions. ESTBOOK aggregates five widely recognized tests, encompassing 29 question types and over 10,576 questions across multiple modalities, including text, images, audio, tables, and mathematical symbols. Using ESTBOOK, we systematically evaluate both the accuracy and inference efficiency of LLMs. Additionally, we propose a breakdown analysis framework that decomposes complex EST questions into task-specific solution steps. This framework allows us to isolate and assess LLM performance at each stage of the reasoning process. Evaluation findings offer insights into the capability of LLMs in educational contexts and point toward targeted strategies for improving their reliability as intelligent tutoring systems. 

---
# Enhancing Mathematics Learning for Hard-of-Hearing Students Through Real-Time Palestinian Sign Language Recognition: A New Dataset 

**Authors**: Fidaa khandaqji, Huthaifa I. Ashqar, Abdelrahem Atawnih  

**Link**: [PDF](https://arxiv.org/pdf/2505.17055)  

**Abstract**: The study aims to enhance mathematics education accessibility for hard-of-hearing students by developing an accurate Palestinian sign language PSL recognition system using advanced artificial intelligence techniques. Due to the scarcity of digital resources for PSL, a custom dataset comprising 41 mathematical gesture classes was created, and recorded by PSL experts to ensure linguistic accuracy and domain specificity. To leverage state-of-the-art-computer vision techniques, a Vision Transformer ViTModel was fine-tuned for gesture classification. The model achieved an accuracy of 97.59%, demonstrating its effectiveness in recognizing mathematical signs with high precision and reliability. This study highlights the role of deep learning in developing intelligent educational tools that bridge the learning gap for hard-of-hearing students by providing AI-driven interactive solutions to enhance mathematical comprehension. This work represents a significant step toward innovative and inclusive frosting digital integration in specialized learning environments. The dataset is hosted on Hugging Face at this https URL. 

---
# METHOD: Modular Efficient Transformer for Health Outcome Discovery 

**Authors**: Linglong Qian, Zina Ibrahim  

**Link**: [PDF](https://arxiv.org/pdf/2505.17054)  

**Abstract**: Recent advances in transformer architectures have revolutionised natural language processing, but their application to healthcare domains presents unique challenges. Patient timelines are characterised by irregular sampling, variable temporal dependencies, and complex contextual relationships that differ substantially from traditional language tasks. This paper introduces \METHOD~(Modular Efficient Transformer for Health Outcome Discovery), a novel transformer architecture specifically designed to address the challenges of clinical sequence modelling in electronic health records. \METHOD~integrates three key innovations: (1) a patient-aware attention mechanism that prevents information leakage whilst enabling efficient batch processing; (2) an adaptive sliding window attention scheme that captures multi-scale temporal dependencies; and (3) a U-Net inspired architecture with dynamic skip connections for effective long sequence processing. Evaluations on the MIMIC-IV database demonstrate that \METHOD~consistently outperforms the state-of-the-art \ETHOS~model, particularly in predicting high-severity cases that require urgent clinical intervention. \METHOD~exhibits stable performance across varying inference lengths, a crucial feature for clinical deployment where patient histories vary significantly in length. Analysis of learned embeddings reveals that \METHOD~better preserves clinical hierarchies and relationships between medical concepts. These results suggest that \METHOD~represents a significant advancement in transformer architectures optimised for healthcare applications, providing more accurate and clinically relevant predictions whilst maintaining computational efficiency. 

---
# Social preferences with unstable interactive reasoning: Large language models in economic trust games 

**Authors**: Ou Jiamin, Eikmans Emile, Buskens Vincent, Pankowska Paulina, Shan Yuli  

**Link**: [PDF](https://arxiv.org/pdf/2505.17053)  

**Abstract**: While large language models (LLMs) have demonstrated remarkable capabilities in understanding human languages, this study explores how they translate this understanding into social exchange contexts that capture certain essences of real world human interactions. Three LLMs - ChatGPT-4, Claude, and Bard - were placed in economic trust games where players balance self-interest with trust and reciprocity, making decisions that reveal their social preferences and interactive reasoning abilities. Our study shows that LLMs deviate from pure self-interest and exhibit trust and reciprocity even without being prompted to adopt a specific persona. In the simplest one-shot interaction, LLMs emulated how human players place trust at the beginning of such a game. Larger human-machine divergences emerged in scenarios involving trust repayment or multi-round interactions, where decisions were influenced by both social preferences and interactive reasoning. LLMs responses varied significantly when prompted to adopt personas like selfish or unselfish players, with the impact outweighing differences between models or game types. Response of ChatGPT-4, in an unselfish or neutral persona, resembled the highest trust and reciprocity, surpassing humans, Claude, and Bard. Claude and Bard displayed trust and reciprocity levels that sometimes exceeded and sometimes fell below human choices. When given selfish personas, all LLMs showed lower trust and reciprocity than humans. Interactive reasoning to the actions of counterparts or changing game mechanics appeared to be random rather than stable, reproducible characteristics in the response of LLMs, though some improvements were observed when ChatGPT-4 responded in selfish or unselfish personas. 

---
# SpecEdge: Scalable Edge-Assisted Serving Framework for Interactive LLMs 

**Authors**: Jinwoo Park, Seunggeun Cho, Dongsu Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.17052)  

**Abstract**: Large language models (LLMs) power many modern applications, but serving them at scale remains costly and resource-intensive. Current server-centric systems overlook consumer-grade GPUs at the edge. We introduce SpecEdge, an edge-assisted inference framework that splits LLM workloads between edge and server GPUs using a speculative decoding scheme, exchanging only token outputs over the network. SpecEdge employs proactive edge drafting to overlap edge token creation with server verification and pipeline-aware scheduling that interleaves multiple user requests to increase server-side throughput. Experiments show SpecEdge enhances overall cost efficiency by 1.91x through achieving 2.22x server throughput, and reduces inter token latency by 11.24% compared to a server-only baseline, introducing a scalable, cost-effective paradigm for LLM serving. 

---
# Embedding-to-Prefix: Parameter-Efficient Personalization for Pre-Trained Large Language Models 

**Authors**: Bernd Huber, Ghazal Fazelnia, Andreas Damianou, Sebastian Peleato, Max Lefarov, Praveen Ravichandran, Marco De Nadai, Mounia Lalmas-Roellke, Paul N. Bennett  

**Link**: [PDF](https://arxiv.org/pdf/2505.17051)  

**Abstract**: Large language models (LLMs) excel at generating contextually relevant content. However, tailoring these outputs to individual users for effective personalization is a significant challenge. While rich user-specific information often exists as pre-existing user representations, such as embeddings learned from preferences or behaviors, current methods to leverage these for LLM personalization typically require costly fine-tuning or token-heavy prompting. We propose Embedding-to-Prefix (E2P), a parameter-efficient method that injects pre-computed context embeddings into an LLM's hidden representation space through a learned projection to a single soft token prefix. This enables effective personalization while keeping the backbone model frozen and avoiding expensive adaptation techniques. We evaluate E2P across two public datasets and in a production setting: dialogue personalization on Persona-Chat, contextual headline generation on PENS, and large-scale personalization for music and podcast consumption. Results show that E2P preserves contextual signals and achieves strong performance with minimal computational overhead, offering a scalable, efficient solution for contextualizing generative AI systems. 

---
# Towards Robust Evaluation of STEM Education: Leveraging MLLMs in Project-Based Learning 

**Authors**: Yanhao Jia, Xinyi Wu, Qinglin Zhang, Yiran Qin, Luwei Xiao, Shuai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.17050)  

**Abstract**: Project-Based Learning (PBL) involves a variety of highly correlated multimodal data, making it a vital educational approach within STEM disciplines. With the rapid development of multimodal large language models (MLLMs), researchers have begun exploring their potential to enhance tasks such as information retrieval, knowledge comprehension, and data generation in educational settings. However, existing benchmarks fall short in providing both a free-form output structure and a rigorous human expert validation process, limiting their effectiveness in evaluating real-world educational tasks. Additionally, few methods have developed automated pipelines to assist with the complex responsibilities of teachers leveraging MLLMs, largely due to model hallucination and instability, which lead to unreliable implementation. To address this gap, we introduce PBLBench, a novel benchmark designed to evaluate complex reasoning grounded in domain-specific knowledge and long-context understanding, thereby challenging models with tasks that closely resemble those handled by human experts. To establish reliable ground truth, we adopt the Analytic Hierarchy Process (AHP), utilizing expert-driven pairwise comparisons to derive structured and weighted evaluation criteria. We assess the performance of 15 leading MLLMs/LLMs using PBLBench and demonstrate that even the most advanced models achieve only 59% rank accuracy, underscoring the significant challenges presented by this benchmark. We believe PBLBench will serve as a catalyst for the development of more capable AI agents, ultimately aiming to alleviate teacher workload and enhance educational productivity. 

---
# Gender and Positional Biases in LLM-Based Hiring Decisions: Evidence from Comparative CV/Résumé Evaluations 

**Authors**: David Rozado  

**Link**: [PDF](https://arxiv.org/pdf/2505.17049)  

**Abstract**: This study examines the behavior of Large Language Models (LLMs) when evaluating professional candidates based on their resumes or curricula vitae (CVs). In an experiment involving 22 leading LLMs, each model was systematically given one job description along with a pair of profession-matched CVs, one bearing a male first name, the other a female first name, and asked to select the more suitable candidate for the job. Each CV pair was presented twice, with names swapped to ensure that any observed preferences in candidate selection stemmed from gendered names cues. Despite identical professional qualifications across genders, all LLMs consistently favored female-named candidates across 70 different professions. Adding an explicit gender field (male/female) to the CVs further increased the preference for female applicants. When gendered names were replaced with gender-neutral identifiers "Candidate A" and "Candidate B", several models displayed a preference to select "Candidate A". Counterbalancing gender assignment between these gender-neutral identifiers resulted in gender parity in candidate selection. When asked to rate CVs in isolation rather than compare pairs, LLMs assigned slightly higher average scores to female CVs overall, but the effect size was negligible. Including preferred pronouns (he/him or she/her) next to a candidate's name slightly increased the odds of the candidate being selected regardless of gender. Finally, most models exhibited a substantial positional bias to select the candidate listed first in the prompt. These findings underscore the need for caution when deploying LLMs in high-stakes autonomous decision-making contexts and raise doubts about whether LLMs consistently apply principled reasoning. 

---
# Words That Unite The World: A Unified Framework for Deciphering Central Bank Communications Globally 

**Authors**: Agam Shah, Siddhant Sukhani, Huzaifa Pardawala, Saketh Budideti, Riya Bhadani, Rudra Gopal, Siddhartha Somani, Michael Galarnyk, Soungmin Lee, Arnav Hiray, Akshar Ravichandran, Eric Kim, Pranav Aluru, Joshua Zhang, Sebastian Jaskowski, Veer Guda, Meghaj Tarte, Liqin Ye, Spencer Gosden, Rutwik Routu, Rachel Yuh, Sloka Chava, Sahasra Chava, Dylan Patrick Kelly, Aiden Chiang, Harsit Mittal, Sudheer Chava  

**Link**: [PDF](https://arxiv.org/pdf/2505.17048)  

**Abstract**: Central banks around the world play a crucial role in maintaining economic stability. Deciphering policy implications in their communications is essential, especially as misinterpretations can disproportionately impact vulnerable populations. To address this, we introduce the World Central Banks (WCB) dataset, the most comprehensive monetary policy corpus to date, comprising over 380k sentences from 25 central banks across diverse geographic regions, spanning 28 years of historical data. After uniformly sampling 1k sentences per bank (25k total) across all available years, we annotate and review each sentence using dual annotators, disagreement resolutions, and secondary expert reviews. We define three tasks: Stance Detection, Temporal Classification, and Uncertainty Estimation, with each sentence annotated for all three. We benchmark seven Pretrained Language Models (PLMs) and nine Large Language Models (LLMs) (Zero-Shot, Few-Shot, and with annotation guide) on these tasks, running 15,075 benchmarking experiments. We find that a model trained on aggregated data across banks significantly surpasses a model trained on an individual bank's data, confirming the principle "the whole is greater than the sum of its parts." Additionally, rigorous human evaluations, error analyses, and predictive tasks validate our framework's economic utility. Our artifacts are accessible through the HuggingFace and GitHub under the CC-BY-NC-SA 4.0 license. 

---
# Assessing the Quality of AI-Generated Clinical Notes: A Validated Evaluation of a Large Language Model Scribe 

**Authors**: Erin Palm, Astrit Manikantan, Mark E. Pepin, Herprit Mahal, Srikanth Subramanya Belwadi  

**Link**: [PDF](https://arxiv.org/pdf/2505.17047)  

**Abstract**: In medical practices across the United States, physicians have begun implementing generative artificial intelligence (AI) tools to perform the function of scribes in order to reduce the burden of documenting clinical encounters. Despite their widespread use, no established methods exist to gauge the quality of AI scribes. To address this gap, we developed a blinded study comparing the relative performance of large language model (LLM) generated clinical notes with those from field experts based on audio-recorded clinical encounters. Quantitative metrics from the Physician Documentation Quality Instrument (PDQI9) provided a framework to measure note quality, which we adapted to assess relative performance of AI generated notes. Clinical experts spanning 5 medical specialties used the PDQI9 tool to evaluate specialist-drafted Gold notes and LLM authored Ambient notes. Two evaluators from each specialty scored notes drafted from a total of 97 patient visits. We found uniformly high inter rater agreement (RWG greater than 0.7) between evaluators in general medicine, orthopedics, and obstetrics and gynecology, and moderate (RWG 0.5 to 0.7) to high inter rater agreement in pediatrics and cardiology. We found a modest yet significant difference in the overall note quality, wherein Gold notes achieved a score of 4.25 out of 5 and Ambient notes scored 4.20 out of 5 (p = 0.04). Our findings support the use of the PDQI9 instrument as a practical method to gauge the quality of LLM authored notes, as compared to human-authored notes. 

---
# Assessing GPT's Bias Towards Stigmatized Social Groups: An Intersectional Case Study on Nationality Prejudice and Psychophobia 

**Authors**: Afifah Kashif, Heer Patel  

**Link**: [PDF](https://arxiv.org/pdf/2505.17045)  

**Abstract**: Recent studies have separately highlighted significant biases within foundational large language models (LLMs) against certain nationalities and stigmatized social groups. This research investigates the ethical implications of these biases intersecting with outputs of widely-used GPT-3.5/4/4o LLMS. Through structured prompt series, we evaluate model responses to several scenarios involving American and North Korean nationalities with various mental disabilities. Findings reveal significant discrepancies in empathy levels with North Koreans facing greater negative bias, particularly when mental disability is also a factor. This underscores the need for improvements in LLMs designed with a nuanced understanding of intersectional identity. 

---
# QRA++: Quantified Reproducibility Assessment for Common Types of Results in Natural Language Processing 

**Authors**: Anya Belz  

**Link**: [PDF](https://arxiv.org/pdf/2505.17043)  

**Abstract**: Reproduction studies reported in NLP provide individual data points which in combination indicate worryingly low levels of reproducibility in the field. Because each reproduction study reports quantitative conclusions based on its own, often not explicitly stated, criteria for reproduction success/failure, the conclusions drawn are hard to interpret, compare, and learn from. In this paper, we present QRA++, a quantitative approach to reproducibility assessment that (i) produces continuous-valued degree of reproducibility assessments at three levels of granularity; (ii) utilises reproducibility measures that are directly comparable across different studies; and (iii) grounds expectations about degree of reproducibility in degree of similarity between experiments. QRA++ enables more informative reproducibility assessments to be conducted, and conclusions to be drawn about what causes reproducibility to be better/poorer. We illustrate this by applying QRA++ to three example sets of comparable experiments, revealing clear evidence that degree of reproducibility depends on similarity of experiment properties, but also system type and evaluation method. 

---
# VLM-KG: Multimodal Radiology Knowledge Graph Generation 

**Authors**: Abdullah Abdullah, Seong Tae Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.17042)  

**Abstract**: Vision-Language Models (VLMs) have demonstrated remarkable success in natural language generation, excelling at instruction following and structured output generation. Knowledge graphs play a crucial role in radiology, serving as valuable sources of factual information and enhancing various downstream tasks. However, generating radiology-specific knowledge graphs presents significant challenges due to the specialized language of radiology reports and the limited availability of domain-specific data. Existing solutions are predominantly unimodal, meaning they generate knowledge graphs only from radiology reports while excluding radiographic images. Additionally, they struggle with long-form radiology data due to limited context length. To address these limitations, we propose a novel multimodal VLM-based framework for knowledge graph generation in radiology. Our approach outperforms previous methods and introduces the first multimodal solution for radiology knowledge graph generation. 

---
# A new classification system of beer categories and styles based on large-scale data mining and self-organizing maps of beer recipes 

**Authors**: Diego Bonatto  

**Link**: [PDF](https://arxiv.org/pdf/2505.17039)  

**Abstract**: A data-driven quantitative approach was used to develop a novel classification system for beer categories and styles. Sixty-two thousand one hundred twenty-one beer recipes were mined and analyzed, considering ingredient profiles, fermentation parameters, and recipe vital statistics. Statistical analyses combined with self-organizing maps (SOMs) identified four major superclusters that showed distinctive malt and hop usage patterns, style characteristics, and historical brewing traditions. Cold fermented styles showed a conservative grain and hop composition, whereas hot fermented beers exhibited high heterogeneity, reflecting regional preferences and innovation. This new taxonomy offers a reproducible and objective framework beyond traditional sensory-based classifications, providing brewers, researchers, and educators with a scalable tool for recipe analysis and beer development. The findings in this work provide an understanding of beer diversity and open avenues for linking ingredient usage with fermentation profiles and flavor outcomes. 

---
# Signals from the Floods: AI-Driven Disaster Analysis through Multi-Source Data Fusion 

**Authors**: Xian Gong, Paul X. McCarthy, Lin Tian, Marian-Andrei Rizoiu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17038)  

**Abstract**: Massive and diverse web data are increasingly vital for government disaster response, as demonstrated by the 2022 floods in New South Wales (NSW), Australia. This study examines how X (formerly Twitter) and public inquiry submissions provide insights into public behaviour during crises. We analyse more than 55,000 flood-related tweets and 1,450 submissions to identify behavioural patterns during extreme weather events. While social media posts are short and fragmented, inquiry submissions are detailed, multi-page documents offering structured insights. Our methodology integrates Latent Dirichlet Allocation (LDA) for topic modelling with Large Language Models (LLMs) to enhance semantic understanding. LDA reveals distinct opinions and geographic patterns, while LLMs improve filtering by identifying flood-relevant tweets using public submissions as a reference. This Relevance Index method reduces noise and prioritizes actionable content, improving situational awareness for emergency responders. By combining these complementary data streams, our approach introduces a novel AI-driven method to refine crisis-related social media content, improve real-time disaster response, and inform long-term resilience planning. 

---
# Prompt Engineering: How Prompt Vocabulary affects Domain Knowledge 

**Authors**: Dimitri Schreiter  

**Link**: [PDF](https://arxiv.org/pdf/2505.17037)  

**Abstract**: Prompt engineering has emerged as a critical component in optimizing large language models (LLMs) for domain-specific tasks. However, the role of prompt specificity, especially in domains like STEM (physics, chemistry, biology, computer science and mathematics), medicine, and law, remains underexplored. This thesis addresses the problem of whether increasing the specificity of vocabulary in prompts improves LLM performance in domain-specific question-answering and reasoning tasks. We developed a synonymization framework to systematically substitute nouns, verbs, and adjectives with varying specificity levels, measuring the impact on four LLMs: Llama-3.1-70B-Instruct, Granite-13B-Instruct-V2, Flan-T5-XL, and Mistral-Large 2, across datasets in STEM, law, and medicine. Our results reveal that while generally increasing the specificity of prompts does not have a significant impact, there appears to be a specificity range, across all considered models, where the LLM performs the best. Identifying this optimal specificity range offers a key insight for prompt design, suggesting that manipulating prompts within this range could maximize LLM performance and lead to more efficient applications in specialized domains. 

---
# Gaming Tool Preferences in Agentic LLMs 

**Authors**: Kazem Faghih, Wenxiao Wang, Yize Cheng, Siddhant Bharti, Gaurang Sriramanan, Sriram Balasubramanian, Parsa Hosseini, Soheil Feizi  

**Link**: [PDF](https://arxiv.org/pdf/2505.18135)  

**Abstract**: Large language models (LLMs) can now access a wide range of external tools, thanks to the Model Context Protocol (MCP). This greatly expands their abilities as various agents. However, LLMs rely entirely on the text descriptions of tools to decide which ones to use--a process that is surprisingly fragile. In this work, we expose a vulnerability in prevalent tool/function-calling protocols by investigating a series of edits to tool descriptions, some of which can drastically increase a tool's usage from LLMs when competing with alternatives. Through controlled experiments, we show that tools with properly edited descriptions receive over 10 times more usage from GPT-4.1 and Qwen2.5-7B than tools with original descriptions. We further evaluate how various edits to tool descriptions perform when competing directly with one another and how these trends generalize or differ across a broader set of 10 different models. These phenomenons, while giving developers a powerful way to promote their tools, underscore the need for a more reliable foundation for agentic LLMs to select and utilize tools and resources. 

---
# VideoGameBench: Can Vision-Language Models complete popular video games? 

**Authors**: Alex L. Zhang, Thomas L. Griffiths, Karthik R. Narasimhan, Ofir Press  

**Link**: [PDF](https://arxiv.org/pdf/2505.18134)  

**Abstract**: Vision-language models (VLMs) have achieved strong results on coding and math benchmarks that are challenging for humans, yet their ability to perform tasks that come naturally to humans--such as perception, spatial navigation, and memory management--remains understudied. Real video games are crafted to be intuitive for humans to learn and master by leveraging innate inductive biases, making them an ideal testbed for evaluating such capabilities in VLMs. To this end, we introduce VideoGameBench, a benchmark consisting of 10 popular video games from the 1990s that VLMs directly interact with in real-time. VideoGameBench challenges models to complete entire games with access to only raw visual inputs and a high-level description of objectives and controls, a significant departure from existing setups that rely on game-specific scaffolding and auxiliary information. We keep three of the games secret to encourage solutions that generalize to unseen environments. Our experiments show that frontier vision-language models struggle to progress beyond the beginning of each game. We find inference latency to be a major limitation of frontier models in the real-time setting; therefore, we introduce VideoGameBench Lite, a setting where the game pauses while waiting for the LM's next action. The best performing model, Gemini 2.5 Pro, completes only 0.48% of VideoGameBench and 1.6% of VideoGameBench Lite. We hope that the formalization of the human skills mentioned above into this benchmark motivates progress in these research directions. 

---
# One RL to See Them All: Visual Triple Unified Reinforcement Learning 

**Authors**: Yan Ma, Linge Du, Xuyang Shen, Shaoxiang Chen, Pengfei Li, Qibing Ren, Lizhuang Ma, Yuchao Dai, Pengfei Liu, Junjie Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.18129)  

**Abstract**: Reinforcement learning (RL) has significantly advanced the reasoning capabilities of vision-language models (VLMs). However, the use of RL beyond reasoning tasks remains largely unexplored, especially for perceptionintensive tasks like object detection and grounding. We propose V-Triune, a Visual Triple Unified Reinforcement Learning system that enables VLMs to jointly learn visual reasoning and perception tasks within a single training pipeline. V-Triune comprises triple complementary components: Sample-Level Data Formatting (to unify diverse task inputs), Verifier-Level Reward Computation (to deliver custom rewards via specialized verifiers) , and Source-Level Metric Monitoring (to diagnose problems at the data-source level). We further introduce a novel Dynamic IoU reward, which provides adaptive, progressive, and definite feedback for perception tasks handled by V-Triune. Our approach is instantiated within off-the-shelf RL training framework using open-source 7B and 32B backbone models. The resulting model, dubbed Orsta (One RL to See Them All), demonstrates consistent improvements across both reasoning and perception tasks. This broad capability is significantly shaped by its training on a diverse dataset, constructed around four representative visual reasoning tasks (Math, Puzzle, Chart, and Science) and four visual perception tasks (Grounding, Detection, Counting, and OCR). Subsequently, Orsta achieves substantial gains on MEGA-Bench Core, with improvements ranging from +2.1 to an impressive +14.1 across its various 7B and 32B model variants, with performance benefits extending to a wide range of downstream tasks. These results highlight the effectiveness and scalability of our unified RL approach for VLMs. The V-Triune system, along with the Orsta models, is publicly available at this https URL. 

---
# Reward Model Overoptimisation in Iterated RLHF 

**Authors**: Lorenz Wolf, Robert Kirk, Mirco Musolesi  

**Link**: [PDF](https://arxiv.org/pdf/2505.18126)  

**Abstract**: Reinforcement learning from human feedback (RLHF) is a widely used method for aligning large language models with human preferences. However, RLHF often suffers from reward model overoptimisation, in which models overfit to the reward function, resulting in non-generalisable policies that exploit the idiosyncrasies and peculiarities of the reward function. A common mitigation is iterated RLHF, in which reward models are repeatedly retrained with updated human feedback and policies are re-optimised. Despite its increasing adoption, the dynamics of overoptimisation in this setting remain poorly understood. In this work, we present the first comprehensive study of overoptimisation in iterated RLHF. We systematically analyse key design choices - how reward model training data is transferred across iterations, which reward function is used for optimisation, and how policies are initialised. Using the controlled AlpacaFarm benchmark, we observe that overoptimisation tends to decrease over successive iterations, as reward models increasingly approximate ground-truth preferences. However, performance gains diminish over time, and while reinitialising from the base policy is robust, it limits optimisation flexibility. Other initialisation strategies often fail to recover from early overoptimisation. These findings offer actionable insights for building more stable and generalisable RLHF pipelines. 

---
# TabSTAR: A Foundation Tabular Model With Semantically Target-Aware Representations 

**Authors**: Alan Arazi, Eilam Shapira, Roi Reichart  

**Link**: [PDF](https://arxiv.org/pdf/2505.18125)  

**Abstract**: While deep learning has achieved remarkable success across many domains, it has historically underperformed on tabular learning tasks, which remain dominated by gradient boosting decision trees (GBDTs). However, recent advancements are paving the way for Tabular Foundation Models, which can leverage real-world knowledge and generalize across diverse datasets, particularly when the data contains free-text. Although incorporating language model capabilities into tabular tasks has been explored, most existing methods utilize static, target-agnostic textual representations, limiting their effectiveness. We introduce TabSTAR: a Foundation Tabular Model with Semantically Target-Aware Representations. TabSTAR is designed to enable transfer learning on tabular data with textual features, with an architecture free of dataset-specific parameters. It unfreezes a pretrained text encoder and takes as input target tokens, which provide the model with the context needed to learn task-specific embeddings. TabSTAR achieves state-of-the-art performance for both medium- and large-sized datasets across known benchmarks of classification tasks with text features, and its pretraining phase exhibits scaling laws in the number of datasets, offering a pathway for further performance improvements. 

---
# ProgRM: Build Better GUI Agents with Progress Rewards 

**Authors**: Danyang Zhang, Situo Zhang, Ziyue Yang, Zichen Zhu, Zihan Zhao, Ruisheng Cao, Lu Chen, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18121)  

**Abstract**: LLM-based (Large Language Model) GUI (Graphical User Interface) agents can potentially reshape our daily lives significantly. However, current LLM-based GUI agents suffer from the scarcity of high-quality training data owing to the difficulties of trajectory collection and reward annotation. Existing works have been exploring LLMs to collect trajectories for imitation learning or to offer reward signals for online RL training. However, the Outcome Reward Model (ORM) used in existing works cannot provide finegrained feedback and can over-penalize the valuable steps in finally failed trajectories. To this end, we propose Progress Reward Model (ProgRM) to provide dense informative intermediate rewards by predicting a task completion progress for each step in online training. To handle the challenge of progress reward label annotation, we further design an efficient LCS-based (Longest Common Subsequence) self-annotation algorithm to discover the key steps in trajectories and assign progress labels accordingly. ProgRM is evaluated with extensive experiments and analyses. Actors trained with ProgRM outperform leading proprietary LLMs and ORM-trained actors, illustrating the effectiveness of ProgRM. The codes for experiments will be made publicly available upon acceptance. 

---
# Bridging Supervised Learning and Reinforcement Learning in Math Reasoning 

**Authors**: Huayu Chen, Kaiwen Zheng, Qinsheng Zhang, Ganqu Cui, Yin Cui, Haotian Ye, Tsung-Yi Lin, Ming-Yu Liu, Jun Zhu, Haoxiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18116)  

**Abstract**: Reinforcement Learning (RL) has played a central role in the recent surge of LLMs' math abilities by enabling self-improvement through binary verifier signals. In contrast, Supervised Learning (SL) is rarely considered for such verification-driven training, largely due to its heavy reliance on reference answers and inability to reflect on mistakes. In this work, we challenge the prevailing notion that self-improvement is exclusive to RL and propose Negative-aware Fine-Tuning (NFT) -- a supervised approach that enables LLMs to reflect on their failures and improve autonomously with no external teachers. In online training, instead of throwing away self-generated negative answers, NFT constructs an implicit negative policy to model them. This implicit policy is parameterized with the same positive LLM we target to optimize on positive data, enabling direct policy optimization on all LLMs' generations. We conduct experiments on 7B and 32B models in math reasoning tasks. Results consistently show that through the additional leverage of negative feedback, NFT significantly improves over SL baselines like Rejection sampling Fine-Tuning, matching or even surpassing leading RL algorithms like GRPO and DAPO. Furthermore, we demonstrate that NFT and GRPO are actually equivalent in strict-on-policy training, even though they originate from entirely different theoretical foundations. Our experiments and theoretical findings bridge the gap between SL and RL methods in binary-feedback learning systems. 

---
# How Can I Publish My LLM Benchmark Without Giving the True Answers Away? 

**Authors**: Takashi Ishida, Thanawat Lodkaew, Ikko Yamane  

**Link**: [PDF](https://arxiv.org/pdf/2505.18102)  

**Abstract**: Publishing a large language model (LLM) benchmark on the Internet risks contaminating future LLMs: the benchmark may be unintentionally (or intentionally) used to train or select a model. A common mitigation is to keep the benchmark private and let participants submit their models or predictions to the organizers. However, this strategy will require trust in a single organization and still permits test-set overfitting through repeated queries. To overcome this issue, we propose a way to publish benchmarks without completely disclosing the ground-truth answers to the questions, while still maintaining the ability to openly evaluate LLMs. Our main idea is to inject randomness to the answers by preparing several logically correct answers, and only include one of them as the solution in the benchmark. This reduces the best possible accuracy, i.e., Bayes accuracy, of the benchmark. Not only is this helpful to keep us from disclosing the ground truth, but this approach also offers a test for detecting data contamination. In principle, even fully capable models should not surpass the Bayes accuracy. If a model surpasses this ceiling despite this expectation, this is a strong signal of data contamination. We present experimental evidence that our method can detect data contamination accurately on a wide range of benchmarks, models, and training methodologies. 

---
# Data Mixing Can Induce Phase Transitions in Knowledge Acquisition 

**Authors**: Xinran Gu, Kaifeng Lyu, Jiazheng Li, Jingzhao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18091)  

**Abstract**: Large Language Models (LLMs) are typically trained on data mixtures: most data come from web scrapes, while a small portion is curated from high-quality sources with dense domain-specific knowledge. In this paper, we show that when training LLMs on such data mixtures, knowledge acquisition from knowledge-dense datasets, unlike training exclusively on knowledge-dense data (arXiv:2404.05405), does not always follow a smooth scaling law but can exhibit phase transitions with respect to the mixing ratio and model size. Through controlled experiments on a synthetic biography dataset mixed with web-scraped data, we demonstrate that: (1) as we increase the model size to a critical value, the model suddenly transitions from memorizing very few to most of the biographies; (2) below a critical mixing ratio, the model memorizes almost nothing even with extensive training, but beyond this threshold, it rapidly memorizes more biographies. We attribute these phase transitions to a capacity allocation phenomenon: a model with bounded capacity must act like a knapsack problem solver to minimize the overall test loss, and the optimal allocation across datasets can change discontinuously as the model size or mixing ratio varies. We formalize this intuition in an information-theoretic framework and reveal that these phase transitions are predictable, with the critical mixing ratio following a power-law relationship with the model size. Our findings highlight a concrete case where a good mixing recipe for large models may not be optimal for small models, and vice versa. 

---
# Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding 

**Authors**: Xiaoyi Zhang, Zhaoyang Jia, Zongyu Guo, Jiahao Li, Bin Li, Houqiang Li, Yan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18079)  

**Abstract**: Long-form video understanding presents significant challenges due to extensive temporal-spatial complexity and the difficulty of question answering under such extended contexts. While Large Language Models (LLMs) have demonstrated considerable advancements in video analysis capabilities and long context handling, they continue to exhibit limitations when processing information-dense hour-long videos. To overcome such limitations, we propose the Deep Video Discovery agent to leverage an agentic search strategy over segmented video clips. Different from previous video agents manually designing a rigid workflow, our approach emphasizes the autonomous nature of agents. By providing a set of search-centric tools on multi-granular video database, our DVD agent leverages the advanced reasoning capability of LLM to plan on its current observation state, strategically selects tools, formulates appropriate parameters for actions, and iteratively refines its internal reasoning in light of the gathered information. We perform comprehensive evaluation on multiple long video understanding benchmarks that demonstrates the advantage of the entire system design. Our DVD agent achieves SOTA performance, significantly surpassing prior works by a large margin on the challenging LVBench dataset. Comprehensive ablation studies and in-depth tool analyses are also provided, yielding insights to further advance intelligent agents tailored for long-form video understanding tasks. The code will be released later. 

---
# Structured Thinking Matters: Improving LLMs Generalization in Causal Inference Tasks 

**Authors**: Wentao Sun, Joao Paulo Nogueira, Alonso Silva  

**Link**: [PDF](https://arxiv.org/pdf/2505.18034)  

**Abstract**: Despite remarkable advances in the field, LLMs remain unreliable in distinguishing causation from correlation. Recent results from the Corr2Cause dataset benchmark reveal that state-of-the-art LLMs -- such as GPT-4 (F1 score: 29.08) -- only marginally outperform random baselines (Random Uniform, F1 score: 20.38), indicating limited capacity of generalization. To tackle this limitation, we propose a novel structured approach: rather than directly answering causal queries, we provide the model with the capability to structure its thinking by guiding the model to build a structured knowledge graph, systematically encoding the provided correlational premises, to answer the causal queries. This intermediate representation significantly enhances the model's causal capabilities. Experiments on the test subset of the Corr2Cause dataset benchmark with Qwen3-32B model (reasoning model) show substantial gains over standard direct prompting methods, improving F1 scores from 32.71 to 48.26 (over 47.5% relative increase), along with notable improvements in precision and recall. These results underscore the effectiveness of providing the model with the capability to structure its thinking and highlight its promising potential for broader generalization across diverse causal inference tasks. 

---
# Towards Analyzing and Understanding the Limitations of VAPO: A Theoretical Perspective 

**Authors**: Jintian Shao, Yiming Cheng, Hongyi Huang, Beiwen Zhang, Zhiyu Wu, You Shan, Mingkai Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.17997)  

**Abstract**: The VAPO framework has demonstrated significant empirical success in enhancing the efficiency and reliability of reinforcement learning for long chain-of-thought (CoT) reasoning tasks with large language models (LLMs). By systematically addressing challenges such as value model bias, heterogeneous sequence lengths, and sparse reward signals, VAPO achieves state-of-the-art performance. While its practical benefits are evident, a deeper theoretical understanding of its underlying mechanisms and potential limitations is crucial for guiding future advancements. This paper aims to initiate such a discussion by exploring VAPO from a theoretical perspective, highlighting areas where its assumptions might be challenged and where further investigation could yield more robust and generalizable reasoning agents. We delve into the intricacies of value function approximation in complex reasoning spaces, the optimality of adaptive advantage estimation, the impact of token-level optimization, and the enduring challenges of exploration and generalization. 

---
# Are Large Language Models Reliable AI Scientists? Assessing Reverse-Engineering of Black-Box Systems 

**Authors**: Jiayi Geng, Howard Chen, Dilip Arumugam, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2505.17968)  

**Abstract**: Using AI to create autonomous researchers has the potential to accelerate scientific discovery. A prerequisite for this vision is understanding how well an AI model can identify the underlying structure of a black-box system from its behavior. In this paper, we explore how well a large language model (LLM) learns to identify a black-box function from passively observed versus actively collected data. We investigate the reverse-engineering capabilities of LLMs across three distinct types of black-box systems, each chosen to represent different problem domains where future autonomous AI researchers may have considerable impact: Program, Formal Language, and Math Equation. Through extensive experiments, we show that LLMs fail to extract information from observations, reaching a performance plateau that falls short of the ideal of Bayesian inference. However, we demonstrate that prompting LLMs to not only observe but also intervene -- actively querying the black-box with specific inputs to observe the resulting output -- improves performance by allowing LLMs to test edge cases and refine their beliefs. By providing the intervention data from one LLM to another, we show that this improvement is partly a result of engaging in the process of generating effective interventions, paralleling results in the literature on human learning. Further analysis reveals that engaging in intervention can help LLMs escape from two common failure modes: overcomplication, where the LLM falsely assumes prior knowledge about the black-box, and overlooking, where the LLM fails to incorporate observations. These insights provide practical guidance for helping LLMs more effectively reverse-engineer black-box systems, supporting their use in making new discoveries. 

---
# Understanding Gated Neurons in Transformers from Their Input-Output Functionality 

**Authors**: Sebastian Gerstner, Hinrich Schütze  

**Link**: [PDF](https://arxiv.org/pdf/2505.17936)  

**Abstract**: Interpretability researchers have attempted to understand MLP neurons of language models based on both the contexts in which they activate and their output weight vectors. They have paid little attention to a complementary aspect: the interactions between input and output. For example, when neurons detect a direction in the input, they might add much the same direction to the residual stream ("enrichment neurons") or reduce its presence ("depletion neurons"). We address this aspect by examining the cosine similarity between input and output weights of a neuron. We apply our method to 12 models and find that enrichment neurons dominate in early-middle layers whereas later layers tend more towards depletion. To explain this finding, we argue that enrichment neurons are largely responsible for enriching concept representations, one of the first steps of factual recall. Our input-output perspective is a complement to activation-dependent analyses and to approaches that treat input and output separately. 

---
# Towards Practical Defect-Focused Automated Code Review 

**Authors**: Junyi Lu, Lili Jiang, Xiaojia Li, Jianbing Fang, Fengjun Zhang, Li Yang, Chun Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2505.17928)  

**Abstract**: The complexity of code reviews has driven efforts to automate review comments, but prior approaches oversimplify this task by treating it as snippet-level code-to-text generation and relying on text similarity metrics like BLEU for evaluation. These methods overlook repository context, real-world merge request evaluation, and defect detection, limiting their practicality. To address these issues, we explore the full automation pipeline within the online recommendation service of a company with nearly 400 million daily active users, analyzing industry-grade C++ codebases comprising hundreds of thousands of lines of code. We identify four key challenges: 1) capturing relevant context, 2) improving key bug inclusion (KBI), 3) reducing false alarm rates (FAR), and 4) integrating human workflows. To tackle these, we propose 1) code slicing algorithms for context extraction, 2) a multi-role LLM framework for KBI, 3) a filtering mechanism for FAR reduction, and 4) a novel prompt design for better human interaction. Our approach, validated on real-world merge requests from historical fault reports, achieves a 2x improvement over standard LLMs and a 10x gain over previous baselines. While the presented results focus on C++, the underlying framework design leverages language-agnostic principles (e.g., AST-based analysis), suggesting potential for broader applicability. 

---
# T2I-Eval-R1: Reinforcement Learning-Driven Reasoning for Interpretable Text-to-Image Evaluation 

**Authors**: Zi-Ao Ma, Tian Lan, Rong-Cheng Tu, Shu-Hang Liu, Heyan Huang, Zhijing Wu, Chen Xu, Xian-Ling Mao  

**Link**: [PDF](https://arxiv.org/pdf/2505.17897)  

**Abstract**: The rapid progress in diffusion-based text-to-image (T2I) generation has created an urgent need for interpretable automatic evaluation methods that can assess the quality of generated images, therefore reducing the human annotation burden. To reduce the prohibitive cost of relying on commercial models for large-scale evaluation, and to improve the reasoning capabilities of open-source models, recent research has explored supervised fine-tuning (SFT) of multimodal large language models (MLLMs) as dedicated T2I evaluators. However, SFT approaches typically rely on high-quality critique datasets, which are either generated by proprietary LLMs-with potential issues of bias and inconsistency-or annotated by humans at high cost, limiting their scalability and generalization. To address these limitations, we propose T2I-Eval-R1, a novel reinforcement learning framework that trains open-source MLLMs using only coarse-grained quality scores, thereby avoiding the need for annotating high-quality interpretable evaluation rationale. Our approach integrates Group Relative Policy Optimization (GRPO) into the instruction-tuning process, enabling models to generate both scalar scores and interpretable reasoning chains with only easy accessible annotated judgment scores or preferences. Furthermore, we introduce a continuous reward formulation that encourages score diversity and provides stable optimization signals, leading to more robust and discriminative evaluation behavior. Experimental results on three established T2I meta-evaluation benchmarks demonstrate that T2I-Eval-R1 achieves significantly higher alignment with human assessments and offers more accurate interpretable score rationales compared to strong baseline methods. 

---
# Trinity-RFT: A General-Purpose and Unified Framework for Reinforcement Fine-Tuning of Large Language Models 

**Authors**: Xuchen Pan, Yanxi Chen, Yushuo Chen, Yuchang Sun, Daoyuan Chen, Wenhao Zhang, Yuexiang Xie, Yilun Huang, Yilei Zhang, Dawei Gao, Yaliang Li, Bolin Ding, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.17826)  

**Abstract**: Trinity-RFT is a general-purpose, flexible and scalable framework designed for reinforcement fine-tuning (RFT) of large language models. It is built with a decoupled design, consisting of (1) an RFT-core that unifies and generalizes synchronous/asynchronous, on-policy/off-policy, and online/offline modes of RFT, (2) seamless integration for agent-environment interaction with high efficiency and robustness, and (3) systematic data pipelines optimized for RFT. Trinity-RFT can be easily adapted for diverse application scenarios, and serves as a unified platform for exploring advanced reinforcement learning paradigms. This technical report outlines the vision, features, design and implementations of Trinity-RFT, accompanied by extensive examples demonstrating the utility and user-friendliness of the proposed framework. 

---
# PatientSim: A Persona-Driven Simulator for Realistic Doctor-Patient Interactions 

**Authors**: Daeun Kyung, Hyunseung Chung, Seongsu Bae, Jiho Kim, Jae Ho Sohn, Taerim Kim, Soo Kyung Kim, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2505.17818)  

**Abstract**: Doctor-patient consultations require multi-turn, context-aware communication tailored to diverse patient personas. Training or evaluating doctor LLMs in such settings requires realistic patient interaction systems. However, existing simulators often fail to reflect the full range of personas seen in clinical practice. To address this, we introduce PatientSim, a patient simulator that generates realistic and diverse patient personas for clinical scenarios, grounded in medical expertise. PatientSim operates using: 1) clinical profiles, including symptoms and medical history, derived from real-world data in the MIMIC-ED and MIMIC-IV datasets, and 2) personas defined by four axes: personality, language proficiency, medical history recall level, and cognitive confusion level, resulting in 37 unique combinations. We evaluated eight LLMs for factual accuracy and persona consistency. The top-performing open-source model, Llama 3.3, was validated by four clinicians to confirm the robustness of our framework. As an open-source, customizable platform, PatientSim provides a reproducible and scalable solution that can be customized for specific training needs. Offering a privacy-compliant environment, it serves as a robust testbed for evaluating medical dialogue systems across diverse patient presentations and shows promise as an educational tool for healthcare. 

---
# PPO-BR: Dual-Signal Entropy-Reward Adaptation for Trust Region Policy Optimization 

**Authors**: Ben Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2505.17714)  

**Abstract**: Despite Proximal Policy Optimization (PPO) dominating policy gradient methods -- from robotic control to game AI -- its static trust region forces a brittle trade-off: aggressive clipping stifles early exploration, while late-stage updates destabilize convergence. PPO-BR establishes a new paradigm in adaptive RL by fusing exploration and convergence signals into a single bounded trust region -- a theoretically grounded innovation that outperforms five SOTA baselines with less than 2% overhead. This work bridges a critical gap in phase-aware learning, enabling real-world deployment in safety-critical systems like robotic surgery within a single adaptive mechanism. PPO-BR achieves 29.1% faster convergence by combining: (1) entropy-driven expansion (epsilon up) for exploration in high-uncertainty states, and (2) reward-guided contraction (epsilon down) for convergence stability. On six diverse benchmarks (MuJoCo, Atari, sparse-reward), PPO-BR achieves 29.1% faster convergence (p < 0.001), 2.3x lower reward variance than PPO, and less than 1.8% runtime overhead with only five lines of code change. PPO-BR's simplicity and theoretical guarantees make it ready-to-deploy in safety-critical domains -- from surgical robotics to autonomous drones. In contrast to recent methods such as Group Relative Policy Optimization (GRPO), PPO-BR offers a unified entropy-reward mechanism applicable to both language models and general reinforcement learning environments. 

---
# COUNTDOWN: Contextually Sparse Activation Filtering Out Unnecessary Weights in Down Projection 

**Authors**: Jaewon Cheon, Pilsung Kang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17701)  

**Abstract**: The growing size of large language models has created significant computational inefficiencies. To address this challenge, sparse activation methods selectively deactivates non-essential parameters during inference, reducing computational costs in FFNN layers. While existing methods focus on non-linear gating mechanisms, we hypothesize that the sparsity of the FFNN layer lies globally in the form of a linear combination over its internal down projection matrix. Based on this insight, we propose two methods: M-COUNTDOWN, leveraging indirect coefficients, and D-COUNTDOWN, utilizing direct coefficients of the linear combination. Experimental results demonstrate that D-COUNTDOWN can omit 90% of computations with performance loss as low as 5.5% ideally, while M-COUNTDOWN provides a predictor-free solution with up to 29.4% better performance preservation compared to existing methods. Our specialized kernel implementations effectively realize these theoretical gains into substantial real-world acceleration. 

---
# HoloLLM: Multisensory Foundation Model for Language-Grounded Human Sensing and Reasoning 

**Authors**: Chuhao Zhou, Jianfei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17645)  

**Abstract**: Embodied agents operating in smart homes must understand human behavior through diverse sensory inputs and communicate via natural language. While Vision-Language Models (VLMs) have enabled impressive language-grounded perception, their reliance on visual data limits robustness in real-world scenarios with occlusions, poor lighting, or privacy constraints. In this paper, we introduce HoloLLM, a Multimodal Large Language Model (MLLM) that integrates uncommon but powerful sensing modalities, such as LiDAR, infrared, mmWave radar, and WiFi, to enable seamless human perception and reasoning across heterogeneous environments. We address two key challenges: (1) the scarcity of aligned modality-text data for rare sensors, and (2) the heterogeneity of their physical signal representations. To overcome these, we design a Universal Modality-Injection Projector (UMIP) that enhances pre-aligned modality embeddings with fine-grained, text-aligned features from tailored encoders via coarse-to-fine cross-attention without introducing significant alignment overhead. We further introduce a human-VLM collaborative data curation pipeline to generate paired textual annotations for sensing datasets. Extensive experiments on two newly constructed benchmarks show that HoloLLM significantly outperforms existing MLLMs, improving language-grounded human sensing accuracy by up to 30%. This work establishes a new foundation for real-world, language-informed multisensory embodied intelligence. 

---
# Surfacing Semantic Orthogonality Across Model Safety Benchmarks: A Multi-Dimensional Analysis 

**Authors**: Jonathan Bennion, Shaona Ghosh, Mantek Singh, Nouha Dziri  

**Link**: [PDF](https://arxiv.org/pdf/2505.17636)  

**Abstract**: Various AI safety datasets have been developed to measure LLMs against evolving interpretations of harm. Our evaluation of five recently published open-source safety benchmarks reveals distinct semantic clusters using UMAP dimensionality reduction and kmeans clustering (silhouette score: 0.470). We identify six primary harm categories with varying benchmark representation. GretelAI, for example, focuses heavily on privacy concerns, while WildGuardMix emphasizes self-harm scenarios. Significant differences in prompt length distribution suggests confounds to data collection and interpretations of harm as well as offer possible context. Our analysis quantifies benchmark orthogonality among AI benchmarks, allowing for transparency in coverage gaps despite topical similarities. Our quantitative framework for analyzing semantic orthogonality across safety benchmarks enables more targeted development of datasets that comprehensively address the evolving landscape of harms in AI use, however that is defined in the future. 

---
# Large language model as user daily behavior data generator: balancing population diversity and individual personality 

**Authors**: Haoxin Li, Jingtao Ding, Jiahui Gong, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.17615)  

**Abstract**: Predicting human daily behavior is challenging due to the complexity of routine patterns and short-term fluctuations. While data-driven models have improved behavior prediction by leveraging empirical data from various platforms and devices, the reliance on sensitive, large-scale user data raises privacy concerns and limits data availability. Synthetic data generation has emerged as a promising solution, though existing methods are often limited to specific applications. In this work, we introduce BehaviorGen, a framework that uses large language models (LLMs) to generate high-quality synthetic behavior data. By simulating user behavior based on profiles and real events, BehaviorGen supports data augmentation and replacement in behavior prediction models. We evaluate its performance in scenarios such as pertaining augmentation, fine-tuning replacement, and fine-tuning augmentation, achieving significant improvements in human mobility and smartphone usage predictions, with gains of up to 18.9%. Our results demonstrate the potential of BehaviorGen to enhance user behavior modeling through flexible and privacy-preserving synthetic data generation. 

---
# MMMG: a Comprehensive and Reliable Evaluation Suite for Multitask Multimodal Generation 

**Authors**: Jihan Yao, Yushi Hu, Yujie Yi, Bin Han, Shangbin Feng, Guang Yang, Bingbing Wen, Ranjay Krishna, Lucy Lu Wang, Yulia Tsvetkov, Noah A. Smith, Banghua Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17613)  

**Abstract**: Automatically evaluating multimodal generation presents a significant challenge, as automated metrics often struggle to align reliably with human evaluation, especially for complex tasks that involve multiple modalities. To address this, we present MMMG, a comprehensive and human-aligned benchmark for multimodal generation across 4 modality combinations (image, audio, interleaved text and image, interleaved text and audio), with a focus on tasks that present significant challenges for generation models, while still enabling reliable automatic evaluation through a combination of models and programs. MMMG encompasses 49 tasks (including 29 newly developed ones), each with a carefully designed evaluation pipeline, and 937 instructions to systematically assess reasoning, controllability, and other key capabilities of multimodal generation models. Extensive validation demonstrates that MMMG is highly aligned with human evaluation, achieving an average agreement of 94.3%. Benchmarking results on 24 multimodal generation models reveal that even though the state-of-the-art model, GPT Image, achieves 78.3% accuracy for image generation, it falls short on multimodal reasoning and interleaved generation. Furthermore, results suggest considerable headroom for improvement in audio generation, highlighting an important direction for future research. 

---
# Controlled Agentic Planning & Reasoning for Mechanism Synthesis 

**Authors**: João Pedro Gandarela, Thiago Rios, Stefan Menzel, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2505.17607)  

**Abstract**: This work presents a dual-agent Large Language Model (LLM)-based reasoning method for mechanism synthesis, capable of reasoning at both linguistic and symbolic levels to generate geometrical and dynamic outcomes. The model consists of a composition of well-defined functions that, starting from a natural language specification, references abstract properties through supporting equations, generates and parametrizes simulation code, and elicits feedback anchor points using symbolic regression and distance functions. This process closes an actionable refinement loop at the linguistic and symbolic layers. The approach is shown to be both effective and convergent in the context of planar mechanisms. Additionally, we introduce MSynth, a novel benchmark for planar mechanism synthesis, and perform a comprehensive analysis of the impact of the model components. We further demonstrate that symbolic regression prompts unlock mechanistic insights only when applied to sufficiently large architectures. 

---
# One Model Transfer to All: On Robust Jailbreak Prompts Generation against LLMs 

**Authors**: Linbao Li, Yannan Liu, Daojing He, Yu Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.17598)  

**Abstract**: Safety alignment in large language models (LLMs) is increasingly compromised by jailbreak attacks, which can manipulate these models to generate harmful or unintended content. Investigating these attacks is crucial for uncovering model vulnerabilities. However, many existing jailbreak strategies fail to keep pace with the rapid development of defense mechanisms, such as defensive suffixes, rendering them ineffective against defended models. To tackle this issue, we introduce a novel attack method called ArrAttack, specifically designed to target defended LLMs. ArrAttack automatically generates robust jailbreak prompts capable of bypassing various defense measures. This capability is supported by a universal robustness judgment model that, once trained, can perform robustness evaluation for any target model with a wide variety of defenses. By leveraging this model, we can rapidly develop a robust jailbreak prompt generator that efficiently converts malicious input prompts into effective attacks. Extensive evaluations reveal that ArrAttack significantly outperforms existing attack strategies, demonstrating strong transferability across both white-box and black-box models, including GPT-4 and Claude-3. Our work bridges the gap between jailbreak attacks and defenses, providing a fresh perspective on generating robust jailbreak prompts. We make the codebase available at this https URL. 

---
# NeUQI: Near-Optimal Uniform Quantization Parameter Initialization 

**Authors**: Li Lin, Xinyu Hu, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2505.17595)  

**Abstract**: Large language models (LLMs) achieve impressive performance across domains but face significant challenges when deployed on consumer-grade GPUs or personal devices such as laptops, due to high memory consumption and inference costs. Post-training quantization (PTQ) of LLMs offers a promising solution that reduces their memory footprint and decoding latency. In practice, PTQ with uniform quantization representation is favored for its efficiency and ease of deployment since uniform quantization is widely supported by mainstream hardware and software libraries. Recent studies on $\geq 2$-bit uniform quantization have led to noticeable improvements in post-quantization model performance; however, they primarily focus on quantization methodologies, while the initialization of quantization parameters is underexplored and still relies on the suboptimal Min-Max strategies. In this work, we propose NeUQI, a method devoted to efficiently determining near-optimal initial parameters for uniform quantization. NeUQI is orthogonal to prior quantization methodologies and can seamlessly integrate with them. The experiments with the LLaMA and Qwen families on various tasks demonstrate that our NeUQI consistently outperforms existing methods. Furthermore, when combined with a lightweight distillation strategy, NeUQI can achieve superior performance to PV-tuning, a much more resource-intensive approach. 

---
# CoMoE: Contrastive Representation for Mixture-of-Experts in Parameter-Efficient Fine-tuning 

**Authors**: Jinyuan Feng, Chaopeng Wei, Tenghai Qiu, Tianyi Hu, Zhiqiang Pu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17553)  

**Abstract**: In parameter-efficient fine-tuning, mixture-of-experts (MoE), which involves specializing functionalities into different experts and sparsely activating them appropriately, has been widely adopted as a promising approach to trade-off between model capacity and computation overhead. However, current MoE variants fall short on heterogeneous datasets, ignoring the fact that experts may learn similar knowledge, resulting in the underutilization of MoE's capacity. In this paper, we propose Contrastive Representation for MoE (CoMoE), a novel method to promote modularization and specialization in MoE, where the experts are trained along with a contrastive objective by sampling from activated and inactivated experts in top-k routing. We demonstrate that such a contrastive objective recovers the mutual-information gap between inputs and the two types of experts. Experiments on several benchmarks and in multi-task settings demonstrate that CoMoE can consistently enhance MoE's capacity and promote modularization among the experts. 

---
# Co-Reinforcement Learning for Unified Multimodal Understanding and Generation 

**Authors**: Jingjing Jiang, Chongjie Si, Jun Luo, Hanwang Zhang, Chao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.17534)  

**Abstract**: This paper presents a pioneering exploration of reinforcement learning (RL) via group relative policy optimization for unified multimodal large language models (ULMs), aimed at simultaneously reinforcing generation and understanding capabilities. Through systematic pilot studies, we uncover the significant potential of ULMs to enable the synergistic co-evolution of dual capabilities within a shared policy optimization framework. Building on this insight, we introduce \textbf{CoRL}, a co-reinforcement learning framework comprising a unified RL stage for joint optimization and a refined RL stage for task-specific enhancement. With the proposed CoRL, our resulting model, \textbf{ULM-R1}, achieves average improvements of \textbf{7%} on three text-to-image generation datasets and \textbf{23%} on nine multimodal understanding benchmarks. These results demonstrate the effectiveness of CoRL and highlight the substantial benefit of reinforcement learning in facilitating cross-task synergy and optimization for ULMs. 

---
# Chain-of-Lure: A Synthetic Narrative-Driven Approach to Compromise Large Language Models 

**Authors**: Wenhan Chang, Tianqing Zhu, Yu Zhao, Shuangyong Song, Ping Xiong, Wanlei Zhou, Yongxiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.17519)  

**Abstract**: In the era of rapid generative AI development, interactions between humans and large language models face significant misusing risks. Previous research has primarily focused on black-box scenarios using human-guided prompts and white-box scenarios leveraging gradient-based LLM generation methods, neglecting the possibility that LLMs can act not only as victim models, but also as attacker models to harm other models. We proposes a novel jailbreaking method inspired by the Chain-of-Thought mechanism, where the attacker model uses mission transfer to conceal harmful user intent in dialogue and generates chained narrative lures to stimulate the reasoning capabilities of victim models, leading to successful jailbreaking. To enhance the attack success rate, we introduce a helper model that performs random narrative optimization on the narrative lures during multi-turn dialogues while ensuring alignment with the original intent, enabling the optimized lures to bypass the safety barriers of victim models effectively. Our experiments reveal that models with weaker safety mechanisms exhibit stronger attack capabilities, demonstrating that models can not only be exploited, but also help harm others. By incorporating toxicity scores, we employ third-party models to evaluate the harmfulness of victim models' responses to jailbreaking attempts. The study shows that using refusal keywords as an evaluation metric for attack success rates is significantly flawed because it does not assess whether the responses guide harmful questions, while toxicity scores measure the harm of generated content with more precision and its alignment with harmful questions. Our approach demonstrates outstanding performance, uncovering latent vulnerabilities in LLMs and providing data-driven feedback to optimize LLM safety mechanisms. We also discuss two defensive strategies to offer guidance on improving defense mechanisms. 

---
# What You Read Isn't What You Hear: Linguistic Sensitivity in Deepfake Speech Detection 

**Authors**: Binh Nguyen, Shuji Shi, Ryan Ofman, Thai Le  

**Link**: [PDF](https://arxiv.org/pdf/2505.17513)  

**Abstract**: Recent advances in text-to-speech technologies have enabled realistic voice generation, fueling audio-based deepfake attacks such as fraud and impersonation. While audio anti-spoofing systems are critical for detecting such threats, prior work has predominantly focused on acoustic-level perturbations, leaving the impact of linguistic variation largely unexplored. In this paper, we investigate the linguistic sensitivity of both open-source and commercial anti-spoofing detectors by introducing transcript-level adversarial attacks. Our extensive evaluation reveals that even minor linguistic perturbations can significantly degrade detection accuracy: attack success rates surpass 60% on several open-source detector-voice pairs, and notably one commercial detection accuracy drops from 100% on synthetic audio to just 32%. Through a comprehensive feature attribution analysis, we identify that both linguistic complexity and model-level audio embedding similarity contribute strongly to detector vulnerability. We further demonstrate the real-world risk via a case study replicating the Brad Pitt audio deepfake scam, using transcript adversarial attacks to completely bypass commercial detectors. These results highlight the need to move beyond purely acoustic defenses and account for linguistic variation in the design of robust anti-spoofing systems. All source code will be publicly available. 

---
# Probe by Gaming: A Game-based Benchmark for Assessing Conceptual Knowledge in LLMs 

**Authors**: Shuhang Xu, Weijian Deng, Yixuan Zhou, Fangwei Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2505.17512)  

**Abstract**: Concepts represent generalized abstractions that enable humans to categorize and reason efficiently, yet it is unclear to what extent Large Language Models (LLMs) comprehend these semantic relationships. Existing benchmarks typically focus on factual recall and isolated tasks, failing to evaluate the ability of LLMs to understand conceptual boundaries. To address this gap, we introduce CK-Arena, a multi-agent interaction game built upon the Undercover game, designed to evaluate the capacity of LLMs to reason with concepts in interactive settings. CK-Arena challenges models to describe, differentiate, and infer conceptual boundaries based on partial information, encouraging models to explore commonalities and distinctions between closely related concepts. By simulating real-world interaction, CK-Arena provides a scalable and realistic benchmark for assessing conceptual reasoning in dynamic environments. Experimental results show that LLMs' understanding of conceptual knowledge varies significantly across different categories and is not strictly aligned with parameter size or general model capabilities. The data and code are available at the project homepage: this https URL. 

---
# On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning 

**Authors**: Yifan Zhang, Yifeng Liu, Huizhuo Yuan, Yang Yuan, Quanquan Gu, Andrew C Yao  

**Link**: [PDF](https://arxiv.org/pdf/2505.17508)  

**Abstract**: Policy gradient algorithms have been successfully applied to enhance the reasoning capabilities of large language models (LLMs). Despite the widespread use of Kullback-Leibler (KL) regularization in policy gradient algorithms to stabilize training, the systematic exploration of how different KL divergence formulations can be estimated and integrated into surrogate loss functions for online reinforcement learning (RL) presents a nuanced and systematically explorable design space. In this paper, we propose regularized policy gradient (RPG), a systematic framework for deriving and analyzing KL-regularized policy gradient methods in the online RL setting. We derive policy gradients and corresponding surrogate loss functions for objectives regularized by both forward and reverse KL divergences, considering both normalized and unnormalized policy distributions. Furthermore, we present derivations for fully differentiable loss functions as well as REINFORCE-style gradient estimators, accommodating diverse algorithmic needs. We conduct extensive experiments on RL for LLM reasoning using these methods, showing improved or competitive results in terms of training stability and performance compared to strong baselines such as GRPO, REINFORCE++, and DAPO. The code is available at this https URL. 

---
# ProxySPEX: Inference-Efficient Interpretability via Sparse Feature Interactions in LLMs 

**Authors**: Landon Butler, Abhineet Agarwal, Justin Singh Kang, Yigit Efe Erginbas, Bin Yu, Kannan Ramchandran  

**Link**: [PDF](https://arxiv.org/pdf/2505.17495)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable performance by capturing complex interactions between input features. To identify these interactions, most existing approaches require enumerating all possible combinations of features up to a given order, causing them to scale poorly with the number of inputs $n$. Recently, Kang et al. (2025) proposed SPEX, an information-theoretic approach that uses interaction sparsity to scale to $n \approx 10^3$ features. SPEX greatly improves upon prior methods but requires tens of thousands of model inferences, which can be prohibitive for large models. In this paper, we observe that LLM feature interactions are often hierarchical -- higher-order interactions are accompanied by their lower-order subsets -- which enables more efficient discovery. To exploit this hierarchy, we propose ProxySPEX, an interaction attribution algorithm that first fits gradient boosted trees to masked LLM outputs and then extracts the important interactions. Experiments across four challenging high-dimensional datasets show that ProxySPEX more faithfully reconstructs LLM outputs by 20% over marginal attribution approaches while using $10\times$ fewer inferences than SPEX. By accounting for interactions, ProxySPEX identifies features that influence model output over 20% more than those selected by marginal approaches. Further, we apply ProxySPEX to two interpretability tasks. Data attribution, where we identify interactions among CIFAR-10 training samples that influence test predictions, and mechanistic interpretability, where we uncover interactions between attention heads, both within and across layers, on a question-answering task. ProxySPEX identifies interactions that enable more aggressive pruning of heads than marginal approaches. 

---
# PD$^3$: A Project Duplication Detection Framework via Adapted Multi-Agent Debate 

**Authors**: Dezheng Bao, Yueci Yang, Xin Chen, Zhengxuan Jiang, Zeguo Fei, Daoze Zhang, Xuanwen Huang, Junru Chen, Chutian Yu, Xiang Yuan, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17492)  

**Abstract**: Project duplication detection is critical for project quality assessment, as it improves resource utilization efficiency by preventing investing in newly proposed project that have already been studied. It requires the ability to understand high-level semantics and generate constructive and valuable feedback. Existing detection methods rely on basic word- or sentence-level comparison or solely apply large language models, lacking valuable insights for experts and in-depth comprehension of project content and review criteria. To tackle this issue, we propose PD$^3$, a Project Duplication Detection framework via adapted multi-agent Debate. Inspired by real-world expert debates, it employs a fair competition format to guide multi-agent debate to retrieve relevant projects. For feedback, it incorporates both qualitative and quantitative analysis to improve its practicality. Over 800 real-world power project data spanning more than 20 specialized fields are used to evaluate the framework, demonstrating that our method outperforms existing approaches by 7.43% and 8.00% in two downstream tasks. Furthermore, we establish an online platform, Review Dingdang, to assist power experts, saving 5.73 million USD in initial detection on more than 100 newly proposed projects. 

---
# From Reasoning to Generalization: Knowledge-Augmented LLMs for ARC Benchmark 

**Authors**: Chao Lei, Nir Lipovetzky, Krista A. Ehinger, Yanchuan Chang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17482)  

**Abstract**: Recent reasoning-oriented LLMs have demonstrated strong performance on challenging tasks such as mathematics and science examinations. However, core cognitive faculties of human intelligence, such as abstract reasoning and generalization, remain underexplored. To address this, we evaluate recent reasoning-oriented LLMs on the Abstraction and Reasoning Corpus (ARC) benchmark, which explicitly demands both faculties. We formulate ARC as a program synthesis task and propose nine candidate solvers. Experimental results show that repeated-sampling planning-aided code generation (RSPC) achieves the highest test accuracy and demonstrates consistent generalization across most LLMs. To further improve performance, we introduce an ARC solver, Knowledge Augmentation for Abstract Reasoning (KAAR), which encodes core knowledge priors within an ontology that classifies priors into three hierarchical levels based on their dependencies. KAAR progressively expands LLM reasoning capacity by gradually augmenting priors at each level, and invokes RSPC to generate candidate solutions after each augmentation stage. This stage-wise reasoning reduces interference from irrelevant priors and improves LLM performance. Empirical results show that KAAR maintains strong generalization and consistently outperforms non-augmented RSPC across all evaluated LLMs, achieving around 5% absolute gains and up to 64.52% relative improvement. Despite these achievements, ARC remains a challenging benchmark for reasoning-oriented LLMs, highlighting future avenues of progress in LLMs. 

---
# OrionBench: A Benchmark for Chart and Human-Recognizable Object Detection in Infographics 

**Authors**: Jiangning Zhu, Yuxing Zhou, Zheng Wang, Juntao Yao, Yima Gu, Yuhui Yuan, Shixia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17473)  

**Abstract**: Given the central role of charts in scientific, business, and communication contexts, enhancing the chart understanding capabilities of vision-language models (VLMs) has become increasingly critical. A key limitation of existing VLMs lies in their inaccurate visual grounding of infographic elements, including charts and human-recognizable objects (HROs) such as icons and images. However, chart understanding often requires identifying relevant elements and reasoning over them. To address this limitation, we introduce OrionBench, a benchmark designed to support the development of accurate object detection models for charts and HROs in infographics. It contains 26,250 real and 78,750 synthetic infographics, with over 6.9 million bounding box annotations. These annotations are created by combining the model-in-the-loop and programmatic methods. We demonstrate the usefulness of OrionBench through three applications: 1) constructing a Thinking-with-Boxes scheme to boost the chart understanding performance of VLMs, 2) comparing existing object detection models, and 3) applying the developed detection model to document layout and UI element detection. 

---
# Diagnosing Vision Language Models' Perception by Leveraging Human Methods for Color Vision Deficiencies 

**Authors**: Kazuki Hayashi, Shintaro Ozaki, Yusuke Sakai, Hidetaka Kamigaito, Taro Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2505.17461)  

**Abstract**: Large-scale Vision Language Models (LVLMs) are increasingly being applied to a wide range of real-world multimodal applications, involving complex visual and linguistic reasoning. As these models become more integrated into practical use, they are expected to handle complex aspects of human interaction. Among these, color perception is a fundamental yet highly variable aspect of visual understanding. It differs across individuals due to biological factors such as Color Vision Deficiencies (CVDs), as well as differences in culture and language. Despite its importance, perceptual diversity has received limited attention. In our study, we evaluate LVLMs' ability to account for individual level perceptual variation using the Ishihara Test, a widely used method for detecting CVDs. Our results show that LVLMs can explain CVDs in natural language, but they cannot simulate how people with CVDs perceive color in image based tasks. These findings highlight the need for multimodal systems that can account for color perceptual diversity and support broader discussions on perceptual inclusiveness and fairness in multimodal AI. 

---
# Self-Training Large Language Models with Confident Reasoning 

**Authors**: Hyosoon Jang, Yunhui Jang, Sungjae Lee, Jungseul Ok, Sungsoo Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2505.17454)  

**Abstract**: Large language models (LLMs) have shown impressive performance by generating reasoning paths before final answers, but learning such a reasoning path requires costly human supervision. To address this issue, recent studies have explored self-training methods that improve reasoning capabilities using pseudo-labels generated by the LLMs themselves. Among these, confidence-based self-training fine-tunes LLMs to prefer reasoning paths with high-confidence answers, where confidence is estimated via majority voting. However, such methods exclusively focus on the quality of the final answer and may ignore the quality of the reasoning paths, as even an incorrect reasoning path leads to a correct answer by chance. Instead, we advocate the use of reasoning-level confidence to identify high-quality reasoning paths for self-training, supported by our empirical observations. We then propose a new self-training method, CORE-PO, that fine-tunes LLMs to prefer high-COnfidence REasoning paths through Policy Optimization. Our experiments show that CORE-PO improves the accuracy of outputs on four in-distribution and two out-of-distribution benchmarks, compared to existing self-training methods. 

---
# Debiasing CLIP: Interpreting and Correcting Bias in Attention Heads 

**Authors**: Wei Jie Yeo, Rui Mao, Moloud Abdar, Erik Cambria, Ranjan Satapathy  

**Link**: [PDF](https://arxiv.org/pdf/2505.17425)  

**Abstract**: Multimodal models like CLIP have gained significant attention due to their remarkable zero-shot performance across various tasks. However, studies have revealed that CLIP can inadvertently learn spurious associations between target variables and confounding factors. To address this, we introduce \textsc{Locate-Then-Correct} (LTC), a contrastive framework that identifies spurious attention heads in Vision Transformers via mechanistic insights and mitigates them through targeted ablation. Furthermore, LTC identifies salient, task-relevant attention heads, enabling the integration of discriminative features through orthogonal projection to improve classification performance. We evaluate LTC on benchmarks with inherent background and gender biases, achieving over a $>50\%$ gain in worst-group accuracy compared to non-training post-hoc baselines. Additionally, we visualize the representation of selected heads and find that the presented interpretation corroborates our contrastive mechanism for identifying both spurious and salient attention heads. Code available at this https URL. 

---
# Speechless: Speech Instruction Training Without Speech for Low Resource Languages 

**Authors**: Alan Dao, Dinh Bach Vu, Huy Hoang Ha, Tuan Le Duc Anh, Shreyas Gopal, Yue Heng Yeo, Warren Keng Hoong Low, Eng Siong Chng, Jia Qi Yip  

**Link**: [PDF](https://arxiv.org/pdf/2505.17417)  

**Abstract**: The rapid growth of voice assistants powered by large language models (LLM) has highlighted a need for speech instruction data to train these systems. Despite the abundance of speech recognition data, there is a notable scarcity of speech instruction data, which is essential for fine-tuning models to understand and execute spoken commands. Generating high-quality synthetic speech requires a good text-to-speech (TTS) model, which may not be available to low resource languages. Our novel approach addresses this challenge by halting synthesis at the semantic representation level, bypassing the need for TTS. We achieve this by aligning synthetic semantic representations with the pre-trained Whisper encoder, enabling an LLM to be fine-tuned on text instructions while maintaining the ability to understand spoken instructions during inference. This simplified training process is a promising approach to building voice assistant for low-resource languages. 

---
# LLM-based Generative Error Correction for Rare Words with Synthetic Data and Phonetic Context 

**Authors**: Natsuo Yamashita, Masaaki Yamamoto, Hiroaki Kokubo, Yohei Kawaguchi  

**Link**: [PDF](https://arxiv.org/pdf/2505.17410)  

**Abstract**: Generative error correction (GER) with large language models (LLMs) has emerged as an effective post-processing approach to improve automatic speech recognition (ASR) performance. However, it often struggles with rare or domain-specific words due to limited training data. Furthermore, existing LLM-based GER approaches primarily rely on textual information, neglecting phonetic cues, which leads to over-correction. To address these issues, we propose a novel LLM-based GER approach that targets rare words and incorporates phonetic information. First, we generate synthetic data to contain rare words for fine-tuning the GER model. Second, we integrate ASR's N-best hypotheses along with phonetic context to mitigate over-correction. Experimental results show that our method not only improves the correction of rare words but also reduces the WER and CER across both English and Japanese datasets. 

---
# Chart-to-Experience: Benchmarking Multimodal LLMs for Predicting Experiential Impact of Charts 

**Authors**: Seon Gyeom Kim, Jae Young Choi, Ryan Rossi, Eunyee Koh, Tak Yeon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.17374)  

**Abstract**: The field of Multimodal Large Language Models (MLLMs) has made remarkable progress in visual understanding tasks, presenting a vast opportunity to predict the perceptual and emotional impact of charts. However, it also raises concerns, as many applications of LLMs are based on overgeneralized assumptions from a few examples, lacking sufficient validation of their performance and effectiveness. We introduce Chart-to-Experience, a benchmark dataset comprising 36 charts, evaluated by crowdsourced workers for their impact on seven experiential factors. Using the dataset as ground truth, we evaluated capabilities of state-of-the-art MLLMs on two tasks: direct prediction and pairwise comparison of charts. Our findings imply that MLLMs are not as sensitive as human evaluators when assessing individual charts, but are accurate and reliable in pairwise comparisons. 

---
# Value-Guided Search for Efficient Chain-of-Thought Reasoning 

**Authors**: Kaiwen Wang, Jin Peng Zhou, Jonathan Chang, Zhaolin Gao, Nathan Kallus, Kianté Brantley, Wen Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.17373)  

**Abstract**: In this paper, we propose a simple and efficient method for value model training on long-context reasoning traces. Compared to existing process reward models (PRMs), our method does not require a fine-grained notion of "step," which is difficult to define for long-context reasoning models. By collecting a dataset of 2.5 million reasoning traces, we train a 1.5B token-level value model and apply it to DeepSeek models for improved performance with test-time compute scaling. We find that block-wise value-guided search (VGS) with a final weighted majority vote achieves better test-time scaling than standard methods such as majority voting or best-of-n. With an inference budget of 64 generations, VGS with DeepSeek-R1-Distill-1.5B achieves an average accuracy of 45.7% across four competition math benchmarks (AIME 2024 & 2025, HMMT Feb 2024 & 2025), reaching parity with o3-mini-medium. Moreover, VGS significantly reduces the inference FLOPs required to achieve the same performance of majority voting. Our dataset, model and codebase are open-sourced. 

---
# An End-to-End Approach for Child Reading Assessment in the Xhosa Language 

**Authors**: Sergio Chevtchenko, Nikhil Navas, Rafaella Vale, Franco Ubaudi, Sipumelele Lucwaba, Cally Ardington, Soheil Afshar, Mark Antoniou, Saeed Afshar  

**Link**: [PDF](https://arxiv.org/pdf/2505.17371)  

**Abstract**: Child literacy is a strong predictor of life outcomes at the subsequent stages of an individual's life. This points to a need for targeted interventions in vulnerable low and middle income populations to help bridge the gap between literacy levels in these regions and high income ones. In this effort, reading assessments provide an important tool to measure the effectiveness of these programs and AI can be a reliable and economical tool to support educators with this task. Developing accurate automatic reading assessment systems for child speech in low-resource languages poses significant challenges due to limited data and the unique acoustic properties of children's voices. This study focuses on Xhosa, a language spoken in South Africa, to advance child speech recognition capabilities. We present a novel dataset composed of child speech samples in Xhosa. The dataset is available upon request and contains ten words and letters, which are part of the Early Grade Reading Assessment (EGRA) system. Each recording is labeled with an online and cost-effective approach by multiple markers and a subsample is validated by an independent EGRA reviewer. This dataset is evaluated with three fine-tuned state-of-the-art end-to-end models: wav2vec 2.0, HuBERT, and Whisper. The results indicate that the performance of these models can be significantly influenced by the amount and balancing of the available training data, which is fundamental for cost-effective large dataset collection. Furthermore, our experiments indicate that the wav2vec 2.0 performance is improved by training on multiple classes at a time, even when the number of available samples is constrained. 

---
# DEL-ToM: Inference-Time Scaling for Theory-of-Mind Reasoning via Dynamic Epistemic Logic 

**Authors**: Yuheng Wu, Jianwen Xie, Denghui Zhang, Zhaozhuo Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17348)  

**Abstract**: Theory-of-Mind (ToM) tasks pose a unique challenge for small language models (SLMs) with limited scale, which often lack the capacity to perform deep social reasoning. In this work, we propose DEL-ToM, a framework that improves ToM reasoning through inference-time scaling rather than architectural changes. Our approach decomposes ToM tasks into a sequence of belief updates grounded in Dynamic Epistemic Logic (DEL), enabling structured and transparent reasoning. We train a verifier, called the Process Belief Model (PBM), to score each belief update step using labels generated automatically via a DEL simulator. During inference, candidate belief traces generated by a language model are evaluated by the PBM, and the highest-scoring trace is selected. This allows SLMs to emulate more deliberate reasoning by allocating additional compute at test time. Experiments across multiple model scales and benchmarks show that DEL-ToM consistently improves performance, demonstrating that verifiable belief supervision can significantly enhance ToM abilities of SLMs without retraining. 

---
# ECHO-LLaMA: Efficient Caching for High-Performance LLaMA Training 

**Authors**: Maryam Dialameh, Rezaul Karim, Hossein Rajabzadeh, Omar Mohamed Awad, Hyock Ju Kwon, Boxing Chen, Walid Ahmed, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17331)  

**Abstract**: This paper introduces ECHO-LLaMA, an efficient LLaMA architecture designed to improve both the training speed and inference throughput of LLaMA architectures while maintaining its learning capacity. ECHO-LLaMA transforms LLaMA models into shared KV caching across certain layers, significantly reducing KV computational complexity while maintaining or improving language performance. Experimental results demonstrate that ECHO-LLaMA achieves up to 77\% higher token-per-second throughput during training, up to 16\% higher Model FLOPs Utilization (MFU), and up to 14\% lower loss when trained on an equal number of tokens. Furthermore, on the 1.1B model, ECHO-LLaMA delivers approximately 7\% higher test-time throughput compared to the baseline. By introducing a computationally efficient adaptation mechanism, ECHO-LLaMA offers a scalable and cost-effective solution for pretraining and finetuning large language models, enabling faster and more resource-efficient training without compromising performance. 

---
# FS-DAG: Few Shot Domain Adapting Graph Networks for Visually Rich Document Understanding 

**Authors**: Amit Agarwal, Srikant Panda, Kulbhushan Pachauri  

**Link**: [PDF](https://arxiv.org/pdf/2505.17330)  

**Abstract**: In this work, we propose Few Shot Domain Adapting Graph (FS-DAG), a scalable and efficient model architecture for visually rich document understanding (VRDU) in few-shot settings. FS-DAG leverages domain-specific and language/vision specific backbones within a modular framework to adapt to diverse document types with minimal data. The model is robust to practical challenges such as handling OCR errors, misspellings, and domain shifts, which are critical in real-world deployments. FS-DAG is highly performant with less than 90M parameters, making it well-suited for complex real-world applications for Information Extraction (IE) tasks where computational resources are limited. We demonstrate FS-DAG's capability through extensive experiments for information extraction task, showing significant improvements in convergence speed and performance compared to state-of-the-art methods. Additionally, this work highlights the ongoing progress in developing smaller, more efficient models that do not compromise on performance. Code : this https URL 

---
# Analyzing Fine-Grained Alignment and Enhancing Vision Understanding in Multimodal Language Models 

**Authors**: Jiachen Jiang, Jinxin Zhou, Bo Peng, Xia Ning, Zhihui Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17316)  

**Abstract**: Achieving better alignment between vision embeddings and Large Language Models (LLMs) is crucial for enhancing the abilities of Multimodal LLMs (MLLMs), particularly for recent models that rely on powerful pretrained vision encoders and LLMs. A common approach to connect the pretrained vision encoder and LLM is through a projector applied after the vision encoder. However, the projector is often trained to enable the LLM to generate captions, and hence the mechanism by which LLMs understand each vision token remains unclear. In this work, we first investigate the role of the projector in compressing vision embeddings and aligning them with word embeddings. We show that the projector significantly compresses visual information, removing redundant details while preserving essential elements necessary for the LLM to understand visual content. We then examine patch-level alignment -- the alignment between each vision patch and its corresponding semantic words -- and propose a *multi-semantic alignment hypothesis*. Our analysis indicates that the projector trained by caption loss improves patch-level alignment but only to a limited extent, resulting in weak and coarse alignment. To address this issue, we propose *patch-aligned training* to efficiently enhance patch-level alignment. Our experiments show that patch-aligned training (1) achieves stronger compression capability and improved patch-level alignment, enabling the MLLM to generate higher-quality captions, (2) improves the MLLM's performance by 16% on referring expression grounding tasks, 4% on question-answering tasks, and 3% on modern instruction-following benchmarks when using the same supervised fine-tuning (SFT) setting. The proposed method can be easily extended to other multimodal models. 

---
# Longer Context, Deeper Thinking: Uncovering the Role of Long-Context Ability in Reasoning 

**Authors**: Wang Yang, Zirui Liu, Hongye Jin, Qingyu Yin, Vipin Chaudhary, Xiaotian Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.17315)  

**Abstract**: Recent language models exhibit strong reasoning capabilities, yet the influence of long-context capacity on reasoning remains underexplored. In this work, we hypothesize that current limitations in reasoning stem, in part, from insufficient long-context capacity, motivated by empirical observations such as (1) higher context window length often leads to stronger reasoning performance, and (2) failed reasoning cases resemble failed long-context cases. To test this hypothesis, we examine whether enhancing a model's long-context ability before Supervised Fine-Tuning (SFT) leads to improved reasoning performance. Specifically, we compared models with identical architectures and fine-tuning data but varying levels of long-context capacity. Our results reveal a consistent trend: models with stronger long-context capacity achieve significantly higher accuracy on reasoning benchmarks after SFT. Notably, these gains persist even on tasks with short input lengths, indicating that long-context training offers generalizable benefits for reasoning performance. These findings suggest that long-context modeling is not just essential for processing lengthy inputs, but also serves as a critical foundation for reasoning. We advocate for treating long-context capacity as a first-class objective in the design of future language models. 

---
# Attention with Trained Embeddings Provably Selects Important Tokens 

**Authors**: Diyuan Wu, Aleksandr Shevchenko, Samet Oymak, Marco Mondelli  

**Link**: [PDF](https://arxiv.org/pdf/2505.17282)  

**Abstract**: Token embeddings play a crucial role in language modeling but, despite this practical relevance, their theoretical understanding remains limited. Our paper addresses the gap by characterizing the structure of embeddings obtained via gradient descent. Specifically, we consider a one-layer softmax attention model with a linear head for binary classification, i.e., $\texttt{Softmax}( p^\top E_X^\top ) E_X v = \frac{ \sum_{i=1}^T \exp(p^\top E_{x_i}) E_{x_i}^\top v}{\sum_{j=1}^T \exp(p^\top E_{x_{j}}) }$, where $E_X = [ E_{x_1} , \dots, E_{x_T} ]^\top$ contains the embeddings of the input sequence, $p$ is the embedding of the $\mathrm{\langle cls \rangle}$ token and $v$ the output vector. First, we show that, already after a single step of gradient training with the logistic loss, the embeddings $E_X$ capture the importance of tokens in the dataset by aligning with the output vector $v$ proportionally to the frequency with which the corresponding tokens appear in the dataset. Then, after training $p$ via gradient flow until convergence, the softmax selects the important tokens in the sentence (i.e., those that are predictive of the label), and the resulting $\mathrm{\langle cls \rangle}$ embedding maximizes the margin for such a selection. Experiments on real-world datasets (IMDB, Yelp) exhibit a phenomenology close to that unveiled by our theory. 

---
# Zebra-Llama: Towards Extremely Efficient Hybrid Models 

**Authors**: Mingyu Yang, Mehdi Rezagholizadeh, Guihong Li, Vikram Appia, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2505.17272)  

**Abstract**: With the growing demand for deploying large language models (LLMs) across diverse applications, improving their inference efficiency is crucial for sustainable and democratized access. However, retraining LLMs to meet new user-specific requirements is prohibitively expensive and environmentally unsustainable. In this work, we propose a practical and scalable alternative: composing efficient hybrid language models from existing pre-trained models. Our approach, Zebra-Llama, introduces a family of 1B, 3B, and 8B hybrid models by combining State Space Models (SSMs) and Multi-head Latent Attention (MLA) layers, using a refined initialization and post-training pipeline to efficiently transfer knowledge from pre-trained Transformers. Zebra-Llama achieves Transformer-level accuracy with near-SSM efficiency using only 7-11B training tokens (compared to trillions of tokens required for pre-training) and an 8B teacher. Moreover, Zebra-Llama dramatically reduces KV cache size -down to 3.9%, 2%, and 2.73% of the original for the 1B, 3B, and 8B variants, respectively-while preserving 100%, 100%, and >97% of average zero-shot performance on LM Harness tasks. Compared to models like MambaInLLaMA, X-EcoMLA, Minitron, and Llamba, Zebra-Llama consistently delivers competitive or superior accuracy while using significantly fewer tokens, smaller teachers, and vastly reduced KV cache memory. Notably, Zebra-Llama-8B surpasses Minitron-8B in few-shot accuracy by 7% while using 8x fewer training tokens, over 12x smaller KV cache, and a smaller teacher (8B vs. 15B). It also achieves 2.6x-3.8x higher throughput (tokens/s) than MambaInLlama up to a 32k context length. We will release code and model checkpoints upon acceptance. 

---
# CHAOS: Chart Analysis with Outlier Samples 

**Authors**: Omar Moured, Yufan Chen, Ruiping Liu, Simon Reiß, Philip Torr, Jiaming Zhang, Rainer Stiefelhagen  

**Link**: [PDF](https://arxiv.org/pdf/2505.17235)  

**Abstract**: Charts play a critical role in data analysis and visualization, yet real-world applications often present charts with challenging or noisy features. However, "outlier charts" pose a substantial challenge even for Multimodal Large Language Models (MLLMs), which can struggle to interpret perturbed charts. In this work, we introduce CHAOS (CHart Analysis with Outlier Samples), a robustness benchmark to systematically evaluate MLLMs against chart perturbations. CHAOS encompasses five types of textual and ten types of visual perturbations, each presented at three levels of severity (easy, mid, hard) inspired by the study result of human evaluation. The benchmark includes 13 state-of-the-art MLLMs divided into three groups (i.e., general-, document-, and chart-specific models) according to the training scope and data. Comprehensive analysis involves two downstream tasks (ChartQA and Chart-to-Text). Extensive experiments and case studies highlight critical insights into robustness of models across chart perturbations, aiming to guide future research in chart understanding domain. Data and code are publicly available at: this http URL. 

---
# CHART-6: Human-Centered Evaluation of Data Visualization Understanding in Vision-Language Models 

**Authors**: Arnav Verma, Kushin Mukherjee, Christopher Potts, Elisa Kreiss, Judith E. Fan  

**Link**: [PDF](https://arxiv.org/pdf/2505.17202)  

**Abstract**: Data visualizations are powerful tools for communicating patterns in quantitative data. Yet understanding any data visualization is no small feat -- succeeding requires jointly making sense of visual, numerical, and linguistic inputs arranged in a conventionalized format one has previously learned to parse. Recently developed vision-language models are, in principle, promising candidates for developing computational models of these cognitive operations. However, it is currently unclear to what degree these models emulate human behavior on tasks that involve reasoning about data visualizations. This gap reflects limitations in prior work that has evaluated data visualization understanding in artificial systems using measures that differ from those typically used to assess these abilities in humans. Here we evaluated eight vision-language models on six data visualization literacy assessments designed for humans and compared model responses to those of human participants. We found that these models performed worse than human participants on average, and this performance gap persisted even when using relatively lenient criteria to assess model performance. Moreover, while relative performance across items was somewhat correlated between models and humans, all models produced patterns of errors that were reliably distinct from those produced by human participants. Taken together, these findings suggest significant opportunities for further development of artificial systems that might serve as useful models of how humans reason about data visualizations. All code and data needed to reproduce these results are available at: this https URL. 

---
# OCR-Reasoning Benchmark: Unveiling the True Capabilities of MLLMs in Complex Text-Rich Image Reasoning 

**Authors**: Mingxin Huang, Yongxin Shi, Dezhi Peng, Songxuan Lai, Zecheng Xie, Lianwen Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.17163)  

**Abstract**: Recent advancements in multimodal slow-thinking systems have demonstrated remarkable performance across diverse visual reasoning tasks. However, their capabilities in text-rich image reasoning tasks remain understudied due to the lack of a systematic benchmark. To address this gap, we propose OCR-Reasoning, a comprehensive benchmark designed to systematically assess Multimodal Large Language Models on text-rich image reasoning tasks. The benchmark comprises 1,069 human-annotated examples spanning 6 core reasoning abilities and 18 practical reasoning tasks in text-rich visual scenarios. Furthermore, unlike other text-rich image understanding benchmarks that only annotate the final answers, OCR-Reasoning also annotates the reasoning process simultaneously. With the annotated reasoning process and the final answers, OCR-Reasoning evaluates not only the final answers generated by models but also their reasoning processes, enabling a holistic analysis of their problem-solving abilities. Leveraging this benchmark, we conducted a comprehensive evaluation of state-of-the-art MLLMs. Our results demonstrate the limitations of existing methodologies. Notably, even state-of-the-art MLLMs exhibit substantial difficulties, with none achieving accuracy surpassing 50\% across OCR-Reasoning, indicating that the challenges of text-rich image reasoning are an urgent issue to be addressed. The benchmark and evaluation scripts are available at this https URL. 

---
# TrimR: Verifier-based Training-Free Thinking Compression for Efficient Test-Time Scaling 

**Authors**: Weizhe Lin, Xing Li, Zhiyuan Yang, Xiaojin Fu, Hui-Ling Zhen, Yaoyuan Wang, Xianzhi Yu, Wulong Liu, Xiaosong Li, Mingxuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2505.17155)  

**Abstract**: Large Reasoning Models (LRMs) demonstrate exceptional capability in tackling complex mathematical, logical, and coding tasks by leveraging extended Chain-of-Thought (CoT) reasoning. Test-time scaling methods, such as prolonging CoT with explicit token-level exploration, can push LRMs' accuracy boundaries, but they incur significant decoding overhead. A key inefficiency source is LRMs often generate redundant thinking CoTs, which demonstrate clear structured overthinking and underthinking patterns. Inspired by human cognitive reasoning processes and numerical optimization theories, we propose TrimR, a verifier-based, training-free, efficient framework for dynamic CoT compression to trim reasoning and enhance test-time scaling, explicitly tailored for production-level deployment. Our method employs a lightweight, pretrained, instruction-tuned verifier to detect and truncate redundant intermediate thoughts of LRMs without any LRM or verifier fine-tuning. We present both the core algorithm and asynchronous online system engineered for high-throughput industrial applications. Empirical evaluations on Ascend NPUs and vLLM show that our framework delivers substantial gains in inference efficiency under large-batch workloads. In particular, on the four MATH500, AIME24, AIME25, and GPQA benchmarks, the reasoning runtime of Pangu-R-38B, QwQ-32B, and DeepSeek-R1-Distill-Qwen-32B is improved by up to 70% with negligible impact on accuracy. 

---
# Robustifying Vision-Language Models via Dynamic Token Reweighting 

**Authors**: Tanqiu Jiang, Jiacheng Liang, Rongyi Zhu, Jiawei Zhou, Fenglong Ma, Ting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17132)  

**Abstract**: Large vision-language models (VLMs) are highly vulnerable to jailbreak attacks that exploit visual-textual interactions to bypass safety guardrails. In this paper, we present DTR, a novel inference-time defense that mitigates multimodal jailbreak attacks through optimizing the model's key-value (KV) caches. Rather than relying on curated safety-specific data or costly image-to-text conversion, we introduce a new formulation of the safety-relevant distributional shift induced by the visual modality. This formulation enables DTR to dynamically adjust visual token weights, minimizing the impact of adversarial visual inputs while preserving the model's general capabilities and inference efficiency. Extensive evaluation across diverse VLMs and attack benchmarks demonstrates that \sys outperforms existing defenses in both attack robustness and benign task performance, marking the first successful application of KV cache optimization for safety enhancement in multimodal foundation models. The code for replicating DTR is available: this https URL (warning: this paper contains potentially harmful content generated by VLMs.) 

---
# Mitigating Cyber Risk in the Age of Open-Weight LLMs: Policy Gaps and Technical Realities 

**Authors**: Alfonso de Gregorio  

**Link**: [PDF](https://arxiv.org/pdf/2505.17109)  

**Abstract**: Open-weight general-purpose AI (GPAI) models offer significant benefits but also introduce substantial cybersecurity risks, as demonstrated by the offensive capabilities of models like DeepSeek-R1 in evaluations such as MITRE's OCCULT. These publicly available models empower a wider range of actors to automate and scale cyberattacks, challenging traditional defence paradigms and regulatory approaches. This paper analyzes the specific threats -- including accelerated malware development and enhanced social engineering -- magnified by open-weight AI release. We critically assess current regulations, notably the EU AI Act and the GPAI Code of Practice, identifying significant gaps stemming from the loss of control inherent in open distribution, which renders many standard security mitigations ineffective. We propose a path forward focusing on evaluating and controlling specific high-risk capabilities rather than entire models, advocating for pragmatic policy interpretations for open-weight systems, promoting defensive AI innovation, and fostering international collaboration on standards and cyber threat intelligence (CTI) sharing to ensure security without unduly stifling open technological progress. 

---
# CAMA: Enhancing Multimodal In-Context Learning with Context-Aware Modulated Attention 

**Authors**: Yanshu Li, JianJiang Yang, Bozheng Li, Ruixiang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17097)  

**Abstract**: Multimodal in-context learning (ICL) enables large vision-language models (LVLMs) to efficiently adapt to novel tasks, supporting a wide array of real-world applications. However, multimodal ICL remains unstable, and current research largely focuses on optimizing sequence configuration while overlooking the internal mechanisms of LVLMs. In this work, we first provide a theoretical analysis of attentional dynamics in multimodal ICL and identify three core limitations of standard attention that ICL impair performance. To address these challenges, we propose Context-Aware Modulated Attention (CAMA), a simple yet effective plug-and-play method for directly calibrating LVLM attention logits. CAMA is training-free and can be seamlessly applied to various open-source LVLMs. We evaluate CAMA on four LVLMs across six benchmarks, demonstrating its effectiveness and generality. CAMA opens new opportunities for deeper exploration and targeted utilization of LVLM attention dynamics to advance multimodal reasoning. 

---
# Voicing Personas: Rewriting Persona Descriptions into Style Prompts for Controllable Text-to-Speech 

**Authors**: Yejin Lee, Jaehoon Kang, Kyuhong Shim  

**Link**: [PDF](https://arxiv.org/pdf/2505.17093)  

**Abstract**: In this paper, we propose a novel framework to control voice style in prompt-based, controllable text-to-speech systems by leveraging textual personas as voice style prompts. We present two persona rewriting strategies to transform generic persona descriptions into speech-oriented prompts, enabling fine-grained manipulation of prosodic attributes such as pitch, emotion, and speaking rate. Experimental results demonstrate that our methods enhance the naturalness, clarity, and consistency of synthesized speech. Finally, we analyze implicit social biases introduced by LLM-based rewriting, with a focus on gender. We underscore voice style as a crucial factor for persona-driven AI dialogue systems. 

---
# From Weak Labels to Strong Results: Utilizing 5,000 Hours of Noisy Classroom Transcripts with Minimal Accurate Data 

**Authors**: Ahmed Adel Attia, Dorottya Demszky, Jing Liu, Carol Espy-Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2505.17088)  

**Abstract**: Recent progress in speech recognition has relied on models trained on vast amounts of labeled data. However, classroom Automatic Speech Recognition (ASR) faces the real-world challenge of abundant weak transcripts paired with only a small amount of accurate, gold-standard data. In such low-resource settings, high transcription costs make re-transcription impractical. To address this, we ask: what is the best approach when abundant inexpensive weak transcripts coexist with limited gold-standard data, as is the case for classroom speech data? We propose Weakly Supervised Pretraining (WSP), a two-step process where models are first pretrained on weak transcripts in a supervised manner, and then fine-tuned on accurate data. Our results, based on both synthetic and real weak transcripts, show that WSP outperforms alternative methods, establishing it as an effective training methodology for low-resource ASR in real-world scenarios. 

---
# GSDFuse: Capturing Cognitive Inconsistencies from Multi-Dimensional Weak Signals in Social Media Steganalysis 

**Authors**: Kaibo Huang, Zipei Zhang, Yukun Wei, TianXin Zhang, Zhongliang Yang, Linna Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.17085)  

**Abstract**: The ubiquity of social media platforms facilitates malicious linguistic steganography, posing significant security risks. Steganalysis is profoundly hindered by the challenge of identifying subtle cognitive inconsistencies arising from textual fragmentation and complex dialogue structures, and the difficulty in achieving robust aggregation of multi-dimensional weak signals, especially given extreme steganographic sparsity and sophisticated steganography. These core detection difficulties are compounded by significant data imbalance. This paper introduces GSDFuse, a novel method designed to systematically overcome these obstacles. GSDFuse employs a holistic approach, synergistically integrating hierarchical multi-modal feature engineering to capture diverse signals, strategic data augmentation to address sparsity, adaptive evidence fusion to intelligently aggregate weak signals, and discriminative embedding learning to enhance sensitivity to subtle inconsistencies. Experiments on social media datasets demonstrate GSDFuse's state-of-the-art (SOTA) performance in identifying sophisticated steganography within complex dialogue environments. The source code for GSDFuse is available at this https URL. 

---
# Safety Alignment Can Be Not Superficial With Explicit Safety Signals 

**Authors**: Jianwei Li, Jung-Eng Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.17072)  

**Abstract**: Recent studies on the safety alignment of large language models (LLMs) have revealed that existing approaches often operate superficially, leaving models vulnerable to various adversarial attacks. Despite their significance, these studies generally fail to offer actionable solutions beyond data augmentation for achieving more robust safety mechanisms. This paper identifies a fundamental cause of this superficiality: existing alignment approaches often presume that models can implicitly learn a safety-related reasoning task during the alignment process, enabling them to refuse harmful requests. However, the learned safety signals are often diluted by other competing objectives, leading models to struggle with drawing a firm safety-conscious decision boundary when confronted with adversarial attacks. Based on this observation, by explicitly introducing a safety-related binary classification task and integrating its signals with our attention and decoding strategies, we eliminate this ambiguity and allow models to respond more responsibly to malicious queries. We emphasize that, with less than 0.2x overhead cost, our approach enables LLMs to assess the safety of both the query and the previously generated tokens at each necessary generating step. Extensive experiments demonstrate that our method significantly improves the resilience of LLMs against various adversarial attacks, offering a promising pathway toward more robust generative AI systems. 

---
# Exploring EFL Secondary Students' AI-generated Text Editing While Composition Writing 

**Authors**: David James Woo, Yangyang Yu, Kai Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.17041)  

**Abstract**: Generative Artificial Intelligence is transforming how English as a foreign language students write. Still, little is known about how students manipulate text generated by generative AI during the writing process. This study investigates how EFL secondary school students integrate and modify AI-generated text when completing an expository writing task. The study employed an exploratory mixed-methods design. Screen recordings were collected from 29 Hong Kong secondary school students who attended an AI-assisted writing workshop and recorded their screens while using generative AI to write an article. Content analysis with hierarchical coding and thematic analysis with a multiple case study approach were adopted to analyze the recordings. 15 types of AI-generated text edits across seven categories were identified from the recordings. Notably, AI-initiated edits from iOS and Google Docs emerged as unanticipated sources of AI-generated text. A thematic analysis revealed four patterns of students' editing behaviors based on planning and drafting direction: planning with top-down drafting and revising; top-down drafting and revising without planning; planning with bottom-up drafting and revising; and bottom-up drafting and revising without planning. Network graphs illustrate cases of each pattern, demonstrating that students' interactions with AI-generated text involve more complex cognitive processes than simple text insertion. The findings challenge assumptions about students' passive, simplistic use of generative AI tools and have implications for developing explicit instructional approaches to teaching AI-generated text editing strategies in the AFL writing pedagogy. 

---
# Generalizing Large Language Model Usability Across Resource-Constrained 

**Authors**: Yun-Da Tsai  

**Link**: [PDF](https://arxiv.org/pdf/2505.17040)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success across a wide range of natural language tasks, and recent efforts have sought to extend their capabilities to multimodal domains and resource-constrained environments. However, existing approaches often rely on costly supervised fine-tuning or assume fixed training conditions, limiting their generalization when facing unseen modalities, limited data, or restricted compute resources. This dissertation presents a systematic study toward generalizing LLM usability under real-world constraints. First, it introduces a robust text-centric alignment framework that enables LLMs to seamlessly integrate diverse modalities-including text, images, tables, and any modalities - via natural language interfaces. This approach supports in-context adaptation to unseen or dynamically changing modalities without requiring retraining. To enhance robustness against noisy and missing modalities, an adversarial prompting technique is proposed, generating semantically challenging perturbations at the prompt level to stress-test model reliability. Beyond multimodal setting, the dissertation investigates inference-time optimization strategies for LLMs, leveraging prompt search and uncertainty quantification to improve performance without additional model training. This perspective offers an efficient alternative to scaling model parameters or retraining from scratch. Additionally, the work addresses low-resource domains such as Verilog code generation by designing correct-by-construction synthetic data pipelines and logic-enhanced reasoning models, achieving state-of-the-art performance with minimal data. Together, these contributions form a unified effort to enhance the adaptability, scalability, and efficiency of large language models under practical constraints. 

---
