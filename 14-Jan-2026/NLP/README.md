# Modeling LLM Agent Reviewer Dynamics in Elo-Ranked Review System 

**Authors**: Hsiang-Wei Huang, Junbin Lu, Kuang-Ming Chen, Jenq-Neng Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08829)  

**Abstract**: In this work, we explore the Large Language Model (LLM) agent reviewer dynamics in an Elo-ranked review system using real-world conference paper submissions. Multiple LLM agent reviewers with different personas are engage in multi round review interactions moderated by an Area Chair. We compare a baseline setting with conditions that incorporate Elo ratings and reviewer memory. Our simulation results showcase several interesting findings, including how incorporating Elo improves Area Chair decision accuracy, as well as reviewers' adaptive review strategy that exploits our Elo system without improving review effort. Our code is available at this https URL. 

---
# Multiplex Thinking: Reasoning via Token-wise Branch-and-Merge 

**Authors**: Yao Tang, Li Dong, Yaru Hao, Qingxiu Dong, Furu Wei, Jiatao Gu  

**Link**: [PDF](https://arxiv.org/pdf/2601.08808)  

**Abstract**: Large language models often solve complex reasoning tasks more effectively with Chain-of-Thought (CoT), but at the cost of long, low-bandwidth token sequences. Humans, by contrast, often reason softly by maintaining a distribution over plausible next steps. Motivated by this, we propose Multiplex Thinking, a stochastic soft reasoning mechanism that, at each thinking step, samples K candidate tokens and aggregates their embeddings into a single continuous multiplex token. This preserves the vocabulary embedding prior and the sampling dynamics of standard discrete generation, while inducing a tractable probability distribution over multiplex rollouts. Consequently, multiplex trajectories can be directly optimized with on-policy reinforcement learning (RL). Importantly, Multiplex Thinking is self-adaptive: when the model is confident, the multiplex token is nearly discrete and behaves like standard CoT; when it is uncertain, it compactly represents multiple plausible next steps without increasing sequence length. Across challenging math reasoning benchmarks, Multiplex Thinking consistently outperforms strong discrete CoT and RL baselines from Pass@1 through Pass@1024, while producing shorter sequences. The code and checkpoints are available at this https URL. 

---
# Spatial Context Improves the Integration of Text with Remote Sensing for Mapping Environmental Variables 

**Authors**: Valerie Zermatten, Chiara Vanalli, Gencer Sumbul, Diego Marcos, Devis Tuia  

**Link**: [PDF](https://arxiv.org/pdf/2601.08750)  

**Abstract**: Recent developments in natural language processing highlight text as an emerging data source for ecology. Textual resources carry unique information that can be used in complementarity with geospatial data sources, thus providing insights at the local scale into environmental conditions and properties hidden from more traditional data sources. Leveraging textual information in a spatial context presents several challenges. First, the contribution of textual data remains poorly defined in an ecological context, and it is unclear for which tasks it should be incorporated. Unlike ubiquitous satellite imagery or environmental covariates, the availability of textual data is sparse and irregular; its integration with geospatial data is not straightforward. In response to these challenges, this work proposes an attention-based approach that combines aerial imagery and geolocated text within a spatial neighbourhood, i.e. integrating contributions from several nearby observations. Our approach combines vision and text representations with a geolocation encoding, with an attention-based module that dynamically selects spatial neighbours that are useful for predictive this http URL proposed approach is applied to the EcoWikiRS dataset, which combines high-resolution aerial imagery with sentences extracted from Wikipedia describing local environmental conditions across Switzerland. Our model is evaluated on the task of predicting 103 environmental variables from the SWECO25 data cube. Our approach consistently outperforms single-location or unimodal, i.e. image-only or text-only, baselines. When analysing variables by thematic groups, results show a significant improvement in performance for climatic, edaphic, population and land use/land cover variables, underscoring the benefit of including the spatial context when combining text and image data. 

---
# To Retrieve or To Think? An Agentic Approach for Context Evolution 

**Authors**: Rubing Chen, Jian Wang, Wenjie Li, Xiao-Yong Wei, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.08747)  

**Abstract**: Current context augmentation methods, such as retrieval-augmented generation, are essential for solving knowledge-intensive reasoning this http URL, they typically adhere to a rigid, brute-force strategy that executes retrieval at every step. This indiscriminate approach not only incurs unnecessary computational costs but also degrades performance by saturating the context with irrelevant noise. To address these limitations, we introduce Agentic Context Evolution (ACE), a framework inspired by human metacognition that dynamically determines whether to seek new evidence or reason with existing knowledge. ACE employs a central orchestrator agent to make decisions strategically via majority this http URL aims to alternate between activating a retriever agent for external retrieval and a reasoner agent for internal analysis and refinement. By eliminating redundant retrieval steps, ACE maintains a concise and evolved context. Extensive experiments on challenging multi-hop QA benchmarks demonstrate that ACE significantly outperforms competitive baselines in accuracy while achieving efficient token this http URL work provides valuable insights into advancing context-evolved generation for complex, knowledge-intensive tasks. 

---
# TableCache: Primary Foreign Key Guided KV Cache Precomputation for Low Latency Text-to-SQL 

**Authors**: Jinbo Su, Yuxuan Hu, Cuiping Li, Hong Chen, Jia Li, Lintao Ma, Jing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08743)  

**Abstract**: In Text-to-SQL tasks, existing LLM-based methods often include extensive database schemas in prompts, leading to long context lengths and increased prefilling latency. While user queries typically focus on recurrent table sets-offering an opportunity for KV cache sharing across queries-current inference engines, such as SGLang and vLLM, generate redundant prefix cache copies when processing user queries with varying table orders. To address this inefficiency, we propose precomputing table representations as KV caches offline and querying the required ones online. A key aspect of our approach is the computation of table caches while preserving primary foreign key relationships between tables. Additionally, we construct a Table Trie structure to facilitate efficient KV cache lookups during inference. To enhance cache performance, we introduce a cache management system with a query reranking strategy to improve cache hit rates and a computation loading pipeline for parallelizing model inference and cache loading. Experimental results show that our proposed TableCache achieves up to a 3.62x speedup in Time to First Token (TTFT) with negligible performance degradation. 

---
# Inferring Latent Intentions: Attributional Natural Language Inference in LLM Agents 

**Authors**: Xin Quan, Jiafeng Xiong, Marco Valentino, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2601.08742)  

**Abstract**: Attributional inference, the ability to predict latent intentions behind observed actions, is a critical yet underexplored capability for large language models (LLMs) operating in multi-agent environments. Traditional natural language inference (NLI), in fact, fails to capture the nuanced, intention-driven reasoning essential for complex interactive systems. To address this gap, we introduce Attributional NLI (Att-NLI), a framework that extends NLI with principles from social psychology to assess an agent's capacity for abductive intentional inference (generating hypotheses about latent intentions), and subsequent deductive verification (drawing valid logical conclusions). We instantiate Att-NLI via a textual game, Undercover-V, experimenting with three types of LLM agents with varying reasoning capabilities and access to external tools: a standard NLI agent using only deductive inference, an Att-NLI agent employing abductive-deductive inference, and a neuro-symbolic Att-NLI agent performing abductive-deductive inference with external theorem provers. Extensive experiments demonstrate a clear hierarchy of attributional inference capabilities, with neuro-symbolic agents consistently outperforming others, achieving an average win rate of 17.08%. Our results underscore the role that Att-NLI can play in developing agents with sophisticated reasoning capabilities, highlighting, at the same time, the potential impact of neuro-symbolic AI in building rational LLM agents acting in multi-agent environments. 

---
# From Rows to Reasoning: A Retrieval-Augmented Multimodal Framework for Spreadsheet Understanding 

**Authors**: Anmol Gulati, Sahil Sen, Waqar Sarguroh, Kevin Paul  

**Link**: [PDF](https://arxiv.org/pdf/2601.08741)  

**Abstract**: Large Language Models (LLMs) struggle to reason over large-scale enterprise spreadsheets containing thousands of numeric rows, multiple linked sheets, and embedded visual content such as charts and receipts. Prior state-of-the-art spreadsheet reasoning approaches typically rely on single-sheet compression or full-context encoding, which limits scalability and fails to reflect how real users interact with complex, multimodal workbooks. We introduce FRTR-Bench, the first large-scale benchmark for multimodal spreadsheet reasoning, comprising 30 enterprise-grade Excel workbooks spanning nearly four million cells and more than 50 embedded images. To address these challenges, we present From Rows to Reasoning (FRTR), an advanced, multimodal retrieval-augmented generation framework that decomposes Excel workbooks into granular row, column, and block embeddings, employs hybrid lexical-dense retrieval with Reciprocal Rank Fusion (RRF), and integrates multimodal embeddings to reason over both numerical and visual information. We tested FRTR on six LLMs, achieving 74% answer accuracy on FRTR-Bench with Claude Sonnet 4.5, a substantial improvement over prior state-of-the-art approaches that reached only 24%. On the SpreadsheetLLM benchmark, FRTR achieved 87% accuracy with GPT-5 while reducing token usage by roughly 50% compared to context-compression methods. 

---
# PrivGemo: Privacy-Preserving Dual-Tower Graph Retrieval for Empowering LLM Reasoning with Memory Augmentation 

**Authors**: Xingyu Tan, Xiaoyang Wang, Qing Liu, Xiwei Xu, Xin Yuan, Liming Zhu, Wenjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08739)  

**Abstract**: Knowledge graphs (KGs) provide structured evidence that can ground large language model (LLM) reasoning for knowledge-intensive question answering. However, many practical KGs are private, and sending retrieved triples or exploration traces to closed-source LLM APIs introduces leakage risk. Existing privacy treatments focus on masking entity names, but they still face four limitations: structural leakage under semantic masking, uncontrollable remote interaction, fragile multi-hop and multi-entity reasoning, and limited experience reuse for stability and efficiency. To address these issues, we propose PrivGemo, a privacy-preserving retrieval-augmented framework for KG-grounded reasoning with memory-guided exposure control. PrivGemo uses a dual-tower design to keep raw KG knowledge local while enabling remote reasoning over an anonymized view that goes beyond name masking to limit both semantic and structural exposure. PrivGemo supports multi-hop, multi-entity reasoning by retrieving anonymized long-hop paths that connect all topic entities, while keeping grounding and verification on the local KG. A hierarchical controller and a privacy-aware experience memory further reduce unnecessary exploration and remote interactions. Comprehensive experiments on six benchmarks show that PrivGemo achieves overall state-of-the-art results, outperforming the strongest baseline by up to 17.1%. Furthermore, PrivGemo enables smaller models (e.g., Qwen3-4B) to achieve reasoning performance comparable to that of GPT-4-Turbo. 

---
# RAGShaper: Eliciting Sophisticated Agentic RAG Skills via Automated Data Synthesis 

**Authors**: Zhengwei Tao, Bo Li, Jialong Wu, Guochen Yan, Huanyao Zhang, Jiahao Xu, Haitao Mi, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08699)  

**Abstract**: Agentic Retrieval-Augmented Generation (RAG) empowers large language models to autonomously plan and retrieve information for complex problem-solving. However, the development of robust agents is hindered by the scarcity of high-quality training data that reflects the noise and complexity of real-world retrieval environments. Conventional manual annotation is unscalable and often fails to capture the dynamic reasoning strategies required to handle retrieval failures. To bridge this gap, we introduce RAGShaper, a novel data synthesis framework designed to automate the construction of RAG tasks and robust agent trajectories. RAGShaper incorporates an InfoCurator to build dense information trees enriched with adversarial distractors spanning Perception and Cognition levels. Furthermore, we propose a constrained navigation strategy that forces a teacher agent to confront these distractors, thereby eliciting trajectories that explicitly demonstrate error correction and noise rejection. Comprehensive experiments confirm that models trained on our synthesized corpus significantly outperform existing baselines, exhibiting superior robustness in noise-intensive and complex retrieval tasks. 

---
# Nationality and Region Prediction from Names: A Comparative Study of Neural Models and Large Language Models 

**Authors**: Keito Inoshita  

**Link**: [PDF](https://arxiv.org/pdf/2601.08692)  

**Abstract**: Predicting nationality from personal names has practical value in marketing, demographic research, and genealogical studies. Conventional neural models learn statistical correspondences between names and nationalities from task-specific training data, posing challenges in generalizing to low-frequency nationalities and distinguishing similar nationalities within the same region. Large language models (LLMs) have the potential to address these challenges by leveraging world knowledge acquired during pre-training. In this study, we comprehensively compare neural models and LLMs on nationality prediction, evaluating six neural models and six LLM prompting strategies across three granularity levels (nationality, region, and continent), with frequency-based stratified analysis and error analysis. Results show that LLMs outperform neural models at all granularity levels, with the gap narrowing as granularity becomes coarser. Simple machine learning methods exhibit the highest frequency robustness, while pre-trained models and LLMs show degradation for low-frequency nationalities. Error analysis reveals that LLMs tend to make ``near-miss'' errors, predicting the correct region even when nationality is incorrect, whereas neural models exhibit more cross-regional errors and bias toward high-frequency classes. These findings indicate that LLM superiority stems from world knowledge, model selection should consider required granularity, and evaluation should account for error quality beyond accuracy. 

---
# QuantEval: A Benchmark for Financial Quantitative Tasks in Large Language Models 

**Authors**: Zhaolu Kang, Junhao Gong, Wenqing Hu, Shuo Yin, Kehan Jiang, Zhicheng Fang, Yingjie He, Chunlei Meng, Rong Fu, Dongyang Chen, Leqi Zheng, Eric Hanchen Jiang, Yunfei Feng, Yitong Leng, Junfan Zhu, Xiaoyou Chen, Xi Yang, Richeng Xuan  

**Link**: [PDF](https://arxiv.org/pdf/2601.08689)  

**Abstract**: Large Language Models (LLMs) have shown strong capabilities across many domains, yet their evaluation in financial quantitative tasks remains fragmented and mostly limited to knowledge-centric question answering. We introduce QuantEval, a benchmark that evaluates LLMs across three essential dimensions of quantitative finance: knowledge-based QA, quantitative mathematical reasoning, and quantitative strategy coding. Unlike prior financial benchmarks, QuantEval integrates a CTA-style backtesting framework that executes model-generated strategies and evaluates them using financial performance metrics, enabling a more realistic assessment of quantitative coding ability. We evaluate some state-of-the-art open-source and proprietary LLMs and observe substantial gaps to human experts, particularly in reasoning and strategy coding. Finally, we conduct large-scale supervised fine-tuning and reinforcement learning experiments on domain-aligned data, demonstrating consistent improvements. We hope QuantEval will facilitate research on LLMs' quantitative finance capabilities and accelerate their practical adoption in real-world trading workflows. We additionally release the full deterministic backtesting configuration (asset universe, cost model, and metric definitions) to ensure strict reproducibility. 

---
# Lessons from the Field: An Adaptable Lifecycle Approach to Applied Dialogue Summarization 

**Authors**: Kushal Chawla, Chenyang Zhu, Pengshan Cai, Sangwoo Cho, Scott Novotney, Ayushman Singh, Jonah Lewis, Keasha Safewright, Alfy Samuel, Erin Babinsky, Shi-Xiong Zhang, Sambit Sahu  

**Link**: [PDF](https://arxiv.org/pdf/2601.08682)  

**Abstract**: Summarization of multi-party dialogues is a critical capability in industry, enhancing knowledge transfer and operational effectiveness across many domains. However, automatically generating high-quality summaries is challenging, as the ideal summary must satisfy a set of complex, multi-faceted requirements. While summarization has received immense attention in research, prior work has primarily utilized static datasets and benchmarks, a condition rare in practical scenarios where requirements inevitably evolve. In this work, we present an industry case study on developing an agentic system to summarize multi-party interactions. We share practical insights spanning the full development lifecycle to guide practitioners in building reliable, adaptable summarization systems, as well as to inform future research, covering: 1) robust methods for evaluation despite evolving requirements and task subjectivity, 2) component-wise optimization enabled by the task decomposition inherent in an agentic architecture, 3) the impact of upstream data bottlenecks, and 4) the realities of vendor lock-in due to the poor transferability of LLM prompts. 

---
# Analyzing Bias in False Refusal Behavior of Large Language Models for Hate Speech Detoxification 

**Authors**: Kyuri Im, Shuzhou Yuan, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2601.08668)  

**Abstract**: While large language models (LLMs) have increasingly been applied to hate speech detoxification, the prompts often trigger safety alerts, causing LLMs to refuse the task. In this study, we systematically investigate false refusal behavior in hate speech detoxification and analyze the contextual and linguistic biases that trigger such refusals. We evaluate nine LLMs on both English and multilingual datasets, our results show that LLMs disproportionately refuse inputs with higher semantic toxicity and those targeting specific groups, particularly nationality, religion, and political ideology. Although multilingual datasets exhibit lower overall false refusal rates than English datasets, models still display systematic, language-dependent biases toward certain targets. Based on these findings, we propose a simple cross-translation strategy, translating English hate speech into Chinese for detoxification and back, which substantially reduces false refusals while preserving the original content, providing an effective and lightweight mitigation approach. 

---
# RULERS: Locked Rubrics and Evidence-Anchored Scoring for Robust LLM Evaluation 

**Authors**: Yihan Hong, Huaiyuan Yao, Bolin Shen, Wanpeng Xu, Hua Wei, Yushun Dong  

**Link**: [PDF](https://arxiv.org/pdf/2601.08654)  

**Abstract**: The LLM-as-a-Judge paradigm promises scalable rubric-based evaluation, yet aligning frozen black-box models with human standards remains a challenge due to inherent generation stochasticity. We reframe judge alignment as a criteria transfer problem and isolate three recurrent failure modes: rubric instability caused by prompt sensitivity, unverifiable reasoning that lacks auditable evidence, and scale misalignment with human grading boundaries. To address these issues, we introduce RULERS (Rubric Unification, Locking, and Evidence-anchored Robust Scoring), a compiler-executor framework that transforms natural language rubrics into executable specifications. RULERS operates by compiling criteria into versioned immutable bundles, enforcing structured decoding with deterministic evidence verification, and applying lightweight Wasserstein-based post-hoc calibration, all without updating model parameters. Extensive experiments on essay and summarization benchmarks demonstrate that RULERS significantly outperforms representative baselines in human agreement, maintains strong stability against adversarial rubric perturbations, and enables smaller models to rival larger proprietary judges. Overall, our results suggest that reliable LLM judging requires executable rubrics, verifiable evidence, and calibrated scales rather than prompt phrasing alone. Code is available at this https URL. 

---
# Safe Language Generation in the Limit 

**Authors**: Antonios Anastasopoulos, Giuseppe Ateniese, Evgenios M. Kornaropoulos  

**Link**: [PDF](https://arxiv.org/pdf/2601.08648)  

**Abstract**: Recent results in learning a language in the limit have shown that, although language identification is impossible, language generation is tractable. As this foundational area expands, we need to consider the implications of language generation in real-world settings.
This work offers the first theoretical treatment of safe language generation. Building on the computational paradigm of learning in the limit, we formalize the tasks of safe language identification and generation. We prove that under this model, safe language identification is impossible, and that safe language generation is at least as hard as (vanilla) language identification, which is also impossible. Last, we discuss several intractable and tractable cases. 

---
# A Parallel Cross-Lingual Benchmark for Multimodal Idiomaticity Understanding 

**Authors**: Dilara Torunoğlu-Selamet, Dogukan Arslan, Rodrigo Wilkens, Wei He, Doruk Eryiğit, Thomas Pickard, Adriana S. Pagano, Aline Villavicencio, Gülşen Eryiğit, Ágnes Abuczki, Aida Cardoso, Alesia Lazarenka, Dina Almassova, Amalia Mendes, Anna Kanellopoulou, Antoni Brosa-Rodríguez, Baiba Saulite, Beata Wojtowicz, Bolette Pedersen, Carlos Manuel Hidalgo-Ternero, Chaya Liebeskind, Danka Jokić, Diego Alves, Eleni Triantafyllidi, Erik Velldal, Fred Philippy, Giedre Valunaite Oleskeviciene, Ieva Rizgeliene, Inguna Skadina, Irina Lobzhanidze, Isabell Stinessen Haugen, Jauza Akbar Krito, Jelena M. Marković, Johanna Monti, Josue Alejandro Sauca, Kaja Dobrovoljc, Kingsley O. Ugwuanyi, Laura Rituma, Lilja Øvrelid, Maha Tufail Agro, Manzura Abjalova, Maria Chatzigrigoriou, María del Mar Sánchez Ramos, Marija Pendevska, Masoumeh Seyyedrezaei, Mehrnoush Shamsfard, Momina Ahsan, Muhammad Ahsan Riaz Khan, Nathalie Carmen Hau Norman, Nilay Erdem Ayyıldız, Nina Hosseini-Kivanani, Noémi Ligeti-Nagy, Numaan Naeem, Olha Kanishcheva, Olha Yatsyshyna, Daniil Orel, Petra Giommarelli, Petya Osenova, Radovan Garabik, Regina E. Semou, Rozane Rebechi, Salsabila Zahirah Pranida, Samia Touileb, Sanni Nimb, Sarfraz Ahmad, Sarvinoz Nematkhonova, Shahar Golan, Shaoxiong Ji, Sopuruchi Christian Aboh, Srdjan Sucur, Stella Markantonatou, Sussi Olsen, Vahide Tajalli, Veronika Lipp, Voula Giouli, Yelda Yeşildal Eraydın, Zahra Saaberi, Zhuohan Xie  

**Link**: [PDF](https://arxiv.org/pdf/2601.08645)  

**Abstract**: Potentially idiomatic expressions (PIEs) construe meanings inherently tied to the everyday experience of a given language community. As such, they constitute an interesting challenge for assessing the linguistic (and to some extent cultural) capabilities of NLP systems. In this paper, we present XMPIE, a parallel multilingual and multimodal dataset of potentially idiomatic expressions. The dataset, containing 34 languages and over ten thousand items, allows comparative analyses of idiomatic patterns among language-specific realisations and preferences in order to gather insights about shared cultural aspects. This parallel dataset allows to evaluate model performance for a given PIE in different languages and whether idiomatic understanding in one language can be transferred to another. Moreover, the dataset supports the study of PIEs across textual and visual modalities, to measure to what extent PIE understanding in one modality transfers or implies in understanding in another modality (text vs. image). The data was created by language experts, with both textual and visual components crafted under multilingual guidelines, and each PIE is accompanied by five images representing a spectrum from idiomatic to literal meanings, including semantically related and random distractors. The result is a high-quality benchmark for evaluating multilingual and multimodal idiomatic language understanding. 

---
# Moral Lenses, Political Coordinates: Towards Ideological Positioning of Morally Conditioned LLMs 

**Authors**: Chenchen Yuan, Bolei Ma, Zheyu Zhang, Bardh Prenkaj, Frauke Kreuter, Gjergji Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2601.08634)  

**Abstract**: While recent research has systematically documented political orientation in large language models (LLMs), existing evaluations rely primarily on direct probing or demographic persona engineering to surface ideological biases. In social psychology, however, political ideology is also understood as a downstream consequence of fundamental moral intuitions. In this work, we investigate the causal relationship between moral values and political positioning by treating moral orientation as a controllable condition. Rather than simply assigning a demographic persona, we condition models to endorse or reject specific moral values and evaluate the resulting shifts on their political orientations, using the Political Compass Test. By treating moral values as lenses, we observe how moral conditioning actively steers model trajectories across economic and social dimensions. Our findings show that such conditioning induces pronounced, value-specific shifts in models' political coordinates. We further notice that these effects are systematically modulated by role framing and model scale, and are robust across alternative assessment instruments instantiating the same moral value. This highlights that effective alignment requires anchoring political assessments within the context of broader social values including morality, paving the way for more socially grounded alignment techniques. 

---
# Get away with less: Need of source side data curation to build parallel corpus for low resource Machine Translation 

**Authors**: Saumitra Yadav, Manish Shrivastava  

**Link**: [PDF](https://arxiv.org/pdf/2601.08629)  

**Abstract**: Data curation is a critical yet under-researched step in the machine translation training paradigm. To train translation systems, data acquisition relies primarily on human translations and digital parallel sources or, to a limited degree, synthetic generation. But, for low-resource languages, human translation to generate sufficient data is prohibitively expensive. Therefore, it is crucial to develop a framework that screens source sentences to form efficient parallel text, ensuring optimal MT system performance in low-resource environments. We approach this by evaluating English-Hindi bi-text to determine effective sentence selection strategies for optimal MT system training. Our extensively tested framework, (Lexical And Linguistically Informed Text Analysis) LALITA, targets source sentence selection using lexical and linguistic features to curate parallel corpora. We find that by training mostly on complex sentences from both existing and synthetic datasets, our method significantly improves translation quality. We test this by simulating low-resource data availabilty with curated datasets of 50K to 800K English sentences and report improved performances on all data sizes. LALITA demonstrates remarkable efficiency, reducing data needs by more than half across multiple languages (Hindi, Odia, Nepali, Norwegian Nynorsk, and German). This approach not only reduces MT systems training cost by reducing training data requirement, but also showcases LALITA's utility in data augmentation. 

---
# How Order-Sensitive Are LLMs? OrderProbe for Deterministic Structural Reconstruction 

**Authors**: Yingjie He, Zhaolu Kang, Kehan Jiang, Qianyuan Zhang, Jiachen Qian, Chunlei Meng, Yujie Feng, Yuan Wang, Jiabao Dou, Aming Wu, Leqi Zheng, Pengxiang Zhao, Jiaxin Liu, Zeyu Zhang, Lei Wang, Guansu Wang, Qishi Zhan, Xiaomin He, Meisheng Zhang, Jianyuan Ni  

**Link**: [PDF](https://arxiv.org/pdf/2601.08626)  

**Abstract**: Large language models (LLMs) excel at semantic understanding, yet their ability to reconstruct internal structure from scrambled inputs remains underexplored. Sentence-level restoration is ill-posed for automated evaluation because multiple valid word orders often exist. We introduce OrderProbe, a deterministic benchmark for structural reconstruction using fixed four-character expressions in Chinese, Japanese, and Korean, which have a unique canonical order and thus support exact-match scoring. We further propose a diagnostic framework that evaluates models beyond recovery accuracy, including semantic fidelity, logical validity, consistency, robustness sensitivity, and information density. Experiments on twelve widely used LLMs show that structural reconstruction remains difficult even for frontier systems: zero-shot recovery frequently falls below 35%. We also observe a consistent dissociation between semantic recall and structural planning, suggesting that structural robustness is not an automatic byproduct of semantic competence. 

---
# GraphSearch: Agentic Search-Augmented Reasoning for Zero-Shot Graph Learning 

**Authors**: Jiajin Liu, Yuanfu Sun, Dongzhe Fan, Qiaoyu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2601.08621)  

**Abstract**: Recent advances in search-augmented large reasoning models (LRMs) enable the retrieval of external knowledge to reduce hallucinations in multistep reasoning. However, their ability to operate on graph-structured data, prevalent in domains such as e-commerce, social networks, and scientific citations, remains underexplored. Unlike plain text corpora, graphs encode rich topological signals that connect related entities and can serve as valuable priors for retrieval, enabling more targeted search and improved reasoning efficiency. Yet, effectively leveraging such structure poses unique challenges, including the difficulty of generating graph-expressive queries and ensuring reliable retrieval that balances structural and semantic relevance. To address this gap, we introduce GraphSearch, the first framework that extends search-augmented reasoning to graph learning, enabling zero-shot graph learning without task-specific fine-tuning. GraphSearch combines a Graph-aware Query Planner, which disentangles search space (e.g., 1-hop, multi-hop, or global neighbors) from semantic queries, with a Graph-aware Retriever, which constructs candidate sets based on topology and ranks them using a hybrid scoring function. We further instantiate two traversal modes: GraphSearch-R, which recursively expands neighborhoods hop by hop, and GraphSearch-F, which flexibly retrieves across local and global neighborhoods without hop constraints. Extensive experiments across diverse benchmarks show that GraphSearch achieves competitive or even superior performance compared to supervised graph learning methods, setting state-of-the-art results in zero-shot node classification and link prediction. These findings position GraphSearch as a flexible and generalizable paradigm for agentic reasoning over graphs. 

---
# ExpSeek: Self-Triggered Experience Seeking for Web Agents 

**Authors**: Wenyuan Zhang, Xinghua Zhang, Haiyang Yu, Shuaiyi Nie, Bingli Wu, Juwei Yue, Tingwen Liu, Yongbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.08605)  

**Abstract**: Experience intervention in web agents emerges as a promising technical paradigm, enhancing agent interaction capabilities by providing valuable insights from accumulated experiences. However, existing methods predominantly inject experience passively as global context before task execution, struggling to adapt to dynamically changing contextual observations during agent-environment interaction. We propose ExpSeek, which shifts experience toward step-level proactive seeking: (1) estimating step-level entropy thresholds to determine intervention timing using the model's intrinsic signals; (2) designing step-level tailor-designed experience content. Experiments on Qwen3-8B and 32B models across four challenging web agent benchmarks demonstrate that ExpSeek achieves absolute improvements of 9.3% and 7.5%, respectively. Our experiments validate the feasibility and advantages of entropy as a self-triggering signal, reveal that even a 4B small-scale experience model can significantly boost the performance of larger agent models. 

---
# Ministral 3 

**Authors**: Alexander H. Liu, Kartik Khandelwal, Sandeep Subramanian, Victor Jouault, Abhinav Rastogi, Adrien Sadé, Alan Jeffares, Albert Jiang, Alexandre Cahill, Alexandre Gavaudan, Alexandre Sablayrolles, Amélie Héliou, Amos You, Andy Ehrenberg, Andy Lo, Anton Eliseev, Antonia Calvi, Avinash Sooriyarachchi, Baptiste Bout, Baptiste Rozière, Baudouin De Monicault, Clémence Lanfranchi, Corentin Barreau, Cyprien Courtot, Daniele Grattarola, Darius Dabert, Diego de las Casas, Elliot Chane-Sane, Faruk Ahmed, Gabrielle Berrada, Gaëtan Ecrepont, Gauthier Guinet, Georgii Novikov, Guillaume Kunsch, Guillaume Lample, Guillaume Martin, Gunshi Gupta, Jan Ludziejewski, Jason Rute, Joachim Studnia, Jonas Amar, Joséphine Delas, Josselin Somerville Roberts, Karmesh Yadav, Khyathi Chandu, Kush Jain, Laurence Aitchison, Laurent Fainsin, Léonard Blier, Lingxiao Zhao, Louis Martin, Lucile Saulnier, Luyu Gao, Maarten Buyl, Margaret Jennings, Marie Pellat, Mark Prins, Mathieu Poirée, Mathilde Guillaumin, Matthieu Dinot, Matthieu Futeral, Maxime Darrin, Maximilian Augustin, Mia Chiquier, Michel Schimpf, Nathan Grinsztajn, Neha Gupta, Nikhil Raghuraman, Olivier Bousquet, Olivier Duchenne, Patricia Wang, Patrick von Platen, Paul Jacob, Paul Wambergue, Paula Kurylowicz, Pavankumar Reddy Muddireddy, Philomène Chagniot, Pierre Stock, Pravesh Agrawal, Quentin Torroba, Romain Sauvestre, Roman Soletskyi, Rupert Menneer, Sagar Vaze, Samuel Barry, Sanchit Gandhi, Siddhant Waghjale, Siddharth Gandhi, Soham Ghosh, Srijan Mishra, Sumukh Aithal, Szymon Antoniak, Teven Le Scao, Théo Cachet, Theo Simon Sorg, Thibaut Lavril, Thiziri Nait Saada, Thomas Chabal, Thomas Foubert, Thomas Robert  

**Link**: [PDF](https://arxiv.org/pdf/2601.08584)  

**Abstract**: We introduce the Ministral 3 series, a family of parameter-efficient dense language models designed for compute and memory constrained applications, available in three model sizes: 3B, 8B, and 14B parameters. For each model size, we release three variants: a pretrained base model for general-purpose use, an instruction finetuned, and a reasoning model for complex problem-solving. In addition, we present our recipe to derive the Ministral 3 models through Cascade Distillation, an iterative pruning and continued training with distillation technique. Each model comes with image understanding capabilities, all under the Apache 2.0 license. 

---
# DeepResearch Bench II: Diagnosing Deep Research Agents via Rubrics from Expert Report 

**Authors**: Ruizhe Li, Mingxuan Du, Benfeng Xu, Chiwei Zhu, Xiaorui Wang, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2601.08536)  

**Abstract**: Deep Research Systems (DRS) aim to help users search the web, synthesize information, and deliver comprehensive investigative reports. However, how to rigorously evaluate these systems remains under-explored. Existing deep-research benchmarks often fall into two failure modes. Some do not adequately test a system's ability to analyze evidence and write coherent reports. Others rely on evaluation criteria that are either overly coarse or directly defined by LLMs (or both), leading to scores that can be biased relative to human experts and are hard to verify or interpret. To address these issues, we introduce Deep Research Bench II, a new benchmark for evaluating DRS-generated reports. It contains 132 grounded research tasks across 22 domains; for each task, a system must produce a long-form research report that is evaluated by a set of 9430 fine-grained binary rubrics in total, covering three dimensions: information recall, analysis, and presentation. All rubrics are derived from carefully selected expert-written investigative articles and are constructed through a four-stage LLM+human pipeline that combines automatic extraction with over 400 human-hours of expert review, ensuring that the criteria are atomic, verifiable, and aligned with human expert judgment. We evaluate several state-of-the-art deep-research systems on Deep Research Bench II and find that even the strongest models satisfy fewer than 50% of the rubrics, revealing a substantial gap between current DRSs and human experts. 

---
# Algorithmic Stability in Infinite Dimensions: Characterizing Unconditional Convergence in Banach Spaces 

**Authors**: Przemysław Spyra  

**Link**: [PDF](https://arxiv.org/pdf/2601.08512)  

**Abstract**: The distinction between conditional, unconditional, and absolute convergence in infinite-dimensional spaces has fundamental implications for computational algorithms. While these concepts coincide in finite dimensions, the Dvoretzky-Rogers theorem establishes their strict separation in general Banach spaces. We present a comprehensive characterization theorem unifying seven equivalent conditions for unconditional convergence: permutation invariance, net convergence, subseries tests, sign stability, bounded multiplier properties, and weak uniform convergence. These theoretical results directly inform algorithmic stability analysis, governing permutation invariance in gradient accumulation for Stochastic Gradient Descent and justifying coefficient thresholding in frame-based signal processing. Our work bridges classical functional analysis with contemporary computational practice, providing rigorous foundations for order-independent and numerically robust summation processes. 

---
# STAR: Detecting Inference-time Backdoors in LLM Reasoning via State-Transition Amplification Ratio 

**Authors**: Seong-Gyu Park, Sohee Park, Jisu Lee, Hyunsik Na, Daeseon Choi  

**Link**: [PDF](https://arxiv.org/pdf/2601.08511)  

**Abstract**: Recent LLMs increasingly integrate reasoning mechanisms like Chain-of-Thought (CoT). However, this explicit reasoning exposes a new attack surface for inference-time backdoors, which inject malicious reasoning paths without altering model parameters. Because these attacks generate linguistically coherent paths, they effectively evade conventional detection. To address this, we propose STAR (State-Transition Amplification Ratio), a framework that detects backdoors by analyzing output probability shifts. STAR exploits the statistical discrepancy where a malicious input-induced path exhibits high posterior probability despite a low prior probability in the model's general knowledge. We quantify this state-transition amplification and employ the CUSUM algorithm to detect persistent anomalies. Experiments across diverse models (8B-70B) and five benchmark datasets demonstrate that STAR exhibits robust generalization capabilities, consistently achieving near-perfect performance (AUROC $\approx$ 1.0) with approximately $42\times$ greater efficiency than existing baselines. Furthermore, the framework proves robust against adaptive attacks attempting to bypass detection. 

---
# STAGE: A Benchmark for Knowledge Graph Construction, Question Answering, and In-Script Role-Playing over Movie Screenplays 

**Authors**: Qiuyu Tian, Yiding Li, Fengyi Chen, Zequn Liu, Youyong Kong, Fan Guo, Yuyao Li, Jinjing Shen, Zhijing Xie, Yiyun Luo, Xin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08510)  

**Abstract**: Movie screenplays are rich long-form narratives that interleave complex character relationships, temporally ordered events, and dialogue-driven interactions. While prior benchmarks target individual subtasks such as question answering or dialogue generation, they rarely evaluate whether models can construct a coherent story world and use it consistently across multiple forms of reasoning and generation. We introduce STAGE (Screenplay Text, Agents, Graphs and Evaluation), a unified benchmark for narrative understanding over full-length movie screenplays. STAGE defines four tasks: knowledge graph construction, scene-level event summarization, long-context screenplay question answering, and in-script character role-playing, all grounded in a shared narrative world representation. The benchmark provides cleaned scripts, curated knowledge graphs, and event- and character-centric annotations for 150 films across English and Chinese, enabling holistic evaluation of models' abilities to build world representations, abstract and verify narrative events, reason over long narratives, and generate character-consistent responses. 

---
# It's All About the Confidence: An Unsupervised Approach for Multilingual Historical Entity Linking using Large Language Models 

**Authors**: Cristian Santini, Marieke Van Erp, Mehwish Alam  

**Link**: [PDF](https://arxiv.org/pdf/2601.08500)  

**Abstract**: Despite the recent advancements in NLP with the advent of Large Language Models (LLMs), Entity Linking (EL) for historical texts remains challenging due to linguistic variation, noisy inputs, and evolving semantic conventions. Existing solutions either require substantial training data or rely on domain-specific rules that limit scalability. In this paper, we present MHEL-LLaMo (Multilingual Historical Entity Linking with Large Language MOdels), an unsupervised ensemble approach combining a Small Language Model (SLM) and an LLM. MHEL-LLaMo leverages a multilingual bi-encoder (BELA) for candidate retrieval and an instruction-tuned LLM for NIL prediction and candidate selection via prompt chaining. Our system uses SLM's confidence scores to discriminate between easy and hard samples, applying an LLM only for hard cases. This strategy reduces computational costs while preventing hallucinations on straightforward cases. We evaluate MHEL-LLaMo on four established benchmarks in six European languages (English, Finnish, French, German, Italian and Swedish) from the 19th and 20th centuries. Results demonstrate that MHEL-LLaMo outperforms state-of-the-art models without requiring fine-tuning, offering a scalable solution for low-resource historical EL. The implementation of MHEL-LLaMo is available on Github. 

---
# BenchOverflow: Measuring Overflow in Large Language Models via Plain-Text Prompts 

**Authors**: Erin Feiglin, Nir Hutnik, Raz Lapid  

**Link**: [PDF](https://arxiv.org/pdf/2601.08490)  

**Abstract**: We investigate a failure mode of large language models (LLMs) in which plain-text prompts elicit excessive outputs, a phenomenon we term Overflow. Unlike jailbreaks or prompt injection, Overflow arises under ordinary interaction settings and can lead to elevated serving cost, latency, and cross-user performance degradation, particularly when scaled across many requests. Beyond usability, the stakes are economic and environmental: unnecessary tokens increase per-request cost and energy consumption, compounding into substantial operational spend and carbon footprint at scale. Moreover, Overflow represents a practical vector for compute amplification and service degradation in shared environments. We introduce BenchOverflow, a model-agnostic benchmark of nine plain-text prompting strategies that amplify output volume without adversarial suffixes or policy circumvention. Using a standardized protocol with a fixed budget of 5000 new tokens, we evaluate nine open- and closed-source models and observe pronounced rightward shifts and heavy tails in length distributions. Cap-saturation rates (CSR@1k/3k/5k) and empirical cumulative distribution functions (ECDFs) quantify tail risk; within-prompt variance and cross-model correlations show that Overflow is broadly reproducible yet heterogeneous across families and attack vectors. A lightweight mitigation-a fixed conciseness reminder-attenuates right tails and lowers CSR for all strategies across the majority of models. Our findings position length control as a measurable reliability, cost, and sustainability concern rather than a stylistic quirk. By enabling standardized comparison of length-control robustness across models, BenchOverflow provides a practical basis for selecting deployments that minimize resource waste and operating expense, and for evaluating defenses that curb compute amplification without eroding task performance. 

---
# Surgical Refusal Ablation: Disentangling Safety from Intelligence via Concept-Guided Spectral Cleaning 

**Authors**: Tony Cristofano  

**Link**: [PDF](https://arxiv.org/pdf/2601.08489)  

**Abstract**: Safety-aligned language models systematically refuse harmful requests. While activation steering can modulate refusal, ablating the raw "refusal vector" calculated from contrastive harmful and harmless prompts often causes collateral damage and distribution drift. We argue this degradation occurs because the raw vector is polysemantic, entangling the refusal signal with core capability circuits and linguistic style.
We introduce Surgical Refusal Ablation (SRA) to distill these steering directions. SRA constructs a registry of independent Concept Atoms representing protected capabilities and stylistic confounds, then uses ridge-regularized spectral residualization to orthogonalize the refusal vector against these directions. This yields a clean refusal direction that targets refusal-relevant structure while minimizing disruption to the model's semantic geometry.
Across five models (Qwen3-VL and Ministral series), SRA achieves deep refusal reduction (0-2%) with negligible perplexity impact on Wikitext-2 (mean delta PPL approx. 0.02) and minimal distribution drift. Notably, standard ablation on Qwen3-VL-4B induces severe drift (first-token KL = 2.088), whereas SRA maintains the original distribution (KL = 0.044) while achieving the same 0% refusal rate. Using teacher-forced perplexity on GSM8K and MBPP as a high-resolution capability proxy, we show SRA preserves math and code distributions. These results suggest that common "model damage" is often "Ghost Noise," defined as the spectral bleeding of the dirty refusal direction into capability subspaces. 

---
# Do You Understand How I Feel?: Towards Verified Empathy in Therapy Chatbots 

**Authors**: Francesco Dettori, Matteo Forasassi, Lorenzo Veronese, Livia Lestingi, Vincenzo Scotti, Matteo Giovanni Rossi  

**Link**: [PDF](https://arxiv.org/pdf/2601.08477)  

**Abstract**: Conversational agents are increasingly used as support tools along mental therapeutic pathways with significant societal impacts. In particular, empathy is a key non-functional requirement in therapeutic contexts, yet current chatbot development practices provide no systematic means to specify or verify it. This paper envisions a framework integrating natural language processing and formal verification to deliver empathetic therapy chatbots. A Transformer-based model extracts dialogue features, which are then translated into a Stochastic Hybrid Automaton model of dyadic therapy sessions. Empathy-related properties can then be verified through Statistical Model Checking, while strategy synthesis provides guidance for shaping agent behavior. Preliminary results show that the formal model captures therapy dynamics with good fidelity and that ad-hoc strategies improve the probability of satisfying empathy requirements. 

---
# sui-1: Grounded and Verifiable Long-Form Summarization 

**Authors**: Benedikt Droste, Jan Philipp Harries, Maximilian Idahl, Björn Plüster  

**Link**: [PDF](https://arxiv.org/pdf/2601.08472)  

**Abstract**: Large language models frequently generate plausible but unfaithful summaries that users cannot verify against source text, a critical limitation in compliance-sensitive domains such as government and legal analysis. We present sui-1, a 24B parameter model that produces abstractive summaries with inline citations, enabling users to trace each claim to its source sentence. Our synthetic data pipeline combines chain-of-thought prompting with multi-stage verification, generating over 22,000 high-quality training examples across five languages from diverse sources including parliamentary documents, web text, and Wikipedia. Evaluation shows sui-1 significantly outperforms all tested open-weight baselines, including models with 3x more parameters. These results demonstrate that task-specific training substantially outperforms scale alone for citation-grounded summarization. Model weights and an interactive demo are publicly available. 

---
# JudgeRLVR: Judge First, Generate Second for Efficient Reasoning 

**Authors**: Jiangshan Duo, Hanyu Li, Hailin Zhang, Yudong Wang, Sujian Li, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.08468)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has become a standard paradigm for reasoning in Large Language Models. However, optimizing solely for final-answer correctness often drives models into aimless, verbose exploration, where they rely on exhaustive trial-and-error tactics rather than structured planning to reach solutions. While heuristic constraints like length penalties can reduce verbosity, they often truncate essential reasoning steps, creating a difficult trade-off between efficiency and verification. In this paper, we argue that discriminative capability is a prerequisite for efficient generation: by learning to distinguish valid solutions, a model can internalize a guidance signal that prunes the search space. We propose JudgeRLVR, a two-stage judge-then-generate paradigm. In the first stage, we train the model to judge solution responses with verifiable answers. In the second stage, we fine-tune the same model with vanilla generating RLVR initialized from the judge. Compared to Vanilla RLVR using the same math-domain training data, JudgeRLVR achieves a better quality--efficiency trade-off for Qwen3-30B-A3B: on in-domain math, it delivers about +3.7 points average accuracy gain with -42\% average generation length; on out-of-domain benchmarks, it delivers about +4.5 points average accuracy improvement, demonstrating enhanced generalization. 

---
# Fine-Mem: Fine-Grained Feedback Alignment for Long-Horizon Memory Management 

**Authors**: Weitao Ma, Xiaocheng Feng, Lei Huang, Xiachong Feng, Zhanyu Ma, Jun Xu, Jiuchong Gao, Jinghua Hao, Renqing He, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2601.08435)  

**Abstract**: Effective memory management is essential for large language model agents to navigate long-horizon tasks. Recent research has explored using Reinforcement Learning to develop specialized memory manager agents. However, existing approaches rely on final task performance as the primary reward, which results in severe reward sparsity and ineffective credit assignment, providing insufficient guidance for individual memory operations. To this end, we propose Fine-Mem, a unified framework designed for fine-grained feedback alignment. First, we introduce a Chunk-level Step Reward to provide immediate step-level supervision via auxiliary chunk-specific question answering tasks. Second, we devise Evidence-Anchored Reward Attribution to redistribute global rewards by anchoring credit to key memory operations, based on the specific memory items utilized as evidence in reasoning. Together, these components enable stable policy optimization and align local memory operations with the long-term utility of memory. Experiments on Memalpha and MemoryAgentBench demonstrate that Fine-Mem consistently outperforms strong baselines, achieving superior success rates across various sub-tasks. Further analysis reveals its adaptability and strong generalization capabilities across diverse model configurations and backbones. 

---
# Silence the Judge: Reinforcement Learning with Self-Verifier via Latent Geometric Clustering 

**Authors**: Nonghai Zhang, Weitao Ma, Zhanyu Ma, Jun Xu, Jiuchong Gao, Jinghua Hao, Renqing He, Jingwen Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.08427)  

**Abstract**: Group Relative Policy Optimization (GRPO) significantly enhances the reasoning performance of Large Language Models (LLMs). However, this success heavily relies on expensive external verifiers or human rules. Such dependency not only leads to significant computational costs and training latency, but also yields sparse rewards that hinder optimization efficiency. To address these challenges, we propose Latent-GRPO, a framework that derives intrinsic rewards directly from latent space geometry. Crucially, our empirical analysis reveals a compelling geometric property: terminal token representations of correct reasoning trajectories form dense clusters with high intra-class similarity, whereas incorrect trajectories remain scattered as outliers. In light of this discovery, we introduce the Iterative Robust Centroid Estimation (IRCE) algorithm, which generates dense, continuous rewards by mitigating magnitude fluctuations via spherical projection and estimating a robust ``truth centroid'' through iterative aggregation. Experimental results on multiple datasets show that our method maintains model performance while achieving a training speedup of over 2x compared to baselines. Furthermore, extensive results demonstrate strong generalization ability and robustness. The code will be released soon. 

---
# PATS: Personality-Aware Teaching Strategies with Large Language Model Tutors 

**Authors**: Donya Rooein, Sankalan Pal Chowdhury, Mariia Eremeeva, Yuan Qin, Debora Nozza, Mrinmaya Sachan, Dirk Hovy  

**Link**: [PDF](https://arxiv.org/pdf/2601.08402)  

**Abstract**: Recent advances in large language models (LLMs) demonstrate their potential as educational tutors. However, different tutoring strategies benefit different student personalities, and mismatches can be counterproductive to student outcomes. Despite this, current LLM tutoring systems do not take into account student personality traits. To address this problem, we first construct a taxonomy that links pedagogical methods to personality profiles, based on pedagogical literature. We simulate student-teacher conversations and use our framework to let the LLM tutor adjust its strategy to the simulated student personality. We evaluate the scenario with human teachers and find that they consistently prefer our approach over two baselines. Our method also increases the use of less common, high-impact strategies such as role-playing, which human and LLM annotators prefer significantly. Our findings pave the way for developing more personalized and effective LLM use in educational applications. 

---
# Detecting Mental Manipulation in Speech via Synthetic Multi-Speaker Dialogue 

**Authors**: Run Chen, Wen Liang, Ziwei Gong, Lin Ai, Julia Hirschberg  

**Link**: [PDF](https://arxiv.org/pdf/2601.08342)  

**Abstract**: Mental manipulation, the strategic use of language to covertly influence or exploit others, is a newly emerging task in computational social reasoning. Prior work has focused exclusively on textual conversations, overlooking how manipulative tactics manifest in speech. We present the first study of mental manipulation detection in spoken dialogues, introducing a synthetic multi-speaker benchmark SPEECHMENTALMANIP that augments a text-based dataset with high-quality, voice-consistent Text-to-Speech rendered audio. Using few-shot large audio-language models and human annotation, we evaluate how modality affects detection accuracy and perception. Our results reveal that models exhibit high specificity but markedly lower recall on speech compared to text, suggesting sensitivity to missing acoustic or prosodic cues in training. Human raters show similar uncertainty in the audio setting, underscoring the inherent ambiguity of manipulative speech. Together, these findings highlight the need for modality-aware evaluation and safety alignment in multimodal dialogue systems. 

---
# CLaS-Bench: A Cross-Lingual Alignment and Steering Benchmark 

**Authors**: Daniil Gurgurov, Yusser Al Ghussin, Tanja Baeumel, Cheng-Ting Chou, Patrick Schramowski, Marius Mosbach, Josef van Genabith, Simon Ostermann  

**Link**: [PDF](https://arxiv.org/pdf/2601.08331)  

**Abstract**: Understanding and controlling the behavior of large language models (LLMs) is an increasingly important topic in multilingual NLP. Beyond prompting or fine-tuning, , i.e.,~manipulating internal representations during inference, has emerged as a more efficient and interpretable technique for adapting models to a target language. Yet, no dedicated benchmarks or evaluation protocols exist to quantify the effectiveness of steering techniques. We introduce CLaS-Bench, a lightweight parallel-question benchmark for evaluating language-forcing behavior in LLMs across 32 languages, enabling systematic evaluation of multilingual steering methods. We evaluate a broad array of steering techniques, including residual-stream DiffMean interventions, probe-derived directions, language-specific neurons, PCA/LDA vectors, Sparse Autoencoders, and prompting baselines. Steering performance is measured along two axes: language control and semantic relevance, combined into a single harmonic-mean steering score. We find that across languages simple residual-based DiffMean method consistently outperforms all other methods. Moreover, a layer-wise analysis reveals that language-specific structure emerges predominantly in later layers and steering directions cluster based on language family. CLaS-Bench is the first standardized benchmark for multilingual steering, enabling both rigorous scientific analysis of language representations and practical evaluation of steering as a low-cost adaptation alternative. 

---
# AgriAgent: Contract-Driven Planning and Capability-Aware Tool Orchestration in Real-World Agriculture 

**Authors**: Bo Yang, Yu Zhang, Yunkui Chen, Lanfei Feng, Xiao Xu, Nueraili Aierken, Shijian Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.08308)  

**Abstract**: Intelligent agent systems in real-world agricultural scenarios must handle diverse tasks under multimodal inputs, ranging from lightweight information understanding to complex multi-step execution. However, most existing approaches rely on a unified execution paradigm, which struggles to accommodate large variations in task complexity and incomplete tool availability commonly observed in agricultural environments. To address this challenge, we propose AgriAgent, a two-level agent framework for real-world agriculture. AgriAgent adopts a hierarchical execution strategy based on task complexity: simple tasks are handled through direct reasoning by modality-specific agents, while complex tasks trigger a contract-driven planning mechanism that formulates tasks as capability requirements and performs capability-aware tool orchestration and dynamic tool generation, enabling multi-step and verifiable execution with failure recovery. Experimental results show that AgriAgent achieves higher execution success rates and robustness on complex tasks compared to existing tool-centric agent baselines that rely on unified execution paradigms. All code, data will be released at after our work be accepted to promote reproducible research. 

---
# Enhancing Sentiment Classification and Irony Detection in Large Language Models through Advanced Prompt Engineering Techniques 

**Authors**: Marvin Schmitt, Anne Schwerk, Sebastian Lempert  

**Link**: [PDF](https://arxiv.org/pdf/2601.08302)  

**Abstract**: This study investigates the use of prompt engineering to enhance large language models (LLMs), specifically GPT-4o-mini and gemini-1.5-flash, in sentiment analysis tasks. It evaluates advanced prompting techniques like few-shot learning, chain-of-thought prompting, and self-consistency against a baseline. Key tasks include sentiment classification, aspect-based sentiment analysis, and detecting subtle nuances such as irony. The research details the theoretical background, datasets, and methods used, assessing performance of LLMs as measured by accuracy, recall, precision, and F1 score. Findings reveal that advanced prompting significantly improves sentiment analysis, with the few-shot approach excelling in GPT-4o-mini and chain-of-thought prompting boosting irony detection in gemini-1.5-flash by up to 46%. Thus, while advanced prompting techniques overall improve performance, the fact that few-shot prompting works best for GPT-4o-mini and chain-of-thought excels in gemini-1.5-flash for irony detection suggests that prompting strategies must be tailored to both the model and the task. This highlights the importance of aligning prompt design with both the LLM's architecture and the semantic complexity of the task. 

---
# D$^2$Plan: Dual-Agent Dynamic Global Planning for Complex Retrieval-Augmented Reasoning 

**Authors**: Kangcheng Luo, Tinglang Wu, Yansong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2601.08282)  

**Abstract**: Recent search-augmented LLMs trained with reinforcement learning (RL) can interleave searching and reasoning for multi-hop reasoning tasks. However, they face two critical failure modes as the accumulating context becomes flooded with both crucial evidence and irrelevant information: (1) ineffective search chain construction that produces incorrect queries or omits retrieval of critical information, and (2) reasoning hijacking by peripheral evidence that causes models to misidentify distractors as valid evidence. To address these challenges, we propose **D$^2$Plan**, a **D**ual-agent **D**ynamic global **Plan**ning paradigm for complex retrieval-augmented reasoning. **D$^2$Plan** operates through the collaboration of a *Reasoner* and a *Purifier*: the *Reasoner* constructs explicit global plans during reasoning and dynamically adapts them based on retrieval feedback; the *Purifier* assesses retrieval relevance and condenses key information for the *Reasoner*. We further introduce a two-stage training framework consisting of supervised fine-tuning (SFT) cold-start on synthesized trajectories and RL with plan-oriented rewards to teach LLMs to master the **D$^2$Plan** paradigm. Extensive experiments demonstrate that **D$^2$Plan** enables more coherent multi-step reasoning and stronger resilience to irrelevant information, thereby achieving superior performance on challenging QA benchmarks. 

---
# Discovery and Reinforcement of Tool-Integrated Reasoning Chains via Rollout Trees 

**Authors**: Kun Li, Zenan Xu, Junan Li, Zengrui Jin, Jinghao Deng, Zexuan Qiu, Bo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.08274)  

**Abstract**: Tool-Integrated Reasoning has emerged as a key paradigm to augment Large Language Models (LLMs) with computational capabilities, yet integrating tool-use into long Chain-of-Thought (long CoT) remains underexplored, largely due to the scarcity of training data and the challenge of integrating tool-use without compromising the model's intrinsic long-chain reasoning. In this paper, we introduce DART (Discovery And Reinforcement of Tool-Integrated Reasoning Chains via Rollout Trees), a reinforcement learning framework that enables spontaneous tool-use during long CoT reasoning without human annotation. DART operates by constructing dynamic rollout trees during training to discover valid tool-use opportunities, branching out at promising positions to explore diverse tool-integrated trajectories. Subsequently, a tree-based process advantage estimation identifies and credits specific sub-trajectories where tool invocation positively contributes to the solution, effectively reinforcing these beneficial behaviors. Extensive experiments on challenging benchmarks like AIME and GPQA-Diamond demonstrate that DART significantly outperforms existing methods, successfully harmonizing tool execution with long CoT reasoning. 

---
# Med-CoReasoner: Reducing Language Disparities in Medical Reasoning via Language-Informed Co-Reasoning 

**Authors**: Fan Gao, Sherry T. Tong, Jiwoong Sohn, Jiahao Huang, Junfeng Jiang, Ding Xia, Piyalitt Ittichaiwong, Kanyakorn Veerakanjana, Hyunjae Kim, Qingyu Chen, Edison Marrese Taylor, Kazuma Kobayashi, Akkiko Aizawa, Irene Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.08267)  

**Abstract**: While reasoning-enhanced large language models perform strongly on English medical tasks, a persistent multilingual gap remains, with substantially weaker reasoning in local languages, limiting equitable global medical deployment. To bridge this gap, we introduce Med-CoReasoner, a language-informed co-reasoning framework that elicits parallel English and local-language reasoning, abstracts them into structured concepts, and integrates local clinical knowledge into an English logical scaffold via concept-level alignment and retrieval. This design combines the structural robustness of English reasoning with the practice-grounded expertise encoded in local languages. To evaluate multilingual medical reasoning beyond multiple-choice settings, we construct MultiMed-X, a benchmark covering seven languages with expert-annotated long-form question answering and natural language inference tasks, comprising 350 instances per language. Experiments across three benchmarks show that Med-CoReasoner improves multilingual reasoning performance by an average of 5%, with particularly substantial gains in low-resource languages. Moreover, model distillation and expert evaluation analysis further confirm that Med-CoReasoner produces clinically sound and culturally grounded reasoning traces. 

---
# User-Oriented Multi-Turn Dialogue Generation with Tool Use at scale 

**Authors**: Jungho Cho, Minbyul Jeong, Sungrae Park  

**Link**: [PDF](https://arxiv.org/pdf/2601.08225)  

**Abstract**: The recent paradigm shift toward large reasoning models (LRMs) as autonomous agents has intensified the demand for sophisticated, multi-turn tool-use capabilities. Yet, existing datasets and data-generation approaches are limited by static, predefined toolsets that cannot scale to the complexity of open-ended human-agent collaboration. To address this, we initially developed a framework for automated task-oriented multi-turn dialogue generation at scale, utilizing an LRM-based simulator to dynamically generate high-value, domain-specific tools to solve specified tasks. However, we observe that a purely task-oriented design often results in "solely task-solving" trajectories, where the agent completes the objective with minimal interaction, failing to generate the high turn-count conversations seen in realistic scenarios. To bridge this gap, we shift toward a user-oriented simulation paradigm. By decoupling task generation from a dedicated user simulator that mimics human behavioral rules - such as incremental request-making and turn-by-turn feedback - we facilitate more authentic, extended multi-turn dialogues that reflect the iterative nature of real-world problem solving. Our generation pipeline operates as a versatile, plug-and-play module capable of initiating generation from any state, ensuring high scalability in producing extended tool-use data. Furthermore, by facilitating multiple task completions within a single trajectory, it yields a high-density dataset that reflects the multifaceted demands of real-world human-agent interaction. 

---
# Towards Principled Design of Mixture-of-Experts Language Models under Memory and Inference Constraints 

**Authors**: Seng Pei Liew, Kenta Shinzato, Yuyang Dong  

**Link**: [PDF](https://arxiv.org/pdf/2601.08215)  

**Abstract**: Modern Mixture-of-Experts (MoE) language models are designed based on total parameters (memory footprint) and active parameters (inference cost). However, we find these two factors alone are insufficient to describe an optimal architecture. Through a systematic study, we demonstrate that MoE performance is primarily determined by total parameters ($N_{total}$) and expert sparsity ($s:=n_{exp}/n_{topk}$).
Moreover, $n_{exp}$ and $n_{topk}$ do not "cancel out" within the sparsity ratio; instead, a larger total number of experts slightly penalizes performance by forcing a reduction in core model dimensions (depth and width) to meet memory constraints. This motivates a simple principle for MoE design which maximizes $N_{total}$ while minimizing $s$ (maximizing $n_{topk}$) and $n_{exp}$ under the given constraints. Our findings provide a robust framework for resolving architectural ambiguity and guiding MoE design. 

---
# Generation-Augmented Generation: A Plug-and-Play Framework for Private Knowledge Injection in Large Language Models 

**Authors**: Rongji Li, Jian Xu, Xueqing Chen, Yisheng Yang, Jiayi Wang, Xingyu Chen, Chunyu Xie, Dawei Leng, Xu-Yao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08209)  

**Abstract**: In domains such as biomedicine, materials, and finance, high-stakes deployment of large language models (LLMs) requires injecting private, domain-specific knowledge that is proprietary, fast-evolving, and under-represented in public pretraining. However, the two dominant paradigms for private knowledge injection each have pronounced drawbacks: fine-tuning is expensive to iterate, and continual updates risk catastrophic forgetting and general-capability regression; retrieval-augmented generation (RAG) keeps the base model intact but is brittle in specialized private corpora due to chunk-induced evidence fragmentation, retrieval drift, and long-context pressure that yields query-dependent prompt inflation. Inspired by how multimodal LLMs align heterogeneous modalities into a shared semantic space, we propose Generation-Augmented Generation (GAG), which treats private expertise as an additional expert modality and injects it via a compact, representation-level interface aligned to the frozen base model, avoiding prompt-time evidence serialization while enabling plug-and-play specialization and scalable multi-domain composition with reliable selective activation. Across two private scientific QA benchmarks (immunology adjuvant and catalytic materials) and mixed-domain evaluations, GAG improves specialist performance over strong RAG baselines by 15.34% and 14.86% on the two benchmarks, respectively, while maintaining performance on six open general benchmarks and enabling near-oracle selective activation for scalable multi-domain deployment. 

---
# Triplets Better Than Pairs: Towards Stable and Effective Self-Play Fine-Tuning for LLMs 

**Authors**: Yibo Wang, Hai-Long Sun, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, Lijun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08198)  

**Abstract**: Recently, self-play fine-tuning (SPIN) has been proposed to adapt large language models to downstream applications with scarce expert-annotated data, by iteratively generating synthetic responses from the model itself. However, SPIN is designed to optimize the current reward advantages of annotated responses over synthetic responses at hand, which may gradually vanish during iterations, leading to unstable optimization. Moreover, the utilization of reference policy induces a misalignment issue between the reward formulation for training and the metric for generation. To address these limitations, we propose a novel Triplet-based Self-Play fIne-tuNing (T-SPIN) method that integrates two key designs. First, beyond current advantages, T-SPIN additionally incorporates historical advantages between iteratively generated responses and proto-synthetic responses produced by the initial policy. Even if the current advantages diminish, historical advantages remain effective, stabilizing the overall optimization. Second, T-SPIN introduces the entropy constraint into the self-play framework, which is theoretically justified to support reference-free fine-tuning, eliminating the training-generation discrepancy. Empirical results on various tasks demonstrate not only the superior performance of T-SPIN over SPIN, but also its stable evolution during iterations. Remarkably, compared to supervised fine-tuning, T-SPIN achieves comparable or even better performance with only 25% samples, highlighting its effectiveness when faced with scarce annotated data. 

---
# Evaluating Implicit Regulatory Compliance in LLM Tool Invocation via Logic-Guided Synthesis 

**Authors**: Da Song, Yuheng Huang, Boqi Chen, Tianshuo Cong, Randy Goebel, Lei Ma, Foutse Khomh  

**Link**: [PDF](https://arxiv.org/pdf/2601.08196)  

**Abstract**: The integration of large language models (LLMs) into autonomous agents has enabled complex tool use, yet in high-stakes domains, these systems must strictly adhere to regulatory standards beyond simple functional correctness. However, existing benchmarks often overlook implicit regulatory compliance, thus failing to evaluate whether LLMs can autonomously enforce mandatory safety constraints. To fill this gap, we introduce LogiSafetyGen, a framework that converts unstructured regulations into Linear Temporal Logic oracles and employs logic-guided fuzzing to synthesize valid, safety-critical traces. Building on this framework, we construct LogiSafetyBench, a benchmark comprising 240 human-verified tasks that require LLMs to generate Python programs that satisfy both functional objectives and latent compliance rules. Evaluations of 13 state-of-the-art (SOTA) LLMs reveal that larger models, despite achieving better functional correctness, frequently prioritize task completion over safety, which results in non-compliant behavior. 

---
# Prompt-Based Clarity Evaluation and Topic Detection in Political Question Answering 

**Authors**: Lavanya Prahallad, Sai Utkarsh Choudarypally, Pragna Prahallad, Pranathi Prahallad  

**Link**: [PDF](https://arxiv.org/pdf/2601.08176)  

**Abstract**: Automatic evaluation of large language model (LLM) responses requires not only factual correctness but also clarity, particularly in political question-answering. While recent datasets provide human annotations for clarity and evasion, the impact of prompt design on automatic clarity evaluation remains underexplored. In this paper, we study prompt-based clarity evaluation using the CLARITY dataset from the SemEval 2026 shared task. We compare a GPT-3.5 baseline provided with the dataset against GPT-5.2 evaluated under three prompting strategies: simple prompting, chain-of-thought prompting, and chain-of-thought with few-shot examples. Model predictions are evaluated against human annotations using accuracy and class-wise metrics for clarity and evasion, along with hierarchical exact match. Results show that GPT-5.2 consistently outperforms the GPT-3.5 baseline on clarity prediction, with accuracy improving from 56 percent to 63 percent under chain-of-thought with few-shot prompting. Chain-of-thought prompting yields the highest evasion accuracy at 34 percent, though improvements are less stable across fine-grained evasion categories. We further evaluate topic identification and find that reasoning-based prompting improves accuracy from 60 percent to 74 percent relative to human annotations. Overall, our findings indicate that prompt design reliably improves high-level clarity evaluation, while fine-grained evasion and topic detection remain challenging despite structured reasoning prompts. 

---
# Relational Knowledge Distillation Using Fine-tuned Function Vectors 

**Authors**: Andrea Kang, Yingnian Wu, Hongjing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2601.08169)  

**Abstract**: Representing relations between concepts is a core prerequisite for intelligent systems to make sense of the world. Recent work using causal mediation analysis has shown that a small set of attention heads encodes task representation in in-context learning, captured in a compact representation known as the function vector. We show that fine-tuning function vectors with only a small set of examples (about 20 word pairs) yields better performance on relation-based word-completion tasks than using the original vectors derived from causal mediation analysis. These improvements hold for both small and large language models. Moreover, the fine-tuned function vectors yield improved decoding performance for relation words and show stronger alignment with human similarity judgments of semantic relations. Next, we introduce the composite function vector - a weighted combination of fine-tuned function vectors - to extract relational knowledge and support analogical reasoning. At inference time, inserting this composite vector into LLM activations markedly enhances performance on challenging analogy problems drawn from cognitive science and SAT benchmarks. Our results highlight the potential of activation patching as a controllable mechanism for encoding and manipulating relational knowledge, advancing both the interpretability and reasoning capabilities of large language models. 

---
# SwiftMem: Fast Agentic Memory via Query-aware Indexing 

**Authors**: Anxin Tian, Yiming Li, Xing Li, Hui-Ling Zhen, Lei Chen, Xianzhi Yu, Zhenhua Dong, Mingxuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2601.08160)  

**Abstract**: Agentic memory systems have become critical for enabling LLM agents to maintain long-term context and retrieve relevant information efficiently. However, existing memory frameworks suffer from a fundamental limitation: they perform exhaustive retrieval across the entire storage layer regardless of query characteristics. This brute-force approach creates severe latency bottlenecks as memory grows, hindering real-time agent interactions. We propose SwiftMem, a query-aware agentic memory system that achieves sub-linear retrieval through specialized indexing over temporal and semantic dimensions. Our temporal index enables logarithmic-time range queries for time-sensitive retrieval, while the semantic DAG-Tag index maps queries to relevant topics through hierarchical tag structures. To address memory fragmentation during growth, we introduce an embedding-tag co-consolidation mechanism that reorganizes storage based on semantic clusters to improve cache locality. Experiments on LoCoMo and LongMemEval benchmarks demonstrate that SwiftMem achieves 47$\times$ faster search compared to state-of-the-art baselines while maintaining competitive accuracy, enabling practical deployment of memory-augmented LLM agents. 

---
# WISE-Flow: Workflow-Induced Structured Experience for Self-Evolving Conversational Service Agents 

**Authors**: Yuqing Zhou, Zhuoer Wang, Jie Yuan, Hong Wang, Samson Koelle, Ziwei Zhu, Wei Niu  

**Link**: [PDF](https://arxiv.org/pdf/2601.08158)  

**Abstract**: Large language model (LLM)-based agents are widely deployed in user-facing services but remain error-prone in new tasks, tend to repeat the same failure patterns, and show substantial run-to-run variability. Fixing failures via environment-specific training or manual patching is costly and hard to scale. To enable self-evolving agents in user-facing service environments, we propose WISE-Flow, a workflow-centric framework that converts historical service interactions into reusable procedural experience by inducing workflows with prerequisite-augmented action blocks. At deployment, WISE-Flow aligns the agent's execution trajectory to retrieved workflows and performs prerequisite-aware feasibility reasoning to achieve state-grounded next actions. Experiments on ToolSandbox and $\tau^2$-bench show consistent improvement across base models. 

---
# Mechanisms are Transferable: Data-Efficient Low-Resource Adaptation via Circuit-Targeted Supervised Fine-Tuning 

**Authors**: Khumaisa Nur'aini, Ayu Purwarianti, Alham Fikri Aji, Derry Wijaya  

**Link**: [PDF](https://arxiv.org/pdf/2601.08146)  

**Abstract**: Adapting LLMs to low-resource languages is difficult: labeled data is scarce, full-model fine-tuning is unstable, and continued cross-lingual tuning can cause catastrophic forgetting. We propose Circuit-Targeted Supervised Fine-Tuning (CT-SFT): a counterfactual-free adaptation of CD-T (Contextual Decomposition Transformer) that uses a label-balanced mean baseline and task-directional relevance scoring to identify a sparse set of task-relevant attention heads in a proxy-language checkpoint, then transfer learns to a target language by updating only those heads (plus LayerNorm) via head-level gradient masking. Across NusaX-Senti and XNLI, CT-SFT improves cross-lingual accuracy over continued full fine-tuning while updating only a small subset of model parameters. We find an editing-preserving trade-off: harder transfers favor editing circuit heads, while easier transfers often favor near-zero (i.e., low-relevance heads) updates, preserving the source mechanism. CT-SFT also substantially reduces catastrophic forgetting, preserving proxy/source-language competence during transfer. 

---
# Qalb: Largest State-of-the-Art Urdu Large Language Model for 230M Speakers with Systematic Continued Pre-training 

**Authors**: Muhammad Taimoor Hassan, Jawad Ahmed, Muhammad Awais  

**Link**: [PDF](https://arxiv.org/pdf/2601.08141)  

**Abstract**: Despite remarkable progress in large language models, Urdu-a language spoken by over 230 million people-remains critically underrepresented in modern NLP systems. Existing multilingual models demonstrate poor performance on Urdu-specific tasks, struggling with the language's complex morphology, right-to-left Nastaliq script, and rich literary traditions. Even the base LLaMA-3.1 8B-Instruct model shows limited capability in generating fluent, contextually appropriate Urdu text. We introduce Qalb, an Urdu language model developed through a two-stage approach: continued pre-training followed by supervised fine-tuning. Starting from LLaMA 3.1 8B, we perform continued pre-training on a dataset of 1.97 billion tokens. This corpus comprises 1.84 billion tokens of diverse Urdu text-spanning news archives, classical and contemporary literature, government documents, and social media-combined with 140 million tokens of English Wikipedia data to prevent catastrophic forgetting. We then fine-tune the resulting model on the Alif Urdu-instruct dataset. Through extensive evaluation on Urdu-specific benchmarks, Qalb demonstrates substantial improvements, achieving a weighted average score of 90.34 and outperforming the previous state-of-the-art Alif-1.0-Instruct model (87.1) by 3.24 points, while also surpassing the base LLaMA-3.1 8B-Instruct model by 44.64 points. Qalb achieves state-of-the-art performance with comprehensive evaluation across seven diverse tasks including Classification, Sentiment Analysis, and Reasoning. Our results demonstrate that continued pre-training on diverse, high-quality language data, combined with targeted instruction fine-tuning, effectively adapts foundation models to low-resource languages. 

---
# How Reliable are Confidence Estimators for Large Reasoning Models? A Systematic Benchmark on High-Stakes Domains 

**Authors**: Reza Khanmohammadi, Erfan Miahi, Simerjot Kaur, Ivan Brugere, Charese H. Smiley, Kundan Thind, Mohammad M. Ghassemi  

**Link**: [PDF](https://arxiv.org/pdf/2601.08134)  

**Abstract**: The miscalibration of Large Reasoning Models (LRMs) undermines their reliability in high-stakes domains, necessitating methods to accurately estimate the confidence of their long-form, multi-step outputs. To address this gap, we introduce the Reasoning Model Confidence estimation Benchmark (RMCB), a public resource of 347,496 reasoning traces from six popular LRMs across different architectural families. The benchmark is constructed from a diverse suite of datasets spanning high-stakes domains, including clinical, financial, legal, and mathematical reasoning, alongside complex general reasoning benchmarks, with correctness annotations provided for all samples. Using RMCB, we conduct a large-scale empirical evaluation of over ten distinct representation-based methods, spanning sequential, graph-based, and text-based architectures. Our central finding is a persistent trade-off between discrimination (AUROC) and calibration (ECE): text-based encoders achieve the best AUROC (0.672), while structurally-aware models yield the best ECE (0.148), with no single method dominating both. Furthermore, we find that increased architectural complexity does not reliably outperform simpler sequential baselines, suggesting a performance ceiling for methods relying solely on chunk-level hidden states. This work provides the most comprehensive benchmark for this task to date, establishing rigorous baselines and demonstrating the limitations of current representation-based paradigms. 

---
# Attention Projection Mixing and Exogenous Anchors 

**Authors**: Jonathan Su  

**Link**: [PDF](https://arxiv.org/pdf/2601.08131)  

**Abstract**: Transformers that reuse early-layer attention projections as residuals face a fundamental tension: the first layer must simultaneously serve as a stable reference for all deeper layers and as an effective computational block. To resolve this, we propose ExoFormer, which learns dedicated exogenous anchor projections outside the sequential layer stack, decoupling the anchor role from computational refinement. Through a unified normalized mixing framework (studying different coefficient granularities: elementwise, headwise, scalar) across all attention pathways (queries, keys, values, and gate logits), ExoFormer variants consistently outperform their internal-anchor counterparts. Moreover, the dynamic variant achieves a 2.13-point increase in downstream accuracy over the baseline and demonstrates superior data efficiency, matching baseline validation loss with 1.84x fewer tokens. ExoFormer also achieves a 2x reduction in attention sink compared to standard Gated Attention. Paradoxically, all ExoFormer variants exhibit signs of representation collapse. We explain this via an Offloading Hypothesis: external anchors preserve essential token identity, allowing layers to specialize exclusively in computational refinement. We release codes and models to facilitate future research. 

---
# Debiasing Large Language Models via Adaptive Causal Prompting with Sketch-of-Thought 

**Authors**: Bowen Li, Ziqi Xu, Jing Ren, Renqiang Luo, Xikun Zhang, Xiuzhen Zhang, Yongli Ren, Feng Xia  

**Link**: [PDF](https://arxiv.org/pdf/2601.08108)  

**Abstract**: Despite notable advancements in prompting methods for Large Language Models (LLMs), such as Chain-of-Thought (CoT), existing strategies still suffer from excessive token usage and limited generalisability across diverse reasoning tasks. To address these limitations, we propose an Adaptive Causal Prompting with Sketch-of-Thought (ACPS) framework, which leverages structural causal models to infer the causal effect of a query on its answer and adaptively select an appropriate intervention (i.e., standard front-door and conditional front-door adjustments). This design enables generalisable causal reasoning across heterogeneous tasks without task-specific retraining. By replacing verbose CoT with concise Sketch-of-Thought, ACPS enables efficient reasoning that significantly reduces token usage and inference cost. Extensive experiments on multiple reasoning benchmarks and LLMs demonstrate that ACPS consistently outperforms existing prompting baselines in terms of accuracy, robustness, and computational efficiency. 

---
# Query Suggestion for Retrieval-Augmented Generation via Dynamic In-Context Learning 

**Authors**: Fabian Spaeh, Tianyi Chen, Chen-Hao Chiang, Bin Shen  

**Link**: [PDF](https://arxiv.org/pdf/2601.08105)  

**Abstract**: Retrieval-augmented generation with tool-calling agents (agentic RAG) has become increasingly powerful in understanding, processing, and responding to user queries. However, the scope of the grounding knowledge is limited and asking questions that exceed this scope may lead to issues like hallucination. While guardrail frameworks aim to block out-of-scope questions (Rodriguez et al., 2024), no research has investigated the question of suggesting answerable queries in order to complete the user interaction.
In this paper, we initiate the study of query suggestion for agentic RAG. We consider the setting where user questions are not answerable, and the suggested queries should be similar to aid the user interaction. Such scenarios are frequent for tool-calling LLMs as communicating the restrictions of the tools or the underlying datasets to the user is difficult, and adding query suggestions enhances the interaction with the RAG agent. As opposed to traditional settings for query recommendations such as in search engines, ensuring that the suggested queries are answerable is a major challenge due to the RAG's multi-step workflow that demands a nuanced understanding of the RAG as a whole, which the executing LLM lacks. As such, we introduce robust dynamic few-shot learning which retrieves examples from relevant workflows. We show that our system can be self-learned, for instance on prior user queries, and is therefore easily applicable in practice. We evaluate our approach on three benchmark datasets based on two unlabeled question datasets collected from real-world user queries. Experiments on real-world datasets confirm that our method produces more relevant and answerable suggestions, outperforming few-shot and retrieval-only baselines, and thus enable safer, more effective user interaction with agentic RAG. 

---
# AdaJudge: Adaptive Multi-Perspective Judging for Reward Modeling 

**Authors**: Yongliang Miao, Yangyang Liang, Mengnan Du  

**Link**: [PDF](https://arxiv.org/pdf/2601.08097)  

**Abstract**: Reward modeling is essential for aligning large language models with human preferences, yet predominant architectures rely on a static pooling strategy to condense sequences into scalar scores. This paradigm, however, suffers from two key limitations: a static inductive bias that misaligns with task-dependent preference signals, and a representational mismatch, as the backbone is optimized for generation rather than fine-grained discrimination. To address this, we propose AdaJudge, a unified framework that jointly adapts representation and aggregation. AdaJudge first refines backbone representations into a discrimination-oriented space via gated refinement blocks. It then replaces the static readout with an adaptive multi-view pooling module that dynamically routes and combines evidence. Extensive experiments on RM-Bench and JudgeBench show that AdaJudge outperforms strong off-the-shelf reward models and traditional pooling baselines. 

---
# Calibration Is Not Enough: Evaluating Confidence Estimation Under Language Variations 

**Authors**: Yuxi Xia, Dennis Ulmer, Terra Blevins, Yihong Liu, Hinrich Schütze, Benjamin Roth  

**Link**: [PDF](https://arxiv.org/pdf/2601.08064)  

**Abstract**: Confidence estimation (CE) indicates how reliable the answers of large language models (LLMs) are, and can impact user trust and decision-making. Existing work evaluates CE methods almost exclusively through calibration, examining whether stated confidence aligns with accuracy, or discrimination, whether confidence is ranked higher for correct predictions than incorrect ones. However, these facets ignore pitfalls of CE in the context of LLMs and language variation: confidence estimates should remain consistent under semantically equivalent prompt or answer variations, and should change when the answer meaning differs. Therefore, we present a comprehensive evaluation framework for CE that measures their confidence quality on three new aspects: robustness of confidence against prompt perturbations, stability across semantic equivalent answers, and sensitivity to semantically different answers. In our work, we demonstrate that common CE methods for LLMs often fail on these metrics: methods that achieve good performance on calibration or discrimination are not robust to prompt variations or are not sensitive to answer changes. Overall, our framework reveals limitations of existing CE evaluations relevant for real-world LLM use cases and provides practical guidance for selecting and designing more reliable CE methods. 

---
# Universal computation is intrinsic to language model decoding 

**Authors**: Alex Lewandowski, Marlos C. Machado, Dale Schuurmans  

**Link**: [PDF](https://arxiv.org/pdf/2601.08061)  

**Abstract**: Language models now provide an interface to express and often solve general problems in natural language, yet their ultimate computational capabilities remain a major topic of scientific debate. Unlike a formal computer, a language model is trained to autoregressively predict successive elements in human-generated text. We prove that chaining a language model's autoregressive output is sufficient to perform universal computation. That is, a language model can simulate the execution of any algorithm on any input. The challenge of eliciting desired computational behaviour can thus be reframed in terms of programmability: the ease of finding a suitable prompt. Strikingly, we demonstrate that even randomly initialized language models are capable of universal computation before training. This implies that training does not give rise to computational expressiveness -- rather, it improves programmability, enabling a natural language interface for accessing these intrinsic capabilities. 

---
# Reasoning Beyond Chain-of-Thought: A Latent Computational Mode in Large Language Models 

**Authors**: Zhenghao He, Guangzhi Xiong, Bohan Liu, Sanchit Sinha, Aidong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08058)  

**Abstract**: Chain-of-Thought (CoT) prompting has improved the reasoning performance of large language models (LLMs), but it remains unclear why it works and whether it is the unique mechanism for triggering reasoning in large language models. In this work, we study this question by directly analyzing and intervening on the internal representations of LLMs with Sparse Autoencoders (SAEs), identifying a small set of latent features that are causally associated with LLM reasoning behavior. Across multiple model families and reasoning benchmarks, we find that steering a single reasoning-related latent feature can substantially improve accuracy without explicit CoT prompting. For large models, latent steering achieves performance comparable to standard CoT prompting while producing more efficient outputs. We further observe that this reasoning-oriented internal state is triggered early in generation and can override prompt-level instructions that discourage explicit reasoning. Overall, our results suggest that multi-step reasoning in LLMs is supported by latent internal activations that can be externally activated, while CoT prompting is one effective, but not unique, way of activating this mechanism rather than its necessary cause. 

---
# LLM Review: Enhancing Creative Writing via Blind Peer Review Feedback 

**Authors**: Weiyue Li, Mingxiao Song, Zhenda Shen, Dachuan Zhao, Yunfan Long, Yi Li, Yongce Li, Ruyi Yang, Mengyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08003)  

**Abstract**: Large Language Models (LLMs) often struggle with creative generation, and multi-agent frameworks that improve reasoning through interaction can paradoxically hinder creativity by inducing content homogenization. We introduce LLM Review, a peer-review-inspired framework implementing Blind Peer Review: agents exchange targeted feedback while revising independently, preserving divergent creative trajectories. To enable rigorous evaluation, we propose SciFi-100, a science fiction writing dataset with a unified framework combining LLM-as-a-judge scoring, human annotation, and rule-based novelty metrics. Experiments demonstrate that LLM Review consistently outperforms multi-agent baselines, and smaller models with our framework can surpass larger single-agent models, suggesting interaction structure may substitute for model scale. 

---
# Is Sentiment Banana-Shaped? Exploring the Geometry and Portability of Sentiment Concept Vectors 

**Authors**: Laurits Lyngbaek, Pascale Feldkamp, Yuri Bizzoni, Kristoffer L. Nielbo, Kenneth Enevoldsen  

**Link**: [PDF](https://arxiv.org/pdf/2601.07995)  

**Abstract**: Use cases of sentiment analysis in the humanities often require contextualized, continuous scores. Concept Vector Projections (CVP) offer a recent solution: by modeling sentiment as a direction in embedding space, they produce continuous, multilingual scores that align closely with human judgments. Yet the method's portability across domains and underlying assumptions remain underexplored. We evaluate CVP across genres, historical periods, languages, and affective dimensions, finding that concept vectors trained on one corpus transfer well to others with minimal performance loss. To understand the patterns of generalization, we further examine the linearity assumption underlying CVP. Our findings suggest that while CVP is a portable approach that effectively captures generalizable patterns, its linearity assumption is approximate, pointing to potential for further development. 

---
# DYCP: Dynamic Context Pruning for Long-Form Dialogue with LLMs 

**Authors**: Nayoung Choi, Jonathan Zhang, Jinho D. Choi  

**Link**: [PDF](https://arxiv.org/pdf/2601.07994)  

**Abstract**: Large Language Models (LLMs) often exhibit increased response latency and degraded answer quality as dialogue length grows, making effective context management essential. However, existing methods rely on extra LLM calls to build memory or perform offline memory construction without considering the current user utterance, which can introduce inefficiencies or disrupt conversational continuity. We introduce DyCP, a lightweight context management method that dynamically segment and retrieve relevant memory at query time. It preserves the sequential structure of dialogue without predefined topic boundaries and supports efficient, adaptive context retrieval. Across three long-form dialogue benchmarks, LoCoMo, MT-Bench+, and SCM4LLMs, and multiple LLMs, DyCP consistently improves answer quality while reducing response latency. We also examine the gap between modern LLMs' expanded context windows and their actual long-context processing capacity, highlighting the continued importance of effective context management. 

---
# From Word Sequences to Behavioral Sequences: Adapting Modeling and Evaluation Paradigms for Longitudinal NLP 

**Authors**: Adithya V Ganesan, Vasudha Varadarajan, Oscar NE Kjell, Whitney R Ringwald, Scott Feltman, Benjamin J Luft, Roman Kotov, Ryan L Boyd, H Andrew Schwartz  

**Link**: [PDF](https://arxiv.org/pdf/2601.07988)  

**Abstract**: While NLP typically treats documents as independent and unordered samples, in longitudinal studies, this assumption rarely holds: documents are nested within authors and ordered in time, forming person-indexed, time-ordered $\textit{behavioral sequences}$. Here, we demonstrate the need for and propose a longitudinal modeling and evaluation paradigm that consequently updates four parts of the NLP pipeline: (1) evaluation splits aligned to generalization over people ($\textit{cross-sectional}$) and/or time ($\textit{prospective}$); (2) accuracy metrics separating between-person differences from within-person dynamics; (3) sequence inputs to incorporate history by default; and (4) model internals that support different $\textit{coarseness}$ of latent state over histories (pooled summaries, explicit dynamics, or interaction-based models). We demonstrate the issues ensued by traditional pipeline and our proposed improvements on a dataset of 17k daily diary transcripts paired with PTSD symptom severity from 238 participants, finding that traditional document-level evaluation can yield substantially different and sometimes reversed conclusions compared to our ecologically valid modeling and evaluation. We tie our results to a broader discussion motivating a shift from word-sequence evaluation toward $\textit{behavior-sequence}$ paradigms for NLP. 

---
# VULCA-Bench: A Multicultural Vision-Language Benchmark for Evaluating Cultural Understanding 

**Authors**: Haorui Yu, Ramon Ruiz-Dolz, Diji Yang, Hang He, Fengrui Zhang, Qiufeng Yi  

**Link**: [PDF](https://arxiv.org/pdf/2601.07986)  

**Abstract**: We introduce VULCA-Bench, a multicultural art-critique benchmark for evaluating Vision-Language Models' (VLMs) cultural understanding beyond surface-level visual perception. Existing VLM benchmarks predominantly measure L1-L2 capabilities (object recognition, scene description, and factual question answering) while under-evaluate higher-order cultural interpretation. VULCA-Bench contains 7,410 matched image-critique pairs spanning eight cultural traditions, with Chinese-English bilingual coverage. We operationalise cultural understanding using a five-layer framework (L1-L5, from Visual Perception to Philosophical Aesthetics), instantiated as 225 culture-specific dimensions and supported by expert-written bilingual critiques. Our pilot results indicate that higher-layer reasoning (L3-L5) is consistently more challenging than visual and technical analysis (L1-L2). The dataset, evaluation scripts, and annotation tools are available under CC BY 4.0 in the supplementary materials. 

---
# Multilingual, Multimodal Pipeline for Creating Authentic and Structured Fact-Checked Claim Dataset 

**Authors**: Z. Melce Hüsünbeyi, Virginie Mouilleron, Leonie Uhling, Daniel Foppe, Tatjana Scheffler, Djamé Seddah  

**Link**: [PDF](https://arxiv.org/pdf/2601.07985)  

**Abstract**: The rapid proliferation of misinformation across online platforms underscores the urgent need for robust, up-to-date, explainable, and multilingual fact-checking resources. However, existing datasets are limited in scope, often lacking multimodal evidence, structured annotations, and detailed links between claims, evidence, and verdicts. This paper introduces a comprehensive data collection and processing pipeline that constructs multimodal fact-checking datasets in French and German languages by aggregating ClaimReview feeds, scraping full debunking articles, normalizing heterogeneous claim verdicts, and enriching them with structured metadata and aligned visual content. We used state-of-the-art large language models (LLMs) and multimodal LLMs for (i) evidence extraction under predefined evidence categories and (ii) justification generation that links evidence to verdicts. Evaluation with G-Eval and human assessment demonstrates that our pipeline enables fine-grained comparison of fact-checking practices across different organizations or media markets, facilitates the development of more interpretable and evidence-grounded fact-checking models, and lays the groundwork for future research on multilingual, multimodal misinformation verification. 

---
# Cross-Cultural Expert-Level Art Critique Evaluation with Vision-Language Models 

**Authors**: Haorui Yu, Ramon Ruiz-Dolz, Xuehang Wen, Fengrui Zhang, Qiufeng Yi  

**Link**: [PDF](https://arxiv.org/pdf/2601.07984)  

**Abstract**: Vision-Language Models (VLMs) excel at visual perception, yet their ability to interpret cultural meaning in art remains under-validated. We present a tri-tier evaluation framework for cross-cultural art-critique assessment: Tier I computes automated coverage and risk indicators offline; Tier II applies rubric-based scoring using a single primary judge across five dimensions; and Tier III calibrates the Tier II aggregate score to human ratings via isotonic regression, yielding a 5.2% reduction in MAE on a 152-sample held-out set. The framework outputs a calibrated cultural-understanding score for model selection and cultural-gap diagnosis, together with dimension-level diagnostics and risk indicators. We evaluate 15 VLMs on 294 expert anchors spanning six cultural traditions. Key findings are that (i) automated metrics are unreliable proxies for cultural depth, (ii) Western samples score higher than non-Western samples under our sampling and rubric, and (iii) cross-judge scale mismatch makes naive score averaging unreliable, motivating a single primary judge with explicit calibration. Dataset and code are available in the supplementary materials. 

---
# Explaining Generalization of AI-Generated Text Detectors Through Linguistic Analysis 

**Authors**: Yuxi Xia, Kinga Stańczak, Benjamin Roth  

**Link**: [PDF](https://arxiv.org/pdf/2601.07974)  

**Abstract**: AI-text detectors achieve high accuracy on in-domain benchmarks, but often struggle to generalize across different generation conditions such as unseen prompts, model families, or domains. While prior work has reported these generalization gaps, there are limited insights about the underlying causes. In this work, we present a systematic study aimed at explaining generalization behavior through linguistic analysis. We construct a comprehensive benchmark that spans 6 prompting strategies, 7 large language models (LLMs), and 4 domain datasets, resulting in a diverse set of human- and AI-generated texts. Using this dataset, we fine-tune classification-based detectors on various generation settings and evaluate their cross-prompt, cross-model, and cross-dataset generalization. To explain the performance variance, we compute correlations between generalization accuracies and feature shifts of 80 linguistic features between training and test conditions. Our analysis reveals that generalization performance for specific detectors and evaluation conditions is significantly associated with linguistic features such as tense usage and pronoun frequency. 

---
# Knowing But Not Doing: Convergent Morality and Divergent Action in LLMs 

**Authors**: Jen-tse Huang, Jiantong Qin, Xueli Qiu, Sharon Levy, Michelle R. Kaufman, Mark Dredze  

**Link**: [PDF](https://arxiv.org/pdf/2601.07972)  

**Abstract**: Value alignment is central to the development of safe and socially compatible artificial intelligence. However, how Large Language Models (LLMs) represent and enact human values in real-world decision contexts remains under-explored. We present ValAct-15k, a dataset of 3,000 advice-seeking scenarios derived from Reddit, designed to elicit ten values defined by Schwartz Theory of Basic Human Values. Using both the scenario-based questions and the traditional value questionnaire, we evaluate ten frontier LLMs (five from U.S. companies, five from Chinese ones) and human participants ($n = 55$). We find near-perfect cross-model consistency in scenario-based decisions (Pearson $r \approx 1.0$), contrasting sharply with the broad variability observed among humans ($r \in [-0.79, 0.98]$). Yet, both humans and LLMs show weak correspondence between self-reported and enacted values ($r = 0.4, 0.3$), revealing a systematic knowledge-action gap. When instructed to "hold" a specific value, LLMs' performance declines up to $6.6%$ compared to merely selecting the value, indicating a role-play aversion. These findings suggest that while alignment training yields normative value convergence, it does not eliminate the human-like incoherence between knowing and acting upon values. 

---
# A Human-Centric Pipeline for Aligning Large Language Models with Chinese Medical Ethics 

**Authors**: Haoan Jin, Han Ying, Jiacheng Ji, Hanhui Xu, Mengyue Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.07954)  

**Abstract**: Recent advances in large language models have enabled their application to a range of healthcare tasks. However, aligning LLMs with the nuanced demands of medical ethics, especially under complex real world scenarios, remains underexplored. In this work, we present MedES, a dynamic, scenario-centric benchmark specifically constructed from 260 authoritative Chinese medical, ethical, and legal sources to reflect the challenges in clinical decision-making. To facilitate model alignment, we introduce a guardian-in-the-loop framework that leverages a dedicated automated evaluator (trained on expert-labeled data and achieving over 97% accuracy within our domain) to generate targeted prompts and provide structured ethical feedback. Using this pipeline, we align a 7B-parameter LLM through supervised fine-tuning and domain-specific preference optimization. Experimental results, conducted entirely within the Chinese medical ethics context, demonstrate that our aligned model outperforms notably larger baselines on core ethical tasks, with observed improvements in both quality and composite evaluation metrics. Our work offers a practical and adaptable framework for aligning LLMs with medical ethics in the Chinese healthcare domain, and suggests that similar alignment pipelines may be instantiated in other legal and cultural environments through modular replacement of the underlying normative corpus. 

---
# EmbeddingRWKV: State-Centric Retrieval with Reusable States 

**Authors**: Haowen Hou, Jie Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.07861)  

**Abstract**: Current Retrieval-Augmented Generation (RAG) systems typically employ a traditional two-stage pipeline: an embedding model for initial retrieval followed by a reranker for refinement. However, this paradigm suffers from significant inefficiency due to the lack of shared information between stages, leading to substantial redundant computation. To address this limitation, we propose \textbf{State-Centric Retrieval}, a unified retrieval paradigm that utilizes "states" as a bridge to connect embedding models and rerankers. First, we perform state representation learning by fine-tuning an RWKV-based LLM, transforming it into \textbf{EmbeddingRWKV}, a unified model that serves as both an embedding model and a state backbone for extracting compact, reusable states. Building upon these reusable states, we further design a state-based reranker to fully leverage precomputed information. During reranking, the model processes only query tokens, decoupling inference cost from document length and yielding a 5.4$\times$--44.8$\times$ speedup. Furthermore, we observe that retaining all intermediate layer states is unnecessary; with a uniform layer selection strategy, our model maintains 98.62\% of full-model performance using only 25\% of the layers. Extensive experiments demonstrate that State-Centric Retrieval achieves high-quality retrieval and reranking results while significantly enhancing overall system efficiency. Code is available at \href{this https URL}{our GitHub repository}. 

---
# APEX-SWE 

**Authors**: Abhi Kottamasu, Akul Datta, Aakash Barthwal, Chirag Mahapatra, Ajay Arun, Adarsh Hiremath, Brendan Foody, Bertie Vidgen  

**Link**: [PDF](https://arxiv.org/pdf/2601.08806)  

**Abstract**: We introduce the AI Productivity Index for Software Engineering (APEX-SWE), a benchmark for assessing whether frontier AI models can execute economically valuable software engineering work. Unlike existing evaluations that focus on narrow, well-defined tasks, APEX-SWE assesses two novel task types that reflect real-world software engineering work: (1) Integration tasks (n=100), which require constructing end-to-end systems across heterogeneous cloud primitives, business applications, and infrastructure-as-code services, and (2) Observability tasks (n=100), which require debugging production failures using telemetry signals such as logs and dashboards, as well as unstructured context. We evaluated eight frontier models on APEX-SWE. Gemini 3 Pro (Thinking = High) performs best, with a Pass@1 score of 25\%. Our analysis shows that strong performance is primarily driven by epistemic reasoning, defined as the ability to distinguish between assumptions and verified facts, combined with agency to resolve uncertainty prior to acting. We open-source the APEX-SWE evaluation harness and a dev set (n=50). 

---
# Asymptotic Universal Alignment: A New Alignment Framework via Test-Time Scaling 

**Authors**: Yang Cai, Weiqiang Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.08777)  

**Abstract**: Aligning large language models (LLMs) to serve users with heterogeneous and potentially conflicting preferences is a central challenge for personalized and trustworthy AI. We formalize an ideal notion of universal alignment through test-time scaling: for each prompt, the model produces $k\ge 1$ candidate responses and a user selects their preferred one. We introduce $(k,f(k))$-robust alignment, which requires the $k$-output model to have win rate $f(k)$ against any other single-output model, and asymptotic universal alignment (U-alignment), which requires $f(k)\to 1$ as $k\to\infty$. Our main result characterizes the optimal convergence rate: there exists a family of single-output policies whose $k$-sample product policies achieve U-alignment at rate $f(k)=\frac{k}{k+1}$, and no method can achieve a faster rate in general.
We show that popular post-training methods, including Nash learning from human feedback (NLHF), can fundamentally underutilize the benefits of test-time scaling. Even though NLHF is optimal for $k=1$, sampling from the resulting (often deterministic) policy cannot guarantee win rates above $\tfrac{1}{2}$ except for an arbitrarily small slack. This stems from a lack of output diversity: existing alignment methods can collapse to a single majority-preferred response, making additional samples redundant. In contrast, our approach preserves output diversity and achieves the optimal test-time scaling rate. In particular, we propose a family of symmetric multi-player alignment games and prove that any symmetric Nash equilibrium policy of the $(k+1)$-player alignment game achieves the optimal $(k,\frac{k}{k+1})$-robust alignment. Finally, we provide theoretical convergence guarantees for self-play learning dynamics in these games and extend the framework to opponents that also generate multiple responses. 

---
# Rewarding the Rare: Uniqueness-Aware RL for Creative Problem Solving in LLMs 

**Authors**: Zhiyuan Hu, Yucheng Wang, Yufei He, Jiaying Wu, Yilun Zhao, See-Kiong Ng, Cynthia Breazeal, Anh Tuan Luu, Hae Won Park, Bryan Hooi  

**Link**: [PDF](https://arxiv.org/pdf/2601.08763)  

**Abstract**: Reinforcement learning (RL) has become a central paradigm for post-training large language models (LLMs), particularly for complex reasoning tasks, yet it often suffers from exploration collapse: policies prematurely concentrate on a small set of dominant reasoning patterns, improving pass@1 while limiting rollout-level diversity and gains in pass@k. We argue that this failure stems from regularizing local token behavior rather than diversity over sets of solutions. To address this, we propose Uniqueness-Aware Reinforcement Learning, a rollout-level objective that explicitly rewards correct solutions that exhibit rare high-level strategies. Our method uses an LLM-based judge to cluster rollouts for the same problem according to their high-level solution strategies, ignoring superficial variations, and reweights policy advantages inversely with cluster size. As a result, correct but novel strategies receive higher rewards than redundant ones. Across mathematics, physics, and medical reasoning benchmarks, our approach consistently improves pass@$k$ across large sampling budgets and increases the area under the pass@$k$ curve (AUC@$K$) without sacrificing pass@1, while sustaining exploration and uncovering more diverse solution strategies at scale. 

---
# Parallel Context-of-Experts Decoding for Retrieval Augmented Generation 

**Authors**: Giulio Corallo, Paolo Papotti  

**Link**: [PDF](https://arxiv.org/pdf/2601.08670)  

**Abstract**: Retrieval Augmented Generation faces a trade-off: concatenating documents in a long prompt enables multi-document reasoning but creates prefill bottlenecks, while encoding document KV caches separately offers speed but breaks cross-document interaction. We propose Parallel Context-of-Experts Decoding (Pced), a training-free framework that shifts evidence aggregation from the attention mechanism to the decoding. Pced treats retrieved documents as isolated "experts", synchronizing their predictions via a novel retrieval-aware contrastive decoding rule that weighs expert logits against the model prior. This approach recovers cross-document reasoning capabilities without constructing a shared attention across documents. 

---
# Learner-Tailored Program Repair: A Solution Generator with Iterative Edit-Driven Retrieval Enhancement 

**Authors**: Zhenlong Dai, Zhuoluo Zhao, Hengning Wang, Xiu Tang, Sai Wu, Chang Yao, Zhipeng Gao, Jingyuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.08545)  

**Abstract**: With the development of large language models (LLMs) in the field of programming, intelligent programming coaching systems have gained widespread attention. However, most research focuses on repairing the buggy code of programming learners without providing the underlying causes of the bugs. To address this gap, we introduce a novel task, namely \textbf{LPR} (\textbf{L}earner-Tailored \textbf{P}rogram \textbf{R}epair). We then propose a novel and effective framework, \textbf{\textsc{\MethodName{}}} (\textbf{L}earner-Tailored \textbf{S}olution \textbf{G}enerator), to enhance program repair while offering the bug descriptions for the buggy code. In the first stage, we utilize a repair solution retrieval framework to construct a solution retrieval database and then employ an edit-driven code retrieval approach to retrieve valuable solutions, guiding LLMs in identifying and fixing the bugs in buggy code. In the second stage, we propose a solution-guided program repair method, which fixes the code and provides explanations under the guidance of retrieval solutions. Moreover, we propose an Iterative Retrieval Enhancement method that utilizes evaluation results of the generated code to iteratively optimize the retrieval direction and explore more suitable repair strategies, improving performance in practical programming coaching scenarios. The experimental results show that our approach outperforms a set of baselines by a large margin, validating the effectiveness of our framework for the newly proposed LPR task. 

---
# What If TSF: A Benchmark for Reframing Forecasting as Scenario-Guided Multimodal Forecasting 

**Authors**: Jinkwan Jang, Hyunbin Jin, Hyungjin Park, Kyubyung Chae, Taesup Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.08509)  

**Abstract**: Time series forecasting is critical to real-world decision making, yet most existing approaches remain unimodal and rely on extrapolating historical patterns. While recent progress in large language models (LLMs) highlights the potential for multimodal forecasting, existing benchmarks largely provide retrospective or misaligned raw context, making it unclear whether such models meaningfully leverage textual inputs. In practice, human experts incorporate what-if scenarios with historical evidence, often producing distinct forecasts from the same observations under different scenarios. Inspired by this, we introduce What If TSF (WIT), a multimodal forecasting benchmark designed to evaluate whether models can condition their forecasts on contextual text, especially future scenarios. By providing expert-crafted plausible or counterfactual scenarios, WIT offers a rigorous testbed for scenario-guided multimodal forecasting. The benchmark is available at this https URL. 

---
# An Under-Explored Application for Explainable Multimodal Misogyny Detection in code-mixed Hindi-English 

**Authors**: Sargam Yadav, Abhishek Kaushik, Kevin Mc Daid  

**Link**: [PDF](https://arxiv.org/pdf/2601.08457)  

**Abstract**: Digital platforms have an ever-expanding user base, and act as a hub for communication, business, and connectivity. However, this has also allowed for the spread of hate speech and misogyny. Artificial intelligence models have emerged as an effective solution for countering online hate speech but are under explored for low resource and code-mixed languages and suffer from a lack of interpretability. Explainable Artificial Intelligence (XAI) can enhance transparency in the decisions of deep learning models, which is crucial for a sensitive domain such as hate speech detection. In this paper, we present a multi-modal and explainable web application for detecting misogyny in text and memes in code-mixed Hindi and English. The system leverages state-of-the-art transformer-based models that support multilingual and multimodal settings. For text-based misogyny identification, the system utilizes XLM-RoBERTa (XLM-R) and multilingual Bidirectional Encoder Representations from Transformers (mBERT) on a dataset of approximately 4,193 comments. For multimodal misogyny identification from memes, the system utilizes mBERT + EfficientNet, and mBERT + ResNET trained on a dataset of approximately 4,218 memes. It also provides feature importance scores using explainability techniques including Shapley Additive Values (SHAP) and Local Interpretable Model Agnostic Explanations (LIME). The application aims to serve as a tool for both researchers and content moderators, to promote further research in the field, combat gender based digital violence, and ensure a safe digital space. The system has been evaluated using human evaluators who provided their responses on Chatbot Usability Questionnaire (CUQ) and User Experience Questionnaire (UEQ) to determine overall usability. 

---
# Deconstructing Pre-training: Knowledge Attribution Analysis in MoE and Dense Models 

**Authors**: Bo Wang, Junzhuo Li, Hong Chen, Yuanlin Chu, Yuxuan Fan, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2601.08383)  

**Abstract**: Mixture-of-Experts (MoE) architectures decouple model capacity from per-token computation, enabling scaling beyond the computational limits imposed by dense scaling laws. Yet how MoE architectures shape knowledge acquisition during pre-training, and how this process differs from dense architectures, remains unknown. To address this issue, we introduce Gated-LPI (Log-Probability Increase), a neuron-level attribution metric that decomposes log-probability increase across neurons. We present a time-resolved comparison of knowledge acquisition dynamics in MoE and dense architectures, tracking checkpoints over 1.2M training steps (~ 5.0T tokens) and 600K training steps (~ 2.5T tokens), respectively. Our experiments uncover three patterns: (1) Low-entropy backbone. The top approximately 1% of MoE neurons capture over 45% of positive updates, forming a high-utility core, which is absent in the dense baseline. (2) Early consolidation. The MoE model locks into a stable importance profile within < 100K steps, whereas the dense model remains volatile throughout training. (3) Functional robustness. Masking the ten most important MoE attention heads reduces relational HIT@10 by < 10%, compared with > 50% for the dense model, showing that sparsity fosters distributed -- rather than brittle -- knowledge storage. These patterns collectively demonstrate that sparsity fosters an intrinsically stable and distributed computational backbone from early in training, helping bridge the gap between sparse architectures and training-time interpretability. 

---
# PosIR: Position-Aware Heterogeneous Information Retrieval Benchmark 

**Authors**: Ziyang Zeng, Dun Zhang, Yu Yan, Xu Sun, Yudong Zhou, Yuqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08363)  

**Abstract**: While dense retrieval models have achieved remarkable success, rigorous evaluation of their sensitivity to the position of relevant information (i.e., position bias) remains largely unexplored. Existing benchmarks typically employ position-agnostic relevance labels, conflating the challenge of processing long contexts with the bias against specific evidence locations. To address this challenge, we introduce PosIR (Position-Aware Information Retrieval), a comprehensive benchmark designed to diagnose position bias in diverse retrieval scenarios. PosIR comprises 310 datasets spanning 10 languages and 31 domains, constructed through a rigorous pipeline that ties relevance to precise reference spans, enabling the strict disentanglement of document length from information position. Extensive experiments with 10 state-of-the-art embedding models reveal that: (1) Performance on PosIR in long-context settings correlates poorly with the MMTEB benchmark, exposing limitations in current short-text benchmarks; (2) Position bias is pervasive and intensifies with document length, with most models exhibiting primacy bias while certain models show unexpected recency bias; (3) Gradient-based saliency analysis further uncovers the distinct internal attention mechanisms driving these positional preferences. In summary, PosIR serves as a valuable diagnostic framework to foster the development of position-robust retrieval systems. 

---
# When KV Cache Reuse Fails in Multi-Agent Systems: Cross-Candidate Interaction is Crucial for LLM Judges 

**Authors**: Sichu Liang, Zhenglin Wang, Jiajia Chu, Pengfei Xia, Hui Zang, Deyu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.08343)  

**Abstract**: Multi-agent LLM systems routinely generate multiple candidate responses that are aggregated by an LLM judge. To reduce the dominant prefill cost in such pipelines, recent work advocates KV cache reuse across partially shared contexts and reports substantial speedups for generation agents. In this work, we show that these efficiency gains do not transfer uniformly to judge-centric inference. Across GSM8K, MMLU, and HumanEval, we find that reuse strategies that are effective for execution agents can severely perturb judge behavior: end-task accuracy may appear stable, yet the judge's selection becomes highly inconsistent with dense prefill. We quantify this risk using Judge Consistency Rate (JCR) and provide diagnostics showing that reuse systematically weakens cross-candidate attention, especially for later candidate blocks. Our ablation further demonstrates that explicit cross-candidate interaction is crucial for preserving dense-prefill decisions. Overall, our results identify a previously overlooked failure mode of KV cache reuse and highlight judge-centric inference as a distinct regime that demands dedicated, risk-aware system design. 

---
# Demystifying the Slash Pattern in Attention: The Role of RoPE 

**Authors**: Yuan Cheng, Fengzhuo Zhang, Yunlong Hou, Cunxiao Du, Chao Du, Tianyu Pang, Aixin Sun, Zhuoran Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08297)  

**Abstract**: Large Language Models (LLMs) often exhibit slash attention patterns, where attention scores concentrate along the $\Delta$-th sub-diagonal for some offset $\Delta$. These patterns play a key role in passing information across tokens. But why do they emerge? In this paper, we demystify the emergence of these Slash-Dominant Heads (SDHs) from both empirical and theoretical perspectives. First, by analyzing open-source LLMs, we find that SDHs are intrinsic to models and generalize to out-of-distribution prompts. To explain the intrinsic emergence, we analyze the queries, keys, and Rotary Position Embedding (RoPE), which jointly determine attention scores. Our empirical analysis reveals two characteristic conditions of SDHs: (1) Queries and keys are almost rank-one, and (2) RoPE is dominated by medium- and high-frequency components. Under these conditions, queries and keys are nearly identical across tokens, and interactions between medium- and high-frequency components of RoPE give rise to SDHs. Beyond empirical evidence, we theoretically show that these conditions are sufficient to ensure the emergence of SDHs by formalizing them as our modeling assumptions. Particularly, we analyze the training dynamics of a shallow Transformer equipped with RoPE under these conditions, and prove that models trained via gradient descent exhibit SDHs. The SDHs generalize to out-of-distribution prompts. 

---
# MemoBrain: Executive Memory as an Agentic Brain for Reasoning 

**Authors**: Hongjin Qian, Zhao Cao, Zheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.08079)  

**Abstract**: Complex reasoning in tool-augmented agent frameworks is inherently long-horizon, causing reasoning traces and transient tool artifacts to accumulate and strain the bounded working context of large language models. Without explicit memory mechanisms, such accumulation disrupts logical continuity and undermines task alignment. This positions memory not as an auxiliary efficiency concern, but as a core component for sustaining coherent, goal-directed reasoning over long horizons.
We propose MemoBrain, an executive memory model for tool-augmented agents that constructs a dependency-aware memory over reasoning steps, capturing salient intermediate states and their logical relations. Operating as a co-pilot alongside the reasoning agent, MemoBrain organizes reasoning progress without blocking execution and actively manages the working context. Specifically, it prunes invalid steps, folds completed sub-trajectories, and preserves a compact, high-salience reasoning backbone under a fixed context budget. Together, these mechanisms enable explicit cognitive control over reasoning trajectories rather than passive context accumulation.
We evaluate MemoBrain on challenging long-horizon benchmarks, including GAIA, WebWalker, and BrowseComp-Plus, demonstrating consistent improvements over strong baselines. 

---
# Exploiting DINOv3-Based Self-Supervised Features for Robust Few-Shot Medical Image Segmentation 

**Authors**: Guoping Xu, Jayaram K. Udupa, Weiguo Lu, You Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08078)  

**Abstract**: Deep learning-based automatic medical image segmentation plays a critical role in clinical diagnosis and treatment planning but remains challenging in few-shot scenarios due to the scarcity of annotated training data. Recently, self-supervised foundation models such as DINOv3, which were trained on large natural image datasets, have shown strong potential for dense feature extraction that can help with the few-shot learning challenge. Yet, their direct application to medical images is hindered by domain differences. In this work, we propose DINO-AugSeg, a novel framework that leverages DINOv3 features to address the few-shot medical image segmentation challenge. Specifically, we introduce WT-Aug, a wavelet-based feature-level augmentation module that enriches the diversity of DINOv3-extracted features by perturbing frequency components, and CG-Fuse, a contextual information-guided fusion module that exploits cross-attention to integrate semantic-rich low-resolution features with spatially detailed high-resolution features. Extensive experiments on six public benchmarks spanning five imaging modalities, including MRI, CT, ultrasound, endoscopy, and dermoscopy, demonstrate that DINO-AugSeg consistently outperforms existing methods under limited-sample conditions. The results highlight the effectiveness of incorporating wavelet-domain augmentation and contextual fusion for robust feature representation, suggesting DINO-AugSeg as a promising direction for advancing few-shot medical image segmentation. Code and data will be made available on this https URL. 

---
# Semantic Gravity Wells: Why Negative Constraints Backfire 

**Authors**: Shailesh Rana  

**Link**: [PDF](https://arxiv.org/pdf/2601.08070)  

**Abstract**: Negative constraints (instructions of the form "do not use word X") represent a fundamental test of instruction-following capability in large language models. Despite their apparent simplicity, these constraints fail with striking regularity, and the conditions governing failure have remained poorly understood. This paper presents the first comprehensive mechanistic investigation of negative instruction failure. We introduce semantic pressure, a quantitative measure of the model's intrinsic probability of generating the forbidden token, and demonstrate that violation probability follows a tight logistic relationship with pressure ($p=\sigma(-2.40+2.27\cdot P_0)$; $n=40{,}000$ samples; bootstrap $95%$ CI for slope: $[2.21,,2.33]$). Through layer-wise analysis using the logit lens technique, we establish that the suppression signal induced by negative instructions is present but systematically weaker in failures: the instruction reduces target probability by only 5.2 percentage points in failures versus 22.8 points in successes -- a $4.4\times$ asymmetry. We trace this asymmetry to two mechanistically distinct failure modes. In priming failure (87.5% of violations), the instruction's explicit mention of the forbidden word paradoxically activates rather than suppresses the target representation. In override failure (12.5%), late-layer feed-forward networks generate contributions of $+0.39$ toward the target probability -- nearly $4\times$ larger than in successes -- overwhelming earlier suppression signals. Activation patching confirms that layers 23--27 are causally responsible: replacing these layers' activations flips the sign of constraint effects. These findings reveal a fundamental tension in negative constraint design: the very act of naming a forbidden word primes the model to produce it. 

---
# FigEx2: Visual-Conditioned Panel Detection and Captioning for Scientific Compound Figures 

**Authors**: Jifeng Song, Arun Das, Pan Wang, Hui Ji, Kun Zhao, Yufei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08026)  

**Abstract**: Scientific compound figures combine multiple labeled panels into a single image, but captions in real pipelines are often missing or only provide figure-level summaries, making panel-level understanding difficult. In this paper, we propose FigEx2, visual-conditioned framework that localizes panels and generates panel-wise captions directly from the compound figure. To mitigate the impact of diverse phrasing in open-ended captioning, we introduce a noise-aware gated fusion module that adaptively filters token-level features to stabilize the detection query space. Furthermore, we employ a staged optimization strategy combining supervised learning with reinforcement learning (RL), utilizing CLIP-based alignment and BERTScore-based semantic rewards to enforce strict multimodal consistency. To support high-quality supervision, we curate BioSci-Fig-Cap, a refined benchmark for panel-level grounding, alongside cross-disciplinary test suites in physics and chemistry. Experimental results demonstrate that FigEx2 achieves a superior 0.726 mAP@0.5:0.95 for detection and significantly outperforms Qwen3-VL-8B by 0.51 in METEOR and 0.24 in BERTScore. Notably, FigEx2 exhibits remarkable zero-shot transferability to out-of-distribution scientific domains without any fine-tuning. 

---
# Cultural Compass: A Framework for Organizing Societal Norms to Detect Violations in Human-AI Conversations 

**Authors**: Myra Cheng, Vinodkumar Prabhakaran, Alice Oh, Hayk Stepanyan, Aishwarya Verma, Charu Kalia, Erin MacMurray van Liemt, Sunipa Dev  

**Link**: [PDF](https://arxiv.org/pdf/2601.07973)  

**Abstract**: Generative AI models ought to be useful and safe across cross-cultural contexts. One critical step toward this goal is understanding how AI models adhere to sociocultural norms. While this challenge has gained attention in NLP, existing work lacks both nuance and coverage in understanding and evaluating models' norm adherence. We address these gaps by introducing a taxonomy of norms that clarifies their contexts (e.g., distinguishing between human-human norms that models should recognize and human-AI interactional norms that apply to the human-AI interaction itself), specifications (e.g., relevant domains), and mechanisms (e.g., modes of enforcement). We demonstrate how our taxonomy can be operationalized to automatically evaluate models' norm adherence in naturalistic, open-ended settings. Our exploratory analyses suggest that state-of-the-art models frequently violate norms, though violation rates vary by model, interactional context, and country. We further show that violation rates also vary by prompt intent and situational framing. Our taxonomy and demonstrative evaluation pipeline enable nuanced, context-sensitive evaluation of cultural norm adherence in realistic settings. 

---
# Towards Specialized Generalists: A Multi-Task MoE-LoRA Framework for Domain-Specific LLM Adaptation 

**Authors**: Yuxin Yang, Aoxiong Zeng, Xiangquan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.07935)  

**Abstract**: The rapid evolution of Large Language Models (LLMs) has shifted focus from general-purpose capabilities to domain-specific expertise. However, adapting LLMs to specialized fields such as medicine presents two challenge: (1) the "Stability-Plasticity Dilemma", where the model must acquire complex clinical knowledge without suffering from catastrophic forgetting of general world knowledge; and (2) "Task Interference", where disparate sub-tasks, such as medical diagnosis, report summarization, and drug-drug interaction prediction, compete for limited low-rank parameter space. In this paper, we propose Med-MoE-LoRA, a novel framework that integrates Mixture-of-Experts (MoE) with Low-Rank Adaptation (LoRA) to enable efficient multi-task domain adaptation, especially for medical scenarios. Drawing inspiration from recent advances, our framework employs an asymmetric expert distribution where deeper layers are equipped with a higher density of LoRA experts to capture complex semantic abstractions. We further introduce a "Knowledge-Preservation Plugin", inspired by LoRA MoE, to isolate and protect general-purpose reasoning. By utilizing soft merging with adaptive routing and rank-wise decoupling, Med-MoE-LoRA achieves superior performance in medical benchmarks while reducing interference. Experimental results demonstrate that our approach consistently outperforms standard LoRA and conventional MoE architectures across multiple clinical NLP tasks while retaining the model's general cognitive capabilities. 

---
# KVzap: Fast, Adaptive, and Faithful KV Cache Pruning 

**Authors**: Simon Jegou, Maximilian Jeblick  

**Link**: [PDF](https://arxiv.org/pdf/2601.07891)  

**Abstract**: Growing context lengths in transformer-based language models have made the key-value (KV) cache a critical inference bottleneck. While many KV cache pruning methods have been proposed, they have not yet been adopted in major inference engines due to speed--accuracy trade-offs. We introduce KVzap, a fast, input-adaptive approximation of KVzip that works in both prefilling and decoding. On Qwen3-8B, Llama-3.1-8B-Instruct, and Qwen3-32B across long-context and reasoning tasks, KVzap achieves $2$--$4\times$ KV cache compression with negligible accuracy loss and achieves state-of-the-art performance on the KVpress leaderboard. Code and models are available at this https URL. 

---
# Sliced-Wasserstein Distribution Alignment Loss Improves the Ultra-Low-Bit Quantization of Large Language Models 

**Authors**: Deyu Cao, Yixin Yin, Samin Aref  

**Link**: [PDF](https://arxiv.org/pdf/2601.07878)  

**Abstract**: The benefits of most large language models come with steep and often hidden economic and environmental costs due to their resource usage inefficiency during deployment. Model quantization improves energy and memory efficiency through representing model parameters by lower-precision values. However, compression below 4-bits often distorts activation distributions and degrades performance. We address this challenge by introducing a sliced Wasserstein loss function for distribution-aware calibration in ultra-low-bit post-training quantization. The proposed loss aligns the output distributions of full-precision and quantized models under random linear projections, complementing standard mean-squared error loss without adding any computational overhead during inference. Our proposed loss function can be incorporated with any post-training quantization framework that has a retraining component. We demonstrate the performance gains of our proposed model by incorporating it with two frontier methods known as OmniQuant and TesseraQ. Compared to these two baselines, the proposed loss consistently improves both perplexity and downstream task accuracy across multiple ultra-low-bit settings. Our proposed loss function recovers 4.12-20.37% of the OmniQuant's lost accuracy on the language model LLaMA-2-7B, 0.93-7.65% on OPT-6.7B, and 2.26-6.20% on LLaMA-2-13B. TesseraQ's accuracy degradation is recovered by 3.63-7.63% in relative terms when augmented by our proposed loss function. Taken together, these results demonstrate that distributional alignment provides a simple yet effective performance boost that can push the limits of frontier quantization methods. Our method is available on GitHub to facilitate future progress in ultra-low-bit quantization. 

---
