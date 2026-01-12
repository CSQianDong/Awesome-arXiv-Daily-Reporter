# Retrieval-Augmented Multi-LLM Ensemble for Industrial Part Specification Extraction 

**Authors**: Muzakkiruddin Ahmed Mohammed, John R. Talburt, Leon Claasssens, Adriaan Marais  

**Link**: [PDF](https://arxiv.org/pdf/2601.05266)  

**Abstract**: Industrial part specification extraction from unstructured text remains a persistent challenge in manufacturing, procurement, and maintenance, where manual processing is both time-consuming and error-prone. This paper introduces a retrieval-augmented multi-LLM ensemble framework that orchestrates nine state-of-the-art Large Language Models (LLMs) within a structured three-phase pipeline. RAGsemble addresses key limitations of single-model systems by combining the complementary strengths of model families including Gemini (2.0, 2.5, 1.5), OpenAI (GPT-4o, o4-mini), Mistral Large, and Gemma (1B, 4B, 3n-e4b), while grounding outputs in factual data using FAISS-based semantic retrieval. The system architecture consists of three stages: (1) parallel extraction by diverse LLMs, (2) targeted research augmentation leveraging high-performing models, and (3) intelligent synthesis with conflict resolution and confidence-aware scoring. RAG integration provides real-time access to structured part databases, enabling the system to validate, refine, and enrich outputs through similarity-based reference retrieval. Experimental results using real industrial datasets demonstrate significant gains in extraction accuracy, technical completeness, and structured output quality compared to leading single-LLM baselines. Key contributions include a scalable ensemble architecture for industrial domains, seamless RAG integration throughout the pipeline, comprehensive quality assessment mechanisms, and a production-ready solution suitable for deployment in knowledge-intensive manufacturing environments. 

---
# LLM2IR: simple unsupervised contrastive learning makes long-context LLM great retriever 

**Authors**: Xiaocong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.05262)  

**Abstract**: Modern dense information retrieval (IR) models usually rely on costly large-scale pretraining. In this paper, we introduce LLM2IR, an efficient unsupervised contrastive learning framework to convert any decoder-only large language model (LLM) to an information retrieval model. Despite its simplicity, the effectiveness is proven among different LLMs on multiple IR benchmarks including LoCo, LongEmbed and BEIR. We also find that models with a longer context length tend to have a stronger IR capacity by comparing task performances of models in the same model family. Our work not only provides an effective way to build IR models on the state-of-the-art LLMs, but also shed light on the relationship between information retrieval ability and model context length, which helps the design of better information retrievers. 

---
# ACR: Adaptive Context Refactoring via Context Refactoring Operators for Multi-Turn Dialogue 

**Authors**: Jiawei Shen, Jia Zhu, Hanghui Guo, Weijie Shi, Yue Cui, Qingyu Niu, Guoqing Ma, Yidan Liang, Jingjiang Liu, Yiling Wang, Shimin Di, Jiajie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.05589)  

**Abstract**: Large Language Models (LLMs) have shown remarkable performance in multi-turn dialogue. However, in multi-turn dialogue, models still struggle to stay aligned with what has been established earlier, follow dependencies across many turns, and avoid drifting into incorrect facts as the interaction grows longer. Existing approaches primarily focus on extending the context window, introducing external memory, or applying context compression, yet these methods still face limitations such as \textbf{contextual inertia} and \textbf{state drift}. To address these challenges, we propose the \textbf{A}daptive \textbf{C}ontext \textbf{R}efactoring \textbf{(ACR)} Framework, which dynamically monitors and reshapes the interaction history to mitigate contextual inertia and state drift actively. ACR is built on a library of context refactoring operators and a teacher-guided self-evolving training paradigm that learns when to intervene and how to refactor, thereby decoupling context management from the reasoning process. Extensive experiments on multi-turn dialogue demonstrate that our method significantly outperforms existing baselines while reducing token consumption. 

---
# Don't Break the Cache: An Evaluation of Prompt Caching for Long-Horizon Agentic Tasks 

**Authors**: Elias Lumer, Faheem Nizar, Akshaya Jangiti, Kevin Frank, Anmol Gulati, Mandar Phadate, Vamse Kumar Subbiah  

**Link**: [PDF](https://arxiv.org/pdf/2601.06007)  

**Abstract**: Recent advancements in Large Language Model (LLM) agents have enabled complex multi-turn agentic tasks requiring extensive tool calling, where conversations can span dozens of API calls with increasingly large context windows. However, although major LLM providers offer prompt caching to reduce cost and latency, its benefits for agentic workloads remain underexplored in the research literature. To our knowledge, no prior work quantifies these cost savings or compares caching strategies for multi-turn agentic tasks. We present a comprehensive evaluation of prompt caching across three major LLM providers (OpenAI, Anthropic, and Google) and compare three caching strategies, including full context caching, system prompt only caching, and caching that excludes dynamic tool results. We evaluate on DeepResearchBench, a multi-turn agentic benchmark where agents autonomously execute real-world web search tool calls to answer complex research questions, measuring both API cost and time to first token (TTFT) across over 500 agent sessions with 10,000-token system prompts. Our results demonstrate that prompt caching reduces API costs by 45-80% and improves time to first token by 13-31% across providers. We find that strategic prompt cache block control, such as placing dynamic content at the end of the system prompt, avoiding dynamic traditional function calling, and excluding dynamic tool results, provides more consistent benefits than naive full-context caching, which can paradoxically increase latency. Our analysis reveals nuanced variations in caching behavior across providers, and we provide practical guidance for implementing prompt caching in production agentic systems. 

---
# Can large language models interpret unstructured chat data on dynamic group decision-making processes? Evidence on joint destination choice 

**Authors**: Sung-Yoo Lim, Koki Sato, Kiyoshi Takami, Giancarlos Parady, Eui-Jin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.05582)  

**Abstract**: Social activities result from complex joint activity-travel decisions between group members. While observing the decision-making process of these activities is difficult via traditional travel surveys, the advent of new types of data, such as unstructured chat data, can help shed some light on these complex processes. However, interpreting these decision-making processes requires inferring both explicit and implicit factors. This typically involves the labor-intensive task of manually annotating dialogues to capture context-dependent meanings shaped by the social and cultural norms. This study evaluates the potential of Large Language Models (LLMs) to automate and complement human annotation in interpreting decision-making processes from group chats, using data on joint eating-out activities in Japan as a case study. We designed a prompting framework inspired by the knowledge acquisition process, which sequentially extracts key decision-making factors, including the group-level restaurant choice set and outcome, individual preferences of each alternative, and the specific attributes driving those preferences. This structured process guides the LLM to interpret group chat data, converting unstructured dialogues into structured tabular data describing decision-making factors. To evaluate LLM-driven outputs, we conduct a quantitative analysis using a human-annotated ground truth dataset and a qualitative error analysis to examine model limitations. Results show that while the LLM reliably captures explicit decision-making factors, it struggles to identify nuanced implicit factors that human annotators readily identified. We pinpoint specific contexts when LLM-based extraction can be trusted versus when human oversight remains essential. These findings highlight both the potential and limitations of LLM-based analysis for incorporating non-traditional data sources on social activities. 

---
# MemBuilder: Reinforcing LLMs for Long-Term Memory Construction via Attributed Dense Rewards 

**Authors**: Zhiyu Shen, Ziming Wu, Fuming Lai, Shaobing Lian, Yanghui Rao  

**Link**: [PDF](https://arxiv.org/pdf/2601.05488)  

**Abstract**: Maintaining consistency in long-term dialogues remains a fundamental challenge for LLMs, as standard retrieval mechanisms often fail to capture the temporal evolution of historical states. While memory-augmented frameworks offer a structured alternative, current systems rely on static prompting of closed-source models or suffer from ineffective training paradigms with sparse rewards. We introduce MemBuilder, a reinforcement learning framework that trains models to orchestrate multi-dimensional memory construction with attributed dense rewards. MemBuilder addresses two key challenges: (1) Sparse Trajectory-Level Rewards: we employ synthetic session-level question generation to provide dense intermediate rewards across extended trajectories; and (2) Multi-Dimensional Memory Attribution: we introduce contribution-aware gradient weighting that scales policy updates based on each component's downstream impact. Experimental results show that MemBuilder enables a 4B-parameter model to outperform state-of-the-art closed-source baselines, exhibiting strong generalization across long-term dialogue benchmarks. 

---
# Fusion Matters: Length-Aware Analysis of Positional-Encoding Fusion in Transformers 

**Authors**: Mohamed Amine Hallam, Kuo-Kun Tseng  

**Link**: [PDF](https://arxiv.org/pdf/2601.05807)  

**Abstract**: Transformers require positional encodings to represent sequence order, yet most prior work focuses on designing new positional encodings rather than examining how positional information is fused with token embeddings. In this paper, we study whether the fusion mechanism itself affects performance, particularly in long-sequence settings. We conduct a controlled empirical study comparing three canonical fusion strategies--element-wise addition, concatenation with projection, and scalar gated fusion--under identical Transformer architectures, data splits, and random seeds. Experiments on three text classification datasets spanning short (AG News), medium (IMDB), and long (ArXiv) sequences show that fusion choice has negligible impact on short texts but produces consistent gains on long documents. To verify that these gains are structural rather than stochastic, we perform paired-seed analysis and cross-dataset comparison across sequence-length regimes. Additional experiments on the ArXiv dataset indicate that the benefit of learnable fusion generalizes across multiple positional encoding families. Finally, we explore a lightweight convolutional gating mechanism that introduces local inductive bias at the fusion level, evaluated on long documents only. Our results indicate that positional-encoding fusion is a non-trivial design choice for long-sequence Transformers and should be treated as an explicit modeling decision rather than a fixed default. 

---
