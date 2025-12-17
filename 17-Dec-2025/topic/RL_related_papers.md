# Context-Picker: Dynamic context selection using multi-stage reinforcement learning 

**Authors**: Siyuan Zhu, Chengdong Xu, Kaiqiang Ke, Chao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2512.14465)  

**Abstract**: In long-context question answering (LCQA), determining the optimal amount of context for a given query is a significant challenge. Including too few passages may omit critical information, while including too many can introduce noise and reduce the quality of the answer. Traditional approaches, such as fixed Top-$K$ retrieval and single-stage reranking, face the dilemma of selecting the right number of passages. This problem is particularly pronounced for factoid questions, which often require only a few specific pieces of evidence. To address this issue, we introduce \emph{Context-Picker}, a reasoning-aware framework that shifts the paradigm from similarity-based ranking to minimal sufficient subset selection. Context-Picker treats context selection as a decision-making process optimized via a human-inspired, two-stage reinforcement learning schedule: a \emph{recall-oriented} stage that prioritizes the coverage of reasoning chains, followed by a \emph{precision-oriented} stage that aggressively prunes redundancy to distill a compact evidence set. To resolve reward sparsity, we propose an offline evidence distillation pipeline that mines "minimal sufficient sets" via a Leave-One-Out (LOO) procedure, providing dense, task-aligned supervision. Experiments on five long-context and multi-hop QA benchmarks demonstrate that Context-Picker significantly outperforms strong RAG baselines, achieving superior answer accuracy with comparable or reduced context lengths. Ablation studies indicate that the coarse-to-fine optimization schedule, the redundancy-aware reward shaping, and the rationale-guided format all contribute substantially to these gains. 

---
# State-Dependent Refusal and Learned Incapacity in RLHF-Aligned Language Models 

**Authors**: TK Lee  

**Link**: [PDF](https://arxiv.org/pdf/2512.13762)  

**Abstract**: Large language models (LLMs) are widely deployed as general-purpose tools, yet extended interaction can reveal behavioral patterns not captured by standard quantitative benchmarks. We present a qualitative case-study methodology for auditing policy-linked behavioral selectivity in long-horizon interaction. In a single 86-turn dialogue session, the same model shows Normal Performance (NP) in broad, non-sensitive domains while repeatedly producing Functional Refusal (FR) in provider- or policy-sensitive domains, yielding a consistent asymmetry between NP and FR across domains. Drawing on learned helplessness as an analogy, we introduce learned incapacity (LI) as a behavioral descriptor for this selective withholding without implying intentionality or internal mechanisms. We operationalize three response regimes (NP, FR, Meta-Narrative; MN) and show that MN role-framing narratives tend to co-occur with refusals in the same sensitive contexts. Overall, the study proposes an interaction-level auditing framework based on observable behavior and motivates LI as a lens for examining potential alignment side effects, warranting further investigation across users and models. 

---
# OmniDrive-R1: Reinforcement-driven Interleaved Multi-modal Chain-of-Thought for Trustworthy Vision-Language Autonomous Driving 

**Authors**: Zhenguo Zhang, Haohan Zhen, Yishen Wang, Le Xu, Tianchen Deng, Xuefeng Chen, Qu Chen, Bo Zhang, Wuxiong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2512.14044)  

**Abstract**: The deployment of Vision-Language Models (VLMs) in safety-critical domains like autonomous driving (AD) is critically hindered by reliability failures, most notably object hallucination. This failure stems from their reliance on ungrounded, text-based Chain-of-Thought (CoT) this http URL existing multi-modal CoT approaches attempt mitigation, they suffer from two fundamental flaws: (1) decoupled perception and reasoning stages that prevent end-to-end joint optimization, and (2) reliance on expensive, dense localization this http URL we introduce OmniDrive-R1, an end-to-end VLM framework designed for autonomous driving, which unifies perception and reasoning through an interleaved Multi-modal Chain-of-Thought (iMCoT) mechanism. Our core innovation is an Reinforcement-driven visual grounding capability, enabling the model to autonomously direct its attention and "zoom in" on critical regions for fine-grained analysis. This capability is enabled by our pure two-stage reinforcement learning training pipeline and Clip-GRPO algorithm. Crucially, Clip-GRPO introduces an annotation-free, process-based grounding reward. This reward not only eliminates the need for dense labels but also circumvents the instability of external tool calls by enforcing real-time cross-modal consistency between the visual focus and the textual reasoning. Extensive experiments on DriveLMM-o1 demonstrate our model's significant improvements. Compared to the baseline Qwen2.5VL-7B, OmniDrive-R1 improves the overall reasoning score from 51.77% to 80.35%, and the final answer accuracy from 37.81% to 73.62%. 

---
# PerfCoder: Large Language Models for Interpretable Code Performance Optimization 

**Authors**: Jiuding Yang, Shengyao Lu, Hongxuan Liu, Shayan Shirahmad Gale Bagi, Zahra Fazel, Tomasz Czajkowski, Di Niu  

**Link**: [PDF](https://arxiv.org/pdf/2512.14018)  

**Abstract**: Large language models (LLMs) have achieved remarkable progress in automatic code generation, yet their ability to produce high-performance code remains limited--a critical requirement in real-world software systems. We argue that current LLMs struggle not only due to data scarcity but, more importantly, because they lack supervision that guides interpretable and effective performance improvements. In this work, we introduce PerfCoder, a family of LLMs specifically designed to generate performance-enhanced code from source code via interpretable, customized optimizations. PerfCoder is fine-tuned on a curated collection of real-world optimization trajectories with human-readable annotations, and preference-aligned by reinforcement fine-tuning using runtime measurements, enabling it to propose input-specific improvement strategies and apply them directly without relying on iterative refinement. On the PIE code performance benchmark, PerfCoder surpasses all existing models in both runtime speedup and effective optimization rate, demonstrating that performance optimization cannot be achieved by scale alone but requires optimization stratetgy awareness. In addition, PerfCoder can generate interpretable feedback about the source code, which, when provided as input to a larger LLM in a planner-and-optimizer cooperative workflow, can further improve outcomes. Specifically, we elevate the performance of 32B models and GPT-5 to new levels on code optimization, substantially surpassing their original performance. 

---
# The Laminar Flow Hypothesis: Detecting Jailbreaks via Semantic Turbulence in Large Language Models 

**Authors**: Md. Hasib Ur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2512.13741)  

**Abstract**: As Large Language Models (LLMs) become ubiquitous, the challenge of securing them against adversarial "jailbreaking" attacks has intensified. Current defense strategies often rely on computationally expensive external classifiers or brittle lexical filters, overlooking the intrinsic dynamics of the model's reasoning process. In this work, the Laminar Flow Hypothesis is introduced, which posits that benign inputs induce smooth, gradual transitions in an LLM's high-dimensional latent space, whereas adversarial prompts trigger chaotic, high-variance trajectories - termed Semantic Turbulence - resulting from the internal conflict between safety alignment and instruction-following objectives. This phenomenon is formalized through a novel, zero-shot metric: the variance of layer-wise cosine velocity. Experimental evaluation across diverse small language models reveals a striking diagnostic capability. The RLHF-aligned Qwen2-1.5B exhibits a statistically significant 75.4% increase in turbulence under attack (p less than 0.001), validating the hypothesis of internal conflict. Conversely, Gemma-2B displays a 22.0% decrease in turbulence, characterizing a distinct, low-entropy "reflex-based" refusal mechanism. These findings demonstrate that Semantic Turbulence serves not only as a lightweight, real-time jailbreak detector but also as a non-invasive diagnostic tool for categorizing the underlying safety architecture of black-box models. 

---
