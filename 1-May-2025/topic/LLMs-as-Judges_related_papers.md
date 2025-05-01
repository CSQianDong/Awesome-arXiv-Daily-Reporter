# Automatic Legal Writing Evaluation of LLMs 

**Authors**: Ramon Pires, Roseval Malaquias Junior, Rodrigo Nogueira  

**Link**: [PDF](https://arxiv.org/pdf/2504.21202)  

**Abstract**: Despite the recent advances in Large Language Models, benchmarks for evaluating legal writing remain scarce due to the inherent complexity of assessing open-ended responses in this domain. One of the key challenges in evaluating language models on domain-specific tasks is finding test datasets that are public, frequently updated, and contain comprehensive evaluation guidelines. The Brazilian Bar Examination meets these requirements. We introduce oab-bench, a benchmark comprising 105 questions across seven areas of law from recent editions of the exam. The benchmark includes comprehensive evaluation guidelines and reference materials used by human examiners to ensure consistent grading. We evaluate the performance of four LLMs on oab-bench, finding that Claude-3.5 Sonnet achieves the best results with an average score of 7.93 out of 10, passing all 21 exams. We also investigated whether LLMs can serve as reliable automated judges for evaluating legal writing. Our experiments show that frontier models like OpenAI's o1 achieve a strong correlation with human scores when evaluating approved exams, suggesting their potential as reliable automated evaluators despite the inherently subjective nature of legal writing assessment. The source code and the benchmark -- containing questions, evaluation guidelines, model-generated responses, and their respective automated evaluations -- are publicly available. 

---
# BiasGuard: A Reasoning-enhanced Bias Detection Tool For Large Language Models 

**Authors**: Zhiting Fan, Ruizhe Chen, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.21299)  

**Abstract**: Identifying bias in LLM-generated content is a crucial prerequisite for ensuring fairness in LLMs. Existing methods, such as fairness classifiers and LLM-based judges, face limitations related to difficulties in understanding underlying intentions and the lack of criteria for fairness judgment. In this paper, we introduce BiasGuard, a novel bias detection tool that explicitly analyzes inputs and reasons through fairness specifications to provide accurate judgments. BiasGuard is implemented through a two-stage approach: the first stage initializes the model to explicitly reason based on fairness specifications, while the second stage leverages reinforcement learning to enhance its reasoning and judgment capabilities. Our experiments, conducted across five datasets, demonstrate that BiasGuard outperforms existing tools, improving accuracy and reducing over-fairness misjudgments. We also highlight the importance of reasoning-enhanced decision-making and provide evidence for the effectiveness of our two-stage optimization pipeline. 

---
