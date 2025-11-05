# Oolong: Evaluating Long Context Reasoning and Aggregation Capabilities 

**Authors**: Amanda Bertsch, Adithya Pratapa, Teruko Mitamura, Graham Neubig, Matthew R. Gormley  

**Link**: [PDF](https://arxiv.org/pdf/2511.02817)  

**Abstract**: As model context lengths continue to grow, concerns about whether models effectively use the full context length have persisted. While several carefully designed long-context evaluations have recently been released, these evaluations tend to rely on retrieval from one or more sections of the context, which allows nearly all of the context tokens to be disregarded as noise. This represents only one type of task that might be performed with long context. We introduce Oolong, a benchmark of long-context reasoning tasks that require analyzing individual chunks of text on an atomic level, and then aggregating these analyses to answer distributional questions. Oolong is separated into two task sets: Oolong-synth, a set of naturalistic synthetic tasks, where we can easily ablate components of the reasoning problem; and Oolong-real, a downstream setting which requires reasoning over real-world conversational data. Oolong requires models to reason over large quantities of examples, to perform both classification and counting in-context, and to reason over temporal and user relations. Even frontier models struggle on Oolong, with GPT-5, Claude-Sonnet-4, and Gemini-2.5-Pro all achieving less than 50% accuracy on both splits at 128K. We release the data and evaluation harness for Oolong to enable further development of models that can reason over large quantities of text. 

---
