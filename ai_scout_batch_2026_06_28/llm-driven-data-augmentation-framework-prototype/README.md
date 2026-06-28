Problem: Lack of diverse and sufficient training data is a common bottleneck in developing robust machine learning models. Manually creating large, varied datasets is time-consuming, expensive, and often biased. This limits model generalization and performance, especially in niche domains or for underrepresented classes, leading to suboptimal model accuracy and robustness.

Approach: This framework leverages Large Language Models (LLMs) to automate and enhance data augmentation. By providing initial data samples and clear instructions, the LLM generates new, synthetic variations that retain key characteristics (e.g., sentiment, topic) but introduce linguistic diversity. This significantly expands the dataset's size and variability, leading to more resilient and accurate downstream models. The prototype demonstrates this by augmenting text-based sentiment data for product reviews.

Usage:
1. Install dependencies: `pip install -r requirements.txt`
2. Set your Google Gemini API key as an environment variable: `export GEMINI_API
