This project demonstrates a multi-modal embedding fusion technique for information retrieval using Google's Gemini API. Traditional retrieval systems often struggle with content that combines different modalities, like text and images, leading to fragmented representations.

**Problem:** Effectively retrieve relevant multi-modal documents when queries themselves can be multi-modal. Simply concatenating embeddings from separate text and image models often fails to capture the intricate semantic relationships between modalities.

**Approach:** We leverage a powerful multi-modal foundation model (Gemini 2.5 Flash) to inherently fuse information from text and images into a single, cohesive embedding. Both the query and the documents are processed by the same model, which generates a unified vector representation. Retrieval is then performed by calculating the cosine similarity between the query embedding and document embeddings, identifying the most semantically relevant items.

**Usage:**
1.  **Set up your environment:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    pip install -r requirements.txt
    ```
2.  **Obtain Google Gemini API Key:** Visit Google AI Studio and create an API key.
3.  **Configure API Key:** Create a `.env` file in the project root with your API key:
    ```
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    ```
4.  **Run the demo:**
    ```bash
    python main.py
    ```
    The script will generate embeddings for sample multi-modal documents and a query, then print the retrieval results.
