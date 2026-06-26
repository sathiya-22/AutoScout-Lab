```python
import os
import google.generativeai as genai
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from dotenv import load_dotenv

from config import Settings

# Load environment variables from .env file
load_dotenv()

def create_dummy_image(text: str) -> Image.Image:
    """Creates a simple dummy image with text for demonstration."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new('RGB', (200, 100), color = (73, 109, 137))
        d = ImageDraw.Draw(img)
        try:
            # Try to load a default font, fall back if not available
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default() # Fallback to default font
        d.text((10,10), text, fill=(255,255,0), font=font)
        return img
    except ImportError:
        print("Pillow not installed, returning a placeholder. Install with `pip install Pillow` for full functionality.")
        return None # In a real scenario, handle this gracefully

def get_image_bytes(image: Image.Image) -> bytes:
    """Converts a PIL Image to bytes."""
    if image is None:
        return b'' # Return empty bytes if image creation failed
    byte_arr = BytesIO()
    image.save(byte_arr, format='PNG')
    return byte_arr.getvalue()

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculates the cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def main():
    settings = Settings()

    # Configure the Google Generative AI SDK
    genai.configure(api_key=settings.api_key)

    # Initialize the generative model for embedding content
    # gemini-2.5-flash is used as it supports multi-modal content embedding
    model = genai.GenerativeModel(settings.model_name)

    print(f"Using model: {settings.model_name}")

    # Sample multi-modal documents
    documents = [
        {
            "id": "doc1",
            "text": "A serene landscape with mountains and a clear blue lake.",
            "image": create_dummy_image("Mountains & Lake")
        },
        {
            "id": "doc2",
            "text": "A bustling city street at night, full of bright lights and cars.",
            "image": create_dummy_image("City Night")
        },
        {
            "id": "doc3",
            "text": "A close-up of a delicious-looking pizza with various toppings.",
            "image": create_dummy_image("Delicious Pizza")
        },
        {
            "id": "doc4",
            "text": "An abstract painting with vibrant colors and geometric shapes.",
            "image": create_dummy_image("Abstract Art")
        },
    ]

    # Generate embeddings for documents
    document_embeddings = {}
    print("\nGenerating document embeddings...")
    for doc in documents:
        content_parts = [genai.types.text_part(doc["text"])]
        if doc["image"]:
            content_parts.append(genai.types.image_part(get_image_bytes(doc["image"])))

        response = model.embed_content(
            model=settings.model_name, # Specify model again for embed_content
            content=content_parts,
            task_type="RETRIEVAL_DOCUMENT"
        )
        document_embeddings[doc["id"]] = np.array(response['embedding'])
        print(f"  - Embedded {doc['id']}")

    # Sample multi-modal query
    query_text = "Find images of food items."
    query_image = create_dummy_image("Food Query")
    query_content_parts = [genai.types.text_part(query_text)]
    if query_image:
        query_content_parts.append(genai.types.image_part(get_image_bytes(query_image)))

    # Generate embedding for the query
    print("\nGenerating query embedding...")
    query_response = model.embed_content(
        model=settings.model_name,
        content=query_content_parts,
        task_type="RETRIEVAL_QUERY"
    )
    query_embedding = np.array(query_response['embedding'])
    print("  - Query embedded successfully.")

    # Perform retrieval by calculating similarity
    print("\nCalculating similarities for retrieval...")
    similarities = []
    for doc_id, doc_embed in document_embeddings.items():
        sim = cosine_similarity(query_embedding, doc_embed)
        similarities.append({"doc_id": doc_id, "similarity": sim})

    # Sort by similarity in descending order
    retrieval_results = sorted(similarities, key=lambda x: x["similarity"], reverse=True)

    print("\n--- Retrieval Results (Highest Similarity First) ---")
    for result in retrieval_results:
        print(f"Document ID: {result['doc_id']}, Similarity: {result['similarity']:.4f}")

    print("\nDemo complete. Remember to replace dummy images with actual multi-modal content for real-world use.")

if __name__ == "__main__":
    main()
```
