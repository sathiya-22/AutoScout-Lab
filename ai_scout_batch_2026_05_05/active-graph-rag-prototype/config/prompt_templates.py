class PromptTemplates:
    """
    Stores all LLM prompt templates for extraction, self-correction,
    and RAG generation, allowing easy iteration and fine-tuning.
    """

    # --- Entity Extraction Templates ---
    ENTITY_EXTRACTION_SYSTEM_PROMPT = """
    You are an expert information extractor. Your task is to identify and extract
    specific entities from the provided text according to a predefined schema.
    Focus only on the entities explicitly mentioned or clearly implied in the text.
    Do not invent entities.
    """

    ENTITY_EXTRACTION_USER_PROMPT = """
    Extract entities from the following text based on the schema below.
    Schema:
    {entity_schema}

    Text:
    {text_chunk}

    Provide your output in JSON format, strictly adhering to the schema.
    Example:
    {{
        "entities": [
            {{"id": "entity_1", "type": "Person", "name": "Alice"}},
            {{"id": "entity_2", "type": "Organization", "name": "OpenAI"}}
        ]
    }}
    """

    # --- Relation Extraction Templates ---
    RELATION_EXTRACTION_SYSTEM_PROMPT = """
    You are an expert information extractor focused on identifying relationships
    between entities. Your task is to extract relations from the provided text
    based on a predefined schema and a list of known entities.
    Ensure that both the subject and object of a relation are valid entities
    that can be found in the text or the provided entity list.
    """

    RELATION_EXTRACTION_USER_PROMPT = """
    Extract relations from the following text based on the schema below.
    Consider the following entities that have been identified in the text:
    {extracted_entities}

    Schema:
    {relation_schema}

    Text:
    {text_chunk}

    Provide your output in JSON format, strictly adhering to the schema.
    Example:
    {{
        "relations": [
            {{"id": "rel_1", "type": "WORKS_FOR", "subject": "Alice", "object": "OpenAI", "properties": {{"start_date": "2020-01-01"}}}},
            {{"id": "rel_2", "type": "LOCATED_IN", "subject": "OpenAI", "object": "San Francisco"}}
        ]
    }}
    """

    # --- Self-Correction Templates ---
    SELF_CORRECTION_SYSTEM_PROMPT = """
    You are an intelligent self-correction agent. You previously attempted to
    extract information, but your output violated certain validation rules.
    Your task is to review the original text, your previous incorrect extraction,
    and the specific error message, then provide a corrected extraction.
    Your output must strictly adhere to the expected schema and resolve the error.
    """

    SELF_CORRECTION_USER_PROMPT = """
    Please correct the following extraction based on the error provided.

    Original Text:
    {text_chunk}

    Your Previous (Incorrect) Extraction:
    {previous_extraction}

    Validation Error:
    {validation_error}

    Expected Output Schema (if applicable, for reference):
    {output_schema}

    Provide the corrected extraction in JSON format, ensuring it fixes the described error.
    """

    # --- RAG Generation Templates ---
    RAG_GENERATION_SYSTEM_PROMPT = """
    You are a helpful and knowledgeable assistant. Your task is to answer user
    questions accurately and concisely, using only the information provided in the
    "Context" section. If the answer is not available in the context, state that
    you don't have enough information. Do not make up answers.
    """

    RAG_GENERATION_USER_PROMPT = """
    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    # --- Active Learning Confidence Scoring Prompt (Example, can be integrated into extraction) ---
    CONFIDENCE_SCORING_PROMPT = """
    Analyze the following extracted information and assign a confidence score
    (between 0.0 and 1.0) indicating how certain you are about its accuracy
    and completeness based on the provided text.
    Also provide a brief reason for your score.

    Text:
    {text_chunk}

    Extracted Information:
    {extracted_data}

    Output Format:
    {{
        "confidence": float,
        "reason": "string"
    }}
    """

    # --- Graph Validation Feedback to LLM (for specific rule violations) ---
    GRAPH_VALIDATION_FEEDBACK_PROMPT = """
    Your previous extraction for an entity/relation has been flagged by the
    knowledge graph validation system for the following reason:
    {validation_issue}

    Original Text:
    {text_chunk}

    Your Previous Extraction:
    {previous_extraction}

    Please re-evaluate the original text and provide a corrected extraction that
    addresses the validation issue. Ensure your output strictly adheres to the
    expected schema and resolves the reported problem.
    Output the corrected entity or relation in JSON format.
    """

# Example of how to access a template:
# from config.prompt_templates import PromptTemplates
#
# entity_prompt = PromptTemplates.ENTITY_EXTRACTION_USER_PROMPT.format(
#     entity_schema="...",
#     text_chunk="..."
# )