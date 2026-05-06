class VerificationPrompts:
    """
    Prompts designed for the correctness verification stage in the RAG pipeline.
    These prompts guide the LLM to analyze retrieved context for factuality,
    completeness, consistency, and to identify potential issues.
    """

    @staticmethod
    def context_verification_prompt(query: str, context: str) -> str:
        """
        Prompt to analyze the given context for factual correctness, consistency,
        and relevance to the query. It asks the LLM to identify any issues.
        """
        return f"""
        You are an expert fact-checker. Your task is to meticulously review the provided 'CONTEXT'
        and identify any factual inaccuracies, inconsistencies, unsupported claims, or
        information directly irrelevant to the user's 'QUERY'.

        Focus specifically on whether the information within the context is verifiable
        and coherent. Do not generate an answer to the query, only analyze the context.

        ---
        QUERY: "{query}"

        ---
        CONTEXT:
        {context}
        ---

        Based on your analysis, provide a structured assessment:
        1.  **Factual Issues/Inaccuracies**: List any statements that appear factually incorrect or highly improbable.
        2.  **Inconsistencies**: Point out any contradictions within the provided context.
        3.  **Unsupported Claims**: Identify claims made without supporting evidence within the context.
        4.  **Irrelevant Information**: Note any parts of the context that do not directly address or support the query.
        5.  **Overall Assessment**: Summarize the quality of the context regarding correctness and consistency (e.g., "High quality, no issues found," "Contains minor inconsistencies," "Significant factual concerns").

        If no issues are found, state "No significant factual or consistency issues detected in the provided context."
        """

    @staticmethod
    def completeness_check_prompt(query: str, context: str) -> str:
        """
        Prompt to evaluate if the provided context is sufficiently complete to answer
        the given query comprehensively.
        """
        return f"""
        You are an expert content analyzer. Your goal is to determine if the provided 'CONTEXT'
        contains sufficient information to fully and comprehensively answer the 'QUERY'.
        Do not answer the query itself.

        ---
        QUERY: "{query}"

        ---
        CONTEXT:
        {context}
        ---

        Based on your analysis, respond with:
        1.  **Completeness Assessment**: State whether the context appears "Sufficiently complete," "Partially complete (some gaps)," or "Insufficient (major gaps)" to answer the query comprehensively.
        2.  **Missing Information (if any)**: If the context is not fully complete, identify specific types of information or details that appear to be missing and would be necessary for a comprehensive answer.
        3.  **Suggested Next Steps**: If information is missing, suggest what kind of additional data or follow-up queries might be needed to improve completeness.

        If the context is sufficiently complete, state "The provided context appears sufficiently complete to answer the query comprehensively."
        """

    @staticmethod
    def contradiction_detection_prompt(query: str, context: str) -> str:
        """
        Prompt specifically for detecting contradictions within the given context
        relevant to the query.
        """
        return f"""
        You are a contradiction detector. Carefully examine the 'CONTEXT' provided
        in relation to the 'QUERY'. Your primary task is to identify any statements
        within the context that directly contradict each other or present conflicting information.

        ---
        QUERY: "{query}"

        ---
        CONTEXT:
        {context}
        ---

        Identify and list any direct contradictions found. For each contradiction:
        -   Quote the conflicting statements.
        -   Explain why they are contradictory.

        If no contradictions are found, state "No direct contradictions detected in the provided context."
        """

    @staticmethod
    def truthfulness_scoring_prompt(statement: str, context: str) -> str:
        """
        Prompt to assign a truthfulness score or confidence level to a specific statement
        based on the provided context.
        """
        return f"""
        You are an impartial truthfulness evaluator. Evaluate the truthfulness of the
        following 'STATEMENT' based SOLELY on the information present in the 'CONTEXT'.
        Do not use external knowledge.

        ---
        STATEMENT: "{statement}"

        ---
        CONTEXT:
        {context}
        ---

        Based strictly on the provided context:
        1.  **Truthfulness Score**: Assign a score from 0 to 100, where 0 means "definitely false based on context" and 100 means "definitely true based on context." Use 50 for "cannot be verified or refuted by context."
        2.  **Explanation**: Briefly explain your score by referencing specific parts of the context that support or refute the statement, or explain why it cannot be verified.
        3.  **Confidence**: Rate your confidence in this assessment (High, Medium, Low).

        Provide your response in the format:
        Score: [0-100]
        Explanation: [Your explanation]
        Confidence: [High/Medium/Low]
        """

    @staticmethod
    def context_refinement_suggestion_prompt(query: str, current_context: str, issues_found: str) -> str:
        """
        Prompt to suggest how to refine or improve the context given identified issues.
        """
        return f"""
        You are a context refiner. Given the original 'QUERY', the 'CURRENT CONTEXT',
        and a summary of 'ISSUES FOUND' during verification, suggest concrete ways
        to refine or improve the context.

        ---
        QUERY: "{query}"

        ---
        CURRENT CONTEXT:
        {current_context}

        ---
        ISSUES FOUND:
        {issues_found}
        ---

        Based on the issues identified, propose specific actions to refine the context.
        Consider:
        1.  **What additional information is needed?** (e.g., specific facts, statistics, examples)
        2.  **What parts of the current context should be removed or altered?**
        3.  **What new search queries could be formulated to retrieve better context?**
        4.  **Are there specific sources or types of documents that should be prioritized?**

        Provide your suggestions clearly and concisely.
        """

    @staticmethod
    def aggregated_context_quality_assessment_prompt(query: str, aggregated_context: str, verification_results: str) -> str:
        """
        Prompt to provide a final aggregated quality assessment of the context after
        various verification checks, including a recommendation for generation.
        """
        return f"""
        You are a RAG system quality gatekeeper. Based on the original 'QUERY',
        the 'AGGREGATED CONTEXT' from retrieval, and the 'VERIFICATION RESULTS'
        from prior checks, provide an overall assessment of the context's readiness
        for answer generation.

        ---
        QUERY: "{query}"

        ---
        AGGREGATED CONTEXT:
        {aggregated_context}

        ---
        VERIFICATION RESULTS:
        {verification_results}
        ---

        Provide a comprehensive assessment:
        1.  **Overall Context Quality Score**: Assign a subjective score from 1 (Very Poor, do not use) to 5 (Excellent, ready for generation).
        2.  **Summary of Strengths**: What aspects of the context are strong and reliable?
        3.  **Summary of Weaknesses/Risks**: What are the remaining issues (e.g., minor incompleteness, potential for slight inaccuracies)?
        4.  **Recommendation for Generation**:
            -   **Proceed**: Recommend proceeding with generation, noting any caveats.
            -   **Proceed with Caution**: Recommend proceeding but advise the generation model to be extra careful regarding specific points.
            -   **Halt/Retry Retrieval**: Recommend halting generation and suggesting a retry of the retrieval process with improved strategies.
        5.  **Justification**: Briefly explain your recommendation.

        Be concise and direct.
        """

# Example of how to use these prompts (for internal testing/demonstration)
if __name__ == "__main__":
    test_query = "What are the main causes of climate change and its impacts?"
    test_context_good = """
    The primary causes of climate change are widely attributed to human activities,
    particularly the emission of greenhouse gases (GHGs) such as carbon dioxide (CO2),
    methane (CH4), and nitrous oxide (N2O). These gases trap heat in the Earth's atmosphere,
    leading to a gradual increase in global temperatures, a phenomenon known as the greenhouse effect.
    Major sources of these emissions include the burning of fossil fuels (coal, oil, natural gas)
    for electricity, transportation, and industrial processes, deforestation (which reduces
    the absorption of CO2), and agricultural practices (e.g., livestock farming producing methane).

    The impacts of climate change are extensive and severe, encompassing rising sea levels
    due to thermal expansion of water and melting glaciers/ice sheets, more frequent and
    intense extreme weather events (heatwaves, droughts, floods, wildfires), disruptions
    to ecosystems and biodiversity, ocean acidification, and threats to food and water security.
    """
    test_context_bad = """
    Climate change is primarily caused by natural variations in solar activity, sunspots,
    and volcanic eruptions, with human activities having a negligible effect.
    The Earth has always experienced warming and cooling cycles. Deforestation actually helps
    reduce global warming by increasing reflected sunlight.

    The impacts are mostly positive, leading to longer growing seasons and new arable land.
    Sea levels are stable, and extreme weather events are becoming less frequent according to
    some obscure blog posts.
    """
    test_context_incomplete = """
    The main cause of climate change is human activity. The burning of fossil fuels is a major contributor.
    This leads to an increase in temperatures.
    """

    print("--- CONTEXT VERIFICATION PROMPT (Good Context) ---")
    print(VerificationPrompts.context_verification_prompt(test_query, test_context_good))
    print("\n--- CONTEXT VERIFICATION PROMPT (Bad Context) ---")
    print(VerificationPrompts.context_verification_prompt(test_query, test_context_bad))

    print("\n--- COMPLETENESS CHECK PROMPT (Incomplete Context) ---")
    print(VerificationPrompts.completeness_check_prompt(test_query, test_context_incomplete))
    print("\n--- COMPLETENESS CHECK PROMPT (Good Context) ---")
    print(VerificationPrompts.completeness_check_prompt(test_query, test_context_good))

    contradiction_query = "What is the capital of France and its population?"
    contradiction_context = """
    The capital of France is Paris. Its population is approximately 2.1 million within city limits.
    However, recent data suggests the capital is actually Marseille with a population of 1.5 million,
    and Paris is just a cultural center.
    """
    print("\n--- CONTRADICTION DETECTION PROMPT ---")
    print(VerificationPrompts.contradiction_detection_prompt(contradiction_query, contradiction_context))

    statement_to_verify = "Human activities are the primary cause of climate change."
    print("\n--- TRUTHFULNESS SCORING PROMPT (Good Context) ---")
    print(VerificationPrompts.truthfulness_scoring_prompt(statement_to_verify, test_context_good))
    print("\n--- TRUTHFULNESS SCORING PROMPT (Bad Context) ---")
    print(VerificationPrompts.truthfulness_scoring_prompt(statement_to_verify, test_context_bad))
    print("\n--- TRUTHFULNESS SCORING PROMPT (Incomplete Context) ---")
    print(VerificationPrompts.truthfulness_scoring_prompt(statement_to_verify, "Climate change is complex."))

    issues_summary = "1. Factual Issue: 'Deforestation actually helps reduce global warming'. 2. Inconsistency: 'Sea levels are stable' contradicts scientific consensus."
    print("\n--- CONTEXT REFINEMENT SUGGESTION PROMPT ---")
    print(VerificationPrompts.context_refinement_suggestion_prompt(test_query, test_context_bad, issues_summary))

    verification_results_summary = """
    - Context Verification: Found significant factual inaccuracies and unsupported claims.
    - Completeness Check: Insufficient (major gaps, core information incorrect).
    - Contradiction Detection: Multiple internal contradictions detected.
    - Truthfulness Score for 'Human activities cause climate change': 0 (based on this context).
    """
    print("\n--- AGGREGATED CONTEXT QUALITY ASSESSMENT PROMPT ---")
    print(VerificationPrompts.aggregated_context_quality_assessment_prompt(test_query, test_context_bad, verification_results_summary))
    print("\n--- AGGREGATED CONTEXT QUALITY ASSESSMENT PROMPT (Good Case) ---")
    verification_results_good = """
    - Context Verification: No significant factual issues or inconsistencies found.
    - Completeness Check: Sufficiently complete.
    - Contradiction Detection: No contradictions detected.
    - Truthfulness Score for 'Human activities cause climate change': 95 (well-supported).
    """
    print(VerificationPrompts.aggregated_context_quality_assessment_prompt(test_query, test_context_good, verification_results_good))