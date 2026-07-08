# Major problem of week 2026-W28

**Difficulty integrating LLMs with existing tools and data**  (id: `difficulty-integrating-llms-with-existing-tools-and-data`, signal: 275)

Developers face challenges in seamlessly integrating LLMs and AI agents with existing software, particularly for tasks like reading and editing common file formats (e.g., Microsoft Office files) or accessing structured data. This requires custom tooling and connectors, increasing development effort and complexity.

## Why this one

The problem of integrating LLMs with existing tools and data affects a vast number of developers across almost every industry, as most real-world applications require interaction with established systems. Its severity is high because it directly impedes the practical deployment and utility of AI agents, turning promising prototypes into integration nightmares. While some custom solutions exist, a standardized, open-source approach is largely absent, making this a prime candidate for a widely beneficial project. A small project can feasibly tackle a subset of common data formats and tools, demonstrating a path forward.

## Sources

- https://news.ycombinator.com/item?id=48807225
- https://github.com/perber/leafwiki/issues/1273
- https://github.com/keephq/keep/issues/6618

Daily prototype: https://github.com/sathiya-22/difficulty-integrating-llms-with-existing-tools-and-data-2026-07-07

---

## Problem
Developers are struggling to integrate Large Language Models (LLMs) and AI agents with existing enterprise tools and data sources. This includes common file formats like Microsoft Office documents (Word, Excel, PowerPoint), PDFs, and structured data in databases or legacy systems. The current landscape requires significant custom development for each integration, leading to increased development effort, complexity, and a barrier to widespread AI adoption in practical business scenarios.

## Evidence
*   **High Signal:** The problem `difficulty-integrating-llms-with-existing-tools-and-data` has the highest signal (275) among the scouted problems, indicating significant community interest and pain points.
*   **Community Discussions:** Multiple sources, including Hacker News threads (`https://news.ycombinator.com/item?id=48807225`) and GitHub issues (`https://github.com/perber/leafwiki/issues/1273`, `https://github.com/keephq/keep/issues/6618`), highlight the need for better integration capabilities.
*   **Existing Prototypes:** The `status: prototyped` and `prototype_repo: https://github.com/sathiya-22/difficulty-integrating-llms-with-existing-tools-and-data-2026-07-07` indicate that developers are already attempting to solve this, but a robust, open-source solution is still lacking.

## Proposed solution
We propose building an open-source `AI Agent Tooling Kit` (tentative name: `AgentConnect`) that provides a standardized, extensible framework for AI agents to interact with common file formats and data sources. This kit will offer a collection of pre-built 'connectors' or 'tools' that agents can utilize, abstracting away the complexities of parsing, reading, writing, and manipulating data within these formats. The focus will be on creating a modular system where new connectors can be easily added by the community.

## MVP scope
The MVP will focus on enabling AI agents to read and extract information from two common, complex document types:

1.  **Microsoft Word Documents (.docx):**
    *   **Tool 1: `read_word_document(file_path)`:** An agent tool that takes a `.docx` file path and returns its full text content, preserving basic formatting (e.g., paragraphs, headings).
    *   **Tool 2: `extract_sections_word(file_path, section_titles)`:** An agent tool that takes a `.docx` file path and a list of section titles (e.g., 'Introduction', 'Conclusion') and returns the text content of those specific sections.

2.  **PDF Documents (.pdf):**
    *   **Tool 3: `read_pdf_document(file_path)`:** An agent tool that takes a `.pdf` file path and returns its full text content, handling basic text extraction.
    *   **Tool 4: `extract_tables_pdf(file_path, page_numbers)`:** An agent tool that takes a `.pdf` file path and optional page numbers, returning extracted tables in a structured format (e.g., list of lists or pandas DataFrame).

The MVP will be implemented in Python, leveraging existing libraries like `python-docx` and `PyPDF2` (or similar) for the underlying parsing, and designed to be easily integrated with popular agent frameworks (e.g., LangChain, LlamaIndex) through a simple function-calling interface.

## Milestones

*   **Milestone 1: Core Document Parsing (2 weeks)**
    *   Set up project structure, CI/CD, and basic documentation.
    *   Implement `read_word_document` tool for `.docx` files.
    *   Implement `read_pdf_document` tool for `.pdf` files.
    *   Develop unit tests for both tools.

*   **Milestone 2: Structured Extraction for Word (2 weeks)**
    *   Implement `extract_sections_word` tool for `.docx` files.
    *   Enhance `read_word_document` to optionally return metadata (e.g., author, creation date).
    *   Integrate with a basic LLM agent framework (e.g., LangChain) to demonstrate tool usage.

*   **Milestone 3: Structured Extraction for PDF (2 weeks)**
    *   Implement `extract_tables_pdf` tool for `.pdf` files.
    *   Improve PDF text extraction robustness (e.g., handling scanned PDFs via OCR if feasible within scope).
    *   Refine LLM agent integration examples and add comprehensive documentation for all MVP tools.

*   **Milestone 4: Community & Extensibility (1 week)**
    *   Publish to PyPI.
    *   Create clear guidelines for contributing new connectors/tools.
    *   Develop a simple template for adding new file format handlers.
    *   Gather initial community feedback and plan for future iterations (e.g., Excel, databases, editing capabilities).
