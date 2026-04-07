import logging
from typing import Dict, Any, List, Optional

# Assuming these modules exist in the same 'context' directory as siblings
from .semantic_context_retriever import SemanticContextRetriever
from .changelog_processor import ChangelogProcessor
from .tree_context_model import TreeContextModel

# Set up logging for the module
logger = logging.getLogger(__name__)
# Basic config if not already set up by the main application
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class ContextAggregator:
    """
    Synthesizes a coherent and relevant working context for the LLM at any given moment.
    It integrates semantic context, changelog-based updates, and tree-based context models.
    """

    def __init__(
        self,
        semantic_retriever: Optional[SemanticContextRetriever] = None,
        changelog_processor: Optional[ChangelogProcessor] = None,
        tree_model: Optional[TreeContextModel] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes the ContextAggregator with its sub-components.

        Args:
            semantic_retriever: An instance of SemanticContextRetriever. If None, a default
                                 instance will be attempted.
            changelog_processor: An instance of ChangelogProcessor. If None, a default
                                  instance will be attempted.
            tree_model: An instance of TreeContextModel. If None, a default
                        instance will be attempted.
            config: Optional configuration dictionary for context aggregation,
                    e.g., retrieval counts, scores, depths.
        """
        self.config = config if config is not None else {}

        # Initialize sub-components. If not provided, create default instances.
        # Handle potential import/instantiation errors gracefully.
        self.semantic_retriever = None
        if semantic_retriever:
            self.semantic_retriever = semantic_retriever
        else:
            try:
                self.semantic_retriever = SemanticContextRetriever()
                logger.debug("Default SemanticContextRetriever initialized.")
            except ImportError:
                logger.error("SemanticContextRetriever module not found or failed to import.")
            except Exception as e:
                logger.error(f"Failed to initialize SemanticContextRetriever: {e}", exc_info=True)

        self.changelog_processor = None
        if changelog_processor:
            self.changelog_processor = changelog_processor
        else:
            try:
                self.changelog_processor = ChangelogProcessor()
                logger.debug("Default ChangelogProcessor initialized.")
            except ImportError:
                logger.error("ChangelogProcessor module not found or failed to import.")
            except Exception as e:
                logger.error(f"Failed to initialize ChangelogProcessor: {e}", exc_info=True)

        self.tree_model = None
        if tree_model:
            self.tree_model = tree_model
        else:
            try:
                self.tree_model = TreeContextModel()
                logger.debug("Default TreeContextModel initialized.")
            except ImportError:
                logger.error("TreeContextModel module not found or failed to import.")
            except Exception as e:
                logger.error(f"Failed to initialize TreeContextModel: {e}", exc_info=True)

        if not any([self.semantic_retriever, self.changelog_processor, self.tree_model]):
            logger.warning("ContextAggregator initialized with no active context sources.")
        else:
            logger.info("ContextAggregator initialized with active sources.")

    def _format_context_section(self, title: str, content: Any) -> str:
        """Helper to format a context section for readability."""
        if not content:
            return ""
        # Ensure content is string for formatting, strip leading/trailing whitespace
        content_str = str(content).strip()
        if not content_str:
            return ""
        return f"\n--- {title.upper()} ---\n{content_str}\n"

    def get_aggregated_context(
        self,
        current_query: str,
        recent_interactions: Optional[List[Dict[str, Any]]] = None,
        agent_state: Optional[Dict[str, Any]] = None,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """
        Synthesizes a comprehensive context string for the LLM based on various sources.

        Args:
            current_query: The immediate query or task from the user/agent. This is crucial
                           for relevance filtering across sub-components.
            recent_interactions: A list of recent conversation turns or actions.
                                 Each dict might contain 'role' and 'content'.
            agent_state: The current persistent state of the agent (e.g., task goals,
                         user preferences).
            max_tokens: The maximum desired token length for the aggregated context.
                        This is a target; actual length might vary slightly before
                        fine-grained optimization by ContextOptimizer.
            kwargs: Additional parameters to pass to sub-component context retrieval methods.
                    Specific kwargs can be nested, e.g., 'semantic_retriever_kwargs'.

        Returns:
            A string representing the aggregated context ready for the LLM.
            Returns an empty string if no context can be aggregated.
        """
        context_parts = []
        
        # 1. Add current query and recent interactions as immediate context
        if current_query:
            context_parts.append(self._format_context_section("CURRENT TASK/QUERY", current_query))
        
        if recent_interactions:
            interaction_summary_lines = []
            for i in recent_interactions:
                role = i.get('role', 'unknown')
                content = i.get('content', '')
                if content:
                    interaction_summary_lines.append(f"[{role.upper()}] {content}")
            if interaction_summary_lines:
                context_parts.append(self._format_context_section("RECENT INTERACTIONS", "\n".join(interaction_summary_lines)))

        # 2. Retrieve Semantic Context
        if self.semantic_retriever:
            try:
                semantic_context_items = self.semantic_retriever.retrieve_context(
                    query=current_query,
                    num_items=self.config.get("semantic_retrieval_count", 5),
                    min_score=self.config.get("semantic_min_score", 0.7),
                    **kwargs.get("semantic_retriever_kwargs", {})
                )
                if semantic_context_items:
                    # Assuming items are dicts with 'content' and potentially 'source'
                    formatted_items = [
                        f"Source: {item.get('source', 'N/A')}\nContent: {item['content']}"
                        for item in semantic_context_items if 'content' in item
                    ]
                    semantic_context = "\n---\n".join(formatted_items)
                    context_parts.append(self._format_context_section("SEMANTIC CONTEXT (Retrieved Memory)", semantic_context))
                logger.debug(f"Retrieved {len(semantic_context_items)} semantic context items.")
            except Exception as e:
                logger.warning(f"Error retrieving semantic context: {e}", exc_info=True)
        else:
            logger.debug("SemanticContextRetriever not active, skipping semantic context retrieval.")

        # 3. Process Changelog-Based Context
        if self.changelog_processor:
            try:
                # Assuming process_changelogs can take a timeframe or specific filters
                changelog_data = self.changelog_processor.process_changelogs(
                    current_query=current_query, # Pass query for relevance
                    num_entries=self.config.get("changelog_entry_count", 3),
                    lookback_days=self.config.get("changelog_lookback_days", 7),
                    **kwargs.get("changelog_processor_kwargs", {})
                )
                if changelog_data:
                    # Assuming changelog_data is a list of formatted strings or objects to be formatted
                    formatted_changelog = "\n---\n".join([str(entry) for entry in changelog_data])
                    context_parts.append(self._format_context_section("CHANGELOG UPDATES (Environment Changes)", formatted_changelog))
                logger.debug(f"Processed {len(changelog_data)} changelog entries.")
            except Exception as e:
                logger.warning(f"Error processing changelogs: {e}", exc_info=True)
        else:
            logger.debug("ChangelogProcessor not active, skipping changelog context.")

        # 4. Get Non-Linear/Tree-Based Context
        if self.tree_model:
            try:
                # Assuming tree_model can focus on relevant nodes based on query/state
                tree_context_data = self.tree_model.get_relevant_context(
                    focus_query=current_query,
                    agent_state=agent_state,
                    depth=self.config.get("tree_context_depth", 2),
                    **kwargs.get("tree_model_kwargs", {})
                )
                if tree_context_data:
                    # Assuming tree_context_data is already a string or easily stringifiable
                    context_parts.append(self._format_context_section("STRUCTURED CONTEXT (Tree Model)", tree_context_data))
                logger.debug("Retrieved tree-based context.")
            except Exception as e:
                logger.warning(f"Error retrieving tree-based context: {e}", exc_info=True)
        else:
            logger.debug("TreeContextModel not active, skipping tree-based context.")

        # 5. Add Agent State (if available and relevant)
        if agent_state:
            # Filter agent_state to only include highly relevant parts for context,
            # or summarize it if too large. Avoid sending very large, raw structures
            # like full memory dumps or deeply nested objects unless specifically needed.
            relevant_state = {}
            for k, v in agent_state.items():
                if k not in ["long_term_memory_raw_dump", "previous_conversations_full", "raw_chat_history"]:
                    # Simple heuristic: if it's a collection, check size; otherwise, include.
                    if isinstance(v, (list, dict, set)):
                        if len(str(v)) < 500: # Arbitrary size limit for direct inclusion
                            relevant_state[k] = v
                        else:
                            # Too large, maybe include a summary or just a placeholder
                            relevant_state[k] = f"<{type(v).__name__} (too large to display)>"
                    else:
                        relevant_state[k] = v

            state_summary_lines = [f"- {k}: {v}" for k, v in relevant_state.items()]
            if state_summary_lines:
                context_parts.append(self._format_context_section("AGENT PERSISTENT STATE", "\n".join(state_summary_lines)))
            else:
                logger.debug("Agent state provided but no relevant summary generated or was empty.")

        # Combine all parts
        full_context = "\n\n".join(context_parts).strip()

        # Placeholder for ContextOptimizer integration:
        # In a full system, a ContextOptimizer would tokenize the `full_context`
        # and intelligently trim/summarize it to fit `max_tokens` precisely.
        # For this prototype, we'll just log a warning if it's estimated to be too long.
        estimated_tokens = len(full_context) // 4 # Rough estimate for English text
        if estimated_tokens > max_tokens:
             logger.warning(
                 f"Aggregated context ({estimated_tokens} tokens, raw length: {len(full_context)} chars) "
                 f"exceeds target max_tokens ({max_tokens}). "
                 f"ContextOptimizer would normally handle truncation/summarization here."
             )
        
        logger.info(f"Aggregated context generated (estimated tokens: {estimated_tokens}, raw length: {len(full_context)} chars).")
        return full_context