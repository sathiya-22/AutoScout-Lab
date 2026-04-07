```python
import uuid
from typing import Dict, Any, List, Optional, Union

class Node:
    """Represents a single node in the tree-based context model."""
    def __init__(self, node_id: str, type: str, content: Any, parent_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        if not node_id:
            raise ValueError("Node ID cannot be empty.")
        if not type:
            raise ValueError("Node type cannot be empty.")

        self.id: str = node_id
        self.type: str = type
        self.content: Any = content
        self.parent_id: Optional[str] = parent_id
        self.children_ids: List[str] = []
        self.metadata: Dict[str, Any] = metadata if metadata is not None else {}

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the node to a dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Node':
        """Deserializes a dictionary into a Node object."""
        try:
            node = cls(
                node_id=data["id"],
                type=data["type"],
                content=data["content"],
                parent_id=data.get("parent_id"),
                metadata=data.get("metadata")
            )
            node.children_ids = data.get("children_ids", [])
            return node
        except KeyError as e:
            raise ValueError(f"Missing key in node data for deserialization: {e}") from e

    def __repr__(self):
        return (f"Node(id='{self.id}', type='{self.type}', "
                f"parent='{self.parent_id if self.parent_id else 'None'}', "
                f"children_count={len(self.children_ids)})")


class TreeContextModel:
    """
    Manages a non-linear, tree-based context model for AI agents.
    Represents complex information (e.g., codebase structure as an AST,
    project dependencies, conversation trees) as a graph or tree,
    allowing the agent to traverse and focus on specific, relevant nodes,
    preventing linear context overflow and drift.
    """
    ROOT_NODE_ID = "root"

    def __init__(self, initial_root_content: Any = "Initial context root") -> None:
        """
        Initializes the TreeContextModel.
        A root node is always created to ensure the tree has a consistent top-level anchor.

        Args:
            initial_root_content: The content for the default root node.
        """
        self.nodes: Dict[str, Node] = {}
        if self.ROOT_NODE_ID not in self.nodes:
            self._add_root_node(initial_root_content)

    def _add_root_node(self, content: Any) -> Node:
        """Initializes the root node of the tree. Internal use only."""
        if self.ROOT_NODE_ID in self.nodes:
            # This should ideally not happen if __init__ logic is correct,
            # but acts as a safeguard.
            raise ValueError(f"Root node '{self.ROOT_NODE_ID}' already exists.")
        root_node = Node(node_id=self.ROOT_NODE_ID, type="root", content=content)
        self.nodes[self.ROOT_NODE_ID] = root_node
        return root_node

    def add_node(self, parent_id: str, node_type: str, content: Any, metadata: Optional[Dict[str, Any]] = None) -> Optional[Node]:
        """
        Adds a new node to the tree as a child of the specified parent.

        Args:
            parent_id: The ID of the parent node.
            node_type: The type of the new node (e.g., 'conversation_turn', 'code_block', 'task_step').
            content: The content associated with the node.
            metadata: Optional dictionary for additional metadata.

        Returns:
            The newly created Node object, or None if the parent_id is not found.
        """
        parent_node = self.nodes.get(parent_id)
        if not parent_node:
            print(f"Error: Parent node with ID '{parent_id}' not found. Cannot add node.")
            return None

        new_node_id = str(uuid.uuid4())
        try:
            new_node = Node(node_id=new_node_id, type=node_type, content=content, parent_id=parent_id, metadata=metadata)
            self.nodes[new_node_id] = new_node
            parent_node.children_ids.append(new_node_id)
            return new_node
        except ValueError as e:
            print(f"Error creating new node: {e}")
            return None

    def get_node(self, node_id: str) -> Optional[Node]:
        """
        Retrieves a node by its ID.

        Args:
            node_id: The ID of the node to retrieve.

        Returns:
            The Node object if found, otherwise None.
        """
        return self.nodes.get(node_id)

    def update_node_content(self, node_id: str, new_content: Any) -> bool:
        """
        Updates the content of an existing node.

        Args:
            node_id: The ID of the node to update.
            new_content: The new content for the node.

        Returns:
            True if the node was updated, False otherwise.
        """
        node = self.nodes.get(node_id)
        if node:
            node.content = new_content
            return True
        print(f"Error: Node with ID '{node_id}' not found for update.")
        return False
    
    def update_node_metadata(self, node_id: str, new_metadata: Dict[str, Any]) -> bool:
        """
        Updates the metadata of an existing node. Merges new metadata with existing.

        Args:
            node_id: The ID of the node to update.
            new_metadata: The new metadata (will be merged into existing metadata).

        Returns:
            True if the node was updated, False otherwise.
        """
        node = self.nodes.get(node_id)
        if node:
            node.metadata.update(new_metadata)
            return True
        print(f"Error: Node with ID '{node_id}' not found for metadata update.")
        return False

    def delete_node(self, node_id: str) -> bool:
        """
        Deletes a node and all its descendants from the tree.

        Args:
            node_id: The ID of the node to delete.

        Returns:
            True if the node was deleted, False otherwise.
        """
        if node_id == self.ROOT_NODE_ID:
            print("Error: Cannot delete the root node directly.")
            return False

        node_to_delete = self.nodes.get(node_id)
        if not node_to_delete:
            print(f"Error: Node with ID '{node_id}' not found for deletion.")
            return False

        # Recursively delete children first
        for child_id in list(node_to_delete.children_ids): # Iterate over a copy to allow modification
            self.delete_node(child_id)

        # Remove the node from its parent's children list
        if node_to_delete.parent_id and node_to_delete.parent_id in self.nodes:
            try:
                self.nodes[node_to_delete.parent_id].children_ids.remove(node_id)
            except ValueError:
                # This can happen if consistency is broken, e.g., child_id not in parent's children_ids
                print(f"Warning: Node '{node_id}' not found in parent '{node_to_delete.parent_id}' children list.")
                pass

        # Finally, delete the node itself
        del self.nodes[node_id]
        return True

    def _get_subtree_representation(self, start_node_id: str, current_depth: int, max_depth: int, visited: Optional[set] = None) -> Optional[Dict[str, Any]]:
        """
        Recursively generates a dictionary representation of a subtree.
        This is used for extracting a hierarchical context chunk.
        """
        if visited is None:
            visited = set()

        # Stop if already visited (to prevent infinite loops in case of malformed graph, though parent/child enforces tree)
        # or if max_depth is exceeded.
        if start_node_id in visited or current_depth > max_depth:
            return None

        node = self.nodes.get(start_node_id)
        if not node:
            return None

        visited.add(start_node_id)

        node_representation = {
            "id": node.id,
            "type": node.type,
            "content": node.content,
            "metadata": node.metadata,
            "children": []
        }

        if current_depth < max_depth:
            for child_id in node.children_ids:
                child_repr = self._get_subtree_representation(child_id, current_depth + 1, max_depth, visited)
                if child_repr:
                    node_representation["children"].append(child_repr)

        return node_representation

    def get_context_chunk(self, focus_node_id: str = ROOT_NODE_ID, max_depth: int = 2, max_siblings: int = 2) -> Optional[Dict[str, Any]]:
        """
        Extracts a relevant 'chunk' of context centered around a focus node.
        This function aims to provide a localized, yet hierarchical view
        to prevent context overflow for LLMs.

        Args:
            focus_node_id: The ID of the node to center the context around. Defaults to ROOT_NODE_ID.
            max_depth: How many levels down from the focus node (relative depth from focus)
                       to include descendants in the 'descendant_context'.
            max_siblings: How many sibling nodes (before and after) to include
                          from the focus node's level in 'sibling_nodes'.

        Returns:
            A dictionary representing the focused context, or None if the focus node is not found.
            The structure will include the focus node, its parent (if any), relevant siblings,
            and a subtree of its descendants.
        """
        focus_node = self.get_node(focus_node_id)
        if not focus_node:
            print(f"Error: Focus node with ID '{focus_node_id}' not found.")
            return None

        context_repr: Dict[str, Any] = {}

        # 1. Include the focus node itself
        context_repr["focus_node"] = {
            "id": focus_node.id,
            "type": focus_node.type,
            "content": focus_node.content,
            "metadata": focus_node.metadata,
        }

        # 2. Include parent and relevant siblings
        if focus_node.parent_id and focus_node.parent_id in self.nodes:
            parent_node = self.nodes[focus_node.parent_id]
            
            context_repr["parent_node"] = {
                "id": parent_node.id,
                "type": parent_node.type,
                "content": parent_node.content,
                "metadata": parent_node.metadata,
            }

            # Gather siblings
            parent_children_ids = parent_node.children_ids
            try:
                focus_index = parent_children_ids.index(focus_node_id)
            except ValueError:
                # Should not happen if parent_id and children_ids are consistent
                focus_index = -1 # Indicate not found, so no siblings
                print(f"Warning: Focus node '{focus_node_id}' not found in parent's children list.")

            if focus_index != -1:
                siblings = []
                # Siblings before the focus node
                for i in range(max(0, focus_index - max_siblings), focus_index):
                    sibling = self.nodes.get(parent_children_ids[i])
                    if sibling:
                        siblings.append({"id": sibling.id, "type": sibling.type, "content": sibling.content, "metadata": sibling.metadata})
                # Siblings after the focus node
                for i in range(focus_index + 1, min(len(parent_children_ids), focus_index + 1 + max_siblings)):
                    sibling = self.nodes.get(parent_children_ids[i])
                    if sibling:
                        siblings.append({"id": sibling.id, "type": sibling.type, "content": sibling.content, "metadata": sibling.metadata})
                
                if siblings:
                    context_repr["sibling_nodes"] = siblings

        # 3. Include children/descendants from the focus node
        # Use _get_subtree_representation starting from the focus node itself,
        # but only extract its 'children' part for descendant_context.
        children_subtree_root = self._get_subtree_representation(focus_node_id, current_depth=0, max_depth=max_depth)
        if children_subtree_root and "children" in children_subtree_root:
            context_repr["descendant_context"] = children_subtree_root["children"]
        else:
            context_repr["descendant_context"] = [] # Ensure it's always a list if no children

        return context_repr

    def get_full_tree_representation(self) -> Optional[Dict[str, Any]]:
        """
        Returns a full, nested dictionary representation of the entire tree
        starting from the root.
        """
        return self._get_subtree_representation(self.ROOT_NODE_ID, current_depth=0, max_depth=float('inf')) # type: ignore

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the entire tree to a dictionary of nodes."""
        return {node_id: node.to_dict() for node_id, node in self.nodes.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TreeContextModel':
        """
        Deserializes a dictionary into a TreeContextModel object.
        Ensures a root node is present even if not explicitly in the deserialized data.
        """
        model = cls("Temporary Root") # Create with a temporary root, which will be cleared
        model.nodes.clear() # Clear the temporary root to load actual data

        if not data:
            print("Warning: Deserializing from empty data. A default root node will be created.")
            model._add_root_node("Default root for empty loaded tree")
            return model

        for node_id, node_data in data.items():
            try:
                model.nodes[node_id] = Node.from_dict(node_data)
            except ValueError as e:
                print(f"Error loading node '{node_id}': {e}. Skipping this node.")
        
        # Ensure a root node with the standard ID exists after loading.
        # This handles cases where the serialized data might not have the expected root.
        if cls.ROOT_NODE_ID not in model.nodes:
            print(f"Warning: Deserialized tree does not contain a node with ID '{cls.ROOT_NODE_ID}'. "
                  "A default root node will be created to ensure tree integrity.")
            model._add_root_node("Default root for loaded tree (placeholder)")
            
        return model

    def __len__(self) -> int:
        """Returns the total number of nodes in the tree."""
        return len(self.nodes)

    def __contains__(self, node_id: str) -> bool:
        """Checks if a node with the given ID exists in the tree."""
        return node_id in self.nodes

    def __repr__(self):
        root_status = f"'{self.ROOT_NODE_ID}' exists" if self.ROOT_NODE_ID in self.nodes else "N/A"
        return f"TreeContextModel(nodes={len(self.nodes)}, root_status={root_status})"

```