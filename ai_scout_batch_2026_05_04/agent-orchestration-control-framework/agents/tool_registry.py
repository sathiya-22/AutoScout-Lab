class ToolRegistry:
    """
    Manages and provides access to external tools that agents can utilize.
    Promotes discoverability and controlled access to external functionalities.
    """

    def __init__(self):
        """
        Initializes the ToolRegistry with an empty dictionary to store tools.
        """
        self._tools = {}  # Stores tool_name: callable_tool

    def register_tool(self, tool_name: str, tool_callable):
        """
        Registers a tool with a unique name.

        Args:
            tool_name (str): The unique name for the tool.
            tool_callable: The callable object representing the tool's functionality.
                           This can be a function, a method, or an object with a __call__ method.

        Raises:
            TypeError: If the tool_callable is not actually callable.
            ValueError: If a tool with the given name is already registered.
        """
        if not callable(tool_callable):
            raise TypeError(f"Tool '{tool_name}' must be a callable object.")
        if tool_name in self._tools:
            raise ValueError(f"Tool with name '{tool_name}' is already registered.")
        
        self._tools[tool_name] = tool_callable

    def get_tool(self, tool_name: str):
        """
        Retrieves a registered tool by its name.

        Args:
            tool_name (str): The name of the tool to retrieve.

        Returns:
            callable: The callable object associated with the tool_name.

        Raises:
            KeyError: If no tool with the given name is found in the registry.
        """
        if tool_name not in self._tools:
            raise KeyError(f"Tool '{tool_name}' not found in the registry.")
        return self._tools[tool_name]

    def unregister_tool(self, tool_name: str):
        """
        Unregisters a tool by its name.

        Args:
            tool_name (str): The name of the tool to unregister.

        Raises:
            KeyError: If no tool with the given name is found for unregistration.
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
        else:
            raise KeyError(f"Tool '{tool_name}' not found in the registry for unregistration.")

    def list_tools(self) -> list[str]:
        """
        Returns a list of all names of the tools currently registered.

        Returns:
            list[str]: A list containing the names of all registered tools.
        """
        return list(self._tools.keys())

    def clear(self):
        """
        Clears all registered tools from the registry.
        """
        self._tools.clear()

    def __len__(self):
        """
        Returns the number of tools currently registered.
        """
        return len(self._tools)

    def __contains__(self, tool_name: str):
        """
        Checks if a tool with the given name is registered.
        """
        return tool_name in self._tools