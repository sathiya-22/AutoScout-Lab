# Major problem of week 2026-W40

**MCP server filesystem path resolution errors**  (id: `mcp-server-filesystem-path-resolution-errors`, signal: 85)

Developers face issues with MCP filesystem servers, including 'Not connected' errors and 'workspaceFolder can not be resolved' errors, particularly on Windows and MacOS. This problem impacts agent ops and deployment, as it prevents agents from correctly accessing and managing files, hindering their ability to perform tasks that require file system interaction.

## Why this one

The 'MCP server filesystem path resolution errors' problem is the most significant due to its high signal, broad impact across agent ops and deployment, and fundamental nature. File system access is a core capability for almost any agentic system, and failures here render agents effectively useless for many practical tasks. The problem affects developers on common operating systems like Windows and MacOS, indicating a widespread issue without a robust existing solution, making it an ideal candidate for a focused open-source project.

## Sources

- https://github.com/modelcontextprotocol/servers/issues/3051
- https://github.com/modelcontextprotocol/servers/issues/1613
- https://github.com/modelcontextprotocol/servers/issues/644
- https://github.com/modelcontextprotocol/servers/issues/470

Daily prototype: https://github.com/sathiya-22/mcp-server-filesystem-path-resolution-errors-2026-09-29

---

## Problem
Developers building agentic systems using Model Context Protocol (MCP) filesystem servers frequently encounter critical errors such as 'Not connected' and 'workspaceFolder can not be resolved'. These issues are particularly prevalent on Windows and MacOS environments. This directly prevents agents from correctly accessing, reading, and managing files, severely hindering their ability to perform tasks that require any form of file system interaction. This impacts agent ops, deployment, and the overall reliability of agentic applications.

## Evidence
- **High Signal:** The problem has a 'signal' score of 85, indicating a strong community awareness and impact.
- **Multiple Sources:** Evidenced by several GitHub issues on the `modelcontextprotocol/servers` repository: [#3051](https://github.com/modelcontextprotocol/servers/issues/3051), [#1613](https://github.com/modelcontextprotocol/servers/issues/1613), [#644](https://github.com/modelcontextprotocol/servers/issues/644), [#470](https://github.com/modelcontextprotocol/servers/issues/470).
- **Operating System Specificity:** The issues are explicitly reported on Windows and MacOS, highlighting platform-specific challenges in path resolution and connection management.
- **Impact on Core Functionality:** The inability to resolve `workspaceFolder` or maintain a connection directly prevents fundamental file operations, which are essential for most agentic tasks.

## Proposed solution
Develop a robust, cross-platform MCP filesystem server client/wrapper that specifically addresses and mitigates path resolution and connection stability issues. This solution will provide a more resilient layer for agentic systems to interact with the local filesystem, abstracting away OS-specific quirks and improving error handling. It will focus on canonicalizing paths, ensuring consistent connection states, and providing clear diagnostics.

## MVP scope
1.  **Cross-platform Path Normalization Utility:** Implement a utility function or module that takes a given path and normalizes it to a canonical, OS-agnostic format, suitable for MCP server consumption. This should handle Windows drive letters, forward/backward slashes, and relative paths consistently.
2.  **Connection Health Check and Reconnection Logic:** Develop a client-side mechanism to periodically check the MCP filesystem server connection status. If the connection is lost or not established, implement an automatic (configurable) reconnection strategy with exponential backoff.
3.  **Enhanced Error Reporting:** When path resolution or connection errors occur, provide more detailed, actionable error messages that include the original path, the normalized path (if applicable), the OS, and potential causes/solutions.
4.  **Basic File Operations Wrapper:** Create a simple wrapper around core MCP filesystem operations (e.g., `read_file`, `list_directory`) that utilizes the path normalization and connection logic to demonstrate resilience.
5.  **Test Suite for Windows/MacOS:** Develop a comprehensive test suite that specifically targets path resolution and connection stability on both Windows and MacOS environments, using mocked MCP server responses where necessary.

## Milestones
### Milestone 1: Path Normalization & Basic Wrapper (2 weeks)
-   Design and implement `PathNormalizer` module supporting Windows and Unix-like paths.
-   Integrate `PathNormalizer` into a basic `MCPFilesystemClient` wrapper for `read_file` and `list_directory`.
-   Develop unit tests for path normalization on various OS path formats.
-   Initial documentation for `PathNormalizer`.

### Milestone 2: Connection Resilience & Enhanced Errors (3 weeks)
-   Implement connection health check and automatic reconnection logic within `MCPFilesystemClient`.
-   Enhance error handling to provide detailed diagnostics for connection and path resolution failures.
-   Develop integration tests to simulate connection drops and verify reconnection.
-   Update documentation with connection management details and error handling best practices.

### Milestone 3: Cross-Platform Validation & Release Candidate (2 weeks)
-   Set up CI/CD to run tests on Windows and MacOS environments.
-   Address any platform-specific bugs identified during cross-platform testing.
-   Refine API and documentation based on testing and user feedback.
-   Prepare release candidate, including examples for common agentic use cases.

### Milestone 4: Public Release & Community Engagement (1 week)
-   Public release of the open-source library.
-   Announce on relevant community forums (e.g., MCP, agentic AI communities).
-   Monitor issues and gather feedback for future improvements.
