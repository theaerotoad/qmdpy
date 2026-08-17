import sys
import shlex
import io
import contextlib
from typing import Optional

try:
    from mcp.server.fastmcp import FastMCP
    FAST_MCP_AVAILABLE = True
except ImportError:
    FAST_MCP_AVAILABLE = False
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        import mcp.types as types
        MCP_AVAILABLE = True
    except ImportError:
        MCP_AVAILABLE = False

def execute_qmd_command(command_str: str, config_path: Optional[str] = None) -> str:
    from qmd.main import build_parser, execute_command
    from qmd.config import load_config
    from qmd.store import Store
    import logging
    
    # Suppress noisy HTTP logs that might pollute the XML output
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    if not command_str:
        return "Error: command cannot be empty."

    try:
        args_list = shlex.split(command_str)
    except ValueError as e:
        return f"Error parsing command string: {e}"

    parser = build_parser()
    f = io.StringIO()
    
    try:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            args = parser.parse_args(args_list)
            
            # The prompt says return the xml-formatted version of the outputs.
            # We dynamically inject --xml if applicable to ensure clean LLM-friendly output.
            if hasattr(args, 'xml') and not getattr(args, 'json', False):
                args.xml = True
                
            config = load_config(getattr(args, "config", config_path))
            store = Store(config)
            execute_command(args, store)
    except SystemExit:
        # Commands like search might exit early or argparse might exit on --help
        pass
    except Exception as e:
        import traceback
        traceback.print_exc(file=f)

    output = f.getvalue()
    if not output:
        output = "Command executed successfully with no output."
    return output

def run_mcp_server(config_path: Optional[str] = None):
    if FAST_MCP_AVAILABLE:
        mcp = FastMCP("qmd")

        @mcp.tool()
        def qmd(command: str) -> str:
            """
            Search and inspect local document knowledge bases using QMD.
            
            Primary Commands:
              - search "query" [--deep] [--session <id>] : Hybrid search. Use --deep for reranked doc grouping.
              - read "<target>" : Inspect chunk(s) or ranges (e.g. 'Books:doc.epub:10-15', 'qmd://Books/doc.epub:3', or '10-15').
              - outline "<target>" : Heading table of contents and chunk sequence map.
              - tree [collection] [-p pattern] : Directory tree of indexed documents.
              - grep "pattern" [-p path] : Exact substring or regex search.
              - collections : List configured collections.
              - guide : Output research workflow and command decision matrix.

            Outputs are automatically formatted in XML with copy-pasteable read="..." attributes.
            """
            return execute_qmd_command(command, config_path)
            
        mcp.run(transport='stdio')
        
    elif MCP_AVAILABLE:
        app = Server("qmd-mcp")

        @app.list_tools()
        async def list_tools() -> list[types.Tool]:
            return [
                types.Tool(
                    name="qmd",
                    description="Search and inspect local document knowledge bases. Commands: search, read, outline, tree, grep, collections, guide. Automatically outputs XML.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Full command string to pass to qmd (excluding the 'qmd' binary name)."
                            }
                        },
                        "required": ["command"]
                    }
                )
            ]

        @app.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
            if name != "qmd":
                raise ValueError(f"Unknown tool: {name}")

            command_str = arguments.get("command", "")
            output = execute_qmd_command(command_str, config_path)
            return [types.TextContent(type="text", text=output)]

        async def main_async():
            async with stdio_server() as (read_stream, write_stream):
                await app.run(read_stream, write_stream, app.create_initialization_options())

        import asyncio
        asyncio.run(main_async())
        
    else:
        print("MCP is not installed. Please install it with: pip install mcp>=1.0.0")
        sys.exit(1)

