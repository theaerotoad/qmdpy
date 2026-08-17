import sys
import shlex
import io
import re
import contextlib
from typing import Optional, Any

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

def execute_qmd_command(command_str: str, config_path: Optional[str] = None, store: Optional[Any] = None) -> str:
    from qmd.main import build_parser, execute_command
    from qmd.config import load_config
    from qmd.store import Store
    import logging
    
    # Suppress noisy HTTP logs that might pollute the XML output
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    if not command_str or not command_str.strip():
        return "Error: command cannot be empty."

    raw_cmd = command_str.strip()

    # Strip XML attribute wrappers e.g. read="qmd read '...'" or outline="..."
    attr_match = re.match(r'^(?:read|outline|expand|resume)\s*=\s*["\'](.*)["\']$', raw_cmd, re.IGNORECASE)
    if attr_match:
        raw_cmd = attr_match.group(1).strip()

    # Strip tool_call tags and tool prefixes e.g. <|tool_call>call:...<tool_call|>
    raw_cmd = re.sub(r'^<\|?(?:tool_call|tool_calls)[^>]*>(?:call:)?', '', raw_cmd, flags=re.IGNORECASE).strip()
    raw_cmd = re.sub(r'<\|?/(?:tool_call|tool_calls)?[^>]*>$', '', raw_cmd, flags=re.IGNORECASE).strip()
    raw_cmd = re.sub(r'^call:\s*', '', raw_cmd, flags=re.IGNORECASE).strip()

    # Strip surrounding backticks e.g. `qmd search "..."`
    if raw_cmd.startswith('`') and raw_cmd.endswith('`') and len(raw_cmd) >= 2:
        raw_cmd = raw_cmd[1:-1].strip()

    # Strip XML tag wrappers e.g. <command>...</command>
    tag_match = re.match(r'^<[^>]+>(.*)</[^>]+>$', raw_cmd, re.DOTALL)
    if tag_match:
        raw_cmd = tag_match.group(1).strip()

    try:
        args_list = shlex.split(raw_cmd)
    except ValueError as e:
        return f"Error parsing command string: {e}"

    if not args_list:
        return "Error: command cannot be empty."

    # Strip leading 'qmd' or python invocation
    if args_list[0] in ("qmd", "./qmd", "qmd.exe"):
        args_list = args_list[1:]
    elif len(args_list) >= 3 and args_list[0] in ("python", "python3") and args_list[1] == "-m" and args_list[2] == "qmd.main":
        args_list = args_list[3:]

    if not args_list:
        return "Error: no subcommand provided."

    parser = build_parser()
    f = io.StringIO()
    
    try:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            args = parser.parse_args(args_list)
            
            # Dynamically inject --xml if applicable to ensure clean LLM-friendly output
            if hasattr(args, 'xml') and not getattr(args, 'json', False):
                args.xml = True
                
            active_store = store
            if active_store is None:
                config = load_config(getattr(args, "config", config_path))
                active_store = Store(config)
            execute_command(args, active_store)
    except SystemExit:
        # Commands might exit on help or termination
        pass
    except Exception as e:
        import traceback
        traceback.print_exc(file=f)

    output = f.getvalue().strip()
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

