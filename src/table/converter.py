<<<<<<< SEARCH
def _convert_single_table(table_tag) -> str:
    """Converts a BeautifulSoup table tag into a Markdown table string."""
    rows = table_tag.find_all("tr")
    if not rows:
        return ""

    grid = []
    max_cols = 0

    for row in rows:
        cells = row.find_all(["th", "td"])
        row_data = []
        for cell in cells:
            text = cell.get_text(strip=True).replace("\n", " ")
            row_data.append(text)
        if row_data:
            grid.append(row_data)
            max_cols = max(max_cols, len(row_data))

    if not grid:
        return ""

    # Pad rows to ensure rectangular grid
    for row in grid:
        while len(row) < max_cols:
            row.append("")

    # Build Markdown table
    md_lines = []
    # Header
    header = grid[0]
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("| " + " | ".join(["---"] * max_cols) + " |")

    # Data rows
    for row in grid[1:]:
        md_lines.append("| " + " | ".join(row) + " |")

    return "\n".join(md_lines)
=======
def _convert_single_table(table_tag) -> str:
    """Converts a BeautifulSoup table tag into a Markdown table string with delimiter escaping and span support."""
    rows = table_tag.find_all("tr")
    if not rows:
        return ""

    grid = []
    max_cols = 0

    for row in rows:
        cells = row.find_all(["th", "td"])
        row_data = []
        for cell in cells:
            # Escape literal pipe characters to prevent breaking Markdown column alignment
            text = cell.get_text(separator=" ", strip=True).replace("\n", " ")
            text = text.replace("|", "\\|")

            # Handle colspans predicted by SLANet by padding trailing empty cells
            colspan = 1
            if cell.has_attr("colspan"):
                try:
                    colspan = max(1, int(cell["colspan"]))
                except ValueError:
                    colspan = 1

            row_data.append(text)
            for _ in range(colspan - 1):
                row_data.append("")

        if row_data:
            grid.append(row_data)
            max_cols = max(max_cols, len(row_data))

    if not grid or max_cols == 0:
        return ""

    # Pad rows to ensure rectangular grid
    for row in grid:
        while len(row) < max_cols:
            row.append("")

    # Build Markdown table
    md_lines = []
    header = grid[0]
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("| " + " | ".join(["---"] * max_cols) + " |")

    for row in grid[1:]:
        md_lines.append("| " + " | ".join(row) + " |")

    return "\n".join(md_lines)
>>>>>>> REPLACE
