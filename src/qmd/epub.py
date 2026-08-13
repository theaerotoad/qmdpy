#!/usr/bin/env python3
"""
Lightweight EPUB to single Markdown file converter.
Extracts HTML/XHTML spine items from an EPUB archive, converts HTML elements
to ATX-style Markdown, and handles images via export, reference, or base64 encoding.
"""

import argparse
import base64
from html.parser import HTMLParser
import os
import posixpath
import re
from pathlib import Path
from typing import Union, Optional
import xml.etree.ElementTree as ET
import zipfile


class EPUBHTMLToMarkdown(HTMLParser):
    """HTML to ATX Markdown converter using standard library HTMLParser."""

    def __init__(self, epub_zip, current_html_path, image_mode, export_dir, output_md_dir):
        super().__init__()
        self.epub_zip = epub_zip
        self.current_html_path = current_html_path
        self.image_mode = image_mode  # 'exportdir', 'refer', or 'base64'
        self.export_dir = export_dir
        self.output_md_dir = output_md_dir

        self.output = []
        self.list_stack = []
        self.link_stack = []
        self.header_stack = []
        self.skip_depth = 0

    def emit(self, text):
        if self.skip_depth > 0:
            return
        if self.link_stack:
            self.link_stack[-1]['buf'].append(text)
        elif self.header_stack:
            self.header_stack[-1]['buf'].append(text)
        else:
            self.output.append(text)

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        tag = tag.lower()

        if tag in ('script', 'style', 'head', 'svg'):
            self.skip_depth += 1
            return
        if self.skip_depth > 0:
            return

        if tag in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
            level = int(tag[1])
            self.header_stack.append({'level': level, 'buf': []})
        elif tag == 'p':
            if not self.header_stack:
                self.emit("\n\n")
            else:
                self.emit(" ")
        elif tag == 'br':
            if self.header_stack or self.link_stack:
                self.emit(" ")
            else:
                self.emit("\n")
        elif tag == 'hr':
            if not self.header_stack:
                self.emit("\n\n---\n\n")
        elif tag in ('strong', 'b'):
            if not self.header_stack:
                self.emit("**")
        elif tag in ('em', 'i'):
            if not self.header_stack:
                self.emit("*")
        elif tag == 'code':
            if not self.header_stack:
                self.emit("`")
        elif tag == 'blockquote':
            if not self.header_stack:
                self.emit("\n\n> ")
        elif tag == 'a':
            href = attrs_dict.get('href', '').strip()
            anchor_id = attrs_dict.get('id', '').strip() or attrs_dict.get('name', '').strip()
            self.link_stack.append({'href': href, 'id': anchor_id, 'buf': []})
        elif tag in ('sup', 'sub'):
            self.emit(f"<{tag}>")
        elif tag in ('ul', 'ol'):
            self.list_stack.append({'type': tag, 'index': 0})
            self.emit("\n")
        elif tag == 'li':
            self.emit("\n")
            indent = "  " * max(0, len(self.list_stack) - 1)
            if self.list_stack:
                curr = self.list_stack[-1]
                if curr['type'] == 'ol':
                    curr['index'] += 1
                    self.emit(f"{indent}{curr['index']}. ")
                else:
                    self.emit(f"{indent}* ")
            else:
                self.emit("* ")
        elif tag == 'img':
            src = attrs_dict.get('src', '')
            alt = attrs_dict.get('alt', '')
            if src:
                img_md = self.process_image(src, alt)
                self.emit(f"\n\n{img_md}\n\n")

    def handle_endtag(self, tag):
        tag = tag.lower()

        if tag in ('script', 'style', 'head', 'svg'):
            if self.skip_depth > 0:
                self.skip_depth -= 1
            return
        if self.skip_depth > 0:
            return

        if tag in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
            if self.header_stack:
                header_info = self.header_stack.pop()
                raw_text = "".join(header_info['buf'])
                clean_text = re.sub(r'[*_`]', '', raw_text)
                clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                if clean_text:
                    level = header_info['level']
                    self.output.append(f"\n\n{'#' * level} {clean_text}\n\n")
        elif tag == 'p':
            if not self.header_stack:
                self.emit("\n\n")
        elif tag in ('strong', 'b'):
            if not self.header_stack:
                self.emit("**")
        elif tag in ('em', 'i'):
            if not self.header_stack:
                self.emit("*")
        elif tag == 'code':
            if not self.header_stack:
                self.emit("`")
        elif tag == 'a':
            if self.link_stack:
                link_info = self.link_stack.pop()
                raw_text = "".join(link_info['buf'])
                clean_text = re.sub(r'[*_`]', '', raw_text)
                link_text = re.sub(r'\s+', ' ', clean_text).strip()
                href = link_info['href']

                if href and link_text:
                    if href.startswith(('http://', 'https://', 'mailto:', 'ftp://')):
                        self.emit(f"[{link_text}]({href})")
                    elif '#' in href:
                        _, anchor = href.split('#', 1)
                        self.emit(f"[{link_text}](#{anchor})")
                    else:
                        self.emit(link_text)
                elif link_text:
                    self.emit(link_text)
        elif tag in ('sup', 'sub'):
            self.emit(f"</{tag}>")
        elif tag in ('ul', 'ol'):
            if self.list_stack:
                self.list_stack.pop()
            self.emit("\n")

    def handle_data(self, data):
        if self.skip_depth > 0:
            return
        text = re.sub(r'[ \t\r\n]+', ' ', data)
        self.emit(text)

    def process_image(self, src, alt):
        html_dir = posixpath.dirname(self.current_html_path)
        img_zip_path = posixpath.normpath(posixpath.join(html_dir, src)) if html_dir else src
        filename = posixpath.basename(img_zip_path)

        if self.image_mode == 'refer':
            return f"![{alt}]({filename})"

        if self.image_mode == 'base64':
            try:
                data = self.epub_zip.read(img_zip_path)
                ext = posixpath.splitext(filename)[1].lower().lstrip('.')
                mime_ext = 'jpeg' if ext == 'jpg' else ext
                mime = f"image/{mime_ext}"
                b64 = base64.b64encode(data).decode('utf-8')
                return f"![{alt}](data:{mime};base64,{b64})"
            except KeyError:
                return f"![{alt}]({filename})"

        if self.image_mode == 'exportdir':
            try:
                data = self.epub_zip.read(img_zip_path)
                out_img_path = Path(self.export_dir) / filename
                out_img_path.parent.mkdir(parents=True, exist_ok=True)
                out_img_path.write_bytes(data)

                if self.output_md_dir:
                    rel_path = os.path.relpath(out_img_path, self.output_md_dir)
                else:
                    rel_path = str(out_img_path)
                return f"![{alt}]({rel_path.replace(os.sep, '/')})"
            except KeyError:
                return f"![{alt}]({filename})"

        return f"![{alt}]({filename})"

    def get_markdown(self):
        raw = "".join(self.output)
        lines = raw.splitlines()
        cleaned_lines = []

        for line in lines:
            stripped = line.rstrip()
            if not stripped:
                cleaned_lines.append('')
            else:
                if re.match(r'^\s+(\*|\d+\.)', stripped):
                    cleaned_lines.append(stripped)
                else:
                    cleaned_lines.append(stripped.lstrip())

        text = "\n".join(cleaned_lines)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()


def normalize_headings(markdown_text: str, epub_title: Optional[str] = None) -> str:
    """
    Ensures the document has a single L1 heading (#) at the top.
    Normalizes body heading levels so that the highest-level body headings
    start at L2 (##), preserving relative hierarchy.
    """
    lines = markdown_text.splitlines()

    first_heading_idx = None
    first_heading_level = None
    first_heading_text = None

    for idx, line in enumerate(lines):
        match = re.match(r'^(#{1,6})\s+(.*)$', line)
        if match:
            first_heading_idx = idx
            first_heading_level = len(match.group(1))
            first_heading_text = match.group(2).strip()
            break

    if first_heading_idx is None:
        if epub_title:
            body = markdown_text.strip()
            return f"# {epub_title}\n\n{body}".strip()
        return markdown_text.strip()

    has_top_l1 = False
    if first_heading_level == 1:
        if epub_title:
            t_norm = epub_title.lower().strip()
            h_norm = first_heading_text.lower().strip()
            if h_norm == t_norm or h_norm in t_norm or t_norm in h_norm:
                has_top_l1 = True
        else:
            if first_heading_idx == 0 or all(not l.strip() for l in lines[:first_heading_idx]):
                has_top_l1 = True

    body_heading_levels = []
    for idx, line in enumerate(lines):
        if has_top_l1 and idx == first_heading_idx:
            continue
        match = re.match(r'^(#{1,6})\s+', line)
        if match:
            body_heading_levels.append(len(match.group(1)))

    has_l1_target = has_top_l1 or bool(epub_title)
    target_min_body_level = 2 if has_l1_target else 1

    shift = 0
    if body_heading_levels:
        min_body_level = min(body_heading_levels)
        shift = target_min_body_level - min_body_level

    new_lines = []
    if not has_top_l1 and epub_title:
        new_lines.append(f"# {epub_title}")
        new_lines.append("")

    for idx, line in enumerate(lines):
        if has_top_l1 and idx == first_heading_idx:
            new_lines.append(line)
        else:
            match = re.match(r'^(#{1,6})\s+(.*)$', line)
            if match:
                hashes, rest = match.groups()
                old_level = len(hashes)
                new_level = max(target_min_body_level, min(6, old_level + shift))
                new_lines.append(f"{'#' * new_level} {rest.strip()}")
            else:
                new_lines.append(line)

    result = "\n".join(new_lines).strip()
    return re.sub(r'\n{3,}', '\n\n', result)


def convert_epub_to_markdown(
    epub_path: Union[str, Path],
    image_mode: str = 'refer',
    export_dir: Optional[Union[str, Path]] = None,
    output_md_dir: Optional[Union[str, Path]] = None
) -> str:
    """
    Converts an EPUB file into Markdown string content.
    """
    epub_path = Path(epub_path)
    output_md_dir = Path(output_md_dir) if output_md_dir else None

    with zipfile.ZipFile(epub_path, 'r') as z:
        # Step 1: Locate OPF package file from META-INF/container.xml
        container_bytes = z.read('META-INF/container.xml')
        container_root = ET.fromstring(container_bytes)
        ns = {'c': 'urn:oasis:names:tc:opendocument:xmlns:container'}
        rootfile_elem = container_root.find('.//c:rootfile', ns)

        if rootfile_elem is None or 'full-path' not in rootfile_elem.attrib:
            raise ValueError("Invalid EPUB: Missing rootfile in META-INF/container.xml")

        opf_path = rootfile_elem.attrib['full-path']
        opf_bytes = z.read(opf_path)
        opf_root = ET.fromstring(opf_bytes)
        opf_dir = posixpath.dirname(opf_path)

        # Extract title from metadata
        epub_title = None
        title_elem = opf_root.find('.//{*}title')
        if title_elem is not None and title_elem.text:
            epub_title = title_elem.text.strip()

        # Step 2: Parse manifest items
        manifest = {}
        for item in opf_root.findall('.//{*}manifest/{*}item'):
            item_id = item.attrib.get('id')
            href = item.attrib.get('href')
            if item_id and href:
                full_href = posixpath.normpath(posixpath.join(opf_dir, href)) if opf_dir else href
                manifest[item_id] = full_href

        # Step 3: Parse spine for chronological reading order
        spine_items = []
        for itemref in opf_root.findall('.//{*}spine/{*}itemref'):
            idref = itemref.attrib.get('idref')
            if idref in manifest:
                spine_items.append(manifest[idref])

        # Step 4: Convert HTML contents in spine sequence
        md_chapters = []
        for html_path in spine_items:
            try:
                content = z.read(html_path).decode('utf-8', errors='ignore')
                parser = EPUBHTMLToMarkdown(
                    epub_zip=z,
                    current_html_path=html_path,
                    image_mode=image_mode,
                    export_dir=export_dir,
                    output_md_dir=output_md_dir
                )
                parser.feed(content)
                chapter_md = parser.get_markdown()
                if chapter_md:
                    md_chapters.append(chapter_md)
            except KeyError:
                continue

        final_markdown = "\n\n---\n\n".join(md_chapters)
        final_markdown = re.sub(r'\n{3,}', '\n\n', final_markdown)
        final_markdown = normalize_headings(final_markdown, epub_title=epub_title)
        final_markdown = re.sub(r'\n{3,}', '\n\n', final_markdown)
        return final_markdown


def main():
    parser = argparse.ArgumentParser(
        description="Lightweight converter from EPUB to a single ATX-styled Markdown file."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to the input .epub file"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Path to the output .md file"
    )

    img_group = parser.add_mutually_exclusive_group()
    img_group.add_argument(
        "--exportdir",
        type=str,
        metavar="DIR",
        help="Export image files to specified directory and link them"
    )
    img_group.add_argument(
        "--refer",
        action="store_true",
        help="Indicate image presence using image filename reference only"
    )
    img_group.add_argument(
        "--base64",
        action="store_true",
        help="Embed images directly as base64 Data URIs in Markdown"
    )

    args = parser.parse_args()

    image_mode = 'refer'
    export_dir = None

    if args.exportdir:
        image_mode = 'exportdir'
        export_dir = args.exportdir
    elif args.base64:
        image_mode = 'base64'
    elif args.refer:
        image_mode = 'refer'

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    md_content = convert_epub_to_markdown(
        epub_path=args.input,
        image_mode=image_mode,
        export_dir=export_dir,
        output_md_dir=output_path.parent
    )
    output_path.write_text(md_content, encoding='utf-8')
    print(f"Successfully converted '{args.input}' -> '{output_path}'")


if __name__ == '__main__':
    main()
