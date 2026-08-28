#!/usr/bin/env python3
"""
Lightweight EPUB to single Markdown file converter.
Extracts HTML/XHTML spine items from an EPUB archive, converts HTML elements
to ATX-style Markdown, preserves Table of Contents (NCX / EPUB3 Nav) heading hierarchies,
and handles images via export, reference, or base64 encoding.
"""

import argparse
import base64
from html.parser import HTMLParser
import os
import posixpath
import re
import urllib.parse
from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple, Set
import xml.etree.ElementTree as ET
import zipfile


class TOCEntry:
    """Represents an entry in the EPUB Table of Contents."""

    def __init__(
        self,
        title: str,
        level: int,
        file_path: str,
        anchor: Optional[str] = None,
        order: int = 0
    ):
        self.title = title.strip()
        self.level = max(1, int(level))
        self.file_path = file_path
        self.anchor = anchor.strip() if anchor else None
        self.order = order

    def __repr__(self):
        return f"TOCEntry(title={self.title!r}, level={self.level}, file={self.file_path!r}, anchor={self.anchor!r})"


def _resolve_toc_href(href: str, base_dir: str) -> Tuple[str, Optional[str]]:
    """Resolves relative TOC href into a normalized zip file path and optional anchor ID."""
    unquoted = urllib.parse.unquote(href.strip())
    if '#' in unquoted:
        file_part, anchor_part = unquoted.split('#', 1)
        anchor = anchor_part.strip() or None
    else:
        file_part = unquoted
        anchor = None

    if file_part:
        norm_file = posixpath.normpath(posixpath.join(base_dir, file_part)) if base_dir else posixpath.normpath(file_part)
    else:
        norm_file = ""

    norm_file = norm_file.lstrip('/')
    return norm_file, anchor


def _parse_ncx(ncx_bytes: bytes, ncx_path: str) -> List[TOCEntry]:
    """Parses EPUB 2 NCX XML table of contents."""
    entries: List[TOCEntry] = []
    ncx_dir = posixpath.dirname(ncx_path)

    def _local_tag(elem):
        return elem.tag.split('}')[-1].lower() if isinstance(elem.tag, str) else ""

    order = 0

    def traverse(elem, current_level):
        nonlocal order
        for child in elem:
            if _local_tag(child) == 'navpoint':
                title = ""
                src = ""
                for sub in child:
                    ltag = _local_tag(sub)
                    if ltag == 'navlabel':
                        for t in sub:
                            if _local_tag(t) == 'text' and t.text:
                                title = t.text.strip()
                    elif ltag == 'content':
                        src = sub.attrib.get('src', '').strip()

                if title and src:
                    fpath, anchor = _resolve_toc_href(src, ncx_dir)
                    order += 1
                    entries.append(TOCEntry(title, current_level, fpath, anchor, order=order))

                traverse(child, current_level + 1)

    try:
        root = ET.fromstring(ncx_bytes)
        for child in root:
            if _local_tag(child) == 'navmap':
                traverse(child, 1)
                break
        if not entries:
            for elem in root.iter():
                if _local_tag(elem) == 'navmap':
                    traverse(elem, 1)
                    break
    except Exception:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(ncx_bytes, 'html.parser')
            order = 0

            def traverse_bs4(nav_point, current_level):
                nonlocal order
                label_elem = nav_point.find('navlabel')
                text = ""
                if label_elem and label_elem.find('text'):
                    text = label_elem.find('text').get_text(strip=True)
                content_elem = nav_point.find('content')
                src = content_elem.get('src', '') if content_elem else ''
                if text and src:
                    fpath, anchor = _resolve_toc_href(src, ncx_dir)
                    order += 1
                    entries.append(TOCEntry(text, current_level, fpath, anchor, order=order))
                for child in nav_point.find_all('navpoint', recursive=False):
                    traverse_bs4(child, current_level + 1)

            navmap = soup.find('navmap')
            if navmap:
                for np in navmap.find_all('navpoint', recursive=False):
                    traverse_bs4(np, 1)
        except Exception:
            pass

    return entries


def _parse_nav_doc(nav_bytes: bytes, nav_path: str) -> List[TOCEntry]:
    """Parses EPUB 3 Navigation Document (nav.xhtml)."""
    entries: List[TOCEntry] = []
    nav_dir = posixpath.dirname(nav_path)

    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(nav_bytes, 'html.parser')

        toc_nav = None
        for nav in soup.find_all('nav'):
            etype = (nav.get('epub:type') or nav.get('type') or '').lower()
            role = (nav.get('role') or '').lower()
            nav_id = (nav.get('id') or '').lower()
            if 'toc' in etype or 'doc-toc' in role or 'toc' in nav_id:
                toc_nav = nav
                break
        if toc_nav is None:
            toc_nav = soup.find('nav')
        if toc_nav is None:
            return []

        order = 0

        def parse_list(list_elem, current_level):
            nonlocal order
            for li in list_elem.find_all('li', recursive=False):
                a_tag = li.find('a', recursive=False) or li.find('a')
                if a_tag:
                    title = a_tag.get_text(strip=True)
                    href = a_tag.get('href', '')
                    if title and href:
                        fpath, anchor = _resolve_toc_href(href, nav_dir)
                        order += 1
                        entries.append(TOCEntry(title, current_level, fpath, anchor, order=order))
                nested_list = li.find(['ol', 'ul'], recursive=False) or li.find(['ol', 'ul'])
                if nested_list:
                    parse_list(nested_list, current_level + 1)

        top_list = toc_nav.find(['ol', 'ul'])
        if top_list:
            parse_list(top_list, 1)
    except Exception:
        pass

    return entries


def extract_epub_toc(z: zipfile.ZipFile, opf_root: ET.Element, opf_dir: str) -> List[TOCEntry]:
    """Extracts Table of Contents entries from an EPUB package."""
    manifest = {}
    properties_map = {}
    media_types = {}

    for item in opf_root.findall('.//{*}manifest/{*}item'):
        item_id = item.attrib.get('id')
        href = item.attrib.get('href')
        props = item.attrib.get('properties', '')
        media_type = item.attrib.get('media-type', '')
        if item_id and href:
            full_href = posixpath.normpath(posixpath.join(opf_dir, href)) if opf_dir else href
            full_href = full_href.lstrip('/')
            manifest[item_id] = full_href
            properties_map[item_id] = props.split()
            media_types[item_id] = media_type

    # 1. Try EPUB 3 Navigation Document
    for item_id, props in properties_map.items():
        if 'nav' in props and item_id in manifest:
            try:
                nav_bytes = z.read(manifest[item_id])
                entries = _parse_nav_doc(nav_bytes, manifest[item_id])
                if entries:
                    return entries
            except Exception:
                pass

    # 2. Try EPUB 2 NCX Document
    spine_elem = opf_root.find('.//{*}spine')
    ncx_id = spine_elem.attrib.get('toc') if spine_elem is not None else None
    if ncx_id and ncx_id in manifest:
        try:
            ncx_bytes = z.read(manifest[ncx_id])
            entries = _parse_ncx(ncx_bytes, manifest[ncx_id])
            if entries:
                return entries
        except Exception:
            pass

    for item_id, mtype in media_types.items():
        if mtype == 'application/x-dtbncx+xml' or item_id.lower() in ('ncx', 'toc.ncx') or manifest[item_id].endswith('.ncx'):
            try:
                ncx_bytes = z.read(manifest[item_id])
                entries = _parse_ncx(ncx_bytes, manifest[item_id])
                if entries:
                    return entries
            except Exception:
                pass

    # 3. Try Guide references or items named toc
    for ref in opf_root.findall('.//{*}guide/{*}reference'):
        ref_type = ref.attrib.get('type', '').lower()
        ref_href = ref.attrib.get('href', '')
        if 'toc' in ref_type and ref_href:
            full_href = posixpath.normpath(posixpath.join(opf_dir, ref_href)) if opf_dir else ref_href
            full_href = full_href.split('#')[0].lstrip('/')
            try:
                toc_bytes = z.read(full_href)
                entries = _parse_nav_doc(toc_bytes, full_href)
                if entries:
                    return entries
            except Exception:
                pass

    return []


def _is_quote_block(block: str) -> bool:
    """Checks if a markdown block appears to be a quote or blockquote."""
    s = block.strip()
    if not s:
        return False
    if s.startswith('>'):
        return True
    # Strip leading bullet/list/quote markers and whitespace/ellipsis
    clean = re.sub(r'^[….\s\-–—*]+', '', s)
    quote_chars = ('"', '“', '‘', '«', '”', '’', "'")
    if clean.startswith(quote_chars) or s.endswith(quote_chars):
        return True
    return False


def _is_all_caps_heading_candidate(block: str) -> bool:
    """Checks if a block is a standalone ALL CAPS heading candidate."""
    s = block.strip()
    if not s or '\n' in s:
        return False
    if s.startswith(('#', '-', '*', '>', '|', '`', '!', '<')):
        return False
    # Must contain uppercase letters
    letters = re.findall(r'[a-zA-Z]', s)
    if len(letters) < 3:
        return False
    if not s.isupper():
        return False
    # Should not end with terminating punctuation of a sentence
    if s.endswith(('.', '?', '!', ',', ';', ':')):
        return False
    # Heading shouldn't be too long
    words = s.split()
    if len(words) > 14 or len(s) > 100:
        return False
    return True


def promote_all_caps_headings(markdown_text: str, default_level: int = 3) -> str:
    """
    Promotes standalone ALL CAPS lines to Markdown headings (default level 3),
    unless the immediately preceding block is a quote (where ALL CAPS indicates attribution).
    """
    blocks = re.split(r'\n{2,}', markdown_text.strip())
    if not blocks or blocks == ['']:
        return markdown_text.strip()

    processed_blocks = []
    prev_block = None

    for block in blocks:
        stripped = block.strip()
        if not stripped:
            continue

        if _is_all_caps_heading_candidate(stripped):
            if prev_block and _is_quote_block(prev_block):
                # Quote attribution: keep as body text
                processed_blocks.append(stripped)
            else:
                # Standalone section subheading
                processed_blocks.append(f"{'#' * default_level} {stripped}")
        else:
            processed_blocks.append(stripped)

        prev_block = stripped

    return "\n\n".join(processed_blocks).strip()


def _normalize_title_for_match(t: str) -> str:
    """Normalizes title string for relaxed text matching."""
    t = re.sub(r'<[^>]+>', '', t)
    t = re.sub(r'[*_`~#]', '', t)
    t = re.sub(r'[^\w\s]', '', t.lower())
    return re.sub(r'\s+', ' ', t).strip()


def _titles_match(t1: str, t2: str) -> bool:
    """Checks if two titles match closely enough to represent the same heading."""
    n1 = _normalize_title_for_match(t1)
    n2 = _normalize_title_for_match(t2)
    if not n1 or not n2:
        return False
    if n1 == n2:
        return True

    # Strip common prefixes like 'chapter 1', 'part 1', etc.
    clean_pfx = lambda s: re.sub(r'^(chapter|part|section|book|unit)\s+([0-9]+|[ivxldcm]+)[:.\s-]*', '', s).strip()
    c1 = clean_pfx(n1)
    c2 = clean_pfx(n2)
    if c1 and c2 and c1 == c2:
        return True
    if c1 and c1 == n2:
        return True
    if c2 and c2 == n1:
        return True

    # Substring check with constrained length difference to avoid false positive matches on body text
    w1 = n1.split()
    w2 = n2.split()
    if n1 in n2 or n2 in n1:
        if abs(len(w1) - len(w2)) <= 3 and min(len(w1), len(w2)) >= 2:
            return True

    return False


class EPUBHTMLToMarkdown(HTMLParser):
    """HTML to ATX Markdown converter with Table of Contents awareness."""

    def __init__(
        self,
        epub_zip,
        current_html_path: str,
        image_mode: str = 'refer',
        export_dir: Optional[Union[str, Path]] = None,
        output_md_dir: Optional[Union[str, Path]] = None,
        toc_entries: Optional[List[TOCEntry]] = None
    ):
        super().__init__()
        self.epub_zip = epub_zip
        self.current_html_path = current_html_path
        self.image_mode = image_mode  # 'exportdir', 'refer', or 'base64'
        self.export_dir = export_dir
        self.output_md_dir = output_md_dir
        self.toc_entries = toc_entries or []

        self.anchor_to_toc: Dict[str, TOCEntry] = {}
        self.file_level_toc: Optional[TOCEntry] = None

        for entry in self.toc_entries:
            if entry.anchor:
                self.anchor_to_toc[entry.anchor] = entry
                self.anchor_to_toc[entry.anchor.lower()] = entry
            else:
                if self.file_level_toc is None:
                    self.file_level_toc = entry

        self.used_toc_entries: Set[int] = set()
        self.pending_toc_entry: Optional[TOCEntry] = None
        self.has_emitted_any_heading = False

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

        anchor_id = attrs_dict.get('id', '').strip() or attrs_dict.get('name', '').strip()
        if anchor_id:
            if anchor_id in self.anchor_to_toc:
                entry = self.anchor_to_toc[anchor_id]
                if id(entry) not in self.used_toc_entries:
                    self.pending_toc_entry = entry
            elif anchor_id.lower() in self.anchor_to_toc:
                entry = self.anchor_to_toc[anchor_id.lower()]
                if id(entry) not in self.used_toc_entries:
                    self.pending_toc_entry = entry

        role = attrs_dict.get('role', '').lower()
        aria_level = attrs_dict.get('aria-level', '')
        is_aria_heading = (role == 'heading' and aria_level.isdigit())
        cls = attrs_dict.get('class', '').lower()
        is_heading_class = bool(re.search(r'\b(chapter[-_]?(title|num|number)?|part[-_]?(title|num|number)?|heading[-_]?[1-6]|h[1-6]|section[-_]?title|chaptitle)\b', cls))

        if tag in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6') or is_aria_heading:
            if is_aria_heading:
                raw_level = max(1, min(6, int(aria_level)))
            else:
                raw_level = int(tag[1])

            assigned_level = raw_level
            matched_toc = None

            if self.pending_toc_entry:
                assigned_level = self.pending_toc_entry.level
                matched_toc = self.pending_toc_entry
                self.used_toc_entries.add(id(self.pending_toc_entry))
                self.pending_toc_entry = None
            elif not self.has_emitted_any_heading and self.file_level_toc and id(self.file_level_toc) not in self.used_toc_entries:
                assigned_level = self.file_level_toc.level
                matched_toc = self.file_level_toc
                self.used_toc_entries.add(id(self.file_level_toc))

            self.header_stack.append({'level': assigned_level, 'raw_level': raw_level, 'buf': [], 'toc_entry': matched_toc})
        elif (tag in ('p', 'div', 'section') and (self.pending_toc_entry or (not self.has_emitted_any_heading and self.file_level_toc and is_heading_class))) and not self.header_stack:
            toc_target = self.pending_toc_entry or self.file_level_toc
            if toc_target and id(toc_target) not in self.used_toc_entries:
                self.header_stack.append({
                    'level': toc_target.level,
                    'raw_level': toc_target.level,
                    'buf': [],
                    'toc_entry': toc_target,
                    'is_pseudo': True
                })
                self.used_toc_entries.add(id(toc_target))
                self.pending_toc_entry = None
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
            anchor_id_a = attrs_dict.get('id', '').strip() or attrs_dict.get('name', '').strip()
            self.link_stack.append({'href': href, 'id': anchor_id_a, 'buf': []})
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

        if (tag in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'section')) and self.header_stack:
            header_info = self.header_stack.pop()
            raw_text = "".join(header_info['buf'])
            clean_text = re.sub(r'[*_`]', '', raw_text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()

            level = header_info['level']
            toc_entry = header_info.get('toc_entry')
            is_pseudo = header_info.get('is_pseudo', False)

            if not toc_entry:
                for entry in self.toc_entries:
                    if id(entry) not in self.used_toc_entries and _titles_match(clean_text, entry.title):
                        level = entry.level
                        toc_entry = entry
                        self.used_toc_entries.add(id(entry))
                        break

            if is_pseudo:
                if toc_entry and _titles_match(clean_text, toc_entry.title):
                    title_to_render = clean_text if clean_text else toc_entry.title
                    self.output.append(f"\n\n{'#' * level} {title_to_render}\n\n")
                    self.has_emitted_any_heading = True
                elif clean_text and len(clean_text) < 120 and not clean_text.endswith(('.', '?', '!')):
                    self.output.append(f"\n\n{'#' * level} {clean_text}\n\n")
                    self.has_emitted_any_heading = True
                else:
                    if toc_entry:
                        if _titles_match(clean_text, toc_entry.title) or len(clean_text.split()) < 15:
                            self.output.append(f"\n\n{'#' * level} {toc_entry.title}\n\n")
                        else:
                            self.output.append(f"\n\n{'#' * level} {toc_entry.title}\n\n{raw_text}\n\n")
                        self.has_emitted_any_heading = True
                    else:
                        self.output.append(f"\n\n{raw_text}\n\n")
            else:
                if clean_text:
                    self.output.append(f"\n\n{'#' * level} {clean_text}\n\n")
                    self.has_emitted_any_heading = True
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
        img_zip_path = img_zip_path.lstrip('/')
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

        if not self.has_emitted_any_heading and self.file_level_toc and id(self.file_level_toc) not in self.used_toc_entries:
            stripped_raw = raw.strip()
            if stripped_raw:
                heading_line = f"{'#' * self.file_level_toc.level} {self.file_level_toc.title}"
                blocks = re.split(r'\n{2,}', stripped_raw)
                first_block = blocks[0].strip() if blocks else ""

                if first_block and _titles_match(self.file_level_toc.title, first_block):
                    blocks[0] = heading_line
                    raw = "\n\n".join(blocks)
                elif len(blocks) >= 2 and _titles_match(self.file_level_toc.title, first_block + " " + blocks[1].strip()):
                    blocks = [heading_line] + blocks[2:]
                    raw = "\n\n".join(blocks)
                else:
                    raw = f"\n\n{heading_line}\n\n" + stripped_raw

                self.used_toc_entries.add(id(self.file_level_toc))
                self.has_emitted_any_heading = True

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


def deduplicate_headings_and_titles(markdown_text: str) -> str:
    """
    Removes duplicate consecutive headings and plain-text paragraphs immediately
    following a heading that repeat the heading's title.
    """
    blocks = re.split(r'\n{2,}', markdown_text.strip())
    if not blocks or blocks == ['']:
        return markdown_text.strip()

    cleaned_blocks = []
    last_heading_text = None

    for block in blocks:
        stripped = block.strip()
        if not stripped:
            continue

        heading_match = re.match(r'^(#{1,6})\s+(.*)$', stripped)
        if heading_match:
            h_text = heading_match.group(2).strip()
            if last_heading_text and _titles_match(last_heading_text, h_text):
                # Consecutive duplicate heading, skip it
                continue
            last_heading_text = h_text
            cleaned_blocks.append(stripped)
            continue

        if stripped == '---':
            if cleaned_blocks and cleaned_blocks[-1] == '---':
                continue
            cleaned_blocks.append('---')
            continue

        if last_heading_text:
            lines = stripped.splitlines()
            first_line = lines[0].strip()

            if _titles_match(last_heading_text, stripped) or _titles_match(last_heading_text, first_line):
                if len(lines) == 1 or _titles_match(last_heading_text, stripped):
                    # Entire block repeats the heading title; drop it
                    continue
                else:
                    # First line repeats heading title; drop first line and keep remainder
                    rest = "\n".join(lines[1:]).strip()
                    if rest:
                        cleaned_blocks.append(rest)
                    last_heading_text = None
                    continue

        last_heading_text = None
        cleaned_blocks.append(stripped)

    res = "\n\n".join(cleaned_blocks).strip()
    return re.sub(r'(?:\n\n---\n\n)+', '\n\n---\n\n', res)


def normalize_headings(markdown_text: str, epub_title: Optional[str] = None) -> str:
    """
    Ensures the document has a single L1 heading (#) at the top.
    Normalizes body heading levels so that the highest-level body headings
    start at L2 (##), preserving relative hierarchy and eliminating duplicates.
    """
    markdown_text = deduplicate_headings_and_titles(markdown_text)
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
    result = re.sub(r'\n{3,}', '\n\n', result)
    return deduplicate_headings_and_titles(result)


def convert_epub_to_markdown(
    epub_path: Union[str, Path],
    image_mode: str = 'refer',
    export_dir: Optional[Union[str, Path]] = None,
    output_md_dir: Optional[Union[str, Path]] = None,
    promote_caps_headings: bool = True
) -> str:
    """Converts an EPUB file into Markdown string content using TOC metadata."""
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
                manifest[item_id] = full_href.lstrip('/')

        # Step 3: Extract Table of Contents entries
        toc_entries = extract_epub_toc(z, opf_root, opf_dir)
        toc_by_file: Dict[str, List[TOCEntry]] = {}
        for entry in toc_entries:
            clean_file = entry.file_path.lstrip('/')
            toc_by_file.setdefault(clean_file, []).append(entry)

        # Step 4: Parse spine for chronological reading order
        spine_items = []
        for itemref in opf_root.findall('.//{*}spine/{*}itemref'):
            idref = itemref.attrib.get('idref')
            if idref in manifest:
                spine_items.append(manifest[idref])

        # Step 5: Convert HTML contents in spine sequence
        md_chapters = []
        for html_path in spine_items:
            try:
                content = z.read(html_path).decode('utf-8', errors='ignore')
                file_toc = toc_by_file.get(html_path, [])
                parser = EPUBHTMLToMarkdown(
                    epub_zip=z,
                    current_html_path=html_path,
                    image_mode=image_mode,
                    export_dir=export_dir,
                    output_md_dir=output_md_dir,
                    toc_entries=file_toc
                )
                parser.feed(content)
                chapter_md = parser.get_markdown()
                if chapter_md:
                    md_chapters.append(chapter_md)
            except KeyError:
                continue

        final_markdown = "\n\n---\n\n".join(md_chapters)
        final_markdown = re.sub(r'\n{3,}', '\n\n', final_markdown)
        if promote_caps_headings:
            final_markdown = promote_all_caps_headings(final_markdown, default_level=3)
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