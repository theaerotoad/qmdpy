"""
Mathematical formula extraction and LaTeX conversion for documents (MathML and OMML).
"""
from typing import Optional


def _mathml_to_latex(node) -> str:
    try:
        import mathml_to_latex
        from lxml import etree
        math_xml = etree.tostring(node, encoding='unicode')
        if hasattr(mathml_to_latex, 'convert'):
            return mathml_to_latex.convert(math_xml)
        elif hasattr(mathml_to_latex, 'mathml_to_latex'):
            return mathml_to_latex.mathml_to_latex(math_xml)
        else:
            return "".join(node.itertext())
    except ImportError:
        return "".join(node.itertext())
    except Exception:
        return "".join(node.itertext())


def _omml_to_latex(node) -> str:
    MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
    res = ""
    for child in node:
        if not isinstance(child.tag, str):
            continue
        tag = child.tag.split('}')[-1]
        if tag == 'f':
            num = child.find(f"{{{MATH_NS}}}num")
            den = child.find(f"{{{MATH_NS}}}den")
            num_tex = _omml_to_latex(num) if num is not None else ""
            den_tex = _omml_to_latex(den) if den is not None else ""
            res += f"\\frac{{{num_tex}}}{{{den_tex}}}"
        elif tag == 'rad':
            deg = child.find(f"{{{MATH_NS}}}deg")
            e = child.find(f"{{{MATH_NS}}}e")
            deg_tex = _omml_to_latex(deg) if deg is not None and len(deg) else ""
            e_tex = _omml_to_latex(e) if e is not None else ""
            if deg_tex:
                res += f"\\sqrt[{deg_tex}]{{{e_tex}}}"
            else:
                res += f"\\sqrt{{{e_tex}}}"
        elif tag == 'sSup':
            e = child.find(f"{{{MATH_NS}}}e")
            sup = child.find(f"{{{MATH_NS}}}sup")
            e_tex = _omml_to_latex(e) if e is not None else ""
            sup_tex = _omml_to_latex(sup) if sup is not None else ""
            res += f"{e_tex}^{{{sup_tex}}}"
        elif tag == 'sSub':
            e = child.find(f"{{{MATH_NS}}}e")
            sub = child.find(f"{{{MATH_NS}}}sub")
            e_tex = _omml_to_latex(e) if e is not None else ""
            sub_tex = _omml_to_latex(sub) if sub is not None else ""
            res += f"{e_tex}_{{{sub_tex}}}"
        elif tag == 'sSubSup':
            e = child.find(f"{{{MATH_NS}}}e")
            sub = child.find(f"{{{MATH_NS}}}sub")
            sup = child.find(f"{{{MATH_NS}}}sup")
            e_tex = _omml_to_latex(e) if e is not None else ""
            sub_tex = _omml_to_latex(sub) if sub is not None else ""
            sup_tex = _omml_to_latex(sup) if sup is not None else ""
            res += f"{e_tex}_{{{sub_tex}}}^{{{sup_tex}}}"
        elif tag == 'd':
            dPr = child.find(f"{{{MATH_NS}}}dPr")
            beg_chr = "("
            end_chr = ")"
            sep_chr = "|"
            if dPr is not None:
                beg_node = dPr.find(f"{{{MATH_NS}}}begChr")
                if beg_node is not None: beg_chr = beg_node.get(f"{{{MATH_NS}}}val", "(")
                end_node = dPr.find(f"{{{MATH_NS}}}endChr")
                if end_node is not None: end_chr = end_node.get(f"{{{MATH_NS}}}val", ")")
                sep_node = dPr.find(f"{{{MATH_NS}}}sepChr")
                if sep_node is not None: sep_chr = sep_node.get(f"{{{MATH_NS}}}val", "|")
            
            if beg_chr == "{": beg_chr = "\\{"
            if end_chr == "}": end_chr = "\\}"
            
            e_nodes = child.findall(f"{{{MATH_NS}}}e")
            e_tex = f" {sep_chr} ".join(_omml_to_latex(e) for e in e_nodes)
            res += f"\\left{beg_chr} {e_tex} \\right{end_chr}"
        elif tag == 'nary':
            naryPr = child.find(f"{{{MATH_NS}}}naryPr")
            op_tex = "\\int"
            if naryPr is not None:
                chr_node = naryPr.find(f"{{{MATH_NS}}}chr")
                if chr_node is not None:
                    val = chr_node.get(f"{{{MATH_NS}}}val")
                    if val == "∑": op_tex = "\\sum"
                    elif val == "∏": op_tex = "\\prod"
                    elif val == "∐": op_tex = "\\coprod"
                    elif val == "∪": op_tex = "\\bigcup"
                    elif val == "∩": op_tex = "\\bigcap"

            sub = child.find(f"{{{MATH_NS}}}sub")
            sup = child.find(f"{{{MATH_NS}}}sup")
            e = child.find(f"{{{MATH_NS}}}e")
            sub_tex = _omml_to_latex(sub) if sub is not None else ""
            sup_tex = _omml_to_latex(sup) if sup is not None else ""
            e_tex = _omml_to_latex(e) if e is not None else ""
            
            res += f"{op_tex}"
            if sub_tex: res += f"_{{{sub_tex}}}"
            if sup_tex: res += f"^{{{sup_tex}}}"
            res += f" {e_tex}"
        elif tag == 'func':
            fName = child.find(f"{{{MATH_NS}}}fName")
            e = child.find(f"{{{MATH_NS}}}e")
            fName_tex = _omml_to_latex(fName) if fName is not None else ""
            e_tex = _omml_to_latex(e) if e is not None else ""
            res += f"{fName_tex}{e_tex}"
        elif tag == 'm':
            mr_nodes = child.findall(f"{{{MATH_NS}}}mr")
            rows = []
            for mr in mr_nodes:
                e_nodes = mr.findall(f"{{{MATH_NS}}}e")
                rows.append(" & ".join(_omml_to_latex(e) for e in e_nodes))
            matrix_content = " \\\\ ".join(rows)
            res += f"\\begin{{matrix}} {matrix_content} \\end{{matrix}}"
        elif tag == 'eqArr':
            e_nodes = child.findall(f"{{{MATH_NS}}}e")
            lines = [ _omml_to_latex(e) for e in e_nodes ]
            res += " \\\\ ".join(lines)
        elif tag == 'acc':
            accPr = child.find(f"{{{MATH_NS}}}accPr")
            chr_val = "^"
            if accPr is not None:
                chr_node = accPr.find(f"{{{MATH_NS}}}chr")
                if chr_node is not None:
                    chr_val = chr_node.get(f"{{{MATH_NS}}}val", "^")
            e = child.find(f"{{{MATH_NS}}}e")
            e_tex = _omml_to_latex(e) if e is not None else ""
            if chr_val in ("⃗", "\u20d7"): res += f"\\vec{{{e_tex}}}"
            elif chr_val in ("^", "̂", "\u0302"): res += f"\\hat{{{e_tex}}}"
            elif chr_val in ("‾", "¯", "\u0304"): res += f"\\bar{{{e_tex}}}"
            elif chr_val in ("˜", "̃", "\u0303"): res += f"\\tilde{{{e_tex}}}"
            elif chr_val in ("˙", "̇", "\u0307"): res += f"\\dot{{{e_tex}}}"
            elif chr_val in ("¨", "̈", "\u0308"): res += f"\\ddot{{{e_tex}}}"
            else: res += f"{e_tex}"
        elif tag == 'limLow':
            e = child.find(f"{{{MATH_NS}}}e")
            lim = child.find(f"{{{MATH_NS}}}lim")
            e_tex = _omml_to_latex(e) if e is not None else ""
            lim_tex = _omml_to_latex(lim) if lim is not None else ""
            res += f"\\mathop{{{e_tex}}}_{{{lim_tex}}}"
        elif tag == 't':
            res += child.text or ""
        else:
            res += _omml_to_latex(child)
    return res


def _extract_text_and_math(node) -> str:
    """Recursively extracts text and math formulas from an lxml node (e.g. DOCX CT_P)."""
    MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
    MML_NS = "http://www.w3.org/1998/Math/MathML"
    WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    DRAWING_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
    
    if not hasattr(node, "tag"):
        return ""

    tag = node.tag
    if isinstance(tag, str):
        if tag == f"{{{MATH_NS}}}oMathPara":
            tex = _omml_to_latex(node)
            return f"$$ {tex} $$"
        elif tag == f"{{{MATH_NS}}}oMath":
            tex = _omml_to_latex(node)
            return f"${tex}$"
        elif tag == f"{{{MML_NS}}}math":
            tex = _mathml_to_latex(node)
            return f"${tex}$"
        
        # Explicit text nodes: Word, Word-Delete, Math, PPTX
        elif tag in (f"{{{WORD_NS}}}t", f"{{{WORD_NS}}}delText", f"{{{MATH_NS}}}t", f"{{{DRAWING_NS}}}t"):
            return node.text or ""
        # Handle whitespace explicitly
        elif tag == f"{{{WORD_NS}}}tab":
            return "\t"
        elif tag == f"{{{WORD_NS}}}br":
            return "\n"
    
    res = []
    for child in node:
        res.append(_extract_text_and_math(child))
        
    return "".join(res)