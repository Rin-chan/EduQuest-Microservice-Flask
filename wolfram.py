import sympy as sp
import os

def detect_math_question(question, expected_answer=None, student_answer=None):
    import re
    text = " ".join(filter(None, [question or "", expected_answer or "", student_answer or ""]))
    text = text.lower()
    math_keywords = [
        r"\\lim",
        r"\\int",
        r"\\sum",
        r"\\sqrt",
        r"\\frac",
        "limit",
        "derivative",
        "integral",
        "polynomial",
        "equation",
        "solve",
        "compute",
        "evaluate",
        "function",
        "matrix",
        "vector",
        "algebra",
        "calculus",
        "trigonometric",
        "logarithm",
        "exponential",
    ]
    if any(keyword in text for keyword in math_keywords):
        return True
    if re.search(r"\b\d+\s*[\+\-\*/\^]\s*\d+\b", text):
        return True
    return False

def wolfram_compute(expr: str):
    """
    Uses Wolfram LLM API as a calculator.
    Returns raw text result.
    """

    try:
        import requests
    except Exception as e:
        return f"WOLFRAM ERROR: requests not installed ({e})"

    app_id = os.getenv("WOLFRAM_ALPHA_APP_ID")
    url = (
        "https://www.wolframalpha.com/api/v1/llm-api"
        f"?appid={app_id}&input={expr}"
    )

    r = requests.get(url, timeout=30)
    return r.text.strip()

def normalize_number(x):
    try:
        return float(sp.sympify(x))
    except:
        return None


def extract_numeric(text):
    """
    Try extracting a number from Wolfram output.
    """
    import re
    nums = re.findall(r"-?\d+\.?\d*", text)
    return float(nums[-1]) if nums else None


def extract_expected_number(expected_text):
    """
    Extract a numeric expected value from a freeform expected-answer string.
    Returns a float or None.
    """
    if expected_text is None:
        return None
    if isinstance(expected_text, (int, float)):
        return float(expected_text)
    import re
    nums = re.findall(r"-?\d+\.?\d*", expected_text)
    return float(nums[-1]) if nums else None

def normalize_and_compute(statement: str):
    """
    Improved normalization and computation pipeline.
    Tries multiple strategies: LaTeX cleanup, caret-to-**, simplify/cancel, and symbolic limit evaluation.
    Returns (value or None, normalized_expr or None)
    """
    import re
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
        convert_xor,
    )

    transforms = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )

    if not statement:
        return None, None

    s = statement
    # Convert common LaTeX constructs
    s = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\\(|\\\)", "", s)
    s = s.replace('\\to', 'to').replace('->', 'to')
    s = s.replace('\\', '')
    s = s.replace('^', '**')
    s = s.replace('$', '')

    # Try to detect limit forms
    m = re.search(r"lim_?\{?\s*([a-zA-Z]+)\s*(?:to|->)\s*([^\}\)\s]+)\}?\s*(?:\(?\s*)?(.*)", s, flags=re.IGNORECASE)
    if m:
        var = m.group(1)
        point = m.group(2)
        expr = m.group(3).strip()
        # If the regex consumed a leading '(', we may end up with an unmatched trailing ')'.
        # Rebalance by prepending '(' when needed so sympy can parse correctly.
        if expr.startswith('(') and expr.endswith(')'):
            expr = expr[1:-1].strip()
        if expr.count('(') < expr.count(')'):
            expr = '(' + expr
        try:
            sym_var = sp.symbols(var)
            sym_expr = parse_expr(expr, transformations=transforms)
            # attempt simplification/cancellation to remove removable singularities
            try:
                simp = sp.simplify(sym_expr)
            except Exception as e:
                simp = sym_expr
            try:
                val = sp.limit(simp, sym_var, parse_expr(point, transformations=transforms))
            except Exception as e:
                val = sp.limit(sym_expr, sym_var, parse_expr(point, transformations=transforms))
            try:
                return float(val), expr
            except Exception as e:
                return None, expr
        except Exception as e:
            return None, None

    # Fallback: try to evaluate as equation 'expr = num'
    try:
        m2 = re.match(r"\s*(.*?)\s*=\s*(-?\d+\.?\d*)\s*$", s)
        if m2:
            lhs = m2.group(1)
            sym_lhs = parse_expr(lhs, transformations=transforms)
            try:
                return float(sym_lhs), lhs
            except Exception:
                return None, lhs
    except Exception:
        pass

    # Final fallback: try to parse and evaluate directly
    try:
        sym_expr = parse_expr(s, transformations=transforms)
        try:
            return float(sym_expr), s
        except Exception:
            return None, s
    except Exception:
        return None, None
    
def verify_claim(statement: str, expected=None):
    """
    Convert statement → compute → compare.
    Checks if the claim's stated answer (if any) matches the computation.
    """

    import re
    # Extract the number stated in the claim itself (e.g., "=20" or "=39")
    stmt_eq_match = re.search(r"=\s*(-?\d+\.?\d*)\s*$", statement)
    stmt_provided_num = None
    if stmt_eq_match:
        stmt_provided_num = float(stmt_eq_match.group(1))
    
    # Strip trailing '= number' from statement before normalization (e.g. '\lim...)=39')
    stmt_no_eq = re.sub(r"\s*=\s*-?\d+\.?\d*\s*$", "", statement)

    # --------------------------------------------------
    # Symbolic equality verification
    # --------------------------------------------------
    if "=" in stmt_no_eq:
        try:
            lhs, rhs = stmt_no_eq.split("=", 1)

            lhs = lhs.strip()
            rhs = rhs.strip()

            # remove latex wrappers
            lhs = lhs.replace("\\cdot", "*")
            rhs = rhs.replace("\\cdot", "*")

            # handle limits specially
            if "\\lim" in lhs or "\\lim" in rhs:
                lhs_val, _ = normalize_and_compute(lhs)
                rhs_val, _ = normalize_and_compute(rhs)

                if lhs_val is not None and rhs_val is not None:
                    if abs(lhs_val - rhs_val) < 1e-6:
                        return True, f"LOCAL LIMIT EQUALITY: {lhs_val} == {rhs_val}"

            else:
                from sympy.parsing.sympy_parser import (
                    parse_expr,
                    standard_transformations,
                    implicit_multiplication_application,
                    convert_xor,
                )

                transforms = standard_transformations + (
                    implicit_multiplication_application,
                    convert_xor,
                )

                lhs_expr = parse_expr(lhs, transformations=transforms)
                rhs_expr = parse_expr(rhs, transformations=transforms)

                diff = sp.simplify(lhs_expr - rhs_expr)

                if diff == 0:
                    return True, f"LOCAL EQUALITY VERIFIED: {lhs} == {rhs}"

        except Exception:
            pass

    # First try improved local normalization/evaluation (handles LaTeX limits etc.)
    local_val, normalized_input = normalize_and_compute(stmt_no_eq)
    if local_val is not None:
        # If the claim provides a specific number, verify against that; else use expected_answer
        if stmt_provided_num is not None:
            comparison_val = stmt_provided_num
        else:
            comparison_val = extract_expected_number(expected)
        
        expected_num = comparison_val  # For legacy code below
        # Attempt to call Wolfram as well (best-effort) so logs include input/output.
        wolfram_input = normalized_input or stmt_no_eq or statement
        wolfram_output = None
        try:
            wolfram_output = wolfram_compute(wolfram_input)
        except Exception as e:
            wolfram_output = f"WOLFRAM ERROR: {e}"

        # Build combined debug string including local compute and wolfram IO
        debug_str = f"LOCAL COMPUTE: {local_val} | WOLFRAM INPUT: {wolfram_input} | WOLFRAM OUTPUT: {wolfram_output}"
        if expected_num is None:
            return True, debug_str
        return abs(float(local_val) - float(expected_num)) < 1e-6, debug_str

    # fallback to Wolfram
    wolfram_input = normalized_input or stmt_no_eq or statement
    wolfram_output = wolfram_compute(wolfram_input)

    if not wolfram_output:
        return False, wolfram_output

    low = wolfram_output.lower()

    if (
        'could not understand' in low
        or 'did not understand' in low
        or "couldn't understand" in low
    ):
        return False, wolfram_output

    # NEW: accept algebraic equalities Wolfram verified
    if "\nResult:\nTrue" in wolfram_output:
        return True, wolfram_output

    computed_value = extract_numeric(wolfram_output)

    if computed_value is None:
        return False, wolfram_output

    # normalize expected if it's a freeform string (like the long justification)
    expected_num = extract_expected_number(expected)
    if stmt_provided_num is not None:
        return (
            abs(float(local_val) - float(stmt_provided_num)) < 1e-6,
            debug_str,
        )

    if expected_num is None:
        return True, debug_str

    # only compare to expected answer when claim itself is a final answer
    if "\\lim" in statement and "=" in statement:
        return (
            abs(float(local_val) - float(expected_num)) < 1e-6,
            debug_str,
        )

    return True, debug_str