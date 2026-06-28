import sympy as sp
import os
import re
import requests

def wolfram_compute(expr: str):
    """
    Uses Wolfram LLM API as a calculator.
    Returns raw text result.
    """

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
    nums = re.findall(r"-?\d+\.?\d*", expected_text)
    return float(nums[-1]) if nums else None

def normalize_and_compute(statement: str):
    """
    Improved normalization and computation pipeline.
    Tries multiple strategies: LaTeX cleanup, caret-to-**, simplify/cancel, and symbolic limit evaluation.
    Returns (value or None, normalized_expr or None)
    """
    if not statement:
        return None, None

    if any(tok in statement for tok in ["<=", ">=", "<", ">", "\\le", "\\ge", "\\leq", "\\geq"]):
        try:
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

            s = statement

            # Normalize LaTeX
            s = s.replace("\\leq", "<=")
            s = s.replace("\\geq", ">=")
            s = s.replace("\\le", "<=")
            s = s.replace("\\ge", ">=")

            s = s.replace("\\cdot", "*")
            s = s.replace("\\times", "*")
            s = s.replace("\\sin", "sin")
            s = s.replace("\\cos", "cos")
            s = s.replace("\\tan", "tan")
            s = s.replace("\\ln", "log")
            s = s.replace("^", "**")

            s = re.sub(r"\|([^|]+)\|", r"Abs(\1)", s)

            s = s.replace("\\text", "")
            s = s.replace("{", "")
            s = s.replace("}", "")
            s = " ".join(s.split())

            operator = None

            if "<=" in s:
                operator = "<="
            elif ">=" in s:
                operator = ">="
            elif "<" in s:
                operator = "<"
            elif ">" in s:
                operator = ">"

            if operator is not None:

                parts = [p.strip() for p in s.split(operator)]

                if len(parts) == 3:

                    left = parse_expr(parts[0], transformations=transforms)
                    middle = parse_expr(parts[1], transformations=transforms)
                    right = parse_expr(parts[2], transformations=transforms)

                    if operator == "<=":
                        ok = (
                            sp.simplify(left <= middle) == True and
                            sp.simplify(middle <= right) == True
                        )

                    elif operator == ">=":
                        ok = (
                            sp.simplify(left >= middle) == True and
                            sp.simplify(middle >= right) == True
                        )

                    elif operator == "<":
                        ok = (
                            sp.simplify(left < middle) == True and
                            sp.simplify(middle < right) == True
                        )

                    else:
                        ok = (
                            sp.simplify(left > middle) == True and
                            sp.simplify(middle > right) == True
                        )

                    if ok:
                        return True, "LOCAL INEQUALITY VERIFIED"

                elif len(parts) == 2:

                    left = parse_expr(parts[0], transformations=transforms)
                    right = parse_expr(parts[1], transformations=transforms)

                    if operator == "<=":
                        ok = sp.simplify(left <= right) == True

                    elif operator == ">=":
                        ok = sp.simplify(left >= right) == True

                    elif operator == "<":
                        ok = sp.simplify(left < right) == True

                    else:
                        ok = sp.simplify(left > right) == True

                    if ok:
                        return True, "LOCAL INEQUALITY VERIFIED"

        except Exception:
            pass

    s = statement
    # Convert common LaTeX constructs
    s = s.replace("\\dfrac", "\\frac")
    s = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\\(|\\\)", "", s)
    s = s.replace('\\to', 'to').replace('->', 'to')
    s = s.replace('\\', '')
    s = s.replace('^', '**')
    s = re.sub(r"\|([^|]+)\|", r"Abs(\1)", s)
    s = s.replace('$', '')

    # Try to detect limit forms
    m = re.search(
        r"(?:lim_?\{?\s*([a-zA-Z]+)\s*(?:to|->|→)\s*([^\}\)\s]+)\}?|"
        r"lim\s*\(?\s*([a-zA-Z]+)\s*(?:to|->|→)\s*([^\)\s]+)\)?)\s*(.*)",
        s,
        flags=re.IGNORECASE,
    )

    if m:
        var = m.group(1) or m.group(3)
        point = m.group(2) or m.group(4)
        expr = m.group(5).strip()
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

    debug_str = ""
    
    import re
    # Extract the number stated in the claim itself (e.g., "=20" or "=39")
    stmt_eq_match = re.search(r"=\s*(-?\d+\.?\d*)\s*$", statement)
    stmt_provided_num = None
    if stmt_eq_match:
        stmt_provided_num = float(stmt_eq_match.group(1))
    
    # Strip trailing '= number' from statement before normalization (e.g. '\lim...)=39')
    stmt_no_eq = re.sub(r"\s*=\s*-?\d+\.?\d*\s*$", "", statement)

    # First try improved local normalization/evaluation (handles LaTeX limits etc.)
    local_val, normalized_input = normalize_and_compute(stmt_no_eq)
    if isinstance(local_val, bool):
        return local_val, normalized_input

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

    # Accept algebraic equalities
    if "\nResult:\nTrue" in wolfram_output:
        return True, wolfram_output

    computed_value = extract_numeric(wolfram_output)

    if computed_value is None:
        return False, wolfram_output

    # If the statement itself supplied a numeric answer (=39, =2, etc.)
    if stmt_provided_num is not None:
        # Prefer local computation if available
        if local_val is not None:
            return (
                abs(local_val - stmt_provided_num) < 1e-6,
                debug_str,
            )

        # Otherwise compare Wolfram's numeric result
        return (
            abs(computed_value - stmt_provided_num) < 1e-6,
            wolfram_output,
        )

    # Compare against expected answer only if one exists
    expected_num = extract_expected_number(expected)

    if expected_num is not None:
        if local_val is not None:
            return (
                abs(local_val - expected_num) < 1e-6,
                debug_str,
            )

        return (
            abs(computed_value - expected_num) < 1e-6,
            wolfram_output,
        )

    # Nothing numeric to compare against, but computation succeeded
    return True, debug_str if debug_str else wolfram_output