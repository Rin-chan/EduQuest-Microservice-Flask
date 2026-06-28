import os
import sys
sys.path.append("..")
from llm import LLM

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
llm = LLM(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=float(os.getenv("AZURE_OPENAI_TEMPERATURE")),
)

wolfram_alpha=os.getenv("WOLFRAM_ALPHA_QUERY")

def test1():
    question = r"Evaluate \(\lim_{x\to 5}(2x^2-3x+4)\), and state which limit laws justify your steps."
    expected_answer = r"Expected answer: \(39\). A valid justification uses simple-function limits, power/product law for \(x^2\), constant multiple law, and sum/difference laws: \(\lim_{x\to5}(2x^2-3x+4)=2\cdot 25-3\cdot 5+4=39\). Common misconception: substituting without noting why it is valid; this is acceptable only because polynomials are continuous / by repeated limit laws."
    good_student_answer = r"""Using the Sum Law, Difference Law, Constant Multiple Law, and Power Law:

    lim(x→5)(2x² - 3x + 4)
    = 2·lim(x→5)(x²) - 3·lim(x→5)(x) + lim(x→5)(4)

    By the Power Law:
    = 2·(lim(x→5)x)² - 3·(5) + 4

    Since lim(x→5)x = 5:
    = 2·(5²) - 15 + 4
    = 50 - 15 + 4
    = 39

    Therefore, lim(x→5)(2x² - 3x + 4) = 39.
    """
    med_student_answer = r"Substitute 5 into the expression: 2(25) - 15 + 4 = 39."
    bad_student_answer = r"39"
    wrong_student_answer = r"20 using limit laws because you plug in 5 directly" 
    half_wrong_student_answer = r"""Using the Sum Law:

    lim(x→5)(2x² - 3x + 4)
    = 2(5²) - 3(5) + 4
    = 50 - 15 + 4
    = 39

    Therefore the limit is 39.
    """

    student_answer = good_student_answer

    return question, expected_answer, student_answer

def test2():
    question = r"Show that \(\lim_{x\to 1}\dfrac{x^2-1}{x-1}=2\). Why is canceling \((x-1)\) valid here?"
    expected_answer = r"Expected answer: factor \(x^2-1=(x-1)(x+1)\), so for \(x\neq1\), \(\frac{x^2-1}{x-1}=x+1\). By the replacement law, the two functions have the same limit at \(x=1\) because they agree for all \(x\neq1\) near 1. Hence \(\lim_{x\to1}\frac{x^2-1}{x-1}=\lim_{x\to1}(x+1)=2\). Common misconception: saying the quotient law applies directly even though the denominator limit is 0; it does not."
    good_student_answer = r"""
    Factor the numerator:

    x² - 1 = (x - 1)(x + 1).

    Therefore,

    lim(x→1) (x² - 1)/(x - 1)
    = lim(x→1) (x + 1),

    because (x - 1) cancels for all x ≠ 1.

    Evaluating the remaining expression gives

    = 1 + 1
    = 2.

    Canceling (x - 1) is valid because a limit depends on values approaching x = 1, not the value at x = 1 itself. Since the two expressions are identical whenever x ≠ 1, they have the same limit.
    """
    med_student_answer = r"""
    Factor x² - 1 into (x - 1)(x + 1).

    Cancel (x - 1), giving x + 1.

    Substitute x = 1:

    1 + 1 = 2.

    The cancellation works because x approaches 1 but is not equal to 1.
    """
    bad_student_answer = r"2"
    wrong_student_answer = r"""
    Substitute x = 1 directly:

    (1² - 1)/(1 - 1)
    = 0/0
    = 0.

    Therefore the limit is 0 because the numerator and denominator are both zero.
    """
    half_wrong_student_answer = r"""
    Since x² - 1 = (x - 1)(x + 1),

    (x² - 1)/(x - 1) = x + 1.

    So the limit is

    1 + 1 = 2.

    We can cancel (x - 1) because both the numerator and denominator contain the same factor.
    """
    student_answer = good_student_answer

    return question, expected_answer, student_answer

def test3():
    question = r"Use the squeeze theorem to prove that \(\lim_{x\to 0} x\sin(1/x)=0\)."
    expected_answer = r"Expected answer: since \(|\sin(1/x)|\le1\), we have \(|x\sin(1/x)|\le |x|\). Therefore \(-|x|\le x\sin(1/x)\le |x|\). Also, \(\lim_{x\to0}(-|x|)=0\) and \(\lim_{x\to0}|x|=0\). By the squeeze theorem, \(\lim_{x\to0}x\sin(1/x)=0\). This question connects absolute value limits with the squeeze theorem. Common misconception: trying direct substitution into \(\sin(1/x)\), but \(\sin(1/x)\) has no limit at 0."
    good_student_answer = r"""
    Since −1 ≤ sin(1/x) ≤ 1, we have |sin(1/x)| ≤ 1.
    Multiplying both sides by |x| (which is nonnegative) gives |x sin(1/x)| ≤ |x|.

    Therefore,
    −|x| ≤ x sin(1/x) ≤ |x|.

    Also,
    lim(x→0)(−|x|)=0
    and
    lim(x→0)|x|=0.

    By the Squeeze Theorem,
    lim(x→0)x sin(1/x)=0.
    """
    med_student_answer = r"""
    Since |sin(1/x)| ≤ 1,
    x sin(1/x) is trapped between −|x| and |x|.

    So by the Squeeze Theorem,
    lim x→0 x sin(1/x)=0.
    """
    bad_student_answer = r"The limit is 0 because x goes to 0, so x sin(1/x) also goes to 0."
    wrong_student_answer = r"Since sin(1/x) does not have a limit as x approaches 0, the limit of x sin(1/x) does not exist." 
    half_wrong_student_answer = r"""
    Since sin(1/x) is between −1 and 1,

    −1 ≤ x sin(1/x) ≤ 1.

    By the Squeeze Theorem, the limit is 0.
    """

    student_answer = good_student_answer

    return question, expected_answer, student_answer

# Set test to use
question, expected_answer, student_answer = test3()

# From /generate_short_ans_score
result = llm.generate_mathematical_claims(question, expected_answer, student_answer)

is_math_question = result.get("is_math_question", False)

print("LLM RESULT:", result)
print("is_math_question:", is_math_question)

claims = result.get("claims", [])
methods_used = result.get("methods_used", [])
score, breakdown = llm.generate_score_answer(
    claims=claims,
    question=question,
    expected_answer=expected_answer,
    methods_used=methods_used,
    student_answer=student_answer,
    is_math_question=is_math_question,
)

print("\nFINAL SCORE:", score)
print("\nBREAKDOWN:")
for item in breakdown:
    print(item)