import os
import sys
sys.path.append("..")
from llm import LLM
from wolfram import detect_math_question

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
llm = LLM(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=float(os.getenv("AZURE_OPENAI_TEMPERATURE")),
)

wolfram_alpha=os.getenv("WOLFRAM_ALPHA_QUERY")

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

Therefore, lim(x→5)(2x² - 3x + 4) = 39."""
med_student_answer = r"Substitute 5 into the expression: 2(25) - 15 + 4 = 39."
bad_student_answer = r"39"
wrong_student_answer = r"20 using limit laws because you plug in 5 directly" 
half_wrong_student_answer = r"""Using the Sum Law:

lim(x→5)(2x² - 3x + 4)
= 2(5²) - 3(5) + 4
= 50 - 15 + 4
= 39

Therefore the limit is 39."""

student_answer = wrong_student_answer

# From /generate_short_ans_score
result = llm.generate_mathematical_claims(question, expected_answer, student_answer)

combined_text = " ".join([question or "", expected_answer or "", student_answer or ""]).lower()
programming_keywords = [
    "program",
    "pseudocode",
    "if statement",
    "if condition",
    "boolean",
    "true",
    "false",
    "print",
    "return",
    "display",
]
is_programming_question = any(keyword in combined_text for keyword in programming_keywords)
is_math_question = (
    not is_programming_question
    and (
        result.get("is_math_question", False) or detect_math_question(
    question, expected_answer=expected_answer, student_answer=student_answer
        )
    )
)

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