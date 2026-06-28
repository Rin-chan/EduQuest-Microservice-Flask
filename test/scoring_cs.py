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
    question = r"A user enters `25` for `date`. Evaluate `thedayaftertomorrow = int(date) + 2` and give the printed result."
    expected_answer = r"Correct answer: 27. The `input()` function returns a string, and `int(date)` changes `'25'` to `25`, so `25 + 2 = 27`."
    good_student_answer = r"""date is 25.
    int(date) converts it to the integer 25.
    Then 25 + 2 = 27.
    Therefore, thedayaftertomorrow is 27, so the printed result is 27.
    """
    med_student_answer = r"int(25) + 2 = 27, so the printed result is 27."
    bad_student_answer = r"27"
    wrong_student_answer = r"int(date) converts '25' to an integer, but adding 2 gives 26. So the printed result is 26." 
    half_wrong_student_answer = r"The printed result is 252 because the program joins 25 and 2 together."

    student_answer = good_student_answer

    return question, expected_answer, student_answer

def test2():
    question = r"For `tom = 19`, show why `print(tom > 18)` prints `True`."
    expected_answer = r"A correct response should state that 19 is greater than 18, so the Boolean expression `tom > 18` is true and Python prints `True`. A common misconception is writing `true` in lowercase, but Python uses `True`."
    good_student_answer = r"""
    tom is 19. The expression tom > 18 checks whether 19 is greater than 18.
    Since 19 > 18 is true, print(tom > 18) prints True.
    """
    med_student_answer = r"19 > 18 is true, so prints True."
    bad_student_answer = r"True"
    wrong_student_answer = r"The program prints 19 because print(tom > 18) prints the value of tom."
    half_wrong_student_answer = r"tom is 19, but 19 > 18 is False, so the program prints False."

    student_answer = good_student_answer

    return question, expected_answer, student_answer

def test3():
    question = r"Suppose `horizon_dist = 1000` and `vertical_dist = 2500`. Compute `travel_dist` and decide whether the taxi message is printed."
    expected_answer = r"Correct answer: `travel_dist = 3500`, and the taxi message is printed because `3500 > 3000` is `True`. This uses both arithmetic and the `if` condition."
    good_student_answer = r"travel_dist = 1000 + 2500 = 3500. Since 3500 > 3000, the condition is True, so the taxi message is printed."
    med_student_answer = r"travel_dist is 3500, the taxi message is printed."
    bad_student_answer = r"travel_dist = 3500"
    wrong_student_answer = r"travel_dist = 2500000, so the taxi message is printed." 
    half_wrong_student_answer = r"travel_dist = 3500, but the taxi message is not printed because 3500 is less than 3000."

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