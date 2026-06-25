from flask.cli import load_dotenv
from langchain_core.prompts import PromptTemplate
from output_parser import parser
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
import json
import os
from wolfram import verify_claim
    
class LLM:
    
    def __init__(self, azure_deployment, openai_api_version, temperature):
        
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        
        self.model = AzureChatOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            openai_api_version=openai_api_version,
            azure_deployment=azure_deployment,
            temperature=temperature,
        )

        # Use a low-temperature instance for scoring to reduce invention
        self.scoring_model = AzureChatOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            openai_api_version=openai_api_version,
            azure_deployment=azure_deployment,
            temperature=0.0,
        )

    def generate_questions_and_answers(self, document_content, num_questions, difficulty):
        prompt = PromptTemplate(
            template="You are a helpful learning assistant for students. Your goal is to facilitate their learning by "
                    "testing their understanding of the content from a lecture note. Based on the provided lecture "
                    "document, generate {num_questions} questions. Ensure that these questions are of {difficulty} "
                    "difficulty. A question should include a list of 4 answers if it is an MCQ,  each answer has an indication "
                    "whether it is a correct answer and a reason to justify why this answer is correct or incorrect. "
                    "Otherwise, all of the questions should be a multi select question and there can be more than one correct answer. "
                    "The possible answers does not have to be solely from the content of the document. You may also "
                    "generate other possible answers depending on the difficulty level.\n\n"
                    "ADDITIONAL REQUIREMENTS:\n"
                    "1) Avoid definition-only questions (max 20% if difficulty is Easy).\n"
                    "2) Ensure coverage across different topics/sections of the document; do not cluster on one topic.\n"
                    "3) Use plausible distractors based on common misconceptions.\n"
                    "4) For Medium/Hard, prioritize application and scenario-based questions.\n"
                    "5) Include at least 1 question that requires reasoning across multiple concepts.\n"
                    "6) Keep wording concise and unambiguous.\n"
                    "7) Keep answer options parallel in length and style.\n"
                    "8) Target distribution (approx.):\n"
                    "- Easy: Remember 30%, Understand 30%, Apply 20%, Analyze 10%, Evaluate 5%, Create 5%\n"
                    "- Medium: Remember 15%, Understand 25%, Apply 25%, Analyze 20%, Evaluate 10%, Create 5%\n"
                    "- Hard: Remember 10%, Understand 15%, Apply 25%, Analyze 25%, Evaluate 15%, Create 10%\n\n"
                    "QUESTION TYPE TAGGING:\n"
                    "Set question_type to one of: mcq, matching, categorising, latex_mcq.\n"
                    "Use latex_mcq for calculation-based questions; format math using LaTeX.\n"
                    "If question_type is matching or categorising, include a structured_data object:\n"
                    "- matching: {{\"pairs\": [{{\"left\": \"...\", \"right\": \"...\"}}]}}\n"
                    "- categorising: {{\"categories\": [{{\"name\": \"...\", \"items\": [\"...\"]}}]}}\n"
                    "For every question, include a short hint in a field named \"hint\".\n"
                    "Always include the standard 4-answer list for compatibility.\n"
                    "{format_instructions} \n\n"
                    "Below is the content of the lecture document:\n\n{document_content}",
            input_variables=["num_questions", "difficulty", "document_content"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        chain = prompt | self.model | parser

        result = chain.invoke({
            "num_questions": num_questions,
            "difficulty": difficulty,
            "document_content": document_content
        })

        try:
            questions = result.get("questions", []) if isinstance(result, dict) else []
            for question in questions:
                hint = question.get("hint")
                if hint:
                    continue

                question_type = question.get("question_type", "mcq")
                if question_type == "matching":
                    question["hint"] = "Match each pair based on the core definitions."
                elif question_type == "categorising":
                    question["hint"] = "Group each item by the category definitions."
                elif question_type == "latex_mcq":
                    question["hint"] = "Set up the calculation carefully and check units."
                else:
                    question["hint"] = "Recall the key concept and eliminate distractors."
        except Exception:
            pass

        return result
    
    def generate_short_ans_questions_and_answers(self, document_content, num_questions, difficulty):
        prompt = prompt = PromptTemplate(
            template="You are a helpful learning assistant for students. Your goal is to facilitate their learning by "
                    "testing their understanding of the content from a lecture note. Based on the provided lecture "
                    "document, generate {num_questions} questions. Ensure that these questions are of {difficulty} "
                    "difficulty. A question should be short-answer questions that require a free-text response. "
                    "The possible answers does not have to be solely from the content of the document. You may also "
                    "generate other possible answers depending on the difficulty level.\n\n"
                    "ADDITIONAL REQUIREMENTS:\n"
                    "1) Avoid definition-only questions (max 20% if difficulty is Easy).\n"
                    "2) Include questions which ask the student to solve and proof an equation (at least 40%)."
                    "3) Ensure coverage across different topics/sections of the document; do not cluster on one topic.\n"
                    "4) Use plausible distractors based on common misconceptions.\n"
                    "5) For Medium/Hard, prioritize application questions.\n"
                    "6) Include at least 1 question that requires reasoning across multiple concepts.\n"
                    "7) Keep wording concise and unambiguous.\n"
                    "8) Keep answer options parallel in length and style.\n"
                    "9) Target distribution (approx.):\n"
                    "- Easy: Remember 30%, Understand 30%, Apply 20%, Analyze 10%, Evaluate 5%, Create 5%\n"
                    "- Medium: Remember 15%, Understand 25%, Apply 25%, Analyze 20%, Evaluate 10%, Create 5%\n"
                    "- Hard: Remember 10%, Understand 15%, Apply 25%, Analyze 25%, Evaluate 15%, Create 10%\n\n"
                    "QUESTION TYPE TAGGING:\n"
                    "Set question_type to one of: short_ans, latex_short_ans.\n"
                    "Use latex_short_ans for calculation-based questions; format math using LaTeX.\n"
                    "For every question, include a short hint in a field named \"hint\".\n"
                    "{format_instructions} \n\n"
                    "Below is the content of the lecture document:\n\n{document_content}",
            input_variables=["num_questions", "difficulty", "document_content"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        chain = prompt | self.model | parser

        result = chain.invoke({
            "num_questions": num_questions,
            "difficulty": difficulty,
            "document_content": document_content
        })

        return result

    def generate_personalised_feedback(self, attempt_data):
        """
        Generate personalised, educational feedback for student's quest attempt
        """
        # Calculate basic statistics
        answers_list = attempt_data.get('answers', [])
        questions_map = {}
        for ans in answers_list:
            question_id = ans.get('question_id')
            if question_id is None:
                continue
            if question_id not in questions_map:
                questions_map[question_id] = {
                    "total_correct": 0,
                    "selected_correct": 0
                }
            answer_is_correct = ans.get('answer_is_correct')
            if answer_is_correct is None:
                answer_is_correct = ans.get('is_correct') and not ans.get('is_selected') is False
            if answer_is_correct:
                questions_map[question_id]["total_correct"] += 1
                if ans.get('is_selected'):
                    questions_map[question_id]["selected_correct"] += 1

        total_count = len(questions_map) or 0
        correct_count = sum(
            1 for stats in questions_map.values()
            if stats["total_correct"] > 0 and stats["selected_correct"] == stats["total_correct"]
        )
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0

        prompt = PromptTemplate(
            template="You are an educational tutor. Analyze this student's quiz attempt and provide detailed, "
             "constructive, and encouraging feedback.\n\n"
             "STUDENT PERFORMANCE SUMMARY:\n"
             "- Total Questions: {total_questions}\n"
             "- Correct Answers: {correct_answers}\n"
             "- Accuracy: {accuracy}%\n\n"
             "DETAILED ANSWERS:\n{attempt_data}\n\n"
             "INSTRUCTIONS:\n"
             "Provide feedback in the following JSON format (return ONLY valid JSON, no markdown, no extra text):\n"
             "{{\n"
             '  "quest_summary": {{\n'
             '    "overall_bloom_rating": 1,\n'
             '    "overall_bloom_level": "Remember",\n'
             '    "summary": "2-3 sentence summary of performance across the quest."\n'
             "  }},\n"
             '  "subtopic_feedback": [\n'
             "    {{\n"
             '      "subtopic": "Subtopic name",\n'
             '      "bloom_rating": 2,\n'
             '      "bloom_level": "Understand",\n'
             '      "evidence": "Short evidence grounded in the student answers.",\n'
             '      "improvement_focus": "One sentence on what to improve in this subtopic."\n'
             "    }}\n"
             "  ],\n"
             '  "study_tips": [\n'
             '    "Practical study tip 1",\n'
             '    "Practical study tip 2"\n'
             "  ]\n"
             "}}\n\n"
             "BLOOM SCALE (STRICT 1-6)\n"
             "1 = Remember\n"
             "2 = Understand\n"
             "3 = Apply\n"
             "4 = Analyse\n"
             "5 = Evaluate\n"
             "6 = Create\n\n"
             "IMPORTANT GUIDELINES:\n"
             "1. Use ONLY the bloom levels listed and map them strictly to the 1-6 ratings\n"
             "2. Infer subtopics by grouping related questions; use concise subtopic names\n"
             "3. Provide 3-8 subtopic entries depending on coverage\n"
             "4. The quest summary should be 2-3 sentences and match the overall bloom rating\n"
             "5. Provide 3-6 study tips as a list, focused on the weakest subtopics\n"
             "6. Use an encouraging, supportive tone - emphasize growth mindset\n"
             "7. Return ONLY the JSON object, no additional text before or after",
            input_variables=["total_questions", "correct_answers", "accuracy", "attempt_data"]
        )

        chain = prompt | self.model

        try:
            result = chain.invoke({
                "total_questions": total_count,
                "correct_answers": correct_count,
                "accuracy": f"{accuracy:.1f}",
                "attempt_data": json.dumps(answers_list, indent=2)
            })

            # Parse the response content as JSON
            content = result.content.strip()
            print("[LLM Raw]", content)
            # Remove markdown code blocks if present
            if content.startswith('```json'):
                content = content.replace('```json', '').replace('```', '').strip()
            elif content.startswith('```'):
                content = content.replace('```', '').strip()
            
            
            feedback_json = json.loads(content)
            
            # Validate required keys
            required_keys = ['quest_summary', 'subtopic_feedback', 'study_tips']
            if not all(key in feedback_json for key in required_keys):
                raise ValueError("Missing required feedback fields")

            print(f"[LLM Feedback] Generated successfully")
            return feedback_json

        except json.JSONDecodeError as e:
            print(f"[LLM Feedback Error] JSON parsing failed: {str(e)}")

            # Return fallback structure
            return {
                "quest_summary": {
                    "overall_bloom_rating": 2,
                    "overall_bloom_level": "Understand",
                    "summary": f"You completed the quest with {accuracy:.1f}% accuracy. Review the topics you missed and "
                               "practice applying them in new contexts for better mastery."
                },
                "subtopic_feedback": [],
                "study_tips": [
                    "Review the lesson notes for the questions you missed.",
                    "Redo similar practice questions to reinforce understanding.",
                    "Explain key concepts in your own words to check understanding."
                ]
            }

        except Exception as e:
            print(f"[LLM Feedback Error] Unexpected error: {str(e)}")

            # Return minimal fallback structure
            return {
                "quest_summary": {
                    "overall_bloom_rating": 1,
                    "overall_bloom_level": "Remember",
                    "summary": "Completed the quest. Review the materials and keep practicing."
                },
                "subtopic_feedback": [],
                "study_tips": ["Keep practicing to improve your understanding."]
            }
        
    def generate_personalised_shortans_feedback(self, attempt_data):
        """
        Generate personalised, educational feedback for student's quest attempt
        """
        # Calculate basic statistics
        answers_list = attempt_data.get('answers', [])
        total_score = 0
        for ans in answers_list:
            total_score += ans.get('score_achieved')

        total_count = len(answers_list) or 0
        accuracy = (total_score / total_count * 100) if total_count > 0 else 0

        prompt = PromptTemplate(
            template="You are an educational tutor. Analyze this student's quiz attempt and provide detailed, "
             "constructive, and encouraging feedback.\n\n"
             "STUDENT PERFORMANCE SUMMARY:\n"
             "- Total Questions: {total_questions}\n"
             "- Total Score: {total_score}\n"
             "- Accuracy: {accuracy}%\n\n"
             "DETAILED ANSWERS:\n{attempt_data}\n\n"
             "INSTRUCTIONS:\n"
             "Provide feedback in the following JSON format (return ONLY valid JSON, no markdown, no extra text):\n"
             "{{\n"
             '  "quest_summary": {{\n'
             '    "overall_bloom_rating": 1,\n'
             '    "overall_bloom_level": "Remember",\n'
             '    "summary": "2-3 sentence summary of performance across the quest."\n'
             "  }},\n"
             '  "subtopic_feedback": [\n'
             "    {{\n"
             '      "subtopic": "Subtopic name",\n'
             '      "bloom_rating": 2,\n'
             '      "bloom_level": "Understand",\n'
             '      "evidence": "Short evidence grounded in the student answers.",\n'
             '      "improvement_focus": "One sentence on what to improve in this subtopic."\n'
             "    }}\n"
             "  ],\n"
             '  "study_tips": [\n'
             '    "Practical study tip 1",\n'
             '    "Practical study tip 2"\n'
             "  ]\n"
             "}}\n\n"
             "BLOOM SCALE (STRICT 1-6)\n"
             "1 = Remember\n"
             "2 = Understand\n"
             "3 = Apply\n"
             "4 = Analyse\n"
             "5 = Evaluate\n"
             "6 = Create\n\n"
             "IMPORTANT GUIDELINES:\n"
             "1. Use ONLY the bloom levels listed and map them strictly to the 1-6 ratings\n"
             "2. Infer subtopics by grouping related questions; use concise subtopic names\n"
             "3. Provide 3-8 subtopic entries depending on coverage\n"
             "4. The quest summary should be 2-3 sentences and match the overall bloom rating\n"
             "5. Provide 3-6 study tips as a list, focused on the weakest subtopics\n"
             "6. Use an encouraging, supportive tone - emphasize growth mindset\n"
             "7. Return ONLY the JSON object, no additional text before or after",
            input_variables=["total_questions", "correct_answers", "accuracy", "attempt_data"]
        )

        chain = prompt | self.model

        try:
            result = chain.invoke({
                "total_questions": total_count,
                "total_score": total_score,
                "accuracy": f"{accuracy:.1f}",
                "attempt_data": json.dumps(answers_list, indent=2)
            })

            # Parse the response content as JSON
            content = result.content.strip()
            print("[LLM Raw]", content)
            # Remove markdown code blocks if present
            if content.startswith('```json'):
                content = content.replace('```json', '').replace('```', '').strip()
            elif content.startswith('```'):
                content = content.replace('```', '').strip()
            
            
            feedback_json = json.loads(content)
            
            # Validate required keys
            required_keys = ['quest_summary', 'subtopic_feedback', 'study_tips']
            if not all(key in feedback_json for key in required_keys):
                raise ValueError("Missing required feedback fields")

            print(f"[LLM Feedback] Generated successfully")
            return feedback_json

        except json.JSONDecodeError as e:
            print(f"[LLM Feedback Error] JSON parsing failed: {str(e)}")

            # Return fallback structure
            return {
                "quest_summary": {
                    "overall_bloom_rating": 2,
                    "overall_bloom_level": "Understand",
                    "summary": f"You completed the quest with {accuracy:.1f}% accuracy. Review the topics you missed and "
                               "practice applying them in new contexts for better mastery."
                },
                "subtopic_feedback": [],
                "study_tips": [
                    "Review the lesson notes for the questions you missed.",
                    "Redo similar practice questions to reinforce understanding.",
                    "Explain key concepts in your own words to check understanding."
                ]
            }

        except Exception as e:
            print(f"[LLM Feedback Error] Unexpected error: {str(e)}")

            # Return minimal fallback structure
            return {
                "quest_summary": {
                    "overall_bloom_rating": 1,
                    "overall_bloom_level": "Remember",
                    "summary": "Completed the quest. Review the materials and keep practicing."
                },
                "subtopic_feedback": [],
                "study_tips": ["Keep practicing to improve your understanding."]
            }

    def generate_bonus_game(self, document_content, game_type):
        if game_type == "matching":
            prompt = PromptTemplate(
                template="You are a learning assistant. Create a matching pairs mini-game based on the document.\n"
                         "Return ONLY valid JSON, no markdown.\n\n"
                         "FORMAT:\n"
                         "{{\n"
                         '  "game_type": "matching",\n'
                         '  "prompt": "...",\n'
                         '  "pairs": [\n'
                         '    {{"left": "...", "right": "..."}},\n'
                         '    {{"left": "...", "right": "..."}},\n'
                         '    {{"left": "...", "right": "..."}},\n'
                         '    {{"left": "...", "right": "..."}}\n'
                         '  ],\n'
                         '  "hint": "..."\n'
                         "}}\n\n"
                         "RULES:\n"
                         "- Generate 4 pairs.\n"
                         "- Keep text concise (<= 8 words each).\n"
                         "- Pairs must be clearly matched from the document.\n"
                         "- Avoid obscure or minor details.\n\n"
                         "DOCUMENT:\n{document_content}",
                input_variables=["document_content"]
            )
        else:
            prompt = PromptTemplate(
                template="You are a learning assistant. Create an ordering sequence mini-game based on the document.\n"
                         "Return ONLY valid JSON, no markdown.\n\n"
                         "FORMAT:\n"
                         "{{\n"
                         '  "game_type": "ordering",\n'
                         '  "prompt": "...",\n'
                         '  "items": ["...", "...", "...", "..."],\n'
                         '  "answer_order": [0, 1, 2, 3],\n'
                         '  "hint": "..."\n'
                         "}}\n\n"
                         "RULES:\n"
                         "- Generate 4 items in correct order in the items list.\n"
                         "- answer_order must be the correct index order (0..3).\n"
                         "- Use a process or sequence from the document.\n"
                         "- Keep items concise (<= 8 words each).\n\n"
                         "DOCUMENT:\n{document_content}",
                input_variables=["document_content"]
            )

        chain = prompt | self.model
        result = chain.invoke({
            "document_content": document_content
        })

        content = result.content.strip()
        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '').strip()
        elif content.startswith('```'):
            content = content.replace('```', '').strip()

        return json.loads(content)

    def generate_score_answer(self, claims, question, expected_answer, methods_used=None, student_answer=None, is_math_question=True):
        """
        Score using an LLM-based rubric. For each claim, produce evidence via
        `verify_claim` for math questions; skip Wolfram and use a generic rubric
        for non-math questions.
        Returns (score:int, breakdown:list).
        """

        evidence = []
        if is_math_question:
            for c in claims:
                stmt = c.get('statement')
                verified, wolfram_out = verify_claim(stmt, expected_answer)
                evidence.append({
                    'statement': stmt,
                    'type': c.get('type'),
                    'verified': bool(verified),
                    'wolfram': wolfram_out,
                })
        else:
            non_math_claims = claims or [{
                'statement': student_answer or '',
                'type': 'answer',
            }]
            for c in non_math_claims:
                stmt = c.get('statement')
                evidence.append({
                    'statement': stmt,
                    'type': c.get('type'),
                    'verified': True,
                    'wolfram': 'SKIPPED (non-math question)',
                })

        # LLM-driven scoring: provide a strict prompt that requires the model to
        # evaluate the Question and Student answer while using the Wolfram evidence
        # as guidance. The model must return strict JSON with score and breakdown.
        score_prompt = PromptTemplate.from_template("""
            You are a strict grader.

            Use the verified claims as factual evidence.

            If is_math_question, math questions:
            - Correctness is most important.
            - Mathematical justification should increase the score.
            - Missing justification should reduce the score.
            - Incorrect reasoning should reduce the score.
            - A correct final answer with no explanation should receive partial credit.

            If not is_math_question, non-math questions:
            - Evaluate correctness, completeness, relevance, and clarity.

            Return:
            {{
            "score": 0-10,
            "max_score": 10,
            "breakdown": [...]
            }}
                                                    
            Is math question: {is_math_question}
            Question: {question}
            Expected answer: {expected_answer}
            Student answer: {student_answer}
            LLM methods_used: {methods_used}
            Per-claim evidence: {evidence}
            """
        )
        
        # Provide methods_used derived only from the student's text to avoid the
        # LLM inventing justification that boosts score.
        methods_from_student = (student_answer or '').strip()
        chain = score_prompt | self.scoring_model | JsonOutputParser()
        parsed = chain.invoke({
            'question': question,
            'expected_answer': expected_answer,
            'student_answer': student_answer or '',
            'methods_used': methods_from_student,
            'is_math_question': is_math_question,
            'evidence': json.dumps(evidence)
        })

        # Validate LLM output and enforce hard guards for obvious rule violations.
        score = parsed.get('score', 0)
        print("[Generate Score Answer] RAW SCORE:", parsed.get("score"))
        breakdown = parsed.get('breakdown', []) or []

        # Normalize and validate types
        try:
            score = int(score)
        except Exception:
            score = 0

        # Ensure breakdown length matches number of claims; if not, rebuild deterministically
        if len(breakdown) != len(evidence):
            # Fallback: produce a safe breakdown using unverified/verified flags
            safe_bd = []
            for ev in evidence:
                safe_bd.append({
                    'statement': ev['statement'],
                    'verified': bool(ev['verified']),
                    'wolfram': ev.get('wolfram'),
                    'points': 0 if not ev['verified'] else 4,
                    'reason': 'fallback breakdown' if ev['verified'] else 'unverified',
                })
            breakdown = safe_bd

        # Enforce: unverified claims must have 0 points
        for i, ev in enumerate(evidence):
            if not ev['verified']:
                if breakdown[i].get('points', 0) != 0:
                    breakdown[i]['points'] = 0
                    breakdown[i]['reason'] = 'unverified (enforced)'

        strict_kw = [k.lower() for k in ['polynomial', 'polynomials', 'continuous', 'continuity']]
        general_kw = [k.lower() for k in [
            'direct substitution',
            'directly substitute',
            'substitute',
            'substitution',
            'term by term',
            'sum law',
            'difference law',
            'power law',
            'constant multiple law',
            'limit law',
            'limit laws'
        ]]
        plug_in_kw = [k.lower() for k in ['plug in', 'plugging in', 'plugging', 'plugged in']]
        incorrect_kw = [k.lower() for k in ['squeeze', 'squeeze theorem', 'bounded']]
        methods_text = (student_answer or '').lower()

        for i, ev in enumerate(evidence):
            if not ev['verified']:
                continue
            
            # Determine allowed range based on keywords (new mapping)
            # For verified claims with good methods, give them credit towards the higher end.
            min_allowed = 2
            max_allowed = 10  # default: any verified claim can theoretically get up to 10
            
            if any(k in methods_text for k in strict_kw):
                # Strict keywords (continuous, polynomial): range [8, 10]
                min_allowed = 8
                max_allowed = 10
            elif any(k in methods_text for k in general_kw):
                # General keywords (substitute, direct substitution): range [7, 9]
                # Verified answer with concrete method deserves high credit (just below strict theory)
                min_allowed = 7
                max_allowed = 9
            elif any(k in methods_text for k in plug_in_kw):
                # Plug-in keywords: range [4, 6]
                min_allowed = 4
                max_allowed = 6
            else:
                import re
                sa_stripped = methods_text.strip()
                numeric_only = bool(re.fullmatch(r"-?\d+(?:\.\d+)?", sa_stripped)) or (
                    len(sa_stripped) <= 6 and bool(re.fullmatch(r"[\d\s\-\+\*/\.()]+", sa_stripped))
                )
                if numeric_only:
                    # Numeric only but verified: range [5, 6] (deserves credit for correct answer)
                    min_allowed = 5
                    max_allowed = 6
                elif any(k in methods_text for k in incorrect_kw):
                    # Incorrect methods: range [0, 2]
                    min_allowed = 0
                    max_allowed = 2
                else:
                    # Fallback: range [1, 3]
                    min_allowed = 1
                    max_allowed = 3

            try:
                llm_points = int(breakdown[i].get('points', 0))
            except Exception:
                llm_points = 0
            
            # Enforce range: clip LLM score to [min_allowed, max_allowed]
            original_points = llm_points
            clamped_points = max(min_allowed, min(llm_points, max_allowed))
            
            # Always update points and reason if there's a mismatch (including boosting low scores)
            breakdown[i]['points'] = clamped_points
            if clamped_points != original_points:
                old_reason = breakdown[i].get('reason', '')
                breakdown[i]['reason'] = f"{old_reason} (validator: clamped {original_points} to [{min_allowed}, {max_allowed}])"
            elif min_allowed > 0:
                # Even if no clamping needed, annotate the validator range for clarity
                old_reason = breakdown[i].get('reason', '')
                breakdown[i]['reason'] = f"{old_reason} (validator range: [{min_allowed}, {max_allowed}])"

        # Only enforce score=0 if nothing verified
        if is_math_question:
            any_verified = any(
                ev.get("verified", False)
                for ev in evidence
            )

            if not any_verified:
                score = 0

        # Clamp to valid range
        score = max(0, min(10, int(score)))
        print(f"[Generate Score Answer] Actual Score:{score}")

        return int(score), breakdown
    
    def generate_mathematical_claims(self, question, expected_answer, student_answer):
        prompt = PromptTemplate.from_template("""
            You extract VERIFIABLE mathematical claims from a student answer.

            IMPORTANT: 
            - Extract only concrete claims that CAN be verified by computation (limits, equations, computed values)
            - DO NOT extract purely theoretical/justification statements like:
            * "polynomials are continuous"
            * "limit laws apply"
            * "the function is continuous"
            * "by squeeze theorem"
            * Any statement that cannot be directly evaluated mathematically
            - Only extract: specific limit evaluations, equations with values, step-by-step calculations

            Return STRICT JSON:

            {{
            "claims": [
                {{
                "statement": "verifiable expression or equation (e.g., '\\\\lim_{{x\\\\to 5}}(...)=39' or '2(5)^2-3(5)+4=39')",
                "type": "limit|equation|other"
                }}
            ],
            "methods_used": ["only actual methods mentioned, not invented ones"]
            }}

            Question:
            {question}

            Expected:
            {expected_answer}

            Student:
            {student_answer}
            """
        )


        chain = prompt | self.scoring_model | JsonOutputParser()

        result = chain.invoke({
            "question": question,
            "expected_answer": expected_answer,
            "student_answer": student_answer
        })

        return result
