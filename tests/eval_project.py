import os
import json
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from app.models import RagEngine

# Load environment variables (.env file)
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
JUDGE_MODEL = "llama-3.3-70b-versatile"


# 1. Test dataset (16 Handbook Questions)

TEST_DATASET = [
    {
        "question": "What is the primary role of a Personal Tutor?",
        "ground_truth": "The primary role of a Personal Tutor is to discuss academic or pastoral issues with students, offer guidance on progress, feedback, module choices, and career options."
    },
    {
        "question": "Who are the Senior Tutors overseeing pastoral support in the School of Computer Science?",
        "ground_truth": "The Senior Tutors are Todd Waugh Ambridge (t.waughambridge@bham.ac.uk) and Ian Batten (i.g.batten@bham.ac.uk)."
    },
    {
        "question": "What is the direct email address to contact the Computer Science Wellbeing Officers?",
        "ground_truth": "You can contact them directly via cswelfare@contacts.bham.ac.uk. "
    },
    {
        "question": "What are the standard teaching hours for Monday through Friday at the UK campus?",
        "ground_truth": "Monday, Tuesday, Thursday: 09:00-19:00; Wednesday: 09:00-13:00; Friday: 09:00-18:00. "
    },
    {
        "question": "Where is the Education Support Office (ESO) located, and what are its opening hours?",
        "ground_truth": "Room UG44 (Building Y9) on the Upper Ground floor of the Computer Science Building; open Monday to Friday, 09:00–17:00 UK time.  "
    },
    {
        "question": "What are the module pass marks for undergraduate and postgraduate taught programmes?",
        "ground_truth": "The undergraduate pass mark is 40%, and the postgraduate pass mark is 50%.  "
    },
    {
        "question": "How many total credits are required for a BSc degree, an MSci degree, and an MSc degree?",
        "ground_truth": "A BSc requires 360 credits, an MSci requires 480 credits, and an MSc requires 180 credits (120 taught component + 60 project component).  "
    },
    {
        "question": "How many hours of work are expected for a standard 20-credit module?",
        "ground_truth": "200 hours of total work (including contact time, continuous assessment, and private study).  "
    },
    {
        "question": "What is the minimum number of credits an undergraduate student must pass to progress to the next year of study?",
        "ground_truth": "A minimum of 100 credits (out of 120).  "
    },
    {
        "question": "What penalty is applied for late coursework submissions if late submissions are permitted?",
        "ground_truth": "A lateness penalty of 5% per working day, up to a final cut-off of 1 week after the submission date. "
    },
    {
        "question": "What phone numbers should students use to contact Security Services for emergencies and non-emergencies?",
        "ground_truth": "Emergency: 0121 414 4444 (ext. 44444); Non-emergency: 0121 414 3000 (ext. 43000).  "
    },
    {
        "question": "By when must an undergraduate student request a transfer to a Study Abroad variant of their programme?",
        "ground_truth": "Before the end of Week 3 of Year 2.  "
    },
    {
        "question": "What overall mark in Year 2 must a student achieve to proceed to the Year 3 Study Abroad module?",
        "ground_truth": "An overall second-year mark of 55% or higher, with no outstanding reassessments.  "
    },
    {
        "question": "What is the name of the official hacking and cyber security student club at the University of Birmingham?",
        "ground_truth": "A Finite Number of Monkeys (AFNOM).  "
    },
    {
        "question": "What are the two key components of the End Point Assessment (EPA) for Degree Apprenticeship students?",
        "ground_truth": "1. Gateway Meeting; 2. EPA Presentation.  "
    },
    {
        "question": "What software system is used by the university to monitor student attendance at in-person teaching sessions?",
        "ground_truth": "MyAttendance.  "
    },

]


# 2. GROQ judge prompt and evaluation logic

JUDGE_SYSTEM_PROMPT = """
You are an expert academic AI evaluator grading a Retrieval-Augmented Generation (RAG) system.
Grade the system's output across four metrics on a scale from 0.0 to 1.0:

1. Faithfulness: Is the generated answer grounded strictly ONLY in the retrieved context? (1.0 = fully grounded, 0.0 = halluncinated/unsupported).
2. Answer Relevancy: Does the answer directly address the user's question? (1.0 = completely direct, 0.0 = off-topic/evasive).
3. Context Precision: Are the retrieved contexts relevant and free from unnecessary noise? (1.0 = clean & relevant, 0.0 = completely irrelevant).
4. Context Recall: Does the retrieved context contain all the facts present in the ground truth answer? (1.0 = full facts retrieved, 0.0 = missing key facts).

CRITICAL: Output ONLY a valid JSON object matching this schema. No markdown wrappers or extra commentary.
{
  "faithfulness": float,
  "answer_relevancy": float,
  "context_precision": float,
  "context_recall": float,
  "reasoning": "brief 1-sentence summary of evaluation"
}
"""


def evaluate_with_groq(question, retrieved_context, generated_answer, ground_truth):
    """Sends a single response payload to Groq to act as judge."""
    user_eval_prompt = f"""
    [USER QUESTION]: {question}
    [GROUND TRUTH ANSWER]: {ground_truth}
    [RETRIEVED CONTEXT]: {retrieved_context}
    [GENERATED ANSWER]: {generated_answer}
    """

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_eval_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"[-] Evaluation API call failed: {e}")
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "reasoning": "Evaluation failed due to API error."
        }


# 3. Execution
def run_evaluation():
    print("[*] Initializing RagEngine...")
    engine = RagEngine()

    results_log = []

    print(f"[*] Running evaluation on {len(TEST_DATASET)} test queries using model '{JUDGE_MODEL}'...\n")

    for idx, item in enumerate(TEST_DATASET, start=1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"[{idx}/{len(TEST_DATASET)}] Evaluating Query: '{question}'")

        # Step 1: Run query through production RAG pipeline
        answer_text, citations = engine.search_and_generate(question, ['public'])

        # Step 2: Format retrieved citations as context string
        if citations:
            formatted_context = "\n".join([
                f"- Page {c.get('page', 'N/A')} ({c.get('filename', 'Doc')}): {c.get('text', '')[:300]}..."
                for c in citations
            ])
        else:
            formatted_context = "No context retrieved / Refusal triggered."

        # Step 3: Grade using Groq Judge
        scores = evaluate_with_groq(question, formatted_context, answer_text, ground_truth)

        # Step 4: Record output row
        row = {
            "Question": question,
            "Ground Truth": ground_truth,
            "Generated Answer": answer_text,
            "Faithfulness": scores.get("faithfulness", 0.0),
            "Answer Relevancy": scores.get("answer_relevancy", 0.0),
            "Context Precision": scores.get("context_precision", 0.0),
            "Context Recall": scores.get("context_recall", 0.0),
            "Judge Reasoning": scores.get("reasoning", "")
        }
        results_log.append(row)

    # Step 5: Process and display summary
    df = pd.DataFrame(results_log)

    print("\n" + "=" * 60)
    print("             CUSTOM GROQ EVALUATION SUMMARY RESULTS           ")
    print("=" * 60)
    print(f"Mean Faithfulness:      {df['Faithfulness'].mean():.2f}")
    print(f"Mean Answer Relevancy:  {df['Answer Relevancy'].mean():.2f}")
    print(f"Mean Context Precision: {df['Context Precision'].mean():.2f}")
    print(f"Mean Context Recall:    {df['Context Recall'].mean():.2f}")
    print("=" * 60)

    # Export to CSV for figures and analysis
    output_filename = "rag_evaluation_results.csv"
    df.to_csv(output_filename, index=False)
    print(f"\n[+] Detailed evaluation report saved to: {output_filename}")


if __name__ == "__main__":
    run_evaluation()