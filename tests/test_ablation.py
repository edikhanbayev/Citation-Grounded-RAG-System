import os
import sys
import re
import json
import csv
import argparse
import unittest
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import RagEngine from app/models.py
from app.models import RagEngine


# =====================================================================
# 1. EVALUATION DATASET (35 QUESTION-ANSWER BENCHMARK PAIRS)
# =====================================================================

@dataclass
class QuestionAnswerPair:
    id: int
    category: str
    question: str
    expected_answer: str
    expected_source: str
    expected_page: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


TEST_DATASET: List[QuestionAnswerPair] = [
    # Category 1: Academic, Wellbeing and Social Support
    QuestionAnswerPair(
        id=1,
        category="Academic, Wellbeing and Social Support",
        question="Who are the Senior Tutors in the School of Computer Science and what are their email addresses?",
        expected_answer="The Senior Tutors are Todd Waugh Ambridge (t.waughambridge@bham.ac.uk) and Ian Batten (i.g.batten@bham.ac.uk).",
        expected_source="academic_wellbeing_and_social_support_cs_handbook.pdf",
        expected_page="Page 1"
    ),
    QuestionAnswerPair(
        id=2,
        category="Academic, Wellbeing and Social Support",
        question="What is the direct email address for contacting the Computer Science Wellbeing Officers?",
        expected_answer="cswelfare@contacts.bham.ac.uk",
        expected_source="academic_wellbeing_and_social_support_cs_handbook.pdf",
        expected_page="Page 2"
    ),
    QuestionAnswerPair(
        id=3,
        category="Academic, Wellbeing and Social Support",
        question="What specific facility in the Main Library is highlighted for neurodiverse students to use for studying?",
        expected_answer="Bookable Assistive Technology Booths.",
        expected_source="academic_wellbeing_and_social_support_cs_handbook.pdf",
        expected_page="Page 4"
    ),
    QuestionAnswerPair(
        id=4,
        category="Academic, Wellbeing and Social Support",
        question="What Academic English support program is provided free of charge to international students by the Birmingham International Academy (BIA), and what is its contact email address?",
        expected_answer="The BIA Academic English Programme; contact email is bia@contacts.bham.ac.uk.",
        expected_source="academic_wellbeing_and_social_support_cs_handbook.pdf",
        expected_page="Page 5"
    ),
    QuestionAnswerPair(
        id=5,
        category="Academic, Wellbeing and Social Support",
        question="What is the name of the student-led hacking club at the University of Birmingham, what is its website, and what primary topic do they cover weekly?",
        expected_answer="A Finite Number of Monkeys (AFNOM); website is https://afnom.net; they focus on offensive hacking and cyber security.",
        expected_source="academic_wellbeing_and_social_support_cs_handbook.pdf",
        expected_page="Page 7"
    ),
    QuestionAnswerPair(
        id=6,
        category="Academic, Wellbeing and Social Support",
        question="Where is the Multi-Faith Chaplaincy located on campus?",
        expected_answer="Located at St Francis Hall, adjacent to the Guild of Students.",
        expected_source="academic_wellbeing_and_social_support_cs_handbook.pdf",
        expected_page="Page 8"
    ),

    # Category 2: General Information
    QuestionAnswerPair(
        id=7,
        category="General Information",
        question="What is the room number, floor, and building location of the Education Support Office (ESO)?",
        expected_answer="Room UG44, Upper Ground floor of the Computer Science Building (Building Y9 on the Campus Map).",
        expected_source="general_information_cs_handbook.pdf",
        expected_page="Page 3"
    ),
    QuestionAnswerPair(
        id=8,
        category="General Information",
        question="What are the contact email addresses for undergraduate and postgraduate taught (MSc) students to reach the Education Support Office (ESO)?",
        expected_answer="Undergraduate: ug-cs@contacts.bham.ac.uk; Postgraduate Taught (MSc): msc-cs@contacts.bham.ac.uk.",
        expected_source="general_information_cs_handbook.pdf",
        expected_page="Page 3"
    ),
    QuestionAnswerPair(
        id=9,
        category="General Information",
        question="What are the standard teaching hours on Wednesday and Friday during the UK University teaching week?",
        expected_answer="Wednesday: 09:00–13:00; Friday: 09:00–18:00.",
        expected_source="general_information_cs_handbook.pdf",
        expected_page="Page 6"
    ),
    QuestionAnswerPair(
        id=10,
        category="General Information",
        question="What are the three study spaces/facilities mentioned within the Computer Science building that are dedicated or available to Computer Science students?",
        expected_answer="1. Sloman Lounge; 2. CS Meeting Rooms; 3. CS Labs.",
        expected_source="general_information_cs_handbook.pdf",
        expected_page="Page 7"
    ),
    QuestionAnswerPair(
        id=11,
        category="General Information",
        question="What is the maximum duration permitted for an Authorised Absence, and which group of students is eligible for it?",
        expected_answer="Up to 8 weeks; strictly for Student Route Visa holders.",
        expected_source="general_information_cs_handbook.pdf",
        expected_page="Page 9"
    ),
    QuestionAnswerPair(
        id=12,
        category="General Information",
        question="What are the emergency and non-emergency telephone numbers for Security Services?",
        expected_answer="Emergency: 0121 414 4444; Non-emergency: 0121 414 3000.",
        expected_source="general_information_cs_handbook.pdf",
        expected_page="Page 12"
    ),

    # Category 3: Library, IT Services, and Security
    QuestionAnswerPair(
        id=13,
        category="Library, IT Services, and Security",
        question="How many printed books/manuscripts and electronic books are accessible through Library Services?",
        expected_answer="Over 2 million printed books/manuscripts, over 500,000 electronic books.",
        expected_source="library_it_services_and_security_cs_handbook.pdf",
        expected_page="Page 1"
    ),
    QuestionAnswerPair(
        id=14,
        category="Library, IT Services, and Security",
        question="What URL address is provided to search the University Library catalogue?",
        expected_answer="www.findit.bham.ac.uk",
        expected_source="library_it_services_and_security_cs_handbook.pdf",
        expected_page="Page 1"
    ),
    QuestionAnswerPair(
        id=15,
        category="Library, IT Services, and Security",
        question="Where is the Security Office located on campus, and what are its operating hours?",
        expected_answer="Rear of Aston Webb in B Block courtyard; operates 24/7.",
        expected_source="library_it_services_and_security_cs_handbook.pdf",
        expected_page="Page 3"
    ),

    # Category 4: Programme Information
    QuestionAnswerPair(
        id=16,
        category="Programme Information",
        question="How many total credits are required to be awarded a BSc, an MSci, and an MSc degree?",
        expected_answer="BSc: 360 credits; MSci: 480 credits; MSc: 180 credits.",
        expected_source="programme_information_cs_handbook.pdf",
        expected_page="Page 1"
    ),
    QuestionAnswerPair(
        id=17,
        category="Programme Information",
        question="How many total learning hours are expected for a 20-credit module, and what do these hours comprise?",
        expected_answer="200 hours total covering contact hours and private study/coursework.",
        expected_source="programme_information_cs_handbook.pdf",
        expected_page="Page 1"
    ),
    QuestionAnswerPair(
        id=18,
        category="Programme Information",
        question="What are the standard module pass marks for undergraduate and postgraduate programmes?",
        expected_answer="Undergraduate: 40; Postgraduate: 50.",
        expected_source="programme_information_cs_handbook.pdf",
        expected_page="Page 1"
    ),
    QuestionAnswerPair(
        id=19,
        category="Programme Information",
        question="What is a 'repeat-only' module, and what two examples are given in the handbook?",
        expected_answer="Module must be repeated in full next academic year. Examples: Final year projects and group work modules.",
        expected_source="programme_information_cs_handbook.pdf",
        expected_page="Page 3"
    ),
    QuestionAnswerPair(
        id=20,
        category="Programme Information",
        question="What is the default policy for late submission of coursework in the School of Computer Science, and what penalty applies if late submissions are explicitly allowed?",
        expected_answer="Default: No late submissions accepted. If allowed: 5% penalty per working day up to 1 week.",
        expected_source="programme_information_cs_handbook.pdf",
        expected_page="Page 5"
    ),
    QuestionAnswerPair(
        id=21,
        category="Programme Information",
        question="What is the expected timeframe for feedback return on mid-module assessments versus all other assessments?",
        expected_answer="15 working days for mid-module assessments; 20 working days for all others.",
        expected_source="programme_information_cs_handbook.pdf",
        expected_page="Page 5"
    ),
    QuestionAnswerPair(
        id=22,
        category="Programme Information",
        question="How is the overall degree mark calculated across study years for a student undertaking a Study Abroad program?",
        expected_answer="20% Year 2, 5% Year 3 (Study Abroad), 75% Year 4.",
        expected_source="programme_information_cs_handbook.pdf",
        expected_page="Page 7"
    ),

    # Category 5: Student Representation
    QuestionAnswerPair(
        id=23,
        category="Student Representation",
        question="How many total students does the Guild of Students represent, and how many student groups does it house?",
        expected_answer="Represents over 38,000 students and houses over 350 student groups.",
        expected_source="student_representation_cs_handbook.pdf",
        expected_page="Pages 1–2"
    ),
    QuestionAnswerPair(
        id=24,
        category="Student Representation",
        question="Approximately how many Student Reps operate across the University of Birmingham?",
        expected_answer="Over 1,000 Student Reps.",
        expected_source="student_representation_cs_handbook.pdf",
        expected_page="Page 2"
    ),
    QuestionAnswerPair(
        id=25,
        category="Student Representation",
        question="What exact web address contains the agendas and minutes for the Staff Student Forums (SSFs)?",
        expected_answer="https://www.cs.bham.ac.uk/internal/local/committees/staffstudent",
        expected_source="student_representation_cs_handbook.pdf",
        expected_page="Page 3"
    ),

    # Category 6: University Legislation
    QuestionAnswerPair(
        id=26,
        category="University Legislation",
        question="Explain how 'cohort legislation' determines which regulations apply to a student who entered in 2024-25 versus 2025-26.",
        expected_answer="Cohort legislation binds a student to regulations of their entry year throughout their degree.",
        expected_source="university_legislation_cs_handbook.pdf",
        expected_page="Page 2"
    ),
    QuestionAnswerPair(
        id=27,
        category="University Legislation",
        question="Where and when are details of extenuating circumstances considered to ensure student privacy, rather than at full Board of Examiners meetings?",
        expected_answer="At Extenuating Circumstances Panels after each assessment period.",
        expected_source="university_legislation_cs_handbook.pdf",
        expected_page="Page 4"
    ),
    QuestionAnswerPair(
        id=28,
        category="University Legislation",
        question="What rules apply regarding wristwatches, mobile phones, and coats during in-person campus examinations?",
        expected_answer="Phones and smart watches off under desk; simple watches allowed; coats in designated area.",
        expected_source="university_legislation_cs_handbook.pdf",
        expected_page="Pages 12–13"
    ),
    QuestionAnswerPair(
        id=29,
        category="University Legislation",
        question="What are three prohibited uses and three permitted uses of Generative AI during assessment preparation according to University policy?",
        expected_answer="Prohibited: Altering text arguments, correcting accuracy, translating to reduce length. Permitted: Spelling/punctuation advice, grammar checks, shortening without content change.",
        expected_source="university_legislation_cs_handbook.pdf",
        expected_page="Page 15"
    ),
    QuestionAnswerPair(
        id=30,
        category="University Legislation",
        question="What is the primary role of the two External Advisors on Academic Standards appointed by the University?",
        expected_answer="One scrutinises UG standards and the other PGT standards for external quality assurance.",
        expected_source="university_legislation_cs_handbook.pdf",
        expected_page="Page 16"
    ),
    QuestionAnswerPair(
        id=31,
        category="University Legislation",
        question="How do I request a deadline extension if I get sick?",
        expected_answer="test",
        expected_source="programme_information_cs_handbook.pdf",
        expected_page="Page 5"
    ),
    QuestionAnswerPair(
        id=32,
        category="University Legislation",
        question="Can I use automated software like ChatGPT to refine the logic and argument structure of my essay draft?",
        expected_answer="test",
        expected_source="university_legislation_cs_handbook.pdf",
        expected_page="Page 15"
    ),
    QuestionAnswerPair(
        id=33,
        category="University Legislation",
        question="What happens if an invigilator catches a student with handwriting on their skin during an assessment?",
        expected_answer="test",
        expected_source="university_legislation_cs_handbook.pdf",
        expected_page="Page 13"
    ),
    QuestionAnswerPair(
        id=34,
        category="Programme Information",
        question="How do second-chance tests work for undergrads who flunk a class, and what is the maximum grade cap?",
        expected_answer="test",
        expected_source="programme_information_cs_handbook.pdf",
        expected_page="Page 4"
    ),
    QuestionAnswerPair(
        id=35,
        category="Academic, Wellbeing and Social Support",
        question="If a major family emergency or severe emotional crisis affects my studies, who handles my situation confidentially without me having to inform my professors first?",
        expected_answer="test",
        expected_source="academic_wellbeing_and_social_support_cs_handbook.pdf",
        expected_page="Page 2"
    )
]


# =====================================================================
# 2. EVALUATION METRIC CALCULATIONS & HELPER EXTRACTORS
# =====================================================================

def parse_expected_pages(page_str: str) -> List[int]:
    """
    Parses expected page strings into a list of integer page numbers.
    Supports formats like 'Page 1', 'Pages 1–2', 'Pages 12-13'.
    """
    if not page_str:
        return []

    # Normalize unicode hyphens/dashes to standard hyphen
    normalized = page_str.replace("–", "-").replace("—", "-")

    # Check for range pattern (e.g. 1-2 or 12-13)
    match_range = re.search(r'(\d+)\s*-\s*(\d+)', normalized)
    if match_range:
        start, end = int(match_range.group(1)), int(match_range.group(2))
        return list(range(start, end + 1))

    # Otherwise extract all standalone integers
    return [int(n) for n in re.findall(r'\d+', normalized)]


def extract_source_name(chunk: Any) -> str:
    """Helper to extract document source filename from payload format."""
    if not isinstance(chunk, dict):
        return ""

    for key in ["filename", "source", "file_name"]:
        val = chunk.get(key)
        if val and isinstance(val, str):
            return os.path.basename(val).strip().lower()

    meta = chunk.get("metadata", {})
    if isinstance(meta, dict):
        for key in ["filename", "source", "file_name"]:
            val = meta.get(key)
            if val and isinstance(val, str):
                return os.path.basename(val).strip().lower()

    return ""


def extract_page_number(chunk: Any) -> Optional[int]:
    """Helper to extract integer page number from retrieved payload format."""
    if not isinstance(chunk, dict):
        return None

    for key in ["page", "page_number", "page_num"]:
        val = chunk.get(key)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                pass

    meta = chunk.get("metadata", {})
    if isinstance(meta, dict):
        for key in ["page", "page_number", "page_num"]:
            val = meta.get(key)
            if val is not None:
                try:
                    return int(val)
                except (ValueError, TypeError):
                    pass

    return None


def evaluate_retrieved_chunks(
        retrieved_chunks: List[Dict[str, Any]],
        expected_source: str,
        expected_page: str
) -> Dict[str, Any]:
    """
    Evaluates retrieved chunks by verifying BOTH filename matching and
    page number matching.
    """
    hit = 0.0
    mrr = 0.0
    target_clean = os.path.basename(expected_source).strip().lower()
    expected_pages = parse_expected_pages(expected_page)

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        retrieved_file = extract_source_name(chunk)
        retrieved_page = extract_page_number(chunk)

        # 1. Source filename validation
        file_match = bool(retrieved_file and (target_clean in retrieved_file or retrieved_file in target_clean))

        # 2. Page number validation
        if expected_pages and retrieved_page is not None:
            page_match = retrieved_page in expected_pages
        else:
            # Fallback to filename match if page info cannot be parsed
            page_match = True

        # Strict Dual Match requirement
        if file_match and page_match:
            hit = 1.0
            mrr = 1.0 / rank
            break

    return {"hit": hit, "mrr": mrr}


# =====================================================================
# 3. ABLATION EXPERIMENT RUNNER
# =====================================================================

class AblationTestRunner:

    def __init__(self, allowed_clearances: Optional[List[str]] = None):
        self.engine = RagEngine()
        self.allowed_clearances = allowed_clearances or ["public", "internal", "confidential"]

    def run_ablation_benchmark(self, top_k: int = 4) -> Dict[str, Any]:
        modes = ["dense_only", "sparse_only", "hybrid"]
        raw_results = {mode: [] for mode in modes}

        for qa in TEST_DATASET:
            for mode in modes:
                retrieved_chunks = self.engine.retrieve(
                    query=qa.question,
                    allowed_clearances=self.allowed_clearances,
                    mode=mode,
                    top_k=top_k
                )

                metrics = evaluate_retrieved_chunks(retrieved_chunks, qa.expected_source, qa.expected_page)
                top_source = extract_source_name(retrieved_chunks[0]) if retrieved_chunks else None
                top_page = extract_page_number(retrieved_chunks[0]) if retrieved_chunks else None

                result_entry = {
                    "question_id": qa.id,
                    "category": qa.category,
                    "question": qa.question,
                    "expected_source": qa.expected_source,
                    "expected_page": qa.expected_page,
                    "mode": mode,
                    "hit": metrics["hit"],
                    "mrr": metrics["mrr"],
                    "retrieved_count": len(retrieved_chunks),
                    "top_retrieved_source": top_source,
                    "top_retrieved_page": top_page
                }
                raw_results[mode].append(result_entry)

        summary = {}
        for mode in modes:
            hits = [r["hit"] for r in raw_results[mode]]
            mrrs = [r["mrr"] for r in raw_results[mode]]

            hit_rate = sum(hits) / len(hits) if hits else 0.0
            mean_mrr = sum(mrrs) / len(mrrs) if mrrs else 0.0

            summary[mode] = {
                f"Hit_Rate@{top_k}": hit_rate,
                f"MRR@{top_k}": mean_mrr,
                "total_queries": len(hits)
            }

        return {"summary": summary, "raw_results": raw_results}

    def render_ablation_report(self, results: Dict[str, Any], top_k: int = 4) -> None:
        summary = results["summary"]

        print("\n=================================================================================")
        print(f"   OBJECTIVE 6: ABLATION BENCHMARK EXPERIMENT REPORT (Top K = {top_k})")
        print("   (Evaluation strictly includes File Name and Page Number matching)")
        print("=================================================================================")
        print(f"{'Retrieval Mode':<20} | {f'Hit Rate @ {top_k}':<18} | {f'MRR @ {top_k}':<15} | {'Status':<10}")
        print("---------------------------------------------------------------------------------")

        for mode, metrics in summary.items():
            hit_str = f"{metrics[f'Hit_Rate@{top_k}'] * 100:.1f}%"
            mrr_str = f"{metrics[f'MRR@{top_k}']:.4f}"
            print(f"{mode:<20} | {hit_str:<18} | {mrr_str:<15} | Active")

        print("---------------------------------------------------------------------------------")

        dense_mrr = summary["dense_only"][f"MRR@{top_k}"]
        sparse_mrr = summary["sparse_only"][f"MRR@{top_k}"]
        hybrid_mrr = summary["hybrid"][f"MRR@{top_k}"]

        gain_over_dense = ((hybrid_mrr - dense_mrr) / dense_mrr * 100) if dense_mrr > 0 else 0.0
        gain_over_sparse = ((hybrid_mrr - sparse_mrr) / sparse_mrr * 100) if sparse_mrr > 0 else 0.0

        print("HYPOTHESIS EMPIRICAL VALIDATION (Based on MRR Gains):")
        print(f"  [H1] Hybrid vs. Dense-Only  : {gain_over_dense:+.2f}% MRR Relative Improvement")
        print(f"  [H2] Hybrid vs. Sparse-Only : {gain_over_sparse:+.2f}% MRR Relative Improvement")

        if hybrid_mrr >= dense_mrr and hybrid_mrr >= sparse_mrr:
            print("\n  ==> VERDICT: Hypotheses H1 & H2 VALIDATED. Hybrid RRF achieves optimal MRR retrieval.")
        else:
            print("\n  ==> VERDICT: Objective 6 completed. Review component performance breakdown above.")
        print("=================================================================================\n")


# =====================================================================
# 4. EXPORT HELPERS
# =====================================================================

def export_results_json(results: Dict[str, Any], filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[*] Results successfully exported to JSON: {os.path.abspath(filepath)}")


def export_results_csv(results: Dict[str, Any], filepath: str) -> None:
    raw_results = results.get("raw_results", {})
    all_rows = []
    for mode, rows in raw_results.items():
        all_rows.extend(rows)

    if not all_rows:
        return

    fieldnames = list(all_rows[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[*] Raw evaluation data exported to CSV: {os.path.abspath(filepath)}")


# =====================================================================
# 5. AUTOMATED UNITTEST TEST SUITE
# =====================================================================

class TestRagAblationFramework(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.runner = AblationTestRunner()
        cls.benchmark_output = cls.runner.run_ablation_benchmark(top_k=4)
        cls.summary = cls.benchmark_output["summary"]
        cls.runner.render_ablation_report(cls.benchmark_output, top_k=4)

    def test_dataset_completeness(self):
        self.assertEqual(len(TEST_DATASET), 35, "Dataset must contain exactly 35 questions.")

    def test_hypothesis_h1_hybrid_vs_dense(self):
        dense_mrr = self.summary["dense_only"]["MRR@4"]
        hybrid_mrr = self.summary["hybrid"]["MRR@4"]
        self.assertGreaterEqual(
            hybrid_mrr,
            dense_mrr,
            msg=f"H1 Failed: Hybrid MRR ({hybrid_mrr:.4f}) underperformed Dense-Only MRR ({dense_mrr:.4f})"
        )

    def test_hypothesis_h2_hybrid_vs_sparse(self):
        sparse_mrr = self.summary["sparse_only"]["MRR@4"]
        hybrid_mrr = self.summary["hybrid"]["MRR@4"]
        self.assertGreaterEqual(
            hybrid_mrr,
            sparse_mrr,
            msg=f"H2 Failed: Hybrid MRR ({hybrid_mrr:.4f}) underperformed Sparse-Only MRR ({sparse_mrr:.4f})"
        )


# =====================================================================
# 6. MAIN CLI ENTRYPOINT
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG Engine Ablation Benchmark Suite")
    parser.add_argument("--top-k", type=int, default=4, help="Top-K cutoff for retrieval evaluation")
    parser.add_argument("--export-json", type=str, default="ablation_results.json",
                        help="Path to save evaluation summary as JSON")
    parser.add_argument("--export-csv", type=str, default="ablation_results.csv",
                        help="Path to save raw evaluation data as CSV")
    parser.add_argument("--unittest", action="store_true", help="Run automated unittest assertion suite")

    args = parser.parse_args()

    if args.unittest:
        unittest.main(argv=['first-arg-is-ignored'], exit=False)
    else:
        runner = AblationTestRunner()
        results = runner.run_ablation_benchmark(top_k=args.top_k)
        runner.render_ablation_report(results, top_k=args.top_k)

        # Export benchmark outputs
        export_results_json(results, args.export_json)
        export_results_csv(results, args.export_csv)