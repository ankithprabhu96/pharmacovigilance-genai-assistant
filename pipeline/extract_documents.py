import json
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document


# DATA_DIR = Path("data")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

class DocumentExtractor:

    def __init__(self):
        self.documents = []

    ####################################################################
    # Pharmacovigilance Cases
    ####################################################################
    def load_cases(self):

        df = pd.read_csv(DATA_DIR / "pharmacovigilance_cases.csv")

        for _, row in df.iterrows():

            text = f"""
PHARMACOVIGILANCE CASE

Case ID: {row['case_id']}

Patient ID: {row['patient_id']}

Drug Name: {row['drug_name']}
Drug Code: {row['drug_code']}

Adverse Event:
{row['adverse_event']}

Severity:
{row['severity']}

Serious:
{row['serious']}

Priority Score:
{row['priority_score']}

Report Date:
{row['report_date']}

Reporter Type:
{row['reporter_type']}

Narrative:
{row['narrative']}
"""

            self.documents.append(
                Document(
                    page_content=text.strip(),
                    metadata={
                        "source": "pharmacovigilance_cases.csv",
                        "case_id": row["case_id"],
                        "patient_id": row["patient_id"],
                        "drug_name": row["drug_name"],
                    },
                )
            )

    ####################################################################
    # Medical Review Notes
    ####################################################################
    def load_medical_reviews(self):

        df = pd.read_csv(DATA_DIR / "case_medical_review_notes.csv")

        for _, row in df.iterrows():

            text = f"""
MEDICAL CASE REVIEW

Review ID:
{row['review_id']}

Case ID:
{row['case_id']}

Reviewer Provider ID:
{row['reviewer_provider_id']}

Review Date:
{row['review_date']}

Seriousness Confirmed:
{row['seriousness_confirmed']}

Expectedness:
{row['expectedness']}

Listedness:
{row['listedness']}

Naranjo Score:
{row['naranjo_score']}

WHO UMC Category:
{row['who_umc_category']}

Review Notes:
{row['review_notes']}

Time To Review:
{row['time_to_review_hours']} hours

QA Flag:
{row['qa_flag']}

Escalated To Physician:
{row['escalated_to_physician']}

Labeling Impact:
{row['labeling_impact']}

Aggregate Report Inclusion:
{row['aggregate_report_inclusion']}
"""

            self.documents.append(
                Document(
                    page_content=text.strip(),
                    metadata={
                        "source": "case_medical_review_notes.csv",
                        "case_id": row["case_id"],
                        "review_id": row["review_id"],
                    },
                )
            )

    ####################################################################
    # MedWatch Regulatory Reports
    ####################################################################
    def load_regulatory_reports(self):

        df = pd.read_csv(DATA_DIR / "medwatch_regulatory_reports.csv")

        for _, row in df.iterrows():

            text = f"""
FDA MEDWATCH REPORT

Report ID:
{row['report_id']}

Case ID:
{row['case_id']}

Patient ID:
{row['patient_id']}

FDA Report Type:
{row['fda_report_type']}

Submission Date:
{row['submission_date']}

Reporter Country:
{row['reporter_country']}

Reporter Qualification:
{row['reporter_qualification']}

Event Outcome:
{row['event_outcome']}

Causality Assessment:
{row['causality_assessment']}

Dechallenge Performed:
{row['dechallenge_performed']}

Rechallenge Performed:
{row['rechallenge_performed']}

Seriousness Criteria:
{row['seriousness_criteria']}

Narrative Summary:
{row['narrative_summary']}

Primary SOC:
{row['primary_soc']}

Regulatory Deadline:
{row['regulatory_deadline']}

Submission Status:
{row['submission_status']}

Acknowledgment Date:
{row['acknowledgment_date']}

Follow Up Required:
{row['follow_up_required']}

Medical Review Priority:
{row['medical_review_priority']}
"""

            self.documents.append(
                Document(
                    page_content=text.strip(),
                    metadata={
                        "source": "medwatch_regulatory_reports.csv",
                        "case_id": row["case_id"],
                        "report_id": row["report_id"],
                    },
                )
            )

    ####################################################################
    # Patients
    ####################################################################
    def load_patients(self):

        df = pd.read_csv(DATA_DIR / "patients.csv")

        for _, row in df.iterrows():

            text = f"""
PATIENT INFORMATION

Patient ID:
{row['patient_id']}

Name:
{row['first_name']} {row['last_name']}

Age:
{row['age']}

Gender:
{row['gender']}

Ethnicity:
{row['ethnicity']}

BMI:
{row['bmi']}

Smoker:
{row['smoker']}

Insurance:
{row['insurance_type']}

Number of Chronic Conditions:
{row['num_chronic_conditions']}

Diabetes:
{row['has_diabetes']}
"""

            self.documents.append(
                Document(
                    page_content=text.strip(),
                    metadata={
                        "source": "patients.csv",
                        "patient_id": row["patient_id"],
                    },
                )
            )

    ####################################################################
    # Prescriptions
    ####################################################################
    def load_prescriptions(self):

        df = pd.read_csv(DATA_DIR / "prescriptions.csv")

        for _, row in df.iterrows():

            text = f"""
PRESCRIPTION

Prescription ID:
{row['prescription_id']}

Patient ID:
{row['patient_id']}

Provider ID:
{row['provider_id']}

Drug Name:
{row['drug_name']}

Drug Code:
{row['drug_code']}

Dosage:
{row['dosage']}

Start Date:
{row['start_date']}

End Date:
{row['end_date']}

Refills:
{row['refills']}

Pharmacy City:
{row['pharmacy_city']}
"""

            self.documents.append(
                Document(
                    page_content=text.strip(),
                    metadata={
                        "source": "prescriptions.csv",
                        "patient_id": row["patient_id"],
                        "drug_name": row["drug_name"],
                    },
                )
            )

    ####################################################################
    # Providers
    ####################################################################
    def load_providers(self):

        df = pd.read_csv(DATA_DIR / "providers.csv")

        for _, row in df.iterrows():

            text = f"""
PROVIDER

Provider ID:
{row['provider_id']}

Provider Name:
{row['name']}

Specialty:
{row['specialty']}

Department:
{row['department']}

Hospital ID:
{row['hospital_id']}

Years of Experience:
{row['years_experience']}

Email:
{row['email']}
"""

            self.documents.append(
                Document(
                    page_content=text.strip(),
                    metadata={
                        "source": "providers.csv",
                        "provider_id": row["provider_id"],
                    },
                )
            )

    ####################################################################
    # Medical Knowledge Base
    ####################################################################
    def load_knowledge_base(self):

        with open(DATA_DIR / "medical_knowledge_base.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:

            text = f"""
MEDICAL KNOWLEDGE BASE

Question:
{item['question']}

Answer:
{item['answer']}

Category:
{item['category']}

Source:
{item['source']}
"""

            self.documents.append(
                Document(
                    page_content=text.strip(),
                    metadata={
                        "source": "medical_knowledge_base.json",
                        "kb_id": item["id"],
                        "category": item["category"],
                    },
                )
            )

    ####################################################################
    # Load Everything
    ####################################################################
    def load_all_documents(self):

        self.load_cases()
        self.load_medical_reviews()
        self.load_regulatory_reports()
        self.load_patients()
        self.load_prescriptions()
        self.load_providers()
        self.load_knowledge_base()

        return self.documents


if __name__ == "__main__":

    extractor = DocumentExtractor()

    docs = extractor.load_all_documents()

    print(f"\nTotal Documents: {len(docs)}\n")

    print(docs[0])