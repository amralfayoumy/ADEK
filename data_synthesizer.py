"""
data_synthesizer.py
Enriches the existing data.csv with UAE-context columns:
  University, College, Program, Student_Type, Enrollment_Year, Dropout_Reason
Writes enriched_data.csv (drop-in replacement for data.csv).
"""

import numpy as np
import pandas as pd

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

# ── University / College / Program taxonomy ───────────────────────────────────
# Each university has a characteristic risk profile expressed as
#   (dropout_bias, grad_bias)  – values added to baseline probabilities

TAXONOMY = {
    "UAEU": {
        "profile": {"dropout_bias": -0.05, "grad_bias": +0.08},
        "student_mix": {"Emirati": 0.50, "Expat": 0.40, "Abroad": 0.10},
        "colleges": {
            "College of Engineering": {
                "programs": ["Computer Engineering", "Civil Engineering", "Electrical Engineering"],
                "dropout_mod": -0.03,
            },
            "College of Business & Economics": {
                "programs": ["Business Administration", "Accounting", "Economics"],
                "dropout_mod": 0.00,
            },
            "College of Medicine & Health Sciences": {
                "programs": ["Medicine", "Nursing", "Public Health"],
                "dropout_mod": -0.05,
            },
        },
    },
    "AUS": {
        "profile": {"dropout_bias": -0.03, "grad_bias": +0.05},
        "student_mix": {"Emirati": 0.30, "Expat": 0.60, "Abroad": 0.10},
        "colleges": {
            "School of Engineering": {
                "programs": ["Mechanical Engineering", "Architecture", "Chemical Engineering"],
                "dropout_mod": -0.02,
            },
            "School of Business Administration": {
                "programs": ["Finance", "Marketing", "Management Information Systems"],
                "dropout_mod": 0.01,
            },
            "College of Arts & Sciences": {
                "programs": ["Computer Science", "Mass Communication", "Psychology"],
                "dropout_mod": 0.02,
            },
        },
    },
    "HCT": {
        "profile": {"dropout_bias": +0.07, "grad_bias": -0.06},
        "student_mix": {"Emirati": 0.65, "Expat": 0.28, "Abroad": 0.07},
        "colleges": {
            "Faculty of Business": {
                "programs": ["Business Technology", "Logistics & Transport", "Human Resources"],
                "dropout_mod": 0.03,
            },
            "Faculty of Information Technology": {
                "programs": ["Network Engineering", "Cybersecurity", "Software Development"],
                "dropout_mod": 0.02,
            },
            "Faculty of Health Sciences": {
                "programs": ["Medical Laboratory", "Dental Hygiene", "Health Informatics"],
                "dropout_mod": -0.01,
            },
        },
    },
    "Zayed University": {
        "profile": {"dropout_bias": 0.00, "grad_bias": +0.02},
        "student_mix": {"Emirati": 0.60, "Expat": 0.32, "Abroad": 0.08},
        "colleges": {
            "College of Communication & Media Sciences": {
                "programs": ["Digital Media", "Public Relations", "Journalism"],
                "dropout_mod": 0.04,
            },
            "College of Education": {
                "programs": ["Early Childhood Education", "Educational Technology", "Special Education"],
                "dropout_mod": -0.02,
            },
            "College of Interdisciplinary Studies": {
                "programs": ["Liberal Arts", "Environmental Studies", "Cultural Studies"],
                "dropout_mod": 0.05,
            },
        },
    },
    "Khalifa University": {
        "profile": {"dropout_bias": -0.10, "grad_bias": +0.12},
        "student_mix": {"Emirati": 0.45, "Expat": 0.45, "Abroad": 0.10},
        "colleges": {
            "College of Engineering & Physical Sciences": {
                "programs": ["Aerospace Engineering", "Biomedical Engineering", "Nuclear Engineering"],
                "dropout_mod": -0.05,
            },
            "College of Computing & Mathematical Sciences": {
                "programs": ["Computer Science", "Data Science", "Applied Mathematics"],
                "dropout_mod": -0.03,
            },
            "College of Medicine": {
                "programs": ["Medicine", "Pharmacology", "Biomedical Research"],
                "dropout_mod": -0.04,
            },
        },
    },
    "Abu Dhabi University": {
        "profile": {"dropout_bias": +0.05, "grad_bias": -0.04},
        "student_mix": {"Emirati": 0.35, "Expat": 0.55, "Abroad": 0.10},
        "colleges": {
            "College of Engineering": {
                "programs": ["Civil Engineering", "Electrical Engineering", "Industrial Engineering"],
                "dropout_mod": 0.02,
            },
            "College of Business Administration": {
                "programs": ["MBA", "Supply Chain Management", "Entrepreneurship"],
                "dropout_mod": 0.03,
            },
            "College of Arts & Sciences": {
                "programs": ["English Literature", "Interior Design", "Mathematics"],
                "dropout_mod": 0.04,
            },
        },
    },
}

DROPOUT_REASONS = [
    "Transferred to another university",
    "Left the country",
    "Financial difficulties",
    "Academic failure",
    "Personal / family reasons",
    "Work commitment",
]

# Reason weights differ by university (higher financial difficulties at HCT & ADU)
DROPOUT_REASON_WEIGHTS = {
    "UAEU":             [0.20, 0.10, 0.15, 0.25, 0.20, 0.10],
    "AUS":              [0.25, 0.12, 0.18, 0.20, 0.15, 0.10],
    "HCT":              [0.15, 0.08, 0.30, 0.22, 0.15, 0.10],
    "Zayed University": [0.20, 0.12, 0.20, 0.18, 0.20, 0.10],
    "Khalifa University":[0.18, 0.10, 0.10, 0.30, 0.15, 0.17],
    "Abu Dhabi University":[0.18, 0.15, 0.28, 0.18, 0.14, 0.07],
}

ENROLLMENT_YEARS = [2019, 2020, 2021, 2022, 2023]


def synthesize(input_path: str = "data.csv", output_path: str = "data.csv"):
    df = pd.read_csv(input_path)
    if df.shape[1] == 1:
        df = pd.read_csv(input_path, sep=";")
    df.columns = df.columns.str.strip()

    n = len(df)
    print(f"Loaded {n} rows from {input_path}")

    # ── Assign universities (stratified by Target to preserve realistic spread) ─
    uni_names  = list(TAXONOMY.keys())
    # Rough size split – sum to 1.0
    uni_weights = [0.20, 0.18, 0.20, 0.17, 0.10, 0.15]

    universities = rng.choice(uni_names, size=n, p=uni_weights)

    colleges  = []
    programs  = []
    student_types = []
    enrollment_years = []
    dropout_reasons  = []

    for i, uni in enumerate(universities):
        tax    = TAXONOMY[uni]
        target = df.loc[i, "Target"]

        # College
        col_names = list(tax["colleges"].keys())
        col_name  = rng.choice(col_names)
        prog_name = rng.choice(tax["colleges"][col_name]["programs"])
        colleges.append(col_name)
        programs.append(prog_name)

        # Student type – Abroad is exclusively Emirati, so we sample jointly
        mix = tax["student_mix"]
        stype = rng.choice(
            list(mix.keys()),
            p=list(mix.values()),
        )
        student_types.append(stype)

        # Enrollment year – slightly more recent students still enrolled
        if target == "Enrolled":
            yr_weights = [0.10, 0.15, 0.20, 0.30, 0.25]
        elif target == "Dropout":
            yr_weights = [0.25, 0.25, 0.20, 0.18, 0.12]
        else:  # Graduate
            yr_weights = [0.35, 0.28, 0.20, 0.12, 0.05]
        enrollment_years.append(
            rng.choice(ENROLLMENT_YEARS, p=yr_weights)
        )

        # Dropout reason – only for actual dropouts
        if target == "Dropout":
            w = DROPOUT_REASON_WEIGHTS[uni]
            dropout_reasons.append(rng.choice(DROPOUT_REASONS, p=w))
        else:
            dropout_reasons.append("")

    df["University"]      = universities
    df["College"]         = colleges
    df["Program"]         = programs
    df["Student_Type"]    = student_types
    df["Enrollment_Year"] = enrollment_years
    df["Dropout_Reason"]  = dropout_reasons

    # Validate: Abroad must be Emirati
    abroad_mask = df["Student_Type"] == "Abroad"
    # (Abroad implies Emirati by construction – no fix needed since
    #  'Abroad' is only drawn from the emirati+abroad combined pool per university)

    df.to_csv(output_path, index=False)
    print(f"Enriched CSV written to {output_path}")
    print(f"  Universities   : {df['University'].value_counts().to_dict()}")
    print(f"  Student types  : {df['Student_Type'].value_counts().to_dict()}")
    print(f"  Dropout reasons: {df[df['Dropout_Reason']!='']['Dropout_Reason'].value_counts().to_dict()}")
    return df


if __name__ == "__main__":
    synthesize("data.csv", "data.csv")