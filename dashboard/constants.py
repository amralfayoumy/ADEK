import importlib.util

HAS_STATSMODELS = importlib.util.find_spec("statsmodels") is not None

PAGE_OPTIONS = [
    ":material/insights: Overview",
    ":material/public: Macro-Economic",
    ":material/balance: University Comparison",
    ":material/account_balance: University Deep Dive",
    ":material/school: College / Program Deep Dive",
    ":material/groups: Emirati vs Expats",
    ":material/flight_takeoff: Students Abroad",
    ":material/warning: At-Risk Students",
    ":material/manage_search: Student Deep-Dive",
    ":material/psychology: Predict New Student",
    ":material/monitoring: Model Management",
]
PAGES = PAGE_OPTIONS

OUTCOME_LABEL_MAP = {
    "Dropout": "Dropout",
    "Enrolled": "Pending",
    "Graduate": "Graduate",
}
OUTCOME_RAW_ORDER = ["Dropout", "Enrolled", "Graduate"]
OUTCOME_DISPLAY_ORDER = [OUTCOME_LABEL_MAP[c] for c in OUTCOME_RAW_ORDER]

COLOR_MAP = {"Dropout": "#f87171", "Pending": "#fbbf24", "Graduate": "#34d399"}
RISK_MAP = {"High": "#f87171", "Medium": "#fbbf24", "Low": "#34d399"}
STYPE_MAP = {"Emirati": "#60a5fa", "Expat": "#a78bfa", "Abroad": "#34d399"}
UNI_COLORS = ["#60a5fa", "#a78bfa", "#34d399", "#fbbf24", "#f87171", "#fb923c"]

COURSE_MAP = {
    33: "Biofuel Production Technologies",
    171: "Animation & Multimedia Design",
    8014: "Social Service (evening)",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equinculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising & Marketing Management",
    9773: "Journalism & Communication",
    9853: "Basic Education",
    9991: "Management (evening)",
}


def display_outcome(label):
    return OUTCOME_LABEL_MAP.get(label, label)
