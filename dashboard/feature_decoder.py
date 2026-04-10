from __future__ import annotations

import pandas as pd

MARITAL_STATUS = {
    1: "Single",
    2: "Married",
    3: "Widower",
    4: "Divorced",
    5: "Facto union",
    6: "Legally separated",
}

APPLICATION_MODE = {
    1: "1st phase - general contingent",
    2: "Ordinance No. 612/93",
    5: "1st phase - special contingent (Azores Island)",
    7: "Holders of other higher courses",
    10: "Ordinance No. 854-B/99",
    15: "International student (bachelor)",
    16: "1st phase - special contingent (Madeira Island)",
    17: "2nd phase - general contingent",
    18: "3rd phase - general contingent",
    26: "Ordinance No. 533-A/99, item b2) (Different Plan)",
    27: "Ordinance No. 533-A/99, item b3 (Other Institution)",
    39: "Over 23 years old",
    42: "Transfer",
    43: "Change of course",
    44: "Technological specialization diploma holders",
    51: "Change of institution/course",
    53: "Short cycle diploma holders",
    57: "Change of institution/course (International)",
}

COURSE = {
    33: "Biofuel Production Technologies",
    171: "Animation and Multimedia Design",
    8014: "Social Service (evening attendance)",
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
    9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication",
    9853: "Basic Education",
    9991: "Management (evening attendance)",
}

DAYTIME_EVENING = {1: "Daytime", 0: "Evening"}

PREVIOUS_QUALIFICATION = {
    1: "Secondary education",
    2: "Higher education - bachelor's degree",
    3: "Higher education - degree",
    4: "Higher education - master's",
    5: "Higher education - doctorate",
    6: "Frequency of higher education",
    9: "12th year of schooling - not completed",
    10: "11th year of schooling - not completed",
    12: "Other - 11th year of schooling",
    14: "10th year of schooling",
    15: "10th year of schooling - not completed",
    19: "Basic education 3rd cycle (9th/10th/11th year) or equiv.",
    38: "Basic education 2nd cycle (6th/7th/8th year) or equiv.",
    39: "Technological specialization course",
    40: "Higher education - degree (1st cycle)",
    42: "Professional higher technical course",
    43: "Higher education - master (2nd cycle)",
}

NACIONALITY = {
    1: "Portuguese",
    2: "German",
    6: "Spanish",
    11: "Italian",
    13: "Dutch",
    14: "English",
    17: "Lithuanian",
    21: "Angolan",
    22: "Cape Verdean",
    24: "Guinean",
    25: "Mozambican",
    26: "Santomean",
    32: "Turkish",
    41: "Brazilian",
    62: "Romanian",
    100: "Moldova (Republic of)",
    101: "Mexican",
    103: "Ukrainian",
    105: "Russian",
    108: "Cuban",
    109: "Colombian",
}

MOTHER_QUALIFICATION = {
    1: "Secondary Education - 12th Year of Schooling or Eq.",
    2: "Higher Education - Bachelor's Degree",
    3: "Higher Education - Degree",
    4: "Higher Education - Master's",
    5: "Higher Education - Doctorate",
    6: "Frequency of Higher Education",
    9: "12th Year of Schooling - Not Completed",
    10: "11th Year of Schooling - Not Completed",
    11: "7th Year (Old)",
    12: "Other - 11th Year of Schooling",
    14: "10th Year of Schooling",
    18: "General commerce course",
    19: "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.",
    22: "Technical-professional course",
    26: "7th year of schooling",
    27: "2nd cycle of the general high school course",
    29: "9th Year of Schooling - Not Completed",
    30: "8th year of schooling",
    34: "Unknown",
    35: "Can't read or write",
    36: "Can read without having a 4th year of schooling",
    37: "Basic education 1st cycle (4th/5th year) or equiv.",
    38: "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.",
    39: "Technological specialization course",
    40: "Higher education - degree (1st cycle)",
    41: "Specialized higher studies course",
    42: "Professional higher technical course",
    43: "Higher Education - Master (2nd cycle)",
    44: "Higher Education - Doctorate (3rd cycle)",
}

FATHER_QUALIFICATION = {
    1: "Secondary Education - 12th Year of Schooling or Eq.",
    2: "Higher Education - Bachelor's Degree",
    3: "Higher Education - Degree",
    4: "Higher Education - Master's",
    5: "Higher Education - Doctorate",
    6: "Frequency of Higher Education",
    9: "12th Year of Schooling - Not Completed",
    10: "11th Year of Schooling - Not Completed",
    11: "7th Year (Old)",
    12: "Other - 11th Year of Schooling",
    13: "2nd year complementary high school course",
    14: "10th Year of Schooling",
    18: "General commerce course",
    19: "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.",
    20: "Complementary High School Course",
    22: "Technical-professional course",
    25: "Complementary High School Course - not concluded",
    26: "7th year of schooling",
    27: "2nd cycle of the general high school course",
    29: "9th Year of Schooling - Not Completed",
    30: "8th year of schooling",
    31: "General Course of Administration and Commerce",
    33: "Supplementary Accounting and Administration",
    34: "Unknown",
    35: "Can't read or write",
    36: "Can read without having a 4th year of schooling",
    37: "Basic education 1st cycle (4th/5th year) or equiv.",
    38: "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.",
    39: "Technological specialization course",
    40: "Higher education - degree (1st cycle)",
    41: "Specialized higher studies course",
    42: "Professional higher technical course",
    43: "Higher Education - Master (2nd cycle)",
    44: "Higher Education - Doctorate (3rd cycle)",
}

MOTHER_OCCUPATION = {
    0: "Student",
    1: "Legislative/executive directors and managers",
    2: "Specialists in intellectual and scientific activities",
    3: "Intermediate level technicians and professions",
    4: "Administrative staff",
    5: "Personal services, security and sellers",
    6: "Farmers and skilled workers in agriculture/fisheries/forestry",
    7: "Skilled workers in industry/construction/crafts",
    8: "Machine operators and assembly workers",
    9: "Unskilled workers",
    10: "Armed forces professions",
    90: "Other situation",
    99: "Blank",
    122: "Health professionals",
    123: "Teachers",
    125: "ICT specialists",
    131: "Science and engineering technicians",
    132: "Health technicians",
    134: "Legal/social/sports/cultural technicians",
    141: "Office workers and data processing operators",
    143: "Accounting/financial/registry operators",
    144: "Other administrative support staff",
    151: "Personal service workers",
    152: "Sellers",
    153: "Personal care workers",
    171: "Skilled construction workers (except electricians)",
    173: "Precision instrument/jewelry/artisan workers",
    175: "Food/wood/clothing and related workers",
    191: "Cleaning workers",
    192: "Unskilled agriculture/fisheries/forestry workers",
    193: "Unskilled extractive/construction/manufacturing/transport workers",
    194: "Meal preparation assistants",
}

FATHER_OCCUPATION = {
    0: "Student",
    1: "Legislative/executive directors and managers",
    2: "Specialists in intellectual and scientific activities",
    3: "Intermediate level technicians and professions",
    4: "Administrative staff",
    5: "Personal services, security and sellers",
    6: "Farmers and skilled workers in agriculture/fisheries/forestry",
    7: "Skilled workers in industry/construction/crafts",
    8: "Machine operators and assembly workers",
    9: "Unskilled workers",
    10: "Armed forces professions",
    90: "Other situation",
    99: "Blank",
    101: "Armed forces officers",
    102: "Armed forces sergeants",
    103: "Other armed forces personnel",
    112: "Directors of administrative and commercial services",
    114: "Hotel/catering/trade and other services directors",
    121: "Physical sciences/mathematics/engineering specialists",
    122: "Health professionals",
    123: "Teachers",
    124: "Finance/accounting/administration/public relations specialists",
    131: "Science and engineering technicians",
    132: "Health technicians",
    134: "Legal/social/sports/cultural technicians",
    135: "ICT technicians",
    141: "Office workers and data processing operators",
    143: "Accounting/financial/registry operators",
    144: "Other administrative support staff",
    151: "Personal service workers",
    152: "Sellers",
    153: "Personal care workers",
    154: "Protection and security personnel",
    161: "Market-oriented farmers and skilled agriculture workers",
    163: "Subsistence farmers/fishermen/hunters/gatherers",
    171: "Skilled construction workers (except electricians)",
    172: "Skilled metallurgy/metalworking workers",
    174: "Skilled electricity and electronics workers",
    175: "Food/wood/clothing and related workers",
    181: "Fixed plant and machine operators",
    182: "Assembly workers",
    183: "Vehicle drivers and mobile equipment operators",
    192: "Unskilled agriculture/fisheries/forestry workers",
    193: "Unskilled extractive/construction/manufacturing/transport workers",
    194: "Meal preparation assistants",
    195: "Street vendors and street service providers",
}

YES_NO = {1: "Yes", 0: "No"}
GENDER = {1: "Male", 0: "Female"}

FEATURE_VALUE_MAPS = {
    "Marital status": MARITAL_STATUS,
    "Marital Status": MARITAL_STATUS,
    "Application mode": APPLICATION_MODE,
    "Course": COURSE,
    "Daytime/evening attendance": DAYTIME_EVENING,
    "Previous qualification": PREVIOUS_QUALIFICATION,
    "Nacionality": NACIONALITY,
    "Mother's qualification": MOTHER_QUALIFICATION,
    "Father's qualification": FATHER_QUALIFICATION,
    "Mother's occupation": MOTHER_OCCUPATION,
    "Father's occupation": FATHER_OCCUPATION,
    "Displaced": YES_NO,
    "Educational special needs": YES_NO,
    "Debtor": YES_NO,
    "Tuition fees up to date": YES_NO,
    "Gender": GENDER,
    "Scholarship holder": YES_NO,
    "International": YES_NO,
}


def _normalize_code(value):
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return value
        try:
            parsed = float(value)
            if parsed.is_integer():
                return int(parsed)
            return parsed
        except ValueError:
            return value

    if isinstance(value, float) and value.is_integer():
        return int(value)

    return value


def decode_feature_value(column: str, value):
    if pd.isna(value):
        return value

    mapping = FEATURE_VALUE_MAPS.get(column)
    if mapping is None:
        return value

    key = _normalize_code(value)
    return mapping.get(key, value)


def decode_dataframe_features(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    decoded = df.copy()
    cols = columns if columns is not None else decoded.columns.tolist()

    for col in cols:
        if col in decoded.columns and col in FEATURE_VALUE_MAPS:
            decoded[col] = decoded[col].apply(lambda value: decode_feature_value(col, value))

    return decoded
