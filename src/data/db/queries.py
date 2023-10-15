CREATE_BUSINESS_DESC_DEF = """
    CREATE TABLE IF NOT EXISTS business_desc (
        ticker TEXT PRIMARY KEY,
        business TEXT NOT NULL
    );
"""

CREATE_GICS_PROB_DEF = """
    CREATE TABLE IF NOT EXISTS {} (
        ticker TEXT,
        classification TEXT,
        prob FLOAT
    );
"""

CREATE_GICS_MAPPING_DEF = """
    CREATE TABLE IF NOT EXISTS gics_mapping (
        ticker TEXT,
        sector TEXT,
        industry_group TEXT,
        industry TEXT,
        subindustry TEXT
    );
"""