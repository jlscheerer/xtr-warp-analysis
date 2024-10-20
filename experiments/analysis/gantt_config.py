from analysis.gantt_plot import GanttColor

LOTTE_K_VALUES = [10, 100, 1000]
XTR_TOKEN_TOP_K_VALUES = [1_000, 40_000]

PLAID_LATENCY_GROUPS = {
    "Query Encoding": ["Query Encoding"],
    "Candidate Generation": ["Candidate Generation"],
    "Filtering": ["Filtering"],
    "Decompression": ["Decompress Residuals"],
    "Scoring": ["Scoring", "Sorting"]
}

PLAID_COLOR_MAP = {
    "Query Encoding": GanttColor.BLUE,
    "Candidate Generation": GanttColor.ORANGE,
    "Filtering": GanttColor.RED,
    "Decompression": GanttColor.GREEN,
    "Scoring": GanttColor.PURPLE,
}

XTR_LATENCY_GROUPS = {
    "Query Encoding": ["Query Encoding"],
    "Token Retrieval": ["search_batched", "enumerate_scores"],
    "Scoring": ["Estimate Missing Similarity", "get_did2scores", "add_ems", "get_final_score", "sort_scores"],
}

XTR_COLOR_MAP = {
    "Query Encoding": GanttColor.BLUE,
    "Token Retrieval": GanttColor.ORANGE,
    "Scoring": GanttColor.PURPLE,
}

WARP_LATENCY_GROUPS = {
    "Query Encoding": ["Query Encoding"],
    "Candidate Generation": ["Candidate Generation", "top-k Precompute"],
    "Decompression": ["Decompression"],
    "Scoring": ["Build Matrix"]
}

WARP_COLOR_MAP = {
    "Query Encoding": GanttColor.BLUE,
    "Candidate Generation": GanttColor.ORANGE,
    "Filtering": GanttColor.RED,
    "Decompression": GanttColor.GREEN,
    "Scoring": GanttColor.PURPLE,
}