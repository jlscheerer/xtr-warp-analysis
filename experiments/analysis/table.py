from analysis import PRETTY_NAMES, metric_round

import re
import json
from dataclasses import dataclass, field

@dataclass
class Footnote:
    symbol: str
    note: str

@dataclass
class TableEntry:
    text: str
    underline: bool = False
    bold: bool = False

    def __repr__(self):
        text = self.text
        if self.bold:
            tt = text.split(" ")
            remains = " ".join(tt[1:])
            text = f"\\textbf{{{tt[0]}}} {remains}"
        if self.underline:
            tt = text.split(" ")
            remains = " ".join(tt[1:])
            text = f"\\underline{{{tt[0]}}} {remains}"
        return text

@dataclass
class TableRow:
    data: list[TableEntry]
    gray_background: bool = False
    gray_text: bool = False

    def values(self):
        values = []
        for entry in self.data:
            if isinstance(entry, str):
                text = entry
            else: text = entry.text
            try:
                value = text.split(" ")[0]
                values.append(float(value))
            except: values.append(None)
        return values

    def __getitem__(self, key):
        if isinstance(self.data[key], str):
            self.data[key] = TableEntry(self.data[key])
        return self.data[key]


@dataclass
class TableGroup:
    rows: list[TableRow]
    gray_text: bool = False

    def bold_max(self):
        values = []
        for row in self.rows:
            values.append(row.values())
        for col_id in range(len(values[0])):
            max_v = values[0][col_id]
            if max_v is None:
                continue
            for row_id in range(len(values)):
                max_v = max(max_v, values[row_id][col_id])
            for row_id in range(len(values)):
                if values[row_id][col_id] == max_v:
                    self.rows[row_id][col_id].bold = True
        return self

@dataclass
class Table:
    columns: list[str]
    groups: list[TableGroup]
    footer: list[Footnote] = field(default_factory=list)
    caption: str = ""
    label: str = ""

    def save(self, filename):
        with open(f"output/{filename}", "w") as file:
            file.write(f"{self}")

    def __repr__(self):
        header = Table._make_table_header(self.columns)
        table = Table._make_table_groups(self)
        footer = Table._make_table_footer(self.footer, self.caption, self.label)
        return f"{header}\n{table}\n{footer}"

    def underline_max(self):
        max_v = [None for _ in range(len(self.groups[0].rows[0].data))]
        for group in self.groups:
            values = []
            for row in group.rows:
                for col_id, value in enumerate(row.values()):
                    if max_v[col_id] is None or max_v[col_id] < value:
                        max_v[col_id] = value

        for group in self.groups:
            for row in group.rows:
                for col_id, value in enumerate(row.values()):
                    if max_v[col_id] is not None and max_v[col_id] == value:
                        row[col_id].underline = True
        return self

    @staticmethod
    def _make_table_header(columns):
        columns_c = "c" * len(columns)
        columns_n = " & ".join(columns)
        return f"""\\begin{{table}}[h]
\\centering
\\resizebox{{\\columnwidth}}{{!}}{{
\\begin{{tabular}}{{@{{}}l|{columns_c}!{{\\vrule width 1pt}}c@{{}}}}
\\toprule
& {columns_n} & Avg. \\\\
\\midrule
        """

    @staticmethod
    def _make_table_footer(footer: list[Footnote], caption, label):
        stringified = " \\quad ".join([f"$\\{note.symbol}$: {note.note}" for note in footer])
        if len(footer) != 0:
            footer = f"""
    \\begin{{tablenotes}}
    \\scriptsize
        \\item {stringified}
    \\end{{tablenotes}}
        """
        else:
            footer = ""

        return f"""\\bottomrule
    \\end{{tabular}}
    }}{footer}   
    \\caption{{{caption}}}
    \\label{{{label}}}
\\end{{table}}"""

    @staticmethod
    def _make_table_entry(entry: TableEntry, gray_text: bool):
        if gray_text:
            return f"{{\\color{{gray}} {entry}}}"
        return str(entry)

    @staticmethod
    def _make_table_row(group: TableGroup, row: TableRow):
        prefix = ""
        if row.gray_background:
            prefix = "\\rowcolor[gray]{0.90}"
        data = " & ".join([Table._make_table_entry(entry, gray_text=row.gray_text or group.gray_text) for entry in row.data])
        return f"{prefix}{data}\\\\"

    @staticmethod
    def _make_table_group(group: TableGroup):
        return "\n".join([
            Table._make_table_row(group, row) for row in group.rows
        ])

    @staticmethod
    def _make_table_groups(table):
        return " \\midrule\n".join([
            Table._make_table_group(group) for group in table.groups
        ])

@dataclass
class Template:
    title: str
    caption: str
    
    groups: list[list]

    keys: dict[str, str]
    data: dict[str, float]

    notes: dict[str, dict]

    def __init__(self, filename):
        with open(f"templates/{filename}.json", "r") as file:
            template = json.loads(file.read())
        self.data = template["data"]
        table = template["table"]

        self.groups = table["groups"]
        self.title = table["title"]
        self.caption = table["caption"]
        self.keys = table["keys"]
        if "notes" in table:
            self.notes = table["notes"]
        else: self.notes = dict()

        rev_prety_names = {y: x for x, y in PRETTY_NAMES.items()}

        self.reverse_lookup = dict()
        for key, value in self.keys.items():
            if value in rev_prety_names:
                rev_name = rev_prety_names[value]
                self.reverse_lookup[rev_name] = key

    def footnotes(self, model):
        symbols = []
        for note in self.notes.values():
            if model in note["models"]:
                symbols.append(note["symbol"])
        return symbols

    def get_score(self, model, dataset):
        return self.data[model][self.reverse_lookup[dataset]]

def format_model_name(text):
    matches = list(re.findall("\\_([a-zA-Z0-9]+)", text))
    if len(matches) == 0:
        return text
    for match in matches:
        text = text.replace(f"_{match}", f"$_\\text{{{match}}}$")
    return text

def compute_row_avg(row: list[TableEntry]):
    values = []
    for entry in row:
        try:
            if isinstance(entry, str):
                text = entry
            else: text = entry.text
            text = text.split(" ")[0]
            values.append(float(text))
        except: continue
    avg = sum(values) / len(values)
    return f"{metric_round(avg, fact=1)}"