"""Builds the labelled dataset.

Malicious: SQL injection payloads from PayloadsAllTheThings (MIT licence,
github.com/swisskyrepo/PayloadsAllTheThings), deduplicated, with at most PER_FILE taken
from each list so the huge UNION SELECT list does not dominate. Each payload keeps the
name of its list ("family"), so the model can be tested on families it never saw.

Benign: (1) generated form input of the kind a web application receives -- names, emails,
searches, addresses, comments, numbers -- including cases a naive filter gets wrong:
apostrophes in names (O'Brien), SQL keywords in plain English ("drop me a line", "union
bank"); and (2) short fragments of real English prose from public-domain books (Project
Gutenberg), which are full of apostrophes, quotes, semicolons, dashes and the words "or",
"and" and "select". Without the prose, the model learns that anything unlike the generated
templates is an attack (it flagged 38 of 40 hand-written benign inputs).
"""
import random
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

SOURCE = ("https://raw.githubusercontent.com/swisskyrepo/PayloadsAllTheThings/master/"
          "SQL%20Injection/Intruder/")
FILES = ["Auth_Bypass.txt", "Auth_Bypass2.txt", "FUZZDB_MSSQL-WHERE_Time.txt",
         "FUZZDB_MSSQL.txt", "FUZZDB_MSSQL_Enumeration.txt", "FUZZDB_MYSQL.txt",
         "FUZZDB_MySQL-WHERE_Time.txt", "FUZZDB_MySQL_ReadLocalFiles.txt", "FUZZDB_Oracle.txt",
         "FUZZDB_Postgres_Enumeration.txt", "Generic_ErrorBased.txt", "Generic_Fuzz.txt",
         "Generic_TimeBased.txt", "Generic_UnionSelect.txt", "SQL-Injection",
         "SQLi_Polyglots.txt", "payloads-sql-blind-MSSQL-INSERT",
         "payloads-sql-blind-MSSQL-WHERE", "payloads-sql-blind-MySQL-INSERT",
         "payloads-sql-blind-MySQL-ORDER_BY", "payloads-sql-blind-MySQL-WHERE"]
PER_FILE = 150
# Public-domain books (Project Gutenberg): Pride and Prejudice, Alice in Wonderland,
# The Adventures of Sherlock Holmes, A Tale of Two Cities.
BOOKS = [1342, 11, 1661, 98]
BOOK_URL = "https://www.gutenberg.org/cache/epub/{n}/pg{n}.txt"


def fetch(raw: Path) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = raw / name
        if not target.exists():
            with urllib.request.urlopen(SOURCE + urllib.parse.quote(name)) as r:
                target.write_bytes(r.read())


def fetch_books(raw: Path) -> list[str]:
    texts = []
    for n in BOOKS:
        target = raw / f"book-{n}.txt"
        if not target.exists():
            with urllib.request.urlopen(BOOK_URL.format(n=n)) as r:
                target.write_bytes(r.read())
        text = target.read_text(encoding="utf-8-sig", errors="replace")
        start = text.find("*** START OF")
        end = text.find("*** END OF")
        texts.append(text[text.find("\n", start) + 1:end] if start >= 0 else text)
    return texts


def prose(texts: list[str], n: int, rng: random.Random) -> list[dict]:
    """Fragments of 1-14 words, like a search box or comment field would receive."""
    words = [w for t in texts for w in t.replace("\u2019", "'").replace("\u201c", '"')
             .replace("\u201d", '"').split()]
    out = set()
    while len(out) < n:
        i = rng.randrange(len(words) - 14)
        out.add(" ".join(words[i:i + rng.randint(1, 14)]))
    return [{"text": t, "label": 0, "family": "prose"} for t in sorted(out)]


def malicious(raw: Path, rng: random.Random) -> list[dict]:
    rows, seen = [], set()
    for name in FILES:
        lines = [x.strip() for x in (raw / name).read_text(errors="replace").splitlines()]
        lines = [x for x in dict.fromkeys(lines) if x and not x.startswith("#") and x not in seen]
        rng.shuffle(lines)
        for x in lines[:PER_FILE]:
            seen.add(x)
            rows.append({"text": x, "label": 1, "family": name.removesuffix(".txt")})
    return rows


FIRST = ["Aarav", "Priya", "Rahul", "Ananya", "John", "Mary", "Sean", "Fatima", "Wei",
         "Carlos", "Ishita", "Arjun", "Sara", "Kabir", "Meera", "Liam", "Zoe", "Omar"]
LAST = ["Sharma", "O'Brien", "D'Souza", "Patel", "O'Neil", "Smith", "Khan", "D'Angelo",
        "Iyer", "Garcia", "Nguyen", "Mehta", "O'Connor", "Reddy", "McDonald", "Karn"]
WORDS = ["laptop", "running shoes", "union bank", "select committee", "drop shipping",
         "order history", "table lamp", "update profile", "insert coin", "delete account",
         "where to eat", "best pizza", "rock 'n' roll", "kids' toys", "men's jacket",
         "or else", "and more", "null hypothesis", "1=1 meaning", "sleep tracker",
         "benchmark results", "script writing", "exec summary", "char dham yatra",
         "group by size", "having fun", "case study", "waitfor it", "join the club"]
SENTENCES = ["Can't log in to my account, please help.", "I'd like to drop my course.",
             "Please select a date and time; I'm free after 5 - 6 pm.",
             "Great product! Would buy again :)", "Is the union canteen open on Sunday?",
             "Order #4521 hasn't arrived yet -- any update?", "It's 50% off, isn't it?",
             "Drop me a line at the office.", "Where's the nearest ATM?",
             "My password reset link doesn't work.", "The 'Submit' button is greyed out.",
             "Delete my old address and insert the new one, please.",
             "Table for 4 at 8 pm, or 9 if that's full.", "He said \"see you tomorrow\".",
             "Select the best option; then click next.", "C++ vs. Java -- which is faster?"]
STREETS = ["MG Road", "Baker St.", "Rajpur Rd", "5th Ave", "Church St", "Park Lane"]


def benign(n: int, rng: random.Random) -> list[dict]:
    makers = [
        lambda: f"{rng.choice(FIRST)} {rng.choice(LAST)}",
        lambda: f"{rng.choice(FIRST).lower()}.{rng.choice(LAST).lower().replace(chr(39), '')}"
                f"{rng.randint(1, 99)}@{rng.choice(['gmail.com', 'upes.ac.in', 'yahoo.com'])}",
        lambda: rng.choice(WORDS),
        lambda: f"{rng.choice(WORDS)} under {rng.randint(5, 50) * 100}",
        lambda: rng.choice(SENTENCES),
        lambda: f"{rng.randint(1, 999)}, {rng.choice(STREETS)}, Dehradun - {rng.randint(248001, 248199)}",
        lambda: str(rng.randint(1, 100000)),
        lambda: f"+91 {rng.randint(70000, 99999)} {rng.randint(10000, 99999)}",
        lambda: f"{rng.randint(2020, 2027)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        lambda: rng.choice(["admin", "john_doe", "priya-s", "user2024", "k.arjun"]),
        lambda: f"{rng.choice(SENTENCES)} {rng.choice(SENTENCES)}",
    ]
    out = {rng.choice(makers)() for _ in range(n * 3)}
    return [{"text": t, "label": 0, "family": "benign"} for t in sorted(out)[:n]]


def build(raw: Path, seed: int = 7, n_benign: int = 2000, n_prose: int = 6000) -> pd.DataFrame:
    rng = random.Random(seed)
    fetch(raw)
    rows = malicious(raw, rng) + benign(n_benign, rng) + prose(fetch_books(raw), n_prose, rng)
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
