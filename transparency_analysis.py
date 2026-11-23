"""
Primary transparency analysis: how often do open-access biomedical articles
DISCLOSE the use of generative AI, computed by us from the full text.

Runs on the Europe PMC REST API (free, no key) over open-access articles with
full text. For a random sample of each year it fetches the JATS full-text XML,
looks only in acknowledgements, methods, declaration and author-note sections,
and counts an article as "disclosing" when a usage verb sits next to a named
GenAI tool in the same paragraph, or when an explicit AI-disclosure section
names a tool. Negative statements ("no AI was used") are not counted.

On the same sample it measures the vocabulary signal of LLM help in the
ABSTRACT with the 291 rare excess style words of Kobak et al. (2025), and
estimates the excess over the pre-ChatGPT trend, so that the disclosure gap
(articles with estimated LLM help per article that declares it) is computed
as in He & Bu (2026).

Outputs
  transparency_by_year.csv   one row per year (rates with 95% Wilson CIs)
  disclosure_details.csv     one row per disclosing article: year, pmcid,
                             tools, location, statement words, AI-as-topic flag

Requires bibliometric_analysis.py in the same folder (for the Kobak word list).

HOW TO RUN
  pip install requests pandas numpy
  python transparency_analysis.py
Runtime: about one second per article. Test first with SAMPLE_PER_YEAR = 100.
"""

import math
import random
import re
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import xml.etree.ElementTree as ET

from bibliometric_analysis import RARE_WORDS, TOKEN   # Kobak et al. (2025)

# ---------------------------------------------------------------- config
YEARS = list(range(2020, 2027))     # 2020-2022 = pre-ChatGPT baseline
SAMPLE_PER_YEAR = 2000              # disclosure is rare (~0.1-3%): use >= 2000 for stable rates
ARTICLES_PER_DAY = 10               # stratified sample: random days, random articles per day
SEED = 42
SLEEP = 0.15
SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
FULLTEXT = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

RARE = set(RARE_WORDS)
SESSION = requests.Session()
TODAY = date.today()

# ---------------------------------------------------------------- patterns
# Product names that are also common words or personal names (Claude, Gemini,
# Bard, Llama, Mistral, Copilot) only count with a qualifier, to avoid
# acknowledgements such as "we thank Claude Martin for his help".
TOOLS = (
    r"chat\s?gpt|\bgpt-?(?:3\.5|4o?|4\.\d|5)\b|openai|"
    r"large language models?|\bllms?\b|"
    r"generative (?:ai|artificial intelligence)|\bgen-?ai\b|"
    r"claude(?:\.ai|\s*\d|\s*\(anthropic\)|\s+(?:sonnet|opus|haiku))|anthropic|"
    r"(?:google\s+)?gemini(?:\s*\d|\s+(?:pro|ultra|flash|advanced)|\s*\(google\))|google gemini|"
    r"google bard|bard\s*\(google\)|"
    r"(?:microsoft|github)\s+copilot|"
    r"deepseek|meta llama|llama[\s-]?\d|mistral(?:\s+ai|\s*\d|\s+large)|"
    r"perplexity(?:\.ai| ai)?|quillbot|grammarly(?: ?go)?|deepl(?: write)?"
)
USE_VERB = (
    r"used|using|use of|employ(?:ed|ing)?|utili[sz](?:ed|ing)|assisted|aided|"
    r"generated (?:with|using|by)|prepared (?:with|using)|"
    r"with the (?:help|assistance|aid|use|support) of|drafted (?:with|using)|"
    r"edited (?:with|using)|polished (?:with|using)|improved (?:with|using)|"
    r"proofread(?:ing)? (?:with|using)"
)
PAT_TOOL = re.compile(TOOLS, re.IGNORECASE)
PAT_USE = re.compile(
    rf"(?:{USE_VERB})[^.;]{{0,80}}?(?:{TOOLS})|(?:{TOOLS})[^.;]{{0,80}}?(?:{USE_VERB})",
    re.IGNORECASE)
PAT_DECL_HEAD = re.compile(
    r"use of (?:ai|generative ai|artificial intelligence|large language)|"
    r"\bai\b.{0,20}(?:disclosure|statement|declaration)|generative ai|"
    r"declaration of (?:generative )?(?:ai|artificial intelligence)", re.IGNORECASE)
NEGATION = re.compile(
    r"\b(?:no|not|without|never|neither|none|nor)\b|n't", re.IGNORECASE)
SECTION_TITLES = re.compile(
    r"method|material|acknowledg|declar|disclos|competing|conflict|"
    r"author contribution|funding|data availability|ethic|statement", re.IGNORECASE)

TOOL_NAMES = [
    ("ChatGPT/GPT", r"chat\s?gpt|gpt|openai"),
    ("Claude", r"claude|anthropic"),
    ("Gemini/Bard", r"gemini|bard"),
    ("Copilot", r"copilot"),
    ("DeepSeek", r"deepseek"),
    ("Llama", r"llama"),
    ("Mistral", r"mistral"),
    ("Perplexity", r"perplexity"),
    ("QuillBot/Grammarly/DeepL", r"quillbot|grammarly|deepl"),
    ("Unspecified LLM/GenAI", r"language model|llm|generative|gen-?ai"),
]


# ---------------------------------------------------------------- helpers
def get(url, params=None, tries=3):
    for t in range(tries):
        try:
            r = SESSION.get(url, params=params, timeout=90)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(1.5 * (t + 1))
    return None


def local(tag):
    """Element name without namespace."""
    return tag.split("}")[-1] if isinstance(tag, str) else ""


def text_of(el):
    return " ".join(" ".join(el.itertext()).split())


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def canonical_tools(text):
    found = {m.group(0).lower() for m in PAT_TOOL.finditer(text)}
    names = [name for name, pat in TOOL_NAMES
             if any(re.search(pat, f) for f in found)]
    if len(names) > 1 and "Unspecified LLM/GenAI" in names:
        names.remove("Unspecified LLM/GenAI")       # a named tool is more specific
    return ";".join(names)


# ---------------------------------------------------------------- sampling
def query_for_day(day):
    d = day.isoformat()
    return (f'(OPEN_ACCESS:y) AND (HAS_FT:y) AND (PUB_TYPE:"research-article") '
            f'AND (FIRST_PDATE:[{d} TO {d}])')


def year_total(year):
    q = (f'(OPEN_ACCESS:y) AND (HAS_FT:y) AND (PUB_TYPE:"research-article") '
         f'AND (FIRST_PDATE:[{year}-01-01 TO {year}-12-31])')
    r = get(SEARCH, {"query": q, "format": "json", "pageSize": 1, "resultType": "lite"})
    return r.json().get("hitCount") if r else None


def sample_pmcids(year, n, rng):
    """Stratified random sample: random days of the year, then random
    articles within each day (avoids taking only the first search results)."""
    last = min(date(year, 12, 31), TODAY - timedelta(days=1))
    days = [date(year, 1, 1) + timedelta(days=i)
            for i in range((last - date(year, 1, 1)).days + 1)]
    rng.shuffle(days)
    chosen = []
    for day in days:
        if len(chosen) >= n:
            break
        r = get(SEARCH, {"query": query_for_day(day), "format": "json",
                         "pageSize": 1000, "resultType": "lite"})
        if r is None:
            continue
        hits = [it["pmcid"] for it in r.json().get("resultList", {}).get("result", [])
                if it.get("pmcid")]
        rng.shuffle(hits)
        chosen.extend(hits[:min(ARTICLES_PER_DAY, n - len(chosen))])
        time.sleep(SLEEP)
    return chosen


# ---------------------------------------------------------------- analysis
def candidate_paragraphs(root):
    """(location, paragraph text) from sections where disclosures appear.
    Table and figure footnotes are excluded."""
    out = []
    for el in root.iter():
        name = local(el.tag)
        if name == "ack":
            loc = "acknowledgements"
        elif name == "author-notes":
            loc = "declarations"
        elif name == "fn-group":
            loc = "declarations"
        elif name == "sec":
            title_el = next((c for c in el if local(c.tag) == "title"), None)
            title = text_of(title_el) if title_el is not None else ""
            if PAT_DECL_HEAD.search(title):
                loc = "dedicated AI section"
            elif re.search(r"method|material", title, re.IGNORECASE):
                loc = "methods"
            elif SECTION_TITLES.search(title):
                loc = "declarations"
            else:
                continue
        else:
            continue
        paras = [p for p in el.iter() if local(p.tag) == "p"] or [el]
        out.extend((loc, text_of(p)) for p in paras)
    return out


def is_disclosure(loc, para):
    for m in PAT_USE.finditer(para):
        window = para[max(0, m.start() - 50): m.end() + 15]
        if not NEGATION.search(window):
            return True
    if loc == "dedicated AI section" and PAT_TOOL.search(para) and not NEGATION.search(para):
        return True
    return False


def analyse_article(pmcid):
    r = get(FULLTEXT.format(pmcid=pmcid))
    if r is None or not r.text.strip():
        return None
    try:
        root = ET.fromstring(r.content)
    except ET.ParseError:
        return None

    abstract = " ".join(text_of(a) for a in root.iter() if local(a.tag) == "abstract")
    title = " ".join(text_of(t) for t in root.iter() if local(t.tag) == "article-title")
    if not abstract:
        return None
    signal = bool(set(TOKEN.findall(abstract.lower())) & RARE)
    ai_topic = bool(PAT_TOOL.search(title + " " + abstract))   # article is ABOUT AI

    for loc, para in candidate_paragraphs(root):
        if is_disclosure(loc, para):
            return {"disclosed": True, "signal": signal, "ai_topic": ai_topic,
                    "tools": canonical_tools(para), "location": loc,
                    "words": len(para.split())}
    return {"disclosed": False, "signal": signal, "ai_topic": ai_topic,
            "tools": "", "location": "", "words": 0}


def excess_signal(df):
    """Kobak-style excess of the abstract signal over the pre-ChatGPT trend."""
    s = df.set_index("year")["signal_rate_pct"]
    f21, f22 = s.get(2021), s.get(2022)
    slope = max(0.0, f22 - f21) if f21 is not None and f22 is not None else 0.0
    return [s[y] - (f22 + slope * (y - 2022)) if y > 2022 and f22 is not None else np.nan
            for y in df["year"]]


# ---------------------------------------------------------------- main
def main():
    rng = random.Random(SEED)
    rows, details = [], []
    for year in YEARS:
        total = year_total(year)
        ids = sample_pmcids(year, SAMPLE_PER_YEAR, rng)
        print(f"== {year} == OA research articles: {total}; sampling {len(ids)}")
        n = n_disc = n_disc_writing = n_sig = n_topic = 0
        for pmcid in ids:
            res = analyse_article(pmcid)
            time.sleep(SLEEP)
            if res is None:
                continue
            n += 1
            n_sig += res["signal"]
            n_topic += res["ai_topic"]
            if res["disclosed"]:
                n_disc += 1
                n_disc_writing += not res["ai_topic"]
                details.append({"year": year, "pmcid": pmcid, **{k: res[k] for k in
                                ("tools", "location", "words", "ai_topic")}})
        n_not_topic = n - n_topic
        lo, hi = wilson(n_disc, n)
        rows.append({
            "year": year, "oa_articles": total, "n_analysed": n,
            "disclosure_rate_pct": 100 * n_disc / n if n else np.nan,
            "disclosure_ci_low": lo, "disclosure_ci_high": hi,
            # excluding articles whose topic is AI (methods use of AI as the object of study)
            "disclosure_rate_non_ai_topic_pct":
                100 * n_disc_writing / n_not_topic if n_not_topic else np.nan,
            "signal_rate_pct": 100 * n_sig / n if n else np.nan,
            "partial_year": year >= TODAY.year,
        })
        print(f"   analysed {n}; disclosed {n_disc} ({rows[-1]['disclosure_rate_pct']:.2f}%); "
              f"abstract signal {rows[-1]['signal_rate_pct']:.1f}%")

    out = pd.DataFrame(rows)
    out["excess_signal_pct"] = excess_signal(out)
    # gap only defined when there is a positive excess and at least one disclosure
    valid = (out["excess_signal_pct"] > 0) & (out["disclosure_rate_non_ai_topic_pct"] > 0)
    out["gap_excess_over_declared"] = np.where(
        valid, out["excess_signal_pct"] / out["disclosure_rate_non_ai_topic_pct"], np.nan)
    out.to_csv("transparency_by_year.csv", index=False)

    det = pd.DataFrame(details)
    det.to_csv("disclosure_details.csv", index=False)
    print(out.to_string(index=False))

    if not det.empty:
        d = det[~det.ai_topic]
        print("\nStatement characteristics (non-AI-topic articles; compare with Driessen et al.):")
        print(f"  mean words: {d.words.mean():.0f}")
        print("  location:", (d.location.value_counts(normalize=True) * 100).round(1).to_dict())
        tools = d.tools.str.split(";").explode()
        print("  tools:", (tools.value_counts() / len(d) * 100).round(1).to_dict())
    print("\nSaved transparency_by_year.csv and disclosure_details.csv")


if __name__ == "__main__":
    main()