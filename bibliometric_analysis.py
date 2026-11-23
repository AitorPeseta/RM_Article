"""
Primary bibliometric analysis of GenAI's footprint in the biomedical and
life-sciences literature, run on the OpenAlex API.

Outputs (one row per year):
  1. volume_by_year.csv     publication volume
  2. adoption_by_year.csv   excess-vocabulary frequencies and the lower-bound
                            estimate of LLM-assisted abstracts (Kobak et al.)
  3. citation_by_year.csv   citations of papers with vs without the LLM marker
  4. diversity_by_year.csv  topic diversity (Shannon entropy of OpenAlex topics)
  + adoption_trend.png and citation_trend.png

METHOD FOR ADOPTION (Kobak et al., Sci. Adv. 11, eadt3813, 2025)
  Two groups of LLM "style" words are used, exactly as in the paper:
    - COMMON_WORDS: the 10 frequent style words used by Kobak et al.
    - RARE_WORDS:   the 291 excess style words with frequency < 0.02 in 2024,
                    reproduced from results/excess_words.csv and
                    results/yearly-counts.csv.gz of github.com/berenslab/llm-excess-vocab
  For each group, f(t) is the share of abstracts in year t containing at least
  one of its words. The counterfactual (expected without LLMs) is the
  pre-ChatGPT trend:   q(t) = f(2022) + max(0, f(2022) - f(2021)) * (t - 2022)
  The excess is Delta(t) = f(t) - q(t), and the estimate is the mean of the two
  groups' Delta, a LOWER BOUND on the share of abstracts processed with LLMs.

HOW TO RUN
  pip install requests pandas numpy matplotlib
  (optional) get a free OpenAlex API key and put it in API_KEY below
  python bibliometric_analysis.py
"""

import re
import time
import math
from collections import Counter
from datetime import date

import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- config
API_KEY = ""                        # free OpenAlex key (recommended); leave "" to run without
YEARS = list(range(2020, 2027))     # 2020..2026; 2020-2022 = pre-ChatGPT baseline
SAMPLE_PER_YEAR = 10000             # OpenAlex maximum for `sample`
SEED = 42
PER_PAGE = 100                      # OpenAlex maximum per page
# Scope: biomedical and life sciences = OpenAlex domains "Life Sciences" + "Health Sciences".
# Add "Physical Sciences" here if you want all natural sciences.
DOMAINS = ["Life Sciences", "Health Sciences"]

# Kobak et al. (2025): 10 common excess style words
COMMON_WORDS = [
    "exhibited", "crucial", "additionally", "within", "notably",
    "insights", "comprehensive", "across", "particularly", "enhancing",
]

# Kobak et al. (2025): 291 rare excess style words (frequency < 0.02 in 2024)
RARE_WORDS = [
    "accentuates", "acknowledges", "acknowledging", "addresses", "adept", "adhered", "adhering", "advancement",
    "advancements", "advancing", "advocates", "advocating", "affirming", "afflicted", "aiding", "akin",
    "align", "aligning", "aligns", "alongside", "amidst", "assessments", "attains", "attributed",
    "augmenting", "avenue", "avenues", "bolster", "bolstered", "bolstering", "broader", "burgeoning",
    "capabilities", "capitalizing", "categorized", "categorizes", "categorizing", "combating", "commendable", "compelling",
    "complicates", "complicating", "comprehending", "comprising", "consequently", "consolidates", "contributing", "conversely",
    "correlating", "crafted", "crafting", "culminating", "customizing", "delineates", "delve", "delved",
    "delves", "delving", "demonstrating", "dependability", "dependable", "detailing", "detrimentally", "diminishes",
    "diminishing", "discern", "discerned", "discernible", "discerning", "displaying", "disrupts", "distinctions",
    "distinctive", "elevate", "elevates", "elevating", "elucidate", "elucidates", "elucidating", "embracing",
    "emerges", "emphasises", "emphasising", "emphasize", "emphasizes", "emphasizing", "employing", "employs",
    "empowers", "emulating", "emulation", "enabling", "encapsulates", "encompass", "encompassed", "encompasses",
    "encompassing", "endeavors", "endeavours", "enduring", "enhancements", "enhances", "ensuring", "equipping",
    "escalating", "evaluates", "evolving", "exacerbating", "examines", "exceeding", "excels", "exceptional",
    "exceptionally", "exerting", "exhibiting", "exhibits", "expedite", "expediting", "exploration", "explores",
    "facilitated", "facilitates", "facilitating", "featuring", "formidable", "fostering", "fosters", "foundational",
    "furnish", "garnered", "garnering", "gauged", "grappling", "groundbreaking", "groundwork", "harness",
    "harnesses", "harnessing", "heighten", "heightened", "hinder", "hinges", "hinting", "hold",
    "holds", "illuminates", "illuminating", "imbalances", "impacting", "impede", "impeding", "imperative",
    "impressive", "inadequately", "incorporates", "incorporating", "influencing", "inherent", "initially", "innovative",
    "inquiries", "integrates", "integrating", "integration", "interconnectedness", "interplay", "intricacies", "intricate",
    "intricately", "introduces", "invaluable", "investigates", "involves", "juxtaposed", "leverages", "leveraging",
    "maintaining", "merges", "methodologies", "meticulous", "meticulously", "multifaceted", "necessitate", "necessitates",
    "necessitating", "necessity", "notable", "noteworthy", "nuanced", "nuances", "offering", "optimizing",
    "orchestrating", "outlines", "overlook", "overlooking", "paving", "persist", "pinpoint", "pinpointed",
    "pinpointing", "pioneering", "pioneers", "pivotal", "poised", "pose", "posed", "poses",
    "posing", "predominantly", "preserving", "pressing", "promise", "pronounced", "propelling", "realm",
    "realms", "recognizing", "refine", "refines", "refining", "remarkable", "renowned", "revealing",
    "reveals", "revolutionize", "revolutionizing", "revolves", "scrutinize", "scrutinized", "scrutinizing", "seamless",
    "seamlessly", "seeks", "serves", "serving", "shaping", "shedding", "showcased", "showcases",
    "showcasing", "signifying", "solidify", "spanned", "spanning", "spurred", "stands", "stemming",
    "strategically", "streamline", "streamlined", "streamlines", "streamlining", "struggle", "substantiated", "substantiates",
    "surged", "surmount", "surpass", "surpassed", "surpasses", "surpassing", "swift", "swiftly",
    "thorough", "transformative", "typically", "ultimately", "uncharted", "uncovering", "underexplored", "underscore",
    "underscored", "underscores", "underscoring", "unexplored", "unlocking", "unparalleled", "unraveling", "unveil",
    "unveiled", "unveiling", "unveils", "uphold", "upholding", "urging", "utilizes", "varying",
    "versatility", "warranting", "yielding",
]

COMMON = set(COMMON_WORDS)
RARE = set(RARE_WORDS)
assert len(COMMON) == 10 and len(RARE) == 291

BASE = "https://api.openalex.org"
SESSION = requests.Session()
TOKEN = re.compile(r"\b\w\w+\b")    # same token pattern as Kobak et al. (CountVectorizer)
CURRENT_YEAR = date.today().year


# ---------------------------------------------------------------- API helpers
def get_json(url, params, tries=5):
    params = dict(params)
    if API_KEY:
        params["api_key"] = API_KEY
    for t in range(tries):
        r = SESSION.get(url, params=params, timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:            # rate limit: wait and retry
            time.sleep(5 * (t + 1))
            continue
        time.sleep(2 * (t + 1))
    r.raise_for_status()


def resolve_domain_filter():
    """Look up the OpenAlex ids of DOMAINS and build a working filter string."""
    data = get_json(f"{BASE}/domains", {"per_page": 100})
    ids = []
    for d in data["results"]:
        if d["display_name"] in DOMAINS:
            ids.append(d["id"].rstrip("/").split("/")[-1])
    if len(ids) != len(DOMAINS):
        raise SystemExit(f"Could not find all domains {DOMAINS}; got {ids}")
    # try the short-id format first, then the full-URL format
    for fmt in ("{}", "https://openalex.org/domains/{}"):
        flt = "primary_topic.domain.id:" + "|".join(fmt.format(i) for i in ids)
        try:
            test = get_json(f"{BASE}/works", {"filter": f"{flt},publication_year:2021",
                                              "per_page": 1})
            if test["meta"]["count"] > 0:
                print(f"Domain filter: {flt}")
                return flt
        except requests.HTTPError:
            pass
    raise SystemExit("The domain filter did not return results; check the OpenAlex docs.")


def year_filter(domain_filter, year, with_abstract=True):
    f = f"{domain_filter},publication_year:{year},type:article"
    if with_abstract:
        f += ",has_abstract:true"
    return f


def volume_for_year(domain_filter, year):
    """Total number of articles that year (all articles, with or without abstract)."""
    data = get_json(f"{BASE}/works",
                    {"filter": year_filter(domain_filter, year, with_abstract=False),
                     "per_page": 1})
    return data["meta"]["count"]


def reconstruct_abstract(inv):
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos)).lower()


def sample_year(domain_filter, year):
    """Random sample of the year's articles with abstract, using basic paging
    (basic paging reaches exactly the 10,000-result maximum of `sample`)."""
    params = {
        "filter": year_filter(domain_filter, year),
        "sample": SAMPLE_PER_YEAR,
        "seed": SEED,
        "per_page": PER_PAGE,
        "select": "id,abstract_inverted_index,cited_by_count,primary_topic",
    }
    n_pages = math.ceil(SAMPLE_PER_YEAR / PER_PAGE)
    seen = set()
    for page in range(1, n_pages + 1):
        params["page"] = page
        data = get_json(f"{BASE}/works", params)
        if not data["results"]:
            break
        for w in data["results"]:
            if w["id"] in seen:            # guard against duplicates
                continue
            seen.add(w["id"])
            text = reconstruct_abstract(w.get("abstract_inverted_index"))
            topic = (w.get("primary_topic") or {}).get("display_name")
            yield text, w.get("cited_by_count", 0), topic
        time.sleep(0.15)


# ---------------------------------------------------------------- measures
def word_groups(text):
    toks = set(TOKEN.findall(text.lower()))
    return bool(toks & COMMON), bool(toks & RARE)


def entropy_bits(counter):
    total = sum(counter.values())
    if total == 0:
        return float("nan"), float("nan")
    p = np.array(list(counter.values()), dtype=float) / total
    h = float(-(p * np.log2(p)).sum())
    h_norm = h / math.log2(len(counter)) if len(counter) > 1 else float("nan")
    return h, h_norm


def kobak_excess(adopt):
    """Delta(t) = f(t) - [f(2022) + max(0, f(2022) - f(2021)) * (t - 2022)]."""
    a = adopt.set_index("year")
    for g in ("common", "rare"):
        f21, f22 = a.loc[2021, f"freq_{g}"], a.loc[2022, f"freq_{g}"]
        slope = max(0.0, f22 - f21)
        expected = [f22 + slope * (y - 2022) if y > 2022 else np.nan for y in a.index]
        a[f"expected_{g}"] = expected
        a[f"delta_{g}"] = a[f"freq_{g}"] - a[f"expected_{g}"]
    a["delta_llm_lower_bound"] = (a["delta_common"] + a["delta_rare"]) / 2
    return a.reset_index()


# ---------------------------------------------------------------- main
def main():
    domain_filter = resolve_domain_filter()
    vol_rows, adopt_rows, cite_rows, div_rows = [], [], [], []

    for year in YEARS:
        print(f"== {year} ==")
        partial = year >= CURRENT_YEAR
        vol_rows.append({"year": year,
                         "n_articles": volume_for_year(domain_filter, year),
                         "partial_year": partial})

        n = n_common = n_rare = 0
        cites_sig, cites_nosig = [], []
        topics = Counter()

        for text, cites, topic in sample_year(domain_filter, year):
            if not text:
                continue
            n += 1
            has_common, has_rare = word_groups(text)
            n_common += has_common
            n_rare += has_rare
            # per-paper LLM marker: at least one rare style word
            (cites_sig if has_rare else cites_nosig).append(cites)
            if topic:
                topics[topic] += 1

        adopt_rows.append({"year": year, "n_sampled": n,
                           "freq_common": n_common / n if n else np.nan,
                           "freq_rare": n_rare / n if n else np.nan,
                           "partial_year": partial})
        cite_rows.append({"year": year,
                          "median_cites_marker": np.median(cites_sig) if cites_sig else np.nan,
                          "median_cites_no_marker": np.median(cites_nosig) if cites_nosig else np.nan,
                          "mean_cites_marker": np.mean(cites_sig) if cites_sig else np.nan,
                          "mean_cites_no_marker": np.mean(cites_nosig) if cites_nosig else np.nan,
                          "n_marker": len(cites_sig), "n_no_marker": len(cites_nosig)})
        h, h_norm = entropy_bits(topics)
        div_rows.append({"year": year, "n_topics": len(topics),
                         "topic_entropy_bits": h, "topic_entropy_normalised": h_norm})
        print(f"   sampled {n}; rare-word share {100 * n_rare / max(n, 1):.1f}%")

    vol = pd.DataFrame(vol_rows)
    adopt = kobak_excess(pd.DataFrame(adopt_rows))
    cite = pd.DataFrame(cite_rows)
    div = pd.DataFrame(div_rows)

    vol.to_csv("volume_by_year.csv", index=False)
    adopt.to_csv("adoption_by_year.csv", index=False)
    cite.to_csv("citation_by_year.csv", index=False)
    div.to_csv("diversity_by_year.csv", index=False)
    print(adopt[["year", "freq_common", "freq_rare", "delta_llm_lower_bound"]].to_string(index=False))

    # -------- plots
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(adopt.year, 100 * adopt.freq_common, marker="o", label="Common style words (10)")
    ax.plot(adopt.year, 100 * adopt.freq_rare, marker="s", label="Rare style words (291)")
    ax.axvline(2022.9, ls="--", color="0.5")
    ax.text(2022.95, ax.get_ylim()[1] * 0.95, "ChatGPT", fontsize=8, va="top")
    ax.set_xlabel("Year"); ax.set_ylabel("% of abstracts")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig("adoption_trend.png", dpi=200)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(cite.year, cite.median_cites_marker, marker="o", label="With LLM marker")
    ax.plot(cite.year, cite.median_cites_no_marker, marker="s", label="Without marker")
    ax.set_xlabel("Year of publication"); ax.set_ylabel("Median citations")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig("citation_trend.png", dpi=200)
    print("\nSaved the CSV files, adoption_trend.png and citation_trend.png")


if __name__ == "__main__":
    main()