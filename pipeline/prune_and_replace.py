#!/usr/bin/env python3
"""
persian-pragmatics-dataset: Prunes near-duplicates (Similarity >= 85%) and synthesizes
novel replacement items via live Gemini 3.8 Flash to restore exact 5,000 / 1,000 balance.
"""

import json, csv, random, re, sys, time
from collections import defaultdict, Counter
from difflib import SequenceMatcher
from pathlib import Path

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.antigravity_client import call_antigravity_gemini
from pipeline.prompts import CATEGORY_PROMPTS, SYSTEM_PROMPT
from pipeline.generate_with_llm import clean_llm_json, validate_quality_gate, call_with_retry
from pipeline.scenarios_v2 import SCENARIOS_V2

data_dir = Path(__file__).parent.parent / "data"

def get_words(t):
    return set("".join(c if c.isalnum() else " " for c in t).split())

def find_near_duplicate_indices(items, max_sim=0.85):
    utts = [x["utterance"].strip() for x in items]
    word_sets = [get_words(u) for u in utts]
    
    inv_idx = defaultdict(list)
    for idx, ws in enumerate(word_sets):
        for w in ws:
            inv_idx[w].append(idx)
            
    to_remove = set()
    checked = set()
    
    for i, ws in enumerate(word_sets):
        if i in to_remove:
            continue
        candidates = set()
        for w in ws:
            candidates.update(inv_idx[w])
        for j in candidates:
            if j > i and j not in to_remove:
                pair = (i, j)
                if pair not in checked:
                    checked.add(pair)
                    ws_j = word_sets[j]
                    inter = len(ws & ws_j)
                    union = len(ws | ws_j)
                    jaccard = inter / union if union > 0 else 0
                    if jaccard >= max_sim:
                        ratio = SequenceMatcher(None, utts[i], utts[j]).ratio()
                        if ratio >= max_sim:
                            to_remove.add(j) # remove the latter duplicate
                            
    return to_remove

def main():
    print("Loading current dataset...")
    all_items = []
    for split in ["train", "test"]:
        with open(data_dir / f"{split}.jsonl", "r", encoding="utf-8") as f:
            all_items.extend([json.loads(l) for l in f if l.strip()])
            
    print(f"Total current items: {len(all_items):,}")
    
    # 1. Identify near duplicates
    remove_indices = find_near_duplicate_indices(all_items, max_sim=0.85)
    print(f"Found {len(remove_indices)} near-duplicates (Similarity >= 85%). Pruning...")
    
    pruned_items = [item for idx, item in enumerate(all_items) if idx not in remove_indices]
    print(f"Remaining pristine items: {len(pruned_items):,}")
    
    # 2. Count deficit per category to reach 750 per category (6,000 total)
    counts = Counter(x["category"] for x in pruned_items)
    deficits = {cat: 750 - counts.get(cat, 0) for cat in CATEGORY_PROMPTS.keys()}
    total_needed = sum(deficits.values())
    print(f"Total replacement items needed: {total_needed} across categories: {deficits}")
    
    # Existing utterances to guard against creating duplicates
    seen_utts = set(x["utterance"].strip() for x in pruned_items)
    seen_prefixes = {}
    for x in pruned_items:
        cat = x["category"]
        pfx2 = " ".join(x["utterance"].split()[:2])
        seen_prefixes.setdefault(cat, {})
        seen_prefixes[cat][pfx2] = seen_prefixes[cat].get(pfx2, 0) + 1

    # 3. Live synthesis of replacements via Gemini 3.8 Flash
    new_items = []
    sample_id = len(all_items) + 1
    
    for cat, needed in deficits.items():
        if needed <= 0:
            continue
        print(f"\nSynthesizing {needed} novel replacements for category: {cat}...")
        scenarios = SCENARIOS_V2[cat]
        collected = 0
        scen_idx = 0
        
        while collected < needed:
            scenario = scenarios[scen_idx % len(scenarios)]
            prompt = CATEGORY_PROMPTS[cat]["prompt"].format(scenario=scenario)
            
            raw_res = call_with_retry(prompt)
            parsed = clean_llm_json(raw_res)
            
            if not parsed or not validate_quality_gate(parsed, seen_prefixes):
                time.sleep(0.3)
                continue
                
            utt = parsed["utterance"].strip()
            if utt in seen_utts:
                continue
                
            # Extra check against high similarity with any existing
            words_utt = get_words(utt)
            too_similar = False
            for prev_u in seen_utts:
                # Fast check: word overlap
                ws_prev = get_words(prev_u)
                if len(words_utt & ws_prev) / len(words_utt | ws_prev) >= 0.85:
                    if SequenceMatcher(None, utt, prev_u).ratio() >= 0.85:
                        too_similar = True
                        break
            if too_similar:
                continue
                
            seen_utts.add(utt)
            pfx2 = " ".join(utt.split()[:2])
            seen_prefixes.setdefault(cat, {})
            seen_prefixes[cat][pfx2] = seen_prefixes[cat].get(pfx2, 0) + 1
            
            rec = {
                "id": f"{cat}-{sample_id:05d}",
                "category": cat,
                "context": parsed.get("context", f"موقعیت در {scenario}").strip(),
                "utterance": utt,
                "surface_meaning": parsed.get("surface_meaning", "").strip(),
                "pragmatic_intent": parsed.get("pragmatic_intent", "").strip(),
                "correct_response": parsed.get("correct_response", "").strip(),
                "naive_llm_response": parsed.get("naive_llm_response", "").strip()
            }
            new_items.append(rec)
            pruned_items.append(rec)
            sample_id += 1
            collected += 1
            scen_idx += 1
            sys.stdout.write(f"\r  [{collected}/{needed}] Novel: «{utt[:35]}...»")
            sys.stdout.flush()
            time.sleep(0.4)
            
    print(f"\n\nTotal items after replacement: {len(pruned_items):,}")
    assert len(pruned_items) == 6000, f"Expected 6000, got {len(pruned_items)}"
    
    # 4. Final Stratified Split (5,000 train / 1,000 test) with global shuffle
    random.seed(42)
    by_cat = defaultdict(list)
    for x in pruned_items:
        by_cat[x["category"]].append(x)
        
    train, test = [], []
    for cat, pool in sorted(by_cat.items()):
        assert len(pool) == 750, f"{cat} has {len(pool)} items, expected 750"
        random.shuffle(pool)
        train.extend(pool[:625])
        test.extend(pool[625:])
        
    random.shuffle(train)
    random.shuffle(test)
    
    assert len(train) == 5000 and len(test) == 1000
    assert len(set(x["utterance"] for x in train) & set(x["utterance"] for x in test)) == 0
    
    # 5. Save clean splits
    fields = list(train[0].keys())
    for name, split in [("train", train), ("test", test)]:
        with open(data_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for x in split:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        with open(data_dir / f"{name}.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
            w.writeheader()
            w.writerows(split)
            
    print("✓ Success! Re-split and exported with zero near-duplicates (Similarity < 85%).")

if __name__ == "__main__":
    main()
