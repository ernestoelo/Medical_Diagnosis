from transformers import pipeline
from typing import List
import re
import unicodedata
import json
import os


from transformers import pipeline
from typing import List
import re
import unicodedata
import json
import os


# helper normalization at module level for reuse when loading map
def normalize_text(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize('NFD', s)
    s = ''.join(ch for ch in s if unicodedata.category(ch) != 'Mn')
    return s.strip()


# load canonical map from data/spanish_map.json (fallback to embedded defaults)
CANONICAL_MAP = {}
try:
    base = os.path.dirname(__file__)
    json_path = os.path.join(base, '..', 'data', 'spanish_map.json')
    json_path = os.path.normpath(json_path)
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as fh:
            raw = json.load(fh)
        # normalize keys
        for k, v in raw.items():
            CANONICAL_MAP[normalize_text(k)] = v
    else:
        CANONICAL_MAP = {}
except Exception:
    CANONICAL_MAP = {}


# Initialize NER pipeline when module is imported
try:
    print("Loading biomedical NER model (may take a while on first run)")
    # Use explicit aggregation to reduce subword fragmentation from the tokenizer
    ner_pipeline = pipeline('ner', model='d4data/biomedical-ner-all', aggregation_strategy='simple')
    print("NER model loaded successfully.")
except Exception as e:
    print(f"Error loading NER model (first run may require internet): {e}")
    ner_pipeline = None


def extract_nodes_from_query(user_query: str) -> List[str]:
    """Extract symptom/condition candidates from a Spanish user query.

    Uses a biomedical NER model plus simple phrase heuristics. Returns a
    deduplicated, normalized list of symptom candidate strings.
    """
    if not ner_pipeline:
        print("NER pipeline not available; cannot extract nodes.")
        return []

    try:
        ner_results = ner_pipeline(user_query)

        # use module-level normalize_text

        lead_patterns = re.compile(r'^(desde hace|desde|tengo( la)?|tengo|me |hace)\s*')
        trailing_patterns = re.compile(r'\s*(desde ayer|ayer|anoche)$')

        stop_words = {'desde', 'tengo', 'hace', 'como', 'con', 'me', 'siento', 'ayer'}
        symptom_keywords = [
            'fiebre', 'temperatura', 'dolor de cabeza', 'cefalea', 'migra', 'tos',
            'tos seca', 'tos productiva', 'tos persistente', 'diarrea', 'nausea', 'nauseas',
            'náuseas', 'vómitos', 'vomitos', 'mareo', 'mareos', 'vertigo', 'fatiga', 'astenia',
            'dolor', 'dolor muscular', 'dolores musculares', 'mialgia', 'dolor abdominal',
            'dolor en el pecho', 'dolor toracico', 'sangrado', 'palpitaciones',
            'dificultad respiratoria', 'perdida de apetito', 'pérdida de apetito',
            'escalofrios', 'escalofríos', 'garganta', 'dolor de garganta', 'vision borrosa',
            'visión borrosa', 'mareo al levantarse', 'mareo cuando me levanto'
        ]

        # Phrase heuristics: split by commas and conjunctions
        parts = re.split(r'[;,]| y | e | and ', user_query)
        heuristic_found = []
        for part in parts:
            p = part.strip()
            if not p:
                continue
            pnorm = normalize_text(p)
            for kw in symptom_keywords:
                if kw in pnorm:
                    clean = lead_patterns.sub('', pnorm)
                    clean = trailing_patterns.sub('', clean)
                    clean = re.sub(r'\s+', ' ', clean).strip()
                    heuristic_found.append(clean)
                    break

        # Extract from NER
        extracted_nodes = []
        for result in ner_results:
            # pipeline aggregation returns 'word' or 'entity_group' and spans
            w = result.get('word') or result.get('entity_group') or result.get('entity') or ''
            if not w:
                continue
            w = w.replace('\n', ' ').replace('\t', ' ').replace('▁', ' ')
            w = re.sub(r'\s+', ' ', w).strip()
            if not w:
                continue
            wl = normalize_text(w)
            if wl in stop_words:
                continue
            w_clean = lead_patterns.sub('', wl)
            w_clean = trailing_patterns.sub('', w_clean)
            w_clean = re.sub(r'\s+', ' ', w_clean).strip()
            # Filter out tiny fragments that are likely tokenization artifacts
            if not w_clean:
                continue
            if len(w_clean) <= 3:
                # discard if it's a prefix of a stop word (e.g., 'aye' from 'ayer')
                if any(sw.startswith(w_clean) for sw in stop_words):
                    continue
                # common micro-artifacts to drop
                if w_clean in {'des', 'ten', 'aye', 'dos'}:
                    continue
            extracted_nodes.append(w_clean)

        def is_valid_token(w: str) -> bool:
            if not w:
                return False
            if w.startswith('##'):
                return False
            if len(w.strip()) <= 2:
                return False
            if not re.search(r'[a-zA-ZñÑáéíóúÁÉÍÓÚ]', w):
                return False
            return True

        valid_ner = [w for w in extracted_nodes if is_valid_token(w)]

        # use CANONICAL_MAP loaded from data/spanish_map.json (or empty if not found)
        combined = []
        seen = set()
        for w in valid_ner:
            w0 = CANONICAL_MAP.get(w, w)
            key = normalize_text(w0)
            if key in seen:
                continue
            seen.add(key)
            combined.append(w0)

        for h in heuristic_found:
            # apply same small-fragment filter to heuristics
            if not h:
                continue
            hk = h.strip()
            if len(hk) <= 3 and any(sw.startswith(hk) for sw in stop_words):
                continue
            keyh = normalize_text(hk)
            if keyh not in seen:
                seen.add(keyh)
                combined.append(hk)

        # Apply canonical mapping to all combined items (use normalized keys)
        mapped = []
        for item in combined:
            key = normalize_text(item)
            # exact mapping first
            mapped_item = CANONICAL_MAP.get(key)
            if not mapped_item:
                # try substring match: if any canonical key is contained in the
                # item, prefer that mapping (helps with phrases like 'se me inflama la garganta')
                for cm_k, cm_v in CANONICAL_MAP.items():
                    if cm_k in key and len(cm_k) >= 3:
                        mapped_item = cm_v
                        break
            if not mapped_item:
                mapped_item = item
            mapped.append(mapped_item)

        # --- Additional cleaning rules ---
        # Drop isolated intensity/adjective fragments (e.g. 'alta', 'altas')
        INTENSITY_ADJS = {'alta', 'altas', 'leve', 'leves', 'moderada', 'moderado'}

        # Adjective/noise blacklist: drop common adjectives or modifiers
        ADJ_NOISE = {'persistente', 'persistentes', 'intermitente', 'intermitentes', 'inter', 'extremo', 'extrema', 'extremos', 'extremas', 'leve', 'leves', 'moderado', 'moderada'}

        # Drop spans that contain temporal or connector stopwords (likely noisy)
        BAD_SPAN_WORDS = re.compile(r"\b(dia|dias|tengo|desde|hace|ayer|anoche)\b")

        filtered = []
        for it in mapped:
            it_norm = normalize_text(it)
            # if the span contains a known symptom keyword, keep it even if it
            # also contains temporal markers (helps keep 'tos ... dias' cases)
            if any(kw in it_norm for kw in symptom_keywords):
                filtered.append(it)
                continue
            # drop pure intensity adjectives
            if it_norm in INTENSITY_ADJS:
                continue
            # drop adjective noise
            if it_norm in ADJ_NOISE:
                continue
            # drop spans that contain temporal/connector words or look too long
            if BAD_SPAN_WORDS.search(it_norm):
                continue
            if len(it_norm.split()) > 6:
                continue
            filtered.append(it)

        mapped = filtered

        # Remove short fragments that are substrings or prefixes of longer items
        cleaned = []
        for i, a in enumerate(mapped):
            a_norm = normalize_text(a)
            tokens_a = set(a_norm.split())
            is_sub = False
            for j, b in enumerate(mapped):
                if i == j:
                    continue
                b_norm = normalize_text(b)
                # if a is very short and appears inside b, drop a
                if len(a_norm) <= 4 and a_norm in b_norm:
                    is_sub = True
                    break
                # if tokens of a subset of tokens of b and a shorter, drop a
                if len(tokens_a) < len(b_norm.split()) and tokens_a.issubset(set(b_norm.split())):
                    is_sub = True
                    break
            if not is_sub:
                cleaned.append(a)

        # If fragments indicate a body-part and there's a nearby pain verb,
        # construct a canonical 'dolor en el <parte>' phrase and remove
        # the fragmented pieces (handles e.g. 'duel' + 'pecho').
        pain_verbs = {'duele', 'duelo', 'doler', 'dolor', 'duel'}
        body_parts = {'pecho', 'abdomen', 'estomago', 'cabeza', 'pierna', 'brazo'}
        final_merge = []
        seen_body = set()
        # detect presence of pain-verb-like fragments or tokens
        has_pain_fragment = any(any(v in normalize_text(x) for v in pain_verbs) for x in mapped)
        for item in cleaned:
            n = normalize_text(item)
            # if item is a naked body part and we detected a pain fragment, merge
            if n in body_parts and has_pain_fragment:
                merged_phrase = f"dolor en el {n}"
                if merged_phrase not in final_merge:
                    final_merge.append(merged_phrase)
                seen_body.add(n)
                continue
            # skip tiny stray fragments that are purely body-part tokens if we've merged
            if n in seen_body:
                continue
            final_merge.append(item)

        cleaned = final_merge

        # --- Fragment joiner: try to merge adjacent short fragments when the
        # combined phrase appears in the original user query (normalized).
        norm_query = normalize_text(user_query)
        joined = []
        i = 0
        while i < len(cleaned):
            cur = cleaned[i]
            cur_k = normalize_text(cur)
            # if next exists, try to join
            if i + 1 < len(cleaned):
                nxt = cleaned[i + 1]
                nxt_k = normalize_text(nxt)
                combo = f"{cur_k} {nxt_k}"
                combo_no_space = f"{cur_k}{nxt_k}"
                # prefer the original substring if present
                if combo in norm_query or combo_no_space in norm_query:
                    # reconstruct from original query substring if possible
                    # fall back to the combined normalized phrase
                    if combo in norm_query:
                        chosen = combo
                    else:
                        chosen = combo.replace('  ', ' ').strip()
                    joined.append(chosen)
                    i += 2
                    continue
            joined.append(cur)
            i += 1

        cleaned = joined

        # Prioritize known symptom keywords (use canonical_map where possible)
        sk_norm = {normalize_text(k): k for k in symptom_keywords}
        prioritized = []
        rest = []
        for item in cleaned:
            key = normalize_text(item)
            if key in sk_norm:
                # prefer canonical mapping if exists
                prioritized.append(CANONICAL_MAP.get(key, sk_norm[key]))
            else:
                rest.append(item)

        merged = prioritized + rest

        # Remove tiny verb-like fragments: drop tokens of length < 4 unless
        # they appear in an allowlist derived from canonical map or common
        # short symptoms (e.g., 'tos'). This reduces leftover fragments like
        # 'du', 'ye', 'ay'.
        allow_short = {normalize_text(v) for v in CANONICAL_MAP.values()}
        allow_short.update({'tos'})
        filtered_merged = []
        for it in merged:
            k = normalize_text(it)
            if len(k) < 4 and k not in allow_short:
                # drop tiny fragments that are unlikely to be real symptoms
                continue
            filtered_merged.append(it)

        merged = filtered_merged

        final = []
        seen2 = set()
        for n in merged:
            nn = n.strip()
            if not nn:
                continue
            key = normalize_text(nn)
            if key in seen2:
                continue
            seen2.add(key)
            final.append(nn)

        # Ensure common short symptom keywords detected in the original query
        # are present in the output (helps when NER/heuristics missed them).
        norm_query = normalize_text(user_query)
        COMMON_KEYWORDS = ['tos', 'tos seca', 'tos productiva', 'vomitos', 'vómitos', 'nausea', 'garganta', 'dolor', 'escalofrio', 'vision borrosa']
        for ck in COMMON_KEYWORDS:
            if ck in norm_query:
                mapped_ck = CANONICAL_MAP.get(ck, ck)
                if normalize_text(mapped_ck) not in seen2:
                    final.append(mapped_ck)
                    seen2.add(normalize_text(mapped_ck))

        # --- Final-pass merger: combine adjacent fragments into known
        # multi-word canonical phrases when possible. This handles cases like
        # ['dolor','abdominal'] -> 'dolor abdominal' when the combo is present
        # in the canonical map or in the original query.
        merged_final = []
        i = 0
        while i < len(final):
            cur = final[i]
            cur_k = normalize_text(cur)
            if i + 1 < len(final):
                nxt = final[i + 1]
                nxt_k = normalize_text(nxt)
                combo = f"{cur_k} {nxt_k}"
                # if combo is a known canonical key or appears verbatim in query,
                # prefer the canonical mapping
                if combo in CANONICAL_MAP or combo in norm_query:
                    chosen = CANONICAL_MAP.get(combo, combo)
                    merged_final.append(chosen)
                    i += 2
                    continue
            merged_final.append(cur)
            i += 1

        # --- Merge non-adjacent tokens into multi-word canonicals when the
        # canonical phrase exists in the normalized query. This handles cases
        # like ['abdominal','dolor'] -> 'dolor abdominal' regardless of order.
        changed = True
        while changed:
            changed = False
            norms = [normalize_text(x) for x in merged_final]
            norm_set = set(norms)
            for cm_k, cm_v in CANONICAL_MAP.items():
                if ' ' not in cm_k:
                    continue
                if cm_k not in norm_query:
                    continue
                tokens = cm_k.split()
                # all tokens present as separate fragments?
                if all(t in norm_set for t in tokens):
                    # remove first occurrence of each token from merged_final
                    to_remove = set(tokens)
                    new_list = []
                    removed = set()
                    for it, nk in zip(merged_final, norms):
                        if nk in to_remove and nk not in removed:
                            removed.add(nk)
                            continue
                        new_list.append(it)
                    # append canonical mapped value
                    if cm_v not in new_list:
                        new_list.append(cm_v)
                    merged_final = new_list
                    changed = True
                    break

        # --- Drop short noisy fragments: remove tokens of length <=4 that are
        # substrings of any longer extracted token (reduces 'borro', 'levant').
        cleaned_final = []
        norms = [normalize_text(x) for x in merged_final]
        for idx, item in enumerate(merged_final):
            nk = norms[idx]
            if len(nk) <= 4:
                # keep if in allowlist
                if nk in allow_short:
                    cleaned_final.append(item)
                    continue
                # if this short fragment is a substring of any longer item, drop
                is_sub = False
                for jdx, other_k in enumerate(norms):
                    if jdx == idx:
                        continue
                    if len(other_k) > len(nk) and nk in other_k:
                        is_sub = True
                        break
                if is_sub:
                    continue
            cleaned_final.append(item)

        return cleaned_final
    except Exception as e:
        print(f"Unexpected error extracting nodes: {e}")
        return []

        merged = prioritized + rest

        # Remove tiny verb-like fragments: drop tokens of length < 4 unless
        # they appear in an allowlist derived from canonical map or common
        # short symptoms (e.g., 'tos'). This reduces leftover fragments like
        # 'du', 'ye', 'ay'.
        allow_short = {normalize_text(v) for v in CANONICAL_MAP.values()}
        allow_short.update({'tos'})
        filtered_merged = []
        for it in merged:
            k = normalize_text(it)
            if len(k) < 4 and k not in allow_short:
                # drop tiny fragments that are unlikely to be real symptoms
                continue
            filtered_merged.append(it)

        merged = filtered_merged

        final = []
        seen2 = set()
        for n in merged:
            nn = n.strip()
            if not nn:
                continue
            key = normalize_text(nn)
            if key in seen2:
                continue
            seen2.add(key)
            final.append(nn)

        return final
    except Exception as e:
        print(f"Unexpected error extracting nodes: {e}")
        return []
