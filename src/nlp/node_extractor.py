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
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    json_path = os.path.join(project_root, 'data', 'spanish_map.json')
    json_path = os.path.normpath(json_path)
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as fh:
            raw = json.load(fh)
        for k, v in raw.items():
            CANONICAL_MAP[normalize_text(k)] = v
    else:
        CANONICAL_MAP = {}
except Exception:
    CANONICAL_MAP = {}


# Initialize NER pipeline when module is imported
try:
    print("Loading biomedical NER model (may take a while on first run)")
    ner_pipeline = pipeline('ner', model='d4data/biomedical-ner-all', aggregation_strategy='simple')
    print("NER model loaded successfully.")
except Exception as e:
    print(f"Error loading NER model (first run may require internet): {e}")
    ner_pipeline = None


def extract_nodes_from_query(user_query: str) -> List[str]:
    if not ner_pipeline:
        print("NER pipeline not available; cannot extract nodes.")
        return []

    try:
        ner_results = ner_pipeline(user_query)

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

        extracted_nodes = []
        for result in ner_results:
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
            if not w_clean:
                continue
            if len(w_clean) <= 3:
                if any(sw.startswith(w_clean) for sw in stop_words):
                    continue
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
            if not h:
                continue
            hk = h.strip()
            if len(hk) <= 3 and any(sw.startswith(hk) for sw in stop_words):
                continue
            keyh = normalize_text(hk)
            if keyh not in seen:
                seen.add(keyh)
                combined.append(hk)

        mapped = []
        for item in combined:
            key = normalize_text(item)
            mapped_item = CANONICAL_MAP.get(key)
            if not mapped_item:
                for cm_k, cm_v in CANONICAL_MAP.items():
                    if cm_k in key and len(cm_k) >= 3:
                        mapped_item = cm_v
                        break
            if not mapped_item:
                mapped_item = item
            mapped.append(mapped_item)

        INTENSITY_ADJS = {'alta', 'altas', 'leve', 'leves', 'moderada', 'moderado'}
        ADJ_NOISE = {'persistente', 'persistentes', 'intermitente', 'intermitentes', 'inter', 'extremo', 'extrema', 'extremos', 'extremas', 'leve', 'leves', 'moderado', 'moderada'}
        BAD_SPAN_WORDS = re.compile(r"\b(dia|dias|tengo|desde|hace|ayer|anoche)\b")

        filtered = []
        for it in mapped:
            it_norm = normalize_text(it)
            if any(kw in it_norm for kw in symptom_keywords):
                filtered.append(it)
                continue
            if it_norm in INTENSITY_ADJS:
                continue
            if it_norm in ADJ_NOISE:
                continue
            if BAD_SPAN_WORDS.search(it_norm):
                continue
            if len(it_norm.split()) > 6:
                continue
            filtered.append(it)

        mapped = filtered

        cleaned = []
        for i, a in enumerate(mapped):
            a_norm = normalize_text(a)
            tokens_a = set(a_norm.split())
            is_sub = False
            for j, b in enumerate(mapped):
                if i == j:
                    continue
                b_norm = normalize_text(b)
                if len(a_norm) <= 4 and a_norm in b_norm:
                    is_sub = True
                    break
                if len(tokens_a) < len(b_norm.split()) and tokens_a.issubset(set(b_norm.split())):
                    is_sub = True
                    break
            if not is_sub:
                cleaned.append(a)

        pain_verbs = {'duele', 'duelo', 'doler', 'dolor', 'duel'}
        body_parts = {'pecho', 'abdomen', 'estomago', 'cabeza', 'pierna', 'brazo'}
        final_merge = []
        seen_body = set()
        has_pain_fragment = any(any(v in normalize_text(x) for v in pain_verbs) for x in mapped)
        for item in cleaned:
            n = normalize_text(item)
            if n in body_parts and has_pain_fragment:
                merged_phrase = f"dolor en el {n}"
                if merged_phrase not in final_merge:
                    final_merge.append(merged_phrase)
                seen_body.add(n)
                continue
            if n in seen_body:
                continue
            final_merge.append(item)

        cleaned = final_merge

        norm_query = normalize_text(user_query)
        joined = []
        i = 0
        while i < len(cleaned):
            cur = cleaned[i]
            cur_k = normalize_text(cur)
            if i + 1 < len(cleaned):
                nxt = cleaned[i + 1]
                nxt_k = normalize_text(nxt)
                combo = f"{cur_k} {nxt_k}"
                combo_no_space = f"{cur_k}{nxt_k}"
                if combo in norm_query or combo_no_space in norm_query:
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

        sk_norm = {normalize_text(k): k for k in symptom_keywords}
        prioritized = []
        rest = []
        for item in cleaned:
            key = normalize_text(item)
            if key in sk_norm:
                prioritized.append(CANONICAL_MAP.get(key, sk_norm[key]))
            else:
                rest.append(item)

        merged = prioritized + rest

        allow_short = {normalize_text(v) for v in CANONICAL_MAP.values()}
        allow_short.update({'tos'})
        filtered_merged = []
        for it in merged:
            k = normalize_text(it)
            if len(k) < 4 and k not in allow_short:
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

        norm_query = normalize_text(user_query)
        COMMON_KEYWORDS = ['tos', 'tos seca', 'tos productiva', 'vomitos', 'vómitos', 'nausea', 'garganta', 'dolor', 'escalofrio', 'vision borrosa']
        for ck in COMMON_KEYWORDS:
            if ck in norm_query:
                mapped_ck = CANONICAL_MAP.get(ck, ck)
                if normalize_text(mapped_ck) not in seen2:
                    final.append(mapped_ck)
                    seen2.add(normalize_text(mapped_ck))

        merged_final = []
        i = 0
        while i < len(final):
            cur = final[i]
            cur_k = normalize_text(cur)
            if i + 1 < len(final):
                nxt = final[i + 1]
                nxt_k = normalize_text(nxt)
                combo = f"{cur_k} {nxt_k}"
                if combo in CANONICAL_MAP or combo in norm_query:
                    chosen = CANONICAL_MAP.get(combo, combo)
                    merged_final.append(chosen)
                    i += 2
                    continue
            merged_final.append(cur)
            i += 1

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
                if all(t in norm_set for t in tokens):
                    to_remove = set(tokens)
                    new_list = []
                    removed = set()
                    for it, nk in zip(merged_final, norms):
                        if nk in to_remove and nk not in removed:
                            removed.add(nk)
                            continue
                        new_list.append(it)
                    if cm_v not in new_list:
                        new_list.append(cm_v)
                    merged_final = new_list
                    changed = True
                    break

        cleaned_final = []
        norms = [normalize_text(x) for x in merged_final]
        for idx, item in enumerate(merged_final):
            nk = norms[idx]
            if len(nk) <= 4:
                if nk in allow_short:
                    cleaned_final.append(item)
                    continue
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
