from transformers import pipeline
import re
import unicodedata

# Simple standalone test extractor (won't modify the broken source file)

try:
    print('Cargando NER para prueba (puede tardar)...')
    ner = pipeline('ner', model='d4data/biomedical-ner-all', grouped_entities=True)
    print('NER cargado (test).')
except Exception as e:
    print('No se pudo cargar el NER:', e)
    ner = None


def normalize_text(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize('NFD', s)
    s = ''.join(ch for ch in s if unicodedata.category(ch) != 'Mn')
    return s.strip()


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


def extract_nodes_from_query_test(user_query: str):
    if ner is None:
        return []
    ner_results = ner(user_query)

    stop_words = {'desde', 'tengo', 'hace', 'como', 'con', 'me', 'siento'}
    symptom_keywords = [
        'fiebre', 'temperatura', 'dolor de cabeza', 'cefalea', 'migra', 'tos', 'diarrea',
        'nausea', 'nauseas', 'mareo', 'vertigo', 'fatiga', 'astenia', 'dolor', 'dolor muscular',
        'mialgia', 'dolor abdominal', 'dolor en el pecho', 'sangrado', 'palpitaciones',
        'dificultad respiratoria', 'tos seca', 'tos productiva', 'perdida de apetito', 'pérdida de apetito'
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
                clean = re.sub(r'^(desde hace|desde|tengo|tengo la|me )\\s*', '', pnorm)
                clean = re.sub(r'\\s+', ' ', clean).strip()
                heuristic_found.append(clean)
                break

    extracted_nodes = []
    for result in ner_results:
        w = result.get('word') or result.get('entity_group') or ''
        if not w:
            continue
        w = w.replace('\n', ' ').replace('\t', ' ').replace('▁', ' ')
        w = re.sub(r'\s+', ' ', w).strip()
        if not w:
            continue
        wl = normalize_text(w)
        if wl in stop_words:
            continue
        extracted_nodes.append(w)

    valid_ner = [w for w in extracted_nodes if is_valid_token(w)]

    filtered_extracted = []
    seen = set()
    for w in valid_ner:
        key = normalize_text(w)
        if key in seen:
            continue
        seen.add(key)
        filtered_extracted.append(w)

    combined = []
    if not filtered_extracted:
        combined = heuristic_found[:]
    else:
        combined = filtered_extracted[:]
        for h in heuristic_found:
            keyh = normalize_text(h)
            if keyh not in {normalize_text(x) for x in combined}:
                combined.append(h)

    unique_nodes = []
    seen2 = set()
    for n in combined:
        nn = n.strip()
        if not nn:
            continue
        key = unicodedata.normalize('NFD', nn).casefold()
        if key in seen2:
            continue
        seen2.add(key)
        unique_nodes.append(nn)

    return unique_nodes


if __name__ == '__main__':
    q = "Tengo fiebre, dolor de cabeza y dolores musculares desde ayer"
    print('INPUT:', q)
    out = extract_nodes_from_query_test(q)
    print('EXTRACTED:', out)
