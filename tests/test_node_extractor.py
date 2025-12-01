from src.node_extractor import extract_nodes_from_query


TEST_CASES = [
    (
        "Tengo fiebres altas y cefalea, me siento mareado",
        ["fiebre", "dolor de cabeza", "mareo"],
    ),
    (
        "Tengo fiebre, dolor de cabeza y dolores musculares desde ayer",
        ["fiebre", "dolor de cabeza", "dolor muscular"],
    ),
    (
        "Desde hace dos días tengo náuseas y vómitos",
        ["nausea"],
    ),
    (
        "Me duele el pecho y tengo dificultad respiratoria",
        ["dolor en el pecho", "dificultad respiratoria"],
    ),
    (
        "Tos seca persistente desde hace 3 días",
        ["tos"],
    ),
    (
        "Se me inflama la garganta y tengo tos productiva",
        ["dolor de garganta", "tos"],
    ),
    (
        "Me duele mucho la garganta y tengo tos productiva",
        ["dolor de garganta", "tos"],
    ),
    (
        "He tenido escalofríos y visión borrosa",
        ["escalofrio", "vision borrosa"],
    ),
    (
        "Me siento con náuseas, vómitos y pérdida de apetito",
        ["nausea", "vomitos", "perdida de apetito"],
    ),
]


def run_tests():
    failures = []
    for idx, (q, expected) in enumerate(TEST_CASES, 1):
        out = extract_nodes_from_query(q)
        out_norm = [s.lower() for s in out]
        missing = [e for e in expected if not any(e in s for s in out_norm)]
        if missing:
            failures.append((q, expected, out))
            print(f"[FAIL] Case {idx}: missing {missing}. Output: {out}")
        else:
            print(f"[ OK ] Case {idx}: matched. Output: {out}")

    if failures:
        print(f"\n{len(failures)} test(s) failed.")
        return 1
    print("\nAll tests passed.")
    return 0


if __name__ == '__main__':
    raise SystemExit(run_tests())
