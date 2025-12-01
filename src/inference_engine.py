import os
import json
import requests
from typing import List, Dict, Any, Optional
import unicodedata
import torch
import os
from typing import Callable
try:
    from transformers import pipeline
except Exception:
    pipeline = None
from .itoe_loader import load_metadata
from .itoe_inference import load_itoe_model, ItoE_Inference

def get_possible_diagnoses(nodes: List[str]) -> List[Dict[str, Any]]:
    """
    Usa la API de Inferencia de Hugging Face manualmente con `requests` para obtener
    posibles diagnósticos basados en una lista de síntomas (nodos).

    Args:
        nodes: Una lista de síntomas o condiciones médicas.

    Returns:
        Una lista de diccionarios, donde cada diccionario representa un posible
        diagnóstico con su justificación y probabilidad.
    """
    if not nodes:
        return []

    # Intentar usar un modelo ItôE local si existe
    project_root = os.path.dirname(os.path.dirname(__file__))
    model_dir = os.path.join(project_root, 'itoe_model')
    metadata_path = os.path.join(project_root, 'itoe_metadata.json')
    itoe_model: Optional[ItoE_Inference] = None
    metadata: Optional[Dict[str, Any]] = None
    try:
        if os.path.exists(model_dir):
            metadata = load_metadata(metadata_path) if os.path.exists(metadata_path) else None
            itoe_model = load_itoe_model(model_dir, metadata_path=metadata_path)
    except Exception as e:
        print(f"Advertencia: no se pudo cargar ItôE local: {e}")

    # Preferencia por uso local configurada vía env var `USE_LOCAL_ITOE` (default: true)
    use_local_itoe = os.getenv('USE_LOCAL_ITOE', '1').lower() in ('1', 'true', 'yes')

    symptoms_str = ", ".join(nodes)

    prompt = f"""<s>[INST] Actúa como un experto en diagnóstico diferencial.
Basándote en la siguiente lista de síntomas: [{symptoms_str}].

Proporciona un ranking de 3 posibles diagnósticos.
Para cada diagnóstico, incluye:
1. "diagnostico": El nombre de la enfermedad o condición.
2. "probabilidad": Una estimación de la probabilidad en porcentaje (ej. "75%").
3. "justificacion": Una breve explicación de por qué los síntomas encajan con ese diagnóstico.

Devuelve el resultado como un único objeto JSON (una lista de diccionarios). No incluyas absolutamente nada de texto antes o después del JSON.
[/INST]"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "return_full_text": False, # Importante para no recibir el prompt de vuelta
        }
    }

    # Si se prefiere el ItôE local y existe, úsalo antes que la API
    if use_local_itoe and itoe_model is not None and metadata is not None:
        try:
            return _itoe_diagnoses(nodes, itoe_model, metadata)
        except Exception as e:
            print(f"Error usando ItôE local para inferencia (preferido por USE_LOCAL_ITOE): {e}")

    token = os.getenv("HUGGINGFACE_API_TOKEN")
    repo_id = os.getenv("HUGGINGFACE_MODEL")  # ejemplo: 'gpt2' o 'bigscience/bloom'
    use_api = bool(token and repo_id)
    if use_api:
        # Preferir el endpoint de inference API; si el usuario necesita usar router, puede configurar HUGGINGFACE_API_ENDPOINT
        api_endpoint = os.getenv("HUGGINGFACE_API_ENDPOINT", "https://api-inference.huggingface.co/models")
        api_url = f"{api_endpoint}/{repo_id}"
        headers = {"Authorization": f"Bearer {token}"}

    try:
        response = requests.post(api_url, headers=headers, json=payload)

        # Manejo explícito de 404 (modelo no encontrado)
        if response.status_code == 404:
            print(f"Error: Modelo '{repo_id}' no encontrado en el endpoint {api_url} (404).")
            print("Comprueba que el nombre del modelo sea correcto y que tu token tenga permisos de acceso.")
            print("Sugerencia: establece la variable de entorno `HUGGINGFACE_MODEL` a un modelo disponible o eliminala para usar el fallback local.")
            return _rule_based_diagnoses(nodes)

        response.raise_for_status()  # Lanza un error para otras respuestas 4xx/5xx

        response_data = response.json()

        # La respuesta puede venir en diferentes formatos según el modelo/endpoint
        # Intentamos extraer texto generado de varias formas seguras
        generated_text = None
        if isinstance(response_data, list) and response_data:
            # formato: [{"generated_text": "..."}]
            generated_text = response_data[0].get('generated_text') or response_data[0].get('generated_text', None)
        elif isinstance(response_data, dict):
            # algunos endpoints devuelven {'generated_text': '...'} o {'text': '...'}
            generated_text = response_data.get('generated_text') or response_data.get('text') or response_data.get('generated_text', None)

        if not generated_text:
            # Si no viene texto, intentamos usar la respuesta cruda como cadena
            generated_text = json.dumps(response_data)

        # A veces el modelo aún puede devolver un JSON malformado, lo limpiamos
        json_start_index = generated_text.find('[')
        if json_start_index == -1:
            print(f"Error: La respuesta del LLM no contenía un JSON válido. Respuesta: {generated_text}")
            return _rule_based_diagnoses(nodes)

        json_str = generated_text[json_start_index:]

        return json.loads(json_str)

    except requests.exceptions.RequestException as e:
        print(f"Error en la llamada a la API de Hugging Face: {e}")
        # Intenta imprimir más detalles si están disponibles en la respuesta
        if 'response' in locals() and hasattr(response, 'text') and response.text:
            print(f"Respuesta del servidor: {response.text}")
        return _rule_based_diagnoses(nodes)
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error al procesar la respuesta JSON del LLM: {e}")
        if 'response_data' in locals():
            print(f"Respuesta recibida: {response_data}")
        return _rule_based_diagnoses(nodes)
    except Exception as e:
        print(f"Ha ocurrido un error inesperado en la inferencia: {e}")
        return _rule_based_diagnoses(nodes)


def _rule_based_diagnoses(nodes: List[str]) -> List[Dict[str, Any]]:
    """
    Fallback simple basado en reglas para ofrecer diagnósticos aproximados
    cuando la API de Hugging Face no está disponible o falla.
    """
    s = " ".join(nodes).lower()
    results: List[Dict[str, Any]] = []

    # Regla 1: fiebre + dolor muscular/cuerpo cortado -> infección viral (gripe, etc.)
    if 'fiebre' in s and ('dolor muscular' in s or 'cuerpo cortado' in s or 'dolor' in s):
        results.append({
            'diagnostico': 'Infección viral (por ejemplo, gripe)',
            'probabilidad': '60%',
            'justificacion': 'Fiebre con dolores musculares y sensación de cuerpo cortado es típica de infecciones virales agudas.'
        })

    # Regla 2: dolor de cabeza prominente
    if 'dolor de cabeza' in s or 'cefalea' in s:
        results.append({
            'diagnostico': 'Cefalea tensional o migraña',
            'probabilidad': '25%',
            'justificacion': 'El dolor de cabeza es un síntoma común en cefaleas tensionales y migrañas; la presencia de fiebre apunta más hacia infección, por eso menor probabilidad.'
        })

    # Regla 3: si hay fiebre persistente considerar infección bacteriana o complicación
    if 'fiebre' in s and 'alta' in s:
        results.insert(0, {
            'diagnostico': 'Infección bacteriana (posible)',
            'probabilidad': '30%',
            'justificacion': 'Fiebre alta sostenida puede indicar infección bacteriana que requiere evaluación médica.'
        })

    # Si no hay resultados suficientes, añadir alternativas genéricas
    if not results:
        results = [
            {
                'diagnostico': 'Síntomas inespecíficos',
                'probabilidad': '50%',
                'justificacion': 'Los síntomas descritos son vagos; se recomienda evaluación médica y monitorización.'
            }
        ]

    # Limitar a 3 resultados
    return results[:3]


def _itoe_diagnoses(nodes: List[str], model: ItoE_Inference, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Usa el modelo ItôE local para generar diagnósticos.
    Espera `metadata` con `ent2id` y `rel2id`.
    """
    ent2id = metadata.get('ent2id', {})
    id2ent = {v: k for k, v in ent2id.items()}
    rel2id = metadata.get('rel2id', {})

    # Normalización ligera (quita tildes, pasa a minúsculas)
    def normalize(s: str) -> str:
        s = s.lower().strip()
        s = unicodedata.normalize('NFD', s)
        s = ''.join(ch for ch in s if unicodedata.category(ch) != 'Mn')
        return s

    norm_ent = {normalize(k): v for k, v in ent2id.items()}

    symptom_ids = []
    for token in nodes:
        t = normalize(token)
        # exact match
        if t in norm_ent:
            symptom_ids.append(norm_ent[t])
            continue
        # substring match (only for longer tokens)
        if len(t) > 3:
            for ent_norm, idx in norm_ent.items():
                if t in ent_norm:
                    symptom_ids.append(idx)
                    break

    # If no symptom IDs found, try a small Spanish->English symptom map
    if not symptom_ids:
        quick_trans = {
            'fiebre': 'fever',
            'fiebre alta': 'fever',
            'temperatura elevada': 'fever',
            'dolor de cabeza': 'headache',
            'cefalea': 'headache',
            'migraña': 'migraine',
            'dolor muscular': 'muscle pain',
            'mialgia': 'muscle pain',
            'cuerpo cortado': 'body ache',
            'tos': 'cough',
            'tos seca': 'dry cough',
            'tos productiva': 'productive cough',
            'diarrea': 'diarrhea',
            'náuseas': 'nausea',
            'mareo': 'dizziness',
            'vértigo': 'dizziness',
            'fatiga': 'fatigue',
            'astenia': 'fatigue',
            'dolor abdominal': 'abdominal pain',
            'dolor en el pecho': 'chest pain',
            'dolor torácico': 'chest pain',
            'sangrado': 'bleeding',
            'palpitaciones': 'palpitations',
            'pérdida de apetito': 'loss of appetite',
            'pérdida de peso': 'weight loss',
            'problemas respiratorios': 'breathing problems',
            'dificultad respiratoria': 'shortness of breath'
        }

        for token in nodes:
            t = normalize(token)
            for k, v in quick_trans.items():
                if normalize(k) == t or (len(t) > 3 and normalize(k) in t) or (len(t) > 3 and t in normalize(k)):
                    # try to find the translated english form in the ent vocabulary
                    trans = normalize(v)
                    # direct match
                    if trans in norm_ent:
                        symptom_ids.append(norm_ent[trans])
                        break
                    # substring match in ent names
                    for ent_norm, idx in norm_ent.items():
                        if trans in ent_norm:
                            symptom_ids.append(idx)
                            break
            if symptom_ids:
                break

    # If still no symptom IDs, try automatic translation (Helsinki opus-mt-es-en)
    if not symptom_ids and os.getenv('ENABLE_ES2EN_TRANSLATION', '1') == '1' and pipeline is not None:
        # lazy translator
        _translator = None
        def get_translator() -> Callable[[str], str]:
            nonlocal _translator
            if _translator is None:
                try:
                    _translator = pipeline('translation', model='Helsinki-NLP/opus-mt-es-en')
                except Exception as e:
                    print(f"Advertencia: no se pudo inicializar el traductor: {e}")
                    _translator = None
            return _translator

        for token in nodes:
            try:
                translator = get_translator()
                if translator is None:
                    break
                res = translator(token, max_length=256)
                if isinstance(res, list) and res:
                    trans_text = res[0].get('translation_text', '')
                elif isinstance(res, dict):
                    trans_text = res.get('translation_text', '')
                else:
                    trans_text = str(res)
                tnorm = normalize(trans_text)
                if tnorm in norm_ent:
                    symptom_ids.append(norm_ent[tnorm])
                    break
                if len(tnorm) > 3:
                    for ent_norm, idx in norm_ent.items():
                        if tnorm in ent_norm:
                            symptom_ids.append(idx)
                            break
                if symptom_ids:
                    break
            except Exception as e:
                print(f"Advertencia: fallo en traducción automática: {e}")
                break

    if not symptom_ids:
        return _rule_based_diagnoses(nodes)

    # preparar tensores
    num_ent = len(ent2id)
    todos_nodos = torch.arange(num_ent, dtype=torch.long)
    # Preferir relación 'association' si existe
    if 'association' in rel2id:
        r_id = rel2id['association']
    elif 'co_morbidity' in rel2id:
        r_id = rel2id['co_morbidity']
    else:
        # fallback: primer valor
        r_id = next(iter(rel2id.values())) if rel2id else 0

    r_tensor = torch.tensor([r_id] * num_ent, dtype=torch.long)

    energia_total = torch.zeros(num_ent, dtype=torch.float32)
    with torch.no_grad():
        for s_id in symptom_ids:
            h_tensor = torch.tensor([s_id] * num_ent)
            energia = model.calcular_energia(h_tensor, r_tensor, todos_nodos)
            energia_total += energia

    # Usar logits (-energía) para seleccionar candidatos y renormalizar con softmax sobre top-N
    logits = -energia_total
    top_k = min(50, num_ent)
    top_logits, indices = torch.topk(logits, k=top_k)

    resultados = []
    # construir resultados filtrando entidades idénticas a los síntomas y
    # priorizando entidades que parezcan enfermedades (heurística mejorada)
    def looks_like_disease(name: str) -> bool:
        low = name.lower()
        keywords = [
            'disease', 'syndrome', 'infection', 'fever', 'pneumonia', 'cancer',
            'neoplasm', 'sepsis', 'diabetes', 'hepatitis', 'anemia', 'pain',
            'viral', 'bacterial', 'unspecified', 'acute', 'chronic', 'disorder'
        ]
        for k in keywords:
            if k in low:
                return True
        # also check for common clinical suffixes
        clinical = ['itis', 'oma', 'covid', 'influenza', 'bronchitis', 'pneumonitis']
        for c in clinical:
            if c in low:
                return True
        return False

    # Renormalizar sobre top-N para presentar porcentajes interpretable
    top_n = min(10, top_logits.size(0))
    top_logits_n = top_logits[:top_n]
    top_indices = indices[:top_n]
    renormed = torch.softmax(top_logits_n, dim=0)
    # Filtrar los candidatos que coinciden con los síntomas o son ruido, luego renormalizar
    filtered_logits = []
    filtered_indices = []
    for i in range(top_n):
        idx_val = int(top_indices[i].item())
        nombre = id2ent.get(idx_val, str(idx_val))
        if idx_val in symptom_ids:
            continue
        if len(nombre) < 4:
            continue
        filtered_logits.append(float(top_logits_n[i].item()))
        filtered_indices.append(idx_val)

    if not filtered_logits:
        return _rule_based_diagnoses(nodes)

    # renormalizar sobre los logits filtrados
    tensor_logits = torch.tensor(filtered_logits, dtype=torch.float32)
    renormed = torch.softmax(tensor_logits, dim=0)

    for ren_score, idx_val in zip(renormed, filtered_indices):
        nombre = id2ent.get(idx_val, str(idx_val))
        resultados.append({
            'diagnostico': nombre,
            'probabilidad': f"{float(ren_score.item()):.4%}",
            'justificacion': 'Basado en las relaciones y embeddings estocásticos del grafo.'
        })
        disease_like = [r for r in resultados if looks_like_disease(r['diagnostico'])]
        if len(disease_like) >= 3:
            break
        if len(resultados) >= 10:
            break

    if not resultados:
        return _rule_based_diagnoses(nodes)

    return resultados
