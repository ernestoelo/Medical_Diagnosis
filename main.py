import os
from dotenv import load_dotenv
from src.node_extractor import extract_nodes_from_query
from src.inference_engine import get_possible_diagnoses

def main():
    """
    Función principal que orquesta el pipeline de diagnóstico.
    """
    # Cargar la clave de API desde el archivo .env
    load_dotenv()
    if not os.getenv("HUGGINGFACE_API_TOKEN"):
        print("Error: La variable de entorno HUGGINGFACE_API_TOKEN no está configurada.")
        print("Por favor, crea un archivo .env y añade tu token: HUGGINGFACE_API_TOKEN='hf_..._aqui'")
        return

    # 1. Consulta del usuario (ejemplo)
    user_query = "Desde hace dos días tengo fiebre no muy alta, me duele la cabeza y siento el cuerpo cortado, como con dolor muscular."

    print(f"Analizando la consulta: '{user_query}'")
    print("-" * 30)

    # 2. Extracción de Nodos (Síntomas)
    extracted_symptoms = extract_nodes_from_query(user_query)
    
    if not extracted_symptoms:
        print("No se pudieron extraer síntomas de la consulta.")
        return

    print(f"Síntomas extraídos: {', '.join(extracted_symptoms)}")
    print("-" * 30)

    # 3. Motor de Inferencia para obtener diagnósticos
    print("Buscando posibles diagnósticos con la API de Hugging Face...")
    diagnoses = get_possible_diagnoses(extracted_symptoms)

    if not diagnoses:
        print("No se pudieron obtener diagnósticos para los síntomas proporcionados.")
        return
        
    print("-" * 30)
    print("Resultados del Diagnóstico Diferencial:")
    print("-" * 30)

    # 4. Mostrar resultados
    for i, diagnosis in enumerate(diagnoses, 1):
        print(f"Opción {i}: {diagnosis.get('diagnostico', 'N/A')}")
        print(f"  - Probabilidad estimada: {diagnosis.get('probabilidad', 'N/A')}")
        print(f"  - Justificación: {diagnosis.get('justificacion', 'N/A')}")
        print("\n")

if __name__ == "__main__":
    main()
