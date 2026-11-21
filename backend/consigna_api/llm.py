"""Modelo de lenguaje por API (Anthropic Messages). Es opcional y nunca calcula montos ni fechas.

Si el servicio falla o devuelve algo sospechoso, quien llama conserva el texto
determinista original: no se pierden datos ni se bloquea la plataforma.
"""
from __future__ import annotations

import json
import re

import requests

API_URL = "https://api.anthropic.com/v1/messages"


class LLMNoDisponible(Exception):
    pass


class ClienteLLM:
    def __init__(self, api_key: str, modelo: str, url: str = API_URL, timeout: int = 60):
        self.api_key, self.modelo, self.url, self.timeout = api_key, modelo, url, timeout

    @property
    def configurado(self) -> bool:
        return bool(self.api_key)

    def completar(self, system: str, usuario: str, max_tokens: int = 2000) -> str:
        if not self.api_key:
            raise LLMNoDisponible("El modelo de lenguaje no está configurado.")
        try:
            r = requests.post(
                self.url, timeout=self.timeout,
                headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": self.modelo, "max_tokens": max_tokens, "system": system,
                      "messages": [{"role": "user", "content": usuario}]})
            r.raise_for_status()
            return "".join(b.get("text", "") for b in r.json()["content"] if b.get("type") == "text").strip()
        except (requests.RequestException, KeyError, ValueError) as e:
            raise LLMNoDisponible(f"El servicio del modelo de lenguaje no respondió ({type(e).__name__}).") from e


_NUM = re.compile(r"\d[\d.,]*")


def _numeros(texto: str) -> set[str]:
    return {n.rstrip(".,") for n in _NUM.findall(texto)}


def sin_cifras_nuevas(original: str, nuevo: str) -> bool:
    """El modelo no puede introducir cifras, fechas ni montos que no estaban en el original."""
    return _numeros(nuevo) <= _numeros(original)


def mejorar_explicaciones(hallazgos: list[dict], llm) -> list[dict]:
    """Reescribe explicación y alternativa en lenguaje claro. No recibe el contrato, solo los
    hallazgos; no puede cambiar niveles de riesgo. Lanza LLMNoDisponible ante cualquier problema."""
    base = [{"id": h["id"], "titulo": h["title"], "nivel": h["level"], "explicacion": h["explanation"], "alternativa": h["alternative"]} for h in hallazgos]
    system = ("Reescribes explicaciones de riesgo de contratos de consignación de vehículos en Chile, en español claro "
              "para una persona sin formación jurídica. No cambies el nivel de riesgo, no agregues cifras, plazos ni citas "
              "legales, y no des asesoría jurídica. Responde SOLO un JSON: "
              '{"<id>": {"explicacion": "...", "alternativa": "..."}} con los mismos ids recibidos.')
    crudo = llm.completar(system, json.dumps(base, ensure_ascii=False))
    inicio, fin = crudo.find("{"), crudo.rfind("}")
    try:
        nuevo = json.loads(crudo[inicio:fin + 1])
    except ValueError as e:
        raise LLMNoDisponible("La respuesta del modelo no fue un JSON válido.") from e
    salida = []
    for h in hallazgos:
        r = nuevo.get(h["id"]) if isinstance(nuevo, dict) else None
        h = dict(h)
        if isinstance(r, dict):
            for campo, clave in (("explanation", "explicacion"), ("alternative", "alternativa")):
                t = r.get(clave)
                if isinstance(t, str) and 10 <= len(t) <= 1200 and sin_cifras_nuevas(h[campo], t):
                    h[campo] = t
        salida.append(h)
    return salida


def pulir_borrador(borrador: str, llm) -> str:
    system = ("Mejoras la redacción de borradores de reclamo en español formal y claro. Conserva EXACTAMENTE todas las cifras, "
              "fechas, nombres, patentes y citas normativas; no agregues fundamentos jurídicos ni afirmaciones nuevas; "
              "mantén los campos en blanco (____). Devuelve solo el texto del borrador.")
    nuevo = llm.completar(system, borrador, max_tokens=2500)
    if not nuevo or not sin_cifras_nuevas(borrador, nuevo):
        raise LLMNoDisponible("El texto devuelto por el modelo modificaba cifras o fechas y se descartó.")
    return nuevo
