import json
import os
from datetime import datetime
from langchain_core.tools import tool

PEDIDOS = {
    "PED-1234": {"estado": "enviado", "fecha_entrega": "15/06/2026", "total": 89.99},
    "PED-5678": {"estado": "en preparación", "fecha_entrega": "18/06/2026", "total": 45.50},
    "PED-9999": {"estado": "entregado", "fecha_entrega": "01/06/2026", "total": 120.00},
    "PED-0001": {"estado": "cancelado", "fecha_entrega": "N/A", "total": 33.75},
}


@tool
def buscar_pedido(pedido_id: str) -> str:
    """Busca el estado de un pedido por su ID. Ejemplo: buscar_pedido('PED-1234')"""
    pedido = PEDIDOS.get(pedido_id.upper())
    if pedido:
        return (
            f"Pedido {pedido_id.upper()}: estado='{pedido['estado']}', "
            f"fecha de entrega estimada={pedido['fecha_entrega']}, "
            f"total={pedido['total']}€"
        )
    return f"Pedido {pedido_id} no encontrado. Verifica el ID e inténtalo de nuevo."


@tool
def calcular_reembolso(total: float, porcentaje: float) -> str:
    """Calcula el importe de un reembolso parcial dado el total y el porcentaje."""
    reembolso = round(total * porcentaje / 100, 2)
    return f"Reembolso del {porcentaje}% sobre {total}€: {reembolso}€"


@tool
def escalar_a_humano(motivo: str) -> str:
    """Registra el caso y lo escala a un agente humano cuando el asistente no puede resolver el problema."""
    casos_path = "casos_escalados.json"

    caso = {
        "timestamp": datetime.now().isoformat(),
        "motivo": motivo,
        "estado": "pendiente",
    }

    casos = []
    if os.path.exists(casos_path):
        with open(casos_path, "r", encoding="utf-8") as f:
            casos = json.load(f)

    casos.append(caso)

    with open(casos_path, "w", encoding="utf-8") as f:
        json.dump(casos, f, ensure_ascii=False, indent=2)

    return (
        f"Caso escalado correctamente. Un agente humano se pondrá en contacto contigo "
        f"en las próximas 2 horas hábiles. Motivo registrado: '{motivo}'"
    )
