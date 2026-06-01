from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from agente import agente, checkpointer

app = FastAPI(title="Asistente de Soporte", version="1.0.0")


class MensajeRequest(BaseModel):
    session_id: str
    mensaje: str


@app.post("/chat")
def chat(request: MensajeRequest):
    config = {"configurable": {"thread_id": request.session_id}}
    resultado = agente.invoke(
        {"mensajes": [HumanMessage(content=request.mensaje)]},
        config=config,
    )
    return {"respuesta": resultado["mensajes"][-1].content}


@app.get("/chat/{session_id}/historial")
def historial(session_id: str):
    config = {"configurable": {"thread_id": session_id}}
    state = checkpointer.get(config)
    if state is None:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")

    mensajes = state["channel_values"].get("mensajes", [])
    historial_formateado = []
    for m in mensajes:
        if isinstance(m, HumanMessage):
            historial_formateado.append({"rol": "usuario", "contenido": m.content})
        elif isinstance(m, AIMessage) and m.content:
            historial_formateado.append({"rol": "asistente", "contenido": m.content})

    return {"session_id": session_id, "mensajes": historial_formateado}


@app.delete("/chat/{session_id}")
def limpiar_sesion(session_id: str):
    # MemorySaver es in-memory; en producción usar PostgresCheckpointer para borrado real
    return {"mensaje": f"Sesión {session_id} cerrada (memoria in-memory, se limpia al reiniciar)"}


@app.get("/health")
def health():
    return {"status": "ok"}
