import operator
from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from tools import buscar_pedido, calcular_reembolso, escalar_a_humano

load_dotenv()


class EstadoSoporte(TypedDict):
    mensajes: Annotated[Sequence[BaseMessage], operator.add]


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

tools = [buscar_pedido, calcular_reembolso, escalar_a_humano]
modelo = ChatOpenAI(model="gpt-4o", temperature=0)
modelo_con_tools = modelo.bind_tools(tools)


def nodo_llm(estado: EstadoSoporte) -> dict:
    ultimo_humano = next(
        (m.content for m in reversed(estado["mensajes"]) if isinstance(m, HumanMessage)),
        "",
    )
    docs = retriever.invoke(ultimo_humano)
    contexto = "\n\n".join(d.page_content for d in docs)

    system = SystemMessage(
        content=f"""Eres un asistente de soporte al cliente amable, preciso y empático.

Tienes acceso a las siguientes herramientas:
- buscar_pedido: consulta el estado de un pedido por su ID (formato PED-XXXX)
- calcular_reembolso: calcula el importe de un reembolso parcial
- escalar_a_humano: registra el caso y lo escala a un agente humano

Responde preguntas sobre políticas usando este contexto de la base de conocimiento:

{contexto}

Reglas:
- Si el usuario pregunta por un pedido, usa buscar_pedido con el ID que indique.
- Si necesitas calcular un reembolso, usa calcular_reembolso.
- Si el problema no tiene solución automática o el cliente lo solicita, usa escalar_a_humano.
- Si no tienes información suficiente, dilo claramente. No inventes datos."""
    )

    mensajes_con_system = [system] + list(estado["mensajes"])
    respuesta = modelo_con_tools.invoke(mensajes_con_system)
    return {"mensajes": [respuesta]}


def debe_continuar(estado: EstadoSoporte) -> str:
    ultimo = estado["mensajes"][-1]
    if hasattr(ultimo, "tool_calls") and ultimo.tool_calls:
        return "usar_tool"
    return END


nodo_tools = ToolNode(tools)

grafo = StateGraph(EstadoSoporte)
grafo.add_node("llm", nodo_llm)
grafo.add_node("tools", nodo_tools)
grafo.set_entry_point("llm")
grafo.add_conditional_edges("llm", debe_continuar, {"usar_tool": "tools", END: END})
grafo.add_edge("tools", "llm")

checkpointer = MemorySaver()
agente = grafo.compile(checkpointer=checkpointer)
