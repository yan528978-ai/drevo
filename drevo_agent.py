"""
Древо — минимальный рабочий скелет слоя "Ветви".

Требования:
    pip install smolagents litellm requests feedparser pydantic networkx schedule --no-cache-dir
    ollama pull qwen2.5:14b
    ollama serve

Запуск:
    python drevo_agent.py           # один цикл и выход
    python drevo_agent.py --loop    # регулярный запуск (раз в час, см. INTERVAL_HOURS)
"""

import sys
import time
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import networkx as nx
import schedule
from pydantic import BaseModel, Field
from smolagents import CodeAgent, LiteLLMModel, tool
import requests
import feedparser


class Cycle(BaseModel):
    """Один цикл Древа: Наблюдение -> Анализ -> Решение -> Действие -> Результат -> Обратная связь."""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    task: str
    result: str
    source: Optional[str] = None
    feedback: Optional[str] = None


STATE_FILE = Path("drevo_state.jsonl")


def save_cycle(cycle: Cycle) -> None:
    with STATE_FILE.open("a", encoding="utf-8") as f:
        f.write(cycle.model_dump_json() + "\n")


@tool
def fetch_url_text(url: str) -> str:
    """Скачивает страницу по URL и возвращает первые 2000 символов текста.

    Args:
        url: адрес страницы для загрузки.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text[:2000]
    except Exception as e:
        return f"Ошибка загрузки {url}: {e}"


@tool
def fetch_rss_headlines(feed_url: str, limit: int = 10) -> str:
    """Читает RSS-ленту и возвращает список последних заголовков с кратким описанием.

    Args:
        feed_url: адрес RSS-ленты.
        limit: сколько последних записей вернуть.
    """
    try:
        feed = feedparser.parse(feed_url)
        if feed.bozo and not feed.entries:
            return f"Не удалось разобрать ленту {feed_url}: {feed.bozo_exception}"
        lines = []
        for entry in feed.entries[:limit]:
            title = entry.get("title", "без заголовка")
            summary = entry.get("summary", "")[:200]
            lines.append(f"- {title}: {summary}")
        return "\n".join(lines) if lines else "Записи не найдены"
    except Exception as e:
        return f"Ошибка чтения ленты {feed_url}: {e}"


def build_agent() -> CodeAgent:
    model = LiteLLMModel(model_id="ollama/qwen2.5:14b", api_base="http://localhost:11434")
    return CodeAgent(tools=[fetch_url_text, fetch_rss_headlines], model=model)


def observe(feed_url: str, limit: int = 10) -> str:
    return fetch_rss_headlines(feed_url=feed_url, limit=limit)


def orient_and_decide(agent: CodeAgent, observation: str, question: str) -> str:
    prompt = (
        f"Вот собранные данные:\n{observation}\n\n"
        f"Задача: {question}\n"
        "Отвечай только на основе приведённых данных, не придумывай факты."
    )
    return str(agent.run(prompt))


def act(decision: str) -> None:
    print("\n--- Действие (Act) ---")
    print(decision)


def run_ooda_cycle(agent: CodeAgent, feed_url: str, question: str) -> Cycle:
    observation = observe(feed_url)
    decision = orient_and_decide(agent, observation, question)
    act(decision)
    cycle = Cycle(task=question, result=decision, source=feed_url)
    save_cycle(cycle)
    update_knowledge_graph(decision, cycle.timestamp)
    return cycle


GRAPH_FILE = Path("drevo_graph.json")


def load_graph() -> nx.Graph:
    if GRAPH_FILE.exists():
        data = json.loads(GRAPH_FILE.read_text(encoding="utf-8"))
        return nx.node_link_graph(data)
    return nx.Graph()


def save_graph(graph: nx.Graph) -> None:
    data = nx.node_link_data(graph)
    GRAPH_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def update_knowledge_graph(decision: str, timestamp: datetime) -> nx.Graph:
    topics = [t.strip() for t in decision.split(",") if t.strip()]
    graph = load_graph()
    for topic in topics:
        if graph.has_node(topic):
            graph.nodes[topic]["count"] = graph.nodes[topic].get("count", 1) + 1
            graph.nodes[topic]["last_seen"] = timestamp.isoformat()
        else:
            graph.add_node(topic, count=1, first_seen=timestamp.isoformat(), last_seen=timestamp.isoformat())
    for i in range(len(topics)):
        for j in range(i + 1, len(topics)):
            a, b = topics[i], topics[j]
            if graph.has_edge(a, b):
                graph[a][b]["weight"] += 1
            else:
                graph.add_edge(a, b, weight=1)
    save_graph(graph)
    return graph


INTERVAL_HOURS = 1


def run_once() -> None:
    agent = build_agent()
    feed_url = "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml"
    question = "Выдели главные экологические угрозы или события из этих новостей"
    cycle = run_ooda_cycle(agent, feed_url, question)
    print(f"\nЦикл сохранён в {STATE_FILE.resolve()}")
    graph = load_graph()
    print(f"Граф знаний: {graph.number_of_nodes()} тем, {graph.number_of_edges()} связей "
          f"(файл: {GRAPH_FILE.resolve()})")


if __name__ == "__main__":
    if "--loop" in sys.argv:
        print(f"Режим регулярного запуска: каждые {INTERVAL_HOURS} ч. Остановить — Ctrl+C.")
        run_once()
        schedule.every(INTERVAL_HOURS).hours.do(run_once)
        while True:
            schedule.run_pending()
            time.sleep(60)
    else:
        run_once()
