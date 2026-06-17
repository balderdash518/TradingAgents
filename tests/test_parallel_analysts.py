from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from tradingagents.graph.analyst_execution import ANALYST_NODE_SPECS
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.setup import GraphSetup


@tool
def sample_tool(value: str) -> str:
    """Return a deterministic sample payload."""
    return f"tool:{value}"


class _DummyLLM:
    def bind_tools(self, tools):
        return self

    def invoke(self, _input):
        return AIMessage(content="dummy response")


def _tool_nodes():
    node = ToolNode([sample_tool])
    return {
        "market": node,
        "social": node,
        "news": node,
        "fundamentals": node,
    }


def test_isolated_parallel_analyst_keeps_messages_private():
    setup = GraphSetup(
        _DummyLLM(),
        _DummyLLM(),
        _tool_nodes(),
        ConditionalLogic(),
        analyst_concurrency_limit=3,
    )
    spec = ANALYST_NODE_SPECS["market"]
    calls = {"count": 0}

    def analyst_node(state):
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "sample_tool",
                                "args": {"value": "AAPL"},
                                "id": "call_sample_tool",
                            }
                        ],
                    )
                ],
                "market_report": "",
            }

        assert any(getattr(message, "name", "") == "sample_tool" for message in state["messages"])
        return {
            "messages": [AIMessage(content="market report")],
            "market_report": "market report",
        }

    node = setup._create_isolated_analyst_node(
        spec=spec,
        analyst_node=analyst_node,
        tool_node=ToolNode([sample_tool]),
    )

    result = node({"messages": [HumanMessage(content="AAPL")]})

    assert result == {"market_report": "market report"}
    assert "messages" not in result
    assert calls["count"] == 2


def test_parallel_analyst_graph_compiles():
    setup = GraphSetup(
        _DummyLLM(),
        _DummyLLM(),
        _tool_nodes(),
        ConditionalLogic(),
        analyst_concurrency_limit=3,
    )

    workflow = setup.setup_graph(["market", "news", "fundamentals"])

    workflow.compile()
