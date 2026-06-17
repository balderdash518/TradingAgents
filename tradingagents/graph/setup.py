# TradingAgents/graph/setup.py

from collections.abc import Callable
from typing import Any

from langchain_core.messages import ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents import (
    create_aggressive_debator,
    create_bear_researcher,
    create_bull_researcher,
    create_conservative_debator,
    create_fundamentals_analyst,
    create_market_analyst,
    create_msg_delete,
    create_neutral_debator,
    create_news_analyst,
    create_portfolio_manager,
    create_research_manager,
    create_sentiment_analyst,
    create_trader,
)
from tradingagents.agents.utils.agent_states import AgentState

from .analyst_execution import build_analyst_execution_plan
from .conditional_logic import ConditionalLogic


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        tool_nodes: dict[str, ToolNode],
        conditional_logic: ConditionalLogic,
        analyst_concurrency_limit: int = 1,
        max_analyst_tool_rounds: int = 20,
    ):
        """Initialize with required components."""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.conditional_logic = conditional_logic
        self.analyst_concurrency_limit = analyst_concurrency_limit
        self.max_analyst_tool_rounds = max_analyst_tool_rounds

    def setup_graph(
        self, selected_analysts=("market", "news", "fundamentals")
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
                - "social": Sentiment analyst, optional
        """
        plan = build_analyst_execution_plan(
            selected_analysts,
            concurrency_limit=self.analyst_concurrency_limit,
        )

        analyst_factories = {
            "market": lambda: create_market_analyst(self.quick_thinking_llm),
            "social": lambda: create_sentiment_analyst(self.quick_thinking_llm),
            "news": lambda: create_news_analyst(self.quick_thinking_llm),
            "fundamentals": lambda: create_fundamentals_analyst(self.quick_thinking_llm),
        }

        # Create researcher and manager nodes
        bull_researcher_node = create_bull_researcher(self.quick_thinking_llm)
        bear_researcher_node = create_bear_researcher(self.quick_thinking_llm)
        research_manager_node = create_research_manager(self.deep_thinking_llm)
        trader_node = create_trader(self.quick_thinking_llm)

        # Create risk analysis nodes
        aggressive_analyst = create_aggressive_debator(self.quick_thinking_llm)
        neutral_analyst = create_neutral_debator(self.quick_thinking_llm)
        conservative_analyst = create_conservative_debator(self.quick_thinking_llm)
        portfolio_manager_node = create_portfolio_manager(self.deep_thinking_llm)

        # Create workflow
        workflow = StateGraph(AgentState)

        analyst_nodes = {
            spec.key: analyst_factories[spec.key]()
            for spec in plan.specs
        }
        parallel_analysts = plan.concurrency_limit > 1 and len(plan.specs) > 1

        # Add analyst nodes to the graph.
        #
        # The legacy graph runs analysts one-by-one and can safely use the
        # shared `messages` channel for tool loops. In parallel mode, each
        # analyst executes its own isolated tool loop inside a wrapper node and
        # only merges its report field back into the parent graph. This avoids
        # different analysts racing on "last message" tool-call routing.
        if parallel_analysts:
            for spec in plan.specs:
                workflow.add_node(
                    spec.agent_node,
                    self._create_isolated_analyst_node(
                        spec=spec,
                        analyst_node=analyst_nodes[spec.key],
                        tool_node=self.tool_nodes[spec.key],
                    ),
                )
            workflow.add_node("Msg Clear Analysts", create_msg_delete())
        else:
            for spec in plan.specs:
                workflow.add_node(spec.agent_node, analyst_nodes[spec.key])
                workflow.add_node(spec.clear_node, create_msg_delete())
                workflow.add_node(spec.tool_node, self.tool_nodes[spec.key])

        # Add other nodes
        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Aggressive Analyst", aggressive_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Conservative Analyst", conservative_analyst)
        workflow.add_node("Portfolio Manager", portfolio_manager_node)

        # Define edges
        if parallel_analysts:
            self._add_parallel_analyst_edges(workflow, plan)
        else:
            self._add_sequential_analyst_edges(workflow, plan)

        # Add remaining edges
        workflow.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bear Researcher": "Bear Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bull Researcher": "Bull Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Aggressive Analyst")
        workflow.add_conditional_edges(
            "Aggressive Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Conservative Analyst": "Conservative Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Conservative Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Neutral Analyst": "Neutral Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Aggressive Analyst": "Aggressive Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )

        workflow.add_edge("Portfolio Manager", END)

        return workflow

    def _create_isolated_analyst_node(
        self,
        spec,
        analyst_node: Callable[[dict[str, Any]], dict[str, Any]],
        tool_node: ToolNode,
    ) -> Callable[..., dict[str, Any]]:
        """Run one analyst's tool loop with a private message history.

        The ordinary graph stores all tool-call messages in the shared
        `messages` channel. That is fine when analysts run sequentially, but it
        is unsafe for parallel analysts because ToolNode dispatch reads the
        latest AI message. This wrapper keeps each analyst's messages local and
        returns only the analyst's report field to the parent graph.
        """

        def isolated_analyst_node(state, config=None):
            local_messages = list(state.get("messages", []))
            report = ""

            for _ in range(self.max_analyst_tool_rounds):
                local_state = dict(state)
                local_state["messages"] = list(local_messages)

                analyst_update = analyst_node(local_state)
                if not isinstance(analyst_update, dict):
                    raise TypeError(
                        f"{spec.agent_node} returned {type(analyst_update).__name__}; "
                        "expected dict"
                    )

                new_messages = list(analyst_update.get("messages", []))
                local_messages.extend(new_messages)
                report = analyst_update.get(spec.report_key, "")

                if report:
                    return {spec.report_key: report}

                last_message = new_messages[-1] if new_messages else (
                    local_messages[-1] if local_messages else None
                )
                tool_calls = getattr(last_message, "tool_calls", None)
                if not tool_calls:
                    return {spec.report_key: report}

                local_messages.extend(
                    self._invoke_tool_calls(tool_node, tool_calls, config=config)
                )

            raise RuntimeError(
                f"{spec.agent_node} exceeded max_analyst_tool_rounds="
                f"{self.max_analyst_tool_rounds}"
            )

        return isolated_analyst_node

    def _invoke_tool_calls(self, tool_node: ToolNode, tool_calls, config=None) -> list[ToolMessage]:
        messages = []
        for call in tool_calls:
            name = call["name"]
            tool = tool_node.tools_by_name.get(name)
            if tool is None:
                available = ", ".join(tool_node.tools_by_name)
                messages.append(
                    ToolMessage(
                        content=f"Error: {name} is not a valid tool. Available tools: {available}",
                        name=name,
                        tool_call_id=call["id"],
                        status="error",
                    )
                )
                continue

            try:
                response = tool.invoke({**call, "type": "tool_call"}, config=config)
            except Exception as exc:
                messages.append(
                    ToolMessage(
                        content=f"Error: {type(exc).__name__}: {exc}",
                        name=name,
                        tool_call_id=call["id"],
                        status="error",
                    )
                )
                continue

            if isinstance(response, ToolMessage):
                messages.append(response)
            else:
                messages.append(
                    ToolMessage(
                        content=str(response),
                        name=name,
                        tool_call_id=call["id"],
                    )
                )
        return messages

    def _add_parallel_analyst_edges(self, workflow: StateGraph, plan) -> None:
        specs = plan.specs
        limit = max(1, plan.concurrency_limit)
        batches = [specs[i:i + limit] for i in range(0, len(specs), limit)]

        for batch_index, batch in enumerate(batches):
            batch_nodes = [spec.agent_node for spec in batch]
            if batch_index == 0:
                for node in batch_nodes:
                    workflow.add_edge(START, node)
            else:
                previous_nodes = [spec.agent_node for spec in batches[batch_index - 1]]
                for node in batch_nodes:
                    workflow.add_edge(previous_nodes, node)

            if batch_index == len(batches) - 1:
                workflow.add_edge(batch_nodes, "Msg Clear Analysts")
                workflow.add_edge("Msg Clear Analysts", "Bull Researcher")

    def _add_sequential_analyst_edges(self, workflow: StateGraph, plan) -> None:
        # Start with the first analyst
        workflow.add_edge(START, plan.specs[0].agent_node)

        # Connect analysts in sequence
        for i, spec in enumerate(plan.specs):
            current_analyst = spec.agent_node
            current_tools = spec.tool_node
            current_clear = spec.clear_node

            # Add conditional edges for current analyst
            workflow.add_conditional_edges(
                current_analyst,
                getattr(self.conditional_logic, f"should_continue_{spec.key}"),
                [current_tools, current_clear],
            )
            workflow.add_edge(current_tools, current_analyst)

            # Connect to next analyst or to Bull Researcher if this is the last analyst
            if i < len(plan.specs) - 1:
                workflow.add_edge(current_clear, plan.specs[i + 1].agent_node)
            else:
                workflow.add_edge(current_clear, "Bull Researcher")
