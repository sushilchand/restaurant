"""LangGraph restaurant ordering workflow and command-line interface."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from groq import Groq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from restaurant.constants import GROQ_API_KEY, LLM_MODEL_NAME

DEFAULT_MODEL = "llama-3.3-70b-versatile"


class RestaurantState(TypedDict, total=False):
    customer_request: str
    messages: Annotated[list[AnyMessage], add_messages]
    order: list[dict[str, Any]]
    status: str
    order_status: str
    customer_message: str
    total: str
    invoice: dict[str, Any]
    refund: dict[str, Any]
    refund_amount: str
    attempt_counts: dict[str, int]
    completed_nodes: list[str]
    cook_failures: int
    serve_failures: int
    last_error: str


@dataclass
class MenuItem:
    name: str
    quantity: int
    price: Decimal


class MenuCatalog:
    def __init__(self, menu_path: Path) -> None:
        with menu_path.open(encoding="utf-8") as menu_file:
            data = json.load(menu_file)

        self.currency = str(data.get("currency", "USD"))
        self.items: dict[str, MenuItem] = {}
        for entry in data["items"]:
            item = MenuItem(
                name=str(entry["name"]),
                quantity=int(entry["quantity"]),
                price=Decimal(str(entry["price"])),
            )
            key = item.name.casefold()
            if key in self.items:
                raise ValueError(f"Duplicate menu item: {item.name}")
            if item.quantity < 0 or item.price < 0:
                raise ValueError(f"Invalid stock or price for {item.name}")
            self.items[key] = item

        self._lock = Lock()

    def names(self) -> list[str]:
        return [item.name for item in self.items.values()]

    def reserve(self, order: list[dict[str, Any]]) -> dict[str, int]:
        """Atomically reserve all lines, returning any unavailable quantities."""
        with self._lock:
            shortages: dict[str, int] = {}
            for line in order:
                item = self.items[line["name"].casefold()]
                if line["quantity"] > item.quantity:
                    shortages[item.name] = item.quantity

            if shortages:
                return shortages

            for line in order:
                item = self.items[line["name"].casefold()]
                item.quantity -= line["quantity"]
            return {}

    def release(self, order: list[dict[str, Any]]) -> None:
        """Return reserved stock when preparation fails before cooking succeeds."""
        with self._lock:
            for line in order:
                item = self.items[line["name"].casefold()]
                item.quantity += line["quantity"]


class InvalidOrderError(ValueError):
    """Raised when a customer request contains an invalid item or quantity."""


class OrderInterpreter:
    def __init__(self, menu: MenuCatalog, api_key: str | None, model: str) -> None:
        self.menu = menu
        self.model = model
        self.client = Groq(api_key=api_key) if api_key else None

    def interpret(
        self, request: str, messages: list[AnyMessage] | None = None
    ) -> list[dict[str, Any]]:
        if self.client is None:
            return self._interpret_locally(request)

        conversation = []
        for message in messages or []:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                continue
            conversation.append({"role": role, "content": str(message.content)})

        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract every item and quantity from the full conversation, "
                        "using the latest user message to revise earlier requests. "
                        "Return JSON with boolean 'valid', an 'items' array, and an "
                        "optional 'reason'. Every item must exactly match a menu "
                        "name and have a positive integer quantity. If any requested "
                        "item is unknown or any quantity is invalid, set valid to "
                        "false and do not silently omit it. Available items: "
                        f"{', '.join(self.menu.names())}."
                    ),
                },
                *conversation,
            ],
        )
        content = completion.choices[0].message.content
        if not content:
            raise ValueError("The language model returned an empty response")
        result = json.loads(content)
        if isinstance(result, dict) and result.get("valid") is False:
            raise InvalidOrderError(
                str(result.get("reason") or "The order contains an invalid item or quantity.")
            )
        items = result.get("items") if isinstance(result, dict) else None
        if not isinstance(items, list):
            raise ValueError("The language model response did not contain an items list")
        return items

    def _interpret_locally(self, request: str) -> list[dict[str, Any]]:
        quantity_words = {
            "zero": 0,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
        }
        item_patterns = []
        known_tokens = set()
        for item in self.menu.items.values():
            name = re.escape(item.name)
            if not item.name.casefold().endswith("s"):
                name += "s?"
                known_tokens.add(f"{item.name.casefold()}s")
            item_patterns.append((item, name))
            known_tokens.update(re.findall(r"[a-z]+", item.name.casefold()))

        if re.search(r"\b(?:negative|minus)\b", request, re.IGNORECASE):
            raise InvalidOrderError("Quantities must be positive integers.")

        for item, name in item_patterns:
            quantity_before_or_after = (
                rf"(?<![\w.])(?:-\s*\d+(?:\.\d+)?|\d+\.\d+|0+)\s*(?:x\s*)?{name}\b"
                rf"|\b{name}\s*(?:x\s*)?(?:-\s*\d+(?:\.\d+)?|\d+\.\d+|0+)(?!\w)"
                rf"|\bzero\s*(?:x\s*)?{name}\b"
            )
            if re.search(quantity_before_or_after, request, re.IGNORECASE):
                raise InvalidOrderError(
                    f"Quantity for {item.name} must be a positive integer."
                )

        allowed_words = {
            "i", "d", "would", "like", "want", "please", "give", "me", "can",
            "could", "get", "have", "order", "to", "and", "also", "some", "a",
            "an", "for", "of", "the", "with", "then", "that", "make", "it", "x",
            *quantity_words,
        }
        unknown_items = sorted(
            {
                word
                for word in re.findall(r"[a-z]+", request.casefold())
                if word not in known_tokens and word not in allowed_words
            }
        )
        if unknown_items:
            raise InvalidOrderError(
                f"Unknown menu item(s): {', '.join(unknown_items)}."
            )

        lines: list[dict[str, Any]] = []
        for item, name in sorted(item_patterns, key=lambda pair: len(pair[0].name), reverse=True):
            numeric_before = re.search(
                rf"\b(\d+)\s*(?:x\s*)?{name}\b", request, re.IGNORECASE
            )
            numeric_after = re.search(
                rf"\b{name}\s*(?:x\s*)?(\d+)\b", request, re.IGNORECASE
            )
            word_pattern = "|".join(sorted(quantity_words, key=len, reverse=True))
            word_before = re.search(
                rf"\b({word_pattern})\s*(?:x\s*)?{name}\b", request, re.IGNORECASE
            )
            word_after = re.search(
                rf"\b{name}\s*(?:x\s*)?({word_pattern})\b", request, re.IGNORECASE
            )
            match = numeric_before or numeric_after
            if match:
                lines.append({"name": item.name, "quantity": int(match.group(1))})
            elif word_before or word_after:
                word_match = word_before or word_after
                lines.append(
                    {"name": item.name, "quantity": quantity_words[word_match.group(1).casefold()]}
                )
            elif re.search(rf"\b{name}\b", request, re.IGNORECASE):
                lines.append({"name": item.name, "quantity": 1})
        return lines


class RestaurantWorkflow:
    def __init__(
        self,
        menu_path: Path | None = None,
        *,
        model: str | None = None,
        api_key: str | None = None,
        max_attempts: int = 3,
        base_delay: float = 0.5,
        sleeper: Callable[[float], None] = time.sleep,
        operations: dict[str, Callable[..., Any]] | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if base_delay < 0:
            raise ValueError("base_delay cannot be negative")

        default_menu = Path(__file__).with_name("menu.json")
        self.menu = MenuCatalog(menu_path or default_menu)
        effective_api_key = GROQ_API_KEY if api_key is None else api_key
        effective_model = model or LLM_MODEL_NAME or DEFAULT_MODEL
        self.interpreter = OrderInterpreter(self.menu, effective_api_key, effective_model)
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.sleeper = sleeper
        self.operations = operations or {}

        graph = StateGraph(RestaurantState)
        graph.add_node("confirm_order", self.confirm_order)
        graph.add_node("ask_for_less_quantity", self.ask_for_less_quantity)
        graph.add_node("cook", self.cook)
        graph.add_node("serve", self.serve)
        graph.add_node("refund", self.refund)
        graph.add_node("recommend_another_restaurant", self.recommend_another_restaurant)
        graph.add_node("billing", self.billing)
        graph.add_edge(START, "confirm_order")
        graph.add_conditional_edges(
            "confirm_order",
            self._route_confirmation,
            {
                "cook": "cook",
                "adjustment": "ask_for_less_quantity",
                "recommend": "recommend_another_restaurant",
                "billing": "billing",
            },
        )
        graph.add_conditional_edges(
            "cook",
            self._route_cooking,
            {"serve": "serve", "cook": "cook", "refund": "refund"},
        )
        graph.add_conditional_edges(
            "serve",
            self._route_serving,
            {"billing": "billing", "cook": "cook", "refund": "refund"},
        )
        graph.add_edge("refund", "billing")
        graph.add_edge("recommend_another_restaurant", "billing")
        graph.add_edge("ask_for_less_quantity", "billing")
        graph.add_edge("billing", END)
        self.graph = graph.compile()

    def invoke(
        self,
        customer_request: str,
        messages: list[AnyMessage] | None = None,
    ) -> RestaurantState:
        conversation = [*(messages or []), HumanMessage(content=customer_request)]
        return self.graph.invoke(
            {"customer_request": customer_request, "messages": conversation}
        )

    def _run_with_retries(
        self,
        stage: str,
        operation: Callable[[], Any],
        attempt_counts: dict[str, int],
        max_attempts: int | None = None,
    ) -> Any:
        attempt_limit = self.max_attempts if max_attempts is None else max_attempts
        for attempt in range(1, attempt_limit + 1):
            try:
                result = operation()
                attempt_counts[stage] = attempt
                return result
            except InvalidOrderError:
                attempt_counts[stage] = attempt
                raise
            except Exception:
                attempt_counts[stage] = attempt
                if attempt == attempt_limit:
                    raise
                self.sleeper(self.base_delay * (2 ** (attempt - 1)))
        raise RuntimeError(f"{stage} failed without an exception")

    @staticmethod
    def _next_node(state: RestaurantState, node: str, **updates: Any) -> RestaurantState:
        completed = [*state.get("completed_nodes", []), node]
        message = updates.get("customer_message")
        if isinstance(message, str) and message:
            updates["messages"] = [AIMessage(content=message)]
        return {**updates, "completed_nodes": completed}

    def confirm_order(self, state: RestaurantState) -> RestaurantState:
        counts = dict(state.get("attempt_counts", {}))
        try:
            order = self._run_with_retries(
                "confirm_order",
                lambda: self.interpreter.interpret(
                    state.get("customer_request", ""), state.get("messages", [])
                ),
                counts,
            )
        except InvalidOrderError as error:
            return self._invalid_order(state, counts, str(error))
        except Exception as error:
            return self._next_node(
                state,
                "confirm_order",
                status="confirmation_failed",
                order_status="failed",
                last_error=str(error),
                customer_message="Order confirmation failed after three attempts.",
                attempt_counts=counts,
            )

        if not order:
            return self._next_node(
                state,
                "confirm_order",
                status="invalid_order",
                order_status="failed",
                customer_message=(
                    "I could not match that request to the menu. Available items: "
                    f"{', '.join(self.menu.names())}."
                ),
                attempt_counts=counts,
            )

        normalized: list[dict[str, Any]] = []
        for line in order:
            if not isinstance(line, dict):
                return self._invalid_order(state, counts)
            name = str(line.get("name", "")).casefold()
            quantity = line.get("quantity")
            if (
                name not in self.menu.items
                or isinstance(quantity, bool)
                or not isinstance(quantity, int)
                or quantity < 1
            ):
                return self._invalid_order(state, counts)
            normalized.append({"name": self.menu.items[name].name, "quantity": quantity})

        shortages = self.menu.reserve(normalized)
        if shortages:
            details = ", ".join(f"{name}: {available} available" for name, available in shortages.items())
            return self._next_node(
                state,
                "confirm_order",
                order=normalized,
                status="needs_adjustment",
                order_status="failed",
                customer_message=(
                    f"We do not have enough stock ({details}). Please enter a lower quantity."
                ),
                attempt_counts=counts,
            )

        priced_order = []
        total = Decimal("0.00")
        for line in normalized:
            item = self.menu.items[line["name"].casefold()]
            subtotal = item.price * line["quantity"]
            total += subtotal
            priced_order.append(
                {
                    **line,
                    "unit_price": f"{item.price:.2f}",
                    "subtotal": f"{subtotal:.2f}",
                }
            )
        return self._next_node(
            state,
            "confirm_order",
            order=priced_order,
            status="confirmed",
            order_status="partially_completed",
            total=f"{total:.2f}",
            customer_message="Order confirmed.",
            attempt_counts=counts,
        )

    def _invalid_order(
        self,
        state: RestaurantState,
        counts: dict[str, int],
        message: str = "I could not match that order to valid menu items and quantities.",
    ) -> RestaurantState:
        return self._next_node(
            state,
            "confirm_order",
            status="invalid_order",
            order_status="failed",
            customer_message=message,
            attempt_counts=counts,
        )

    def ask_for_less_quantity(self, state: RestaurantState) -> RestaurantState:
        shortage_message = state.get("customer_message", "")
        customer_message = (
            f"{shortage_message} Please enter a lower quantity, or leave the order blank to stop."
            if shortage_message
            else "Please enter a lower quantity, or leave the order blank to stop."
        )
        return self._next_node(
            state,
            "ask_for_less_quantity",
            customer_message=customer_message,
        )

    def cook(self, state: RestaurantState) -> RestaurantState:
        counts = dict(state.get("attempt_counts", {}))
        operation = self.operations.get("cook", lambda order: "Order cooked.")
        counts["cook"] = counts.get("cook", 0) + 1
        try:
            message = operation(state["order"])
        except Exception as error:
            failures = state.get("cook_failures", 0) + 1
            retry = failures < 2
            if retry:
                self.sleeper(self.base_delay * (2 ** (failures - 1)))
            was_partially_prepared = state.get("cooked_once", False)
            if not retry and not was_partially_prepared:
                self.menu.release(state["order"])
            return self._next_node(
                state,
                "cook",
                status="cook_retry" if retry else "cook_failed",
                order_status=(
                    "partially_completed"
                    if retry or was_partially_prepared
                    else "failed"
                ),
                cook_failures=failures,
                last_error=str(error),
                refund_amount=state.get("total") if not retry else state.get("refund_amount"),
                customer_message=(
                    "The kitchen had a problem. Retrying preparation."
                    if retry
                    else "We could not prepare the order after two failed cooking attempts."
                ),
                attempt_counts=counts,
            )
        return self._next_node(
            state,
            "cook",
            status="cooked",
            order_status="partially_completed",
            cooked_once=True,
            cook_failures=state.get("cook_failures", 0),
            last_error="",
            customer_message=str(message),
            attempt_counts=counts,
        )

    def serve(self, state: RestaurantState) -> RestaurantState:
        counts = dict(state.get("attempt_counts", {}))
        operation = self.operations.get("serve", lambda order: "Order served.")
        counts["serve"] = counts.get("serve", 0) + 1
        try:
            message = operation(state["order"])
        except Exception as error:
            failures = state.get("serve_failures", 0) + 1
            retry = failures < 2
            if retry:
                self.sleeper(self.base_delay * (2 ** (failures - 1)))
            return self._next_node(
                state,
                "serve",
                status="serve_retry" if retry else "serve_failed",
                order_status="partially_completed",
                serve_failures=failures,
                refund_amount=state.get("total") if not retry else state.get("refund_amount"),
                last_error=str(error),
                customer_message=(
                    "We could not serve the order. We will prepare it again and retry service."
                    if retry
                    else "We could not serve the order after the retry. A refund is required."
                ),
                attempt_counts=counts,
            )
        return self._next_node(
            state,
            "serve",
            status="served",
            order_status="completed",
            serve_failures=state.get("serve_failures", 0),
            last_error="",
            customer_message=str(message),
            attempt_counts=counts,
        )

    def refund(self, state: RestaurantState) -> RestaurantState:
        counts = dict(state.get("attempt_counts", {}))
        amount = state.get("refund_amount", state.get("total", "0.00"))
        refund_record: dict[str, Any] = {
            "status": "refund_required",
            "amount": amount,
            "currency": self.menu.currency,
            "reason": state.get("status", "order_failed"),
        }
        refund_handler = self.operations.get("refund")
        if refund_handler is not None:
            try:
                result = self._run_with_retries(
                    "refund", lambda: refund_handler(amount), counts
                )
                refund_record.update(status="refunded", result=str(result))
                message = f"A refund of {self.menu.currency} {amount} has been processed."
            except Exception as error:
                refund_record.update(status="refund_failed", error=str(error))
                message = (
                    f"A refund of {self.menu.currency} {amount} is due. "
                    "Please contact staff to complete it."
                )
        else:
            message = (
                f"A refund of {self.menu.currency} {amount} is due. "
                "Please contact staff to complete it."
            )

        return self._next_node(
            state,
            "refund",
            refund=refund_record,
            customer_message=message,
            attempt_counts=counts,
        )

    def recommend_another_restaurant(self, state: RestaurantState) -> RestaurantState:
        return self._next_node(
            state,
            "recommend_another_restaurant",
            order_status="failed",
            customer_message=(
                "We could not confirm your order after three attempts. "
                "Please try another restaurant."
            ),
        )

    def billing(self, state: RestaurantState) -> RestaurantState:
        counts = dict(state.get("attempt_counts", {}))
        try:
            invoice = self._run_with_retries(
                "billing", lambda: self._create_invoice(state), counts
            )
        except Exception as error:
            return self._next_node(
                state,
                "billing",
                status="billing_failed",
                last_error=str(error),
                customer_message="We could not prepare the bill. Please contact staff.",
                attempt_counts=counts,
            )
        return self._next_node(
            state,
            "billing",
            invoice=invoice,
            attempt_counts=counts,
            customer_message=(
                f"Your order is ready. Total due: {self.menu.currency} "
                f"{invoice['total_due']}."
                if invoice["status"] == "ready_for_payment"
                else state.get("customer_message", "No bill was created.")
            ),
        )

    def _create_invoice(self, state: RestaurantState) -> dict[str, Any]:
        if state.get("order_status") != "completed":
            return {
                "status": "not_billed",
                "currency": self.menu.currency,
                "total_due": "0.00",
                "lines": [],
                "refund": state.get("refund"),
            }
        return {
            "order_id": str(uuid.uuid4()),
            "status": "ready_for_payment",
            "currency": self.menu.currency,
            "total_due": state["total"],
            "lines": state["order"],
        }

    @staticmethod
    def _route_confirmation(state: RestaurantState) -> str:
        if state.get("status") == "confirmed":
            return "cook"
        if state.get("status") == "needs_adjustment":
            return "adjustment"
        if state.get("status") == "confirmation_failed":
            return "recommend"
        return "billing"

    @staticmethod
    def _route_cooking(state: RestaurantState) -> str:
        if state.get("status") == "cooked":
            return "serve"
        if state.get("status") == "cook_retry":
            return "cook"
        return "refund"

    @staticmethod
    def _route_serving(state: RestaurantState) -> str:
        if state.get("status") == "served":
            return "billing"
        if state.get("status") == "serve_retry":
            return "cook"
        return "refund"


def _print_result(state: RestaurantState) -> None:
    payload = dict(state)
    payload["messages"] = [
        {"role": message.type, "content": message.content}
        for message in state.get("messages", [])
    ]
    print(json.dumps(payload, indent=2, ensure_ascii=True, default=str))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Take and fulfill restaurant orders")
    parser.add_argument("--request", help="Submit one order request and exit")
    parser.add_argument("--menu", type=Path, help="Path to a menu JSON file")
    parser.add_argument("--model", help="Override LLM_MODEL_NAME from constants.py")
    parser.add_argument("--retry-delay", type=float, default=0.5)
    args = parser.parse_args(argv)

    try:
        workflow = RestaurantWorkflow(
            menu_path=args.menu,
            model=args.model,
            base_delay=args.retry_delay,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Could not start restaurant workflow: {error}", file=sys.stderr)
        raise SystemExit(2) from error

    if args.request is not None:
        _print_result(workflow.invoke(args.request))
        return

    try:
        conversation: list[AnyMessage] = []
        request = input("What would you like to order? ").strip()
        while request:
            state = workflow.invoke(request, messages=conversation)
            conversation = state.get("messages", [])
            print(state.get("customer_message", ""))
            if state.get("status") != "needs_adjustment":
                if state.get("invoice", {}).get("status") == "ready_for_payment":
                    _print_result(state)
                break
            request = input("Please enter your revised order: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nOrder session ended.")


if __name__ == "__main__":
    main()