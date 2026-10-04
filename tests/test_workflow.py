import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from restaurant.main import RestaurantWorkflow


class RestaurantWorkflowTests(unittest.TestCase):
    def make_workflow(self, **kwargs):
        kwargs.setdefault("base_delay", 0)
        kwargs.setdefault("sleeper", lambda _: None)
        kwargs.setdefault("api_key", "")
        return RestaurantWorkflow(**kwargs)

    def test_successful_order_ends_with_billing(self):
        workflow = self.make_workflow()

        result = workflow.invoke("2 burgers and 1 fries")

        self.assertEqual(result["status"], "served")
        self.assertEqual(result["order_status"], "completed")
        self.assertEqual(result["invoice"]["status"], "ready_for_payment")
        self.assertEqual(result["invoice"]["total_due"], "29.00")
        self.assertEqual(result["completed_nodes"][-1], "billing")
        self.assertEqual(result["messages"][0].type, "human")

    def test_unavailable_quantity_requests_adjustment_then_bills_nothing(self):
        workflow = self.make_workflow()

        result = workflow.invoke("99 burger")

        self.assertEqual(result["status"], "needs_adjustment")
        self.assertEqual(result["order_status"], "failed")
        self.assertIn("8 available", result["customer_message"])
        self.assertEqual(result["invoice"]["status"], "not_billed")
        self.assertEqual(result["completed_nodes"][-1], "billing")

    def test_unknown_item_fails_the_whole_order_without_reserving_stock(self):
        for request in ("1 pizza", "1 burger and 2 pizza"):
            with self.subTest(request=request):
                workflow = self.make_workflow()

                result = workflow.invoke(request)

                self.assertEqual(result["status"], "invalid_order")
                self.assertEqual(result["order_status"], "failed")
                self.assertEqual(workflow.menu.items["burger"].quantity, 8)
                self.assertEqual(result["invoice"]["status"], "not_billed")
                self.assertEqual(result["completed_nodes"][-1], "billing")

    def test_invalid_quantities_fail_without_reserving_stock(self):
        for request in (
            "0 burger",
            "-2 burger",
            "1.5 burger",
            "zero burger",
            "negative two burgers",
        ):
            with self.subTest(request=request):
                workflow = self.make_workflow()

                result = workflow.invoke(request)

                self.assertEqual(result["status"], "invalid_order")
                self.assertEqual(result["order_status"], "failed")
                self.assertEqual(workflow.menu.items["burger"].quantity, 8)

    def test_word_number_quantity_is_parsed_as_an_integer(self):
        workflow = self.make_workflow()

        result = workflow.invoke("two burgers")

        self.assertEqual(result["order_status"], "completed")
        self.assertEqual(result["order"][0]["quantity"], 2)

    def test_quantity_revision_keeps_annotated_conversation_history(self):
        workflow = self.make_workflow()

        first_result = workflow.invoke("99 burgers")
        revised_result = workflow.invoke("2 burgers", messages=first_result["messages"])

        self.assertEqual(revised_result["order_status"], "completed")
        human_messages = [
            message.content
            for message in revised_result["messages"]
            if message.type == "human"
        ]
        self.assertEqual(human_messages, ["99 burgers", "2 burgers"])

    def test_order_interpreter_sends_retained_messages_to_llm(self):
        workflow = self.make_workflow()
        captured = {}

        def create_completion(**kwargs):
            captured.update(kwargs)
            message = SimpleNamespace(content='{"items": []}')
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        workflow.interpreter.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_completion))
        )
        history = [
            HumanMessage(content="99 burgers"),
            AIMessage(content="Only 8 burgers are available."),
            HumanMessage(content="Then make it 2."),
        ]

        workflow.interpreter.interpret("Then make it 2.", history)

        self.assertEqual(
            [
                (message["role"], message["content"])
                for message in captured["messages"][1:]
            ],
            [
                ("user", "99 burgers"),
                ("assistant", "Only 8 burgers are available."),
                ("user", "Then make it 2."),
            ],
        )

    def test_llm_invalid_order_decision_fails_without_retry(self):
        workflow = self.make_workflow()
        message = SimpleNamespace(
            content='{"valid": false, "reason": "pizza is not on the menu", "items": []}'
        )
        workflow.interpreter.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_kwargs: SimpleNamespace(
                        choices=[SimpleNamespace(message=message)]
                    )
                )
            )
        )

        result = workflow.invoke("1 pizza")

        self.assertEqual(result["status"], "invalid_order")
        self.assertEqual(result["order_status"], "failed")
        self.assertEqual(result["attempt_counts"]["confirm_order"], 1)
        self.assertEqual(workflow.menu.items["burger"].quantity, 8)

    def test_cook_failure_twice_requests_refund(self):
        calls = 0
        delays = []

        def failing_cook(_order):
            nonlocal calls
            calls += 1
            raise RuntimeError("oven unavailable")

        workflow = self.make_workflow(
            base_delay=0.25,
            sleeper=delays.append,
            operations={"cook": failing_cook},
        )
        result = workflow.invoke("1 burger")

        self.assertEqual(calls, 2)
        self.assertEqual(delays, [0.25])
        self.assertEqual(result["attempt_counts"]["cook"], 2)
        self.assertEqual(result["status"], "cook_failed")
        self.assertEqual(result["order_status"], "failed")
        self.assertEqual(result["invoice"]["status"], "not_billed")
        self.assertEqual(result["refund"]["status"], "refund_required")
        self.assertEqual(result["refund"]["amount"], "12.50")
        self.assertEqual(workflow.menu.items["burger"].quantity, 8)
        self.assertIn("refund", result["completed_nodes"])
        self.assertEqual(result["completed_nodes"][-1], "billing")

    def test_cook_retry_can_recover_and_complete_order(self):
        calls = 0
        delays = []

        def flaky_cook(_order):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("temporary oven issue")
            return "cooked after retry"

        workflow = self.make_workflow(
            base_delay=0.25,
            sleeper=delays.append,
            operations={"cook": flaky_cook},
        )
        result = workflow.invoke("1 burger")

        self.assertEqual(calls, 2)
        self.assertEqual(delays, [0.25])
        self.assertEqual(result["order_status"], "completed")
        self.assertEqual(result["invoice"]["status"], "ready_for_payment")
        self.assertEqual(result["completed_nodes"][-1], "billing")

    def test_serve_failure_recooks_then_requests_refund_after_second_failure(self):
        cook_calls = 0
        serve_calls = 0
        delays = []

        def cook(_order):
            nonlocal cook_calls
            cook_calls += 1
            return "cooked"

        def serve(_order):
            nonlocal serve_calls
            serve_calls += 1
            raise RuntimeError("delivery handoff unavailable")

        workflow = self.make_workflow(
            base_delay=0.25,
            sleeper=delays.append,
            operations={"cook": cook, "serve": serve},
        )
        result = workflow.invoke("1 burger")

        self.assertEqual(cook_calls, 2)
        self.assertEqual(serve_calls, 2)
        self.assertEqual(delays, [0.25])
        self.assertEqual(result["status"], "serve_failed")
        self.assertEqual(result["order_status"], "partially_completed")
        self.assertEqual(result["refund"]["amount"], "12.50")
        self.assertEqual(result["completed_nodes"][-1], "billing")
        self.assertLess(
            result["completed_nodes"].index("serve", 1),
            result["completed_nodes"].index("cook", 2),
        )

    def test_confirmation_failure_three_times_recommends_another_restaurant(self):
        workflow = self.make_workflow(base_delay=0.25)
        delays = []
        workflow.sleeper = delays.append

        with patch.object(
            workflow.interpreter,
            "interpret",
            side_effect=RuntimeError("LLM unavailable"),
        ) as interpret:
            result = workflow.invoke("1 burger")

        self.assertEqual(interpret.call_count, 3)
        self.assertEqual(delays, [0.25, 0.5])
        self.assertEqual(result["attempt_counts"]["confirm_order"], 3)
        self.assertEqual(result["order_status"], "failed")
        self.assertIn("another restaurant", result["customer_message"])
        self.assertIn("recommend_another_restaurant", result["completed_nodes"])
        self.assertEqual(result["completed_nodes"][-1], "billing")


if __name__ == "__main__":
    unittest.main()
