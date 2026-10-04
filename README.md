# Restaurant Order Workflow

A small restaurant ordering CLI built with LangGraph. It checks menu inventory,
asks customers to revise unavailable quantities, retries transient operation
failures with exponential backoff, and creates an invoice as the final graph
node.

## Run

Python 3.12 or newer is required. From the project directory:

```sh
uv sync
uv run restaurant
```

Enter an order such as `2 burgers and 1 fries`. If there is not enough stock,
the CLI asks for a revised order. To submit one request without interactive
prompts:

```sh
uv run restaurant --request "2 burgers and 1 fries"
```

The workflow reads `GROQ_API_KEY` and `LLM_MODEL_NAME` from
`src/restaurant/constants.py` (loaded from `.env`). Without a Groq key, a local
menu-aware parser supports orders written with menu item names and quantities.
If `LLM_MODEL_NAME` is unset, the model defaults to
`llama-3.3-70b-versatile`; override it with `--model`.

The menu and starting inventory are in `src/restaurant/menu.json`. Each
workflow instance reserves inventory when an order is confirmed. Retry attempts
default to three, with delays of 0.5 and 1 second before attempts two and three;
use `--retry-delay 0` to disable waiting while testing.

Each result includes `order_status`: `completed`, `partially_completed`, or
`failed`. Confirmation gets three attempts; two failed cooking attempts request
a refund. A failed serving attempt triggers a recook and one more serving
attempt; a second failure requests a refund. The graph routes every outcome
through `billing` as its final node. Conversation messages use LangGraph's
annotated message reducer and are retained when the interactive CLI asks for a
revised quantity.

Unknown menu items and nonpositive, fractional, or otherwise invalid quantities
fail confirmation without reserving stock. The local parser also accepts
integer quantities written as words, such as `two burgers`.

No payment processor is configured by default. A failure produces a
`refund_required` record and tells the customer the amount due. To process real
refunds, provide a `refund` operation when constructing `RestaurantWorkflow`;
that operation is retried up to three times.
