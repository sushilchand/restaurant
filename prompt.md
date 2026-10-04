I am creating a restaurant system where customer will order something from the menu.
You need to createa a menu as a json where the name of the food, quantity, price will be present.
Create a langgraph to achive this where the nodes are  like confirm_order, cook, serve, billing.
Also create conditional edges like if quantity is not available then ask client again for less quantity. also handle failures and try to reattempt them 3 times with exponential backoffs. The last edge of the langgraph should be billing. So create all this based on classes and methods using OOPS concepts and give me a running code. Make sure to use langgraph and LLM to create all this with State, nodes and edges
Also each order should have a status whether order was completed, partially completed, failed

If serve fails then it should retry cook again and if again serve fails then go to user to refund the amount
also if cook fails twice then also refund the amount
also if confirm order fails 3 times then also fail and ask user to go to anohter restaurant
for the state of langgraph there should be annotated message for LLM and user so that complete context remains with LLM

the agent should also dedect if any order contains invalid item or invalid quantity. in those case the order should fail
