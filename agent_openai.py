import os
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9000/v1",          # proxy, not api.openai.com
    api_key=os.environ["OPENAI_API_KEY"],          # real key; proxy forwards it
    default_headers={
        "X-SBus-Agent-Id":   "worker-demo",
        "X-SBus-Session-Id": "session-demo-001",
    },
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content":
            "You are a Django engineer. You have read-only access to "
            "models_state (schema) and read-write to orm_query and test_fixture."},
        {"role": "user", "content":
            "Please update the orm_query to add a select_related, and update "
            "the matching test_fixture to cover the new behaviour."},
    ],
)

print("=" * 60)
print("Completion:")
print(response.choices[0].message.content)
print("=" * 60)

print()
print("Check sbus-proxy logs — you should see:")
print("  - 'input-side extraction' with shards [models_state, orm_query, test_fixture]")
print("  - 'DeliveryLog registered' with status=200")
