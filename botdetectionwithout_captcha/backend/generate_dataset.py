import pandas as pd
import random

data = []

for i in range(1000):

    if random.random() > 0.5:
        # HUMAN (with noise)
        data.append({
            "text": random.choice([
                "hello how are you",
                "login portal access",
                "student dashboard",
                "checking results"
            ]),
            "mouse_moves": random.randint(50, 800),
            "clicks": random.randint(1, 30),
            "avg_typing_speed": random.randint(80, 400),
            "scroll_depth": random.randint(50, 1500),
            "time_spent": random.randint(5, 120),
            "label": 0
        })

    else:
        # BOT (with noise)
        data.append({
            "text": random.choice([
                "script login",
                "auto submit form",
                "bot access",
                "test automation"
            ]),
            "mouse_moves": random.randint(0, 200),   # overlap
            "clicks": random.randint(0, 10),
            "avg_typing_speed": random.randint(200, 800),
            "scroll_depth": random.randint(0, 500),
            "time_spent": random.randint(0, 30),
            "label": 1
        })

df = pd.DataFrame(data)

# shuffle
df = df.sample(frac=1).reset_index(drop=True)

df.to_csv("rf_dataset.csv", index=False)

print("✅ Real-world dataset created")