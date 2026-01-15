import json
import os

os.makedirs("data/finetune", exist_ok=True)

dataset = [
    {
        "instruction": "Explain the objective of the recovery index analysis for airlines.",
        "input": "",
        "output": "The objective of the recovery index analysis is to evaluate how quickly and effectively airlines recovered from the COVID-19 disruption by comparing financial and operational performance against 2019 baseline levels."
    },
    {
        "instruction": "Why is 2019 used as the baseline year in airline recovery studies?",
        "input": "",
        "output": "The year 2019 is used as the baseline because it represents the last full year of normal airline operations before COVID-19 disruptions, making it a reliable reference point for recovery comparisons."
    },
    {
        "instruction": "Explain passenger traffic recovery in the airline industry.",
        "input": "",
        "output": "Passenger traffic recovery refers to the gradual return of passenger volumes to pre-COVID levels, reflecting improvements in travel demand, route availability, and consumer confidence."
    },
    {
        "instruction": "Why is load factor an important indicator of airline recovery?",
        "input": "",
        "output": "Load factor measures how efficiently airlines fill available seats. A higher load factor during recovery indicates better demand management, operational efficiency, and improved revenue generation."
    },
    {
        "instruction": "Compare recovery challenges faced by Indian airlines versus Gulf airlines.",
        "input": "",
        "output": "Indian airlines faced challenges such as high domestic competition and cost sensitivity, while Gulf airlines benefited from strong international networks, government support, and hub-based operations that accelerated recovery."
    }
]

with open("data/finetune/airline_recovery_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

print("Airline recovery fine-tuning dataset created successfully")
