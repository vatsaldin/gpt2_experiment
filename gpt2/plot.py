# pip install matplotlib
import json, matplotlib.pyplot as plt

with open("loss_log.json") as f:
    log = json.load(f)

plt.figure(figsize=(10, 6))
plt.plot(log["steps"], log["train"], alpha=0.3, label="train")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_curve.png")
plt.show()