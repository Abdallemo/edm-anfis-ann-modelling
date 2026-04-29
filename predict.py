import pandas as pd

from ann import NeuralNetwork

model = NeuralNetwork.load("models/dataset2_model.pkl")

raw_machine_settings = pd.DataFrame(
    [[750, 10, 40, 30]], columns=["ton", "duty_cycle", "peak_current", "voltage"]
)

# raw_machine_settings = pd.DataFrame(
#     [[60, 12, 8, 9]], columns=["volt", "ip", "ton", "toff"]
# )


prediction = model.predict(raw_machine_settings)

print(f"Predicted Surface Roughness: {prediction[0]:.4f}")
