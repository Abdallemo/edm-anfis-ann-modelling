import pandas as pd

from ann import NeuralNetwork


def main():
    print("=== Surface Roughness Predictor ===")

    model_path = input(
        "Enter the path to the model (e.g., models/dataset1_model.pkl): "
    ).strip()

    try:
        model = NeuralNetwork.load(model_path)
    except Exception as e:
        print(f"\nError: Could not load the model from '{model_path}'.")
        print(f"Details: {e}")
        return

    print(
        f"\nModel loaded successfully! Please provide the {len(model.feature_names)} required inputs."
    )
    print("-" * 40)

    user_inputs = []
    for feature in model.feature_names:
        while True:
            raw_input = input(f"Enter value for '{feature}': ")
            try:
                val = float(raw_input)
                user_inputs.append(val)
                break
            except ValueError:
                print("  -> Invalid input. Please enter a valid number.")

    raw_machine_settings = pd.DataFrame([user_inputs], columns=model.feature_names)

    try:
        prediction = model.predict(raw_machine_settings)

        print("\n" + "=" * 40)
        print("PREDICTION RESULTS")
        print("=" * 40)
        print(f"Model Used : {model_path}")
        print(f"Inputs     : {dict(zip(model.feature_names, user_inputs))}")

        if isinstance(prediction, (list, tuple, pd.Series)) or (
            hasattr(prediction, "shape") and prediction > 0
        ):
            print(f"Output     : {prediction:.4f}")
        else:
            print(f"Output     : {prediction:.4f}")

        print("=" * 40)

    except Exception as e:
        print(f"\nError during prediction: {e}")


if __name__ == "__main__":
    main()
