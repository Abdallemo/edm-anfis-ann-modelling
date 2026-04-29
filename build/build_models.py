from build.build_pipeline import build_and_save_model


def main():
    build_and_save_model(
        csv_path="./datasets/dataset1.csv",
        model_name="dataset1_model",
        dataset_type="dataset1-type",
    )

    build_and_save_model(
        csv_path="./datasets/dataset2.csv",
        model_name="dataset2_model",
        dataset_type="dataset2-type",
    )

    build_and_save_model(
        csv_path="./datasets/titanium_ti-6ai-4v.csv",
        model_name="titanium_ti_6ai_4v_model",
        dataset_type="dataset1-type",
    )


if __name__ == "__main__":
    main()
