from src.pipeline import run

if __name__ == "__main__":
    df, report = run(
        data_path="data/ecommerce_data.xlsx",
        output_dir="output"
    )
