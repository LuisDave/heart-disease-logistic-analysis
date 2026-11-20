from src.run_analysis import main


if __name__ == "__main__":
    summary = main()
    print("\nAnalysis complete. Key results:")
    for key, value in summary.items():
        print(f"- {key}: {value}")
