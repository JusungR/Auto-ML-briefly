"""코드로 직접 배치 스코어링을 호출하는 예시."""
from auto_ml import load_config
from auto_ml.scoring.runner import run_scoring


def main() -> None:
    config = load_config("configs/example.yaml")
    output_path = run_scoring(config)
    print(f"scored output: {output_path}")


if __name__ == "__main__":
    main()
