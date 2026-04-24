"""코드로 직접 학습 파이프라인을 호출하는 예시."""
from auto_ml import AutoMLPipeline, load_config


def main() -> None:
    config = load_config("configs/example.yaml")
    outputs = AutoMLPipeline(config).run()
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
