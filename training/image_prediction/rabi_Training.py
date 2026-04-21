try:
    from training.image_prediction.train_rabi import main
except ImportError:  # pragma: no cover
    from train_rabi import main


if __name__ == "__main__":
    main(default_season="rabi")
