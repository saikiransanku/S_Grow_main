try:
    from training.image_prediction.train_all_season import main
except ImportError:  # pragma: no cover
    from train_all_season import main


if __name__ == "__main__":
    main(default_season="all_season")
