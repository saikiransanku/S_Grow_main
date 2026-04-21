try:
    from training.image_prediction.train_kharif import main
except ImportError:  # pragma: no cover
    from train_kharif import main


if __name__ == "__main__":
    main(default_season="kharif")
