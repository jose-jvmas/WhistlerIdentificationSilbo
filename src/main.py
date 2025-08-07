import os
import utils
import argparse

from experiments import perform_experiments


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Whistler identification on silbo gomero",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--encoder",
        "-e",
        type=str,
        choices=["whisper", "wav2vec", "mfcc_mean", "mfcc_delta", "mfcc_deltadelta"],
        required=True,
        help="Audio encoder",
    )
    parser.add_argument(
        "--param",
        "-p",
        type=str,
        required=True,
        help="Parameter of the encoder",
    )
    args, _ = parser.parse_known_args()

    if args.encoder == "whisper":
        assert args.param in [
            "base",
            "medium",
            "small",
            "tiny",
        ], f"{args.param} not a valid parameter for encoder {args.encoder}"
    elif args.encoder == "wav2vec":
        args.param = int(args.param)
        assert args.param in [
            64,
            256,
            1024,
            4096,
        ], f"{args.param} not a valid parameter for encoder {args.encoder}"
    else:
        args.param = int(args.param)
        assert args.param > 0, f"{args.param} not a valid parameter for encoder {args.encoder}"

    return args


if __name__ == "__main__":
    try:
        print("Started")

        # Parse arguments and print them
        args = parse_arguments()
        for k, v in vars(args).items():
            print(f"{k}: {v}")

        # Obtain embeddings using the specified model
        if args.encoder == "whisper":
            silbo_dataset = utils.load_CSV_dataset(src_folder = "features", encoder = args.encoder, param = args.param)
        elif args.encoder == "wav2vec":
            silbo_dataset = utils.load_CSV_dataset(src_folder = "features", encoder = args.encoder, param = args.param)
        elif "mfcc" in args.encoder:
            silbo_dataset = utils.load_CSV_dataset(src_folder = "features", encoder = args.encoder, param = args.param)


        # Results file:
        if not os.path.exists("results"):
            os.makedirs("results")

        Perform whistler classification experiments
        perform_experiments(
            in_dataset=silbo_dataset,
            args=args,
            res_file=f"results/Results_{args.encoder}-{args.param}.csv",
        )

    except Exception as err:
        print(f"Error: {err}")

    finally:
        print("Finished")
