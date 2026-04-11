#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logger.info("Downloading artifact")
    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info("Reading dataframe")
    df = pd.read_csv(artifact_local_path)

    logger.info("Cleaning data")
    # Perform basic cleaning operations
    df = df.drop_duplicates()
    df = df.dropna(subset=["price"])
    df = df[(df['price'] >= args.min_price) & (df['price'] <= args.max_price)]

    # Add this boundary filter
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    logger.info("Cleaned data has %s rows and %s columns", *df.shape)
    logger.info("Saving cleaned data")
    df.to_csv(args.output_artifact, index=False)

    logger.info("Creating artifact")
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )
    artifact.add_file(args.output_artifact)

    logger.info("Uploading artifact to W&B") 
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Input artifact to be cleaned",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Output artifact after cleaning",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price for filtering",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum price for filtering",
        required=True
    )


    args = parser.parse_args()

    go(args)
