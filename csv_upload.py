import boto3
s3 = boto3.client("s3")
local_file = "/Users/anurag_77y/MLOPs-capstone-project/notebooks/IMDB.csv"
bucket_name = "mlopsimdb"
s3_file = "data/raw/IMDB.csv"

s3.upload_file(
    local_file,
    bucket_name,
    s3_file
)

print("Uploaded successfully!")