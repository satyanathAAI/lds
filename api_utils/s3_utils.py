import boto3


# Initialize a session using Amazon S3
s3_client = boto3.client(
    "s3",
)


def get_presigned_url(bucket_name: str, object_key: str) -> str:
    try:
        presigned_url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": object_key},
            ExpiresIn=86400,  # URL expires in one day
        )
        return presigned_url
    except Exception as e:
        print(f"Error generating pre-signed URL: {e}")
        response = None
    print(response)


def check_folder_exists(bucket_name, folder_name):

    # List objects with the folder prefix
    response = s3_client.list_objects_v2(
        Bucket=bucket_name, Prefix=folder_name, Delimiter="/"
    )

    # Check if any objects are returned
    print(response)
    if "Contents" in response or "CommonPrefixes" in response:
        return True
    return False


def upload_file_to_S3(bucket_name, filename, prefix_key):
    s3_client.upload_file(filename, bucket_name, prefix_key)
    s3_url = f"https://lds-test-public.s3.amazonaws.com/{prefix_key}"
    return s3_url
