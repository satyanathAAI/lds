from pymongo import MongoClient


CLUSTER_ENDPOINT = (
    "docdb-2024-08-21-06-19-45.cluster-c5yo62eims1o.us-east-1.docdb.amazonaws.com"
)
PORT = 27017
USERNAME = "ldsadmin"
PASSWORD = "lds2connect"
DATABASE_NAME = "ldstest"
SSL_CA_CERT = "/home/ubuntu/lds/global-bundle.pem"
uri = f"mongodb://{USERNAME}:{PASSWORD}@{CLUSTER_ENDPOINT}:{PORT}/{DATABASE_NAME}?ssl=true"

# Connect to the DocumentDB cluster
client = MongoClient(uri, tls=True, tlsCAFile=SSL_CA_CERT)


def insert_to_mongo(record, collection_name):
    db = client[DATABASE_NAME]
    collection = db[collection_name]

    try:
        collection.insert_one(record)
        print("record inserted")
    except Exception as e:
        print(f"Error inserting to mongo collection {collection_name}")


def check_cache(videoname: str, collection_name: str) -> bool:
    db = client[DATABASE_NAME]
    collection = db[collection_name]
    search_query = {"filename": videoname}
  

    # Define the projection to return only the value of the filename field
    projection = {"_id": 0}

    # Find the document with the specified filename and project the result
    result = collection.find_one(search_query, projection)
   

    if result is not None:
        del result["filename"]

        return result, True
    else:
        return None, False
