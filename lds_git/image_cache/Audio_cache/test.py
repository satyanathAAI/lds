import pymongo
from pymongo import MongoClient
CLUSTER_ENDPOINT="docdb-2024-08-21-06-19-45.cluster-c5yo62eims1o.us-east-1.docdb.amazonaws.com"
PORT = 27017
USERNAME = 'ldsadmin'
PASSWORD = 'lds2connect'
DATABASE_NAME = 'ldstest'
SSL_CA_CERT = '/home/ubuntu/lds/global-bundle.pem'
uri = f"mongodb://{USERNAME}:{PASSWORD}@{CLUSTER_ENDPOINT}:{PORT}/{DATABASE_NAME}?ssl=true"

# Connect to the DocumentDB cluster
mongoclient = MongoClient(uri, tls=True, tlsCAFile=SSL_CA_CERT)

db = mongoclient[DATABASE_NAME]
collection = db['audio']
# collections = db.list_collection_names()
# document = {"name": "satya", "value": 42}
# document['age']='23'
# collection.insert_one(document)
# print("Collections:", collections)
# collection = db[collection_name]
search_query = {"filename": "suffering_and_joy.mp3"}

# Define the projection to return only the value of the filename field
projection = {"_id": 0}

# Find the document with the specified filename and project the result
result = collection.find_one(search_query, projection)

# Print the result
if result:
    print("Filename value:", result)
else:
    print("No document found with the specified filename.")
