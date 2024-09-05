from flask import Flask, request
from flask_cors import CORS
import os
import json
import base64

from constants import Augument_open_api_key, search_rag_api, global_output_fodler_path
from utils import (
    get_summary_from_gpt4turbo,
    get_summary_from_sagemaker,
    prompt2,
    create_folder_save_files_in_main_output_folder,
    send_to_rag,
)
import boto3
import cv2
from keras import metrics
import numpy as np
import tensorflow as tf
import json
import tensorflow as tf
import numpy as np
import os
import random
import tensorflow as tf
from pathlib import Path
from keras import applications
from keras import layers
from keras import losses
import pandas as pd
from keras import optimizers
from keras import metrics
from keras import Model
from keras.applications import resnet
from pymongo import MongoClient
import requests
from urllib.parse import urlparse

# DocDB connection
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
mongoclient = MongoClient(uri, tls=True, tlsCAFile=SSL_CA_CERT)


def check_cache(filename):

    db = mongoclient["ldstest"]
    collection = db["image"]
    search_query = {"filename": filename}

    # Define the projection to return only the value of the filename field
    projection = {"_id": 0}

    # Find the document with the specified filename and project the result
    result = collection.find_one(search_query, projection)
    if result is not None:

        return result, True
    else:
        return None, False


def insert_to_mongo(record, filename):
    db = mongoclient["ldstest"]
    collection = db["image"]
    record["filename"] = filename
    # record_background = jsonpickle.dumps(record)
    # record_to_insert = record_background

    try:
        collection.insert_one(record)
        print("record inserted")
    except Exception as e:
        print(f"Error inserting to mongo collection {collection}")


def extract_bucket_and_key(s3_url):
    # Parse the S3 URL
    parsed_url = urlparse(s3_url)
    # Extract the object key from the path
    object_key = parsed_url.path.lstrip("/")

    return object_key


target_shape = (200, 200)
batch_size = 8
base_cnn = resnet.ResNet50(
    weights="imagenet", input_shape=target_shape + (3,), include_top=False
)

flatten = layers.Flatten()(base_cnn.output)
dense1 = layers.Dense(512, activation="relu")(flatten)
dense1 = layers.BatchNormalization()(dense1)
dense2 = layers.Dense(256, activation="relu")(dense1)
dense2 = layers.BatchNormalization()(dense2)
output = layers.Dense(256)(dense2)
output = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(output)
f = open("/home/ubuntu/lds/lds_image/image_cache/temple.json", "r")

temple_data = json.load(f)

embedding = Model(base_cnn.input, output, name="Embedding")

trainable = False
for layer in base_cnn.layers:
    if layer.name == "conv5_block1_out":
        trainable = True
    layer.trainable = trainable


class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.math.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.math.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
positive_input = layers.Input(name="positive", shape=target_shape + (3,))
negative_input = layers.Input(name="negative", shape=target_shape + (3,))

distances = DistanceLayer()(
    embedding(resnet.preprocess_input(anchor_input)),
    embedding(resnet.preprocess_input(positive_input)),
    embedding(resnet.preprocess_input(negative_input)),
)

siamese_network = Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=distances
)


def preprocess_image(filename):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """

    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    return image


def preprocess_triplets(anchor, positive, negative):
    """
    Given the filenames corresponding to the three images, load and
    preprocess them.
    """

    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative),
    )


class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.6):
        super().__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]


data = pd.read_csv("/home/ubuntu/lds/lds_image/image_cache/data.csv")
data = data.sample(frac=1.0)
data_train = data.iloc[0 : int(0.9 * len(data)), :]
data_test = data.iloc[int(0.9 * len(data)) :, :]
anchors_train = data_train["anchors"]
positives_train = data_train["pos"]
negatives_train = data_train["neg"]
anchors_test = data_test["anchors"]
positives_test = data_test["pos"]
negatives_test = data_test["neg"]

train_dataset = tf.data.Dataset.from_tensor_slices(
    (anchors_train, positives_train, negatives_train)
)
train_dataset = (
    train_dataset.map(preprocess_triplets, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

validation_dataset = tf.data.Dataset.from_tensor_slices(
    (anchors_test, positives_test, negatives_test)
)
validation_dataset = (
    validation_dataset.map(preprocess_triplets, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

siamese_model = SiameseModel(siamese_network)
siamese_model.compile(optimizer=optimizers.Adam(0.0001))
# siamese_model.fit(train_dataset, epochs=3, validation_data=validation_dataset)
# siamese_model.save_weights("weights_new.h5")
siamese_model(next(iter(train_dataset)))
siamese_model.load_weights(
    "/home/ubuntu/lds/lds_image/churchrecognition/weights_new_final.h5"
)


sample = next(iter(train_dataset))


anchor, positive, negative = sample
anchor_embedding, positive_embedding, negative_embedding = (
    embedding(resnet.preprocess_input(anchor)),
    embedding(resnet.preprocess_input(positive)),
    embedding(resnet.preprocess_input(negative)),
)


empty_json = {"ip_details": "", "summary": "", "tags": []}

from bson import json_util


def get_embedding(path):
    img_data = cv2.imread(path)
    img_data = cv2.resize(img_data, (200, 200))
    img_data = img_data / 255.0
    img_data = tf.expand_dims(img_data, axis=0)
    img_embedding = embedding(resnet.preprocess_input(img_data))
    return img_embedding


app = Flask(__name__)
CORS(app)
from io import BytesIO

s3 = boto3.client("s3")


@app.route("/image_upload", methods=["POST"])
def main():
    data = request.get_json()
    s3_url = data["s3_url"]

    img_response = requests.get(s3_url)
    img_response.raise_for_status()  # Check if the request was successful
    image_data = img_response.content
    # Process the image as needed
    image = BytesIO(image_data)
    # For example, if you want to display the image using PIL
    from PIL import Image

    file = Image.open(image).convert("RGB")
    filename = extract_bucket_and_key(s3_url)
    print("filename is :", filename)
    filename = filename.split("/")[-1]

    global empty_json

    os.makedirs("images", exist_ok=True)
    os.makedirs("images_json", exist_ok=True)

    # Get the file name and extension
    file_name_without_extension, extension = os.path.splitext(filename)

    file.save(f"/home/ubuntu/lds/lds_image/image_cache/output_images/{filename}")

    # server to host images to display
    s3.upload_file(
        f"/home/ubuntu/lds/lds_image/image_cache/output_images/{filename}",
        "lds-test-public",
        f"Inputimages/{filename}",
    )
    # s3_output_path = f"https://testlds.s3.amazonaws.com/Output/{filename}"
    s3_output_path = f"https://lds-test-public.s3.amazonaws.com/Inputimages/{filename}"
    # store in local
    img_saved_path = os.path.join("images", filename)
    img_json_saved_path = os.path.join(
        "images_json", f"{file_name_without_extension}.json"
    )

    # store in Main folder
    main_folder_img_summary_saved_path = f"{global_output_fodler_path}/{file_name_without_extension}/{file_name_without_extension}_summary.txt"
    main_folder_img_json_saved_path = f"{global_output_fodler_path}/{file_name_without_extension}/{file_name_without_extension}_json.json"
    img_saved_path = f"/home/ubuntu/lds/lds_image/image_cache/output_images/{filename}"
    # We currently support PNG (.png), JPEG (.jpeg and .jpg), WEBP (.webp), and non-animated GIF (.gif).
    if extension in [".png", ".jpg", ".jpeg", ".webp", ".gif"]:
        # if os.path.exists(main_folder_img_summary_saved_path) and os.path.exists(
        #     main_folder_img_json_saved_path
        # ):
        #     print("files already existed")
        #     with open(main_folder_img_json_saved_path, "r") as f:
        #         json_data = json.load(f)
        #     print("fetched json file from Main OUTPUT folder successfully")
        #     # send to rag
        #     send_to_rag(
        #         main_folder_img_summary_saved_path,
        #         file_name_without_extension,
        #         search_rag_api,
        #     )
        json_data, search = check_cache(filename)
        if search is True:
            json_data["filename"] = filename
            return json_data
        else:
            print("its a new image, getting tags and summary from open ai started")
            # save image in folder
            try:
                answer = get_summary_from_sagemaker(
                    s3_url, Augument_open_api_key, prompt2
                )
                # answer = get_summary_from_gpt4turbo(
                #     s3_url, Augument_open_api_key, prompt2
                # )
                if answer is None:
                    print("error while getting tags and sumary form open ai")
                    empty_json["path"] = s3_output_path
                    return empty_json
                else:

                    # Parse the JSON string into a Python dictionary
                    json_data = answer
                    print(type(json_data))
                    temple_names = list(temple_data.keys())
                    sim_scores = []
                    for temple_name in temple_names:
                        img_embedding = get_embedding(img_saved_path)
                        temple_path = temple_data[temple_name]
                        print("temple path is", temple_path)
                        temple_embedding = get_embedding(temple_path)
                        cosine_similarity = metrics.CosineSimilarity()
                        sim_scores.append(
                            cosine_similarity(img_embedding, temple_embedding)
                        )
                    max_index = np.argmax(sim_scores)
                    print(sim_scores)
                    if sim_scores[max_index] > 0.8:
                        try:
                            json_data["tags"].append(temple_names[max_index])
                        except:
                            pass

                    json_data["path"] = s3_output_path
                    # Saving Json
                    # with open(img_json_saved_path, "w") as f:
                    #     json.dump(json_data, f, indent=4)
                    insert_to_mongo(json_data, filename)
                    # send files to main folder json and summary
                    # create_folder_save_files_in_main_output_folder(
                    #     file_name_without_extension,
                    #     global_output_fodler_path,
                    #     json_data,
                    #     search_rag_api,
                    # )
                    print("completed the process")
                    del json_data['_id']
                    return json.dumps(json_data, default=json_util.default)
            except Exception as e:
                print(e)
                print("error while getting tags and sumary form open ai")
                empty_json["path"] = s3_output_path
                return empty_json
    else:
        return "We currently support PNG (.png), JPEG (.jpeg and .jpg), WEBP (.webp), and non-animated GIF (.gif)."


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=1111)
