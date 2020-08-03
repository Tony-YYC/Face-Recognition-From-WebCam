# -*- encoding: utf8 -*-
import logging
from face import catch_face
from face import recognize_face
from train import train_model
import config

def catch(tag):
    catch_face(tag).catch_video()
    logging.info("catch_face done.")

def train():
    train_model()
    logging.info("train done.")

def predict():
    recognize_face().recognize_video()
    logging.info("predict done.")

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    command = input("input command")
    if command == "train":
        train()
    elif command == "predict":
        predict()
    elif command =="videocam":
        tags = input("tag:\t")
        with open(config.NAME_TXT,"a") as file:
            file.write(tags)
        catch(tags)