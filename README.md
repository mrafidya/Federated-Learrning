# Identification of The Authenticity of Signatures Using The Federated Learning Method

Signature is a sign as a symbol of the name written by the person's own hand as a personal marker. Currently, to identify the authenticity of the signatures must be done manually one another. This will be very challenging if there are thousands or millions of signatures that must identified in a certain time frame. In this case, the process of identify signature authenticity was applied using the federated learning method. Every round of federation learning runs by running an initial learning model in the form of a convolutional neural network by two client with 10 epochs and the learning performance parameters are sent to the server to be combined. Federation learning rounds in this program were twice. In my case, Client 1 stores 941 data images for the learning process and 252 data images for the testing process while client 2 stores 708 images for the learning process and 248 iamges for the testing process. Based on the test results, each client is able to learn with local data and achieve an accuracy of up to 99%.

## Before running the program

Please adjust the dataset name based on your dataset

## How to run the program

1. Run server.py

2. Run client1.py

3. Run client2.py

4. Training result in each round will be shown in graphical form

5. After 2 round, sever will show the aggregated testing accuracy for each round

Please note that this program will use localhost port number 8080
