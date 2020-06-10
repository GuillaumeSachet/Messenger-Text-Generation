# Messenger-Text-Generation
Generate messages from Messenger chat (Facebook) with Recurrent Neural Network (character by character)

## Download messages from Facebook

![](Pictures/1.jpg "First step")
Top Right on facebook, go into parameters and privacy

![](Pictures/2.jpg "Second step")
Into parameters

![](Pictures/3.jpg "Third step")
Your facebook datas -> Download your datas

## Run the notebook

Before running you can change detect_messenger_sentence in preprocessing.py to replace banned french words by other words

The model learns output like : <br />
First_person : .......... <br />
Second_person : ......... <br />
... : ... <br />

The algorithm isn't that smart so it will just learn to write like the people in the chat group.
