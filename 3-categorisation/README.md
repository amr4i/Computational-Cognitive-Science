# word-similarity

## Authors

Aarsh Prakash Agarwal (150004)
Amrit Singhal (150092)

## Running the code:

For running the code use `python q.py`.
Ensure tha the file `combined.csv` is present in the folder 
while running the code.

For getting the similarity values for word2vec model, First
you need to run the word2vec api with the following command with 
`python word2vec-api.py --model <path to model> --binary BINARY --host localhost --port 5000`. Keep the server running while you run `python q.py`.

Since running the queries was taking a lot of time
We prefered storing the similarity values we recieved into 
a file first. 

Therefore, when running for the first time you must comment
the part of the code where we read from the file. 
Running the code takes a while, about an hour.

After that comment the part where you are writing to the file and uncomment the part where you read from it.


## Output
You will get two scatter plots. One for Human Similarity Score VS scaled NGD similarity scores. Other will be Human SImilarity Score VS scaled word2vec similarity score for all the 353 word pairs.
