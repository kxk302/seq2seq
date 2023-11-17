# seq2seq

Took the example code for an LSTM seq2seq model from here (https://keras.io/examples/nlp/lstm_seq2seq/). 
Added two seq2seq models: One has 2 LSTM cells. The other has 2 LSTM layers with 2 LSTM cells per layer.

To run the scripts, create and activate a virtual environment by running the following command:

> python3 -m venv venv;
> pip install -r requirements.txt;

To run the plain seq2seq model (no layers or cells), run the following command:

> python3 ./scripts/seq2seq_plain.py

To run the seq2seq model with 2 cells (no layers), run the following command:

> python3 ./scripts/seq2seq_cells.py

To run the seq2seq model with 2 layers and 2 cells per layer, run the following command:

> python3 ./scripts/seq2seq_layers.py
