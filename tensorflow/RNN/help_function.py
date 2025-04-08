import os
import tensorflow as tf
import re
import nltk
import unicodedata

DATA_DIR = "./data"
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")


def download_and_read(urls):
    texts = []
    for i, url in enumerate(urls):
        filename = "ex1-{:d}.txt".format(i)

        # Check directly in the Keras datasets directory
        keras_datasets_dir = os.path.expanduser("./datasets")
        filepath = os.path.join(keras_datasets_dir, filename)

        if os.path.exists(filepath):
            print(f"File {filename} already exists in Keras datasets, using it.")
            p = filepath
        else:
            print(f"Downloading {filename}...")
            p = tf.keras.utils.get_file(filename, url, cache_dir=".")

        text = open(p, "r").read()
        # remove byte order mark
        text = text.replace("\ufeff", "")
        # remove newlines
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        # add it to the list
        texts.extend(text)
    return texts


def split_train_labels(sequence):
    input_seq = sequence[0:-1]
    output_seq = sequence[1:]
    return input_seq, output_seq


def generate_text(
    model, start_string, char2idx, idx2char, num_generate=1000, temperature=1.0
):
    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Reset the model state
    model.reset_states()

    for i in range(num_generate):
        predictions = model(input_eval)
        # Remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # Using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # Pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        # Check if the predicted_id is within the valid range of idx2char
        if predicted_id in idx2char:
            char = idx2char[predicted_id]
            # Only add ASCII characters (0-127) to avoid strange symbols
            if ord(char) < 128:
                text_generated.append(char)
            else:
                # Replace non-ASCII with space
                text_generated.append(" ")
        else:
            # If not in the dictionary, add a space
            text_generated.append(" ")

    return start_string + "".join(text_generated)


def download_and_read_commond(url):
    # Extract the filename from the URL
    filename = url.split("/")[-1]

    # Check if the file already exists in the datasets directory
    datasets_dir = os.path.expanduser("./datasets")

    # Look for any file matching the pattern *sentences.zip
    existing_zip = None
    if os.path.exists(datasets_dir):
        for file in os.listdir(datasets_dir):
            if file.endswith("sentences.zip"):
                existing_zip = os.path.join(datasets_dir, file)
                print(f"Found existing file: {existing_zip}")
                break

    if existing_zip:
        # Use the existing file
        path = existing_zip
        # Extract if needed
        extracted_dir = os.path.join(datasets_dir, "sentiment labelled sentences")
        if not os.path.exists(extracted_dir):
            import zipfile

            with zipfile.ZipFile(path, "r") as zip_ref:
                zip_ref.extractall(datasets_dir)
    else:
        # Download the file
        print(f"Downloading {filename}...")
        path = tf.keras.utils.get_file(filename, url, extract=True, cache_dir=".")
        extracted_dir = os.path.join(
            os.path.dirname(path), "sentiment labelled sentences"
        )

    # If the directory doesn't exist, try to find it
    if not os.path.exists(extracted_dir):
        # Search in the datasets directory for any folder containing "sentiment"
        for root, dirs, _ in os.walk(datasets_dir):
            for d in dirs:
                if "sentiment" in d.lower():
                    extracted_dir = os.path.join(root, d)
                    break
            if os.path.exists(extracted_dir):
                break

    # Read the labeled sentences
    labeled_sentences = []

    for labeled_filename in os.listdir(extracted_dir):
        if labeled_filename.endswith(".txt"):
            file_path = os.path.join(extracted_dir, labeled_filename)
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    # Each line has format: text<tab>label
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        text, label = parts
                        labeled_sentences.append((text, int(label)))

    return labeled_sentences


def download_and_read_for_mtm(dataset_dir, num_pairs=None):
    """
    Download and read a dataset from a given URL.

    Args:
        url (str): The URL of the dataset.

    Returns:
        list: A list of tuples containing the text and label of each sentence.
    """
    sent_filename = os.path.join(dataset_dir, "treebank-sents.txt")
    poss_filename = os.path.join(dataset_dir, "treebank-poss.txt")
    if not (os.path.exists(sent_filename) and os.path.exists(poss_filename)):

        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        fsents = open(sent_filename, "w")
        fposs = open(poss_filename, "w")
        sentences = nltk.corpus.treebank.tagged_sents()
        for sent in sentences:
            fsents.write(" ".join([w for w, p in sent]) + "\n")
            fposs.write(" ".join([p for w, p in sent]) + "\n")
        fsents.close()
        fposs.close()
    sents, poss = [], []
    with open(sent_filename, "r") as fsent:
        for idx, line in enumerate(fsent):
            sents.append(line.strip())
            if num_pairs is not None and idx >= num_pairs:
                break
    with open(poss_filename, "r") as fposs:
        for idx, line in enumerate(fposs):
            poss.append(line.strip())
            if (
                num_pairs is not None and idx >= num_pairs
            ):  # Fixed: moved the break inside the loop
                break
    return sents, poss


def preprocess_sentence(sent):
    sent = "".join(
        [
            c
            for c in unicodedata.normalize("NFD", sent)
            if unicodedata.category(c) != "Mn"
        ]
    )
    sent = re.sub(r"([!.?])", r" \1", sent)
    sent = re.sub(r"[^a-zA-Z!.?]+", r" ", sent)
    sent = re.sub(r"\s+", " ", sent)
    sent = sent.lower()
    return sent


def download_and_read_translations(num_sent_pairs=10000):
    # Create datasets directory if it doesn't exist
    datasets_dir = os.path.expanduser("./datasets")
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)

    # Path to the French-English dataset
    local_file = os.path.join(datasets_dir, "fra.txt")
    zip_file = os.path.join(datasets_dir, "fra-eng.zip")

    # checkfile if exist
    if os.path.exists(local_file):
        print(f"file {local_file} exist。now useing it")
    else:
        print(f"Downloading French-English dataset...")
        url = "http://www.manythings.org/anki/fra-eng.zip"

        #
        if os.path.exists(zip_file):
            print(f"zip file {zip_file} is exist，now useing it。")
        else:
            # Use requests library with a custom User-Agent if available
            try:
                import requests

                print("Using requests library for download...")
                headers = {
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
                }
                response = requests.get(url, headers=headers, stream=True)
                response.raise_for_status()  # Raise an exception for HTTP errors

                # Save the zip file
                with open(zip_file, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Download complete. Saved to {zip_file}")

            except (ImportError, Exception) as e:
                print(f"Failed to download using requests: {e}")
                print("Trying alternative download method...")

                # Try using curl as a fallback
                import subprocess

                try:
                    print("Using curl for download...")
                    subprocess.run(
                        ["curl", "-A", "Mozilla/5.0", "-L", url, "-o", zip_file],
                        check=True,
                    )
                    print(f"Download complete. Saved to {zip_file}")
                except Exception as e:
                    print(f"Failed to download using curl: {e}")
                    print(
                        "Please download the file manually from http://www.manythings.org/anki/fra-eng.zip"
                    )
                    print(f"and place it in {datasets_dir}")
                    return [], [], []

        # Extract the zip file
        if os.path.exists(zip_file):
            import zipfile

            try:
                with zipfile.ZipFile(zip_file, "r") as zip_ref:
                    zip_ref.extractall(datasets_dir)
                print(f"Extracted zip file to {datasets_dir}")

                # Look for fra.txt in the extracted files
                for root, _, files in os.walk(datasets_dir):
                    for file in files:
                        if file == "fra.txt":
                            src_file = os.path.join(root, file)
                            if (
                                src_file != local_file
                            ):  # Only copy if not already in the right place
                                import shutil

                                shutil.copy(src_file, local_file)
                                print(f"Copied {src_file} to {local_file}")
                            break
            except Exception as e:
                print(f"Failed to extract zip file: {e}")
                return [], [], []

    en_sents, fr_sents_in, fr_sents_out = [], [], []

    # Check if file exists before opening
    if os.path.exists(local_file):
        with open(local_file, "r", encoding="utf-8") as fin:
            for i, line in enumerate(fin):
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    en_sent, fr_sent = parts[0], parts[1]
                    en_sent = [w for w in preprocess_sentence(en_sent).split()]
                    fr_sent = preprocess_sentence(fr_sent)
                    fr_sent_in = [w for w in ("BOS " + fr_sent).split()]
                    fr_sent_out = [w for w in (fr_sent + " EOS").split()]
                    en_sents.append(en_sent)
                    fr_sents_in.append(fr_sent_in)
                    fr_sents_out.append(fr_sent_out)
                    if i >= num_sent_pairs - 1:
                        break
    else:
        print(f"Error: Could not find dataset file at {local_file}")

    return en_sents, fr_sents_in, fr_sents_out
