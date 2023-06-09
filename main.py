import os
import sys
import requests
from bs4 import BeautifulSoup
import pandas as pd
from langchain.utilities import GoogleSearchAPIWrapper
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, Document
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import shutil
import time
import ssl
import re

ps = PorterStemmer()
nltk.download('wordnet')

water_solubility_points = {
    "Highly soluble": 10,
    "Soluble": 8,
    "Slightly soluble": 6,
    "less Soluble": 8,
    "Varies": 0,
    "Unknown": 0,
    "N/A": 0,
    "Poorly soluble": -6,
    "Insoluble": -10
}

RETRIES = 3
DELAY = 5
TIMEOUT = 10
TEXT_FILE_CHUNK_SIZE = 1000
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2 MB
NUM_GOOGLE_RESULTS = 10
MAX_DOWNLOADS = 5


def get_response_with_retry(url, retries=RETRIES, delay=DELAY, timeout=TIMEOUT, headers=None):
    for i in range(retries):
        try:
            return requests.get(url, timeout=timeout, headers=headers)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, ssl.SSLEOFError):
            if i < retries - 1:
                time.sleep(delay)
            else:
                raise


def download(df):
    """
    Download top 5 search results txt files from Google.
    :param df: recipe DataFrame w/ File, Ingredient, and Product
    """
    search = GoogleSearchAPIWrapper()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
    }

    # Iterate over each row and download text files for each search prompt
    for index, row in df.iterrows():
        file_name = row['File']
        ingredient = row['Ingredient']
        product = row['ProductSubType']
        search_prompt = f"role of {ingredient} in {product}"
        print(f"searching for: {search_prompt}")

        # Get search results with metadata
        result_list = search.results(search_prompt, NUM_GOOGLE_RESULTS)
        links = [result.get("link", "") for result in result_list]

        # Create a folder to store the text files
        counter = 1
        while True:
            folder_path = f"./text_files/{file_name}_{counter}"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                break
            counter += 1

        downloads = 0
        for link_index, link in enumerate(links):
            if downloads >= MAX_DOWNLOADS:
                break

            try:
                response = get_response_with_retry(link, headers=headers)
                if response.status_code != 200:
                    print(f"Error: {response.status_code} for {link}")
                    continue

                content_type = response.headers.get('Content-Type')
                content_length = int(response.headers.get('Content-Length', '0'))

                if content_type and ('pdf' in content_type or content_length > MAX_FILE_SIZE):
                    print(f"Skipping {link} as it is either a PDF or larger than 2 MB")
                    continue

                soup = BeautifulSoup(response.text, "html.parser")
                page_title = soup.title.string if soup.title else 'Untitled'
                page_text = soup.get_text()

                # Generate a filename for the text file
                filename = f"link_{downloads}.txt"
                filepath = os.path.join(folder_path, filename)

                # Write the text to the file
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write("Document Name: " + search_prompt + str(downloads) + '\n')
                    f.write("Document link: " + link + '\n')
                    f.write(page_text)
                    print(f"Saved {filename} in folder {folder_path}")

                downloads += 1
            except requests.exceptions.ConnectionError:
                print(f"ConnectionError: Unable to connect to {link}")
            except requests.exceptions.Timeout:
                print(f"TimeoutError: Request timed out for {link}")
            except requests.exceptions.TooManyRedirects:
                print(f"TooManyRedirects: Too many redirects for {link}")
            except requests.exceptions.RequestException as e:
                print(f"RequestException: {e}")


def extract_link(path_to_file):
    """
    Extract source link from txt file.
    :param path_to_file: file path
    :return: source link
    """
    with open(path_to_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        # Change pattern if required
        return lines[1].split(': ')[1].strip()


def stem_words(functionalities):
    """
    Stem functionality words.
    :param functionalities: functionalities to stem
    :return: stemmed functionality
    """
    lemmatizer = WordNetLemmatizer()
    stem_words = [ps.stem(functionality) for functionality in functionalities]
    return stem_words


def break_text_files_into_chunks(directory, chunk_size):
    """
    Break text files into chunks.
    :param directory: directory of files
    :param chunk_size: size of each chunk
    :return: chunks
    """
    # Initialize an empty list to hold the chunks
    chunks = []

    # Loop through all the files in the directory
    for filename in os.listdir(directory):
        # Skip any files that are not text files
        if not filename.endswith('.txt'):
            continue

        # Open the file for reading
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            # Read in the first two lines of the file
            first_two_lines = file.readline() + file.readline()

            # Read in the remaining text
            text = file.read()

            # Split the remaining text into chunks of the specified size
            file_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

            # Prepend the first two lines to each chunk
            file_chunks = [first_two_lines + chunk for chunk in file_chunks]

            # Add the chunks to the list of all chunks
            chunks.extend(file_chunks)

    # Return the list of all chunks
    return chunks


def prompt(df):
    """
    Prompt LlamaIndex for an ingredient's functionality with one txt file at a time.
    :param df: recipe dataframe
    :return: recipe dataframe added stem/original functionality, sources, and total links
    """
    df['Stem_Functionality'] = ""
    df['Original_Functionality'] = ""
    df['Source_Links'] = ""
    df['Total_Links'] = ""

    for index, row in df.iterrows():
        path = "text_files/" + row['File'] + "_1"
        num_files = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])

        embedding_folder_path = path + "/current_embedding_folder/"
        if not os.path.exists(embedding_folder_path):
            os.makedirs(embedding_folder_path)

        links = []
        functionalities = []

        for embedding_file_counter in range(0, num_files):
            # Set the paths for the source file and destination folder
            src_path = path + "/link_" + str(embedding_file_counter) + ".txt"
            print("Reading file: " + src_path)
            links.append(extract_link(src_path))

            dest_path = embedding_folder_path

            # Move the file from the source path to the destination path
            shutil.copy2(src_path, dest_path)

            text_list = break_text_files_into_chunks(embedding_folder_path, TEXT_FILE_CHUNK_SIZE)
            documents = [Document(t) for t in text_list]

            gpt_index = GPTSimpleVectorIndex.from_documents(documents)

            query = "What is the role of " + row['Ingredient'] + " in " + row['Product'].title() + "? " + \
                    "Give a one word answer. The word would be the name of the functionality " + \
                    "of " + row['Ingredient'] + " in " + row[
                        'Product'].title() + "that you find out from the input file." + \
                    "Give the functionality as a common noun."

            print("Prompt: " + query)

            response = gpt_index.query(query)

            print(response)

            functionalities.append(response.response.strip())

            file_path_embedding_done = embedding_folder_path + "link_" + str(embedding_file_counter) + ".txt"

            os.remove(file_path_embedding_done)

        print(links)
        print(functionalities)

        functionalities = list(functionalities)
        stem_functionalities = stem_words(functionalities)

        print(stem_functionalities)

        aggregrated_data = {stem: [stem_functionalities.count(stem),
                                   [link for i, link in enumerate(links)
                                    if stem_functionalities[i] == stem],
                                   [functionality for i, functionality in enumerate(functionalities)
                                    if stem_functionalities[i] == stem]]
                            for stem in stem_functionalities}

        print("Aggregated Data: ", aggregrated_data)
        if not aggregrated_data:        
            aggregrated_data = {"fragranc":[
            1,
            [
                "https://www.google.com"
            ],
            [
                "Unknown"
            ]
        ]
        }
        max_count_functionality = max(aggregrated_data, key=lambda k: aggregrated_data[k][0])
        total_links = aggregrated_data[max_count_functionality][0]
        source_links = aggregrated_data[max_count_functionality][1]
        original_functionality = aggregrated_data[max_count_functionality][2]

        print("Key with highest count:", max_count_functionality)
        print("Original Functionalities:", original_functionality)
        print("Count:", total_links)
        print("Links:", source_links)

        print(index)

        df.loc[index, 'Stem_Functionality'] = str(max_count_functionality)
        df.loc[index, 'Original_Functionality'] = str(original_functionality)
        df.loc[index, 'Total_Links'] = str(total_links)
        df.loc[index, 'Source_Links'] = str(source_links)

    return df


def clean_solubility(combined_df):
    """
    Clean solubility data and add default values.
    :param combined_df: recipe dataframe with solubility data
    :return: recipe dataframe with clean solubility data
    """
    combined_df['Water Solubility'].fillna('Unknown', inplace=True)
    combined_df['Oil Solubility'].fillna('Unknown', inplace=True)
    combined_df.drop(['LogP', 'Remarks'], axis=1, inplace=True)
    combined_df['Water Solubility'] = combined_df['Water Solubility'].map(water_solubility_points)
    combined_df['Oil Solubility'] = combined_df['Oil Solubility'].map(water_solubility_points)
    combined_df['Is Phase Predicted'] = "N"
    combined_df['Predicted Phase'] = "Undefined"
    return combined_df


def predict_phase(combined_df):
    """
    Predict a phase is water, oil, or undefined.
    :param combined_df: recipe dataframe with solubility data
    :return: recipe dataframe with predicted phase
    """
    combined_df = clean_solubility(combined_df)

    start_index = 0
    total_water_solubility = 0
    total_oil_solubility = 0
    for index, row in combined_df.iterrows():
        product = row['Product']
        next_product = combined_df.iloc[index + 1]['Product'] if index < len(combined_df) - 1 else None
        phase = row['Phase']
        next_phase = combined_df.iloc[index + 1]['Phase'] if index < len(combined_df) - 1 else None
        total_water_solubility += combined_df.iloc[index]['Water Solubility']
        total_oil_solubility += combined_df.iloc[index]['Oil Solubility']
        if phase != next_phase or product != next_product:
            if total_water_solubility > total_oil_solubility:
                print(start_index + 2, index + 2, "Water Phase")
                combined_df.loc[start_index:index, 'Is Phase Predicted'] = "Y"
                combined_df.loc[start_index:index, 'Predicted Phase'] = "Water"
            elif total_water_solubility < total_oil_solubility:
                print(start_index + 2, index + 2, "Oil Phase")
                combined_df.loc[start_index:index, 'Is Phase Predicted'] = "Y"
                combined_df.loc[start_index:index, 'Predicted Phase'] = "Oil"
            else:
                print(start_index + 2, index + 2, "Undefined")
            total_water_solubility = 0
            total_oil_solubility = 0
            start_index = index + 1
    return combined_df


def matchOld_confidence(df):
    """
    Calculate base confidence points.
    Assign 1 if functionality from LlamaIndex matches the ingredient-to-functionality mapping,
    0 if otherwise.
    :param df: recipe dataframe with Stem_Functionality
    :return: recipe dataframe added matchOld boolean column
    """
    df['FunctionalityOld'] = df['FunctionalityOld'].fillna('')
    df['FunctionalityOld'] = df['FunctionalityOld'].apply(str.lower)
    df['matchOld'] = df.apply(lambda x: 1 if x.Stem_Functionality in x.FunctionalityOld else 0, axis=1)
    return df


def aggregate(df):
    """
    Generate recipe template for each product subtype.
    :param df: recipe dataframe with predicted phase and confidence points.
    :return: recipe template
    """

    def remove_brackets_and_join(lst):
        return ', '.join(lst)

    def add_recipeNo():
        # create a new column Recipe Number
        df['Recipe No'] = ''

        # initialize the recipe number to 1
        recipe_number = 1

        # loop through the rows of the dataframe
        for i in range(len(df)):
            if i == 0 or df.loc[i, 'ProductSubType'] != df.loc[i - 1, 'ProductSubType']:
                recipe_number = 1
                df.loc[i, 'Recipe No'] = recipe_number
            else:
                # check if the current product is different from the previous row
                if df.loc[i, 'File'] != df.loc[i - 1, 'File'] and (
                        df.loc[i, 'ProductSubType'] == df.loc[i - 1, 'ProductSubType']):
                    # if it is, assign a new recipe number
                    recipe_number += 1
                    df.loc[i, 'Recipe No'] = recipe_number

                else:
                    # if it is not, use the same recipe number as the previous row
                    df.loc[i, 'Recipe No'] = df.loc[i - 1, 'Recipe No']

    def clean_list():
        df['Source_Links'] = df['Source_Links'].apply(lambda x: x.replace('[', '').replace(']', ''))
        df['Original_Functionality'] = df['Original_Functionality'].apply(lambda x: x.replace('[', '').replace(']', ''))
        df['Source_Links'] = df['Source_Links'].apply(lambda x: x.replace("'", ''))
        df['Original_Functionality'] = df['Original_Functionality'].apply(lambda x: x.replace("'", ''))

    def first_group():
        grouped_data = df.groupby(['ProductSubType', 'Recipe No', 'Stem_Functionality', 'Predicted Phase']).agg(
            {'Weight': 'sum', 'Confidence Score %': 'sum', 'Source_Links': lambda x: list(x),
             'Original_Functionality': lambda x: list(set(x))})

        # Calculate the weight proportion % for each ProductSubType and Recipe
        grouped_data['Weight Proportion %'] = grouped_data['Weight'] / grouped_data.groupby(
            ['ProductSubType', 'Recipe No', 'Stem_Functionality'])['Weight'].transform('sum') * 100
        # grouped_data['Document_link'].apply(lambda x: list(flatten(x)))
        grouped_data['Source_Links'] = grouped_data['Source_Links'].apply(remove_brackets_and_join)

        grouped_data['Original_Functionality'] = grouped_data['Original_Functionality'].apply(remove_brackets_and_join)
        grouped_data.drop(['Weight'], axis=1, inplace=True)
        return grouped_data

    def second_group(grouped_data):
        # Group the data again by ProductSubType and calculate the max and min weight proportion %
        product_subtype_data = grouped_data.groupby(['ProductSubType', 'Stem_Functionality', 'Predicted Phase']).agg(
            {'Weight Proportion %': ['min', 'max'], 'Confidence Score %': ['min', 'max'],
             'Source_Links': lambda x: list(x), 'Original_Functionality': lambda x: list(set(x))})
        # Rename the columns to 'min_weight_proportion' and 'max_weight_proportion'
        product_subtype_data.columns = ['Min Weight Proportion %', 'Max Weight Proportion %', 'Min Confidence Score %',
                                        'Max Confidence Score %', 'Source_Links', 'Original_Functionality']

        product_subtype_data['Source_Links'] = product_subtype_data['Source_Links'].apply(remove_brackets_and_join)
        product_subtype_data['Original_Functionality'] = product_subtype_data['Original_Functionality'].apply(
            remove_brackets_and_join)
        product_subtype_data.loc[product_subtype_data['Min Weight Proportion %'] == 0, 'Min Weight Proportion %'] = \
            product_subtype_data['Max Weight Proportion %']
        product_subtype_data.reset_index(inplace=True)
        product_subtype_data['Final Functionality'] = product_subtype_data['Stem_Functionality'].map(mapping)
        product_subtype_data.set_index(['ProductSubType', 'Final Functionality', 'Predicted Phase'], inplace=True)
        product_subtype_data['Min Weight Proportion %'] = product_subtype_data['Min Weight Proportion %'].round()
        product_subtype_data['Max Weight Proportion %'] = product_subtype_data['Max Weight Proportion %'].round()
        product_subtype_data['Min Confidence Score %'] = product_subtype_data['Min Confidence Score %'].round()
        product_subtype_data['Max Confidence Score %'] = product_subtype_data['Max Confidence Score %'].round()
        product_subtype_data.rename(
            columns={"Original_Functionality": "Functionality Group", "Final Functionality": "Functionality"},
            inplace=True)
        product_subtype_data.drop(['Stem_Functionality'], axis=1, inplace=True)
        product_subtype_data.reset_index(inplace=True)
        product_subtype_data['Final Functionality'] = product_subtype_data['Final Functionality'].apply(
            lambda x: x.split(',')[0])
        product_subtype_data['Functionality Group'] = product_subtype_data['Functionality Group'].apply(
            lambda x: set(x.split(',')))
        product_subtype_data['Functionality Group'] = product_subtype_data['Functionality Group'].apply(
            lambda x: set([re.sub(r'[^\w\s]', '', y).strip() for y in x]))
        product_subtype_data['Final Functionality'] = product_subtype_data['Final Functionality'].apply(
            lambda x: re.sub(r'[^\w\s]', '', x))
        product_subtype_data['Functionality Group'] = product_subtype_data['Functionality Group'].apply(
            lambda x: str(x).replace('{', '').replace('}', ''))
        product_subtype_data['Functionality Group'] = product_subtype_data['Functionality Group'].apply(
            lambda x: x.replace("'", ''))
        product_subtype_data.set_index(['ProductSubType', 'Final Functionality', 'Predicted Phase'], inplace=True)
        return product_subtype_data

    add_recipeNo()
    df['Phase'] = df['Phase'].fillna('')
    df['Predicted Phase'] = df.apply(
        lambda x: x['Predicted Phase'] + '_' + x['Phase'] if x['Is Phase Predicted'] == 'N' else x['Predicted Phase'],
        axis=1)
    df.drop(['File', 'Product', 'Phase', 'Ingredient', 'ProductType', 'FunctionalityOld', 'Is Phase Predicted',
             'Unnamed: 0.1', 'Unnamed: 0'], axis=1, inplace=True)
    clean_list()
    mapping = dict(zip(df['Stem_Functionality'], df['Original_Functionality']))
    df['Total_sum'] = df['Total_Links'] + df['matchOld']
    df['Confidence Score %'] = round((df['Total_sum'] / df.groupby(['ProductSubType', 'Recipe No', 'Predicted Phase'])[
        'Total_sum'].transform('sum')) * 100, 2)
    grouped_df = first_group()
    template = second_group(grouped_df)
    return template


if __name__ == '__main__':
    recipesFile = sys.argv[1]
    solubilityFile = sys.argv[2]
    gptOutput = "recipe_func_4_new.csv"
    phasePredFile = "recipe_func_pred_4_new.csv"
    outputFile = "recipe_template_4_new.xlsx"
    df = pd.read_csv(recipesFile)

    # if files are already downloaded, skip the download step
    if "--no-download" not in sys.argv:
        download(df)

    df = prompt(df)
    df.to_csv(gptOutput)

    df = pd.read_csv(gptOutput)
    df = matchOld_confidence(df)
    solubility_data = pd.read_excel(solubilityFile)
    combined_df = pd.merge(df, solubility_data, on='Ingredient', how='left')
    combined_df = predict_phase(combined_df)
    combined_df.to_csv(phasePredFile)

    combined_df = pd.read_csv(phasePredFile)
    recipe_template = aggregate(combined_df)
    recipe_template.to_excel(outputFile, index=True)