import csv
import pickle
import pandas as pd
import sys
import json
from fuzzywuzzy import fuzz

#Precision and recall on salient entity linking: 
#Precision is % of (salient entity, document) pairs that SEL method predicts that are correct
#Recall is % of (document, salient entity) pairs in ground truth that are correctly output by SEL pipeline
def main(ground_truth_path, sel_output_path):

    #Loading in ground truth split
    ground_truth_df = pd.read_csv(ground_truth_path)
    
    #Loading in end to end salient entity linking df
    sel_output_df = pd.read_csv(sel_output_path)

    #Exclude duplicate article texts
    article_texts = ground_truth_df.drop_duplicates(subset="text", keep="first")["text"].to_list()

    #For ReFined include only predicted salient entities
    #sel_output_df = sel_output_df[sel_output_df["predicted_salience"] == 1]
    sel_output_df = sel_output_df.drop_duplicates(subset=["text", "predicted_wiki_ID"])

    #recall demoniator is number of ground truth salient entities
    recall_denominator = len(ground_truth_df[ground_truth_df["entity salience"] == 1])

    #precision denominator is just number of salient entities output by SEL
    precision_denominator = len(sel_output_df)

    #Number of correctly detected salient entities by SED
    numerator = 0

    for idx, text in enumerate(article_texts):
        #ground truth salient entities for current article
        ground_truth_salient_entities = ground_truth_df[(ground_truth_df["text"] == text) & (ground_truth_df["entity salience"] == 1)]["Q_ID"].to_list()

        #salient entities (after linking) output for current article    
        sel_article_predicted_salient_entities = sel_output_df[sel_output_df["text"] == text]["predicted_wiki_ID"].to_list()

        #casting from float
        # sel_article_predicted_salient_entities = [int(x) for x in sel_article_predicted_salient_entities]

        #looping over ground truth salient entities for current article
        for ground_truth_salient_entity in ground_truth_salient_entities:
            #If current ground truth salient wikipedia ID is in list of predicted salient wikipedia ID's for current article, increase precision, recall
            if ground_truth_salient_entity in sel_article_predicted_salient_entities:
                numerator += 1


    #Calculate SEL metrics and print
    precision = float(numerator/precision_denominator)
    recall = float(numerator/recall_denominator)
    f1 = f1_score(precision, recall)
    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(f'F1 Score: {f1}')

    print(f'precision denominator: {precision_denominator}')
    print(f'recall denominator: {recall_denominator}')
    print(f'numerator: {numerator}')

    return {'precision': precision, 'recall': recall, 'f1': f1}    

def f1_score(precision, recall):
    return float((2*precision*recall)/(precision+recall))


if __name__ == "__main__":

    #checking to see if valid ground truth path is passed as sys.argv[1]
    try:
        ground_truth_path = sys.argv[1]
    except:
        raise Exception("sys.argv[1] not present. Must be valid path to a ground truth CSV file for a dataset")
    
    #checking to see if valid output path is passed as sys.argv[2]
    try:
        output_path = sys.argv[2]
    except:
        raise Exception("sys.argv[2] not present. Must be a valid path to a output CSV file which contains predicted salient entities (wikipedia ID's) for a dataset")

    results = main(ground_truth_path, output_path)
    print(results)
