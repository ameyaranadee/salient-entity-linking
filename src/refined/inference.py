import csv
import json
import argparse
import pandas as pd
from refined.inference.processor import Refined
from src.utils.logging_utils import setup_logger

class RefinedInference:
    def __init__(self, model_name="wikipedia_model", entity_set="wikipedia"):
        self.refined = Refined.from_pretrained(
            model_name=model_name,
            entity_set=entity_set
        )
        self.logger = setup_logger("refined_inference", level="INFO")
        
    def process_article(self, article_text):
        return self.refined.process_text(article_text)

    def process_articles(self, input_path, output_path):
        test_data = pd.read_csv(input_path)
            
        test_data_df = test_data.drop_duplicates(subset=['article_text']).reset_index(drop=True)
        spans_df = []
        
        for index, row in test_data_df.iterrows():
            article_spans = self.process_article(row['article_text'])
            
            for span in article_spans:
                candidate_entities_str = None
                if span.candidate_entities:
                    candidate_entities_str = str(span.candidate_entities)

                span_dict = {
                    'article_text': row['article_text'],
                    'date': row['date'],
                    'article_title': row['article_title'],
                    'predicted_entity': span.text,
                    'predicted_wiki_ID': span.predicted_entity.wikidata_entity_id if span.predicted_entity else None,
                    'predicted_wiki_title': span.predicted_entity.wikipedia_entity_title if span.predicted_entity else None,
                    'entity_linking_confidence': span.entity_linking_model_confidence_score,  # Confidence score
                    'candidate_entities': candidate_entities_str,
                    'start': span.start,
                    'end': span.start + span.ln,
                }
                spans_df.append(span_dict)

        spans_df = pd.DataFrame(spans_df)
        spans_df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--entity_set", type=str, required=True)
    args = parser.parse_args()

    refined_inference = RefinedInference(model_name=args.model_name, entity_set=args.entity_set)
    refined_inference.process_articles(args.input_path, args.output_path)

if __name__ == "__main__":
    main()