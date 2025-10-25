import os
import torch
import argparse
import pandas as pd
from typing import Optional, Tuple, List
from transformers import DebertaV2TokenizerFast, DebertaV2ForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from src.utils.logging_utils import setup_logger

class InferenceDataset(Dataset):
    BEGIN_ENTITY_TOKEN = "[BEGIN_ENTITY]"
    END_ENTITY_TOKEN = "[END_ENTITY]"

    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def _parse_offsets(self, start_val, end_val):
        if pd.isna(start_val) or pd.isna(end_val):
            return None, None
        try:
            start = int(start_val)
            end = int(end_val)
            return start, end
        except (ValueError, TypeError):
            return None, None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['text'])
        entity_name = str(row['entity title'])
        start, end = self._parse_offsets(row['start'], row['end'])

        marked = text
        if start is not None and end is not None and 0 <= start < end <= len(text):
            marked = f"{text[:start]}{BEGIN_ENTITY_TOKEN}{text[start:end]}{END_ENTITY_TOKEN}{text[end:]}"
        
        ins = self.tokenizer(
            entity_name,
            marked,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': ins['input_ids'].squeeze(0),
            'attention_mask': ins['attention_mask'].squeeze(0),
        }

class SSPInference:
    def __init__(self, model_path: str, tokenizer_path: str, device=Optional[torch.device] = None, batch_size: int=16, max_length: int=512):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        self.logger = setup_logger("ssp_inference", level="INFO")
        # Load model and tokenizer
        self._load_model_and_tokenizer(model_path, tokenizer_path)

    def _load_model_and_tokenizer(self, model_path: str, tokenizer_path: str):
        try:
            self.logger.info(f"Loading tokenizer from: {tokenizer_path}")
            self.tokenizer = DebertaV2TokenizerFast.from_pretrained(tokenizer_path)
            
            self.logger.info(f"Loading model from: {model_path}")
            self.model = DebertaV2ForSequenceClassification.from_pretrained(
                model_path, 
                num_labels=1
            )
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info(f"Model loaded successfully on device: {self.device}")
            self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            self.logger.error(f"Error loading model/tokenizer: {str(e)}")
            raise

    def _parse_offsets(self, start_val, end_val) -> Tuple[Optional[int], Optional[int]]:
        """Parse start and end offsets, handling invalid values."""
        if pd.isna(start_val) or pd.isna(end_val):
            return None, None
        try:
            start = int(start_val)
            end = int(end_val)
            return start, end
        except (ValueError, TypeError):
            return None, None

    def _load_and_preprocess_data(self, input_path: str) -> pd.DataFrame:
        """Load and preprocess the input data."""
        df = pd.read_csv(input_path)
        
        df.rename(columns={
            'article_text': 'text',
            'predicted_entity': 'entity title'
        }, inplace=True)
        
        # Filter valid rows
        valid_rows = []
        for i, row in df.iterrows():
            try:
                start, end = self._parse_offsets(row.get('start', ''), row.get('end', ''))
                if start is not None and end is not None and 0 <= start < end <= len(text):
                    valid_rows.append(i)
            except (ValueError, TypeError):
                continue
        
        df_filtered = df.loc[valid_rows].reset_index(drop=True)
        self.logger.info(f"Filtered {len(df_filtered)} valid samples from {len(df)} total")
        
        return df_filtered
        
    def _run_batch_inference(self, dataloader: DataLoader) -> List[int]:
        """Run batch inference on the dataloader."""
        predictions = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Running Inference"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                logits = self.model(input_ids, attention_mask=attention_mask).logits
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                preds = (probs > 0.5).astype(int)
                predictions.extend(preds)
        
        return predictions
    
    def run_inference(self, input_path: str, output_path: str):
        self.logger.info(f"Starting SSPinference on: {input_path}")
        
        # Load and preprocess data
        df = self._load_and_preprocess_data(input_path)

        dataset = InferenceDataset(df, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        # Run inference
        predictions = self._run_batch_inference(dataloader)

        df['predicted_salience'] = predictions
        df.to_csv(output_path, index=False)
        
        self.logger.info(f"Inference complete. Output saved to: {output_path}")
        self.logger.info(f"Processed {len(df)} samples")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Run RoBERTa SSP inference")
    parser.add_argument("--input_path", type=str, required=True,
                       help="Path to the input CSV file")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to save the output CSV file")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                       help="Path to the tokenizer")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for inference (default: 16)")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length (default: 512)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    inference = SSPInference(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    inference.run_inference(args.input_path, args.output_path)


if __name__ == "__main__":
    main()