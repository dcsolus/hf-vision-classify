import json
import base64
import re
from llm import Label
from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import datetime
import pytz

class GenerateLabel(Label):
    def __init__(self, input_dir:Path, output_dir:Path, context_prompt_path:Path, llm_deployed_model = 'gpt-4o'):
        super().__init__(context_prompt_path, llm_deployed_model)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_dir_temp = self.output_dir / 'temp'
        self._validate_path()

    def _validate_path(self):
        if not all([isinstance(self.input_dir, Path), isinstance(self.output_dir, Path)]):
            raise ValueError(f'input_dir and output_dir must be a pathlib.Path')        

    def encode_image(self, image_path:Path):
        """Encodes an image to a Base64 string"""
        try:
            with image_path.open('rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None
    
    def parse_json_str(self, json_str:str) -> dict:
        """Extract JSON from a string, handling edge cases."""
        try:
            match = re.search(r"```json\s*(.*?)\s*```", json_str, re.DOTALL)
            if match:
                json_data = match.group(1).strip()
            else:
                json_data = json_str.strip()
            return json.loads(json_data)
        except (json.JSONDecodeError, IndexError) as e:
            print(f"Error parsing JSON: {e}")
            return None

    def process_image(self, image_path:Path) -> None:
        """Processes a single image within multi-threading"""
        base64_image = self.encode_image(image_path)
        if base64_image is None:
            raise ValueError(f"Failed to encode image: {image_path}")
        json_response = self.run(base64_image)
        
        # [cache] Save the JSON response to a file
        response = self.parse_json_str(json_response)
        self.output_dir_temp.mkdir(exist_ok=True, parents=True)
        output_path = self.output_dir_temp / f'{image_path.stem}.json'
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(response, f, ensure_ascii=False, indent=4)

    def label(self, max_workers:int=4):
        """Generates labels for images, using threading for parallel execution."""
        files = list(self.input_dir.glob("*.jpg"))
        if not files:
            print(f'No image found in : {self.input_dir}')
            return
        
        frames = self.load_labels()
        if frames:
            files = sorted([file for file in files if file.stem not in frames])
            
            if not files:
                print(f'No new labels to process')
                return
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.process_image, file): file for file in files}
            
            for future in tqdm(as_completed(futures), total=len(files), desc='Processing...'):
                file = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Failed file {file}: {e}")

        # [cache] Load the JSON responses from the files
        self.gather_label_results()

    def gather_label_results(self):
        """Saves the results to a JSON file"""
        label_files = sorted(self.output_dir_temp.glob('*.json'))

        # Use an OrderedDict to maintain the order of the files
        results = OrderedDict()

        for file in label_files:
            try:
                with file.open('r', encoding='utf-8') as f:
                    data = json.load(f, object_pairs_hook=OrderedDict)
                    results[file.stem] = data
            except Exception as e:
                print(f"Error laoding JSON files: {e}")
            else:
                file.unlink() # delete the file after loading

        # [cache] Save the results to a JSON file
        self.output_dir.mkdir(exist_ok=True)
        tz = pytz.timezone('Asis/Seoul')
        now = datetime.datetime.now(tz)
        timestamp = now.isoformat('T', 'seconds').replace(':','-')
        output_file_path = self.output_dir / f'labels_{timestamp}.json'
        with output_file_path.open('w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        self.output_dir_temp.rmdir()

    def load_labels(self)->set:
        """Loads the labels from a JSON file"""
        label_files = sorted(self.output_dir.glob('labels_*.json'))
        if not label_files:
            return None
        frames = set()
        for file in label_files:
            with file.open('r', encoding='utf-8') as f:
                labels = json.load(f, object_pairs_hook=OrderedDict)
                frames.update(labels.keys())
        return frames

def label_df(label_path:Path)->pd.DataFrame:
    with label_path.open('r', encoding='utf-8') as f:
        label_data = json.load(f)

    return pd.DataFrame.from_dict(label_data, orient='index').reset_index()

if __name__=='__main__':
    context_prompt = Path('prompts') / 'label.md'
    input_dir = Path('frames')
    output_dir = Path('labels')
    
    generate_label = GenerateLabel(
        input_dir=input_dir,
        output_dir=output_dir, 
        context_prompt_path=context_prompt
    )
    
    generate_label.label(max_workers=5)

    # label_file = Path('labels') / 'labels.json'
    # df = label_df(label_file)
    # print(df)