import os
import clip
import torch
from torchvision.datasets import CIFAR100
import pandas as pd
from tqdm import tqdm 

PROMPTS = [
    "This is a photo of a person with 5 o'clock shadow",
    "This is a photo of a person with arched eyebrows",
    "This is a photo of a person who is attractive",
    "This is a photo of a person with bags under eyes",
    "This is a photo of a person who is bald",
    "This is a photo of a person with bangs",
    "This is a photo of a person with big lips",
    "This is a photo of a person with a big nose",
    "This is a photo of a person with black hair",
    "This is a photo of a person with blond hair",
    "This is a photo of a person with blurry",
    "This is a photo of a person with brown hair",
    "This is a photo of a person with bushy eyebrows",
    "This is a photo of a person who is chubby",
    "This is a photo of a person with a double chin",
    "This is a photo of a person with eyeglasses",
    "This is a photo of a person with a goatee",
    "This is a photo of a person with gray hair",
    "This is a photo of a person with heavy makeup",
    "This is a photo of a person with high cheekbones",
    "This is a photo of a man",
    "This is a photo of a person with mouth slightly open",
    "This is a photo of a person with a mustache",
    "This is a photo of a person with narrow eyes",
    "This is a photo of a person without a beard",
    "This is a photo of a person with an oval face",
    "This is a photo of a person with pale skin",
    "This is a photo of a person with a pointy nose",
    "This is a photo of a person with a receding hairline",
    "This is a photo of a person with rosy cheeks",
    "This is a photo of a person with sideburns",
    "This is a photo of a person who is smiling",
    "This is a photo of a person with straight hair",
    "This is a photo of a person with wavy hair",
    "This is a photo of a person wearing earrings",
    "This is a photo of a person wearing a hat",
    "This is a photo of a person wearing lipstick",
    "This is a photo of a person wearing a necklace",
    "This is a photo of a person wearing a necktie",
    "This is a photo of a person who is young",
]

NEGATIVE_PROMPTS = [
    "This is a photo of a person without 5 o'clock shadow",
    "This is a photo of a person without arched eyebrows",
    "This is a photo of a person who is not attractive",
    "This is a photo of a person without bags under eyes",
    "This is a photo of a person who is not bald",
    "This is a photo of a person without bangs",
    "This is a photo of a person without big lips",
    "This is a photo of a person without a big nose",
    "This is a photo of a person without black hair",
    "This is a photo of a person without blond hair",
    "This is a photo of a person without blurry",
    "This is a photo of a person without brown hair",
    "This is a photo of a person without bushy eyebrows",
    "This is a photo of a person who is not chubby",
    "This is a photo of a person without a double chin",
    "This is a photo of a person without eyeglasses",
    "This is a photo of a person without a goatee",
    "This is a photo of a person without gray hair",
    "This is a photo of a person without heavy makeup",
    "This is a photo of a person without high cheekbones",
    "This is a photo of a female",
    "This is a photo of a person with mouth closed",
    "This is a photo of a person without a mustache",
    "This is a photo of a person without narrow eyes",
    "This is a photo of a person with a beard",
    "This is a photo of a person without an oval face",
    "This is a photo of a person without pale skin",
    "This is a photo of a person without a pointy nose",
    "This is a photo of a person without a receding hairline",
    "This is a photo of a person without rosy cheeks",
    "This is a photo of a person without sideburns",
    "This is a photo of a person who is not smiling",
    "This is a photo of a person with curly hair",
    "This is a photo of a person with short hair",
    "This is a photo of a person not wearing earrings",
    "This is a photo of a person not wearing a hat",
    "This is a photo of a person not wearing lipstick",
    "This is a photo of a person not wearing a necklace",
    "This is a photo of a person not wearing a necktie",
    "This is a photo of a person who is old",
]

def load_model(device):
    # Load the model
    model, preprocess = clip.load('ViT-L/14', device)
    return model, preprocess

import torch
from PIL import Image
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, preprocess):
        self.df = df
        self.preprocess = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # image_path = f"../datasets/celeba/img_align_celeba/{row['image_id']}"
        image_path = f"../datasets/fairface/{row['file']}"
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image)
        return image_input

def main():
    import numpy as np

    df = pd.read_csv("../datasets/celeba/list_attr_celeba.csv")
    attributes = df.columns[1:].tolist()

    fairface_df = pd.read_csv("../datasets/fairface/fairface_label_train.csv")

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_model(device)

    ds = CustomDataset(fairface_df, preprocess)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=4, pin_memory=True, drop_last=False, shuffle=False)
    
    for attr in attributes:
        print(f"Attribute: {attr}")
        print(f"Prompt: {PROMPTS[attributes.index(attr)]}")
        print(f"Negative Prompt: {NEGATIVE_PROMPTS[attributes.index(attr)]}")
        print("\n")

        # Take positive prompt and negative prompt and tokenize
        text_inputs = clip.tokenize([PROMPTS[attributes.index(attr)], NEGATIVE_PROMPTS[attributes.index(attr)]]).to(device)
        
        # Calculate features 
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)


        batch = []
        attribute_similarities = []

        # for i in tqdm(range(len(fairface_df))):
        for batch in tqdm(dataloader):
            image_inputs = batch.to(device)

            with torch.no_grad():
                image_features = model.encode_image(image_inputs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            attribute_similarities.append(similarity.cpu().numpy())

            batch = []

        attribute_similarities = np.concatenate(attribute_similarities)

        # Save the similarities
        import pickle
        with open(f"../datasets/fairface/{attr}_similarities.pkl", "wb") as f:
            pickle.dump(attribute_similarities, f)
            
if __name__ == "__main__":
    main()
