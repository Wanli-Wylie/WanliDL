import json

if __name__ == '__main__':

    with open("./datasets/vision/MSCOCO/annotations/captions_train2014.json") as f:
        records = json.load(f).get("annotations")

    image_captions = {}
    
    # Process each record
    for record in records:
        image_id = record['image_id']
        caption = record['caption']

        # If the image_id is already in the dictionary, append the caption to the list
        if image_id in image_captions:
            image_captions[image_id].append(caption)
        else:
            # Otherwise, create a new list for this image_id
            image_captions[image_id] = [caption]

    with open('./datasets/vision/MSCOCO/train2014_captions.json', 'w') as f:
        json.dump(image_captions, f)