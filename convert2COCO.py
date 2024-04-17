import json
import cv2
import os

def initialize_category_mapping(original_json):
    """ Initialize category mapping from a given annotations file. """
    with open(original_json, 'r') as f:
        annotations_data = json.load(f)

    category_id_mapping = {}
    current_category_id = 1

    for video_name, video_data in annotations_data.items():
        for annotation in video_data["annotations"]:
            for qset_id, qset in annotation["query_sets"].items():
                object_title = qset["object_title"]
                if object_title not in category_id_mapping:
                    category_id_mapping[object_title] = current_category_id
                    current_category_id += 1

    return category_id_mapping, current_category_id


def extract_frames_from_videos(original_json, clips_root, frames_save_root):
    # print(original_json, clips_root, frames_save_root)
    """ Extract specific frames from video files and save them as images. """
    with open(original_json, 'r') as f:
        annotations_data = json.load(f)
    
    for video_name, video_data in annotations_data.items():
        video_path = os.path.join(clips_root, video_name + '.mp4')
        # print(video_path)
        if not os.path.isfile(video_path):
            print(f"Video file {video_path} not found.")
            continue
        
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        frame_idx = 0

        while success:
            for annotation in video_data["annotations"]:
                for qset_id, qset in annotation["query_sets"].items():
                    frame_numbers = [qset["query_frame"]] + [resp["frame_number"] for resp in qset.get("response_track", [])]
                    if frame_idx in frame_numbers:
                        frame_file_name = f"{video_name}_frame_{frame_idx}.jpg"
                        # print(frame_file_name)
                        cv2.imwrite(os.path.join(frames_save_root, frame_file_name), image)
            success, image = vidcap.read()
            frame_idx += 1
        vidcap.release()

def convert_to_coco(frames_save_root, original_json, output_path, current_category_id, category_id_mapping):
    """ Convert extracted frames and annotations into the COCO dataset format. """
    with open(original_json, 'r') as f:
        annotations_data = json.load(f)

    coco_format = {"images": [], "annotations": [], "categories": []}
    annotation_id = 1 # 每一張照片都不一樣
    image_id = 1 # 指向相關圖像 image 的 id，因為我們一張圖就只有一個物件要找，所以 id 跟 image_id 一樣。

    for video_name, video_data in annotations_data.items():
        for annotation in video_data["annotations"]:
            for qset_id, qset in annotation["query_sets"].items():
                if not qset["is_valid"]:
                    continue

                frame_numbers = [qset["query_frame"]] + [resp["frame_number"] for resp in qset.get("response_track", [])]
                for frame_num in frame_numbers:
                    frame_file = f"{video_name}_frame_{frame_num}.jpg"
                    frame_path = os.path.join(frames_save_root, frame_file)
                    if not os.path.exists(frame_path):
                        continue

                    image = cv2.imread(frame_path)
                    height, width = image.shape[:2]

                    coco_format["images"].append({
                        "id": image_id,
                        "file_name": frame_file, # 從 original_json 的 "query_frame" 來的！
                        "width": width,
                        "height": height
                    })

                    # for categories's name with object_title
                    object_title = qset["object_title"]
                    if object_title not in category_id_mapping:
                        category_id_mapping[object_title] = current_category_id
                        coco_format["categories"].append({
                            "id": current_category_id,
                            "name": object_title,
                            "supercategory": ""
                        })
                        current_category_id += 1

                    coco_format["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id_mapping[object_title],
                        "bbox": [qset["visual_crop"]["x"], qset["visual_crop"]["y"], qset["visual_crop"]["width"], qset["visual_crop"]["height"]],
                        "iscrowd": 0,
                        "area": qset["visual_crop"]["width"] * qset["visual_crop"]["height"]
                    })
                    annotation_id += 1
                    image_id += 1
    
    # 確保包含所有類別
    for category_name, category_id in category_id_mapping.items():
        if not any(category['id'] == category_id for category in coco_format['categories']):
            coco_format['categories'].append({
                "id": category_id,
                "name": category_name,
                "supercategory": ""
            })

    with open(output_path, 'w') as f:
        json.dump(coco_format, f, indent=4)

    return category_id_mapping, current_category_id

if __name__ == "__main__":
    # 需要將路徑改成自己的 paths
    original_train_json = 'DLCV_vq2d_data/vq_train.json'  # Path to your JSON file with annotations
    original_val_json = 'DLCV_vq2d_data/vq_val.json'  # Path to your JSON file with annotations
    clips_root = 'DLCV_vq2d_data/clips'  # Directory path containing all video files
    # 注意！需要自己先創資料夾喔！
    frames_save_train_root = 'DLCV_vq2d_data/saveTrainCOCO'
    frames_save_val_root = 'DLCV_vq2d_data/saveValCOCO'  # Directory path to save the extracted frames
    output_train_path = 'DLCV_vq2d_data/vq_train_coco.json'
    output_val_path = 'DLCV_vq2d_data/vq_val_coco.json'  # Path to save the COCO format dataset
    
    category_id_mapping = {} # for category
    current_category_id = 1 # category 類別 id
    # 下面兩步驟做如果想要分成兩個程式做，可以自行註解！

    # # Extract frames from videos
    # extract_frames_from_videos(original_json, clips_root, frames_save_root)
    # print("finish step 1: extract_frames_from_videos")

    # Initialize category mapping from training annotations
    category_id_mapping, current_category_id = initialize_category_mapping(original_train_json)
    
    # Process training data
    category_id_mapping, current_category_id = convert_to_coco(frames_save_train_root, original_train_json, output_train_path, current_category_id, category_id_mapping)
    
    # Process validation data
    category_id_mapping, current_category_id = convert_to_coco(frames_save_val_root, original_val_json, output_val_path, current_category_id, category_id_mapping)

    print("Finished processing.")