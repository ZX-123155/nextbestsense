"""Python3.11* code to segment images in a folder using a UI with Segment Anything 2."""

import os
import argparse
import os.path as osp
import sys

import pickle
from tqdm import tqdm

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import Canvas
from tqdm import tqdm

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _find_sam2_repo() -> str:
    candidates = [
        os.environ.get("SAM2_REPO_DIR", ""),
        osp.expanduser("~/sam2"),
        osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), "../../../sam2")),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        candidate = osp.abspath(candidate)
        if osp.isfile(osp.join(candidate, "sam2", "build_sam.py")):
            return candidate
    return ""


SAM2_REPO_DIR = _find_sam2_repo()
if SAM2_REPO_DIR and SAM2_REPO_DIR not in sys.path:
    sys.path.insert(0, SAM2_REPO_DIR)

from sam2.build_sam import build_sam2_video_predictor

DEFAULT_SAM2_PATH = osp.join(SAM2_REPO_DIR, "checkpoints", "sam2.1_hiera_large.pt") if SAM2_REPO_DIR else ""
DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_PATH = os.environ.get("SAM2_CHECKPOINT", DEFAULT_SAM2_PATH)
MODEL_CFG = os.environ.get("SAM2_MODEL_CFG", DEFAULT_MODEL_CFG)
FRUIT_CLASS_VALUE = 255
LEAF_CLASS_VALUE = 128


def _normalize_model_cfg(model_cfg: str) -> str:
    model_cfg = str(model_cfg).strip().replace("\\", "/")
    if model_cfg.startswith("pkg://"):
        model_cfg = model_cfg[len("pkg://"):]
    if model_cfg.startswith("sam2/"):
        model_cfg = model_cfg[len("sam2/"):]
    if "/configs/" in model_cfg:
        model_cfg = model_cfg.split("/configs/", 1)[1]
        model_cfg = f"configs/{model_cfg}"
    if model_cfg.startswith("/"):
        model_cfg = model_cfg.lstrip("/")
    return model_cfg


MODEL_CFG_NAME = _normalize_model_cfg(MODEL_CFG)


def _resolve_model_cfg_file(model_cfg_name: str) -> str:
    candidates = []
    if SAM2_REPO_DIR:
        candidates.append(osp.join(SAM2_REPO_DIR, "sam2", model_cfg_name))
        candidates.append(osp.join(SAM2_REPO_DIR, model_cfg_name))
    candidates.append(model_cfg_name)
    for candidate in candidates:
        if osp.isfile(candidate):
            return candidate
    return ""


MODEL_CFG_FILE = _resolve_model_cfg_file(MODEL_CFG_NAME)
SAM2_POINTS_FILENAME = 'sam2_points_labels.pkl'

# SAM2 UI on multiple frames (video or not)

def save_mask(mask, root_dir, out_path):
    """Save semantic mask (0:bg, 128:leaf, 255:fruit) as PNG image."""
    mask_root = os.path.join(root_dir, 'masks')
    os.makedirs(mask_root, exist_ok=True)
    mask_path = os.path.join(mask_root, out_path)
    mask = mask.reshape(mask.shape[-2], mask.shape[-1]).astype(np.uint8)
    mask_image = Image.fromarray(mask)
    mask_image.save(mask_path)

def show_mask(mask, ax, obj_id=None, random_color=False):
    """Show the mask on the matplotlib axis."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, class_ids, canvas, point_radius=5):
    """Show the points on the canvas."""
    for coord, label, cls_id in zip(coords, labels, class_ids):
        x, y = coord
        if cls_id == 1:
            color = 'green' if label == 1 else 'darkgreen'
        else:
            color = 'orange' if label == 1 else 'brown'
        canvas.create_oval(x - point_radius, y - point_radius, x + point_radius, y + point_radius, fill=color, outline='white')

def on_click(event, points, labels, class_ids, point_type, active_class, canvas):
    """Add a point on the canvas."""
    x, y = event.x, event.y
    points.append([x, y])
    labels.append(point_type[0])
    class_ids.append(active_class[0])
    show_points(np.array([[x, y]]), np.array([point_type[0]]), np.array([active_class[0]]), canvas)

def on_enter(event, inference_state, predictor, ann_frame_idx, ann_obj_id, frame_names, video_dir, points, labels, class_ids):
    """Propagate the masks and save the points and labels."""
    
    points_np = np.array(points, dtype=np.float32)
    labels_np = np.array(labels, np.int32)
    class_ids_np = np.array(class_ids, np.int32)
    
    # save points and labels as pkl file
    root_dir = video_dir.split('/')[:-1]
    root_dir = f'/{os.path.join(*root_dir)}'
    
    points_labels = {'points': points_np, 'labels': labels_np, 'class_ids': class_ids_np}
    points_labels_path = os.path.join(root_dir, SAM2_POINTS_FILENAME)
    with open(points_labels_path, 'wb') as f:
        pickle.dump(points_labels, f)
        
        
    compute_masks(inference_state, predictor, ann_frame_idx, 
                  ann_obj_id, points_np, labels_np, class_ids_np, root_dir, 
                  video_dir, frame_names)
        
    exit()


def compute_masks(inference_state, predictor, 
                  ann_frame_idx, ann_obj_id, 
                  points_np, labels_np, class_ids_np, root_dir,
                  video_dir, frame_names):
    """Compute the masks for the video/multiple frames."""
    
    class_to_obj = {1: 1, 2: 2}
    for cls_id, obj_id in class_to_obj.items():
        cls_mask = class_ids_np == cls_id
        if not np.any(cls_mask):
            continue
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=obj_id,
            points=points_np[cls_mask],
            labels=labels_np[cls_mask],
        )
    
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        
    vis_frame_stride = 1
    for out_frame_idx in tqdm(range(0, len(frame_names), vis_frame_stride), desc="Visualizing masks"):
        frame_img = np.array(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        h, w = frame_img.shape[:2]
        out_mask_fruit = video_segments[out_frame_idx].get(1, np.zeros((1, h, w), dtype=bool))
        out_mask_leaf = video_segments[out_frame_idx].get(2, np.zeros_like(out_mask_fruit))
        fruit_2d = out_mask_fruit.reshape(out_mask_fruit.shape[-2], out_mask_fruit.shape[-1]).astype(bool)
        leaf_2d = out_mask_leaf.reshape(out_mask_leaf.shape[-2], out_mask_leaf.shape[-1]).astype(bool)
        semantic_mask = np.zeros_like(fruit_2d, dtype=np.uint8)
        semantic_mask[leaf_2d] = LEAF_CLASS_VALUE
        semantic_mask[fruit_2d] = FRUIT_CLASS_VALUE
        
        frame_name = os.path.splitext(frame_names[out_frame_idx])[0]
        
        save_mask(semantic_mask, root_dir, f"{frame_name}.png")

def start_app(video_dir, predictor):
    frame_names = [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)

    ann_frame_idx = 0
    ann_obj_id = 1
    
    # see if points exists 
    root_dir = video_dir.split('/')[:-1]
    root_dir = f'/{os.path.join(*root_dir)}'
    points_labels_path = os.path.join(root_dir, SAM2_POINTS_FILENAME)
    if os.path.exists(points_labels_path):
        with open(points_labels_path, 'rb') as f:
            points_obj = pickle.load(f)
        points_np = points_obj['points']
        labels_np = points_obj['labels']
        class_ids_np = points_obj.get("class_ids")
        if class_ids_np is None:
            class_ids_np = np.ones((points_np.shape[0],), dtype=np.int32)
        
        print("No need to annotate, using existing points..")
        compute_masks(inference_state, predictor, ann_frame_idx, 
                      ann_obj_id, points_np, labels_np, class_ids_np, root_dir, 
                      video_dir, frame_names)
        exit()

    # Points and labels storage
    points = []
    labels = []
    class_ids = []

    # Use a list to store point type (1 for positive, 0 for negative)
    point_type = [1]
    # Active semantic class: 1=fruit, 2=leaf
    active_class = [1]

    # Create the main window
    root = tk.Tk()
    root.title("Image Click UI")

    # Load the first image
    img = Image.open(os.path.join(video_dir, frame_names[ann_frame_idx]))
    img_tk = ImageTk.PhotoImage(img)

    # Create a canvas and display the image
    canvas = Canvas(root, width=img.width, height=img.height)
    canvas.pack()
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

    # Bind the click event to add points
    canvas.bind("<Button-1>", lambda event: on_click(event, points, labels, class_ids, point_type, active_class, canvas))

    # Bind the 'p' key to switch to adding positive points
    root.bind('p', lambda event: point_type.__setitem__(0, 1))

    # Bind the 'n' key to switch to adding negative points
    root.bind('n', lambda event: point_type.__setitem__(0, 0))
    # Bind keys to switch semantic class
    root.bind('1', lambda event: active_class.__setitem__(0, 1))
    root.bind('2', lambda event: active_class.__setitem__(0, 2))

    # Bind the Enter key to propagate masks
    root.bind('<Return>', lambda event: on_enter(event, inference_state, predictor, ann_frame_idx, ann_obj_id, frame_names, video_dir, points, labels, class_ids))
    
    
    print("Starting UI... [1:fruit class, 2:leaf class, p:positive, n:negative, Enter:propagate]")
    # Start the Tkinter main loop
    root.mainloop()


def convert_images_to_jpg(data_dir):
    """Goes through an `image_dir` and converts all images to JPG format."""
    
    jpg_dir = os.path.join(data_dir, 'jpg_images')
    os.makedirs(jpg_dir, exist_ok=True)
    img_dir = os.path.join(data_dir, 'images')

    for filename in os.listdir(img_dir):
        file_path = os.path.join(img_dir, filename)
        if 'depth' in filename:
            continue
        
        base, ext = os.path.splitext(filename)
        if ext.lower() not in ['.jpg', '.jpeg']:
            img = Image.open(file_path)
            img = img.convert('RGB')
            img.save(os.path.join(jpg_dir, f"{base}.jpg"), 'JPEG')
        else:
            img = Image.open(file_path)
            img.save(os.path.join(jpg_dir, filename), 'JPEG')
    
    return jpg_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a directory in nerfstudio format and convert them to JPG for UI segmentation.")
    parser.add_argument("--data_dir", type=str, help="Path to the data directory.")
    
    args = parser.parse_args()

    if not osp.isfile(SAM2_PATH):
        raise FileNotFoundError(f"SAM2 checkpoint not found: {SAM2_PATH}. Set SAM2_CHECKPOINT or SAM2_REPO_DIR.")
    if not MODEL_CFG_FILE:
        raise FileNotFoundError(
            f"SAM2 model cfg not found for '{MODEL_CFG}' (normalized: '{MODEL_CFG_NAME}'). "
            "Set SAM2_MODEL_CFG or SAM2_REPO_DIR."
        )
    
    predictor = build_sam2_video_predictor(MODEL_CFG_NAME, SAM2_PATH)

    # Convert images to jpg and store them in 'jpg_images' directory
    jpg_dir = convert_images_to_jpg(args.data_dir)
    # Run the application with the converted jpg images
    start_app(jpg_dir, predictor)


