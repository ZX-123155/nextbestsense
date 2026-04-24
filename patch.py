with open("/home/ras/NextBestSense/FisherRF-ns/nerfstudio/data/datasets/depth_dataset.py", "r") as f:
    text = f.read()

old_text = """        # mde depth is the file path with a mde prefix
        name = filepath.name
        # insert mde after the number but before the depth.png
        mde_name  = name[:name.find("depth")] + "mde_" + name[name.find("depth"):]
        
        mde_filepath = filepath.parent / (mde_name)
        mde_depth_image = get_depth_image_from_path(
            filepath=mde_filepath, height=height, width=width, scale_factor=1.0
        )

        return {"depth_image": depth_image,  "mde_depth_image": mde_depth_image}"""

new_text = """        # mde depth is the file path with a mde prefix
        name = filepath.name
        if "depth" in name:
            mde_name = name[:name.find("depth")] + "mde_" + name[name.find("depth"):]
        else:
            mde_name = "mde_" + name
        
        mde_filepath = filepath.parent / (mde_name)
        if mde_filepath.exists():
            mde_depth_image = get_depth_image_from_path(
                filepath=mde_filepath, height=height, width=width, scale_factor=1.0
            )
        else:
            mde_depth_image = depth_image.clone()

        return {"depth_image": depth_image,  "mde_depth_image": mde_depth_image}"""

text = text.replace(old_text, new_text)

# Also let's handle the exact string without the empty line if it fails
import re
text = re.sub(r'mde_name\s*=\s*name\[:name\.find\("depth"\)\] \+ "mde_" \+ name\[name\.find\("depth"\):\]\s*mde_filepath = filepath\.parent / \(mde_name\)\s*mde_depth_image = get_depth_image_from_path\(\s*filepath=mde_filepath, height=height, width=width, scale_factor=1\.0\s*\)', 
    """if "depth" in name:
            mde_name = name[:name.find("depth")] + "mde_" + name[name.find("depth"):]
        else:
            mde_name = "mde_" + name
        
        mde_filepath = filepath.parent / mde_name
        if mde_filepath.exists():
            mde_depth_image = get_depth_image_from_path(
                filepath=mde_filepath, height=height, width=width, scale_factor=1.0
            )
        else:
            mde_depth_image = depth_image.clone()""", text)


with open("/home/ras/NextBestSense/FisherRF-ns/nerfstudio/data/datasets/depth_dataset.py", "w") as f:
    f.write(text)
