# ARMLab 2024
import os
import subprocess
import glob
from typing import Optional

import rospy
from gaussian_splatting_py.base_splatfacto import ROSSplatfacto


class Splatfacto3D(ROSSplatfacto):
    def __init__(self, data_dir='bunny_blender_dir', render_uncertainty=False,
                 train_split_fraction=0.85, depth_uncertainty_weight=1.0, rgb_uncertainty_weight=1.0,
                 camera_optimizer_mode='off', quats_lr=5e-4, scales_lr=2e-3, opacities_lr=2e-2,
                 uncertainty_object_mask_weight=0.25, uncertainty_mask_gamma=2.0, uncertainty_bg_floor=0.03,
                 uncertainty_fruit_weight=1.0, uncertainty_leaf_weight=0.55, uncertainty_bg_weight=0.10):
        """
        initialize 3DGS. Calls the ns-train cmd to avoid manual copy paste.

        When the model trains, it will train to 2K steps,
        """
        super(Splatfacto3D, self).__init__(
            data_dir,
            render_uncertainty,
            train_split_fraction,
            depth_uncertainty_weight,
            rgb_uncertainty_weight,
            uncertainty_object_mask_weight,
            uncertainty_mask_gamma,
            uncertainty_bg_floor,
            uncertainty_fruit_weight,
            uncertainty_leaf_weight,
            uncertainty_bg_weight,
        )
        self.training_process: Optional[subprocess.Popen] = None
        self.camera_optimizer_mode = camera_optimizer_mode
        self.quats_lr = float(quats_lr)
        self.scales_lr = float(scales_lr)
        self.opacities_lr = float(opacities_lr)

    def _training_env(self):
        env = os.environ.copy()
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
        devel_glob = os.path.join(workspace_root, "devel", "lib", "python*", "dist-packages")
        devel_paths = [p for p in glob.glob(devel_glob) if os.path.isdir(p)]
        if devel_paths:
            existing_pythonpath = env.get("PYTHONPATH", "")
            pythonpath_parts = devel_paths + ([existing_pythonpath] if existing_pythonpath else [])
            env["PYTHONPATH"] = ":".join([p for p in pythonpath_parts if p])
            rospy.loginfo(f"GS training PYTHONPATH includes: {devel_paths}")
        return env

    def start_training(self, data_dir, steps=15000):
        """
        Start training Gaussian Splatting 
        """
        print("Starting training")
        # outputs dir is the same as the data dir with outputs
        outputs_dir = f'{data_dir}/outputs'
        
        command = f"""ns-train depth-splatfacto --data {data_dir} --output-dir {outputs_dir} """ \
                  f"""--viewer.quit-on-train-completion True """ \
                  f"""--max-num-iterations {steps} """ \
                  f"""--pipeline.model.depth-loss-mult 2.0 """ \
                  f"""--pipeline.model.sh-degree 3 """ \
                  f"""--pipeline.model.learn-semantic-parts True """ \
                  f"""--pipeline.model.uncertainty-object-mask-weight {self.uncertainty_object_mask_weight} """ \
                  f"""--pipeline.model.uncertainty-mask-gamma {self.uncertainty_mask_gamma} """ \
                  f"""--pipeline.model.uncertainty-bg-floor {self.uncertainty_bg_floor} """ \
                  f"""--pipeline.model.uncertainty-fruit-weight {self.uncertainty_fruit_weight} """ \
                  f"""--pipeline.model.uncertainty-leaf-weight {self.uncertainty_leaf_weight} """ \
                  f"""--pipeline.model.uncertainty-bg-weight {self.uncertainty_bg_weight} """ \
                  f"""--pipeline.model.camera-optimizer.mode {self.camera_optimizer_mode} """ \
                  f"""--optimizers.quats.optimizer.lr {self.quats_lr} """ \
                  f"""--optimizers.scales.optimizer.lr {self.scales_lr} """ \
                  f"""--optimizers.opacities.optimizer.lr {self.opacities_lr} """ \
                  f"""nerfstudio-data --train-split-fraction {self.train_split_fraction}"""
        rospy.loginfo(command)

        try:
            if os.name == 'nt':
                self.training_process = subprocess.Popen(command, shell=True, env=self._training_env())
            else:
                self.training_process = subprocess.Popen(
                    command,
                    shell=True,
                    executable='/bin/bash',
                    env=self._training_env(),
                )
        except Exception as e:
            rospy.logerr(f"Failed to start GS training process: {e}")
            self.training_process = None
            return False

        rospy.loginfo(f"Started GS training process pid={self.training_process.pid}")

        rospy.loginfo("Starting GS Training.")
        return True

    def is_training_alive(self) -> bool:
        if self.training_process is None:
            return False
        return self.training_process.poll() is None

    def training_return_code(self):
        if self.training_process is None:
            return None
        return self.training_process.poll()

    
if __name__ == "__main__":
    #data_dir = '/home/ras/few_shot_initial_dataset'
    data_dir = '/media/ras/data1/touch-gs-data/bunny_blender_data'
    #data_dir = '/media/ras/data1/touch-gs-data/bunny_real_data'
    splatfacto = Splatfacto3D(data_dir=data_dir)
    splatfacto.start_training(data_dir)