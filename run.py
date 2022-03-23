from utils import general
from models import models

data_dir = "D:\Data\DIRLAB\DIRLAB_clean"
out_dir = "D:\Data\DIRLAB\outtest"

case_id = 8

(
    img_insp,
    img_exp,
    landmarks_insp,
    landmarks_exp,
    mask_exp,
    voxel_size,
) = general.load_image_DIRLab(case_id, "{}\Case".format(data_dir))

kwargs = {}
kwargs["verbose"] = False
kwargs["hyper_regularization"] = False
kwargs["jacobian_regularization"] = False
kwargs["bending_regularization"] = True
kwargs["network_type"] = "SIREN"  # Options are "MLP" and "SIREN"
kwargs["save_folder"] = out_dir + str(case_id)
kwargs["mask"] = mask_exp

ImpReg = models.ImplicitRegistrator(img_exp, img_insp, **kwargs)
ImpReg.fit()
new_landmarks_orig, _ = general.compute_landmarks(
    ImpReg.network, landmarks_insp, image_size=img_insp.shape
)

print(voxel_size)
accuracy_mean, accuracy_std = general.compute_landmark_accuracy(
    new_landmarks_orig, landmarks_exp, voxel_size=voxel_size
)

print("{} {} {}".format(case_id, accuracy_mean, accuracy_std))
