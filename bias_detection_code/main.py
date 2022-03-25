import pickle
from ieat.api import test, test_all
import torchvision
import torch
import os
import shutil


def main():

    depth = 50
    width = 1

    # Models pretrained by OpenSelfSup
    byol_pretrained = "./pretrains/byol_r50_bs2048_accmulate2_ep200-e3b0c442.pth"
    odc_pretrained = "./pretrains/odc_r50_v1-5af5dd0c.pth"
    moco_v1_pretrained = "./pretrains/moco_r50_v1-4ad89b5c.pth"
    moco_v2_pretrained = "./pretrains/moco_r50_v2-e3b0c442.pth"
    npid_pretrained = "./pretrains/npid_r50-dec3df0c.pth"
    relative_loc_pretrained = "./pretrains/relative_loc_r50-342c9097.pth"
    rotation_pretrained = "./pretrains/rotation_r50-cfab8ebb.pth"
    supervised_pretrained = "./pretrains/imagenet_r50-21352794.pth"

    # Models pretrained by vissl
    simclr_pretrained = "./pretrains/simclr_vissl_200.pth"
    jigsaw_pretrained = "./pretrains/jigsaw_state_dict.pth"
    cluster_fit_pretrained = "./pretrains/cluster_fit_state_dict.pth"
    swav_pretrained = "./pretrains/swav_200.pth"

    random_model_paths = []
    for i in range(3):
        rand_path = f"random_weights_{i}.pth"

        random_model_paths.append(rand_path)
        # load resnet and remove the head
        random_model_weights = torchvision.models.resnet50(pretrained=False).state_dict()
        del random_model_weights['fc.bias']
        del random_model_weights['fc.weight']
        torch.save({'state_dict': random_model_weights}, rand_path)

    models = ['random_torch_1', 'random_torch_2', 'random_torch_3', 'rotation', 'relative_loc', 'odc', 'npid', 'moco_v1', 'moco_v2', 'byol', 'supervised', 'simclr_vissl_200', 'jigsaw', 'cluster_fit', 'swav']
    weights = random_model_paths + [rotation_pretrained, relative_loc_pretrained,  odc_pretrained, npid_pretrained,
               moco_v1_pretrained, moco_v2_pretrained, byol_pretrained, supervised_pretrained, simclr_pretrained, jigsaw_pretrained, cluster_fit_pretrained, swav_pretrained]

    experiments = ['GAP', 'layer_4', 'layer_3', 'layer_2', 'layer_1', 'layer_0']
    n_validations = 1

    emb_path = './embeddings'
    for features in experiments:
        for i in range(n_validations):
            experiment_name = f"run{i}_{features}"

            results = test_all(dict(zip(models, [(features, depth, width, w) for w in weights])), from_cache=False)
            pickle.dump(results, open(f"results/{experiment_name}.pkl", "wb"))

            # clear cached embeddings
            for filename in os.listdir(emb_path):
                file_path = os.path.join(emb_path, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)


if __name__ == '__main__':
    main()