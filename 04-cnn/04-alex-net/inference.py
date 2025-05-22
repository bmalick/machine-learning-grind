import torch
import torchvision

def predict(model, img, device):
    patch_size=224
    assert patch_size<256, "patch size should be lower than 256"

    center_crop = torchvision.transforms.CenterCrop((patch_size,patch_size))
    h_flip = torchvision.transforms.RandomHorizontalFlip(p=1.)

    img = torchvision.transforms.Resize((256,256))(img)
    centered_patch = center_crop(img)
    top_left_patch = img[:, :patch_size, :patch_size]
    top_right_patch = img[:, -patch_size:, :patch_size]
    bottom_left_patch = img[:, :patch_size, -patch_size:]
    bottom_right_patch = img[:, -patch_size:, -patch_size:]

    patches = [centered_patch, top_left_patch, top_right_patch, bottom_left_patch, bottom_right_patch]

    for im in patches[1:]:
        patches.append(h_flip(im))

    normalize_transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    patches = [normalize_transform(im) for im in patches]
    alex_net = model.to(device)
    batch = torch.stack(patches).to(device)
    output = alex_net(batch)
    output = output.mean(dim=0)
    pred = output.argmax().item()
    probas = torch.nn.functional.softmax(output, dim=0).detach().cpu().numpy()
    return pred, probas
