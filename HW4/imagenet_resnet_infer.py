import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import Ridge 

# Load the pre-trained ResNet18 model
model = models.resnet18(pretrained=True)
model.eval()  # Set model to evaluation mode

# Define the image preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]   
    )
])

# Load the ImageNet class index mapping
with open("imagenet_class_index.json") as f:
    class_idx = json.load(f)
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
idx2synset = [class_idx[str(k)][0] for k in range(len(class_idx))]
id2label = {v[0]: v[1] for v in class_idx.values()}


#lime
def lime_explanation(image_path, predicted_class, num_samples=1000, num_features=50):
    input_image = Image.open(image_path).convert('RGB')
    input_image_resized = input_image.resize((224, 224))
    input_array = np.array(input_image_resized) / 255.0

    segments = slic(input_array, n_segments=num_features, compactness=10, sigma=1, start_label=0)
    num_superpixels = len(np.unique(segments))

    # Get device of model
    device = next(model.parameters()).device

    perturbations = []
    predictions = []

    for _ in range(num_samples):
        # random select
        mask = np.random.binomial(1, 0.5, size=num_superpixels)
        perturbations.append(mask)

        perturbed_image = input_array.copy()
        for i in range(num_superpixels):
            if mask[i] == 0:
                perturbed_image[segments == i] = 0.5

        perturbed_pil = Image.fromarray((perturbed_image * 255).astype(np.uint8))
        perturbed_tensor = preprocess(perturbed_pil).to(device)
        with torch.no_grad():
            perturbed_output = model(perturbed_tensor.unsqueeze(0))
            perturbed_probs = F.softmax(perturbed_output, dim=1)[0]
        predictions.append(perturbed_probs[predicted_class].item())

    X = np.array(perturbations)
    y = np.array(predictions)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)

    importance = ridge.coef_

    importance_map = np.zeros((224, 224))
    for i in range(num_superpixels):
        importance_map[segments == i] = importance[i]

    return importance_map


#smoothgrad
def smoothgrad_explanation(input_tensor, predicted_class, num_samples=50, noise_level=0.15):
    # Get device of model
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    total_gradients = torch.zeros_like(input_tensor)

    for _ in range(num_samples):
        # add Gaussian noise
        noise = torch.randn_like(input_tensor) * noise_level
        noisy_input = input_tensor + noise
        noisy_input.requires_grad = True

        output = model(noisy_input.unsqueeze(0))
        score = output[0, predicted_class]
        model.zero_grad()
        score.backward()

        total_gradients += noisy_input.grad.data


    smooth_grad = total_gradients / num_samples
    smooth_grad_map = torch.abs(smooth_grad).sum(dim=0).numpy()

    return smooth_grad_map



def visualize_explanations(image_path, lime_map, smoothgrad_map, predicted_label, image_name):
    """Visualize original image and both explanation methods"""
    input_image = Image.open(image_path).convert('RGB')
    input_image_resized = input_image.resize((224, 224))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(input_image_resized)
    axes[0].set_title(f'Original Image\nPredicted: {predicted_label}')
    axes[0].axis('off')

    # LIME explanation
    im1 = axes[1].imshow(lime_map, cmap='RdBu_r')
    axes[1].set_title('LIME Explanation')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # SmoothGrad explanation
    im2 = axes[2].imshow(smoothgrad_map, cmap='hot')
    axes[2].set_title('SmoothGrad Explanation')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()

    # Save figure
    output_path = f"explanation_{image_name.replace('.JPEG', '.png')}"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def compute_ranking_correlation(lime_map, smoothgrad_map):
    """Compute Kendall-Tau and Spearman correlation between two explanation methods"""
    lime_flat = lime_map.flatten()
    smoothgrad_flat = smoothgrad_map.flatten()

    kendall_tau, _ = kendalltau(lime_flat, smoothgrad_flat)
    spearman_corr, _ = spearmanr(lime_flat, smoothgrad_flat)

    return kendall_tau, spearman_corr


imagenet_path = './imagenet_samples'

# List of image file paths
image_paths = os.listdir(imagenet_path)

results = []

for img_path in sorted(image_paths):
    if not img_path.endswith('.JPEG'):
        continue

    print(f"\n{'='*60}")
    print(f"Analyzing: {img_path}")
    print(f"{'='*60}")
    # Open and preprocess the image
    my_img = os.path.join(imagenet_path, img_path)
    input_image = Image.open(my_img).convert('RGB')
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

    # Move the input and model to GPU if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)

    # Get the predicted class index
    _, predicted_idx = torch.max(output, 1)
    predicted_idx = predicted_idx.item()
    predicted_synset = idx2synset[predicted_idx]
    predicted_label = idx2label[predicted_idx]

    print(f"Predicted label: {predicted_synset} ({predicted_label})")

    # Generate explanations
    print("Computing LIME explanation...")
    lime_map = lime_explanation(my_img, predicted_idx)

    print("Computing SmoothGrad explanation...")
    smoothgrad_map = smoothgrad_explanation(input_tensor, predicted_idx)

    # Visualize
    output_path = visualize_explanations(my_img, lime_map, smoothgrad_map, predicted_label, img_path)
    print(f"Saved visualization to: {output_path}")

    # Compute correlations
    kendall, spearman = compute_ranking_correlation(lime_map, smoothgrad_map)
    print(f"Kendall-Tau: {kendall:.4f}, Spearman: {spearman:.4f}")

    results.append({
        'image_name': img_path,
        'predicted_label': predicted_label,
        'predicted_prob': predicted_idx,
        'kendall_tau': kendall,
        'spearman': spearman
    })

# Summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"{'Image':<25} {'Prediction':<20} {'Kendall':<10} {'Spearman':<10}")
print("-" * 65)
for r in results:
    print(f"{r['image_name']:<25} {r['predicted_label']:<20} {r['kendall_tau']:<10.4f} {r['spearman']:<10.4f}")