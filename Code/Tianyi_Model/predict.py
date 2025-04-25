import torch
import torchvision.transforms as transforms
from PIL import Image
import os

# Import modules from our project
from Tianyi_Model.model import InvariantMappingModel, build_encoder
from Tianyi_Model.data_preparation import CIFAR10_MEAN, CIFAR10_STD # Need normalization constants

# Device configuration
# device = torch.device(
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps" if torch.backends.mps.is_available() else "cpu"
# )
# print(f"Using device: {device}")

# --- Model and Data Parameters (MUST match training) ---
# EMBEDDING_DIM = 64
# NUM_CLASSES = 4
# MODEL_SAVE_PATH = './output/invariant_mapping_model.pth' # Path where the model was saved


# --- Image Transformation (MUST match test transformations) ---
# For prediction on a single image, we apply the same normalization
# as the test set during training. We might also resize if input isn't 32x32.
predict_transform = transforms.Compose([
    transforms.Resize((32, 32)), # Ensure image is 32x32
    transforms.ToTensor(),
    # transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

def load_trained_model(model_path, encoder_name, embedding_dim, num_classes, device):
    """
    Loads the trained model parameters from a file.

    Args:
        model_path (str): Path to the saved model state_dict.
        embedding_dim (int): The embedding dimension used during training.
        num_classes (int): The number of classes used during training.

    Returns:
        torch.nn.Module: The loaded model in evaluation mode.
    """
    encoder = build_encoder(encoder_name, embedding_dim)
    model = InvariantMappingModel(encoder, num_classes)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. "
                                f"Please run train.py first to save the model.")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # Set model to evaluation mode
    print(f"Model loaded successfully from {model_path}")
    return model

def predict_image_class(model, image_tensor, device, output_embeding=False):
    """
    Predicts the class of a single image.

    Args:
        model (torch.nn.Module): The trained model.
        image_tensor (torch.Tensor): 3 * 32 * 32

    Returns:
        tuple: A tuple containing (predicted_class_name, confidence).
    """
    # CIFAR-10 class names
    CIFAR10_CLASSES = ['cat', 'dog', 'ship', 'truck']

    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    # Perform inference
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor) # Get logits

    # Get predicted class and confidence
    probabilities = torch.softmax(outputs, dim=1)
    confidence, predicted_index = torch.max(probabilities, dim=1)

    predicted_class_name = CIFAR10_CLASSES[predicted_index.item()]
    confidence_score = confidence.item()

    # You can also get the embedding if needed
    embedding = model.get_embedding(image_tensor)
    if output_embeding:
        return predicted_class_name, confidence_score, embedding
    return predicted_class_name, confidence_score
