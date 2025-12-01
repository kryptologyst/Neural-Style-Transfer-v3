"""Streamlit demo for Neural Style Transfer."""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nst import (
    StyleTransferSampler,
    Config,
    set_seed,
    get_device,
    denormalize_tensor
)


def load_config():
    """Load default configuration."""
    config = Config()
    return config


def tensor_to_image(tensor):
    """Convert tensor to PIL Image."""
    # Denormalize
    tensor = denormalize_tensor(tensor)
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and PIL
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    numpy_image = tensor.permute(1, 2, 0).cpu().numpy()
    numpy_image = (numpy_image * 255).astype(np.uint8)
    
    return Image.fromarray(numpy_image)


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Neural Style Transfer",
        page_icon="🎨",
        layout="wide"
    )
    
    st.title("🎨 Neural Style Transfer Demo")
    st.markdown("Transform your images with artistic styles using deep learning!")
    
    # Load configuration
    config = load_config()
    
    # Sidebar controls
    st.sidebar.header("Settings")
    
    # Model settings
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["vgg19", "fast_nst", "multi_scale"],
        index=0,
        help="Choose the neural style transfer model"
    )
    
    # Training parameters
    num_epochs = st.sidebar.slider(
        "Number of Epochs",
        min_value=50,
        max_value=1000,
        value=500,
        step=50,
        help="Number of optimization iterations"
    )
    
    learning_rate = st.sidebar.slider(
        "Learning Rate",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Learning rate for optimization"
    )
    
    style_weight = st.sidebar.slider(
        "Style Weight",
        min_value=1e4,
        max_value=1e7,
        value=1e6,
        step=1e4,
        format="%.0e",
        help="Weight for style loss"
    )
    
    content_weight = st.sidebar.slider(
        "Content Weight",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="Weight for content loss"
    )
    
    # Device selection
    device_options = ["auto", "cpu"]
    if torch.cuda.is_available():
        device_options.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_options.append("mps")
    
    device = st.sidebar.selectbox(
        "Device",
        device_options,
        index=0,
        help="Device to run the model on"
    )
    
    # Update config
    config.set("model.type", model_type)
    config.set("training.num_epochs", num_epochs)
    config.set("training.learning_rate", learning_rate)
    config.set("training.style_weight", style_weight)
    config.set("training.content_weight", content_weight)
    config.set("system.device", device)
    
    # Main content
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.header("Content Image")
        content_file = st.file_uploader(
            "Upload content image",
            type=["jpg", "jpeg", "png"],
            help="Upload the image you want to stylize"
        )
        
        if content_file is not None:
            content_image = Image.open(content_file).convert("RGB")
            st.image(content_image, caption="Content Image", use_column_width=True)
        else:
            st.info("Please upload a content image")
            content_image = None
    
    with col2:
        st.header("Style Image")
        style_file = st.file_uploader(
            "Upload style image",
            type=["jpg", "jpeg", "png"],
            help="Upload the style reference image"
        )
        
        if style_file is not None:
            style_image = Image.open(style_file).convert("RGB")
            st.image(style_image, caption="Style Image", use_column_width=True)
        else:
            st.info("Please upload a style image")
            style_image = None
    
    with col3:
        st.header("Stylized Result")
        
        if st.button("Generate Stylized Image", disabled=(content_image is None or style_image is None)):
            if content_image is not None and style_image is not None:
                with st.spinner("Generating stylized image..."):
                    try:
                        # Set seed for reproducibility
                        set_seed(42)
                        
                        # Initialize sampler
                        sampler = StyleTransferSampler(config)
                        
                        # Save uploaded images temporarily
                        temp_dir = Path("temp")
                        temp_dir.mkdir(exist_ok=True)
                        
                        content_path = temp_dir / "content.jpg"
                        style_path = temp_dir / "style.jpg"
                        
                        content_image.save(content_path)
                        style_image.save(style_path)
                        
                        # Perform style transfer
                        stylized_tensor = sampler.transfer_style(
                            str(content_path),
                            str(style_path),
                            num_epochs=num_epochs,
                            learning_rate=learning_rate
                        )
                        
                        # Convert to image
                        stylized_image = tensor_to_image(stylized_tensor)
                        
                        # Display result
                        st.image(stylized_image, caption="Stylized Image", use_column_width=True)
                        
                        # Download button
                        st.download_button(
                            label="Download Stylized Image",
                            data=stylized_image.tobytes(),
                            file_name="stylized_image.jpg",
                            mime="image/jpeg"
                        )
                        
                        # Clean up temp files
                        content_path.unlink(missing_ok=True)
                        style_path.unlink(missing_ok=True)
                        
                    except Exception as e:
                        st.error(f"Error during style transfer: {str(e)}")
                        st.exception(e)
            else:
                st.warning("Please upload both content and style images")
    
    # Information section
    st.markdown("---")
    st.header("About Neural Style Transfer")
    
    st.markdown("""
    Neural Style Transfer is a technique that uses deep learning to blend the content of one image 
    with the artistic style of another. This implementation uses a pre-trained VGG19 network to 
    extract features and optimize a target image to match both content and style characteristics.
    
    **How it works:**
    1. **Content Loss**: Ensures the generated image preserves the content structure of the original image
    2. **Style Loss**: Matches the artistic style by comparing Gram matrices of feature activations
    3. **Optimization**: Uses gradient descent to minimize the combined loss function
    
    **Tips for better results:**
    - Use high-quality images with good contrast
    - Style images work best with distinctive artistic patterns
    - Adjust the style weight to balance content preservation vs style transfer
    - More epochs generally lead to better style transfer but take longer
    """)
    
    # Model information
    st.header("Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current Settings")
        st.write(f"- Model Type: {model_type}")
        st.write(f"- Epochs: {num_epochs}")
        st.write(f"- Learning Rate: {learning_rate}")
        st.write(f"- Style Weight: {style_weight:.0e}")
        st.write(f"- Content Weight: {content_weight}")
        st.write(f"- Device: {device}")
    
    with col2:
        st.subheader("Performance Tips")
        st.write("- Use GPU for faster processing")
        st.write("- Reduce epochs for quick previews")
        st.write("- Increase style weight for stronger style transfer")
        st.write("- Higher resolution images take longer to process")


if __name__ == "__main__":
    main()
