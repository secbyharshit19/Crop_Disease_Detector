import streamlit as st
import torch

from PIL import Image
from torchvision import models, transforms


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/tomato_disease_efficientnet_b0_final.pth"

IMAGE_SIZE = 224

# Initial threshold for rejecting low-confidence images.
# We can tune this later using validation + external images.
CONFIDENCE_THRESHOLD = 0.60


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Tomato Disease Detection",
    page_icon="🍅",
    layout="wide"
)


# ============================================================
# DISPLAY NAMES
# ============================================================

display_names = {
    "Bacterial_spot": "Bacterial Spot",

    "Early_blight": "Early Blight",

    "Late_blight": "Late Blight",

    "Leaf_Mold": "Leaf Mold",

    "Septoria_leaf_spot": "Septoria Leaf Spot",

    "Spider_mites Two-spotted_spider_mite":
        "Two-Spotted Spider Mites",

    "Target_Spot": "Target Spot",

    "Tomato_Yellow_Leaf_Curl_Virus":
        "Tomato Yellow Leaf Curl Virus",

    "Tomato_mosaic_virus":
        "Tomato Mosaic Virus",

    "healthy": "Healthy",

    "powdery_mildew": "Powdery Mildew"
}


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    classes = checkpoint["classes"]

    # Create EfficientNet-B0
    model = models.efficientnet_b0(
        weights=None
    )

    # Replace classifier
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features,
        len(classes)
    )

    # Load trained weights
    model.load_state_dict(
        checkpoint["state_dict"]
    )

    model = model.to(device)

    model.eval()

    return model, classes, checkpoint


# Load model
model, classes, checkpoint = load_model()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],

        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# HEADER
# ============================================================

st.title(
    "🍅 Tomato Disease Detection"
)

st.write(
    "Upload a tomato leaf image and the trained "
    "EfficientNet-B0 model will predict the disease."
)


# ============================================================
# MODEL STATUS
# ============================================================

st.info(
    f"Model: EfficientNet-B0 | "
    f"Classes: {checkpoint['num_classes']} | "
    f"Test Accuracy: {checkpoint['test_accuracy']:.2%} | "
    f"Device: {device}"
)


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a tomato leaf image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    try:

        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        # ----------------------------------------------------
        # DISPLAY IMAGE
        # ----------------------------------------------------

        col1, col2 = st.columns(
            2
        )


        with col1:

            st.subheader(
                "Uploaded Image"
            )

            st.image(
                image,
                use_container_width=True
            )


        # ----------------------------------------------------
        # PREPROCESS IMAGE
        # ----------------------------------------------------

        input_tensor = transform(
            image
        ).unsqueeze(0).to(device)


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = model(
                input_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            top_probs, top_indices = torch.topk(
                probabilities,
                k=3,
                dim=1
            )


        # ----------------------------------------------------
        # BEST PREDICTION
        # ----------------------------------------------------

        predicted_idx = (
            top_indices[0][0].item()
        )

        predicted_class = (
            classes[predicted_idx]
        )

        confidence = (
            top_probs[0][0].item()
        )


        # ----------------------------------------------------
        # UNKNOWN IMAGE CHECK
        # ----------------------------------------------------

        is_unknown = (
            confidence < CONFIDENCE_THRESHOLD
        )


        # ====================================================
        # RESULT
        # ====================================================

        with col2:

            st.subheader(
                "Prediction"
            )


            if is_unknown:

                # --------------------------------------------
                # UNKNOWN / UNSUPPORTED IMAGE
                # --------------------------------------------

                st.warning(
                    "⚠️ Unknown / Unrecognized Image"
                )

                st.error(
                    "This image could not be confidently "
                    "identified as one of the 11 trained "
                    "tomato-leaf classes."
                )

                st.metric(
                    "Highest Model Confidence",
                    f"{confidence:.2%}"
                )

                st.write(
                    "The image may be:"
                )

                st.write(
                    "• Not a leaf"
                )

                st.write(
                    "• Not a tomato leaf"
                )

                st.write(
                    "• A tomato disease not included "
                    "in the training dataset"
                )

                st.write(
                    "• Too different from the training images"
                )

                st.write(
                    "• Poor quality or unclear"
                )


            else:

                # --------------------------------------------
                # VALID CLASSIFICATION
                # --------------------------------------------

                st.success(
                    f"🌿 "
                    f"{display_names[predicted_class]}"
                )

                st.metric(
                    "Model Confidence",
                    f"{confidence:.2%}"
                )


        # ====================================================
        # TOP 3 PREDICTIONS
        # ====================================================

        st.subheader(
            "Model's Top 3 Candidates"
        )


        for rank in range(3):

            idx = (
                top_indices[0][rank].item()
            )

            probability = (
                top_probs[0][rank].item()
            )

            class_name = (
                classes[idx]
            )

            display_name = (
                display_names[class_name]
            )

            st.write(
                f"**{rank + 1}. "
                f"{display_name}** — "
                f"{probability:.2%}"
            )


        # ====================================================
        # UNKNOWN WARNING
        # ====================================================

        if is_unknown:

            st.info(
                "Because the highest model confidence "
                f"({confidence:.2%}) is below the "
                f"{CONFIDENCE_THRESHOLD:.0%} threshold, "
                "the result should NOT be treated as a "
                "reliable tomato disease classification."
            )


        # ====================================================
        # CONFIDENCE BAR
        # ====================================================

        st.subheader(
            "Prediction Confidence"
        )

        st.progress(
            min(
                confidence,
                1.0
            )
        )

        st.write(
            f"Highest confidence: "
            f"**{confidence:.2%}**"
        )

        st.write(
            f"Unknown threshold: "
            f"**{CONFIDENCE_THRESHOLD:.0%}**"
        )


    except Exception as e:

        st.error(
            "Unable to process the uploaded image."
        )

        st.exception(e)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander(
    "📊 Model Information"
):

    st.write(
        "**Architecture:** EfficientNet-B0"
    )

    st.write(
        "**Approach:** Transfer Learning / Fine-tuning"
    )

    st.write(
        f"**Number of classes:** "
        f"{checkpoint['num_classes']}"
    )

    st.write(
        f"**Best epoch:** "
        f"{checkpoint['best_epoch']}"
    )

    st.write(
        f"**Validation accuracy:** "
        f"{checkpoint['validation_accuracy']:.2%}"
    )

    st.write(
        f"**Test accuracy:** "
        f"{checkpoint['test_accuracy']:.2%}"
    )

    st.write(
        "**Input size:** 224 × 224"
    )

    st.write(
        f"**Unknown-image threshold:** "
        f"{CONFIDENCE_THRESHOLD:.0%}"
    )

    st.write(
        f"**Device:** {device}"
    )


# ============================================================
# TRAINED CLASSES
# ============================================================

with st.expander(
    "🌱 Trained Disease Classes"
):

    for i, class_name in enumerate(classes):

        st.write(
            f"{i + 1}. "
            f"{display_names[class_name]}"
        )


# ============================================================
# IMPORTANT NOTE
# ============================================================

st.markdown("---")

st.caption(
    "The model is trained on 11 tomato-leaf classes. "
    "Low-confidence images are flagged as unknown or "
    "outside the trained distribution. This rejection "
    "mechanism does not constitute a definitive "
    "non-tomato-leaf detector."
)
