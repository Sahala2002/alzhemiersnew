from django.shortcuts import (
    render,
    redirect,
    get_object_or_404
)

from django.contrib.auth import login, logout
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User

from django.core.files.base import ContentFile

from .forms import (
    MRIUploadForm,
    RegistrationForm
)

from .models import (
    Prediction,
    UserActivity
)

import os
import io
import numpy as np

from PIL import (
    Image,
    UnidentifiedImageError
)

from tensorflow.keras.models import load_model

import tensorflow as tf
import matplotlib.cm as cm


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = (
    r"C:\Users\SAHALA\alzhemiers\disease"
    r"\model\alzheimer_fixed.keras"
)


# ============================================================
# CLASS NAMES
# ============================================================

# IMPORTANT:
# This order MUST match the order used by your trained model.

CLASS_NAMES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented",
]


# ============================================================
# STAGE INFORMATION
# ============================================================

STAGE_INFO = {

    "NonDemented": {
        "title": "Non-Demented",
        "description": (
            "The uploaded MRI was classified by the model "
            "under the Non-Demented category."
        ),
    },

    "VeryMildDemented": {
        "title": "Very Mild Demented",
        "description": (
            "The uploaded MRI was classified by the model "
            "under the Very Mild Demented category."
        ),
    },

    "MildDemented": {
        "title": "Mild Demented",
        "description": (
            "The uploaded MRI was classified by the model "
            "under the Mild Demented category."
        ),
    },

    "ModerateDemented": {
        "title": "Moderate Demented",
        "description": (
            "The uploaded MRI was classified by the model "
            "under the Moderate Demented category."
        ),
    },
}


# ============================================================
# MODEL CACHE
# ============================================================

_model = None


# ============================================================
# GET MODEL
# ============================================================

def get_model():

    global _model

    if _model is None:

        if not os.path.exists(MODEL_PATH):

            raise FileNotFoundError(
                "Alzheimer model file not found at:\n"
                f"{MODEL_PATH}"
            )

        try:

            _model = load_model(
                MODEL_PATH,
                compile=False
            )

            print()
            print("=" * 60)
            print("ALZHEIMER MODEL LOADED SUCCESSFULLY")
            print("=" * 60)

            print(
                "Model path:",
                MODEL_PATH
            )

            print(
                "Input shape:",
                _model.input_shape
            )

            print(
                "Output shape:",
                _model.output_shape
            )

            print("=" * 60)
            print()

        except Exception as e:

            print()
            print("=" * 60)
            print("MODEL LOADING ERROR")
            print("=" * 60)
            print(str(e))
            print("=" * 60)
            print()

            raise

    return _model


# ============================================================
# BASIC IMAGE VALIDATION
# ============================================================

def validate_uploaded_image(image_file):

    """
    Checks whether the uploaded file is a valid image.

    This does NOT determine whether the image is actually
    a brain MRI.
    """

    try:

        if not image_file:

            return (
                False,
                "Please upload an image."
            )

        image_file.seek(0)

        image = Image.open(
            image_file
        )

        # Check image integrity
        image.verify()

        # Re-open after verify
        image_file.seek(0)

        image = Image.open(
            image_file
        )

        width, height = image.size

        # Reject extremely small images
        if width < 100 or height < 100:

            image_file.seek(0)

            return (
                False,
                "Image is too small. "
                "Please upload a clear brain MRI image."
            )

        # Supported formats
        allowed_formats = [
            "JPEG",
            "PNG",
            "JPG",
            "BMP",
            "WEBP"
        ]

        if image.format not in allowed_formats:

            image_file.seek(0)

            return (
                False,
                "Unsupported image format. "
                "Please upload a JPG or PNG image."
            )

        image_file.seek(0)

        return (
            True,
            None
        )

    except UnidentifiedImageError:

        image_file.seek(0)

        return (
            False,
            "The uploaded file is not a valid image. "
            "Please upload a correct brain MRI image."
        )

    except OSError:

        image_file.seek(0)

        return (
            False,
            "The uploaded image is corrupted "
            "or cannot be read."
        )

    except Exception as e:

        image_file.seek(0)

        print(
            "IMAGE VALIDATION ERROR:",
            str(e)
        )

        return (
            False,
            "Unable to validate the uploaded image."
        )


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image_file):

    try:

        image_file.seek(0)

        image = Image.open(
            image_file
        )

        image = image.convert(
            "RGB"
        )

        image = image.resize(
            (224, 224),
            Image.Resampling.LANCZOS
        )

        image_array = np.asarray(
            image,
            dtype=np.float32
        )

        # IMPORTANT:
        # Do NOT divide by 255.
        # Keep preprocessing consistent with the model
        # that you already tested successfully.

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        return image_array

    except UnidentifiedImageError:

        raise ValueError(
            "The uploaded file is not a valid image."
        )

    except Exception as e:

        raise ValueError(
            "Unable to process the uploaded image: "
            f"{str(e)}"
        )


# ============================================================
# PREDICT MRI
# ============================================================

def predict_mri(image_file):

    model = get_model()

    image_array = preprocess_image(
        image_file
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )

    predictions = predictions[0]

    # ========================================================
    # CHECK MODEL OUTPUT
    # ========================================================

    if len(predictions) != len(
        CLASS_NAMES
    ):

        raise ValueError(
            "Model returned "
            f"{len(predictions)} classes, "
            "but CLASS_NAMES contains "
            f"{len(CLASS_NAMES)} classes."
        )

    predictions = np.asarray(
        predictions,
        dtype=np.float32
    )

    # ========================================================
    # CONVERT OUTPUT TO PROBABILITIES IF NECESSARY
    # ========================================================

    total = np.sum(
        predictions
    )

    if (
        np.any(
            ~np.isfinite(
                predictions
            )
        )
        or total <= 0
        or not np.isclose(
            total,
            1.0,
            atol=0.01
        )
    ):

        exp_predictions = np.exp(
            predictions -
            np.max(predictions)
        )

        predictions = (
            exp_predictions /
            np.sum(exp_predictions)
        )

    # ========================================================
    # PREDICTED CLASS
    # ========================================================

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = (
        CLASS_NAMES[
            predicted_index
        ]
    )

    confidence = round(
        float(
            predictions[
                predicted_index
            ] * 100
        ),
        2
    )

    # ========================================================
    # PROBABILITIES
    # ========================================================

    probabilities = {}

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        probabilities[class_name] = round(
            float(
                predictions[index] * 100
            ),
            2
        )

    # ========================================================
    # PRINT RESULT
    # ========================================================

    print()
    print("=" * 60)
    print("MRI PREDICTION")
    print("=" * 60)

    for class_name in CLASS_NAMES:

        print(
            f"{class_name}: "
            f"{probabilities[class_name]}%"
        )

    print("-" * 60)

    print(
        "Predicted:",
        predicted_class
    )

    print(
        "Confidence:",
        f"{confidence}%"
    )

    print("=" * 60)
    print()

    return {

        "predicted_class":
            predicted_class,

        "predicted_index":
            predicted_index,

        "confidence":
            confidence,

        "probabilities":
            probabilities,
    }


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(
    model,
    image_array,
    predicted_index
):

    """
    Generates a Grad-CAM heatmap for the nested
    ResNet50 model.

    Architecture:

        outer Sequential
             |
             +-- sequential
             |
             +-- resnet50
                    |
                    +-- conv5_block3_3_conv
             |
             +-- global_average_pooling2d
             +-- dense
             +-- dropout
             +-- dense_1
    """

    # --------------------------------------------------------
    # Get nested ResNet50 model
    # --------------------------------------------------------

    resnet_model = model.get_layer(
        "resnet50"
    )

    # --------------------------------------------------------
    # Target convolutional layer
    # --------------------------------------------------------

    target_layer = resnet_model.get_layer(
        "conv5_block3_3_conv"
    )

    # --------------------------------------------------------
    # Model that gives:
    #
    # 1. target convolution output
    # 2. ResNet output
    # --------------------------------------------------------

    grad_model = tf.keras.models.Model(
        inputs=resnet_model.input,
        outputs=[
            target_layer.output,
            resnet_model.output
        ]
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # The outer model has a layer named "sequential"
    # before ResNet50.
    #
    # We apply it before sending the image into ResNet.
    # This keeps Grad-CAM consistent with the actual model.
    # --------------------------------------------------------

    preprocessing_layer = model.get_layer(
        "sequential"
    )

    with tf.GradientTape() as tape:

        # Apply the same first layer used by the model
        preprocessed_image = preprocessing_layer(
            image_array,
            training=False
        )

        # Get convolution activations and ResNet output
        conv_outputs, resnet_output = grad_model(
            preprocessed_image,
            training=False
        )

        # ----------------------------------------------------
        # Recreate the outer classification head
        # ----------------------------------------------------

        x = resnet_output

        x = model.get_layer(
            "global_average_pooling2d"
        )(
            x
        )

        x = model.get_layer(
            "dense"
        )(
            x
        )

        x = model.get_layer(
            "dropout"
        )(
            x,
            training=False
        )

        predictions = model.get_layer(
            "dense_1"
        )(
            x
        )

        # Select the predicted class
        class_channel = predictions[
            :,
            predicted_index
        ]

    # --------------------------------------------------------
    # Calculate gradients
    # --------------------------------------------------------

    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    if grads is None:

        raise ValueError(
            "Grad-CAM could not calculate gradients."
        )

    # --------------------------------------------------------
    # Global average pooling of gradients
    # --------------------------------------------------------

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    # Remove batch dimension
    conv_outputs = conv_outputs[0]

    # --------------------------------------------------------
    # Weight feature maps
    # --------------------------------------------------------

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    # --------------------------------------------------------
    # ReLU
    # --------------------------------------------------------

    heatmap = tf.maximum(
        heatmap,
        0
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    max_value = tf.reduce_max(
        heatmap
    )

    if float(max_value) > 0:

        heatmap = (
            heatmap /
            max_value
        )

    heatmap = heatmap.numpy()

    return heatmap


# ============================================================
# CREATE GRAD-CAM IMAGE
# ============================================================

def create_gradcam_image(
    original_image_file,
    heatmap
):

    """
    Creates a colored Grad-CAM overlay over
    the original MRI image.
    """

    # --------------------------------------------------------
    # Open original image
    # --------------------------------------------------------

    original_image_file.seek(0)

    original_image = Image.open(
        original_image_file
    ).convert(
        "RGB"
    )

    # --------------------------------------------------------
    # Resize heatmap to original image size
    # --------------------------------------------------------

    heatmap_image = Image.fromarray(
        np.uint8(
            255 * heatmap
        )
    )

    heatmap_image = heatmap_image.resize(
        original_image.size,
        Image.Resampling.BILINEAR
    )

    # --------------------------------------------------------
    # Convert heatmap to NumPy
    # --------------------------------------------------------

    heatmap_array = np.asarray(
        heatmap_image,
        dtype=np.float32
    )

    heatmap_array = (
        heatmap_array / 255.0
    )

    # --------------------------------------------------------
    # Apply JET colormap
    # --------------------------------------------------------

    colored_heatmap = cm.jet(
        heatmap_array
    )

    colored_heatmap = np.uint8(
        colored_heatmap[:, :, :3] * 255
    )

    colored_heatmap = Image.fromarray(
        colored_heatmap
    ).convert(
        "RGB"
    )

    # --------------------------------------------------------
    # Blend original MRI + heatmap
    # --------------------------------------------------------

    overlay = Image.blend(
        original_image,
        colored_heatmap,
        alpha=0.40
    )

    # --------------------------------------------------------
    # Save image in memory
    # --------------------------------------------------------

    output = io.BytesIO()

    overlay.save(
        output,
        format="JPEG",
        quality=90
    )

    output.seek(0)

    return ContentFile(
        output.read(),
        name="gradcam.jpg"
    )


# ============================================================
# HOME
# ============================================================

@login_required(login_url="login")
def home(request):

    return render(
        request,
        "alzhmri/home.html"
    )


# ============================================================
# ABOUT
# ============================================================

def about(request):

    return render(
        request,
        "alzhmri/about.html"
    )


# ============================================================
# STAGES
# ============================================================

def stages(request):

    return render(
        request,
        "alzhmri/stages.html"
    )


# ============================================================
# UPLOAD MRI
# ============================================================

@login_required(login_url="login")
def upload_mri(request):

    if request.method == "POST":

        form = MRIUploadForm(
            request.POST,
            request.FILES
        )

        if form.is_valid():

            uploaded_image = (
                form.cleaned_data[
                    "uploaded_image"
                ]
            )

            # =================================================
            # BASIC IMAGE VALIDATION
            # =================================================

            is_valid, validation_error = (
                validate_uploaded_image(
                    uploaded_image
                )
            )

            if not is_valid:

                form.add_error(
                    "uploaded_image",
                    validation_error
                )

            else:

                try:

                    # =========================================
                    # PREDICTION
                    # =========================================

                    uploaded_image.seek(0)

                    result = predict_mri(
                        uploaded_image
                    )

                    # =========================================
                    # RESET FILE
                    # =========================================

                    uploaded_image.seek(0)

                    # =========================================
                    # SAVE PREDICTION
                    # =========================================

                    prediction = (
                        Prediction.objects.create(

                            user=request.user,

                            uploaded_image=
                                uploaded_image,

                            predicted_class=
                                result[
                                    "predicted_class"
                                ],

                            confidence=
                                result[
                                    "confidence"
                                ],

                            non_demented_probability=
                                result[
                                    "probabilities"
                                ][
                                    "NonDemented"
                                ],

                            very_mild_probability=
                                result[
                                    "probabilities"
                                ][
                                    "VeryMildDemented"
                                ],

                            mild_probability=
                                result[
                                    "probabilities"
                                ][
                                    "MildDemented"
                                ],

                            moderate_probability=
                                result[
                                    "probabilities"
                                ][
                                    "ModerateDemented"
                                ],
                        )
                    )

                    # =========================================
                    # MRI UPLOAD ACTIVITY
                    # =========================================

                    UserActivity.objects.create(

                        user=request.user,

                        activity_type=
                            "MRI_UPLOAD",

                        description=
                            "MRI image uploaded"
                    )

                    # =========================================
                    # PREDICTION ACTIVITY
                    # =========================================

                    UserActivity.objects.create(

                        user=request.user,

                        activity_type=
                            "PREDICTION",

                        description=(
                            "Prediction generated: "
                            f"{result['predicted_class']} "
                            f"({result['confidence']:.2f}%)"
                        )
                    )

                    # =========================================
                    # GRAD-CAM
                    # =========================================

                    try:

                        print()
                        print("=" * 60)
                        print("GENERATING GRAD-CAM")
                        print("=" * 60)

                        model = get_model()

                        uploaded_image.seek(0)

                        image_array = preprocess_image(
                            uploaded_image
                        )

                        heatmap = generate_gradcam(
                            model,
                            image_array,
                            result[
                                "predicted_index"
                            ]
                        )

                        uploaded_image.seek(0)

                        gradcam_file = (
                            create_gradcam_image(
                                uploaded_image,
                                heatmap
                            )
                        )

                        prediction.gradcam_image.save(
                            f"gradcam_{prediction.id}.jpg",
                            gradcam_file,
                            save=True
                        )

                        print(
                            "Grad-CAM generated successfully."
                        )

                        print(
                            "Grad-CAM file:",
                            prediction.gradcam_image.name
                        )

                        print("=" * 60)
                        print()

                    except Exception as gradcam_error:

                        # IMPORTANT:
                        # Grad-CAM failure must NOT destroy
                        # an otherwise successful prediction.

                        print()
                        print("=" * 60)
                        print("GRAD-CAM ERROR")
                        print("=" * 60)
                        print(
                            str(gradcam_error)
                        )
                        print("=" * 60)
                        print()

                    # =========================================
                    # RESULT PAGE
                    # =========================================

                    return redirect(
                        "result",
                        prediction_id=
                            prediction.id
                    )

                except Exception as e:

                    print()
                    print("=" * 60)
                    print("PREDICTION ERROR")
                    print("=" * 60)
                    print(
                        str(e)
                    )
                    print("=" * 60)
                    print()

                    form.add_error(
                        None,
                        "Prediction failed: "
                        f"{str(e)}"
                    )

    else:

        form = MRIUploadForm()

    return render(
        request,
        "alzhmri/upload.html",
        {
            "form": form
        }
    )


# ============================================================
# RESULT
# ============================================================

@login_required(login_url="login")
def result(
    request,
    prediction_id
):

    prediction = get_object_or_404(
        Prediction,
        id=prediction_id,
        user=request.user
    )

    # ========================================================
    # PROBABILITIES
    # ========================================================

    probabilities = {

        "NonDemented":
            prediction.non_demented_probability,

        "VeryMildDemented":
            prediction.very_mild_probability,

        "MildDemented":
            prediction.mild_probability,

        "ModerateDemented":
            prediction.moderate_probability,
    }

    # ========================================================
    # STAGE INFORMATION
    # ========================================================

    stage_info = STAGE_INFO.get(
        prediction.predicted_class,
        {
            "title":
                prediction.get_predicted_class_display(),

            "description":
                "The model generated this classification "
                "from the uploaded image."
        }
    )

    # ========================================================
    # GRAD-CAM URL
    # ========================================================

    gradcam_url = None

    if prediction.gradcam_image:

        gradcam_url = (
            prediction.gradcam_image.url
        )

    return render(
        request,
        "alzhmri/result.html",
        {

            "prediction":
                prediction,

            "prediction_record":
                prediction,

            "predicted_class":
                prediction.predicted_class,

            "predicted_class_display":
                prediction.get_predicted_class_display(),

            "confidence":
                prediction.confidence,

            "probabilities":
                probabilities,

            "stage_info":
                stage_info,

            "gradcam_url":
                gradcam_url,
        }
    )


# ============================================================
# HISTORY
# ============================================================

@login_required(login_url="login")
def history(request):

    predictions = (
        Prediction.objects
        .filter(
            user=request.user
        )
        .order_by(
            "-created_at"
        )
    )

    return render(
        request,
        "alzhmri/history.html",
        {
            "predictions":
                predictions
        }
    )


# ============================================================
# RECOMMENDATION
# ============================================================

def recommendation(request):

    return render(
        request,
        "alzhmri/recommendation.html"
    )


# ============================================================
# CONTACT
# ============================================================

def contact(request):

    return render(
        request,
        "alzhmri/contact.html"
    )


# ============================================================
# DISCLAIMER
# ============================================================

def disclaimer(request):

    return render(
        request,
        "alzhmri/disclaimer.html"
    )


# ============================================================
# REGISTER
# ============================================================

def register_view(request):

    if request.user.is_authenticated:

        return redirect(
            "home"
        )

    if request.method == "POST":

        form = RegistrationForm(
            request.POST
        )

        if form.is_valid():

            user = form.save()

            # Record registration
            UserActivity.objects.create(

                user=user,

                activity_type=
                    "REGISTER",

                description=
                    "New account registered"
            )

            # Login automatically
            login(
                request,
                user
            )

            return redirect(
                "home"
            )

    else:

        form = RegistrationForm()

    return render(
        request,
        "alzhmri/register.html",
        {
            "form":
                form
        }
    )


# ============================================================
# LOGIN
# ============================================================

def login_view(request):

    if request.user.is_authenticated:

        if request.user.is_staff:

            return redirect(
                "admin_dashboard"
            )

        return redirect(
            "home"
        )

    if request.method == "POST":

        form = AuthenticationForm(
            request,
            data=request.POST
        )

        if form.is_valid():

            user = form.get_user()

            # -----------------------------------------------
            # LOGIN
            # -----------------------------------------------

            login(
                request,
                user
            )

            # -----------------------------------------------
            # RECORD LOGIN ACTIVITY
            # -----------------------------------------------

            UserActivity.objects.create(

                user=user,

                activity_type=
                    "LOGIN",

                description=
                    "User logged in"
            )

            # -----------------------------------------------
            # ADMIN / USER REDIRECT
            # -----------------------------------------------

            if user.is_staff:

                return redirect(
                    "admin_dashboard"
                )

            else:

                return redirect(
                    "home"
                )

    else:

        form = AuthenticationForm()

    return render(
        request,
        "alzhmri/login.html",
        {
            "form": form
        }
    )


# ============================================================
# LOGOUT
# ============================================================

def logout_view(request):

    if request.user.is_authenticated:

        UserActivity.objects.create(

            user=request.user,

            activity_type=
                "LOGOUT",

            description=
                "User logged out"
        )

    logout(
        request
    )

    return redirect(
        "login"
    )


# ============================================================
# INDEX
# ============================================================

def index(request):

    if request.user.is_authenticated:

        return redirect(
            "home"
        )

    return redirect(
        "login"
    )


# ============================================================
# ADMIN DASHBOARD
# ============================================================

@login_required(login_url="login")
def admin_dashboard(request):

    # Only staff users can access dashboard
    if not request.user.is_staff:

        return redirect(
            "home"
        )

    # ========================================================
    # USER COUNTS
    # ========================================================

    total_users = (
        User.objects.count()
    )

    active_users = (
        User.objects
        .filter(
            is_active=True
        )
        .count()
    )

    inactive_users = (
        User.objects
        .filter(
            is_active=False
        )
        .count()
    )

    # ========================================================
    # PREDICTION COUNTS
    # ========================================================

    total_predictions = (
        Prediction.objects.count()
    )

    mild_count = (
        Prediction.objects
        .filter(
            predicted_class=
                "MildDemented"
        )
        .count()
    )

    moderate_count = (
        Prediction.objects
        .filter(
            predicted_class=
                "ModerateDemented"
        )
        .count()
    )

    non_demented_count = (
        Prediction.objects
        .filter(
            predicted_class=
                "NonDemented"
        )
        .count()
    )

    very_mild_count = (
        Prediction.objects
        .filter(
            predicted_class=
                "VeryMildDemented"
        )
        .count()
    )

    # ========================================================
    # ACTIVITIES
    # ========================================================

    total_activities = (
        UserActivity.objects.count()
    )

    # ========================================================
    # USERS
    # ========================================================

    users = (
        User.objects
        .all()
        .order_by(
            "-date_joined"
        )
    )

    # ========================================================
    # RECENT ACTIVITIES
    # ========================================================

    recent_activities = (
        UserActivity.objects
        .select_related("user")
        .all()
        .order_by(
            "-created_at"
        )[:20]
    )

    # ========================================================
    # RECENT PREDICTIONS
    # ========================================================

    recent_predictions = (
        Prediction.objects
        .select_related("user")
        .all()
        .order_by(
            "-created_at"
        )[:10]
    )

    return render(
        request,
        "alzhmri/admin_dashboard.html",
        {

            "total_users":
                total_users,

            "active_users":
                active_users,

            "inactive_users":
                inactive_users,

            "total_predictions":
                total_predictions,

            "total_activities":
                total_activities,

            "mild_count":
                mild_count,

            "moderate_count":
                moderate_count,

            "non_demented_count":
                non_demented_count,

            "very_mild_count":
                very_mild_count,

            "users":
                users,

            "recent_activities":
                recent_activities,

            "recent_predictions":
                recent_predictions,
        }
    )


# ============================================================
# ADMIN USER ACTIVITY
# ============================================================

@login_required(login_url="login")
def admin_user_activity(
    request,
    user_id
):

    # Only staff users
    if not request.user.is_staff:

        return redirect(
            "home"
        )

    selected_user = get_object_or_404(
        User,
        id=user_id
    )

    # User activities
    activities = (
        UserActivity.objects
        .filter(
            user=selected_user
        )
        .order_by(
            "-created_at"
        )
    )

    # User predictions
    predictions = (
        Prediction.objects
        .filter(
            user=selected_user
        )
        .order_by(
            "-created_at"
        )
    )

    return render(
        request,
        "alzhmri/admin_user_activity.html",
        {

            "selected_user":
                selected_user,

            "activities":
                activities,

            "predictions":
                predictions,
        }
    )
def recommendations(request):
    return render(request, "alzhmri/recommendations.html")