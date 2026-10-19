from django.db import models
from django.contrib.auth.models import User


# ============================================================
# PREDICTION MODEL
# ============================================================

class Prediction(models.Model):

    CLASS_CHOICES = [
        ("NonDemented", "Non-Demented"),
        ("VeryMildDemented", "Very Mild Demented"),
        ("MildDemented", "Mild Demented"),
        ("ModerateDemented", "Moderate Demented"),
    ]

    # --------------------------------------------------------
    # USER WHO UPLOADED THE MRI
    # --------------------------------------------------------

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="predictions"
    )

    # --------------------------------------------------------
    # ORIGINAL UPLOADED MRI
    # --------------------------------------------------------

    uploaded_image = models.ImageField(
        upload_to="mri_scans/"
    )

    # --------------------------------------------------------
    # GRAD-CAM IMAGE
    # --------------------------------------------------------

    gradcam_image = models.ImageField(
        upload_to="gradcam/",
        null=True,
        blank=True
    )

    # --------------------------------------------------------
    # PREDICTED CLASS
    # --------------------------------------------------------

    predicted_class = models.CharField(
        max_length=50,
        choices=CLASS_CHOICES
    )

    # --------------------------------------------------------
    # MODEL CONFIDENCE
    # --------------------------------------------------------

    confidence = models.FloatField(
        default=0.0
    )

    # --------------------------------------------------------
    # CLASS PROBABILITIES
    # --------------------------------------------------------

    non_demented_probability = models.FloatField(
        default=0.0
    )

    very_mild_probability = models.FloatField(
        default=0.0
    )

    mild_probability = models.FloatField(
        default=0.0
    )

    moderate_probability = models.FloatField(
        default=0.0
    )

    # --------------------------------------------------------
    # CREATED DATE
    # --------------------------------------------------------

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    # --------------------------------------------------------
    # DEFAULT ORDER
    # --------------------------------------------------------

    class Meta:
        ordering = ["-created_at"]

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    def __str__(self):

        username = (
            self.user.username
            if self.user
            else "Unknown User"
        )

        return (
            f"{username} - "
            f"{self.predicted_class} - "
            f"{self.confidence:.2f}%"
        )


# ============================================================
# USER ACTIVITY MODEL
# ============================================================

class UserActivity(models.Model):

    ACTIVITY_CHOICES = [

        # ----------------------------------------------------
        # ACCOUNT
        # ----------------------------------------------------

        (
            "REGISTER",
            "Registered Account"
        ),

        # ----------------------------------------------------
        # AUTHENTICATION
        # ----------------------------------------------------

        (
            "LOGIN",
            "Logged In"
        ),

        (
            "LOGOUT",
            "Logged Out"
        ),

        # ----------------------------------------------------
        # MRI
        # ----------------------------------------------------

        (
            "MRI_UPLOAD",
            "MRI Uploaded"
        ),

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        (
            "PREDICTION",
            "Prediction Generated"
        ),
    ]

    # --------------------------------------------------------
    # USER
    # --------------------------------------------------------

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="activities"
    )

    # --------------------------------------------------------
    # ACTIVITY TYPE
    # --------------------------------------------------------

    activity_type = models.CharField(
        max_length=30,
        choices=ACTIVITY_CHOICES
    )

    # --------------------------------------------------------
    # ACTIVITY DESCRIPTION
    # --------------------------------------------------------

    description = models.CharField(
        max_length=255,
        blank=True
    )

    # --------------------------------------------------------
    # ACTIVITY DATE
    # --------------------------------------------------------

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    # --------------------------------------------------------
    # DEFAULT ORDER
    # --------------------------------------------------------

    class Meta:
        ordering = ["-created_at"]

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    def __str__(self):

        return (
            f"{self.user.username} - "
            f"{self.get_activity_type_display()}"
        )