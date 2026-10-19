from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


class MRIUploadForm(forms.Form):

    uploaded_image = forms.ImageField(
        required=True,
        label="Upload MRI Image",
        widget=forms.ClearableFileInput(
            attrs={
                "accept": ".jpg,.jpeg,.png",
                "id": "id_uploaded_image",
            }
        )
    )

    def clean_uploaded_image(self):

        image = self.cleaned_data.get(
            "uploaded_image"
        )

        if not image:
            raise forms.ValidationError(
                "Please select an MRI image."
            )

        # ==========================================
        # MAXIMUM FILE SIZE: 10 MB
        # ==========================================

        max_size = 10 * 1024 * 1024

        if image.size > max_size:
            raise forms.ValidationError(
                "Image size must be less than 10 MB."
            )

        # ==========================================
        # ALLOWED FILE EXTENSIONS
        # ==========================================

        allowed_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
        ]

        filename = image.name.lower()

        if not any(
            filename.endswith(ext)
            for ext in allowed_extensions
        ):
            raise forms.ValidationError(
                "Only JPG, JPEG and PNG images are allowed."
            )

        # ==========================================
        # ALLOWED MIME TYPES
        # ==========================================

        allowed_types = [
            "image/jpeg",
            "image/png",
        ]

        if image.content_type not in allowed_types:
            raise forms.ValidationError(
                "Invalid image type. "
                "Please upload JPG, JPEG or PNG."
            )

        return image


# ==================================================
# REGISTRATION FORM
# ==================================================

class RegistrationForm(UserCreationForm):

    email = forms.EmailField(
        required=True
    )

    class Meta:
        model = User
        fields = [
            "username",
            "email",
            "password1",
            "password2",
        ]