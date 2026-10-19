from django.contrib import admin

from .models import Prediction


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):

    list_display = (
        "id",
        "predicted_class",
        "confidence",
        "created_at",
    )

    list_filter = (
        "predicted_class",
        "created_at",
    )

    search_fields = (
        "predicted_class",
    )

    readonly_fields = (
        "created_at",
    )