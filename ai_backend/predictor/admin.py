from django.contrib import admin

from .models import ImagePredictionCache, UploadedLeafImage


@admin.register(ImagePredictionCache)
class ImagePredictionCacheAdmin(admin.ModelAdmin):
    list_display = ("image_hash", "content_type", "file_size", "updated_at")
    search_fields = ("image_hash",)
    readonly_fields = ("created_at", "updated_at")


@admin.register(UploadedLeafImage)
class UploadedLeafImageAdmin(admin.ModelAdmin):
    list_display = (
        "file_name",
        "source_type",
        "image_type",
        "mime_type",
        "file_size",
        "created_at",
    )
    list_filter = ("source_type", "image_type", "mime_type")
    search_fields = ("file_name", "request_id", "image_hash")
    readonly_fields = ("created_at",)
