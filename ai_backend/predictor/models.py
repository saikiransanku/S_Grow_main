from django.db import models


class ImagePredictionCache(models.Model):
    image_hash = models.CharField(max_length=64, unique=True, db_index=True)
    payload = models.JSONField()
    content_type = models.CharField(max_length=120, blank=True, default="")
    file_size = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self) -> str:
        return f"{self.image_hash[:12]}... ({self.content_type or 'unknown'})"


class UploadedLeafImage(models.Model):
    SOURCE_CAMERA = "camera"
    SOURCE_UPLOAD = "upload"
    SOURCE_CHOICES = (
        (SOURCE_CAMERA, "Camera"),
        (SOURCE_UPLOAD, "Upload"),
    )

    request_id = models.CharField(max_length=64, blank=True, default="", db_index=True)
    image_hash = models.CharField(max_length=64, db_index=True)
    file_name = models.CharField(max_length=255, blank=True, default="")
    source_type = models.CharField(
        max_length=24,
        choices=SOURCE_CHOICES,
        default=SOURCE_UPLOAD,
    )
    mime_type = models.CharField(max_length=120, blank=True, default="")
    image_type = models.CharField(max_length=40, blank=True, default="unknown")
    file_size = models.PositiveIntegerField(default=0)
    image_data = models.BinaryField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        source = self.source_type or self.SOURCE_UPLOAD
        image_type = self.image_type or "unknown"
        return f"{source}:{image_type}:{self.file_name or self.image_hash[:12]}"
