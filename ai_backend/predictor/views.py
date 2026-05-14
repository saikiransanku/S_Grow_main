import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import logging
import uuid
from datetime import datetime

from django.http import HttpRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .models import ImagePredictionCache, UploadedLeafImage

logger = logging.getLogger(__name__)


def _parse_major_minor(raw_version: str) -> tuple[int, int]:
    parts = []
    for raw_part in raw_version.split("."):
        digits = "".join(ch for ch in raw_part if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
        if len(parts) == 2:
            break
    while len(parts) < 2:
        parts.append(0)
    return parts[0], parts[1]


def _detect_runtime_incompatibility() -> str | None:
    try:
        numpy_version = version("numpy")
        tensorflow_version = version("tensorflow")
    except PackageNotFoundError:
        return None

    numpy_major, _ = _parse_major_minor(numpy_version)
    tf_major, tf_minor = _parse_major_minor(tensorflow_version)

    if tf_major == 2 and numpy_major >= 2 and tf_minor < 18:
        return (
            "Prediction engine is unavailable in this Python runtime "
            f"(tensorflow {tensorflow_version} with numpy {numpy_version}). "
            "Use the project's Python environment with a NumPy-compatible "
            "TensorFlow installation."
        )

    return None


def _load_prediction_services():
    runtime_error = _detect_runtime_incompatibility()
    if runtime_error:
        raise RuntimeError(runtime_error)

    try:
        from .services import (
            ImageQualityError,
            SSGROW_MODEL_VERSION,
            SSGROW_TRAINED_SEASONS,
            call_llm,
            get_prediction_cache_namespace,
            normalize_requested_season,
            run_prediction,
        )
    except Exception as exc:  # pragma: no cover - depends on local ML runtime
        logger.exception("Prediction services import failed.")
        raise RuntimeError(
            "Prediction engine failed to initialize. "
            "Start the AI backend with the project's Python environment and ensure "
            "TensorFlow is installed with a NumPy-compatible version. "
            f"Root cause: {exc.__class__.__name__}: {exc}"
        ) from exc

    return {
        "model_version": SSGROW_MODEL_VERSION,
        "trained_seasons": SSGROW_TRAINED_SEASONS,
        "image_quality_error": ImageQualityError,
        "call_llm": call_llm,
        "cache_namespace": get_prediction_cache_namespace(),
        "normalize_requested_season": normalize_requested_season,
        "run_prediction": run_prediction,
    }


def _serialize_prediction_result(result, *, model_version: str, trained_seasons) -> dict:
    return {
        "disease_type": result.label,
        "crop_detected": result.crop_detected,
        "confidence_score": result.confidence,
        "prediction_score": round(result.confidence / 100.0, 4),
        "model_version": model_version,
        "trained_seasons": list(trained_seasons),
        "season_used": result.season_used,
        "verification_passed": result.verification_passed,
        "verification_reason": result.verification_reason,
        "seasonal_comparison": result.seasonal_comparison,
        "status_message": result.status_message,
        "diagnosis_status": result.diagnosis_status,
        "override_applied": result.override_applied,
        "override_reason": result.override_reason,
        "model_label_before_override": result.model_label_before_override,
        "model_confidence_before_override": result.model_confidence_before_override,
        "heuristic_lesion_count": result.heuristic_lesion_count,
        "preprocessing_metrics": getattr(result, "preprocessing_metrics", {}),
        "recommended_pesticide": result.recommended_pesticide,
        "active_ingredient": result.active_ingredient,
        "usage_note": result.usage_note,
        "leaf_visual_analysis": result.leaf_visual_analysis,
        "farmer_report": result.farmer_report,
        "farmer_action_plan_markdown": result.farmer_action_plan_markdown,
    }


def _build_prediction_cache_key(
    *,
    image_bytes: bytes,
    requested_season: str | None,
    cache_namespace: str,
    normalize_requested_season,
) -> str:
    normalized_season = normalize_requested_season(requested_season)
    cache_input = b"|".join(
        (
            image_bytes,
            normalized_season.encode("utf-8"),
            cache_namespace.encode("utf-8"),
        )
    )
    return hashlib.sha256(cache_input).hexdigest()


def _normalize_upload_source(raw_source: str | None) -> str:
    value = (raw_source or "").strip().lower()
    if value == UploadedLeafImage.SOURCE_CAMERA:
        return UploadedLeafImage.SOURCE_CAMERA
    return UploadedLeafImage.SOURCE_UPLOAD


def _infer_image_type(*, content_type: str, file_name: str) -> str:
    normalized_content_type = (content_type or "").strip().lower()
    if "/" in normalized_content_type:
        subtype = normalized_content_type.split("/", 1)[1].strip()
        if subtype:
            return subtype

    lowered_name = (file_name or "").strip().lower()
    if "." in lowered_name:
        extension = lowered_name.rsplit(".", 1)[1].strip()
        if extension:
            return extension

    return "unknown"


def _persist_uploaded_image(
    *,
    request_id: str,
    upload_name: str,
    image_bytes: bytes,
    content_type: str,
    source_type: str,
    image_hash: str,
) -> UploadedLeafImage:
    return UploadedLeafImage.objects.create(
        request_id=request_id,
        image_hash=image_hash,
        file_name=upload_name,
        source_type=_normalize_upload_source(source_type),
        mime_type=content_type,
        image_type=_infer_image_type(
            content_type=content_type,
            file_name=upload_name,
        ),
        file_size=len(image_bytes),
        image_data=image_bytes,
    )


@csrf_exempt
def health(_: HttpRequest):
    return JsonResponse({"status": "ok"})


@csrf_exempt
def predict(request: HttpRequest):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    request_id = (request.POST.get("request_id") or "").strip() or str(uuid.uuid4())
    request_name = (request.POST.get("request_name") or "").strip()
    if not request_name:
        request_name = f"Leaf Analysis {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    uploads = request.FILES.getlist("files")
    if not uploads:
        single_upload = request.FILES.get("file")
        if single_upload:
            uploads = [single_upload]

    if not uploads:
        return JsonResponse(
            {"error": "Missing image file", "request_id": request_id},
            status=400,
        )

    season = request.POST.get("season")
    user_message = (request.POST.get("message") or "").strip()
    upload_sources = request.POST.getlist("file_sources")

    try:
        services = _load_prediction_services()
    except RuntimeError as exc:
        logger.exception("Prediction services could not be initialized.")
        return JsonResponse(
            {"error": str(exc), "request_id": request_id},
            status=503,
        )

    results_payload = []
    for index, upload in enumerate(uploads):
        try:
            image_bytes = upload.read()
            source_image_hash = hashlib.sha256(image_bytes).hexdigest()
            source_type = (
                upload_sources[index]
                if index < len(upload_sources)
                else UploadedLeafImage.SOURCE_UPLOAD
            )
            stored_upload = _persist_uploaded_image(
                request_id=request_id,
                upload_name=upload.name,
                image_bytes=image_bytes,
                content_type=upload.content_type or "image/jpeg",
                source_type=source_type,
                image_hash=source_image_hash,
            )
            normalized_requested_season = services["normalize_requested_season"](season)
            cache_key = _build_prediction_cache_key(
                image_bytes=image_bytes,
                requested_season=season,
                cache_namespace=str(services["cache_namespace"]),
                normalize_requested_season=services["normalize_requested_season"],
            )
            cached_row = ImagePredictionCache.objects.filter(image_hash=cache_key).values("payload").first()

            if cached_row and isinstance(cached_row.get("payload"), dict):
                base_payload = dict(cached_row["payload"])
                base_payload["cache_hit"] = True
            else:
                result = services["run_prediction"](
                    image_bytes=image_bytes,
                    content_type=upload.content_type or "image/jpeg",
                    requested_season=season,
                )
                base_payload = _serialize_prediction_result(
                    result,
                    model_version=str(services["model_version"]),
                    trained_seasons=services["trained_seasons"],
                )
                base_payload["requested_season"] = normalized_requested_season
                base_payload["cache_hit"] = False

                ImagePredictionCache.objects.update_or_create(
                    image_hash=cache_key,
                    defaults={
                        "payload": base_payload,
                        "content_type": upload.content_type or "image/jpeg",
                        "file_size": len(image_bytes),
                    },
                )
            if "requested_season" not in base_payload:
                base_payload["requested_season"] = normalized_requested_season
        except services["image_quality_error"] as exc:
            logger.warning("Image quality rejected for upload '%s': %s", upload.name, exc)
            return JsonResponse(
                {
                    "error": str(exc),
                    "request_id": request_id,
                },
                status=400,
            )
        except Exception as exc:
            logger.exception("Prediction failed for upload '%s'.", upload.name)
            return JsonResponse(
                {
                    "error": f"Prediction failed for '{upload.name}': {exc}",
                    "request_id": request_id,
                },
                status=500,
            )

        results_payload.append(
            {
                "image_index": index + 1,
                "file_name": upload.name,
                "image_hash": source_image_hash,
                "image_record_id": stored_upload.id,
                "source_type": stored_upload.source_type,
                "image_type": stored_upload.image_type,
                **base_payload,
            }
        )

    response_payload = {
        "status": "success",
        "request_id": request_id,
        "request_name": request_name,
        "message": user_message,
        "image_count": len(results_payload),
        "results": results_payload,
    }

    if len(results_payload) == 1:
        response_payload.update(results_payload[0])

    return JsonResponse(response_payload)


@csrf_exempt
def chat(request: HttpRequest):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        body = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    message = (body.get("message") or "").strip()
    context = body.get("context") or {}
    profile_name = (body.get("profile_name") or "").strip()
    profile_context = body.get("profile_context") or {}
    advisor_context = body.get("advisor_context") or {}
    conversation_history = body.get("conversation_history") or []
    if not message:
        return JsonResponse({"error": "Missing message"}, status=400)

    try:
        services = _load_prediction_services()
        llm = services["call_llm"](
            prompt=message,
            context=context,
            profile_name=profile_name,
            profile_context=profile_context,
            advisor_context=advisor_context,
            conversation_history=conversation_history,
        )
    except RuntimeError as exc:
        logger.exception("Chat services could not be initialized.")
        return JsonResponse({"error": str(exc)}, status=503)
    except Exception as exc:
        logger.exception("Chat request failed.")
        return JsonResponse({"error": f"Chat request failed: {exc}"}, status=500)

    reply = llm.get("answer") if isinstance(llm, dict) else ""
    if not isinstance(reply, str) or not reply.strip():
        reply = "Unable to generate response right now."
    response_payload = {"status": "success", "reply": reply}
    if isinstance(llm, dict):
        if "advisor_context" in llm:
            response_payload["advisor_context"] = llm.get("advisor_context")
        if "route" in llm:
            response_payload["route"] = llm.get("route")
    return JsonResponse(response_payload)

# Create your views here.
