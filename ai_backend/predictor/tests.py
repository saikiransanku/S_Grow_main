import io
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from unittest.mock import patch

import numpy as np
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, TestCase
from PIL import Image

from . import services, views
from .models import UploadedLeafImage


class CallLLMTests(SimpleTestCase):
    def setUp(self) -> None:
        self.context = {
            "status": "success",
            "results": [
                {
                    "crop_detected": "mango",
                    "disease_type": "sooty_mould_mango",
                    "confidence_score": 89.5,
                    "diagnosis_status": "ok",
                    "leaf_visual_analysis": {
                        "anomalies_textures": {
                            "lesion_summary": "dark fungal patches visible",
                            "chlorosis_halo": "chlorotic halo is not strongly visible",
                            "lesions_detected": 2,
                        }
                    },
                    "farmer_report": {
                        "leaf_name": "Mango",
                        "type_of_disease": "Sooty Mould",
                        "organic_recommendations": [
                            {
                                "name": "Neem oil",
                                "use_case": "early symptom control",
                            }
                        ],
                        "chemical_recommendation": {
                            "name": "Copper fungicide",
                            "usage_note": "follow local label guidance",
                        },
                    },
                }
            ],
        }

    @patch("predictor.services._get_openai_api_key", return_value="sk-test")
    @patch("predictor.services._openai_chat_completion")
    def test_call_llm_sends_structured_ssgrow_prediction_context(
        self,
        mock_openai_chat_completion,
        _mock_api_key,
    ) -> None:
        mock_openai_chat_completion.return_value = (
            "Crop Identified:\nMango\n\n"
            "Disease Prediction:\nSooty Mould\n\n"
            "Prediction Confidence:\n89.5%\n\n"
            "Symptoms Detected:\n* Dark fungal patches are visible.\n\n"
            "Seasonal Model Note:\nThe model was trained across multiple seasons.\n\n"
            "Recommended Actions:\n* Spray neem oil.\n\n"
            "Uncertainty Note:\nThis is an AI prediction and may not always be fully accurate."
        )

        services.call_llm(
            prompt="What treatment do you suggest?",
            context=self.context,
            profile_name="Kiran",
        )

        messages = mock_openai_chat_completion.call_args.kwargs["messages"]
        self.assertEqual(messages[0]["content"], services.SSGROW_SYSTEM_PROMPT)
        self.assertIn('"crop": "Mango"', messages[1]["content"])
        self.assertIn('"disease": "Sooty Mould"', messages[1]["content"])
        self.assertIn('"confidence": "89.5%"', messages[1]["content"])
        self.assertIn(
            f'"model_version": "{services.SSGROW_MODEL_VERSION}"',
            messages[1]["content"],
        )
        self.assertIn('"trained_seasons": ["Kharif", "Rabi", "All Season"]', messages[1]["content"])

    @patch("predictor.services._get_openai_api_key", return_value="sk-test")
    @patch("predictor.services._openai_chat_completion")
    def test_call_llm_uses_structured_cnn_fallback_when_quota_is_exceeded(
        self,
        mock_openai_chat_completion,
        _mock_api_key,
    ) -> None:
        mock_openai_chat_completion.side_effect = HTTPError(
            url="https://api.openai.com/v1/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=io.BytesIO(b'{"error":{"message":"quota exceeded"}}'),
        )

        response = services.call_llm(
            prompt="Explain the result",
            context=self.context,
            profile_name="Kiran",
        )

        answer = response["answer"]
        self.assertIn("Crop Identified:\nMango", answer)
        self.assertIn("Disease Prediction:\nSooty Mould", answer)
        self.assertIn("Prediction Confidence:\n89.5%", answer)
        self.assertIn("Recommended Actions:", answer)
        self.assertIn("cloud AI quota is currently reached", answer)

    @patch("predictor.services._get_openai_api_key", return_value="")
    def test_call_llm_without_prediction_context_requests_image_in_required_format(
        self,
        _mock_api_key,
    ) -> None:
        response = services.call_llm(
            prompt="Can you help me?",
            context=None,
            profile_name="Kiran",
        )

        answer = response["answer"]
        self.assertIn("Crop Identified:\nNot available", answer)
        self.assertIn("Disease Prediction:\nAwaiting clear crop leaf image", answer)
        self.assertIn("Prediction Confidence:\nNot available", answer)
        self.assertIn("* Upload one clear leaf image in good light.", answer)

    def test_call_llm_returns_welcome_message_for_greeting_even_with_context(self) -> None:
        response = services.call_llm(
            prompt="hello",
            context=self.context,
            profile_name="Kiran",
        )

        answer = response["answer"]
        self.assertIn("Hello Kiran!", answer)
        self.assertIn("Welcome back to SSGrow.", answer)
        self.assertIn("I'm your AI crop assistant.", answer)
        self.assertIn("upload a leaf image anytime", answer.lower())
        self.assertNotIn("Crop Identified:", answer)


class SeasonNormalizationTests(SimpleTestCase):
    def test_normalize_requested_season_supports_new_and_legacy_aliases(self) -> None:
        self.assertEqual(services.normalize_requested_season("all_season"), "all_season")
        self.assertEqual(services.normalize_requested_season("all"), "all_season")
        self.assertEqual(services.normalize_requested_season("Kharif"), "kharif")
        self.assertEqual(services.normalize_requested_season("RABI"), "rabi")
        self.assertEqual(services.normalize_requested_season("unexpected"), "auto")


class CropRoutingTests(SimpleTestCase):
    def test_extract_crop_from_label_handles_multi_token_and_prefix_labels(self) -> None:
        self.assertEqual(
            services.extract_crop_from_label("healthy_cherry_including_sour"),
            "cherry_including_sour",
        )
        self.assertEqual(
            services.extract_crop_from_label("bacterial_spot_pepper,_bell"),
            "pepper_bell",
        )
        self.assertEqual(
            services.extract_crop_from_label("cercospora_leaf_spot_gray_leaf_spot_corn_maize"),
            "corn_maize",
        )
        self.assertEqual(
            services.extract_crop_from_label("groundnut_healthy_leaf"),
            "groundnut",
        )

    def test_infer_season_from_crop_uses_supported_season_mapping(self) -> None:
        seasons_by_crop = {
            "potato": {"rabi"},
            "cotton": {"kharif"},
            "sugarcane": {"all_season"},
        }
        self.assertEqual(services.infer_season_from_crop("potato", seasons_by_crop), "rabi")
        self.assertEqual(services.infer_season_from_crop("sugarcane", seasons_by_crop), "all_season")
        self.assertEqual(services.infer_season_from_crop("unknown", seasons_by_crop), "unknown")

    def test_choose_auto_prediction_season_prefers_supported_rabi_crop(self) -> None:
        raw_predictions = {
            "all_season": {"crop_detected": "tomato", "confidence": 42.0},
            "kharif": {"crop_detected": "cotton", "confidence": 58.0},
            "rabi": {"crop_detected": "potato", "confidence": 91.0},
        }
        seasons_by_crop = {
            "tomato": {"all_season"},
            "cotton": {"kharif"},
            "potato": {"rabi"},
        }

        season, crop, _reason = services._choose_auto_prediction_season(
            raw_predictions,
            seasons_by_crop,
        )

        self.assertEqual(season, "rabi")
        self.assertEqual(crop, "potato")

    def test_requested_season_is_rejected_when_routing_points_to_another_dataset(self) -> None:
        raw_predictions = {
            "all_season": {"crop_detected": "mango", "confidence": 97.0},
            "kharif": {"crop_detected": "cotton", "confidence": 61.0},
            "rabi": {"crop_detected": "potato", "confidence": 63.0},
        }
        seasons_by_crop = {
            "mango": {"all_season"},
            "cotton": {"kharif"},
            "potato": {"rabi"},
        }

        season, crop, reason = services._resolve_requested_prediction_season(
            requested_season="rabi",
            raw_predictions=raw_predictions,
            supported_seasons_by_crop=seasons_by_crop,
        )

        self.assertIsNone(season)
        self.assertEqual(crop, "mango")
        self.assertIn("All Season", reason)


class PredictionCacheKeyTests(SimpleTestCase):
    def test_prediction_cache_key_changes_when_requested_season_changes(self) -> None:
        image_bytes = b"sample-leaf-image"
        auto_key = views._build_prediction_cache_key(
            image_bytes=image_bytes,
            requested_season="auto",
            cache_namespace="namespace-a",
            normalize_requested_season=services.normalize_requested_season,
        )
        rabi_key = views._build_prediction_cache_key(
            image_bytes=image_bytes,
            requested_season="rabi",
            cache_namespace="namespace-a",
            normalize_requested_season=services.normalize_requested_season,
        )

        self.assertNotEqual(auto_key, rabi_key)

    def test_prediction_cache_key_changes_when_model_artifacts_change(self) -> None:
        image_bytes = b"sample-leaf-image"
        current_key = views._build_prediction_cache_key(
            image_bytes=image_bytes,
            requested_season="rabi",
            cache_namespace="namespace-a",
            normalize_requested_season=services.normalize_requested_season,
        )
        refreshed_key = views._build_prediction_cache_key(
            image_bytes=image_bytes,
            requested_season="rabi",
            cache_namespace="namespace-b",
            normalize_requested_season=services.normalize_requested_season,
        )

        self.assertNotEqual(current_key, refreshed_key)


class ModelFallbackTests(SimpleTestCase):
    def test_find_exported_h5_fallback_reads_training_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            model_dir = root / "rabi"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "ssgrow_disease_model_v2.keras"
            model_path.write_bytes(b"PK\x03\x04")

            fallback_path = root / "assets" / "models" / "rabi_cnn.h5"
            fallback_path.parent.mkdir(parents=True, exist_ok=True)
            fallback_path.write_bytes(b"HDF5")

            summary_path = model_dir / "training_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "config": {
                            "backend_model_path": str(fallback_path),
                        }
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                services._find_exported_h5_fallback(model_path),
                fallback_path,
            )


class HealthyLesionLogicGateTests(SimpleTestCase):
    def test_high_confidence_healthy_prediction_bypasses_visual_analysis(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ENABLE_HEALTHY_LESION_OVERRIDE": "true",
                "HEALTHY_VISUAL_BYPASS_CONFIDENCE": "95",
            },
            clear=False,
        ):
            self.assertTrue(
                services._should_bypass_healthy_visual_analysis(
                    "healthy_mango_leaf",
                    97.4,
                )
            )
            self.assertFalse(
                services._should_bypass_healthy_visual_analysis(
                    "healthy_mango_leaf",
                    94.9,
                )
            )

    def test_weighted_gate_suppresses_weak_lesion_signal_for_healthy_prediction(self) -> None:
        leaf_visual_analysis = {
            "anomalies_textures": {
                "lesion_summary": "single notable dark lesion detected",
                "dark_center_presence": True,
                "chlorosis_halo": "chlorotic halo is not strongly visible",
                "lesions_detected": 1,
                "metrics": {
                    "lesion_area_percent_of_leaf": 0.18,
                    "yellow_chlorosis_percent_of_leaf": 0.0,
                    "largest_lesions": [{"pixels": 18}],
                },
            }
        }

        with patch.dict(
            os.environ,
            {
                "ENABLE_HEALTHY_LESION_OVERRIDE": "true",
                "HEALTHY_LESION_SIGNAL_THRESHOLD": "0.55",
                "HEALTHY_LESION_SCORE_THRESHOLD": "0.15",
                "HEALTHY_LESION_CONFIDENCE_PENALTY": "0.70",
            },
            clear=False,
        ):
            result = services.apply_logic_gate_override(
                predicted_label="healthy_mango_leaf",
                predicted_confidence=93.0,
                leaf_visual_analysis=leaf_visual_analysis,
            )

        self.assertFalse(result["override_applied"])
        self.assertTrue(result["suppress_lesion_output"])
        self.assertEqual(result["diagnosis_status"], "ok")
        self.assertEqual(result["label"], "healthy_mango_leaf")
        self.assertEqual(result["heuristic_lesion_count"], 0)

    def test_low_confidence_healthy_prediction_does_not_hide_lesion_output(self) -> None:
        leaf_visual_analysis = {
            "anomalies_textures": {
                "lesion_summary": "single notable dark lesion detected",
                "dark_center_presence": True,
                "chlorosis_halo": "chlorotic halo is not strongly visible",
                "lesions_detected": 1,
                "metrics": {
                    "lesion_area_percent_of_leaf": 0.18,
                    "yellow_chlorosis_percent_of_leaf": 0.0,
                    "largest_lesions": [{"pixels": 18}],
                },
            }
        }

        with patch.dict(
            os.environ,
            {
                "ENABLE_HEALTHY_LESION_OVERRIDE": "true",
                "HEALTHY_LESION_SIGNAL_THRESHOLD": "0.55",
                "HEALTHY_LESION_SCORE_THRESHOLD": "0.15",
                "HEALTHY_LESION_CONFIDENCE_PENALTY": "0.70",
            },
            clear=False,
        ):
            result = services.apply_logic_gate_override(
                predicted_label="healthy_mango_leaf",
                predicted_confidence=41.0,
                leaf_visual_analysis=leaf_visual_analysis,
            )

        self.assertFalse(result["override_applied"])
        self.assertFalse(result["suppress_lesion_output"])
        self.assertEqual(result["heuristic_lesion_count"], 1)

    def test_weighted_gate_escalates_only_for_strong_lesion_signal(self) -> None:
        leaf_visual_analysis = {
            "anomalies_textures": {
                "lesion_summary": "distinct necrotic lesions detected",
                "dark_center_presence": True,
                "chlorosis_halo": "faint chlorotic halo likely around lesions",
                "lesions_detected": 3,
                "metrics": {
                    "lesion_area_percent_of_leaf": 1.1,
                    "yellow_chlorosis_percent_of_leaf": 3.8,
                    "largest_lesions": [{"pixels": 52}, {"pixels": 41}],
                },
            }
        }

        with patch.dict(
            os.environ,
            {
                "ENABLE_HEALTHY_LESION_OVERRIDE": "true",
                "HEALTHY_LESION_SIGNAL_THRESHOLD": "0.55",
                "HEALTHY_LESION_SCORE_THRESHOLD": "0.15",
                "HEALTHY_LESION_CONFIDENCE_PENALTY": "0.70",
            },
            clear=False,
        ):
            result = services.apply_logic_gate_override(
                predicted_label="healthy_mango_leaf",
                predicted_confidence=89.0,
                leaf_visual_analysis=leaf_visual_analysis,
            )

        self.assertTrue(result["override_applied"])
        self.assertFalse(result["suppress_lesion_output"])
        self.assertEqual(result["diagnosis_status"], "manual_review_required")
        self.assertEqual(result["label"], services.MANUAL_REVIEW_LABEL)
        self.assertGreater(result["heuristic_lesion_count"], 0)


class MultiStagePipelineHelperTests(SimpleTestCase):
    def _png_bytes(self, *, color: tuple[int, int, int], size: tuple[int, int] = (64, 64)) -> bytes:
        image = Image.new("RGB", size, color=color)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _png_bytes_from_bgr(self, image_bgr: np.ndarray) -> bytes:
        image = Image.fromarray(image_bgr[:, :, ::-1].astype(np.uint8), mode="RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def test_input_gatekeeper_rejects_extreme_glare(self) -> None:
        with patch.dict(
            os.environ,
            {
                "MAX_IMAGE_GLARE_RATIO": "0.12",
            },
            clear=False,
        ):
            with self.assertRaises(services.ImageQualityError):
                services.run_input_gatekeepers(
                    self._png_bytes(color=(255, 255, 255)),
                    "image/png",
                )

    def test_temperature_scaling_softens_overconfident_distribution(self) -> None:
        raw_scores = np.array([0.99, 0.005, 0.005], dtype=np.float32)
        calibrated = services._apply_temperature_scaling(raw_scores, temperature=2.0)

        self.assertAlmostEqual(float(np.sum(calibrated)), 1.0, places=5)
        self.assertLess(float(calibrated[0]), 0.99)
        self.assertGreater(float(calibrated[1]), 0.005)

    def test_leaf_segmentation_is_applied_before_inference(self) -> None:
        image = np.zeros((120, 120, 3), dtype=np.uint8)
        image[:, :] = [20, 40, 120]
        image[18:102, 30:90] = [35, 170, 45]

        segmented, metadata = services._segment_leaf_foreground(image)

        self.assertTrue(metadata["applied"])
        self.assertEqual(segmented.shape[2], image.shape[2])
        self.assertGreater(float(metadata["leaf_coverage_ratio"]), 0.08)
        self.assertLess(float(metadata["leaf_coverage_ratio"]), 0.9)

    def test_adaptive_correction_inpaints_specular_regions_and_clips_gain(self) -> None:
        image = np.zeros((160, 160, 3), dtype=np.uint8)
        image[:, :] = [18, 32, 90]
        image[18:142, 28:132] = [40, 170, 45]
        image[48:92, 60:104] = [255, 255, 255]
        image[72:110, 54:62] = [26, 120, 30]

        corrected, metrics = services._apply_adaptive_leaf_illumination_correction(image)

        self.assertEqual(corrected.shape, image.shape)
        self.assertTrue(metrics["specular_inpaint_applied"])
        self.assertGreater(float(metrics["specular_ratio_before"]), float(metrics["specular_ratio_after"]))
        self.assertGreaterEqual(float(metrics["gain_min"]), 0.6)
        self.assertLessEqual(float(metrics["gain_max"]), 2.2)

    def test_adaptive_correction_warns_when_glare_covers_large_area(self) -> None:
        image = np.zeros((160, 160, 3), dtype=np.uint8)
        image[:, :] = [18, 32, 90]
        image[16:144, 24:136] = [42, 165, 48]
        image[36:112, 46:122] = [255, 255, 255]

        _corrected, metrics = services._apply_adaptive_leaf_illumination_correction(image)

        warnings = metrics["warnings"]
        self.assertTrue(any("Retake is recommended" in item for item in warnings))

    def test_input_gatekeeper_returns_preprocessing_metrics(self) -> None:
        image = np.zeros((160, 160, 3), dtype=np.uint8)
        image[:, :] = [18, 32, 90]
        image[20:144, 30:132] = [38, 172, 44]
        image[48:92, 64:104] = [255, 255, 255]
        image[34:130:8, 40:122] = [28, 138, 36]

        prepared = services.run_input_gatekeepers(
            self._png_bytes_from_bgr(image),
            "image/png",
        )

        self.assertIn("pipeline_version", prepared.preprocessing_metrics)
        self.assertIn("selected_variant", prepared.preprocessing_metrics)
        self.assertIn(
            prepared.preprocessing_metrics["selected_variant"],
            {
                "original",
                "segmented_leaf",
                "adaptive_correction",
                "adaptive_correction_with_segmentation",
            },
        )
        self.assertIn("quality_score_selected", prepared.preprocessing_metrics)
        self.assertIsInstance(prepared.preprocessing_metrics["segmentation_applied"], bool)

    def test_stage1_route_blocks_explicit_wrong_season(self) -> None:
        models = SimpleNamespace(
            route_supported_crop=lambda _image_bytes: services.CropRoutingDecision(
                accepted=True,
                crop="mango",
                crop_confidence=93.0,
                season="all_season",
                reason="Stage-1 crop gate routed this image to mango.",
                source="stage1_crop_classifier",
            )
        )

        decision = services._resolve_stage1_crop_route(
            models=models,
            image_bytes=b"demo",
            requested_season="rabi",
        )

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertFalse(decision.accepted)
        self.assertIn("All Season", decision.reason)

    @patch(
        "predictor.services.build_structured_leaf_visual_analysis",
        return_value={"anomalies_textures": {"lesions_detected": 0}},
    )
    def test_unsupported_result_marks_non_leaf_or_fruit_upload(
        self,
        _mock_visual_analysis,
    ) -> None:
        result = services._build_unsupported_prediction_result(
            crop_detected="others",
            reason="Stage-1 crop gate classified this upload as a non-leaf-or-fruit image with 98.0% confidence.",
            seasonal_comparison={},
            image_bytes=b"demo",
            content_type="image/png",
        )

        self.assertEqual(result.diagnosis_status, "invalid_leaf_or_fruit")
        self.assertEqual(result.label, "invalid_leaf_or_fruit")
        self.assertEqual(result.status_message, services.INVALID_LEAF_OR_FRUIT_MESSAGE)
        self.assertEqual(result.farmer_report["type_of_disease"], "Not a leaf or fruit")
        self.assertIn("upload it again", result.farmer_action_plan_markdown.lower())

    def test_retrieve_agronomy_references_uses_local_docs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            doc_path = Path(tmp_dir) / "tomato_early_blight.md"
            doc_path.write_text(
                (
                    "Tomato early blight management: remove infected leaves, "
                    "improve airflow, and avoid overhead irrigation."
                ),
                encoding="utf-8",
            )

            previous_cache = services.agronomy_reference_index
            services.agronomy_reference_index = None
            try:
                with patch.dict(
                    os.environ,
                    {
                        "ENABLE_AGRONOMY_RETRIEVAL": "true",
                        "AGRONOMY_DOCS_DIR": tmp_dir,
                        "AGRONOMY_RETRIEVAL_MAX_REFERENCES": "2",
                    },
                    clear=False,
                ):
                    references = services.retrieve_agronomy_references(
                        crop_detected="tomato",
                        disease_label="early_blight_tomato",
                        pesticide={
                            "recommended_pesticide": "Mancozeb",
                            "active_ingredient": "mancozeb",
                            "usage_note": "Follow local label guidance.",
                        },
                    )
            finally:
                services.agronomy_reference_index = previous_cache

        self.assertGreaterEqual(len(references), 1)
        self.assertTrue(
            any("remove infected leaves" in item["snippet"].lower() for item in references)
        )

    def test_handle_unknown_crop_rejects_non_plant_image(self) -> None:
        class FakeDiscoveryCNN:
            def predict(self, _image_data):
                return ("unknown", 0.31)

        with tempfile.TemporaryDirectory() as tmp_dir:
            response = services.handle_unknown_crop(
                Image.new("RGB", (32, 32), color=(120, 120, 120)),
                FakeDiscoveryCNN(),
                tmp_dir,
            )

            self.assertEqual(
                response,
                {
                    "status": "rejected",
                    "message": "No recognizable plant features detected.",
                },
            )
            self.assertEqual(list(Path(tmp_dir).iterdir()), [])

    def test_handle_unknown_crop_creates_folder_and_saves_unique_image(self) -> None:
        class FakeDiscoveryCNN:
            def predict(self, _image_data):
                return ("Mango", 0.88)

        with tempfile.TemporaryDirectory() as tmp_dir:
            image = Image.new("RGB", (40, 40), color=(22, 180, 55))

            first = services.handle_unknown_crop(image, FakeDiscoveryCNN(), tmp_dir)
            second = services.handle_unknown_crop(image, FakeDiscoveryCNN(), tmp_dir)

            mango_dir = Path(tmp_dir) / "Mango"
            saved_files = sorted(mango_dir.glob("*.png"))

            self.assertEqual(first["status"], "saved")
            self.assertEqual(second["status"], "saved")
            self.assertTrue(mango_dir.exists())
            self.assertEqual(len(saved_files), 2)
            self.assertNotEqual(saved_files[0].name, saved_files[1].name)
            self.assertIn("Mango", first["message"])


class PredictUploadPersistenceTests(TestCase):
    @patch("predictor.views._load_prediction_services")
    def test_predict_persists_uploaded_image_with_source_and_type(
        self,
        mock_load_prediction_services,
    ) -> None:
        def normalize_requested_season(_value: str | None) -> str:
            return "auto"

        mock_load_prediction_services.return_value = {
            "model_version": "SSGrow-CNN-v2",
            "trained_seasons": ("Kharif", "Rabi", "All Season"),
            "image_quality_error": ValueError,
            "call_llm": None,
            "cache_namespace": "test-namespace",
            "normalize_requested_season": normalize_requested_season,
            "run_prediction": lambda **_kwargs: SimpleNamespace(
                label="onion_purple_blotch",
                crop_detected="onion",
                confidence=78.08,
                season_used="rabi",
                verification_passed=True,
                verification_reason="ok",
                seasonal_comparison={},
                status_message="Prediction completed successfully.",
                diagnosis_status="ok",
                override_applied=False,
                override_reason="",
                model_label_before_override="onion_purple_blotch",
                model_confidence_before_override=78.08,
                heuristic_lesion_count=1,
                recommended_pesticide="Expert review required",
                active_ingredient="n/a",
                usage_note="No exact pesticide match found for this disease label.",
                leaf_visual_analysis={},
                farmer_report={},
                farmer_action_plan_markdown="",
            ),
        }

        response = self.client.post(
            "/api/ai/predict",
            {
                "file": SimpleUploadedFile(
                    "camera-leaf.jpg",
                    b"fake-image-bytes",
                    content_type="image/jpeg",
                ),
                "season": "auto",
                "file_sources": UploadedLeafImage.SOURCE_CAMERA,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["preprocessing_metrics"], {})
        self.assertEqual(UploadedLeafImage.objects.count(), 1)

        stored_image = UploadedLeafImage.objects.get()
        self.assertEqual(stored_image.source_type, UploadedLeafImage.SOURCE_CAMERA)
        self.assertEqual(stored_image.mime_type, "image/jpeg")
        self.assertEqual(stored_image.image_type, "jpeg")
        self.assertEqual(stored_image.file_name, "camera-leaf.jpg")
        self.assertEqual(bytes(stored_image.image_data), b"fake-image-bytes")
