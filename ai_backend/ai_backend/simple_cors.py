import os

from django.http import HttpResponse


class SimpleCorsMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        self.allow_origin = os.getenv("CORS_ALLOW_ORIGIN", "*")

    def __call__(self, request):
        if request.method == "OPTIONS":
            response = HttpResponse(status=200)
        else:
            response = self.get_response(request)

        response["Access-Control-Allow-Origin"] = self.allow_origin
        response["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response
