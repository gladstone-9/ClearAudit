from django.shortcuts import render
from django.views import generic
from django.http import JsonResponse
from django.http import HttpResponse
from django.views import View


class PortalUploadView(View):
    template_name = "portal_upload.html"
    
    def get(self, request):
        return render(request, 'portal_upload.html')  # your HTML form

    def post(self, request):
        csv_file = request.FILES.get('csvFile')
        
        return render(request, "success_upload.html")
    
class SuccessView(generic.base.TemplateView):
    template_name = "success_upload.html"
