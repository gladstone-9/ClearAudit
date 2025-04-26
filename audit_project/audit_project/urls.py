"""
URL configuration for audit_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from audit_app.views import *

from django.conf import settings
from django.conf.urls.static import static


from django.views.generic.base import RedirectView

favicon_view = RedirectView.as_view(url='/static/favicon.ico', permanent=True)


urlpatterns = [
    path('admin/', admin.site.urls),
    path("", PortalUploadView.as_view(), name="portal_upload"),
    path('explore/<int:file_id>/', ExploreStatisticsView.as_view(), name='explore_statistics'),
    path('slider/', range_slider_view, name='range_slider'),
    path('dp-data/', dp_histogram_data, name='dp_histogram_data'),
    path('dp-sum-data/', dp_sum_data, name='dp_sum_data'),
    path('pca-data/', pca_view, name='pca_data'),
    path('simulate-attack/', simulate_attack_view, name='simulate_attack'),
    path('synthetic-data/', synthetic_data_view, name='generate_synthetic_data'),
    path('data-release/', publish_data_release_view, name='publish_data_release'),
]
